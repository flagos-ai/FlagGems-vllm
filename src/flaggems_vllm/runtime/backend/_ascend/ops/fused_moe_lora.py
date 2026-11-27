# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend vector projections for routed MoE LoRA."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fused_moe_lora import _GEMV_LAUNCH, _gemv, _tp_exchange
from flaggems_vllm.runtime.backend._ascend import _fused_moe_lora_cube as cube
from flaggems_vllm.utils.libentry import major_version, minor_version


@triton.jit
def _sorted_routes(
    Routes,
    Ids,
    Enabled,
    Perm,
    Experts,
    Post,
    Count,
    base_program,
    P: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    RAW: tl.constexpr,
    BLOCKS: tl.constexpr,
    BM: tl.constexpr,
    DEVICE_COUNT: tl.constexpr,
):
    program = base_program + tl.program_id(0).to(tl.int64)
    block = program % tl.cdiv(RAW, BM)
    slot = program // tl.cdiv(RAW, BM)
    if DEVICE_COUNT:
        if slot >= tl.load(Count):
            return
    adapter = tl.load(Ids + slot).to(tl.int64)
    if adapter < 0 or adapter >= L:
        return
    if tl.load(Enabled + adapter) == 0:
        return
    count = tl.load(Post + adapter)
    start = block * BM
    if start >= count:
        return
    expert = tl.load(Experts + adapter * BLOCKS + block).to(tl.int64)
    if expert < 0 or expert >= E:
        return
    offsets = start + tl.arange(0, BM)
    pair = tl.load(Perm + adapter * RAW + offsets, offsets < RAW, other=-1)
    live = (offsets < count) & (pair >= 0) & (pair < P)
    # Each scattered code has a private 32-byte hardware transfer block.
    tl.store(
        Routes + tl.where(live, pair, 0).to(tl.int64) * 4,
        1 + adapter * E + expert,
        live,
    )


@triton.jit
def _project_rows(
    X,
    W,
    Y,
    Codes,
    Mapping,
    Experts,
    Enabled,
    Weights,
    base_program,
    TOPK: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    SORTED: tl.constexpr,
    FACTOR: tl.constexpr,
    WEIGHTED: tl.constexpr,
    ADD: tl.constexpr,
    ROUND_DELTA: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    program = base_program + tl.program_id(0).to(tl.int64)
    pair = program // tl.cdiv(N, BN)
    if SORTED:
        code = tl.load(Codes + pair * 4)
        if code == 0:
            return
        weight_index = (code - 1).to(tl.int64)
    else:
        adapter = tl.load(Mapping + pair // TOPK).to(tl.int64)
        if adapter < 0 or adapter >= L:
            return
        expert = tl.load(Experts + pair).to(tl.int64)
        if expert < 0 or expert >= E:
            return
        if tl.load(Enabled + adapter) == 0:
            return
        weight_index = adapter * E + expert
    W += weight_index * N * K
    X += pair // FACTOR * XS
    cols = (program % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    value = tl.full((BN,), 0, tl.float32)
    for start in range(0, K, BK):
        ks = start + k
        x = tl.load(X + ks, ks < K, other=0).to(tl.float32)
        w = tl.load(
            W + cols[:, None].to(tl.int64) * K + ks[None, :],
            (cols[:, None] < N) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        value += tl.sum(w * x[None, :], axis=1)
    if WEIGHTED:
        value *= tl.load(Weights + pair).to(tl.float32)
    if ROUND_DELTA:
        value = value.to(Y.dtype.element_ty).to(tl.float32)
    y = Y + pair * YS + cols
    if ADD:
        value += tl.load(y, cols < N, other=0).to(tl.float32)
    tl.store(y, value, cols < N)


def _launch_tiles(kernel, count, pointers, constants):
    # CANN 9.0 rtKernelLaunch on 910B1 accepts at most 65,535 programs
    # per launch, including all dimensions of a multidimensional grid.
    for base in range(0, count, 65535):
        kernel[(min(count - base, 65535),)](
            *pointers, base, *constants, num_warps=4, num_stages=1
        )


def _needs_staged(output, x, a_list, b_list, offset, pairs):
    # The fused decode kernel writes directly into the public output. Dense,
    # aligned rows keep stores by different programs in disjoint DMA blocks.
    rank, width = a_list[0].shape[-2:]
    if rank > 64 or rank & (rank - 1) or width % 64 or offset % 16:
        return True
    if output.shape[-1] % 16 or any(b.shape[2] % 64 for b in b_list):
        return True
    if any(
        not t.is_contiguous() or t.storage_offset() % 16
        for t in (output, x, *a_list, *b_list)
    ):
        return True
    max_tiles = max(triton.cdiv(b.shape[2], 128) for b in b_list)
    return pairs * max_tiles > 65535


_COMPILED_LAUNCH_HAS_CONSTEXPRS = (
    major_version == 3
    and 3 <= minor_version <= 6
    and hasattr(_gemv, "kernel_cache")
    and hasattr(_gemv, "_launch_options_key")
)
_DECODE_KEY = (
    _gemv._launch_options_key({"num_warps": 4, "num_stages": 1})
    if _COMPILED_LAUNCH_HAS_CONSTEXPRS
    else ()
)


def _decode(
    output, x, a_list, b_list, weights, experts, mapping, enabled, offset, weighted, add
):
    pairs, slices = weights.numel(), len(a_list)
    loras, n_experts, rank, width = a_list[0].shape
    topk = weights.shape[1]
    chunk = 128 if pairs * slices < 16 else 256
    for a, b in zip(a_list, b_list):
        n = b.shape[2]
        if not n:
            continue
        y = (
            output
            if offset == 0 and output.shape[2] == n
            else output[:, :, offset : offset + n]
        )
        constants = (
            pairs,
            topk,
            width,
            rank,
            n,
            loras,
            n_experts,
            slices,
            1,
            x.stride(),
            a.stride(),
            b.stride(),
            y.stride(),
            weights.stride(),
            mapping.stride(0),
            experts.stride(0),
            enabled.stride(0),
            add,
            weighted,
            pairs * rank <= 1024,
            chunk,
            rank,
        )
        grid = (pairs, triton.cdiv(n, chunk), 1)
        pointers = (x, a, b, y, None, None, weights, mapping, experts, enabled)
        # Match LibEntry's normal key, including real pointer alignment, strides
        # and fixed compile options. Every launch still binds the current tensors.
        if _COMPILED_LAUNCH_HAS_CONSTEXPRS:
            key = _gemv.key(pointers, (), [*constants, 64, 64]) + _DECODE_KEY
            entry = _gemv.kernel_cache[output.device.index].get(key)
            if entry is not None:
                entry[0][grid](*pointers, *constants, 64, 64)
                offset += n
                continue
        _gemv[grid](*pointers, *constants, **_GEMV_LAUNCH)
        offset += n


def _ascend_launch(output, *projection_args):
    with runtime.torch_device_fn.device(output.device):
        return _launch_impl(output, *projection_args)


def _launch_impl(
    output,
    x,
    a_list,
    b_list,
    weights,
    sorted_ids,
    experts,
    padded_count,
    mapping,
    ids,
    active_count,
    enabled,
    block_m,
    active_groups,
    weighted,
    sharded,
    offset,
    add_inputs,
):
    # 910B requires dense tiles for arbitrary user views. Preserve row padding
    # in disjoint accumulator rows, then copy back only the public slice.
    device, dtype = output.device, output.dtype
    tokens, topk = weights.shape
    pairs, slices = weights.numel(), len(a_list)
    loras, n_experts, rank, _ = a_list[0].shape
    sorted_mode = sorted_ids is not None
    if (
        sorted_mode
        and not sharded
        and cube.selected(
            output, a_list, b_list, weights, sorted_ids, block_m, active_groups
        )
    ):
        return cube.launch(
            output,
            x,
            a_list,
            b_list,
            weights,
            sorted_ids,
            experts,
            padded_count,
            ids,
            active_count,
            enabled,
            block_m,
            active_groups,
            weighted,
            offset,
            add_inputs,
        )
    if not (sorted_mode or sharded) and not _needs_staged(
        output, x, a_list, b_list, offset, pairs
    ):
        return _decode(
            output,
            x,
            a_list,
            b_list,
            weights,
            experts,
            mapping,
            enabled,
            offset,
            weighted,
            add_inputs,
        )
    small = not (sorted_mode or sharded) and rank <= 64 and pairs * rank <= 1024
    experts, mapping, enabled = (t.contiguous() for t in (experts, mapping, enabled))
    if sorted_mode:
        codes = torch.zeros((pairs, 4), dtype=torch.int64, device=device)
        if sorted_ids.shape[1] and active_groups:
            device_count = active_count.device.type != "cpu"
            _launch_tiles(
                _sorted_routes,
                triton.cdiv(sorted_ids.shape[1], block_m) * active_groups,
                (
                    codes,
                    ids.contiguous(),
                    enabled,
                    sorted_ids.contiguous(),
                    experts,
                    padded_count.contiguous(),
                    active_count if device_count else None,
                ),
                (
                    pairs,
                    loras,
                    n_experts,
                    sorted_ids.shape[1],
                    experts.shape[1],
                    block_m,
                    device_count,
                ),
            )
    else:
        codes = None
    x = x.contiguous()
    weights = weights.contiguous()
    rank_pitch = triton.cdiv(rank, 32) * 32
    # Collectives read inactive rows as well. Local expansion only reads live
    # rows after their shrink projection has written the entire logical rank.
    allocate_cache = torch.zeros if sharded else torch.empty
    cache = allocate_cache(
        (slices, pairs, rank_pitch),
        dtype=torch.float32 if small else dtype,
        device=device,
    )

    def project(source, weight, dest, factor, scale, add, rounded):
        n, k = weight.shape[2:]
        bn, bk = 32, 64
        _launch_tiles(
            _project_rows,
            pairs * triton.cdiv(n, bn),
            (
                source,
                weight.contiguous(),
                dest,
                codes,
                mapping,
                experts,
                enabled,
                weights,
            ),
            (
                topk,
                loras,
                n_experts,
                k,
                n,
                source.stride(0),
                dest.stride(0),
                sorted_mode,
                factor,
                scale,
                add,
                rounded,
                bn,
                bk,
            ),
        )

    for s, a in enumerate(a_list):
        project(x, a, cache[s], 1 if weighted else topk, False, False, False)
    if sharded:
        full_rank = b_list[0].shape[-1]
        send = cache[..., :rank].contiguous().view(slices, tokens, topk, rank)
        cache = _tp_exchange(send, full_rank).view(slices, pairs, full_rank)
    for s, b in enumerate(b_list):
        n = b.shape[2]
        if not n:
            continue
        y = output[:, :, offset : offset + n]
        if (
            y.stride(2) == 1
            and y.stride(0) == topk * y.stride(1)
            and y.stride(1) % 16 == 0
            and y.storage_offset() % 16 == 0
            and n % 16 == 0
        ):
            project(cache[s], b, y.view(pairs, n), 1, weighted, add_inputs, not small)
            offset += n
            continue
        pitch = triton.cdiv(n, 32) * 32
        # A slice can begin on any channel and the public row stride may be odd.
        # Keep stores from different programs on disjoint physical blocks.
        target = torch.empty((pairs, pitch), dtype=dtype, device=device)
        view = target[:, :n].view(tokens, topk, n)
        view.copy_(y)
        project(cache[s], b, target, 1, weighted, add_inputs, not small)
        y.copy_(view)
        offset += n
