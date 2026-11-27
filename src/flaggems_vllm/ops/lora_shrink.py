# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_INDEX_DTYPES = (torch.int32, torch.int64)
_INPUT_DTYPES = (torch.float16, torch.bfloat16)
_OUTPUT_DTYPES = (*_INPUT_DTYPES, torch.float32)


def _configs(configs):
    # The first configuration uses portable tile sizes and one pipeline stage.
    return configs if runtime.device.vendor_name == "nvidia" else configs[:1]


@libentry()
@libtuner(
    configs=_configs(runtime.get_tuned_config("lora_shrink_zero")),
    key=["SIZE"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
@triton.jit
def _zero_kernel(OUT, SIZE, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + i, 0, mask=i < SIZE)


@triton.jit
def _token_rows(SORTED, STARTS, group, pos, count, T: tl.constexpr):
    start = tl.load(STARTS + group).to(tl.int64)
    valid = (pos < count) & (start >= 0) & (start <= T - 1 - pos)
    row = tl.load(SORTED + start + pos, mask=valid, other=-1).to(tl.int64)
    return row, valid & (row >= 0) & (row < T)


@triton.jit
def _slice_weight(weights, slice_id, S: tl.constexpr):
    # The tuple contains the actual tensor arguments for this call. The JIT
    # specializes the tuple arity and the CTA selects exactly one base pointer.
    weight = weights[0]
    for s in tl.static_range(1, S):
        weight = tl.where(slice_id == s, weights[s], weight)
    return weight


@libentry()
@libtuner(
    configs=_configs(runtime.get_tuned_config("lora_shrink_gemv")),
    key=["T", "R", "K", "L", "S", "ACTIVE", "X_ROW", "X_K"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
@triton.jit
def _gemv_kernel(
    X,
    A,
    OUT,
    SORTED,
    COUNTS,
    STARTS,
    IDS,
    scale,
    T: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    ACTIVE: tl.constexpr,
    X_ROW: tl.constexpr,
    X_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACKED: tl.constexpr = False,
):
    pos = tl.program_id(0)
    group = tl.program_id(1)
    if PACKED:
        slice_id = group // ACTIVE
        group = group % ACTIVE
        A = _slice_weight(A, slice_id, S)
        OUT = OUT + slice_id.to(tl.int64) * (T * R)
    adapter = tl.load(IDS + group)
    count = tl.load(COUNTS + group)
    if (adapter < 0) | (adapter >= L) | (pos >= count):
        return
    row, valid = _token_rows(SORTED, STARTS, group, pos, count, T)
    if not valid:
        return
    ranks = tl.program_id(2) * BLOCK_R + tl.arange(0, BLOCK_R)
    a = A + (adapter.to(tl.int64) * R + ranks[:, None]) * K
    acc = tl.full((BLOCK_R,), 0, tl.float32)
    for tile in range(tl.cdiv(K, BLOCK_K)):
        ks = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(X + row * X_ROW + ks * X_K, mask=ks < K, other=0).to(tl.float32)
        w = tl.load(
            a + ks[None, :], mask=(ranks[:, None] < R) & (ks[None, :] < K), other=0
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(OUT + row * R + ranks, acc * scale, mask=ranks < R)


@triton.jit
def _grouped_kernel(
    X,
    A,
    OUT,
    SORTED,
    COUNTS,
    STARTS,
    IDS,
    scale,
    T: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    ACTIVE: tl.constexpr,
    X_ROW: tl.constexpr,
    X_K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PACKED: tl.constexpr = False,
):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    if PACKED:
        slice_id = group // ACTIVE
        group = group % ACTIVE
        A = _slice_weight(A, slice_id, S)
        OUT = OUT + slice_id.to(tl.int64) * (SPLITS * T * R)
    split = tl.program_id(2)
    r_tiles = tl.cdiv(R, BLOCK_R)
    first = tile // r_tiles * BLOCK_M
    count = tl.load(COUNTS + group)
    adapter = tl.load(IDS + group)
    if (first >= count) | (adapter < 0) | (adapter >= L):
        return
    pos = first + tl.arange(0, BLOCK_M)
    rows, valid = _token_rows(SORTED, STARTS, group, pos, count, T)
    ranks = tile % r_tiles * BLOCK_R + tl.arange(0, BLOCK_R)
    x = X + rows[:, None] * X_ROW
    a = A + (adapter.to(tl.int64) * R + ranks[:, None]) * K
    acc = tl.full((BLOCK_M, BLOCK_R), 0, tl.float32)
    for k_block in range(tl.cdiv(K, BLOCK_K * SPLITS)):
        ks = (k_block * SPLITS + split) * BLOCK_K + tl.arange(0, BLOCK_K)
        xt = tl.load(
            x + ks[None, :] * X_K, mask=valid[:, None] & (ks[None, :] < K), other=0
        )
        wt = tl.load(
            a + ks[None, :], mask=(ranks[:, None] < R) & (ks[None, :] < K), other=0
        )
        acc += tl.dot(xt, tl.trans(wt))
    if SPLITS == 1:
        values = acc * scale
    else:
        values = acc
    dst = OUT + (split.to(tl.int64) * T + rows[:, None]) * R + ranks[None, :]
    tl.store(dst, values, mask=valid[:, None] & (ranks[None, :] < R))


_tuned_grouped_kernel = libentry()(
    libtuner(
        configs=_configs(runtime.get_tuned_config("lora_shrink_grouped")),
        key=["T", "R", "K", "L", "S", "ACTIVE", "X_ROW", "X_K", "SPLITS"],
        warmup=5,
        rep=10,
        use_cuda_graph=False,
    )(_grouped_kernel)
)


@libentry()
@libtuner(
    configs=_configs(runtime.get_tuned_config("lora_shrink_reduce")),
    key=["T", "R", "L", "S", "ACTIVE", "SPLITS"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
@triton.jit
def _reduce_kernel(
    PARTIAL,
    OUT,
    SORTED,
    COUNTS,
    STARTS,
    IDS,
    scale,
    T: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    ACTIVE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PACKED: tl.constexpr = False,
):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    if PACKED:
        slice_id = group // ACTIVE
        group = group % ACTIVE
        PARTIAL = PARTIAL + slice_id.to(tl.int64) * (SPLITS * T * R)
        OUT = OUT + slice_id.to(tl.int64) * (T * R)
    r_tiles = tl.cdiv(R, BLOCK_R)
    first = tile // r_tiles * BLOCK_M
    adapter = tl.load(IDS + group)
    count = tl.load(COUNTS + group)
    if (first >= count) | (adapter < 0) | (adapter >= L):
        return
    rows, valid = _token_rows(
        SORTED, STARTS, group, first + tl.arange(0, BLOCK_M), count, T
    )
    ranks = tile % r_tiles * BLOCK_R + tl.arange(0, BLOCK_R)
    offsets = rows[:, None] * R + ranks[None, :]
    mask = valid[:, None] & (ranks[None, :] < R)
    acc = tl.full((BLOCK_M, BLOCK_R), 0, tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(PARTIAL + split * (T * R) + offsets, mask=mask, other=0)
    tl.store(OUT + offsets, acc * scale, mask=mask)


_packed_plans = {}


def _launch(entry, args, grid, key):
    if key is None:
        entry[grid](*args)
        return
    # LibEntry's generic argument key retains a tuple of tensor objects.
    # Cache only the selected tile and grid here. Every launch still passes
    # fresh tensors to Triton, which handles pointer and alignment changes.
    key = (entry, *key)
    plan = _packed_plans.get(key)
    if plan is None:
        with entry.lock:
            plan = _packed_plans.get(key)
            if plan is None:
                tuner = entry.fn
                tuner[grid](*args, PACKED=True)
                options = tuner.best_config.all_kwargs()
                plan = (grid(options), options)
                if len(_packed_plans) >= 256:
                    _packed_plans.clear()
                _packed_plans[key] = plan
                return
    launch_grid, options = plan
    entry.jit_function[launch_grid](*args, PACKED=True, **options)


def _project(inputs, weights, output, grouped, scale, geometry, invariant):
    tokens, rank, hidden, slots, slices, active, _, _ = geometry
    if not rank:
        return
    vendor = runtime.device.vendor_name
    # Bound both kernel argument count and CUDA-like grid-y extents. Larger
    # lists use the same computation, with one typed tensor argument per launch.
    packed = (
        vendor in ("nvidia", "metax", "thead", "hygon")
        and slices <= 8
        and active * slices <= 65535
    )
    if packed:
        output.zero_()
    else:
        size = slices * tokens * rank
        _zero_kernel[lambda meta: (triton.cdiv(size, meta["BLOCK"]),)](output, size)
    if not active or not slots:
        return

    long_gemv = tokens <= 32 and (
        (vendor == "nvidia" and hidden >= 8192)
        or (vendor in ("metax", "thead", "hygon") and hidden >= 4096)
    )
    gemv = not invariant and (tokens <= 8 or long_gemv)
    splits = 8 if invariant and hidden >= 1024 else 1
    key = None
    if packed:
        groups = active * slices
        launches = ((tuple(weights), output),)
        key = (
            inputs.device,
            inputs.dtype,
            output.dtype,
            tuple(m.dtype for m in grouped),
            geometry,
            splits,
        )
    else:
        groups = active
        launches = ((w, output[s]) for s, w in enumerate(weights))

    if gemv:

        def grid(meta):
            return (tokens, groups, triton.cdiv(rank, meta["BLOCK_R"]))

        for weight, out in launches:
            _launch(
                _gemv_kernel,
                (inputs, weight, out, *grouped, scale, *geometry),
                grid,
                key,
            )
        return

    partial = None
    if splits > 1:
        partial = torch.empty(
            ((slices if packed else 1), splits, tokens, rank),
            device=inputs.device,
            dtype=torch.float32,
        )

    def group_grid(meta):
        return (
            triton.cdiv(tokens, meta["BLOCK_M"]) * triton.cdiv(rank, meta["BLOCK_R"]),
            groups,
            splits,
        )

    for weight, out in launches:
        dest = out if splits == 1 else partial
        args = (inputs, weight, dest, *grouped, scale, *geometry, splits)
        if invariant:
            # Do not vary the MMA tile with T. Each token must retain exactly
            # the same K partition and reduction order when regrouped.
            options = dict(
                BLOCK_M=16, BLOCK_R=16, BLOCK_K=64, num_warps=4, num_stages=1
            )
            _grouped_kernel[group_grid(options)](*args, PACKED=packed, **options)
        else:
            _launch(_tuned_grouped_kernel, args, group_grid, key)
        if splits > 1:
            args = (
                partial,
                out,
                *grouped,
                scale,
                tokens,
                rank,
                slots,
                slices,
                active,
                splits,
            )
            _launch(_reduce_kernel, args, lambda meta: group_grid(meta)[:2], key)


def _cpu_control(tensor, name, dtypes):
    if tensor.device.type != "cpu" or tensor.shape != (1,):
        raise ValueError(f"{name} must be a CPU tensor of shape [1]")
    if tensor.dtype not in dtypes:
        raise TypeError(f"invalid dtype for {name}")


def _shared_view_overlaps(output, source):
    if not output.numel() or not source.numel():
        return False
    # Inclusive element offsets in the source that intersect output bytes.
    delta = output.data_ptr() - source.data_ptr()
    elem = source.element_size()
    low = max(0, delta // elem)
    high = (delta + output.nbytes - 1) // elem
    if source.is_contiguous():
        return low <= min(source.numel() - 1, high)
    # Only X may be noncontiguous. Inspect its shorter axis and intersect each
    # strided run on the other axis with [low, high], including broadcast views.
    (rows, row_step), (cols, col_step) = sorted(zip(source.shape, source.stride()))
    high = min(high, (rows - 1) * row_step + (cols - 1) * col_step)
    if low > high:
        return False
    for row in range(rows if row_step else 1):
        start = row * row_step
        if start > high:
            break
        col = max(0, (low - start + col_step - 1) // col_step) if col_step else 0
        if col < cols and low <= start + col * col_step <= high:
            return True
    return False


def _validate(inputs, weights, output, metadata, active):
    device = inputs.device
    if device.type in ("cpu", "meta"):
        raise NotImplementedError("lora_shrink requires an accelerator")
    if not weights or output.ndim != 3:
        raise ValueError(
            "expected at least one weight and output [slices, tokens, rank]"
        )
    tokens, hidden = inputs.shape
    slices, out_tokens, rank = output.shape
    if slices != len(weights) or out_tokens != tokens:
        raise ValueError("output shape must match the weight slices and input tokens")
    if inputs.dtype not in _INPUT_DTYPES or output.dtype not in _OUTPUT_DTYPES:
        raise TypeError("expected FP16/BF16 inputs and FP16/BF16/FP32 output")
    if not output.is_contiguous():
        raise ValueError("output_tensor must be contiguous")
    if output.device != device:
        raise ValueError("output_tensor and inputs must be on the same device")
    if inputs.layout != torch.strided:
        raise ValueError("inputs must be strided")
    if weights[0].ndim not in (3, 4):
        raise ValueError("LoRA weights must have rank 3 or 4")
    slots = weights[0].size(0)
    for weight in weights:
        if weight.dtype != inputs.dtype:
            raise TypeError("all LoRA weights must match the input dtype")
        if weight.shape not in ((slots, rank, hidden), (slots, 1, rank, hidden)):
            raise ValueError("LoRA weights must have shape [L,R,K] or [L,1,R,K]")
        if not weight.is_contiguous():
            raise ValueError("LoRA weights must be contiguous")
    groups = metadata[4].numel()
    if not 0 <= active <= groups:
        raise ValueError("num_active_loras exceeds metadata capacity")
    for tensor, length in zip(metadata, (tokens, tokens, groups, groups + 1, groups)):
        if tensor.shape != (length,) or not tensor.is_contiguous():
            raise ValueError("invalid grouped LoRA metadata shape or layout")
        if tensor.dtype not in _INDEX_DTYPES:
            raise TypeError("LoRA indices, counts, and offsets must be int32 or int64")
    for source in (inputs, *weights, *metadata):
        if source.device != device:
            raise ValueError(
                "all data and grouped metadata must be on the input device"
            )
        if torch._C._overlaps(output, source) and _shared_view_overlaps(output, source):
            raise ValueError("output_tensor must not overlap inputs or metadata")
    return (tokens, rank, hidden, slots, slices, active, *inputs.stride())


def lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    scaling: float,
) -> None:
    """Write scaled per-adapter A projections to output[slice, token, rank].

    A true no-LoRA flag preserves the caller's buffer. Otherwise unassigned rows
    are zero. Only the prefix of groups specified by num_active_loras is active.
    Each live group describes distinct token rows in the sorted index vector.
    """
    logger.debug("GEMS LORA_SHRINK")
    _cpu_control(no_lora_flag_cpu, "no_lora_flag_cpu", (torch.bool,))
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape [T,K]")
    if inputs.size(0) == 0 or no_lora_flag_cpu.item():
        return
    _cpu_control(num_active_loras, "num_active_loras", _INDEX_DTYPES)
    active = num_active_loras.item()
    grouped = (
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
    )
    geometry = _validate(
        inputs, lora_a_weights, output_tensor, (token_lora_mapping, *grouped), active
    )
    invariant = bool(int(os.getenv("VLLM_BATCH_INVARIANT", "0")))
    with runtime.torch_device_fn.device(inputs.device):
        _project(
            inputs, lora_a_weights, output_tensor, grouped, scaling, geometry, invariant
        )
