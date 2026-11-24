# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import math
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

_HAS_ASYNC_LOAD = callable(getattr(tle, "load", None))


@triton.jit
def _rank_key(scores, experts, BF16: tl.constexpr = False):
    # Unsigned comparison orders the float first and then prefers lower IDs.
    scores = tl.where(scores == 0, 0.0, scores)
    if BF16:
        bits = scores.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        ordered = tl.where((bits >> 15) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
        return (ordered << 16) | (1023 - experts).to(tl.uint32)
    else:
        bits = scores.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000)
        return (ordered.to(tl.uint64) << 32) | (1023 - experts).to(tl.uint64)


@triton.jit
def _key_index(keys):
    return 1023 - (keys & 1023).to(tl.int32)


@triton.jit
def _route(
    logits,
    Bias,
    ImageBias,
    ImageMask,
    FixedIds,
    FixedMask,
    Weights,
    Indices,
    rows,
    T,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TOPK: tl.constexpr,
    scale,
    SCORING: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    FIXED_STRIDE: tl.constexpr,
):
    e = tl.arange(0, BN)
    if ROUND_LOGITS:
        logits = logits.to(tl.bfloat16).to(tl.float32)
    if SCORING == "sigmoid":
        scores = tl.sigmoid(logits)
    elif SCORING == "sqrtsoftplus":
        exp = tl.exp(-tl.abs(logits))
        # Preserve small positive scores where 1 + exp would round to one.
        softplus = tl.maximum(logits, 0.0) + tl.where(
            exp < 1.0e-3, exp * (1.0 - 0.5 * exp), tl.log(1.0 + exp)
        )
        scores = tl.sqrt(softplus)
    else:
        # Softmax is monotonic. With renormalization only the selected
        # logits need exponentiation, and the full denominator cancels.
        scores = logits

    slots = tl.arange(0, triton.next_power_of_2(TOPK))
    active = (rows[:, None] < T) & (slots[None, :] < TOPK)
    if FixedIds is not None:
        fixed = tl.load(
            FixedIds + rows[:, None] * FIXED_STRIDE + slots[None, :], active, other=0
        )
        fixed = tl.where((fixed >= 0) & (fixed < E), fixed, -1).to(tl.int32)

    if FixedIds is not None and FixedMask is None:
        ids = fixed
    else:
        ranking = scores
        if Bias is not None:
            bias = tl.load(Bias + e, e < E, other=0)
        else:
            bias = tl.full((BN,), 0, tl.float32)
        if ImageBias is not None:
            image = tl.load(ImageMask + rows, rows < T, other=False)
            image_bias = tl.load(ImageBias + e, e < E, other=0)
            ranking = ranking + tl.where(image[:, None], image_bias[None, :], bias)
        elif Bias is not None:
            ranking = ranking + bias[None, :]
        # Padded graph rows may contain non-finite values. Keep the selection
        # bounded even for such rows, without changing the returned scores.
        ranking = tl.where(ranking == ranking, ranking, -float("inf"))
        eligible = e[None, :] < E
        if GROUPS > TOP_GROUPS:
            width: tl.constexpr = E // GROUPS
            grouped = tl.reshape(ranking, (BM, GROUPS, width))
            if Bias is not None:
                group_max = tl.max(grouped, 2)
                group_lane = tl.arange(0, width)
                first = tl.min(
                    tl.where(grouped == group_max[:, :, None], group_lane, width), 2
                )
                second = tl.max(
                    tl.where(group_lane == first[:, :, None], -float("inf"), grouped), 2
                )
                group_scores = group_max + second
            else:
                group_scores = tl.max(grouped, 2)
            group_scores = tl.where(
                group_scores == group_scores, group_scores, -float("inf")
            )
            group = tl.arange(0, GROUPS)
            group_keys = tl.topk(
                _rank_key(group_scores, group[None, :]),
                triton.next_power_of_2(TOP_GROUPS),
            )
            groups = _key_index(group_keys)
            group_slots = tl.arange(0, triton.next_power_of_2(TOP_GROUPS))
            group_bits = tl.sum(
                tl.where(group_slots[None, :] < TOP_GROUPS, 1 << groups, 0), 1
            )
            eligible = eligible & (((group_bits[:, None] >> (e // width)) & 1) != 0)
        keys = tl.where(
            eligible,
            _rank_key(ranking, e[None, :], SCORING == "softmax" and ROUND_LOGITS),
            0,
        )
        ids = _key_index(tl.topk(keys, triton.next_power_of_2(TOPK)))
        if FixedIds is not None:
            use_fixed = tl.load(FixedMask + rows, rows < T, other=False)
            ids = tl.where(use_fixed[:, None], fixed, ids)

    # The unused slots in a non-power-of-two top-k need an in-range gather
    # index. A fixed expert ID outside [0, E) yields zero weight and ID -1.
    valid = (ids >= 0) & (ids < E) & (slots[None, :] < TOPK)
    selected = tl.gather(scores, tl.where(valid, ids, 0), axis=1)
    if SCORING == "softmax":
        if NORMALIZE:
            maximum = tl.max(tl.where(valid, selected, -float("inf")), 1)
        else:
            maximum = tl.max(tl.where(e[None, :] < E, logits, -float("inf")), 1)
        selected = tl.exp(selected - maximum[:, None])
    selected = tl.where(valid, selected, 0.0)
    if NORMALIZE:
        denom = tl.sum(selected, 1)
        selected = selected / denom[:, None]
    elif SCORING == "softmax":
        denom = tl.sum(
            tl.where(e[None, :] < E, tl.exp(logits - maximum[:, None]), 0.0), 1
        )
        selected = selected / denom[:, None]
    offsets = rows[:, None] * TOPK + slots[None, :]
    tl.store(Weights + offsets, tl.where(valid, selected * scale, 0.0), active)
    tl.store(Indices + offsets, tl.where(valid, ids, -1), active)


@triton.jit
def _project_tile(
    X,
    W,
    row,
    col,
    split,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    ASYNC: tl.constexpr,
    STAGES: tl.constexpr,
):
    split_hidden: tl.constexpr = triton.cdiv(H, SPLITS * BK) * BK
    start = split * split_hidden
    if ASYNC:
        x_tile = tl.make_block_ptr(X, (T, H), (H, 1), (row, start), (BM, BK), (1, 0))
        w_tile = tl.make_block_ptr(W, (H, E), (1, H), (start, col), (BK, BN), (0, 1))
    else:
        rows = row + tl.arange(0, BM)
        experts = col + tl.arange(0, BN)
        k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in tl.range(0, split_hidden, BK, num_stages=STAGES):
        if ASYNC:
            x = tle.load(
                x_tile, boundary_check=(0,), padding_option="zero", is_async=True
            )
            w = tle.load(
                w_tile, boundary_check=(1,), padding_option="zero", is_async=True
            )
            x_tile = tl.advance(x_tile, (0, BK))
            w_tile = tl.advance(w_tile, (BK, 0))
        else:
            hidden = start + tile + k
            x = tl.load(
                X + rows[:, None] * H + hidden,
                (rows[:, None] < T) & (hidden < H),
                other=0,
            )
            w = tl.load(
                W + experts[None, :] * H + hidden[:, None],
                (experts[None, :] < E) & (hidden[:, None] < H),
                other=0,
            )
        acc = tl.dot(x, w, acc)
    return acc


@triton.jit
def _complete_routes(
    Partial,
    Bias,
    ImageBias,
    ImageMask,
    FixedIds,
    FixedMask,
    Weights,
    Indices,
    rows,
    T,
    scale,
    E: tl.constexpr,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    FIXED_STRIDE: tl.constexpr,
):
    block_experts: tl.constexpr = triton.next_power_of_2(E)
    experts = tl.arange(0, block_experts)
    mask = (rows[:, None] < T) & (experts[None, :] < E)
    offsets = rows[:, None] * E + experts
    if SPLITS == 1:
        logits = tl.load(Partial + offsets, mask, other=0).to(tl.float32)
    else:
        part = tl.arange(0, SPLITS)
        logits = tl.sum(
            tl.load(
                Partial + part[:, None, None] * (T * E) + offsets[None, :, :],
                mask[None, :, :],
                other=0,
            ),
            0,
        )
    _route(
        logits,
        Bias,
        ImageBias,
        ImageMask,
        FixedIds,
        FixedMask,
        Weights,
        Indices,
        rows,
        T,
        E,
        ROWS,
        block_experts,
        TOPK,
        scale,
        SCORING,
        ROUND_LOGITS,
        NORMALIZE,
        GROUPS,
        TOP_GROUPS,
        FIXED_STRIDE,
    )


@triton.jit(do_not_specialize=["scale"])
def _megagate_project(
    X,
    W,
    Partial,
    Arrivals,
    Bias,
    ImageBias,
    ImageMask,
    FixedIds,
    FixedMask,
    Weights,
    Indices,
    T,
    scale,
    H: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    ASYNC: tl.constexpr,
    STAGES: tl.constexpr,
    ROUTE_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    FIXED_STRIDE: tl.constexpr,
):
    tiles_n: tl.constexpr = triton.cdiv(E, BN)
    contributors: tl.constexpr = SPLITS * tiles_n
    if Arrivals is not None:
        group = tl.program_id(0) // contributors
        job = tl.program_id(0) % contributors
        split = job // tiles_n
        tile_n = job % tiles_n
    else:
        group = tl.program_id(0)
        tile_n = tl.program_id(1)
        split = tl.program_id(2)
    row = group * BM
    col = tile_n * BN
    acc = _project_tile(
        X, W, row, col, split, T, H, E, BM, BN, BK, SPLITS, ASYNC, STAGES
    )
    m = row + tl.arange(0, BM)
    n = col + tl.arange(0, BN)
    tl.store(
        Partial + (split * T + m[:, None]) * E + n,
        acc,
        (m[:, None] < T) & (n < E),
    )
    if Arrivals is not None:
        # Finish every lane's stores before publishing the tile. The RMWs
        # form an acquire/release chain. The last CTA observes all tiles.
        tl.debug_barrier()
        count = tl.atomic_add(Arrivals + group, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if count == contributors - 1:
            for subrow in range(tl.cdiv(tl.minimum(T - row, BM), ROUTE_ROWS)):
                rows = row + subrow * ROUTE_ROWS + tl.arange(0, ROUTE_ROWS)
                _complete_routes(
                    Partial,
                    Bias,
                    ImageBias,
                    ImageMask,
                    FixedIds,
                    FixedMask,
                    Weights,
                    Indices,
                    rows,
                    T,
                    scale,
                    E,
                    ROUTE_ROWS,
                    SPLITS,
                    TOPK,
                    SCORING,
                    ROUND_LOGITS,
                    NORMALIZE,
                    GROUPS,
                    TOP_GROUPS,
                    FIXED_STRIDE,
                )


@triton.jit(do_not_specialize=["scale"])
def _megagate_route(
    Partial,
    Bias,
    ImageBias,
    ImageMask,
    FixedIds,
    FixedMask,
    Weights,
    Indices,
    T,
    scale,
    E: tl.constexpr,
    ROUTE_ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    ROUND_LOGITS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    FIXED_STRIDE: tl.constexpr,
):
    rows = tl.program_id(0) * ROUTE_ROWS + tl.arange(0, ROUTE_ROWS)
    _complete_routes(
        Partial,
        Bias,
        ImageBias,
        ImageMask,
        FixedIds,
        FixedMask,
        Weights,
        Indices,
        rows,
        T,
        scale,
        E,
        ROUTE_ROWS,
        SPLITS,
        TOPK,
        SCORING,
        ROUND_LOGITS,
        NORMALIZE,
        GROUPS,
        TOP_GROUPS,
        FIXED_STRIDE,
    )


@lru_cache(maxsize=1)
def _load_memset():
    try:
        memset = ctypes.CDLL("libhggc.so").hgMemsetD8Async
    except (AttributeError, OSError):
        return None
    memset.argtypes = (
        ctypes.c_uint64,
        ctypes.c_ubyte,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    memset.restype = ctypes.c_int
    return memset


def _projection_config(tokens, hidden, experts, multiprocessors, async_load, fused):
    if fused:
        bm, bn, bk, warps, stages = 16, 32, 64, 4, 2
    elif tokens >= 8 * multiprocessors and async_load and hidden % 128 == 0:
        bm, bn, bk, warps, stages = 128, 128, 128, 8, 3
    else:
        bm, bn, bk, warps, stages = 32, 64, 64, 4, 3
    tiles = triton.cdiv(tokens, bm) * triton.cdiv(experts, bn)
    split_k = min(
        16 if fused else 4, triton.next_power_of_2(triton.cdiv(multiprocessors, tiles))
    )
    # Very short K slices are dominated by the routing reduction and stores.
    while split_k > 1 and (hidden // split_k < 2 * bk or hidden % (split_k * bk)):
        split_k //= 2
    return bm, bn, bk, warps, stages, split_k


def _check_metadata(name, tensor, shape, dtype, device, *, row_strided=False):
    if tensor is None:
        return
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or tensor.layout != torch.strided
        or not (tensor.stride(-1) == 1 if row_strided else tensor.is_contiguous())
    ):
        raise ValueError(f"Invalid shape, dtype, device, or layout for {name}")


def megagate(
    x,
    weight,
    num_topk,
    *,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    bias=None,
    image_bias=None,
    image_token_mask=None,
    fix_routing_mask=None,
    unmapped_topk_idx=None,
    num_expert_group=1,
    topk_group=1,
    renormalize=True,
    router_logits_dtype=torch.float32,
):
    """Compute a BF16 router projection and expert selection on T-Head PPU.

    Return FP32 routing weights and int64 expert IDs, both of shape
    [tokens, num_topk]. Bias affects selection only. Image rows use
    image_bias in place of bias. Fixed rows preserve the supplied expert
    order, with scores gathered before normalization. If the fixed IDs
    are supplied alone, all rows use them. Equal scores prefer lower IDs.
    Invalid fixed IDs return -1 with weight zero.

    X and weight are contiguous [T, H] and [E, H] BF16 tensors.
    Accumulation is FP32. Set router_logits_dtype to BF16 for models with
    a BF16 GateLinear output. Output tensors and scratch are per call,
    including during CUDA Graph capture.
    """
    if not x.is_cuda:
        raise NotImplementedError("MegaGate requires T-Head PPU tensors")
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.layout != torch.strided
        or weight.layout != torch.strided
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.device != weight.device
    ):
        raise ValueError("Expected contiguous [T, H] and [E, H] on the same device")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise NotImplementedError("MegaGate requires BF16 inputs and weights")
    if router_logits_dtype not in (torch.float32, torch.bfloat16):
        raise NotImplementedError("Router logits must be FP32 or BF16")
    if scoring_func not in ("sqrtsoftplus", "sigmoid", "softmax", "identity"):
        raise NotImplementedError(f"Unsupported scoring function: {scoring_func}")

    tokens, hidden = x.shape
    experts, weight_hidden = weight.shape
    if hidden != weight_hidden or hidden <= 0 or not 1 <= experts <= 512:
        raise ValueError("Expected matching positive H and 1 to 512 experts")
    if max(x.numel(), weight.numel(), tokens * experts) >= 2**31:
        raise ValueError("MegaGate tensors exceed the 32-bit index range")
    if not isinstance(num_topk, int) or not 1 <= num_topk <= min(32, experts):
        raise ValueError("num_topk must be between 1 and min(32, experts)")
    if (
        not math.isfinite(routed_scaling_factor)
        or abs(routed_scaling_factor) > torch.finfo(torch.float32).max
    ):
        raise ValueError("routed_scaling_factor must be finite")
    if (image_bias is None) != (image_token_mask is None):
        raise ValueError("image_bias and image_token_mask must be supplied together")
    if fix_routing_mask is not None and unmapped_topk_idx is None:
        raise ValueError("fix_routing_mask requires unmapped_topk_idx")

    if (
        not isinstance(num_expert_group, int)
        or not isinstance(topk_group, int)
        or not 1 <= topk_group <= num_expert_group <= 32
        or experts % num_expert_group
        or num_topk > topk_group * (experts // num_expert_group)
    ):
        raise ValueError("Invalid expert grouping or top-k")
    if image_bias is not None and num_expert_group != 1:
        raise NotImplementedError("Image routing requires one expert group")
    if num_expert_group > topk_group:
        if experts & (experts - 1) or num_expert_group & (num_expert_group - 1):
            raise NotImplementedError("Grouped routing requires power-of-two tiles")
        if experts == num_expert_group and bias is not None:
            raise ValueError("Biased grouped routing requires two experts per group")
    if scoring_func == "softmax" and (
        bias is not None or image_bias is not None or num_expert_group != 1
    ):
        raise NotImplementedError("Softmax routing supports one group without bias")

    device = x.device
    _check_metadata("bias", bias, (experts,), torch.float32, device)
    _check_metadata("image_bias", image_bias, (experts,), torch.float32, device)
    _check_metadata("image_token_mask", image_token_mask, (tokens,), torch.bool, device)
    _check_metadata("fix_routing_mask", fix_routing_mask, (tokens,), torch.bool, device)
    _check_metadata(
        "unmapped_topk_idx",
        unmapped_topk_idx,
        (tokens, num_topk),
        torch.int64,
        device,
        row_strided=True,
    )
    if unmapped_topk_idx is not None and (
        tokens * unmapped_topk_idx.stride(0) >= 2**31
    ):
        raise ValueError("unmapped_topk_idx exceeds the 32-bit index range")

    weights = torch.empty((tokens, num_topk), dtype=torch.float32, device=device)
    indices = torch.empty((tokens, num_topk), dtype=torch.int64, device=device)
    if tokens == 0:
        return weights, indices

    multiprocessors = torch_device_fn.get_device_properties(
        device
    ).multi_processor_count
    async_load = (
        _HAS_ASYNC_LOAD
        and hidden % 64 == 0
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )
    # Keep the last-CTA epilogue small. Wide, batched decode fills more
    # projection tiles and benefits from independently scheduled routing.
    fused = tokens <= 32 and tokens * triton.next_power_of_2(experts) <= 4096
    memset = _load_memset() if fused else None
    bm, bn, bk, warps, stages, splits = _projection_config(
        tokens, hidden, experts, multiprocessors, async_load, memset is not None
    )
    bm_tiles = triton.cdiv(tokens, bm)
    bn_tiles = triton.cdiv(experts, bn)
    partial_dtype = router_logits_dtype if splits == 1 else torch.float32
    partial = torch.empty((splits, tokens, experts), device=device, dtype=partial_dtype)
    arrivals = None
    route_rows = min(4, triton.next_power_of_2(tokens)) if memset is not None else 1
    router_ptrs = (
        bias,
        image_bias,
        image_token_mask,
        unmapped_topk_idx,
        fix_routing_mask,
        weights,
        indices,
    )
    router_params = (
        num_topk,
        scoring_func,
        router_logits_dtype == torch.bfloat16,
        renormalize,
        num_expert_group,
        topk_group,
        0 if unmapped_topk_idx is None else unmapped_topk_idx.stride(0),
    )
    with torch_device_fn.device(device):
        if memset is not None:
            arrivals = torch.empty((bm_tiles,), device=device, dtype=torch.int32)
            stream = torch_device_fn.current_stream(device)
            status = memset(
                arrivals.data_ptr(),
                0,
                bm_tiles * arrivals.element_size(),
                stream.cuda_stream,
            )
            if status:
                raise RuntimeError(
                    f"MegaGate counter initialization failed with status {status}"
                )
            grid = (bm_tiles * bn_tiles * splits,)
        else:
            grid = (bm_tiles, bn_tiles, splits)
        _megagate_project[grid](
            x,
            weight,
            partial,
            arrivals,
            *router_ptrs,
            tokens,
            routed_scaling_factor,
            hidden,
            experts,
            bm,
            bn,
            bk,
            splits,
            async_load,
            stages,
            route_rows,
            *router_params,
            num_warps=warps,
            num_stages=stages,
        )
        if arrivals is None:
            _megagate_route[(triton.cdiv(tokens, route_rows),)](
                partial,
                *router_ptrs,
                tokens,
                routed_scaling_factor,
                experts,
                route_rows,
                splits,
                *router_params,
                num_warps=1,
            )
    return weights, indices
