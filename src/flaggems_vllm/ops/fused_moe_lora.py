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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)
_VENDOR = runtime.device.vendor_name
_IS_NVIDIA = _VENDOR == "nvidia"
_THEAD_CHECK_FUSED_RANK = tl.constexpr(_VENDOR == "thead")
_METAX_IEEE_BF16_DOT = tl.constexpr(_VENDOR == "metax")
_GROUP_SLICES = _VENDOR in ("nvidia", "metax", "hygon", "thead", "mthreads")
_WIDE_PREFILL = _VENDOR in ("nvidia", "metax", "hygon", "thead", "mthreads")


def _tune(name, key, prune=None):
    if _VENDOR in ("hygon", "thead"):
        # HCU's JIT already specializes buffer accesses by their address range.
        # Launching the fixed configuration directly avoids a second lookup.
        return lambda kernel: kernel
    if not _IS_NVIDIA:
        return libentry()

    def decorate(kernel):
        return libentry()(
            libtuner(
                configs=runtime.get_tuned_config(name),
                key=key,
                prune_configs_by={"early_config_prune": prune} if prune else None,
                restore_value=["Y"],
                warmup=2,
                rep=5,
                use_cuda_graph=False,
            )(kernel)
        )

    return decorate


# Vendor JITs run the same kernels with one explicit launch configuration.
# Only the NVIDIA path samples a tuning space and invokes a benchmark timer.
_GEMV_LAUNCH = (
    {} if _IS_NVIDIA else dict(BLOCK_K=64, BLOCK_N=64, num_warps=4, num_stages=1)
)
_GROUPED_LAUNCH = (
    {} if _IS_NVIDIA else dict(BLOCK_K=32, BLOCK_N=64, num_warps=4, num_stages=1)
)
_PREFILL_LAUNCH = _GROUPED_LAUNCH
if _VENDOR in ("metax", "hygon"):
    _PREFILL_LAUNCH = dict(_GROUPED_LAUNCH, BLOCK_K=64)
elif _VENDOR == "thead":
    _PREFILL_LAUNCH = dict(BLOCK_K=128, BLOCK_N=128, num_warps=4, num_stages=3)
_THEAD_RANK_GROUPED = dict(BLOCK_K=64, BLOCK_N=64, num_warps=4, num_stages=2)
_METAX_RANK_GEMV = dict(BLOCK_K=64, BLOCK_N=32, num_warps=4, num_stages=1)
_PROJECT_LAUNCH = (
    {} if _IS_NVIDIA else dict(BLOCK_K=32, BLOCK_N=32, num_warps=4, num_stages=1)
)


@triton.jit
def _write_rows(
    Y,
    Weights,
    pairs,
    rows_valid,
    cols,
    value,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    YS: tl.constexpr,
    WS: tl.constexpr,
    WEIGHTED: tl.constexpr,
    ADD: tl.constexpr,
):
    token = pairs // TOP_K
    choice = pairs % TOP_K
    if WEIGHTED:
        scale = tl.load(
            Weights + token * WS[0] + choice * WS[1], mask=rows_valid, other=0
        ).to(tl.float32)
        value *= scale[:, None]
    y = Y + (token * YS[0] + choice * YS[1])[:, None] + cols[None, :] * YS[2]
    mask = rows_valid[:, None] & (cols[None, :] < N)
    delta = value.to(Y.dtype.element_ty)
    if ADD:
        delta += tl.load(y, mask, other=0)
    tl.store(y, delta, mask)


@triton.jit
def _slice_ptrs(A, B, Y, A1, B1, PACK: tl.constexpr, N: tl.constexpr, YS: tl.constexpr):
    if PACK == 2:
        index = tl.program_id(2) % 2
        A = tl.where(index == 0, A, A1)
        B = tl.where(index == 0, B, B1)
        Y += index.to(tl.int64) * N * YS
    return A, B, Y


@_tune(
    "fused_moe_lora_gemv",
    ["PAIRS", "K", "R", "N", "SLICES", "PACK", "ADD", "WEIGHTED", "FP32_RANK"],
)
@triton.jit
def _gemv(
    X,
    A,
    B,
    Y,
    A1,
    B1,
    Weights,
    Mapping,
    Experts,
    Enabled,
    PAIRS: tl.constexpr,
    TOP_K: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    SLICES: tl.constexpr,
    PACK: tl.constexpr,
    XS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    YS: tl.constexpr,
    WS: tl.constexpr,
    MS: tl.constexpr,
    ES: tl.constexpr,
    ENABLED_S: tl.constexpr,
    ADD: tl.constexpr,
    WEIGHTED: tl.constexpr,
    FP32_RANK: tl.constexpr,
    CHUNK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    A, B, Y = _slice_ptrs(A, B, Y, A1, B1, PACK, N, YS[2])
    pair = tl.program_id(0).to(tl.int64)
    token = pair // TOP_K
    adapter = tl.load(Mapping + token * MS).to(tl.int64)
    if (adapter < 0) | (adapter >= L):
        return
    expert = tl.load(Experts + pair * ES).to(tl.int64)
    if (expert < 0) | (expert >= E) | (tl.load(Enabled + adapter * ENABLED_S) == 0):
        return
    A += adapter * AS[0] + expert * AS[1]
    B += adapter * BS[0] + expert * BS[1]
    xrow = pair if WEIGHTED else token
    X += xrow * XS[0]
    r = tl.arange(0, BLOCK_R).to(tl.int64)
    k = tl.arange(0, BLOCK_K).to(tl.int64)
    hidden = tl.full((BLOCK_R,), 0, tl.float32)
    for start in range(0, K, BLOCK_K):
        ks = k + start
        x = tl.load(X + ks * XS[1], ks < K, other=0).to(tl.float32)
        a = tl.load(
            A + r[:, None] * AS[2] + ks[None, :] * AS[3],
            (r[:, None] < R) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        hidden += tl.sum(a * x[None, :], axis=1)
    if BLOCK_R > R:
        # Nonfinite activations times a padded zero weight can produce NaN.
        hidden = tl.where(r < R, hidden, 0.0)
    if not FP32_RANK:
        hidden = hidden.to(Y.dtype.element_ty).to(tl.float32)
    choice = pair % TOP_K
    if WEIGHTED:
        scale = tl.load(Weights + token * WS[0] + choice * WS[1]).to(tl.float32)
    yrow = Y + token * YS[0] + choice * YS[1]
    start_n = tl.program_id(1).to(tl.int64) * CHUNK_N
    for tile in range(tl.cdiv(CHUNK_N, BLOCK_N)):
        cols = start_n + tile * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = (cols < N) & (cols < start_n + CHUNK_N)
        b = tl.load(
            B + cols[:, None] * BS[2] + r[None, :] * BS[3],
            mask[:, None] & (r[None, :] < R),
            other=0,
        ).to(tl.float32)
        value = tl.sum(b * hidden[None, :], axis=1)
        if WEIGHTED:
            value *= scale
        if not FP32_RANK:
            value = value.to(Y.dtype.element_ty).to(tl.float32)
        if ADD:
            value += tl.load(yrow + cols * YS[2], mask, other=0).to(tl.float32)
        tl.store(yrow + cols * YS[2], value, mask)


@triton.jit
def _resolve_routes(
    Sorted,
    Experts,
    Post,
    Mapping,
    Ids,
    Active,
    Enabled,
    PAIRS: tl.constexpr,
    TOP_K: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    SORTED: tl.constexpr,
    RAW: tl.constexpr,
    META_M: tl.constexpr,
    BM: tl.constexpr,
    RS: tl.constexpr,
    DEVICE_COUNT: tl.constexpr,
    PACK: tl.constexpr = 1,
):
    row_group = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BM)
    if SORTED:
        slot = tl.program_id(2).to(tl.int64) // PACK
        active = True
        if DEVICE_COUNT:
            active = slot < tl.load(Active)
        adapter = tl.load(Ids + slot * RS[5], mask=active, other=-1).to(tl.int64)
        block = row_group // tl.cdiv(META_M, BM)
        pos = block * META_M + row_group % tl.cdiv(META_M, BM) * BM + lanes
        in_adapter = (adapter >= 0) & (adapter < L)
        count = tl.load(Post + adapter * RS[3], mask=in_adapter, other=0)
        valid = in_adapter & (block * META_M < count)
        expert = tl.load(
            Experts + adapter * RS[1] + block * RS[2], mask=valid, other=-1
        ).to(tl.int64)
        pairs = tl.load(
            Sorted + adapter * RS[0] + pos * RS[7],
            (pos < RAW) & (pos < count) & (pos < (block + 1) * META_M) & valid,
            other=-1,
        ).to(tl.int64)
    else:
        adapter = tl.load(Mapping + row_group // TOP_K * RS[4]).to(tl.int64)
        expert = tl.load(Experts + row_group * RS[2]).to(tl.int64)
        pairs = tl.where(lanes == 0, row_group, -1)
    in_adapter = (adapter >= 0) & (adapter < L)
    enabled = tl.load(Enabled + adapter * RS[6], mask=in_adapter, other=0)
    live = (enabled != 0) & (expert >= 0) & (expert < E)
    valid_pairs = (pairs >= 0) & (pairs < PAIRS)
    return adapter, expert, tl.where(valid_pairs, pairs, 0), valid_pairs, live


@triton.jit
def _lora_dot(a, b, acc=None):
    # C550's BF16 matrix instruction leaves a residual for exact cancellation.
    # Its IEEE FP32 matrix instruction accepts the same BF16 values losslessly.
    if _METAX_IEEE_BF16_DOT and a.dtype == tl.bfloat16:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    return tl.dot(a, b, acc)


@triton.jit
def _dot_rows(
    X,
    W,
    rows,
    valid_rows,
    cols,
    K: tl.constexpr,
    N: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    k = tl.arange(0, BK).to(tl.int64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, K, BK):
        ks = start + k
        x = tl.load(
            X + rows[:, None] * XS[0] + ks[None, :] * XS[1],
            valid_rows[:, None] & (ks[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + ks[:, None] * WS[3] + cols[None, :] * WS[2],
            (ks[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = _lora_dot(x, w, acc)
    return acc


@triton.jit
def _expand_nonfinite_rank(hidden, b):
    # PPU's register-to-register MMA chain can turn a valid infinity into NaN.
    # Accumulate one outer product at a time to bound register use at large rank.
    rows: tl.constexpr = hidden.shape[0]
    cols: tl.constexpr = b.shape[1]
    acc = tl.full((rows, cols), 0, tl.float32)
    for i in range(hidden.shape[1]):
        a_col = tl.gather(hidden, tl.full((rows, 1), i, tl.int32), 1).to(tl.float32)
        b_row = tl.gather(b, tl.full((1, cols), i, tl.int32), 0).to(tl.float32)
        acc += a_col * b_row
    return acc


def _prune_grouped(configs, named_args, **kwargs):
    if named_args["WIDE_PREFILL"]:
        return configs
    return [config for config in configs if config.kwargs["BLOCK_N"] == 64]


@_tune(
    "fused_moe_lora_grouped",
    [
        "PAIRS",
        "K",
        "R",
        "N",
        "SLICES",
        "PACK",
        "ADD",
        "WEIGHTED",
        "META_M",
        "WIDE_PREFILL",
    ],
    prune=_prune_grouped,
)
@triton.jit
def _grouped(
    X,
    A,
    B,
    Y,
    A1,
    B1,
    Weights,
    Sorted,
    Experts,
    Post,
    Mapping,
    Ids,
    Active,
    Enabled,
    PAIRS: tl.constexpr,
    TOP_K: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    SLICES: tl.constexpr,
    PACK: tl.constexpr,
    XS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    YS: tl.constexpr,
    WS: tl.constexpr,
    RS: tl.constexpr,
    DEVICE_COUNT: tl.constexpr,
    RAW: tl.constexpr,
    META_M: tl.constexpr,
    BM: tl.constexpr,
    WEIGHTED: tl.constexpr,
    ADD: tl.constexpr,
    BR: tl.constexpr,
    WIDE_PREFILL: tl.constexpr,
    CHUNK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    A, B, Y = _slice_ptrs(A, B, Y, A1, B1, PACK, N, YS[2])
    adapter, expert, pairs, valid, live = _resolve_routes(
        Sorted,
        Experts,
        Post,
        Mapping,
        Ids,
        Active,
        Enabled,
        PAIRS,
        TOP_K,
        L,
        E,
        True,
        RAW,
        META_M,
        BM,
        RS,
        DEVICE_COUNT,
        PACK,
    )
    if not live:
        return
    rows = pairs if WEIGHTED else pairs // TOP_K
    r = tl.arange(0, BR).to(tl.int64)
    hidden = _dot_rows(
        X,
        A + adapter * AS[0] + expert * AS[1],
        rows,
        valid,
        r,
        K,
        R,
        XS,
        AS,
        BM,
        BR,
        BLOCK_K,
    )
    if BR > R:
        hidden = tl.where(r[None, :] < R, hidden, 0.0)
    hidden = hidden.to(Y.dtype.element_ty)
    if _THEAD_CHECK_FUSED_RANK:
        finite = tl.abs(hidden.to(tl.float32)) < float("inf")
        special = tl.max((~finite).to(tl.int32)) != 0
    B += adapter * BS[0] + expert * BS[1]
    start_n = tl.program_id(1).to(tl.int64) * CHUNK_N
    for tile in range(tl.cdiv(CHUNK_N, BLOCK_N)):
        cols = start_n + tile * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        b = tl.load(
            B + r[:, None] * BS[3] + cols[None, :] * BS[2],
            (r[:, None] < R) & (cols[None, :] < N),
            other=0,
        )
        if _THEAD_CHECK_FUSED_RANK:
            if special:
                value = _expand_nonfinite_rank(hidden, b)
            else:
                value = _lora_dot(hidden, b)
        else:
            value = _lora_dot(hidden, b)
        _write_rows(
            Y, Weights, pairs, valid, cols, value, TOP_K, N, YS, WS, WEIGHTED, ADD
        )


@_tune(
    "fused_moe_lora_project",
    [
        "PAIRS",
        "K",
        "N",
        "SLICES",
        "PACK",
        "SORTED",
        "META_M",
        "FACTOR",
        "ADD",
        "WEIGHTED",
        "SHARDED",
    ],
)
@triton.jit
def _project(
    X,
    W,
    Y,
    W1,
    Weights,
    Sorted,
    Experts,
    Post,
    Mapping,
    Ids,
    Active,
    Enabled,
    PAIRS: tl.constexpr,
    TOP_K: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    E: tl.constexpr,
    SLICES: tl.constexpr,
    PACK: tl.constexpr,
    SLICE_X: tl.constexpr,
    SLICE_Y: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    YS: tl.constexpr,
    ROUTER_S: tl.constexpr,
    RS: tl.constexpr,
    DEVICE_COUNT: tl.constexpr,
    RAW: tl.constexpr,
    META_M: tl.constexpr,
    BM: tl.constexpr,
    SORTED: tl.constexpr,
    FACTOR: tl.constexpr,
    WEIGHTED: tl.constexpr,
    ADD: tl.constexpr,
    SHARDED: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if PACK == 2:
        index = (tl.program_id(2) % 2).to(tl.int64)
        W = tl.where(index == 0, W, W1)
        X += index * SLICE_X
        Y += index * SLICE_Y
    adapter, expert, pairs, valid, live = _resolve_routes(
        Sorted,
        Experts,
        Post,
        Mapping,
        Ids,
        Active,
        Enabled,
        PAIRS,
        TOP_K,
        L,
        E,
        SORTED,
        RAW,
        META_M,
        BM,
        RS,
        DEVICE_COUNT,
        PACK,
    )
    if not live:
        return
    cols = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    value = _dot_rows(
        X,
        W + adapter * WS[0] + expert * WS[1],
        pairs // FACTOR,
        valid,
        cols,
        K,
        N,
        XS,
        WS,
        BM,
        BLOCK_N,
        BLOCK_K,
    )
    _write_rows(
        Y, Weights, pairs, valid, cols, value, TOP_K, N, YS, ROUTER_S, WEIGHTED, ADD
    )


def _check_route_tensors(tensors, device):
    for tensor in tensors:
        if tensor.device != device or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "route indices must be int32 or int64 on the output device"
            )


def _tp_exchange(cache, full_rank):
    from vllm.distributed import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    if cache.shape[-1] == full_rank:
        return tensor_model_parallel_all_reduce(cache)
    gathered = tensor_model_parallel_all_gather(cache, dim=-1)
    if gathered.shape[-1] != full_rank:
        raise ValueError("gathered LoRA rank does not match the B matrices")
    return gathered


def fused_moe_lora(
    output: torch.Tensor,
    qcurr_hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor | None,
    token_lora_mapping: torch.Tensor,
    max_lora_rank: int,
    top_k_num: int,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    adapter_enabled: torch.Tensor,
    shrink_block_size_m: int,
    shrink_block_size_n: int,
    shrink_block_size_k: int,
    shrink_group_size_m: int,
    shrink_num_warps: int,
    shrink_num_stages: int,
    shrink_split_k: int,
    expand_block_size_m: int,
    expand_block_size_n: int,
    expand_block_size_k: int,
    expand_group_size_m: int,
    expand_num_warps: int,
    expand_num_stages: int,
    expand_split_k: int,
    mul_routed_weight: bool = False,
    fully_sharded: bool = False,
    offset: int = 0,
    add_inputs: bool = True,
) -> None:
    """Apply expert- and adapter-specific LoRA projections to the routed pairs."""
    logger.debug("GEMS FUSED_MOE_LORA")
    device = output.device
    if device.type in ("cpu", "meta"):
        raise NotImplementedError("fused_moe_lora requires an accelerator device")
    if fully_sharded and not add_inputs:
        raise ValueError("fully_sharded requires add_inputs=True")
    x = qcurr_hidden_states
    dtype = output.dtype
    if dtype not in (torch.float16, torch.bfloat16) or x.dtype != dtype:
        raise ValueError("input, output and LoRA weights must share FP16 or BF16 dtype")
    if x.device != device or topk_weights.device != device:
        raise ValueError(
            "hidden states, routing weights and output must share a device"
        )
    if topk_weights.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("topk_weights must be FP32, BF16 or FP16")
    if topk_weights.ndim != 2 or output.ndim != 3 or x.ndim != 2:
        raise ValueError("expected [T, top_k] weights, [T, top_k, H] output and 2-D X")
    tokens, top_k = topk_weights.shape
    pairs = tokens * top_k
    factor = 1 if mul_routed_weight else top_k
    if top_k <= 0 or top_k_num != top_k or output.shape[:2] != topk_weights.shape:
        raise ValueError("top-k dimensions must agree")
    if x.shape[0] != pairs // factor:
        raise ValueError(
            "weighted input requires one X row per pair, unweighted per token"
        )
    if any(step == 0 and size > 1 for size, step in zip(output.shape, output.stride())):
        raise ValueError(
            "output must have distinct storage for each routed pair and column"
        )
    slices = len(lora_a_stacked)
    if not slices or slices != len(lora_b_stacked):
        raise ValueError("one A and one B tensor must be supplied per slice")
    if lora_a_stacked[0].ndim != 4 or lora_b_stacked[0].ndim != 4:
        raise ValueError("expected 4-D LoRA weight tensors")
    adapters, experts, rank, width = lora_a_stacked[0].shape
    full_rank = lora_b_stacked[0].shape[-1]
    if rank <= 0 or rank != max_lora_rank or width != x.shape[-1]:
        raise ValueError("A must match max_lora_rank and the hidden state width")
    if (
        (not fully_sharded and rank != full_rank)
        or full_rank % rank
        or full_rank < rank
    ):
        raise ValueError("B must contract the local or gathered rank dimension")
    ends_at = offset
    for a, b in zip(lora_a_stacked, lora_b_stacked):
        if (
            a.shape != (adapters, experts, rank, width)
            or b.ndim != 4
            or b.shape[:2] != (adapters, experts)
            or b.shape[-1] != full_rank
        ):
            raise ValueError("incompatible adapter, expert, rank or hidden dimensions")
        if (
            a.device != device
            or b.device != device
            or a.dtype != dtype
            or b.dtype != dtype
        ):
            raise ValueError("LoRA weights must share the output dtype and device")
        ends_at += b.shape[2]
    if offset < 0 or ends_at > output.shape[-1]:
        raise ValueError("LoRA output slice is outside the output tensor")

    _check_route_tensors((expert_ids, token_lora_mapping, lora_ids), device)
    if (
        adapter_enabled.device != device
        or adapter_enabled.ndim != 1
        or adapter_enabled.numel() < adapters
        or adapter_enabled.dtype
        not in (torch.int32, torch.int64, torch.bool, torch.uint8)
    ):
        raise ValueError("expected an adapter_enabled flag per adapter")
    if token_lora_mapping.ndim != 1 or token_lora_mapping.numel() < tokens:
        raise ValueError("token_lora_mapping must cover every token")
    if (
        lora_ids.ndim != 1
        or num_active_loras.numel() != 1
        or num_active_loras.dtype not in (torch.int32, torch.int64)
        or (num_active_loras.device.type != "cpu" and num_active_loras.device != device)
    ):
        raise ValueError("expected a 1-D active adapter list and one active count")
    sorted_route = sorted_token_ids is not None
    device_count = sorted_route and num_active_loras.device.type != "cpu"
    active_count = num_active_loras if device_count else None
    active_groups = 1
    raw = pairs
    block_m = 16
    wide_prefill = False
    group_count = pairs
    meta_m = shrink_block_size_m
    if sorted_route:
        if num_tokens_post_padded is None:
            raise ValueError("sorted_token_ids requires num_tokens_post_padded")
        _check_route_tensors((sorted_token_ids, num_tokens_post_padded), device)
        if meta_m < 16 or meta_m & (meta_m - 1) or meta_m != expand_block_size_m:
            raise ValueError(
                "sorted shrink and expand block_size_m must match and be a power of two >= 16"
            )
        if (
            sorted_token_ids.ndim != 2
            or expert_ids.ndim != 2
            or num_tokens_post_padded.ndim != 1
            or sorted_token_ids.shape[0] != adapters
            or expert_ids.shape[0] != adapters
            or num_tokens_post_padded.numel() != adapters
        ):
            raise ValueError("sorted route tables must provide one row per adapter")
        raw = sorted_token_ids.shape[1]
        blocks = triton.cdiv(raw, meta_m)
        if expert_ids.shape[1] < blocks:
            raise ValueError("expert_ids must provide one expert per metadata block")
        # At prefill sizes the metadata blocks alone provide parallel work.
        # Retain the whole shrink tile across N instead of rereading X and A.
        wide_prefill = (
            _WIDE_PREFILL
            and not fully_sharded
            and rank <= 64
            and pairs >= 1024
            and blocks * slices >= 128
        )
        block_m = min(meta_m, 64 if wide_prefill else 32)
        group_count = blocks * triton.cdiv(meta_m, block_m)
        active_groups = (
            lora_ids.numel() if device_count else int(num_active_loras.item())
        )
        if active_groups < 0 or active_groups > lora_ids.numel():
            raise ValueError("num_active_loras exceeds the list of adapter IDs")
    elif expert_ids.ndim != 1 or expert_ids.numel() < pairs:
        raise ValueError("naive routing requires a 1-D expert ID per pair")

    if (
        not pairs
        or ends_at == offset
        or (not fully_sharded and not group_count * active_groups)
    ):
        return
    if _VENDOR == "ascend":
        from flaggems_vllm.runtime.backend._ascend.ops.fused_moe_lora import (
            _ascend_launch,
        )

        return _ascend_launch(
            output,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mapping,
            lora_ids,
            num_active_loras,
            adapter_enabled,
            meta_m,
            active_groups,
            mul_routed_weight,
            fully_sharded,
            offset,
            add_inputs,
        )
    route = (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        lora_ids,
        active_count,
        adapter_enabled,
    )
    route_strides = (
        sorted_token_ids.stride(0) if sorted_route else 0,
        expert_ids.stride(0) if sorted_route else 0,
        expert_ids.stride(-1),
        num_tokens_post_padded.stride(0) if sorted_route else 0,
        token_lora_mapping.stride(0),
        lora_ids.stride(0),
        adapter_enabled.stride(0),
        sorted_token_ids.stride(1) if sorted_route else 0,
    )
    common = (adapters, experts, slices)
    pack = 1
    a1 = b1 = None
    if (
        _GROUP_SLICES
        and slices == 2
        and lora_a_stacked[0].stride() == lora_a_stacked[1].stride()
        and lora_b_stacked[0].shape == lora_b_stacked[1].shape
        and lora_b_stacked[0].stride() == lora_b_stacked[1].stride()
    ):
        # The two tensors may have unrelated storage. Select their real
        # pointers in the kernel without building a device pointer table.
        pack = 2
        a1, b1 = lora_a_stacked[1], lora_b_stacked[1]
    with runtime.torch_device_fn.device(device):
        # H20 decode can keep a short, pair-local rank tail in registers.
        # Large grouped projections keep using the tiled workspace.
        compact_decode = (
            _IS_NVIDIA
            and not sorted_route
            and pairs <= 32
            and rank <= 256
            and width <= 128
        )
        if not fully_sharded and (rank <= 128 or compact_decode):
            col = offset
            for i in range(0, slices, pack):
                a, b = lora_a_stacked[i], lora_b_stacked[i]
                n = b.shape[2]
                if not n:
                    continue
                stop = col + n * pack
                y = (
                    output
                    if col == 0 and stop == output.shape[2]
                    else output[:, :, col:stop]
                )
                col = stop
                if sorted_route:
                    # Each CTA reuses its shrink tile across a contiguous column range.
                    chunk_n = triton.cdiv(n, 64) * 64 if wide_prefill else 256
                    grouped_launch = (
                        _PREFILL_LAUNCH if wide_prefill else _GROUPED_LAUNCH
                    )
                    if (
                        _VENDOR == "thead"
                        and rank >= 64
                        and n <= 256
                        and not wide_prefill
                    ):
                        # Short projections need only the column tiles that contain data.
                        chunk_n = triton.cdiv(n, 64) * 64
                        grouped_launch = _THEAD_RANK_GROUPED
                    if wide_prefill and _VENDOR == "metax" and dtype == torch.bfloat16:
                        # Keep enough parallel N tiles for C550's FP32 MMA path.
                        chunk_n = min(chunk_n, 512)
                    _grouped[
                        (group_count, triton.cdiv(n, chunk_n), active_groups * pack)
                    ](
                        x,
                        a,
                        b,
                        y,
                        a1,
                        b1,
                        topk_weights,
                        *route,
                        pairs,
                        top_k,
                        width,
                        rank,
                        n,
                        *common,
                        pack,
                        x.stride(),
                        a.stride(),
                        b.stride(),
                        y.stride(),
                        topk_weights.stride(),
                        route_strides,
                        device_count,
                        raw,
                        meta_m,
                        block_m,
                        mul_routed_weight,
                        add_inputs,
                        max(16, triton.next_power_of_2(rank)),
                        wide_prefill,
                        chunk_n,
                        **grouped_launch,
                    )
                else:
                    fp32_rank = rank <= 64 and pairs * rank <= 1024
                    chunk_n = 128 if pairs * slices < 16 else 256
                    gemv_launch = _GEMV_LAUNCH
                    if _VENDOR == "metax" and rank >= 32:
                        # A shorter column range limits the live rank-by-column tile.
                        chunk_n = 128
                        gemv_launch = _METAX_RANK_GEMV
                    _gemv[(pairs, triton.cdiv(n, chunk_n), pack)](
                        x,
                        a,
                        b,
                        y,
                        a1,
                        b1,
                        topk_weights,
                        token_lora_mapping,
                        expert_ids,
                        adapter_enabled,
                        pairs,
                        top_k,
                        width,
                        rank,
                        n,
                        *common,
                        pack,
                        x.stride(),
                        a.stride(),
                        b.stride(),
                        y.stride(),
                        topk_weights.stride(),
                        token_lora_mapping.stride(0),
                        expert_ids.stride(0),
                        adapter_enabled.stride(0),
                        add_inputs,
                        mul_routed_weight,
                        fp32_rank,
                        chunk_n,
                        triton.next_power_of_2(rank),
                        **gemv_launch,
                    )
            return

        # Collectives address the original pairs, even if alignment orders differ on each rank.
        cache_shape = (slices, tokens, top_k, rank)
        cache = (
            torch.zeros(cache_shape, dtype=dtype, device=device)
            if fully_sharded
            else torch.empty(cache_shape, dtype=dtype, device=device)
        )

        def project(
            source, weight, second_weight, target, row_factor, weighted, accumulate
        ):
            n = weight.shape[2]
            if group_count and active_groups and n:

                def grid(meta):
                    return (
                        group_count,
                        triton.cdiv(n, meta["BLOCK_N"]),
                        active_groups * pack,
                    )

                _project[grid](
                    source,
                    weight,
                    target,
                    second_weight,
                    topk_weights,
                    *route,
                    pairs,
                    top_k,
                    weight.shape[3],
                    n,
                    *common,
                    pack,
                    source.stride(0) if source.ndim == 4 else 0,
                    target.stride(0) if target.ndim == 4 else n * target.stride(2),
                    source.stride()[-2:],
                    weight.stride(),
                    target.stride()[-3:],
                    topk_weights.stride(),
                    route_strides,
                    device_count,
                    raw,
                    meta_m,
                    block_m,
                    sorted_route,
                    row_factor,
                    weighted,
                    accumulate,
                    fully_sharded,
                    **_PROJECT_LAUNCH,
                )

        for i in range(0, slices, pack):
            target = cache if pack == 2 else cache[i]
            project(x, lora_a_stacked[i], a1, target, factor, False, False)
        if fully_sharded:
            cache = _tp_exchange(cache, full_rank)
        col = offset
        for i in range(0, slices, pack):
            b = lora_b_stacked[i]
            end = col + b.shape[2] * pack
            source = cache if pack == 2 else cache[i].view(pairs, full_rank)
            target = (
                output if col == 0 and end == output.shape[2] else output[:, :, col:end]
            )
            project(source, b, b1, target, 1, mul_routed_weight, add_inputs)
            col = end
