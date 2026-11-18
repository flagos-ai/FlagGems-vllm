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

"""Gemma 4 routing using integer reductions and partial top-k selection."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

_HAS_PARTIAL_TOPK = hasattr(tl, "topk")


def _prune_token_tiles(configs, named_args, **kwargs):
    # Large token tiles waste lanes in latency-bound decoding batches.
    limit = 4 if named_args["T"] <= 32 else 16
    return [config for config in configs if config.kwargs["BLOCK_T"] <= limit]


@triton.jit
def _order_key(value):
    bits = value.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, ~(bits ^ -2147483648), bits)


@triton.jit
def _key_value(key):
    bits = tl.where(key < 0, (~key) ^ -2147483648, key)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _select_routes(x, cols, slots, exps, indices, E: tl.constexpr, K: tl.constexpr):
    """Two scalar integer reductions replace the shuffle-heavy argmax pair."""
    available = tl.broadcast_to(cols[None, :] < E, x.shape)
    keys = tl.where(available, _order_key(x), -2147483648)
    peak = tl.max(x, 1)
    for rank in tl.static_range(K):
        winner = tl.max(keys, 1)
        # available also disambiguates the INT_MIN key of an all-ones NaN.
        candidates = available & (keys == winner[:, None])
        expert = tl.min(tl.where(candidates, cols[None, :], 2147483647), 1)
        probability = tl.exp(_key_value(winner) - peak)
        exps = tl.where(slots[None, :] == rank, probability[:, None], exps)
        indices = tl.where(slots[None, :] == rank, expert[:, None], indices)
        available = available & (cols[None, :] != expert[:, None])
        keys = tl.where(available, keys, -2147483648)
    return exps, indices


@triton.jit
def _finish_routes(
    S,
    W,
    I,
    P,
    rows,
    slots,
    exps,
    ids,
    T: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    P0: tl.constexpr,
    HAS_PADDING: tl.constexpr,
):
    denominator = tl.sum(exps, 1)
    denominator = tl.where(denominator > 0, denominator, 1.0)
    valid = (rows[:, None] < T) & (slots[None, :] < K)
    scales = tl.load(S + ids * S0, valid, other=1.0).to(tl.float32)
    weights = exps / denominator[:, None] * scales
    if HAS_PADDING:
        padded = tl.load(P + rows * P0, rows < T, other=1)
        ids = tl.where(padded[:, None], -1, ids)
        weights = tl.where(padded[:, None], 0.0, weights)
    offsets = rows[:, None] * K + slots[None, :]
    tl.store(W + offsets, weights, valid)
    tl.store(I + offsets, ids, valid)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gemma4_routing_select"),
    key=["T", "E", "K", "HAS_PADDING", "CONTIGUOUS"],
    prune_configs_by={"early_config_prune": _prune_token_tiles},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _gemma4_native_kernel(
    G,
    S,
    W,
    I,
    P,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    S0: tl.constexpr,
    P0: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, B)
    slots = tl.arange(0, triton.next_power_of_2(K))
    x = tl.load(
        G + rows[:, None] * G0 + cols[None, :] * G1,
        (rows[:, None] < T) & (cols[None, :] < E),
        other=-float("inf"),
    ).to(tl.float32)
    exps = tl.full((BLOCK_T, triton.next_power_of_2(K)), 0.0, tl.float32)
    ids = tl.full((BLOCK_T, triton.next_power_of_2(K)), 0, tl.int32)
    exps, ids = _select_routes(x, cols, slots, exps, ids, E, K)
    _finish_routes(S, W, I, P, rows, slots, exps, ids, T, K, S0, P0, HAS_PADDING)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gemma4_routing_partial"),
    key=["T", "E", "K", "HAS_PADDING", "CONTIGUOUS"],
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _gemma4_partial_kernel(
    G,
    S,
    W,
    I,
    P,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    S0: tl.constexpr,
    P0: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    PARTIAL: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    x = tl.load(G + row * G0 + cols * G1, cols < E, other=-float("inf")).to(tl.float32)
    key = tl.where(cols < E, _order_key(x), -2147483648)
    # A descending lexicographic key retains lower expert IDs on equal values.
    ordered = (key.to(tl.int64) << 32) | (0xFFFFFFFF - cols.to(tl.int64))
    if PARTIAL:
        selected = tl.topk(ordered, triton.next_power_of_2(K))
        ranks = tl.arange(0, triton.next_power_of_2(K))
    else:
        selected = tl.sort(ordered, descending=True)
        ranks = cols
    ids = (0xFFFFFFFF - (selected & 0xFFFFFFFF)).to(tl.int32)
    logits = _key_value((selected >> 32).to(tl.int32))
    exps = tl.exp(logits - tl.max(x, 0))
    denominator = tl.sum(tl.where(ranks < K, exps, 0.0), 0)
    denominator = tl.where(denominator > 0, denominator, 1.0)
    scales = tl.load(S + ids * S0, ranks < K, other=1.0).to(tl.float32)
    weights = exps / denominator * scales
    if HAS_PADDING:
        padded = tl.load(P + row * P0)
        ids = tl.where(padded, -1, ids)
        weights = tl.where(padded, 0.0, weights)
    tl.store(W + row * K + ranks, weights, ranks < K)
    tl.store(I + row * K + ranks, ids, ranks < K)


def _prepare_routes(gating_output, topk, per_expert_scale, num_warps, is_padding):
    if gating_output.ndim != 2 or per_expert_scale.ndim != 1:
        raise ValueError("Expected logits [T,E] and expert scales [E]")
    t, e = gating_output.shape
    if not isinstance(topk, int) or not 1 <= topk <= e:
        raise ValueError("topk must be an integer in [1, E]")
    if not isinstance(num_warps, int) or num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    if e > 1024 or gating_output.device.type != "cuda":
        raise NotImplementedError("Gemma 4 routing supports CUDA and E <= 1024")
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if gating_output.dtype not in supported or per_expert_scale.dtype not in supported:
        raise NotImplementedError("Expected FP16, BF16 or FP32 logits and scales")
    if (
        per_expert_scale.shape != (e,)
        or per_expert_scale.device != gating_output.device
    ):
        raise ValueError("Expert scales must have shape [E] on the logits device")
    if is_padding is not None and (
        is_padding.shape != (t,)
        or is_padding.dtype != torch.bool
        or is_padding.device != gating_output.device
    ):
        raise ValueError("Padding must be a bool tensor [T] on the logits device")
    weights = torch.empty((t, topk), device=gating_output.device, dtype=torch.float32)
    ids = torch.empty((t, topk), device=gating_output.device, dtype=torch.int32)
    return t, e, weights, ids


def _kernel_args(g, k, s, w, i, p):
    t, e = g.shape
    contiguous = (
        g.is_contiguous() and s.is_contiguous() and (p is None or p.is_contiguous())
    )
    return (
        g,
        s,
        w,
        i,
        i if p is None else p,
        t,
        e,
        k,
        triton.next_power_of_2(e),
        g.stride(0),
        g.stride(1),
        s.stride(0),
        0 if p is None else p.stride(0),
        contiguous,
        p is not None,
    )


def gemma4_fused_routing_kernel_triton(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
    num_warps: int = 1,
    *,
    is_padding: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward routing. Prewarm each shape before CUDA Graph capture.

    num_warps is retained as a compatibility hint; libtuner chooses geometry.
    Hopper's explicit-layout implementation is exported by the backend registrar.
    """
    t, e, w, i = _prepare_routes(
        gating_output, topk, per_expert_scale, num_warps, is_padding
    )
    if not t:
        return w, i
    args = _kernel_args(gating_output, topk, per_expert_scale, w, i, is_padding)
    with torch_device_fn.device(gating_output.device):
        if topk <= 8:
            grid = lambda meta: (triton.cdiv(t, meta["BLOCK_T"]),)
            _gemma4_native_kernel[grid](*args)
        else:
            _gemma4_partial_kernel[(t,)](*args, PARTIAL=_HAS_PARTIAL_TOPK)
    return w, i
