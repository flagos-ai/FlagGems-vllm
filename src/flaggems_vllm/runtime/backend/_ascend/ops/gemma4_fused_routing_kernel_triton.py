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

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@triton.jit
def _take_rows(
    values, indices, rows, BT: tl.constexpr, WIDTH: tl.constexpr, SIZE: tl.constexpr
):
    offsets = tl.reshape(rows[:, None] * WIDTH + indices, (BT * SIZE,))
    values = tl.reshape(values.to(tl.float32), (BT * WIDTH,))
    return tl.reshape(tl.gather(values, offsets, 0), (BT, SIZE))


@libentry()
@triton.jit
def _gemma4_route_ascend(
    G,
    S,
    W,
    I,
    P,
    T,
    E: tl.constexpr,
    K: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    S0: tl.constexpr,
    P0: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    BT: tl.constexpr,
    BE: tl.constexpr,
    LOG_E: tl.constexpr,
    BK: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    rows = tl.arange(0, BT)
    row = tl.program_id(0) * BT + rows
    lanes = tl.arange(0, BE)
    valid_row = row < T
    row_offset = row.to(tl.int64) if WIDE_INDEX else row
    lanes_offset = lanes.to(tl.int64) if WIDE_INDEX else lanes
    raw = tl.load(
        G + row_offset[:, None] * G0 + lanes_offset[None, :] * G1,
        valid_row[:, None] & (lanes[None, :] < E),
        -float("inf"),
    )
    # Pack one radix digit and its current rank into an exact FP32 integer.
    # The rank breaks ties and makes every pass stable, including on padded E.
    if BE == 1:
        ordered = tl.full((BT, 1), 0, tl.int32)
    else:
        if G.dtype.element_ty == tl.float32:
            bits = raw.to(tl.int32, bitcast=True)
            key = tl.where(
                lanes[None, :] < E, bits ^ tl.where(bits < 0, 0, 2147483647), -1
            )
            DIGIT: tl.constexpr = 16 if LOG_E <= 8 else 11
            ROUNDS: tl.constexpr = 2 if LOG_E <= 8 else 3
        else:
            bits = raw.to(tl.int16, bitcast=True).to(tl.int32)
            key = (bits ^ tl.where(bits < 0, 0, 32767)) & 65535
            key = tl.where(lanes[None, :] < E, key, 65535)
            DIGIT: tl.constexpr = 16 if LOG_E <= 8 else 8
            ROUNDS: tl.constexpr = 1 if LOG_E <= 8 else 2
        DMASK: tl.constexpr = (1 << DIGIT) - 1
        code = ((key & DMASK) << LOG_E) | lanes[None, :]
        packed = al.sort(code.to(tl.float32), dim=-1, descending=False)
        ordered = packed.to(tl.int32) & (BE - 1)
        for part in tl.static_range(1, ROUNDS):
            digit = (key >> (part * DIGIT)) & DMASK
            digit = _take_rows(digit, ordered, rows, BT, BE, BE).to(tl.int32)
            code = (digit << LOG_E) | lanes[None, :]
            packed = al.sort(code.to(tl.float32), dim=-1, descending=False)
            ranks = packed.to(tl.int32) & (BE - 1)
            ordered = _take_rows(ordered, ranks, rows, BT, BE, BE).to(tl.int32)
    # Softmax over just the selected scores equals full softmax followed by
    # renormalization over the selected experts. Scales are applied afterward.
    ranks = tl.arange(0, BK)
    slots = tl.broadcast_to(ranks[None, :], (BT, BK))
    ids = _take_rows(ordered, slots, rows, BT, BE, BK).to(tl.int32)
    scores = _take_rows(raw, ids, rows, BT, BE, BK)
    valid_rank = ranks[None, :] < K
    peak = tl.max(tl.where(valid_rank, scores, -float("inf")), axis=1)
    exps = tl.where(valid_rank, tl.exp(scores - peak[:, None]), 0.0)
    den = tl.sum(exps, axis=1)
    den = tl.where(den > 0, den, 1.0)
    mask = valid_row[:, None] & valid_rank
    all_scales = tl.load(S + lanes_offset * S0, lanes < E, 1.0).to(tl.float32)
    weights = (exps / den[:, None]) * tl.reshape(
        tl.gather(all_scales, tl.reshape(ids, (BT * BK,)), 0), (BT, BK)
    )
    if HAS_PADDING:
        padded = tl.load(P + row_offset * P0, valid_row, True).to(tl.int1)
        weights = tl.where(padded[:, None], 0.0, weights)
        ids = tl.where(padded[:, None], -1, ids)
    out = row_offset[:, None] * K + ranks[None, :]
    tl.store(W + out, weights, mask)
    tl.store(I + out, ids, mask)


def _validate(gating_output, topk, per_expert_scale, num_warps, is_padding):
    if gating_output.ndim != 2 or per_expert_scale.ndim != 1:
        raise ValueError("Expected logits [T,E] and expert scales [E]")
    t, e = gating_output.shape
    if not isinstance(topk, int) or not 1 <= topk <= e:
        raise ValueError("topk must be an integer in [1, E]")
    if not isinstance(num_warps, int) or num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if gating_output.dtype not in supported or per_expert_scale.dtype not in supported:
        raise NotImplementedError("Expected FP16, BF16 or FP32 logits and scales")
    if e > 1024 or gating_output.device.type != "npu":
        raise NotImplementedError("Ascend Gemma 4 routing requires NPU and E <= 1024")
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
    return t, e


def gemma4_fused_routing_kernel_triton(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
    num_warps: int = 1,
    *,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route each token to its highest-scoring experts on Ascend."""
    t, e = _validate(gating_output, topk, per_expert_scale, num_warps, is_padding)
    g, s, p = gating_output, per_expert_scale, is_padding
    weights = torch.empty((t, topk), device=g.device, dtype=torch.float32)
    ids = torch.empty((t, topk), device=g.device, dtype=torch.int32)
    if t == 0:
        return weights, ids
    be = triton.next_power_of_2(e)
    if e == 128 and topk == 8:
        # Small batches need enough independent programs; larger ones
        # amortize scheduling and scale loads across more rows.
        bt = 1 if t <= 24 else 2 if t <= 64 else 4 if t <= 512 else 16
    else:
        bt = 1
    grid = (triton.cdiv(t, bt),)
    wide_index = (
        max(
            t * topk,
            (t - 1) * g.stride(0) + (e - 1) * g.stride(1),
            (e - 1) * s.stride(0),
            0 if p is None else (t - 1) * p.stride(0),
        )
        > 2**31 - 1
    )
    with torch_device_fn.device(g.device):
        _gemma4_route_ascend[grid](
            g,
            s,
            weights,
            ids,
            ids if p is None else p,
            t,
            e,
            topk,
            g.stride(0),
            g.stride(1),
            s.stride(0),
            0 if p is None else p.stride(0),
            p is not None,
            bt,
            be,
            be.bit_length() - 1,
            triton.next_power_of_2(topk),
            wide_index,
            num_warps=1,
        )
    return weights, ids
