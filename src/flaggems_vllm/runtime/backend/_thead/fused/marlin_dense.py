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


"""Dense prefill kernels and workspace consumers for T-Head Marlin MoE."""

from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from flag_gems import zero_

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
)
from flaggems_vllm.ops.moe_align_block_size import _moe_align_block_size_triton_4stage
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._thead.fused.marlin_fragments import (
    _TL_QUANT_TYPE_FP4_E2M1,
    _TL_QUANT_TYPE_FP8_E4M3,
    _TL_QUANT_TYPE_UINT4B8,
    decode_fp8_word,
    decode_mx_register,
    e8m0_to_bf16_bits,
    tle_async,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin_quantized import (
    decode_fp8_scaled_bf16_normalized,
)
from flaggems_vllm.utils import libentry

OVERLAP_BLOCK_ROWS = 64
DENSE_ROUTING_ALIGN_WORDS = 4
MIN_OVERLAP_TOKENS = 8192


@libentry()
@triton.jit
def decode_bank(
    weight_bytes,
    scales,
    decoded,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    QE: tl.constexpr,
    QN: tl.constexpr,
    QK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    QTYPE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PAIR_FP8: tl.constexpr = False,
):
    FP8: tl.constexpr = QTYPE == _TL_QUANT_TYPE_FP8_E4M3
    W4: tl.constexpr = (
        QTYPE == _TL_QUANT_TYPE_UINT4B8 or QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    )
    expert = tl.program_id(2).to(tl.int64)
    ns = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    k_base = (tl.program_id(1) * BK).to(tl.int64)
    ks = k_base + tl.arange(0, BK)
    mask = (ns[:, None] < num_columns) & (ks[None, :] < reduction_size)
    if W4:
        kps = k_base // 2 + tl.arange(0, BK // 2)
        q = tl.load(
            weight_bytes + expert * QE + ns[:, None] * QN + kps[None, :] * QK,
            (ns[:, None] < num_columns) & (kps[None, :] * 2 < reduction_size),
            other=0,
        )
        if QTYPE == _TL_QUANT_TYPE_UINT4B8:
            scale = tl.load(
                scales + expert * SE + ns * SN + (k_base // 128) * SG,
                ns < num_columns,
                other=0,
            )[:, None]
            low_bits = (q & 15).to(tl.uint16) | 0x4300
            high_bits = (q >> 4).to(tl.uint16) | 0x4300
            lo = ((low_bits.to(tl.bfloat16, bitcast=True) - 136.0) * scale).to(
                tl.bfloat16
            )
            hi = ((high_bits.to(tl.bfloat16, bitcast=True) - 136.0) * scale).to(
                tl.bfloat16
            )
        else:
            gs = k_base // 32 + tl.arange(0, BK // 32)
            byte = tl.load(
                scales + expert * SE + ns[:, None] * SN + gs[None, :] * SG,
                (ns[:, None] < num_columns) & (gs[None, :] * 32 < reduction_size),
                other=127,
            ).to(tl.uint32)
            bits = e8m0_to_bf16_bits(byte)
            scale = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            scale = tl.reshape(
                tl.broadcast_to(scale[:, :, None], (BN, BK // 32, 16)), (BN, BK // 2)
            )
            if reduction_size >= num_columns:
                # Embed each E2M1 nibble in BF16 before correcting the exponent.
                low_bits = ((q & 15).to(tl.uint16) * 0x1040) & 0x81C0
                high_bits = ((q >> 4).to(tl.uint16) * 0x1040) & 0x81C0
                lo = (
                    low_bits.to(tl.bfloat16, bitcast=True).to(tl.float32) * (2.0**126)
                ).to(tl.bfloat16)
                hi = (
                    high_bits.to(tl.bfloat16, bitcast=True).to(tl.float32) * (2.0**126)
                ).to(tl.bfloat16)
                lo = (lo * scale).to(tl.bfloat16)
                hi = (hi * scale).to(tl.bfloat16)
            else:
                bits = scale.to(tl.uint16, bitcast=True).to(tl.uint32)
                pair_scale = bits | (bits << 16)
                pair, _ = decode_mx_register(q.to(tl.uint32), pair_scale, pair_scale)
                lo = pair.to(tl.uint16).to(tl.bfloat16, bitcast=True)
                hi = (pair >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
        value = tl.reshape(tl.join(lo, hi), (BN, BK))
    else:
        q = tl.load(
            weight_bytes + expert * QE + ns[:, None] * QN + ks[None, :] * QK,
            mask,
            other=0 if FP8 else 128,
        )
        scale = tl.load(
            scales + expert * SE + ns * SN + (k_base // 128) * SG,
            ns < num_columns,
            other=0,
        )[:, None]
        if FP8 and PAIR_FP8:
            q0, q1 = tl.split(tl.reshape(q, (BN, BK // 2, 2)))
            word = q0.to(tl.uint32) | (q1.to(tl.uint32) << 8)
            bits = scale.to(tl.uint16, bitcast=True).to(tl.uint32)
            pair, _ = decode_fp8_word(word, bits | (bits << 16))
            lo = pair.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            hi = (pair >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
            value = tl.reshape(tl.join(lo, hi), (BN, BK))
        elif FP8:
            # Normalize the FP8 value before multiplying by the original scale.
            value = decode_fp8_scaled_bf16_normalized(q, scale)
        else:
            value = ((q.to(tl.int32) - 128).to(tl.bfloat16) * scale).to(tl.bfloat16)
    tl.store(
        decoded
        + expert * num_columns * reduction_size
        + ns[:, None]
        + ks[None, :] * num_columns,
        value,
        mask,
    )


@triton.jit
def dense_w8_bf16_tile(
    activations,
    B,
    output_tile,
    router_weights,
    route_ids,
    block_experts,
    row_tile,
    col_tile,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    VALID,
    AM,
    AK,
    BE,
    BKSTR,
    BNSTR,
    CM,
    CN,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    USE32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    DIRECT_B: tl.constexpr = False,
    FIRST_EXPERT: tl.constexpr = 0,
):
    offsets = row_tile * BM + tl.arange(0, BM)
    tokens = tl.load(route_ids + offsets, offsets < EM, other=VALID)
    expert = tl.load(block_experts + row_tile) - FIRST_EXPERT
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    tokens = tokens.to(offset_type)
    expert = expert.to(offset_type)
    mask = tokens < VALID
    ks = tl.arange(0, BK)
    abase = activations + (tokens[:, None] // TOPK) * AM + ks[None, :] * AK
    bp = tl.make_block_ptr(
        base=B + expert * BE,
        shape=(reduction_size, num_columns),
        strides=(BKSTR, BNSTR),
        offsets=(0, col_tile * BN),
        block_shape=(BK, BN),
        order=(1, 0),
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for k_tile in tl.range(0, tl.cdiv(reduction_size, BK), num_stages=STAGES):
        a = tl.load(
            abase,
            mask=mask[:, None] & (k_tile * BK + ks[None, :] < reduction_size),
            other=0.0,
        )
        if DIRECT_B:
            cols = col_tile * BN + tl.arange(0, BN)
            b = tl.load(
                B
                + expert * BE
                + (k_tile * BK + ks[:, None]) * BKSTR
                + cols[None, :] * BNSTR
            )
        elif num_columns % BN == 0 and reduction_size % BK == 0:
            b = tle_async.load(bp, is_async=True)
        else:
            b = tle_async.load(
                bp, boundary_check=(0, 1), padding_option="zero", is_async=True
            )
        acc = tl.dot(a, b, acc=acc)
        abase += BK * AK
        bp = tl.advance(bp, (BK, 0))
    if MUL:
        w = tl.load(router_weights + tokens, mask, other=0).to(tl.float32)
    else:
        w = tl.full((BM,), 1.0, tl.float32)
    acc *= w[:, None]
    ns = col_tile * BN + tl.arange(0, BN)
    tl.store(
        output_tile + tokens[:, None] * CM + ns[None, :] * CN,
        acc.to(output_tile.dtype.element_ty),
        mask[:, None] & (ns[None, :] < num_columns),
    )


@libentry()
@triton.jit
def dense_w8_bf16_kernel(
    activations,
    B,
    output_tile,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    VALID,
    AM,
    AK,
    BE,
    BKSTR,
    BNSTR,
    CM,
    CN,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    USE32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    STAGES: tl.constexpr,
    DIRECT_B: tl.constexpr = False,
    REVERSE_M: tl.constexpr = False,
):
    """Gather routed activations and load a decoded BF16 bank."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(EM, BM)
    num_n = tl.cdiv(num_columns, BN)
    row_tile, col_tile = tl.swizzle2d(pid // num_n, pid % num_n, num_m, num_n, GM)
    live_rows = tl.load(padded_count)
    if row_tile * BM >= live_rows:
        return
    if REVERSE_M:
        row_tile = live_rows // BM - 1 - row_tile
    dense_w8_bf16_tile(
        activations,
        B,
        output_tile,
        router_weights,
        route_ids,
        block_experts,
        row_tile,
        col_tile,
        num_columns,
        reduction_size,
        EM,
        VALID,
        AM,
        AK,
        BE,
        BKSTR,
        BNSTR,
        CM,
        CN,
        TOPK,
        MUL,
        USE32,
        BM,
        BN,
        BK,
        STAGES,
        DIRECT_B,
    )


@libentry()
@triton.jit
def dense_w8_bf16_expert_window(
    activations,
    B,
    output_tile,
    router_weights,
    route_ids,
    block_experts,
    Prefix,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    VALID,
    AM,
    AK,
    BE,
    BKSTR,
    BNSTR,
    CM,
    CN,
    TOPK: tl.constexpr,
    MUL: tl.constexpr,
    USE32: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    STAGES: tl.constexpr,
    FIRST_EXPERT: tl.constexpr,
    LAST_EXPERT: tl.constexpr,
):
    begin = tl.load(Prefix + FIRST_EXPERT)
    end = tl.load(Prefix + LAST_EXPERT)
    num_m = (end - begin) // BM
    num_n = tl.cdiv(num_columns, BN)
    # Reuse CTAs for overflow tiles when routes concentrate in one window.
    for pid in range(tl.program_id(0), num_m * num_n, tl.num_programs(0)):
        row_tile, col_tile = tl.swizzle2d(pid // num_n, pid % num_n, num_m, num_n, GM)
        row_tile = end // BM - 1 - row_tile
        dense_w8_bf16_tile(
            activations,
            B,
            output_tile,
            router_weights,
            route_ids,
            block_experts,
            row_tile,
            col_tile,
            num_columns,
            reduction_size,
            EM,
            VALID,
            AM,
            AK,
            BE,
            BKSTR,
            BNSTR,
            CM,
            CN,
            TOPK,
            MUL,
            USE32,
            BM,
            BN,
            BK,
            STAGES,
            False,
            FIRST_EXPERT,
        )


def dense_routing_sizes(
    routes: int, experts: int, block_m: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    padded = routes + min(routes, experts) * (block_m - 1)
    sizes = (
        padded,
        triton.cdiv(padded, block_m),
        1,
        (experts + 1) * experts,
        experts + 1,
    )
    # 16-byte slots keep full and tail chunks on the same pointer specialization.
    slots = tuple(
        triton.cdiv(size, DENSE_ROUTING_ALIGN_WORDS) * DENSE_ROUTING_ALIGN_WORDS
        for size in sizes
    )
    return sizes, slots


def align_dense_routes(
    ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    routing: Tuple[torch.Tensor, ...],
) -> None:
    sorted_ids, experts, post, counts, prefix = routing
    zero_(counts)
    zero_(prefix)
    _moe_align_block_size_triton_4stage(
        ids, num_experts, block_m, sorted_ids, experts, post, workspace=(prefix, counts)
    )


@lru_cache(maxsize=16)
def eager_decode_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device, priority=-1)


def decode_dense_bank(
    weight: torch.Tensor,
    scales: torch.Tensor,
    destination: torch.Tensor,
    quant_type_id: int,
    *,
    pair_fp8: bool = False,
) -> None:
    experts, width, packed_k = weight.shape
    reduction = (
        packed_k * 2
        if quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
        else packed_k
    )
    columns = 128 if quant_type_id == QUANT_TYPE_FP8_E4M3 and width >= 128 else 64
    decode_bank[(triton.cdiv(width, columns), triton.cdiv(reduction, 128), experts)](
        weight.view(torch.uint8),
        scales,
        destination,
        width,
        reduction,
        *weight.stride(),
        *scales.stride(),
        quant_type_id,
        columns,
        128,
        PAIR_FP8=pair_fp8,
        num_warps=8 if columns == 128 else 4,
        num_stages=1,
    )


def run_fp8_ordered_overlap(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router: torch.Tensor,
    output: torch.Tensor,
    ids: torch.Tensor,
    input_mul: bool,
    banks: torch.Tensor,
    gateup: torch.Tensor,
    middle: torch.Tensor,
    down: torch.Tensor,
    routing: Tuple[torch.Tensor, ...],
    side: torch.cuda.Stream,
) -> torch.Tensor:
    m, k = hidden.shape
    experts_count, n2, _ = w1.shape
    n = n2 // 2
    topk = ids.size(1)
    routes = m * topk
    half = experts_count // 2
    # Each bank holds one gate/up expert window or one down-projection column tile.
    gate_bank, up_bank = banks.chunk(2)
    gateup = gateup.view(routes, n2)
    down_tile = down.size(-1)

    main = torch.cuda.current_stream(hidden.device)
    for operand in (w1, w2, w1_scale, w2_scale, ids):
        operand.record_stream(side)

    first, second = slice(half, experts_count), slice(0, half)
    gate_weight, gate_scale = w1[:, :n], w1_scale[:, :n]
    up_weight, up_scale = w1[:, n:], w1_scale[:, n:]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        align_dense_routes(ids, experts_count, OVERLAP_BLOCK_ROWS, routing)
    sorted_ids, experts, padded, _, prefix = routing
    decode_dense_bank(
        gate_weight[first], gate_scale[first], gate_bank, QUANT_TYPE_FP8_E4M3
    )
    main.wait_stream(side)
    em = sorted_ids.numel()
    row_tiles = triton.cdiv(em, OVERLAP_BLOCK_ROWS)

    def project_gate(
        bank: torch.Tensor, projected: torch.Tensor, window: slice
    ) -> None:
        dense_w8_bf16_expert_window[(triton.cdiv(row_tiles, 2),)](
            hidden,
            bank,
            projected,
            router,
            sorted_ids,
            experts,
            prefix,
            n,
            k,
            em,
            routes,
            *hidden.stride(),
            k * n,
            n,
            1,
            n2,
            1,
            topk,
            input_mul,
            True,
            OVERLAP_BLOCK_ROWS,
            256,
            32,
            8,
            4,
            num_warps=4,
            num_stages=4,
            FIRST_EXPERT=window.start,
            LAST_EXPERT=window.stop,
        )

    gate_output, up_output = gateup[:, :n], gateup[:, n:]
    for window in (first, second):
        side.wait_stream(main)
        project_gate(gate_bank, gate_output, window)
        with torch.cuda.stream(side):
            decode_dense_bank(
                up_weight[window], up_scale[window], up_bank, QUANT_TYPE_FP8_E4M3
            )
        main.wait_stream(side)
        side.wait_stream(main)
        project_gate(up_bank, up_output, window)
        with torch.cuda.stream(side):
            if window == first:
                decode_dense_bank(
                    gate_weight[second],
                    gate_scale[second],
                    gate_bank,
                    QUANT_TYPE_FP8_E4M3,
                )
            else:
                decode_dense_bank(
                    w2[:, :down_tile],
                    w2_scale[:, :down_tile],
                    gate_bank,
                    QUANT_TYPE_FP8_E4M3,
                )
        main.wait_stream(side)
    silu_and_mul_out(gate_output, up_output, middle)
    for col in range(0, k, down_tile):
        end = col + down_tile
        if end < k:
            side.wait_stream(main)
        dense_w8_bf16_kernel[(row_tiles * (down_tile // 256),)](
            middle,
            gate_bank,
            down,
            router,
            sorted_ids,
            experts,
            padded,
            down_tile,
            n,
            em,
            routes,
            *middle.stride(),
            down_tile * n,
            down_tile,
            1,
            down_tile,
            1,
            1,
            not input_mul,
            True,
            OVERLAP_BLOCK_ROWS,
            256,
            32,
            8,
            4,
            DIRECT_B=True,
            REVERSE_M=True,
            num_warps=4,
            num_stages=4,
        )
        moe_sum(down, output[:, col:end])
        if end < k:
            with torch.cuda.stream(side):
                decode_dense_bank(
                    w2[:, end : end + down_tile],
                    w2_scale[:, end : end + down_tile],
                    up_bank,
                    QUANT_TYPE_FP8_E4M3,
                )
            main.wait_stream(side)
            gate_bank, up_bank = up_bank, gate_bank
    return output


def project_dense_w8(
    activation: torch.Tensor,
    decoded: torch.Tensor,
    projected: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    experts: torch.Tensor,
    padded: torch.Tensor,
    multiply_router: bool,
    projection_topk: int,
    config: Dict[str, int],
    is_down: bool,
) -> None:
    _, width, reduction = decoded.shape
    # Direct loads avoid staging aligned short-K down-projection tiles.
    direct_b = (
        is_down
        and width >= 1024
        and reduction == 256
        and width % config["BLOCK_SIZE_N"] == 0
        and reduction % config["BLOCK_SIZE_K"] == 0
    )
    dense_w8_bf16_kernel[
        (
            triton.cdiv(sorted_ids.numel(), config["BLOCK_SIZE_M"])
            * triton.cdiv(width, config["BLOCK_SIZE_N"]),
        )
    ](
        activation,
        decoded,
        projected,
        topk_weights,
        sorted_ids,
        experts,
        padded,
        width,
        reduction,
        sorted_ids.numel(),
        topk_weights.numel(),
        *activation.stride(),
        decoded.stride(0),
        decoded.stride(2),
        decoded.stride(1),
        projected.stride(-2),
        projected.stride(-1),
        projection_topk,
        multiply_router,
        config["USE_INT32_OFFSETS"],
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["num_stages"],
        DIRECT_B=direct_b,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def run_dense_wna16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool,
    *,
    gate_tile: int,
    down_tile: int,
    scratch: torch.Tensor,
    cache: torch.Tensor,
    activations: torch.Tensor,
    route_storage: torch.Tensor,
    side: Optional[torch.cuda.Stream],
    configs: Tuple[Dict[str, int], Dict[str, int]],
) -> torch.Tensor:
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )

    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n, topk = n2 // 2, topk_ids.shape[1]
    is_w4 = quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
    routes = m * topk
    gateup = cache[: routes * n2].view(m, topk, n2)
    intermediate = activations[: routes * n].view(routes, n)
    down = cache[: routes * down_tile].view(m, topk, down_tile)
    block_m = OVERLAP_BLOCK_ROWS if side is not None else configs[0]["BLOCK_SIZE_M"]
    sizes, slots = dense_routing_sizes(routes, e, block_m)
    routing_buffers = tuple(
        slot[:size]
        for slot, size in zip(route_storage[: sum(slots)].split(slots), sizes)
    )
    if side is not None:
        try:
            return run_fp8_ordered_overlap(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                output,
                topk_ids,
                apply_router_weight_on_input,
                scratch,
                gateup,
                intermediate,
                down,
                routing_buffers,
                side,
            )
        except BaseException:
            # Keep the shared workspace live until queued side-stream work finishes.
            torch.cuda.current_stream(hidden_states.device).wait_stream(side)
            raise
    align_dense_routes(topk_ids, e, block_m, routing_buffers)
    sorted_ids, experts, padded, _, _ = routing_buffers
    if quant_type_id == QUANT_TYPE_FP4_E2M1:
        w1_scale, w2_scale = w1_scale.view(torch.uint8), w2_scale.view(torch.uint8)
    for index, (weight, scales) in enumerate(((w1, w1_scale), (w2, w2_scale))):
        first = index == 0
        output_size, reduction = weight.shape[1:]
        if is_w4:
            reduction *= 2
        config = configs[index]
        tile = gate_tile if first else down_tile
        for col in range(0, output_size, tile):
            stop = min(col + tile, output_size)
            width = stop - col
            qw, ss = weight[:, col:stop], scales[:, col:stop]
            decode_dense_bank(
                qw,
                ss,
                scratch,
                quant_type_id,
                pair_fp8=quant_type_id == QUANT_TYPE_FP8_E4M3,
            )
            decoded = (
                scratch[: e * reduction * width]
                .view(e, reduction, width)
                .transpose(1, 2)
            )
            projected = gateup[:, :, col:stop] if first else down[:, :, :width]
            operands = (
                hidden_states if first else intermediate,
                decoded,
                projected,
            )
            routing = (topk_weights, sorted_ids, experts, padded)
            multiply_router = (
                apply_router_weight_on_input
                if first
                else not apply_router_weight_on_input
            )
            projection_topk = topk if first else 1
            if is_w4:
                invoke_fused_moe_triton_kernel(
                    *operands,
                    None,
                    None,
                    *routing,
                    multiply_router,
                    projection_topk,
                    config,
                    tl.bfloat16,
                )
            else:
                project_dense_w8(
                    *operands,
                    *routing,
                    multiply_router,
                    projection_topk,
                    config,
                    is_down=not first,
                )
            if not first:
                moe_sum(projected, output[:, col:stop])
        if first:
            flat = gateup.view(m * topk, n2)
            silu_and_mul_out(flat[:, :n], flat[:, n:], intermediate)
    return output
