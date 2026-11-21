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

"""Bounded dense decoding and overlapped expert projections."""

from functools import lru_cache
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
)
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    _PPU_DENSE_MIN_TOKENS,
    _TL_QUANT_TYPE_FP4_E2M1,
    _TL_QUANT_TYPE_FP8_E4M3,
    _TL_QUANT_TYPE_UINT4B8,
    DENSE_BLOCK_ROWS,
    decode_fp8_scaled_bf16_normalized,
    decode_int8_scaled_bf16,
    decode_raw_int4_bf16,
    decode_raw_mxfp4_bf16,
    e8m0_to_bf16_bits,
    tle_async,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.policy import MarlinCall
from flaggems_vllm.runtime.backend._thead.fused.marlin.routing import (
    align_grouped_tokens,
)
from flaggems_vllm.utils import libentry

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
            lo, hi = decode_raw_int4_bf16(q, scale)
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
            lo, hi = decode_raw_mxfp4_bf16(q, scale)
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
        if FP8:
            # Normalize the FP8 value before multiplying by the original scale.
            value = decode_fp8_scaled_bf16_normalized(q, scale)
        else:
            value = decode_int8_scaled_bf16(q, scale)
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


@libentry()
@triton.jit
def dense_align_count_histogram(
    route_ids,
    Count,
    Prefix,
    sorted_routes,
    NUM_ROUTES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EM: tl.constexpr,
    RB: tl.constexpr,
    ID_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    expert = tl.arange(0, NUM_EXPERTS)
    row = pid * RB + tl.arange(0, RB)
    valid = row < NUM_ROUTES
    ids = tl.load(route_ids + row, valid, other=0).to(tl.int32)
    val = tl.histogram(ids, NUM_EXPERTS, mask=valid)
    tl.store(Count + (pid + 1) * NUM_EXPERTS + expert, val)
    if pid == 0:
        tl.store(Prefix, 0)
        tl.store(Count + expert, 0)
    o = pid * ID_BLOCK + tl.arange(0, ID_BLOCK)
    tl.store(sorted_routes + o, NUM_ROUTES, o < EM)


def align_dense_fp8_histogram(
    ids: torch.Tensor, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from flaggems_vllm.ops.moe_align_block_size import (
        moe_align_block_size_stage2_vec,
        moe_align_block_size_stage3,
        moe_align_block_size_stage4,
    )

    r = ids.numel()
    e = num_experts
    em = r + (DENSE_BLOCK_ROWS - 1) * e
    mb = triton.cdiv(em, 64)
    rb = triton.next_power_of_2(triton.cdiv(r, e))
    sorted_ids = torch.empty((em,), device=ids.device, dtype=torch.int32)
    experts = torch.empty((mb,), device=ids.device, dtype=torch.int32)
    post = torch.empty((1,), device=ids.device, dtype=torch.int32)
    count = torch.empty((e + 1, e), device=ids.device, dtype=torch.int32)
    prefix = torch.empty((e + 1,), device=ids.device, dtype=torch.int32)
    dense_align_count_histogram[(e,)](
        ids,
        count,
        prefix,
        sorted_ids,
        r,
        e,
        em,
        rb,
        triton.next_power_of_2(triton.cdiv(em, e)),
        num_warps=4,
    )
    moe_align_block_size_stage2_vec[(e,)](count, e)
    moe_align_block_size_stage3[(1,)](post, count, prefix, e, e, 64)

    moe_align_block_size_stage4[(e,)](
        ids, sorted_ids, experts, count, prefix, e, 64, r, rb
    )
    return sorted_ids, experts, post, prefix


@lru_cache(maxsize=16)
def eager_decode_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device=device, priority=-1)


def decode_dense_bank(
    weight: torch.Tensor,
    scales: torch.Tensor,
    destination: torch.Tensor,
    quant_type_id: int,
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
    # Eager callers can share a work queue; graph capture needs its own fork/join stream.
    if torch.cuda.is_current_stream_capturing():
        side = torch.cuda.Stream(device=hidden.device, priority=-1)
    else:
        side = eager_decode_stream(hidden.device)
    for operand in (banks, w1, w2, w1_scale, w2_scale, ids):
        operand.record_stream(side)

    first, second = slice(half, experts_count), slice(0, half)
    gate_weight, gate_scale = w1[:, :n], w1_scale[:, :n]
    up_weight, up_scale = w1[:, n:], w1_scale[:, n:]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        sorted_ids, experts, padded, prefix = align_dense_fp8_histogram(
            ids, experts_count
        )
    decode_dense_bank(
        gate_weight[first], gate_scale[first], gate_bank, QUANT_TYPE_FP8_E4M3
    )
    main.wait_stream(side)
    for metadata in (sorted_ids, experts, padded, prefix):
        metadata.record_stream(main)
    em = sorted_ids.numel()
    row_tiles = triton.cdiv(em, 64)

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
            64,
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
            64,
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
    decode_elements: int,
    projection_width: int,
    workspace_limit: int,
) -> torch.Tensor:
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        get_default_config,
        invoke_fused_moe_triton_kernel,
    )

    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n, topk = n2 // 2, topk_ids.shape[1]
    is_fp8 = quant_type_id == QUANT_TYPE_FP8_E4M3
    is_w4 = quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
    routes = m * topk
    scratch = torch.empty(
        decode_elements, device=hidden_states.device, dtype=torch.bfloat16
    )
    # Gate/up and down do not overlap in time; only the activation survives both GEMMs.
    cache = hidden_states.new_empty((routes * projection_width,))
    gateup = cache[: routes * n2].view(m, topk, n2)
    intermediate = hidden_states.new_empty((routes, n))
    down = cache[: routes * down_tile].view(m, topk, down_tile)
    if (
        is_fp8
        and m >= MIN_OVERLAP_TOKENS
        and e in (128, 256)
        and k == 4096
        and n == 256
        and 4 <= topk <= 8
        and gate_tile == n
        and down_tile <= k // 2
        and k % down_tile == 0
        and down_tile % 256 == 0
        and hidden_states.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        workspace = (
            2 * decode_elements
            + 2 * routes * (projection_width + n)
            + 4 * (e + 1) ** 2
            + 16 * (routes + (DENSE_BLOCK_ROWS - 1) * e)
            + 8
        )
        if workspace <= workspace_limit:
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
            )
    if quant_type_id == QUANT_TYPE_FP4_E2M1:
        w1_scale, w2_scale = w1_scale.view(torch.uint8), w2_scale.view(torch.uint8)
    for index, (weight, scales) in enumerate(((w1, w1_scale), (w2, w2_scale))):
        first = index == 0
        output_size, reduction = weight.shape[1:]
        if is_w4:
            reduction *= 2
        config = get_default_config(
            max(m, _PPU_DENSE_MIN_TOKENS),
            e,
            output_size,
            reduction,
            topk,
            "bf16",
            gemm_stage="gemm1" if first else "gemm2",
        )
        if config["BLOCK_SIZE_M"] != DENSE_BLOCK_ROWS:
            raise NotImplementedError(
                "Dense Marlin workspace requires 64-row expert blocks"
            )
        if first:
            sorted_ids, experts, padded = align_grouped_tokens(
                topk_ids, config["BLOCK_SIZE_M"], e
            )
        tile = gate_tile if first else down_tile
        for col in range(0, output_size, tile):
            stop = min(col + tile, output_size)
            width = stop - col
            qw, ss = weight[:, col:stop], scales[:, col:stop]
            decode_dense_bank(qw, ss, scratch, quant_type_id)
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


def run_dense_batches(
    call: MarlinCall, gate_limit: int, down_limit: int, workspace_limit: int
) -> None:
    gate_tile = min(2 * call.shape.intermediate, gate_limit)
    down_tile = min(call.shape.hidden, down_limit)
    decode_elements = call.shape.experts * max(
        gate_tile * call.shape.hidden, down_tile * call.shape.intermediate
    )
    projection_width = max(2 * call.shape.intermediate, down_tile)
    fixed_bytes = (
        2 * decode_elements
        + 4 * (call.shape.experts + 1) ** 2
        + 16 * (DENSE_BLOCK_ROWS - 1) * call.shape.experts
        + 8
    )
    bytes_per_token = call.shape.top_k * (
        2 * (projection_width + call.shape.intermediate) + 12
    )
    batch_tokens = min(
        call.shape.tokens, (workspace_limit - fixed_bytes) // bytes_per_token
    )
    if batch_tokens < 1:
        raise NotImplementedError("PPU dense MoE workspace exceeds its budget")
    if batch_tokens < call.shape.tokens and call.out is not call.hidden_states:
        # A chunk can overwrite activations needed by a later chunk.
        byte_offset = abs(call.out.data_ptr() - call.hidden_states.data_ptr())
        if 0 < byte_offset < call.hidden_states.nbytes:
            raise ValueError(
                "output must not partially overlap hidden_states in chunked dense MoE"
            )
    for start in range(0, call.shape.tokens, batch_tokens):
        end = min(start + batch_tokens, call.shape.tokens)
        run_dense_wna16(
            call.hidden_states[start:end],
            call.w1,
            call.w2,
            call.w1_scale,
            call.w2_scale,
            call.topk_weights[start:end],
            call.out[start:end],
            call.topk_ids[start:end],
            call.quant_type_id,
            call.apply_router_weight_on_input,
            gate_tile=gate_tile,
            down_tile=down_tile,
            decode_elements=decode_elements,
            projection_width=projection_width,
            workspace_limit=workspace_limit,
        )
