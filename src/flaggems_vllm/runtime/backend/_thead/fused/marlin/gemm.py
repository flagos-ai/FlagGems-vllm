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

"""TLE projection, activation staging and reduction kernels."""

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    _TL_QUANT_TYPE_FP8_E4M3,
    _TL_QUANT_TYPE_UINT4B8,
    _TL_QUANT_TYPE_UINT8B128,
    decode_fp8_scaled_bf16_normalized,
    decode_raw_int4_bf16,
    decode_raw_mxfp4_bf16,
    dequant_weight,
    e8m0_to_bf16_bits,
    store_k16,
    store_k32,
    tle_async,
)
from flaggems_vllm.utils import libentry


@triton.jit
def w4_pair_gemm(
    activations,
    b_block_ptr,
    scales,
    expert,
    ns,
    se,
    sg,
    sn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    PID_M,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STAGES: tl.constexpr,
    QTYPE: tl.constexpr,
):
    # Even and odd K values occupy separate contiguous EM x K/2 planes.
    # The logical width and physical row pitch must agree for these loads.
    ae = tl.make_block_ptr(
        activations,
        shape=(EM, reduction_size // 2),
        strides=(reduction_size // 2, 1),
        offsets=(PID_M * BM, 0),
        block_shape=(BM, BK // 2),
        order=(1, 0),
    )
    ao = tl.make_block_ptr(
        activations + EM * (reduction_size // 2),
        shape=(EM, reduction_size // 2),
        strides=(reduction_size // 2, 1),
        offsets=(PID_M * BM, 0),
        block_shape=(BM, BK // 2),
        order=(1, 0),
    )
    acc = tl.zeros((BN, BM), tl.float32)
    for tile in tl.range(0, tl.cdiv(reduction_size, BK), num_stages=STAGES):
        a0 = tle_async.load(
            ae, boundary_check=(0, 1), padding_option="zero", is_async=True
        )
        a1 = tle_async.load(
            ao, boundary_check=(0, 1), padding_option="zero", is_async=True
        )
        if num_columns % BN == 0 and reduction_size % BK == 0:
            q = tle_async.load(b_block_ptr, is_async=True)
        else:
            q = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
        if QTYPE == _TL_QUANT_TYPE_UINT4B8:
            scale = tl.load(
                scales + expert * se + (tile * BK // GROUP_SIZE) * sg + ns * sn,
                ns < num_columns,
                other=0,
            )[None, :]
            lo, hi = decode_raw_int4_bf16(q, scale)
        else:
            gs = tile * BK // 32 + tl.arange(0, BK // 32)
            byte = tl.load(
                scales + expert * se + gs[:, None] * sg + ns[None, :] * sn,
                (ns[None, :] < num_columns) & (gs[:, None] * 32 < reduction_size),
                other=127,
            ).to(tl.uint32)
            bits = e8m0_to_bf16_bits(byte)
            scale = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            scale = tl.reshape(
                tl.broadcast_to(scale[:, None, :], (BK // 32, 16, BN)),
                (BK // 2, BN),
            )
            lo, hi = decode_raw_mxfp4_bf16(q, scale)
        acc = tl.dot(tl.trans(lo), tl.trans(a0), acc)
        acc = tl.dot(tl.trans(hi), tl.trans(a1), acc)
        ae = tl.advance(ae, (0, BK // 2))
        ao = tl.advance(ao, (0, BK // 2))
        b_block_ptr = tl.advance(b_block_ptr, (BK // 2, 0))
    return acc


@triton.jit
def quantized_gemm_tile(
    a_block_ptr,
    b_block_ptr,
    b_scale_ptr,
    expert,
    offs_n,
    stride_bse,
    stride_bsg,
    stride_bsn,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr = 128,
):
    """Load, decode and multiply one stream of quantized weight tiles."""
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for k_tile in tl.range(
        0, tl.cdiv(reduction_size, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
    ):
        activation = tle_async.load(
            a_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )
        if reduction_size % BLOCK_SIZE_K == 0 and num_columns % BLOCK_SIZE_N == 0:
            b_quantized = tle_async.load(b_block_ptr, is_async=True)
        else:
            b_quantized = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
        weight = dequant_weight(
            b_quantized,
            b_scale_ptr,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        acc = tl.dot(tl.trans(weight), tl.trans(activation), acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))
    return acc


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    A_ROUTE_DIVISOR: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    # One program computes one routed row and one output-N tile.  A
    # 16-row activation tile is used because PPU AIU requires a regular
    # 2D tile; boundary padding leaves only row zero valid.
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    route = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    a_row = route // A_ROUTE_DIVISOR
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + a_row * stride_am,
        shape=(1, reduction_size),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = quantized_gemm_tile(
        a_block_ptr,
        b_block_ptr,
        b_scale_ptr,
        expert,
        offs_n,
        stride_bse,
        stride_bsg,
        stride_bsn,
        num_columns,
        reduction_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        GROUP_SIZE_K,
        PIPELINE_STAGES,
        compute_type,
        QUANT_TYPE,
    )

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        acc *= routed_weight

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_rows = route + offs_m[None, :]
    c_cols = offs_n[:, None]
    c_ptrs = c_ptr + c_rows * stride_cm + c_cols * stride_cn
    tl.store(
        c_ptrs,
        acc.to(compute_type),
        mask=(offs_m[None, :] == 0) & (offs_n[:, None] < num_columns),
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_matvec_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    A_ROUTE_DIVISOR: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    SPLIT_STRIDE: tl.constexpr = 0,
):
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    route = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    a_row = route // A_ROUTE_DIVISOR
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(
            (
                (
                    tl.program_id(1)
                    * tl.cdiv(tl.cdiv(reduction_size, 128), SPLIT_K)
                    * 128
                )
                if SPLIT_K > 1
                else 0
            ),
            pid_n * BLOCK_SIZE_N,
        ),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    if SPLIT_K > 1:
        chunk: tl.constexpr = tl.cdiv(tl.cdiv(reduction_size, BLOCK_SIZE_K), SPLIT_K)
        begin = tl.program_id(1) * chunk
        end = tl.minimum(begin + chunk, tl.cdiv(reduction_size, BLOCK_SIZE_K))
    else:
        begin = 0
        end = tl.cdiv(reduction_size, BLOCK_SIZE_K)
    for k_tile in tl.range(begin, end, num_stages=PIPELINE_STAGES):
        a = tl.load(
            a_ptr + a_row * stride_am + (k_tile * BLOCK_SIZE_K + offs_k) * stride_ak,
            mask=k_tile * BLOCK_SIZE_K + offs_k < reduction_size,
            other=0.0,
        )
        if reduction_size % BLOCK_SIZE_K == 0 and num_columns % BLOCK_SIZE_N == 0:
            b_quantized = tle_async.load(b_block_ptr, is_async=True)
        else:
            b_quantized = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
        b = dequant_weight(
            b_quantized,
            b_scale_ptr,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        acc += tl.sum(a[:, None].to(tl.float32) * b.to(tl.float32), axis=0)
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))

    if MUL_ROUTED_WEIGHT and SPLIT_K == 1:
        acc *= tl.load(topk_weights_ptr + route).to(tl.float32)
    tl.store(
        c_ptr
        + (tl.program_id(1) * SPLIT_STRIDE if SPLIT_K > 1 else 0)
        + route * stride_cm
        + offs_n * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=offs_n < num_columns,
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_reduce_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    TOP_K: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    """Compute GEMM2 and reduce all routed experts into one token row."""
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    token = pid // num_pid_n
    pid_n = pid % num_pid_n
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for topk_index in tl.range(0, TOP_K):
        route = token * TOP_K + topk_index
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        route_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + route * stride_am,
            shape=(1, reduction_size),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(
                (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
                num_columns,
            ),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
            order=(0, 1),
        )

        for k_tile in tl.range(
            0, tl.cdiv(reduction_size, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
        ):
            a = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_quantized = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )

            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            b = dequant_weight(
                b_quantized,
                b_scale_ptr,
                expert,
                k_tile * BLOCK_SIZE_K,
                offs_n,
                stride_bse,
                stride_bsg,
                stride_bsn,
                num_columns,
                reduction_size,
                compute_type,
                GROUP_SIZE_K,
                QUANT_TYPE,
            )
            route_acc = tl.dot(tl.trans(b), tl.trans(a), acc=route_acc)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
            route_acc *= routed_weight
        acc += route_acc

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (token + offs_m[None, :]) * stride_cm + offs_n[:, None] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(compute_type),
        mask=(offs_m[None, :] == 0) & (offs_n[:, None] < num_columns),
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_silu_direct_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    A_ROUTE_DIVISOR: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    BLOCK_SIZE_M: tl.constexpr = 16
    BLOCK_SIZE_K: tl.constexpr = 128
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2

    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.cdiv(num_columns, BLOCK_SIZE_N)
    route = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert = tl.load(topk_ids_ptr + route).to(tl.int64)
    a_row = route // A_ROUTE_DIVISOR
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + a_row * stride_am,
        shape=(1, reduction_size),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_gate_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            2 * num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    b_up_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            2 * num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, num_columns + pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    acc_gate = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for k_tile in tl.range(
        0, tl.cdiv(reduction_size, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
    ):
        a = tle_async.load(
            a_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )
        b_gate_quantized = tle_async.load(
            b_gate_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )
        b_up_quantized = tle_async.load(
            b_up_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
            is_async=True,
        )

        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        gate_b = dequant_weight(
            b_gate_quantized,
            b_scale_ptr,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        up_b = dequant_weight(
            b_up_quantized,
            b_scale_ptr + num_columns * stride_bsn,
            expert,
            k_tile * BLOCK_SIZE_K,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            compute_type,
            GROUP_SIZE_K,
            QUANT_TYPE,
        )
        a_trans = tl.trans(a)
        acc_gate = tl.dot(tl.trans(gate_b), a_trans, acc=acc_gate)
        acc_up = tl.dot(tl.trans(up_b), a_trans, acc=acc_up)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_gate_block_ptr = tl.advance(b_gate_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))
        b_up_block_ptr = tl.advance(b_up_block_ptr, (BLOCK_SIZE_K_WEIGHT, 0))

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        acc_gate *= routed_weight
        acc_up *= routed_weight

    acc = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (route + offs_m[None, :]) * stride_cm + offs_n[:, None] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(compute_type),
        mask=(offs_m[None, :] == 0) & (offs_n[:, None] < num_columns),
    )


@libentry()
@triton.jit
def stage_routed_activations_kernel(
    a_ptr,
    routed_a_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    EM: tl.constexpr,
    reduction_size: tl.constexpr,
    num_valid_tokens,
    stride_am,
    stride_ak,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DEINTERLEAVE_K: tl.constexpr = False,
    PERMUTE_K16: tl.constexpr = False,
    PERMUTE_K32: tl.constexpr = False,
    SKIP_INACTIVE: tl.constexpr = False,
):
    """Gather expert-sorted rows into a contiguous AIU input matrix."""
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if SKIP_INACTIVE:
        # Raw W8 only consumes complete tiles below the padded route count.
        if tl.program_id(0) * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    route_mask = (offs_m < EM) & (offs_m < num_tokens_post_padded)
    routed_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=route_mask,
        other=num_valid_tokens,
    ).to(tl.int64)
    valid = route_mask[:, None] & (routed_token[:, None] < num_valid_tokens)
    activation = tl.load(
        a_ptr
        + (routed_token[:, None] // top_k) * stride_am
        + offs_k[None, :] * stride_ak,
        mask=valid & (offs_k[None, :] < reduction_size),
        other=0.0,
    )
    store_offset = offs_m[:, None].to(tl.int64) * reduction_size + offs_k[None, :]
    if PERMUTE_K32:
        tl.static_assert(
            reduction_size % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K
        )
        store_offset = (
            offs_m[:, None].to(tl.int64) * reduction_size + store_k32(offs_k)[None, :]
        )
    elif PERMUTE_K16:
        tl.static_assert(reduction_size % 16 == 0 and not DEINTERLEAVE_K)
        store_offset = (
            offs_m[:, None].to(tl.int64) * reduction_size + store_k16(offs_k)[None, :]
        )
    elif DEINTERLEAVE_K:
        store_offset = (
            offs_m[:, None].to(tl.int64) * (reduction_size // 2)
            + offs_k[None, :] // 2
            + (offs_k[None, :] % 2).to(tl.int64) * EM * (reduction_size // 2)
        )
    tl.store(
        routed_a_ptr + store_offset,
        activation,
        mask=(offs_m[:, None] < EM) & (offs_k[None, :] < reduction_size),
    )


@libentry()
@triton.jit
def silu_and_stage_routed_kernel(
    intermediate1_ptr,
    routed_intermediate2_ptr,
    sorted_token_ids_ptr,
    num_tokens_post_padded_ptr,
    EM: tl.constexpr,
    num_columns: tl.constexpr,
    num_valid_tokens,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    DEINTERLEAVE_K: tl.constexpr = False,
    PERMUTE_K16: tl.constexpr = False,
    PERMUTE_K32: tl.constexpr = False,
    SKIP_INACTIVE: tl.constexpr = False,
    SKIP_UNUSED_UPPER_ROWS: tl.constexpr = False,
):
    """Fuse SwiGLU with the expert-sorted layout required by GEMM2.

    A separate SiLU call followed by ``_stage_routed_activations_kernel``
    would add one launch and a full intermediate write/read cycle.
    """
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    start = tl.program_id(0) * BLOCK_SIZE_M
    if SKIP_INACTIVE:
        # Raw W8 only consumes complete tiles below the padded route count.
        if start >= num_tokens_post_padded:
            return
    if SKIP_UNUSED_UPPER_ROWS:
        tl.static_assert(BLOCK_SIZE_M == 8)
        if start % 16 == 8:
            # The down GEMM bypasses this subtile if its first row is padding.
            first = tl.load(
                sorted_token_ids_ptr + start,
                (start < EM) & (start < num_tokens_post_padded),
                other=num_valid_tokens,
            )
            if first >= num_valid_tokens:
                return
    offs_m = start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    route_mask = (offs_m < EM) & (offs_m < num_tokens_post_padded)
    routed_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=route_mask,
        other=num_valid_tokens,
    ).to(tl.int64)
    valid = (
        route_mask[:, None]
        & (routed_token[:, None] < num_valid_tokens)
        & (offs_n[None, :] < num_columns)
    )
    row_base = routed_token[:, None] * (2 * num_columns)
    gate = tl.load(
        intermediate1_ptr + row_base + offs_n[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        intermediate1_ptr + row_base + num_columns + offs_n[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    output = (gate * tl.sigmoid(gate)) * up
    store_offset = offs_m[:, None].to(tl.int64) * num_columns + offs_n[None, :]
    if PERMUTE_K32:
        tl.static_assert(
            num_columns % 32 == 0 and not PERMUTE_K16 and not DEINTERLEAVE_K
        )
        store_offset = (
            offs_m[:, None].to(tl.int64) * num_columns + store_k32(offs_n)[None, :]
        )
    elif PERMUTE_K16:
        tl.static_assert(num_columns % 16 == 0 and not DEINTERLEAVE_K)
        store_offset = (
            offs_m[:, None].to(tl.int64) * num_columns + store_k16(offs_n)[None, :]
        )
    elif DEINTERLEAVE_K:
        store_offset = (
            offs_m[:, None].to(tl.int64) * (num_columns // 2)
            + offs_n[None, :] // 2
            + (offs_n[None, :] % 2).to(tl.int64) * EM * (num_columns // 2)
        )
    tl.store(
        routed_intermediate2_ptr + store_offset,
        output,
        mask=(offs_m[:, None] < EM) & (offs_n[None, :] < num_columns),
    )


@libentry()
@triton.jit(do_not_specialize_on_alignment=["routed_a_ptr", "b_ptr", "c_ptr"])
def marlin_moe_gemm_grouped_kernel(
    routed_a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    EM: tl.constexpr,
    num_valid_tokens,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsg,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    compute_type: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    """Expert-grouped quantized GEMM using contiguous TLE AIU transfers."""
    IS_8BIT: tl.constexpr = (
        QUANT_TYPE == _TL_QUANT_TYPE_UINT8B128 or QUANT_TYPE == _TL_QUANT_TYPE_FP8_E4M3
    )
    BLOCK_SIZE_K_WEIGHT: tl.constexpr = BLOCK_SIZE_K if IS_8BIT else BLOCK_SIZE_K // 2
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(num_columns, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    pid_m, pid_n = tl.swizzle2d(
        pid // num_pid_n, pid % num_pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M
    )

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    routed_token = tl.load(
        sorted_token_ids_ptr + offs_m,
        mask=offs_m < EM,
        other=num_valid_tokens,
    ).to(tl.int64)
    token_mask = (offs_m < EM) & (routed_token < num_valid_tokens)

    a_block_ptr = tl.make_block_ptr(
        base=routed_a_ptr,
        shape=(EM, reduction_size),
        strides=(reduction_size, 1),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + expert * stride_be,
        shape=(
            (reduction_size if IS_8BIT else tl.cdiv(reduction_size, 2)),
            num_columns,
        ),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K_WEIGHT, BLOCK_SIZE_N),
        order=(0, 1),
    )
    if not IS_8BIT and compute_type == tl.bfloat16 and BLOCK_SIZE_M <= 32:
        acc = w4_pair_gemm(
            routed_a_ptr,
            b_block_ptr,
            b_scale_ptr,
            expert,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            EM,
            pid_m,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_K,
            PIPELINE_STAGES,
            QUANT_TYPE,
        )
    else:
        acc = quantized_gemm_tile(
            a_block_ptr,
            b_block_ptr,
            b_scale_ptr,
            expert,
            offs_n,
            stride_bse,
            stride_bsg,
            stride_bsn,
            num_columns,
            reduction_size,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            GROUP_SIZE_K,
            PIPELINE_STAGES,
            compute_type,
            QUANT_TYPE,
            BLOCK_SIZE_K,
        )

    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + routed_token,
            mask=token_mask,
            other=0.0,
        )
        acc *= routed_weight[None, :]

    c_ptrs = c_ptr + routed_token[None, :] * stride_cm + offs_n[:, None] * stride_cn
    if num_columns % BLOCK_SIZE_N == 0:
        tl.store(c_ptrs, acc.to(compute_type), mask=token_mask[None, :])
    else:
        tl.store(
            c_ptrs,
            acc.to(compute_type),
            mask=token_mask[None, :] & (offs_n[:, None] < num_columns),
        )


@libentry()
@triton.jit
def reduce_split_silu_kernel(
    partials,
    output_tile,
    router_weights,
    num_columns: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    MUL: tl.constexpr,
    BN: tl.constexpr,
):
    r = tl.program_id(0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    gate = tl.full((BN,), 0, tl.float32)
    up = tl.full((BN,), 0, tl.float32)
    for part in tl.static_range(SPLIT_COUNT):
        gate += tl.load(
            partials + part * NUM_ROUTES * 2 * num_columns + r * 2 * num_columns + ns,
            ns < num_columns,
            other=0,
        )
        up += tl.load(
            partials
            + part * NUM_ROUTES * 2 * num_columns
            + r * 2 * num_columns
            + num_columns
            + ns,
            ns < num_columns,
            other=0,
        )
    if MUL:
        w = tl.load(router_weights + r)
        gate *= w
        up *= w
    # Match the original GEMM1 rounding before computing SiLU.
    gate = gate.to(output_tile.dtype.element_ty).to(tl.float32)
    up = up.to(output_tile.dtype.element_ty).to(tl.float32)
    result = tl.fdiv(gate, 1.0 + tl.exp(-gate)) * up
    tl.store(output_tile + r * num_columns + ns, result, ns < num_columns)


@libentry()
@triton.jit
def stage_bf16_activation_pairs_kernel(
    activations,
    Out,
    SortedTokenIds,
    NumTokensPostPadded,
    EM: tl.constexpr,
    K_PAIRS: tl.constexpr,
    num_valid_tokens,
    stride_am_pairs,
    TOP_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SKIP_UNUSED_UPPER_ROWS: tl.constexpr = False,
    PERMUTE_K32: tl.constexpr = False,
):
    padded = tl.load(NumTokensPostPadded)
    start = tl.program_id(0) * BLOCK_SIZE_M
    if start >= padded:
        return
    m = (start + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    k = tl.program_id(1) * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    live = (m < EM) & (m < padded)
    token = tl.load(SortedTokenIds + m, live, num_valid_tokens).to(tl.int64)
    valid = live & (token < num_valid_tokens)
    a = activations.to(tl.pointer_type(tl.uint32))
    out = Out.to(tl.pointer_type(tl.uint32))
    pairs = tl.load(
        a + (token[:, None] // TOP_K) * stride_am_pairs + k[None, :],
        valid[:, None] & (k[None, :] < K_PAIRS),
        other=0,
    )
    # Permute whole BF16 pairs before the contiguous store.
    if PERMUTE_K32:
        grouped = tl.reshape(pairs, (BLOCK_SIZE_M, BLOCK_SIZE_K // 16, 4, 4))
        pairs = tl.reshape(
            tl.permute(grouped, (0, 1, 3, 2)), (BLOCK_SIZE_M, BLOCK_SIZE_K)
        )
    else:
        grouped = tl.reshape(pairs, (BLOCK_SIZE_M, BLOCK_SIZE_K // 8, 2, 2, 2))
        pairs = tl.reshape(
            tl.permute(grouped, (0, 1, 4, 2, 3)), (BLOCK_SIZE_M, BLOCK_SIZE_K)
        )
    write_rows = m < EM
    if SKIP_UNUSED_UPPER_ROWS:
        tl.static_assert(BLOCK_SIZE_M == 16)
        # The raw 16-row GEMM reads only eight rows when the ninth slot is padding.
        upper = start + 8
        upper_id = tl.load(
            SortedTokenIds + upper, (upper < EM) & (upper < padded), num_valid_tokens
        )
        write_rows &= (m < upper) | (upper_id < num_valid_tokens)
    tl.store(
        out + m[:, None] * K_PAIRS + k[None, :],
        pairs,
        write_rows[:, None] & (k[None, :] < K_PAIRS),
    )


@libentry()
@triton.jit
def tiny_fp8_down_reduce(
    Activation,
    Weight,
    Scale,
    route_ids,
    Router,
    Out,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    TOPK: tl.constexpr,
    SE,
    SN,
    SG,
    BN: tl.constexpr,
    MUL: tl.constexpr,
):
    token = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, reduction_size)
    total = tl.zeros((BN,), tl.float32)
    for j in tl.static_range(TOPK):
        route = token * TOPK + j
        expert = tl.load(route_ids + route).to(tl.int32)
        activation = tl.load(Activation + route * reduction_size + ks).to(tl.float32)
        quantized = tl.load(
            Weight
            + expert * (num_columns * reduction_size)
            + col[:, None] * reduction_size
            + ks[None, :]
        )
        scale = tl.load(
            Scale + expert * SE + col[:, None] * SN + (ks[None, :] // 128) * SG
        )
        weight = decode_fp8_scaled_bf16_normalized(quantized, scale)
        route_sum = tl.sum(activation[None, :] * weight.to(tl.float32), axis=1)
        if MUL:
            route_weight = tl.load(Router + route).to(tl.float32)
        else:
            route_weight = 1.0
        # Keep the BF16 boundary and route order of GEMM2 followed by moe_sum.
        total += (route_sum * route_weight).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + token * num_columns + col, total)
