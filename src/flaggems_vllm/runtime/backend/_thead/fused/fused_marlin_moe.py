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

"""T-Head PPU specialization for fused Marlin MoE W4A16 INT4."""

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl

try:
    from triton.experimental.tle import language as tle_async
except ImportError:  # pragma: no cover - requires a TLE-enabled FlagTree build
    tle_async = None

from flaggems_vllm import runtime
from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_UINT4B8,
    _RouterWeightPlacement,
    _invoke_w4a16_int4_moe_gemm,
    _invoke_w4a16_int4_moe_gemm_silu,
    _router_weight_placement,
    _select_w4a16_int4_kernel_policy,
    _stack_8,
    fused_marlin_moe as _generic_fused_marlin_moe,
    w4a16_int4_pack,
)
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.utils import libentry

_PPU_DIRECT_ROUTE_LIMIT = 32

@triton.jit
def _ppu_dequant_int4(b_packed, scale, compute_type: tl.constexpr):
    """Unpack PPU-friendly interleaved INT4 and apply one group scale.

    ``_pack_w_interleave`` stores eight logical K sub-tiles in the bit order
    0, 16, 4, 20, 8, 24, 12, 28.  Extracting in that order restores logical
    K order before ``_stack_8`` concatenates the sub-tiles.
    """
    b0 = (((b_packed >> 0) & 0xF).to(compute_type) - 8.0) * scale
    b1 = (((b_packed >> 16) & 0xF).to(compute_type) - 8.0) * scale
    b2 = (((b_packed >> 4) & 0xF).to(compute_type) - 8.0) * scale
    b3 = (((b_packed >> 20) & 0xF).to(compute_type) - 8.0) * scale
    b4 = (((b_packed >> 8) & 0xF).to(compute_type) - 8.0) * scale
    b5 = (((b_packed >> 24) & 0xF).to(compute_type) - 8.0) * scale
    b6 = (((b_packed >> 12) & 0xF).to(compute_type) - 8.0) * scale
    b7 = (((b_packed >> 28) & 0xF).to(compute_type) - 8.0) * scale
    return b0, b1, b2, b3, b4, b5, b6, b7


if tle_async is not None:

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
    def _ppu_w4a16_int4_moe_gemm_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
    ):
        # One program computes one routed row and one output-N tile.  A
        # 16-row activation tile is used because PPU AIU requires a regular
        # 2D tile; boundary padding leaves only row zero valid.
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = 16

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        route = pid // num_pid_n
        pid_n = pid % num_pid_n
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_row = route // A_ROUTE_DIVISOR
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + a_row * stride_am,
            shape=(1, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        # Packed B is physically and logically [E, K/8, N].
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(K // 8, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for k_tile in tl.range(
            0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
        ):
            a = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_packed = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )

            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            scale_group = k_tile * BLOCK_SIZE_K // GROUP_SIZE_K
            scale = tl.load(
                b_scale_ptr
                + expert * stride_bse
                + scale_group * stride_bsg
                + offs_n * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
            bs = _ppu_dequant_int4(b_packed, scale, compute_type)
            b = _stack_8(bs, BLOCK_SIZE_K_PACK, BLOCK_SIZE_N)
            acc = tl.dot(tl.trans(b), tl.trans(a), acc=acc)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_PACK, 0))

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
            mask=(offs_m[None, :] == 0) & (offs_n[:, None] < N),
        )

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
    def _ppu_w4a16_int4_moe_gemm_reduce_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
    ):
        """Compute GEMM2 and reduce all routed experts into one token row."""
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = 16

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        token = pid // num_pid_n
        pid_n = pid % num_pid_n
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for topk_index in tl.range(0, TOP_K):
            route = token * TOP_K + topk_index
            expert = tl.load(topk_ids_ptr + route).to(tl.int64)
            route_acc = tl.zeros(
                (BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32
            )
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr + route * stride_am,
                shape=(1, K),
                strides=(stride_am, stride_ak),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                order=(1, 0),
            )
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr + expert * stride_be,
                shape=(K // 8, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_SIZE_N),
                block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
                order=(1, 0),
            )

            for k_tile in tl.range(
                0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
            ):
                a = tle_async.load(
                    a_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
                b_packed = tle_async.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )

                offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                scale_group = k_tile * BLOCK_SIZE_K // GROUP_SIZE_K
                scale = tl.load(
                    b_scale_ptr
                    + expert * stride_bse
                    + scale_group * stride_bsg
                    + offs_n * stride_bsn,
                    mask=offs_n < N,
                    other=0.0,
                )[None, :]
                bs = _ppu_dequant_int4(b_packed, scale, compute_type)
                b = _stack_8(bs, BLOCK_SIZE_K_PACK, BLOCK_SIZE_N)
                route_acc = tl.dot(
                    tl.trans(b), tl.trans(a), acc=route_acc
                )

                a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
                b_block_ptr = tl.advance(
                    b_block_ptr, (BLOCK_SIZE_K_PACK, 0)
                )

            if MUL_ROUTED_WEIGHT:
                routed_weight = tl.load(topk_weights_ptr + route).to(
                    tl.float32
                )
                route_acc *= routed_weight
            acc += route_acc

        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + (token + offs_m[None, :]) * stride_cm
            + offs_n[:, None] * stride_cn
        )
        tl.store(
            c_ptrs,
            acc.to(compute_type),
            mask=(offs_m[None, :] == 0) & (offs_n[:, None] < N),
        )

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["a_ptr", "b_ptr", "c_ptr"])
    def _ppu_w4a16_int4_moe_gemm_silu_direct_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
        APPLY_ROUTER_WEIGHT_BEFORE_SILU: tl.constexpr,
        PIPELINE_STAGES: tl.constexpr,
        compute_type: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        BLOCK_SIZE_M: tl.constexpr = 16
        BLOCK_SIZE_K: tl.constexpr = 128
        BLOCK_SIZE_K_PACK: tl.constexpr = 16

        pid = tl.program_id(0)
        num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
        route = pid // num_pid_n
        pid_n = pid % num_pid_n
        expert = tl.load(topk_ids_ptr + route).to(tl.int64)
        a_row = route // A_ROUTE_DIVISOR
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + a_row * stride_am,
            shape=(1, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_gate_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(K // 8, 2 * N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
        )
        b_up_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(K // 8, 2 * N),
            strides=(stride_bk, stride_bn),
            offsets=(0, N + pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
        )
        acc_gate = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for k_tile in tl.range(
            0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
        ):
            a = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_gate_packed = tle_async.load(
                b_gate_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_up_packed = tle_async.load(
                b_up_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )

            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            scale_group = k_tile * BLOCK_SIZE_K // GROUP_SIZE_K
            scale_base = (
                b_scale_ptr
                + expert * stride_bse
                + scale_group * stride_bsg
                + offs_n * stride_bsn
            )
            scale_gate = tl.load(
                scale_base, mask=offs_n < N, other=0.0
            )[None, :]
            scale_up = tl.load(
                scale_base + N * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
            gate_bs = _ppu_dequant_int4(
                b_gate_packed, scale_gate, compute_type
            )
            up_bs = _ppu_dequant_int4(b_up_packed, scale_up, compute_type)
            gate_b = _stack_8(gate_bs, BLOCK_SIZE_K_PACK, BLOCK_SIZE_N)
            up_b = _stack_8(up_bs, BLOCK_SIZE_K_PACK, BLOCK_SIZE_N)
            a_trans = tl.trans(a)
            acc_gate = tl.dot(tl.trans(gate_b), a_trans, acc=acc_gate)
            acc_up = tl.dot(tl.trans(up_b), a_trans, acc=acc_up)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_gate_block_ptr = tl.advance(
                b_gate_block_ptr, (BLOCK_SIZE_K_PACK, 0)
            )
            b_up_block_ptr = tl.advance(
                b_up_block_ptr, (BLOCK_SIZE_K_PACK, 0)
            )

        if APPLY_ROUTER_WEIGHT_BEFORE_SILU:
            routed_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
            acc_gate *= routed_weight
            acc_up *= routed_weight

        acc = (acc_gate * tl.sigmoid(acc_gate)) * acc_up
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + (route + offs_m[None, :]) * stride_cm
            + offs_n[:, None] * stride_cn
        )
        tl.store(
            c_ptrs,
            acc.to(compute_type),
            mask=(offs_m[None, :] == 0) & (offs_n[:, None] < N),
        )


    @triton.jit
    def _ppu_stage_routed_activations_kernel(
        a_ptr,
        routed_a_ptr,
        sorted_token_ids_ptr,
        num_tokens_post_padded_ptr,
        EM: tl.constexpr,
        K: tl.constexpr,
        num_valid_tokens,
        stride_am,
        stride_ak,
        top_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """Gather expert-sorted rows into a contiguous AIU input matrix."""
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
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
            mask=valid & (offs_k[None, :] < K),
            other=0.0,
        )
        tl.store(
            routed_a_ptr + offs_m[:, None] * K + offs_k[None, :],
            activation,
            mask=(offs_m[:, None] < EM) & (offs_k[None, :] < K),
        )

    @triton.jit
    def _ppu_silu_and_stage_routed_kernel(
        intermediate1_ptr,
        routed_intermediate2_ptr,
        sorted_token_ids_ptr,
        num_tokens_post_padded_ptr,
        EM: tl.constexpr,
        N: tl.constexpr,
        num_valid_tokens,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        """Fuse SwiGLU with the expert-sorted layout required by GEMM2."""
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
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
            & (offs_n[None, :] < N)
        )
        row_base = routed_token[:, None] * (2 * N)
        gate = tl.load(
            intermediate1_ptr + row_base + offs_n[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            intermediate1_ptr + row_base + N + offs_n[None, :],
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        output = (gate * tl.sigmoid(gate)) * up
        tl.store(
            routed_intermediate2_ptr + offs_m[:, None] * N + offs_n[None, :],
            output,
            mask=(offs_m[:, None] < EM) & (offs_n[None, :] < N),
        )

    @libentry()
    @triton.jit(
        do_not_specialize_on_alignment=["routed_a_ptr", "b_ptr", "c_ptr"]
    )
    def _ppu_w4a16_int4_moe_gemm_grouped_kernel(
        routed_a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
    ):
        """Expert-grouped W4A16 GEMM using contiguous TLE AIU transfers."""
        BLOCK_SIZE_K_PACK: tl.constexpr = BLOCK_SIZE_K // 8
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        routed_token = tl.load(
            sorted_token_ids_ptr + offs_m,
            mask=offs_m < EM,
            other=num_valid_tokens,
        ).to(tl.int64)
        token_mask = (offs_m < EM) & (routed_token < num_valid_tokens)
        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        if expert == -1:
            zeros = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=compute_type)
            tl.store(
                c_ptr
                + routed_token[None, :] * stride_cm
                + offs_n[:, None] * stride_cn,
                zeros,
                mask=token_mask[None, :] & (offs_n[:, None] < N),
            )
            return

        a_block_ptr = tl.make_block_ptr(
            base=routed_a_ptr,
            shape=(EM, K),
            strides=(K, 1),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + expert * stride_be,
            shape=(K // 8, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K_PACK, BLOCK_SIZE_N),
            order=(1, 0),
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for k_tile in tl.range(
            0, tl.cdiv(K, BLOCK_SIZE_K), num_stages=PIPELINE_STAGES
        ):
            activation = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b_packed = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            scale_idx = k_tile * BLOCK_SIZE_K // GROUP_SIZE_K
            scale = tl.load(
                b_scale_ptr
                + expert * stride_bse
                + scale_idx * stride_bsg
                + offs_n * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            )[None, :]
            bs = _ppu_dequant_int4(b_packed, scale, compute_type)
            b = _stack_8(bs, BLOCK_SIZE_K_PACK, BLOCK_SIZE_N)
            acc = tl.dot(tl.trans(b), tl.trans(activation), acc=acc)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K_PACK, 0))

        if MUL_ROUTED_WEIGHT:
            routed_weight = tl.load(
                topk_weights_ptr + routed_token,
                mask=token_mask,
                other=0.0,
            )
            acc *= routed_weight[None, :]

        tl.store(
            c_ptr
            + routed_token[None, :] * stride_cm
            + offs_n[:, None] * stride_cn,
            acc.to(compute_type),
            mask=token_mask[None, :] & (offs_n[:, None] < N),
        )
def _select_ppu_direct_block_n(n: int) -> int:
    if n <= 32:
        return 32
    if n < 512:
        return 64
    return 128


def _select_ppu_grouped_config(M: int, K: int, N: int):
    if 256 <= M <= 512 and K > N:
        return 128, 1, 4, 3
    if M < 512:
        return 256, 1, 8, 3
    if M == 2048 and K > N:
        return 256, 8, 8, 2
    return 256, 8, 8, 3


def _select_ppu_grouped_block_m(M: int, E: int, top_k: int) -> int:
    routes = M * top_k
    if routes <= 16 * E:
        return 16
    if routes <= 32 * E:
        return 32
    return 64


def _use_ppu_direct_route(
    M: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    routes = M * top_k
    max_output_n = max(hidden_size, 2 * intermediate_size)
    block_n = _select_ppu_direct_block_n(max_output_n)
    n_tiles = triton.cdiv(max_output_n, block_n)
    max_routes_by_grid = 65535 // n_tiles
    return routes <= min(_PPU_DIRECT_ROUTE_LIMIT, max_routes_by_grid)


def _align_ppu_grouped_tokens(
    topk_ids: torch.Tensor,
    block_m: int,
    num_experts: int,
):
    # FlagGems-vLLM is integrated into vLLM, and the T-Head CUDA extension's
    # alignment kernel avoids the high fixed cost of the generic Triton path.
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as vllm_moe_align_block_size,
    )

    return vllm_moe_align_block_size(
        topk_ids,
        block_m,
        num_experts,
        expert_map=None,
        ignore_invalid_experts=True,
    )


def _invoke_ppu_w4a16_int4_moe_gemm_direct(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    a_route_divisor: int,
    group_size: int,
    compute_type,
):
    if tle_async is None:
        raise RuntimeError("PPU W4A16 AIU path requires Triton TLE")

    K = A.size(1)
    N = B.size(2)
    routes = topk_ids.numel()
    if C.ndim == 3:
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)
    block_n = _select_ppu_direct_block_n(N)
    # The PPU pipeline pass allocates ``num_stages - 1`` loop buffers.  Three
    # scheduling stages therefore provide the two buffers required to overlap
    # the next AIU copy with the current tile's unpack/dequantize/dot work.
    pipeline_stages = 3 if K > 128 else 1
    grid = (routes * triton.cdiv(N, block_n),)

    _ppu_w4a16_int4_moe_gemm_direct_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        stride_cm,
        stride_cn,
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        A_ROUTE_DIVISOR=a_route_divisor,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
    )


def _invoke_ppu_w4a16_int4_moe_gemm_reduce_direct(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    group_size: int,
    compute_type,
):
    if tle_async is None:
        raise RuntimeError("PPU W4A16 AIU path requires Triton TLE")

    K = A.size(1)
    N = B.size(2)
    top_k = topk_ids.size(1)
    tokens = A.size(0) // top_k
    if tokens == 1:
        block_n = 128
    elif tokens == 2:
        block_n = 32
    else:
        block_n = 32
    pipeline_stages = 3 if K > 128 else 1
    grid = (tokens * triton.cdiv(N, block_n),)

    _ppu_w4a16_int4_moe_gemm_reduce_direct_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        TOP_K=top_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
    )


def _invoke_ppu_w4a16_int4_moe_gemm_silu_direct(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    apply_router_weight_before_silu: bool,
    a_route_divisor: int,
    group_size: int,
    compute_type,
):
    if tle_async is None:
        raise RuntimeError("PPU W4A16 AIU path requires Triton TLE")

    K = A.size(1)
    N = C.size(1)
    routes = topk_ids.numel()
    block_n = 32 if routes >= 12 else _select_ppu_direct_block_n(N)
    pipeline_stages = 3 if K > 128 else 1
    grid = (routes * triton.cdiv(N, block_n),)

    _ppu_w4a16_int4_moe_gemm_silu_direct_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        topk_ids,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        B_scale.stride(0),
        B_scale.stride(1),
        B_scale.stride(2),
        A_ROUTE_DIVISOR=a_route_divisor,
        GROUP_SIZE_K=group_size,
        APPLY_ROUTER_WEIGHT_BEFORE_SILU=apply_router_weight_before_silu,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
    )


def _stage_ppu_grouped_activations(
    A: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    em: int,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
) -> torch.Tensor:
    routed_a = torch.empty((em, A.size(1)), dtype=A.dtype, device=A.device)
    grid = (triton.cdiv(em, block_m), triton.cdiv(A.size(1), 128))
    _ppu_stage_routed_activations_kernel[grid](
        A,
        routed_a,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        K=A.size(1),
        num_valid_tokens=num_valid_tokens,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=128,
        num_warps=4,
        num_stages=1,
    )
    return routed_a


def _silu_and_stage_ppu_grouped(
    intermediate1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    em: int,
    num_valid_tokens: int,
    block_m: int,
) -> torch.Tensor:
    n = intermediate1.size(1) // 2
    routed = torch.empty((em, n), dtype=intermediate1.dtype, device=intermediate1.device)
    grid = (triton.cdiv(em, block_m), triton.cdiv(n, 128))
    _ppu_silu_and_stage_routed_kernel[grid](
        intermediate1,
        routed,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        N=n,
        num_valid_tokens=num_valid_tokens,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=128,
        num_warps=4,
        num_stages=1,
    )
    return routed


def _invoke_ppu_w4a16_int4_moe_gemm_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    group_size: int,
    compute_type,
    input_is_routed: bool = False,
):
    if tle_async is None:
        raise RuntimeError("PPU W4A16 grouped AIU path requires Triton TLE")
    em = sorted_token_ids.size(0)
    num_valid_tokens = C.size(0) * C.size(1) if C.ndim == 3 else A.size(0) * top_k
    if input_is_routed:
        routed_a = A
    else:
        routed_a = _stage_ppu_grouped_activations(
            A,
            sorted_token_ids,
            num_tokens_post_padded,
            em=em,
            num_valid_tokens=num_valid_tokens,
            top_k=top_k,
            block_m=block_m,
        )

    n = B.size(2)
    batch_m = C.size(0) if C.ndim == 3 else A.size(0)
    block_n, group_m, num_warps, pipeline_stages = _select_ppu_grouped_config(
        batch_m, A.size(1), n
    )
    if C.ndim == 3:
        stride_cm = C.stride(1)
        stride_cn = C.stride(2)
    else:
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    _ppu_w4a16_int4_moe_gemm_grouped_kernel[grid](
        routed_a,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N=n,
        K=A.size(1),
        EM=em,
        num_valid_tokens=num_valid_tokens,
        stride_be=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        stride_bse=B_scale.stride(0),
        stride_bsg=B_scale.stride(1),
        stride_bsn=B_scale.stride(2),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=128,
        GROUP_SIZE_M=group_m,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        num_warps=num_warps,
        num_stages=pipeline_stages,
    )


def fused_marlin_moe_w4a16_int4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "silu",
    group_size: int = 128,
    apply_router_weight_on_input: bool = False,
    inplace: bool = False,
    swap_ab: bool = True,
) -> torch.Tensor:
    assert activation == "silu"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert hidden_states.is_contiguous()
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1

    M = hidden_states.size(0)
    K = hidden_states.size(1)
    E = w1.size(0)
    intermediate_size = w1.size(1) // 2
    top_k_num = topk_ids.size(1)

    assert w1.shape == (E, 2 * intermediate_size, K // 2)
    assert w2.shape == (E, K, intermediate_size // 2)
    assert K % group_size == 0
    assert intermediate_size % group_size == 0
    assert w1_scale.shape == (E, 2 * intermediate_size, K // group_size)
    assert w2_scale.shape == (E, K, intermediate_size // group_size)
    assert w1_scale.dtype == hidden_states.dtype
    assert w2_scale.dtype == hidden_states.dtype
    assert topk_weights.shape == topk_ids.shape

    block_size_k = group_size
    # Compute_type for the kernel.
    if hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.bfloat16

    w1_packed, w2_packed, w1_scale_packed, w2_scale_packed = w4a16_int4_pack(
        w1,
        w2,
        w1_scale,
        w2_scale,
        block_size_k=block_size_k,
        cached=True,
    )

    policy = _select_w4a16_int4_kernel_policy(
        hidden_states.device,
        M,
        E,
        top_k_num,
        swap_ab,
        apply_router_weight_on_input,
    )
    is_ppu = runtime.device.vendor_name == "thead"
    use_ppu_direct_route = (
        is_ppu
        and tle_async is not None
        and group_size == 128
        and _use_ppu_direct_route(M, top_k_num, K, intermediate_size)
    )
    use_ppu_reduce_direct = use_ppu_direct_route and M <= 2
    block_m = policy.block_m
    if is_ppu and not use_ppu_direct_route:
        block_m = _select_ppu_grouped_block_m(M, E, top_k_num)
    # Direct-route is faster for sparse routed-token batches. Larger PPU
    # batches stay on the expert-grouped W4A16 kernel. Do not force the fused
    # grouped variant: its two live accumulators increase register pressure.
    use_fused_gemm1_silu = policy.use_fused_gemm1_silu or use_ppu_direct_route
    move_router_weight_before_gemm2 = (
        policy.move_router_weight_before_gemm2
        or (
            is_ppu
            and use_fused_gemm1_silu
            and not apply_router_weight_on_input
            and M >= 512
        )
    )
    router_weight_placement = _router_weight_placement(
        apply_router_weight_on_input,
        move_router_weight_before_gemm2,
    )
    mul_routed_weight_in_gemm2 = router_weight_placement == _RouterWeightPlacement.none

    intermediate_cache1 = None
    intermediate_cache3 = None
    if not use_ppu_reduce_direct:
        cache13_size = M * top_k_num * K
        if not use_fused_gemm1_silu:
            cache13_size = max(
                cache13_size, M * top_k_num * 2 * intermediate_size
            )
        cache13 = torch.empty(
            cache13_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if not use_fused_gemm1_silu:
            intermediate_cache1 = cache13[
                : M * top_k_num * 2 * intermediate_size
            ].view(M * top_k_num, 2 * intermediate_size)
        intermediate_cache3 = cache13[: M * top_k_num * K].view(
            M, top_k_num, K
        )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    ppu_grouped_intermediate2 = None

    if use_ppu_direct_route:
        sorted_token_ids = expert_ids = num_tokens_post_padded = None
    elif is_ppu:
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _align_ppu_grouped_tokens(topk_ids, block_m, E)
        )
    else:
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=block_m,
            num_experts=E,
            expert_map=None,
        )

    if use_fused_gemm1_silu:
        if use_ppu_direct_route:
            _invoke_ppu_w4a16_int4_moe_gemm_silu_direct(
                A=hidden_states,
                B=w1_packed,
                C=intermediate_cache2,
                B_scale=w1_scale_packed,
                topk_weights=(
                    topk_weights if apply_router_weight_on_input else None
                ),
                topk_ids=topk_ids,
                apply_router_weight_before_silu=apply_router_weight_on_input,
                a_route_divisor=top_k_num,
                group_size=group_size,
                compute_type=compute_type,
            )
        else:
            _invoke_w4a16_int4_moe_gemm_silu(
                A=hidden_states,
                B=w1_packed,
                C=intermediate_cache2,
                B_scale=w1_scale_packed,
                topk_weights=(
                    topk_weights
                    if router_weight_placement != _RouterWeightPlacement.none
                    else None
                ),
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                router_weight=router_weight_placement,
                top_k=top_k_num,
                block_m=block_m,
                block_size_k=block_size_k,
                group_size=group_size,
                compute_type=compute_type,
                swap_ab=swap_ab,
            )
    else:
        assert intermediate_cache1 is not None
        if use_ppu_direct_route:
            _invoke_ppu_w4a16_int4_moe_gemm_direct(
                A=hidden_states,
                B=w1_packed,
                C=intermediate_cache1,
                B_scale=w1_scale_packed,
                topk_weights=(topk_weights if apply_router_weight_on_input else None),
                topk_ids=topk_ids,
                mul_routed_weight=apply_router_weight_on_input,
                a_route_divisor=top_k_num,
                group_size=group_size,
                compute_type=compute_type,
            )
        elif is_ppu:
            _invoke_ppu_w4a16_int4_moe_gemm_grouped(
                A=hidden_states,
                B=w1_packed,
                C=intermediate_cache1,
                B_scale=w1_scale_packed,
                topk_weights=(topk_weights if apply_router_weight_on_input else None),
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=apply_router_weight_on_input,
                top_k=top_k_num,
                block_m=block_m,
                group_size=group_size,
                compute_type=compute_type,
            )
        else:
            _invoke_w4a16_int4_moe_gemm(
                A=hidden_states,
                B=w1_packed,
                C=intermediate_cache1,
                B_scale=w1_scale_packed,
                topk_weights=topk_weights if apply_router_weight_on_input else None,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=apply_router_weight_on_input,
                top_k=top_k_num,
                block_m=block_m,
                block_size_k=block_size_k,
                group_size=group_size,
                compute_type=compute_type,
                swap_ab=swap_ab,
            )
        if is_ppu and not use_ppu_direct_route:
            ppu_grouped_intermediate2 = _silu_and_stage_ppu_grouped(
                intermediate_cache1,
                sorted_token_ids,
                num_tokens_post_padded,
                em=sorted_token_ids.size(0),
                num_valid_tokens=M * top_k_num,
                block_m=block_m,
            )
        else:
            gate = intermediate_cache1[:, :intermediate_size]
            up = intermediate_cache1[:, intermediate_size:]
            silu_and_mul_out(gate, up, intermediate_cache2)

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    if use_ppu_direct_route:
        if use_ppu_reduce_direct:
            _invoke_ppu_w4a16_int4_moe_gemm_reduce_direct(
                A=intermediate_cache2,
                B=w2_packed,
                C=out_hidden_states,
                B_scale=w2_scale_packed,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                mul_routed_weight=mul_routed_weight_in_gemm2,
                group_size=group_size,
                compute_type=compute_type,
            )
        else:
            assert intermediate_cache3 is not None
            _invoke_ppu_w4a16_int4_moe_gemm_direct(
                A=intermediate_cache2,
                B=w2_packed,
                C=intermediate_cache3,
                B_scale=w2_scale_packed,
                topk_weights=(
                    topk_weights if mul_routed_weight_in_gemm2 else None
                ),
                topk_ids=topk_ids,
                mul_routed_weight=mul_routed_weight_in_gemm2,
                a_route_divisor=1,
                group_size=group_size,
                compute_type=compute_type,
            )
    elif is_ppu:
        assert ppu_grouped_intermediate2 is not None
        assert intermediate_cache3 is not None
        _invoke_ppu_w4a16_int4_moe_gemm_grouped(
            A=ppu_grouped_intermediate2,
            B=w2_packed,
            C=intermediate_cache3,
            B_scale=w2_scale_packed,
            topk_weights=(topk_weights if mul_routed_weight_in_gemm2 else None),
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=mul_routed_weight_in_gemm2,
            top_k=1,
            block_m=block_m,
            group_size=group_size,
            compute_type=compute_type,
            input_is_routed=True,
        )
    else:
        assert intermediate_cache3 is not None
        _invoke_w4a16_int4_moe_gemm(
            A=intermediate_cache2,
            B=w2_packed,
            C=intermediate_cache3,
            B_scale=w2_scale_packed,
            topk_weights=topk_weights if mul_routed_weight_in_gemm2 else None,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=mul_routed_weight_in_gemm2,
            top_k=1,
            block_m=block_m,
            block_size_k=block_size_k,
            group_size=group_size,
            compute_type=compute_type,
            swap_ab=swap_ab,
        )

    if not use_ppu_reduce_direct:
        assert intermediate_cache3 is not None
        moe_sum(intermediate_cache3, out_hidden_states)

    return out_hidden_states




def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Use the PPU AIU path for supported W4A16 INT4 inputs."""

    activation_str = "silu"
    if activation is not None:
        for attr in ("value", "name"):
            value = getattr(activation, attr, None)
            if isinstance(value, str):
                activation_str = value.lower()
                break
        if isinstance(activation, str):
            activation_str = activation.lower()

    use_ppu_w4a16 = (
        tle_async is not None
        and quant_type_id == QUANT_TYPE_UINT4B8
        and activation_str == "silu"
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and w1.dtype == torch.uint8
        and w2.dtype == torch.uint8
        and bias1 is None
        and bias2 is None
        and w1_zeros is None
        and w2_zeros is None
        and expert_map is None
        and input_dtype is None
        and clamp_limit is None
        and input_global_scale1 is None
        and input_global_scale2 is None
        and global_scale1 is None
        and global_scale2 is None
        and g_idx1 is None
        and g_idx2 is None
        and sort_indices1 is None
        and sort_indices2 is None
        and (global_num_experts == -1 or global_num_experts == w1.size(0))
        and group_size == 128
        and w1_scale.dtype == hidden_states.dtype
        and w2_scale.dtype == hidden_states.dtype
    )
    if use_ppu_w4a16:
        if inplace and output is not None:
            raise ValueError("Cannot pass both inplace=True and output")
        result = fused_marlin_moe_w4a16_int4(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation_str,
            group_size=group_size,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
        )
        if output is not None:
            output.copy_(result)
            return output
        return result

    return _generic_fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        bias1=bias1,
        bias2=bias2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=quant_type_id,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        activation=activation,
        activation_func=activation_func,
        moe_sum=moe_sum,
        expert_map=expert_map,
        input_global_scale1=input_global_scale1,
        input_global_scale2=input_global_scale2,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        is_k_full=is_k_full,
        output=output,
        input_dtype=input_dtype,
        inplace=inplace,
        clamp_limit=clamp_limit,
        group_size=group_size,
    )


__all__ = ["fused_marlin_moe"]
