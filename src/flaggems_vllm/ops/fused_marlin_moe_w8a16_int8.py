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

# SPDX-License-Identifier: Apache-2.0
"""Fused Marlin MoE for group-wise INT8 weights and A16 activations."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
import triton
import triton.language as tl


class QuantMode(Enum):
    """Quantization modes supported by QC-MoE."""

    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    W8A16 = "w8a16"  # INT8 weight, FP16 activation
    W4A16 = "w4a16"  # INT4 weight, FP16 activation


@dataclass
class QuantConfig:
    """Configuration for MoE quantization."""

    mode: QuantMode = QuantMode.FP16
    group_size: int = 128
    has_zero_point: bool = True
    per_channel_quant: bool = False

    @property
    def w_nbits(self) -> int:
        """Get weight bit width from mode."""
        if self.mode == QuantMode.W4A16:
            return 4
        elif self.mode in (QuantMode.W8A16, QuantMode.INT8, QuantMode.FP8):
            return 8
        return 16

    @property
    def use_int4(self) -> bool:
        return self.mode == QuantMode.W4A16

    @property
    def use_int8(self) -> bool:
        return self.mode in (QuantMode.W8A16, QuantMode.INT8)


def _build_w8a16_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_warps=4,
            num_stages=1,
            maxnreg=64,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_warps=4,
            num_stages=1,
            maxnreg=80,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
            num_warps=4,
            num_stages=1,
            maxnreg=96,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=8, num_stages=1
        ),
    ]


_W8A16_AUTOTUNE_CONFIGS = _build_w8a16_autotune_configs()


def _build_w8a16_fused_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=64,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=80,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=96,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
    ]


_W8A16_FUSED_AUTOTUNE_CONFIGS = _build_w8a16_fused_autotune_configs()


def _mxq_b2_autotune_prepin_enabled() -> bool:
    return True


def _prepend_b2_mid_autotune_configs(configs: list) -> list:
    """Put 128x128 / 64x128 winners first — faster autotune for T=64～512."""
    if not _mxq_b2_autotune_prepin_enabled():
        return configs
    pins = [
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
    ]
    keys = {
        (
            c.kwargs.get("BLOCK_SIZE_N"),
            c.kwargs.get("BLOCK_SIZE_K"),
            c.num_warps,
            c.num_stages,
        )
        for c in configs
    }
    prefix = []
    for cfg in pins:
        key = (
            cfg.kwargs.get("BLOCK_SIZE_N"),
            cfg.kwargs.get("BLOCK_SIZE_K"),
            cfg.num_warps,
            cfg.num_stages,
        )
        if key not in keys:
            prefix.append(cfg)
            keys.add(key)
    return prefix + configs


def _build_w8a16_fused_large_autotune_configs():
    base = [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
    ]
    return _prepend_b2_mid_autotune_configs(base)


_W8A16_FUSED_LARGE_AUTOTUNE_CONFIGS = _build_w8a16_fused_large_autotune_configs()


def _build_w8a16_down_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=64,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=80,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=96,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_warps=4,
            num_stages=1,
            maxnreg=96,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1
        ),
    ]


_W8A16_DOWN_AUTOTUNE_CONFIGS = _build_w8a16_down_autotune_configs()


def _build_w8a16_unified_moe_autotune_configs():
    """Autotune for ``*_unified_moe`` (MI grid, T<=MI_MAX only).

    Six candidates match 170907 / 183339 repro (T=1 Gems ~0.137 ms).  Larger
    search spaces slow autotune and can pick worse tiles on T=1.
    """
    return [
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_I_TILE": 32, "BLOCK_K_H": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_I_TILE": 32, "BLOCK_K_H": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_I_TILE": 32, "BLOCK_K_H": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 64, "BLOCK_I_TILE": 64, "BLOCK_K_H": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_I_TILE": 64, "BLOCK_K_H": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_N": 128, "BLOCK_I_TILE": 128, "BLOCK_K_H": 128},
            num_warps=8,
            num_stages=1,
        ),
    ]


_W8A16_UNIFIED_MOE_AUTOTUNE_CONFIGS = _build_w8a16_unified_moe_autotune_configs()


_BSKS_UNIFIED_MOE_KH = {
    c.kwargs["BLOCK_K_H"] for c in _W8A16_UNIFIED_MOE_AUTOTUNE_CONFIGS
}


_BSKS_UNIFIED_MOE_IT = {
    c.kwargs["BLOCK_I_TILE"] for c in _W8A16_UNIFIED_MOE_AUTOTUNE_CONFIGS
}


def _use_unified_moe_kernel() -> bool:
    return True


@triton.autotune(
    configs=_W8A16_AUTOTUNE_CONFIGS,
    key=["M_padded", "Nw1", "H", "T"],
)
@triton.jit
def fused_moe_kernel_w8a16_gateup(
    A,  # (T, H) bf16, indexed by token_id
    W1_q,  # (E, Nw1, H) uint8
    W1_scales,  # (E, Nw1, H_groups) bf16
    W1_zp,  # (E, Nw1, H_groups) uint8 or empty
    GATEUP,  # (M_padded, Nw1) bf16, output indexed by dispatch_idx
    sorted_token_ids,
    expert_ids_per_block,
    M_padded,
    T,
    Nw1,
    H,
    stride_a_t,
    stride_a_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_s_e,
    stride_s_n,
    stride_s_k,
    stride_zp_e,
    stride_zp_n,
    stride_zp_k,
    stride_gu_m,
    stride_gu_n,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    has_zp: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
    SWAP_AB: tl.constexpr,
    compute_type: tl.constexpr,
):
    """gate_up = W1[expert] @ x, written to GATEUP[dispatch_idx, :]. Full N coverage."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < Nw1

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_start in range(0, H, BLOCK_SIZE_K):
        k_indices = k_start + offs_k
        k_mask = k_indices < H
        if even_Ks:
            if SWAP_AB:
                a = tl.load(
                    A
                    + token_ids[None, :] * stride_a_t
                    + k_indices[:, None] * stride_a_k,
                    mask=token_mask[None, :],
                    other=0.0,
                )
            else:
                a = tl.load(
                    A
                    + token_ids[:, None] * stride_a_t
                    + k_indices[None, :] * stride_a_k,
                    mask=token_mask[:, None],
                    other=0.0,
                )
            b_int = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + offs_n[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=n_mask[:, None],
                other=128,
            ).to(tl.float32)
        else:
            if SWAP_AB:
                a = tl.load(
                    A
                    + token_ids[None, :] * stride_a_t
                    + k_indices[:, None] * stride_a_k,
                    mask=token_mask[None, :] & k_mask[:, None],
                    other=0.0,
                )
            else:
                a = tl.load(
                    A
                    + token_ids[:, None] * stride_a_t
                    + k_indices[None, :] * stride_a_k,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
            b_int = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + offs_n[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=n_mask[:, None] & k_mask[None, :],
                other=128,
            ).to(tl.float32)

        group_idx = k_start // group_size
        s = tl.load(
            W1_scales
            + expert_id * stride_s_e
            + offs_n * stride_s_n
            + group_idx * stride_s_k,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        if has_zp:
            zp = tl.load(
                W1_zp
                + expert_id * stride_zp_e
                + offs_n * stride_zp_n
                + group_idx * stride_zp_k,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            b_deq = (b_int - zp[:, None]) * s[:, None]
        else:
            b_deq = (b_int - 128.0) * s[:, None]

        if SWAP_AB:
            accumulator += tl.dot(b_deq.to(a.dtype), a)
        else:
            accumulator += tl.dot(a, tl.trans(b_deq.to(a.dtype)))

    if SWAP_AB:
        out_ptrs = (
            GATEUP + offs_m[None, :] * stride_gu_m + offs_n[:, None] * stride_gu_n
        )
        out_mask = token_mask[None, :] & n_mask[:, None]
    else:
        out_ptrs = (
            GATEUP + offs_m[:, None] * stride_gu_m + offs_n[None, :] * stride_gu_n
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, accumulator.to(compute_type), mask=out_mask)


@triton.jit
def silu_mul_kernel(
    GATEUP,  # (M_padded, 2*I) bf16
    INTER,  # (M_padded, I) bf16
    sorted_token_ids,
    sorted_weights,
    M_padded,
    T,
    I,
    stride_gu_m,
    stride_gu_n,
    stride_inter_m,
    stride_inter_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    APPLY_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """SwiGLU: intermediate[m, i] = silu(gate_up[m, i]) * gate_up[m, i + I]."""
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)

    m_mask = offs_m < M_padded
    token_ids = tl.load(sorted_token_ids + offs_m, mask=m_mask, other=T)
    m_mask = m_mask & (token_ids < T)
    i_mask = offs_i < I
    full_mask = m_mask[:, None] & i_mask[None, :]

    gate_ptr = GATEUP + offs_m[:, None] * stride_gu_m + offs_i[None, :] * stride_gu_n
    up_ptr = (
        GATEUP + offs_m[:, None] * stride_gu_m + (offs_i + I)[None, :] * stride_gu_n
    )

    gate = tl.load(gate_ptr, mask=full_mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr, mask=full_mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up
    if APPLY_ROUTED_WEIGHT:
        route_weights = tl.load(sorted_weights + offs_m, mask=m_mask, other=0.0).to(
            tl.float32
        )
        result *= route_weights[:, None]

    out_ptr = (
        INTER + offs_m[:, None] * stride_inter_m + offs_i[None, :] * stride_inter_n
    )
    tl.store(out_ptr, result.to(compute_type), mask=full_mask)


@triton.autotune(
    configs=_W8A16_FUSED_AUTOTUNE_CONFIGS,
    key=["M_padded", "I", "H", "T"],
)
@triton.jit
def fused_moe_kernel_w8a16_gateup_silu(
    A,  # (T, H) bf16, indexed by token_id
    W1_q,  # (E, 2*I, H) uint8 — first I rows: gate, last I rows: up
    W1_scales,  # (E, 2*I, H_groups) bf16
    W1_zp,  # (E, 2*I, H_groups) uint8 or empty
    INTER,  # (M_padded, I) bf16, fused output (silu(gate) * up)
    sorted_token_ids,
    expert_ids_per_block,
    sorted_weights,
    M_padded,
    T,
    I,  # half of Nw1; gate is rows [0,I), up is rows [I,2I)
    H,
    stride_a_t,
    stride_a_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_s_e,
    stride_s_n,
    stride_s_k,
    stride_zp_e,
    stride_zp_n,
    stride_zp_k,
    stride_inter_m,
    stride_inter_n,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    has_zp: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
    APPLY_ROUTED_WEIGHT: tl.constexpr,
    SWAP_AB: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Fused gate-up GEMM + SwiGLU (Optimization B2).

    Output shape is (M_padded, I), i.e. ONLY the intermediate dim — gate_up
    buffer is never materialized.  Writes silu(gate_acc) * up_acc directly.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < I
    up_offs_n = offs_n + I  # up rows live at [I, 2I) along the N axis of W1

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    if SWAP_AB:
        gate_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_start in range(0, H, BLOCK_SIZE_K):
        k_indices = k_start + offs_k
        k_mask = k_indices < H
        if even_Ks:
            if SWAP_AB:
                a = tl.load(
                    A
                    + token_ids[None, :] * stride_a_t
                    + k_indices[:, None] * stride_a_k,
                    mask=token_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
            else:
                a = tl.load(
                    A
                    + token_ids[:, None] * stride_a_t
                    + k_indices[None, :] * stride_a_k,
                    mask=token_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
            b_int_gate = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + offs_n[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=n_mask[:, None],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
            b_int_up = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + up_offs_n[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=n_mask[:, None],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            if SWAP_AB:
                a = tl.load(
                    A
                    + token_ids[None, :] * stride_a_t
                    + k_indices[:, None] * stride_a_k,
                    mask=token_mask[None, :] & k_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
            else:
                a = tl.load(
                    A
                    + token_ids[:, None] * stride_a_t
                    + k_indices[None, :] * stride_a_k,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
            b_int_gate = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + offs_n[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=n_mask[:, None] & k_mask[None, :],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
            b_int_up = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + up_offs_n[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=n_mask[:, None] & k_mask[None, :],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)

        if group_size >= BLOCK_SIZE_K and (group_size % BLOCK_SIZE_K) == 0:
            group_idx = k_start // group_size
            s_gate = tl.load(
                W1_scales
                + expert_id * stride_s_e
                + offs_n * stride_s_n
                + group_idx * stride_s_k,
                mask=n_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            s_up = tl.load(
                W1_scales
                + expert_id * stride_s_e
                + up_offs_n * stride_s_n
                + group_idx * stride_s_k,
                mask=n_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            if has_zp:
                zp_gate = tl.load(
                    W1_zp
                    + expert_id * stride_zp_e
                    + offs_n * stride_zp_n
                    + group_idx * stride_zp_k,
                    mask=n_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                zp_up = tl.load(
                    W1_zp
                    + expert_id * stride_zp_e
                    + up_offs_n * stride_zp_n
                    + group_idx * stride_zp_k,
                    mask=n_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                b_deq_gate = (b_int_gate - zp_gate[:, None]) * s_gate[:, None]
                b_deq_up = (b_int_up - zp_up[:, None]) * s_up[:, None]
            else:
                b_deq_gate = (b_int_gate - 128.0) * s_gate[:, None]
                b_deq_up = (b_int_up - 128.0) * s_up[:, None]

            if SWAP_AB:
                gate_acc += tl.dot(b_deq_gate.to(a.dtype), a)
                up_acc += tl.dot(b_deq_up.to(a.dtype), a)
            else:
                gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
                up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))
        else:
            scale_groups = k_indices // group_size
            scale_mask = n_mask[:, None] & k_mask[None, :]
            s_gate = tl.load(
                W1_scales
                + expert_id * stride_s_e
                + offs_n[:, None] * stride_s_n
                + scale_groups[None, :] * stride_s_k,
                mask=scale_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            s_up = tl.load(
                W1_scales
                + expert_id * stride_s_e
                + up_offs_n[:, None] * stride_s_n
                + scale_groups[None, :] * stride_s_k,
                mask=scale_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            if has_zp:
                zp_gate = tl.load(
                    W1_zp
                    + expert_id * stride_zp_e
                    + offs_n[:, None] * stride_zp_n
                    + scale_groups[None, :] * stride_zp_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                zp_up = tl.load(
                    W1_zp
                    + expert_id * stride_zp_e
                    + up_offs_n[:, None] * stride_zp_n
                    + scale_groups[None, :] * stride_zp_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                b_deq_gate = (b_int_gate - zp_gate) * s_gate
                b_deq_up = (b_int_up - zp_up) * s_up
            else:
                b_deq_gate = (b_int_gate - 128.0) * s_gate
                b_deq_up = (b_int_up - 128.0) * s_up

            if SWAP_AB:
                gate_acc += tl.dot(b_deq_gate.to(a.dtype), a)
                up_acc += tl.dot(b_deq_up.to(a.dtype), a)
            else:
                gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
                up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))

    silu_gate = gate_acc * tl.sigmoid(gate_acc)
    result = silu_gate * up_acc
    if APPLY_ROUTED_WEIGHT:
        weights = tl.load(sorted_weights + offs_m, mask=token_mask, other=0.0).to(
            tl.float32
        )
        result = result * (weights[None, :] if SWAP_AB else weights[:, None])

    if SWAP_AB:
        out_ptrs = (
            INTER + offs_m[None, :] * stride_inter_m + offs_n[:, None] * stride_inter_n
        )
        out_mask = token_mask[None, :] & n_mask[:, None]
    else:
        out_ptrs = (
            INTER + offs_m[:, None] * stride_inter_m + offs_n[None, :] * stride_inter_n
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, result.to(compute_type), mask=out_mask)


@triton.autotune(
    configs=_W8A16_FUSED_LARGE_AUTOTUNE_CONFIGS,
    key=["M_padded", "I", "H", "T"],
)
@triton.jit
def fused_moe_kernel_w8a16_gateup_silu_large(
    A,
    W1_q,
    W1_scales,
    INTER,
    sorted_token_ids,
    expert_ids_per_block,
    sorted_weights,
    M_padded,
    T,
    I,
    H,
    stride_a_t,
    stride_a_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_s_e,
    stride_s_n,
    stride_s_k,
    stride_inter_m,
    stride_inter_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    APPLY_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Large-token fast path for has_zp=False, group_size=128 W8A16 gateup_silu."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < I
    up_offs_n = offs_n + I

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_start in range(0, H, BLOCK_SIZE_K):
        k_indices = k_start + offs_k
        a = tl.load(
            A + token_ids[:, None] * stride_a_t + k_indices[None, :] * stride_a_k,
            mask=token_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        b_int_gate = tl.load(
            W1_q
            + expert_id * stride_w1_e
            + offs_n[:, None] * stride_w1_n
            + k_indices[None, :] * stride_w1_k,
            mask=n_mask[:, None],
            other=128,
            eviction_policy="evict_first",
        ).to(tl.float32)
        b_int_up = tl.load(
            W1_q
            + expert_id * stride_w1_e
            + up_offs_n[:, None] * stride_w1_n
            + k_indices[None, :] * stride_w1_k,
            mask=n_mask[:, None],
            other=128,
            eviction_policy="evict_first",
        ).to(tl.float32)

        group_idx = k_start // 128
        s_gate = tl.load(
            W1_scales
            + expert_id * stride_s_e
            + offs_n * stride_s_n
            + group_idx * stride_s_k,
            mask=n_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        s_up = tl.load(
            W1_scales
            + expert_id * stride_s_e
            + up_offs_n * stride_s_n
            + group_idx * stride_s_k,
            mask=n_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        b_deq_gate = (b_int_gate - 128.0) * s_gate[:, None]
        b_deq_up = (b_int_up - 128.0) * s_up[:, None]

        gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
        up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))

    silu_gate = gate_acc * tl.sigmoid(gate_acc)
    result = silu_gate * up_acc
    if APPLY_ROUTED_WEIGHT:
        weights = tl.load(sorted_weights + offs_m, mask=token_mask, other=0.0).to(
            tl.float32
        )
        result = result * weights[:, None]

    out_ptrs = (
        INTER + offs_m[:, None] * stride_inter_m + offs_n[None, :] * stride_inter_n
    )
    tl.store(out_ptrs, result.to(compute_type), mask=n_mask[None, :])


@triton.autotune(
    configs=_W8A16_DOWN_AUTOTUNE_CONFIGS,
    key=["M_padded", "H", "I", "T", "SMALL_TOKEN_MXQ_PATH"],
    # IMPORTANT: this kernel uses `tl.atomic_add` to accumulate into OUT.
    # Triton's autotuner re-runs the kernel ~warmup+rep times per Config to
    # measure latency.  Without `reset_to_zero`, OUT would be summed hundreds
    # of times during calibration and the final result would be a huge
    # multiple of the correct value.  `reset_to_zero=["OUT"]` zeroes OUT
    # before each calibration run; cached subsequent runs are NOT reset.
    reset_to_zero=["OUT"],
)
@triton.jit
def fused_moe_kernel_w8a16_down(
    INTER,  # (M_padded, I) bf16, indexed by dispatch_idx
    W2_q,  # (E, H, I) uint8
    W2_scales,  # (E, H, I_groups) bf16
    W2_zp,  # (E, H, I_groups) uint8 or empty
    OUT,  # (T, H) bf16, atomic_add target
    sorted_token_ids,
    expert_ids_per_block,
    topk_weights,
    M_padded,
    T,
    H,
    I,
    stride_inter_m,
    stride_inter_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_s_e,
    stride_s_n,
    stride_s_k,
    stride_zp_e,
    stride_zp_n,
    stride_zp_k,
    stride_out_t,
    stride_out_n,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    has_zp: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    even_Ks: tl.constexpr,
    DOWN_GRID_N_FIRST: tl.constexpr,
    INTER_PREWEIGHTED: tl.constexpr,
    SMALL_TOKEN_MXQ_PATH: tl.constexpr,
    SWAP_AB: tl.constexpr,
    compute_type: tl.constexpr,
):
    """y = W2[expert] @ intermediate, output[token] += weight * y. Full H coverage."""
    if DOWN_GRID_N_FIRST:
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    n_mask = offs_n < H
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_start in range(0, I, BLOCK_SIZE_K):
        k_indices = k_start + offs_k
        k_mask = k_indices < I
        if SWAP_AB:
            a_ptrs = (
                INTER
                + offs_m[None, :] * stride_inter_m
                + k_indices[:, None] * stride_inter_k
            )
            a_mask = token_mask[None, :]
        else:
            a_ptrs = (
                INTER
                + offs_m[:, None] * stride_inter_m
                + k_indices[None, :] * stride_inter_k
            )
            a_mask = token_mask[:, None]
        if not even_Ks:
            if SWAP_AB:
                a_mask = a_mask & k_mask[:, None]
            else:
                a_mask = a_mask & k_mask[None, :]
        if even_Ks:
            if SMALL_TOKEN_MXQ_PATH:
                a = tl.load(
                    a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last"
                )
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices[None, :] * stride_w2_k,
                    mask=n_mask[:, None],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                a = tl.load(
                    a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first"
                )
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices[None, :] * stride_w2_k,
                    mask=n_mask[:, None],
                    other=128,
                    eviction_policy="evict_last",
                ).to(tl.float32)
        else:
            if SMALL_TOKEN_MXQ_PATH:
                a = tl.load(
                    a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_last"
                )
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices[None, :] * stride_w2_k,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                a = tl.load(
                    a_ptrs, mask=a_mask, other=0.0, eviction_policy="evict_first"
                )
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices[None, :] * stride_w2_k,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=128,
                    eviction_policy="evict_last",
                ).to(tl.float32)

        if group_size >= BLOCK_SIZE_K and (group_size % BLOCK_SIZE_K) == 0:
            group_idx = k_start // group_size
            if SMALL_TOKEN_MXQ_PATH:
                s = tl.load(
                    W2_scales
                    + expert_id * stride_s_e
                    + offs_n * stride_s_n
                    + group_idx * stride_s_k,
                    mask=n_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                s = tl.load(
                    W2_scales
                    + expert_id * stride_s_e
                    + offs_n * stride_s_n
                    + group_idx * stride_s_k,
                    mask=n_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)

            if has_zp:
                if SMALL_TOKEN_MXQ_PATH:
                    zp = tl.load(
                        W2_zp
                        + expert_id * stride_zp_e
                        + offs_n * stride_zp_n
                        + group_idx * stride_zp_k,
                        mask=n_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                else:
                    zp = tl.load(
                        W2_zp
                        + expert_id * stride_zp_e
                        + offs_n * stride_zp_n
                        + group_idx * stride_zp_k,
                        mask=n_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                b_deq = (b_int - zp[:, None]) * s[:, None]
            else:
                b_deq = (b_int - 128.0) * s[:, None]

            if SWAP_AB:
                accumulator += tl.dot(b_deq.to(a.dtype), a)
            else:
                accumulator += tl.dot(a, tl.trans(b_deq.to(a.dtype)))
        else:
            scale_groups = k_indices // group_size
            scale_mask = n_mask[:, None] & k_mask[None, :]
            if SMALL_TOKEN_MXQ_PATH:
                s = tl.load(
                    W2_scales
                    + expert_id * stride_s_e
                    + offs_n[:, None] * stride_s_n
                    + scale_groups[None, :] * stride_s_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                s = tl.load(
                    W2_scales
                    + expert_id * stride_s_e
                    + offs_n[:, None] * stride_s_n
                    + scale_groups[None, :] * stride_s_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)

            if has_zp:
                if SMALL_TOKEN_MXQ_PATH:
                    zp = tl.load(
                        W2_zp
                        + expert_id * stride_zp_e
                        + offs_n[:, None] * stride_zp_n
                        + scale_groups[None, :] * stride_zp_k,
                        mask=scale_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                else:
                    zp = tl.load(
                        W2_zp
                        + expert_id * stride_zp_e
                        + offs_n[:, None] * stride_zp_n
                        + scale_groups[None, :] * stride_zp_k,
                        mask=scale_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                b_deq = (b_int - zp) * s
            else:
                b_deq = (b_int - 128.0) * s

            if SWAP_AB:
                accumulator += tl.dot(b_deq.to(a.dtype), a)
            else:
                accumulator += tl.dot(a, tl.trans(b_deq.to(a.dtype)))

    if not INTER_PREWEIGHTED:
        weights = tl.load(topk_weights + offs_m, mask=token_mask, other=0.0).to(
            tl.float32
        )
        accumulator = accumulator * (weights[None, :] if SWAP_AB else weights[:, None])

    if SWAP_AB:
        out_ptrs = (
            OUT + token_ids[None, :] * stride_out_t + offs_n[:, None] * stride_out_n
        )
        out_mask = token_mask[None, :] & n_mask[:, None]
    else:
        out_ptrs = (
            OUT + token_ids[:, None] * stride_out_t + offs_n[None, :] * stride_out_n
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
    tl.atomic_add(out_ptrs, accumulator.to(compute_type), mask=out_mask)


@triton.jit
def fused_moe_kernel_w8a16_down_gs128(
    INTER,
    W2_q,
    W2_scales,
    OUT,
    sorted_token_ids,
    expert_ids_per_block,
    topk_weights,
    M_padded,
    T,
    H,
    I,
    stride_inter_m,
    stride_inter_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_s_e,
    stride_s_n,
    stride_s_k,
    stride_out_t,
    stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    DOWN_GRID_N_FIRST: tl.constexpr,
    INTER_PREWEIGHTED: tl.constexpr,
    SMALL_TOKEN_MXQ_PATH: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Down fast path: I=1024, gs=128, no zp — fixed 128x128 tiles (experimental, default off)."""
    if DOWN_GRID_N_FIRST:
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    n_mask = offs_n < H
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_start in range(0, I, BLOCK_SIZE_K):
        k_indices = k_start + offs_k
        group_idx = k_start // BLOCK_SIZE_K
        if SMALL_TOKEN_MXQ_PATH:
            a = tl.load(
                INTER
                + offs_m[:, None] * stride_inter_m
                + k_indices[None, :] * stride_inter_k,
                mask=token_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            b_int = tl.load(
                W2_q
                + expert_id * stride_w2_e
                + offs_n[:, None] * stride_w2_n
                + k_indices[None, :] * stride_w2_k,
                mask=n_mask[:, None],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
            s = tl.load(
                W2_scales
                + expert_id * stride_s_e
                + offs_n * stride_s_n
                + group_idx * stride_s_k,
                mask=n_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            a = tl.load(
                INTER
                + offs_m[:, None] * stride_inter_m
                + k_indices[None, :] * stride_inter_k,
                mask=token_mask[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            b_int = tl.load(
                W2_q
                + expert_id * stride_w2_e
                + offs_n[:, None] * stride_w2_n
                + k_indices[None, :] * stride_w2_k,
                mask=n_mask[:, None],
                other=128,
                eviction_policy="evict_last",
            ).to(tl.float32)
            s = tl.load(
                W2_scales
                + expert_id * stride_s_e
                + offs_n * stride_s_n
                + group_idx * stride_s_k,
                mask=n_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)

        b_deq = (b_int - 128.0) * s[:, None]
        accumulator += tl.dot(a, tl.trans(b_deq.to(a.dtype)))

    if not INTER_PREWEIGHTED:
        weights = tl.load(topk_weights + offs_m, mask=token_mask, other=0.0).to(
            tl.float32
        )
        accumulator = accumulator * weights[:, None]

    out_ptrs = OUT + token_ids[:, None] * stride_out_t + offs_n[None, :] * stride_out_n
    out_mask = token_mask[:, None] & n_mask[None, :]
    tl.atomic_add(out_ptrs, accumulator.to(compute_type), mask=out_mask)


@triton.jit
def fused_moe_kernel_w8a16_gateup_silu_large_h4096(
    A,
    W1_q,
    W1_scales,
    INTER,
    sorted_token_ids,
    expert_ids_per_block,
    sorted_weights,
    M_padded,
    T,
    I,
    H,
    stride_a_t,
    stride_a_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_s_e,
    stride_s_n,
    stride_s_k,
    stride_inter_m,
    stride_inter_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    APPLY_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
):
    """gateup_silu_large fast path: H=4096, gs=128 (experimental, default off)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < I
    up_offs_n = offs_n + I

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    for k_start in range(0, H, BLOCK_SIZE_K):
        k_indices = k_start + offs_k
        group_idx = k_start // BLOCK_SIZE_K
        a = tl.load(
            A + token_ids[:, None] * stride_a_t + k_indices[None, :] * stride_a_k,
            mask=token_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        b_int_gate = tl.load(
            W1_q
            + expert_id * stride_w1_e
            + offs_n[:, None] * stride_w1_n
            + k_indices[None, :] * stride_w1_k,
            mask=n_mask[:, None],
            other=128,
            eviction_policy="evict_first",
        ).to(tl.float32)
        b_int_up = tl.load(
            W1_q
            + expert_id * stride_w1_e
            + up_offs_n[:, None] * stride_w1_n
            + k_indices[None, :] * stride_w1_k,
            mask=n_mask[:, None],
            other=128,
            eviction_policy="evict_first",
        ).to(tl.float32)

        s_gate = tl.load(
            W1_scales
            + expert_id * stride_s_e
            + offs_n * stride_s_n
            + group_idx * stride_s_k,
            mask=n_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        s_up = tl.load(
            W1_scales
            + expert_id * stride_s_e
            + up_offs_n * stride_s_n
            + group_idx * stride_s_k,
            mask=n_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        b_deq_gate = (b_int_gate - 128.0) * s_gate[:, None]
        b_deq_up = (b_int_up - 128.0) * s_up[:, None]

        gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
        up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))

    silu_gate = gate_acc * tl.sigmoid(gate_acc)
    result = silu_gate * up_acc
    if APPLY_ROUTED_WEIGHT:
        weights = tl.load(sorted_weights + offs_m, mask=token_mask, other=0.0).to(
            tl.float32
        )
        result = result * weights[:, None]

    out_ptrs = (
        INTER + offs_m[:, None] * stride_inter_m + offs_n[None, :] * stride_inter_n
    )
    tl.store(out_ptrs, result.to(compute_type), mask=n_mask[None, :])


@triton.autotune(
    configs=_W8A16_UNIFIED_MOE_AUTOTUNE_CONFIGS,
    key=["M_padded", "H", "I", "T", "SMALL_TOKEN_MXQ_PATH"],
    reset_to_zero=["OUT"],
)
@triton.jit
def fused_moe_kernel_w8a16_unified_moe(
    A,
    W1_q,
    W1_scales,
    W1_zp,
    W2_q,
    W2_scales,
    W2_zp,
    OUT,
    sorted_token_ids,
    expert_ids_per_block,
    topk_weights,
    M_padded,
    T,
    I,
    H,
    stride_a_t,
    stride_a_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_s1_e,
    stride_s1_n,
    stride_s1_k,
    stride_zp1_e,
    stride_zp1_n,
    stride_zp1_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_s2_e,
    stride_s2_n,
    stride_s2_k,
    stride_zp2_e,
    stride_zp2_n,
    stride_zp2_k,
    stride_out_t,
    stride_out_n,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_I_TILE: tl.constexpr,
    BLOCK_K_H: tl.constexpr,
    even_Ks_h: tl.constexpr,
    even_Ks_i: tl.constexpr,
    has_zp_w1: tl.constexpr,
    has_zp_w2: tl.constexpr,
    DOWN_GRID_N_FIRST: tl.constexpr,
    SMALL_TOKEN_MXQ_PATH: tl.constexpr,
    INTER_PREWEIGHTED: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One launch: BSM-matched SwiGLU MoE without materializing ``(M_padded, I)`` in HBM.

    Grid **(num_blocks_m, I_tiles)** — same contract as ``gateup_silu``: each CTA owns one
    ``(m_block, i_tile)``, runs **one** reduction over ``H`` for gate/up, then applies W2
    across ``H`` output tiles via ``atomic_add`` (same as split down, without intermediate
    write/read).
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_i = pid_i * BLOCK_I_TILE + tl.arange(0, BLOCK_I_TILE)
    i_mask = offs_i < I
    up_offs_i = offs_i + I

    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)

    weights_row = tl.load(topk_weights + offs_m, mask=token_mask, other=0.0).to(
        tl.float32
    )

    offs_k_h = tl.arange(0, BLOCK_K_H)
    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_I_TILE), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_I_TILE), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K_H):
        k_indices = k_start + offs_k_h
        k_mask = k_indices < H
        if even_Ks_h:
            a = tl.load(
                A + token_ids[:, None] * stride_a_t + k_indices[None, :] * stride_a_k,
                mask=token_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            b_int_gate = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + offs_i[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=i_mask[:, None],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
            b_int_up = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + up_offs_i[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=i_mask[:, None],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            a = tl.load(
                A + token_ids[:, None] * stride_a_t + k_indices[None, :] * stride_a_k,
                mask=token_mask[:, None] & k_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            b_int_gate = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + offs_i[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=i_mask[:, None] & k_mask[None, :],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)
            b_int_up = tl.load(
                W1_q
                + expert_id * stride_w1_e
                + up_offs_i[:, None] * stride_w1_n
                + k_indices[None, :] * stride_w1_k,
                mask=i_mask[:, None] & k_mask[None, :],
                other=128,
                eviction_policy="evict_first",
            ).to(tl.float32)

        if group_size >= BLOCK_K_H and (group_size % BLOCK_K_H) == 0:
            group_idx = k_start // group_size
            # scale 数据量小（BLOCK_I_TILE 个标量），在 k_start 循环中每隔
            # group_size/BLOCK_K_H 轮才更新一次，保留在 L1/L2 中有利，用 evict_last。
            s_gate = tl.load(
                W1_scales
                + expert_id * stride_s1_e
                + offs_i * stride_s1_n
                + group_idx * stride_s1_k,
                mask=i_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            s_up = tl.load(
                W1_scales
                + expert_id * stride_s1_e
                + up_offs_i * stride_s1_n
                + group_idx * stride_s1_k,
                mask=i_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)

            if has_zp_w1:
                zp_gate = tl.load(
                    W1_zp
                    + expert_id * stride_zp1_e
                    + offs_i * stride_zp1_n
                    + group_idx * stride_zp1_k,
                    mask=i_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                zp_up = tl.load(
                    W1_zp
                    + expert_id * stride_zp1_e
                    + up_offs_i * stride_zp1_n
                    + group_idx * stride_zp1_k,
                    mask=i_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                b_deq_gate = (b_int_gate - zp_gate[:, None]) * s_gate[:, None]
                b_deq_up = (b_int_up - zp_up[:, None]) * s_up[:, None]
            else:
                b_deq_gate = (b_int_gate - 128.0) * s_gate[:, None]
                b_deq_up = (b_int_up - 128.0) * s_up[:, None]

            gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
            up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))
        else:
            scale_groups = k_indices // group_size
            scale_mask = i_mask[:, None] & k_mask[None, :]
            s_gate = tl.load(
                W1_scales
                + expert_id * stride_s1_e
                + offs_i[:, None] * stride_s1_n
                + scale_groups[None, :] * stride_s1_k,
                mask=scale_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            s_up = tl.load(
                W1_scales
                + expert_id * stride_s1_e
                + up_offs_i[:, None] * stride_s1_n
                + scale_groups[None, :] * stride_s1_k,
                mask=scale_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            if has_zp_w1:
                zp_gate = tl.load(
                    W1_zp
                    + expert_id * stride_zp1_e
                    + offs_i[:, None] * stride_zp1_n
                    + scale_groups[None, :] * stride_zp1_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                zp_up = tl.load(
                    W1_zp
                    + expert_id * stride_zp1_e
                    + up_offs_i[:, None] * stride_zp1_n
                    + scale_groups[None, :] * stride_zp1_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                b_deq_gate = (b_int_gate - zp_gate) * s_gate
                b_deq_up = (b_int_up - zp_up) * s_up
            else:
                b_deq_gate = (b_int_gate - 128.0) * s_gate
                b_deq_up = (b_int_up - 128.0) * s_up

            gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
            up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))

    silu_gate = gate_acc * tl.sigmoid(gate_acc)
    inter = silu_gate * up_acc
    if INTER_PREWEIGHTED:
        inter = inter * weights_row[:, None]

    inter_typed = inter.to(compute_type)
    k_indices_i = offs_i
    k_mask_i = i_mask
    group_idx_i = pid_i * BLOCK_I_TILE // group_size

    num_h_tiles = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    for h_tile_idx in range(num_h_tiles):
        offs_n = h_tile_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offs_n < H

        if even_Ks_i:
            if SMALL_TOKEN_MXQ_PATH:
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices_i[None, :] * stride_w2_k,
                    mask=n_mask[:, None],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices_i[None, :] * stride_w2_k,
                    mask=n_mask[:, None],
                    other=128,
                    eviction_policy="evict_last",
                ).to(tl.float32)
        else:
            if SMALL_TOKEN_MXQ_PATH:
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices_i[None, :] * stride_w2_k,
                    mask=n_mask[:, None] & k_mask_i[None, :],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                b_int = tl.load(
                    W2_q
                    + expert_id * stride_w2_e
                    + offs_n[:, None] * stride_w2_n
                    + k_indices_i[None, :] * stride_w2_k,
                    mask=n_mask[:, None] & k_mask_i[None, :],
                    other=128,
                    eviction_policy="evict_last",
                ).to(tl.float32)

        if group_size >= BLOCK_I_TILE and (group_size % BLOCK_I_TILE) == 0:
            s2 = tl.load(
                W2_scales
                + expert_id * stride_s2_e
                + offs_n * stride_s2_n
                + group_idx_i * stride_s2_k,
                mask=n_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)

            if has_zp_w2:
                zp2 = tl.load(
                    W2_zp
                    + expert_id * stride_zp2_e
                    + offs_n * stride_zp2_n
                    + group_idx_i * stride_zp2_k,
                    mask=n_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                b_deq = (b_int - zp2[:, None]) * s2[:, None]
            else:
                b_deq = (b_int - 128.0) * s2[:, None]

            partial = tl.dot(inter_typed, tl.trans(b_deq.to(inter_typed.dtype)))
        else:
            scale_groups_i = k_indices_i // group_size
            scale_mask_i = n_mask[:, None] & k_mask_i[None, :]
            if SMALL_TOKEN_MXQ_PATH:
                s2 = tl.load(
                    W2_scales
                    + expert_id * stride_s2_e
                    + offs_n[:, None] * stride_s2_n
                    + scale_groups_i[None, :] * stride_s2_k,
                    mask=scale_mask_i,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                s2 = tl.load(
                    W2_scales
                    + expert_id * stride_s2_e
                    + offs_n[:, None] * stride_s2_n
                    + scale_groups_i[None, :] * stride_s2_k,
                    mask=scale_mask_i,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)

            if has_zp_w2:
                zp2 = tl.load(
                    W2_zp
                    + expert_id * stride_zp2_e
                    + offs_n[:, None] * stride_zp2_n
                    + scale_groups_i[None, :] * stride_zp2_k,
                    mask=scale_mask_i,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                b_deq = (b_int - zp2) * s2
            else:
                b_deq = (b_int - 128.0) * s2

            partial = tl.dot(inter_typed, tl.trans(b_deq.to(inter_typed.dtype)))

        if not INTER_PREWEIGHTED:
            partial = partial * weights_row[:, None]

        out_ptrs = (
            OUT + token_ids[:, None] * stride_out_t + offs_n[None, :] * stride_out_n
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
        tl.atomic_add(out_ptrs, partial.to(compute_type), mask=out_mask)


@triton.autotune(
    configs=_W8A16_DOWN_AUTOTUNE_CONFIGS,
    key=["M_padded", "H", "I", "T", "SMALL_TOKEN_MXQ_PATH"],
    reset_to_zero=["OUT"],
)
@triton.jit
def fused_moe_kernel_w8a16_unified_moe_per_m(
    A,
    W1_q,
    W1_scales,
    W1_zp,
    INTER,
    W2_q,
    W2_scales,
    W2_zp,
    OUT,
    sorted_token_ids,
    expert_ids_per_block,
    topk_weights,
    M_padded,
    T,
    I,
    H,
    stride_a_t,
    stride_a_k,
    stride_w1_e,
    stride_w1_n,
    stride_w1_k,
    stride_s1_e,
    stride_s1_n,
    stride_s1_k,
    stride_zp1_e,
    stride_zp1_n,
    stride_zp1_k,
    stride_inter_m,
    stride_inter_k,
    stride_w2_e,
    stride_w2_n,
    stride_w2_k,
    stride_s2_e,
    stride_s2_n,
    stride_s2_k,
    stride_zp2_e,
    stride_zp2_n,
    stride_zp2_k,
    stride_out_t,
    stride_out_n,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    even_Ks_h: tl.constexpr,
    even_Ks_i: tl.constexpr,
    has_zp_w1: tl.constexpr,
    has_zp_w2: tl.constexpr,
    INTER_PREWEIGHTED: tl.constexpr,
    SMALL_TOKEN_MXQ_PATH: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One launch per BSM M-block: gateup+SwiGLU -> INTER, then down -> OUT.

    Grid ``(num_blocks_m,)``.  Semantics match split ``gateup_silu`` + ``down`` but
    fused in a single kernel so CUDA Graph captures one launch.  INTER is a
    short-lived workspace (same as split B2).
    """
    pid_m = tl.program_id(0)
    block_start = pid_m * BLOCK_SIZE_M
    if block_start >= M_padded:
        return

    offs_m = block_start + tl.arange(0, BLOCK_SIZE_M)
    token_ids = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = token_ids < T
    expert_id = tl.load(expert_ids_per_block + pid_m).to(tl.int64)
    weights_row = tl.load(topk_weights + offs_m, mask=token_mask, other=0.0).to(
        tl.float32
    )

    offs_k_h = tl.arange(0, BLOCK_SIZE_K)

    # ---- Phase 1: W1 + SwiGLU, write (M_padded, I) tiles to INTER ----
    for i_start in range(0, I, BLOCK_SIZE_K):
        offs_i = i_start + tl.arange(0, BLOCK_SIZE_K)
        i_mask = offs_i < I
        up_offs_i = offs_i + I

        gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for k_start in range(0, H, BLOCK_SIZE_K):
            k_indices = k_start + offs_k_h
            k_mask = k_indices < H
            if even_Ks_h:
                a = tl.load(
                    A
                    + token_ids[:, None] * stride_a_t
                    + k_indices[None, :] * stride_a_k,
                    mask=token_mask[:, None],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                b_int_gate = tl.load(
                    W1_q
                    + expert_id * stride_w1_e
                    + offs_i[:, None] * stride_w1_n
                    + k_indices[None, :] * stride_w1_k,
                    mask=i_mask[:, None],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                b_int_up = tl.load(
                    W1_q
                    + expert_id * stride_w1_e
                    + up_offs_i[:, None] * stride_w1_n
                    + k_indices[None, :] * stride_w1_k,
                    mask=i_mask[:, None],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
            else:
                a = tl.load(
                    A
                    + token_ids[:, None] * stride_a_t
                    + k_indices[None, :] * stride_a_k,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
                b_int_gate = tl.load(
                    W1_q
                    + expert_id * stride_w1_e
                    + offs_i[:, None] * stride_w1_n
                    + k_indices[None, :] * stride_w1_k,
                    mask=i_mask[:, None] & k_mask[None, :],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                b_int_up = tl.load(
                    W1_q
                    + expert_id * stride_w1_e
                    + up_offs_i[:, None] * stride_w1_n
                    + k_indices[None, :] * stride_w1_k,
                    mask=i_mask[:, None] & k_mask[None, :],
                    other=128,
                    eviction_policy="evict_first",
                ).to(tl.float32)

            if group_size >= BLOCK_SIZE_K and (group_size % BLOCK_SIZE_K) == 0:
                group_idx = k_start // group_size
                s_gate = tl.load(
                    W1_scales
                    + expert_id * stride_s1_e
                    + offs_i * stride_s1_n
                    + group_idx * stride_s1_k,
                    mask=i_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                s_up = tl.load(
                    W1_scales
                    + expert_id * stride_s1_e
                    + up_offs_i * stride_s1_n
                    + group_idx * stride_s1_k,
                    mask=i_mask,
                    other=0.0,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                if has_zp_w1:
                    zp_gate = tl.load(
                        W1_zp
                        + expert_id * stride_zp1_e
                        + offs_i * stride_zp1_n
                        + group_idx * stride_zp1_k,
                        mask=i_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                    zp_up = tl.load(
                        W1_zp
                        + expert_id * stride_zp1_e
                        + up_offs_i * stride_zp1_n
                        + group_idx * stride_zp1_k,
                        mask=i_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                    b_deq_gate = (b_int_gate - zp_gate[:, None]) * s_gate[:, None]
                    b_deq_up = (b_int_up - zp_up[:, None]) * s_up[:, None]
                else:
                    b_deq_gate = (b_int_gate - 128.0) * s_gate[:, None]
                    b_deq_up = (b_int_up - 128.0) * s_up[:, None]
                gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
                up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))
            else:
                scale_groups = k_indices // group_size
                scale_mask = i_mask[:, None] & k_mask[None, :]
                s_gate = tl.load(
                    W1_scales
                    + expert_id * stride_s1_e
                    + offs_i[:, None] * stride_s1_n
                    + scale_groups[None, :] * stride_s1_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                s_up = tl.load(
                    W1_scales
                    + expert_id * stride_s1_e
                    + up_offs_i[:, None] * stride_s1_n
                    + scale_groups[None, :] * stride_s1_k,
                    mask=scale_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                if has_zp_w1:
                    zp_gate = tl.load(
                        W1_zp
                        + expert_id * stride_zp1_e
                        + offs_i[:, None] * stride_zp1_n
                        + scale_groups[None, :] * stride_zp1_k,
                        mask=scale_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    zp_up = tl.load(
                        W1_zp
                        + expert_id * stride_zp1_e
                        + up_offs_i[:, None] * stride_zp1_n
                        + scale_groups[None, :] * stride_zp1_k,
                        mask=scale_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    b_deq_gate = (b_int_gate - zp_gate) * s_gate
                    b_deq_up = (b_int_up - zp_up) * s_up
                else:
                    b_deq_gate = (b_int_gate - 128.0) * s_gate
                    b_deq_up = (b_int_up - 128.0) * s_up
                gate_acc += tl.dot(a, tl.trans(b_deq_gate.to(a.dtype)))
                up_acc += tl.dot(a, tl.trans(b_deq_up.to(a.dtype)))

        silu_gate = gate_acc * tl.sigmoid(gate_acc)
        inter = silu_gate * up_acc
        if INTER_PREWEIGHTED:
            inter = inter * weights_row[:, None]
        inter_ptrs = (
            INTER + offs_m[:, None] * stride_inter_m + offs_i[None, :] * stride_inter_k
        )
        inter_mask = token_mask[:, None] & i_mask[None, :]
        tl.store(inter_ptrs, inter.to(compute_type), mask=inter_mask)

    # ---- Phase 2: down projection (same as fused_moe_kernel_w8a16_down) ----
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for n_start in range(0, H, BLOCK_SIZE_N):
        offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offs_n < H
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k_start in range(0, I, BLOCK_SIZE_K):
            k_indices = k_start + offs_k
            k_mask = k_indices < I
            if even_Ks_i:
                if SMALL_TOKEN_MXQ_PATH:
                    a = tl.load(
                        INTER
                        + offs_m[:, None] * stride_inter_m
                        + k_indices[None, :] * stride_inter_k,
                        mask=token_mask[:, None],
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    b_int = tl.load(
                        W2_q
                        + expert_id * stride_w2_e
                        + offs_n[:, None] * stride_w2_n
                        + k_indices[None, :] * stride_w2_k,
                        mask=n_mask[:, None],
                        other=128,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                else:
                    a = tl.load(
                        INTER
                        + offs_m[:, None] * stride_inter_m
                        + k_indices[None, :] * stride_inter_k,
                        mask=token_mask[:, None],
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    b_int = tl.load(
                        W2_q
                        + expert_id * stride_w2_e
                        + offs_n[:, None] * stride_w2_n
                        + k_indices[None, :] * stride_w2_k,
                        mask=n_mask[:, None],
                        other=128,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
            else:
                if SMALL_TOKEN_MXQ_PATH:
                    a = tl.load(
                        INTER
                        + offs_m[:, None] * stride_inter_m
                        + k_indices[None, :] * stride_inter_k,
                        mask=token_mask[:, None] & k_mask[None, :],
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    b_int = tl.load(
                        W2_q
                        + expert_id * stride_w2_e
                        + offs_n[:, None] * stride_w2_n
                        + k_indices[None, :] * stride_w2_k,
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=128,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                else:
                    a = tl.load(
                        INTER
                        + offs_m[:, None] * stride_inter_m
                        + k_indices[None, :] * stride_inter_k,
                        mask=token_mask[:, None] & k_mask[None, :],
                        other=0.0,
                        eviction_policy="evict_first",
                    )
                    b_int = tl.load(
                        W2_q
                        + expert_id * stride_w2_e
                        + offs_n[:, None] * stride_w2_n
                        + k_indices[None, :] * stride_w2_k,
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=128,
                        eviction_policy="evict_last",
                    ).to(tl.float32)

            if group_size >= BLOCK_SIZE_K and (group_size % BLOCK_SIZE_K) == 0:
                group_idx = k_start // group_size
                if SMALL_TOKEN_MXQ_PATH:
                    s = tl.load(
                        W2_scales
                        + expert_id * stride_s2_e
                        + offs_n * stride_s2_n
                        + group_idx * stride_s2_k,
                        mask=n_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                else:
                    s = tl.load(
                        W2_scales
                        + expert_id * stride_s2_e
                        + offs_n * stride_s2_n
                        + group_idx * stride_s2_k,
                        mask=n_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                if has_zp_w2:
                    if SMALL_TOKEN_MXQ_PATH:
                        zp = tl.load(
                            W2_zp
                            + expert_id * stride_zp2_e
                            + offs_n * stride_zp2_n
                            + group_idx * stride_zp2_k,
                            mask=n_mask,
                            other=0.0,
                            eviction_policy="evict_first",
                        ).to(tl.float32)
                    else:
                        zp = tl.load(
                            W2_zp
                            + expert_id * stride_zp2_e
                            + offs_n * stride_zp2_n
                            + group_idx * stride_zp2_k,
                            mask=n_mask,
                            other=0.0,
                            eviction_policy="evict_last",
                        ).to(tl.float32)
                    b_deq = (b_int - zp[:, None]) * s[:, None]
                else:
                    b_deq = (b_int - 128.0) * s[:, None]
                accumulator += tl.dot(a, tl.trans(b_deq.to(a.dtype)))
            else:
                scale_groups = k_indices // group_size
                scale_mask = n_mask[:, None] & k_mask[None, :]
                if SMALL_TOKEN_MXQ_PATH:
                    s = tl.load(
                        W2_scales
                        + expert_id * stride_s2_e
                        + offs_n[:, None] * stride_s2_n
                        + scale_groups[None, :] * stride_s2_k,
                        mask=scale_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                else:
                    s = tl.load(
                        W2_scales
                        + expert_id * stride_s2_e
                        + offs_n[:, None] * stride_s2_n
                        + scale_groups[None, :] * stride_s2_k,
                        mask=scale_mask,
                        other=0.0,
                        eviction_policy="evict_last",
                    ).to(tl.float32)
                if has_zp_w2:
                    if SMALL_TOKEN_MXQ_PATH:
                        zp = tl.load(
                            W2_zp
                            + expert_id * stride_zp2_e
                            + offs_n[:, None] * stride_zp2_n
                            + scale_groups[None, :] * stride_zp2_k,
                            mask=scale_mask,
                            other=0.0,
                            eviction_policy="evict_first",
                        ).to(tl.float32)
                    else:
                        zp = tl.load(
                            W2_zp
                            + expert_id * stride_zp2_e
                            + offs_n[:, None] * stride_zp2_n
                            + scale_groups[None, :] * stride_zp2_k,
                            mask=scale_mask,
                            other=0.0,
                            eviction_policy="evict_last",
                        ).to(tl.float32)
                    b_deq = (b_int - zp) * s
                else:
                    b_deq = (b_int - 128.0) * s
                accumulator += tl.dot(a, tl.trans(b_deq.to(a.dtype)))

        if not INTER_PREWEIGHTED:
            accumulator = accumulator * weights_row[:, None]
        out_ptrs = (
            OUT + token_ids[:, None] * stride_out_t + offs_n[None, :] * stride_out_n
        )
        out_mask = token_mask[:, None] & n_mask[None, :]
        tl.atomic_add(out_ptrs, accumulator.to(compute_type), mask=out_mask)


def _mxq_split_small_large_threshold() -> int:
    """Token count bound for down-kernel *small* eviction / INTER traffic policy.

    Default **512** (stable baseline ``20260515_170907``).  opt2 (split=64) regressed
    T=64 vLLM 1.29x->0.82x; opt3 decoupled eviction=64 without improving BF16 mid-batch.
    """
    return 512


def _mxq_small_token_mxq_path(num_valid_tokens: int) -> bool:
    """``SMALL_TOKEN_MXQ_PATH`` constexpr: T bound tied to split threshold (default 512)."""
    return num_valid_tokens <= _mxq_split_small_large_threshold()


def _mxq_triton_jit_fn(kernel):
    """Return the bare ``@triton.jit`` under an ``@triton.autotune`` wrapper.

      Bucket-pin launches pass ``BLOCK_SIZE_*`` manually; calling the autotuner
    entrypoint with those kwargs raises "Conflicting meta-parameters".
    """
    return getattr(kernel, "fn", kernel)


def _mxq_b2_bucket_pin_enabled() -> bool:
    return False


def _mxq_b2_pin_min_tokens() -> int:
    return 64


def _mxq_b2_pin_max_tokens() -> int:
    return 512


def _mxq_b2_gateup_large_pin(num_valid_tokens: int) -> Optional[dict]:
    """Fixed gateup tile **only** for mid batch (default T=64～512).

    T≥1024 must use autotune — pinning 128×128 there regressed 231618 vs 170907
    (e.g. T=1024 Gems 11.3 ms vs 6.5 ms).  T≤16 stay on autotune / MI (T≤MI_MAX).
    """
    if not _mxq_b2_bucket_pin_enabled():
        return None
    lo, hi = _mxq_b2_pin_min_tokens(), _mxq_b2_pin_max_tokens()
    if lo <= num_valid_tokens <= hi:
        return {
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "num_warps": 8,
            "num_stages": 3,
        }
    return None


def _mxq_b2_down_pin(num_valid_tokens: int) -> Optional[dict]:
    """Fixed down tile for mid batch only; large T uses autotune (see gateup pin)."""
    if not _mxq_b2_bucket_pin_enabled():
        return None
    lo, hi = _mxq_b2_pin_min_tokens(), _mxq_b2_pin_max_tokens()
    if lo <= num_valid_tokens <= hi:
        return {
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "num_warps": 8,
            "num_stages": 3,
        }
    return None


def _launch_w8a16_gateup_silu(
    x,
    W1_q,
    W1_scales,
    zp1,
    intermediate,
    sorted_token_ids,
    expert_ids_per_block,
    sorted_weights,
    *,
    num_post_padded: int,
    num_valid_tokens: int,
    I: int,
    H: int,
    BLOCK_SIZE_M: int,
    preweight_intermediate: bool,
    compute_type,
    quant_config: Any,
    has_zp_w1: bool,
    even_Ks_gateup: bool,
    stride_zp_e: int,
    stride_zp_n: int,
    stride_zp_k: int,
) -> None:
    """Autotuned gateup_silu, or fixed 128×128 tile when bucket pin is on."""
    num_blocks_m = num_post_padded // BLOCK_SIZE_M
    swap_ab = 1 < num_valid_tokens <= 16
    pin = _mxq_b2_gateup_large_pin(num_valid_tokens)
    if pin is not None:
        bsn = pin["BLOCK_SIZE_N"]

        def _grid_pin(META):
            del META
            return (num_blocks_m, (I + bsn - 1) // bsn)

        _mxq_triton_jit_fn(fused_moe_kernel_w8a16_gateup_silu)[_grid_pin](
            x,
            W1_q,
            W1_scales,
            zp1,
            intermediate,
            sorted_token_ids,
            expert_ids_per_block,
            sorted_weights,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            I=I,
            H=H,
            stride_a_t=x.stride(0),
            stride_a_k=x.stride(1),
            stride_w1_e=W1_q.stride(0),
            stride_w1_n=W1_q.stride(1),
            stride_w1_k=W1_q.stride(2),
            stride_s_e=W1_scales.stride(0),
            stride_s_n=W1_scales.stride(1),
            stride_s_k=W1_scales.stride(2),
            stride_zp_e=stride_zp_e,
            stride_zp_n=stride_zp_n,
            stride_zp_k=stride_zp_k,
            stride_inter_m=intermediate.stride(0),
            stride_inter_n=intermediate.stride(1),
            group_size=quant_config.group_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=pin["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=pin["BLOCK_SIZE_K"],
            has_zp=has_zp_w1,
            use_int8_w8a16=quant_config.use_int8,
            even_Ks=even_Ks_gateup,
            APPLY_ROUTED_WEIGHT=preweight_intermediate,
            SWAP_AB=swap_ab,
            compute_type=compute_type,
            num_warps=pin["num_warps"],
            num_stages=pin["num_stages"],
        )
        return

    def _grid_gateup_silu(META):
        return (num_blocks_m, triton.cdiv(I, META["BLOCK_SIZE_N"]))

    fused_moe_kernel_w8a16_gateup_silu[_grid_gateup_silu](
        x,
        W1_q,
        W1_scales,
        zp1,
        intermediate,
        sorted_token_ids,
        expert_ids_per_block,
        sorted_weights,
        M_padded=num_post_padded,
        T=num_valid_tokens,
        I=I,
        H=H,
        stride_a_t=x.stride(0),
        stride_a_k=x.stride(1),
        stride_w1_e=W1_q.stride(0),
        stride_w1_n=W1_q.stride(1),
        stride_w1_k=W1_q.stride(2),
        stride_s_e=W1_scales.stride(0),
        stride_s_n=W1_scales.stride(1),
        stride_s_k=W1_scales.stride(2),
        stride_zp_e=stride_zp_e,
        stride_zp_n=stride_zp_n,
        stride_zp_k=stride_zp_k,
        stride_inter_m=intermediate.stride(0),
        stride_inter_n=intermediate.stride(1),
        group_size=quant_config.group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        has_zp=has_zp_w1,
        use_int8_w8a16=quant_config.use_int8,
        even_Ks=even_Ks_gateup,
        APPLY_ROUTED_WEIGHT=preweight_intermediate,
        SWAP_AB=swap_ab,
        compute_type=compute_type,
    )


def _mxq_preweight_intermediate(num_valid_tokens: int) -> bool:
    return num_valid_tokens <= 512


def _mxq_use_down_gs128_fast(
    num_valid_tokens: int,
    I: int,
    quant_config: Any,
    has_zp_w2: bool,
    even_Ks_down: bool,
) -> bool:
    del num_valid_tokens, I, quant_config, has_zp_w2, even_Ks_down
    return False


def _mxq_use_gateup_large_h4096_fast(
    num_valid_tokens: int,
    H: int,
    quant_config: Any,
    has_zp_w1: bool,
    even_Ks: bool,
) -> bool:
    del num_valid_tokens, H, quant_config, has_zp_w1, even_Ks
    return False


def _launch_w8a16_gateup_silu_large(
    x,
    W1_q,
    W1_scales,
    intermediate,
    sorted_token_ids,
    expert_ids_per_block,
    sorted_weights,
    *,
    num_post_padded: int,
    num_valid_tokens: int,
    I: int,
    H: int,
    BLOCK_SIZE_M: int,
    preweight_intermediate: bool,
    compute_type,
    quant_config: Any,
    has_zp_w1: bool,
    even_Ks_gateup_large: bool,
) -> None:
    """Autotuned gateup_silu_large, or H=4096 fixed-tile + unrolled K when enabled."""
    num_blocks_m = num_post_padded // BLOCK_SIZE_M
    bsn_fast = 64
    bsk_fast = 128

    pin = _mxq_b2_gateup_large_pin(num_valid_tokens)
    if pin is not None:

        def _grid_pin(META):
            del META
            bsn = pin["BLOCK_SIZE_N"]
            return (num_blocks_m, (I + bsn - 1) // bsn)

        _mxq_triton_jit_fn(fused_moe_kernel_w8a16_gateup_silu_large)[_grid_pin](
            x,
            W1_q,
            W1_scales,
            intermediate,
            sorted_token_ids,
            expert_ids_per_block,
            sorted_weights,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            I=I,
            H=H,
            stride_a_t=x.stride(0),
            stride_a_k=x.stride(1),
            stride_w1_e=W1_q.stride(0),
            stride_w1_n=W1_q.stride(1),
            stride_w1_k=W1_q.stride(2),
            stride_s_e=W1_scales.stride(0),
            stride_s_n=W1_scales.stride(1),
            stride_s_k=W1_scales.stride(2),
            stride_inter_m=intermediate.stride(0),
            stride_inter_n=intermediate.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=pin["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=pin["BLOCK_SIZE_K"],
            APPLY_ROUTED_WEIGHT=preweight_intermediate,
            compute_type=compute_type,
            num_warps=pin["num_warps"],
            num_stages=pin["num_stages"],
        )
        return

    if _mxq_use_gateup_large_h4096_fast(
        num_valid_tokens, H, quant_config, has_zp_w1, even_Ks_gateup_large
    ):

        def _grid_h4096(META):
            del META
            return (num_blocks_m, triton.cdiv(I, bsn_fast))

        fused_moe_kernel_w8a16_gateup_silu_large_h4096[_grid_h4096](
            x,
            W1_q,
            W1_scales,
            intermediate,
            sorted_token_ids,
            expert_ids_per_block,
            sorted_weights,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            I=I,
            H=H,
            stride_a_t=x.stride(0),
            stride_a_k=x.stride(1),
            stride_w1_e=W1_q.stride(0),
            stride_w1_n=W1_q.stride(1),
            stride_w1_k=W1_q.stride(2),
            stride_s_e=W1_scales.stride(0),
            stride_s_n=W1_scales.stride(1),
            stride_s_k=W1_scales.stride(2),
            stride_inter_m=intermediate.stride(0),
            stride_inter_n=intermediate.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=bsn_fast,
            BLOCK_SIZE_K=bsk_fast,
            APPLY_ROUTED_WEIGHT=preweight_intermediate,
            compute_type=compute_type,
            num_warps=4,
            num_stages=1,
        )
        return

    def _grid_gateup_silu(META):
        return (num_blocks_m, triton.cdiv(I, META["BLOCK_SIZE_N"]))

    fused_moe_kernel_w8a16_gateup_silu_large[_grid_gateup_silu](
        x,
        W1_q,
        W1_scales,
        intermediate,
        sorted_token_ids,
        expert_ids_per_block,
        sorted_weights,
        M_padded=num_post_padded,
        T=num_valid_tokens,
        I=I,
        H=H,
        stride_a_t=x.stride(0),
        stride_a_k=x.stride(1),
        stride_w1_e=W1_q.stride(0),
        stride_w1_n=W1_q.stride(1),
        stride_w1_k=W1_q.stride(2),
        stride_s_e=W1_scales.stride(0),
        stride_s_n=W1_scales.stride(1),
        stride_s_k=W1_scales.stride(2),
        stride_inter_m=intermediate.stride(0),
        stride_inter_n=intermediate.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        APPLY_ROUTED_WEIGHT=preweight_intermediate,
        compute_type=compute_type,
    )


def _launch_w8a16_down(
    intermediate,
    W2_q,
    W2_scales,
    zp2,
    output,
    sorted_token_ids,
    expert_ids_per_block,
    sorted_weights,
    *,
    num_post_padded: int,
    num_valid_tokens: int,
    H: int,
    I: int,
    BLOCK_SIZE_M: int,
    quant_config: Any,
    has_zp_w2: bool,
    even_Ks_down: bool,
    down_grid_n_first: bool,
    preweight_intermediate: bool,
    small_token_mxq_path: bool,
    compute_type,
    stride_zp_e: int,
    stride_zp_n: int,
    stride_zp_k: int,
) -> None:
    """Autotuned down, or I=1024 gs=128 fixed-tile + unrolled K when enabled."""
    num_blocks_m = num_post_padded // BLOCK_SIZE_M
    swap_ab = 1 < num_valid_tokens <= 64
    bsn_fast = bsk_fast = 128

    pin = _mxq_b2_down_pin(num_valid_tokens)
    if pin is not None:
        bsn = pin["BLOCK_SIZE_N"]
        bsk = pin["BLOCK_SIZE_K"]

        def _grid_down_pin(META):
            del META
            h_tiles = (H + bsn - 1) // bsn
            if down_grid_n_first:
                return (h_tiles, num_blocks_m)
            return (num_blocks_m, h_tiles)

        _mxq_triton_jit_fn(fused_moe_kernel_w8a16_down)[_grid_down_pin](
            intermediate,
            W2_q,
            W2_scales,
            zp2,
            output,
            sorted_token_ids,
            expert_ids_per_block,
            sorted_weights,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            H=H,
            I=I,
            stride_inter_m=intermediate.stride(0),
            stride_inter_k=intermediate.stride(1),
            stride_w2_e=W2_q.stride(0),
            stride_w2_n=W2_q.stride(1),
            stride_w2_k=W2_q.stride(2),
            stride_s_e=W2_scales.stride(0),
            stride_s_n=W2_scales.stride(1),
            stride_s_k=W2_scales.stride(2),
            stride_zp_e=stride_zp_e,
            stride_zp_n=stride_zp_n,
            stride_zp_k=stride_zp_k,
            stride_out_t=output.stride(0),
            stride_out_n=output.stride(1),
            group_size=quant_config.group_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=bsn,
            BLOCK_SIZE_K=bsk,
            has_zp=has_zp_w2,
            use_int8_w8a16=quant_config.use_int8,
            even_Ks=even_Ks_down,
            DOWN_GRID_N_FIRST=down_grid_n_first,
            INTER_PREWEIGHTED=preweight_intermediate,
            SMALL_TOKEN_MXQ_PATH=small_token_mxq_path,
            SWAP_AB=swap_ab,
            compute_type=compute_type,
            num_warps=pin["num_warps"],
            num_stages=pin["num_stages"],
        )
        return

    if _mxq_use_down_gs128_fast(
        num_valid_tokens, I, quant_config, has_zp_w2, even_Ks_down
    ):

        def _grid_down_gs128(META):
            del META
            h_tiles = triton.cdiv(H, bsn_fast)
            if down_grid_n_first:
                return (h_tiles, num_blocks_m)
            return (num_blocks_m, h_tiles)

        fused_moe_kernel_w8a16_down_gs128[_grid_down_gs128](
            intermediate,
            W2_q,
            W2_scales,
            output,
            sorted_token_ids,
            expert_ids_per_block,
            sorted_weights,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            H=H,
            I=I,
            stride_inter_m=intermediate.stride(0),
            stride_inter_k=intermediate.stride(1),
            stride_w2_e=W2_q.stride(0),
            stride_w2_n=W2_q.stride(1),
            stride_w2_k=W2_q.stride(2),
            stride_s_e=W2_scales.stride(0),
            stride_s_n=W2_scales.stride(1),
            stride_s_k=W2_scales.stride(2),
            stride_out_t=output.stride(0),
            stride_out_n=output.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=bsn_fast,
            BLOCK_SIZE_K=bsk_fast,
            DOWN_GRID_N_FIRST=down_grid_n_first,
            INTER_PREWEIGHTED=preweight_intermediate,
            SMALL_TOKEN_MXQ_PATH=small_token_mxq_path,
            compute_type=compute_type,
            num_warps=8,
            num_stages=1,
        )
        return

    def _grid_down(META):
        h_tiles = triton.cdiv(H, META["BLOCK_SIZE_N"])
        if down_grid_n_first:
            return (h_tiles, num_blocks_m)
        return (num_blocks_m, h_tiles)

    fused_moe_kernel_w8a16_down[_grid_down](
        intermediate,
        W2_q,
        W2_scales,
        zp2,
        output,
        sorted_token_ids,
        expert_ids_per_block,
        sorted_weights,
        M_padded=num_post_padded,
        T=num_valid_tokens,
        H=H,
        I=I,
        stride_inter_m=intermediate.stride(0),
        stride_inter_k=intermediate.stride(1),
        stride_w2_e=W2_q.stride(0),
        stride_w2_n=W2_q.stride(1),
        stride_w2_k=W2_q.stride(2),
        stride_s_e=W2_scales.stride(0),
        stride_s_n=W2_scales.stride(1),
        stride_s_k=W2_scales.stride(2),
        stride_zp_e=stride_zp_e,
        stride_zp_n=stride_zp_n,
        stride_zp_k=stride_zp_k,
        stride_out_t=output.stride(0),
        stride_out_n=output.stride(1),
        group_size=quant_config.group_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        has_zp=has_zp_w2,
        use_int8_w8a16=quant_config.use_int8,
        even_Ks=even_Ks_down,
        DOWN_GRID_N_FIRST=down_grid_n_first,
        INTER_PREWEIGHTED=preweight_intermediate,
        SMALL_TOKEN_MXQ_PATH=small_token_mxq_path,
        SWAP_AB=swap_ab,
        compute_type=compute_type,
    )


def _mxq_down_grid_n_first(num_valid_tokens: int) -> bool:
    return num_valid_tokens >= 64


def _mxq_fused_gateup_silu_large_min_tokens() -> int:
    """Min valid tokens to select ``gateup_silu_large`` (no zp, gs=128 fast path)."""
    return 1024


def _mxq_unified_mi_max_tokens() -> int:
    """Max tokens for unified MI (no INTER).  Default **1** (170907): only T=1 fused."""
    return 1


def _mxq_use_unified_mi_fusion(num_valid_tokens: int) -> bool:
    """True → MI grid ``(M, I_tile)`` — only T<=MI_MAX (default 1)."""
    return num_valid_tokens <= _mxq_unified_mi_max_tokens()


def _prepare_bsm_routing_mxq_cached(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_size_m: int,
):
    return _prepare_bsm_routing(
        topk_ids, topk_weights, num_tokens, top_k, num_experts, block_size_m
    )


def _mxq_alloc_intermediate_buffer(
    device: torch.device,
    m_padded: int,
    i_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty((m_padded, i_dim), dtype=dtype, device=device)


@triton.jit
def moe_direct_route_padded_kernel(
    topk_ids,
    topk_weights,
    stride_tid,
    stride_tk,
    stride_wt,
    stride_wk,
    out_token_ids,
    out_expert_ids,
    out_weights,
    num_dispatch,
    num_tokens,
    top_k: tl.constexpr,
    block_size_m: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Expand routes directly into one padded BSM block per dispatch."""
    row_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dispatch_ids = row_ids // block_size_m
    lanes = row_ids - dispatch_ids * block_size_m
    valid_rows = dispatch_ids < num_dispatch
    route_mask = valid_rows & (lanes == 0)

    token_ids = dispatch_ids // top_k
    topk_slots = dispatch_ids - token_ids * top_k
    expert_ids = tl.load(
        topk_ids + token_ids * stride_tid + topk_slots * stride_tk,
        mask=route_mask,
        other=0,
    )
    weights = tl.load(
        topk_weights + token_ids * stride_wt + topk_slots * stride_wk,
        mask=route_mask,
        other=0.0,
    )

    padded_token_ids = tl.where(route_mask, token_ids, num_tokens)
    padded_weights = tl.where(route_mask, weights, 0.0)
    tl.store(
        out_token_ids + row_ids,
        padded_token_ids.to(tl.int64),
        mask=valid_rows,
    )
    tl.store(out_weights + row_ids, padded_weights, mask=valid_rows)
    tl.store(
        out_expert_ids + dispatch_ids,
        expert_ids.to(tl.int64),
        mask=route_mask,
    )


def _prepare_direct_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
    block_size_m: int,
):
    """Build BSM routing in one launch when expert reuse is negligible."""
    num_dispatch = num_tokens * top_k
    num_post_padded = num_dispatch * block_size_m
    device = topk_ids.device
    sorted_token_ids = torch.empty(num_post_padded, dtype=torch.int64, device=device)
    expert_ids_per_block = torch.empty(num_dispatch, dtype=torch.int64, device=device)
    sorted_weights = torch.empty(
        num_post_padded, dtype=topk_weights.dtype, device=device
    )
    route_block = 256
    grid = (triton.cdiv(num_post_padded, route_block),)
    moe_direct_route_padded_kernel[grid](
        topk_ids,
        topk_weights,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        sorted_token_ids,
        expert_ids_per_block,
        sorted_weights,
        num_dispatch,
        num_tokens,
        top_k,
        block_size_m,
        BLOCK=route_block,
        num_warps=4,
    )
    return sorted_token_ids, expert_ids_per_block, sorted_weights, num_post_padded


@triton.jit
def moe_bsm_route_count_kernel(
    topk_ids,
    stride_tid,
    stride_tk,
    counts,
    num_dispatch,
    top_k_ptr,
):
    pid = tl.program_id(0)
    if pid >= num_dispatch:
        return
    tk = tl.load(top_k_ptr).to(tl.int32)
    t = pid // tk
    rk = pid - t * tk
    eid = tl.load(topk_ids + t * stride_tid + rk * stride_tk).to(tl.int32)
    tl.atomic_add(counts + eid, 1)


@triton.jit
def moe_bsm_route_scatter_kernel(
    topk_ids,
    topk_weights,
    stride_tid,
    stride_tk,
    stride_wt,
    stride_wk,
    new_offsets,
    cursor,
    out_tid,
    out_w,
    num_dispatch,
    num_tokens,
    top_k_ptr,
):
    pid = tl.program_id(0)
    if pid >= num_dispatch:
        return
    tk = tl.load(top_k_ptr).to(tl.int32)
    t = pid // tk
    rk = pid - t * tk
    eid = tl.load(topk_ids + t * stride_tid + rk * stride_tk).to(tl.int32)
    tok = t.to(tl.int64)
    w = tl.load(topk_weights + t * stride_wt + rk * stride_wk)
    slot = tl.atomic_add(cursor + eid, 1)
    base = tl.load(new_offsets + eid.to(tl.int64))
    pos = base + slot.to(tl.int64)
    tl.store(out_tid + pos, tok)
    tl.store(out_w + pos, w)


def _prepare_bsm_routing_triton(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_size_m: int,
):
    """BSM routing via bucket histogram + atomic scatter (B1)."""
    device = topk_ids.device
    num_dispatch = num_tokens * top_k
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    top_k_ptr = torch.tensor([top_k], dtype=torch.int32, device=device)

    grid = (num_dispatch,)
    moe_bsm_route_count_kernel[grid](
        topk_ids,
        topk_ids.stride(0),
        topk_ids.stride(1),
        counts,
        num_dispatch,
        top_k_ptr,
    )

    counts_i64 = counts.to(torch.int64)
    padded_counts = ((counts_i64 + block_size_m - 1) // block_size_m) * block_size_m
    num_post_padded = int(padded_counts.sum().item())

    new_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    new_offsets[1:] = padded_counts.cumsum(0)

    sorted_token_ids_out = torch.full(
        (num_post_padded,), num_tokens, dtype=torch.int64, device=device
    )
    sorted_weights_out = torch.zeros(
        num_post_padded, dtype=topk_weights.dtype, device=device
    )
    cursor = torch.zeros(num_experts, dtype=torch.int32, device=device)

    moe_bsm_route_scatter_kernel[grid](
        topk_ids,
        topk_weights,
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        new_offsets,
        cursor,
        sorted_token_ids_out,
        sorted_weights_out,
        num_dispatch,
        num_tokens,
        top_k_ptr,
    )

    block_starts = torch.arange(
        0, num_post_padded, block_size_m, dtype=torch.int64, device=device
    )
    expert_ids_per_block = torch.searchsorted(new_offsets, block_starts, right=True) - 1

    return (
        sorted_token_ids_out,
        expert_ids_per_block,
        sorted_weights_out,
        num_post_padded,
    )


def _prepare_bsm_routing_py(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_size_m: int,
):
    """Legacy BSM routing: stable argsort by expert + PyTorch scatter."""
    device = topk_ids.device
    num_dispatch = num_tokens * top_k

    flat_token_ids = (
        torch.arange(num_tokens, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .contiguous()
        .view(-1)
    )
    flat_expert_ids = topk_ids.contiguous().view(-1).to(torch.int64)
    flat_weights = topk_weights.contiguous().view(-1)

    sort_indices = torch.argsort(flat_expert_ids, stable=True)
    sorted_tids_unpadded = flat_token_ids[sort_indices]
    sorted_eids_unpadded = flat_expert_ids[sort_indices]
    sorted_w_unpadded = flat_weights[sort_indices]

    counts = torch.bincount(sorted_eids_unpadded, minlength=num_experts)
    padded_counts = ((counts + block_size_m - 1) // block_size_m) * block_size_m
    num_post_padded = int(padded_counts.sum().item())

    new_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    new_offsets[1:] = padded_counts.cumsum(0)
    old_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    old_offsets[1:] = counts.cumsum(0)

    expert_idx = sorted_eids_unpadded
    pos_within = (
        torch.arange(num_dispatch, device=device, dtype=torch.int64)
        - old_offsets[expert_idx]
    )
    new_positions = new_offsets[expert_idx] + pos_within

    sorted_token_ids_out = torch.full(
        (num_post_padded,), num_tokens, dtype=torch.int64, device=device
    )
    sorted_weights_out = torch.zeros(
        num_post_padded, dtype=flat_weights.dtype, device=device
    )
    sorted_token_ids_out[new_positions] = sorted_tids_unpadded
    sorted_weights_out[new_positions] = sorted_w_unpadded

    blocks_per_expert = padded_counts // block_size_m
    expert_ids_per_block = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int64),
        blocks_per_expert,
    )

    return (
        sorted_token_ids_out,
        expert_ids_per_block,
        sorted_weights_out,
        num_post_padded,
    )


def _prepare_bsm_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_size_m: int,
):
    """Build BSM-aligned routing tensors.

    Returns
    -------
    sorted_token_ids : (num_post_padded,) int64
        Token index per row; padding rows store the sentinel value `num_tokens`.
    expert_ids_per_block : (num_blocks,) int64
        Expert index per BSM block (one entry per block, num_blocks =
        num_post_padded // block_size_m).
    sorted_weights : (num_post_padded,) same dtype as topk_weights
        Routing weights; padding rows store 0.0.
    num_post_padded : int
    """
    if 1 != 0:
        return _prepare_bsm_routing_triton(
            topk_ids, topk_weights, num_tokens, top_k, num_experts, block_size_m
        )
    return _prepare_bsm_routing_py(
        topk_ids, topk_weights, num_tokens, top_k, num_experts, block_size_m
    )


def _bsm_block_m_for_avg_load(avg_tokens_per_expert: int, num_tokens: int) -> int:
    """Map average routed load per expert to a routing block size."""
    if avg_tokens_per_expert <= 16:
        if num_tokens <= 4:
            return 4
        if num_tokens <= 64:
            return 8
        return 16
    if avg_tokens_per_expert <= 32:
        return 32
    if avg_tokens_per_expert <= 48:
        return 48
    return 64


def _select_bsm_block_m(num_tokens: int, num_experts: int, top_k: int) -> int:
    experts = max(int(num_experts), 1)
    avg_tokens_per_expert = (int(num_tokens) * int(top_k)) // experts
    return _bsm_block_m_for_avg_load(avg_tokens_per_expert, int(num_tokens))


def _mxq_bsm_avg_load_max_tokens() -> int:
    """Max tokens for avg-load BSM routing; keeps T=1024 on BSM32 by default."""
    return 1024


def _select_bsm_block_m_rollback_large_path(_num_tokens: int) -> int:
    return 64


def invoke_fused_moe_full_swiglu(
    x: torch.Tensor,
    W1_q: torch.Tensor,
    W1_scales: torch.Tensor,
    W1_zeros: Optional[torch.Tensor],
    W2_q: torch.Tensor,
    W2_scales: torch.Tensor,
    W2_zeros: Optional[torch.Tensor],
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids_per_block: torch.Tensor,
    sorted_weights: torch.Tensor,
    num_post_padded: int,
    num_valid_tokens: int,
    quant_config: Any,
) -> None:
    """Full SwiGLU MoE: gate_up = W1@x, h = silu(gate)*up, output = W2@h, weighted sum.

    Layout convention (matches ``fused_experts_impl``):
        - W1: (E, 2*I, H)  — gate-up combined, first I rows are gate, last I rows are up
        - W2: (E, H, I)    — down projection
    Routing tensors come from ``_prepare_bsm_routing`` so each BSM block
    contains rows belonging to a single expert.

    ``output`` must be pre-zeroed.

    Execution paths (most-specific first):

      - **Unified MoE** (``_use_unified_moe_kernel()``): small batch uses single-kernel
        ``*_unified_moe`` (MI, no INTER buffer).  Larger batch uses split512-equivalent
        **B2** (``gateup_silu`` [``_large``] + ``down``, 2 launches / one CUDA-Graph chain)
        unified flag is unset).

        writes ``(M_padded, I)``; down reads it (no ``(M_padded, 2*I)`` gate_up buffer).

        silu_mul, down GEMM.
    """
    if not x.is_contiguous():
        x = x.contiguous()

    T, H = x.shape
    Nw1 = W1_q.shape[1]
    assert Nw1 % 2 == 0, "W1.shape[1] must be 2*intermediate_size"
    intermediate_size = Nw1 // 2
    H_w2 = W2_q.shape[1]
    I_w2 = W2_q.shape[2]
    assert H_w2 == H, f"W2.shape[1]={H_w2} must equal H={H}"
    assert (
        I_w2 == intermediate_size
    ), f"W2.shape[2]={I_w2} must equal intermediate_size={intermediate_size}"

    # SMALL_TOKEN_MXQ_PATH: T<=split (default 512); same bound as routing split threshold.
    small_token_mxq_path = _mxq_small_token_mxq_path(num_valid_tokens)

    # NOTE: Tile K/N/warps/stages come from each kernel's `@triton.autotune`.
    # BLOCK_SIZE_M is inferred from routing: one BSM block row count per program.
    BLOCK_SIZE_M = num_post_padded // max(int(expert_ids_per_block.numel()), 1)

    # Splitting gateup and SwiGLU for the mid-token range lowers register pressure
    # and lets each stage use a tile specialized for the short-M workload.
    use_fused_gateup_silu = True
    three_kernel_min_tokens = 64
    three_kernel_max_tokens = 1024
    if (
        three_kernel_min_tokens > 0
        and three_kernel_min_tokens <= num_valid_tokens <= three_kernel_max_tokens
    ):
        use_fused_gateup_silu = False

    # Conservative even_Ks: True iff every BSK candidate in the relevant
    # autotune list divides the contraction dim.  Computed per-kernel because
    # the fused kernel uses a different config list.
    _bsks_legacy = {c.kwargs["BLOCK_SIZE_K"] for c in _W8A16_AUTOTUNE_CONFIGS}
    _bsks_fused = {c.kwargs["BLOCK_SIZE_K"] for c in _W8A16_FUSED_AUTOTUNE_CONFIGS}
    _bsks_fused_large = {
        c.kwargs["BLOCK_SIZE_K"] for c in _W8A16_FUSED_LARGE_AUTOTUNE_CONFIGS
    }
    _bsks_down = {c.kwargs["BLOCK_SIZE_K"] for c in _W8A16_DOWN_AUTOTUNE_CONFIGS}
    if use_fused_gateup_silu:
        even_Ks_gateup = all((H % bsk) == 0 for bsk in _bsks_fused)
    else:
        even_Ks_gateup = all((H % bsk) == 0 for bsk in _bsks_legacy)
    even_Ks_gateup_large = all((H % bsk) == 0 for bsk in _bsks_fused_large)
    even_Ks_down = all((intermediate_size % bsk) == 0 for bsk in _bsks_down)

    if x.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif x.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    has_zp_w1 = (
        quant_config.has_zero_point
        and W1_zeros is not None
        and W1_zeros.numel() > 0
        and W1_zeros.dim() == 3
    )
    has_zp_w2 = (
        quant_config.has_zero_point
        and W2_zeros is not None
        and W2_zeros.numel() > 0
        and W2_zeros.dim() == 3
    )

    if W1_zeros is None or W1_zeros.numel() == 0 or W1_zeros.dim() != 3:
        zp1 = torch.empty(0, dtype=torch.uint8, device=x.device)
        s_zp1_e = s_zp1_n = s_zp1_k = 0
    else:
        zp1 = W1_zeros
        s_zp1_e = W1_zeros.stride(0)
        s_zp1_n = W1_zeros.stride(1)
        s_zp1_k = W1_zeros.stride(2)

    if W2_zeros is None or W2_zeros.numel() == 0 or W2_zeros.dim() != 3:
        zp2 = torch.empty(0, dtype=torch.uint8, device=x.device)
        s_zp2_e = s_zp2_n = s_zp2_k = 0
    else:
        zp2 = W2_zeros
        s_zp2_e = W2_zeros.stride(0)
        s_zp2_n = W2_zeros.stride(1)
        s_zp2_k = W2_zeros.stride(2)

    num_blocks_m = num_post_padded // BLOCK_SIZE_M

    down_grid_n_first = _mxq_down_grid_n_first(num_valid_tokens)
    preweight_intermediate = _mxq_preweight_intermediate(num_valid_tokens)

    even_Ks_unified_h = all((H % bsk) == 0 for bsk in _BSKS_UNIFIED_MOE_KH)
    even_Ks_unified_i = all(
        (intermediate_size % bsk) == 0 for bsk in _BSKS_UNIFIED_MOE_IT
    )
    use_unified_moe_path = (
        _use_unified_moe_kernel()
        and quant_config.use_int8
        and not quant_config.use_int4
        and even_Ks_unified_h
        and even_Ks_unified_i
        and use_fused_gateup_silu
    )

    if use_unified_moe_path:
        # T<=MI_MAX → MI; else split B2 (170907).
        if _mxq_use_unified_mi_fusion(num_valid_tokens):

            def _grid_unified_mi(META):
                return (
                    num_blocks_m,
                    triton.cdiv(intermediate_size, META["BLOCK_I_TILE"]),
                )

            fused_moe_kernel_w8a16_unified_moe[_grid_unified_mi](
                x,
                W1_q,
                W1_scales,
                zp1,
                W2_q,
                W2_scales,
                zp2,
                output,
                sorted_token_ids,
                expert_ids_per_block,
                sorted_weights,
                M_padded=num_post_padded,
                T=num_valid_tokens,
                I=intermediate_size,
                H=H,
                stride_a_t=x.stride(0),
                stride_a_k=x.stride(1),
                stride_w1_e=W1_q.stride(0),
                stride_w1_n=W1_q.stride(1),
                stride_w1_k=W1_q.stride(2),
                stride_s1_e=W1_scales.stride(0),
                stride_s1_n=W1_scales.stride(1),
                stride_s1_k=W1_scales.stride(2),
                stride_zp1_e=s_zp1_e,
                stride_zp1_n=s_zp1_n,
                stride_zp1_k=s_zp1_k,
                stride_w2_e=W2_q.stride(0),
                stride_w2_n=W2_q.stride(1),
                stride_w2_k=W2_q.stride(2),
                stride_s2_e=W2_scales.stride(0),
                stride_s2_n=W2_scales.stride(1),
                stride_s2_k=W2_scales.stride(2),
                stride_zp2_e=s_zp2_e,
                stride_zp2_n=s_zp2_n,
                stride_zp2_k=s_zp2_k,
                stride_out_t=output.stride(0),
                stride_out_n=output.stride(1),
                group_size=quant_config.group_size,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                even_Ks_h=even_Ks_unified_h,
                even_Ks_i=even_Ks_unified_i,
                has_zp_w1=has_zp_w1,
                has_zp_w2=has_zp_w2,
                DOWN_GRID_N_FIRST=down_grid_n_first,
                SMALL_TOKEN_MXQ_PATH=small_token_mxq_path,
                INTER_PREWEIGHTED=preweight_intermediate,
                compute_type=compute_type,
            )
        else:
            unified_inter = _mxq_alloc_intermediate_buffer(
                x.device, num_post_padded, intermediate_size, x.dtype
            )
            unified_large_mode = "b2"

            if unified_large_mode == "per_m":

                def _grid_unified_per_m(META):
                    return (num_blocks_m,)

                fused_moe_kernel_w8a16_unified_moe_per_m[_grid_unified_per_m](
                    x,
                    W1_q,
                    W1_scales,
                    zp1,
                    unified_inter,
                    W2_q,
                    W2_scales,
                    zp2,
                    output,
                    sorted_token_ids,
                    expert_ids_per_block,
                    sorted_weights,
                    M_padded=num_post_padded,
                    T=num_valid_tokens,
                    I=intermediate_size,
                    H=H,
                    stride_a_t=x.stride(0),
                    stride_a_k=x.stride(1),
                    stride_w1_e=W1_q.stride(0),
                    stride_w1_n=W1_q.stride(1),
                    stride_w1_k=W1_q.stride(2),
                    stride_s1_e=W1_scales.stride(0),
                    stride_s1_n=W1_scales.stride(1),
                    stride_s1_k=W1_scales.stride(2),
                    stride_zp1_e=s_zp1_e,
                    stride_zp1_n=s_zp1_n,
                    stride_zp1_k=s_zp1_k,
                    stride_inter_m=unified_inter.stride(0),
                    stride_inter_k=unified_inter.stride(1),
                    stride_w2_e=W2_q.stride(0),
                    stride_w2_n=W2_q.stride(1),
                    stride_w2_k=W2_q.stride(2),
                    stride_s2_e=W2_scales.stride(0),
                    stride_s2_n=W2_scales.stride(1),
                    stride_s2_k=W2_scales.stride(2),
                    stride_zp2_e=s_zp2_e,
                    stride_zp2_n=s_zp2_n,
                    stride_zp2_k=s_zp2_k,
                    stride_out_t=output.stride(0),
                    stride_out_n=output.stride(1),
                    group_size=quant_config.group_size,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    even_Ks_h=even_Ks_gateup,
                    even_Ks_i=even_Ks_down,
                    has_zp_w1=has_zp_w1,
                    has_zp_w2=has_zp_w2,
                    INTER_PREWEIGHTED=preweight_intermediate,
                    SMALL_TOKEN_MXQ_PATH=small_token_mxq_path,
                    compute_type=compute_type,
                )
            else:
                # B2: same kernels/autotune as split512 rollback (best large-batch).
                large_gateup_enabled = 1 != 0
                large_gateup_threshold = _mxq_fused_gateup_silu_large_min_tokens()
                use_large_gateup_silu = (
                    large_gateup_enabled
                    and num_valid_tokens >= large_gateup_threshold
                    and quant_config.use_int8
                    and not quant_config.use_int4
                    and quant_config.group_size == 128
                    and not has_zp_w1
                    and even_Ks_gateup_large
                )

                if use_large_gateup_silu:
                    _launch_w8a16_gateup_silu_large(
                        x,
                        W1_q,
                        W1_scales,
                        unified_inter,
                        sorted_token_ids,
                        expert_ids_per_block,
                        sorted_weights,
                        num_post_padded=num_post_padded,
                        num_valid_tokens=num_valid_tokens,
                        I=intermediate_size,
                        H=H,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        preweight_intermediate=preweight_intermediate,
                        compute_type=compute_type,
                        quant_config=quant_config,
                        has_zp_w1=has_zp_w1,
                        even_Ks_gateup_large=even_Ks_gateup_large,
                    )
                else:
                    _launch_w8a16_gateup_silu(
                        x,
                        W1_q,
                        W1_scales,
                        zp1,
                        unified_inter,
                        sorted_token_ids,
                        expert_ids_per_block,
                        sorted_weights,
                        num_post_padded=num_post_padded,
                        num_valid_tokens=num_valid_tokens,
                        I=intermediate_size,
                        H=H,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        preweight_intermediate=preweight_intermediate,
                        compute_type=compute_type,
                        quant_config=quant_config,
                        has_zp_w1=has_zp_w1,
                        even_Ks_gateup=even_Ks_gateup,
                        stride_zp_e=s_zp1_e,
                        stride_zp_n=s_zp1_n,
                        stride_zp_k=s_zp1_k,
                    )

                _launch_w8a16_down(
                    unified_inter,
                    W2_q,
                    W2_scales,
                    zp2,
                    output,
                    sorted_token_ids,
                    expert_ids_per_block,
                    sorted_weights,
                    num_post_padded=num_post_padded,
                    num_valid_tokens=num_valid_tokens,
                    H=H,
                    I=intermediate_size,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    quant_config=quant_config,
                    has_zp_w2=has_zp_w2,
                    even_Ks_down=even_Ks_down,
                    down_grid_n_first=down_grid_n_first,
                    preweight_intermediate=preweight_intermediate,
                    small_token_mxq_path=small_token_mxq_path,
                    compute_type=compute_type,
                    stride_zp_e=s_zp2_e,
                    stride_zp_n=s_zp2_n,
                    stride_zp_k=s_zp2_k,
                )
        return

    intermediate = _mxq_alloc_intermediate_buffer(
        x.device, num_post_padded, intermediate_size, x.dtype
    )
    if use_fused_gateup_silu:
        large_gateup_enabled = 1 != 0
        large_gateup_threshold = _mxq_fused_gateup_silu_large_min_tokens()
        use_large_gateup_silu = (
            large_gateup_enabled
            and num_valid_tokens >= large_gateup_threshold
            and quant_config.use_int8
            and not quant_config.use_int4
            and quant_config.group_size == 128
            and not has_zp_w1
            and even_Ks_gateup_large
        )

        # ============ B2 path: fused gate-up + SwiGLU, 2 kernels total ============
        # Kernel 1: gate-up GEMM with SwiGLU fused, writes (M_padded, I) directly.
        if use_large_gateup_silu:
            _launch_w8a16_gateup_silu_large(
                x,
                W1_q,
                W1_scales,
                intermediate,
                sorted_token_ids,
                expert_ids_per_block,
                sorted_weights,
                num_post_padded=num_post_padded,
                num_valid_tokens=num_valid_tokens,
                I=intermediate_size,
                H=H,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                preweight_intermediate=preweight_intermediate,
                compute_type=compute_type,
                quant_config=quant_config,
                has_zp_w1=has_zp_w1,
                even_Ks_gateup_large=even_Ks_gateup_large,
            )
        else:
            _launch_w8a16_gateup_silu(
                x,
                W1_q,
                W1_scales,
                zp1,
                intermediate,
                sorted_token_ids,
                expert_ids_per_block,
                sorted_weights,
                num_post_padded=num_post_padded,
                num_valid_tokens=num_valid_tokens,
                I=intermediate_size,
                H=H,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                preweight_intermediate=preweight_intermediate,
                compute_type=compute_type,
                quant_config=quant_config,
                has_zp_w1=has_zp_w1,
                even_Ks_gateup=even_Ks_gateup,
                stride_zp_e=s_zp1_e,
                stride_zp_n=s_zp1_n,
                stride_zp_k=s_zp1_k,
            )
    else:
        # ============ Legacy path: gate-up GEMM -> silu_mul -> down (3 kernels) ============
        gate_up = torch.empty((num_post_padded, Nw1), dtype=x.dtype, device=x.device)

        def _grid_gateup(META):
            return (num_blocks_m, triton.cdiv(Nw1, META["BLOCK_SIZE_N"]))

        fused_moe_kernel_w8a16_gateup[_grid_gateup](
            x,
            W1_q,
            W1_scales,
            zp1,
            gate_up,
            sorted_token_ids,
            expert_ids_per_block,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            Nw1=Nw1,
            H=H,
            stride_a_t=x.stride(0),
            stride_a_k=x.stride(1),
            stride_w1_e=W1_q.stride(0),
            stride_w1_n=W1_q.stride(1),
            stride_w1_k=W1_q.stride(2),
            stride_s_e=W1_scales.stride(0),
            stride_s_n=W1_scales.stride(1),
            stride_s_k=W1_scales.stride(2),
            stride_zp_e=s_zp1_e,
            stride_zp_n=s_zp1_n,
            stride_zp_k=s_zp1_k,
            stride_gu_m=gate_up.stride(0),
            stride_gu_n=gate_up.stride(1),
            group_size=quant_config.group_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            has_zp=has_zp_w1,
            use_int8_w8a16=quant_config.use_int8,
            even_Ks=even_Ks_gateup,
            SWAP_AB=num_valid_tokens == three_kernel_min_tokens,
            compute_type=compute_type,
        )

        SWIGLU_BSM = 32
        SWIGLU_BSI = 256
        grid2 = (
            triton.cdiv(num_post_padded, SWIGLU_BSM),
            triton.cdiv(intermediate_size, SWIGLU_BSI),
        )
        silu_mul_kernel[grid2](
            gate_up,
            intermediate,
            sorted_token_ids,
            sorted_weights,
            M_padded=num_post_padded,
            T=num_valid_tokens,
            I=intermediate_size,
            stride_gu_m=gate_up.stride(0),
            stride_gu_n=gate_up.stride(1),
            stride_inter_m=intermediate.stride(0),
            stride_inter_n=intermediate.stride(1),
            BLOCK_SIZE_M=SWIGLU_BSM,
            BLOCK_SIZE_I=SWIGLU_BSI,
            APPLY_ROUTED_WEIGHT=preweight_intermediate,
            compute_type=compute_type,
            num_warps=4,
            num_stages=1,
        )

        del gate_up

    # ---------------- down GEMM (full N=H, atomic_add) ----------------
    _launch_w8a16_down(
        intermediate,
        W2_q,
        W2_scales,
        zp2,
        output,
        sorted_token_ids,
        expert_ids_per_block,
        sorted_weights,
        num_post_padded=num_post_padded,
        num_valid_tokens=num_valid_tokens,
        H=H,
        I=intermediate_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        quant_config=quant_config,
        has_zp_w2=has_zp_w2,
        even_Ks_down=even_Ks_down,
        down_grid_n_first=down_grid_n_first,
        preweight_intermediate=preweight_intermediate,
        small_token_mxq_path=small_token_mxq_path,
        compute_type=compute_type,
        stride_zp_e=s_zp2_e,
        stride_zp_n=s_zp2_n,
        stride_zp_k=s_zp2_k,
    )


def fused_marlin_moe_w8a16_int8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    group_size: int = 128,
    inplace: bool = False,
) -> torch.Tensor:
    """Run fused Marlin MoE with groupwise INT8 weights and A16 activations.

    This is the optimized full-SwiGLU path migrated from ``fused_moe_mxq.py``.
    It consumes the vLLM/Marlin-style public API tensors directly:
    ``w1=(E, 2I, H)`` and ``w2=(E, H, I)`` with uint8 groupwise W8A16 scales.
    """
    assert hidden_states.dtype in (torch.float16, torch.bfloat16)
    assert hidden_states.is_contiguous()
    assert w1.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert w1.stride(-1) == 1 and w2.stride(-1) == 1
    assert topk_weights.shape == topk_ids.shape

    num_tokens = hidden_states.size(0)
    num_experts = w1.size(0)
    top_k_num = topk_ids.size(1)

    if inplace:
        output = hidden_states
        output.zero_()
    else:
        output = torch.zeros_like(hidden_states)

    quant_config = QuantConfig(
        mode=QuantMode.W8A16,
        group_size=group_size,
        has_zero_point=w1_zeros is not None or w2_zeros is not None,
        per_channel_quant=False,
    )

    if num_tokens <= 16:
        bsm_block_m = (
            1
            if num_tokens == 16
            else _select_bsm_block_m(num_tokens, num_experts, top_k_num)
        )
        routing = _prepare_direct_routing(
            topk_ids,
            topk_weights,
            num_tokens,
            top_k_num,
            bsm_block_m,
        )
    else:
        routing = None

    split_th = _mxq_split_small_large_threshold()
    if routing is None and num_tokens <= max(split_th, _mxq_bsm_avg_load_max_tokens()):
        bsm_block_m = _select_bsm_block_m(num_tokens, num_experts, top_k_num)
    elif routing is None:
        bsm_block_m = _select_bsm_block_m_rollback_large_path(num_tokens)

    if routing is None:
        routing = _prepare_bsm_routing_mxq_cached(
            topk_ids,
            topk_weights,
            num_tokens,
            top_k_num,
            num_experts,
            bsm_block_m,
        )
    sorted_token_ids, expert_ids_per_block, sorted_weights, num_post_padded = routing

    invoke_fused_moe_full_swiglu(
        hidden_states,
        w1,
        w1_scale,
        w1_zeros,
        w2,
        w2_scale,
        w2_zeros,
        output,
        sorted_token_ids,
        expert_ids_per_block,
        sorted_weights,
        num_post_padded,
        num_tokens,
        quant_config,
    )
    return output


__all__ = ["fused_marlin_moe_w8a16_int8"]
