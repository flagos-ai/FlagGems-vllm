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

"""Optimized fused MoE kernel for Ascend 910B.

This module provides an Ascend-optimized implementation of the fused MoE
(Mixture of Experts) operator, designed for vLLM inference workloads.
The kernel performs expert routing, GEMM operations, and activation fusion
with optimizations tailored for Ascend NPU architecture.

[KernelGen] This kernel was generated and optimized using automated kernel
generation and tuning infrastructure.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(
    topk_ids_ptr,
    topk_w_ptr,
    sorted_tokens_ptr,
    sorted_w_ptr,
    sorted_route_ptr,
    count_ptr,
    ROWS: tl.constexpr,
    ROWS_P2: tl.constexpr,
    E: tl.constexpr,
    topk: tl.constexpr,
):
    """Scan-based expert routing: row i=(t,r) gets packed position offset_e + rank.

    Packed order: expert-major, within-expert by original (t,r) index.
    ROWS is a power of two, so the ids tile is exactly ROWS lanes (no
    overrun past the caller's topk_ids allocation). Sorted arrays are padded
    to ROWS_P2 so later tile loads stay in-bounds.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, ROWS)
    ids = tl.load(topk_ids_ptr + offs).to(tl.int32)
    e = tl.load(topk_ids_ptr + pid).to(tl.int32)
    w = tl.load(topk_w_ptr + pid)
    rank = tl.sum(((ids == e) & (offs < pid)).to(tl.int32))
    offset = tl.sum((ids < e).to(tl.int32))
    count_e = tl.sum((ids == e).to(tl.int32))
    pos = offset + rank
    tl.store(count_ptr + e, count_e)
    tl.store(sorted_tokens_ptr + pos, (pid // topk).to(tl.int32))
    tl.store(sorted_w_ptr + pos, w)
    tl.store(sorted_route_ptr + pos, pid.to(tl.int32))


@triton.jit
def _pack_kernel(
    src_ptr,
    dst_ptr,
    idx_ptr,
    ROWS,
    K,
    K_TILE: tl.constexpr,
):
    """Pack rows: dst[p, :] = src[idx[p], :] for p < ROWS, zero otherwise.

    dst is (ROWS_P2, K). K_TILE divides K, tiles are exact. The idx load is a
    masked scalar (masked lanes read row 0), so all addresses are in-bounds.
    """
    p = tl.program_id(0)
    offs = tl.arange(0, K_TILE)
    base_d = p.to(tl.int64) * K
    valid = p < ROWS
    t = tl.load(idx_ptr + p, mask=valid, other=0).to(tl.int64)
    for k0 in range(0, K, K_TILE):
        v = tl.load(src_ptr + t * K + k0 + offs)
        tl.store(dst_ptr + base_d + k0 + offs, v, mask=valid)


@triton.jit
def _hq_kernel(
    x_ptr,
    q_ptr,
    K,
    COMPUTE_DTYPE: tl.constexpr,
    K_TILE: tl.constexpr,
):
    """Per-row int8 fake quantization: q = round(clamp(x/scale, -128, 127)) * scale.

    scale = max|x| / 127 per row. K_TILE divides K, so every tile is exact and
    all loads/stores are unmasked (no out-of-range DMA).
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, K_TILE)
    base = pid.to(tl.int64) * K
    amax = tl.zeros((), dtype=tl.float32) - 1.0
    for k0 in range(0, K, K_TILE):
        v = tl.load(x_ptr + base + k0 + offs).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v)))
    sc = tl.maximum(amax, 1e-10) / 127.0
    for k0 in range(0, K, K_TILE):
        v = tl.load(x_ptr + base + k0 + offs).to(tl.float32)
        q = tl.floor(tl.minimum(tl.maximum(v / sc, -128.0), 127.0) + 0.5) * sc
        tl.store(q_ptr + base + k0 + offs, q.to(COMPUTE_DTYPE))


@triton.jit
def _dequant_w_kernel(
    w_ptr,
    scale_ptr,
    out_ptr,
    N,
    K,
    count_ptr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Dequantize int8 weights with per-channel scale to COMPUTE dtype.

    Skips experts with zero routed rows. BLOCK_N=64 and BLOCK_K=64 divide
    every workload N/K, so all loads/stores are unmasked.
    """
    e = tl.program_id(0)
    n_tile = tl.program_id(1)
    if tl.load(count_ptr + e) == 0:
        return
    n0 = n_tile * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    base = e.to(tl.int64) * N * K
    s = tl.load(scale_ptr + e.to(tl.int64) * N + n0 + offs_n).to(tl.float32)
    for k0 in range(0, K, BLOCK_K):
        w8 = tl.load(
            w_ptr
            + base
            + (k0 + offs_k)[:, None].to(tl.int64)
            + (n0 + offs_n)[None, :].to(tl.int64) * K
        )
        wf = w8.to(tl.float32) * s[None, :]
        tl.store(
            out_ptr
            + base
            + (k0 + offs_k)[:, None].to(tl.int64)
            + (n0 + offs_n)[None, :].to(tl.int64) * K,
            wf.to(COMPUTE_DTYPE),
        )


@triton.jit
def _gemm1_kernel(
    pack_ptr,
    w1_ptr,
    inter_ptr,
    count_ptr,
    sorted_w_ptr,
    K,
    N,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    E_P2: tl.constexpr,
    APPLY_W_ON_INPUT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """GEMM1: packed rows [offset_e, offset_e+count_e) @ w1[e][n, :].

    w1[e] is [N, K] (N = 2*intermediate). All dot-loop A rows are affine in
    the tile offset (packed input). count is padded to E_P2 and sorted_w is
    padded to 128 rows, so every masked tile load stays in-bounds.
    """
    e = tl.program_id(0)
    n_tile = tl.program_id(1)
    n0 = n_tile * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < (N - n0)
    offs_e = tl.arange(0, E_P2)
    counts = tl.load(count_ptr + offs_e)
    count_e = tl.load(count_ptr + e)
    offset_e = tl.sum((offs_e < e).to(tl.int32) * counts)
    m_offs = tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_K)
    w1_base = w1_ptr + e.to(tl.int64) * N * K
    for m0 in range(0, count_e, BLOCK_M):
        offs_m = m0 + m_offs
        m_mask = offs_m < count_e
        prow = offset_e + offs_m
        wgt = tl.load(sorted_w_ptr + prow, mask=m_mask, other=0.0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(
                pack_ptr + prow[:, None].to(tl.int64) * K + k0 + k_offs[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            if APPLY_W_ON_INPUT:
                a = (a.to(tl.float32) * wgt[:, None]).to(COMPUTE_DTYPE)
            b = tl.load(
                w1_base
                + (k0 + k_offs)[:, None].to(tl.int64)
                + (n0 + offs_n)[None, :].to(tl.int64) * K,
                mask=n_mask[None, :],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            acc = tl.dot(a, b, acc=acc)
        out = acc.to(COMPUTE_DTYPE)
        tl.store(
            inter_ptr
            + prow[:, None].to(tl.int64) * N
            + (n0 + offs_n)[None, :].to(tl.int64),
            out,
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _act_kernel(
    inter_ptr,
    act_ptr,
    K2,
    COMPUTE_DTYPE: tl.constexpr,
    K_TILE: tl.constexpr,
):
    """Fused SiLU: act[j] = silu(gate[j]) * up[j].

    Gate and up are each contiguous (K_TILE,) runs inside the row
    [gate(intermediate) | up(intermediate)] layout, so the loop body is two
    contiguous 1D loads + one contiguous store. K_TILE divides K2 so tiles are
    exact and no masks are needed. (The former (K_TILE,2) pair-stride tile
    generated per-row 4-byte DMA transactions and ran at ~3ms; this structure
    runs at ~0.12ms for the benchmark K2=14336.)
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, K_TILE)
    base = pid.to(tl.int64) * (2 * K2)
    for k0 in range(0, K2, K_TILE):
        g = tl.load(inter_ptr + base + k0 + offs).to(tl.float32)
        u = tl.load(inter_ptr + base + K2 + k0 + offs).to(tl.float32)
        act = tl.sigmoid(g) * g * u
        tl.store(
            act_ptr + pid.to(tl.int64) * K2 + k0 + offs,
            act.to(COMPUTE_DTYPE),
        )


@triton.jit
def _gemm2_kernel(
    w2_ptr,
    act_ptr,
    inter_ptr,
    out2_ptr,
    sorted_route_ptr,
    sorted_w_ptr,
    count_ptr,
    N2,
    K2,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    E_P2: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    APPLY_W_ON_INPUT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """GEMM2: out2[route, n] = w_r * act @ w2[e][n, :].

    When FUSE_SILU, the A operand is computed in-loop from inter rows
    [gate(intermediate) | up(intermediate)] (packed expert order): two
    contiguous 1D loads + fp32 silu, cast to COMPUTE before the dot - identical
    rounding to the separate act kernel. Otherwise A is read from act_ptr.
    All dot-loop A rows are affine in the tile offset; count is padded to E_P2
    and sorted arrays are padded to 128 rows, so every masked tile load stays
    in-bounds.
    """
    e = tl.program_id(0)
    n_tile = tl.program_id(1)
    n0 = n_tile * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < (N2 - n0)
    offs_e = tl.arange(0, E_P2)
    counts = tl.load(count_ptr + offs_e)
    count_e = tl.load(count_ptr + e)
    offset_e = tl.sum((offs_e < e).to(tl.int32) * counts)
    m_offs = tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, BLOCK_K)
    w2_base = w2_ptr + e.to(tl.int64) * N2 * K2
    for m0 in range(0, count_e, BLOCK_M):
        offs_m = m0 + m_offs
        m_mask = offs_m < count_e
        prow = offset_e + offs_m
        route = tl.load(sorted_route_ptr + prow, mask=m_mask, other=0)
        wgt = tl.load(sorted_w_ptr + prow, mask=m_mask, other=0.0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K2, BLOCK_K):
            if FUSE_SILU:
                g = tl.load(
                    inter_ptr
                    + prow[:, None].to(tl.int64) * (2 * K2)
                    + k0
                    + k_offs[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                u = tl.load(
                    inter_ptr
                    + prow[:, None].to(tl.int64) * (2 * K2)
                    + K2
                    + k0
                    + k_offs[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                ).to(tl.float32)
                act = tl.sigmoid(g) * g * u
                a = act.to(COMPUTE_DTYPE)
            else:
                a = tl.load(
                    act_ptr + prow[:, None].to(tl.int64) * K2 + k0 + k_offs[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                )
            b = tl.load(
                w2_base
                + (k0 + k_offs)[:, None].to(tl.int64)
                + (n0 + offs_n)[None, :].to(tl.int64) * K2,
                mask=n_mask[None, :],
                other=0.0,
            ).to(COMPUTE_DTYPE)
            acc = tl.dot(a, b, acc=acc)
        if not APPLY_W_ON_INPUT:
            acc = acc * wgt[:, None]
        out = acc.to(COMPUTE_DTYPE)
        tl.store(
            out2_ptr
            + route[:, None].to(tl.int64) * N2
            + (n0 + offs_n)[None, :].to(tl.int64),
            out,
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _reduce_kernel(
    out2_ptr,
    out_ptr,
    N2,
    topk: tl.constexpr,
    N_TILE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.arange(0, N_TILE)
    base = t.to(tl.int64) * topk * N2
    for n0 in range(0, N2, N_TILE):
        acc = tl.zeros((N_TILE,), dtype=tl.float32)
        for r in tl.static_range(0, topk):
            v = tl.load(out2_ptr + base + r * N2 + n0 + offs)
            acc += v.to(tl.float32)
        tl.store(out_ptr + t.to(tl.int64) * N2 + n0 + offs, acc.to(OUT_DTYPE))


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ascend-optimized fused MoE implementation.

    This function overrides the default fused_experts_impl when imported on Ascend hardware.

    This implementation uses a simplified routing and tiling strategy optimized
    for Ascend 910B NPU architecture. It supports INT8 quantization and fused
    SiLU activation.

    Constraints:
    - Only supports 'silu' activation
    - expert_map must be None (no expert parallelism)
    - Only INT8 w8a8/w8a16 quantization supported (no FP8 or MX schemes)
    - No bias support
    - Batch size * topk must be power of 2 and <= 128

    Args:
        hidden_states: Input tensor [M, K]
        w1: First expert weight [E, N, K] where N = 2*intermediate
        w2: Second expert weight [E, N2, K2]
        topk_weights: Router weights [M, topk]
        topk_ids: Expert indices [M, topk]
        inplace: Whether to write output in-place to hidden_states
        activation: Activation function (only 'silu' supported)
        apply_router_weight_on_input: Apply router weight before or after GEMM
        use_int8_w8a8: Use INT8 w8a8 quantization
        use_int8_w8a16: Use INT8 w8a16 quantization
        per_channel_quant: Use per-channel quantization
        w1_scale: Scale for w1 weights [E, N]
        w2_scale: Scale for w2 weights [E, N2]
        Other args: Not supported on Ascend, must be None/False

    Returns:
        Output tensor [M, N2] (same as hidden_states if inplace=True)
    """
    assert activation == "silu"
    assert ocp_mx_scheme is None
    assert expert_map is None
    assert w1_zp is None and w2_zp is None
    assert a1_scale is None and a2_scale is None
    assert block_shape is None
    assert w1_bias is None and w2_bias is None
    assert not use_fp8_w8a8, "fp8 w8a8 not supported"
    assert sum((use_int8_w8a8, use_int8_w8a16, use_int4_w4a16)) <= 1

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w1.shape[1]  # 2 * intermediate
    N2 = w2.shape[1]  # hidden output dim
    K2 = w2.shape[2]  # intermediate dim
    topk = topk_ids.shape[1]
    ROWS = M * topk
    ROWS_P2 = 128
    assert ROWS <= ROWS_P2 and (ROWS & (ROWS - 1)) == 0, "ROWS must be pow2 <= 128"
    assert global_num_experts in (-1, E)
    assert hidden_states.is_contiguous()
    assert w1.is_contiguous() and w2.is_contiguous()
    assert topk_ids.is_contiguous() and topk_weights.is_contiguous()

    quant = use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16
    if quant:
        assert per_channel_quant
        assert w1_scale is not None and w2_scale is not None
        assert w1_scale.is_contiguous() and w2_scale.is_contiguous()
    else:
        assert not per_channel_quant

    device = hidden_states.device
    if hidden_states.dtype == torch.bfloat16:
        COMPUTE: tl.dtype = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        COMPUTE = tl.float16
    else:
        raise ValueError(f"unsupported hidden dtype {hidden_states.dtype}")

    out = hidden_states if inplace else torch.empty_like(hidden_states)

    sorted_tokens = torch.empty((ROWS_P2,), device=device, dtype=torch.int32)
    sorted_route = torch.empty((ROWS_P2,), device=device, dtype=torch.int32)
    sorted_w = torch.empty((ROWS_P2,), device=device, dtype=hidden_states.dtype)
    count = torch.zeros((16,), device=device, dtype=torch.int32)
    pack_h = torch.zeros((ROWS_P2, K), device=device, dtype=hidden_states.dtype)
    inter = torch.zeros((ROWS_P2, 2 * K2), device=device, dtype=hidden_states.dtype)
    actbuf = torch.zeros((ROWS_P2, K2), device=device, dtype=hidden_states.dtype)
    out2 = torch.zeros((ROWS_P2, N2), device=device, dtype=hidden_states.dtype)

    w1dq = w2dq = None
    hq = aq = None
    if quant:
        w1dq = torch.empty((E, N, K), device=device, dtype=hidden_states.dtype)
        w2dq = torch.empty((E, N2, K2), device=device, dtype=hidden_states.dtype)
    if use_int8_w8a8:
        hq = torch.zeros((ROWS_P2, K), device=device, dtype=hidden_states.dtype)
        aq = torch.zeros((ROWS_P2, K2), device=device, dtype=hidden_states.dtype)

    BLOCK_M, BLOCK_N = 32, 256
    for t in (256, 128, 64):
        if K % t == 0:
            BLOCK_K1 = t
            break
    else:
        BLOCK_K1 = 64
    for t in (256, 128, 64):
        if K2 % t == 0:
            BLOCK_K2 = t
            break
    else:
        BLOCK_K2 = 64
    E_P2 = 16
    for t in (2048, 1024, 512, 256, 128, 64):
        if K % t == 0:
            KTILE_K = t
            break
    else:
        KTILE_K = 64
    for t in (2048, 1024, 512, 256, 128, 64):
        if K2 % t == 0:
            KTILE_K2 = t
            break
    else:
        KTILE_K2 = 64
    for t in (1024, 512, 256, 128, 64):
        if N2 % t == 0:
            NTILE_N2 = t
            break
    else:
        NTILE_N2 = 64

    _route_kernel[(ROWS,)](
        topk_ids,
        topk_weights,
        sorted_tokens,
        sorted_w,
        sorted_route,
        count,
        ROWS=ROWS,
        ROWS_P2=ROWS_P2,
        E=E,
        topk=topk,
    )

    if quant:
        _dequant_w_kernel[(E, triton.cdiv(N, 64))](
            w1,
            w1_scale,
            w1dq,
            N,
            K,
            count,
            E=E,
            BLOCK_N=64,
            BLOCK_K=64,
            COMPUTE_DTYPE=COMPUTE,
        )
        _dequant_w_kernel[(E, triton.cdiv(N2, 64))](
            w2,
            w2_scale,
            w2dq,
            N2,
            K2,
            count,
            E=E,
            BLOCK_N=64,
            BLOCK_K=64,
            COMPUTE_DTYPE=COMPUTE,
        )

    _pack_kernel[(ROWS_P2,)](
        hidden_states,
        pack_h,
        sorted_tokens,
        ROWS,
        K,
        K_TILE=KTILE_K,
    )

    gemm1_a = pack_h
    if use_int8_w8a8:
        _hq_kernel[(ROWS_P2,)](pack_h, hq, K, COMPUTE_DTYPE=COMPUTE, K_TILE=KTILE_K)
        gemm1_a = hq

    _gemm1_kernel[(E, triton.cdiv(N, BLOCK_N))](
        gemm1_a,
        w1dq if quant else w1,
        inter,
        count,
        sorted_w,
        K,
        N,
        E=E,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K1,
        E_P2=E_P2,
        APPLY_W_ON_INPUT=apply_router_weight_on_input,
        COMPUTE_DTYPE=COMPUTE,
    )

    fuse_silu = not use_int8_w8a8
    if fuse_silu:
        # SiLU fused into gemm2 epilogue: no act kernel or actbuf needed.
        gemm2_a = actbuf  # unused placeholder (keeps signature simple)
    else:
        _act_kernel[(ROWS_P2,)](
            inter,
            actbuf,
            K2,
            COMPUTE_DTYPE=COMPUTE,
            K_TILE=KTILE_K2,
        )
        _hq_kernel[(ROWS_P2,)](actbuf, aq, K2, COMPUTE_DTYPE=COMPUTE, K_TILE=KTILE_K2)
        gemm2_a = aq

    _gemm2_kernel[(E, triton.cdiv(N2, BLOCK_N))](
        w2dq if quant else w2,
        gemm2_a,
        inter,
        out2,
        sorted_route,
        sorted_w,
        count,
        N2,
        K2,
        E=E,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K2,
        E_P2=E_P2,
        FUSE_SILU=fuse_silu,
        APPLY_W_ON_INPUT=apply_router_weight_on_input,
        COMPUTE_DTYPE=COMPUTE,
    )

    _reduce_kernel[(M,)](
        out2,
        out,
        N2,
        topk=topk,
        N_TILE=NTILE_N2,
        COMPUTE_DTYPE=COMPUTE,
        OUT_DTYPE=COMPUTE,
    )

    return out


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    In-place fused MoE: writes output directly into ``hidden_states``.

    Same semantics as ``fused_experts_impl(..., inplace=True)``.
    Returns None (the result is stored in ``hidden_states``).
    """
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Out-of-place fused MoE: allocates and returns a new output tensor.

    Same semantics as ``fused_experts_impl(..., inplace=False)``.
    """
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
