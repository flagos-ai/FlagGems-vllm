from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
from flaggems_vllm.ops.FLA.triton_ops_helper import autotune_cache_kwargs, exp2
from flaggems_vllm.ops.FLA.utils import use_cuda_graph

# =============================================================================
# Fused K1: cumsum + intra-chunk attention + WY inverse + w/u/qg/kg/bg
# =============================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_kernel_k1_fused(
    q,
    k,
    a,
    b,
    v,
    gk,
    Aqk,
    Aqb,
    Aab_inv,
    qg_out,
    kg_out,
    bg_out,
    w_out,
    u_out,
    g_last_out,
    scale,
    cu_seqlens,
    chunk_offsets,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    RCP_LN2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_t_global = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    o_i = tl.arange(0, BT)
    m_i = o_i[:, None] >= o_i[None, :]  # Lower triangular mask (bool)
    m_e = o_i[:, None] > o_i[None, :]
    valid_len = min(T - i_t * BT, BT)
    m_mid = (o_i == (valid_len // 2)).to(tl.float32)
    m_last = (o_i == (valid_len - 1)).to(tl.float32)

    offset_base = (bos * H + i_h) * K

    # =========================================================================
    # Phase 1: Accumulate A matrices [BT, BT] over K tiles
    # FP32 for critical paths (qk/qb), FP16 for WY inverse path (ak/ab)
    # =========================================================================
    b_A_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_A_qb = tl.zeros([BT, BT], dtype=tl.float32)
    b_A_ak = tl.zeros(
        [BT, BT], dtype=tl.float32
    )  # FP32 accumulator, convert to FP16 later
    b_A_ab = tl.zeros(
        [BT, BT], dtype=tl.float32
    )  # FP32 accumulator, convert to FP16 later

    for i_k in range(tl.cdiv(K, BK)):
        p_gk = tl.make_block_ptr(
            gk + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        # Load and scale in FP32 for higher precision cumsum
        b_gk_s = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32) * RCP_LN2

        # Cumsum in FP32 for precision (keep FP32 throughout)
        m_cumsum = m_i.to(tl.float32)
        b_gi = tl.dot(m_cumsum, b_gk_s)  # Keep FP32
        b_ge = b_gi - b_gk_s  # Exclusive cumsum (FP32)

        b_off = tl.sum(m_mid[:, None] * b_gi, 0)
        b_gi_c = b_gi - b_off[None, :]
        b_ge_c = b_ge - b_off[None, :]

        p_q = tl.make_block_ptr(
            q + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_a = tl.make_block_ptr(
            a + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_b = tl.make_block_ptr(
            b + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        # Load without conversion - preserve input precision (FP16/BF16)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))

        # Compute exp2 in FP32 for numerical safety (keep FP32, no conversion)
        b_gi_c_exp = exp2(b_gi_c)
        b_gi_c_neg_exp = exp2(-b_gi_c)
        b_ge_c_exp = exp2(b_ge_c)

        # Mixed precision: input(FP16/BF16) * exp(FP32) -> FP32 automatically
        q_ops = (b_q * scale) * b_gi_c_exp
        k_ops = b_k * b_gi_c_neg_exp
        a_ops = b_a * b_ge_c_exp
        b_ops = b_b * b_gi_c_neg_exp

        # Accumulate in FP32 (BF16×BF16 inputs, FP32 accumulator)
        b_A_qk += tl.dot(q_ops, tl.trans(k_ops), out_dtype=tl.float32)
        b_A_qb += tl.dot(q_ops, tl.trans(b_ops), out_dtype=tl.float32)
        b_A_ak += tl.dot(a_ops, tl.trans(k_ops), out_dtype=tl.float32)
        b_A_ab += tl.dot(a_ops, tl.trans(b_ops), out_dtype=tl.float32)
    # Apply masks: keep FP32, convert at storage time
    b_A_qk = tl.where(m_i, b_A_qk, 0)
    b_A_qb = tl.where(m_i, b_A_qb, 0)
    b_A_ak = tl.where(m_e, b_A_ak, 0).to(tl.float16)  # Convert to FP16 for WY inverse
    b_A_ab = tl.where(m_e, b_A_ab, 0).to(tl.float16)  # Convert to FP16 for WY inverse
    # =========================================================================
    # Phase 2: Geometric series inverse (fp32→fp16 direct, skip bf16)
    # =========================================================================
    # Solve (I - L)^{-1} using geometric series product: O(log N) vs O(N)
    # Identity: (I - L)^{-1} = (I + L)(I + L^2)(I + L^4)(I + L^8)...
    # Proof: (I - L)(I + L) = I - L^2, (I - L^2)(I + L^2) = I - L^4, etc.

    # Extract strict lower triangular part (already FP16)
    b_L = tl.where(o_i[:, None] > o_i[None, :], b_A_ab, 0)
    # Precompute identity matrix to avoid redundant computation in loop
    b_I = (o_i[:, None] == o_i[None, :]).to(tl.float16)
    b_Ai = b_I
    L_power = b_L  # L^1

    # 4 iterations for BT=16: covers (I+L)(I+L^2)(I+L^4)(I+L^8)
    # While L^8 often underflows in fp16, empirical tests show 4 iterations
    # preserve model quality better than 3, likely due to:
    #   - Edge cases with higher ||L|| in certain layers/positions
    #   - Accumulated fp16 rounding errors benefit from extra precision
    # Triton automatically unrolls this loop for optimal performance
    for _ in range(4):
        b_Ai = tl.dot(b_Ai, b_I + L_power, out_dtype=tl.float16)
        L_power = tl.dot(L_power, L_power, out_dtype=tl.float16)  # L → L^2 → L^4 → L^8

    # Solve (I - L)^{-1} @ A_ak in fp16
    b_Aak_solved = tl.dot(b_Ai, b_A_ak, out_dtype=tl.float16)

    out_A_base = (bos * H + i_h) * BT
    p_Aqk = tl.make_block_ptr(
        Aqk + out_A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    p_Aqb = tl.make_block_ptr(
        Aqb + out_A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    p_Aai = tl.make_block_ptr(
        Aab_inv + out_A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Aqk, b_A_qk.to(p_Aqk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aqb, b_A_qb.to(p_Aqb.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Aai, b_Ai.to(p_Aai.dtype.element_ty), boundary_check=(0, 1))

    # =========================================================================
    # Phase 3: Per-K outputs (qg, kg, bg, w) and g_last (bf16 computation)
    # =========================================================================
    if IS_VARLEN:
        g_last_base = i_t_global * H + i_h
    else:
        g_last_base = (i_b * tl.cdiv(T, BT) + i_t) * H + i_h
    o_k = tl.arange(0, BK)

    for i_k in range(tl.cdiv(K, BK)):
        p_gk = tl.make_block_ptr(
            gk + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        # Load and scale in FP32 for higher precision cumsum
        b_gk_s = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32) * RCP_LN2

        # Cumsum in FP32 for precision (keep FP32 throughout)
        m_cumsum = m_i.to(tl.float32)
        b_gi = tl.dot(m_cumsum, b_gk_s)  # Keep FP32
        b_ge = b_gi - b_gk_s
        b_g_last = tl.sum(m_last[:, None] * b_gi, 0)

        p_q = tl.make_block_ptr(
            q + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_a = tl.make_block_ptr(
            a + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_b = tl.make_block_ptr(
            b + offset_base, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        # Load without conversion - preserve input precision
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))

        # Compute exp2 in FP32, keep FP32 (numerical safety)
        b_gi_exp = exp2(b_gi)
        b_ge_exp = exp2(b_ge)
        b_gi_last_diff = exp2(-b_gi + b_g_last[None, :])

        # Mixed precision: automatically promoted to FP32
        b_qg = (b_q * scale) * b_gi_exp
        b_kg = b_k * b_gi_last_diff
        b_bg = b_b * b_gi_last_diff
        b_ag = b_a * b_ge_exp
        # Convert b_ag from FP32 to FP16 for WY inverse (b_Ai is FP16)
        b_w = tl.dot(b_Ai, b_ag.to(tl.float16), out_dtype=tl.float16)

        p_qg = tl.make_block_ptr(
            qg_out + offset_base,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_kg = tl.make_block_ptr(
            kg_out + offset_base,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_bg = tl.make_block_ptr(
            bg_out + offset_base,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w_out + offset_base,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_bg, b_bg.to(p_bg.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

        tl.store(
            g_last_out + g_last_base * K + i_k * BK + o_k,
            b_g_last,
            mask=(o_k + i_k * BK) < K,
        )

    # =========================================================================
    # Phase 4: u = (Ai @ A_ak) @ v per V tile (bf16 computation)
    # =========================================================================
    v_base = (bos * H + i_h) * V
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u_out + v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # b_Aak_solved is FP16, convert b_v to match
        b_u = tl.dot(b_Aak_solved, b_v.to(tl.float16), out_dtype=tl.float16)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Fused K2: state propagation + v_new + output
# =============================================================================


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps)
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_kernel_k2_fused(
    # inputs (from K1)
    kg,
    bg,
    w,
    u,
    v,
    qg,
    Aqk,
    Aqb,
    g_last,
    # outputs
    o,
    h0,
    ht,
    # control
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused K2: state recurrence + v_new + output in a single sequential pass."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    # State in registers: [K, BV]
    # Use FP32 for state to avoid type conversion in loop
    BK: tl.constexpr = K
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    # Base pointers
    k_base = (bos * H + i_h) * K
    v_base = (bos * H + i_h) * V
    A_base = (bos * H + i_h) * BT

    # g_last layout: [(boh + i_t) * H + i_h, K]
    if IS_VARLEN:
        chunk_offset_base = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        chunk_offset_base = i_n * tl.cdiv(T, BT)

    for i_t in range(NT):
        # --- v_new = w @ h + u ---
        p_w = tl.make_block_ptr(
            w + k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )

        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_u = tl.load(p_u, boundary_check=(0, 1))
        # dot with low precision inputs (BF16/FP16), FP32 accumulator
        b_v_new_fp32 = tl.dot(b_w, b_h.to(b_w.dtype), out_dtype=tl.float32) + b_u.to(
            tl.float32
        )

        # --- output = qg @ h + Aqk @ v + Aqb @ v_new ---
        p_qg = tl.make_block_ptr(
            qg + k_base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        # dot with low precision inputs, FP32 accumulator
        b_o = tl.dot(b_qg, b_h.to(b_qg.dtype), out_dtype=tl.float32)

        p_Aqk = tl.make_block_ptr(
            Aqk + A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        p_Aqb = tl.make_block_ptr(
            Aqb + A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        p_v = tl.make_block_ptr(
            v + v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )

        b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
        b_Aqb = tl.load(p_Aqb, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # dot with low precision, FP32 accumulator
        b_o += tl.dot(b_Aqk, b_v, out_dtype=tl.float32)
        b_o += tl.dot(b_Aqb, b_v_new_fp32.to(b_Aqb.dtype), out_dtype=tl.float32)
        p_o = tl.make_block_ptr(
            o + v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # --- state update: h = h * decay + kg^T @ v + bg^T @ v_new ---
        o_k = tl.arange(0, BK)
        g_last_idx = ((chunk_offset_base + i_t) * H + i_h) * K + o_k
        b_g_last = tl.load(g_last + g_last_idx, mask=(o_k < K), other=0)
        # FP32 state computation
        b_h = b_h * exp2(b_g_last.to(tl.float32))[:, None]

        p_kg = tl.make_block_ptr(
            kg + k_base, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1)
        )
        p_bg = tl.make_block_ptr(
            bg + k_base, (K, T), (1, H * K), (0, i_t * BT), (BK, BT), (0, 1)
        )

        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        b_bg = tl.load(p_bg, boundary_check=(0, 1))

        # dot with low precision, FP32 accumulator
        b_h += tl.dot(b_kg, b_v, out_dtype=tl.float32)
        b_h += tl.dot(b_bg, b_v_new_fp32.to(b_bg.dtype), out_dtype=tl.float32)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Python entry point
# =============================================================================


def chunk_dplr_verifier(
    q: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 16,
) -> tuple[bool, str]:
    """Check if inputs meet the requirements for fused DPLR kernel.

    Args:
        q: Query tensor
        v: Value tensor
        chunk_size: Chunk size

    Returns:
        (can_use, reason): Whether fused kernel can be used and reason if not
    """
    K, V = q.shape[-1], v.shape[-1]

    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0

    if chunk_size != 16:
        return False, f"chunk_size must be 16, got {chunk_size}"

    if not is_power_of_2(K) or K < 32:
        return False, f"K must be power of 2 and >= 32, got K={K}"

    if not is_power_of_2(V) or V < 32:
        return False, f"V must be power of 2 and >= 32, got V={V}"

    if q.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype must be bfloat16 or float16, got {q.dtype}"

    return True, ""


def chunk_dplr_fwd_k1_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    chunk_size: int = 16,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    """Fused K1: cumsum + intra attention + WY inverse + w/u/qg/kg/bg."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    RCP_LN2_VAL = 1.4426950408889634

    if chunk_indices is None:
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
        )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    chunk_offsets = None
    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    # Storage dtype matches input dtype (FP16/BF16), following CUDA reference implementation
    storage_dtype = q.dtype

    # Allocate output buffers - all use storage_dtype for memory efficiency
    Aqk = q.new_empty(B, T, H, BT, dtype=storage_dtype)
    Aqb = q.new_empty(B, T, H, BT, dtype=storage_dtype)
    Aab_inv = q.new_empty(B, T, H, BT, dtype=storage_dtype)
    qg_out = torch.empty_like(q, dtype=storage_dtype)
    kg_out = torch.empty_like(k, dtype=storage_dtype)
    bg_out = torch.empty_like(b, dtype=storage_dtype)
    w_out = torch.empty_like(a, dtype=storage_dtype)
    u_out = torch.empty_like(v, dtype=storage_dtype)
    # g_last also uses storage_dtype (FP16/BF16 sufficient for chunk-level gates)
    g_last_out = torch.empty(
        B * NT * H if cu_seqlens is None else NT * H,
        K,
        device=q.device,
        dtype=storage_dtype,
    )

    # Use original fused kernel with Neumann series
    grid = (NT, B * H)
    chunk_dplr_fwd_kernel_k1_fused[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        v=v,
        gk=gk,
        Aqk=Aqk,
        Aqb=Aqb,
        Aab_inv=Aab_inv,
        qg_out=qg_out,
        kg_out=kg_out,
        bg_out=bg_out,
        w_out=w_out,
        u_out=u_out,
        g_last_out=g_last_out,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        RCP_LN2=RCP_LN2_VAL,
    )

    return Aqk, Aqb, Aab_inv, qg_out, kg_out, bg_out, w_out, u_out, g_last_out


def chunk_dplr_fwd_k2_fused(
    kg: torch.Tensor,
    bg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    qg: torch.Tensor,
    Aqk: torch.Tensor,
    Aqb: torch.Tensor,
    g_last: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 16,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Fused K2: state propagation + output computation."""
    B, T, H, K = kg.shape
    V = v.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    o = torch.empty_like(v)
    final_state = (
        kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * H)

    chunk_offsets = None
    if cu_seqlens is not None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    chunk_dplr_fwd_kernel_k2_fused[grid](
        kg=kg,
        bg=bg,
        w=w,
        u=u,
        v=v,
        qg=qg,
        Aqk=Aqk,
        Aqb=Aqb,
        g_last=g_last,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return o, final_state


def chunk_dplr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
):
    """Fused DPLR (Delta rule with Per-head Low-rank) attention forward pass.

    Implements the DPLR recurrence using two fused Triton kernels (K1 + K2) for
    efficient GPU computation. K1 performs intra-chunk attention with WY inverse,
    and K2 performs inter-chunk recurrence.

    Fused kernel constraints (raises NotImplementedError if not met):
    - chunk_size must be exactly 16 (WY iteration hardcoded for 4 iterations)
    - K and V must be powers of 2 and >= 32 (Triton tl.zeros requirement)
    - dtype must be bfloat16 or float16
    - varlen mode requires cu_seqlens to be provided

    Args:
        q: Query tensor, shape [B, T, H, K] where B=batch, T=seq_len,
           H=num_heads, K=head_dim
        k: Key tensor, shape [B, T, H, K]
        v: Value tensor, shape [B, T, H, V] where V=value_dim
        a: Alpha coefficients for low-rank updates, shape [B, T, H, K]
        b: Beta coefficients for low-rank updates, shape [B, T, H, K]
        gk: Gate values for exponential decay, shape [B, T, H, K]
        scale: Scaling factor
        initial_state: Initial hidden state [B, H, K, V] (optional)
        output_final_state: Whether to return final state
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        chunk_size: Chunk size (default: 16)

    Returns:
        output: Output tensor [B, T, H, V]
        final_state: Final hidden state [B, H, K, V] (if output_final_state=True)
    """
    # Check if we can use the fused kernel
    can_use_fused, reason = chunk_dplr_verifier(q, v, chunk_size)

    if not can_use_fused:
        # TODO: Fallback to naive implementation
        # This will be implemented when naive version is added
        raise NotImplementedError(
            f"Fused kernel requirements not met: {reason}. "
            f"Naive implementation fallback not yet implemented."
        )

    # Use fused Triton kernel implementation
    Aqk, Aqb, Aab_inv, qg, kg, bg, w, u, g_last = chunk_dplr_fwd_k1_fused(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    o, final_state = chunk_dplr_fwd_k2_fused(
        kg=kg,
        bg=bg,
        w=w,
        u=u,
        v=v,
        qg=qg,
        Aqk=Aqk,
        Aqb=Aqb,
        g_last=g_last,
        scale=1.0,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
