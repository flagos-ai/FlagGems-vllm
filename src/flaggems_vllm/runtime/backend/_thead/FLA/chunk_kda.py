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

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets

__all__ = ["chunk_kda"]

RCP_LN2 = 1.4426950216
KFIRST_INTERNAL = True
GLAST_PREFETCH = True


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "USE_GATE_IN_KERNEL": lambda args: args["A_log"] is not None,
        "USE_QK_L2NORM": lambda args: args["use_qk_l2norm"],
        "APPLY_BETA_SIGMOID": lambda args: args["apply_beta_sigmoid"],
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": 32, "BV": 64}, num_warps=1, num_stages=1),
        triton.Config({"BK": 64, "BV": 64}, num_warps=2, num_stages=1),
    ],
    key=["H", "HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _thead_kda_fwd_intra_kernel(
    q,
    k,
    v,
    g,
    beta,
    w,
    u,
    qg,
    kg,
    Aqk,
    g_last,
    A_log,
    dt_bias,
    lower_bound,
    scale,
    g_scale,
    l2norm_eps,
    use_qk_l2norm,
    apply_beta_sigmoid,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
):
    i_tg, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_tg * 2).to(tl.int32), tl.load(
            chunk_indices + i_tg * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        i_gc = i_tg  # g_last row = position in chunk_indices
    else:
        i_t = i_tg
        bos, eos = i_b * T, i_b * T + T
        i_gc = i_b * tl.num_programs(0) + i_t  # g_last row = b * NT + t

    if i_t * BT >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    Aqk += (bos * HV + i_hv) * BT
    w += (bos * HV + i_hv) * K
    u += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    g_last += (i_gc * HV + i_hv) * K
    if STORE_QG:
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_i = tl.arange(0, BT)
    o_c = i_t * BT + o_i
    m_c = o_c < T
    r_last = tl.minimum(BT, T - i_t * BT) - 1

    # Phase 0: L2 norm on q/k (optional) + beta sigmoid (optional)
    if USE_QK_L2NORM:
        b_q_ss = tl.zeros([BT], dtype=tl.float32)
        b_k_ss = tl.zeros([BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_k = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_q_ss += tl.sum(b_q * b_q, 1)
            b_k_ss += tl.sum(b_k * b_k, 1)

        b_q_rstd = 1.0 / tl.sqrt(b_q_ss + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(b_k_ss + l2norm_eps)

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    if APPLY_BETA_SIGMOID:
        b_beta = tl.sigmoid(b_beta)

    # Phase 1: cumsum(g) + intra-chunk Aqk/Akk
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Akk = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_GATE_IN_KERNEL:
        b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * b_q_rstd[:, None]
            b_k = b_k * b_k_rstd[:, None]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                p_dt = tl.make_block_ptr(
                    dt_bias + i_hv * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
                )
                b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
                b_g = b_g + b_bias[None, :]
            if USE_LOWER_BOUND:
                b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
            else:
                b_g = -b_A * _softplus(b_g) * g_scale
        else:
            b_g = b_g * g_scale
        b_g = tl.cumsum(b_g, axis=0)

        o_k = i_k * BK + tl.arange(0, BK)
        b_gn = tl.sum(tl.where(o_i[:, None] == r_last, b_g, 0.0), 0)
        tl.store(g_last + o_k, b_gn, mask=o_k < K)

        b_gq = tl.where(m_c[:, None], exp2(b_g), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_g), 0.0)

        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_k * b_gq, b_kgt)

    # Causal mask
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)

    # Phase 3: w, u, qg, kg -- the gated cumsum is RECOMPUTED from raw g in
    # the exact same op order as Phase 1, replacing the g_out readback
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Ai.to(b_vb.dtype), b_vb)
        tl.store(p_u, b_u.to(u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        b_gk = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                p_dt = tl.make_block_ptr(
                    dt_bias + i_hv * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
                )
                b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
                b_gk = b_gk + b_bias[None, :]
            if USE_LOWER_BOUND:
                b_gk = (lower_bound * g_scale) * tl.sigmoid(b_A * b_gk)
            else:
                b_gk = -b_A * _softplus(b_gk) * g_scale
        else:
            b_gk = b_gk * g_scale
        b_gk = tl.cumsum(b_gk, axis=0)
        b_kb = b_k * b_beta[:, None] * exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_qg_out = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * b_q_rstd[:, None]
            b_qg_val = b_q * exp2(b_gk)
            tl.store(p_qg_out, b_qg_val.to(qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            # last cumsum row from the recomputed tile (same value as g_last)
            b_gn = tl.sum(tl.where(o_i[:, None] == r_last, b_gk, 0.0), 0)
            b_kg_val = b_k * tl.where(m_c[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg_out = tl.make_block_ptr(
                kg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_kg_out, b_kg_val.to(kg.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_w = tl.dot(b_Ai.to(b_kb.to(b_k.dtype).dtype), b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(w.dtype.element_ty), boundary_check=(0, 1))


def _thead_kda_fwd_intra(
    q,
    k,
    v,
    g,
    beta,
    scale,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=16,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm=True,
    apply_beta_sigmoid=True,
):
    """Fused intra-chunk computation. Returns (w, u, qg, kg, Aqk, g_last)."""
    B, T_len, H, K = q.shape
    HV = g.shape[2]
    V = v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HV)

    g_last = torch.empty(B * NT, HV, K, device=q.device, dtype=torch.float32)
    w = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    u = torch.empty(B, T_len, HV, V, device=q.device, dtype=q.dtype)
    qg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    kg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    Aqk = torch.empty(B, T_len, HV, BT, device=q.device, dtype=q.dtype)

    _thead_kda_fwd_intra_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        Aqk=Aqk,
        g_last=g_last,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        scale=scale,
        g_scale=RCP_LN2,
        l2norm_eps=1e-6,
        use_qk_l2norm=use_qk_l2norm,
        apply_beta_sigmoid=apply_beta_sigmoid,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
    )
    return w, u, qg, kg, Aqk, g_last


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": 32}, num_warps=2, num_stages=3),
        triton.Config({"BV": 64}, num_warps=8, num_stages=2),
    ],
    key=["HV", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _thead_kda_fwd_h_o_kernel(
    kg,
    w,
    u,
    gk,
    qg,
    Aqk,
    o,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    H_VFIRST: tl.constexpr,
    GLAST_PREFETCH: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        i_cb = tl.load(chunk_offsets + i_n)  # this sequence's first g_last row
    else:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        i_cb = i_n * NT

    kg += (bos * HV + i_h).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    u += (bos * HV + i_h).to(tl.int64) * V
    gk += i_h.to(tl.int64) * K  # g_last [n_chunks, HV, K]: row = i_cb + i_t
    qg += (bos * HV + i_h).to(tl.int64) * K
    Aqk += (bos * HV + i_h).to(tl.int64) * BT
    o += (bos * HV + i_h).to(tl.int64) * V

    o_t0 = tl.arange(0, BT)
    o_k1 = tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)

    if H_VFIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        b_h0_1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if STATE_V_FIRST == H_VFIRST:
            b_h1 += b_h0_1
        else:
            b_h1 += tl.trans(b_h0_1)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            b_h0_2 = tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
            if STATE_V_FIRST == H_VFIRST:
                b_h2 += b_h0_2
            else:
                b_h2 += tl.trans(b_h0_2)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            b_h0_3 = tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
            if STATE_V_FIRST == H_VFIRST:
                b_h3 += b_h0_3
            else:
                b_h3 += tl.trans(b_h0_3)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            b_h0_4 = tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
            if STATE_V_FIRST == H_VFIRST:
                b_h4 += b_h0_4
            else:
                b_h4 += tl.trans(b_h0_4)

    if GLAST_PREFETCH:
        b_gn_next1 = tl.load(
            gk + i_cb * (HV * K) + o_k1, mask=(o_k1 < K) & (NT > 0), other=0.0
        ).to(tl.float32)
        if K > 64:
            o_k2 = 64 + o_k1
            b_gn_next2 = tl.load(
                gk + i_cb * (HV * K) + o_k2, mask=(o_k2 < K) & (NT > 0), other=0.0
            ).to(tl.float32)
        if K > 128:
            o_k3 = 128 + o_k1
            b_gn_next3 = tl.load(
                gk + i_cb * (HV * K) + o_k3, mask=(o_k3 < K) & (NT > 0), other=0.0
            ).to(tl.float32)
        if K > 192:
            o_k4 = 192 + o_k1
            b_gn_next4 = tl.load(
                gk + i_cb * (HV * K) + o_k4, mask=(o_k4 < K) & (NT > 0), other=0.0
            ).to(tl.float32)

    for i_t in range(NT):
        o_t = i_t * BT + o_t0
        m_t = o_t < T
        i_gc = i_cb + i_t

        if GLAST_PREFETCH:
            b_gk_last1 = b_gn_next1
            i_gn = i_gc + 1
            m_gn = (i_t + 1) < NT
            b_gn_next1 = tl.load(
                gk + i_gn * (HV * K) + o_k1, mask=(o_k1 < K) & m_gn, other=0.0
            ).to(tl.float32)
            if K > 64:
                b_gk_last2 = b_gn_next2
                b_gn_next2 = tl.load(
                    gk + i_gn * (HV * K) + o_k2, mask=(o_k2 < K) & m_gn, other=0.0
                ).to(tl.float32)
            if K > 128:
                b_gk_last3 = b_gn_next3
                b_gn_next3 = tl.load(
                    gk + i_gn * (HV * K) + o_k3, mask=(o_k3 < K) & m_gn, other=0.0
                ).to(tl.float32)
            if K > 192:
                b_gk_last4 = b_gn_next4
                b_gn_next4 = tl.load(
                    gk + i_gn * (HV * K) + o_k4, mask=(o_k4 < K) & m_gn, other=0.0
                ).to(tl.float32)

        # v_new = u - w @ h
        b_w = tl.load(
            w + o_t[:, None] * (HV * K) + o_k1[None, :], mask=m_t[:, None], other=0.0
        )
        if H_VFIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            b_w = tl.load(
                w + o_t[:, None] * (HV * K) + (64 + o_k1)[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            if H_VFIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            b_w = tl.load(
                w + o_t[:, None] * (HV * K) + (128 + o_k1)[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            if H_VFIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            b_w = tl.load(
                w + o_t[:, None] * (HV * K) + (192 + o_k1)[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            if H_VFIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v = (
            tl.load(
                u + o_t[:, None] * (HV * V) + o_v[None, :], mask=m_t[:, None], other=0.0
            )
            - b_v
        )

        # output = scale * qg @ h + Aqk @ v_new
        b_qg = tl.load(
            qg + o_t[:, None] * (HV * K) + o_k1[None, :], mask=m_t[:, None], other=0.0
        )
        if H_VFIRST:
            b_o = tl.dot(b_qg, tl.trans(b_h1).to(b_qg.dtype))
        else:
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype))
        if K > 64:
            b_qg = tl.load(
                qg + o_t[:, None] * (HV * K) + (64 + o_k1)[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            if H_VFIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h2).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h2.to(b_qg.dtype))
        if K > 128:
            b_qg = tl.load(
                qg + o_t[:, None] * (HV * K) + (128 + o_k1)[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            if H_VFIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h3).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h3.to(b_qg.dtype))
        if K > 192:
            b_qg = tl.load(
                qg + o_t[:, None] * (HV * K) + (192 + o_k1)[None, :],
                mask=m_t[:, None],
                other=0.0,
            )
            if H_VFIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h4).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h4.to(b_qg.dtype))
        b_o *= scale

        b_Aqk = tl.load(
            Aqk + o_t[:, None] * (HV * BT) + o_t0[None, :], mask=m_t[:, None], other=0.0
        )
        # b_o += tl.dot(b_Aqk.to(b_v.dtype), b_v)
        b_v = b_v.to(Aqk.dtype.element_ty)
        b_o += tl.dot(b_Aqk, b_v)

        tl.store(
            o + o_t[:, None] * (HV * V) + o_v[None, :],
            b_o.to(o.dtype.element_ty),
            mask=m_t[:, None],
        )

        if not GLAST_PREFETCH:
            b_gk_last1 = tl.load(
                gk + i_gc * (HV * K) + o_k1, mask=(o_k1 < K), other=0.0
            ).to(tl.float32)
        if H_VFIRST:
            b_h1 *= exp2(b_gk_last1)[None, :]
        else:
            b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            if not GLAST_PREFETCH:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + i_gc * (HV * K) + o_k2, mask=(o_k2 < K), other=0.0
                ).to(tl.float32)
            if H_VFIRST:
                b_h2 *= exp2(b_gk_last2)[None, :]
            else:
                b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            if not GLAST_PREFETCH:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + i_gc * (HV * K) + o_k3, mask=(o_k3 < K), other=0.0
                ).to(tl.float32)
            if H_VFIRST:
                b_h3 *= exp2(b_gk_last3)[None, :]
            else:
                b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            if not GLAST_PREFETCH:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + i_gc * (HV * K) + o_k4, mask=(o_k4 < K), other=0.0
                ).to(tl.float32)
            if H_VFIRST:
                b_h4 *= exp2(b_gk_last4)[None, :]
            else:
                b_h4 *= exp2(b_gk_last4)[:, None]

        # state update: h += kg^T @ v_new
        b_v = b_v.to(kg.dtype.element_ty)
        b_kg = tl.load(
            kg + o_k1[:, None] + o_t[None, :] * (HV * K), mask=m_t[None, :], other=0.0
        )
        if H_VFIRST:
            b_h1 += tl.trans(tl.dot(b_kg, b_v))
        else:
            b_h1 += tl.dot(b_kg, b_v)
        if K > 64:
            b_kg = tl.load(
                kg + (64 + o_k1)[:, None] + o_t[None, :] * (HV * K),
                mask=m_t[None, :],
                other=0.0,
            )
            if H_VFIRST:
                b_h2 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h2 += tl.dot(b_kg, b_v)
        if K > 128:
            b_kg = tl.load(
                kg + (128 + o_k1)[:, None] + o_t[None, :] * (HV * K),
                mask=m_t[None, :],
                other=0.0,
            )
            if H_VFIRST:
                b_h3 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h3 += tl.dot(b_kg, b_v)
        if K > 192:
            b_kg = tl.load(
                kg + (192 + o_k1)[:, None] + o_t[None, :] * (HV * K),
                mask=m_t[None, :],
                other=0.0,
            )
            if H_VFIRST:
                b_h4 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h4 += tl.dot(b_kg, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        if STATE_V_FIRST == H_VFIRST:
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        else:
            tl.store(
                p_ht, tl.trans(b_h1).to(p_ht.dtype.element_ty), boundary_check=(0, 1)
            )
        if K > 64:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            if STATE_V_FIRST == H_VFIRST:
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            else:
                tl.store(
                    p_ht,
                    tl.trans(b_h2).to(p_ht.dtype.element_ty),
                    boundary_check=(0, 1),
                )
        if K > 128:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            if STATE_V_FIRST == H_VFIRST:
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            else:
                tl.store(
                    p_ht,
                    tl.trans(b_h3).to(p_ht.dtype.element_ty),
                    boundary_check=(0, 1),
                )
        if K > 192:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            if STATE_V_FIRST == H_VFIRST:
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            else:
                tl.store(
                    p_ht,
                    tl.trans(b_h4).to(p_ht.dtype.element_ty),
                    boundary_check=(0, 1),
                )


def _thead_kda_fwd_h_o(
    kg,
    w,
    u,
    gk,
    qg,
    Aqk,
    scale,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=16,
):
    """Fused state propagation + output. gk is g_last [n_chunks, HV, K]."""
    B, T, HV, K = kg.shape
    V = u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if cu_seqlens is None:
        N = B
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    final_state = None
    if output_final_state:
        if state_v_first:
            final_state = kg.new_empty(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = kg.new_empty(N, HV, K, V, dtype=torch.float32)

    o = torch.empty(B, T, HV, V, device=kg.device, dtype=u.dtype)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    _thead_kda_fwd_h_o_kernel[grid](
        kg=kg,
        w=w,
        u=u,
        gk=gk,
        qg=qg,
        Aqk=Aqk,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        H_VFIRST=state_v_first and not KFIRST_INTERNAL,
        GLAST_PREFETCH=GLAST_PREFETCH,
    )
    return o, final_state


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    BT = chunk_size

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    w, u, qg, kg, Aqk, g_last = _thead_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        lower_bound=lower_bound,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        apply_beta_sigmoid=use_beta_sigmoid_in_kernel,
    )

    return _thead_kda_fwd_h_o(
        kg=kg,
        w=w,
        u=u,
        gk=g_last,
        qg=qg,
        Aqk=Aqk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
    )
