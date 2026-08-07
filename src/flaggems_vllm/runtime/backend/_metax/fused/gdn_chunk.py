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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
from flaggems_vllm.ops.FLA.triton_ops_helper import exp
from flaggems_vllm.ops.FLA.utils import SUPPRESS_LEVEL
from flaggems_vllm.ops.FLA.wy_fast import recompute_w_u_fwd
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

FLA_CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# 1. Fused cumsum + scaled_dot_kkt + solve_tril
# ---------------------------------------------------------------------------
def _prune_unsafe_kloop_configs(configs, named_args, *args, **kwargs):
    K = named_args.get("K", kwargs.get("K"))
    if K is None:
        return configs
    pruned = [c for c in configs if not (c.num_warps > 2 and c.kwargs.get("BK", K) < K)]
    return pruned or configs[:1]


@libentry()
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [1]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": _prune_unsafe_kloop_configs},
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel(
    g_in,
    g_out,
    k,
    beta,
    A,
    A_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # ---------- cumsum ----------
    p_g_in = tl.make_block_ptr(
        g_in + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g_out = tl.make_block_ptr(
        g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g_in, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    tl.store(p_g_out, b_g.to(p_g_out.dtype.element_ty), boundary_check=(0,))

    # ---------- KKT (write L to A) ----------
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))
    b_g_diff = b_g[:, None] - b_g[None, :]
    b_A = b_A * exp(b_g_diff)
    m_A_kkt = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A_kkt, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()

    # ---------- solve_tril (read A, write A_inv), 16x16 blocked ----------
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_base = A + (bos * H + i_h) * BT
    A_inv_base = A_inv + (bos * H + i_h) * BT

    p_A_11 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_A_22 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_A_33 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_A_44 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I

    p_A_21 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)

    DOT_PRECISION: tl.constexpr = "ieee"
    b_Ai_21 = -tl.dot(
        tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION),
        b_Ai_11,
        input_precision=DOT_PRECISION,
    )
    b_Ai_32 = -tl.dot(
        tl.dot(b_Ai_33, b_A_32, input_precision=DOT_PRECISION),
        b_Ai_22,
        input_precision=DOT_PRECISION,
    )
    b_Ai_43 = -tl.dot(
        tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION),
        b_Ai_33,
        input_precision=DOT_PRECISION,
    )
    b_Ai_31 = -tl.dot(
        b_Ai_33,
        tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_42 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b_Ai_41 = -tl.dot(
        b_Ai_44,
        tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION)
        + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        + tl.dot(b_A_43, b_Ai_31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    p_Ai_11 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    tl.store(
        p_Ai_11,
        b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_33,
        b_Ai_33.to(p_Ai_33.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_44,
        b_Ai_44.to(p_Ai_44.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_31,
        b_Ai_31.to(p_Ai_31.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_32,
        b_Ai_32.to(p_Ai_32.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_41,
        b_Ai_41.to(p_Ai_41.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_42,
        b_Ai_42.to(p_Ai_42.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_43,
        b_Ai_43.to(p_Ai_43.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cumsum(g) + KKT(L) + solve_tril(L -> inv). Returns (g_cumsum, A_inv).

    Only BT == 64 is supported (solve_tril is hard-coded as 4x4 blocks of 16).
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    assert BT == 64, "fused cumsum_kkt_solve_tril only supports chunk size 64."
    output_dtype = output_dtype or k.dtype
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_out = torch.empty_like(g)
    A = torch.empty(B, T, H, BT, device=g.device, dtype=torch.float32)
    A_inv = torch.zeros(B, T, H, BT, device=g.device, dtype=output_dtype)

    def grid(meta):
        return (NT, H if cu_seqlens is not None else B * H)

    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel[grid](
        g_in=g,
        g_out=g_out,
        k=k,
        beta=beta,
        A=A,
        A_inv=A_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
    )
    return g_out, A_inv


# ---------------------------------------------------------------------------
# 2. chunk_delta_h with K-major state layout (no tl.trans in the recurrence)
# ---------------------------------------------------------------------------
@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [1]
        for BV in [32, 64]
    ],
    key=[
        "H",
        "K",
        "V",
        "BT",
        "IS_VARLEN",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
    ],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_kmajor(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (K, V),
                (V, 1),
                (64, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (K, V),
                (V, 1),
                (192, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_kmajor[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state


# ---------------------------------------------------------------------------
# 3. chunk_o consuming the (K, V) h layout
# ---------------------------------------------------------------------------
@libentry()
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK, BV in [(32, 32), (32, 64), (64, 32), (64, 64)]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        # h is stored as (K, V)
        p_h = tl.make_block_ptr(
            h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
        )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))

    # scale is applied only once at the end
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    def grid(meta):
        return (
            triton.cdiv(V, meta["BV"]),
            NT,
            H if cu_seqlens is not None else B * H,
        )

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o


# ---------------------------------------------------------------------------
# 4. Patched forward: fused cumsum+kkt+solve_tril -> recompute_w_u ->
#    K-major chunk_delta_h -> chunk_o
# ---------------------------------------------------------------------------
def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    **kwargs,
):
    logger.debug("METAX GDN CHUNK FWD (K-major, fused cumsum+kkt+solve_tril)")
    # Ensure contiguity
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if initial_state is not None and not initial_state.is_contiguous():
        initial_state = initial_state.contiguous()
    if cu_seqlens is not None and not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=FLA_CHUNK_SIZE,
        output_dtype=k.dtype,
    )
    # obtain WY representation. u is actually the new v.
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new
