# Copyright (c) 2025 FlagGems. All rights reserved.
# Fused cumsum + KKT + solve_tril for chunk_gated_delta_rule. Returns g_out, A_inv; w_u is separate.
# License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
from flaggems_vllm.ops.FLA.solve_tril import FLA_TRIL_PRECISION
from flaggems_vllm.ops.FLA.triton_ops_helper import exp
from flaggems_vllm.ops.FLA.utils import is_tma_supported
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args.get("USE_G", True),
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
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
    USE_G: tl.constexpr,
    USE_TMA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
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
    if USE_G:
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

    # ---------- solve_tril (read A, write A_inv) ----------
    # GCU: the 16x16 block-merge formulation chains [16,16] fp32 ieee tl.dot
    # ops that fail gcu.matmul attribute conversion on every fresh compile
    # (varlen specialization, cleared caches, the standalone solve_tril
    # entry); only stale cached binaries of the non-varlen form run. Use the
    # plain row-recurrence forward substitution instead: no tl.dot, verified
    # against solve_triangular on GCU (maxdiff <= 7.3e-4), valid for BT in
    # {16, 32, 64}.
    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_base = A + (bos * H + i_h) * BT
    A_inv_base = A_inv + (bos * H + i_h) * BT
    p_L = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_Ai = tl.load(p_L, boundary_check=(0, 1))
    b_Ai = -tl.where(m_A, b_Ai, 0)
    for i in range(2, min(BT, T - i_t * BT)):
        b_a = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i)
        b_a += tl.sum(b_a[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
    b_Ai += m_I
    p_Ai = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(
        p_Ai,
        b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = None,
    *,
    use_g_in_kkt: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused kernel: cumsum(g) + KKT(L) + solve_tril(L -> inv). Returns (g_out, A_inv).
    w_u stays a separate kernel (e.g. recompute_w_u_fwd) for HGMMA."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    output_dtype = output_dtype or k.dtype
    # GCU: Triton reads int64 buffers with garbage values; normalize to int32
    # (sequence offsets are < 2^31 here).
    if cu_seqlens is not None and cu_seqlens.dtype == torch.int64:
        cu_seqlens = cu_seqlens.to(torch.int32)
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    if chunk_indices is not None and chunk_indices.dtype == torch.int64:
        chunk_indices = chunk_indices.to(torch.int32)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_out = torch.empty_like(g)
    A = torch.empty(B, T, H, BT, device=g.device, dtype=torch.float32)
    A_inv = torch.zeros(B, T, H, BT, device=g.device, dtype=output_dtype)

    def grid(meta):
        # Varlen kernels derive the sequence from chunk_indices. The batch id
        # from i_bh is unused there, so H programs are sufficient for B > 1 too.
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
        IS_VARLEN=cu_seqlens is not None,
        USE_G=use_g_in_kkt,
        USE_TMA=is_tma_supported and BT == 64,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return g_out, A_inv
