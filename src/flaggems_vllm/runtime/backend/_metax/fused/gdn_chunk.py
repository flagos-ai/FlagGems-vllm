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
import triton.backends.metax.compiler as backend

_original_compile = backend.metax.translate_llvmir_to_mcfatbin


# Disable backend LICM only for the GDN H-state kernel.
def _compile_without_licm(src, mxcc, maca, options):
    name = backend.maca_get_kernel_name(src)
    if name == "chunk_gated_delta_rule_fwd_kernel_h_vmajor":
        options += " -mllvm -metaxgpu-disable-licm=true"
    return _original_compile(src, mxcc, maca, options)


backend.metax.translate_llvmir_to_mcfatbin = _compile_without_licm

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
from flaggems_vllm.ops.FLA.triton_ops_helper import exp
from flaggems_vllm.ops.FLA.utils import SUPPRESS_LEVEL, tensor_cache
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


@triton.jit
def _inv_unit_lower_16(D, m_strict, I16, DP: tl.constexpr):
    """Invert a 16x16 unit-lower-triangular matrix in registers."""
    Q = -tl.where(m_strict, D, 0.0)
    S = I16 + Q
    Q = tl.dot(Q, Q, input_precision=DP)
    S = S + tl.dot(S, Q, input_precision=DP)
    Q = tl.dot(Q, Q, input_precision=DP)
    S = S + tl.dot(S, Q, input_precision=DP)
    Q = tl.dot(Q, Q, input_precision=DP)
    S = S + tl.dot(S, Q, input_precision=DP)
    return S


@libentry()
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [1, 2, 3]
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

    DOT_PRECISION: tl.constexpr = "ieee"

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

    g_t = tl.trans(tl.reshape(b_g, (4, 16)))
    g_x, g_y = tl.split(tl.reshape(g_t, (16, 2, 2)))
    b_g0, b_g2 = tl.split(g_x)
    b_g1, b_g3 = tl.split(g_y)

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    bt_t = tl.trans(tl.reshape(b_beta, (4, 16)))
    bt_x, bt_y = tl.split(tl.reshape(bt_t, (16, 2, 2)))
    b_beta0, b_beta2 = tl.split(bt_x)
    b_beta1, b_beta3 = tl.split(bt_y)

    # ---------- KKT ----------
    b_A_11 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_22 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_33 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_44 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_21 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_31 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_32 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_41 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_42 = tl.zeros([16, 16], dtype=tl.float32)
    b_A_43 = tl.zeros([16, 16], dtype=tl.float32)
    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    for i_k in range(tl.cdiv(K, BK)):
        p_k0 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0)
        )
        p_k1 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        p_k2 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        p_k3 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )

        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))
        b_kb0 = b_k0 * b_beta0[:, None].to(b_k0.dtype)
        b_kb1 = b_k1 * b_beta1[:, None].to(b_k1.dtype)
        b_kb2 = b_k2 * b_beta2[:, None].to(b_k2.dtype)
        b_kb3 = b_k3 * b_beta3[:, None].to(b_k3.dtype)
        b_k0t = tl.trans(b_k0)
        b_k1t = tl.trans(b_k1)
        b_k2t = tl.trans(b_k2)
        b_k3t = tl.trans(b_k3)
        b_A_11 += tl.dot(b_kb0, b_k0t)
        b_A_22 += tl.dot(b_kb1, b_k1t)
        b_A_33 += tl.dot(b_kb2, b_k2t)
        b_A_44 += tl.dot(b_kb3, b_k3t)
        b_A_21 += tl.dot(b_kb1, b_k0t)
        b_A_31 += tl.dot(b_kb2, b_k0t)
        b_A_32 += tl.dot(b_kb2, b_k1t)
        b_A_41 += tl.dot(b_kb3, b_k0t)
        b_A_42 += tl.dot(b_kb3, b_k1t)
        b_A_43 += tl.dot(b_kb3, b_k2t)

    b_A_11 = b_A_11 * exp(b_g0[:, None] - b_g0[None, :])
    b_A_22 = b_A_22 * exp(b_g1[:, None] - b_g1[None, :])
    b_A_33 = b_A_33 * exp(b_g2[:, None] - b_g2[None, :])
    b_A_44 = b_A_44 * exp(b_g3[:, None] - b_g3[None, :])
    b_A_21 = b_A_21 * exp(b_g1[:, None] - b_g0[None, :])
    b_A_31 = b_A_31 * exp(b_g2[:, None] - b_g0[None, :])
    b_A_32 = b_A_32 * exp(b_g2[:, None] - b_g1[None, :])
    b_A_41 = b_A_41 * exp(b_g3[:, None] - b_g0[None, :])
    b_A_42 = b_A_42 * exp(b_g3[:, None] - b_g1[None, :])
    b_A_43 = b_A_43 * exp(b_g3[:, None] - b_g2[None, :])

    # ---------- solve_tril ----------
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    I16 = (o_i[:, None] == o_i[None, :]).to(tl.float32)
    A_inv_base = A_inv + (bos * H + i_h) * BT

    b_Ai_11 = _inv_unit_lower_16(b_A_11, m_A, I16, DOT_PRECISION)
    b_Ai_22 = _inv_unit_lower_16(b_A_22, m_A, I16, DOT_PRECISION)
    b_Ai_33 = _inv_unit_lower_16(b_A_33, m_A, I16, DOT_PRECISION)
    b_Ai_44 = _inv_unit_lower_16(b_A_44, m_A, I16, DOT_PRECISION)

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
    """Fused cumsum(g) + KKT(L) + solve_tril(L -> inv).

    Returns (g_cumsum, A_inv). Only BT=64 is supported.
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    assert BT == 64, "fused cumsum_kkt_solve_tril only supports chunk size 64."
    output_dtype = output_dtype or k.dtype
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Gate cumsum must remain fp32; bf16 error is amplified by the later exp.
    g_out = torch.empty_like(g, dtype=torch.float32)
    A_inv = torch.zeros(B, T, H, BT, device=g.device, dtype=output_dtype)

    def grid(meta):
        return (NT, H if cu_seqlens is not None else B * H)

    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel[grid](
        g_in=g,
        g_out=g_out,
        k=k,
        beta=beta,
        A_inv=A_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT
    )
    return g_out, A_inv

def _n_bucket(N: int) -> int:
    """Map request count to autotune tiers: 1, 2-8, 9-24, and 25+."""
    if N <= 1:
        return 0
    if N <= 8:
        return 1
    if N <= 24:
        return 2
    return 3


@tensor_cache
def _skew_bucket_cached(chunk_offsets: torch.Tensor) -> int:
    """Map max/average chunk-count skew to an identity-cached autotune tier."""
    N = chunk_offsets.numel() - 1
    if N <= 1:
        return 0
    nt = chunk_offsets[1:] - chunk_offsets[:-1]
    stats = torch.stack([nt.max(), nt.sum()])
    max_nt, sum_nt = stats.tolist()  # One sync per chunk_offsets identity.
    if sum_nt <= 0:
        return 0
    avg_nt = sum_nt / N
    skew = max_nt / avg_nt if avg_nt > 0 else 1.0
    if skew <= 1.15:
        return 0
    elif skew <= 1.5:
        return 1
    elif skew <= 2.5:
        return 2
    else:
        return 3


def _skew_bucket(chunk_offsets: torch.Tensor | None, N: int) -> int:
    """Return the fwd_h tail-skew tier; uniform and single-request inputs use 0."""
    if chunk_offsets is None or N <= 1:
        return 0
    return _skew_bucket_cached(chunk_offsets)

# ---------------------------------------------------------------------------
# 2. chunk_delta_h with V-major state layout
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
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64, 128]
        for BV in [16, 32, 64]
        for num_warps in [2, 4]
        # num_stages=2 triggers a MetaX pipeliner map::at failure.
        for num_stages in [3]
    ],
    key=[
        "H",
        "K",
        "V",
        "BT",
        "IS_VARLEN",
        "USE_INITIAL_STATE",
        "STORE_FINAL_STATE",
        "N_BUCKET",
        "SKEW_BUCKET",
    ],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_vmajor(
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
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    N_BUCKET: tl.constexpr,
    SKEW_BUCKET: tl.constexpr,
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

    NK: tl.constexpr = tl.cdiv(K, BK)

    # [BV, BK] V-major accumulators.
    b_h1 = tl.zeros([BV, BK], dtype=tl.float32)
    if K > BK:
        b_h2 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h2 = b_h1
    if K > 2 * BK:
        b_h3 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h3 = b_h1
    if K > 3 * BK:
        b_h4 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h4 = b_h1
    if K > 4 * BK:
        b_h5 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h5 = b_h1
    if K > 5 * BK:
        b_h6 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h6 = b_h1
    if K > 6 * BK:
        b_h7 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h7 = b_h1
    if K > 7 * BK:
        b_h8 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h8 = b_h1

    # Calculate offsets.
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * V * K
    if STORE_FINAL_STATE:
        ht = ht + i_nh * V * K

    # Load initial state.
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        b_h1 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, BK), (BV, BK), (1, 0))
            b_h2 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 2 * BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 2 * BK), (BV, BK), (1, 0))
            b_h3 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 3 * BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 3 * BK), (BV, BK), (1, 0))
            b_h4 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 4 * BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 4 * BK), (BV, BK), (1, 0))
            b_h5 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 5 * BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 5 * BK), (BV, BK), (1, 0))
            b_h6 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 6 * BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 6 * BK), (BV, BK), (1, 0))
            b_h7 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
        if K > 7 * BK:
            p_h0 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 7 * BK), (BV, BK), (1, 0))
            b_h8 += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    # Main recurrence.
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                (i_v * BV, 0), (BV, BK), (1, 0))
        tl.store(p_h, b_h1.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h2.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > 2 * BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, 2 * BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h3.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > 3 * BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, 3 * BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h4.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > 4 * BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, 4 * BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h5.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > 5 * BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, 5 * BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h6.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > 6 * BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, 6 * BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h7.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        if K > 7 * BK:
            p_h = tl.make_block_ptr(h + i_t.to(tl.int64) * stride_h, (V, K), (K, 1),
                                    (i_v * BV, 7 * BK), (BV, BK), (1, 0))
            tl.store(p_h, b_h8.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_v = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK), (1, 0))
        b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h1).to(k.dtype.element_ty))
        if K > BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h2).to(k.dtype.element_ty))
        if K > 2 * BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 2 * BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h3).to(k.dtype.element_ty))
        if K > 3 * BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 3 * BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h4).to(k.dtype.element_ty))
        if K > 4 * BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 4 * BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h5).to(k.dtype.element_ty))
        if K > 5 * BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 5 * BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h6).to(k.dtype.element_ty))
        if K > 6 * BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 6 * BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h7).to(k.dtype.element_ty))
        if K > 7 * BK:
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 7 * BK), (BT, BK), (1, 0))
            b_v += tl.dot(tl.load(p_w, boundary_check=(0, 1)), tl.trans(b_h8).to(k.dtype.element_ty))

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(v_new, (T, V), (stride_v, 1),
                                     (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t.to(tl.int64) * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > BK:
                b_h2 *= b_g_last
            if K > 2 * BK:
                b_h3 *= b_g_last
            if K > 3 * BK:
                b_h4 *= b_g_last
            if K > 4 * BK:
                b_h5 *= b_g_last
            if K > 5 * BK:
                b_h6 *= b_g_last
            if K > 6 * BK:
                b_h7 *= b_g_last
            if K > 7 * BK:
                b_h8 *= b_g_last

        if USE_GK:
            o_k = tl.arange(0, BK)
            b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                           mask=(o_k < K), other=0.0)
            b_h1 *= exp(b_gk)[None, :]
            if K > BK:
                o_k = BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h2 *= exp(b_gk)[None, :]
            if K > 2 * BK:
                o_k = 2 * BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h3 *= exp(b_gk)[None, :]
            if K > 3 * BK:
                o_k = 3 * BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h4 *= exp(b_gk)[None, :]
            if K > 4 * BK:
                o_k = 4 * BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h5 *= exp(b_gk)[None, :]
            if K > 5 * BK:
                o_k = 5 * BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h6 *= exp(b_gk)[None, :]
            if K > 6 * BK:
                o_k = 6 * BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h7 *= exp(b_gk)[None, :]
            if K > 7 * BK:
                o_k = 7 * BK + tl.arange(0, BK)
                b_gk = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
                               mask=(o_k < K), other=0.0)
                b_h8 *= exp(b_gk)[None, :]

        b_v = b_v.to(k.dtype.element_ty)

        # Update H with trans(v_new) @ K; this form avoids a MetaX map::at failure.
        b_vt = tl.trans(b_v)  # [BV, BT]
        for i_k in tl.range(NK, num_stages=2):
            p_k = tl.make_block_ptr(
                k, (T, K), (stride_k, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_kv = tl.dot(b_vt, tl.load(p_k, boundary_check=(0, 1)))
            if i_k == 0:
                b_h1 += b_kv
            if K > BK and i_k == 1:
                b_h2 += b_kv
            if K > 2 * BK and i_k == 2:
                b_h3 += b_kv
            if K > 3 * BK and i_k == 3:
                b_h4 += b_kv
            if K > 4 * BK and i_k == 4:
                b_h5 += b_kv
            if K > 5 * BK and i_k == 5:
                b_h6 += b_kv
            if K > 6 * BK and i_k == 6:
                b_h7 += b_kv
            if K > 7 * BK and i_k == 7:
                b_h8 += b_kv


    # Store final state.
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 2 * BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 2 * BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 3 * BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 3 * BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 4 * BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 4 * BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h5.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 5 * BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 5 * BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h6.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 6 * BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 6 * BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h7.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 7 * BK:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 7 * BK), (BV, BK), (1, 0))
            tl.store(p_ht, b_h8.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h_vmajor(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    assert K <= 256, "head dim > 256 unsupported"

    h = k.new_empty(B, NT, H, V, K)  # V-major layout, matches vLLM
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u) if save_new_value else None

    N_BUCKET = _n_bucket(N)
    SKEW_BUCKET = _skew_bucket(chunk_offsets if cu_seqlens is not None else None, N)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_vmajor[grid](
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
        N_BUCKET=N_BUCKET,
        SKEW_BUCKET=SKEW_BUCKET,
        pipeline="cpasync-mixed",
    )
    return h, v_new, final_state


# ---------------------------------------------------------------------------
# 3. chunk_o consuming the V-major [V, K] state layout
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
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [4]
        for num_stages in [1, 2, 3]
    ],
    key=["H", "K", "V", "BT", "IS_VARLEN", "N_BUCKET"],
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
    N_BUCKET: tl.constexpr,
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

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    # h is stored as [B, NT, H, V, K].
    h += (i_tg * H + i_h).to(tl.int64) * V * K

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1)
        )
        p_h = tl.make_block_ptr(
            h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.trans(tl.load(p_h, boundary_check=(0, 1)))

        b_o += tl.dot(b_q, b_h)
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

    # Apply scale once at the end.
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

    N = (len(cu_seqlens) - 1) if cu_seqlens is not None else B
    N_BUCKET = _n_bucket(N)

    o = torch.empty_like(v)

    def grid(meta):
        return (
            triton.cdiv(V, meta["BV"]),
            NT,
            B * H,
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
        N_BUCKET=N_BUCKET,
    )
    return o



# ---------------------------------------------------------------------------
# 4. Patched forward: fused cumsum+kkt+solve_tril -> recompute_w_u ->
#    V-major chunk_delta_h -> chunk_o
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
    logger.debug("METAX GDN CHUNK FWD (V-major, fused cumsum+kkt+solve_tril)")
    # Ensure contiguity.
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
    # Obtain WY representation; u is the new value.
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_vmajor(
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
