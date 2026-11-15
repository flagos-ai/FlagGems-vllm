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

"""BT=16 inference kernels for KDA prefill.

Two forward paths with identical semantics share one public entry, ``chunk_kda``:

* TLE path (``hygon_chunk_kda_fwd_infer``): TLE-optimized kernels.
* Triton fallback (``chunk_kda_fwd_infer_triton``): portable plain-Triton
  kernels shared verbatim with ``flaggems_vllm.ops.FLA.chunk_kda``.

``chunk_kda`` validates inputs first, then dispatches: TLE when
available, and Triton otherwise. Set ``FLAGGEMS_CHUNK_KDA_BACKEND`` to
``tle`` or ``triton`` to force a backend; the default is ``auto``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.FLA.chunk_kda import (
    _validate_chunk_kda_inputs,
    chunk_kda_fwd_infer_triton,
)
from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_KDA = True
    except ImportError:
        tle = None
        HAS_TLE_KDA = False
else:
    tle = None
    HAS_TLE_KDA = False

__all__ = ["chunk_kda"]

# =============================================================================
# Shared helpers
# =============================================================================

RCP_LN2 = 1.4426950216
_BACKEND_ENV = "FLAGGEMS_CHUNK_KDA_BACKEND"
_BACKEND_ALIASES = {
    "auto": "auto",
    "tle": "tle",
    "generic_tle": "tle",
    "triton": "triton",
    "triton_fuse": "triton",
}


def _chunk_kda_backend() -> str:
    value = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    try:
        return _BACKEND_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(
            f"invalid {_BACKEND_ENV}={value!r}; expected one of: {choices}"
        ) from exc


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


_FP16_DOT_PRECISION = tl.constexpr("ieee")


def _allocate_triton_workspace(size: int, _alignment: int, _stream) -> torch.Tensor:
    return torch.empty(size, device=flaggems_vllm.device, dtype=torch.int8)


# =============================================================================
# TLE path -- default when available
# =============================================================================

if HAS_TLE_KDA:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=num_warps, num_stages=num_stages)
            for num_warps in [2, 4, 8]
            for num_stages in [2, 4, 8]
        ],
        key=["H", "HV", "K", "BT"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _hygon_kda_fwd_intra_kernel(
        q,
        k,
        g,
        beta,
        ws,
        Aqk,
        Akk,
        g_out,
        A_log,
        dt_bias,
        lower_bound,
        scale,
        g_scale,
        l2norm_eps,
        cu_seqlens,
        chunk_indices,
        T,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        i_t, i_bh = tl.program_id(0), tl.program_id(1)
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
                chunk_indices + i_t * 2 + 1
            ).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
        else:
            bos = i_bh // HV * T

        if i_t * BT >= T:
            return

        q += (bos * H + i_h) * K
        k += (bos * H + i_h) * K
        g += (bos * HV + i_hv) * K
        g_out += (bos * HV + i_hv) * K
        Aqk += (bos * HV + i_hv) * BT
        Akk += (bos * HV + i_hv) * BT
        ws += (bos * HV + i_hv) * 3 * K
        beta += bos * HV + i_hv

        o_i = tl.arange(0, BT)
        o_c = i_t * BT + o_i
        m_c = o_c < T

        # Reuse q/k/g cumsum from shared memory across the intra-chunk phases.
        q_buf = tle.gpu.alloc([BT, K], dtype=q.dtype.element_ty, scope=tle.gpu.smem)
        k_buf = tle.gpu.alloc([BT, K], dtype=k.dtype.element_ty, scope=tle.gpu.smem)
        gc_buf = tle.gpu.alloc([BT, K], dtype=tl.float32, scope=tle.gpu.smem)

        rows = tl.broadcast_to(tl.arange(0, BT)[:, None], (BT, K))
        cols = tl.broadcast_to(tl.arange(0, K)[None, :], (BT, K))
        q_sp = tle.gpu.local_ptr(q_buf, (rows, cols))
        k_sp = tle.gpu.local_ptr(k_buf, (rows, cols))
        gc_sp = tle.gpu.local_ptr(gc_buf, (rows, cols))

        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        tl.store(q_sp, b_q)
        tl.store(k_sp, b_k)

        b_qf = b_q.to(tl.float32)
        b_kf = b_k.to(tl.float32)

        b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
        p_dt = tl.make_block_ptr(dt_bias + i_hv * K, (K,), (1,), (0,), (K,), (0,))
        b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
        b_g = b_g + b_bias[None, :]
        # FlashKDA-compatible safe gate; lower_bound is required by the verifier.
        b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
        tl.store(gc_sp, b_g)
        one_row = tl.broadcast_to(tl.arange(0, 1)[:, None], (1, K))
        col_row = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
        b_acc = tl.zeros([1, K], dtype=tl.float32)
        for r in tl.static_range(BT):
            rp = tle.gpu.local_ptr(
                gc_buf, (tl.broadcast_to(one_row + r, (1, K)), col_row)
            )
            b_acc = b_acc + tl.load(rp)
            tl.store(rp, b_acc)

        p_g_out = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        tl.store(
            p_g_out, tl.load(gc_sp).to(g_out.dtype.element_ty), boundary_check=(0, 1)
        )

        # Intra-chunk Aqk/Akk plus triangular solve.
        b_gq = tl.where(m_c[:, None], exp2(tl.load(gc_sp)), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-tl.load(gc_sp)), 0.0)

        # Keep b_gq/b_gk in fp32: exp2(±cumsum) can exceed fp16 max (65504), casting would overflow.
        # For bfloat16, bf16 range (3.4e38) is sufficient, so cast is safe.
        if q.dtype.element_ty == tl.float16:
            b_kgt = tl.trans(b_kf * b_gk)
            b_Aqk = tl.dot(
                b_qf * b_gq,
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )
            b_Akk = tl.dot(
                b_kf * b_gq,
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )
        else:
            b_kgt = tl.trans(b_kf * b_gk).to(b_k.dtype)
            b_Aqk = tl.dot(
                (b_qf * b_gq).to(b_q.dtype),
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )
            b_Akk = tl.dot(
                (b_kf * b_gq).to(b_k.dtype),
                b_kgt,
                input_precision=_FP16_DOT_PRECISION,
                out_dtype=tl.float32,
            )

        b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
        b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

        p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.sigmoid(tl.load(p_beta, boundary_check=(0,)).to(tl.float32))

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

        p_Akk_out = tl.make_block_ptr(
            Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))

        # Pack w, qg, and kg into one workspace at columns 0, K, and 2*K.
        b_k3 = tl.load(k_sp).to(tl.float32) * b_k_rstd[:, None]
        b_gk3 = tl.load(gc_sp)
        b_kb = b_k3 * b_beta[:, None] * exp2(b_gk3)
        p_w = tl.make_block_ptr(
            ws, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        tl.store(p_w, b_kb.to(ws.dtype.element_ty), boundary_check=(0, 1))

        b_q3 = tl.load(q_sp).to(tl.float32) * b_q_rstd[:, None]
        b_qg_val = b_q3 * exp2(b_gk3)
        p_qg = tl.make_block_ptr(
            ws, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K), (BT, K), (1, 0)
        )
        tl.store(p_qg, b_qg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

        last_local = tl.minimum(BT, T - i_t * BT) - 1
        gn_rows = tl.broadcast_to(last_local + tl.zeros([1, K], dtype=tl.int32), (1, K))
        gn_cols = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
        b_gn = tl.load(tle.gpu.local_ptr(gc_buf, (gn_rows, gn_cols)))
        b_kg_val = b_k3 * tl.where(m_c[:, None], exp2(b_gn - b_gk3), 0)
        p_kg = tl.make_block_ptr(
            ws, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 2 * K), (BT, K), (1, 0)
        )
        tl.store(p_kg, b_kg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

    def _hygon_kda_fwd_intra(
        q,
        k,
        g,
        beta,
        scale,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=16,
        lower_bound=None,
        A_log=None,
        dt_bias=None,
    ):
        B, T_len, H, K = q.shape
        HV = g.shape[2]
        BT = chunk_size

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
        grid = (NT, B * HV)

        g_out = torch.empty(B, T_len, HV, K, device=q.device, dtype=torch.float32)
        ws = torch.empty(B, T_len, HV, 3 * K, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(B, T_len, HV, BT, device=q.device, dtype=q.dtype)
        Akk = torch.zeros(B, T_len, HV, BT, device=q.device, dtype=q.dtype)

        _hygon_kda_fwd_intra_kernel[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            ws=ws,
            Aqk=Aqk,
            Akk=Akk,
            g_out=g_out,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            scale=scale,
            g_scale=RCP_LN2,
            l2norm_eps=1e-6,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T_len,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
        )
        return ws, Aqk, Akk, g_out

    # -----------------------------------------------------------------------------
    # Kernel 2: state propagation + output
    # -----------------------------------------------------------------------------

    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
            "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[
            triton.Config({"BV": BV}, num_warps=num_warps)
            for BV in [32, 64]
            for num_warps in [2, 4]
        ],
        key=["HV", "K", "V", "BT"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _hygon_kda_fwd_state_output_kernel(
        kg,
        v,
        beta,
        gk,
        Aqk,
        Akk,
        o,
        ws,
        h0,
        ht,
        cu_seqlens,
        scale,
        T,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        STATE_V_FIRST: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        i_v, i_nh = tl.program_id(0), tl.program_id(1)

        if IS_VARLEN:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            NT = tl.cdiv(T, BT)
        else:
            i_n = i_nh // HV
            i_h = i_nh % HV
            bos = i_n * T
            NT = tl.cdiv(T, BT)

        v += (bos * HV + i_h).to(tl.int64) * V
        beta += bos * HV + i_h
        gk += (bos * HV + i_h).to(tl.int64) * K
        Aqk += (bos * HV + i_h).to(tl.int64) * BT
        Akk += (bos * HV + i_h).to(tl.int64) * BT
        o += (bos * HV + i_h).to(tl.int64) * V
        ws_base = ws + (bos * HV + i_h).to(tl.int64) * 3 * K

        kg_dtype = kg.dtype.element_ty

        # Initial state.
        if USE_INITIAL_STATE:
            if STATE_V_FIRST:
                p_h0_1 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
                )
                b_h1 = tl.trans(tl.load(p_h0_1, boundary_check=(0, 1))).to(tl.float32)
                if K > 64:
                    p_h0_2 = tl.make_block_ptr(
                        h0 + i_nh * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 64),
                        (BV, 64),
                        (1, 0),
                    )
                    b_h2 = tl.trans(tl.load(p_h0_2, boundary_check=(0, 1))).to(
                        tl.float32
                    )
                if K > 128:
                    p_h0_3 = tl.make_block_ptr(
                        h0 + i_nh * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 128),
                        (BV, 64),
                        (1, 0),
                    )
                    b_h3 = tl.trans(tl.load(p_h0_3, boundary_check=(0, 1))).to(
                        tl.float32
                    )
                if K > 192:
                    p_h0_4 = tl.make_block_ptr(
                        h0 + i_nh * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 192),
                        (BV, 64),
                        (1, 0),
                    )
                    b_h4 = tl.trans(tl.load(p_h0_4, boundary_check=(0, 1))).to(
                        tl.float32
                    )
            else:
                p_h0_1 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
                )
                b_h1 = tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                if K > 64:
                    p_h0_2 = tl.make_block_ptr(
                        h0 + i_nh * K * V,
                        (K, V),
                        (V, 1),
                        (64, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_h2 = tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                if K > 128:
                    p_h0_3 = tl.make_block_ptr(
                        h0 + i_nh * K * V,
                        (K, V),
                        (V, 1),
                        (128, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_h3 = tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                if K > 192:
                    p_h0_4 = tl.make_block_ptr(
                        h0 + i_nh * K * V,
                        (K, V),
                        (V, 1),
                        (192, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    b_h4 = tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_h1 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 64:
                b_h2 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 128:
                b_h3 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 192:
                b_h4 = tl.zeros([64, BV], dtype=tl.float32)

        for i_t in tl.range(NT):
            last_idx = tl.minimum(i_t * BT + BT, T) - 1

            # Load w / qg / kg tiles from the packed workspace (offsets 0/K/2K).
            p_w1 = tl.make_block_ptr(
                ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
            )
            b_w1 = tl.load(p_w1, boundary_check=(0, 1))
            p_qg1 = tl.make_block_ptr(
                ws_base, (T, 3 * K), (HV * 3 * K, 1), (i_t * BT, K), (BT, 64), (1, 0)
            )
            b_qg1 = tl.load(p_qg1, boundary_check=(0, 1))
            p_kg1 = tl.make_block_ptr(
                ws_base,
                (T, 3 * K),
                (HV * 3 * K, 1),
                (i_t * BT, 2 * K),
                (BT, 64),
                (1, 0),
            )
            b_kg1 = tl.load(p_kg1, boundary_check=(0, 1))
            if K > 64:
                p_w2 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, 64),
                    (BT, 64),
                    (1, 0),
                )
                b_w2 = tl.load(p_w2, boundary_check=(0, 1))
                p_qg2 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, K + 64),
                    (BT, 64),
                    (1, 0),
                )
                b_qg2 = tl.load(p_qg2, boundary_check=(0, 1))
                p_kg2 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, 2 * K + 64),
                    (BT, 64),
                    (1, 0),
                )
                b_kg2 = tl.load(p_kg2, boundary_check=(0, 1))
            if K > 128:
                p_w3 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, 128),
                    (BT, 64),
                    (1, 0),
                )
                b_w3 = tl.load(p_w3, boundary_check=(0, 1))
                p_qg3 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, K + 128),
                    (BT, 64),
                    (1, 0),
                )
                b_qg3 = tl.load(p_qg3, boundary_check=(0, 1))
                p_kg3 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, 2 * K + 128),
                    (BT, 64),
                    (1, 0),
                )
                b_kg3 = tl.load(p_kg3, boundary_check=(0, 1))
            if K > 192:
                p_w4 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, 192),
                    (BT, 64),
                    (1, 0),
                )
                b_w4 = tl.load(p_w4, boundary_check=(0, 1))
                p_qg4 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, K + 192),
                    (BT, 64),
                    (1, 0),
                )
                b_qg4 = tl.load(p_qg4, boundary_check=(0, 1))
                p_kg4 = tl.make_block_ptr(
                    ws_base,
                    (T, 3 * K),
                    (HV * 3 * K, 1),
                    (i_t * BT, 2 * K + 192),
                    (BT, 64),
                    (1, 0),
                )
                b_kg4 = tl.load(p_kg4, boundary_check=(0, 1))

            # Load Aqk / Akk / gk tiles.
            p_Aqk = tl.make_block_ptr(
                Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
            )
            b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
            p_Akk = tl.make_block_ptr(
                Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
            )
            b_Akk = tl.load(p_Akk, boundary_check=(0, 1))
            p_gk1 = tl.make_block_ptr(
                gk, (T, K), (HV * K, 1), (last_idx, 0), (1, 64), (1, 0)
            )
            b_gk1 = tl.load(p_gk1, boundary_check=(0, 1)).reshape([64])
            if K > 64:
                p_gk2 = tl.make_block_ptr(
                    gk, (T, K), (HV * K, 1), (last_idx, 64), (1, 64), (1, 0)
                )
                b_gk2 = tl.load(p_gk2, boundary_check=(0, 1)).reshape([64])
            if K > 128:
                p_gk3 = tl.make_block_ptr(
                    gk, (T, K), (HV * K, 1), (last_idx, 128), (1, 64), (1, 0)
                )
                b_gk3 = tl.load(p_gk3, boundary_check=(0, 1)).reshape([64])
            if K > 192:
                p_gk4 = tl.make_block_ptr(
                    gk, (T, K), (HV * K, 1), (last_idx, 192), (1, 64), (1, 0)
                )
                b_gk4 = tl.load(p_gk4, boundary_check=(0, 1)).reshape([64])

            # v and beta need elementwise work: vb = v * sigmoid(beta).
            p_v = tl.make_block_ptr(
                v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
            b_vb = (b_v.to(tl.float32) * b_beta_f[:, None]).to(b_v.dtype)

            b_h1_bf = b_h1.to(kg_dtype)
            if K > 64:
                b_h2_bf = b_h2.to(kg_dtype)
            if K > 128:
                b_h3_bf = b_h3.to(kg_dtype)
            if K > 192:
                b_h4_bf = b_h4.to(kg_dtype)

            # v_new = Akk_inv @ (v*beta - w @ h)
            b_kh = tl.dot(b_w1, b_h1_bf).to(tl.float32)
            if K > 64:
                b_kh += tl.dot(b_w2, b_h2_bf).to(tl.float32)
            if K > 128:
                b_kh += tl.dot(b_w3, b_h3_bf).to(tl.float32)
            if K > 192:
                b_kh += tl.dot(b_w4, b_h4_bf).to(tl.float32)
            b_diff = b_vb.to(tl.float32) - b_kh
            b_v = tl.dot(b_Akk, b_diff.to(kg_dtype)).to(tl.float32)

            # output = scale * qg @ h + Aqk @ v_new
            b_qh = tl.dot(b_qg1, b_h1_bf).to(tl.float32)
            if K > 64:
                b_qh += tl.dot(b_qg2, b_h2_bf).to(tl.float32)
            if K > 128:
                b_qh += tl.dot(b_qg3, b_h3_bf).to(tl.float32)
            if K > 192:
                b_qh += tl.dot(b_qg4, b_h4_bf).to(tl.float32)
            b_o = scale * b_qh
            b_v_cast = b_v.to(kg_dtype)
            b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

            p_o = tl.make_block_ptr(
                o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

            # state decay + update: h = h * exp2(gk_last) + kg^T @ v_new
            b_h1 = b_h1 * exp2(b_gk1)[:, None] + tl.dot(tl.trans(b_kg1), b_v_cast).to(
                tl.float32
            )
            if K > 64:
                b_h2 = b_h2 * exp2(b_gk2)[:, None] + tl.dot(
                    tl.trans(b_kg2), b_v_cast
                ).to(tl.float32)
            if K > 128:
                b_h3 = b_h3 * exp2(b_gk3)[:, None] + tl.dot(
                    tl.trans(b_kg3), b_v_cast
                ).to(tl.float32)
            if K > 192:
                b_h4 = b_h4 * exp2(b_gk4)[:, None] + tl.dot(
                    tl.trans(b_kg4), b_v_cast
                ).to(tl.float32)

        # Final state.
        if STORE_FINAL_STATE:
            if STATE_V_FIRST:
                p_ht1 = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
                )
                tl.store(
                    p_ht1,
                    tl.trans(b_h1).to(p_ht1.dtype.element_ty),
                    boundary_check=(0, 1),
                )
                if K > 64:
                    p_ht2 = tl.make_block_ptr(
                        ht + i_nh * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 64),
                        (BV, 64),
                        (1, 0),
                    )
                    tl.store(
                        p_ht2,
                        tl.trans(b_h2).to(p_ht2.dtype.element_ty),
                        boundary_check=(0, 1),
                    )
                if K > 128:
                    p_ht3 = tl.make_block_ptr(
                        ht + i_nh * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 128),
                        (BV, 64),
                        (1, 0),
                    )
                    tl.store(
                        p_ht3,
                        tl.trans(b_h3).to(p_ht3.dtype.element_ty),
                        boundary_check=(0, 1),
                    )
                if K > 192:
                    p_ht4 = tl.make_block_ptr(
                        ht + i_nh * K * V,
                        (V, K),
                        (K, 1),
                        (i_v * BV, 192),
                        (BV, 64),
                        (1, 0),
                    )
                    tl.store(
                        p_ht4,
                        tl.trans(b_h4).to(p_ht4.dtype.element_ty),
                        boundary_check=(0, 1),
                    )
            else:
                p_ht1 = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_ht1, b_h1.to(p_ht1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    p_ht2 = tl.make_block_ptr(
                        ht + i_nh * K * V,
                        (K, V),
                        (V, 1),
                        (64, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    tl.store(
                        p_ht2, b_h2.to(p_ht2.dtype.element_ty), boundary_check=(0, 1)
                    )
                if K > 128:
                    p_ht3 = tl.make_block_ptr(
                        ht + i_nh * K * V,
                        (K, V),
                        (V, 1),
                        (128, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    tl.store(
                        p_ht3, b_h3.to(p_ht3.dtype.element_ty), boundary_check=(0, 1)
                    )
                if K > 192:
                    p_ht4 = tl.make_block_ptr(
                        ht + i_nh * K * V,
                        (K, V),
                        (V, 1),
                        (192, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    tl.store(
                        p_ht4, b_h4.to(p_ht4.dtype.element_ty), boundary_check=(0, 1)
                    )

    def _hygon_kda_fwd_state_output(
        kg: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        Akk: torch.Tensor,
        gk: torch.Tensor,
        Aqk: torch.Tensor,
        scale: float | None,
        ws: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        state_v_first: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, _, HV, K = kg.shape
        T_actual = v.shape[1]
        V = v.shape[-1]
        BT = chunk_size

        if K > 256:
            raise ValueError(f"KDA K must be <= 256, got {K}")

        if cu_seqlens is None:
            N = B
        else:
            N = len(cu_seqlens) - 1

        final_state = None
        if output_final_state:
            if state_v_first:
                final_state = kg.new_zeros(N, HV, V, K, dtype=torch.float32)
            else:
                final_state = kg.new_zeros(N, HV, K, V, dtype=torch.float32)

        o = torch.zeros(B, T_actual, HV, V, device=kg.device, dtype=v.dtype)

        h0_arg = (
            initial_state
            if initial_state is not None
            else kg.new_empty(1, dtype=torch.float32)
        )
        ht_arg = (
            final_state
            if final_state is not None
            else kg.new_empty(1, dtype=torch.float32)
        )

        grid = lambda meta: (triton.cdiv(V, meta["BV"]), N * HV)
        _hygon_kda_fwd_state_output_kernel[grid](
            kg=kg,
            v=v,
            beta=beta,
            gk=gk,
            Aqk=Aqk,
            Akk=Akk,
            o=o,
            ws=ws,
            h0=h0_arg,
            ht=ht_arg,
            cu_seqlens=cu_seqlens,
            scale=scale,
            T=T_actual,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            STATE_V_FIRST=state_v_first,
        )

        return o, final_state


def hygon_chunk_kda_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
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
    triton.set_allocator(_allocate_triton_workspace)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    ws, Aqk, Akk, g_cumsum = _hygon_kda_fwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    K = q.shape[-1]
    return _hygon_kda_fwd_state_output(
        kg=ws[:, :, :, 2 * K :],
        v=v,
        beta=beta,
        Akk=Akk,
        gk=g_cumsum,
        Aqk=Aqk,
        ws=ws,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )


# =============================================================================
# Public entry: validate, then dispatch (TLE if available, else Triton)
# =============================================================================


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
    allow_neg_eigval: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inference-only implementation of chunk Kimi Delta Attention.

    Inputs use seq-first layout: q/k ``[B, T, H, K]``, v/g ``[B, T, HV, *]``,
    and beta ``[B, T, HV]``. q/k L2 norm, gate activation, and beta sigmoid are
    computed inside the kernels. Backend selection defaults to automatic and
    can be forced with ``FLAGGEMS_CHUNK_KDA_BACKEND``.
    """
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")

    _validate_chunk_kda_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )

    backend = _chunk_kda_backend()
    if backend == "tle" and not HAS_TLE_KDA:
        raise RuntimeError(f"{_BACKEND_ENV}=tle requires Triton TLE >= 3.6.0")

    if backend in {"auto", "tle"} and HAS_TLE_KDA:
        return hygon_chunk_kda_fwd_infer(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    return chunk_kda_fwd_infer_triton(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )
