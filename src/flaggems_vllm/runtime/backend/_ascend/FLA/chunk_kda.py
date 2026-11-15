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

* TLE path (``ascend_chunk_kda_fwd_infer``): TLE-optimized kernels.
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

from flaggems_vllm.ops.FLA.chunk_kda import (
    _validate_chunk_kda_inputs,
    chunk_kda_fwd_infer_triton,
)
from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 5, 0):
    try:
        import triton.experimental.tle as tle

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
MAX_GRID_BLOCKS = 32768

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


# =============================================================================
# TLE path -- default when available
# =============================================================================

if HAS_TLE_KDA:

    # ------------------------------------------------------------------
    # Kernel 1: intra-chunk Aqk/Akk + triangular inverse + workspace packing
    # ------------------------------------------------------------------

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.jit(do_not_specialize=["T"])
    def _ascend_kda_fwd_intra_kernel(
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
        NC,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        CPT: tl.constexpr,
        IS_VARLEN: tl.constexpr,
    ):
        i_tc, i_bh = tl.program_id(0), tl.program_id(1)
        i_hv = i_bh % HV
        i_h = i_hv // (HV // H)

        for c in tl.static_range(CPT):
            i_t = i_tc * CPT + c
            if i_t < NC:
                if IS_VARLEN:
                    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
                        chunk_indices + i_t * 2 + 1
                    ).to(tl.int32)
                    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
                    T = tl.load(cu_seqlens + i_n + 1).to(tl.int32) - bos
                else:
                    bos = i_bh // HV * T

                if i_t * BT < T:
                    p_q = q + (bos * H + i_h).to(tl.int64) * K
                    p_k = k + (bos * H + i_h).to(tl.int64) * K
                    p_g = g + (bos * HV + i_hv).to(tl.int64) * K
                    p_g_out = g_out + (bos * HV + i_hv).to(tl.int64) * K
                    p_Aqk = Aqk + (bos * HV + i_hv).to(tl.int64) * BT
                    p_Akk = Akk + (bos * HV + i_hv).to(tl.int64) * BT
                    p_ws = ws + (bos * HV + i_hv).to(tl.int64) * 3 * K
                    p_beta = beta + bos * HV + i_hv

                    o_i = tl.arange(0, BT)
                    o_c = i_t * BT + o_i
                    m_c = o_c < T
                    o_k = tl.arange(0, K)

                    # [BT, K] tiles; q/k/g are the real (unpadded) [B, T, H/HV, K] tensors.
                    ptrs_qk = o_c[:, None] * (H * K) + o_k[None, :]
                    ptrs_g = o_c[:, None] * (HV * K) + o_k[None, :]
                    b_q = tl.load(p_q + ptrs_qk, mask=m_c[:, None], other=0.0)
                    b_k = tl.load(p_k + ptrs_qk, mask=m_c[:, None], other=0.0)
                    b_g = tl.load(p_g + ptrs_g, mask=m_c[:, None], other=0.0).to(
                        tl.float32
                    )

                    b_qf = b_q.to(tl.float32)
                    b_kf = b_k.to(tl.float32)

                    b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
                    b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

                    b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
                    b_bias = tl.load(dt_bias + i_hv * K + o_k).to(tl.float32)
                    b_g = b_g + b_bias[None, :]
                    # FlashKDA-compatible safe gate; lower_bound is required by the verifier.
                    b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)

                    acc = tl.zeros((K,), dtype=tl.float32)
                    b_gc = b_g
                    for r in tl.static_range(BT):
                        row = tl.reshape(
                            tle.dsa.extract_slice(b_gc, (r, 0), (1, K), (1, 1)), (K,)
                        )
                        acc += row
                        b_gc = tle.dsa.insert_slice(
                            b_gc, tl.reshape(acc, (1, K)), (r, 0), (1, K), (1, 1)
                        )

                    if IS_VARLEN:
                        tl.store(
                            p_g_out + o_c[:, None] * (HV * K) + o_k[None, :],
                            b_gc,
                            mask=m_c[:, None],
                        )
                    else:
                        tl.store(p_g_out + o_c[:, None] * (HV * K) + o_k[None, :], b_gc)

                    b_gq = tl.where(m_c[:, None], exp2(b_gc), 0.0)
                    b_gk = tl.where(m_c[:, None], exp2(-b_gc), 0.0)

                    b_kg_t = tl.trans((b_kf * b_gk).to(tl.bfloat16))
                    b_Aqk = tl.dot(
                        (b_qf * b_gq).to(tl.bfloat16), b_kg_t, out_dtype=tl.float32
                    )
                    b_Akk = tl.dot(
                        (b_kf * b_gq).to(tl.bfloat16), b_kg_t, out_dtype=tl.float32
                    )

                    b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
                    b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

                    b_beta = tl.sigmoid(
                        tl.load(p_beta + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
                    )

                    m_Aqk = o_i[:, None] >= o_i[None, :]
                    m_Akk = o_i[:, None] > o_i[None, :]
                    m_I = o_i[:, None] == o_i[None, :]

                    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
                    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

                    if IS_VARLEN:
                        tl.store(
                            p_Aqk + o_c[:, None] * (HV * BT) + o_i[None, :],
                            b_Aqk.to(Aqk.dtype.element_ty),
                            mask=m_c[:, None],
                        )
                    else:
                        tl.store(
                            p_Aqk + o_c[:, None] * (HV * BT) + o_i[None, :],
                            b_Aqk.to(Aqk.dtype.element_ty),
                        )

                    b_L = b_Akk.to(tl.bfloat16)
                    b_Ai = m_I.to(tl.float32) - b_Akk
                    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float32).to(tl.bfloat16)
                    b_Ai = b_Ai + tl.dot(
                        b_Ai.to(tl.bfloat16), b_L2, out_dtype=tl.float32
                    )
                    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float32).to(tl.bfloat16)
                    b_Ai = b_Ai + tl.dot(
                        b_Ai.to(tl.bfloat16), b_L4, out_dtype=tl.float32
                    )
                    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float32).to(tl.bfloat16)
                    b_Ai = b_Ai + tl.dot(
                        b_Ai.to(tl.bfloat16), b_L8, out_dtype=tl.float32
                    )

                    if IS_VARLEN:
                        tl.store(
                            p_Akk + o_c[:, None] * (HV * BT) + o_i[None, :],
                            b_Ai.to(Akk.dtype.element_ty),
                            mask=m_c[:, None],
                        )
                    else:
                        tl.store(
                            p_Akk + o_c[:, None] * (HV * BT) + o_i[None, :],
                            b_Ai.to(Akk.dtype.element_ty),
                        )

                    # Pack w, qg, kg into the workspace at column offsets 0, K and 2*K.
                    ptrs_ws = o_c[:, None] * (HV * 3 * K) + o_k[None, :]
                    b_k3 = b_kf * b_k_rstd[:, None]
                    b_kb = b_k3 * b_beta[:, None] * exp2(b_gc)
                    b_q3 = b_qf * b_q_rstd[:, None]
                    b_qg_val = b_q3 * exp2(b_gc)
                    last_local = tl.minimum(BT, T - i_t * BT) - 1
                    b_gn = tl.sum(tl.where(o_i[:, None] == last_local, b_gc, 0.0), 0)
                    b_kg_val = b_k3 * tl.where(
                        m_c[:, None], exp2(b_gn[None, :] - b_gc), 0.0
                    )
                    if IS_VARLEN:
                        tl.store(
                            p_ws + ptrs_ws,
                            b_kb.to(ws.dtype.element_ty),
                            mask=m_c[:, None],
                        )
                        tl.store(
                            p_ws + ptrs_ws + K,
                            b_qg_val.to(ws.dtype.element_ty),
                            mask=m_c[:, None],
                        )
                        tl.store(
                            p_ws + ptrs_ws + 2 * K,
                            b_kg_val.to(ws.dtype.element_ty),
                            mask=m_c[:, None],
                        )
                    else:
                        tl.store(p_ws + ptrs_ws, b_kb.to(ws.dtype.element_ty))
                        tl.store(p_ws + ptrs_ws + K, b_qg_val.to(ws.dtype.element_ty))
                        tl.store(
                            p_ws + ptrs_ws + 2 * K, b_kg_val.to(ws.dtype.element_ty)
                        )

    def _ascend_kda_fwd_intra(
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
        BHV = B * HV

        # Cap the linear grid at MAX_GRID_BLOCKS.
        # Each program then handles CPT consecutive chunks; CPT == 1 reproduces
        # the original (NT, B*HV) grid exactly.
        CPT = triton.cdiv(NT * BHV, MAX_GRID_BLOCKS)
        grid0 = triton.cdiv(NT, CPT)
        while grid0 * BHV > MAX_GRID_BLOCKS:
            CPT += 1
            grid0 = triton.cdiv(NT, CPT)
        grid = (grid0, BHV)

        # Pad the workspace T dimension so full BT tiles can always be stored.
        T_padded = NT * BT
        g_out = torch.empty(B, T_padded, HV, K, device=q.device, dtype=torch.float32)
        ws = torch.empty(B, T_padded, HV, 3 * K, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(B, T_padded, HV, BT, device=q.device, dtype=q.dtype)
        Akk = torch.zeros(B, T_padded, HV, BT, device=q.device, dtype=q.dtype)

        _ascend_kda_fwd_intra_kernel[grid](
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
            NC=NT,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            CPT=CPT,
        )

        return ws, Aqk, Akk, g_out

    # ------------------------------------------------------------------
    # Kernel 2: state propagation + output
    # ------------------------------------------------------------------

    def _pick_bv(K: int) -> int:
        return 64 if K <= 128 else 32

    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
            "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.jit(do_not_specialize=["T"])
    def _ascend_kda_fwd_state_output_kernel(
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

        o_t = tl.arange(0, BT)
        o_k = tl.arange(0, K)
        o_v = tl.arange(0, BV)
        m_v = i_v * BV + o_v < V

        # Initial state h [K, BV] fp32.
        if USE_INITIAL_STATE:
            h0 += i_nh.to(tl.int64) * K * V
            if STATE_V_FIRST:
                # h0 is [N, HV, V, K]
                b_h = tl.trans(
                    tl.load(
                        h0 + (i_v * BV + o_v)[:, None] * K + o_k[None, :],
                        mask=m_v[:, None],
                        other=0.0,
                    )
                ).to(tl.float32)
            else:
                # h0 is [N, HV, K, V]
                b_h = tl.load(
                    h0 + o_k[:, None] * V + (i_v * BV + o_v)[None, :],
                    mask=m_v[None, :],
                    other=0.0,
                ).to(tl.float32)
        else:
            b_h = tl.zeros((K, BV), dtype=tl.float32)

        # Sequential recurrence across chunks.
        for i_t in range(NT):
            o_c = i_t * BT + o_t
            m_c = o_c < T

            ptrs_ws = o_c[:, None] * (HV * 3 * K) + o_k[None, :]
            b_w = tl.load(ws_base + ptrs_ws)
            b_qg = tl.load(ws_base + ptrs_ws + K)
            b_kg = tl.load(ws_base + ptrs_ws + 2 * K)
            b_Aqk = tl.load(Aqk + o_c[:, None] * (HV * BT) + o_t[None, :])
            b_Akk = tl.load(Akk + o_c[:, None] * (HV * BT) + o_t[None, :])
            if IS_VARLEN:
                z = m_c[:, None]
                b_w = tl.where(z, b_w, 0.0)
                b_qg = tl.where(z, b_qg, 0.0)
                b_kg = tl.where(z, b_kg, 0.0)
                b_Aqk = tl.where(z, b_Aqk, 0.0)
                b_Akk = tl.where(z, b_Akk, 0.0)

            last_idx = tl.minimum(i_t * BT + BT, T) - 1
            b_gk_last = tl.load(gk + last_idx * (HV * K) + o_k)

            b_v = tl.load(
                v + o_c[:, None] * (HV * V) + (i_v * BV + o_v)[None, :],
                mask=m_c[:, None],
                other=0.0,
            )
            b_beta = tl.sigmoid(
                tl.load(beta + o_c * HV, mask=m_c, other=0.0).to(tl.float32)
            )
            b_vb = b_v.to(tl.float32) * b_beta[:, None]

            b_h_cast = b_h.to(tl.bfloat16)
            b_kh = tl.dot(b_w.to(tl.bfloat16), b_h_cast, out_dtype=tl.float32)
            b_diff = b_vb - b_kh
            b_v_new = tl.dot(
                b_Akk.to(tl.bfloat16), b_diff.to(tl.bfloat16), out_dtype=tl.float32
            )

            # o = scale * qg @ h + Aqk @ v_new
            b_qh = tl.dot(b_qg.to(tl.bfloat16), b_h_cast, out_dtype=tl.float32)
            b_o = scale * b_qh + tl.dot(
                b_Aqk.to(tl.bfloat16), b_v_new.to(tl.bfloat16), out_dtype=tl.float32
            )
            tl.store(
                o + o_c[:, None] * (HV * V) + (i_v * BV + o_v)[None, :],
                b_o.to(o.dtype.element_ty),
                mask=m_c[:, None],
            )

            # h = h * exp2(gk_last) + kg^T @ v_new
            b_h = b_h * exp2(b_gk_last)[:, None] + tl.dot(
                tl.trans(b_kg).to(tl.bfloat16),
                b_v_new.to(tl.bfloat16),
                out_dtype=tl.float32,
            )

        # Final state.
        if STORE_FINAL_STATE:
            ht += i_nh.to(tl.int64) * K * V
            if STATE_V_FIRST:
                tl.store(
                    ht + (i_v * BV + o_v)[:, None] * K + o_k[None, :],
                    tl.trans(b_h).to(ht.dtype.element_ty),
                    mask=m_v[:, None],
                )
            else:
                tl.store(
                    ht + o_k[:, None] * V + (i_v * BV + o_v)[None, :],
                    b_h.to(ht.dtype.element_ty),
                    mask=m_v[None, :],
                )

    def _ascend_kda_fwd_state_output(
        kg,
        v,
        beta,
        Akk,
        gk,
        Aqk,
        scale,
        ws=None,
        initial_state=None,
        output_final_state=False,
        state_v_first=True,
        cu_seqlens=None,
        chunk_size=16,
    ):
        B, _, HV, K = kg.shape
        T_actual = v.shape[1]
        V = v.shape[-1]
        BT = Akk.shape[-1]

        if K > 256:
            raise ValueError(f"KDA K must be <= 256, got {K}")

        if cu_seqlens is None:
            N = B
        else:
            N = len(cu_seqlens) - 1

        final_state = None
        if output_final_state:
            if state_v_first:
                final_state = v.new_zeros(N, HV, V, K, dtype=torch.float32)
            else:
                final_state = v.new_zeros(N, HV, K, V, dtype=torch.float32)

        o = torch.zeros(B, T_actual, HV, V, device=v.device, dtype=v.dtype)

        h0_arg = (
            initial_state
            if initial_state is not None
            else v.new_empty(1, dtype=torch.float32)
        )
        ht_arg = (
            final_state
            if final_state is not None
            else v.new_empty(1, dtype=torch.float32)
        )

        grid = (triton.cdiv(V, _pick_bv(K)), N * HV)
        _ascend_kda_fwd_state_output_kernel[grid](
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
            BV=_pick_bv(K),
            STATE_V_FIRST=state_v_first,
        )

        return o, final_state

    def ascend_tle_chunk_kda_fwd_infer(
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
        if scale is None:
            scale = q.shape[-1] ** -0.5

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        ws, Aqk, Akk, g_cumsum = _ascend_kda_fwd_intra(
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
        return _ascend_kda_fwd_state_output(
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
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Inference-only implementation of chunk Kimi Delta Attention (Ascend).

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
        raise RuntimeError(
            f"{_BACKEND_ENV}=tle requires Triton TLE >= 3.5.0 "
            "(FlagTree triton on Ascend)"
        )

    if backend in {"auto", "tle"} and HAS_TLE_KDA:
        return ascend_tle_chunk_kda_fwd_infer(
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
            chunk_indices=chunk_indices,
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
