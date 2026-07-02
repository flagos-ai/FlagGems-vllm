"""BT=16 inference kernels for KDA prefill.

Two forward paths with identical semantics share one public entry, ``chunk_kda``:

* TLE path (``chunk_kda_fwd_infer``): TMA-accelerated, warp-specialized fused
  kernels. Selected automatically when the Triton TLE extension is available.
* Triton fallback (``chunk_kda_fwd_infer_triton``): portable plain-Triton
  kernels used when TLE is unavailable (e.g. CI).

``chunk_kda`` validates inputs first, then dispatches: TLE if available,
otherwise Triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
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


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


_FP16_DOT_PRECISION = tl.constexpr("ieee")
_FP16_DOT_PRECISION_REFRESHED = False


def _refresh_fp16_dot_precision() -> None:
    global _FP16_DOT_PRECISION, _FP16_DOT_PRECISION_REFRESHED
    if _FP16_DOT_PRECISION_REFRESHED:
        return
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
            if major >= 8:
                _FP16_DOT_PRECISION = tl.constexpr("tf32")
    except Exception:
        _FP16_DOT_PRECISION = tl.constexpr("ieee")
    _FP16_DOT_PRECISION_REFRESHED = True


def _allocate_triton_workspace(size: int, _alignment: int, _stream) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


# =============================================================================
# TLE path (fused, TMA + warp-specialized) -- default when available
# =============================================================================

if HAS_TLE_KDA:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[
            triton.Config(
                {}, num_warps=num_warps, num_stages=num_stages, maxnreg=maxnreg
            )
            for num_warps in [2, 4, 8]
            for num_stages in [2, 4, 8]
            for maxnreg in [None, 32, 64, 72]
        ],
        key=["H", "HV", "K", "BT"],
    )
    @triton.jit(do_not_specialize=["T"])
    def _kda_fwd_intra_kernel(
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
        b_q = tle.load(p_q, boundary_check=(0, 1), is_async=True)
        b_k = tle.load(p_k, boundary_check=(0, 1), is_async=True)
        tl.store(q_sp, b_q)
        tl.store(k_sp, b_k)

        b_qf = b_q.to(tl.float32)
        b_kf = b_k.to(tl.float32)

        b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

        b_g = tle.load(p_g, boundary_check=(0, 1), is_async=True).to(tl.float32)
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

    def _kda_fwd_intra(
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

        # Pad the workspace T dimension so TMA descriptors can read full BT tiles.
        T_padded = NT * BT
        g_out = torch.empty(B, T_padded, HV, K, device=q.device, dtype=torch.float32)
        ws = torch.empty(B, T_padded, HV, 3 * K, device=q.device, dtype=q.dtype)
        Aqk = torch.empty(B, T_padded, HV, BT, device=q.device, dtype=q.dtype)
        Akk = torch.zeros(B, T_padded, HV, BT, device=q.device, dtype=q.dtype)

        _kda_fwd_intra_kernel[grid](
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

    @triton.jit
    def _kda_state_output_load_producer(
        writer,
        ws_desc,
        v_ptr,
        beta_ptr,
        gk_desc,
        Aqk_desc,
        Akk_desc,
        K,
        T,
        HV: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        NT,
        i_v,
    ):
        for i_t in tl.range(NT):
            slot = writer.acquire(i_t)
            last_idx = tl.minimum(i_t * BT + BT, T) - 1

            # The workspace packs w, qg, and kg at offsets 0, K, and 2*K.
            tle.gpu.copy(ws_desc, slot.w1, [BT, 64], [i_t * BT, 0])
            tle.gpu.copy(ws_desc, slot.qg1, [BT, 64], [i_t * BT, K])
            tle.gpu.copy(ws_desc, slot.kg1, [BT, 64], [i_t * BT, 2 * K])
            tle.gpu.copy(Aqk_desc, slot.Aqk, [BT, BT], [i_t * BT, 0])
            tle.gpu.copy(Akk_desc, slot.Akk, [BT, BT], [i_t * BT, 0])
            tle.gpu.copy(gk_desc, slot.gk1, [1, 64], [last_idx, 0])
            if K > 64:
                tle.gpu.copy(ws_desc, slot.w2, [BT, 64], [i_t * BT, 64])
                tle.gpu.copy(ws_desc, slot.qg2, [BT, 64], [i_t * BT, K + 64])
                tle.gpu.copy(ws_desc, slot.kg2, [BT, 64], [i_t * BT, 2 * K + 64])
                tle.gpu.copy(gk_desc, slot.gk2, [1, 64], [last_idx, 64])
            if K > 128:
                tle.gpu.copy(ws_desc, slot.w3, [BT, 64], [i_t * BT, 128])
                tle.gpu.copy(ws_desc, slot.qg3, [BT, 64], [i_t * BT, K + 128])
                tle.gpu.copy(ws_desc, slot.kg3, [BT, 64], [i_t * BT, 2 * K + 128])
                tle.gpu.copy(gk_desc, slot.gk3, [1, 64], [last_idx, 128])
            if K > 192:
                tle.gpu.copy(ws_desc, slot.w4, [BT, 64], [i_t * BT, 192])
                tle.gpu.copy(ws_desc, slot.qg4, [BT, 64], [i_t * BT, K + 192])
                tle.gpu.copy(ws_desc, slot.kg4, [BT, 64], [i_t * BT, 2 * K + 192])
                tle.gpu.copy(gk_desc, slot.gk4, [1, 64], [last_idx, 192])

            # v and beta need elementwise work, so keep them as regular loads.
            p_v = tl.make_block_ptr(
                v_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            p_beta = tl.make_block_ptr(beta_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
            b_vb = (b_v.to(tl.float32) * b_beta_f[:, None]).to(b_v.dtype)
            tl.store(tle.gpu.local_ptr(slot.vb), b_vb)

            writer.commit(i_t)

    @triton.jit
    def _kda_state_output_mma_consumer(
        load_reader,
        store_writer,
        h0,
        ht,
        gk,
        scale,
        i_v,
        i_nh,
        T,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        NT,
        kg_dtype: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
        STATE_V_FIRST: tl.constexpr,
    ):
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

        # Sequential recurrence across chunks.
        for i_t in tl.range(NT):
            wait = load_reader.wait(i_t)
            slot = wait.slot

            b_w1 = tl.load(tle.gpu.local_ptr(slot.w1))
            b_vb = tl.load(tle.gpu.local_ptr(slot.vb))
            b_qg1 = tl.load(tle.gpu.local_ptr(slot.qg1))
            b_kg1 = tl.load(tle.gpu.local_ptr(slot.kg1))
            b_Aqk = tl.load(tle.gpu.local_ptr(slot.Aqk))
            b_Akk = tl.load(tle.gpu.local_ptr(slot.Akk))
            b_gk1 = tl.load(tle.gpu.local_ptr(slot.gk1)).reshape([64])
            if K > 64:
                b_w2 = tl.load(tle.gpu.local_ptr(slot.w2))
                b_qg2 = tl.load(tle.gpu.local_ptr(slot.qg2))
                b_kg2 = tl.load(tle.gpu.local_ptr(slot.kg2))
                b_gk2 = tl.load(tle.gpu.local_ptr(slot.gk2)).reshape([64])
            if K > 128:
                b_w3 = tl.load(tle.gpu.local_ptr(slot.w3))
                b_qg3 = tl.load(tle.gpu.local_ptr(slot.qg3))
                b_kg3 = tl.load(tle.gpu.local_ptr(slot.kg3))
                b_gk3 = tl.load(tle.gpu.local_ptr(slot.gk3)).reshape([64])
            if K > 192:
                b_w4 = tl.load(tle.gpu.local_ptr(slot.w4))
                b_qg4 = tl.load(tle.gpu.local_ptr(slot.qg4))
                b_kg4 = tl.load(tle.gpu.local_ptr(slot.kg4))
                b_gk4 = tl.load(tle.gpu.local_ptr(slot.gk4)).reshape([64])

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

            out_slot = store_writer.acquire(i_t)
            tl.store(tle.gpu.local_ptr(out_slot.output), b_o)
            store_writer.commit(i_t)

            load_reader.release(i_t)

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

    @triton.jit
    def _kda_state_output_store_consumer(
        store_reader,
        o_ptr,
        T,
        HV: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        BV: tl.constexpr,
        NT,
        i_v,
    ):
        for i_t in tl.range(NT):
            store_wait = store_reader.wait(i_t)
            slot = store_wait.slot
            b_o = tl.load(tle.gpu.local_ptr(slot.output))
            p_o = tl.make_block_ptr(
                o_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
            store_reader.release(i_t)

    PIPE_STAGES = tl.constexpr(4)

    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
            "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @triton.autotune(
        configs=[triton.Config({"BV": BV}, num_warps=4) for BV in [32, 64, 128]],
        key=[
            "HV",
            "K",
            "V",
            "BT",
        ],  # warp_specialize fixes num_warps=4, only BV is tuned
    )
    @triton.jit(do_not_specialize=["T"])
    def _kda_fwd_state_output_kernel(
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

        # TMA descriptors.
        ws_desc = tl.make_tensor_descriptor(
            ws_base, shape=[T, 3 * K], strides=[HV * 3 * K, 1], block_shape=[BT, 64]
        )
        gk_desc = tl.make_tensor_descriptor(
            gk, shape=[T, K], strides=[HV * K, 1], block_shape=[1, 64]
        )
        Aqk_desc = tl.make_tensor_descriptor(
            Aqk, shape=[T, BT], strides=[HV * BT, 1], block_shape=[BT, BT]
        )
        Akk_desc = tl.make_tensor_descriptor(
            Akk, shape=[T, BT], strides=[HV * BT, 1], block_shape=[BT, BT]
        )

        # Allocate only the K blocks needed by this specialization.
        w1_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
        )
        vb_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, BV], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
        )
        qg1_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
        )
        kg1_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
        )
        Aqk_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, BT], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
        )
        Akk_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, BT], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
        )
        gk1_smem = tle.gpu.alloc(
            [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem
        )
        if K > 64:
            w2_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            qg2_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            kg2_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            gk2_smem = tle.gpu.alloc(
                [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem
            )
        if K > 128:
            w3_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            qg3_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            kg3_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            gk3_smem = tle.gpu.alloc(
                [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem
            )
        if K > 192:
            w4_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            qg4_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            kg4_smem = tle.gpu.alloc(
                [PIPE_STAGES, BT, 64], dtype=kg.dtype.element_ty, scope=tle.gpu.smem
            )
            gk4_smem = tle.gpu.alloc(
                [PIPE_STAGES, 1, 64], dtype=tl.float32, scope=tle.gpu.smem
            )
        out_smem = tle.gpu.alloc(
            [PIPE_STAGES, BT, BV], dtype=tl.float32, scope=tle.gpu.smem
        )

        # Build pipe slots that match the allocated blocks.
        if K <= 64:
            load_pipe = tle.pipe(
                capacity=PIPE_STAGES,
                scope="cta",
                name="kda_load",
                w1=w1_smem,
                vb=vb_smem,
                qg1=qg1_smem,
                kg1=kg1_smem,
                Aqk=Aqk_smem,
                Akk=Akk_smem,
                gk1=gk1_smem,
            )
        elif K <= 128:
            load_pipe = tle.pipe(
                capacity=PIPE_STAGES,
                scope="cta",
                name="kda_load",
                w1=w1_smem,
                w2=w2_smem,
                vb=vb_smem,
                qg1=qg1_smem,
                qg2=qg2_smem,
                kg1=kg1_smem,
                kg2=kg2_smem,
                Aqk=Aqk_smem,
                Akk=Akk_smem,
                gk1=gk1_smem,
                gk2=gk2_smem,
            )
        elif K <= 192:
            load_pipe = tle.pipe(
                capacity=PIPE_STAGES,
                scope="cta",
                name="kda_load",
                w1=w1_smem,
                w2=w2_smem,
                w3=w3_smem,
                vb=vb_smem,
                qg1=qg1_smem,
                qg2=qg2_smem,
                qg3=qg3_smem,
                kg1=kg1_smem,
                kg2=kg2_smem,
                kg3=kg3_smem,
                Aqk=Aqk_smem,
                Akk=Akk_smem,
                gk1=gk1_smem,
                gk2=gk2_smem,
                gk3=gk3_smem,
            )
        else:
            load_pipe = tle.pipe(
                capacity=PIPE_STAGES,
                scope="cta",
                name="kda_load",
                w1=w1_smem,
                w2=w2_smem,
                w3=w3_smem,
                w4=w4_smem,
                vb=vb_smem,
                qg1=qg1_smem,
                qg2=qg2_smem,
                qg3=qg3_smem,
                qg4=qg4_smem,
                kg1=kg1_smem,
                kg2=kg2_smem,
                kg3=kg3_smem,
                kg4=kg4_smem,
                Aqk=Aqk_smem,
                Akk=Akk_smem,
                gk1=gk1_smem,
                gk2=gk2_smem,
                gk3=gk3_smem,
                gk4=gk4_smem,
            )
        store_pipe = tle.pipe(
            capacity=PIPE_STAGES,
            scope="cta",
            name="kda_store",
            output=out_smem,
        )
        tle.gpu.warp_specialize(
            [
                (
                    _kda_state_output_load_producer,
                    (
                        load_pipe.writer(),
                        ws_desc,
                        v,
                        beta,
                        gk_desc,
                        Aqk_desc,
                        Akk_desc,
                        K,
                        T,
                        HV,
                        V,
                        BT,
                        BV,
                        NT,
                        i_v,
                    ),
                ),
                (
                    _kda_state_output_mma_consumer,
                    (
                        load_pipe.reader(),
                        store_pipe.writer(),
                        h0,
                        ht,
                        gk,
                        scale,
                        i_v,
                        i_nh,
                        T,
                        HV,
                        K,
                        V,
                        BT,
                        BV,
                        NT,
                        kg.dtype.element_ty,
                        USE_INITIAL_STATE,
                        STORE_FINAL_STATE,
                        STATE_V_FIRST,
                    ),
                ),
                (
                    _kda_state_output_store_consumer,
                    (store_pipe.reader(), o, T, HV, V, BT, BV, NT, i_v),
                ),
            ],
            [4, 1],
            [240, 32],
        )

    def _kda_fwd_state_output(
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
        _kda_fwd_state_output_kernel[grid](
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


def chunk_kda_fwd_infer(
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
    _refresh_fp16_dot_precision()

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    ws, Aqk, Akk, g_cumsum = _kda_fwd_intra(
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
    return _kda_fwd_state_output(
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
# Triton fallback path (portable, used when TLE is unavailable)
# =============================================================================


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
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [16, 32, 64]
        for BV in [16, 32, 64]
        for num_warps in [1, 2, 4]
        for num_stages in [1, 2, 4]
    ],
    key=["H", "HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_intra_triton_kernel(
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
    Akk,
    g_out,
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
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
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

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    g_out += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    w += (bos * HV + i_hv) * K
    u += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    if STORE_QG:
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_i = tl.arange(0, BT)
    o_c = i_t * BT + o_i
    m_c = o_c < T

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

        p_g_out = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        tl.store(p_g_out, b_g.to(g_out.dtype.element_ty), boundary_check=(0, 1))

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

    # Phase 2: Solve (I + L)^{-1} via parallel prefix
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

    # Phase 3: w, u, qg, kg
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
        p_gk = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
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
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            last_idx = tl.minimum(i_t * BT + BT, T) - 1
            b_gn = tl.load(g_out + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )
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


def _kda_fwd_intra_triton(
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
    """Fused intra-chunk computation. Returns (w, u, qg, kg, Aqk, Akk, g_cumsum)."""
    B, T_len, H, K = q.shape
    HV = g.shape[2]
    V = v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HV)

    g_out = torch.empty(B, T_len, HV, K, device=q.device, dtype=torch.float32)
    w = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    u = torch.empty(B, T_len, HV, V, device=q.device, dtype=q.dtype)
    qg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    kg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    Aqk = torch.empty(B, T_len, HV, BT, device=q.device, dtype=q.dtype)
    Akk = torch.zeros(B, T_len, HV, BT, device=q.device, dtype=q.dtype)

    _kda_fwd_intra_triton_kernel[grid](
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
        Akk=Akk,
        g_out=g_out,
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
    return w, u, qg, kg, Aqk, Akk, g_out


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
        for num_warps in [2, 4]
    ],
    key=["HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_h_o_triton_kernel(
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
    else:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    kg += (bos * HV + i_h).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    u += (bos * HV + i_h).to(tl.int64) * V
    gk += (bos * HV + i_h).to(tl.int64) * K
    qg += (bos * HV + i_h).to(tl.int64) * K
    Aqk += (bos * HV + i_h).to(tl.int64) * BT
    o += (bos * HV + i_h).to(tl.int64) * V

    if STATE_V_FIRST:
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
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # v_new = u - w @ h
        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_u = tl.make_block_ptr(
            u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

        # output = scale * qg @ h + Aqk @ v_new
        p_qg = tl.make_block_ptr(
            qg, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg, tl.trans(b_h1).to(b_qg.dtype))
        else:
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype))
        if K > 64:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h2).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h2.to(b_qg.dtype))
        if K > 128:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h3).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h3.to(b_qg.dtype))
        if K > 192:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h4).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h4.to(b_qg.dtype))
        b_o *= scale

        p_Aqk = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
        b_o += tl.dot(b_Aqk.to(b_v.dtype), b_v)

        p_o = tl.make_block_ptr(
            o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # decay: h *= exp2(gk_last)
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(
            gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.0
        ).to(tl.float32)
        if STATE_V_FIRST:
            b_h1 *= exp2(b_gk_last1)[None, :]
        else:
            b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            o_k2 = 64 + o_k1
            b_gk_last2 = tl.load(
                gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h2 *= exp2(b_gk_last2)[None, :]
            else:
                b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            o_k3 = 128 + o_k1
            b_gk_last3 = tl.load(
                gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h3 *= exp2(b_gk_last3)[None, :]
            else:
                b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            o_k4 = 192 + o_k1
            b_gk_last4 = tl.load(
                gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h4 *= exp2(b_gk_last4)[None, :]
            else:
                b_h4 *= exp2(b_gk_last4)[:, None]

        # state update: h += kg^T @ v_new
        b_v = b_v.to(kg.dtype.element_ty)
        p_kg = tl.make_block_ptr(
            kg, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_kg, b_v))
        else:
            b_h1 += tl.dot(b_kg, b_v)
        if K > 64:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h2 += tl.dot(b_kg, b_v)
        if K > 128:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h3 += tl.dot(b_kg, b_v)
        if K > 192:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
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
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _kda_fwd_h_o_triton(
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
    """Fused state propagation + output."""
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
            final_state = kg.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = kg.new_zeros(N, HV, K, V, dtype=torch.float32)

    o = torch.zeros(B, T, HV, V, device=kg.device, dtype=u.dtype)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    _kda_fwd_h_o_triton_kernel[grid](
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
    )
    return o, final_state


def chunk_kda_fwd_infer_triton(
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
    """Plain-Triton inference forward for chunk KDA (TLE-free fallback)."""
    BT = chunk_size

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    w, u, qg, kg, Aqk, Akk, g_cumsum = _kda_fwd_intra_triton(
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

    return _kda_fwd_h_o_triton(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
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


# =============================================================================
# Input validation (shared by both paths)
# =============================================================================


def _validate_chunk_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
    state_v_first: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    safe_gate: bool,
    lower_bound: float | None,
) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError("chunk_kda TLE path only supports inference/no-grad mode")
    if chunk_size != 16:
        raise ValueError(f"chunk_kda TLE path requires chunk_size=16, got {chunk_size}")
    supported_dtypes = (torch.bfloat16, torch.float16)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype not in supported_dtypes:
            raise ValueError(
                f"chunk_kda TLE path requires {name} dtype to be bf16 or fp16, "
                f"got {tensor.dtype}"
            )
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4:
        raise ValueError("q, k, v, and g must be 4D tensors in [B, T, H, D] layout")
    if beta.ndim != 3:
        raise ValueError("beta must be a 3D tensor in [B, T, HV] layout")

    B, T, H, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, HV, V = v.shape
    if (Bk, Tk, Hk, Kk) != (B, T, H, K):
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if (Bv, Tv) != (B, T):
        raise ValueError("v must share B and T dimensions with q/k")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"g must have shape {(B, T, HV, K)}, got {tuple(g.shape)}")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}")
    if K not in (64, 128, 192, 256):
        raise ValueError(
            f"chunk_kda TLE path requires K in {{64, 128, 192, 256}}, got {K}"
        )
    if V <= 0:
        raise ValueError(f"chunk_kda TLE path requires V > 0, got {V}")
    if HV < H or HV % H != 0:
        raise ValueError(f"chunk_kda TLE path requires HV % H == 0, got H={H}, HV={HV}")

    if not use_qk_l2norm_in_kernel:
        raise ValueError("chunk_kda TLE path requires use_qk_l2norm_in_kernel=True")
    if not use_gate_in_kernel:
        raise ValueError("chunk_kda TLE path requires use_gate_in_kernel=True")
    if not use_beta_sigmoid_in_kernel:
        raise ValueError("chunk_kda TLE path requires use_beta_sigmoid_in_kernel=True")
    if allow_neg_eigval:
        raise ValueError("chunk_kda TLE path does not support allow_neg_eigval=True")
    if not safe_gate:
        raise ValueError("chunk_kda TLE path requires safe_gate=True")
    if lower_bound is None:
        raise ValueError("chunk_kda TLE path requires lower_bound")
    if A_log is None:
        raise ValueError("chunk_kda TLE path requires A_log")
    if A_log.device != q.device:
        raise ValueError("A_log must be on the same device as q")
    if A_log.numel() != HV:
        raise ValueError(f"A_log.numel() must be HV={HV}, got {A_log.numel()}")
    if dt_bias is None:
        raise ValueError("chunk_kda TLE path requires dt_bias")
    if dt_bias.device != q.device:
        raise ValueError("dt_bias must be on the same device as q")
    if dt_bias.numel() != HV * K:
        raise ValueError(
            f"dt_bias.numel() must be HV*K={HV * K}, got {dt_bias.numel()}"
        )

    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if B != 1:
            raise ValueError("cu_seqlens packed varlen inputs must use B=1")

    if initial_state is not None:
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
        expected_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        if tuple(initial_state.shape) != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, "
                f"got {tuple(initial_state.shape)}"
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
    computed inside the kernels. When the Triton TLE extension is available the
    fused TLE path is used; otherwise it falls back to a plain-Triton path with
    identical semantics.
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

    if HAS_TLE_KDA:
        return chunk_kda_fwd_infer(
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
