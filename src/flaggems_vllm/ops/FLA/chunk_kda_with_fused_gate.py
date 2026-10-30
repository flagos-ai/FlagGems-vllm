# SPDX-License-Identifier: Apache-2.0
# Triton stages adapted from vLLM 0.24.0 FLA kernels (Apache-2.0).
# The original FLA portions retain the following MIT license notice:
# MIT License
#
# Copyright (c) 2023-2025 Songlin Yang, Yu Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Inference KDA prefill with softplus gate and fixed 64-token chunks.

The public ABI follows vLLM 0.24.0, including V-first FP32 final states and
the output alias of contiguous v. All arithmetic and copies use Triton."""

import math
import statistics

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.data_movement import contiguous_copy, copy_to, fill
from flaggems_vllm.ops.FLA.triton_ops_helper import exp, exp2, log
from flaggems_vllm.utils import libentry, libtuner


def _stage_bench(fn, quantiles):
    """Tune with device events, without any backend Graph benchmark helper."""
    api = runtime.torch_device_fn
    fn()
    api.synchronize()
    samples = []
    for _ in range(5):
        start, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    ordered = sorted(samples)
    return (
        [ordered[round(q * (len(ordered) - 1))] for q in quantiles]
        if quantiles
        else statistics.median(samples)
    )


@libentry()
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@libtuner(
    configs=runtime.get_tuned_config("ckfg_inter"),
    key=["H", "K", "BC", "IS_VARLEN"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_inter(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
    i_i, i_j = (i_c // NC, i_c % NC)
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
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    Aqk += (bos * H + i_h) * BT
    p_b = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,)
    )
    b_b = tl.load(p_b, boundary_check=(0,), padding_option="zero")
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0)
        )
        b_kt = tl.make_block_ptr(
            k, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )
        p_gk = tl.make_block_ptr(
            g, (K, T), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1)
        )
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero") * exp2(
            b_g - b_gn[None, :]
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1), padding_option="zero")
        b_kt = tl.load(b_kt, boundary_check=(0, 1), padding_option="zero")
        b_ktg = b_kt * exp2(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_ktg)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_qg = b_q * exp2(b_g - b_gn[None, :]) * scale
        b_Aqk += tl.dot(b_qg, b_ktg)
    b_A *= b_b[:, None]
    p_A = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@libtuner(
    configs=runtime.get_tuned_config("ckfg_intra"),
    key=["BK", "BT"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_intra(
    q,
    k,
    g,
    beta,
    A,
    Aqk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
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
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + o_i < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC
    p_q = tl.make_block_ptr(
        q + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_t * BT + i_i * BC, 0),
        (BC, BK),
        (1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = b_k * tl.load(p_b, mask=m_A, other=0)[:, None]
    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_ktg = b_kt[None, :] * exp2(b_g - b_gk[None, :])
        b_A = tl.sum(b_k * b_ktg, 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        b_Aqk = tl.sum(b_q * b_ktg, 1)
        b_Aqk = tl.where(o_i >= j, b_Aqk * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        tl.store(Aqk + o_A + j, b_Aqk, mask=m_A)
        p_kt += H * K
        p_gk += H * K


@libentry()
@triton.heuristics(
    {
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config("ckfg_recompute"),
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_recompute(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
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
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT >= T:
        return
    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,), padding_option="zero")
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_kb = b_k * b_b[:, None]
        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_gk = tl.load(p_gk, boundary_check=(0, 1), padding_option="zero")
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_qg = tl.make_block_ptr(
                qg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(
                gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.0
            )
            b_kg = b_k * exp2(b_gn - b_gk)
            p_kg = tl.make_block_ptr(
                kg + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@libtuner(
    configs=runtime.get_tuned_config("ckfg_output"),
    key=["BT"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_output(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
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
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT >= T:
        return
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_g = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + (i_tg * H + i_h) * K * V,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        if i_k >= 0:
            b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config("ckfg_gate"),
    key=["H", "D", "BT", "IS_VARLEN"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_gate(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    beta,
    threshold,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_b, i_h = (i_bh // H, i_bh % H)
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
        bos = i_b * T
    if i_t * BT >= T:
        return
    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_y = tl.make_block_ptr(
        y + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    if HAS_BIAS:
        o_d = i_d * BD + tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]
    b_a = -tl.exp(tl.load(A + i_h).to(tl.float32))
    b_g_scaled = b_g * beta
    b_softplus = tl.where(
        b_g_scaled > threshold, b_g, 1.0 / beta * log(1.0 + tl.exp(b_g_scaled))
    )
    b_gate = b_a * b_softplus
    o_t = tl.arange(0, BT)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@libentry()
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@libtuner(
    configs=runtime.get_tuned_config("ckfg_inverse"),
    key=["H", "BT", "IS_VARLEN"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_inverse(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = (tl.program_id(0), tl.program_id(1))
    i_b, i_h = (i_bh // H, i_bh % H)
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
        bos, eos = (i_b * T, i_b * T + T)
    if i_t * BT >= T:
        return
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT
    p_A_11 = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
    p_A_22 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_A_33 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_A_44 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)
    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H * BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a_33 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 32)
        b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a_44 = -tl.load(A + (i_t * BT + i) * H * BT + o_i + 48)
        b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
    b_Ai_11 += m_I
    b_Ai_22 += m_I
    b_Ai_33 += m_I
    b_Ai_44 += m_I
    p_A_21 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_A_31 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_A_32 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_A_41 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_A_42 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_A_43 = tl.make_block_ptr(
        A, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
    )
    b_A_21 = tl.load(p_A_21, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_A_31 = tl.load(p_A_31, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_A_32 = tl.load(p_A_32, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_A_41 = tl.load(p_A_41, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_A_42 = tl.load(p_A_42, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    b_A_43 = tl.load(p_A_43, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
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
        Ai, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_33 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p_Ai_44 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_31 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
    )
    p_Ai_32 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
    )
    p_Ai_41 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
    )
    p_Ai_42 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
    )
    p_Ai_43 = tl.make_block_ptr(
        Ai, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
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
    configs=runtime.get_tuned_config("ckfg_state"),
    key=["H", "K", "V", "BT"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit(do_not_specialize=["T"])
def _ckfg_state(
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
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = (tl.program_id(0), tl.program_id(1))
    i_n, i_h = (i_nh // H, i_nh % H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = (i_n * T, i_n * T + T)
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
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
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t.to(tl.int64) * stride_h,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, 64),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 64),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 128),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t.to(tl.int64) * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 192),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero") - b_v
        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
        if USE_G:
            m_t = i_t.to(tl.int64) * BT + tl.arange(0, BT) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,), padding_option="zero")
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
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
                gk + (bos + last_idx) * H * K + i_h * K + o_k1, mask=o_k1 < K, other=0.0
            )
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=o_k2 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=o_k3 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=o_k4 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp(b_gk_last4)[None, :]
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
            b_h4 += tl.trans(tl.dot(b_k, b_v))
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


_CHUNK = 64
_SUBCHUNK = 16
_RCP_LN2 = 1.4426950408889634
_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ckfg_norm"),
    key=["N", "K"],
    use_cuda_graph=False,
    do_bench=_stage_bench,
)
@triton.jit
def _norm_kernel(
    X, Y, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, BLOCK)
    mask = (row[:, None] < N) & (d[None, :] < K)
    x = tl.load(X + row[:, None] * K + d[None, :], mask, other=0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 1) + 1e-6)[:, None]
    tl.store(Y + row[:, None] * K + d[None, :], y, mask)


@libentry()
@triton.jit
def _metadata_init(
    CU,
    OFF,
    IDX,
    N: tl.constexpr,
    NT: tl.constexpr,
    SENTINEL: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
):
    n = tl.arange(0, BN)
    start = tl.load(CU + n, n < N, other=0)
    end = tl.load(CU + n + 1, n < N, other=0)
    count = tl.where(n < N, tl.cdiv(end - start, 64), 0)
    prefix = tl.cumsum(count, 0)
    tl.store(OFF + n + 1, prefix, n < N)
    tl.store(OFF, 0)
    t = tl.arange(0, BT)
    tl.store(IDX + 2 * t, 0, t < NT)
    tl.store(IDX + 2 * t + 1, SENTINEL, t < NT)


@libentry()
@triton.jit
def _metadata_expand(CU, OFF, IDX):
    n = tl.program_id(0)
    start = tl.load(CU + n)
    end = tl.load(CU + n + 1)
    offset = tl.load(OFF + n)
    for j in range(tl.cdiv(end - start, 64)):
        tl.store(IDX + (offset + j) * 2, n)
        tl.store(IDX + (offset + j) * 2 + 1, j)


def _metadata(cu, t, device):
    n = cu.numel() - 1
    nt = triton.cdiv(t, _CHUNK) + n
    indices = torch.empty((nt, 2), device=device, dtype=cu.dtype)
    offsets = torch.empty((n + 1,), device=device, dtype=cu.dtype)
    _metadata_init[(1,)](
        cu,
        offsets,
        indices,
        n,
        nt,
        triton.cdiv(t, _CHUNK) + 1,
        triton.next_power_of_2(max(n, 1)),
        triton.next_power_of_2(max(nt, 1)),
        num_warps=4,
    )
    _metadata_expand[(n,)](cu, offsets, indices, num_warps=1)
    return indices, offsets


def _validate(q, k, v, raw_g, beta, a, bias, state, cu, scale):
    if any(x.ndim != 4 for x in (q, k, v, raw_g)):
        raise ValueError("q/k/v/raw_g must be four-dimensional seq-first tensors")
    b, t, h, dk = q.shape
    if k.shape != q.shape or raw_g.shape != q.shape:
        raise ValueError("q/k/raw_g must have the same [B,T,H,K] shape")
    if v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("v/beta must share the B/T/H dimensions of q")
    if not 0 < dk <= 256 or h < 1 or v.shape[-1] < 1:
        raise NotImplementedError("KDA requires 1<=K<=256, H>0 and V>0")
    if q.dtype not in _FLOATS[:2] or k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError("q/k/v must have the same float16 or bfloat16 dtype")
    tensors = (q, k, v, raw_g, beta, a, bias, state)
    for x in tensors:
        if x is None:
            continue
        if x.device != q.device:
            raise ValueError("KDA inputs must be on one device")
        if x.dtype not in _FLOATS:
            raise NotImplementedError(
                "gate, beta, parameters and state must be floating tensors"
            )
        if x.requires_grad:
            raise NotImplementedError("chunk_kda_with_fused_gate is inference-only")
    if a.numel() != h or (bias is not None and bias.numel() != h * dk):
        raise ValueError("A_log/bias must contain H/H*K elements")
    if cu is not None:
        if b != 1 or cu.ndim != 1 or cu.numel() < 1:
            raise ValueError(
                "packed varlen requires B=1 and a one-dimensional cu_seqlens"
            )
        if cu.dtype not in (torch.int32, torch.int64) or cu.device != q.device:
            raise ValueError("cu_seqlens must be device int32/int64")
    n = b if cu is None else cu.numel() - 1
    if n == 0 and t != 0:
        raise ValueError("nonempty tokens require at least one sequence")
    if state is not None and state.shape != (n, h, v.shape[-1], dk):
        raise ValueError("initial_state must have shape [N,H,V,K]")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return n


def chunk_kda_with_fused_gate(
    q,
    k,
    v,
    raw_g,
    beta,
    A_log,
    g_bias,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **kwargs,
):
    """Return output and optional [N,H,V,K] FP32 final state.

    Beta is already activated. Packed cu_seqlens must be monotone, start at 0
    and end at T. Like vLLM 0.24.0, extra kwargs do not select a safe gate.
    """
    n = _validate(q, k, v, raw_g, beta, A_log, g_bias, initial_state, cu_seqlens, scale)
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    scale = dk ** (-0.5) if scale is None else scale
    q, k, v, raw_g, beta = map(contiguous_copy, (q, k, v, raw_g, beta))
    a = contiguous_copy(A_log).view(-1)
    bias = contiguous_copy(g_bias).view(-1) if g_bias is not None else None
    state = contiguous_copy(initial_state) if initial_state is not None else None
    final = (
        torch.empty((n, h, dv, dk), dtype=torch.float32, device=q.device)
        if output_final_state
        else None
    )
    if t == 0 or b == 0:
        if final is not None:
            if state is None:
                fill(final, 0)
            else:
                copy_to(contiguous_copy(state, dtype=torch.float32), final)
        return (v, final)
    if use_qk_l2norm_in_kernel:
        qn, kn = (torch.empty_like(q), torch.empty_like(k))
        for x, y in ((q, qn), (k, kn)):
            _norm_kernel[lambda meta: (triton.cdiv(b * t * h, meta["ROWS"]),)](
                x, y, b * t * h, dk, triton.next_power_of_2(dk)
            )
        q, k = (qn, kn)
    cu = contiguous_copy(cu_seqlens) if cu_seqlens is not None else None
    if cu is None:
        indices, offsets = (None, None)
        nt = triton.cdiv(t, _CHUNK)
    else:
        indices, offsets = _metadata(cu, t, q.device)
        nt = indices.shape[0]
    gate = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    _ckfg_gate[lambda meta: (triton.cdiv(dk, meta["BD"]), nt, b * h)](
        g=raw_g,
        A=a,
        y=gate,
        g_bias=bias,
        cu_seqlens=cu,
        chunk_indices=indices,
        cumsum_scale=_RCP_LN2,
        beta=1.0,
        threshold=20.0,
        T=t,
        H=h,
        D=dk,
        BT=_CHUNK,
    )
    matrix_shape = (b, t, h, _CHUNK)
    akk = fill(torch.empty(matrix_shape, dtype=torch.float32, device=q.device), 0)
    aqk = fill(torch.empty_like(akk), 0)
    common = dict(
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=akk,
        Aqk=aqk,
        scale=scale,
        cu_seqlens=cu,
        chunk_indices=indices,
        T=t,
        H=h,
        K=dk,
        BT=_CHUNK,
        BC=_SUBCHUNK,
    )
    _ckfg_inter[nt, 16, b * h](**common, NC=4)
    _ckfg_intra[nt, 4, b * h](**common, BK=max(triton.next_power_of_2(dk), 16))
    inverse = fill(torch.empty(matrix_shape, dtype=k.dtype, device=q.device), 0)
    _ckfg_inverse[nt, b * h](
        A=akk,
        Ai=inverse,
        cu_seqlens=cu,
        chunk_indices=indices,
        T=t,
        H=h,
        BT=_CHUNK,
        USE_TMA=False,
        DOT_PRECISION="ieee",
    )
    w, u, kg = (torch.empty_like(k), torch.empty_like(v), torch.empty_like(k))
    _ckfg_recompute[nt, b * h](
        q=None,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=inverse,
        gk=gate,
        cu_seqlens=cu,
        chunk_indices=indices,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
        BK=64,
        BV=64,
        DOT_PRECISION="ieee",
    )
    history = torch.empty((b, nt, h, dv, dk), dtype=k.dtype, device=q.device)
    v_new = torch.empty_like(u)
    state_args = dict(
        k=kg,
        v=u,
        w=w,
        v_new=v_new,
        gk=gate,
        h=history,
        h0=state,
        ht=final,
        cu_seqlens=cu,
        chunk_offsets=offsets,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
    )
    state_grid = lambda meta: (triton.cdiv(dv, meta["BV"]), n * h)
    _ckfg_state[state_grid](**state_args, g=None, Hg=h, USE_EXP2=True)
    _ckfg_output[lambda meta: (triton.cdiv(dv, meta["BV"]), nt, b * h)](
        q=q,
        v=v_new,
        g=gate,
        h=history,
        o=v,
        A=aqk,
        cu_seqlens=cu,
        chunk_indices=indices,
        scale=scale,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=_CHUNK,
    )
    return (v, final)
