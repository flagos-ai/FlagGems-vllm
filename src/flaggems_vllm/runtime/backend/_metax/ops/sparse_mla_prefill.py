# Copyright 2026 FlagOS Contributors

# SPDX-License-Identifier: Apache-2.0
"MetaX DeepSeek V4 sparse prefill kernels, replay and TP headpacking."

import importlib
import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._metax.ops import sparse_mla_policy as _tuning
from flaggems_vllm.runtime.backend._metax.ops.sparse_mla_policy import local_launch

logger = logging.getLogger(__name__)


@triton.jit
def qk_states(
    Q,
    KV,
    I,
    SINK,
    LENS,
    P,
    A,
    F,
    MAX,
    LSE,
    SQ,
    SKV,
    TOPK: tl.constexpr,
    NC: tl.constexpr,
    SINK_ON: tl.constexpr,
    LEN_ON: tl.constexpr,
    SCALE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, 16)
    d = tl.arange(0, 256)
    k = tl.arange(0, 16)
    qp = Q + t * 16 * 512 + h[:, None] * 512 + d[None, :]
    q0 = tl.load(qp, eviction_policy="evict_first")
    q1 = tl.load(qp + 256, eviction_policy="evict_first")
    max_log = tl.full([16], -float("inf"), tl.float32)
    sum_exp = tl.full([16], 0.0, tl.float32)
    topk_len = tl.load(LENS + t) if LEN_ON else TOPK
    for ck in range(tl.cdiv(topk_len, 16)):
        ki = ck * 16 + k
        ids = tl.load(I + t * TOPK + ki, ki < topk_len, other=-1)
        valid = (ids < SKV) & (ids >= 0)
        ids = tl.where(valid, ids, 0)
        kp = KV + d[:, None] + ids[None, :] * 512
        kv0 = tl.load(kp, cache_modifier=".cg")
        kv1 = tl.load(kp + 256, cache_modifier=".cg")
        qk = tl.dot(q0, kv0, out_dtype=tl.float32)
        qk = tl.dot(q1, kv1, qk, out_dtype=tl.float32)
        qk *= SCALE
        qk = tl.where(valid[None, :], qk, -float("inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk
        tl.store(
            P + ((t * NC + ck) * 16 + h[:, None]) * 16 + k[None, :],
            exp_qk.to(tl.bfloat16),
        )
        tl.store(A + (t * NC + ck) * 16 + h, alpha)
        max_log = new_max
    valid_mask = max_log != -float("inf")
    max_log = tl.where(valid_mask, max_log, -float("inf"))
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    if SINK_ON:
        sink = tl.load(SINK + h)
        factor = tl.math.exp(max_log) / (tl.math.exp(orig_lse) + tl.math.exp(sink))
    else:
        factor = 1.0 / sum_exp
    tl.store(F + t * 16 + h, factor)
    tl.store(MAX + t * 16 + h, max_log)
    tl.store(LSE + t * 16 + h, lse_out)


@triton.jit
def pv_replay(
    KV,
    I,
    LENS,
    P,
    A,
    F,
    MAX,
    OUT,
    SQ,
    SKV,
    TOPK: tl.constexpr,
    NC: tl.constexpr,
    LEN_ON: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    vs = tl.program_id(1) * 256
    h = tl.arange(0, 16)
    d = tl.arange(0, 256)
    k = tl.arange(0, 16)
    acc = tl.zeros([16, 256], tl.float32)
    topk_len = tl.load(LENS + t) if LEN_ON else TOPK
    for ck in range(tl.cdiv(topk_len, 16)):
        ki = ck * 16 + k
        ids = tl.load(I + t * TOPK + ki, ki < topk_len, other=-1)
        valid = (ids < SKV) & (ids >= 0)
        ids = tl.where(valid, ids, 0)
        vp = KV + (vs + d[:, None]) + ids[None, :] * 512
        v = tl.load(vp, cache_modifier=".cg")
        p = tl.load(P + ((t * NC + ck) * 16 + h[:, None]) * 16 + k[None, :])
        a = tl.load(A + (t * NC + ck) * 16 + h)
        acc = tl.dot(p, v.trans(), acc * a[:, None], out_dtype=tl.float32)
    factor = tl.load(F + t * 16 + h)
    valid_out = tl.load(MAX + t * 16 + h) != -float("inf")
    out = tl.where(valid_out[:, None], acc * factor[:, None], 0.0)
    tl.store(
        OUT + t * 16 * 512 + h[:, None] * 512 + (vs + d[None, :]), out.to(tl.bfloat16)
    )


def buffers(q, topk):
    t = q.shape[0]
    nc = triton.cdiv(topk, 16)
    return (
        torch.empty((t, nc, 16, 16), device=q.device, dtype=torch.bfloat16),
        torch.empty((t, nc, 16), device=q.device, dtype=torch.float32),
        torch.empty((t, 16), device=q.device, dtype=torch.float32),
    )


def run(q, kv, idx, sink, lens, out, ml, lse, work, qw=2, pw=2, sm_scale=512**-0.5):
    if not (q.shape[1:] == (16, 512) and q.is_contiguous()):
        raise ValueError(
            "Invalid operator argument: q.shape[1:] == (16, 512) and q.is_contiguous()"
        )
    if not (kv.shape[1:] == (1, 512) and kv.is_contiguous()):
        raise ValueError(
            "Invalid operator argument: kv.shape[1:] == (1, 512) and kv.is_contiguous()"
        )
    if not (idx.is_contiguous() and out.is_contiguous()):
        raise ValueError(
            "Invalid operator argument: idx.is_contiguous() and out.is_contiguous()"
        )
    t = q.shape[0]
    n = idx.shape[-1]
    nc = triton.cdiv(n, 16)
    p, a, f = work
    k1 = qk_states[(t,)](
        q,
        kv,
        idx,
        sink,
        lens,
        p,
        a,
        f,
        ml,
        lse,
        t,
        kv.shape[0],
        n,
        nc,
        sink is not None,
        lens is not None,
        sm_scale,
        num_warps=qw,
        num_stages=1,
    )
    k2 = pv_replay[(t, 2)](
        kv,
        idx,
        lens,
        p,
        a,
        f,
        ml,
        out,
        t,
        kv.shape[0],
        n,
        nc,
        lens is not None,
        num_warps=pw,
        num_stages=1,
    )
    return [k1, k2]


_native = importlib.import_module("flaggems_vllm.ops.flashmla_sparse")

TLE_FLASHMLA_PREFILL_BH = 16

TLE_FLASHMLA_PREFILL_BK = 16

_can_use_tle_flash_mla_sparse_fwd = _native._can_use_tle_flash_mla_sparse_fwd

_set_triton_descriptor_allocator = _native._set_triton_descriptor_allocator

_tle_flashmla_prefill_fwd = getattr(_native, "_tle_flashmla_prefill_fwd", None)

TLE_FLASHMLA_PREFILL_PAIR_BLOCKS = _native.TLE_FLASHMLA_PREFILL_PAIR_BLOCKS

TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS = _native.TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS

triton_flash_mla_sparse_fwd = local_launch(
    _native.triton_flash_mla_sparse_fwd,
    {"BK": 16, "BH": 16, "num_warps": 2, "num_stages": 2},
)


@triton.jit
def triton_flash_mla_sparse_fwd_halfd_serial(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    V_START: tl.constexpr,
    WRITE_STATS: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc = tl.zeros([BH, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_idx = BK * ck + offs_t
        t_msk = t_idx < topk_len
        kv_ids = tl.load(t_base + t_idx, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk

        # For exact 256/256 split, value dimension is the same physical bf16 K tile.
        # Reusing kv_blk0/kv_blk1 avoids an extra [256,BK] live tile.
        val_blk = kv_blk0 if V_START == 0 else kv_blk1
        acc = tl.dot(
            exp_qk.to(tl.bfloat16),
            val_blk.trans(),
            acc * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    if WRITE_STATS:
        tl.store(max_log_base + offs_h, max_log)
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    if WRITE_STATS:
        tl.store(l_base + offs_h, lse_out)
    if HAVE_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + offs_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp
    out_vals = tl.where(valid_mask[:, None], acc * factor[:, None], 0.0)
    o_ptr = o_base + offs_h[:, None] * stride_oh + (V_START + offs_d)[None, :]
    tl.store(o_ptr, out_vals.to(tl.bfloat16))


@triton.jit
def triton_flash_mla_sparse_fwd_halfd_qkfold_blockdiag(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    real_h = offs_h % 8
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    q_ptr = q_base + real_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + real_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc = tl.zeros([BH, BDP], dtype=tl.float32)
    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        t_idx = BK * ck + offs_t
        t_msk = t_idx < topk_len
        kv_ids = tl.load(t_base + t_idx, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        kv_ptr = kv_base + offs_d[:, None] + kv_ids[None, :] * stride_kvn
        kv_blk0 = tl.load(kv_ptr, cache_modifier=".cg")
        kv_blk1 = tl.load(kv_ptr + BDP, cache_modifier=".cg")
        qk = tl.dot(q_blk0, kv_blk0, out_dtype=tl.float32)
        qk = tl.dot(q_blk1, kv_blk1, qk, out_dtype=tl.float32)
        if DQK == 576:
            tkv_ptr = tkv_base + offs_td[:, None] + kv_ids[None, :] * stride_kvn
            tkv_blk = tl.load(tkv_ptr, cache_modifier=".cg")
            qk = tl.dot(tq_blk, tkv_blk, qk, out_dtype=tl.float32)
        qk *= sm_scale
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))
        exp_qk = tl.math.exp(qk - new_max[:, None])
        sum_qk = tl.sum(exp_qk, axis=1)
        alpha = tl.math.exp(max_log - new_max)
        sum_exp = sum_exp * alpha + sum_qk

        prob = exp_qk.to(tl.bfloat16)
        zero_prob = tl.zeros([BH, BK], dtype=tl.bfloat16)
        prob_lo = tl.where(offs_h[:, None] < 8, prob, zero_prob)
        prob_hi = tl.where(offs_h[:, None] >= 8, prob, zero_prob)
        prob_pair = tl.reshape(tl.join(prob_lo, prob_hi), [BH, BK * 2])
        kv_pair = tl.reshape(tl.join(kv_blk0, kv_blk1), [BDP, BK * 2])
        acc = tl.dot(
            prob_pair,
            kv_pair.trans(),
            acc * alpha[:, None],
            out_dtype=tl.float32,
        )
        max_log = new_max

    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    tl.store(max_log_base + offs_h, max_log)
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)
    if HAVE_ATTN_SINK:
        # High virtual rows carry V[256:512] for the same real head and must
        # use the same sink normalization as rows 0..7.
        sink = tl.load(attn_sink_ptr + real_h)
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp
    out_vals = tl.where(valid_mask[:, None], acc * factor[:, None], 0.0)
    v_half = (offs_h >= 8).to(tl.int64) * BDP
    o_ptr = o_base + real_h[:, None] * stride_oh + v_half[:, None] + offs_d[None, :]
    tl.store(o_ptr, out_vals.to(tl.bfloat16))


def _flash_mla_sparse_halfd_serial_enabled() -> bool:
    return _tuning.DEFAULT_POLICY.prefill_half_d


def _flash_mla_sparse_qkfold_blockdiag_enabled() -> bool:
    return _tuning.DEFAULT_POLICY.prefill_blockdiag


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    max_logits_out: Optional[torch.Tensor] = None,
    lse_out: Optional[torch.Tensor] = None,
    real_num_heads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    logger.debug("GEMS_METAX FLASH_MLA_SPARSE_FWD")
    # MetaX local-headpack extension: 2-D single-KV-head indices and explicit
    # output buffers preserve the deployed staged/serial arithmetic. Keep the
    # common 3-D indices path below unchanged.
    if indices.ndim == 2:
        _validate_local_prefill(
            q,
            kv,
            indices,
            d_v,
            attn_sink,
            topk_length,
            out,
            max_logits_out,
            lse_out,
            real_num_heads,
        )
        return _sparse_prefill_local(
            q,
            kv,
            indices.unsqueeze(1),
            sm_scale,
            attn_sink,
            topk_length,
            out,
            max_logits_out,
            lse_out,
        )
    if not (q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()):
        raise ValueError(
            "Invalid operator argument: q.is_contiguous() and kv.is_contiguous() and "
            "indices.is_contiguous()"
        )
    if not (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and (indices.dtype == torch.int32)
    ):
        raise ValueError(
            "Invalid operator argument: q.dtype == torch.bfloat16 and kv.dtype == "
            "torch.bfloat16 and (indices.dtype == torch.int32)"
        )
    (SQ, HQ, DQK) = q.shape
    (SKV, HKV, _) = kv.shape
    if not (d_v == 512):
        raise ValueError("Unsupported d_v")
    DV = d_v
    if not (kv.shape[-1] == DQK):
        raise ValueError("Invalid operator argument: kv.shape[-1] == DQK")
    (_, _, TOPK) = indices.shape
    if not (indices.shape == (SQ, HKV, TOPK)):
        raise ValueError("Invalid operator argument: indices.shape == (SQ, HKV, TOPK)")
    if attn_sink is not None:
        if not (attn_sink.is_contiguous()):
            raise ValueError("Invalid operator argument: attn_sink.is_contiguous()")
        if not (attn_sink.dtype == torch.float32):
            raise ValueError(
                "Invalid operator argument: attn_sink.dtype == torch.float32"
            )
        if not (attn_sink.shape == (HQ,)):
            raise ValueError("attn_sink error shape")
    if topk_length is not None:
        if not (topk_length.is_contiguous()):
            raise ValueError("Invalid operator argument: topk_length.is_contiguous()")
        if not (topk_length.dtype == torch.int32):
            raise ValueError(
                "Invalid operator argument: topk_length.dtype == torch.int32"
            )
        if not (topk_length.shape == (SQ,)):
            raise ValueError("topk_length error shape")
    if not (HKV == 1):
        raise ValueError("h_kv is expected to be 1")
    if not (HQ == 16 or HQ == 64 or HQ == 128):
        raise ValueError("Unsupported h_q")
    if not (DQK == 576 or DQK == 512):
        raise ValueError("Unsupported d_qk")
    _ = SKV
    D = DV
    TD = DQK - D
    DP = triton.next_power_of_2(D)
    HAVE_TAIL = TD > 0
    TDP = triton.next_power_of_2(TD) if HAVE_TAIL else 1
    G = HQ // HKV
    BH = TLE_FLASHMLA_PREFILL_BH
    RH = G // BH
    BK = TLE_FLASHMLA_PREFILL_BK
    if out is None:
        output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    else:
        if not (out.device == q.device and out.dtype == q.dtype):
            raise ValueError(
                "Invalid operator argument: out.device == q.device and out.dtype == q.dtype"
            )
        if not (out.shape == (SQ, HQ, DV)):
            raise ValueError(f"out error shape {tuple(out.shape)} != {(SQ, HQ, DV)}")
        output = out
    if max_logits_out is None:
        max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    else:
        if not (
            max_logits_out.device == q.device and max_logits_out.dtype == torch.float32
        ):
            raise ValueError(
                "Invalid operator argument: max_logits_out.device == q.device and "
                "max_logits_out.dtype == torch.float32"
            )
        if not (max_logits_out.shape == (SQ, HQ)):
            raise ValueError("max_logits_out error shape")
        max_logits = max_logits_out
    if lse_out is None:
        lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    else:
        if not (lse_out.device == q.device and lse_out.dtype == torch.float32):
            raise ValueError(
                "Invalid operator argument: lse_out.device == q.device and lse_out.dtype == torch.float32"
            )
        if not (lse_out.shape == (SQ, HQ)):
            raise ValueError("lse_out error shape")
        lse = lse_out

    def triton_grid(META):
        return (triton.cdiv(HQ, META["BH"]) * SQ,)

    if _flash_mla_sparse_halfd_serial_enabled() and (
        SQ > 1
        and HQ == 16
        and (HKV == 1)
        and (DQK in (512, 576))
        and (DV == 512)
        and (TOPK > 0)
        and q.is_contiguous()
        and kv.is_contiguous()
        and indices.is_contiguous()
        and (attn_sink is None or attn_sink.is_contiguous())
        and (topk_length is None or topk_length.is_contiguous())
    ):
        if _flash_mla_sparse_qkfold_blockdiag_enabled() and real_num_heads == 8:
            triton_flash_mla_sparse_fwd_halfd_qkfold_blockdiag[triton_grid](
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                sm_scale,
                output,
                max_logits,
                lse,
                q.stride(1),
                q.stride(0),
                kv.stride(1),
                kv.stride(0),
                indices.stride(1),
                indices.stride(0),
                output.stride(1),
                output.stride(0),
                max_logits.stride(0),
                lse.stride(0),
                SQ,
                HQ,
                DQK,
                SKV,
                TOPK,
                attn_sink is not None,
                topk_length is not None,
                BK=16,
                BH=16,
                num_warps=4,
                num_stages=1,
            )
        else:
            for _v_start, _write_stats in ((0, True), (256, False)):
                triton_flash_mla_sparse_fwd_halfd_serial[triton_grid](
                    q,
                    kv,
                    indices,
                    attn_sink,
                    topk_length,
                    sm_scale,
                    output,
                    max_logits,
                    lse,
                    q.stride(1),
                    q.stride(0),
                    kv.stride(1),
                    kv.stride(0),
                    indices.stride(1),
                    indices.stride(0),
                    output.stride(1),
                    output.stride(0),
                    max_logits.stride(0),
                    lse.stride(0),
                    SQ,
                    HQ,
                    DQK,
                    SKV,
                    TOPK,
                    attn_sink is not None,
                    topk_length is not None,
                    BK=16,
                    BH=16,
                    V_START=_v_start,
                    WRITE_STATS=_write_stats,
                    num_warps=2,
                    num_stages=1,
                )
        return (output, max_logits, lse)
    if _can_use_tle_flash_mla_sparse_fwd(q, kv, indices, d_v, topk_length):
        from triton.tools.tensor_descriptor import TensorDescriptor

        _set_triton_descriptor_allocator(q.device)
        q_desc = TensorDescriptor(
            q, shape=[SQ * HQ, DQK], strides=[DQK, 1], block_shape=[BH, DP // 2]
        )
        if HAVE_TAIL:
            tq_desc = TensorDescriptor(
                q, shape=[SQ * HQ, DQK], strides=[DQK, 1], block_shape=[BH, TDP]
            )
        else:
            tq_desc = q_desc
        output_desc = TensorDescriptor(
            output, shape=[SQ * HQ, D], strides=[D, 1], block_shape=[BH, DP // 2]
        )
        _tle_flashmla_prefill_fwd[triton_grid](
            q_desc,
            tq_desc,
            output_desc,
            kv,
            indices,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits,
            lse,
            SQ,
            HQ,
            DQK,
            SKV,
            TOPK,
            attn_sink is not None,
            topk_length is not None,
            D,
            TD,
            DP,
            TDP,
            G,
            HKV,
            RH,
            HAVE_TAIL,
            BK,
            BH,
            TLE_FLASHMLA_PREFILL_PAIR_BLOCKS,
            num_warps=TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS,
            num_stages=1,
        )
        return (output, max_logits, lse)
    triton_flash_mla_sparse_fwd[triton_grid](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        SQ,
        HQ,
        DQK,
        SKV,
        TOPK,
        attn_sink is not None,
        topk_length is not None,
    )
    return (output, max_logits, lse)


def _validate_local_prefill(
    q,
    kv,
    indices,
    d_v,
    attn_sink,
    topk_length,
    out,
    max_logits_out,
    lse_out,
    real_num_heads,
):
    """Validate the explicit MetaX 2-D-index, caller-owned-output extension.

    This is the full physical 16-head local computation, not a padded-head
    shortcut. Inputs and outputs must be non-overlapping buffers. Lifetimes and
    stream ordering remain the caller's responsibility, including graph replay.
    No tensor contents are read and no buffers are allocated or written here.
    """
    if q.ndim != 3 or q.shape[1] != 16 or q.shape[2] not in (512, 576):
        raise ValueError("local prefill requires q shape [rows, 16, 512 or 576]")
    rows, heads, head_dim = q.shape
    if rows <= 0 or d_v != 512 or real_num_heads is not None:
        raise ValueError("local prefill requires nonempty rows, d_v=512 and full heads")
    if kv.ndim != 3 or kv.shape[0] <= 0 or kv.shape[1:] != (1, head_dim):
        raise ValueError("local prefill requires nonempty kv shape [keys, 1, head_dim]")
    if indices.ndim != 2 or indices.shape[0] != rows or indices.shape[1] <= 0:
        raise ValueError("local prefill requires nonempty indices shape [rows, topk]")
    required = (
        (q, q.shape, torch.bfloat16, "q"),
        (kv, kv.shape, torch.bfloat16, "kv"),
        (indices, indices.shape, torch.int32, "indices"),
        (out, (rows, heads, d_v), torch.bfloat16, "out"),
        (max_logits_out, (rows, heads), torch.float32, "max_logits_out"),
        (lse_out, (rows, heads), torch.float32, "lse_out"),
    )
    optional = (
        (attn_sink, (heads,), torch.float32, "attn_sink"),
        (topk_length, (rows,), torch.int32, "topk_length"),
    )
    for tensor, shape, dtype, name in required + tuple(
        item for item in optional if item[0] is not None
    ):
        if tensor is None:
            raise ValueError(f"local prefill requires caller-owned {name}")
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"invalid local prefill {name}: shape/dtype/device/layout")


def _sparse_prefill_local(
    q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse
):
    logger.debug("GEMS_METAX SPARSE_PREFILL_LOCAL")
    (SQ, HQ, DQK) = q.shape
    (SKV, TOPK) = (kv.shape[0], indices.shape[-1])
    grid = (SQ,)
    if (
        _tuning.DEFAULT_POLICY.prefill_staged
        and DQK == 512
        and (q.dtype == torch.bfloat16)
        and (kv.dtype == torch.bfloat16)
        and kv.is_contiguous()
        and (indices.dtype == torch.int32)
    ):
        work = buffers(q, TOPK)
        run(
            q,
            kv,
            indices,
            attn_sink,
            topk_length,
            out,
            max_logits,
            lse,
            work,
            qw=2,
            pw=2,
            sm_scale=sm_scale,
        )
    else:
        for _v_start, _write_stats in ((0, True), (256, False)):
            triton_flash_mla_sparse_fwd_halfd_serial[grid](
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                sm_scale,
                out,
                max_logits,
                lse,
                q.stride(1),
                q.stride(0),
                kv.stride(1),
                kv.stride(0),
                indices.stride(1),
                indices.stride(0),
                out.stride(1),
                out.stride(0),
                max_logits.stride(0),
                lse.stride(0),
                SQ,
                HQ,
                DQK,
                SKV,
                TOPK,
                attn_sink is not None,
                topk_length is not None,
                BK=16,
                BH=16,
                V_START=_v_start,
                WRITE_STATS=_write_stats,
                num_warps=2,
                num_stages=1,
            )
    return (out, max_logits, lse)
