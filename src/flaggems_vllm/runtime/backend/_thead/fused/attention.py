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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@triton.jit
def _flash_int8_pack_kv(
    K,
    V,
    KP,
    VP,
    N: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PK: tl.constexpr,
    SK: tl.constexpr,
    HK: tl.constexpr,
    PV: tl.constexpr,
    SV: tl.constexpr,
    HV: tl.constexpr,
):
    n = tl.program_id(0) * 32 + tl.arange(0, 32)
    d = tl.arange(0, D)
    h = tl.program_id(1)
    k = tl.load(
        K + (n[:, None] // PAGE) * PK + (n[:, None] % PAGE) * SK + h * HK + d[None, :],
        n[:, None] < N,
        0,
    )
    v = tl.load(
        V + (n[:, None] // PAGE) * PV + (n[:, None] % PAGE) * SV + h * HV + d[None, :],
        n[:, None] < N,
        0,
    )
    off = (h * N + n[:, None]) * D + d[None, :]
    tl.store(KP + off, k, n[:, None] < N)
    tl.store(VP + off, v.to(tl.float16), n[:, None] < N)


@libentry()
@triton.jit
def _flash_int8_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    ALIBI,
    sq: tl.constexpr,
    hq: tl.constexpr,
    sk: tl.constexpr,
    hk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    hv: tl.constexpr,
    pv: tl.constexpr,
    so: tl.constexpr,
    ho: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    table_stride: tl.constexpr,
    alibi_stride: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    CAUSAL: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    CAP: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HALF_PV: tl.constexpr,
    BATCH: tl.constexpr,
    COMPACT: tl.constexpr,
    FOLD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile, batch, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head if FOLD else head // GROUP
    query_tile: tl.constexpr = BM // GROUP if FOLD else BM
    if COMPACT:
        batches = tl.arange(0, triton.next_power_of_2(BATCH))
        begins = tl.load(CUQ + batches, batches < BATCH, 0)
        ends = tl.load(CUQ + batches + 1, batches < BATCH, 0)
        counts = tl.cdiv(ends - begins, query_tile)
        cumulative = tl.cumsum(counts)
        # The host grid bounds sum(ceil(q_len / query_tile)); excess tiles
        # map past the final request and are skipped by the query-length guard.
        batch = tl.minimum(
            tl.sum(((tile >= cumulative) & (batches < BATCH)).to(tl.int32)), BATCH - 1
        )
        tile -= tl.sum(tl.where(batches < batch, counts, 0))
    q_start = tl.load(CUQ + batch)
    nq = tl.load(CUQ + batch + 1) - q_start
    if PAGED:
        k_start = 0
        nk = tl.load(USED + batch)
    else:
        k_start = tl.load(CUK + batch)
        nk = tl.load(CUK + batch + 1) - k_start
    rows = tile * BM + tl.arange(0, BM)
    m = rows // GROUP if FOLD else rows
    h = kv_head * GROUP + rows % GROUP if FOLD else tl.full((BM,), head, tl.int32)
    d = tl.arange(0, D)
    if tile * query_tile < nq:
        q = tl.load(
            Q + (q_start + m[:, None]) * sq + h[:, None] * hq + d[None, :],
            m[:, None] < nq,
            0,
        )
        q_scale = tl.load(QS + batch * qs0 + h * qs1 + (m // 128) * qs2, m < nq, 0)
        maximum = tl.full((BM,), float("-inf"), tl.float32)
        denom = tl.full((BM,), 0, tl.float32)
        acc = tl.full((BM, D), 0, tl.float32)
        if HAS_ALIBI:
            slope = tl.load(ALIBI + batch * alibi_stride + h)
        # Skip KV blocks that are masked for every query in this tile.
        first = 0
        end = nk
        if LEFT >= 0:
            first = tl.maximum(0, tile * query_tile + nk - nq - LEFT) // BN
        if CAUSAL:
            end = tl.minimum(end, (tile + 1) * query_tile + nk - nq)
        if RIGHT >= 0:
            end = tl.minimum(end, (tile + 1) * query_tile + nk - nq + RIGHT)
        for start in range(first, tl.cdiv(tl.maximum(end, 0), BN)):
            n = start * BN + tl.arange(0, BN)
            if PAGED:
                page = tl.load(TABLE + batch * table_stride + n // PAGE, n < nk, 0)
                k_row = page * pk + (n % PAGE) * sk
                v_row = page * pv + (n % PAGE) * sv
            else:
                k_row = (k_start + n) * sk
                v_row = (k_start + n) * sv
            k = tl.load(
                K + k_row[None, :] + kv_head * hk + d[:, None], n[None, :] < nk, 0
            )
            ks = tl.load(KS + batch * ks0 + kv_head * ks1 + start * BN // 128 * ks2)
            vs = tl.load(VS + batch * vs0 + kv_head * vs1 + start * BN // 128 * vs2)
            scores = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
            scores = scores * (q_scale * ks * SCALE)[:, None]
            if CAP > 0:
                scores = CAP * (2 / (1 + tl.exp(2 * (-scores / CAP))) - 1)
            position = m + nk - nq
            if HAS_ALIBI:
                scores -= slope[:, None] * tl.abs(position[:, None] - n[None, :])
            valid = (m[:, None] < nq) & (n[None, :] < nk)
            if CAUSAL:
                valid &= n[None, :] <= position[:, None]
            if LEFT >= 0:
                valid &= n[None, :] >= position[:, None] - LEFT
            if RIGHT >= 0:
                valid &= n[None, :] <= position[:, None] + RIGHT
            scores = tl.where(valid, scores * 1.4426950408889634, float("-inf"))
            tile_max = tl.max(scores, 1)
            new_max = tl.maximum(maximum, tile_max)
            safe_max = tl.where(new_max == float("-inf"), 0, new_max)
            alpha = tl.exp2(maximum - safe_max)
            if HALF_PV:
                p = tl.exp2(scores - safe_max[:, None])
                denom = denom * alpha + tl.sum(p, 1)
            else:
                # Normalize within this KV tile so probability quantization needs
                # only a constant multiply. beta brings its sum/PV into the running scale.
                safe_tile = tl.where(tile_max == float("-inf"), 0, tile_max)
                p = tl.exp2(scores - safe_tile[:, None])
                beta = tl.exp2(tile_max - safe_max)
                denom = denom * alpha + tl.sum(p, 1) * beta
                p_scale = beta * (1.0 / 255)
                # Signed INT8 stores 0..255 levels with a -128 zero point.
                p_int8 = (tl.floor(p * 255 + 0.5) - 128).to(tl.int8)
            v = tl.load(
                V + v_row[:, None] + kv_head * hv + d[None, :], n[:, None] < nk, 0
            )
            if HALF_PV:
                if vs2 == 0:
                    acc = tl.dot(
                        p.to(tl.float16), v.to(tl.float16), acc * alpha[:, None]
                    )
                else:
                    partial = tl.dot(
                        p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float32
                    )
                    acc = acc * alpha[:, None] + partial * vs
            else:
                correction = tl.dot(
                    tl.full((BM, BN), -128, tl.int8), v, out_dtype=tl.int32
                )
                partial = tl.dot(p_int8, v, -correction, out_dtype=tl.int32).to(
                    tl.float32
                )
                acc = acc * alpha[:, None] + partial * (p_scale * vs)[:, None]
            maximum = new_max
        result = acc / tl.where(denom > 0, denom, 1)[:, None]
        if HALF_PV and vs2 == 0:
            v_scale = tl.load(VS + batch * vs0 + kv_head * vs1, nk > 0, 0)
            result *= v_scale
        tl.store(
            O + (q_start + m[:, None]) * so + h[:, None] * ho + d[None, :],
            result,
            m[:, None] < nq,
        )
        if WRITE_LSE:
            lse = tl.where(
                denom > 0, maximum * 0.6931471805599453 + tl.log(denom), float("inf")
            )
            tl.store(LSE + h * TOTAL_Q + q_start + m, lse, m < nq)


def flash_attn_varlen_func_w8a8_int8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """PPU INT8 variable-length attention, inference forward only.

    Q is [total_q, heads, D]; K/V are [total_k, kv_heads, D] or paged
    [pages, page_size, kv_heads, D]. D is 64 or 128, heads must be a multiple
    of kv_heads, and the last dimension must be contiguous. Q/K/V are INT8.
    Required FP32 descales have shape [batch, heads, ceil(max_length / 128)]
    (kv_heads for K/V), indexed by logical sequence position even for paged KV.
    Each real value is the INT8 value multiplied by its descale.

    QK uses INT8 dot with INT32 accumulation. For paged KV with max query
    length >= 128, K/V are packed by KV head and V is converted to FP16;
    PV uses FP16 operands with FP32 accumulation. Other paths quantize
    probabilities to 256 levels and use INT8 PV with zero-point correction.
    Softmax and online accumulation use FP32. The long-query workspace
    requires three bytes per element of the physical K cache.
    Output is BF16 by default, or uses the supplied FP16/BF16 out buffer.
    LSE is [heads, total_q], FP32. Fully masked rows return zero and LSE +inf,
    following FlashAttention's convention. Inputs and out must not overlap.
    """
    if dropout_p != 0 or return_attn_probs or q_v is not None:
        raise NotImplementedError(
            "Only attention inference without dropout is supported"
        )
    if scheduler_metadata is not None or s_aux is not None or num_splits != 0:
        raise NotImplementedError(
            "Scheduler, auxiliary and split-KV modes are unsupported"
        )
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("Context parallel attention is unsupported")
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise NotImplementedError("Q/K/V must use INT8")
    if q.shape[-1] not in (64, 128):
        raise NotImplementedError("Only head dimensions 64 and 128 are supported")
    if seqused_k is not None and block_table is None:
        raise NotImplementedError("seqused_k requires paged KV")

    batch = cu_seqlens_q.numel() - 1
    total, heads, dim = q.shape
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    if total == 0:
        return (out, lse) if return_softmax_lse else out
    left, right = (-1, -1) if window_size is None else window_size
    paged = block_table is not None
    # Packed non-causal prefill benefits from more query rows per program.
    block_m = 64 if not paged and not causal and max_seqlen_q >= 512 else 16
    # Pack query heads sharing a KV head into MMA rows, reusing K/V loads.
    group = heads // k.shape[-2]
    fold = paged and group > 1 and group <= 16 and group & (group - 1) == 0
    if fold:
        block_m = 32 if max_seqlen_q > 16 else 16
    query_tile = block_m // group if fold else block_m
    grid_heads = k.shape[-2] if fold else heads
    num_warps = 8 if block_m == 64 and dim == 128 else 4
    compact = paged and max_seqlen_q > 16 and total * 2 < batch * max_seqlen_q
    grid = (
        (triton.cdiv(total, query_tile) + batch - 1, 1, grid_heads)
        if compact
        else (triton.cdiv(max_seqlen_q, query_tile), batch, grid_heads)
    )
    half_pv = paged and max_seqlen_q >= 128
    with torch_device_fn.device(q.device):
        if half_pv and k.shape[0] > 0:
            # Preserve physical page indices while making each head contiguous.
            # Packing and the INT8-to-FP16 V conversion are part of this call.
            n = k.shape[0] * k.shape[1]
            kp = torch.empty_strided(
                k.shape,
                (k.shape[1] * dim, dim, n * dim, 1),
                dtype=torch.int8,
                device=k.device,
            )
            vp = torch.empty_strided(
                kp.shape, kp.stride(), dtype=torch.float16, device=v.device
            )
            _flash_int8_pack_kv[(triton.cdiv(n, 32), k.shape[2])](
                k,
                v,
                kp,
                vp,
                n,
                dim,
                k.shape[1],
                *k.stride()[:3],
                *v.stride()[:3],
            )
            k, v = kp, vp
        _flash_int8_fwd[grid](
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_k,
            block_table,
            q_descale,
            k_descale,
            v_descale,
            alibi_slopes,
            q.stride(0),
            q.stride(1),
            k.stride(-3),
            k.stride(-2),
            k.stride(0) if paged else 0,
            v.stride(-3),
            v.stride(-2),
            v.stride(0) if paged else 0,
            out.stride(0),
            out.stride(1),
            *q_descale.stride(),
            *k_descale.stride(),
            *v_descale.stride(),
            block_table.stride(0) if paged else 0,
            (
                alibi_slopes.stride(0)
                if alibi_slopes is not None and alibi_slopes.ndim == 2
                else 0
            ),
            total,
            heads // k.shape[-2],
            dim,
            k.shape[1] if paged else 0,
            paged,
            causal,
            left,
            right,
            softcap,
            dim**-0.5 if softmax_scale is None else softmax_scale,
            alibi_slopes is not None,
            return_softmax_lse,
            HALF_PV=half_pv,
            BATCH=batch,
            COMPACT=compact,
            FOLD=fold,
            BM=block_m,
            BN=128 if half_pv else 64,
            num_warps=num_warps,
            num_stages=3 if half_pv else 1,
        )
    return (out, lse) if return_softmax_lse else out
