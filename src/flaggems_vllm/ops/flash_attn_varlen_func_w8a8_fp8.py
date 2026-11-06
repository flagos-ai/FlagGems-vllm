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
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, tl_extra_shim
from flaggems_vllm.utils.device_info import get_device_capability

logger = logging.getLogger(__name__)


def _select_decode_splits(batch, kv_heads, chunks, max_k, block_n, num_sms):
    # Target one wave of grouped CTAs, with at least four KV tiles per split.
    tasks = max(1, batch * kv_heads * chunks)
    desired = max(1, (num_sms + tasks - 1) // tasks)
    kv_tiles = max(1, (max_k + block_n - 1) // block_n)
    limit = min(8, desired, max(1, kv_tiles // 4))
    splits = 1
    for candidate in (2, 4, 8):
        if candidate <= limit:
            splits = candidate
    return splits


@triton.jit
def apply_alibi(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    is_causal: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slope: tl.constexpr = None,
):
    if is_alibi:
        if is_causal:
            # The row independent alibi bias renders the same attention output
            # as with the standard alibi because softmax is shift invariant, i.e.,
            # softmax(A + bias + const) = softamx(A + bias). The following two
            # biases are no different if causal is true.
            # bias_1 = [
            #   -4, -3, -2,  X, X,
            #   -4, -3, -2, -1, X,
            #   -4, -3, -2, -1, 0,
            # ]
            # bias_2 = [
            #   -2, -1, 0,  X,  X,
            #   -3, -2, -1, 0,  X,
            #   -4, -3, -2, -1, 0,
            # ]
            bias = alibi_slope * (-max_seqlen_k + 1 + col_idx[None, :]).to(tl.float32)
            S += bias
        else:
            bias = -alibi_slope * tl.abs(
                col_idx[None, :] - max_seqlen_k + max_seqlen_q - row_idx[:, None]
            ).to(tl.float32)
            S += bias

    return S


@triton.jit
def apply_mask(
    S,
    col_idx,
    row_idx,
    max_seqlen_q,
    max_seqlen_k,
    window_size_left,
    window_size_right,
    is_even_mn: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
):
    need_mask = is_causal | is_local | (not is_even_mn)
    if need_mask:
        # Extra care should be taken to void one-off errors: both col_lb and col_rb are inclusive!
        col_lb = max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left)
        col_rb = min(
            max_seqlen_k - 1, row_idx + max_seqlen_k - max_seqlen_q + window_size_right
        )

        if is_causal:
            S = tl.where(col_idx[None, :] > col_rb[:, None], float("-inf"), S)

        if is_local:
            S = tl.where(
                (col_idx[None, :] > col_rb[:, None])
                | (col_idx[None, :] < col_lb[:, None]),
                float("-inf"),
                S,
            )

        if (not is_local) & (not is_causal) & (not is_even_mn):
            S = tl.where(col_idx[None, :] >= max_seqlen_k, float("-inf"), S)

    return S


@triton.jit
def softmax_rescale(
    O_acc,
    S,
    row_max,
    row_sum,
    softmax_scale_log2e: tl.constexpr,
    is_border: tl.constexpr,
    fp8_p_max: tl.constexpr,
    acc_rescale=1.0,
):
    prev_max = row_max
    row_max = tl.maximum(row_max, tl.max(S, 1))

    if is_border:
        cur_max = tl.where(row_max == float("-inf"), 0, row_max)
    else:
        cur_max = row_max

    p_scale = tl.math.exp2((prev_max - cur_max) * softmax_scale_log2e)
    row_sum *= p_scale
    O_acc *= (p_scale * acc_rescale)[:, None]

    # Scale E4M3 probabilities by 256 in the exponent domain.
    p_log2_scale: tl.constexpr = 8.0
    max_scaled = tl.where(
        row_max == float("-inf"),
        0,
        row_max * softmax_scale_log2e - p_log2_scale,
    )
    P = tl.math.exp2(S * softmax_scale_log2e - max_scaled[:, None])
    row_sum = row_sum + tl.sum(P, 1)
    return O_acc, P, row_max, row_sum


@triton.jit
def _load_fp8_block_descales(
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    bid,
    hid,
    kv_hid,
    row_start,
    col_start,
):
    # Block quantization stores one descale per
    # logical 128-token Q/K/V block.  These scales correct FP8 QK and FP8 PV.
    q_block = row_start // 128
    kv_block = col_start // 128
    q_descale = tl.load(
        q_descale_ptr
        + bid * q_descale_batch_stride
        + hid * q_descale_head_stride
        + q_block * q_descale_block_stride
    ).to(tl.float32)
    k_descale = tl.load(
        k_descale_ptr
        + bid * k_descale_batch_stride
        + kv_hid * k_descale_head_stride
        + kv_block * k_descale_block_stride
    ).to(tl.float32)
    v_descale = tl.load(
        v_descale_ptr
        + bid * v_descale_batch_stride
        + kv_hid * v_descale_head_stride
        + kv_block * v_descale_block_stride
    ).to(tl.float32)
    return q_descale, k_descale, v_descale


@triton.jit
def _fp8_pv_dot(
    P,
    V,
    acc,
    v_descale,
    fp8_p_max: tl.constexpr,
    fp8_dtype: tl.constexpr,
    precise_p: tl.constexpr = False,
):
    # The scaled row sum cancels the factor of 256 during normalization.
    P_fp8 = P.to(fp8_dtype)
    pv = tl.dot(P_fp8, V, out_dtype=tl.float32)
    if precise_p:
        # Short KV sequences and sparse boundary rows amplify P rounding.
        # Correct its residual with another FP8 dot, not a BF16 fallback.
        P_residual = ((P - P_fp8.to(tl.float32)) * 32.0).to(fp8_dtype)
        pv += tl.dot(P_residual, V, out_dtype=tl.float32) * (1.0 / 32.0)
    return acc + pv * v_descale


@triton.jit
def _fp16_pv_dot(
    P,
    V,
    acc,
    v_descale,
    fp8_p_max: tl.constexpr,
    fp8_dtype: tl.constexpr,
    precise_p: tl.constexpr = False,
):
    # Keep FP8 storage, but avoid probability quantization and its second
    # residual dot in the accuracy-sensitive wide-head prefill path.
    pv = tl.dot(P.to(tl.float16), V.to(tl.float16), out_dtype=tl.float32)
    return acc + pv * v_descale


@triton.jit
def _fp16_pv_dot_residual(
    P,
    V,
    acc,
    v_descale,
    fp8_p_max: tl.constexpr,
    fp8_dtype: tl.constexpr,
    precise_p: tl.constexpr = False,
):
    # Large V scales can amplify FP16 probability rounding near cancellation.
    # Preserve the FP32 probability residual with a second FP16 dot. Scaling
    # E4M3 V by 1/32 is exact in FP16, including its smallest nonzero value.
    P_hi = P.to(tl.float16)
    P_lo = ((P - P_hi.to(tl.float32)) * 32.0).to(tl.float16)
    V_hi = V.to(tl.float16)
    pv = tl.dot(P_hi, V_hi, out_dtype=tl.float32)
    pv = tl.dot(
        P_lo,
        (V_hi * (1.0 / 32.0)).to(tl.float16),
        acc=pv,
        out_dtype=tl.float32,
    )
    return acc + pv * v_descale


@triton.jit
def _fusion_load_kv(
    K,
    V,
    TABLE,
    bid,
    kv_head,
    col,
    length,
    table_row,
    k_page,
    k_row,
    k_head,
    v_page,
    v_row,
    v_head,
    PAGE: tl.constexpr,
    D: tl.constexpr,
):
    physical = tl.load(
        TABLE + bid * table_row + col // PAGE,
        mask=col < length,
        other=0,
    ).to(tl.int64)
    dims = tl.arange(0, D)
    key_offset = physical * k_page + (col % PAGE) * k_row + kv_head * k_head
    value_offset = physical * v_page + (col % PAGE) * v_row + kv_head * v_head
    key = tl.load(
        K + key_offset[None, :] + dims[:, None],
        mask=col[None, :] < length,
        other=0.0,
    )
    value = tl.load(
        V + value_offset[:, None] + dims[None, :],
        mask=col[:, None] < length,
        other=0.0,
    )
    return key, value


@triton.jit
def _fusion_softmax_pv(
    acc,
    row_max,
    row_sum,
    scores,
    value,
    v_scale,
    SCALE: tl.constexpr,
    BORDER: tl.constexpr,
):
    previous = row_max
    row_max = tl.maximum(row_max, tl.max(scores, 1))
    current = tl.where(row_max == -float("inf"), 0.0, row_max) if BORDER else row_max
    alpha = tl.exp2((previous - current) * (SCALE * 1.4426950408889634))
    row_sum *= alpha
    acc *= alpha[:, None]
    maximum_scaled = tl.where(
        row_max == -float("inf"),
        0.0,
        row_max * (SCALE * 1.4426950408889634) - 8.0,
    )
    probability = tl.exp2(
        scores * (SCALE * 1.4426950408889634) - maximum_scaled[:, None]
    )
    row_sum += tl.sum(probability, 1)
    acc += (
        tl.dot(probability.to(tl.float16), value.to(tl.float16), out_dtype=tl.float32)
        * v_scale
    )
    return acc, row_max, row_sum


@triton.jit
def _short_fused_fwd(
    Q,
    K,
    V,
    CU,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    OUT,
    LSE,
    q_row,
    q_head,
    k_page,
    k_row,
    k_head,
    v_page,
    v_row,
    v_head,
    table_row,
    qs_b,
    qs_h,
    qs_n,
    ks_b,
    ks_h,
    ks_n,
    vs_b,
    vs_h,
    vs_n,
    B: tl.constexpr,
    SPLITS: tl.constexpr,
    MAX_Q: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    FUSED_HEADS: tl.constexpr,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
    CAUSAL: tl.constexpr = True,
    D: tl.constexpr = 256,
):
    token_tile = tl.program_id(0)
    bid = tl.program_id(1)
    head_group = tl.program_id(2) // SPLITS
    split = tl.program_id(2) % SPLITS
    chunks: tl.constexpr = G // FUSED_HEADS
    token_rows: tl.constexpr = ROWS // FUSED_HEADS
    kv_head = head_group // chunks
    head_chunk = head_group % chunks
    first = tl.load(CU + bid)
    q_len = tl.load(CU + bid + 1) - first
    if q_len <= 1 or q_len > MAX_Q or token_tile * token_rows >= q_len:
        return
    k_len = tl.load(USED + bid)

    # Token-major rows keep the fused Q heads adjacent in packed Q storage.
    matrix_rows = tl.arange(0, ROWS)
    query_row = token_tile * token_rows + matrix_rows // FUSED_HEADS
    head = kv_head * G + head_chunk * FUSED_HEADS + matrix_rows % FUSED_HEADS
    dims = tl.arange(0, D)
    query = tl.load(
        Q
        + (first + query_row[:, None]) * q_row
        + head[:, None] * q_head
        + dims[None, :],
        mask=query_row[:, None] < q_len,
        other=0.0,
    ).to(tl.float16)
    q_scale = tl.load(
        QS + bid * qs_b + head * qs_h + (query_row // 128) * qs_n,
        mask=query_row < q_len,
        other=1.0,
    ).to(tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    row_max = tl.full((ROWS,), -float("inf"), tl.float32)
    row_sum = tl.zeros((ROWS,), tl.float32)
    split_blocks = tl.cdiv(tl.cdiv(k_len, BN), SPLITS)
    first_block = split * split_blocks
    end_block = tl.minimum(tl.cdiv(k_len, BN), (split + 1) * split_blocks)
    if CAUSAL:
        end_block = tl.minimum(
            end_block,
            tl.cdiv((token_tile + 1) * token_rows + k_len - q_len, BN),
        )
    end_block = tl.maximum(end_block, first_block)
    boundary_blocks = tl.minimum(
        end_block - first_block, tl.cdiv(token_rows, BN) + 1 if CAUSAL else 1
    )
    for step in tl.range(0, boundary_blocks):
        block = end_block - 1 - step
        col = block * BN + tl.arange(0, BN)
        key, value = _fusion_load_kv(
            K,
            V,
            TABLE,
            bid,
            kv_head,
            col,
            k_len,
            table_row,
            k_page,
            k_row,
            k_head,
            v_page,
            v_row,
            v_head,
            PAGE,
            D,
        )
        k_scale = tl.load(KS + bid * ks_b + kv_head * ks_h + (block * BN // 128) * ks_n)
        v_scale = tl.load(VS + bid * vs_b + kv_head * vs_h + (block * BN // 128) * vs_n)
        scores = tl.dot(query, key.to(tl.float16), out_dtype=tl.float32)
        scores *= q_scale[:, None] * k_scale
        visible = col[None, :] < k_len
        if CAUSAL:
            visible = visible & (col[None, :] <= query_row[:, None] + k_len - q_len)
        scores = tl.where(visible, scores, -float("inf"))
        acc, row_max, row_sum = _fusion_softmax_pv(
            acc, row_max, row_sum, scores, value, v_scale, SCALE, True
        )

    for block in tl.range(
        end_block - boundary_blocks - 1, first_block - 1, -1, num_stages=STAGES
    ):
        col = block * BN + tl.arange(0, BN)
        key, value = _fusion_load_kv(
            K,
            V,
            TABLE,
            bid,
            kv_head,
            col,
            k_len,
            table_row,
            k_page,
            k_row,
            k_head,
            v_page,
            v_row,
            v_head,
            PAGE,
            D,
        )
        k_scale = tl.load(KS + bid * ks_b + kv_head * ks_h + (block * BN // 128) * ks_n)
        v_scale = tl.load(VS + bid * vs_b + kv_head * vs_h + (block * BN // 128) * vs_n)
        scores = tl.dot(query, key.to(tl.float16), out_dtype=tl.float32)
        scores *= q_scale[:, None] * k_scale
        acc, row_max, row_sum = _fusion_softmax_pv(
            acc, row_max, row_sum, scores, value, v_scale, SCALE, False
        )

    empty = (row_sum == 0) | (row_sum != row_sum)
    acc /= tl.where(empty, 1.0, row_sum)[:, None]
    lse = tl.where(
        empty,
        -float("inf"),
        row_max * SCALE + tl.log(row_sum * (1.0 / 256.0)),
    )
    partial_token = (split * B + bid) * MAX_Q + query_row
    tl.store(
        OUT + (partial_token[:, None] * H + head[:, None]) * D + dims[None, :],
        acc,
        mask=query_row[:, None] < q_len,
    )
    tl.store(
        LSE + ((split * B + bid) * H + head) * MAX_Q + query_row,
        lse,
        mask=query_row < q_len,
    )


@triton.jit
def _short_fused_combine(
    PARTIAL,
    PARTIAL_LSE,
    CU,
    OUT,
    LSE,
    out_row,
    out_head,
    total_q,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    MAX_Q: tl.constexpr,
):
    row = tl.program_id(0)
    bid = tl.program_id(1)
    head = tl.program_id(2)
    first = tl.load(CU + bid)
    q_len = tl.load(CU + bid + 1) - first
    if q_len <= 1 or q_len > MAX_Q or row >= q_len:
        return
    split = tl.arange(0, SPLITS)
    partial_lse = tl.load(PARTIAL_LSE + ((split * B + bid) * H + head) * MAX_Q + row)
    maximum = tl.max(partial_lse, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(partial_lse - safe_maximum)
    denominator = tl.sum(weights, 0)
    dims = tl.arange(0, D)
    partial = tl.load(
        PARTIAL
        + (((split[:, None] * B + bid) * MAX_Q + row) * H + head) * D
        + dims[None, :]
    )
    value = tl.sum(partial * weights[:, None], 0) / tl.where(
        denominator > 0.0, denominator, 1.0
    )
    lse = tl.where(denominator > 0.0, maximum + tl.log(denominator), float("inf"))
    tl.store(OUT + (first + row) * out_row + head * out_head + dims, value)
    tl.store(LSE + head * total_q + first + row, lse)


@triton.jit
def _flash_varlen_grouped_decode_fwd(
    Q,
    K,
    V,
    CU,
    USED,
    TABLE,
    QS,
    KS,
    VS,
    OUT,
    LSE,
    PART,
    PLSE,
    q_row,
    q_head,
    k_page,
    k_row,
    k_head,
    v_page,
    v_row,
    v_head,
    table_row,
    qs_b,
    qs_h,
    ks_b,
    ks_h,
    ks_n,
    vs_b,
    vs_h,
    vs_n,
    out_row,
    out_head,
    total_q,
    H: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    PAGE: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    WORKERS: tl.constexpr,
    BM: tl.constexpr = 16,
    D: tl.constexpr = 256,
    PART_BY_BATCH: tl.constexpr = False,
    PART_ROWS: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    GROUP_TILE: tl.constexpr = 8
    chunks: tl.constexpr = tl.cdiv(G, GROUP_TILE)
    kh = tl.program_id(1) // chunks
    chunk = tl.program_id(1) % chunks
    worker = tl.program_id(2)
    token = tl.load(CU + bid)
    q_end = tl.load(CU + bid + 1)
    if q_end - token != 1:
        return
    partial_token = bid if PART_BY_BATCH else token
    partial_rows = PART_ROWS if PART_BY_BATCH else total_q
    length = tl.load(USED + bid)
    tiles = tl.cdiv(length, BN)
    tiles_per_split = tl.maximum(tl.cdiv(tiles, SPLITS), 1)
    active_splits = tl.cdiv(tiles, tiles_per_split)
    if worker >= tl.maximum(active_splits, 1):
        return
    local_group = tl.arange(0, BM)
    group = chunk * GROUP_TILE + local_group
    valid_heads = (local_group < GROUP_TILE) & (group < G)
    heads = kh * G + group
    dims = tl.arange(0, D)
    q = tl.load(
        Q + token * q_row + heads[:, None] * q_head + dims[None, :],
        mask=valid_heads[:, None],
        other=0.0,
    )
    q_scale = tl.load(QS + bid * qs_b + heads * qs_h, mask=valid_heads, other=1).to(
        tl.float32
    )
    if WORKERS == 1:
        merged_maximum = tl.full((BM,), -float("inf"), tl.float32)
        merged_denominator = tl.zeros((BM,), tl.float32)
        merged_acc = tl.zeros((BM, D), tl.float32)
    for split in range(worker, tl.maximum(active_splits, 1), WORKERS):
        first = split * tiles_per_split
        last = tl.minimum(first + tiles_per_split, tiles)
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        for tile in range(first, last):
            positions = tile * BN + tl.arange(0, BN)
            tl.static_assert(BN % PAGE == 0)
            # A KV tile starts on a page boundary. Load each page ID once
            # and share it across that page's token offsets.
            logical_pages = tile * (BN // PAGE) + tl.arange(0, BN // PAGE)
            page_ids = tl.load(
                TABLE + bid * table_row + logical_pages,
                mask=logical_pages * PAGE < length,
                other=0,
            )
            page = tl.broadcast_to(page_ids[:, None], (BN // PAGE, PAGE))
            page = page.reshape(BN).to(tl.int64)
            k_offsets = page * k_page + (positions % PAGE) * k_row + kh * k_head
            v_offsets = page * v_page + (positions % PAGE) * v_row + kh * v_head
            key = tl.load(
                K + k_offsets[None, :] + dims[:, None],
                mask=positions[None, :] < length,
                other=0.0,
            )
            value = tl.load(
                V + v_offsets[:, None] + dims[None, :],
                mask=positions[:, None] < length,
                other=0.0,
            )
            k_scale = tl.load(KS + bid * ks_b + kh * ks_h + (tile * BN // 128) * ks_n)
            v_scale = tl.load(VS + bid * vs_b + kh * vs_h + (tile * BN // 128) * vs_n)
            scores = tl.dot(q, key).to(tl.float32)
            scores *= q_scale[:, None] * k_scale * (SCALE * 1.4426950408889634)
            scores = tl.where(positions[None, :] < length, scores, -float("inf"))
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp2(maximum - new_maximum)
            probabilities = tl.exp2(scores - new_maximum[:, None]) * 256.0
            denominator = denominator * alpha + tl.sum(probabilities, 1)
            acc *= alpha[:, None]
            # Correct FP8 probability rounding for short sequences as in the general path.
            acc = _fp8_pv_dot(
                probabilities, value, acc, v_scale, 448.0, tl.float8e4nv, precise_p=True
            )
            maximum = new_maximum
        empty = denominator == 0
        result = acc / tl.where(empty, 1.0, denominator)[:, None]
        lse = tl.where(
            empty,
            float("inf"),
            maximum * 0.6931471805599453 + tl.log(denominator / 256.0),
        )
        # Empty partitions are never materialized. A sole nonempty partition is final.
        if WORKERS == 1:
            new_maximum = tl.maximum(merged_maximum, maximum)
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            old_weight = tl.exp2(merged_maximum - safe_maximum)
            new_weight = tl.exp2(maximum - safe_maximum)
            merged_acc = merged_acc * old_weight[:, None] + acc * new_weight[:, None]
            merged_denominator = (
                merged_denominator * old_weight + denominator * new_weight
            )
            merged_maximum = new_maximum
        elif active_splits <= 1:
            tl.store(
                OUT + token * out_row + heads[:, None] * out_head + dims[None, :],
                result,
                mask=valid_heads[:, None],
            )
            tl.store(LSE + heads * total_q + token, lse, mask=valid_heads)
        else:
            tl.store(
                PART
                + ((split * partial_rows + partial_token) * H + heads[:, None]) * D
                + dims[None, :],
                result,
                mask=valid_heads[:, None],
            )
            tl.store(
                PLSE + (split * H + heads) * partial_rows + partial_token,
                lse,
                mask=valid_heads,
            )
    if WORKERS == 1:
        empty = merged_denominator == 0
        result = merged_acc / tl.where(empty, 1.0, merged_denominator)[:, None]
        lse = tl.where(
            empty,
            float("inf"),
            merged_maximum * 0.6931471805599453 + tl.log(merged_denominator / 256.0),
        )
        tl.store(
            OUT + token * out_row + heads[:, None] * out_head + dims[None, :],
            result,
            mask=valid_heads[:, None],
        )
        tl.store(LSE + heads * total_q + token, lse, mask=valid_heads)


@triton.jit
def _flash_varlen_grouped_decode_combine(
    OUT,
    LSE,
    PART,
    PLSE,
    CU,
    USED,
    total_q,
    out_row,
    out_head,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr = 64,
    PART_BY_BATCH: tl.constexpr = False,
    PART_ROWS: tl.constexpr = 0,
):
    bid = tl.program_id(0)
    head = tl.program_id(1)
    d_start = tl.program_id(2) * BD
    token = tl.load(CU + bid)
    if tl.load(CU + bid + 1) - token != 1:
        return
    partial_token = bid if PART_BY_BATCH else token
    partial_rows = PART_ROWS if PART_BY_BATCH else total_q
    length = tl.load(USED + bid)
    tiles = tl.cdiv(length, BN)
    per_split = tl.maximum(tl.cdiv(tiles, SPLITS), 1)
    active = tl.cdiv(tiles, per_split)
    if active <= 1:
        return
    splits = tl.arange(0, BS)
    dims = d_start + tl.arange(0, BD)
    lse = tl.load(
        PLSE + (splits * H + head) * partial_rows + partial_token,
        mask=splits < active,
        other=-float("inf"),
    )
    maximum = tl.max(lse, 0)
    weights = tl.exp(lse - maximum)
    weight_sum = tl.sum(weights, 0)
    values = tl.load(
        PART
        + ((splits[:, None] * partial_rows + partial_token) * H + head) * D
        + dims[None, :],
        mask=splits[:, None] < active,
        other=0,
    )
    result = tl.sum(values * weights[:, None], 0) / weight_sum
    tl.store(OUT + token * out_row + head * out_head + dims, result)
    if d_start == 0:
        tl.store(LSE + head * total_q + token, maximum + tl.log(weight_sum))


@triton.jit
def apply_softcap(S, softcap, is_softcap: tl.constexpr):
    if is_softcap:
        S = tl_extra_shim.tanh(S * softcap)

    return S


def block_m_splitkv_heuristic(headdim):
    return 128 if headdim <= 128 else 64


def block_n_splitkv_heuristic(headdim):
    return 64 if headdim <= 64 else 32


def is_even_mn(M, N, BM, BN, WL, WR):
    if M % BM == 0 and N % BN == 0:
        if M % N == 0 or N % M == 0:
            if (WL == -1 or WL % BN == 0) and (WR == -1 or WR % BN == 0):
                return True
    return False


def block_m_splitkv_heuristic_spec_args(args):
    return 128 if args["d"] <= 128 else 64


def block_n_splitkv_heuristic_spec_args(args):
    return 64 if args["d"] <= 64 else 32


def is_even_mn_spec_args(args):
    if (
        args["seqlen_q"] % args["BLOCK_M"] == 0
        and args["seqlen_k"] % args["BLOCK_N"] == 0
    ):
        if (
            args["seqlen_q"] % args["seqlen_k"] == 0
            or args["seqlen_k"] % args["seqlen_q"] == 0
        ):
            if (
                args["window_size_left"] == -1
                or args["window_size_left"] % args["BLOCK_N"] == 0
            ) and (
                args["window_size_right"] == -1
                or args["window_size_right"] % args["BLOCK_N"] == 0
            ):
                return True
    return False


def fwd_configs_w8a8():
    # use a dedicated config space for FP8 QK.  The original FA2
    # configs were picked for fp16/bf16 QK, while FP8 often benefits from trying
    # larger N tiles and more shape-specific choices.  num_stages is deliberately
    # left as a tunable dimension instead of forcing one pipeline depth globally.
    return [
        # D64 candidates: small shapes often prefer smaller M, while long-S
        # QK-heavy cases can benefit from larger N.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        # D128 candidates: keep the original-style narrow-N option, but also
        # test wider N because QK is FP8 and may tolerate a different balance.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ]


def prune_fwd_configs_w8a8(configs, nargs, **kwargs):
    d = nargs["d"]
    out = []
    for cfg in configs:
        BM = cfg.kwargs["BLOCK_M"]
        BN = cfg.kwargs["BLOCK_N"]
        w = cfg.num_warps
        s = cfg.num_stages

        # keep D64 and D128 candidate sets separate to avoid trying
        # configs that are known to have poor register/occupancy balance.
        if d <= 64:
            if (BM, BN, w, s) in {
                (128, 32, 4, 2),
                (128, 128, 8, 2),
                (128, 64, 4, 2),
                (128, 64, 4, 3),
                (128, 128, 4, 2),
                (128, 128, 4, 3),
            }:
                out.append(cfg)
        else:
            if (BM, BN, w, s) in {
                (128, 32, 8, 2),
                (128, 32, 8, 3),
                (128, 64, 8, 2),
                (128, 64, 8, 3),
                (128, 128, 4, 2),
                (128, 128, 8, 2),
                (64, 128, 8, 3),
            }:
                out.append(cfg)

    return out


def flash_fwd_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.autotune(
    configs=fwd_configs_w8a8(),
    prune_configs_by={"early_config_prune": prune_fwd_configs_w8a8},
    key=["b", "h", "seqlen_q", "seqlen_k", "d", "is_dropout"],
)
@triton.heuristics(
    values={
        "BLOCK_K": flash_fwd_kernel_heur_block_k,
        "PRE_LOAD_V": lambda args: False,
        "IS_EVEN_MN": lambda args: is_even_mn(
            args["seqlen_q"],
            args["seqlen_k"],
            args["BLOCK_M"],
            args["BLOCK_N"],
            args["window_size_left"],
            args["window_size_right"],
        ),
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k,
    cu_seqlens_k_ptr,
    is_seqused_k,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q: tl.constexpr,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    # For D128, split the output dimension across two D64 CTAs. D64 or
    # non-split-D uses a single CTA.
    d_split = tl.program_id(2)
    d_start = d_split * BLOCK_D
    hid = bh % h
    bid = bh // h
    num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    # We draw a minimum covering frame on the attention map that this CTA is assigned to process.
    # The frame edges are rounded to multiples of BLOCK_M and BLOCK_N for rows and columns respectively.

    col_min = 0
    if is_local:
        col_min = max(0, m_block * BLOCK_M + seqlen_k - seqlen_q - window_size_left)
        if not IS_EVEN_MN:
            # round left
            col_min = (col_min // BLOCK_N) * BLOCK_N

    col_max = seqlen_k
    if is_causal or is_local:
        col_max += (m_block - num_m_blocks + 1) * BLOCK_M
        if is_local:
            col_max += window_size_right
        col_max = min(seqlen_k, col_max)

    if not IS_EVEN_MN:
        # round right
        col_max = tl.cdiv(col_max, BLOCK_N) * BLOCK_N

    if (not is_causal) and (not is_local):
        if IS_EVEN_MN:
            masking_cols: tl.constexpr = 0
        else:
            masking_cols: tl.constexpr = BLOCK_N
    elif (
        is_causal | is_local
    ) and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_cols: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) * BLOCK_N
    else:
        # local
        masking_cols: tl.constexpr = (tl.cdiv(BLOCK_M, BLOCK_N) + 1) * BLOCK_N

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    q_batch_stride = tl.multiple_of(q_batch_stride, d * h)
    q_ptr += bid * q_batch_stride + hid * q_head_stride
    row_start = m_block * BLOCK_M
    row_idx = row_start + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & d == BLOCK_K:
        Q = tl.load(q_ptr + q_off, cache_modifier=".cg")
    else:
        Q = tl.load(q_ptr + q_off, mask=qmask, cache_modifier=".cg")

    # Use a [BLOCK_M, BLOCK_D] PV accumulator. For D128, BLOCK_D is 64,
    # so one CTA does not hold a [BM, 128] accumulator, reducing register pressure.
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    k_batch_stride = tl.multiple_of(k_batch_stride, d * hk)
    h_hk_ratio = h // hk
    kv_hid = hid // h_hk_ratio
    k_ptr += bid * k_batch_stride
    k_ptr += kv_hid * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += kv_hid * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    # Load only the 64-wide V slice for the current D split; QK still uses
    # the full BLOCK_K of 128.
    v_d = d_start + tl.arange(0, BLOCK_D)
    v_dmask = v_d < d
    v_offset = tl.arange(0, BLOCK_N)[:, None] * k_row_stride + v_d[None, :]

    p_bk0 = k_ptr + k_offset
    p_bv0 = v_ptr + v_offset

    if is_causal | is_local | (not IS_EVEN_MN):
        # Cut short masking cols if there's not enough cols out there
        masking_cols = min(col_max - col_min, masking_cols)
        for col_shift in tl.range(0, masking_cols, step=BLOCK_N):
            col_start = col_max - col_shift - BLOCK_N
            col_start = tl.multiple_of(col_start, BLOCK_N)
            off = col_start * k_row_stride
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_bk0 + off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
            elif d == BLOCK_K:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(p_bk0 + off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
            else:
                col_idx = col_start + tl.arange(0, BLOCK_N)
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_bk0 + off,
                    mask=kvmask[None, :] & dmask[:, None],
                    cache_modifier=".cg",
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & v_dmask[None, :],
                        cache_modifier=".cg",
                    )
            q_descale, k_descale, v_descale = _load_fp8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            # QK is FP8*FP8 tensor-core compute; the per-block
            # Q/K descales restore the original score magnitude before softmax.
            S = tl.dot(Q, K, out_dtype=tl.float32, allow_tf32=False)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = col_start + tl.arange(0, BLOCK_N)
            row_idx = row_start + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            # tl.store(p_bp0 + col_start, S)
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=is_local,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
                fp8_p_max=fp8_p_max,
            )

            if not PRE_LOAD_V:
                off = col_start * k_row_stride
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_bv0 + off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(p_bv0 + off, mask=kvmask[:, None], cache_modifier=".cg")
                else:
                    kvmask = col_idx < seqlen_k
                    V = tl.load(
                        p_bv0 + off,
                        mask=kvmask[:, None] & v_dmask[None, :],
                        cache_modifier=".cg",
                    )
            acc_ = _fp8_pv_dot(P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)

    for col_start in tl.range(
        col_min, col_max - masking_cols, step=BLOCK_N, num_stages=num_stages
    ):
        col_start = tl.multiple_of(col_start, BLOCK_N)
        off = col_start * k_row_stride
        if d == BLOCK_K:
            K = tl.load(p_bk0 + off, cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
        else:
            K = tl.load(p_bk0 + off, mask=dmask[:, None], cache_modifier=".cg")
            if PRE_LOAD_V:
                V = tl.load(p_bv0 + off, mask=v_dmask[None, :], cache_modifier=".cg")

        q_descale, k_descale, v_descale = _load_fp8_block_descales(
            q_descale_ptr,
            k_descale_ptr,
            v_descale_ptr,
            q_descale_batch_stride,
            q_descale_head_stride,
            q_descale_block_stride,
            k_descale_batch_stride,
            k_descale_head_stride,
            k_descale_block_stride,
            v_descale_batch_stride,
            v_descale_head_stride,
            v_descale_block_stride,
            bid,
            hid,
            kv_hid,
            row_start,
            col_start,
        )
        # use FP8*FP8 tensor-core path for QK and apply the
        # block descales before online softmax.
        S = tl.dot(Q, K, out_dtype=tl.float32)
        S *= q_descale * k_descale
        S = apply_softcap(S, softcap, is_softcap)
        col_idx = col_start + tl.arange(0, BLOCK_N)
        row_idx = row_start + tl.arange(0, BLOCK_M)
        S = apply_alibi(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            is_causal=is_causal,
            is_alibi=is_alibi,
            alibi_slope=alibi_slope,
        )
        S = apply_mask(
            S,
            col_idx,
            row_idx,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
            is_even_mn=True,
            is_causal=False,
            is_local=is_local,
        )

        acc_, P, rowmax_, rowsum_ = softmax_rescale(
            acc_,
            S,
            rowmax_,
            rowsum_,
            softmax_scale_log2e=scale_softmax_log2,
            is_border=is_local,
            fp8_p_max=fp8_p_max,
        )

        if not PRE_LOAD_V:
            off = col_start * k_row_stride
            if d == BLOCK_K:
                V = tl.load(p_bv0 + off, cache_modifier=".cg")
            else:
                V = tl.load(p_bv0 + off, mask=v_dmask[None, :], cache_modifier=".cg")
        acc_ = _fp8_pv_dot(P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)

    # LSE
    # Undo the E4M3 probability scale for LSE only. Output normalization uses
    # the scaled row sum to cancel the matching scale in the PV accumulator.
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_ * (1.0 / 256.0)),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)
    acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_batch_stride = tl.multiple_of(o_batch_stride, d * h)
    o_ptr += bid * o_batch_stride
    o_ptr += hid * o_head_stride
    # Each split-D CTA writes only its assigned D64 output slice.
    o_cols = d_start + tl.arange(0, BLOCK_D)
    o_dmask = o_cols < d
    o_offset = row_idx[:, None] * o_row_stride + o_cols[None, :]

    if IS_EVEN_MN & (d == BLOCK_K) & (not SPLIT_D):
        tl.store(o_ptr + o_offset, out)
    else:
        tl.store(
            o_ptr + o_offset, out, mask=(row_idx[:, None] < seqlen_q) & o_dmask[None, :]
        )

    # Write back lse
    p_lse = softmax_lse_ptr + (bid * h + hid) * seqlen_q
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)

    # Both split-D CTAs produce the same LSE, so only D split 0 writes it
    # to avoid duplicate stores to the same address.
    lse_write_mask = d_split == 0
    if IS_EVEN_MN:
        tl.store(p_lse + row_idx, lse, mask=lse_write_mask)
    else:
        tl.store(p_lse + row_idx, lse, mask=(row_idx < seqlen_q) & lse_write_mask)


def flash_fwd_splitkv_kernel_heur_block_k(args):
    return triton.next_power_of_2(args["d"])


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": block_m_splitkv_heuristic_spec_args,
        "BLOCK_N": block_n_splitkv_heuristic_spec_args,
        "BLOCK_K": flash_fwd_splitkv_kernel_heur_block_k,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
        "PRE_LOAD_V": lambda args: True,
        "IS_EVEN_MN": is_even_mn_spec_args,
    }
)
@triton.jit(
    do_not_specialize=["seqlen_q", "seqlen_k", "seqlen_q_rounded", "seqlen_k_rounded"]
)
def flash_fwd_splitkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b: tl.constexpr,
    bk: tl.constexpr,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride: tl.constexpr,
    # kernel params
    IS_EVEN_MN: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    blocks_per_split: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    m_block = tl.program_id(0)
    split_id = tl.program_id(1)
    bid = tl.program_id(2) // h
    hid = tl.program_id(2) % h

    split_block_min = split_id * blocks_per_split
    split_block_max = split_block_min + blocks_per_split

    n_block_max = tl.cdiv(seqlen_k, BLOCK_N)
    if is_causal:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + seqlen_k - seqlen_q + window_size_right,
                BLOCK_N,
            ),
        )

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0

    if not is_causal:
        if IS_EVEN_MN:
            masking_block_min = n_block_max
        else:
            masking_block_min = n_block_max - 1
    elif is_causal and IS_EVEN_MN:  # causal implies window_size_right is zero
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        masking_block_min = n_block_max - tl.cdiv(BLOCK_M, BLOCK_N) - 1

    q_ptr += bid * q_batch_stride
    q_ptr += hid * q_head_stride
    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    q_off = row_idx[:, None] * q_row_stride + tl.arange(0, BLOCK_K)[None, :]
    p_qm = q_ptr + q_off
    dmask = tl.arange(0, BLOCK_K) < d
    qmask = dmask[None, :] & (row_idx[:, None] < seqlen_q)
    if IS_EVEN_MN & BLOCK_K == d:
        Q = tl.load(p_qm, cache_modifier=".cg")
    else:
        Q = tl.load(p_qm, mask=qmask, cache_modifier=".cg")

    h_hk_ratio = h // hk
    kv_hid = hid // h_hk_ratio
    k_ptr += bid * k_batch_stride
    k_ptr += kv_hid * k_head_stride
    v_ptr += bid * k_batch_stride
    v_ptr += kv_hid * k_head_stride

    k_offset = (
        tl.arange(0, BLOCK_N)[None, :] * k_row_stride + tl.arange(0, BLOCK_K)[:, None]
    )
    p_k0 = k_ptr + k_offset

    v_offset = (
        tl.arange(0, BLOCK_N)[:, None] * k_row_stride + tl.arange(0, BLOCK_K)[None, :]
    )
    p_v0 = v_ptr + v_offset

    acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if split_block_max <= masking_block_min:
        # no masking needed
        for n_block in tl.range(
            split_block_min, split_block_max, num_stages=num_stages
        ):
            kv_off = n_block * BLOCK_N * k_row_stride
            if d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
            else:
                K = tl.load(
                    p_k0 + kv_off, mask=dmask[:, None], cache_modifier=".cg", other=0.0
                )
            if PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            row_start = m_block * BLOCK_M
            col_start = n_block * BLOCK_N
            q_descale, k_descale, v_descale = _load_fp8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            # split-KV QK also applies per-block descales.
            S = tl.dot(Q, K, out_dtype=tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=False,
                fp8_p_max=fp8_p_max,
            )

            if not PRE_LOAD_V:
                if d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            # split-KV PV uses an in-kernel FP8 P tile and FP8 V.
            acc_ = _fp8_pv_dot(P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)
    else:
        for n_block in tl.range(split_block_min, min(split_block_max, n_block_max)):
            kv_off = n_block * BLOCK_N * k_row_stride
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            if IS_EVEN_MN & d == BLOCK_K:
                K = tl.load(p_k0 + kv_off, cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
            elif d == BLOCK_K:
                kvmask = col_idx < seqlen_k
                K = tl.load(p_k0 + kv_off, mask=kvmask[None, :], cache_modifier=".cg")
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
            else:
                kvmask = col_idx < seqlen_k
                K = tl.load(
                    p_k0 + kv_off,
                    mask=dmask[:, None] & kvmask[None, :],
                    cache_modifier=".cg",
                    other=0.0,
                )
                if PRE_LOAD_V:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )

            row_start = m_block * BLOCK_M
            col_start = n_block * BLOCK_N
            q_descale, k_descale, v_descale = _load_fp8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                row_start,
                col_start,
            )
            # masked split-KV QK applies per-block descales.
            S = tl.dot(Q, K, out_dtype=tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
                is_even_mn=IS_EVEN_MN,
                is_causal=is_causal,
                is_local=False,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=(is_causal or is_local),
                fp8_p_max=fp8_p_max,
            )

            if not PRE_LOAD_V:
                if IS_EVEN_MN & d == BLOCK_K:
                    V = tl.load(p_v0 + kv_off, cache_modifier=".cg")
                elif d == BLOCK_K:
                    V = tl.load(
                        p_v0 + kv_off, mask=kvmask[:, None], cache_modifier=".cg"
                    )
                else:
                    V = tl.load(
                        p_v0 + kv_off,
                        mask=dmask[None, :] & kvmask[:, None],
                        cache_modifier=".cg",
                        other=0.0,
                    )
            # masked split-KV PV runs as FP8 P * FP8 V.
            acc_ = _fp8_pv_dot(P, V, acc_, v_descale, fp8_p_max, v_ptr.type.element_ty)

    # LSE
    lse = tl.where(
        rowsum_ == 0 | (rowsum_ != rowsum_),
        float("-inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_ * (1.0 / 256.0)),
    )
    inv_sum = tl.where(rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_)

    # Rescale output
    acc_ *= inv_sum[:, None]

    # Write back output
    # o_splits layout = (n_splits, batch_size, num_heads, seqlen_q, head_size)
    # grid = (seq_block, split, batch * head)
    o_split_ptr = o_ptr
    # + split, batch, head offsets, seq_block offsets are already added in row_idx
    o_split_ptr += (split_id * tl.num_programs(2) + tl.program_id(2)) * seqlen_q * d
    o_split_offset = row_idx[:, None] * d + tl.arange(0, BLOCK_K)
    o_split_ptr = tl.multiple_of(o_split_ptr, d)
    p_om = o_split_ptr + o_split_offset

    if IS_EVEN_MN & BLOCK_K == d:
        tl.store(p_om, acc_, cache_modifier=".cg")
    else:
        tl.store(p_om, acc_, mask=qmask, cache_modifier=".cg")

    # Write back lse
    # lse_splits layout = (n_splits, batch_size, num_heads, seqlen_q)
    lse_split_ptr = softmax_lse_ptr
    # + split, batch, head, seq_block offsets
    lse_split_ptr += (
        split_id * tl.num_programs(2) + tl.program_id(2)
    ) * seqlen_q + m_block * BLOCK_M

    if IS_EVEN_MN:
        tl.store(lse_split_ptr + tl.arange(0, BLOCK_M), lse, cache_modifier=".cg")
    else:
        tl.store(
            lse_split_ptr + tl.arange(0, BLOCK_M),
            lse,
            mask=row_idx < seqlen_q,
            cache_modifier=".cg",
        )


@libentry()
@triton.jit
def flash_fwd_splitkv_combine_kernel(
    out_ptr,
    lse_ptr,
    out_splits_ptr,
    lse_splits_ptr,
    head_size: tl.constexpr,
    out_split_stride,
    lse_split_stride,
    out_b_stride,
    out_s_stride,
    out_h_stride,
    n_splits,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    q_total,
    MAX_N_SPLITS: tl.constexpr,
    num_heads,
    seqlen_q,
):
    pid = tl.program_id(0)
    lse_splits_ptr += pid * BLOCK_M
    lse_ptr += pid * BLOCK_M
    out_splits_ptr += pid * BLOCK_M * head_size

    # Subtracting maximum from each of the split lse's for better numerical stability
    lse_split_offset = (
        tl.arange(0, BLOCK_M)[:, None]
        + tl.arange(0, MAX_N_SPLITS)[None, :] * lse_split_stride
    )
    lse_split_mask = (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] < q_total) & (
        tl.arange(0, MAX_N_SPLITS)[None, :] < n_splits
    )
    lse_splits = tl.load(
        lse_splits_ptr + lse_split_offset, mask=lse_split_mask, other=float("-inf")
    )
    max_lse = tl.max(lse_splits, 1)

    # Sum exp(lse(i) - max_lse) over all split i to obtain Z=sumexp(QK) up to a scaled factor exp(-max_lse)
    Zi_scaled = tl.exp(lse_splits - max_lse[:, None])
    Z_scaled = tl.sum(Zi_scaled, 1)
    Zi_Z = Zi_scaled / Z_scaled[:, None]

    # Write back LSE
    lse = tl.log(Z_scaled) + max_lse
    out_mask = pid * BLOCK_M + tl.arange(0, BLOCK_M) < q_total
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=out_mask)

    out_split_offset = (
        tl.arange(0, BLOCK_M)[:, None, None] * head_size
        + tl.arange(0, MAX_N_SPLITS)[None, :, None] * out_split_stride
        + tl.arange(0, BLOCK_K)[None, None, :]
    )
    out_split_mask = (
        (pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None, None] < q_total)
        & (tl.arange(0, MAX_N_SPLITS)[None, :, None] < n_splits)
        & (tl.arange(0, BLOCK_K)[None, None, :] < head_size)
    )
    out_splits = tl.load(
        out_splits_ptr + out_split_offset, mask=out_split_mask, other=0.0
    )
    out = tl.sum(Zi_Z[:, :, None] * out_splits, 1)
    out = out.to(out_ptr.type.element_ty)

    # Write back output.
    # The split buffers flatten [batch, head, query], but `out` is a BSHD view
    # whose storage order need not follow that flattening. Decode each row into
    # batch/head/query and use the real output strides.
    row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    batch = row // (num_heads * seqlen_q)
    head = (row // seqlen_q) % num_heads
    query = row % seqlen_q
    out_offset = (
        batch[:, None] * out_b_stride
        + head[:, None] * out_h_stride
        + query[:, None] * out_s_stride
        + tl.arange(0, BLOCK_K)[None, :]
    )
    dmask = tl.arange(0, BLOCK_K) < head_size
    tl.store(out_ptr + out_offset, out, mask=out_mask[:, None] & dmask[None, :])


@triton.jit
def virtual_to_cache_offset(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    block_size,
    k_row_stride,
    k_page_stride,
    boundary_check: tl.constexpr = False,
):
    # virtual_index is the kv sequence index in the current batch element
    # page_table_ptr is already pointed at current batch element's block table entry
    # block_size is the size of each block in the page table
    virtual_page_index = virtual_index // block_size
    page_offset = virtual_index % block_size
    if boundary_check:
        page_block_index = tl.load(
            page_table_ptr + virtual_page_index,
            mask=virtual_index < max_virtual_index,
            other=0,
        ).to(tl.int64)
    else:
        page_block_index = tl.load(page_table_ptr + virtual_page_index).to(tl.int64)
    return page_block_index * k_page_stride + page_offset * k_row_stride


@triton.jit
def _load_paged_k_tile_late_v(
    k_ptr_base,
    cache_rows,
    virtual_index,
    max_virtual_index,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    dims = tl.arange(0, BLOCK_K)
    mask = virtual_index[None, :] < max_virtual_index
    if d != BLOCK_K:
        mask = mask & (dims[:, None] < d)
    return tl.load(
        k_ptr_base + cache_rows[None, :] + dims[:, None],
        mask=mask,
        other=0.0,
    )


@triton.jit
def _load_paged_v_tile_late_v(
    v_ptr_base,
    cache_rows,
    virtual_index,
    max_virtual_index,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    dims = tl.arange(0, BLOCK_K)
    mask = virtual_index[:, None] < max_virtual_index
    if d != BLOCK_K:
        mask = mask & (dims[None, :] < d)
    return tl.load(
        v_ptr_base + cache_rows[:, None] + dims[None, :],
        mask=mask,
        other=0.0,
    )


@triton.jit
def load_from_kvcache(
    virtual_index,
    max_virtual_index,
    page_table_ptr,
    k_ptr_base,
    v_ptr_base,
    block_size,
    d: tl.constexpr,
    k_row_stride,
    k_page_stride,
    BLOCK_K: tl.constexpr,
    boundary_check: tl.constexpr = False,
):
    cache_offset = virtual_to_cache_offset(
        virtual_index,
        max_virtual_index,
        page_table_ptr,
        block_size,
        k_row_stride,
        k_page_stride,
        boundary_check,
    )
    k_offset = tl.arange(0, BLOCK_K)[:, None] + cache_offset[None, :]
    v_offset = tl.arange(0, BLOCK_K)[None, :] + cache_offset[:, None]
    if d == BLOCK_K:
        bK_mask = virtual_index[None, :] < max_virtual_index[None, :]
        bV_mask = virtual_index[:, None] < max_virtual_index[:, None]
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    else:
        bK_mask = (tl.arange(0, BLOCK_K)[:, None] < d) & (
            virtual_index[None, :] < max_virtual_index[None, :]
        )
        bV_mask = (tl.arange(0, BLOCK_K)[None, :] < d) & (
            virtual_index[:, None] < max_virtual_index[:, None]
        )
        bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
        bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
    return bK, bV


@triton.jit
def query_tile_prefix_kernel(
    cu_q_ptr,
    prefix_ptr,
    batch_map_ptr,
    tile_map_ptr,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    bids = tl.arange(0, BLOCK_B)
    starts = tl.load(cu_q_ptr + bids, mask=bids < B, other=0)
    ends = tl.load(cu_q_ptr + bids + 1, mask=bids < B, other=0)
    counts = tl.cdiv(ends - starts, BLOCK_M)
    prefix = tl.cumsum(counts)
    tl.store(prefix_ptr + bids + 1, prefix, mask=bids < B)
    tl.store(prefix_ptr, 0)
    local_tiles = tl.arange(0, BLOCK_T)
    positions = (prefix - counts)[:, None] + local_tiles[None, :]
    mask = (bids[:, None] < B) & (local_tiles[None, :] < counts[:, None])
    tl.store(batch_map_ptr + positions, bids[:, None], mask=mask)
    tl.store(tile_map_ptr + positions, local_tiles[None, :], mask=mask)


@triton.jit
def _check_varlen_v_scales(
    v_descale_ptr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    safe_flags_ptr,
    v_batch_stride,
    v_head_stride,
    v_block_stride,
    max_seqlen_k,
    HK: tl.constexpr,
    HAS_SEQUSED_K: tl.constexpr,
    SCAN_BLOCKS: tl.constexpr,
):
    # One CTA checks each request/KV-head pair for this invocation. The flag
    # is rewritten on every graph replay; no input-value cache is retained.
    bid = tl.program_id(0)
    kv_hid = tl.program_id(1)
    if HAS_SEQUSED_K:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_len = k_eos - k_bos
    logical_blocks = tl.cdiv(k_len, 128)
    declared_blocks = tl.cdiv(max_seqlen_k, 128)
    scale_index = tl.arange(0, SCAN_BLOCKS)
    valid = (scale_index < logical_blocks) & (scale_index < declared_blocks)
    scale = tl.load(
        v_descale_ptr
        + bid * v_batch_stride
        + kv_hid * v_head_stride
        + scale_index * v_block_stride,
        mask=valid,
        other=1.0,
    ).to(tl.float32)
    in_range = (scale >= 0.00390625) & (scale <= 0.0625)
    safe = tl.min(tl.where(valid, in_range, True).to(tl.int32), 0) == 1
    safe = (
        safe & (k_len >= 0) & (k_len <= max_seqlen_k) & (logical_blocks <= SCAN_BLOCKS)
    )
    tl.store(safe_flags_ptr + bid * HK + kv_hid, safe.to(tl.int32))


@triton.jit
def _scale_group_fwd_tile(
    bQ,
    acc,
    row_max,
    row_sum,
    q_scale,
    k_scale,
    frame_ratio,
    n_block,
    row_idx,
    q_len,
    k_len,
    page_table_ptr,
    k_ptr_base,
    v_ptr_base,
    k_row_stride,
    k_page_stride,
    block_size: tl.constexpr,
    d: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    is_alibi: tl.constexpr,
    alibi_slope,
    MASKED: tl.constexpr,
):
    # Reuse the quantization-group scales for both reverse-order BN64 tiles.
    col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    cache_rows = virtual_to_cache_offset(
        col_idx,
        k_len,
        page_table_ptr,
        block_size,
        k_row_stride,
        k_page_stride,
        boundary_check=MASKED,
    )
    bK = _load_paged_k_tile_late_v(
        k_ptr_base,
        cache_rows,
        col_idx,
        k_len,
        d,
        BLOCK_K,
    )
    scores = tl.dot(bQ.to(tl.float16), bK.to(tl.float16), out_dtype=tl.float32)
    scores *= q_scale * k_scale
    scores = apply_softcap(scores, softcap, is_softcap)
    scores = apply_alibi(
        scores,
        col_idx,
        row_idx,
        q_len,
        k_len,
        is_causal=is_causal,
        is_alibi=is_alibi,
        alibi_slope=alibi_slope,
    )
    scores = apply_mask(
        scores,
        col_idx,
        row_idx,
        q_len,
        k_len,
        window_size_left,
        window_size_right,
        is_even_mn=not MASKED,
        is_causal=is_causal and MASKED,
        is_local=is_local,
    )
    acc, probabilities, row_max, row_sum = softmax_rescale(
        acc,
        scores,
        row_max,
        row_sum,
        softmax_scale_log2e=scale_softmax_log2,
        is_border=MASKED or is_local,
        fp8_p_max=448.0,
        acc_rescale=frame_ratio,
    )
    bV = _load_paged_v_tile_late_v(
        v_ptr_base,
        cache_rows,
        col_idx,
        k_len,
        d,
        BLOCK_K,
    )
    acc = tl.dot(
        probabilities.to(tl.float16),
        bV.to(tl.float16),
        acc=acc,
        out_dtype=tl.float32,
    )
    return acc, row_max, row_sum


@libentry()
@triton.jit(
    do_not_specialize=[
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "b",
        "bk",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "total_q",
    ]
)
def flash_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    p_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    q_batch_stride,
    k_batch_stride,
    v_batch_stride,
    o_batch_stride,
    is_cu_seqlens_q: tl.constexpr,
    cu_seqlens_q_ptr,
    is_cu_seqlens_k: tl.constexpr,
    cu_seqlens_k_ptr,
    is_seqused_k: tl.constexpr,
    seqused_k_ptr,
    # sizes
    b,
    bk,
    h: tl.constexpr,
    hk: tl.constexpr,
    h_hk_ratio: tl.constexpr,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    seqlen_k_rounded,
    d: tl.constexpr,
    d_rounded: tl.constexpr,
    # scaling factors
    is_softcap: tl.constexpr,
    softcap: tl.constexpr,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_descale_ptr,
    q_descale_batch_stride,
    q_descale_head_stride,
    q_descale_block_stride,
    k_descale_batch_stride,
    k_descale_head_stride,
    k_descale_block_stride,
    v_descale_batch_stride,
    v_descale_head_stride,
    v_descale_block_stride,
    use_fa3_fp8_scales: tl.constexpr,
    fp8_p_max: tl.constexpr,
    # dropout
    is_dropout: tl.constexpr,
    p_dropout: tl.constexpr,
    rp_dropout: tl.constexpr,
    p_dropout_in_uint8_t: tl.constexpr,
    philox_args,
    return_softmax: tl.constexpr,
    # causal and swa
    is_causal: tl.constexpr,
    is_local: tl.constexpr,
    window_size_left: tl.constexpr,
    window_size_right: tl.constexpr,
    seqlenq_ngroups_swapped: tl.constexpr,
    is_paged: tl.constexpr,
    # alibi
    is_alibi: tl.constexpr,
    alibi_slopes_ptr,
    alibi_slopes_batch_stride: tl.constexpr,
    # block table
    total_q,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    block_size: tl.constexpr,
    k_page_stride,
    # kernel params
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_D: tl.constexpr,
    PRECISE_SHORT_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr = 1,
    SKIP_SINGLE_Q: tl.constexpr = False,
    SKIP_SHORT_Q: tl.constexpr = False,
    PAGED_PIPELINE_STAGES: tl.constexpr = 1,
    DELAY_PAGED_V: tl.constexpr = False,
    GROUP_FP8_SCALES: tl.constexpr = False,
    FULL_FP32_QK: tl.constexpr = False,
    UNIFIED_K_LENGTHS: tl.constexpr = False,
    PV_DOT: tl.constexpr = _fp8_pv_dot,
    QUERY_TILE_PREFIX=None,
    QUERY_BATCH_MAP=None,
    QUERY_LOCAL_TILE_MAP=None,
    COMPACT_QUERY_SCHEDULE: tl.constexpr = False,
    REVERSE_QUERY_TILES: tl.constexpr = False,
    SCALE_AWARE_GUARD: tl.constexpr = False,
    SCALE_AWARE_ACC: tl.constexpr = False,
    SCALE_SCAN_BLOCKS: tl.constexpr = 1,
    SCALE_SAFE_FLAGS=None,
    PRECHECK_SCALE_GUARD: tl.constexpr = False,
):
    if COMPACT_QUERY_SCHEDULE:
        query_tile = tl.program_id(0)
        if query_tile >= tl.load(QUERY_TILE_PREFIX + b):
            return
        bid = tl.load(QUERY_BATCH_MAP + query_tile)
        m_block = tl.load(QUERY_LOCAL_TILE_MAP + query_tile)
        kv_split = tl.program_id(1)
    else:
        m_block = tl.program_id(0)
        if NUM_KV_SPLITS > 1:
            bid = tl.program_id(1) % b
            kv_split = tl.program_id(1) // b
        else:
            bid = tl.program_id(1)
            kv_split = 0
    # The varlen grid has only three dimensions, so encode both the head
    # and D split in program_id(2).
    hd = tl.program_id(2)
    hid = hd % h
    d_split = hd // h
    d_start = d_split * BLOCK_D
    # num_m_blocks = tl.cdiv(seqlen_q, BLOCK_M)

    if is_cu_seqlens_q:
        q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
        q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
        q_len = q_eos - q_bos
        # Current request's start offset in the batched Q
        q_offset = q_bos * q_row_stride
        o_offset = q_bos * o_row_stride
        lse_offset = q_bos * 1
    else:
        q_len = seqlen_q
        q_offset = bid * q_batch_stride
        o_offset = bid * o_batch_stride
        lse_offset = bid * seqlen_q

    if REVERSE_QUERY_TILES and is_causal and not COMPACT_QUERY_SCHEDULE:
        if q_len > 512:
            # Schedule the longest causal query tiles first. Compact scheduling
            # keeps the local tile indices supplied by its existing map.
            num_m_tiles = tl.cdiv(q_len, BLOCK_M)
            if m_block >= num_m_tiles:
                return
            m_block = num_m_tiles - 1 - m_block

    if SKIP_SINGLE_Q and q_len == 1:
        return
    if SKIP_SHORT_Q and q_len <= 32:
        return

    if is_cu_seqlens_k:
        k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
        k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
        k_len_cache = k_eos - k_bos
        # k_offset = k_bos * k_row_stride
    else:
        k_len_cache = seqlen_k
        # k_offset = bid * k_batch_stride

    if is_seqused_k:
        k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
    else:
        k_len = k_len_cache

    # Specialize the full KV loop, rather than putting a runtime branch
    # around every PV dot and disabling the long-sequence pipeline.
    if not UNIFIED_K_LENGTHS and (k_len <= 128) != PRECISE_SHORT_K:
        return

    # Noop CTA
    if m_block * BLOCK_M >= q_len:
        return

    if SCALE_AWARE_GUARD:
        if PRECHECK_SCALE_GUARD:
            # Reuse the per-invocation result for every query tile and GQA head.
            all_scales_safe = (
                tl.load(SCALE_SAFE_FLAGS + bid * hk + (hid // h_hk_ratio)) != 0
            )
        else:
            # Select exactly one static loop variant using current device scales.
            # Positive bounded scales avoid zero division and extreme frame ratios.
            scale_index = tl.arange(0, SCALE_SCAN_BLOCKS)
            logical_blocks = tl.cdiv(k_len, 128)
            declared_blocks = tl.cdiv(seqlen_k, 128)
            scale_mask = (scale_index < logical_blocks) & (
                scale_index < declared_blocks
            )
            guard_scale = tl.load(
                v_descale_ptr
                + bid * v_descale_batch_stride
                + (hid // h_hk_ratio) * v_descale_head_stride
                + scale_index * v_descale_block_stride,
                mask=scale_mask,
                other=1.0,
            ).to(tl.float32)
            # The upper bound limits the FP16-P rounding contribution to about
            # 2^-11 * 448 / 16 = 0.0137 in the normalized output. Other errors
            # still require the independent reference checks.
            scale_in_range = (guard_scale >= 0.00390625) & (guard_scale <= 0.0625)
            all_scales_safe = (
                tl.min(tl.where(scale_mask, scale_in_range, True).to(tl.int32), 0) == 1
            )
            all_scales_safe = (
                all_scales_safe
                & (k_len >= 0)
                & (k_len <= seqlen_k)
                & (logical_blocks <= SCALE_SCAN_BLOCKS)
            )
        if all_scales_safe != SCALE_AWARE_ACC:
            return

    # is_even_mn = (q_len % BLOCK_M == 0) and (k_len % BLOCK_N == 0)
    is_even_mn: tl.constexpr = False

    if is_local:
        n_block_min = max(
            0, (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N
        )
    else:
        n_block_min = 0

    n_block_max = tl.cdiv(k_len, BLOCK_N)
    if is_causal or is_local:
        n_block_max = min(
            n_block_max,
            tl.cdiv(
                (m_block + 1) * BLOCK_M + k_len - q_len + window_size_right, BLOCK_N
            ),
        )

    if NUM_KV_SPLITS > 1:
        # Partition logical KV tiles, independently of physical cache pages.
        # Some causal/local rows have no keys in a split; they contribute zero.
        split_blocks = tl.cdiv(tl.cdiv(k_len, BLOCK_N), NUM_KV_SPLITS)
        n_block_min = max(n_block_min, kv_split * split_blocks)
        n_block_max = max(n_block_min, min(n_block_max, (kv_split + 1) * split_blocks))
        o_ptr += kv_split * total_q * h * d
        softmax_lse_ptr += kv_split * h * total_q

    # Locate the page table entry for the current batch element
    if is_paged:
        page_table_ptr += bid * page_table_batch_stride
    # Calculate the starting offset of q for the current head
    q_row_offset = hid * q_head_stride
    # Calculate the starting offset of k and v for the current head
    kv_hid = hid // h_hk_ratio
    k_row_offset = kv_hid * k_head_stride
    # Shift the k, v pointers to align with the current head
    k_ptr_base = k_ptr + k_row_offset
    v_ptr_base = v_ptr + k_row_offset

    gQ = tl.make_block_ptr(
        base=q_ptr + q_offset + q_row_offset,
        shape=(q_len, d),
        strides=(q_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    bQ = tl.load(
        gQ.advance([m_block * BLOCK_M, 0]),
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # BLOCK_D selects a full-width or split-D PV accumulator.
    acc_ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    if SCALE_AWARE_ACC:
        # acc_ represents the unnormalized output divided by this scale.
        previous_v_descale = tl.full((), 1.0, tl.float32)
    rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

    if is_alibi:
        alibi_offset = bid * alibi_slopes_batch_stride + hid
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
        alibi_slope /= scale_softmax
    else:
        alibi_slope = 0.0

    if not is_causal and not is_local:
        n_masking_steps = 1
    elif is_even_mn:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N)
    else:
        n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1

    n_masking_steps = min(n_block_max - n_block_min, n_masking_steps)

    row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    if GROUP_FP8_SCALES and SCALE_AWARE_ACC:
        tl.static_assert((is_paged and FULL_FP32_QK))
        tl.static_assert(BLOCK_N == 64 and 128 % BLOCK_M == 0)
        tl.static_assert((not SPLIT_D))
        # The current Q tile never crosses a logical 128-token scale block.
        q_descale = tl.load(
            q_descale_ptr
            + bid * q_descale_batch_stride
            + hid * q_descale_head_stride
            + (m_block * BLOCK_M // 128) * q_descale_block_stride
        ).to(tl.float32)
        # Empty KV splits must not access even an unused scale block.
        if n_block_max > n_block_min:
            # Complete the masked prefix down to a quantization-group boundary.
            # This keeps every subsequent full group entirely in the unmasked loop.
            masked_start = max(n_block_min, ((n_block_max - n_masking_steps) // 2) * 2)
            for kv_group in tl.range(
                tl.cdiv(n_block_max, 2) - 1,
                masked_start // 2 - 1,
                -1,
            ):
                k_descale = tl.load(
                    k_descale_ptr
                    + bid * k_descale_batch_stride
                    + kv_hid * k_descale_head_stride
                    + kv_group * k_descale_block_stride
                ).to(tl.float32)
                v_descale = tl.load(
                    v_descale_ptr
                    + bid * v_descale_batch_stride
                    + kv_hid * v_descale_head_stride
                    + kv_group * v_descale_block_stride
                ).to(tl.float32)
                group_ratio = tl.where(
                    previous_v_descale == v_descale,
                    1.0,
                    previous_v_descale / v_descale,
                )
                # A partial top group can start at its even tile. Apply the
                # frame conversion to the first valid tile, exactly once.
                first_block = min(n_block_max - 1, kv_group * 2 + 1)
                for sub_tile in tl.range(0, 2, loop_unroll_factor=1):
                    n_block = kv_group * 2 + 1 - sub_tile
                    if n_block < n_block_max and n_block >= masked_start:
                        frame_ratio = tl.where(n_block == first_block, group_ratio, 1.0)
                        acc_, rowmax_, rowsum_ = _scale_group_fwd_tile(
                            bQ,
                            acc_,
                            rowmax_,
                            rowsum_,
                            q_descale,
                            k_descale,
                            frame_ratio,
                            n_block,
                            row_idx,
                            q_len,
                            k_len,
                            page_table_ptr,
                            k_ptr_base,
                            v_ptr_base,
                            k_row_stride,
                            k_page_stride,
                            block_size,
                            d,
                            BLOCK_K,
                            BLOCK_N,
                            scale_softmax_log2,
                            is_causal,
                            is_local,
                            window_size_left,
                            window_size_right,
                            is_softcap,
                            softcap,
                            is_alibi,
                            alibi_slope,
                            MASKED=True,
                        )
                previous_v_descale = v_descale

            # Full interior groups have two valid tiles, with no per-tile scale
            # load, ratio division, or group-transition branch.
            for kv_group in tl.range(
                masked_start // 2 - 1,
                tl.cdiv(n_block_min, 2) - 1,
                -1,
                num_stages=PAGED_PIPELINE_STAGES,
            ):
                k_descale = tl.load(
                    k_descale_ptr
                    + bid * k_descale_batch_stride
                    + kv_hid * k_descale_head_stride
                    + kv_group * k_descale_block_stride
                ).to(tl.float32)
                v_descale = tl.load(
                    v_descale_ptr
                    + bid * v_descale_batch_stride
                    + kv_hid * v_descale_head_stride
                    + kv_group * v_descale_block_stride
                ).to(tl.float32)
                group_ratio = tl.where(
                    previous_v_descale == v_descale,
                    1.0,
                    previous_v_descale / v_descale,
                )
                for sub_tile in tl.range(0, 2, loop_unroll_factor=1):
                    n_block = kv_group * 2 + 1 - sub_tile
                    frame_ratio = tl.where(sub_tile == 0, group_ratio, 1.0)
                    acc_, rowmax_, rowsum_ = _scale_group_fwd_tile(
                        bQ,
                        acc_,
                        rowmax_,
                        rowsum_,
                        q_descale,
                        k_descale,
                        frame_ratio,
                        n_block,
                        row_idx,
                        q_len,
                        k_len,
                        page_table_ptr,
                        k_ptr_base,
                        v_ptr_base,
                        k_row_stride,
                        k_page_stride,
                        block_size,
                        d,
                        BLOCK_K,
                        BLOCK_N,
                        scale_softmax_log2,
                        is_causal,
                        is_local,
                        window_size_left,
                        window_size_right,
                        is_softcap,
                        softcap,
                        is_alibi,
                        alibi_slope,
                        MASKED=False,
                    )
                previous_v_descale = v_descale

            # An odd split/window lower bound leaves only the high tile of its
            # final group. Do not read the tile owned by the preceding KV split.
            if n_block_min % 2 != 0 and masked_start > n_block_min:
                kv_group = n_block_min // 2
                k_descale = tl.load(
                    k_descale_ptr
                    + bid * k_descale_batch_stride
                    + kv_hid * k_descale_head_stride
                    + kv_group * k_descale_block_stride
                ).to(tl.float32)
                v_descale = tl.load(
                    v_descale_ptr
                    + bid * v_descale_batch_stride
                    + kv_hid * v_descale_head_stride
                    + kv_group * v_descale_block_stride
                ).to(tl.float32)
                group_ratio = tl.where(
                    previous_v_descale == v_descale,
                    1.0,
                    previous_v_descale / v_descale,
                )
                n_block = n_block_min
                acc_, rowmax_, rowsum_ = _scale_group_fwd_tile(
                    bQ,
                    acc_,
                    rowmax_,
                    rowsum_,
                    q_descale,
                    k_descale,
                    group_ratio,
                    n_block,
                    row_idx,
                    q_len,
                    k_len,
                    page_table_ptr,
                    k_ptr_base,
                    v_ptr_base,
                    k_row_stride,
                    k_page_stride,
                    block_size,
                    d,
                    BLOCK_K,
                    BLOCK_N,
                    scale_softmax_log2,
                    is_causal,
                    is_local,
                    window_size_left,
                    window_size_right,
                    is_softcap,
                    softcap,
                    is_alibi,
                    alibi_slope,
                    MASKED=False,
                )
                previous_v_descale = v_descale
    else:
        n_block = n_block_max - 1
        for step in tl.range(0, n_masking_steps):
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            if is_paged:
                if DELAY_PAGED_V:
                    # Carry only the token row offsets across QK and softmax.
                    cache_rows = virtual_to_cache_offset(
                        col_idx,
                        k_len,
                        page_table_ptr,
                        block_size,
                        k_row_stride,
                        k_page_stride,
                        boundary_check=True,
                    )
                    bK = _load_paged_k_tile_late_v(
                        k_ptr_base,
                        cache_rows,
                        col_idx,
                        k_len,
                        d,
                        BLOCK_K,
                    )
                else:
                    bK, bV = load_from_kvcache(
                        col_idx,
                        k_len,
                        page_table_ptr,
                        k_ptr_base,
                        v_ptr_base,
                        block_size,
                        d,
                        k_row_stride,
                        k_page_stride,
                        BLOCK_K=BLOCK_K,
                        boundary_check=True,
                    )
            else:
                start_n = n_block * BLOCK_N
                k_ptr_seq = k_ptr_base + k_bos * k_row_stride
                v_ptr_seq = v_ptr_base + k_bos * k_row_stride
                gK = tl.make_block_ptr(
                    base=k_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                )
                gV = tl.make_block_ptr(
                    base=v_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    # For non-paged varlen attention, load only the V slice for
                    # the current D split.
                    offsets=(start_n, d_start),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(0, 1),
                )
                bK = tl.load(gK, boundary_check=(0, 1), padding_option="zero")
                bK = tl.trans(bK)
                bV = tl.load(gV, boundary_check=(0, 1), padding_option="zero")
            q_descale, k_descale, v_descale = _load_fp8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
            )
            if FULL_FP32_QK:
                S = tl.dot(bQ.to(tl.float16), bK.to(tl.float16), out_dtype=tl.float32)
            else:
                S = tl.dot(bQ, bK, out_dtype=tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                is_even_mn=is_even_mn,
                is_causal=is_causal,
                is_local=is_local,
            )

            if SCALE_AWARE_ACC:
                # Fold the frame change into online-softmax rescaling, keeping
                # one FP32 accumulator instead of a separate FP32 PV result.
                # Keep repeated logical-block scales exact. Other positive scale
                # ratios are bounded by 16 by the entry guard.
                acc_rescale = tl.where(
                    previous_v_descale == v_descale,
                    1.0,
                    previous_v_descale / v_descale,
                )
            else:
                acc_rescale = 1.0
            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=True,
                fp8_p_max=fp8_p_max,
                acc_rescale=acc_rescale,
            )

            if is_paged and DELAY_PAGED_V:
                # Convert and consume V only after probabilities are available.
                bV = _load_paged_v_tile_late_v(
                    v_ptr_base,
                    cache_rows,
                    col_idx,
                    k_len,
                    d,
                    BLOCK_K,
                )
            if SCALE_AWARE_ACC:
                acc_ = tl.dot(
                    P.to(tl.float16),
                    bV.to(tl.float16),
                    acc=acc_,
                    out_dtype=tl.float32,
                )
                previous_v_descale = v_descale
            else:
                acc_ = PV_DOT(
                    P,
                    bV,
                    acc_,
                    v_descale,
                    fp8_p_max,
                    v_ptr.type.element_ty,
                    precise_p=is_causal or is_local or PRECISE_SHORT_K,
                )
            n_block -= 1

        for n_block in tl.range(
            n_block_max - n_masking_steps - 1,
            n_block_min - 1,
            step=-1,
            num_stages=PAGED_PIPELINE_STAGES if is_paged else num_stages,
        ):
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            if is_paged:
                if DELAY_PAGED_V:
                    # Carry only the token row offsets across QK and softmax.
                    cache_rows = virtual_to_cache_offset(
                        col_idx,
                        k_len,
                        page_table_ptr,
                        block_size,
                        k_row_stride,
                        k_page_stride,
                        boundary_check=False,
                    )
                    bK = _load_paged_k_tile_late_v(
                        k_ptr_base,
                        cache_rows,
                        col_idx,
                        k_len,
                        d,
                        BLOCK_K,
                    )
                else:
                    bK, bV = load_from_kvcache(
                        col_idx,
                        k_len,
                        page_table_ptr,
                        k_ptr_base,
                        v_ptr_base,
                        block_size,
                        d,
                        k_row_stride,
                        k_page_stride,
                        BLOCK_K=BLOCK_K,
                    )
            else:
                start_n = n_block * BLOCK_N
                k_ptr_seq = k_ptr_base + k_bos * k_row_stride
                v_ptr_seq = v_ptr_base + k_bos * k_row_stride
                gK = tl.make_block_ptr(
                    base=k_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                )
                gV = tl.make_block_ptr(
                    base=v_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    # For non-paged varlen attention, load only the V slice for
                    # the current D split.
                    offsets=(start_n, d_start),
                    block_shape=(BLOCK_N, BLOCK_D),
                    order=(0, 1),
                )
                # Sequence tiles are complete here, but head_dim may be padded.
                bK = tl.load(gK, boundary_check=(1,), padding_option="zero")
                bK = tl.trans(bK)
                bV = tl.load(gV, boundary_check=(1,), padding_option="zero")
            q_descale, k_descale, v_descale = _load_fp8_block_descales(
                q_descale_ptr,
                k_descale_ptr,
                v_descale_ptr,
                q_descale_batch_stride,
                q_descale_head_stride,
                q_descale_block_stride,
                k_descale_batch_stride,
                k_descale_head_stride,
                k_descale_block_stride,
                v_descale_batch_stride,
                v_descale_head_stride,
                v_descale_block_stride,
                bid,
                hid,
                kv_hid,
                m_block * BLOCK_M,
                n_block * BLOCK_N,
            )
            if FULL_FP32_QK:
                S = tl.dot(bQ.to(tl.float16), bK.to(tl.float16), out_dtype=tl.float32)
            else:
                S = tl.dot(bQ, bK, out_dtype=tl.float32)
            S *= q_descale * k_descale
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                is_even_mn=True,
                is_causal=False,
                is_local=is_local,
            )

            if SCALE_AWARE_ACC:
                # Fold the frame change into online-softmax rescaling, keeping
                # one FP32 accumulator instead of a separate FP32 PV result.
                # Keep repeated logical-block scales exact. Other positive scale
                # ratios are bounded by 16 by the entry guard.
                acc_rescale = tl.where(
                    previous_v_descale == v_descale,
                    1.0,
                    previous_v_descale / v_descale,
                )
            else:
                acc_rescale = 1.0
            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=is_local,
                fp8_p_max=fp8_p_max,
                acc_rescale=acc_rescale,
            )
            if is_paged and DELAY_PAGED_V:
                # Convert and consume V only after probabilities are available.
                bV = _load_paged_v_tile_late_v(
                    v_ptr_base,
                    cache_rows,
                    col_idx,
                    k_len,
                    d,
                    BLOCK_K,
                )
            if SCALE_AWARE_ACC:
                acc_ = tl.dot(
                    P.to(tl.float16),
                    bV.to(tl.float16),
                    acc=acc_,
                    out_dtype=tl.float32,
                )
                previous_v_descale = v_descale
            else:
                acc_ = PV_DOT(
                    P,
                    bV,
                    acc_,
                    v_descale,
                    fp8_p_max,
                    v_ptr.type.element_ty,
                    precise_p=PRECISE_SHORT_K,
                )

    # LSE
    empty_row = (rowsum_ == 0) | (rowsum_ != rowsum_)
    lse = tl.where(
        empty_row,
        float("-inf") if NUM_KV_SPLITS > 1 else float("inf"),
        rowmax_ * scale_softmax + tl.log(rowsum_ * (1.0 / 256.0)),
    )
    inv_sum = tl.where(empty_row, 1.0, 1.0 / rowsum_)

    if SCALE_AWARE_ACC:
        inv_sum *= previous_v_descale
    acc_ *= inv_sum[:, None]

    out = acc_.to(o_ptr.type.element_ty)  # noqa

    # Write back output
    o_row_offset = hid * o_head_stride

    gO = tl.make_block_ptr(
        base=o_ptr + o_offset + o_row_offset,
        shape=(q_len, d),
        strides=(o_row_stride, 1),
        # Each varlen split-D CTA writes only its current D64 output slice.
        offsets=(0, d_start),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1))

    # Write back lse
    # lse shape: [h, total_q]
    softmax_lse_ptr += hid * total_q
    lse_row_offset = lse_offset + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    # All varlen split-D CTAs produce the same LSE, so only D split 0 writes
    # it back.
    tl.store(
        softmax_lse_ptr + lse_row_offset,
        lse,
        mask=(lse_row_offset < (lse_offset + q_len)) & (d_split == 0),
    )


@libentry()
@triton.jit
def flash_varlen_splitkv_combine_kernel(
    out_ptr,
    lse_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    total_q,
    out_row_stride,
    out_head_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    hid = row % H
    token = row // H
    d_start = tl.program_id(1) * BLOCK_D
    splits = tl.arange(0, BLOCK_S)
    dims = d_start + tl.arange(0, BLOCK_D)
    partial_lse = tl.load(
        partial_lse_ptr + (splits * H + hid) * total_q + token,
        mask=splits < NUM_KV_SPLITS,
        other=float("-inf"),
    )
    max_lse = tl.max(partial_lse, 0)
    # All-empty rows must avoid -inf - -inf and retain the public +inf LSE.
    safe_max = tl.where(max_lse == float("-inf"), 0.0, max_lse)
    weights = tl.exp(partial_lse - safe_max)
    weight_sum = tl.sum(weights, 0)
    weights /= tl.where(weight_sum == 0.0, 1.0, weight_sum)
    partial_out = tl.load(
        partial_out_ptr
        + ((splits[:, None] * total_q + token) * H + hid) * D
        + dims[None, :],
        mask=(splits[:, None] < NUM_KV_SPLITS) & (dims[None, :] < D),
        other=0.0,
    )
    result = tl.sum(partial_out * weights[:, None], 0)
    tl.store(
        out_ptr + token * out_row_stride + hid * out_head_stride + dims,
        result,
        mask=dims < D,
    )
    lse = tl.where(weight_sum == 0.0, float("inf"), safe_max + tl.log(weight_sum))
    if d_start == 0:
        tl.store(lse_ptr + hid * total_q + token, lse)


def CHECK_DEVICE(x):
    assert x.device.type == runtime.device.name


# Q/K/V are FP8 inputs while O stays FP16/BF16. The FP32 descale tensors carry
# the per-block quantization scales used by the QK and PV tensor-core matmuls.
_FP8_DTYPES = (torch.float8_e4m3fn,)
_HIGH_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _normalize_dense_descale(descale, batch_size, num_heads, nblocks, device, name):
    if descale is None:
        raise ValueError(f"{name} is required for FP8 attention")
    if descale.device != device:
        raise ValueError(f"{name} must be on the same device as q")
    if descale.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if descale.ndim == 0:
        descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
    elif descale.ndim == 1:
        if descale.numel() == 1:
            descale = descale.reshape(1, 1, 1).expand(batch_size, num_heads, nblocks)
        else:
            if descale.numel() != num_heads:
                raise ValueError(f"{name} 1D scale must have H elements")
            descale = descale.reshape(1, num_heads, 1).expand(
                batch_size, num_heads, nblocks
            )
    elif descale.ndim == 2:
        if descale.shape != (batch_size, num_heads):
            raise ValueError(f"{name} 2D scale must be [B, H]")
        descale = descale[:, :, None].expand(batch_size, num_heads, nblocks)
    else:
        if descale.ndim != 3 or descale.shape != (
            batch_size,
            num_heads,
            nblocks,
        ):
            raise ValueError(f"{name} must be [B, H, nblocks]")
    return descale


class fwd_params:
    # This positional ABI is also consumed by backend-specific kernels.
    # Keep the inference-only dropout and FP8 metadata slots for those callers.
    __slots__ = (
        # pointers and strides
        "q_ptr",
        "k_ptr",
        "v_ptr",
        "o_ptr",
        "p_ptr",
        "softmax_lse_ptr",
        "q_row_stride",
        "k_row_stride",
        "v_row_stride",
        "q_head_stride",
        "k_head_stride",
        "v_head_stride",
        "o_row_stride",
        "o_head_stride",
        "q_batch_stride",
        "k_batch_stride",
        "v_batch_stride",
        "o_batch_stride",
        "is_cu_seqlens_q",
        "cu_seqlens_q_ptr",
        "is_cu_seqlens_k",
        "cu_seqlens_k_ptr",
        "is_seqused_k",
        "seqused_k_ptr",
        # sizes
        "b",
        "bk",
        "h",
        "hk",
        "h_hk_ratio",
        "seqlen_q",
        "seqlen_k",
        "seqlen_q_rounded",
        "seqlen_k_rounded",
        "d",
        "d_rounded",
        # scaling factors
        "is_softcap",
        "softcap",
        "scale_softmax",
        "scale_softmax_log2",
        "q_descale_ptr",
        "k_descale_ptr",
        "v_descale_ptr",
        "q_descale_batch_stride",
        "q_descale_head_stride",
        "q_descale_block_stride",
        "k_descale_batch_stride",
        "k_descale_head_stride",
        "k_descale_block_stride",
        "v_descale_batch_stride",
        "v_descale_head_stride",
        "v_descale_block_stride",
        "use_fa3_fp8_scales",
        "fp8_p_max",
        # dropout
        "is_dropout",
        "p_dropout",
        "rp_dropout",
        "p_dropout_in_uint8_t",
        "philox_args",
        "return_softmax",
        # masking
        "is_causal",
        "is_local",
        "window_size_left",
        "window_size_right",
        "seqlenq_ngroups_swapped",
        "is_paged",
        # alibi
        "is_alibi",
        "alibi_slopes_ptr",
        "alibi_slopes_batch_stride",
        # block table
        "total_q",
        "page_table_ptr",
        "page_table_batch_stride",
        "block_size",
        "k_page_stride",
    )

    def __init__(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k,
        cu_seqlens_k_ptr,
        is_seqused_k,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h,
        hk,
        h_hk_ratio,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d,
        d_rounded,
        # scaling factors
        is_softcap,
        softcap,
        scale_softmax,
        scale_softmax_log2,
        q_descale_ptr,
        k_descale_ptr,
        v_descale_ptr,
        q_descale_batch_stride,
        q_descale_head_stride,
        q_descale_block_stride,
        k_descale_batch_stride,
        k_descale_head_stride,
        k_descale_block_stride,
        v_descale_batch_stride,
        v_descale_head_stride,
        v_descale_block_stride,
        use_fa3_fp8_scales,
        fp8_p_max,
        # dropout
        is_dropout,
        p_dropout,
        rp_dropout,
        p_dropout_in_uint8_t,
        philox_args,
        return_softmax,
        # masking
        is_causal,
        is_local,
        window_size_left,
        window_size_right,
        seqlenq_ngroups_swapped,
        is_paged,
        # alibi
        is_alibi,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride,
        block_size,
        k_page_stride,
    ):
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.o_ptr = o_ptr
        self.p_ptr = p_ptr
        self.softmax_lse_ptr = softmax_lse_ptr
        self.q_row_stride = q_row_stride
        self.k_row_stride = k_row_stride
        self.v_row_stride = v_row_stride
        self.q_head_stride = q_head_stride
        self.k_head_stride = k_head_stride
        self.v_head_stride = v_head_stride
        self.o_row_stride = o_row_stride
        self.o_head_stride = o_head_stride
        self.q_batch_stride = q_batch_stride
        self.k_batch_stride = k_batch_stride
        self.v_batch_stride = v_batch_stride
        self.o_batch_stride = o_batch_stride
        self.is_cu_seqlens_q = is_cu_seqlens_q
        self.cu_seqlens_q_ptr = cu_seqlens_q_ptr
        self.is_cu_seqlens_k = is_cu_seqlens_k
        self.cu_seqlens_k_ptr = cu_seqlens_k_ptr
        self.is_seqused_k = is_seqused_k
        self.seqused_k_ptr = seqused_k_ptr
        # sizes
        self.b = b
        self.bk = bk
        self.h = h
        self.hk = hk
        self.h_hk_ratio = h_hk_ratio
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.seqlen_q_rounded = seqlen_q_rounded
        self.seqlen_k_rounded = seqlen_k_rounded
        self.d = d
        self.d_rounded = d_rounded
        # scaling factors
        self.is_softcap = is_softcap
        self.softcap = softcap
        self.scale_softmax = scale_softmax
        self.scale_softmax_log2 = scale_softmax_log2
        self.q_descale_ptr = q_descale_ptr
        self.k_descale_ptr = k_descale_ptr
        self.v_descale_ptr = v_descale_ptr
        self.q_descale_batch_stride = q_descale_batch_stride
        self.q_descale_head_stride = q_descale_head_stride
        self.q_descale_block_stride = q_descale_block_stride
        self.k_descale_batch_stride = k_descale_batch_stride
        self.k_descale_head_stride = k_descale_head_stride
        self.k_descale_block_stride = k_descale_block_stride
        self.v_descale_batch_stride = v_descale_batch_stride
        self.v_descale_head_stride = v_descale_head_stride
        self.v_descale_block_stride = v_descale_block_stride
        self.use_fa3_fp8_scales = use_fa3_fp8_scales
        self.fp8_p_max = fp8_p_max
        # dropout
        self.is_dropout = is_dropout
        self.p_dropout = p_dropout
        self.rp_dropout = rp_dropout
        self.p_dropout_in_uint8_t = p_dropout_in_uint8_t
        self.philox_args = philox_args
        self.return_softmax = return_softmax
        # masking
        self.is_causal = is_causal
        self.is_local = is_local
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped
        self.is_paged = is_paged
        # alibi
        self.is_alibi = is_alibi
        self.alibi_slopes_ptr = alibi_slopes_ptr
        self.alibi_slopes_batch_stride = alibi_slopes_batch_stride
        # block table
        self.total_q = total_q
        self.page_table_ptr = page_table_ptr
        self.page_table_batch_stride = page_table_batch_stride
        self.block_size = block_size
        self.k_page_stride = k_page_stride

    def args(self):
        return tuple(getattr(self, k) for k in self.__slots__)


def _get_varlen_fwd_config(
    args,
    head_size,
    use_varlen_split_d,
    is_paged,
    is_causal,
    window_size_left,
    window_size_right,
    is_alibi,
    is_softcap,
    is_dropout,
    total_q,
    batch_size,
    num_heads,
    max_seqlen_q,
):
    total_rows = total_q * num_heads
    num_sms = torch_device_fn.get_device_properties(
        runtime.device.name
    ).multi_processor_count
    avg_rows_per_sm = total_rows / num_sms
    avg_rows_per_batch = total_q / batch_size
    avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)

    if avg_rows_per_cta > 64:
        varlen_fwd_config_str = "mha_block_128"
    elif avg_rows_per_cta > 32:
        varlen_fwd_config_str = "mha_block_64"
    elif avg_rows_per_cta > 16:
        varlen_fwd_config_str = "mha_block_32"
    else:
        varlen_fwd_config_str = "mha_block_16"

    cfg = runtime.get_heuristic_config(varlen_fwd_config_str)
    cfg_params = {
        "BLOCK_M": cfg["BLOCK_M"](args),
        "BLOCK_N": cfg["BLOCK_N"](args),
        # FP8 QK reduction tiles require at least 32 elements. Padding is
        # internal to the kernel; the public tensor shape and scale stay intact.
        "BLOCK_K": max(32, triton.next_power_of_2(head_size)),
        "BLOCK_D": (
            64 if use_varlen_split_d else max(32, triton.next_power_of_2(head_size))
        ),
        "SPLIT_D": use_varlen_split_d,
        "num_warps": cfg["num_warps"](args),
        "num_stages": 1 if not is_paged else cfg["num_stages"](args),
    }

    if head_size not in (64, 128) and not is_paged:
        # Bound register use for new dimensions and avoid oversized query
        # tiles on ragged batches. Wide heads use D64 output slices.
        cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)

    # Hopper FP8 tuning favors wider KV tiles and eight warps. Keep unmeasured
    # feature combinations on the generic runtime configuration.
    is_standard_attention = (
        not is_causal and window_size_left == -1 and window_size_right == -1
    ) or (is_causal and window_size_left == -1 and window_size_right == 0)
    is_hopper = (
        runtime.device.vendor_name == "nvidia" and get_device_capability()[0] == 9
    )
    if (
        is_hopper
        and is_paged
        and head_size == 256
        and max_seqlen_q > 512
        and is_standard_attention
        and not is_alibi
        and not is_softcap
    ):
        # Keep QK shared by all output channels. Explicit FP16 promotion
        # enables Hopper WGMMA while retaining full FP32 QK accumulation.
        # One FP16 PV dot avoids the probability rounding and residual-dot
        # cost of the FP8 path; Q/K/V storage remains FP8.
        cfg_params.update(
            BLOCK_D=256,
            SPLIT_D=False,
            PV_DOT=_fp16_pv_dot,
            FULL_FP32_QK=True,
        )
        if avg_rows_per_cta > 16:
            cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)
    if (
        is_hopper
        and not is_paged
        and is_standard_attention
        and not is_alibi
        and not is_softcap
    ) and head_size in (64, 128):
        cfg_params.update(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128 if head_size == 64 and not is_causal else 64,
                "num_warps": 8,
                "num_stages": 1,
            }
        )
        # Small ragged queries benefit from more CTAs and lower register use.
        # Uniform packed batches take the dense fast path before this dispatch.
        use_small_q_config = (head_size == 64 and max_seqlen_q <= 64) or (
            head_size == 128 and max_seqlen_q <= 1024
        )
        if use_small_q_config:
            cfg_params.update(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "num_warps": 4,
                    "num_stages": 1,
                }
            )

        # Match the CTA footprint to the causal query length. Short queries
        # need more CTAs; long queries amortize masking over wider KV tiles.
        if head_size == 64 and is_causal:
            if max_seqlen_q <= 512:
                cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4)
            else:
                cfg_params.update(BLOCK_N=128, num_warps=4, num_stages=3)
        elif head_size == 128 and not is_causal and max_seqlen_q > 1024:
            # Pipeline the unmasked KV loop while retaining split-D to keep
            # the FP32 accumulator's register footprint bounded.
            cfg_params["num_stages"] = 3

    return cfg_params


def _normalize_varlen_window(
    max_seqlen_q,
    max_seqlen_k,
    alibi_slopes,
    is_causal,
    window_size_left,
    window_size_right,
):
    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    if window_size_right >= max_seqlen_k:
        window_size_right = -1

    is_local = window_size_left >= 0
    return is_causal, is_local, window_size_left, window_size_right


def mha_varlan_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    leftpad_k,
    page_table,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    p_dropout,
    softmax_scale,
    zero_tensors,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    gen,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    fp8_p_max=448.0,
    *,
    window_normalizer=_normalize_varlen_window,
    get_config=_get_varlen_fwd_config,
    pv_dot=_fp8_pv_dot,
    allow_group_swap=True,
    num_splits=0,
):
    if p_dropout != 0.0 or return_softmax:
        raise NotImplementedError(
            "dropout and attention probabilities are not supported"
        )
    if fp8_p_max != 448.0:
        raise ValueError("E4M3FN attention requires fp8_p_max=448.0")
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    if zero_tensors:
        raise NotImplementedError("zero_tensors is not supported")
    q_device = q.device
    q_dtype = q.dtype
    v_dtype = v.dtype
    # varlen forward also consumes FP8 Q/K/V and uses explicit
    # descale tensors, matching the dense kernel contract.
    assert (
        q_dtype in _FP8_DTYPES
    ), "W8A8 FP8 FlashAttention requires torch.float8_e4m3fn"
    assert (
        k.dtype == q_dtype
    ), "W8A8 FP8 FlashAttention expects q and k to use the same fp8 dtype"
    assert v_dtype == q_dtype, "W8A8 FP8 FlashAttention expects v to be fp8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"

    assert cu_seqlens_q.dtype == torch.int32
    assert cu_seqlens_q.is_contiguous()

    assert cu_seqlens_k.dtype == torch.int32
    assert cu_seqlens_k.is_contiguous()

    is_paged = page_table is not None
    if not is_paged:
        page_table = torch.empty((0, 0), device=q_device, dtype=torch.int32)

    # q shape: [total_q_tokens, num_heads, head_size]
    # k shape:
    #   paged_kv: [num_pages, block_size, num_heads_k, head_size]
    # batch_size, number of sentences
    total_q, num_heads, head_size = q.size()
    num_heads_k = k.size(2) if is_paged else k.size(1)
    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size > 0, "batch_size must be positive"
    block_size = k.size(1) if is_paged else 1
    num_pages = k.size(0) if is_paged else 0
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)
    k_batch_stride = k.stride(0)
    v_batch_stride = v.stride(0)

    assert k.size() == v.size()
    assert cu_seqlens_q.size() == (batch_size + 1,)
    assert cu_seqlens_k.size() == (batch_size + 1,)

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        # output stays high precision although V is FP8.
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (total_q, num_heads, head_size)

    if seqused_k is not None:
        assert seqused_k.is_contiguous()
        assert seqused_k.size() == (batch_size,)

    is_causal, is_local, window_size_left, window_size_right = window_normalizer(
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes,
        is_causal,
        window_size_left,
        window_size_right,
    )

    # Optimize all single-query sequences by swapping the query-group and sequence dimensions
    seqlenq_ngroups_swapped = (
        allow_group_swap
        and max_seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
    )
    q_groups = num_heads // num_heads_k
    if seqlenq_ngroups_swapped:
        logger.debug("Swapping query groups and sequence dimensions")
        q = (
            q.reshape((batch_size, num_heads_k, q_groups, head_size))
            .transpose(1, 2)
            .reshape(batch_size * q_groups, num_heads_k, head_size)
        )
        max_seqlen_q = q_groups
        num_heads = num_heads_k
        cu_seqlens_q = None
        q_batch_stride = q.stride(0) * max_seqlen_q
        k_batch_stride = k.stride(0)
        v_batch_stride = v.stride(0)
    else:
        q_batch_stride = 0
        k_batch_stride = 0
        v_batch_stride = 0
        o_batch_stride = 0

    total_q = q.size(0)

    assert leftpad_k is None, "leftpad_k is not supported."
    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"

    assert q.shape == (total_q, num_heads, head_size)
    if is_paged:
        assert k.shape == (num_pages, block_size, num_heads_k, head_size)
        assert v.shape == (num_pages, block_size, num_heads_k, head_size)
    assert k.stride() == v.stride()

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32) if head_size <= 192 else 256
    seqlen_q_rounded = round_multiple(max_seqlen_q, 128)
    seqlen_k_rounded = round_multiple(max_seqlen_k, 32)

    M_LOG2E = 1.4426950408889634074
    if softcap > 0.0:
        is_softcap = True
        adjusted_scale_softmax = softcap
        adjusted_softcap = softmax_scale / softcap
        adjusted_scale_softmax_log2e = softcap * M_LOG2E
    else:
        is_softcap = False
        adjusted_softcap = 0.0
        adjusted_scale_softmax = softmax_scale
        adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

    # Set alibi params
    if alibi_slopes is not None:
        assert alibi_slopes.device == q_device
        assert alibi_slopes.dtype in (torch.float,)
        assert alibi_slopes.stride(-1) == 1
        assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
            batch_size,
            num_heads,
        )
        alibi_slopes_batch_stride = (
            alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
        )
        is_alibi = True
    else:
        alibi_slopes_batch_stride = 0
        is_alibi = False

    # Prepare params to kernel
    with torch_device_fn.device(q_device):
        if out is not None:
            out_ = out
            if seqlenq_ngroups_swapped:
                out = torch.empty_like(q, dtype=torch.bfloat16)
        else:
            out_ = None
            out = torch.empty_like(q, dtype=torch.bfloat16)

        if seqlenq_ngroups_swapped:
            o_batch_stride = out.stride(0) * max_seqlen_q

        lse = torch.empty((num_heads, total_q), dtype=torch.float, device=q_device)
        num_kv_splits = max(1, num_splits)
        is_h20_or_h800 = runtime.device.vendor_name == "nvidia" and any(
            name in torch_device_fn.get_device_properties(runtime.device.name).name
            for name in ("H20", "H800")
        )
        use_short_prefill = (
            (
                (
                    is_h20_or_h800
                    and get_device_capability()[0] == 9
                    and is_paged
                    and block_size in (16, 32)
                    and head_size == 256
                    and q_groups in (4, 8)
                    and batch_size > 1
                    and max_seqlen_q > 512
                )
                and num_splits == 0
                and num_kv_splits == 1
                and seqused_k is not None
                and not allow_group_swap
                and not is_alibi
                and not is_softcap
            )
            and not is_local
            and is_causal
            and window_size_left == -1
            and window_size_right == 0
        )
        use_grouped_decode = (
            runtime.device.vendor_name == "nvidia"
            and get_device_capability()[0] == 9
            and is_paged
            and block_size in (16, 32)
            and head_size == 256
            and max_seqlen_q == 1
            and q_groups in (4, 8, 16, 32)
            and seqused_k is not None
            and not allow_group_swap
            and not is_alibi
            and not is_softcap
        ) and not is_local
        use_hybrid_decode = (
            (
                runtime.device.vendor_name == "nvidia"
                and get_device_capability()[0] == 9
                and is_paged
                and block_size in (16, 32)
                and head_size == 256
                and max_seqlen_q > 1
                and (
                    (batch_size >= 32 and batch_size * num_heads_k >= 48)
                    or (
                        batch_size >= 3 and 2 * batch_size * max_seqlen_q >= 3 * total_q
                    )
                )
                and q_groups in (4, 8, 16, 32)
                and seqused_k is not None
                and not allow_group_swap
                and not is_alibi
                and not is_softcap
            )
            and not is_local
            and window_size_left == -1
            and (
                (is_causal and window_size_right == 0)
                or (not is_causal and window_size_right == -1)
            )
            and num_splits == 0
        )
        if use_short_prefill:
            # Decode, short prefill, and long prefill have disjoint output rows.
            use_hybrid_decode = True
        decode_chunks = triton.cdiv(q_groups, 8)
        decode_bn = 128
        decode_splits = num_kv_splits
        decode_workers = num_kv_splits
        if use_grouped_decode or use_hybrid_decode:
            num_sms = torch_device_fn.get_device_properties(
                runtime.device.name
            ).multi_processor_count
            if num_splits == 0:
                if use_hybrid_decode and q_groups in (4, 8):
                    # Preserve the validated schedule for existing mixed requests.
                    decode_splits = 32
                else:
                    decode_splits = _select_decode_splits(
                        batch_size,
                        num_heads_k,
                        decode_chunks,
                        max_seqlen_k,
                        decode_bn,
                        num_sms,
                    )
                if use_grouped_decode:
                    num_kv_splits = decode_splits
            decode_workers = min(
                decode_splits,
                triton.cdiv(2 * num_sms, batch_size * num_heads_k * decode_chunks),
            )
        if num_kv_splits > 1 and not (use_grouped_decode and decode_workers == 1):
            partial_out = torch.empty(
                (num_kv_splits, total_q, num_heads, head_size),
                dtype=torch.float32,
                device=q_device,
            )
            partial_lse = torch.empty(
                (num_kv_splits, num_heads, total_q),
                dtype=torch.float32,
                device=q_device,
            )

        # Keep the shared backend ABI without allocating unused inference state.
        is_dropout = False
        p_dropout = 1.0
        p_dropout_in_uint8_t = 255
        rp_dropout = 1.0
        philox_args = None
        p = None

        q_nblocks = triton.cdiv(max_seqlen_q, 128)
        k_nblocks = triton.cdiv(max_seqlen_k, 128)
        # varlen kernels use [B, H, block] descales indexed by
        # request id and local block id.  Paged KV uses the same logical blocks.
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        if use_grouped_decode or use_hybrid_decode:
            if use_hybrid_decode:
                if decode_workers > 1:
                    decode_partials = torch.empty(
                        (decode_splits, batch_size, num_heads, head_size),
                        dtype=torch.float32,
                        device=q_device,
                    )
                    decode_partial_lse = torch.empty(
                        (decode_splits, num_heads, batch_size),
                        dtype=torch.float32,
                        device=q_device,
                    )
                else:
                    decode_partials, decode_partial_lse = out, lse
            else:
                decode_partials = partial_out if decode_workers > 1 else out
                decode_partial_lse = partial_lse if decode_workers > 1 else lse
            _flash_varlen_grouped_decode_fwd[
                (batch_size, num_heads_k * decode_chunks, decode_workers)
            ](
                q,
                k,
                v,
                cu_seqlens_q,
                seqused_k,
                page_table,
                q_descale,
                k_descale,
                v_descale,
                out,
                lse,
                decode_partials,
                decode_partial_lse,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                page_table.stride(0),
                q_descale.stride(0),
                q_descale.stride(1),
                k_descale.stride(0),
                k_descale.stride(1),
                k_descale.stride(2),
                v_descale.stride(0),
                v_descale.stride(1),
                v_descale.stride(2),
                out.stride(0),
                out.stride(1),
                total_q,
                H=num_heads,
                HK=num_heads_k,
                G=q_groups,
                PAGE=block_size,
                SPLITS=decode_splits,
                SCALE=softmax_scale,
                BN=decode_bn,
                WORKERS=decode_workers,
                num_warps=4,
                num_stages=1,
                PART_BY_BATCH=use_hybrid_decode,
                PART_ROWS=batch_size,
            )
            if decode_workers > 1:
                _flash_varlen_grouped_decode_combine[(batch_size, num_heads, 4)](
                    out,
                    lse,
                    decode_partials,
                    decode_partial_lse,
                    cu_seqlens_q,
                    seqused_k,
                    total_q,
                    out.stride(0),
                    out.stride(1),
                    H=num_heads,
                    D=head_size,
                    SPLITS=decode_splits,
                    BS=triton.next_power_of_2(decode_splits),
                    BN=decode_bn,
                    num_warps=4,
                    PART_BY_BATCH=use_hybrid_decode,
                    PART_ROWS=batch_size,
                )
            if use_grouped_decode:
                unused = None
                return out, q, k, v, lse, philox_args, unused, p

        if use_short_prefill:
            short_heads = q_groups
            short_rows = 128
            short_capacity = 32
            short_token_rows = short_rows // short_heads
            short_splits = _select_decode_splits(
                batch_size,
                num_heads_k,
                triton.cdiv(short_capacity, short_token_rows),
                max_seqlen_k,
                64,
                num_sms,
            )
            short_partial = torch.empty(
                (short_splits, batch_size, short_capacity, num_heads, head_size),
                dtype=torch.float32,
                device=q_device,
            )
            short_lse = torch.empty(
                (short_splits, batch_size, num_heads, short_capacity),
                dtype=torch.float32,
                device=q_device,
            )
            _short_fused_fwd[
                (
                    triton.cdiv(short_capacity, short_token_rows),
                    batch_size,
                    num_heads_k * short_splits,
                )
            ](
                q,
                k,
                v,
                cu_seqlens_q,
                seqused_k,
                page_table,
                q_descale,
                k_descale,
                v_descale,
                short_partial,
                short_lse,
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                page_table.stride(0),
                q_descale.stride(0),
                q_descale.stride(1),
                q_descale.stride(2),
                k_descale.stride(0),
                k_descale.stride(1),
                k_descale.stride(2),
                v_descale.stride(0),
                v_descale.stride(1),
                v_descale.stride(2),
                B=batch_size,
                SPLITS=short_splits,
                MAX_Q=short_capacity,
                H=num_heads,
                G=q_groups,
                PAGE=block_size,
                SCALE=softmax_scale,
                FUSED_HEADS=short_heads,
                ROWS=short_rows,
                BN=64,
                STAGES=1,
                num_warps=8,
                num_stages=1,
            )
            _short_fused_combine[(short_capacity, batch_size, num_heads)](
                short_partial,
                short_lse,
                cu_seqlens_q,
                out,
                lse,
                out.stride(0),
                out.stride(1),
                total_q,
                B=batch_size,
                H=num_heads,
                D=head_size,
                SPLITS=short_splits,
                MAX_Q=short_capacity,
                num_warps=4,
                num_stages=1,
            )

        kernel_out = partial_out[0] if num_kv_splits > 1 else out
        kernel_lse = partial_lse[0] if num_kv_splits > 1 else lse
        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            kernel_out,  # o_ptr,
            p,  # p_ptr,
            kernel_lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            kernel_out.stride(-3),  # o_row_stride,
            kernel_out.stride(-2),  # o_head_stride,
            q_batch_stride,  # q_batch_stride,
            k_batch_stride,  # k_batch_stride,
            v_batch_stride,  # v_batch_stride,
            o_batch_stride,  # o_batch_stride,
            cu_seqlens_q is not None,  # is_cu_seqlens_q,
            cu_seqlens_q,  # cu_seqlens_q_ptr,
            seqused_k is None,  # is_cu_seqlens_k,
            cu_seqlens_k,  # cu_seqlens_k_ptr,
            seqused_k is not None,  # is_seqused_k,
            seqused_k,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            k_batch_size,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            max_seqlen_q,  # seqlen_q,
            max_seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            True,  # use_fa3_fp8_scales,
            fp8_p_max,  # fp8_p_max,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            total_q,  # total_q,
            page_table,  # page_table_ptr,
            page_table_batch_stride,  # page_table_batch_stride,
            block_size,  # block_size,
            k.stride(0) if is_paged else 0,  # k_page_stride,
        )

        if runtime.device.vendor_name == "iluvatar":
            params.k_ptr = k.view(k.shape[0], k.shape[1], -1)
            params.v_ptr = v.view(v.shape[0], v.shape[1], -1)
        logger.debug("kernel: flash_varlen_fwd")
        # Packed wide heads use D64 output slices. Paged D256 instead
        # keeps QK shared with a full-width PV accumulator.
        use_varlen_split_d = head_size > 64 and not is_paged
        args = tuple(getattr(params, k) for k in params.__slots__)

        cfg_params = get_config(
            args,
            head_size,
            use_varlen_split_d,
            is_paged,
            is_causal,
            window_size_left,
            window_size_right,
            is_alibi,
            is_softcap,
            is_dropout,
            total_q,
            batch_size,
            num_heads,
            max_seqlen_q,
        )
        if (
            (cfg_params.get("FULL_FP32_QK", False))
            and batch_size == 1
            and num_heads == 4
            and num_heads_k == 1
            and is_causal
            and total_q >= 4096
            and num_kv_splits == 1
        ):
            # Wider query tiles improve KV reuse for long single-request MQA.
            cfg_params.update(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=1)
        if is_h20_or_h800 and cfg_params.get("FULL_FP32_QK", False):
            # H20 and H800 use smaller query tiles and asynchronous KV loads.
            cfg_params.update(
                BLOCK_M=64,
                BLOCK_N=64,
                num_warps=4,
                PAGED_PIPELINE_STAGES=2,
            )
        if (
            runtime.device.vendor_name == "nvidia"
            and "H800"
            in torch_device_fn.get_device_properties(runtime.device.name).name
            and is_paged
            and head_size == 256
            and max_seqlen_q > 512
            and cfg_params.get("FULL_FP32_QK", False)
        ):
            cfg_params["PAGED_PIPELINE_STAGES"] = 1
            cfg_params["DELAY_PAGED_V"] = True
        cfg_params["REVERSE_QUERY_TILES"] = (
            (is_h20_or_h800 and cfg_params.get("FULL_FP32_QK", False))
            and is_causal
            and not is_local
        )
        cfg_params["SKIP_SINGLE_Q"] = use_hybrid_decode
        cfg_params["SKIP_SHORT_Q"] = use_short_prefill
        args = cfg_params.pop("args", args)
        num_d_splits = triton.cdiv(head_size, cfg_params["BLOCK_D"])
        if num_kv_splits > 1:
            cfg_params["NUM_KV_SPLITS"] = num_kv_splits
        grid = (
            triton.cdiv(max_seqlen_q, cfg_params["BLOCK_M"]),
            batch_size * num_kv_splits,
            num_heads * num_d_splits,
        )
        max_query_tiles = triton.cdiv(total_q, cfg_params["BLOCK_M"]) + batch_size - 1
        max_tiles_per_sequence = triton.cdiv(max_seqlen_q, cfg_params["BLOCK_M"])
        use_compact_schedule = (
            runtime.device.vendor_name == "nvidia"
            and get_device_capability()[0] == 9
            and is_paged
            and head_size == 256
            and batch_size >= 32
            and cu_seqlens_q is not None
            and batch_size * max_tiles_per_sequence >= 8 * max_query_tiles
            and triton.next_power_of_2(batch_size)
            * triton.next_power_of_2(max_tiles_per_sequence)
            <= 16384
        )
        if use_compact_schedule:
            prefix = torch.empty((batch_size + 1,), dtype=torch.int32, device=q_device)
            batch_map = torch.empty(
                (max_query_tiles,), dtype=torch.int32, device=q_device
            )
            local_tile_map = torch.empty_like(batch_map)
            query_tile_prefix_kernel[(1,)](
                cu_seqlens_q,
                prefix,
                batch_map,
                local_tile_map,
                B=batch_size,
                BLOCK_M=cfg_params["BLOCK_M"],
                BLOCK_B=triton.next_power_of_2(batch_size),
                BLOCK_T=triton.next_power_of_2(
                    triton.cdiv(max_seqlen_q, cfg_params["BLOCK_M"])
                ),
                num_warps=4,
            )
            # Sum(ceil(q_len/M)) <= ceil(total_q/M) + B - 1, without a
            # device-to-host read or changing the recorded input dimensions.
            grid = (
                triton.cdiv(total_q, cfg_params["BLOCK_M"]) + batch_size - 1,
                num_kv_splits,
                num_heads * num_d_splits,
            )
            cfg_params.update(
                QUERY_TILE_PREFIX=prefix,
                QUERY_BATCH_MAP=batch_map,
                QUERY_LOCAL_TILE_MAP=local_tile_map,
                COMPACT_QUERY_SCHEDULE=True,
            )
        kernel = cfg_params.pop("kernel", flash_varlen_fwd_kernel)[grid]

        logger.debug("Running flash_varlen_fwd_kernel with config: %s", cfg_params)
        if not cfg_params.pop("split_short_k", True):
            short_k_modes = (False,)
        elif max_seqlen_k <= 128:
            short_k_modes = (True,)
        elif (
            not is_paged
            and seqused_k is None
            and k.shape[0] == batch_size * max_seqlen_k
        ):
            # A full packed allocation at the declared maximum is uniform.
            short_k_modes = (False,)
        else:
            # Each CTA runs in exactly one variant, selected by its KV length.
            short_k_modes = (False, True)
        selected_pv_dot = cfg_params.pop("PV_DOT", pv_dot)
        if cfg_params.get("FULL_FP32_QK", False):
            # This path uses the same precise PV calculation for every KV
            # length, so a second short-sequence launch is unnecessary.
            cfg_params["UNIFIED_K_LENGTHS"] = True
            short_k_modes = (False,)
        use_scale_aware_acc = (
            runtime.device.vendor_name == "nvidia"
            and "H800"
            in torch_device_fn.get_device_properties(runtime.device.name).name
            and is_paged
            and head_size == 256
            and max_seqlen_q > 512
            and cfg_params.get("FULL_FP32_QK", False)
        )
        if use_scale_aware_acc:
            # The device guard selects the bounded direct-accumulator path.
            # All other scales retain their value and use residual-corrected PV.
            selected_pv_dot = _fp16_pv_dot_residual
            # Preserve the grouped loop with its original CTA guard for the
            # single-request GQA/Page32 regime. Other eligible layouts share one guard check.
            use_group_scales = (
                batch_size == 1
                and q_groups == 8
                and block_size == 32
                and cfg_params["BLOCK_N"] == 64
            )
            cfg_params["GROUP_FP8_SCALES"] = use_group_scales
            precheck_scale_guard = not use_group_scales
            scale_safe_flags = None
            if precheck_scale_guard:
                # Fresh stream-local workspace is captured by CUDA Graph and
                # overwritten by the check kernel before either main kernel reads it.
                scale_safe_flags = torch.empty(
                    (batch_size, num_heads_k), dtype=torch.int32, device=q_device
                )
                _check_varlen_v_scales[(batch_size, num_heads_k)](
                    v_descale,
                    cu_seqlens_k,
                    seqused_k,
                    scale_safe_flags,
                    v_descale.stride(0),
                    v_descale.stride(1),
                    v_descale.stride(2),
                    max_seqlen_k,
                    HK=num_heads_k,
                    HAS_SEQUSED_K=seqused_k is not None,
                    SCAN_BLOCKS=triton.next_power_of_2(triton.cdiv(max_seqlen_k, 128)),
                    num_warps=4,
                    num_stages=1,
                )
        scale_acc_modes = (True, False) if use_scale_aware_acc else (None,)
        for precise_short_k in short_k_modes:
            for scale_acc_fast in scale_acc_modes:
                launch_config = cfg_params
                if scale_acc_fast is not None:
                    launch_config = dict(
                        cfg_params,
                        SCALE_AWARE_GUARD=True,
                        SCALE_AWARE_ACC=scale_acc_fast,
                        SCALE_SAFE_FLAGS=scale_safe_flags,
                        PRECHECK_SCALE_GUARD=precheck_scale_guard,
                        SCALE_SCAN_BLOCKS=(
                            triton.next_power_of_2(triton.cdiv(max_seqlen_k, 128))
                            if use_group_scales
                            else 1
                        ),
                    )
                kernel(
                    *args,
                    PRECISE_SHORT_K=precise_short_k,
                    PV_DOT=selected_pv_dot,
                    **launch_config,
                )

        if num_kv_splits > 1:
            flash_varlen_splitkv_combine_kernel[
                (total_q * num_heads, triton.cdiv(head_size, 64))
            ](
                out,
                lse,
                partial_out,
                partial_lse,
                total_q,
                out.stride(0),
                out.stride(1),
                H=num_heads,
                D=head_size,
                NUM_KV_SPLITS=num_kv_splits,
                BLOCK_S=triton.next_power_of_2(num_kv_splits),
                BLOCK_D=64,
                num_warps=4,
            )

        if seqlenq_ngroups_swapped:
            out = out.reshape(
                batch_size, max_seqlen_q, num_heads_k, head_size
            ).transpose(1, 2)
            if out_ is not None:
                out_.view(batch_size, num_heads_k, max_seqlen_q, head_size).copy_(out)
                out = out_
            else:
                out = out.reshape(batch_size, num_heads_k * max_seqlen_q, head_size)
            lse = lse.reshape(num_heads_k, batch_size, max_seqlen_q)
            lse = lse.reshape(num_heads_k * max_seqlen_q, batch_size)

        unused = None
    return out, q, k, v, lse, philox_args, unused, p


def mha_fwd(
    q,
    k,
    v,
    out,
    alibi_slopes,
    p_dropout,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    softcap,
    return_softmax,
    disable_splitkv=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    fp8_p_max=448.0,
):
    if p_dropout != 0.0 or return_softmax:
        raise NotImplementedError(
            "dropout and attention probabilities are not supported"
        )
    if fp8_p_max != 448.0:
        raise ValueError("E4M3FN attention requires fp8_p_max=448.0")
    CHECK_DEVICE(q), CHECK_DEVICE(k), CHECK_DEVICE(v)
    q_dtype = q.dtype
    v_dtype = v.dtype
    q_device = q.device
    # The internal dense forward consumes FP8 Q/K/V
    # tensors and the output keeps the caller's high-precision output dtype.
    assert (
        q_dtype in _FP8_DTYPES
    ), "W8A8 FP8 FlashAttention requires torch.float8_e4m3fn"
    assert (
        k.dtype == q_dtype
    ), "W8A8 FP8 FlashAttention expects q and k to use the same fp8 dtype"
    assert v_dtype == q_dtype, "W8A8 FP8 FlashAttention expects v to be fp8"
    assert q.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert k.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    assert v.stride(-1) == 1, "Input tensor must have contiguous last dimension"
    batch_size, seqlen_q, num_heads, head_size = q.size()
    _, seqlen_k, num_heads_k, _ = k.size()

    # Check output shape
    if out is not None:
        assert out.stride(-1) == 1
        # Output stays high precision instead of inheriting the FP8 V dtype.
        assert out.dtype in _HIGH_PRECISION_DTYPES
        assert out.size() == (batch_size, seqlen_q, num_heads, head_size)
        CHECK_DEVICE(out)

    assert (
        head_size % 8 == 0
    ), "head_size must be a multiple of 8, this is ensured by padding!"
    assert (
        num_heads % num_heads_k == 0
    ), "Number of heads in key/value must divide number of heads in query"
    if window_size_left >= seqlen_k:
        window_size_left = -1
    if window_size_right >= seqlen_k:
        window_size_right = -1
    if seqlen_q == 1 and alibi_slopes is None:
        is_causal = False
    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0
    is_local = window_size_left >= 0 and window_size_right >= 0

    seqlenq_ngroups_swapped = (
        seqlen_q == 1
        and alibi_slopes is None
        and num_heads > num_heads_k
        and window_size_left < 0
        and window_size_right < 0
    )
    q_groups = num_heads // num_heads_k

    if seqlenq_ngroups_swapped:
        logger.debug("q_kg swapped.")
        q = q.reshape(batch_size, num_heads_k, q_groups, head_size).transpose(1, 2)
        seqlen_q = q_groups
        num_heads = num_heads_k

    round_multiple = lambda x, m: (x + m - 1) // m * m
    head_size_rounded = round_multiple(head_size, 32)
    seqlen_q_rounded = round_multiple(seqlen_q, 128)
    seqlen_k_rounded = round_multiple(seqlen_k, 32)

    assert (
        head_size <= 256
    ), "FlashAttention forward only supports head dimension at most 256"
    assert head_size == head_size_rounded, "head_size must be rounded to 32"

    def splits_heuristic(num_tasks, num_sms, n_blocks):
        # splits when wave efficiency is low
        n_waves = triton.cdiv(num_tasks, num_sms)
        eff = (num_tasks / num_sms) / n_waves
        if eff > 0.8 or n_waves > 1:
            return 1

        min_blocks_per_split = 2
        best_splits = min(
            triton.cdiv(n_blocks, min_blocks_per_split),
            int(math.floor(1.0 / eff)),
            num_sms,
        )

        return best_splits

    with torch_device_fn.device(q_device):
        # Set softmax params
        lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float, device=q_device
        )

        if out is not None:
            if seqlenq_ngroups_swapped:
                out = out.reshape(
                    batch_size, num_heads_k, q_groups, head_size
                ).transpose(1, 2)
        else:
            # allocate a high-precision output even though all
            # three inputs are FP8.
            out = torch.empty_like(q, dtype=torch.bfloat16)

        # Keep the shared backend ABI without allocating unused inference state.
        is_dropout = False
        p_dropout = 1.0
        p_dropout_in_uint8_t = 255
        rp_dropout = 1.0
        philox_args = None
        p = None

        q_nblocks = triton.cdiv(seqlen_q, 128)
        k_nblocks = triton.cdiv(seqlen_k, 128)
        # Per-block descale tensors are required by the FP8 kernel contract.
        q_descale = _normalize_dense_descale(
            q_descale, batch_size, num_heads, q_nblocks, q_device, "q_descale"
        )
        k_descale = _normalize_dense_descale(
            k_descale, batch_size, num_heads_k, k_nblocks, q_device, "k_descale"
        )
        v_descale = _normalize_dense_descale(
            v_descale, batch_size, num_heads_k, k_nblocks, q_device, "v_descale"
        )

        M_LOG2E = 1.4426950408889634074
        if softcap > 0.0:
            is_softcap = True
            adjusted_scale_softmax = softcap
            adjusted_softcap = softmax_scale / softcap
            adjusted_scale_softmax_log2e = softcap * M_LOG2E
        else:
            is_softcap = False
            adjusted_softcap = 0.0
            adjusted_scale_softmax = softmax_scale
            adjusted_scale_softmax_log2e = softmax_scale * M_LOG2E

        # Set alibi params
        if alibi_slopes is not None:
            assert alibi_slopes.device == q_device
            assert alibi_slopes.dtype in (torch.float,)
            assert alibi_slopes.stride(-1) == 1
            assert alibi_slopes.shape == (num_heads,) or alibi_slopes.shape == (
                batch_size,
                num_heads,
            )
            alibi_slopes_batch_stride = (
                alibi_slopes.stride(0) if alibi_slopes.ndim == 2 else 0
            )
            is_alibi = True
        else:
            alibi_slopes_batch_stride = 0
            is_alibi = False

        # ONLY EVEN_K IS SUPPORTED
        assert head_size == head_size_rounded

        # Do kernel dispatching
        def dispatch(B, H, Q, K, D, params):
            num_sms = torch_device_fn.get_device_properties(
                "cuda"
            ).multi_processor_count
            # For D128, use the split-D dense kernel and partition the output
            # dimension across two D64 CTAs to reduce register pressure in
            # flash_fwd_kernel. The split-KV path does not use split-D.
            use_split_d = D == 128
            # For short sequences, split-KV combine and temporary-tensor overhead
            # usually outweigh the benefit, so S <= 512 uses the dense path.
            disable_splitkv1 = disable_splitkv or seqlen_q <= 512

            # Try splitkv
            if ((not use_split_d)) and not is_local and not disable_splitkv1:
                BM = block_m_splitkv_heuristic(D)
                n_tasks = B * H * triton.cdiv(seqlen_q, BM)
                BN = block_n_splitkv_heuristic(D)
                n_blocks = triton.cdiv(seqlen_k, BN)
                n_splits = splits_heuristic(n_tasks, num_sms, n_blocks)

                if n_splits > 1:
                    logger.debug("kernel: flash_fwd_splitkv")
                    lse_splits = torch.empty(
                        (n_splits, B, H, Q), dtype=torch.float, device=q_device
                    )
                    out_splits = torch.empty(
                        (n_splits, B, H, Q, D), dtype=torch.float, device=q_device
                    )
                    grid = lambda args: (
                        triton.cdiv(Q, args["BLOCK_M"]),
                        n_splits,
                        B * H,
                    )
                    splitkv_kernel = flash_fwd_splitkv_kernel[grid]
                    params.o_ptr = out_splits
                    params.softmax_lse_ptr = lse_splits
                    extra_args = {"blocks_per_split": triton.cdiv(n_blocks, n_splits)}
                    kernel = splitkv_kernel(*params.args(), **extra_args)

                    if D >= 128:
                        BLOCK_M = 4
                    elif D >= 64:
                        BLOCK_M = 8
                    else:
                        BLOCK_M = 16
                    BLOCK_K = triton.next_power_of_2(D)
                    grid = lambda args: (triton.cdiv(B * H * Q, BLOCK_M),)
                    combine_kernel = flash_fwd_splitkv_combine_kernel[grid]
                    combine_args = {
                        "out_ptr": out,
                        "lse_ptr": lse,
                        "head_size": head_size,
                        "out_split_stride": out_splits.stride(0),
                        "lse_split_stride": lse_splits.stride(0),
                        "out_b_stride": out.stride(0),
                        "out_s_stride": out.stride(-3),
                        # Head stride is the second-to-last dim of the BSHD view;
                        # stride(-1) is the contiguous head-size stride.
                        "out_h_stride": out.stride(-2),
                        "out_splits_ptr": out_splits,
                        "lse_splits_ptr": lse_splits,
                        "n_splits": n_splits,
                        "BLOCK_M": BLOCK_M,
                        "BLOCK_K": BLOCK_K,
                        "q_total": B * H * Q,
                        "MAX_N_SPLITS": triton.next_power_of_2(n_splits),
                        "num_heads": H,
                        "seqlen_q": Q,
                    }
                    combine_kernel(**combine_args)
                    return kernel

            # Last option: flash_fwd
            logger.debug("kernel: flash_fwd")
            grid = lambda args: (
                triton.cdiv(Q, args["BLOCK_M"]),
                H * B,
                2 if use_split_d else 1,
            )
            kernel = flash_fwd_kernel[grid]
            # For D128 split-D, QK still uses the full BLOCK_K with head_dim=128.
            # BLOCK_D=64 applies only to PV and output stores; D64 or non-split-D
            # retains the original BLOCK_K.
            extra_args = {
                "BLOCK_D": 64 if use_split_d else triton.next_power_of_2(D),
                "SPLIT_D": use_split_d,
            }
            kernel = kernel(*params.args(), **extra_args)
            return kernel

        params = fwd_params(
            q,  # q_ptr,
            k,  # k_ptr,
            v,  # v_ptr,
            out,  # o_ptr,
            p,  # p_ptr,
            lse,  # softmax_lse_ptr,
            q.stride(-3),  # q_row_stride,
            k.stride(-3),  # k_row_stride,
            v.stride(-3),  # v_row_stride,
            q.stride(-2),  # q_head_stride,
            k.stride(-2),  # k_head_stride,
            v.stride(-2),  # v_head_stride,
            out.stride(-3),  # o_row_stride,
            out.stride(-2),  # o_head_stride,
            q.stride(0),  # q_batch_stride,
            k.stride(0),  # k_batch_stride,
            v.stride(0),  # v_batch_stride,
            out.stride(0),  # o_batch_stride,
            False,  # is_cu_seqlens_q,
            None,  # cu_seqlens_q_ptr,
            False,  # is_cu_seqlens_k,
            None,  # cu_seqlens_k_ptr,
            False,  # is_seqused_k,
            None,  # seqused_k_ptr,
            # sizes
            batch_size,  # b,
            0,  # bk,
            num_heads,  # h,
            num_heads_k,  # hk,
            num_heads // num_heads_k,  # h_hk_ratio,
            seqlen_q,  # seqlen_q,
            seqlen_k,  # seqlen_k,
            seqlen_q_rounded,  # seqlen_q_rounded,
            seqlen_k_rounded,  # seqlen_k_rounded,
            head_size,  # d,
            head_size_rounded,  # d_rounded,
            # scaling factors
            is_softcap,
            adjusted_softcap,  # softcap,
            adjusted_scale_softmax,  # scale_softmax,
            adjusted_scale_softmax_log2e,  # scale_softmax_log2,
            # pass Q/K/V per-block descale metadata to every
            # compute kernel so QK and PV can both use FP8 tensor cores.
            q_descale,  # q_descale_ptr,
            k_descale,  # k_descale_ptr,
            v_descale,  # v_descale_ptr,
            q_descale.stride(0),  # q_descale_batch_stride,
            q_descale.stride(1),  # q_descale_head_stride,
            q_descale.stride(2),  # q_descale_block_stride,
            k_descale.stride(0),  # k_descale_batch_stride,
            k_descale.stride(1),  # k_descale_head_stride,
            k_descale.stride(2),  # k_descale_block_stride,
            v_descale.stride(0),  # v_descale_batch_stride,
            v_descale.stride(1),  # v_descale_head_stride,
            v_descale.stride(2),  # v_descale_block_stride,
            True,  # use_fa3_fp8_scales,
            fp8_p_max,  # fp8_p_max,
            # dropout
            is_dropout,
            p_dropout,
            rp_dropout,
            p_dropout_in_uint8_t,
            philox_args,
            return_softmax,
            # causal and swa
            is_causal,  # is_causal,
            is_local,  # is_local,
            window_size_left,  # window_size_left,
            window_size_right,  # window_size_right,
            seqlenq_ngroups_swapped,  # seqlenq_ngroups_swapped,
            False,  # is_paged,
            # alibi
            is_alibi,  #
            alibi_slopes,  # alibi_slopes_ptr,
            alibi_slopes_batch_stride,  # alibi_slopes_batch_stride,
            # block table params
            0,  # total_q,
            None,  # page_table_ptr,
            0,  # page_table_batch_stride,
            0,  # block_size,
            0,  # k_page_stride,
        )

        # Move TxD to last dims for correct stride in Triton tt.load
        if runtime.device.vendor_name == "iluvatar":
            params.q_ptr = q.transpose(1, 2)
            params.k_ptr = k.transpose(1, 2)
            params.v_ptr = v.transpose(1, 2)
        dispatch(batch_size, num_heads, seqlen_q, seqlen_k, head_size, params)

        if seqlenq_ngroups_swapped:
            out = out.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            q = q.transpose(1, 2).reshape(
                (batch_size, 1, num_heads_k * seqlen_q, head_size)
            )
            lse = lse.reshape((batch_size, num_heads_k * seqlen_q, 1))

        unused = None

    return out, q, k, v, lse, philox_args, unused, p


def _flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Compatibility arguments from the shared FlashAttention API.
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
    *,
    fp8_dtypes=_FP8_DTYPES,
    allow_gqa=False,
    allow_scheduler_args=False,
    use_dense=True,
    varlen_fwd=mha_varlan_fwd,
):
    """Compute variable-length FlashAttention-2 with block-wise FP8 Q/K/V.

    Args:
        q: Packed FP8 query tensor in ``[total_q, heads, head_dim]`` layout.
        k: Packed FP8 key tensor in ``[total_k, heads, head_dim]`` layout.
        v: Packed FP8 value tensor with the same shape as ``k``.
        max_seqlen_q: Maximum query sequence length in the batch.
        cu_seqlens_q: Cumulative query sequence lengths with shape ``[batch + 1]``.
        max_seqlen_k: Maximum key sequence length in the batch.
        cu_seqlens_k: Cumulative key sequence lengths with shape ``[batch + 1]``.
        q_descale: Query descales normalized to ``[batch, q_heads, q_blocks]``.
        k_descale: Key descales normalized to ``[batch, kv_heads, kv_blocks]``.
        v_descale: Value descales normalized to ``[batch, kv_heads, kv_blocks]``.
        softmax_scale: Score scale. Defaults to ``1 / sqrt(head_dim)``.
        causal: Whether to apply a causal mask.
        out: Optional packed FP16/BF16 output tensor.
        scheduler_metadata: Optional backend-specific scheduling hint. NVIDIA
            FA2 computes its own schedule and does not read this FA3 metadata.
        num_splits: Number of KV partitions on NVIDIA. Zero uses the default
            schedule, which may partition single-token requests in a mixed
            batch. Explicit positive values retain that many logical KV
            partitions. Multiple partitions use a stable FP32 merge.

    The public signature matches ``flash_attn_varlen_func``. Descales are
    applied per logical 128-token block. The returned tensor uses ``out.dtype``
    when supplied and BF16 otherwise. Q/K/V must use ``torch.float8_e4m3fn``.
    Backend entry points select head grouping and launch strategy.
    Head dimensions must be multiples of 8 between 8 and 256, inclusive.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported by this inference path")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported")
    if q_v is not None:
        raise NotImplementedError("q_v is not supported")
    if scheduler_metadata is not None and not allow_scheduler_args:
        raise NotImplementedError("scheduler arguments are not supported")
    if s_aux is not None:
        raise NotImplementedError("attention sinks are not supported")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("context parallel attention is not supported")
    if fa_version != 2:
        raise RuntimeError("Only FA2 is implemented.")
    if not isinstance(num_splits, int) or isinstance(num_splits, bool):
        raise TypeError("num_splits must be a Python integer")
    if num_splits < 0:
        raise ValueError("num_splits must be nonnegative")
    if num_splits > 0 and not allow_scheduler_args:
        raise RuntimeError("num_splits > 0 is not implemented.")
    assert (
        cu_seqlens_k is not None or seqused_k is not None
    ), "cu_seqlens_k or seqused_k must be provided"
    assert (
        cu_seqlens_k is None or seqused_k is None
    ), "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert (
        block_table is None or seqused_k is not None
    ), "seqused_k must be provided if block_table is provided"
    if seqused_k is not None and block_table is None:
        raise NotImplementedError("seqused_k without a paged KV cache is not supported")

    if not isinstance(max_seqlen_q, int) or not isinstance(max_seqlen_k, int):
        raise TypeError("max_seqlen_q and max_seqlen_k must be Python integers")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise NotImplementedError("empty sequences are not supported")
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, heads, head_dim]")
    expected_kv_ndim = 4 if block_table is not None else 3
    if k.ndim != expected_kv_ndim or v.ndim != expected_kv_ndim:
        raise ValueError("k and v rank does not match the selected cache layout")
    if (
        q.dtype != torch.float8_e4m3fn
        or q.dtype not in fp8_dtypes
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("q, k, and v must all have dtype torch.float8_e4m3fn")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise NotImplementedError("q, k, and v must be contiguous in head_dim")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    head_size = q.shape[-1]
    if not 8 <= head_size <= 256 or head_size % 8 != 0:
        raise NotImplementedError("head_dim must be a multiple of 8 between 8 and 256")
    if k.shape[-1] != head_size:
        raise ValueError("q, k, and v must have the same head_dim")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.device != q.device:
        raise ValueError("cu_seqlens_q must be an int32 tensor on q.device")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.dtype != torch.int32 or cu_seqlens_k.device != q.device
    ):
        raise ValueError("cu_seqlens_k must be an int32 tensor on q.device")
    if seqused_k is not None and (
        seqused_k.dtype != torch.int32 or seqused_k.device != q.device
    ):
        raise ValueError("seqused_k must be an int32 tensor on q.device")
    if block_table is not None and (
        block_table.dtype != torch.int32 or block_table.device != q.device
    ):
        raise ValueError("block_table must be an int32 tensor on q.device")
    if out is not None:
        if out.shape != q.shape or out.device != q.device:
            raise ValueError("out must match q shape and device")
        if out.dtype not in _HIGH_PRECISION_DTYPES:
            raise TypeError("out must have dtype torch.float16 or torch.bfloat16")
        if out.stride(-1) != 1:
            raise NotImplementedError("out must be contiguous in head_dim")
        if out.data_ptr() in (q.data_ptr(), k.data_ptr(), v.data_ptr()):
            raise ValueError("out must not alias q, k, or v")

    num_heads_k = k.shape[2] if block_table is not None else k.shape[1]
    if q.shape[1] != num_heads_k and not allow_gqa:
        raise NotImplementedError("GQA is not supported by this W8A8 path")
    if num_heads_k <= 0 or q.shape[1] % num_heads_k:
        raise ValueError("query heads must be divisible by KV heads")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])

    batch_size = cu_seqlens_q.numel() - 1
    assert batch_size > 0, "batch_size must be positive"
    if cu_seqlens_q.ndim != 1 or not cu_seqlens_q.is_contiguous():
        raise ValueError("cu_seqlens_q must be a contiguous 1D tensor")
    if cu_seqlens_k is not None and (
        cu_seqlens_k.ndim != 1
        or cu_seqlens_k.numel() != batch_size + 1
        or not cu_seqlens_k.is_contiguous()
    ):
        raise ValueError("cu_seqlens_k must be contiguous with shape [batch + 1]")
    if seqused_k is not None and (
        seqused_k.ndim != 1
        or seqused_k.numel() != batch_size
        or not seqused_k.is_contiguous()
    ):
        raise ValueError("seqused_k must be contiguous with shape [batch]")
    if block_table is not None and (
        block_table.ndim != 2
        or block_table.shape[0] != batch_size
        or block_table.stride(-1) != 1
    ):
        raise ValueError(
            "block_table must be contiguous in its last dimension with shape [batch, pages]"
        )
    # Other head dimensions use the padded varlen kernel below; this guard
    # restricts only the tuned dense shortcut, not public dimension support.
    uniform_nonpaged = (
        use_dense
        and q.shape[1] == num_heads_k
        and num_splits == 0
        and head_size in (64, 128)
        and block_table is None
        and cu_seqlens_k is not None
        and seqused_k is None
        and not return_softmax_lse
        and max_seqlen_q % 128 == 0
        and max_seqlen_k % 128 == 0
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and (out is None or out.is_contiguous())
        and q.shape[0] == batch_size * max_seqlen_q
        and k.shape[0] == batch_size * max_seqlen_k
    )
    if uniform_nonpaged:
        # Tile-aligned uniform packed batches are dense tensors without padding.
        # Reuse the tuned dense kernels, including their split-KV paths.
        q_dense = q.view(batch_size, max_seqlen_q, q.shape[1], q.shape[2])
        k_dense = k.view(batch_size, max_seqlen_k, k.shape[1], k.shape[2])
        v_dense = v.view(batch_size, max_seqlen_k, v.shape[1], v.shape[2])
        out_dense = (
            None
            if out is None
            else out.view(batch_size, max_seqlen_q, out.shape[1], out.shape[2])
        )
        dense_result = mha_fwd(
            q_dense,
            k_dense,
            v_dense,
            out_dense,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            False,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            fp8_p_max=448.0,
        )
        return out if out is not None else dense_result[0].view(q.shape)

    cu_seqlens_k_arg = (
        torch.empty_like(cu_seqlens_q) if cu_seqlens_k is None else cu_seqlens_k
    )
    varlen_kwargs = {}
    if allow_scheduler_args:
        # Keep the logical query heads intact so their individual descales
        # stay aligned. FA3 scheduler metadata is advisory, not attention data.
        varlen_kwargs.update(num_splits=num_splits, allow_group_swap=False)
    result = varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k_arg,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        real_window_size[0],
        real_window_size[1],
        softcap,
        False,
        None,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        fp8_p_max=448.0,
        **varlen_kwargs,
    )
    return (result[0], result[4]) if return_softmax_lse else result[0]


def flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Compatibility arguments from the shared FlashAttention API.
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
    """Compute block-scaled E4M3FN MHA/GQA/MQA on NVIDIA Hopper or newer.

    Q/K/V must use ``torch.float8_e4m3fn`` with FP32 block descales.
    ``scheduler_metadata`` is an opaque FA3 scheduling hint and is not consumed
    by this FA2 implementation. ``num_splits`` controls actual KV partitioning.
    """
    if runtime.device.vendor_name != "nvidia" or get_device_capability()[0] < 9:
        raise NotImplementedError("W8A8 FP8 attention requires NVIDIA Hopper or newer")
    return _flash_attn_varlen_func_w8a8_fp8(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        q_v=q_v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table,
        return_softmax_lse=return_softmax_lse,
        out=out,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        num_splits=num_splits,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        fa_version=fa_version,
        allow_gqa=True,
        allow_scheduler_args=True,
    )
