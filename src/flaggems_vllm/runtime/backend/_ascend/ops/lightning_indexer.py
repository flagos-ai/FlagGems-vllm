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

"""TLE implementation of the Ascend paged-attention Lightning Indexer.

Dispatch is split into three semantic paths: short K directly materializes the full
index set; single-token long K builds 512-key proposals in parallel along K while
multi-token long K builds them along query rows; both share the Stage2 merge tree.

Each proposal is stored as two FP32 words: the score and a bit-packed INT32 key
index. Intermediate merges must keep both; only the final TopK step emits the
index alone. Sorting, merging, and unpacking reuse the FlagTree PR #1065 CustomOps.
"""

import logging

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle
    from triton.experimental.tle.language.dsa.ascend.custom_ops import (
        SORT_IMPL_BASE,
        SORT_IMPL_S4096_K129_512,
    )
except (AttributeError, ImportError) as exc:
    tle = None
    SORT_IMPL_BASE = tl.constexpr(0)
    SORT_IMPL_S4096_K129_512 = tl.constexpr(1)
    _PR1065_IMPORT_ERROR = exc
else:
    _PR1065_IMPORT_ERROR = None

pipe = al.PIPE
_COMPILER_OPTIONS = {
    "use_bytecode": True,  # CustomOp bitcode is provided by the installed PR toolchain
    "enable_auto_bind_sub_block": False,
    "enable_ubuf_saving": True,
}


@triton.jit
def lightning_indexer_tnd_pa_stage1_kernel(
    q_ptr,
    k_ptr,
    weights_ptr,
    wsp_ptr,
    out_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    block_table_ptr,
    stride_qt,
    stride_qn,
    stride_kbn,
    stride_wt,
    stride_out_0: tl.int64,
    stride_block_table_b,
    query_head_num: tl.constexpr,
    head_dim: tl.constexpr,
    REQ_NUM: tl.constexpr,
    Q_TILE: tl.constexpr = 4,
    C_TILE: tl.constexpr = 4,
    K_TILE: tl.constexpr = 128,
    TOPK: tl.constexpr = 2048,
    MB: tl.constexpr = 3,
    OUTPUT_RAW_SCORES: tl.constexpr = False,
    GROUPWISE_FP32_REDUCTION: tl.constexpr = False,
    HOST_SPECIALIZE_CAUSAL_SCHEDULE: tl.constexpr = True,
    SINGLE_REQ_EXTRA_CORES: tl.constexpr = 0,
    SINGLE_REQ_PREFIX_ROWS: tl.constexpr = 0,
):
    """Build sorted 512-key proposal lists for multi-token long-K query rows.

    Each MIX program owns ``Q_TILE`` rows. The Cube computes QK in 128-key blocks;
    two AIV sub-blocks reduce over heads with weights and mask invalid tail lanes
    with ``-inf``. The score word is kept for the Stage2 merge.
    """

    T_TILE: tl.constexpr = 2  # one Cube pass covers two query rows
    # one local sort group always covers 512 keys
    wsp_nstride: tl.constexpr = C_TILE * K_TILE
    wsp_mstride: tl.constexpr = (
        T_TILE * query_head_num * wsp_nstride
    )  # workspace stride for two rows of QK
    m_coef: tl.constexpr = (
        Q_TILE // al.sub_vec_num()
    )  # query rows actually owned by each AIV sub-block
    q_step: tl.constexpr = Q_TILE // T_TILE  # Cube batches needed per query tile
    N_TILE: tl.constexpr = 16  # reduce heads in groups of 16
    n_step: tl.constexpr = (
        query_head_num // N_TILE
    )  # reductions needed to cover all heads
    core_nums = tl.num_programs(
        0
    )  # derive the cross-round query stride from the actual grid
    core_id = tl.program_id(0)
    b = 0
    t_i = (
        core_id * Q_TILE
    )  # assign query tiles by grid first, then map to cumulative TND requests
    pre_len_q = 0
    cur_len_q = tl.load(seq_lens_q_ptr)
    seq_len_q = cur_len_q
    cur_len_k = tl.load(seq_lens_k_ptr)
    query_stride = Q_TILE * core_nums
    schedule_end_q = cur_len_q
    if REQ_NUM == 1:
        if HOST_SPECIALIZE_CAUSAL_SCHEDULE:
            if SINGLE_REQ_EXTRA_CORES != 0:
                if core_id < SINGLE_REQ_EXTRA_CORES:
                    t_i = core_id * Q_TILE
                    query_stride = SINGLE_REQ_EXTRA_CORES * Q_TILE
                    schedule_end_q = SINGLE_REQ_PREFIX_ROWS
                else:
                    short_core_id = core_id - SINGLE_REQ_EXTRA_CORES
                    t_i = SINGLE_REQ_PREFIX_ROWS + short_core_id * Q_TILE
                    query_stride = (core_nums - SINGLE_REQ_EXTRA_CORES) * Q_TILE
            # the two-segment path uses host constexprs, dropping device-side div/mod
        else:
            query_tiles = (cur_len_q + Q_TILE - 1) // Q_TILE
            base_tiles = query_tiles // core_nums
            extra_tiles = query_tiles % core_nums
            if extra_tiles != 0:
                if core_id < extra_tiles:
                    t_i = core_id * Q_TILE
                    query_stride = extra_tiles * Q_TILE
                    schedule_end_q = extra_tiles * (base_tiles + 1) * Q_TILE
                else:
                    short_core_id = core_id - extra_tiles
                    prefix_tiles = extra_tiles * (base_tiles + 1)
                    t_i = (prefix_tiles + short_core_id) * Q_TILE
                    query_stride = (core_nums - extra_tiles) * Q_TILE
                    schedule_end_q = query_tiles * Q_TILE
            # the generic path keeps device-side two-pool scheduling for any tile count
    tmp_buf = tl.zeros([wsp_nstride * 4], dtype=tl.float32)
    sorted_pairs_buf = tl.zeros([2 * wsp_nstride], dtype=tl.float32)
    for i in tl.static_range(MB):
        al.sync_block_set(
            "vector", "cube", i, pipe.PIPE_MTE2, pipe.PIPE_FIX
        )  # mark every ring slot Cube-writable before the first round
    db_flag = 0
    if REQ_NUM > 1:
        while t_i >= cur_len_q and b < REQ_NUM - 1:
            q_tail = seq_len_q % Q_TILE
            if q_tail:
                t_i -= (
                    Q_TILE - q_tail
                )  # back up to the next segment real start when a TND tail block is short of Q_TILE
            b += 1
            pre_len_q = cur_len_q
            cur_len_q = tl.load(seq_lens_q_ptr + b)
            seq_len_q = cur_len_q - pre_len_q
            cur_len_k = tl.load(seq_lens_k_ptr + b)
    if t_i >= cur_len_q:
        b = REQ_NUM
    while b < REQ_NUM:
        act_len_k = (
            cur_len_k - (cur_len_q - t_i) + 1
        )  # causal-aligned length for sparse_mode=3
        k_blk_cnt = (
            (act_len_k + K_TILE - 1) // K_TILE if act_len_k + Q_TILE - 1 > TOPK else 0
        )  # rows already covered by TopK need no sorting
        q_block_0 = tl.load(
            q_ptr
            + t_i * stride_qt
            + (tl.arange(0, T_TILE * query_head_num) * stride_qn)[:, None]
            + tl.arange(0, head_dim)[None, :]
        )
        q_block_1 = tl.load(
            q_ptr
            + t_i * stride_qt
            + T_TILE * query_head_num * head_dim
            + (tl.arange(0, T_TILE * query_head_num) * stride_qn)[:, None]
            + tl.arange(0, head_dim)[None, :]
        )

        weight_lanes = tl.arange(0, m_coef * query_head_num)
        weight_rows = t_i + m_coef * al.sub_vec_id() + weight_lanes // query_head_num
        weight_offsets = (
            weight_rows * stride_wt + weight_lanes % query_head_num
        )  # skip inter-token padding via the real row stride
        weight_block = tl.load(
            weights_ptr + weight_offsets,
            mask=weight_rows < cur_len_q,
            other=0.0,
        ).to(tl.float32)

        q_itr = 0
        q_itr_b = 0
        for t in tl.static_range(m_coef):
            tile_offset = m_coef * al.sub_vec_id() + t
            if t_i + tile_offset < cur_len_q:
                q_itr += 1
                if act_len_k + tile_offset <= TOPK:
                    q_itr_b += 1
        for k_t in tl.range(0, k_blk_cnt, C_TILE):
            core_offset = (
                q_step * wsp_mstride * (core_nums * (db_flag % MB) + core_id)
            )  # MB ring slots are isolated per program
            remain_k_blk_cnt = k_blk_cnt - k_t
            k_itr = C_TILE if remain_k_blk_cnt >= C_TILE else remain_k_blk_cnt
            al.sync_block_wait(
                "vector", "cube", (db_flag % MB), pipe.PIPE_MTE2, pipe.PIPE_FIX
            )  # wait for AIV consumption before the Cube reuses a slot
            for k_i in range(k_itr):
                actual_k_block_i = tl.load(
                    block_table_ptr + b * stride_block_table_b + k_t + k_i
                )  # paged indirection through the block table
                k_block_ptr = tl.make_block_ptr(
                    base=k_ptr + actual_k_block_i * stride_kbn,
                    shape=(K_TILE, head_dim),
                    strides=(head_dim, 1),
                    offsets=(0, 0),
                    block_shape=(K_TILE, head_dim),
                    order=(1, 0),
                )
                k_block = tl.load(k_block_ptr)
                qk_block = tl.dot(q_block_0, tl.trans(k_block))
                qk_block_0 = libdevice.relu(qk_block)
                tl.store(
                    wsp_ptr
                    + core_offset
                    + k_i * K_TILE
                    + (tl.arange(0, T_TILE * query_head_num) * wsp_nstride)[:, None]
                    + tl.arange(0, K_TILE)[None, :],
                    qk_block_0,
                )

                qk_block = tl.dot(q_block_1, tl.trans(k_block))
                qk_block_1 = libdevice.relu(qk_block)
                tl.store(
                    wsp_ptr
                    + core_offset
                    + k_i * K_TILE
                    + wsp_mstride
                    + (tl.arange(0, T_TILE * query_head_num) * wsp_nstride)[:, None]
                    + tl.arange(0, K_TILE)[None, :],
                    qk_block_1,
                )

            al.sync_block_set(
                "cube", "vector", (db_flag % MB), pipe.PIPE_FIX, pipe.PIPE_MTE2
            )  # publish this slot QK data to the AIV
            in_offset = core_offset + tl.arange(0, N_TILE * wsp_nstride)
            if OUTPUT_RAW_SCORES:
                out_offsets = (
                    t_i * stride_out_0 + k_t * K_TILE + tl.arange(0, wsp_nstride)
                )
            else:
                out_offsets = (
                    t_i * stride_out_0
                    + 2 * k_t * K_TILE
                    + tl.arange(0, 2 * wsp_nstride)
                )
            al.sync_block_wait(
                "cube", "vector", (db_flag % MB), pipe.PIPE_FIX, pipe.PIPE_MTE2
            )  # wait for the Cube write before the AIV reads
            for q_i in range(q_itr_b, q_itr):
                qk_slice = tl.load(
                    wsp_ptr
                    + in_offset
                    + (
                        m_coef * al.sub_vec_id() * query_head_num
                        + (q_i * n_step + 0) * N_TILE
                    )
                    * wsp_nstride
                )
                weight = al.extract_slice(
                    weight_block, ((q_i * n_step + 0) * N_TILE,), (N_TILE,), (1,)
                )[:, None]
                qk_contribution = tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight
                if GROUPWISE_FP32_REDUCTION:
                    tmp_reduce_res_block = tl.sum(qk_contribution, 0)
                else:
                    qk_scale = qk_contribution

                qk_slice = tl.load(
                    wsp_ptr
                    + in_offset
                    + (
                        m_coef * al.sub_vec_id() * query_head_num
                        + (q_i * n_step + 1) * N_TILE
                    )
                    * wsp_nstride
                )
                weight = al.extract_slice(
                    weight_block, ((q_i * n_step + 1) * N_TILE,), (N_TILE,), (1,)
                )[:, None]
                qk_contribution = tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight
                if GROUPWISE_FP32_REDUCTION:
                    tmp_reduce_res_block += tl.sum(qk_contribution, 0)
                else:
                    qk_scale += qk_contribution

                if n_step == 4:
                    qk_slice = tl.load(
                        wsp_ptr
                        + in_offset
                        + (
                            m_coef * al.sub_vec_id() * query_head_num
                            + (q_i * n_step + 2) * N_TILE
                        )
                        * wsp_nstride
                    )
                    weight = al.extract_slice(
                        weight_block, ((q_i * n_step + 2) * N_TILE,), (N_TILE,), (1,)
                    )[:, None]
                    qk_contribution = (
                        tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight
                    )
                    if GROUPWISE_FP32_REDUCTION:
                        tmp_reduce_res_block += tl.sum(qk_contribution, 0)
                    else:
                        qk_scale += qk_contribution

                    qk_slice = tl.load(
                        wsp_ptr
                        + in_offset
                        + (
                            m_coef * al.sub_vec_id() * query_head_num
                            + (q_i * n_step + 3) * N_TILE
                        )
                        * wsp_nstride
                    )
                    weight = al.extract_slice(
                        weight_block, ((q_i * n_step + 3) * N_TILE,), (N_TILE,), (1,)
                    )[:, None]
                    qk_contribution = (
                        tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight
                    )
                    if GROUPWISE_FP32_REDUCTION:
                        tmp_reduce_res_block += tl.sum(qk_contribution, 0)
                    else:
                        qk_scale += qk_contribution

                if not GROUPWISE_FP32_REDUCTION:
                    tmp_reduce_res_block = tl.sum(qk_scale, 0)
                if k_t + C_TILE >= k_blk_cnt:
                    tmp_reduce_res_block = tl.where(
                        k_t * K_TILE + tl.arange(0, wsp_nstride)
                        < act_len_k + m_coef * al.sub_vec_id() + q_i,
                        tmp_reduce_res_block,
                        float("-inf"),
                    )  # invalid lanes of the tail group must sort after all real scores
                if OUTPUT_RAW_SCORES:
                    tl.store(
                        out_ptr
                        + out_offsets
                        + (m_coef * al.sub_vec_id() + q_i) * stride_out_0,
                        tmp_reduce_res_block,
                    )
                else:
                    sorted_pairs_buf = tle.dsa.ascend.raw(
                        "sort_1d_pack",
                        tl.reshape(tmp_reduce_res_block, (wsp_nstride,)),
                        tmp_buf,
                        True,
                        wsp_nstride,
                        k_t
                        * K_TILE,  # the proposal index must include the 512-key group offset
                        SORT_IMPL_BASE,
                        out=sorted_pairs_buf,
                    )
                    tl.store(
                        out_ptr
                        + out_offsets
                        + (m_coef * al.sub_vec_id() + q_i) * stride_out_0,
                        sorted_pairs_buf,
                    )

            al.sync_block_set(
                "vector", "cube", (db_flag % MB), pipe.PIPE_MTE2, pipe.PIPE_FIX
            )  # return the ring slot after the AIV finishes
            db_flag += 1
        t_i += query_stride  # single-request load-balancing path claims the next tile by its pool stride
        if REQ_NUM > 1:
            while t_i >= cur_len_q and b < REQ_NUM - 1:
                q_tail = seq_len_q % Q_TILE
                if q_tail:
                    t_i -= (
                        Q_TILE - q_tail
                    )  # drop the hole left by the previous tail when crossing requests
                b += 1
                pre_len_q = cur_len_q
                cur_len_q = tl.load(seq_lens_q_ptr + b)
                seq_len_q = cur_len_q - pre_len_q
                cur_len_k = tl.load(seq_lens_k_ptr + b)
        if (REQ_NUM == 1 and t_i >= schedule_end_q) or t_i >= cur_len_q:
            b = REQ_NUM
    for i in tl.static_range(MB):
        al.sync_block_wait(
            "vector", "cube", i, pipe.PIPE_MTE2, pipe.PIPE_FIX
        )  # wait for all AIV consumers to return their slots before exiting


@triton.jit
def lightning_indexer_tnd_pa_prefill_sort_4096_top512_kernel(
    score_ptr,
    proposal_ptr,
    output_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    stride_score_row: tl.int64,
    stride_proposal_row: tl.int64,
    stride_output_row: tl.int64,
    PROPOSAL_GROUPS: tl.constexpr,
    REQ_NUM: tl.constexpr,
    OUTPUT_INDICES: tl.constexpr,
    TOPK: tl.constexpr = 512,
):
    """Trim a 4096-score GM segment down to a Top512 proposal."""

    SEGMENT_KEYS: tl.constexpr = 4096
    PAIR_WORDS: tl.constexpr = 2 * TOPK
    SORT_TMP_WORDS: tl.constexpr = 3 * 4 * TOPK * 2
    tl.static_assert(TOPK == 512)

    task_id = tl.program_id(0)
    t_i = task_id // PROPOSAL_GROUPS
    proposal_group_id = task_id % PROPOSAL_GROUPS
    b = 0
    cur_len_q = tl.load(seq_lens_q_ptr)
    if REQ_NUM > 1:
        while b < REQ_NUM - 1 and t_i >= cur_len_q:
            b += 1
            cur_len_q = tl.load(seq_lens_q_ptr + b)
    cur_len_k = tl.load(seq_lens_k_ptr + b)
    act_len_k = cur_len_k - (cur_len_q - t_i) + 1
    segment_begin = proposal_group_id * SEGMENT_KEYS

    sort_tmp = tl.zeros([SORT_TMP_WORDS], dtype=tl.float32)
    sorted_pairs = tl.zeros([PAIR_WORDS], dtype=tl.float32)
    final_values = tl.zeros([TOPK], dtype=tl.float32)
    final_indices = tl.zeros([TOPK], dtype=tl.int32)
    if OUTPUT_INDICES and act_len_k <= TOPK:
        lanes = tl.arange(0, TOPK)
        direct_indices = tl.where(lanes < act_len_k, lanes, -1)
        tl.store(output_ptr + t_i * stride_output_row + lanes, direct_indices)
    elif segment_begin < act_len_k:
        lanes = tl.arange(0, SEGMENT_KEYS)
        segment_offsets = segment_begin + lanes
        segment_scores = tl.load(
            score_ptr + t_i * stride_score_row + segment_offsets,
            mask=segment_offsets < act_len_k,
            other=float("-inf"),
        )
        sorted_pairs = tle.dsa.ascend.raw(
            "sort_1d_pack",
            segment_scores,
            sort_tmp,
            True,
            TOPK,
            segment_begin,
            SORT_IMPL_S4096_K129_512,
            out=sorted_pairs,
        )
        if OUTPUT_INDICES:
            final_values, final_indices = tle.dsa.ascend.raw(
                "unpack_sort",
                sorted_pairs,
                TOPK,
                out=[final_values, final_indices],
            )
            tl.store(
                output_ptr + t_i * stride_output_row + tl.arange(0, TOPK),
                final_indices,
            )
        else:
            tl.store(
                proposal_ptr
                + t_i * stride_proposal_row
                + proposal_group_id * PAIR_WORDS
                + tl.arange(0, PAIR_WORDS),
                sorted_pairs,
            )


@triton.jit
def lightning_indexer_tnd_pa_prefill_sort_merge_8192_top512_kernel(
    score_ptr,
    output_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    stride_score_row: tl.int64,
    stride_output_row: tl.int64,
    REQ_NUM: tl.constexpr,
    HOST_SPECIALIZE_SINGLE_REQ_LENGTHS: tl.constexpr = False,
    SINGLE_REQ_QUERY_ROWS: tl.constexpr = 0,
    SINGLE_REQ_KEY_TOKENS: tl.constexpr = 0,
    TOPK: tl.constexpr = 512,
):
    """Process 8192 scores in one base sort and emit the global Top512."""

    TOTAL_KEYS: tl.constexpr = 8192
    PAIR_WORDS: tl.constexpr = 2 * TOPK
    SORT_TMP_WORDS: tl.constexpr = 4 * TOTAL_KEYS
    tl.static_assert(TOPK == 512)

    t_i = tl.program_id(0)  # one row per Vector program so proposals are never shared
    b = 0
    if HOST_SPECIALIZE_SINGLE_REQ_LENGTHS:
        cur_len_q = SINGLE_REQ_QUERY_ROWS
        cur_len_k = SINGLE_REQ_KEY_TOKENS
    else:
        cur_len_q = tl.load(seq_lens_q_ptr)
        if REQ_NUM > 1:
            while b < REQ_NUM - 1 and t_i >= cur_len_q:
                b += 1
                cur_len_q = tl.load(seq_lens_q_ptr + b)
        cur_len_k = tl.load(seq_lens_k_ptr + b)
    act_len_k = cur_len_k - (cur_len_q - t_i) + 1

    output_lanes = tl.arange(0, TOPK)
    if act_len_k <= TOPK:
        direct_indices = tl.where(output_lanes < act_len_k, output_lanes, -1)
        tl.store(output_ptr + t_i * stride_output_row + output_lanes, direct_indices)
    else:
        sort_tmp = tl.zeros([SORT_TMP_WORDS], dtype=tl.float32)
        score_lanes = tl.arange(0, TOTAL_KEYS)
        scores = tl.load(
            score_ptr + t_i * stride_score_row + score_lanes,
            mask=score_lanes < act_len_k,
            other=float("-inf"),
        )
        top_pairs = tl.zeros([PAIR_WORDS], dtype=tl.float32)
        top_pairs = tle.dsa.ascend.raw(
            "sort_1d_pack",
            scores,
            sort_tmp,
            True,
            TOPK,
            0,
            SORT_IMPL_BASE,
            out=top_pairs,
        )  # one global sort avoids two proposals plus an extra two-way merge
        final_values = tl.zeros([TOPK], dtype=tl.float32)
        final_indices = tl.zeros([TOPK], dtype=tl.int32)
        final_values, final_indices = tle.dsa.ascend.raw(
            "unpack_sort",
            top_pairs,
            TOPK,
            out=[final_values, final_indices],
        )
        tl.store(
            output_ptr + t_i * stride_output_row + output_lanes,
            final_indices,
        )


@triton.jit
def lightning_indexer_tnd_pa_decode_stage1_kernel(
    q_ptr,
    k_ptr,
    weights_ptr,
    wsp_ptr,
    out_ptr,
    block_table_ptr,
    stride_qt,
    stride_qn,
    stride_kbn,
    stride_wt,
    stride_out_0: tl.int64,
    stride_block_table_b,
    query_head_num: tl.constexpr,
    head_dim: tl.constexpr,
    ACT_LEN_K: tl.constexpr,
    K_BLK_CNT: tl.constexpr,
    C_TILE: tl.constexpr = 4,
    K_TILE: tl.constexpr = 128,
    TOPK: tl.constexpr = 2048,
):
    """Build one compact proposal list per 512-key group for a single decode token.

    The host resolves the token's single owning request and passes its block-table row.
    The grid uses the real K-group count, covering all keys even when they outnumber AI Cores.
    """
    T_TILE: tl.constexpr = 2  # keep the same Cube tile shape as the generic Stage1
    Q_TILE: tl.constexpr = (
        4  # keep the verified static MIX layout; only token 0 is reduced
    )
    N_TILE: tl.constexpr = 16  # reduce 16 query heads per vector pass
    wsp_nstride: tl.constexpr = (
        C_TILE * K_TILE
    )  # each program covers one 512-key proposal group
    wsp_mstride: tl.constexpr = T_TILE * query_head_num * wsp_nstride
    m_coef: tl.constexpr = Q_TILE // al.sub_vec_num()
    q_step: tl.constexpr = Q_TILE // T_TILE
    n_step: tl.constexpr = query_head_num // N_TILE

    core_id = tl.program_id(0)
    k_t = (
        core_id * C_TILE
    )  # one K group per program restores parallelism for a single token
    tmp_buf = tl.zeros([wsp_nstride * 4], dtype=tl.float32)
    sorted_pairs_buf = tl.zeros([2 * wsp_nstride], dtype=tl.float32)

    al.sync_block_set(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # declare the slot writable before the first Cube store
    q_rows = tl.arange(0, T_TILE * query_head_num)
    q_block_0 = tl.load(
        q_ptr + (q_rows * stride_qn)[:, None] + tl.arange(0, head_dim)[None, :]
    )
    q_block_1 = tl.load(
        q_ptr
        + T_TILE * query_head_num * head_dim
        + (q_rows * stride_qn)[:, None]
        + tl.arange(0, head_dim)[None, :]
    )

    weight_lanes = tl.arange(0, m_coef * query_head_num)
    weight_rows = m_coef * al.sub_vec_id() + weight_lanes // query_head_num
    weight_offsets = (
        weight_rows * stride_wt + weight_lanes % query_head_num
    )  # use real strides; token rows must not be assumed densely packed
    weight_block = tl.load(
        weights_ptr + weight_offsets,
        mask=weight_rows < 1,
        other=0.0,
    ).to(tl.float32)

    core_offset = q_step * wsp_mstride * core_id
    remain_k_blk_cnt = K_BLK_CNT - k_t
    k_itr = C_TILE if remain_k_blk_cnt >= C_TILE else remain_k_blk_cnt
    al.sync_block_wait(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # wait for a writable slot before the Cube overwrites the workspace
    for k_i in range(k_itr):
        actual_k_block_i = tl.load(
            block_table_ptr + k_t + k_i
        )  # pointer already targets the active request block-table row
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + actual_k_block_i * stride_kbn,
            shape=(K_TILE, head_dim),
            strides=(head_dim, 1),
            offsets=(0, 0),
            block_shape=(K_TILE, head_dim),
            order=(1, 0),
        )
        k_block = tl.load(k_block_ptr)
        qk_block_0 = libdevice.relu(tl.dot(q_block_0, tl.trans(k_block)))
        tl.store(
            wsp_ptr
            + core_offset
            + k_i * K_TILE
            + (tl.arange(0, T_TILE * query_head_num) * wsp_nstride)[:, None]
            + tl.arange(0, K_TILE)[None, :],
            qk_block_0,
        )
        qk_block_1 = libdevice.relu(tl.dot(q_block_1, tl.trans(k_block)))
        tl.store(
            wsp_ptr
            + core_offset
            + k_i * K_TILE
            + wsp_mstride
            + (tl.arange(0, T_TILE * query_head_num) * wsp_nstride)[:, None]
            + tl.arange(0, K_TILE)[None, :],
            qk_block_1,
        )

    al.sync_block_set(
        "cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_MTE2
    )  # publish the complete 512-key QK tile
    in_offset = core_offset + tl.arange(0, N_TILE * wsp_nstride)
    out_offsets = 2 * k_t * K_TILE + tl.arange(0, 2 * wsp_nstride)
    al.sync_block_wait(
        "cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_MTE2
    )  # wait for the full QK tile before the AIV reduction

    q_itr = 0
    q_itr_b = 0
    for t in tl.static_range(m_coef):
        tile_offset = m_coef * al.sub_vec_id() + t
        if tile_offset < 1:
            q_itr += 1
            if ACT_LEN_K + tile_offset <= TOPK:
                q_itr_b += 1
    for q_i in range(q_itr_b, q_itr):
        qk_slice = tl.load(wsp_ptr + in_offset)
        weight = al.extract_slice(weight_block, (0,), (N_TILE,), (1,))[:, None]
        qk_scale = tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight

        qk_slice = tl.load(wsp_ptr + in_offset + N_TILE * wsp_nstride)
        weight = al.extract_slice(weight_block, (N_TILE,), (N_TILE,), (1,))[:, None]
        qk_scale += tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight

        if n_step == 4:
            qk_slice = tl.load(wsp_ptr + in_offset + 2 * N_TILE * wsp_nstride)
            weight = al.extract_slice(weight_block, (2 * N_TILE,), (N_TILE,), (1,))[
                :, None
            ]
            qk_scale += tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight

            qk_slice = tl.load(wsp_ptr + in_offset + 3 * N_TILE * wsp_nstride)
            weight = al.extract_slice(weight_block, (3 * N_TILE,), (N_TILE,), (1,))[
                :, None
            ]
            qk_scale += tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight

        scores = tl.sum(qk_scale, 0)
        if k_t + C_TILE >= K_BLK_CNT:
            scores = tl.where(
                k_t * K_TILE + tl.arange(0, wsp_nstride) < ACT_LEN_K,
                scores,
                float("-inf"),
            )  # invalid tail lanes must leave TopK contention with -inf in decode
        sorted_pairs_buf = tle.dsa.ascend.raw(
            "sort_1d_pack",
            tl.reshape(scores, (wsp_nstride,)),
            tmp_buf,
            True,
            wsp_nstride,
            k_t
            * K_TILE,  # the shared sort emits device-side indices with the global offset
            SORT_IMPL_BASE,
            out=sorted_pairs_buf,
        )
        tl.store(out_ptr + out_offsets, sorted_pairs_buf)

    al.sync_block_set(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # explicitly close the MIX sync edge
    al.sync_block_wait(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # wait for the final proposal write-out


@triton.jit
def lightning_indexer_tnd_pa_direct_indices_kernel(
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    o_ptr,
    REQ_NUM: tl.constexpr,
    TOP_K: tl.constexpr = 512,
):
    """Materialize the full index set directly when TopK covers all valid keys.

    sparse_mode=3 uses causal alignment: row ``t`` has
    ``key_end - (query_end - t) + 1`` valid keys. The full TopK set is simply
    ``[0, act_len_k)`` followed by ``-1`` padding; no scores are needed.
    """
    core_nums = tl.num_programs(0)
    t_i = tl.program_id(0)
    b = 0
    cur_len_q = tl.load(seq_lens_q_ptr)
    cur_len_k = tl.load(seq_lens_k_ptr)
    if REQ_NUM > 1:
        while b < REQ_NUM - 1 and t_i >= cur_len_q:
            b += 1
            cur_len_q = tl.load(seq_lens_q_ptr + b)
            cur_len_k = tl.load(seq_lens_k_ptr + b)
    if t_i >= cur_len_q:
        b = REQ_NUM

    lanes = tl.arange(0, TOP_K)
    while b < REQ_NUM:
        act_len_k = (
            cur_len_k - (cur_len_q - t_i) + 1
        )  # same causal valid length as the long-K paths
        indices = tl.where(
            lanes < act_len_k, lanes, -1
        )  # with TopK covering everything, sequential indices are the exact set
        tl.store(o_ptr + t_i * TOP_K + lanes, indices)

        t_i += core_nums
        if REQ_NUM > 1:
            while b < REQ_NUM - 1 and t_i >= cur_len_q:
                b += 1
                cur_len_q = tl.load(seq_lens_q_ptr + b)
                cur_len_k = tl.load(seq_lens_k_ptr + b)
        if t_i >= cur_len_q:
            b = REQ_NUM


@triton.jit
def lightning_indexer_tnd_pa_stage2_top512_merge_pairs_kernel(
    proposal_ptr,
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    output_proposal_ptr,
    stride_proposal_row: tl.int64,
    task_base: tl.int32,
    OUTPUT_GROUPS: tl.constexpr,
    ROUND: tl.constexpr,
    REQ_NUM: tl.constexpr,
    INPUT_SPAN: tl.constexpr = 512,
    TOP_K: tl.constexpr = 512,
    OUTPUT_INDICES: tl.constexpr = False,
):
    """Execute one layer of the TopK=512 pair-preserving GM merge tree.

    Each program owns one ``(query row, output group)`` task and merges one to four
    adjacent sorted lists; only the final layer may drop scores and emit raw indices.
    """
    V_I: tl.constexpr = 2
    MERGE_WAYS: tl.constexpr = (
        4  # four-way merge is the largest verified fan-in under the UB budget
    )
    LIST_WORDS: tl.constexpr = (
        V_I * TOP_K
    )  # score/index_bits must stay paired across GM layers
    MERGE_PROPS: tl.constexpr = MERGE_WAYS * TOP_K
    MERGE_WORDS: tl.constexpr = V_I * MERGE_PROPS

    task_id = task_base + tl.program_id(
        0
    )  # grid equals the real merge-task count; do not truncate to the AI-Core count
    t_i = task_id // OUTPUT_GROUPS  # high dimension maps to the query row
    group_id = (
        task_id % OUTPUT_GROUPS
    )  # low dimension maps to the row output group, tail included
    b = 0
    cur_len_q = tl.load(seq_lens_q_ptr)
    if REQ_NUM > 1:
        while b < REQ_NUM - 1 and t_i >= cur_len_q:
            b += 1
            cur_len_q = tl.load(seq_lens_q_ptr + b)
    cur_len_k = tl.load(seq_lens_k_ptr + b)
    act_len_k = (
        cur_len_k - (cur_len_q - t_i) + 1
    )  # each row computes its valid K per causal semantics
    valid_input_lists = tl.cdiv(
        act_len_k, INPUT_SPAN
    )  # list coverage follows the Stage1 proposal granularity; GM padding must not compete
    for _ in tl.static_range(
        ROUND
    ):  # each four-way layer folds the valid input-list count by four
        valid_input_lists = tl.cdiv(valid_input_lists, MERGE_WAYS)
    group_input_begin = (
        group_id * MERGE_WAYS
    )  # each task reads only its own one-to-four contiguous lists
    group_input_lists = tl.maximum(
        0,
        tl.minimum(
            MERGE_WAYS,
            valid_input_lists - group_input_begin,
        ),
    )
    output_offsets = (t_i * OUTPUT_GROUPS + group_id) * LIST_WORDS + tl.arange(
        0, LIST_WORDS
    )
    if (
        OUTPUT_INDICES and act_len_k <= TOP_K
    ):  # the final layer keeps the short-K defensive branch and emits the exact set
        lanes = tl.arange(0, TOP_K)
        short_indices = tl.where(lanes < act_len_k, lanes, -1)
        tl.store(output_proposal_ptr + t_i * TOP_K + lanes, short_indices)
    elif (
        group_input_lists == 0
    ):  # padding tasks must not read out-of-bounds proposals; legal sentinels only
        if OUTPUT_INDICES:
            lanes = tl.arange(0, TOP_K)
            tl.store(
                output_proposal_ptr + t_i * TOP_K + lanes,
                tl.full([TOP_K], -1, tl.int32),
            )
        else:
            tl.store(
                output_proposal_ptr + output_offsets,
                tl.full([LIST_WORDS], float("-inf"), tl.float32),
            )
    elif (
        group_input_lists == 1
    ):  # a single-list tail needs no merge, but middle layers still copy full pairs
        if OUTPUT_INDICES:
            lanes = tl.arange(0, TOP_K)
            singleton_index_words = tl.load(
                proposal_ptr
                + t_i * stride_proposal_row
                + group_input_begin * LIST_WORDS
                + 2 * lanes
                + 1
            )
            singleton_indices = singleton_index_words.to(
                tl.int32,
                bitcast=True,
            )
            tl.store(
                output_proposal_ptr + t_i * TOP_K + lanes,
                singleton_indices,
            )
        else:
            singleton_offsets = group_input_begin * LIST_WORDS + tl.arange(
                0, LIST_WORDS
            )
            singleton_pairs = tl.load(
                proposal_ptr + t_i * stride_proposal_row + singleton_offsets
            )
            tl.store(output_proposal_ptr + output_offsets, singleton_pairs)
    else:
        buffer_lanes = tl.arange(0, MERGE_WORDS)
        source_word_offset = group_input_begin * LIST_WORDS + buffer_lanes
        merge_input = tl.load(
            proposal_ptr + t_i * stride_proposal_row + source_word_offset,
            mask=buffer_lanes < group_input_lists * LIST_WORDS,
            other=float("-inf"),
        )
        merged_pairs = tl.zeros([MERGE_WORDS], dtype=tl.float32)
        consumed = tl.zeros([MERGE_WAYS], dtype=tl.int32)
        merged_pairs, consumed = tle.dsa.ascend.raw(
            "merge_exhaust_sort4",
            merge_input,
            group_input_lists,
            0,
            TOP_K,
            2 * TOP_K,
            3 * TOP_K,
            TOP_K,
            tl.where(group_input_lists > 1, TOP_K, 0),
            tl.where(group_input_lists > 2, TOP_K, 0),
            tl.where(group_input_lists > 3, TOP_K, 0),
            out=[merged_pairs, consumed],
        )
        top_pairs = al.extract_slice(
            merged_pairs,
            (0,),
            (LIST_WORDS,),
            (1,),
        )
        produced = (
            al.get_element(consumed, (0,))
            + al.get_element(consumed, (1,))
            + al.get_element(consumed, (2,))
            + al.get_element(consumed, (3,))
        )
        top_pairs = tl.where(
            tl.arange(0, LIST_WORDS) < 2 * produced,
            top_pairs,
            float("-inf"),
        )  # the exhaustion merge only guarantees the consumed prefix is valid
        if OUTPUT_INDICES:
            final_values = tl.zeros([TOP_K], dtype=tl.float32)
            final_indices = tl.zeros([TOP_K], dtype=tl.int32)
            final_values, final_indices = tle.dsa.ascend.raw(
                "unpack_sort",
                top_pairs,
                TOP_K,
                out=[final_values, final_indices],
            )
            tl.store(
                output_proposal_ptr + t_i * TOP_K + tl.arange(0, TOP_K),
                final_indices,
            )
        else:
            tl.store(output_proposal_ptr + output_offsets, top_pairs)


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor = None,
    actual_seq_lengths_key: torch.Tensor = None,
    block_table: torch.Tensor = None,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 512,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    return_value: bool = False,
):
    """Dispatch the Lightning Indexer by semantic region and workload shape.

    Exact indices are emitted directly when the current request's valid K fits within
    ``sparse_count``; single-token long K uses the K-parallel Stage1 and other long K
    the query-parallel Stage1, both built on the FlagTree PR #1065 CustomOps.
    """
    logger.debug("GEMS_ASCEND LIGHTNING_INDEXER")

    if tle is None:
        raise RuntimeError(
            "Ascend LightningIndexer requires FlagTree PR #1065 CustomOps "
            "(sort_1d_pack, merge_exhaust_sort4, unpack_sort)"
        ) from _PR1065_IMPORT_ERROR

    tensors = (
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
    )
    if any(tensor is None for tensor in tensors):
        raise ValueError("LightningIndexer requires query/key lengths and block_table")
    if any(tensor.device.type != "npu" for tensor in tensors):
        raise RuntimeError("Ascend LightningIndexer requires NPU tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all LightningIndexer inputs must be on the same NPU")
    if query.ndim != 3 or key.ndim != 4 or weights.ndim != 2:
        raise ValueError("expected query [T,H,D], key [B,S,N,D], weights [T,H]")
    if query.shape[0] == 0:
        raise NotImplementedError("empty query is not supported")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise TypeError("query and key must use torch.bfloat16")
    if weights.dtype != torch.bfloat16:
        raise TypeError("weights must use torch.bfloat16")
    if any(
        tensor.dtype != torch.int32
        for tensor in (
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
        )
    ):
        raise TypeError("sequence lengths and block_table must use torch.int32")
    if actual_seq_lengths_query.ndim != 1 or actual_seq_lengths_key.ndim != 1:
        raise ValueError("sequence lengths must be rank-1 cumulative metadata")
    if block_table.ndim != 2:
        raise ValueError("block_table must be rank-2")
    if actual_seq_lengths_query.shape != actual_seq_lengths_key.shape:
        raise ValueError("query and key length metadata must have the same batch size")
    if block_table.shape[0] != actual_seq_lengths_key.shape[0]:
        raise ValueError("block_table batch size must match sequence metadata")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise NotImplementedError(
            "only contiguous LightningIndexer inputs are supported"
        )
    if layout_query != "TND" or layout_key != "PA_BSND":
        raise NotImplementedError(
            "only TND query and PA_BSND key layouts are supported"
        )
    if sparse_count != 512 or sparse_mode != 3:
        raise NotImplementedError(
            "only sparse_count=512 and sparse_mode=3 are supported"
        )
    if pre_tokens != 9223372036854775807 or next_tokens != 9223372036854775807:
        raise NotImplementedError("custom pre_tokens and next_tokens are not supported")
    if return_value:
        raise NotImplementedError("return_value=True is not supported")

    total_query_seqs, query_head_num, head_dim = query.shape
    _, block_size, key_head_num, key_head_dim = key.shape
    if (query_head_num, head_dim) != (64, 128):
        raise NotImplementedError("only 64 query heads with head_dim=128 are supported")
    if (block_size, key_head_num, key_head_dim) != (128, 1, 128):
        raise NotImplementedError("key must use block_size=128, one head, head_dim=128")
    if weights.shape != (total_query_seqs, query_head_num):
        raise ValueError("weights must match query token and head dimensions")
    K_TILE = block_size  # K tile must match the paged-cache block size
    C_TILE = 4  # each local proposal merges four physical blocks
    base_block = K_TILE * C_TILE  # one Stage1 group always covers 512 keys
    req_num = actual_seq_lengths_key.shape[0]
    decode_request = None
    decode_actual_tokens = None
    if total_query_seqs == 1:
        query_ends = [
            int(value) for value in actual_seq_lengths_query.detach().cpu().tolist()
        ]  # single-token dispatch reads only the small cumulative-length metadata, no Torch ops
        key_lengths = [
            int(value) for value in actual_seq_lengths_key.detach().cpu().tolist()
        ]
        previous_end = 0
        active_requests = []
        for request_id, query_end in enumerate(query_ends):
            if query_end > previous_end:
                active_requests.append(request_id)
            previous_end = query_end
        if len(active_requests) != 1:
            raise ValueError(
                "single-token query must belong to exactly one request, got "
                f"cumulative lengths {query_ends}"
            )
        decode_request = active_requests[
            0
        ]  # the single token may still sit in any row of the multi-request metadata
        decode_actual_tokens = key_lengths[
            decode_request
        ]  # the K-parallel kernel reads only its own request length
        max_actual_tokens = max(key_lengths)
    else:
        max_actual_tokens = int(actual_seq_lengths_key.max().item())
    device_index = query.device.index if query.device.index is not None else 0
    device_properties = triton.runtime.driver.active.utils.get_device_properties(
        device_index
    )
    USED_CORES = int(
        device_properties["num_aicore"]
    )  # query the hardware limit at runtime instead of hardcoding one board
    Q_TILE = 4  # each prefill MIX program owns four query rows
    output = torch.empty(
        (total_query_seqs, 1, sparse_count),
        dtype=torch.int32,
        device=query.device,
    )
    values = torch.empty(
        (0,), dtype=query.dtype, device=query.device
    )  # match the official two-tuple contract: an empty value Tensor when return_value=False

    route_tokens = decode_actual_tokens if total_query_seqs == 1 else max_actual_tokens
    if (
        route_tokens <= sparse_count
    ):  # no sorting needed once TopK covers the active request
        lightning_indexer_tnd_pa_direct_indices_kernel[(USED_CORES,)](
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            output,
            REQ_NUM=req_num,
            TOP_K=sparse_count,
            multibuffer=False,
        )
        return output, values

    proposal_input_span = (
        base_block if total_query_seqs == 1 else 4096
    )  # decode keeps 512-key lists; each prefill list covers a 4096-key segment
    max_proposal_lists = (
        max_actual_tokens + proposal_input_span - 1
    ) // proposal_input_span
    max_proposal_tokens = max_proposal_lists * sparse_count
    MB = (
        3 if total_query_seqs > 1 and max_proposal_lists == 2 else 2
    )  # only the two-segment prefill uses three slots; other paths stay lean
    single_req_extra_cores = 0
    single_req_prefix_rows = 0
    host_specialize_causal_schedule = (
        req_num == 1 and total_query_seqs > 1 and max_proposal_lists == 2
    )
    if host_specialize_causal_schedule:
        query_tiles = (total_query_seqs + Q_TILE - 1) // Q_TILE
        base_tiles = query_tiles // USED_CORES
        single_req_extra_cores = query_tiles % USED_CORES
        single_req_prefix_rows = (
            single_req_extra_cores * (base_tiles + 1) * Q_TILE
        )  # derive the two-pool boundary from the real query-row count and dynamic core count
    out = torch.empty(
        (total_query_seqs, 2 * max_proposal_tokens),
        dtype=torch.float32,
        device=query.device,
    )
    raw_scores = None
    fuse_prefill_sort_merge = total_query_seqs > 1 and max_proposal_lists == 2
    host_specialize_sort_lengths = req_num == 1 and fuse_prefill_sort_merge
    if total_query_seqs > 1:
        raw_scores = torch.empty(
            (total_query_seqs, max_proposal_lists * proposal_input_span),
            dtype=torch.float32,
            device=query.device,
        )  # MIX writes 512-score chunks, avoiding a 4096-score SSA tensor splice in one IR
    # QK must stay FP32 before head reduction; narrowing to FP16 changes some rows TopK sets.
    qk_workspace_dtype = torch.float32
    groupwise_fp32_reduction = (
        req_num == 1 and total_query_seqs > 1 and max_proposal_lists == 2
    )  # grouped reduction is enabled only on the current Stage1 bottleneck path
    wsp = torch.empty(
        (MB * USED_CORES * Q_TILE * query_head_num * K_TILE * C_TILE),
        dtype=qk_workspace_dtype,
        device=query.device,
    )

    if total_query_seqs == 1:
        if decode_actual_tokens > sparse_count:
            k_blk_cnt = (decode_actual_tokens + K_TILE - 1) // K_TILE
            k_group_count = (
                k_blk_cnt + C_TILE - 1
            ) // C_TILE  # grid uses the real K-group count instead of a fixed core count
            lightning_indexer_tnd_pa_decode_stage1_kernel[(k_group_count,)](
                query,
                key,
                weights,
                wsp,
                out,
                block_table[decode_request],
                query.stride(0),
                query.stride(1),
                key.stride(0),
                weights.stride(0),
                out.stride(0),
                block_table.stride(0),
                query_head_num,
                head_dim,
                ACT_LEN_K=decode_actual_tokens,
                K_BLK_CNT=k_blk_cnt,
                K_TILE=K_TILE,
                C_TILE=C_TILE,
                TOPK=sparse_count,
                disable_auto_cv_work_space_manage=True,
                unit_flag=False,
                multibuffer=False,
                sync_solver=False,
                **_COMPILER_OPTIONS,
            )
    else:
        lightning_indexer_tnd_pa_stage1_kernel[(USED_CORES,)](
            query,
            key,
            weights,
            wsp,
            raw_scores,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            weights.stride(0),
            raw_scores.stride(0),
            block_table.stride(0),
            query_head_num,
            head_dim,
            REQ_NUM=req_num,
            Q_TILE=Q_TILE,
            K_TILE=K_TILE,
            C_TILE=C_TILE,
            TOPK=sparse_count,
            MB=MB,
            OUTPUT_RAW_SCORES=True,
            GROUPWISE_FP32_REDUCTION=groupwise_fp32_reduction,
            HOST_SPECIALIZE_CAUSAL_SCHEDULE=host_specialize_causal_schedule,
            SINGLE_REQ_EXTRA_CORES=single_req_extra_cores,
            SINGLE_REQ_PREFIX_ROWS=single_req_prefix_rows,
            disable_auto_cv_work_space_manage=True,
            # the path syncs cores explicitly; auto-injected sync breaks the overlap
            disable_auto_inject_block_sync=True,
            unit_flag=False,
            multibuffer=True,
            sync_solver=False,
            **_COMPILER_OPTIONS,
        )
        if fuse_prefill_sort_merge:
            lightning_indexer_tnd_pa_prefill_sort_merge_8192_top512_kernel[
                (total_query_seqs,)
            ](
                raw_scores,
                output,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                raw_scores.stride(0),
                output.stride(0),
                REQ_NUM=req_num,
                HOST_SPECIALIZE_SINGLE_REQ_LENGTHS=host_specialize_sort_lengths,
                SINGLE_REQ_QUERY_ROWS=(
                    total_query_seqs if host_specialize_sort_lengths else 0
                ),
                SINGLE_REQ_KEY_TOKENS=(
                    max_actual_tokens if host_specialize_sort_lengths else 0
                ),
                TOPK=sparse_count,
                multibuffer=False,
                **_COMPILER_OPTIONS,
            )
        else:
            lightning_indexer_tnd_pa_prefill_sort_4096_top512_kernel[
                (total_query_seqs * max_proposal_lists,)
            ](
                raw_scores,
                out,
                output,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                raw_scores.stride(0),
                out.stride(0),
                output.stride(0),
                PROPOSAL_GROUPS=max_proposal_lists,
                REQ_NUM=req_num,
                OUTPUT_INDICES=max_proposal_lists == 1,
                TOPK=sparse_count,
                multibuffer=False,
                **_COMPILER_OPTIONS,
            )

    stage2_actual_tokens = (
        decode_actual_tokens if total_query_seqs == 1 else max_actual_tokens
    )
    merge_ways = 4  # the PR CustomOp merges at most four proposals at once
    input_lists = (
        1
        if fuse_prefill_sort_merge
        else (stage2_actual_tokens + proposal_input_span - 1) // proposal_input_span
    )  # the fused two-segment path already emitted final indices; skip Stage2
    current_proposals = (
        out  # the first layer consumes the full score/index pairs from Stage1
    )
    current_stride = out.stride(
        -2
    )  # strides follow the current GM allocation; layers must not assume identical layouts
    current_groups = input_lists
    merge_round = 0
    while current_groups > 1:
        output_groups = (current_groups + merge_ways - 1) // merge_ways
        merge_tasks = (
            total_query_seqs * output_groups
        )  # 1/2/3-list tail groups also need their own tasks
        final_round = (
            output_groups == 1
        )  # scores are unneeded after the final layer; write INT32 indices directly
        if final_round:
            lightning_indexer_tnd_pa_stage2_top512_merge_pairs_kernel[(merge_tasks,)](
                current_proposals,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                output,
                current_stride,
                0,
                OUTPUT_GROUPS=output_groups,
                ROUND=merge_round,
                REQ_NUM=req_num,
                INPUT_SPAN=proposal_input_span,
                TOP_K=sparse_count,
                OUTPUT_INDICES=True,
                multibuffer=False,
                **_COMPILER_OPTIONS,
            )
        else:
            next_proposals = torch.empty(
                (total_query_seqs, output_groups, 2 * sparse_count),
                dtype=torch.float32,
                device=query.device,
            )
            lightning_indexer_tnd_pa_stage2_top512_merge_pairs_kernel[(merge_tasks,)](
                current_proposals,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                next_proposals,
                current_stride,
                0,
                OUTPUT_GROUPS=output_groups,
                ROUND=merge_round,
                REQ_NUM=req_num,
                INPUT_SPAN=proposal_input_span,
                TOP_K=sparse_count,
                OUTPUT_INDICES=False,
                multibuffer=False,
                **_COMPILER_OPTIONS,
            )
            current_proposals = next_proposals  # non-final layers keep passing pairs; indices alone would be premature
            current_stride = next_proposals.stride(
                0
            )  # the next layer reads with the new allocation row stride
        current_groups = output_groups
        merge_round += 1
    return output, values
