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

"""Ascend 分页注意力 Lightning Indexer 的 TLE 实现。

实现按语义分为三条路径：短 K 直接生成完整索引；单 token 长 K 沿 K 维并行生成
512-key proposal；多 token 长 K 沿 query 行并行生成相同 proposal。两条长 K 路径
共用保持 score/index pair 的 Stage2 归并树。

每个 proposal 使用两个 FP32 word 保存：score 和按位编码的 INT32 key index。
中间归并必须同时保留这两个 word，只有最终 TopK 完成后才能只输出 index。
排序、归并和解包复用 FlagTree PR #1065 提供的公共 proposal CustomOp。
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice

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
    "use_bytecode": True,  # CustomOp bitcode 由安装的 PR 工具链统一提供
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
    """为多 token 长 K query 行构造有序的 512-key proposal 列表。

    每个 MIX program 负责 ``Q_TILE`` 行。Cube 按 128-key block 计算 QK，两个 AIV
    sub-block 对 head 加权归约、用 ``-inf`` 屏蔽尾块无效位置，并输出紧凑 proposal。
    Stage2 归并时继续保留 score word。
    """

    T_TILE: tl.constexpr = (
        2  # 单次 Cube 计算两行 query，匹配两个 AIV sub-block 的消费节奏
    )
    wsp_nstride: tl.constexpr = C_TILE * K_TILE  # 一个局部排序组固定覆盖 512 个 key
    wsp_mstride: tl.constexpr = (
        T_TILE * query_head_num * wsp_nstride
    )  # 两行 QK 的 workspace 跨度
    m_coef: tl.constexpr = (
        Q_TILE // al.sub_vec_num()
    )  # 每个 AIV sub-block 实际负责的 query 行数
    q_step: tl.constexpr = Q_TILE // T_TILE  # 一个 query tile 需要的 Cube 批次数
    N_TILE: tl.constexpr = 16  # 以 16 个 head 为一组做向量加权归约
    n_step: tl.constexpr = query_head_num // N_TILE  # 覆盖全部 query head 的归约次数
    core_nums = tl.num_programs(0)  # 使用实际 launch grid 计算跨轮 query 步长
    core_id = tl.program_id(0)
    b = 0
    t_i = core_id * Q_TILE  # 先按 grid 分配 query tile，再映射到累计 TND request
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
            # 两段路径使用host constexpr，删除设备端整除和取模
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
            # 非两段特化路径保留设备端两池调度，覆盖任意 query tile 数
    tmp_buf = tl.zeros([wsp_nstride * 4], dtype=tl.float32)
    sorted_pairs_buf = tl.zeros([2 * wsp_nstride], dtype=tl.float32)
    for i in tl.static_range(MB):
        al.sync_block_set(
            "vector", "cube", i, pipe.PIPE_MTE2, pipe.PIPE_FIX
        )  # 首轮先把每个环形槽标记为 Cube 可写
    db_flag = 0
    if REQ_NUM > 1:
        while t_i >= cur_len_q and b < REQ_NUM - 1:
            q_tail = seq_len_q % Q_TILE
            if q_tail:
                t_i -= (
                    Q_TILE - q_tail
                )  # TND request 尾块不足 Q_TILE 时回退到下一段真实起点
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
        )  # sparse_mode=3 的 causal 对齐长度
        k_blk_cnt = (
            (act_len_k + K_TILE - 1) // K_TILE if act_len_k + Q_TILE - 1 > TOPK else 0
        )  # TopK 已覆盖的行无需排序
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
        )  # 按真实行 stride 跳过 token 间 padding
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
            )  # MB 环形槽按 program 隔离
            remain_k_blk_cnt = k_blk_cnt - k_t
            k_itr = C_TILE if remain_k_blk_cnt >= C_TILE else remain_k_blk_cnt
            al.sync_block_wait(
                "vector", "cube", (db_flag % MB), pipe.PIPE_MTE2, pipe.PIPE_FIX
            )  # Cube 重用槽前等待 AIV 消费完成
            for k_i in range(k_itr):
                actual_k_block_i = tl.load(
                    block_table_ptr + b * stride_block_table_b + k_t + k_i
                )  # 通过 block table 做分页间接寻址
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
            )  # 发布本槽 QK 数据给 AIV
            in_offset = core_offset + tl.arange(0, N_TILE * wsp_nstride)
            if OUTPUT_RAW_SCORES:
                out_offsets = (
                    t_i * stride_out_0
                    + k_t * K_TILE
                    + tl.arange(0, wsp_nstride)
                )
            else:
                out_offsets = (
                    t_i * stride_out_0
                    + 2 * k_t * K_TILE
                    + tl.arange(0, 2 * wsp_nstride)
                )
            al.sync_block_wait(
                "cube", "vector", (db_flag % MB), pipe.PIPE_FIX, pipe.PIPE_MTE2
            )  # AIV 读取前等待 Cube 写完
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
                qk_contribution = (
                    tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight
                )
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
                qk_contribution = (
                    tl.reshape(qk_slice, (N_TILE, wsp_nstride)) * weight
                )
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
                    )  # 尾组无效 lane 必须排在所有真实 score 之后
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
                        k_t * K_TILE,  # proposal index 必须包含当前 512-key 分组偏移
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
            )  # AIV 完成后归还环形槽
            db_flag += 1
        t_i += query_stride  # 单请求负载均衡路径按各自 core pool 的 stride 领取下一块
        if REQ_NUM > 1:
            while t_i >= cur_len_q and b < REQ_NUM - 1:
                q_tail = seq_len_q % Q_TILE
                if q_tail:
                    t_i -= Q_TILE - q_tail  # 跨 request 时消除上一段尾块造成的累计空洞
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
        )  # 退出前等待所有 AIV consumer 归还槽位


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
    """将 GM 中的 4096-score segment 裁剪为 Top512 proposal。"""

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
    """用公共 base sort 一次处理 8192 score并输出全局 Top512。"""

    TOTAL_KEYS: tl.constexpr = 8192
    PAIR_WORDS: tl.constexpr = 2 * TOPK
    SORT_TMP_WORDS: tl.constexpr = 4 * TOTAL_KEYS
    tl.static_assert(TOPK == 512)

    t_i = tl.program_id(0)  # 每个 Vector program 独占一行，避免 proposal 跨行共享
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
        )  # 一次全局排序避免两条proposal及额外二路归并
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
    """为单个 decode token 的每个 512-key 分组构造一条紧凑 proposal 列表。

    Host 已解析出该 token 唯一所属的 request，并传入对应 block-table 行。grid 使用真实
    K-group 数，因此即使 group 数超过 AI Core 数，也能独立覆盖全部 key。
    """
    T_TILE: tl.constexpr = 2  # 保持与通用 Stage1 相同的 Cube tile 形状
    Q_TILE: tl.constexpr = 4  # 保持已验证的 MIX 静态布局，实际只归约 token 0
    N_TILE: tl.constexpr = 16  # 每次向量归约 16 个 query head
    wsp_nstride: tl.constexpr = (
        C_TILE * K_TILE
    )  # 每个 program 覆盖一个 512-key proposal 组
    wsp_mstride: tl.constexpr = T_TILE * query_head_num * wsp_nstride
    m_coef: tl.constexpr = Q_TILE // al.sub_vec_num()
    q_step: tl.constexpr = Q_TILE // T_TILE
    n_step: tl.constexpr = query_head_num // N_TILE

    core_id = tl.program_id(0)
    k_t = core_id * C_TILE  # 一个 program 独占一个 K group，补足单 token 下的并行度
    tmp_buf = tl.zeros([wsp_nstride * 4], dtype=tl.float32)
    sorted_pairs_buf = tl.zeros([2 * wsp_nstride], dtype=tl.float32)

    al.sync_block_set(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # 首次 Cube 写 workspace 前声明槽可用
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
    )  # 使用真实 stride，不能假设 token 行紧密连续
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
    )  # Cube 覆盖 workspace 前等待槽位可写
    for k_i in range(k_itr):
        actual_k_block_i = tl.load(
            block_table_ptr + k_t + k_i
        )  # 指针已定位到 active request 的 block-table 行
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
    )  # 发布完整 512-key QK tile
    in_offset = core_offset + tl.arange(0, N_TILE * wsp_nstride)
    out_offsets = 2 * k_t * K_TILE + tl.arange(0, 2 * wsp_nstride)
    al.sync_block_wait(
        "cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_MTE2
    )  # AIV 归约前等待完整 QK tile 发布

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
            )  # decode 尾组无效 lane 必须以 -inf 退出 TopK 竞争
        sorted_pairs_buf = tle.dsa.ascend.raw(
            "sort_1d_pack",
            tl.reshape(scores, (wsp_nstride,)),
            tmp_buf,
            True,
            wsp_nstride,
            k_t * K_TILE,  # 公共 sort 在设备侧生成带全局偏移的 index
            SORT_IMPL_BASE,
            out=sorted_pairs_buf,
        )
        tl.store(out_ptr + out_offsets, sorted_pairs_buf)

    al.sync_block_set(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # 显式闭合 MIX 同步边
    al.sync_block_wait(
        "vector", "cube", 0, pipe.PIPE_MTE2, pipe.PIPE_FIX
    )  # 等待最终 proposal 写出完成


@triton.jit
def lightning_indexer_tnd_pa_direct_indices_kernel(
    seq_lens_q_ptr,
    seq_lens_k_ptr,
    o_ptr,
    REQ_NUM: tl.constexpr,
    TOP_K: tl.constexpr = 512,
):
    """当 TopK 覆盖全部有效 key 时，直接物化完整索引集合。

    sparse_mode=3 使用 causal 对齐，行 ``t`` 的有效长度为
    ``key_end - (query_end - t) + 1``。完整 TopK 集合就是 ``[0, act_len_k)``
    后接 ``-1`` padding，无需计算 score。
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
        )  # 与长 K 路径保持相同的 causal 有效长度
        indices = tl.where(
            lanes < act_len_k, lanes, -1
        )  # TopK 覆盖全集时，顺序 index 即为精确集合
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
    """执行 TopK=512 pair-preserving GM 归并树的一层。

    每个 program 独占一个 ``(query row, output group)`` task，归并一至四条相邻有序
    list。非最终层写完整紧凑 pair，只有最终层物化 index；后续层仍需跨组比较 score。
    """
    V_I: tl.constexpr = 2
    MERGE_WAYS: tl.constexpr = 4  # 四路归并是在当前 UB 约束下验证通过的最大 fan-in
    LIST_WORDS: tl.constexpr = V_I * TOP_K  # score/index_bits 必须成对跨 GM 层保存
    MERGE_PROPS: tl.constexpr = MERGE_WAYS * TOP_K
    MERGE_WORDS: tl.constexpr = V_I * MERGE_PROPS

    task_id = task_base + tl.program_id(
        0
    )  # grid 等于真实 merge task 数，不能按 AI Core 数截断
    t_i = task_id // OUTPUT_GROUPS  # 高维映射 query 行
    group_id = task_id % OUTPUT_GROUPS  # 低维映射该行的输出分组，包含尾组
    b = 0
    cur_len_q = tl.load(seq_lens_q_ptr)
    if REQ_NUM > 1:
        while b < REQ_NUM - 1 and t_i >= cur_len_q:
            b += 1
            cur_len_q = tl.load(seq_lens_q_ptr + b)
    cur_len_k = tl.load(seq_lens_k_ptr + b)
    act_len_k = cur_len_k - (cur_len_q - t_i) + 1  # 每行按 causal 语义独立计算有效 K
    valid_input_lists = tl.cdiv(
        act_len_k, INPUT_SPAN
    )  # list 覆盖范围由 Stage1 proposal 粒度决定，禁止 GM padding 参与候选
    for _ in tl.static_range(ROUND):  # 每经过一层四路树，合法输入 list 数按四归一
        valid_input_lists = tl.cdiv(valid_input_lists, MERGE_WAYS)
    group_input_begin = (
        group_id * MERGE_WAYS
    )  # 每个 task 只读取自己负责的连续一至四条 list
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
    ):  # 最终层保留短 K 防御分支，直接输出精确集合
        lanes = tl.arange(0, TOP_K)
        short_indices = tl.where(lanes < act_len_k, lanes, -1)
        tl.store(output_proposal_ptr + t_i * TOP_K + lanes, short_indices)
    elif group_input_lists == 0:  # padding task 不得读取越界 proposal，只写合法哨兵
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
    elif group_input_lists == 1:  # 单路尾组无需归并，但中间层仍要复制完整 pair
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
        )  # exhaustion merge 只保证 consumed 对应的前缀有效
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
    """按语义区域和工作负载 shape 分流 Lightning Indexer。

    当前路由 request 的有效 K 不超过 ``sparse_count`` 时直接生成精确索引；单 query
    token 的长 K 使用 K-parallel Stage1，其他长 K 使用 query-parallel Stage1。长 K
    的局部排序和四路归并树使用 FlagTree PR #1065 的公共 proposal CustomOp。
    """

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
    K_TILE = block_size  # K tile 必须与分页 cache 的 block size 一致
    C_TILE = 4  # 每条局部 proposal 合并四个物理 block
    base_block = K_TILE * C_TILE  # Stage1 单组固定覆盖 512 个 key
    req_num = actual_seq_lengths_key.shape[0]
    decode_request = None
    decode_actual_tokens = None
    if total_query_seqs == 1:
        query_ends = [
            int(value) for value in actual_seq_lengths_query.detach().cpu().tolist()
        ]  # 单 token dispatch 只读取小型累计长度元数据，不用 Torch 计算输出
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
        ]  # 单 token 仍可能位于多 request 元数据中的任意一行
        decode_actual_tokens = key_lengths[
            decode_request
        ]  # K-parallel kernel 只读取所属 request 的长度
        max_actual_tokens = max(key_lengths)
    else:
        max_actual_tokens = int(actual_seq_lengths_key.max().item())
    device_index = query.device.index if query.device.index is not None else 0
    device_properties = triton.runtime.driver.active.utils.get_device_properties(
        device_index
    )
    USED_CORES = int(
        device_properties["num_aicore"]
    )  # 运行时读取硬件资源上限，避免绑定某张卡
    Q_TILE = 4  # 每个 prefill MIX program 负责四行 query
    output = torch.empty(
        (total_query_seqs, 1, sparse_count),
        dtype=torch.int32,
        device=query.device,
    )
    values = torch.empty(
        (0,), dtype=query.dtype, device=query.device
    )  # 对齐官方 return_value=False 时仍返回空 value Tensor 的二元组契约

    route_tokens = (
        decode_actual_tokens if total_query_seqs == 1 else max_actual_tokens
    )
    if route_tokens <= sparse_count:  # TopK 覆盖当前有效 request 时无需排序
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
    )  # decode 保留 512-key list；prefill 每条 list 覆盖一个 4096-key segment
    max_proposal_lists = (
        max_actual_tokens + proposal_input_span - 1
    ) // proposal_input_span
    max_proposal_tokens = max_proposal_lists * sparse_count
    MB = (
        3 if total_query_seqs > 1 and max_proposal_lists == 2 else 2
    )  # 两段 prefill 才用三槽，其他路径保持低开销
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
        )  # 用真实query行数和动态core数生成两池边界
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
        )  # MIX 只写 512-score chunk，避免在同一 IR 中拼接 4096-score SSA tensor
    # QK 在 head reduction 前必须保持 FP32；缩窄到 FP16 会改变部分行的 TopK 集合。
    qk_workspace_dtype = torch.float32
    groupwise_fp32_reduction = (
        req_num == 1
        and total_query_seqs > 1
        and max_proposal_lists == 2
    )  # 分组归约只在当前 Stage1 瓶颈路径启用，其他路径保持原树
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
            ) // C_TILE  # grid 使用真实 K group 数而非固定 core 数
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
            disable_auto_inject_block_sync=True,  # 本路径已有显式跨核同步，避免自动插入的同步破坏流水重叠
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
    merge_ways = 4  # PR CustomOp 最多一次合并四路 proposal
    input_lists = (
        1
        if fuse_prefill_sort_merge
        else (stage2_actual_tokens + proposal_input_span - 1)
        // proposal_input_span
    )  # fused 两段路径已经输出最终 index，不再进入 Stage2
    current_proposals = out  # 首层消费 Stage1 写出的完整 score/index pair
    current_stride = out.stride(
        -2
    )  # stride 随当前 GM allocation 更新，不能假设连续层布局相同
    current_groups = input_lists
    merge_round = 0
    while current_groups > 1:
        output_groups = (current_groups + merge_ways - 1) // merge_ways
        merge_tasks = (
            total_query_seqs * output_groups
        )  # 1/2/3-list 尾组同样必须获得独立 task
        final_round = (
            output_groups == 1
        )  # 最终层之后不再需要 score，可直接写 INT32 index
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
            current_proposals = (
                next_proposals  # 非最终层继续传递 pair，不能提前只保留 index
            )
            current_stride = next_proposals.stride(
                0
            )  # 下一层按新 allocation 的行 stride 读取
        current_groups = output_groups
        merge_round += 1
    return output, values
