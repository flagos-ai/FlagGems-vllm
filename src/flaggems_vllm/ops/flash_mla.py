import logging
import math
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device, error, torch_device_fn
from flaggems_vllm.utils import triton_lang_extension as ext
from flaggems_vllm.utils.device_info import get_device_capability
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_FLASH_MLA = True
    except ImportError:
        tle = None
        HAS_TLE_FLASH_MLA = False
else:
    tle = None
    HAS_TLE_FLASH_MLA = False

vendor_name = device.vendor_name
device = device.name
logger = logging.getLogger(__name__)

FLASH_MLA_META_FIELDS = 8
FLASH_MLA_BLOCK_M = 64
FLASH_MLA_BLOCK_N = 64
FLASH_MLA_FIXED_OVERHEAD_BLOCKS = 5
FLASH_MLA_COMBINE_BLOCK_H = 16


def _set_triton_descriptor_allocator(cuda_device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=cuda_device)

    triton.set_allocator(alloc_fn)


def _flash_mla_tle_variant() -> str:
    variant = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT", "auto").lower()
    legacy = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_TLE")
    if legacy is not None and legacy.lower() in {"0", "false", "off", "no"}:
        return "triton"
    if variant not in {"auto", "triton"}:
        logger.warning("Unknown FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT=%s", variant)
        return "auto"
    return variant


def _can_use_tle_flash_mla(
    q: torch.Tensor,
    block_table: torch.Tensor,
    blocked_k: torch.Tensor,
    block_size: int,
    b: int,
    s_q: int,
    cache_seqlens: torch.Tensor,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    causal: bool,
) -> bool:
    if _flash_mla_tle_variant() == "triton":
        return False
    if not HAS_TLE_FLASH_MLA:
        return False
    if q.device.type != "cuda":
        return False
    major, _ = get_device_capability()
    if major != 9:
        return False
    return (
        causal
        and q.dtype in (torch.bfloat16, torch.float16)
        and blocked_k.dtype == q.dtype
        and block_table.dtype == torch.int32
        and cache_seqlens.dtype == torch.int32
        and q.ndim == 4
        and blocked_k.ndim == 4
        and block_table.ndim == 2
        and cache_seqlens.ndim == 1
        and b == q.shape[0]
        and s_q == q.shape[1]
        and s_q == 1
        and h_q == q.shape[2]
        and h_kv == 1
        and d == q.shape[3]
        and dv == 512
        and d in (512, 576)
        and d >= dv
        and h_q % 64 == 0
        and block_size == 64
        and blocked_k.shape[1] == block_size
        and blocked_k.shape[2] == h_kv
        and blocked_k.shape[3] == d
        and cache_seqlens.shape[0] == b
    )


@triton.jit
def flash_mla_sched_meta_kernel(
    B_seq_len,
    Sched_meta,
    Num_splits,
    BLOCK_B: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FIXED_OVERHEAD_NUM_BLOCKS: tl.constexpr,
    NUM_SM_PARTS: tl.constexpr,
    META_FIELDS: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < BATCH_SIZE
    seqlens = tl.load(B_seq_len + offs_b, mask=mask_b, other=0)
    num_blocks_vec = tl.cdiv(tl.maximum(seqlens, 1), BLOCK_SIZE_N)
    total_num_blocks = tl.sum(
        tl.where(mask_b, num_blocks_vec + FIXED_OVERHEAD_NUM_BLOCKS, 0), axis=0
    )
    payload = tl.cdiv(total_num_blocks, NUM_SM_PARTS) + FIXED_OVERHEAD_NUM_BLOCKS

    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    tl.store(Num_splits, 0)

    for part in tl.static_range(0, NUM_SM_PARTS):
        begin_req_idx = now_req_idx
        begin_block_idx = now_block
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = now_block != 0
        remain_payload = payload

        while (now_req_idx < BATCH_SIZE) & (remain_payload > 0):
            cur_seq_len = tl.load(B_seq_len + now_req_idx)
            cur_num_blocks = tl.cdiv(tl.maximum(cur_seq_len, 1), BLOCK_SIZE_N)
            now_remain_blocks = cur_num_blocks - now_block
            if remain_payload >= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS:
                cum_num_splits += now_n_split_idx + 1
                tl.store(Num_splits + now_req_idx + 1, cum_num_splits)
                remain_payload -= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - FIXED_OVERHEAD_NUM_BLOCKS > 0:
                    now_block += remain_payload - FIXED_OVERHEAD_NUM_BLOCKS
                    now_n_split_idx += 1
                remain_payload = 0

        if now_block > 0:
            end_req_idx = now_req_idx
            end_block_idx = now_block
        else:
            end_req_idx = now_req_idx - 1
            if end_req_idx >= 0:
                end_seq_len = tl.load(B_seq_len + end_req_idx)
                end_block_idx = tl.where(
                    end_seq_len == 0, 0, tl.cdiv(end_seq_len, BLOCK_SIZE_N)
                )
            else:
                end_block_idx = 0

        meta = Sched_meta + part * META_FIELDS
        if begin_req_idx >= BATCH_SIZE:
            tl.store(meta + 0, BATCH_SIZE)
            tl.store(meta + 1, BATCH_SIZE - 1)
            tl.store(meta + 2, 0)
            tl.store(meta + 3, 0)
            tl.store(meta + 4, 0)
            tl.store(meta + 5, 0)
            tl.store(meta + 6, 0)
            tl.store(meta + 7, 0)
        else:
            end_seq_len = tl.load(B_seq_len + end_req_idx)
            last_block_exclusive = tl.where(
                end_seq_len == 0, 0, tl.cdiv(end_seq_len, BLOCK_SIZE_N)
            )
            is_last_req_splitted = (end_block_idx != last_block_exclusive) & (
                end_seq_len != 0
            )
            if begin_req_idx == end_req_idx:
                same_req_split = is_first_req_splitted | is_last_req_splitted
                is_first_req_splitted = same_req_split
                is_last_req_splitted = same_req_split

            tl.store(meta + 0, begin_req_idx)
            tl.store(meta + 1, end_req_idx)
            tl.store(meta + 2, begin_block_idx)
            tl.store(meta + 3, end_block_idx)
            tl.store(meta + 4, begin_split_idx)
            tl.store(meta + 5, is_first_req_splitted.to(tl.int32))
            tl.store(meta + 6, is_last_req_splitted.to(tl.int32))
            tl.store(meta + 7, 0)

@triton.jit
def _flash_mla_ws_kv_producer(
    k0_l_writer,
    k0_r_writer,
    k1_l_writer,
    k1_r_writer,
    valid_writer,
    Kv,
    Kv_desc,
    Block_table,
    B_seq_len,
    begin_req_idx,
    end_req_idx,
    begin_block_idx_meta,
    end_block_idx_meta,
    stride_kv_token,
    stride_block_table_b,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    _ = Kv_desc
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_CHUNK)
    kv_rows = tl.broadcast_to(offs_n[:, None], (BLOCK_N, D_CHUNK))
    pipe_base = 0
    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        block_table_base = batch_idx * stride_block_table_b
        n_pairs = tl.cdiv(end_block_idx - start_block_idx, 2)
        for pair in tl.range(0, n_pairs):
            pipe_idx = pipe_base + pair
            block0 = start_block_idx + pair * 2
            block1 = block0 + 1
            has0 = block0 < end_block_idx
            has1 = block1 < end_block_idx
            page0 = tle.load(Block_table + block_table_base + block0, mask=has0, other=0)
            page1 = tle.load(Block_table + block_table_base + block1, mask=has1, other=0)
            kv_row0 = (page0 * PAGE_SIZE).to(tl.int32)
            kv_row1 = (page1 * PAGE_SIZE).to(tl.int32)

            valid0 = has0 & (block0 * BLOCK_N + offs_n < seq_len)
            valid1 = has1 & (block1 * BLOCK_N + offs_n < seq_len)
            valid_slot = valid_writer.acquire(pipe_idx)
            row0 = tl.full([BLOCK_N], 0, dtype=tl.int32)
            row1 = tl.full([BLOCK_N], 1, dtype=tl.int32)
            tl.store(tle.gpu.local_ptr(valid_slot.valid, (row0, offs_n)), valid0.to(tl.int8))
            tl.store(tle.gpu.local_ptr(valid_slot.valid, (row1, offs_n)), valid1.to(tl.int8))
            valid_writer.commit(pipe_idx)

            k0_l_slot = k0_l_writer.acquire(pipe_idx)
            for tile in tl.static_range(0, HEAD_DIM_V // 2, D_CHUNK):
                kv_cols = tl.broadcast_to((tile + offs_d)[None, :], (BLOCK_N, D_CHUNK))
                k = tl.load(Kv + (kv_row0 + kv_rows) * stride_kv_token + kv_cols)
                tl.store(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)), k)
            k0_l_writer.commit(pipe_idx)

            k0_r_slot = k0_r_writer.acquire(pipe_idx)
            for tile in tl.static_range(HEAD_DIM_V // 2, HEAD_DIM_V, D_CHUNK):
                kv_cols = tl.broadcast_to((tile + offs_d)[None, :], (BLOCK_N, D_CHUNK))
                k = tl.load(Kv + (kv_row0 + kv_rows) * stride_kv_token + kv_cols)
                tl.store(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols)), k)
            if HAVE_TAIL:
                tail_cols = tl.broadcast_to(offs_d[None, :], (BLOCK_N, D_CHUNK))
                k_tail = tl.load(
                    Kv + (kv_row0 + kv_rows) * stride_kv_token + HEAD_DIM_V + tail_cols
                )
                tl.store(tle.gpu.local_ptr(k0_r_slot.sK_tail, (kv_rows, tail_cols)), k_tail)
            k0_r_writer.commit(pipe_idx)

            k1_l_slot = k1_l_writer.acquire(pipe_idx)
            for tile in tl.static_range(0, HEAD_DIM_V // 2, D_CHUNK):
                kv_cols = tl.broadcast_to((tile + offs_d)[None, :], (BLOCK_N, D_CHUNK))
                k = tl.load(Kv + (kv_row1 + kv_rows) * stride_kv_token + kv_cols)
                tl.store(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols)), k)
            k1_l_writer.commit(pipe_idx)

            k1_r_slot = k1_r_writer.acquire(pipe_idx)
            for tile in tl.static_range(HEAD_DIM_V // 2, HEAD_DIM_V, D_CHUNK):
                kv_cols = tl.broadcast_to((tile + offs_d)[None, :], (BLOCK_N, D_CHUNK))
                k = tl.load(Kv + (kv_row1 + kv_rows) * stride_kv_token + kv_cols)
                tl.store(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols)), k)
            if HAVE_TAIL:
                tail_cols = tl.broadcast_to(offs_d[None, :], (BLOCK_N, D_CHUNK))
                k_tail = tl.load(
                    Kv + (kv_row1 + kv_rows) * stride_kv_token + HEAD_DIM_V + tail_cols
                )
                tl.store(tle.gpu.local_ptr(k1_r_slot.sK_tail, (kv_rows, tail_cols)), k_tail)
            k1_r_writer.commit(pipe_idx)
        pipe_base += n_pairs


@triton.jit
def _flash_mla_ws_consumer0(
    k0_l_reader,
    k0_r_qk_reader,
    k1_l_remote_reader,
    valid_reader,
    sM_wg0_writer,
    sM_wg1_reader,
    sP0_writer,
    sP1_reader,
    sL0_writer,
    sL1_reader,
    q_writer,
    q_reader,
    sO_stage,
    Q_desc,
    Q_tail_desc,
    Output_desc,
    O,
    O_accum,
    LSE_accum,
    sm_scale,
    B_seq_len,
    Num_splits,
    begin_req_idx,
    end_req_idx,
    begin_block_idx_meta,
    end_block_idx_meta,
    begin_split_idx,
    is_first_req_splitted,
    is_last_req_splitted,
    head_base,
    head_num,
    stride_o_b,
    stride_o_h,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    offs_dv = tl.arange(0, HALF_DIM_V)
    mask_h = head_offsets < head_num

    kv_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, HALF_DIM_V))
    kv_cols_l = tl.broadcast_to(tl.arange(0, HALF_DIM_V)[None, :], (BLOCK_N, HALF_DIM_V))
    kv_cols_r = tl.broadcast_to((HALF_DIM_V + tl.arange(0, HALF_DIM_V))[None, :], (BLOCK_N, HALF_DIM_V))
    stage_rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, D_CHUNK))
    stage_batch_idx = tl.full((BLOCK_M, D_CHUNK), 0, dtype=tl.int32)
    if HAVE_TAIL:
        kv_rows_tail = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, D_CHUNK))
        kv_cols_tail = tl.broadcast_to(tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK))

    offs_chunk = tl.arange(0, D_CHUNK)
    pipe_base = 0
    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        q_stage = batch_idx - begin_req_idx
        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        n_split_idx = tl.where(batch_idx == begin_req_idx, begin_split_idx, 0)
        no_split_middle = (batch_idx != begin_req_idx) & (batch_idx != end_req_idx)
        no_split_first = (batch_idx == begin_req_idx) & (~is_first_req_splitted)
        no_split_last = (batch_idx == end_req_idx) & (~is_last_req_splitted)
        is_no_split = no_split_middle | no_split_first | no_split_last
        if begin_req_idx == end_req_idx:
            is_no_split = ~is_first_req_splitted
        split_idx = tl.load(Num_splits + batch_idx) + n_split_idx
        q_row = batch_idx * head_num + head_base

        q_write_slot = q_writer.acquire(q_stage)
        tle.gpu.copy(Q_desc, q_write_slot.sQ_l, [BLOCK_M, HALF_DIM_V], [q_row, 0])
        tle.gpu.copy(Q_desc, q_write_slot.sQ_r, [BLOCK_M, HALF_DIM_V], [q_row, HALF_DIM_V])
        if HAVE_TAIL:
            tle.gpu.copy(Q_tail_desc, q_write_slot.sQ_tail, [BLOCK_M, D_CHUNK], [q_row, HEAD_DIM_V])
        q_writer.commit(q_stage)

        q_wait = q_reader.wait(q_stage)
        q_slot = q_wait.slot

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HALF_DIM_V], dtype=tl.float32)

        n_pairs = tl.cdiv(end_block_idx - start_block_idx, 2)
        for pair in tl.range(0, n_pairs):
            pipe_idx = pipe_base + pair
            k0_l_wait = k0_l_reader.wait(pipe_idx)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            k0_l_slot = k0_l_wait.slot
            k0_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            q_l = tl.load(tle.gpu.local_ptr(q_slot.sQ_l))
            qk = tl.dot(q_l, tl.trans(k0_l), qk, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pipe_idx)
            k0_r_slot = k0_r_wait.slot
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols_r)))
            q_r = tl.load(tle.gpu.local_ptr(q_slot.sQ_r))
            qk = tl.dot(q_r, tl.trans(k0_r), qk, out_dtype=tl.float32)
            if HAVE_TAIL:
                k_tail = tl.load(tle.gpu.local_ptr(k0_r_slot.sK_tail, (kv_rows_tail, kv_cols_tail)))
                q_tail = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                qk = tl.dot(q_tail, tl.trans(k_tail), qk, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pipe_idx)
            valid_row = tl.full([BLOCK_N], 0, dtype=tl.int32)
            valid_n = (
                tl.load(tle.gpu.local_ptr(valid_wait.slot.valid, (valid_row, tl.arange(0, BLOCK_N)))) != 0
            )
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))

            local_max = tl.maximum(tl.max(qk, axis=1), e_max)
            sM_slot = sM_wg0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sM_slot.sM), local_max)
            sM_wg0_writer.commit(pipe_idx)

            peer_wait = sM_wg1_reader.wait(pipe_idx)
            merged_max = tl.load(tle.gpu.local_ptr(peer_wait.slot.sM))
            sM_wg1_reader.release(pipe_idx)

            re_scale = tl.exp(e_max - merged_max)
            p = tl.exp(qk - merged_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            acc = acc * re_scale[:, None]

            v_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            acc = tl.dot(p.to(v_l.dtype), v_l, acc, out_dtype=tl.float32)
            if pair != n_pairs - 1:
                k0_l_reader.release(pipe_idx)
                k0_r_qk_reader.release(pipe_idx)

            p_save = p.to(v_l.dtype)
            sP_slot = sP0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_save)
            sP0_writer.commit(pipe_idx)

            peer_p_wait = sP1_reader.wait(pipe_idx)
            peer_p = tl.load(tle.gpu.local_ptr(peer_p_wait.slot.sP))
            k1_l_wait = k1_l_remote_reader.wait(pipe_idx)
            k1_l = tl.load(tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols_l)))
            acc = tl.dot(peer_p, k1_l, acc, out_dtype=tl.float32)
            sP1_reader.release(pipe_idx)
            k1_l_remote_reader.release(pipe_idx)
            valid_reader.release(pipe_idx)
            e_max = merged_max

        l_stage0 = q_stage * 2
        l_stage1 = l_stage0 + 1

        sL_slot = sL0_writer.acquire(l_stage0)
        tl.store(tle.gpu.local_ptr(sL_slot.sL), e_sum)
        sL0_writer.commit(l_stage0)

        peer_l_wait = sL1_reader.wait(l_stage1)
        total_sum = e_sum + tl.load(tle.gpu.local_ptr(peer_l_wait.slot.sL))
        sL1_reader.release(l_stage1)

        valid = total_sum > 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)

        output_row = batch_idx * head_num + head_base
        if is_no_split:
            out_vals = acc * inv_total_sum[:, None]
            out_vals = tl.where(valid[:, None], out_vals, 0.0)
            out_vals_bf16 = out_vals.to(O.dtype.element_ty)
            tl.store(tle.gpu.local_ptr(q_slot.sQ_l), out_vals_bf16, mask=mask_h[:, None])
            tle.gpu.copy(q_slot.sQ_l, Output_desc, [BLOCK_M, HALF_DIM_V], [output_row, 0])
        else:
            for tile in tl.static_range(0, HALF_DIM_V, D_CHUNK):
                out_tile = tle.extract_tile(
                    acc,
                    index=[0, tile // D_CHUNK],
                    tile_shape=(BLOCK_M, D_CHUNK),
                )
                out_tile = tl.where(
                    valid[:, None], out_tile * inv_total_sum[:, None], 0.0
                )
                stage_cols = tl.broadcast_to(((tile + offs_chunk) * 2)[None, :], (BLOCK_M, D_CHUNK))
                stage_ptrs = tle.gpu.local_ptr(sO_stage, (stage_batch_idx, stage_rows, stage_cols))
                oaccum_ptrs = (
                    O_accum
                    + split_idx * stride_oaccum_split
                    + head_offsets[:, None] * stride_oaccum_h
                    + (tile + offs_chunk)[None, :]
                )
                store_mask = tl.broadcast_to(mask_h[:, None], (BLOCK_M, D_CHUNK)).to(tl.int32)
                stage_addr = tl.inline_asm_elementwise(
                    asm="mov.u64 $0, $1;",
                    constraints="=l,l",
                    args=[stage_ptrs],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
                oaccum_addr = tl.inline_asm_elementwise(
                    asm="mov.u64 $0, $1;",
                    constraints="=l,l",
                    args=[oaccum_ptrs],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
                out_bits = out_tile.to(tl.uint32, bitcast=True)
                dummy_store0 = tl.inline_asm_elementwise(
                    asm="""
                    {
                        .reg .b32 tmp;
                        .reg .pred p;
                        setp.ne.u32 p, $4, 0;
                        @p st.shared.b32 [$1], $3;
                        @p ld.shared.b32 tmp, [$1];
                        @p st.global.b32 [$2], tmp;
                        mov.u32 $0, 0;
                    }
                    """,
                    constraints="=r,l,l,r,r",
                    args=[stage_addr, oaccum_addr, out_bits, store_mask],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                dummy_store0 += 0

        if n_pairs > 0:
            last_pipe_idx = pipe_base + n_pairs - 1
            k0_l_reader.release(last_pipe_idx)
            k0_r_qk_reader.release(last_pipe_idx)
        q_reader.release(q_stage)
        pipe_base += n_pairs

@triton.jit
def _flash_mla_ws_consumer1(
    k1_l_qk_reader,
    k1_r_reader,
    k0_r_remote_reader,
    valid_reader,
    sM_wg1_writer,
    sM_wg0_reader,
    sP1_writer,
    sP0_reader,
    sL1_writer,
    sL0_reader,
    q_reader,
    sO_stage,
    Q_desc,
    Q_tail_desc,
    Output_desc,
    O,
    O_accum,
    LSE_accum,
    sm_scale,
    B_seq_len,
    Num_splits,
    begin_req_idx,
    end_req_idx,
    begin_block_idx_meta,
    end_block_idx_meta,
    begin_split_idx,
    is_first_req_splitted,
    is_last_req_splitted,
    head_base,
    head_num,
    stride_o_b,
    stride_o_h,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    unused_q_desc = Q_desc
    unused_q_tail_desc = Q_tail_desc
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    offs_dv = tl.arange(0, HALF_DIM_V)
    mask_h = head_offsets < head_num

    kv_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, HALF_DIM_V))
    kv_cols_l = tl.broadcast_to(tl.arange(0, HALF_DIM_V)[None, :], (BLOCK_N, HALF_DIM_V))
    kv_cols_r = tl.broadcast_to((HALF_DIM_V + tl.arange(0, HALF_DIM_V))[None, :], (BLOCK_N, HALF_DIM_V))
    stage_rows = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, D_CHUNK))
    stage_batch_idx = tl.full((BLOCK_M, D_CHUNK), 0, dtype=tl.int32)
    if HAVE_TAIL:
        kv_rows_tail = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, D_CHUNK))
        kv_cols_tail = tl.broadcast_to(tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK))

    offs_chunk = tl.arange(0, D_CHUNK)
    pipe_base = 0
    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        q_stage = batch_idx - begin_req_idx
        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        n_split_idx = tl.where(batch_idx == begin_req_idx, begin_split_idx, 0)
        no_split_middle = (batch_idx != begin_req_idx) & (batch_idx != end_req_idx)
        no_split_first = (batch_idx == begin_req_idx) & (~is_first_req_splitted)
        no_split_last = (batch_idx == end_req_idx) & (~is_last_req_splitted)
        is_no_split = no_split_middle | no_split_first | no_split_last
        if begin_req_idx == end_req_idx:
            is_no_split = ~is_first_req_splitted
        split_idx = tl.load(Num_splits + batch_idx) + n_split_idx

        q_wait = q_reader.wait(q_stage)
        q_slot = q_wait.slot

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HALF_DIM_V], dtype=tl.float32)

        n_pairs = tl.cdiv(end_block_idx - start_block_idx, 2)
        for pair in tl.range(0, n_pairs):
            pipe_idx = pipe_base + pair
            k1_r_wait = k1_r_reader.wait(pipe_idx)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            k1_r_slot = k1_r_wait.slot
            k1_r = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))
            q_r = tl.load(tle.gpu.local_ptr(q_slot.sQ_r))
            qk = tl.dot(q_r, tl.trans(k1_r), qk, out_dtype=tl.float32)
            if HAVE_TAIL:
                k1_tail = tl.load(tle.gpu.local_ptr(k1_r_slot.sK_tail, (kv_rows_tail, kv_cols_tail)))
                q_tail = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                qk = tl.dot(q_tail, tl.trans(k1_tail), qk, out_dtype=tl.float32)

            k1_l_wait = k1_l_qk_reader.wait(pipe_idx)
            k1_l_slot = k1_l_wait.slot
            k1_l = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols_l)))
            q_l = tl.load(tle.gpu.local_ptr(q_slot.sQ_l))
            qk = tl.dot(q_l, tl.trans(k1_l), qk, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pipe_idx)
            valid_row = tl.full([BLOCK_N], 1, dtype=tl.int32)
            valid_n = (
                tl.load(tle.gpu.local_ptr(valid_wait.slot.valid, (valid_row, tl.arange(0, BLOCK_N)))) != 0
            )
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))
            valid_reader.release(pipe_idx)

            peer_wait = sM_wg0_reader.wait(pipe_idx)
            peer_max = tl.load(tle.gpu.local_ptr(peer_wait.slot.sM))
            sM_wg0_reader.release(pipe_idx)
            local_max = tl.maximum(e_max, tl.max(qk, axis=1))
            merged_max = tl.maximum(local_max, peer_max)
            sM_slot = sM_wg1_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sM_slot.sM), merged_max)
            sM_wg1_writer.commit(pipe_idx)

            re_scale = tl.exp(e_max - merged_max)
            p = tl.exp(qk - merged_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            acc = acc * re_scale[:, None]
            p_b = p.to(k1_r.dtype)

            v_r = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))
            acc = tl.dot(p_b, v_r, acc, out_dtype=tl.float32)
            if pair != n_pairs - 1:
                k1_l_qk_reader.release(pipe_idx)

            sP_slot = sP1_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_b)
            sP1_writer.commit(pipe_idx)

            sP0_wait = sP0_reader.wait(pipe_idx)
            p0 = tl.load(tle.gpu.local_ptr(sP0_wait.slot.sP))
            k0_r_wait = k0_r_remote_reader.wait(pipe_idx)
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols_r)))
            acc = tl.dot(p0, k0_r, acc, out_dtype=tl.float32)
            sP0_reader.release(pipe_idx)
            k0_r_remote_reader.release(pipe_idx)
            if pair != n_pairs - 1:
                k1_r_reader.release(pipe_idx)
            e_max = merged_max

        l_stage0 = q_stage * 2
        l_stage1 = l_stage0 + 1
        sL_slot = sL1_writer.acquire(l_stage1)
        tl.store(tle.gpu.local_ptr(sL_slot.sL), e_sum)
        sL1_writer.commit(l_stage1)
        peer_l_wait = sL0_reader.wait(l_stage0)
        total_sum = e_sum + tl.load(tle.gpu.local_ptr(peer_l_wait.slot.sL))
        sL0_reader.release(l_stage0)
        valid = total_sum > 0.0
        lse_vals = tl.where(valid, tl.log(total_sum) + e_max, float("-inf"))
        inv_total_sum = tl.fdiv(1.0, total_sum)
        output_row = batch_idx * head_num + head_base
        if is_no_split:
            out_vals = acc * inv_total_sum[:, None]
            out_vals = tl.where(valid[:, None], out_vals, 0.0)
            out_vals_bf16 = out_vals.to(O.dtype.element_ty)
            tl.store(tle.gpu.local_ptr(q_slot.sQ_r), out_vals_bf16, mask=mask_h[:, None])
            tle.gpu.copy(q_slot.sQ_r, Output_desc, [BLOCK_M, HALF_DIM_V], [output_row, HALF_DIM_V])
        else:
            for tile in tl.static_range(0, HALF_DIM_V, D_CHUNK):
                out_tile = tle.extract_tile(
                    acc,
                    index=[0, tile // D_CHUNK],
                    tile_shape=(BLOCK_M, D_CHUNK),
                )
                out_tile = tl.where(
                    valid[:, None], out_tile * inv_total_sum[:, None], 0.0
                )
                stage_cols = tl.broadcast_to(((tile + offs_chunk) * 2)[None, :], (BLOCK_M, D_CHUNK))
                stage_ptrs = tle.gpu.local_ptr(sO_stage, (stage_batch_idx, stage_rows, stage_cols))
                oaccum_ptrs = (
                    O_accum
                    + split_idx * stride_oaccum_split
                    + head_offsets[:, None] * stride_oaccum_h
                    + (HALF_DIM_V + tile + offs_chunk)[None, :]
                )
                store_mask = tl.broadcast_to(mask_h[:, None], (BLOCK_M, D_CHUNK)).to(tl.int32)
                stage_addr = tl.inline_asm_elementwise(
                    asm="mov.u64 $0, $1;",
                    constraints="=l,l",
                    args=[stage_ptrs],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
                oaccum_addr = tl.inline_asm_elementwise(
                    asm="mov.u64 $0, $1;",
                    constraints="=l,l",
                    args=[oaccum_ptrs],
                    dtype=tl.uint64,
                    is_pure=True,
                    pack=1,
                )
                out_bits = out_tile.to(tl.uint32, bitcast=True)
                dummy_store1 = tl.inline_asm_elementwise(
                    asm="""
                    {
                        .reg .b32 tmp;
                        .reg .pred p;
                        setp.ne.u32 p, $4, 0;
                        @p st.shared.b32 [$1], $3;
                        @p ld.shared.b32 tmp, [$1];
                        @p st.global.b32 [$2], tmp;
                        mov.u32 $0, 0;
                    }
                    """,
                    constraints="=r,l,l,r,r",
                    args=[stage_addr, oaccum_addr, out_bits, store_mask],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                dummy_store1 += 0
            tl.store(LSE_accum + split_idx * stride_lseaccum_split + head_offsets * stride_lseaccum_h,
                lse_vals, mask=mask_h)
        if n_pairs > 0:
            last_pipe_idx = pipe_base + n_pairs - 1
            k1_l_qk_reader.release(last_pipe_idx)
            k1_r_reader.release(last_pipe_idx)
        q_reader.release(q_stage)
        pipe_base += n_pairs


@triton.jit
def flash_mla_splitkv_ws_tle_kernel(
    Q_desc,
    Q_tail_desc,
    Output_desc,
    Kv_desc,
    Kv,
    Block_table,
    B_seq_len,
    Sched_meta,
    Num_splits,
    O,
    O_accum,
    LSE_accum,
    sm_scale,
    head_num,
    stride_q_b,
    stride_q_h,
    stride_kv_token,
    stride_block_table_b,
    stride_o_b,
    stride_o_h,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_CHUNK: tl.constexpr,
    META_FIELDS: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    partition_idx = tl.program_id(1)
    meta_base = Sched_meta + partition_idx * META_FIELDS
    begin_req_idx = tl.load(meta_base + 0)
    end_req_idx = tl.load(meta_base + 1)
    begin_block_idx_meta = tl.load(meta_base + 2)
    end_block_idx_meta = tl.load(meta_base + 3)
    begin_split_idx = tl.load(meta_base + 4)
    is_first_req_splitted = tl.load(meta_base + 5) != 0
    is_last_req_splitted = tl.load(meta_base + 6) != 0
    head_base = m_block_idx * BLOCK_M
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    HAVE_TAIL: tl.constexpr = HEAD_DIM > HEAD_DIM_V

    sQ_l_smem = tle.gpu.alloc(
        [1, BLOCK_M, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sQ_r_smem = tle.gpu.alloc(
        [1, BLOCK_M, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    if HAVE_TAIL:
        sQ_tail_smem = tle.gpu.alloc(
            [1, BLOCK_M, D_CHUNK],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        q_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_q",
            readers=("wg0", "wg1"),
            sQ_l=sQ_l_smem,
            sQ_r=sQ_r_smem,
            sQ_tail=sQ_tail_smem,
        )
    else:
        q_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_q",
            readers=("wg0", "wg1"),
            sQ_l=sQ_l_smem,
            sQ_r=sQ_r_smem,
        )
    sK0 = tle.gpu.alloc(
        [1, BLOCK_N, HEAD_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK1 = tle.gpu.alloc(
        [1, BLOCK_N, HEAD_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    if HAVE_TAIL:
        sK0_tail = tle.gpu.alloc(
            [1, BLOCK_N, D_CHUNK],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sK1_tail = tle.gpu.alloc(
            [1, BLOCK_N, D_CHUNK],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sP0_smem = sK0_tail
        sP1_smem = sK1_tail
    else:
        sP0_smem = tle.gpu.alloc(
            [1, BLOCK_M, BLOCK_N],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sP1_smem = tle.gpu.alloc(
            [1, BLOCK_M, BLOCK_N],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    valid_smem = tle.gpu.alloc(
        [1, 2, BLOCK_N],
        dtype=tl.int8,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    sM_smem = tle.gpu.alloc(
        [1, BLOCK_M],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    sL_smem = tle.gpu.alloc(
        [2, BLOCK_M],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    k0_l_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="flash_mla_ws_k0_l",
        sK=sK0,
    )
    if HAVE_TAIL:
        k0_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k0_r",
            readers=("qk", "remote"),
            sK=sK0,
            sK_tail=sK0_tail,
        )
    else:
        k0_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k0_r",
            readers=("qk", "remote"),
            sK=sK0,
        )
    k1_l_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="flash_mla_ws_k1_l",
        readers=("qk", "remote"),
        sK=sK1,
    )
    if HAVE_TAIL:
        k1_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k1_r",
            sK=sK1,
            sK_tail=sK1_tail,
        )
    else:
        k1_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k1_r",
            sK=sK1,
        )
    valid_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_valid", readers=("wg0", "wg1"), valid=valid_smem)
    sM_wg0_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_m0", sM=sM_smem)
    sM_wg1_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_m1", sM=sM_smem)
    sP0_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_p0", sP=sP0_smem)
    sP1_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_p1", sP=sP1_smem)
    sL0_pipe = tle.pipe(capacity=2, scope="cta", name="flash_mla_ws_l0", sL=sL_smem)
    sL1_pipe = tle.pipe(capacity=2, scope="cta", name="flash_mla_ws_l1", sL=sL_smem)

    tle.gpu.warp_specialize(
        [
            (
                _flash_mla_ws_consumer0,
                (
                    k0_l_pipe.reader(),
                    k0_r_pipe.reader("qk"),
                    k1_l_pipe.reader("remote", fields=("sK",)),
                    valid_pipe.reader("wg0"),
                    sM_wg0_pipe.writer(),
                    sM_wg1_pipe.reader(),
                    sP0_pipe.writer(),
                    sP1_pipe.reader(),
                    sL0_pipe.writer(),
                    sL1_pipe.reader(),
                    q_pipe.writer(),
                    q_pipe.reader("wg0"),
                    sK0,
                    Q_desc,
                    Q_tail_desc,
                    Output_desc,
                    O,
                    O_accum,
                    LSE_accum,
                    sm_scale,
                    B_seq_len,
                    Num_splits,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    begin_split_idx,
                    is_first_req_splitted,
                    is_last_req_splitted,
                    head_base,
                    head_num,
                    stride_o_b,
                    stride_o_h,
                    stride_oaccum_split,
                    stride_oaccum_h,
                    stride_lseaccum_split,
                    stride_lseaccum_h,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM_V,
                    D_CHUNK,
                    HAVE_TAIL,
                ),
            ),
            (
                _flash_mla_ws_consumer1,
                (
                    k1_l_pipe.reader("qk"),
                    k1_r_pipe.reader(),
                    k0_r_pipe.reader("remote", fields=("sK",)),
                    valid_pipe.reader("wg1"),
                    sM_wg1_pipe.writer(),
                    sM_wg0_pipe.reader(),
                    sP1_pipe.writer(),
                    sP0_pipe.reader(),
                    sL1_pipe.writer(),
                    sL0_pipe.reader(),
                    q_pipe.reader("wg1"),
                    sK1,
                    Q_desc,
                    Q_tail_desc,
                    Output_desc,
                    O,
                    O_accum,
                    LSE_accum,
                    sm_scale,
                    B_seq_len,
                    Num_splits,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    begin_split_idx,
                    is_first_req_splitted,
                    is_last_req_splitted,
                    head_base,
                    head_num,
                    stride_o_b,
                    stride_o_h,
                    stride_oaccum_split,
                    stride_oaccum_h,
                    stride_lseaccum_split,
                    stride_lseaccum_h,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM_V,
                    D_CHUNK,
                    HAVE_TAIL,
                ),
            ),
            (
                _flash_mla_ws_kv_producer,
                (
                    k0_l_pipe.writer(),
                    k0_r_pipe.writer(),
                    k1_l_pipe.writer(),
                    k1_r_pipe.writer(),
                    valid_pipe.writer(),
                    Kv,
                    Kv_desc,
                    Block_table,
                    B_seq_len,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    stride_kv_token,
                    stride_block_table_b,
                    BLOCK_N,
                    PAGE_SIZE,
                    HEAD_DIM_V,
                    D_CHUNK,
                    HAVE_TAIL,
                ),
            ),
        ],
        [4, 4],
        [216, 72],
    )


@triton.jit
def flash_mla_combine_kernel(
    O_accum,
    LSE_accum,
    Num_splits,
    O,
    head_num,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    stride_o_b,
    stride_o_h,
    BLOCK_H: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    h_block_idx = tl.program_id(1)
    offs_h = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < head_num
    offs_d = tl.arange(0, HEAD_DIM_V)

    start_split = tl.load(Num_splits + batch_idx)
    end_split = tl.load(Num_splits + batch_idx + 1)
    my_num_splits = end_split - start_split
    if my_num_splits != 1:
        max_lse = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
        for s in tl.static_range(0, MAX_SPLITS):
            active = s < my_num_splits
            lse_s = tl.load(
                LSE_accum
                + (start_split + s) * stride_lseaccum_split
                + offs_h * stride_lseaccum_h,
                mask=active & mask_h,
                other=float("-inf"),
            )
            max_lse = tl.maximum(max_lse, lse_s)

        acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)
        sum_w = tl.zeros([BLOCK_H], dtype=tl.float32)
        for s in tl.static_range(0, MAX_SPLITS):
            active = s < my_num_splits
            lse_s = tl.load(
                LSE_accum
                + (start_split + s) * stride_lseaccum_split
                + offs_h * stride_lseaccum_h,
                mask=active & mask_h,
                other=float("-inf"),
            )
            w = tl.exp(lse_s - max_lse)
            sum_w += tl.where(active, w, 0.0)
            o_s = tl.load(
                O_accum
                + (start_split + s) * stride_oaccum_split
                + offs_h[:, None] * stride_oaccum_h
                + offs_d[None, :],
                mask=active & mask_h[:, None],
                other=0.0,
            )
            acc += w[:, None] * o_s

        acc = acc * tl.fdiv(1.0, sum_w)[:, None]
        tl.store(
            O + batch_idx * stride_o_b + offs_h[:, None] * stride_o_h + offs_d[None, :],
            acc.to(O.dtype.element_ty),
            mask=mask_h[:, None],
        )


def _try_flash_mla_tle(
    q: torch.Tensor,
    block_table: torch.Tensor,
    blocked_k: torch.Tensor,
    block_size: int,
    b: int,
    s_q: int,
    cache_seqlens: torch.Tensor,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    causal: bool,
) -> torch.Tensor | None:
    if not _can_use_tle_flash_mla(
        q,
        block_table,
        blocked_k,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    ):
        return None

    q_tle = q.contiguous()
    kv_flat = blocked_k.contiguous().view(-1, d)
    block_table_tle = block_table.contiguous()
    cache_seqlens_tle = cache_seqlens.contiguous()
    from triton.tools.tensor_descriptor import TensorDescriptor

    _set_triton_descriptor_allocator(q.device)
    q_flat = q_tle.view(b * s_q * h_q, d)
    num_m_blocks = triton.cdiv(s_q * h_q // h_kv, FLASH_MLA_BLOCK_M)
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sm_parts = max(num_sms // h_kv // num_m_blocks, 1)
    sched_meta = torch.empty(
        (num_sm_parts, FLASH_MLA_META_FIELDS),
        dtype=torch.int32,
        device=q.device,
    )
    num_splits = torch.empty(b + 1, dtype=torch.int32, device=q.device)
    flash_mla_sched_meta_kernel[(1,)](
        cache_seqlens_tle,
        sched_meta,
        num_splits,
        BLOCK_B=triton.next_power_of_2(b),
        BATCH_SIZE=b,
        BLOCK_SIZE_N=FLASH_MLA_BLOCK_N,
        FIXED_OVERHEAD_NUM_BLOCKS=FLASH_MLA_FIXED_OVERHEAD_BLOCKS,
        NUM_SM_PARTS=num_sm_parts,
        META_FIELDS=FLASH_MLA_META_FIELDS,
        num_warps=1,
        num_stages=1,
    )
    total_num_splits = b + num_sm_parts
    out = torch.empty((b * s_q, h_q, dv), dtype=q.dtype, device=q.device)
    out_accum = torch.empty(
        (total_num_splits, h_q, dv), dtype=torch.float32, device=q.device
    )
    lse_accum = torch.empty(
        (total_num_splits, h_q), dtype=torch.float32, device=q.device
    )
    out_flat = out.view(b * s_q * h_q, dv)
    d_chunk = 64
    q_desc = TensorDescriptor(
        q_flat,
        shape=[b * s_q * h_q, d],
        strides=[d, 1],
        block_shape=[FLASH_MLA_BLOCK_M, dv // 2],
    )
    q_tail_desc = TensorDescriptor(
        q_flat,
        shape=[b * s_q * h_q, d],
        strides=[d, 1],
        block_shape=[FLASH_MLA_BLOCK_M, d_chunk],
    )
    output_desc = TensorDescriptor(
        out_flat,
        shape=[b * s_q * h_q, dv],
        strides=[dv, 1],
        block_shape=[FLASH_MLA_BLOCK_M, dv // 2],
    )
    kv_desc = TensorDescriptor(
        kv_flat,
        shape=[kv_flat.shape[0], d],
        strides=[d, 1],
        block_shape=[FLASH_MLA_BLOCK_N, d_chunk],
    )

    flash_mla_splitkv_ws_tle_kernel[(num_m_blocks, num_sm_parts)](
        q_desc,
        q_tail_desc,
        output_desc,
        kv_desc,
        kv_flat,
        block_table_tle,
        cache_seqlens_tle,
        sched_meta,
        num_splits,
        out,
        out_accum,
        lse_accum,
        1 / math.sqrt(d),
        h_q,
        q_tle.stride(0),
        q_tle.stride(2),
        kv_flat.stride(0),
        block_table_tle.stride(0),
        out.stride(0),
        out.stride(1),
        out_accum.stride(0),
        out_accum.stride(1),
        lse_accum.stride(0),
        lse_accum.stride(1),
        BLOCK_M=FLASH_MLA_BLOCK_M,
        BLOCK_N=FLASH_MLA_BLOCK_N,
        PAGE_SIZE=block_size,
        HEAD_DIM_V=dv,
        HEAD_DIM=d,
        D_CHUNK=d_chunk,
        META_FIELDS=FLASH_MLA_META_FIELDS,
        num_warps=4,
        num_stages=1,
    )

    flash_mla_combine_kernel[(b, triton.cdiv(h_q, FLASH_MLA_COMBINE_BLOCK_H))](
        out_accum,
        lse_accum,
        num_splits,
        out,
        h_q,
        out_accum.stride(0),
        out_accum.stride(1),
        lse_accum.stride(0),
        lse_accum.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_H=FLASH_MLA_COMBINE_BLOCK_H,
        HEAD_DIM_V=dv,
        MAX_SPLITS=num_sm_parts,
        num_warps=FLASH_MLA_COMBINE_BLOCK_H,
        num_stages=1,
    )
    return out.view([b, s_q, h_q, dv])


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": h, "BLOCK_N": n}, num_warps=w, num_stages=s)
#         for h in [32, 64, 128]
#         for n in [32, 64, 128]
#         for w in [4, 8]
#         for s in [1, 2]
#     ],
#     key=["head_num"]
# )
@triton.heuristics(
    values={
        "EVEN_H": lambda META: META["head_num"] % META["BLOCK_H"] == 0,
    }
)
@triton.jit
def flash_mla_attn_kernel(
    Q_ptr,
    Kv_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    sm_scale,
    head_num,
    stride_q_bs,
    stride_q_h,
    stride_kv_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_head_id = ext.program_id(0)
    cur_batch_id = ext.program_id(1)
    Req_to_tokens += stride_req_to_tokens_bs * cur_batch_id

    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)

    offs_d_ckv = tl.arange(0, HEAD_DIM_V)
    offs_q_nope = (
        cur_batch_id * stride_q_bs
        + cur_head[:, None] * stride_q_h
        + offs_d_ckv[None, :]
    )

    offs_d_kpe = tl.arange(HEAD_DIM_V, HEAD_DIM)
    offs_q_pe = (
        cur_batch_id * stride_q_bs
        + cur_head[:, None] * stride_q_h
        + offs_d_kpe[None, :]
    )

    if EVEN_H:
        q_nope = tl.load(Q_ptr + offs_q_nope)
        q_pe = tl.load(Q_ptr + offs_q_pe)
    else:
        mask_head = cur_head < head_num
        q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None])
        q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None])

    e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch_id)
    loop_time = cur_batch_seq_len // BLOCK_N
    remainder = cur_batch_seq_len % BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    for i in range(0, loop_time):
        kv_page_number = tl.load(Req_to_tokens + offs_n // PAGE_SIZE)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
        v_c = tl.load(Kv_cache + offs_v_c)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)  # qk_nope

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
        k_pe = tl.load(Kv_cache + offs_k_pe)

        qk = tl.dot(q_pe, k_pe, acc=qk)  # qk_rope
        qk *= sm_scale

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N

    if remainder:
        mask_kvsplit = offs_n < cur_batch_seq_len
        kv_page_number = tl.load(
            Req_to_tokens + offs_n // PAGE_SIZE,
            mask=mask_kvsplit,
            other=0,
        )
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
        v_c = tl.load(Kv_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)  # qk_nope

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
        k_pe = tl.load(Kv_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)

        qk = tl.dot(q_pe, k_pe, acc=qk)  # qk_rope
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)

        e_sum = e_sum * re_scale + tl.sum(p, 1)

    offs_o = (
        cur_batch_id * stride_o_b + cur_head[:, None] * stride_o_h + offs_d_ckv[None, :]
    )
    if EVEN_H:
        tl.store(
            O + offs_o,
            acc / e_sum[:, None],
        )
    else:
        tl.store(O + offs_o, acc / e_sum[:, None], mask=mask_head[:, None])


def flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
):
    logger.debug("GEMS FLASH MLA")
    assert causal, "causal False not supported"
    assert d > dv, "mla with rope dim should be larger than no rope dim"

    tle_out = _try_flash_mla_tle(
        q,
        block_table,
        blocked_k,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    )
    if tle_out is not None:
        return tle_out

    batch_size, s_q, head_num, d = list(q.shape)
    q = q.view([-1, head_num, d]).contiguous()
    blocked_k = blocked_k.view([-1, d]).contiguous()
    block_table = block_table.contiguous()
    cache_seqlens = cache_seqlens.contiguous()

    sm_scale = 1 / math.sqrt(d)

    o = torch.empty([b * s_q, h_q, dv], dtype=q.dtype, device=device)

    major, _ = get_device_capability()
    if major == 9:
        BLOCK_H = 64
        num_stages = 3
    elif major == 8:
        BLOCK_H = 32
        num_stages = 2
    elif major == 7 and vendor_name == "iluvatar":
        BLOCK_H = 32
        num_stages = 1
    elif major == 3 and vendor_name == "mthreads":
        BLOCK_H = 32
        num_stages = 1
    else:
        error.backend_not_support(device)
    BLOCK_N = 64
    grid = (
        triton.cdiv(head_num, BLOCK_H),
        batch_size,
    )
    with torch_device_fn.device(device):
        flash_mla_attn_kernel[grid](
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            o,
            sm_scale,
            head_num,
            # stride
            q.stride(0),
            q.stride(1),
            blocked_k.stride(-2),
            block_table.stride(0),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            BLOCK_H=BLOCK_H,
            BLOCK_N=BLOCK_N,
            PAGE_SIZE=block_size,
            HEAD_DIM_V=dv,
            HEAD_DIM=d,
            num_warps=8,
            num_stages=num_stages,
        )

    return o.view([b, s_q, h_q, dv])
