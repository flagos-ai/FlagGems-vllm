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
    kv0_writer,
    kv1_writer,
    kv2_writer,
    kv3_writer,
    kv4_writer,
    kv5_writer,
    kv6_writer,
    kv7_writer,
    kv_tail_writer,
    Kv_desc,
    Block_table,
    block_table_base,
    start_block_idx,
    end_block_idx,
    seq_len,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
):
    _ = seq_len
    for block_idx in tl.range(start_block_idx, end_block_idx):
        pipe_idx = block_idx - start_block_idx
        page_id = tle.load(Block_table + block_table_base + block_idx)
        kv_row = (page_id * PAGE_SIZE).to(tl.int32)

        kv0_slot = kv0_writer.acquire(pipe_idx)
        tle.gpu.copy(Kv_desc, kv0_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 0])
        kv0_writer.commit(pipe_idx)
        for chunk in tl.static_range(0, 8):
            if chunk == 0:
                pass
            elif chunk == 1:
                kv_slot = kv1_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 64])
                kv1_writer.commit(pipe_idx)
            elif chunk == 2:
                kv_slot = kv2_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 128])
                kv2_writer.commit(pipe_idx)
            elif chunk == 3:
                kv_slot = kv3_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 192])
                kv3_writer.commit(pipe_idx)
            elif chunk == 4:
                kv_slot = kv4_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 256])
                kv4_writer.commit(pipe_idx)
            elif chunk == 5:
                kv_slot = kv5_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 320])
                kv5_writer.commit(pipe_idx)
            elif chunk == 6:
                kv_slot = kv6_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 384])
                kv6_writer.commit(pipe_idx)
            else:
                kv_slot = kv7_writer.acquire(pipe_idx)
                tle.gpu.copy(Kv_desc, kv_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, 448])
                kv7_writer.commit(pipe_idx)

        kv_tail_slot = kv_tail_writer.acquire(pipe_idx)
        tle.gpu.copy(
            Kv_desc, kv_tail_slot.sKV, [BLOCK_N, D_CHUNK], [kv_row, HEAD_DIM_V]
        )
        kv_tail_writer.commit(pipe_idx)


@triton.jit
def _flash_mla_ws_consumer(
    kv0_reader,
    kv1_reader,
    kv2_reader,
    kv3_reader,
    kv4_reader,
    kv5_reader,
    kv6_reader,
    kv7_reader,
    kv_tail_reader,
    Q_desc,
    Q_tail_desc,
    Num_splits,
    O,
    O_accum,
    LSE_accum,
    sm_scale,
    batch_idx,
    split_idx,
    is_no_split,
    start_block_idx,
    end_block_idx,
    seq_len,
    q_row,
    out_row_base,
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
):
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    offs_dv = tl.arange(0, HEAD_DIM_V)
    mask_h = head_offsets < head_num

    kv_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, D_CHUNK))
    kv_cols = tl.broadcast_to(tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK))

    # Q 只在这里 load 一次；后面 QK 阶段 extract_tile(q_nope)，PV 阶段不再读 Q
    q_nope = Q_desc.load([q_row, 0])

    # 注意：如果 Q_tail_desc 已经是 tail view，这里应该是 [q_row, 0]
    # 如果 Q_tail_desc 仍然是完整 Q descriptor，这里才是 [q_row, HEAD_DIM_V]
    q_pe = Q_tail_desc.load([q_row, HEAD_DIM_V])

    e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    for block_idx in tl.range(start_block_idx, end_block_idx):
        pipe_idx = block_idx - start_block_idx

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # =========================
        # QK phase
        # =========================
        wait0 = kv0_reader.wait(pipe_idx)
        slot0 = wait0.slot
        q_d0 = tle.extract_tile(q_nope, index=[0, 0], tile_shape=(BLOCK_M, D_CHUNK))
        k_d0 = tl.load(tle.gpu.local_ptr(slot0.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d0, tl.trans(k_d0), qk, out_dtype=tl.float32)

        wait1 = kv1_reader.wait(pipe_idx)
        slot1 = wait1.slot
        q_d1 = tle.extract_tile(q_nope, index=[0, 1], tile_shape=(BLOCK_M, D_CHUNK))
        k_d1 = tl.load(tle.gpu.local_ptr(slot1.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d1, tl.trans(k_d1), qk, out_dtype=tl.float32)

        wait2 = kv2_reader.wait(pipe_idx)
        slot2 = wait2.slot
        q_d2 = tle.extract_tile(q_nope, index=[0, 2], tile_shape=(BLOCK_M, D_CHUNK))
        k_d2 = tl.load(tle.gpu.local_ptr(slot2.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d2, tl.trans(k_d2), qk, out_dtype=tl.float32)

        wait3 = kv3_reader.wait(pipe_idx)
        slot3 = wait3.slot
        q_d3 = tle.extract_tile(q_nope, index=[0, 3], tile_shape=(BLOCK_M, D_CHUNK))
        k_d3 = tl.load(tle.gpu.local_ptr(slot3.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d3, tl.trans(k_d3), qk, out_dtype=tl.float32)

        wait4 = kv4_reader.wait(pipe_idx)
        slot4 = wait4.slot
        q_d4 = tle.extract_tile(q_nope, index=[0, 4], tile_shape=(BLOCK_M, D_CHUNK))
        k_d4 = tl.load(tle.gpu.local_ptr(slot4.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d4, tl.trans(k_d4), qk, out_dtype=tl.float32)

        wait5 = kv5_reader.wait(pipe_idx)
        slot5 = wait5.slot
        q_d5 = tle.extract_tile(q_nope, index=[0, 5], tile_shape=(BLOCK_M, D_CHUNK))
        k_d5 = tl.load(tle.gpu.local_ptr(slot5.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d5, tl.trans(k_d5), qk, out_dtype=tl.float32)

        wait6 = kv6_reader.wait(pipe_idx)
        slot6 = wait6.slot
        q_d6 = tle.extract_tile(q_nope, index=[0, 6], tile_shape=(BLOCK_M, D_CHUNK))
        k_d6 = tl.load(tle.gpu.local_ptr(slot6.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d6, tl.trans(k_d6), qk, out_dtype=tl.float32)

        wait7 = kv7_reader.wait(pipe_idx)
        slot7 = wait7.slot
        q_d7 = tle.extract_tile(q_nope, index=[0, 7], tile_shape=(BLOCK_M, D_CHUNK))
        k_d7 = tl.load(tle.gpu.local_ptr(slot7.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_d7, tl.trans(k_d7), qk, out_dtype=tl.float32)

        tail_wait = kv_tail_reader.wait(pipe_idx)
        tail_slot = tail_wait.slot
        k_tail = tl.load(tle.gpu.local_ptr(tail_slot.sKV, (kv_rows, kv_cols)))
        qk = tl.dot(q_pe, tl.trans(k_tail), qk, out_dtype=tl.float32)

        # =========================
        # Online softmax
        # =========================
        valid_n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N) < seq_len
        qk *= sm_scale * 1.4426950408889634
        qk = tl.where(valid_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])

        # =========================
        # PV phase: O += P @ V
        # 这里不再读 Q，只读 p 和 slot*.sKV
        # =========================
        acc0 = tle.extract_tile(acc, index=[0, 0], tile_shape=(BLOCK_M, D_CHUNK))
        acc0 *= re_scale[:, None]
        v_d0 = tl.load(tle.gpu.local_ptr(slot0.sKV, (kv_rows, kv_cols)))
        acc0 = tl.dot(p.to(v_d0.dtype), v_d0, acc0, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc0, index=[0, 0])

        acc1 = tle.extract_tile(acc, index=[0, 1], tile_shape=(BLOCK_M, D_CHUNK))
        acc1 *= re_scale[:, None]
        v_d1 = tl.load(tle.gpu.local_ptr(slot1.sKV, (kv_rows, kv_cols)))
        acc1 = tl.dot(p.to(v_d1.dtype), v_d1, acc1, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc1, index=[0, 1])

        acc2 = tle.extract_tile(acc, index=[0, 2], tile_shape=(BLOCK_M, D_CHUNK))
        acc2 *= re_scale[:, None]
        v_d2 = tl.load(tle.gpu.local_ptr(slot2.sKV, (kv_rows, kv_cols)))
        acc2 = tl.dot(p.to(v_d2.dtype), v_d2, acc2, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc2, index=[0, 2])

        acc3 = tle.extract_tile(acc, index=[0, 3], tile_shape=(BLOCK_M, D_CHUNK))
        acc3 *= re_scale[:, None]
        v_d3 = tl.load(tle.gpu.local_ptr(slot3.sKV, (kv_rows, kv_cols)))
        acc3 = tl.dot(p.to(v_d3.dtype), v_d3, acc3, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc3, index=[0, 3])

        acc4 = tle.extract_tile(acc, index=[0, 4], tile_shape=(BLOCK_M, D_CHUNK))
        acc4 *= re_scale[:, None]
        v_d4 = tl.load(tle.gpu.local_ptr(slot4.sKV, (kv_rows, kv_cols)))
        acc4 = tl.dot(p.to(v_d4.dtype), v_d4, acc4, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc4, index=[0, 4])

        acc5 = tle.extract_tile(acc, index=[0, 5], tile_shape=(BLOCK_M, D_CHUNK))
        acc5 *= re_scale[:, None]
        v_d5 = tl.load(tle.gpu.local_ptr(slot5.sKV, (kv_rows, kv_cols)))
        acc5 = tl.dot(p.to(v_d5.dtype), v_d5, acc5, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc5, index=[0, 5])

        acc6 = tle.extract_tile(acc, index=[0, 6], tile_shape=(BLOCK_M, D_CHUNK))
        acc6 *= re_scale[:, None]
        v_d6 = tl.load(tle.gpu.local_ptr(slot6.sKV, (kv_rows, kv_cols)))
        acc6 = tl.dot(p.to(v_d6.dtype), v_d6, acc6, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc6, index=[0, 6])

        acc7 = tle.extract_tile(acc, index=[0, 7], tile_shape=(BLOCK_M, D_CHUNK))
        acc7 *= re_scale[:, None]
        v_d7 = tl.load(tle.gpu.local_ptr(slot7.sKV, (kv_rows, kv_cols)))
        acc7 = tl.dot(p.to(v_d7.dtype), v_d7, acc7, out_dtype=tl.float32)
        acc = tle.insert_tile(acc, acc7, index=[0, 7])

        # 在 release 前，确保所有使用 slot*.sKV 的 WGMMA 已完成
        # 这个 wait 的目的不是为了 qk/acc 数据依赖，而是为了防止 producer 覆盖 sKV

        kv0_reader.release(pipe_idx)
        kv1_reader.release(pipe_idx)
        kv2_reader.release(pipe_idx)
        kv3_reader.release(pipe_idx)
        kv4_reader.release(pipe_idx)
        kv5_reader.release(pipe_idx)
        kv6_reader.release(pipe_idx)
        kv7_reader.release(pipe_idx)
        kv_tail_reader.release(pipe_idx)

        e_sum = e_sum * re_scale + tl.sum(p, axis=1)
        e_max = n_e_max

    valid = e_sum > 0.0
    out_vals = tl.where(valid[:, None], acc * tl.fdiv(1.0, e_sum)[:, None], 0.0)
    lse_vals = tl.where(valid, tl.log(e_sum) + e_max, float("-inf"))

    if is_no_split:
        tl.store(
            O
            + batch_idx * stride_o_b
            + head_offsets[:, None] * stride_o_h
            + offs_dv[None, :],
            out_vals.to(O.dtype.element_ty),
            mask=mask_h[:, None],
        )
    else:
        tl.store(
            O_accum
            + split_idx * stride_oaccum_split
            + head_offsets[:, None] * stride_oaccum_h
            + offs_dv[None, :],
            out_vals,
            mask=mask_h[:, None],
        )
        tl.store(
            LSE_accum
            + split_idx * stride_lseaccum_split
            + head_offsets * stride_lseaccum_h,
            lse_vals,
            mask=mask_h,
        )


@triton.jit
def flash_mla_splitkv_ws_tle_kernel(
    Q_desc,
    Q_tail_desc,
    Kv_desc,
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

    sKV0 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV1 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV2 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV3 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV4 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV5 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV6 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV7 = tle.gpu.alloc([1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV_tail = tle.gpu.alloc(
        [1, BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem
    )
    kv0_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv0", sKV=sKV0
    )
    kv1_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv1", sKV=sKV1
    )
    kv2_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv2", sKV=sKV2
    )
    kv3_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv3", sKV=sKV3
    )
    kv4_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv4", sKV=sKV4
    )
    kv5_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv5", sKV=sKV5
    )
    kv6_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv6", sKV=sKV6
    )
    kv7_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv7", sKV=sKV7
    )
    kv_tail_pipe = tle.pipe(
        capacity=1, scope="cta", name="flash_mla_ws_kv_tail", sKV=sKV_tail
    )

    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
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
        out_row_base = batch_idx * stride_o_b
        block_table_base = batch_idx * stride_block_table_b
        tle.gpu.warp_specialize(
            [
                (
                    _flash_mla_ws_kv_producer,
                    (
                        kv0_pipe.writer(),
                        kv1_pipe.writer(),
                        kv2_pipe.writer(),
                        kv3_pipe.writer(),
                        kv4_pipe.writer(),
                        kv5_pipe.writer(),
                        kv6_pipe.writer(),
                        kv7_pipe.writer(),
                        kv_tail_pipe.writer(),
                        Kv_desc,
                        Block_table,
                        block_table_base,
                        start_block_idx,
                        end_block_idx,
                        seq_len,
                        BLOCK_N,
                        PAGE_SIZE,
                        HEAD_DIM_V,
                        D_CHUNK,
                    ),
                ),
                (
                    _flash_mla_ws_consumer,
                    (
                        kv0_pipe.reader(),
                        kv1_pipe.reader(),
                        kv2_pipe.reader(),
                        kv3_pipe.reader(),
                        kv4_pipe.reader(),
                        kv5_pipe.reader(),
                        kv6_pipe.reader(),
                        kv7_pipe.reader(),
                        kv_tail_pipe.reader(),
                        Q_desc,
                        Q_tail_desc,
                        Num_splits,
                        O,
                        O_accum,
                        LSE_accum,
                        sm_scale,
                        batch_idx,
                        split_idx,
                        is_no_split,
                        start_block_idx,
                        end_block_idx,
                        seq_len,
                        q_row,
                        out_row_base,
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
                    ),
                ),
            ],
            [4],
            [216],
        )


@triton.jit
def flash_mla_splitkv_tle_kernel(
    Q_desc,
    Q_tail_desc,
    Kv_desc,
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

    offs_h = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_h = offs_h < head_num
    offs_dv = tl.arange(0, HEAD_DIM_V)
    kv_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, D_CHUNK))
    kv_chunk_cols = tl.broadcast_to(tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK))
    sKV0 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV1 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV2 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV3 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV4 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV5 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV6 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV7 = tle.gpu.alloc([BLOCK_N, D_CHUNK], dtype=Kv_desc.dtype, scope=tle.gpu.smem)
    sKV_tail = tle.gpu.alloc(
        [BLOCK_N, D_CHUNK],
        dtype=Kv_desc.dtype,
        scope=tle.gpu.smem,
    )

    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
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

        q_row = (batch_idx * head_num + m_block_idx * BLOCK_M).to(tl.int32)
        q_nope = Q_desc.load([q_row, 0])
        q_pe = Q_tail_desc.load([q_row, HEAD_DIM_V])

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        block_table_base = Block_table + batch_idx * stride_block_table_b

        for block_idx in tl.range(start_block_idx, end_block_idx):
            page_id = tle.load(block_table_base + block_idx)
            token_pos = block_idx * BLOCK_N + offs_n
            valid_n = token_pos < seq_len
            kv_row = (page_id * PAGE_SIZE).to(tl.int32)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for chunk in tl.static_range(0, 8):
                d_start = chunk * 64
                q_d = tle.extract_tile(
                    q_nope,
                    index=[0, chunk],
                    tile_shape=(BLOCK_M, D_CHUNK),
                )
                if chunk == 0:
                    tle.gpu.copy(Kv_desc, sKV0, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV0, (kv_rows, kv_chunk_cols)))
                elif chunk == 1:
                    tle.gpu.copy(Kv_desc, sKV1, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV1, (kv_rows, kv_chunk_cols)))
                elif chunk == 2:
                    tle.gpu.copy(Kv_desc, sKV2, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV2, (kv_rows, kv_chunk_cols)))
                elif chunk == 3:
                    tle.gpu.copy(Kv_desc, sKV3, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV3, (kv_rows, kv_chunk_cols)))
                elif chunk == 4:
                    tle.gpu.copy(Kv_desc, sKV4, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV4, (kv_rows, kv_chunk_cols)))
                elif chunk == 5:
                    tle.gpu.copy(Kv_desc, sKV5, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV5, (kv_rows, kv_chunk_cols)))
                elif chunk == 6:
                    tle.gpu.copy(Kv_desc, sKV6, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV6, (kv_rows, kv_chunk_cols)))
                else:
                    tle.gpu.copy(Kv_desc, sKV7, [BLOCK_N, D_CHUNK], [kv_row, d_start])
                    k_d = tl.load(tle.gpu.local_ptr(sKV7, (kv_rows, kv_chunk_cols)))
                qk = tl.dot(q_d, tl.trans(k_d), qk, out_dtype=tl.float32)
            tle.gpu.copy(
                Kv_desc,
                sKV_tail,
                [BLOCK_N, D_CHUNK],
                [kv_row, HEAD_DIM_V],
            )
            k_tail = tl.load(
                tle.gpu.local_ptr(sKV_tail, (kv_rows, kv_chunk_cols))
            )
            qk = tl.dot(q_pe, tl.trans(k_tail), qk, out_dtype=tl.float32)
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, axis=1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            for chunk in tl.static_range(0, 8):
                if chunk == 0:
                    v_d = tl.load(tle.gpu.local_ptr(sKV0, (kv_rows, kv_chunk_cols)))
                elif chunk == 1:
                    v_d = tl.load(tle.gpu.local_ptr(sKV1, (kv_rows, kv_chunk_cols)))
                elif chunk == 2:
                    v_d = tl.load(tle.gpu.local_ptr(sKV2, (kv_rows, kv_chunk_cols)))
                elif chunk == 3:
                    v_d = tl.load(tle.gpu.local_ptr(sKV3, (kv_rows, kv_chunk_cols)))
                elif chunk == 4:
                    v_d = tl.load(tle.gpu.local_ptr(sKV4, (kv_rows, kv_chunk_cols)))
                elif chunk == 5:
                    v_d = tl.load(tle.gpu.local_ptr(sKV5, (kv_rows, kv_chunk_cols)))
                elif chunk == 6:
                    v_d = tl.load(tle.gpu.local_ptr(sKV6, (kv_rows, kv_chunk_cols)))
                else:
                    v_d = tl.load(tle.gpu.local_ptr(sKV7, (kv_rows, kv_chunk_cols)))
                acc_d = tle.extract_tile(
                    acc, index=[0, chunk], tile_shape=(BLOCK_M, D_CHUNK)
                )
                acc_d *= re_scale[:, None]
                acc_d = tl.dot(p.to(v_d.dtype), v_d, acc_d, out_dtype=tl.float32)
                acc = tle.insert_tile(
                    acc, acc_d, index=[0, chunk]
                )
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            e_max = n_e_max

        valid = e_sum > 0.0
        out_vals = tl.where(valid[:, None], acc * tl.fdiv(1.0, e_sum)[:, None], 0.0)
        lse_vals = tl.where(valid, tl.log(e_sum) + e_max, float("-inf"))

        if is_no_split:
            tl.store(
                O
                + batch_idx * stride_o_b
                + offs_h[:, None] * stride_o_h
                + offs_dv[None, :],
                out_vals.to(O.dtype.element_ty),
                mask=mask_h[:, None],
            )
        else:
            split_idx = tl.load(Num_splits + batch_idx) + n_split_idx
            accum_row = (split_idx * head_num + m_block_idx * BLOCK_M).to(tl.int32)
            tl.store(
                O_accum
                + split_idx * stride_oaccum_split
                + offs_h[:, None] * stride_oaccum_h
                + offs_dv[None, :],
                out_vals,
                mask=mask_h[:, None],
            )
            tl.store(
                LSE_accum
                + split_idx * stride_lseaccum_split
                + offs_h * stride_lseaccum_h,
                lse_vals,
                mask=mask_h,
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
    out_accum_flat = out_accum.view(total_num_splits * h_q, dv)
    d_chunk = 64
    q_desc = TensorDescriptor(
        q_flat,
        shape=[b * s_q * h_q, d],
        strides=[d, 1],
        block_shape=[FLASH_MLA_BLOCK_M, dv],
    )
    q_tail_desc = TensorDescriptor(
        q_flat,
        shape=[b * s_q * h_q, d],
        strides=[d, 1],
        block_shape=[FLASH_MLA_BLOCK_M, d_chunk],
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
        kv_desc,
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
