import logging
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

try:
    import triton._C.libtriton as libtriton
    from triton._C.libtriton import ir
except Exception as e:
    print(f"[ERROR] Failed to import Triton: {e}")
    sys.exit(1)

print("Triton information")
print("===================")
print("triton version :", getattr(triton, "__version__", "unknown"))
print("triton path    :", triton.__file__)
print("libtriton path :", libtriton.__file__)

supported = hasattr(ir.builder, "make_nv_mma_shared_encoding_attr")
print("has make_nv_mma_shared_encoding_attr :", supported)

if not supported:
    raise RuntimeError("This Triton/libtriton does not support nv_mma_shared_layout.")

print("\nTLE-compatible Triton detected.")

from flaggems_vllm.runtime import device, error, torch_device_fn  # noqa: E402
from flaggems_vllm.utils import triton_lang_extension as ext  # noqa: E402
from flaggems_vllm.utils.device_info import get_device_capability  # noqa: E402
from flaggems_vllm.utils.triton_version_utils import has_triton_tle  # noqa: E402

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
FLASH_MLA_COMBINE_BLOCK_H = 8
FLASH_MLA_COMBINE_BLOCK_D = 256

_TENSOR_DESCRIPTOR_CLS = None
_CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE: int | None = None
_NUM_SMS_CACHE: dict[int, int] = {}
_FLASH_MLA_TLE_PLAN_CACHE: OrderedDict[
    "FlashMLATLEDecodePlanKey", "FlashMLATLEDecodePlan"
] = OrderedDict()


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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


def _contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _get_cuda_device_index(cuda_device: torch.device) -> int:
    dev_idx = cuda_device.index
    if dev_idx is None:
        dev_idx = torch.cuda.current_device()
    return int(dev_idx)


def _get_tensor_descriptor_cls():
    global _TENSOR_DESCRIPTOR_CLS
    if _TENSOR_DESCRIPTOR_CLS is None:
        from triton.tools.tensor_descriptor import TensorDescriptor

        _TENSOR_DESCRIPTOR_CLS = TensorDescriptor
    return _TENSOR_DESCRIPTOR_CLS


def _ensure_triton_descriptor_allocator(cuda_device: torch.device) -> None:
    global _CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE
    dev_idx = _get_cuda_device_index(cuda_device)
    if _CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE == dev_idx:
        return
    _set_triton_descriptor_allocator(cuda_device)
    _CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE = dev_idx


def _get_num_sms(cuda_device: torch.device) -> int:
    dev_idx = _get_cuda_device_index(cuda_device)
    cached = _NUM_SMS_CACHE.get(dev_idx)
    if cached is not None:
        return cached
    num_sms = torch.cuda.get_device_properties(cuda_device).multi_processor_count
    _NUM_SMS_CACHE[dev_idx] = int(num_sms)
    return int(num_sms)


def _get_plan_cache_size() -> int:
    try:
        return max(
            int(os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_TLE_PLAN_CACHE_SIZE", "32")),
            0,
        )
    except ValueError:
        logger.warning("Invalid FLAGGEMS_VLLM_FLASH_MLA_TLE_PLAN_CACHE_SIZE")
        return 32


def _reuse_tle_output() -> bool:
    return _env_flag("FLAGGEMS_VLLM_FLASH_MLA_TLE_REUSE_OUTPUT", False)


def _force_triton_flash_mla() -> bool:
    return _env_flag("FLAGGEMS_VLLM_FLASH_MLA_FORCE_TRITON", False)


def _same_cuda_device(lhs: torch.device, rhs: torch.device) -> bool:
    return lhs.type == rhs.type == "cuda" and _get_cuda_device_index(
        lhs
    ) == _get_cuda_device_index(rhs)


def _tensor_desc_light_key(t: torch.Tensor, block_shape: tuple[int, int]) -> tuple:
    if t.ndim != 2:
        raise ValueError("TensorDescriptor cache key expects a 2D tensor/view")
    return (
        int(t.data_ptr()),
        int(t.shape[0]),
        int(t.shape[1]),
        int(t.stride(0)),
        int(t.stride(1)),
        t.dtype,
        _get_cuda_device_index(t.device),
        tuple(block_shape),
    )


def _set_triton_descriptor_allocator(cuda_device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=cuda_device)

    triton.set_allocator(alloc_fn)


@triton.jit
def flash_mla_sched_meta_kernel_v3(
    B_seq_len,
    Sched_meta,
    Num_splits,
    CombineReqIds,
    NumCombineReqs,
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
    payload = tl.maximum(
        tl.cdiv(total_num_blocks, NUM_SM_PARTS) + FIXED_OVERHEAD_NUM_BLOCKS,
        FIXED_OVERHEAD_NUM_BLOCKS + 2,
    )

    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    combine_req_count = 0
    tl.store(Num_splits, 0)

    for part in tl.range(0, NUM_SM_PARTS, 1):
        begin_req_idx = now_req_idx
        begin_block_idx = now_block
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = now_block != 0
        remain_payload = payload

        while (now_req_idx < BATCH_SIZE) & (remain_payload > 0):
            cur_seq_len = tl.load(B_seq_len + now_req_idx)
            cur_num_blocks = tl.cdiv(tl.maximum(cur_seq_len, 1), BLOCK_SIZE_N)
            now_remain_blocks = cur_num_blocks - now_block
            if remain_payload + 1 >= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS:
                req_num_splits = now_n_split_idx + 1
                if req_num_splits != 1:
                    tl.store(CombineReqIds + combine_req_count, now_req_idx)
                    combine_req_count += 1
                cum_num_splits += req_num_splits
                tl.store(Num_splits + now_req_idx + 1, cum_num_splits)
                remain_payload -= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - FIXED_OVERHEAD_NUM_BLOCKS > 0:
                    split_blocks = remain_payload - FIXED_OVERHEAD_NUM_BLOCKS
                    # The WS TLE kernel cannot safely handle a one-block
                    # partial split in this schedule.
                    split_blocks = tl.where(
                        (split_blocks > 1) & (now_remain_blocks - split_blocks == 1),
                        split_blocks - 1,
                        split_blocks,
                    )
                    now_block += split_blocks
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
    stride_kv_token,
    stride_block_table_b,
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
        q_stage = batch_idx - begin_req_idx
        q_row = batch_idx * head_num + head_base
        q_slot = q_writer.acquire(q_stage)
        tle.gpu.copy(Q_desc, q_slot.sQ_l, [BLOCK_M, HEAD_DIM_V // 2], [q_row, 0])
        tle.gpu.copy(
            Q_desc, q_slot.sQ_r, [BLOCK_M, HEAD_DIM_V // 2], [q_row, HEAD_DIM_V // 2]
        )
        if HAVE_TAIL:
            tle.gpu.copy(
                Q_tail_desc, q_slot.sQ_tail, [BLOCK_M, D_CHUNK], [q_row, HEAD_DIM_V]
            )
        q_writer.commit(q_stage)

        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        block_table_base = batch_idx * stride_block_table_b
        n_blocks = end_block_idx - start_block_idx
        n_full_pairs = n_blocks // 2
        has_tail = (n_blocks - n_full_pairs * 2) > 0
        n_pair_slots = n_full_pairs + has_tail.to(tl.int32)

        for pair in tl.range(0, n_full_pairs):
            pipe_idx = pipe_base + pair
            k1_pipe_idx = k1_pipe_base + pair
            block0 = start_block_idx + pair * 2
            block1 = block0 + 1
            page0 = tle.load(Block_table + block_table_base + block0)
            page1 = tle.load(Block_table + block_table_base + block1)
            kv_row0 = page0 * PAGE_SIZE
            kv_row1 = page1 * PAGE_SIZE

            k0_l_slot = k0_l_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc, k0_l_slot.sK, [BLOCK_N, HEAD_DIM_V // 2], [kv_row0, 0]
            )
            k0_l_writer.commit(pipe_idx)

            k0_r_slot = k0_r_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc,
                k0_r_slot.sK,
                [BLOCK_N, HEAD_DIM_V // 2],
                [kv_row0, HEAD_DIM_V // 2],
            )
            if HAVE_TAIL:
                tle.gpu.copy(
                    Kv_tail_desc,
                    k0_r_slot.sK_tail,
                    [BLOCK_N, D_CHUNK],
                    [kv_row0, HEAD_DIM_V],
                )
            k0_r_writer.commit(pipe_idx)

            k1_l_slot = k1_l_writer.acquire(k1_pipe_idx)
            tle.gpu.copy(
                Kv_desc, k1_l_slot.sK, [BLOCK_N, HEAD_DIM_V // 2], [kv_row1, 0]
            )
            k1_l_writer.commit(k1_pipe_idx)

            k1_r_slot = k1_r_writer.acquire(k1_pipe_idx)
            tle.gpu.copy(
                Kv_desc,
                k1_r_slot.sK,
                [BLOCK_N, HEAD_DIM_V // 2],
                [kv_row1, HEAD_DIM_V // 2],
            )
            if HAVE_TAIL:
                tle.gpu.copy(
                    Kv_tail_desc,
                    k1_r_slot.sK_tail,
                    [BLOCK_N, D_CHUNK],
                    [kv_row1, HEAD_DIM_V],
                )
            k1_r_writer.commit(k1_pipe_idx)

        if has_tail:
            pipe_idx = pipe_base + n_full_pairs
            block0 = start_block_idx + n_full_pairs * 2
            page0 = tle.load(Block_table + block_table_base + block0)
            kv_row0 = page0 * PAGE_SIZE

            k0_l_slot = k0_l_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc, k0_l_slot.sK, [BLOCK_N, HEAD_DIM_V // 2], [kv_row0, 0]
            )
            k0_l_writer.commit(pipe_idx)

            k0_r_slot = k0_r_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc,
                k0_r_slot.sK,
                [BLOCK_N, HEAD_DIM_V // 2],
                [kv_row0, HEAD_DIM_V // 2],
            )
            if HAVE_TAIL:
                tle.gpu.copy(
                    Kv_tail_desc,
                    k0_r_slot.sK_tail,
                    [BLOCK_N, D_CHUNK],
                    [kv_row0, HEAD_DIM_V],
                )
            k0_r_writer.commit(pipe_idx)

        pipe_base += n_pair_slots
        k1_pipe_base += n_full_pairs


@triton.jit
def _flash_mla_ws_consumer0(
    k0_l_reader,
    k0_r_qk_reader,
    k1_l_remote_reader,
    sM_wg0_writer,
    sM_wg1_reader,
    sP0_writer,
    sP1_reader,
    sL0_writer,
    sL1_reader,
    q_reader,
    sO_stage,
    Output_desc,
    OAccum_desc,
    O,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    mask_h = head_offsets < head_num

    offs_n = tl.arange(0, BLOCK_N)
    kv_rows = tl.broadcast_to(offs_n[:, None], (BLOCK_N, HALF_DIM_V))
    kv_cols = tl.broadcast_to(tl.arange(0, HALF_DIM_V)[None, :], (BLOCK_N, HALF_DIM_V))
    if HAVE_TAIL:
        kv_rows_tail = tl.broadcast_to(offs_n[:, None], (BLOCK_N, D_CHUNK))
        kv_cols_tail = tl.broadcast_to(
            tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK)
        )

    pipe_base = 0
    k1_pipe_base = 0
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

        q_row = (batch_idx * head_num + m_block_idx * BLOCK_M).to(tl.int32)
        q_nope = Q_desc.load([q_row, 0])
        q_pe = Q_tail_desc.load([q_row, HEAD_DIM_V])

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HALF_DIM_V], dtype=tl.float32)

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
            acc = acc * re_scale[:, None]

            p_save = p.to(k0_l.dtype)
            k0_r_qk_reader.release(pipe_idx)

            v_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p_save, v_l, acc, out_dtype=tl.float32)

            sP_slot = sP0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_save)
            sP0_writer.commit(pipe_idx)

            k0_l_reader.release(pipe_idx)

            peer_p_wait = sP1_reader.wait(k1_pipe_idx)
            peer_p = tl.load(tle.gpu.local_ptr(peer_p_wait.slot.sP))
            k1_l_wait = k1_l_remote_reader.wait(k1_pipe_idx)
            k1_l = tl.load(tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(peer_p, k1_l, acc, out_dtype=tl.float32)
            sP1_reader.release(k1_pipe_idx)
            k1_l_remote_reader.release(k1_pipe_idx)
            e_max = merged_max

        if has_tail:
            pipe_idx = pipe_base + n_full_pairs
            block0 = start_block_idx + n_full_pairs * 2
            k0_l_wait = k0_l_reader.wait(pipe_idx)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            k0_l_slot = k0_l_wait.slot
            k0_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            q_l = tl.load(tle.gpu.local_ptr(q_slot.sQ_l))
            qk = tl.dot(q_l, tl.trans(k0_l), qk, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pipe_idx)
            k0_r_slot = k0_r_wait.slot
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols)))
            q_r = tl.load(tle.gpu.local_ptr(q_slot.sQ_r))
            qk = tl.dot(q_r, tl.trans(k0_r), qk, out_dtype=tl.float32)
            if HAVE_TAIL:
                k_tail = tl.load(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail, (kv_rows_tail, kv_cols_tail))
                )
                q_tail = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                qk = tl.dot(q_tail, tl.trans(k_tail), qk, out_dtype=tl.float32)

            valid_n = block0 * BLOCK_N + offs_n < seq_len
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))

            new_max = tl.maximum(tl.max(qk, axis=1), e_max)
            sM_slot = sM_wg0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sM_slot.sM), new_max)
            sM_wg0_writer.commit(pipe_idx)

            re_scale = tl.exp(e_max - new_max)
            p = tl.exp(qk - new_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            acc = acc * re_scale[:, None]

            p_save = p.to(k0_l.dtype)
            k0_r_qk_reader.release(pipe_idx)

            v_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p_save, v_l, acc, out_dtype=tl.float32)

            sP_slot = sP0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_save)
            sP0_writer.commit(pipe_idx)

            e_max = new_max
            k0_l_reader.release(pipe_idx)

        l_stage0 = q_stage * 2
        l_stage1 = l_stage0 + 1

        sL_slot = sL0_writer.acquire(l_stage0)
        tl.store(tle.gpu.local_ptr(sL_slot.sL), e_sum)
        sL0_writer.commit(l_stage0)

        peer_l_wait = sL1_reader.wait(l_stage1)
        total_sum = e_sum + tl.load(tle.gpu.local_ptr(peer_l_wait.slot.sL))
        sL1_reader.release(l_stage1)

        valid = total_sum > 0.0
        safe_total_sum = tl.where(valid, total_sum, 1.0)
        inv_total_sum = tl.fdiv(1.0, safe_total_sum)

        output_row = batch_idx * head_num + head_base
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
                + head_offsets * stride_lseaccum_h,
                lse_vals,
                mask=mask_h,
            )
        q_reader.release(q_stage)
        pipe_base += n_pair_slots
        k1_pipe_base += n_full_pairs


@triton.jit
def flash_mla_splitkv_ws_tle_kernel(
    Q_desc,
    Q_tail_desc,
    Output_desc,
    OAccum_desc,
    Kv_desc,
    Kv_tail_desc,
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
    stride_kv_token,
    stride_block_table_b,
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
    sK0_l = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK0_r = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK1_l = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK1_r = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
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
        sK=sK0_l,
    )
    if HAVE_TAIL:
        k0_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k0_r",
            readers=("qk", "remote"),
            sK=sK0_r,
            sK_tail=sK0_tail,
        )
    else:
        k0_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k0_r",
            readers=("qk", "remote"),
            sK=sK0_r,
        )
    k1_l_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="flash_mla_ws_k1_l",
        readers=("qk", "remote"),
        sK=sK1_l,
    )
    if HAVE_TAIL:
        k1_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k1_r",
            sK=sK1_r,
            sK_tail=sK1_tail,
        )
    else:
        k1_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k1_r",
            sK=sK1_r,
        )
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
                    sM_wg0_pipe.writer(),
                    sM_wg1_pipe.reader(),
                    sP0_pipe.writer(),
                    sP1_pipe.reader(),
                    sL0_pipe.writer(),
                    sL1_pipe.reader(),
                    q_pipe.reader("wg0"),
                    sK0_l,
                    Output_desc,
                    OAccum_desc,
                    O,
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
                    sM_wg1_pipe.writer(),
                    sM_wg0_pipe.reader(),
                    sP1_pipe.writer(),
                    sP0_pipe.reader(),
                    sL1_pipe.writer(),
                    sL0_pipe.reader(),
                    q_pipe.reader("wg1"),
                    sK1_r,
                    Output_desc,
                    OAccum_desc,
                    O,
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
                    q_pipe.writer(),
                    k0_l_pipe.writer(),
                    k0_r_pipe.writer(),
                    k1_l_pipe.writer(),
                    k1_r_pipe.writer(),
                    Q_desc,
                    Q_tail_desc,
                    Kv,
                    Kv_desc,
                    Kv_tail_desc,
                    Block_table,
                    B_seq_len,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    head_base,
                    head_num,
                    stride_kv_token,
                    stride_block_table_b,
                    BLOCK_M,
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
def flash_mla_combine_kernel_compact(
    O_accum,
    LSE_accum,
    Num_splits,
    CombineReqIds,
    NumCombineReqs,
    O,
    head_num,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    stride_o_b,
    stride_o_h,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
):
    task_idx = tl.program_id(0)
    h_block_idx = tl.program_id(1)
    d_block_idx = tl.program_id(2)

    num_tasks = tl.load(NumCombineReqs)
    if task_idx < num_tasks:
        batch_idx = tl.load(CombineReqIds + task_idx)

        offs_h = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_h = offs_h < head_num
        mask_d = offs_d < HEAD_DIM_V

        start_split = tl.load(Num_splits + batch_idx)
        end_split = tl.load(Num_splits + batch_idx + 1)
        my_num_splits = end_split - start_split

        if my_num_splits > 1:
            max_lse = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
            for s in tl.range(0, my_num_splits):
                lse_s = tl.load(
                    LSE_accum
                    + (start_split + s) * stride_lseaccum_split
                    + offs_h * stride_lseaccum_h,
                    mask=mask_h,
                    other=float("-inf"),
                )
                max_lse = tl.maximum(max_lse, lse_s)

            acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
            sum_w = tl.zeros([BLOCK_H], dtype=tl.float32)
            valid_row = max_lse != float("-inf")
            for s in tl.range(0, my_num_splits):
                lse_s = tl.load(
                    LSE_accum
                    + (start_split + s) * stride_lseaccum_split
                    + offs_h * stride_lseaccum_h,
                    mask=mask_h,
                    other=float("-inf"),
                )
                w = tl.where(valid_row, tl.exp(lse_s - max_lse), 0.0)
                sum_w += w

                o_s = tl.load(
                    O_accum
                    + (start_split + s) * stride_oaccum_split
                    + offs_h[:, None] * stride_oaccum_h
                    + offs_d[None, :],
                    mask=mask_h[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += w[:, None] * o_s

            inv_sum = tl.where(sum_w > 0.0, tl.fdiv(1.0, sum_w), 0.0)
            acc = acc * inv_sum[:, None]

            tl.store(
                O
                + batch_idx * stride_o_b
                + offs_h[:, None] * stride_o_h
                + offs_d[None, :],
                acc.to(O.dtype.element_ty),
                mask=mask_h[:, None] & mask_d[None, :],
            )


class FlashMLATLEDecodePlan:
    def __init__(
        self,
        *,
        b: int,
        s_q: int,
        h_q: int,
        h_kv: int,
        d: int,
        dv: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
        reuse_output: bool = False,
    ) -> None:
        self.b = b
        self.s_q = s_q
        self.h_q = h_q
        self.h_kv = h_kv
        self.d = d
        self.dv = dv
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.causal = causal
        self.reuse_output = reuse_output
        self._cache_run_desc = _env_flag(
            "FLAGGEMS_VLLM_FLASH_MLA_TLE_CACHE_RUN_DESCRIPTORS", False
        )
        self._cache_q_desc = _env_flag(
            "FLAGGEMS_VLLM_FLASH_MLA_TLE_CACHE_Q_DESC", False
        )

        self.d_chunk = 64
        self.sm_scale = 1 / math.sqrt(d)
        self.num_m_blocks = triton.cdiv(s_q * h_q // h_kv, FLASH_MLA_BLOCK_M)
        self.num_sms = _get_num_sms(device)
        self.num_sm_parts = max(self.num_sms // h_kv // self.num_m_blocks, 1)
        self.total_num_splits = b + self.num_sm_parts
        self.max_combine_reqs = min(b, self.num_sm_parts)
        self.block_b = triton.next_power_of_2(b)

        self.sched_meta = torch.empty(
            (self.num_sm_parts, FLASH_MLA_META_FIELDS),
            dtype=torch.int32,
            device=device,
        )
        self.num_splits = torch.empty((b + 1,), dtype=torch.int32, device=device)
        self.combine_req_ids = torch.empty(
            (self.max_combine_reqs,), dtype=torch.int32, device=device
        )
        self.num_combine_reqs = torch.empty((1,), dtype=torch.int32, device=device)
        self.out_accum = torch.empty(
            (self.total_num_splits, h_q, dv), dtype=dtype, device=device
        )
        self.lse_accum = torch.empty(
            (self.total_num_splits, h_q), dtype=torch.float32, device=device
        )
        self.out = (
            torch.empty((b * s_q, h_q, dv), dtype=dtype, device=device)
            if reuse_output
            else None
        )

        self.metadata_valid = False
        self._last_cache_seqlens_ref: torch.Tensor | None = None
        self._last_launch_refs = ()
        self._out_accum_flat = None
        self._oaccum_desc = None
        self._oaccum_desc_refs = None
        self._kv_desc_key = None
        self._kv_desc = None
        self._kv_tail_desc = None
        self._kv_desc_refs = None
        self._out_desc_key = None
        self._output_desc = None
        self._out_desc_refs = None
        self._q_desc_key = None
        self._q_desc = None
        self._q_tail_desc = None
        self._q_desc_refs = None
        self._build_oaccum_desc()

    def invalidate_metadata(self) -> None:
        self.metadata_valid = False
        self._last_cache_seqlens_ref = None

    def _build_oaccum_desc(self) -> None:
        TensorDescriptor = _get_tensor_descriptor_cls()
        _ensure_triton_descriptor_allocator(self.device)

        self._out_accum_flat = self.out_accum.view(
            self.total_num_splits * self.h_q,
            self.dv,
        )
        self._oaccum_desc = TensorDescriptor(
            self._out_accum_flat,
            shape=[self.total_num_splits * self.h_q, self.dv],
            strides=[self.dv, 1],
            block_shape=[FLASH_MLA_BLOCK_M, self.dv // 2],
        )
        self._oaccum_desc_refs = (
            self._out_accum_flat,
            self._oaccum_desc,
        )

    def _get_oaccum_desc(self):
        if self._oaccum_desc is None:
            self._build_oaccum_desc()
        return self._oaccum_desc

    def clear_descriptor_cache(self) -> None:
        self._kv_desc_key = None
        self._kv_desc = None
        self._kv_tail_desc = None
        self._kv_desc_refs = None
        self._out_desc_key = None
        self._output_desc = None
        self._out_desc_refs = None
        self._q_desc_key = None
        self._q_desc = None
        self._q_tail_desc = None
        self._q_desc_refs = None

    def _get_kv_descs(self, TensorDescriptor, kv_flat, can_cache: bool):
        kv_block = (FLASH_MLA_BLOCK_N, self.dv // 2)
        kv_tail_block = (FLASH_MLA_BLOCK_N, self.d_chunk)

        if not self._cache_run_desc or not can_cache:
            kv_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_block),
            )
            kv_tail_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_tail_block),
            )
            return kv_desc, kv_tail_desc

        key = (
            _tensor_desc_light_key(kv_flat, kv_block),
            _tensor_desc_light_key(kv_flat, kv_tail_block),
        )
        if key != self._kv_desc_key:
            self._kv_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_block),
            )
            self._kv_tail_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_tail_block),
            )
            self._kv_desc_key = key
            self._kv_desc_refs = (
                kv_flat,
                self._kv_desc,
                self._kv_tail_desc,
            )
        return self._kv_desc, self._kv_tail_desc

    def _get_output_desc(self, TensorDescriptor, out_flat, can_cache: bool):
        block = (FLASH_MLA_BLOCK_M, self.dv // 2)

        if not self._cache_run_desc or not can_cache:
            return TensorDescriptor(
                out_flat,
                shape=[self.b * self.s_q * self.h_q, self.dv],
                strides=[self.dv, 1],
                block_shape=list(block),
            )

        key = _tensor_desc_light_key(out_flat, block)
        if key != self._out_desc_key:
            self._output_desc = TensorDescriptor(
                out_flat,
                shape=[self.b * self.s_q * self.h_q, self.dv],
                strides=[self.dv, 1],
                block_shape=list(block),
            )
            self._out_desc_key = key
            self._out_desc_refs = (
                out_flat,
                self._output_desc,
            )
        return self._output_desc

    def _get_q_descs(self, TensorDescriptor, q_flat, can_cache: bool):
        q_block = (FLASH_MLA_BLOCK_M, self.dv // 2)
        q_tail_block = (FLASH_MLA_BLOCK_M, self.d_chunk)

        if not (self._cache_run_desc and self._cache_q_desc and can_cache):
            q_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_block),
            )
            q_tail_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_tail_block),
            )
            return q_desc, q_tail_desc

        key = (
            _tensor_desc_light_key(q_flat, q_block),
            _tensor_desc_light_key(q_flat, q_tail_block),
        )
        if key != self._q_desc_key:
            self._q_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_block),
            )
            self._q_tail_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_tail_block),
            )
            self._q_desc_key = key
            self._q_desc_refs = (
                q_flat,
                self._q_desc,
                self._q_tail_desc,
            )
        return self._q_desc, self._q_tail_desc

    def plan(self, cache_seqlens: torch.Tensor) -> None:
        if cache_seqlens.dtype != torch.int32:
            raise TypeError("cache_seqlens must be int32")
        if not _same_cuda_device(cache_seqlens.device, self.device):
            raise ValueError("cache_seqlens device mismatch")
        if cache_seqlens.ndim != 1 or cache_seqlens.shape[0] != self.b:
            raise ValueError("cache_seqlens shape mismatch")

        cache_seqlens_tle = _contiguous_if_needed(cache_seqlens)
        flash_mla_sched_meta_kernel_v3[(1,)](
            cache_seqlens_tle,
            self.sched_meta,
            self.num_splits,
            self.combine_req_ids,
            self.num_combine_reqs,
            BLOCK_B=self.block_b,
            BATCH_SIZE=self.b,
            BLOCK_SIZE_N=FLASH_MLA_BLOCK_N,
            FIXED_OVERHEAD_NUM_BLOCKS=FLASH_MLA_FIXED_OVERHEAD_BLOCKS,
            NUM_SM_PARTS=self.num_sm_parts,
            META_FIELDS=FLASH_MLA_META_FIELDS,
            num_warps=1,
            num_stages=1,
        )

        self.metadata_valid = True
        self._last_cache_seqlens_ref = cache_seqlens_tle

    def _check_run_inputs(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        if not (
            _same_cuda_device(q.device, self.device)
            and _same_cuda_device(blocked_k.device, self.device)
            and _same_cuda_device(block_table.device, self.device)
        ):
            raise ValueError("device mismatch")
        if q.dtype != self.dtype or blocked_k.dtype != self.dtype:
            raise TypeError("dtype mismatch")
        if block_table.dtype != torch.int32:
            raise TypeError("block_table must be int32")
        if q.ndim != 4 or tuple(q.shape) != (
            self.b,
            self.s_q,
            self.h_q,
            self.d,
        ):
            raise ValueError("q shape mismatch")
        if (
            blocked_k.ndim != 4
            or blocked_k.shape[1] != self.block_size
            or blocked_k.shape[2] != self.h_kv
            or blocked_k.shape[3] != self.d
        ):
            raise ValueError("blocked_k shape mismatch")
        if block_table.ndim != 2 or block_table.shape[0] != self.b:
            raise ValueError("block_table shape mismatch")

    def _get_out_tensor(self, out: torch.Tensor | None) -> torch.Tensor:
        if out is not None:
            if not _same_cuda_device(out.device, self.device):
                raise ValueError("out device mismatch")
            if out.dtype != self.dtype:
                raise TypeError("out dtype mismatch")
            if tuple(out.shape) != (self.b * self.s_q, self.h_q, self.dv):
                raise ValueError("out shape must be (b * s_q, h_q, dv)")
            return out
        if self.reuse_output:
            if self.out is None:
                self.out = torch.empty(
                    (self.b * self.s_q, self.h_q, self.dv),
                    dtype=self.dtype,
                    device=self.device,
                )
            return self.out
        return torch.empty(
            (self.b * self.s_q, self.h_q, self.dv),
            dtype=self.dtype,
            device=self.device,
        )

    def run(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor | None = None,
        *,
        update_metadata: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_run_inputs(q, blocked_k, block_table)
        if update_metadata:
            if cache_seqlens is None:
                raise ValueError("cache_seqlens is required when update_metadata=True")
            self.plan(cache_seqlens)
        elif not self.metadata_valid or self._last_cache_seqlens_ref is None:
            raise RuntimeError("metadata is not valid; call plan(cache_seqlens) first")

        TensorDescriptor = _get_tensor_descriptor_cls()
        _ensure_triton_descriptor_allocator(self.device)

        q_tle = _contiguous_if_needed(q)
        blocked_k_tle = _contiguous_if_needed(blocked_k)
        kv_flat = blocked_k_tle.view(-1, self.d)
        block_table_tle = _contiguous_if_needed(block_table)
        q_flat = q_tle.view(self.b * self.s_q * self.h_q, self.d)
        out_tle = self._get_out_tensor(out)
        out_flat = out_tle.view(self.b * self.s_q * self.h_q, self.dv)
        oaccum_desc = self._get_oaccum_desc()

        q_desc, q_tail_desc = self._get_q_descs(
            TensorDescriptor,
            q_flat,
            can_cache=(q_tle is q),
        )
        output_desc = self._get_output_desc(
            TensorDescriptor,
            out_flat,
            can_cache=(self.reuse_output and out is None),
        )
        kv_desc, kv_tail_desc = self._get_kv_descs(
            TensorDescriptor,
            kv_flat,
            can_cache=(blocked_k_tle is blocked_k),
        )

        flash_mla_splitkv_ws_tle_kernel[(self.num_m_blocks, self.num_sm_parts)](
            q_desc,
            q_tail_desc,
            output_desc,
            oaccum_desc,
            kv_desc,
            kv_tail_desc,
            kv_flat,
            block_table_tle,
            self._last_cache_seqlens_ref,
            self.sched_meta,
            self.num_splits,
            out_tle,
            self.out_accum,
            self.lse_accum,
            self.sm_scale,
            self.h_q,
            kv_flat.stride(0),
            block_table_tle.stride(0),
            self.lse_accum.stride(0),
            self.lse_accum.stride(1),
            BLOCK_M=FLASH_MLA_BLOCK_M,
            BLOCK_N=FLASH_MLA_BLOCK_N,
            PAGE_SIZE=self.block_size,
            HEAD_DIM_V=self.dv,
            HEAD_DIM=self.d,
            D_CHUNK=self.d_chunk,
            META_FIELDS=FLASH_MLA_META_FIELDS,
            num_warps=4,
            num_stages=1,
        )

        flash_mla_combine_kernel_compact[
            (
                self.max_combine_reqs,
                triton.cdiv(self.h_q, FLASH_MLA_COMBINE_BLOCK_H),
                triton.cdiv(self.dv, FLASH_MLA_COMBINE_BLOCK_D),
            )
        ](
            self.out_accum,
            self.lse_accum,
            self.num_splits,
            self.combine_req_ids,
            self.num_combine_reqs,
            out_tle,
            self.h_q,
            self.out_accum.stride(0),
            self.out_accum.stride(1),
            self.lse_accum.stride(0),
            self.lse_accum.stride(1),
            out_tle.stride(0),
            out_tle.stride(1),
            BLOCK_H=FLASH_MLA_COMBINE_BLOCK_H,
            BLOCK_D=FLASH_MLA_COMBINE_BLOCK_D,
            HEAD_DIM_V=self.dv,
            num_warps=4,
            num_stages=1,
        )

        self._last_launch_refs = (
            q_tle,
            blocked_k_tle,
            kv_flat,
            block_table_tle,
            out_tle,
            out_flat,
            self._out_accum_flat,
            q_desc,
            q_tail_desc,
            output_desc,
            self._oaccum_desc,
            kv_desc,
            kv_tail_desc,
        )
        return out_tle.view(self.b, self.s_q, self.h_q, self.dv)


def _get_flash_mla_tle_decode_plan(
    *,
    b: int,
    s_q: int,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool = True,
    reuse_output: bool = False,
) -> FlashMLATLEDecodePlan:
    key = FlashMLATLEDecodePlanKey(
        device=_get_cuda_device_index(device),
        dtype=dtype,
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        causal=causal,
        reuse_output=reuse_output,
    )
    plan = _FLASH_MLA_TLE_PLAN_CACHE.get(key)
    if plan is not None:
        _FLASH_MLA_TLE_PLAN_CACHE.move_to_end(key)
        return plan

    plan = FlashMLATLEDecodePlan(
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        dtype=dtype,
        device=device,
        causal=causal,
        reuse_output=reuse_output,
    )
    max_size = _get_plan_cache_size()
    if max_size == 0:
        return plan
    _FLASH_MLA_TLE_PLAN_CACHE[key] = plan
    while len(_FLASH_MLA_TLE_PLAN_CACHE) > max_size:
        _FLASH_MLA_TLE_PLAN_CACHE.popitem(last=False)
    return plan


def get_flash_mla_tle_decode_plan(
    *,
    b: int,
    s_q: int,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool = True,
    reuse_output: bool = False,
) -> FlashMLATLEDecodePlan:
    return _get_flash_mla_tle_decode_plan(
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        dtype=dtype,
        device=device,
        causal=causal,
        reuse_output=reuse_output,
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
    if _force_triton_flash_mla() or not HAS_TLE_FLASH_MLA:
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
        causal=causal,
        reuse_output=_reuse_tle_output(),
    )
    return plan.run(
        q=q,
        blocked_k=blocked_k,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        update_metadata=True,
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
        kv_loc = kv_page_number.to(tl.int64) * PAGE_SIZE + (offs_n % PAGE_SIZE).to(
            tl.int64
        )
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
        kv_loc = kv_page_number.to(tl.int64) * PAGE_SIZE + (offs_n % PAGE_SIZE).to(
            tl.int64
        )
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
