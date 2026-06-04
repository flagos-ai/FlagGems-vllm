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


TLE_FLASH_MLA_BK = 64
TLE_FLASH_MLA_BH = 64
TLE_FLASH_MLA_SPLITKV_BH = 16  # Smaller BH for splitkv to avoid register spilling
TLE_FLASH_MLA_2WG_BH = 32
TLE_FLASH_MLA_PAIR_BLOCKS = 2
TLE_FLASH_MLA_WORKER_NUM_WARPS = 4


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
def flash_mla_splitkv_kernel(
    Q_ptr,
    Kv_cache,
    Req_to_tokens,
    B_seq_len,
    O_partial,
    LSE_partial,
    sm_scale,
    head_num,
    stride_q_bs,
    stride_q_h,
    stride_kv_bs,
    stride_req_to_tokens_bs,
    stride_o_split,
    stride_o_b,
    stride_o_h,
    stride_lse_split,
    stride_lse_b,
    stride_lse_h,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    cur_head_id = ext.program_id(0)
    cur_batch_id = ext.program_id(1)
    split_id = ext.program_id(2)
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

    # Split KV: each split processes SPLIT_SIZE blocks
    start_block = split_id * SPLIT_SIZE
    end_block = tl.minimum(start_block + SPLIT_SIZE, cur_batch_seq_len // BLOCK_N)

    offs_n = start_block * BLOCK_N + tl.arange(0, BLOCK_N)
    for i in range(start_block, end_block):
        kv_page_number = tl.load(Req_to_tokens + offs_n // PAGE_SIZE)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
        v_c = tl.load(Kv_cache + offs_v_c)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
        k_pe = tl.load(Kv_cache + offs_k_pe)

        qk = tl.dot(q_pe, k_pe, acc=qk)
        qk *= sm_scale

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N

    # Handle remainder
    remainder_start = end_block * BLOCK_N
    if (
        remainder_start < cur_batch_seq_len
        and split_id == (cur_batch_seq_len // BLOCK_N) // SPLIT_SIZE
    ):
        offs_n = remainder_start + tl.arange(0, BLOCK_N)
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

        qk = tl.dot(q_nope, k_c)

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
        k_pe = tl.load(Kv_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)

        qk = tl.dot(q_pe, k_pe, acc=qk)
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Normalize and store partial output
    valid = e_sum > 0
    inv_sum = tl.fdiv(1.0, e_sum)
    acc = tl.where(valid[:, None], acc * inv_sum[:, None], 0.0)

    # Store partial output
    offs_o = (
        split_id * stride_o_split
        + cur_batch_id * stride_o_b
        + cur_head[:, None] * stride_o_h
        + offs_d_ckv[None, :]
    )
    if EVEN_H:
        tl.store(O_partial + offs_o, acc.to(O_partial.dtype.element_ty))
    else:
        tl.store(
            O_partial + offs_o,
            acc.to(O_partial.dtype.element_ty),
            mask=mask_head[:, None],
        )

    # Store LSE (log-sum-exp)
    lse = tl.where(valid, tl.log(e_sum) + e_max, float("-inf"))
    offs_lse = (
        split_id * stride_lse_split
        + cur_batch_id * stride_lse_b
        + cur_head * stride_lse_h
    )
    if EVEN_H:
        tl.store(LSE_partial + offs_lse, lse)
    else:
        tl.store(LSE_partial + offs_lse, lse, mask=mask_head)


@triton.heuristics(
    values={
        "EVEN_H": lambda META: META["head_num"] % META["BLOCK_H"] == 0,
    }
)
@triton.jit
def flash_mla_splitkv_persistent_tle_kernel(
    Q_ptr,
    Kv_cache,
    Req_to_tokens,
    B_seq_len,
    O_partial,
    LSE_partial,
    sm_scale,
    total_tiles,
    head_num,
    batch_size: tl.constexpr,
    stride_q_bs,
    stride_q_h,
    stride_kv_bs,
    stride_req_to_tokens_bs,
    stride_o_split,
    stride_o_b,
    stride_o_h,
    stride_lse_split,
    stride_lse_b,
    stride_lse_h,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    # Experimental wave=1 path: persistent tile scheduling with TLE-Lite page-table loads.
    pid = ext.program_id(0)
    offs_d_ckv = tl.arange(0, HEAD_DIM_V)
    offs_d_kpe = tl.arange(HEAD_DIM_V, HEAD_DIM)

    for tile_id in tl.range(pid, total_tiles, NUM_SMS):
        split_id = tile_id % NUM_SPLITS
        tile_group = tile_id // NUM_SPLITS
        cur_batch_id = tile_group % batch_size
        cur_head_id = tile_group // batch_size
        req_base = Req_to_tokens + stride_req_to_tokens_bs * cur_batch_id
        cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_head = cur_head < head_num

        offs_q_nope = (
            cur_batch_id * stride_q_bs
            + cur_head[:, None] * stride_q_h
            + offs_d_ckv[None, :]
        )
        offs_q_pe = (
            cur_batch_id * stride_q_bs
            + cur_head[:, None] * stride_q_h
            + offs_d_kpe[None, :]
        )
        if EVEN_H:
            q_nope = tl.load(Q_ptr + offs_q_nope)
            q_pe = tl.load(Q_ptr + offs_q_pe)
        else:
            q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None])
            q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None])

        e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)
        cur_batch_seq_len = tl.load(B_seq_len + cur_batch_id)
        start_block = split_id * SPLIT_SIZE
        end_block = tl.minimum(start_block + SPLIT_SIZE, cur_batch_seq_len // BLOCK_N)
        offs_n = start_block * BLOCK_N + tl.arange(0, BLOCK_N)

        for _ in range(start_block, end_block):
            kv_page_number = tle.load(req_base + offs_n // PAGE_SIZE)
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
            v_c = tl.load(Kv_cache + offs_v_c)
            qk = tl.dot(q_nope, tl.trans(v_c))
            offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
            k_pe = tl.load(Kv_cache + offs_k_pe)
            qk = tl.dot(q_pe, k_pe, acc=qk)
            qk *= sm_scale

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
            offs_n += BLOCK_N

        remainder_start = end_block * BLOCK_N
        if (
            remainder_start < cur_batch_seq_len
            and split_id == (cur_batch_seq_len // BLOCK_N) // SPLIT_SIZE
        ):
            offs_n = remainder_start + tl.arange(0, BLOCK_N)
            mask_kvsplit = offs_n < cur_batch_seq_len
            kv_page_number = tle.load(
                req_base + offs_n // PAGE_SIZE, mask=mask_kvsplit, other=0
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
            v_c = tl.load(Kv_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
            qk = tl.dot(q_nope, tl.trans(v_c))
            offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
            k_pe = tl.load(
                Kv_cache + offs_k_pe,
                mask=mask_kvsplit[None, :],
                other=0.0,
            )
            qk = tl.dot(q_pe, k_pe, acc=qk)
            qk *= sm_scale
            qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        active = start_block * BLOCK_N < cur_batch_seq_len
        acc = acc * tl.fdiv(1.0, e_sum)[:, None]
        offs_o = (
            split_id * stride_o_split
            + cur_batch_id * stride_o_b
            + cur_head[:, None] * stride_o_h
            + offs_d_ckv[None, :]
        )
        tl.store(
            O_partial + offs_o,
            acc.to(O_partial.dtype.element_ty),
            mask=active & mask_head[:, None],
        )
        lse = tl.log(e_sum) + e_max
        offs_lse = (
            split_id * stride_lse_split
            + cur_batch_id * stride_lse_b
            + cur_head * stride_lse_h
        )
        tl.store(LSE_partial + offs_lse, lse, mask=active & mask_head)


@triton.jit
def flash_mla_splitkv_combine_kernel(
    O_partial,
    LSE_partial,
    O,
    num_splits,
    head_num,
    stride_op_split,
    stride_op_b,
    stride_op_h,
    stride_lse_split,
    stride_lse_b,
    stride_lse_h,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    cur_head_id = ext.program_id(0)
    cur_batch_id = ext.program_id(1)

    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_head = cur_head < head_num
    offs_d = tl.arange(0, HEAD_DIM_V)

    # Find max LSE across splits
    max_lse = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        offs_lse = (
            s * stride_lse_split + cur_batch_id * stride_lse_b + cur_head * stride_lse_h
        )
        lse_s = tl.load(LSE_partial + offs_lse, mask=mask_head, other=float("-inf"))
        max_lse = tl.maximum(max_lse, lse_s)

    # Weighted sum
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)
    sum_w = tl.zeros([BLOCK_H], dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        offs_lse = (
            s * stride_lse_split + cur_batch_id * stride_lse_b + cur_head * stride_lse_h
        )
        lse_s = tl.load(LSE_partial + offs_lse, mask=mask_head, other=float("-inf"))
        w = tl.exp(lse_s - max_lse)
        sum_w += w
        offs_o = (
            s * stride_op_split
            + cur_batch_id * stride_op_b
            + cur_head[:, None] * stride_op_h
            + offs_d[None, :]
        )
        o_s = tl.load(O_partial + offs_o, mask=mask_head[:, None], other=0.0)
        acc += w[:, None] * o_s.to(tl.float32)

    inv_sum = tl.fdiv(1.0, sum_w)
    acc = acc * inv_sum[:, None]

    # Store final output
    offs_out = (
        cur_batch_id * stride_o_b + cur_head[:, None] * stride_o_h + offs_d[None, :]
    )
    tl.store(O + offs_out, acc.to(O.dtype.element_ty), mask=mask_head[:, None])


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


if HAS_TLE_FLASH_MLA:

    @triton.jit
    def _tle_flash_mla_dense_producer(
        k0_l_writer,
        k0_r_writer,
        k1_l_writer,
        k1_r_writer,
        valid_writer,
        kv_base,
        req_to_tokens,
        seq_len_ptr,
        stride_kv: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        BK: tl.constexpr,
    ):
        seq_len = tl.load(seq_len_ptr)
        NK = tl.cdiv(seq_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        offs_t = tl.arange(0, BK)
        offs_tile = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))

        for pair in tl.range(NPAIRS):
            ck0 = pair * 2
            ck1 = ck0 + 1
            t_offs0 = ck0 * BK + offs_t
            t_offs1 = ck1 * BK + offs_t
            valid0 = t_offs0 < seq_len
            valid1 = t_offs1 < seq_len

            page0 = tl.load(req_to_tokens + t_offs0 // PAGE_SIZE, valid0, other=0)
            page1 = tl.load(req_to_tokens + t_offs1 // PAGE_SIZE, valid1, other=0)
            kv_offsets0 = (page0 * PAGE_SIZE + t_offs0 % PAGE_SIZE).to(
                tl.int64
            ) * stride_kv
            kv_offsets1 = (page1 * PAGE_SIZE + t_offs1 % PAGE_SIZE).to(
                tl.int64
            ) * stride_kv

            k0_l_slot = k0_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k_ptr = kv_base + kv_offsets0[:, None] + k_cols[None, :]
                k_msk = valid0[:, None] & (k_cols < D)[None, :]
                k_blk = tle.load(k_ptr, mask=k_msk, other=0.0, is_async=True)
                tl.store(
                    tle.gpu.local_ptr(k0_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k_blk,
                    mask=k_msk,
                )
            k0_l_writer.commit(pair)

            k1_r_slot = k1_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k_ptr = kv_base + kv_offsets1[:, None] + k_cols[None, :]
                k_msk = valid1[:, None] & (k_cols < D)[None, :]
                k_blk = tle.load(k_ptr, mask=k_msk, other=0.0, is_async=True)
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k_blk,
                    mask=k_msk,
                )
            if TD > 0:
                offs_td = tl.arange(0, TDP)
                k_tail_ptr = kv_base + kv_offsets1[:, None] + (D + offs_td)[None, :]
                k_tail_msk = valid1[:, None] & (offs_td < TD)[None, :]
                k_tail_blk = tl.load(
                    k_tail_ptr,
                    mask=k_tail_msk,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK_tail),
                    k_tail_blk,
                    mask=k_tail_msk,
                )
            k1_r_writer.commit(pair)

            k0_r_slot = k0_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k_ptr = kv_base + kv_offsets0[:, None] + k_cols[None, :]
                k_msk = valid0[:, None] & (k_cols < D)[None, :]
                k_blk = tle.load(k_ptr, mask=k_msk, other=0.0, is_async=True)
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k_blk,
                    mask=k_msk,
                )
            if TD > 0:
                offs_td = tl.arange(0, TDP)
                k_tail_ptr = kv_base + kv_offsets0[:, None] + (D + offs_td)[None, :]
                k_tail_msk = valid0[:, None] & (offs_td < TD)[None, :]
                k_tail_blk = tl.load(
                    k_tail_ptr,
                    mask=k_tail_msk,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail),
                    k_tail_blk,
                    mask=k_tail_msk,
                )
            k0_r_writer.commit(pair)

            k1_l_slot = k1_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k_ptr = kv_base + kv_offsets1[:, None] + k_cols[None, :]
                k_msk = valid1[:, None] & (k_cols < D)[None, :]
                k_blk = tle.load(k_ptr, mask=k_msk, other=0.0, is_async=True)
                tl.store(
                    tle.gpu.local_ptr(k1_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k_blk,
                    mask=k_msk,
                )
            k1_l_writer.commit(pair)

            valid_slot = valid_writer.acquire(pair)
            row0 = tl.full([BK], 0, dtype=tl.int32)
            row1 = tl.full([BK], 1, dtype=tl.int32)
            tl.store(
                tle.gpu.local_ptr(valid_slot.is_valid, (row0, offs_t)),
                valid0.to(tl.int8),
            )
            tl.store(
                tle.gpu.local_ptr(valid_slot.is_valid, (row1, offs_t)),
                valid1.to(tl.int8),
            )
            valid_writer.commit(pair)

    @triton.jit
    def _tle_flash_mla_dense_consumer0(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k0_l_reader,
        k0_r_qk_reader,
        k1_l_remote_reader,
        valid_reader,
        sM_wg0_writer,
        sM_wg1_reader,
        sS0_writer,
        sS1_reader,
        sL_wg0_writer,
        sL_wg1_reader,
        output_desc,
        q_row,
        seq_len_ptr,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
    ):
        offs_dh = tl.arange(0, DPH)
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [q_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [q_row, DPH])
        if TD > 0:
            tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [q_row, D])
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)

        seq_len = tl.load(seq_len_ptr)
        NPAIRS = tl.cdiv(tl.cdiv(seq_len, BK), 2)
        for pair in tl.range(NPAIRS):
            k0_l_wait = k0_l_reader.wait(pair)
            k0_l_slot = k0_l_wait.slot
            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))

            qk0 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk0 = tl.dot(q_l_blk, tl.trans(k0_l_blk), qk0, out_dtype=tl.float32)
            k0_r_wait = k0_r_qk_reader.wait(pair)
            k0_r_slot = k0_r_wait.slot
            k0_r_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols_r)))
            qk0 = tl.dot(q_r_blk, tl.trans(k0_r_blk), qk0, out_dtype=tl.float32)
            if TD > 0:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k0_t_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK_tail))
                qk0 = tl.dot(q_tail_blk, tl.trans(k0_t_blk), qk0, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pair)
            row0 = tl.full([BK], 0, dtype=tl.int32)
            valid0 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_valid, (row0, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk0 = tl.where(valid0[None, :], qk0, float("-inf"))
            valid_reader.release(pair)

            local_max = tl.maximum(max_prev, tl.max(qk0, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob0 = tl.math.exp2(qk0 * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob0, axis=1)
            acc_l = acc_l * alpha[:, None]
            prob0_b = prob0.to(OUT_DTYPE)

            sM_wg0_slot = sM_wg0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)
            sM_wg0_writer.commit(pair)

            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            acc_l = tl.dot(prob0_b, k0_l_blk, acc_l, out_dtype=tl.float32)
            k0_l_reader.release(pair)
            k0_r_qk_reader.release(pair)

            sM_wg1_wait = sM_wg1_reader.wait(pair)
            max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))
            sM_wg1_reader.release(pair)
            final_scale = tl.math.exp2((local_max - max_next) * log_scale)
            sum_exp = sum_exp * final_scale
            acc_l = acc_l * final_scale[:, None]

            sS0_slot = sS0_writer.acquire(pair)
            tl.store(
                tle.gpu.local_ptr(sS0_slot.sS0),
                (prob0 * final_scale[:, None]).to(OUT_DTYPE),
            )
            sS0_writer.commit(pair)

            sS1_wait = sS1_reader.wait(pair)
            prob1 = tl.load(tle.gpu.local_ptr(sS1_wait.slot.sS1))
            k1_l_wait = k1_l_remote_reader.wait(pair)
            k1_l_blk = tl.load(
                tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols_l))
            )
            acc_l = tl.dot(prob1, k1_l_blk, acc_l, out_dtype=tl.float32)
            sS1_reader.release(pair)
            k1_l_remote_reader.release(pair)
            max_prev = max_next

        sL_wg0_slot = sL_wg0_writer.acquire(0)
        tl.store(tle.gpu.local_ptr(sL_wg0_slot.sL), sum_exp)
        sL_wg0_writer.commit(0)
        sL_wg1_wait = sL_wg1_reader.wait(1)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg1_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg1_reader.release(1)
        out_l_vals = acc_l * tl.fdiv(1.0, total_sum)[:, None]
        tl.store(q_l_smem_ptr, out_l_vals.to(OUT_DTYPE))
        tle.gpu.copy(q_slot.sQ_l, output_desc, [BH, DPH], [q_row, 0])

    @triton.jit
    def _tle_flash_mla_dense_consumer1(
        q_reader,
        k1_r_reader,
        k1_l_qk_reader,
        k0_r_remote_reader,
        valid_reader,
        sM_wg1_writer,
        sM_wg0_reader,
        sS1_writer,
        sS0_reader,
        sL_wg1_writer,
        sL_wg0_reader,
        output_desc,
        q_row,
        seq_len_ptr,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
    ):
        offs_dh = tl.arange(0, DPH)
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))
        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        seq_len = tl.load(seq_len_ptr)
        NPAIRS = tl.cdiv(tl.cdiv(seq_len, BK), 2)
        for pair in tl.range(NPAIRS):
            k1_r_wait = k1_r_reader.wait(pair)
            k1_r_slot = k1_r_wait.slot
            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k1_r_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))

            qk1 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk1 = tl.dot(q_r_blk, tl.trans(k1_r_blk), qk1, out_dtype=tl.float32)
            if TD > 0:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k1_t_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK_tail))
                qk1 = tl.dot(q_tail_blk, tl.trans(k1_t_blk), qk1, out_dtype=tl.float32)
            k1_l_wait = k1_l_qk_reader.wait(pair)
            k1_l_slot = k1_l_wait.slot
            k1_l_blk = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols_l)))
            qk1 = tl.dot(q_l_blk, tl.trans(k1_l_blk), qk1, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pair)
            row1 = tl.full([BK], 1, dtype=tl.int32)
            valid1 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_valid, (row1, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk1 = tl.where(valid1[None, :], qk1, float("-inf"))
            valid_reader.release(pair)

            sM_wg0_wait = sM_wg0_reader.wait(pair)
            candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))
            sM_wg0_reader.release(pair)
            candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))
            max_next = tl.maximum(candidate1, candidate0)
            sM_wg1_slot = sM_wg1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)
            sM_wg1_writer.commit(pair)

            alpha = tl.math.exp2((max_prev - max_next) * log_scale)
            prob1 = tl.math.exp2(qk1 * log_scale - max_next[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob1, axis=1)
            acc_r = acc_r * alpha[:, None]
            prob1_b = prob1.to(OUT_DTYPE)

            acc_r = tl.dot(prob1_b, k1_r_blk, acc_r, out_dtype=tl.float32)
            sS1_slot = sS1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS1_slot.sS1), prob1_b)
            sS1_writer.commit(pair)

            sS0_wait = sS0_reader.wait(pair)
            prob0 = tl.load(tle.gpu.local_ptr(sS0_wait.slot.sS0))
            k0_r_wait = k0_r_remote_reader.wait(pair)
            k0_r_blk = tl.load(
                tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols_r))
            )
            acc_r = tl.dot(prob0, k0_r_blk, acc_r, out_dtype=tl.float32)
            k1_r_reader.release(pair)
            k1_l_qk_reader.release(pair)
            sS0_reader.release(pair)
            k0_r_remote_reader.release(pair)
            max_prev = max_next

        sL_wg1_slot = sL_wg1_writer.acquire(1)
        tl.store(tle.gpu.local_ptr(sL_wg1_slot.sL), sum_exp)
        sL_wg1_writer.commit(1)
        sL_wg0_wait = sL_wg0_reader.wait(0)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg0_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg0_reader.release(0)
        out_r_vals = acc_r * tl.fdiv(1.0, total_sum)[:, None]
        tl.store(q_r_smem_ptr, out_r_vals.to(OUT_DTYPE))
        tle.gpu.copy(q_slot.sQ_r, output_desc, [BH, DPH], [q_row, DPH])

    @triton.jit
    def _tle_flash_mla_dense_2wg_consumer0(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        q,
        kv,
        req_to_tokens,
        seq_len_ptr,
        sM_wg0_writer,
        sM_wg1_reader,
        sS0_writer,
        sS1_reader,
        sL_wg0_writer,
        sL_wg1_reader,
        output_desc,
        q_row,
        log_scale: tl.constexpr,
        Q_STRIDE: tl.constexpr,
        STRIDE_KV: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
    ):
        offs_t = tl.arange(0, BK)
        offs_dh = tl.arange(0, DPH)

        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [q_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [q_row, DPH])
        _ = tq_desc
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)

        seq_len = tl.load(seq_len_ptr)
        NPAIRS = tl.cdiv(tl.cdiv(seq_len, BK), 2)
        for pair in tl.range(NPAIRS):
            ck0 = pair * 2
            ck1 = ck0 + 1
            t_offs0 = ck0 * BK + offs_t
            t_offs1 = ck1 * BK + offs_t
            valid0 = t_offs0 < seq_len
            valid1 = t_offs1 < seq_len
            page0 = tl.load(req_to_tokens + t_offs0 // PAGE_SIZE, valid0, other=0)
            page1 = tl.load(req_to_tokens + t_offs1 // PAGE_SIZE, valid1, other=0)
            kv_offsets0 = (page0 * PAGE_SIZE + t_offs0 % PAGE_SIZE).to(
                tl.int64
            ) * STRIDE_KV
            kv_offsets1 = (page1 * PAGE_SIZE + t_offs1 % PAGE_SIZE).to(
                tl.int64
            ) * STRIDE_KV

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k0_l_blk = tl.load(
                kv + kv_offsets0[:, None] + offs_dh[None, :],
                mask=valid0[:, None] & (offs_dh < D)[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            qk0 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk0 = tl.dot(q_l_blk, tl.trans(k0_l_blk), qk0, out_dtype=tl.float32)
            k0_r_blk = tl.load(
                kv + kv_offsets0[:, None] + (DPH + offs_dh)[None, :],
                mask=valid0[:, None] & (DPH + offs_dh < D)[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            qk0 = tl.dot(q_r_blk, tl.trans(k0_r_blk), qk0, out_dtype=tl.float32)
            if TD > 0:
                offs_td = tl.arange(0, TDP)
                offs_h = tl.arange(0, BH)
                q_tail_blk = tl.load(
                    q + (q_row + offs_h[:, None]) * Q_STRIDE + D + offs_td[None, :],
                    mask=offs_td[None, :] < TD,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                k0_t_blk = tl.load(
                    kv + kv_offsets0[:, None] + (D + offs_td)[None, :],
                    mask=valid0[:, None] & (offs_td < TD)[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                qk0 = tl.dot(q_tail_blk, tl.trans(k0_t_blk), qk0, out_dtype=tl.float32)
            qk0 = tl.where(valid0[None, :], qk0, float("-inf"))

            local_max = tl.maximum(max_prev, tl.max(qk0, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob0 = tl.math.exp2(qk0 * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob0, axis=1)
            acc_l = acc_l * alpha[:, None]
            prob0_b = prob0.to(OUT_DTYPE)

            sM_wg0_slot = sM_wg0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)
            sM_wg0_writer.commit(pair)

            acc_l = tl.dot(prob0_b, k0_l_blk, acc_l, out_dtype=tl.float32)
            sM_wg1_wait = sM_wg1_reader.wait(pair)
            max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))
            sM_wg1_reader.release(pair)
            final_scale = tl.math.exp2((local_max - max_next) * log_scale)
            sum_exp = sum_exp * final_scale
            acc_l = acc_l * final_scale[:, None]

            sS0_slot = sS0_writer.acquire(pair)
            tl.store(
                tle.gpu.local_ptr(sS0_slot.sS0),
                (prob0 * final_scale[:, None]).to(OUT_DTYPE),
            )
            sS0_writer.commit(pair)

            sS1_wait = sS1_reader.wait(pair)
            prob1 = tl.load(tle.gpu.local_ptr(sS1_wait.slot.sS1))
            k1_l_blk = tl.load(
                kv + kv_offsets1[:, None] + offs_dh[None, :],
                mask=valid1[:, None] & (offs_dh < D)[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            acc_l = tl.dot(prob1, k1_l_blk, acc_l, out_dtype=tl.float32)
            sS1_reader.release(pair)
            max_prev = max_next

        sL_wg0_slot = sL_wg0_writer.acquire(0)
        tl.store(tle.gpu.local_ptr(sL_wg0_slot.sL), sum_exp)
        sL_wg0_writer.commit(0)
        sL_wg1_wait = sL_wg1_reader.wait(1)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg1_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg1_reader.release(1)
        out_l_vals = acc_l * tl.fdiv(1.0, total_sum)[:, None]
        tl.store(q_l_smem_ptr, out_l_vals.to(OUT_DTYPE))
        tle.gpu.copy(q_slot.sQ_l, output_desc, [BH, DPH], [q_row, 0])

    @triton.jit
    def _tle_flash_mla_dense_2wg_consumer1(
        q_reader,
        q,
        kv,
        req_to_tokens,
        seq_len_ptr,
        sM_wg1_writer,
        sM_wg0_reader,
        sS1_writer,
        sS0_reader,
        sL_wg1_writer,
        sL_wg0_reader,
        output_desc,
        q_row,
        log_scale: tl.constexpr,
        Q_STRIDE: tl.constexpr,
        STRIDE_KV: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
    ):
        offs_t = tl.arange(0, BK)
        offs_dh = tl.arange(0, DPH)
        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        seq_len = tl.load(seq_len_ptr)
        NPAIRS = tl.cdiv(tl.cdiv(seq_len, BK), 2)
        for pair in tl.range(NPAIRS):
            ck0 = pair * 2
            ck1 = ck0 + 1
            t_offs0 = ck0 * BK + offs_t
            t_offs1 = ck1 * BK + offs_t
            valid0 = t_offs0 < seq_len
            valid1 = t_offs1 < seq_len
            page0 = tl.load(req_to_tokens + t_offs0 // PAGE_SIZE, valid0, other=0)
            page1 = tl.load(req_to_tokens + t_offs1 // PAGE_SIZE, valid1, other=0)
            kv_offsets0 = (page0 * PAGE_SIZE + t_offs0 % PAGE_SIZE).to(
                tl.int64
            ) * STRIDE_KV
            kv_offsets1 = (page1 * PAGE_SIZE + t_offs1 % PAGE_SIZE).to(
                tl.int64
            ) * STRIDE_KV

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k1_r_blk = tl.load(
                kv + kv_offsets1[:, None] + (DPH + offs_dh)[None, :],
                mask=valid1[:, None] & (DPH + offs_dh < D)[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            qk1 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk1 = tl.dot(q_r_blk, tl.trans(k1_r_blk), qk1, out_dtype=tl.float32)
            if TD > 0:
                offs_td = tl.arange(0, TDP)
                offs_h = tl.arange(0, BH)
                q_tail_blk = tl.load(
                    q + (q_row + offs_h[:, None]) * Q_STRIDE + D + offs_td[None, :],
                    mask=offs_td[None, :] < TD,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                k1_t_blk = tl.load(
                    kv + kv_offsets1[:, None] + (D + offs_td)[None, :],
                    mask=valid1[:, None] & (offs_td < TD)[None, :],
                    other=0.0,
                    eviction_policy="evict_first",
                )
                qk1 = tl.dot(q_tail_blk, tl.trans(k1_t_blk), qk1, out_dtype=tl.float32)
            k1_l_blk = tl.load(
                kv + kv_offsets1[:, None] + offs_dh[None, :],
                mask=valid1[:, None] & (offs_dh < D)[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            qk1 = tl.dot(q_l_blk, tl.trans(k1_l_blk), qk1, out_dtype=tl.float32)
            qk1 = tl.where(valid1[None, :], qk1, float("-inf"))

            sM_wg0_wait = sM_wg0_reader.wait(pair)
            candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))
            sM_wg0_reader.release(pair)
            candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))
            max_next = tl.maximum(candidate1, candidate0)
            sM_wg1_slot = sM_wg1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)
            sM_wg1_writer.commit(pair)

            alpha = tl.math.exp2((max_prev - max_next) * log_scale)
            prob1 = tl.math.exp2(qk1 * log_scale - max_next[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob1, axis=1)
            acc_r = acc_r * alpha[:, None]
            prob1_b = prob1.to(OUT_DTYPE)
            acc_r = tl.dot(prob1_b, k1_r_blk, acc_r, out_dtype=tl.float32)

            sS1_slot = sS1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS1_slot.sS1), prob1_b)
            sS1_writer.commit(pair)

            sS0_wait = sS0_reader.wait(pair)
            prob0 = tl.load(tle.gpu.local_ptr(sS0_wait.slot.sS0))
            k0_r_blk = tl.load(
                kv + kv_offsets0[:, None] + (DPH + offs_dh)[None, :],
                mask=valid0[:, None] & (DPH + offs_dh < D)[None, :],
                other=0.0,
                eviction_policy="evict_first",
            )
            acc_r = tl.dot(prob0, k0_r_blk, acc_r, out_dtype=tl.float32)
            sS0_reader.release(pair)
            max_prev = max_next

        sL_wg1_slot = sL_wg1_writer.acquire(1)
        tl.store(tle.gpu.local_ptr(sL_wg1_slot.sL), sum_exp)
        sL_wg1_writer.commit(1)
        sL_wg0_wait = sL_wg0_reader.wait(0)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg0_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg0_reader.release(0)
        out_r_vals = acc_r * tl.fdiv(1.0, total_sum)[:, None]
        tl.store(q_r_smem_ptr, out_r_vals.to(OUT_DTYPE))
        tle.gpu.copy(q_slot.sQ_r, output_desc, [BH, DPH], [q_row, DPH])

    @triton.jit
    def _tle_flash_mla_dense_2wg_fwd(
        q_desc,
        tq_desc,
        output_desc,
        q,
        kv,
        req_to_tokens,
        seq_lens,
        sm_scale: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        DQK: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SEQLEN_PAD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
    ):
        _ = B
        DPH: tl.constexpr = DP // 2
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        h_base = pid_h * BH
        q_row = pid_b * H + h_base
        req_base = req_to_tokens + pid_b * (MAX_SEQLEN_PAD // PAGE_SIZE)
        seq_len_ptr = seq_lens + pid_b

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        _ = tq_desc
        q_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_2wg_sQ",
            readers=("wg0", "wg1"),
            one_shot=True,
            sQ_l=sQ_l_smem,
            sQ_r=sQ_r_smem,
        )
        sS0_smem = tle.gpu.alloc(
            [1, BH, BK], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sS1_smem = tle.gpu.alloc(
            [1, BH, BK], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sM_smem = tle.gpu.alloc(
            [1, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sL_smem = tle.gpu.alloc(
            [2, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sM_wg0_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_2wg_sM0", sM=sM_smem
        )
        sM_wg1_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_2wg_sM1", sM=sM_smem
        )
        sS0_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_2wg_sS0", sS0=sS0_smem
        )
        sS1_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_2wg_sS1", sS1=sS1_smem
        )
        sL_wg0_pipe = tle.pipe(
            capacity=2, scope="cta", name="flash_mla_2wg_sL0", sL=sL_smem
        )
        sL_wg1_pipe = tle.pipe(
            capacity=2, scope="cta", name="flash_mla_2wg_sL1", sL=sL_smem
        )

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634
        tle.gpu.warp_specialize(
            [
                (
                    _tle_flash_mla_dense_2wg_consumer0,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        q,
                        kv,
                        req_base,
                        seq_len_ptr,
                        sM_wg0_pipe.writer(),
                        sM_wg1_pipe.reader(),
                        sS0_pipe.writer(),
                        sS1_pipe.reader(),
                        sL_wg0_pipe.writer(),
                        sL_wg1_pipe.reader(),
                        output_desc,
                        q_row,
                        log_scale,
                        DQK,
                        DQK,
                        PAGE_SIZE,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        BK,
                        BH,
                        DPH,
                        TDP,
                    ),
                ),
                (
                    _tle_flash_mla_dense_2wg_consumer1,
                    (
                        q_pipe.reader("wg1"),
                        q,
                        kv,
                        req_base,
                        seq_len_ptr,
                        sM_wg1_pipe.writer(),
                        sM_wg0_pipe.reader(),
                        sS1_pipe.writer(),
                        sS0_pipe.reader(),
                        sL_wg1_pipe.writer(),
                        sL_wg0_pipe.reader(),
                        output_desc,
                        q_row,
                        log_scale,
                        DQK,
                        DQK,
                        PAGE_SIZE,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        BK,
                        BH,
                        DPH,
                        TDP,
                    ),
                ),
            ],
            [4],
            [216],
        )

    @triton.jit
    def _tle_flash_mla_dense_fwd(
        q_desc,
        tq_desc,
        output_desc,
        kv,
        req_to_tokens,
        seq_lens,
        sm_scale: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        DQK: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SEQLEN_PAD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        PAIR_BLOCKS: tl.constexpr,
    ):
        _ = B
        _ = DQK
        DPH: tl.constexpr = DP // 2
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        h_base = pid_h * BH
        q_row = pid_b * H + h_base
        req_base = req_to_tokens + pid_b * (MAX_SEQLEN_PAD // PAGE_SIZE)
        seq_len_ptr = seq_lens + pid_b

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        if TD > 0:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flash_mla_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flash_mla_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )

        sK0_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sK1_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        if TD > 0:
            sK0_tail_smem = tle.gpu.alloc(
                [1, BK, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            sK1_tail_smem = tle.gpu.alloc(
                [1, BK, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            if BH == BK:
                sS0_smem = sK0_tail_smem
            else:
                sS0_smem = tle.gpu.alloc(
                    [1, BH, BK],
                    dtype=kv.dtype.element_ty,
                    layout=None,
                    scope=tle.gpu.smem,
                )
        else:
            sS0_smem = tle.gpu.alloc(
                [1, BH, BK], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
        sS1_smem = tle.gpu.alloc(
            [1, BH, BK], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        is_valid_smem = tle.gpu.alloc(
            [1, PAIR_BLOCKS, BK],
            dtype=tl.int8,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sM_smem = tle.gpu.alloc(
            [1, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sL_smem = tle.gpu.alloc(
            [2, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        k0_l_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_sK0_l", sK=sK0_smem
        )
        if TD > 0:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flash_mla_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
                sK_tail=sK0_tail_smem,
            )
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flash_mla_sK1_r",
                sK=sK1_smem,
                sK_tail=sK1_tail_smem,
            )
        else:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flash_mla_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
            )
            k1_r_pipe = tle.pipe(
                capacity=1, scope="cta", name="flash_mla_sK1_r", sK=sK1_smem
            )
        k1_l_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_sK1_l",
            readers=("qk", "remote"),
            sK=sK1_smem,
        )
        valid_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_valid",
            readers=("wg0", "wg1"),
            is_valid=is_valid_smem,
        )
        sM_wg0_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_sM0", sM=sM_smem
        )
        sM_wg1_pipe = tle.pipe(
            capacity=1, scope="cta", name="flash_mla_sM1", sM=sM_smem
        )
        sS0_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_sS0", sS0=sS0_smem)
        sS1_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_sS1", sS1=sS1_smem)
        sL_wg0_pipe = tle.pipe(
            capacity=2, scope="cta", name="flash_mla_sL0", sL=sL_smem
        )
        sL_wg1_pipe = tle.pipe(
            capacity=2, scope="cta", name="flash_mla_sL1", sL=sL_smem
        )

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634
        tle.gpu.warp_specialize(
            [
                (
                    _tle_flash_mla_dense_consumer0,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k0_l_pipe.reader(),
                        k0_r_pipe.reader("qk"),
                        k1_l_pipe.reader("remote", fields=("sK",)),
                        valid_pipe.reader("wg0"),
                        sM_wg0_pipe.writer(),
                        sM_wg1_pipe.reader(),
                        sS0_pipe.writer(),
                        sS1_pipe.reader(),
                        sL_wg0_pipe.writer(),
                        sL_wg1_pipe.reader(),
                        output_desc,
                        q_row,
                        seq_len_ptr,
                        log_scale,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        BK,
                        BH,
                        DPH,
                        TDP,
                    ),
                ),
                (
                    _tle_flash_mla_dense_consumer1,
                    (
                        q_pipe.reader("wg1"),
                        k1_r_pipe.reader(),
                        k1_l_pipe.reader("qk"),
                        k0_r_pipe.reader("remote", fields=("sK",)),
                        valid_pipe.reader("wg1"),
                        sM_wg1_pipe.writer(),
                        sM_wg0_pipe.reader(),
                        sS1_pipe.writer(),
                        sS0_pipe.reader(),
                        sL_wg1_pipe.writer(),
                        sL_wg0_pipe.reader(),
                        output_desc,
                        q_row,
                        seq_len_ptr,
                        log_scale,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        BK,
                        BH,
                        DPH,
                        TDP,
                    ),
                ),
                (
                    _tle_flash_mla_dense_producer,
                    (
                        k0_l_pipe.writer(),
                        k0_r_pipe.writer(),
                        k1_l_pipe.writer(),
                        k1_r_pipe.writer(),
                        valid_pipe.writer(),
                        kv,
                        req_base,
                        seq_len_ptr,
                        DQK,
                        PAGE_SIZE,
                        D,
                        TD,
                        DPH,
                        TDP,
                        BK,
                    ),
                ),
            ],
            [4, 4],
            [216, 72],
        )

    @triton.jit
    def _splitkv_direct_kernel(
        Q_ptr,
        kv_ptr,
        req_to_tokens,
        seq_lens,
        o_partial,
        lse_partial,
        sm_scale: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kv: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        DQK: tl.constexpr,
        DV: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SEQLEN_PAD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        """Split-KV main kernel: direct global loads, no pipe, no warp_specialize."""
        DPH: tl.constexpr = DP // 2

        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_s = tl.program_id(2)

        h_base = pid_h * BH
        q_row = pid_b * H + h_base  # row into o_partial / lse_partial

        # --- guard empty split ---
        seq_len = tl.load(seq_lens + pid_b)
        k_start = pid_s * BLOCKS_PER_SPLIT
        NK = tl.cdiv(seq_len, BK)
        k_end = k_start + BLOCKS_PER_SPLIT
        if k_end > NK:
            k_end = NK
        if k_start >= NK:
            return
        num_blocks = k_end - k_start

        # --- load Q once (global -> registers), split into left/right halves ---
        offs_h = tl.arange(0, BH)
        q_heads = h_base + offs_h
        mask_h = q_heads < H
        offs_dph = tl.arange(0, DPH)
        q_batch_off = pid_b * stride_qb

        q_l = tl.load(
            Q_ptr + q_batch_off + q_heads[:, None] * stride_qh + offs_dph[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )  # [BH, DPH]
        q_r = tl.load(
            Q_ptr
            + q_batch_off
            + q_heads[:, None] * stride_qh
            + (DPH + offs_dph)[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )  # [BH, DPH]

        if TD > 0:
            offs_td = tl.arange(0, TDP)
            q_tail = tl.load(
                Q_ptr
                + q_batch_off
                + q_heads[:, None] * stride_qh
                + (DP + offs_td)[None, :],
                mask=mask_h[:, None] & (offs_td < TD)[None, :],
                other=0.0,
            )  # [BH, TDP]

        max_prev = tl.full([BH], float("-inf"), dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        offs_k = tl.arange(0, BK)
        req_base = req_to_tokens + pid_b * (MAX_SEQLEN_PAD // PAGE_SIZE)

        for _ in tl.range(num_blocks):
            ck = k_start + _
            t_offs = ck * BK + offs_k
            valid = t_offs < seq_len

            # page-table lookup -> linear KV offset
            page = tl.load(req_base + t_offs // PAGE_SIZE, mask=valid, other=0)
            kv_offs = (page * PAGE_SIZE + t_offs % PAGE_SIZE).to(tl.int64) * stride_kv

            # load K halves directly from global memory
            k_l = tl.load(
                kv_ptr + kv_offs[:, None] + offs_dph[None, :],
                mask=valid[:, None],
                other=0.0,
            )  # [BK, DPH]
            k_r = tl.load(
                kv_ptr + kv_offs[:, None] + (DPH + offs_dph)[None, :],
                mask=valid[:, None],
                other=0.0,
            )  # [BK, DPH]

            # qk = sum over left/right/tail partitions
            qk = tl.dot(q_l, tl.trans(k_l), out_dtype=tl.float32)
            qk = tl.dot(q_r, tl.trans(k_r), qk, out_dtype=tl.float32)
            if TD > 0:
                offs_tdp = tl.arange(0, TDP)
                k_t_val = tl.load(
                    kv_ptr + kv_offs[:, None] + (DP + offs_tdp)[None, :],
                    mask=valid[:, None] & (offs_tdp < TD)[None, :],
                    other=0.0,
                )  # [BK, TDP]
                qk = tl.dot(q_tail, tl.trans(k_t_val), qk, out_dtype=tl.float32)

            qk *= sm_scale
            qk = tl.where(valid[None, :], qk, float("-inf"))

            # online softmax
            local_max = tl.maximum(max_prev, tl.max(qk, axis=1))
            alpha = tl.exp(max_prev - local_max)
            prob = tl.exp(qk - local_max[:, None])
            sum_exp = sum_exp * alpha + tl.sum(prob, axis=1)
            acc_l = acc_l * alpha[:, None]
            acc_r = acc_r * alpha[:, None]
            acc_l = tl.dot(prob.to(k_l.dtype), k_l, acc_l, out_dtype=tl.float32)
            acc_r = tl.dot(prob.to(k_r.dtype), k_r, acc_r, out_dtype=tl.float32)
            max_prev = local_max

        # --- store partial O / LSE (compatible with _tle_splitkv_combine) ---
        inv_sum = tl.fdiv(1.0, sum_exp)
        offs_dv_l = tl.arange(0, DPH)
        offs_dv_r = DPH + tl.arange(0, DPH)

        TOTAL_ROWS: tl.constexpr = B * H
        stride_o_split = TOTAL_ROWS * DV
        stride_lse_split = TOTAL_ROWS

        o_base = o_partial + pid_s * stride_o_split + q_row * DV
        lse_base = lse_partial + pid_s * stride_lse_split + q_row

        tl.store(
            o_base + offs_h[:, None] * DV + offs_dv_l[None, :],
            (acc_l * inv_sum[:, None]).to(o_partial.dtype.element_ty),
            mask=mask_h[:, None],
        )
        tl.store(
            o_base + offs_h[:, None] * DV + offs_dv_r[None, :],
            (acc_r * inv_sum[:, None]).to(o_partial.dtype.element_ty),
            mask=mask_h[:, None],
        )
        # max_prev is already in scaled space (qk was multiplied by sm_scale)
        lse = tl.where(
            mask_h,
            max_prev + tl.log(sum_exp),
            float("-inf"),
        )
        tl.store(lse_base + offs_h, lse, mask=mask_h)

    @triton.jit
    def _tle_splitkv_producer(
        k_writer,
        kv_base,
        req_to_tokens,
        seq_len_ptr,
        stride_kv: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DV: tl.constexpr,
        TD: tl.constexpr,
        TDP: tl.constexpr,
        BK: tl.constexpr,
        k_start: tl.constexpr,
        k_end_param,
    ):
        seq_len = tl.load(seq_len_ptr)
        NK = tl.cdiv(seq_len, BK)
        k_end = NK if k_end_param > NK else k_end_param
        offs_t = tl.arange(0, BK)
        offs_d = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))

        for ck in tl.range(k_start, k_end):
            t_offs = ck * BK + offs_t
            valid = t_offs < seq_len
            page = tl.load(req_to_tokens + t_offs // PAGE_SIZE, valid, other=0)
            kv_offsets = (page * PAGE_SIZE + t_offs % PAGE_SIZE).to(
                tl.int64
            ) * stride_kv

            k_slot = k_writer.acquire(ck - k_start)
            for tile in tl.static_range(0, DV, 64):
                k_cols = tile + offs_d
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k_ptr = kv_base + kv_offsets[:, None] + k_cols[None, :]
                k_msk = valid[:, None]
                k_blk = tle.load(k_ptr, mask=k_msk, other=0.0, is_async=True)
                tl.store(
                    tle.gpu.local_ptr(k_slot.sK, (kv_tile_rows, k_cols_b)),
                    k_blk,
                    mask=k_msk,
                )
            if TD > 0:
                offs_td = tl.arange(0, TDP)
                k_tail_ptr = kv_base + kv_offsets[:, None] + (DV + offs_td)[None, :]
                k_tail_msk = valid[:, None] & (offs_td < TD)[None, :]
                k_tail_blk = tle.load(
                    k_tail_ptr, mask=k_tail_msk, other=0.0, is_async=True
                )
                tl.store(tle.gpu.local_ptr(k_slot.sK_tail), k_tail_blk, mask=k_tail_msk)

            tl.store(tle.gpu.local_ptr(k_slot.is_valid), valid.to(tl.int8))
            k_writer.commit(ck - k_start)

    @triton.jit
    def _tle_splitkv_consumer(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k_reader,
        o_partial,
        lse_partial,
        q_row,
        seq_len_ptr,
        log_scale: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        k_start: tl.constexpr,
        k_end_param,
        split_idx,
        stride_o_split,
        stride_lse_split,
    ):
        seq_len = tl.load(seq_len_ptr)
        NK = tl.cdiv(seq_len, BK)
        k_end = NK if k_end_param > NK else k_end_param
        num_blocks = k_end - k_start

        offs_dh = tl.arange(0, DPH)
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [q_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [q_row, DPH])
        if TD > 0:
            tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [q_row, 2 * DPH])
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)

        # Q is loop-invariant in decode attention: load once outside the K/V loop
        q_l_blk = tl.load(q_l_smem_ptr)
        q_r_blk = tl.load(q_r_smem_ptr)
        if TD > 0:
            q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))

        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        for i in tl.range(num_blocks):
            k_wait = k_reader.wait(i)
            k_slot = k_wait.slot
            k_l_blk = tl.load(tle.gpu.local_ptr(k_slot.sK, (kv_rows, kv_cols_l)))
            k_r_blk = tl.load(tle.gpu.local_ptr(k_slot.sK, (kv_rows, kv_cols_r)))

            qk = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk = tl.dot(q_l_blk, tl.trans(k_l_blk), qk, out_dtype=tl.float32)
            qk = tl.dot(q_r_blk, tl.trans(k_r_blk), qk, out_dtype=tl.float32)
            if TD > 0:
                k_t_blk = tl.load(tle.gpu.local_ptr(k_slot.sK_tail))
                qk = tl.dot(q_tail_blk, tl.trans(k_t_blk), qk, out_dtype=tl.float32)

            valid_mask = tl.load(tle.gpu.local_ptr(k_slot.is_valid)) != 0
            qk = tl.where(valid_mask[None, :], qk, float("-inf"))

            local_max = tl.maximum(max_prev, tl.max(qk, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob = tl.math.exp2(qk * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob, axis=1)
            acc_l = acc_l * alpha[:, None]
            acc_r = acc_r * alpha[:, None]
            prob_b = prob.to(OUT_DTYPE)

            acc_l = tl.dot(prob_b, k_l_blk, acc_l, out_dtype=tl.float32)
            acc_r = tl.dot(prob_b, k_r_blk, acc_r, out_dtype=tl.float32)
            k_reader.release(i)
            max_prev = local_max

        lse = max_prev * log_scale * 0.6931471805599453 + tl.log(sum_exp)
        inv_sum = tl.fdiv(1.0, sum_exp)
        acc_l = acc_l * inv_sum[:, None]
        acc_r = acc_r * inv_sum[:, None]
        offs_h = tl.arange(0, BH)
        offs_dv_l = tl.arange(0, DPH)
        offs_dv_r = DPH + tl.arange(0, DPH)
        DV2: tl.constexpr = 2 * DPH
        o_base = o_partial + split_idx * stride_o_split + q_row * DV2
        lse_base = lse_partial + split_idx * stride_lse_split + q_row
        tl.store(
            o_base + offs_h[:, None] * DV2 + offs_dv_l[None, :], acc_l.to(OUT_DTYPE)
        )
        tl.store(
            o_base + offs_h[:, None] * DV2 + offs_dv_r[None, :], acc_r.to(OUT_DTYPE)
        )
        tl.store(lse_base + offs_h, lse)

    @triton.jit
    def _tle_persistent_splitkv_producer(
        k_writer,
        valid_writer,
        kv_base,
        req_to_tokens,
        seq_lens,
        stride_kv: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        DV: tl.constexpr,
        TD: tl.constexpr,
        TDP: tl.constexpr,
        BK: tl.constexpr,
        MAX_SEQLEN_PAD: tl.constexpr,
        B: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        NUM_SMS: tl.constexpr,
        total_tiles,
    ):
        offs_t = tl.arange(0, BK)
        offs_d = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))

        pid = tl.program_id(0)
        for tile_linear in tl.range(pid, total_tiles, NUM_SMS):
            pid_s = tile_linear % NUM_SPLITS
            pid_b = (tile_linear // NUM_SPLITS) % B
            tile_iter = tile_linear // NUM_SMS
            req_base = req_to_tokens + pid_b * (MAX_SEQLEN_PAD // PAGE_SIZE)
            seq_len = tl.load(seq_lens + pid_b)
            k_start = pid_s * BLOCKS_PER_SPLIT

            for i in tl.range(BLOCKS_PER_SPLIT):
                stage = tile_iter * BLOCKS_PER_SPLIT + i
                t_offs = (k_start + i) * BK + offs_t
                valid = t_offs < seq_len
                page = tl.load(req_base + t_offs // PAGE_SIZE, valid, other=0)
                kv_offsets = (page * PAGE_SIZE + t_offs % PAGE_SIZE).to(
                    tl.int64
                ) * stride_kv

                k_slot = k_writer.acquire(stage)
                for tile in tl.static_range(0, DV, 64):
                    k_cols = tile + offs_d
                    k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                    k_ptr = kv_base + kv_offsets[:, None] + k_cols[None, :]
                    k_msk = valid[:, None]
                    k_blk = tle.load(k_ptr, mask=k_msk, other=0.0, is_async=True)
                    tl.store(
                        tle.gpu.local_ptr(k_slot.sK, (kv_tile_rows, k_cols_b)),
                        k_blk,
                        mask=k_msk,
                    )
                if TD > 0:
                    offs_td = tl.arange(0, TDP)
                    k_tail_ptr = kv_base + kv_offsets[:, None] + (DV + offs_td)[None, :]
                    k_tail_msk = valid[:, None] & (offs_td < TD)[None, :]
                    k_tail_blk = tle.load(
                        k_tail_ptr, mask=k_tail_msk, other=0.0, is_async=True
                    )
                    tl.store(
                        tle.gpu.local_ptr(k_slot.sK_tail), k_tail_blk, mask=k_tail_msk
                    )
                k_writer.commit(stage)

                valid_slot = valid_writer.acquire(stage)
                tl.store(tle.gpu.local_ptr(valid_slot.is_valid), valid.to(tl.int8))
                valid_writer.commit(stage)

    @triton.jit
    def _tle_persistent_splitkv_consumer(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k_reader,
        valid_reader,
        o_partial,
        lse_partial,
        seq_lens,
        log_scale: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        NUM_SMS: tl.constexpr,
        total_tiles,
    ):
        DV2: tl.constexpr = 2 * DPH
        TOTAL_ROWS: tl.constexpr = B * H

        offs_dh = tl.arange(0, DPH)
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))
        offs_h = tl.arange(0, BH)
        offs_dv_l = tl.arange(0, DPH)
        offs_dv_r = DPH + tl.arange(0, DPH)

        pid = tl.program_id(0)
        for tile_linear in tl.range(pid, total_tiles, NUM_SMS):
            pid_s = tile_linear % NUM_SPLITS
            tile_group = tile_linear // NUM_SPLITS
            pid_b = tile_group % B
            pid_h = tile_group // B
            tile_iter = tile_linear // NUM_SMS
            q_row = pid_b * H + pid_h * BH
            seq_len = tl.load(seq_lens + pid_b)
            active = pid_s * BLOCKS_PER_SPLIT * BK < seq_len

            q_write_slot = q_writer.acquire(tile_iter)
            tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [q_row, 0])
            tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [q_row, DPH])
            if TD > 0:
                tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [q_row, 2 * DPH])
            q_writer.commit(tile_iter)

            q_slot = q_reader.wait(tile_iter).slot
            q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
            q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
            max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
            sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
            acc_l = tl.zeros([BH, DPH], dtype=tl.float32)
            acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

            for i in tl.range(BLOCKS_PER_SPLIT):
                stage = tile_iter * BLOCKS_PER_SPLIT + i
                k_wait = k_reader.wait(stage)
                k_slot = k_wait.slot
                q_l_blk = tl.load(q_l_smem_ptr)
                q_r_blk = tl.load(q_r_smem_ptr)
                k_l_blk = tl.load(tle.gpu.local_ptr(k_slot.sK, (kv_rows, kv_cols_l)))
                k_r_blk = tl.load(tle.gpu.local_ptr(k_slot.sK, (kv_rows, kv_cols_r)))

                qk = tl.full([BH, BK], 0.0, dtype=tl.float32)
                qk = tl.dot(q_l_blk, tl.trans(k_l_blk), qk, out_dtype=tl.float32)
                qk = tl.dot(q_r_blk, tl.trans(k_r_blk), qk, out_dtype=tl.float32)
                if TD > 0:
                    q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                    k_t_blk = tl.load(tle.gpu.local_ptr(k_slot.sK_tail))
                    qk = tl.dot(q_tail_blk, tl.trans(k_t_blk), qk, out_dtype=tl.float32)

                valid_wait = valid_reader.wait(stage)
                valid_mask = tl.load(tle.gpu.local_ptr(valid_wait.slot.is_valid)) != 0
                qk = tl.where(valid_mask[None, :], qk, float("-inf"))
                valid_reader.release(stage)

                local_max = tl.maximum(max_prev, tl.max(qk, axis=1))
                alpha = tl.math.exp2((max_prev - local_max) * log_scale)
                prob = tl.math.exp2(qk * log_scale - local_max[:, None] * log_scale)
                sum_exp = sum_exp * alpha + tl.sum(prob, axis=1)
                acc_l = acc_l * alpha[:, None]
                acc_r = acc_r * alpha[:, None]
                prob_b = prob.to(OUT_DTYPE)
                acc_l = tl.dot(prob_b, k_l_blk, acc_l, out_dtype=tl.float32)
                acc_r = tl.dot(prob_b, k_r_blk, acc_r, out_dtype=tl.float32)
                k_reader.release(stage)
                max_prev = local_max

            lse = max_prev * log_scale * 0.6931471805599453 + tl.log(sum_exp)
            inv_sum = tl.fdiv(1.0, sum_exp)
            acc_l = acc_l * inv_sum[:, None]
            acc_r = acc_r * inv_sum[:, None]
            o_base = o_partial + pid_s * TOTAL_ROWS * DV2 + q_row * DV2
            lse_base = lse_partial + pid_s * TOTAL_ROWS + q_row
            tl.store(
                o_base + offs_h[:, None] * DV2 + offs_dv_l[None, :],
                acc_l.to(OUT_DTYPE),
                mask=active,
            )
            tl.store(
                o_base + offs_h[:, None] * DV2 + offs_dv_r[None, :],
                acc_r.to(OUT_DTYPE),
                mask=active,
            )
            tl.store(lse_base + offs_h, lse, mask=active)

    @triton.jit
    def _tle_persistent_splitkv_fwd(
        q_desc,
        tq_desc,
        kv,
        req_to_tokens,
        seq_lens,
        o_partial,
        lse_partial,
        sm_scale: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        DQK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SEQLEN_PAD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        NUM_SMS: tl.constexpr,
        total_tiles,
    ):
        _ = DQK
        DPH: tl.constexpr = DP // 2
        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sK_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        is_valid_smem = tle.gpu.alloc(
            [1, BK],
            dtype=tl.int8,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        if TD > 0:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            sK_tail_smem = tle.gpu.alloc(
                [1, BK, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="persist_sQ",
                readers=("wg0",),
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
            k_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="persist_sK",
                sK=sK_smem,
                sK_tail=sK_tail_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="persist_sQ",
                readers=("wg0",),
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )
            k_pipe = tle.pipe(capacity=1, scope="cta", name="persist_sK", sK=sK_smem)

        valid_pipe = tle.pipe(
            capacity=1, scope="cta", name="persist_valid", is_valid=is_valid_smem
        )
        log_scale: tl.constexpr = sm_scale * 1.4426950408889634
        tle.gpu.warp_specialize(
            [
                (
                    _tle_persistent_splitkv_consumer,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k_pipe.reader(),
                        valid_pipe.reader(),
                        o_partial,
                        lse_partial,
                        seq_lens,
                        log_scale,
                        TD,
                        kv.dtype.element_ty,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        B,
                        H,
                        BLOCKS_PER_SPLIT,
                        NUM_SPLITS,
                        NUM_SMS,
                        total_tiles,
                    ),
                ),
                (
                    _tle_persistent_splitkv_producer,
                    (
                        k_pipe.writer(),
                        valid_pipe.writer(),
                        kv,
                        req_to_tokens,
                        seq_lens,
                        DQK,
                        PAGE_SIZE,
                        DV,
                        TD,
                        TDP,
                        BK,
                        MAX_SEQLEN_PAD,
                        B,
                        BLOCKS_PER_SPLIT,
                        NUM_SPLITS,
                        NUM_SMS,
                        total_tiles,
                    ),
                ),
            ],
            [4],
            [72],
        )

    @triton.jit
    def _tle_splitkv_fwd(
        q_desc,
        tq_desc,
        kv,
        req_to_tokens,
        seq_lens,
        o_partial,
        lse_partial,
        sm_scale: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        DQK: tl.constexpr,
        D: tl.constexpr,
        DV: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_SEQLEN_PAD: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        BLOCKS_PER_SPLIT: tl.constexpr,
    ):
        _ = B
        _ = DQK
        DPH: tl.constexpr = DP // 2
        pid_h = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_s = tl.program_id(2)
        h_base = pid_h * BH
        q_row = pid_b * H + h_base
        req_base = req_to_tokens + pid_b * (MAX_SEQLEN_PAD // PAGE_SIZE)
        seq_len_ptr = seq_lens + pid_b

        k_start = pid_s * BLOCKS_PER_SPLIT
        seq_len = tl.load(seq_len_ptr)
        NK = tl.cdiv(seq_len, BK)
        k_end = k_start + BLOCKS_PER_SPLIT
        if k_end > NK:
            k_end = NK
        if k_start >= NK:
            return

        TOTAL_ROWS: tl.constexpr = B * H
        stride_o_split = TOTAL_ROWS * DV
        stride_lse_split = TOTAL_ROWS

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        sK_smem = tle.gpu.alloc(
            [1, BK, DP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
        )
        is_valid_smem = tle.gpu.alloc(
            [1, BK],
            dtype=tl.int8,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        if TD > 0:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            sK_tail_smem = tle.gpu.alloc(
                [1, BK, TDP], dtype=kv.dtype.element_ty, layout=None, scope=tle.gpu.smem
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="splitkv_sQ",
                readers=("wg0",),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
            k_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="splitkv_sK",
                sK=sK_smem,
                sK_tail=sK_tail_smem,
                is_valid=is_valid_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="splitkv_sQ",
                readers=("wg0",),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )
            k_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="splitkv_sK",
                sK=sK_smem,
                is_valid=is_valid_smem,
            )

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634
        tle.gpu.warp_specialize(
            [
                (
                    _tle_splitkv_consumer,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k_pipe.reader(),
                        o_partial,
                        lse_partial,
                        q_row,
                        seq_len_ptr,
                        log_scale,
                        TD,
                        kv.dtype.element_ty,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        k_start,
                        k_end,
                        pid_s,
                        stride_o_split,
                        stride_lse_split,
                    ),
                ),
                (
                    _tle_splitkv_producer,
                    (
                        k_pipe.writer(),
                        kv,
                        req_base,
                        seq_len_ptr,
                        DQK,
                        PAGE_SIZE,
                        DV,
                        TD,
                        TDP,
                        BK,
                        k_start,
                        k_end,
                    ),
                ),
            ],
            [4],
            [72],
        )

    @triton.jit
    def _tle_splitkv_combine(
        o_partial,
        lse_partial,
        output,
        NUM_SPLITS: tl.constexpr,
        BH: tl.constexpr,
        TOTAL_ROWS: tl.constexpr,
        DV: tl.constexpr,
        DPH: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row = pid * BH
        offs_h = tl.arange(0, BH)
        stride_split_o = TOTAL_ROWS * DV
        stride_split_lse = TOTAL_ROWS

        max_lse = tl.full([BH], -1.0e30, dtype=tl.float32)
        for s in tl.static_range(NUM_SPLITS):
            lse_s = tl.load(lse_partial + s * stride_split_lse + row + offs_h)
            max_lse = tl.maximum(max_lse, lse_s)

        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)
        sum_w = tl.zeros([BH], dtype=tl.float32)
        offs_dv_l = tl.arange(0, DPH)
        offs_dv_r = DPH + tl.arange(0, DPH)

        for s in tl.static_range(NUM_SPLITS):
            lse_s = tl.load(lse_partial + s * stride_split_lse + row + offs_h)
            w = tl.exp(lse_s - max_lse)
            sum_w += w
            o_base = o_partial + s * stride_split_o + (row + offs_h[:, None]) * DV
            o_l = tl.load(o_base + offs_dv_l[None, :])
            o_r = tl.load(o_base + offs_dv_r[None, :])
            acc_l += w[:, None] * o_l.to(tl.float32)
            acc_r += w[:, None] * o_r.to(tl.float32)

        inv_sum = tl.fdiv(1.0, sum_w)
        acc_l = acc_l * inv_sum[:, None]
        acc_r = acc_r * inv_sum[:, None]
        out_base = output + (row + offs_h[:, None]) * DV
        tl.store(out_base + offs_dv_l[None, :], acc_l.to(tl.bfloat16))
        tl.store(out_base + offs_dv_r[None, :], acc_r.to(tl.bfloat16))


def _flash_mla_tle_enabled() -> bool:
    value = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}


def _flash_mla_tle_variant() -> str:
    # Default to the auto-selected TLE implementation. Set
    # FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT=triton to force the original Triton path.
    return os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_TLE_VARIANT", "auto").lower()


def _flash_mla_auto_select_variant(batch_size: int, seqlen: int) -> str:
    """Auto-select best variant based on batch size and sequence length."""
    if batch_size >= 128 and seqlen >= 4096:
        return "none"
    if batch_size >= 17 and seqlen > 4352:
        return "none"
    if batch_size > 16:
        return "3wg"
    elif batch_size <= 16:
        return "splitkv_direct"  # direct global loads, no warp_specialize
    else:
        return "none"  # use original triton (splitkv_lite) kernel


def _flash_mla_tle_bh(variant=None) -> int:
    if variant is None:
        variant = _flash_mla_tle_variant()
    if variant == "splitkv":
        return TLE_FLASH_MLA_SPLITKV_BH
    if variant == "2wg":
        return TLE_FLASH_MLA_2WG_BH
    return TLE_FLASH_MLA_BH


def _flash_mla_splitkv_lite_split_size(batch_size: int, seqlen: int) -> int:
    def default_split_size() -> int:
        if batch_size >= 128:
            if seqlen <= 4352:
                return 48
            return 112
        if batch_size >= 32:
            return 112
        if batch_size >= 31:
            return 104
        if batch_size >= 27:
            return 96
        if batch_size >= 20:
            return 88
        if batch_size >= 17:
            return 64
        if batch_size <= 4:
            return 24
        if batch_size <= 8:
            return 32
        if batch_size <= 12:
            return 48
        return 64

    value = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_SPLITKV_LITE_SPLIT_SIZE")
    if value is None:
        return default_split_size()
    try:
        split_size = int(value)
    except ValueError:
        return default_split_size()
    return split_size if split_size > 0 else default_split_size()


def _flash_mla_splitkv_lite_combine_bh(batch_size: int) -> int:
    value = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_SPLITKV_LITE_COMBINE_BH")
    if value is None:
        return 8 if batch_size >= 17 else 64
    try:
        combine_bh = int(value)
    except ValueError:
        return 8 if batch_size >= 17 else 64
    return combine_bh if combine_bh > 0 else (8 if batch_size >= 17 else 64)


def _flash_mla_splitkv_lite_combine_warps() -> int:
    value = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_SPLITKV_LITE_COMBINE_WARPS")
    if value is None:
        return 4
    try:
        num_warps = int(value)
    except ValueError:
        return 4
    return num_warps if num_warps > 0 else 4


def _flash_mla_splitkv_lite_main_warps() -> int:
    value = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_SPLITKV_LITE_MAIN_WARPS")
    if value is None:
        return 8
    try:
        num_warps = int(value)
    except ValueError:
        return 8
    return num_warps if num_warps > 0 else 8


def _flash_mla_splitkv_lite_main_stages(batch_size: int, default_stages: int) -> int:
    value = os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_SPLITKV_LITE_MAIN_STAGES")
    if value is None:
        return 2 if batch_size >= 17 else default_stages
    try:
        num_stages = int(value)
    except ValueError:
        return 2 if batch_size >= 17 else default_stages
    return num_stages if num_stages > 0 else (2 if batch_size >= 17 else default_stages)


def _can_use_tle_flash_mla(
    q,
    blocked_k,
    block_table,
    cache_seqlens,
    block_size,
    h_q,
    h_kv,
    d,
    dv,
    variant=None,
):
    return (
        HAS_TLE_FLASH_MLA
        and _flash_mla_tle_enabled()
        and q.device.type == "cuda"
        and h_q % _flash_mla_tle_bh(variant) == 0
        and h_kv == 1
        and d in (512, 576)
        and dv == 512
        and block_size == TLE_FLASH_MLA_BK
        and blocked_k.is_contiguous()
        and block_table.is_contiguous()
        and cache_seqlens.is_contiguous()
    )


def _set_triton_descriptor_allocator(torch_dev: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=torch_dev)

    triton.set_allocator(alloc_fn)


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

    # Non-TLE split-KV path for small batch
    _effective_variant = _flash_mla_tle_variant()
    if _effective_variant == "auto":
        _effective_variant = _flash_mla_auto_select_variant(batch_size, max_seqlen_pad)
    force_original_triton = _effective_variant == "triton"
    # TLE async KV loads do not currently lower correctly through the two-dot attention update.
    # Keep the persistent experiment on long sequences, where it meets the accuracy threshold.
    use_tle_persistent = (
        _effective_variant == "persistent"
        and max_seqlen_pad >= 8192
        and _can_use_tle_flash_mla(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            block_size,
            h_q,
            h_kv,
            d,
            dv,
            _effective_variant,
        )
    )
    if _effective_variant == "persistent" and not use_tle_persistent:
        _effective_variant = _flash_mla_auto_select_variant(batch_size, max_seqlen_pad)
    can_use_tle = _can_use_tle_flash_mla(
        q,
        blocked_k,
        block_table,
        cache_seqlens,
        block_size,
        h_q,
        h_kv,
        d,
        dv,
        _effective_variant,
    )
    use_splitkv_lite = use_tle_persistent or (
        (
            batch_size <= 16
            or (batch_size >= 128 and max_seqlen_pad >= 4096)
            or (batch_size >= 17 and max_seqlen_pad > 4352)
        )
        and max_seqlen_pad >= 4096
        and not force_original_triton
        and (_effective_variant == "none" or not can_use_tle)
    )
    if (
        not use_splitkv_lite
        and os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_SPLITKV_LITE", "0") == "1"
    ):
        use_splitkv_lite = True

    if use_splitkv_lite:
        SPLIT_SIZE = _flash_mla_splitkv_lite_split_size(batch_size, max_seqlen_pad)
        max_nk = triton.cdiv(max_seqlen_pad, BLOCK_N)
        num_splits = triton.cdiv(max_nk, SPLIT_SIZE)
        o_partial = torch.empty(
            [num_splits, batch_size, head_num, dv],
            dtype=q.dtype,
            device=q.device,
        )
        lse_partial = torch.full(
            [num_splits, batch_size, head_num],
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        splitkv_grid = (
            triton.cdiv(head_num, BLOCK_H),
            batch_size,
            num_splits,
        )
        with torch_device_fn.device(device):
            main_kernel = (
                flash_mla_splitkv_persistent_tle_kernel
                if use_tle_persistent
                else flash_mla_splitkv_kernel
            )
            main_grid = (
                (torch.cuda.get_device_properties(q.device).multi_processor_count,)
                if use_tle_persistent
                else splitkv_grid
            )
            main_kernel[main_grid](
                q,
                blocked_k,
                block_table,
                cache_seqlens,
                o_partial,
                lse_partial,
                sm_scale,
                *(
                    (triton.cdiv(head_num, BLOCK_H) * batch_size * num_splits,)
                    if use_tle_persistent
                    else ()
                ),
                head_num,
                *((batch_size,) if use_tle_persistent else ()),
                q.stride(0),
                q.stride(1),
                blocked_k.stride(-2),
                block_table.stride(0),
                o_partial.stride(0),
                o_partial.stride(1),
                o_partial.stride(2),
                lse_partial.stride(0),
                lse_partial.stride(1),
                lse_partial.stride(2),
                BLOCK_H=BLOCK_H,
                BLOCK_N=BLOCK_N,
                PAGE_SIZE=block_size,
                HEAD_DIM_V=dv,
                HEAD_DIM=d,
                SPLIT_SIZE=SPLIT_SIZE,
                **(
                    {
                        "NUM_SPLITS": num_splits,
                        "NUM_SMS": main_grid[0],
                    }
                    if use_tle_persistent
                    else {}
                ),
                num_warps=_flash_mla_splitkv_lite_main_warps(),
                num_stages=_flash_mla_splitkv_lite_main_stages(batch_size, num_stages),
            )
            combine_bh = _flash_mla_splitkv_lite_combine_bh(batch_size)
            combine_grid = (
                triton.cdiv(head_num, combine_bh),
                batch_size,
            )
            flash_mla_splitkv_combine_kernel[combine_grid](
                o_partial,
                lse_partial,
                o,
                num_splits,
                head_num,
                o_partial.stride(0),
                o_partial.stride(1),
                o_partial.stride(2),
                lse_partial.stride(0),
                lse_partial.stride(1),
                lse_partial.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                BLOCK_H=combine_bh,
                HEAD_DIM_V=dv,
                NUM_SPLITS=num_splits,
                num_warps=_flash_mla_splitkv_lite_combine_warps(),
                num_stages=1,
            )
        return o.view([b, s_q, h_q, dv])

    if not force_original_triton and _effective_variant != "none" and can_use_tle:
        from triton.tools.tensor_descriptor import TensorDescriptor

        tle_bh = _flash_mla_tle_bh(_effective_variant)
        tle_grid = (
            triton.cdiv(head_num, tle_bh),
            batch_size,
        )
        DP = triton.next_power_of_2(dv)
        TD = d - dv
        TDP = triton.next_power_of_2(TD) if TD > 0 else 1

        _set_triton_descriptor_allocator(q.device)
        q_desc = TensorDescriptor(
            q,
            shape=[batch_size * head_num, d],
            strides=[d, 1],
            block_shape=[tle_bh, DP // 2],
        )
        if TD > 0:
            tq_desc = TensorDescriptor(
                q,
                shape=[batch_size * head_num, d],
                strides=[d, 1],
                block_shape=[tle_bh, TDP],
            )
        else:
            tq_desc = q_desc
        output_desc = TensorDescriptor(
            o,
            shape=[batch_size * head_num, dv],
            strides=[dv, 1],
            block_shape=[tle_bh, DP // 2],
        )
        with torch_device_fn.device(device):
            if _effective_variant in ("splitkv", "splitkv_direct"):
                BK = TLE_FLASH_MLA_BK
                splitkv_bh = TLE_FLASH_MLA_SPLITKV_BH
                max_nk = triton.cdiv(max_seqlen_pad, BK)
                if max_nk <= 128:
                    BLOCKS_PER_SPLIT = 8
                elif max_nk <= 384:
                    BLOCKS_PER_SPLIT = 16
                else:
                    BLOCKS_PER_SPLIT = 32
                num_splits = triton.cdiv(max_nk, BLOCKS_PER_SPLIT)
                total_rows = batch_size * head_num
                o_partial = torch.empty(
                    [num_splits, total_rows, dv],
                    dtype=q.dtype,
                    device=q.device,
                )
                lse_partial = torch.full(
                    [num_splits, total_rows],
                    float("-inf"),
                    dtype=torch.float32,
                    device=q.device,
                )
                splitkv_q_desc = TensorDescriptor(
                    q,
                    shape=[batch_size * head_num, d],
                    strides=[d, 1],
                    block_shape=[splitkv_bh, DP // 2],
                )
                if TD > 0:
                    splitkv_tq_desc = TensorDescriptor(
                        q,
                        shape=[batch_size * head_num, d],
                        strides=[d, 1],
                        block_shape=[splitkv_bh, TDP],
                    )
                else:
                    splitkv_tq_desc = splitkv_q_desc
                splitkv_grid = (
                    triton.cdiv(head_num, splitkv_bh),
                    batch_size,
                    num_splits,
                )
                if _effective_variant == "splitkv_direct":
                    _splitkv_direct_kernel[splitkv_grid](
                        q,
                        blocked_k,
                        block_table,
                        cache_seqlens,
                        o_partial,
                        lse_partial,
                        sm_scale,
                        q.stride(0),
                        q.stride(1),
                        d,  # stride_kv = d
                        batch_size,
                        head_num,
                        d,
                        dv,
                        TD,
                        DP,
                        TDP,
                        block_size,
                        max_seqlen_pad,
                        BK,
                        splitkv_bh,
                        BLOCKS_PER_SPLIT,
                        num_warps=TLE_FLASH_MLA_WORKER_NUM_WARPS,
                        num_stages=1,
                    )
                else:
                    _tle_splitkv_fwd[splitkv_grid](
                        splitkv_q_desc,
                        splitkv_tq_desc,
                        blocked_k,
                        block_table,
                        cache_seqlens,
                        o_partial,
                        lse_partial,
                        sm_scale,
                        batch_size,
                        head_num,
                        d,
                        d,
                        dv,
                        TD,
                        DP,
                        TDP,
                        block_size,
                        max_seqlen_pad,
                        BK,
                        splitkv_bh,
                        BLOCKS_PER_SPLIT,
                        num_warps=TLE_FLASH_MLA_WORKER_NUM_WARPS,
                        num_stages=1,
                    )
                combine_grid = (triton.cdiv(total_rows, splitkv_bh),)
                _tle_splitkv_combine[combine_grid](
                    o_partial,
                    lse_partial,
                    o,
                    num_splits,
                    splitkv_bh,
                    total_rows,
                    dv,
                    DP // 2,
                    num_warps=4,
                    num_stages=1,
                )
            elif _effective_variant == "3wg":
                _tle_flash_mla_dense_fwd[tle_grid](
                    q_desc,
                    tq_desc,
                    output_desc,
                    blocked_k,
                    block_table,
                    cache_seqlens,
                    sm_scale,
                    batch_size,
                    head_num,
                    d,
                    dv,
                    TD,
                    DP,
                    TDP,
                    block_size,
                    max_seqlen_pad,
                    TLE_FLASH_MLA_BK,
                    tle_bh,
                    TLE_FLASH_MLA_PAIR_BLOCKS,
                    num_warps=TLE_FLASH_MLA_WORKER_NUM_WARPS,
                    num_stages=1,
                )
            else:
                _tle_flash_mla_dense_2wg_fwd[tle_grid](
                    q_desc,
                    tq_desc,
                    output_desc,
                    q,
                    blocked_k,
                    block_table,
                    cache_seqlens,
                    sm_scale,
                    batch_size,
                    head_num,
                    d,
                    dv,
                    TD,
                    DP,
                    TDP,
                    block_size,
                    max_seqlen_pad,
                    TLE_FLASH_MLA_BK,
                    tle_bh,
                    num_warps=TLE_FLASH_MLA_WORKER_NUM_WARPS,
                    num_stages=1,
                )
        return o.view([b, s_q, h_q, dv])

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
