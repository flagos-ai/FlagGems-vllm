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

"""
This backend is specialized for chunk size 64 and K=V=128. The source
preprocessing kernel does not expose the KKT inverse ``A`` expected by the
generic FLA seven-value protocol, so the third result is intentionally
``None`` rather than an unrelated intermediate.
"""

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.ops.FLA.utils import SUPPRESS_LEVEL


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit
def extract_slice(ful, offsets, sizes, strides):
    return tl.extract_slice(ful, offsets, sizes, strides)


@triton.jit
def insert_slice(ful, sub, offsets, sizes, strides):
    return tl.insert_slice(ful, sub, offsets, sizes, strides)


MAX_PROGRAMS_PER_LAUNCH = 8192
SINGLE_KERNEL_MAX_TOTAL_CHUNKS = 13
LLMINFER_GDR_DEFAULT_NUM_AIC = 24


def _get_aic_num() -> int:
    try:
        from triton.runtime.driver import driver

        num_aicore = int(driver.active.utils.get_aicore_num())
        if num_aicore > 0:
            return num_aicore
    except Exception:
        return LLMINFER_GDR_DEFAULT_NUM_AIC
    return LLMINFER_GDR_DEFAULT_NUM_AIC


@triton.jit
def _map_flattened_varlen_block(
    block_pid,
    cu_seqlens,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    block_base = 0
    selected_bos = 0
    selected_t = 0
    selected_block = 0
    for i_n in range(0, N):
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        t_local = eos - bos
        blocks_local = (t_local + BLOCK_SIZE - 1) // BLOCK_SIZE
        selected = (block_pid >= block_base) & (block_pid < block_base + blocks_local)
        selected_bos = tl.where(selected, bos, selected_bos)
        selected_t = tl.where(selected, t_local, selected_t)
        selected_block = tl.where(
            selected,
            block_pid - block_base,
            selected_block,
        )
        block_base += blocks_local
    return block_pid < block_base, selected_bos, selected_t, selected_block


# -----------------------------------------------------------------------------
# Stage 0: chunk-local cumsum
# -----------------------------------------------------------------------------
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    N,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,
):
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    N_CHUNKS: tl.constexpr = BLOCK_T // CHUNK_SIZE

    if IS_VARLEN:
        valid, bos, T, i_block = _map_flattened_varlen_block(
            i_block,
            cu_seqlens,
            N,
            BLOCK_T,
        )
    else:
        valid, bos = True, i_b * T

    if valid:
        if HEAD_FIRST:
            ptr_s = tl.make_block_ptr(
                s + bos * H,
                (H, T),
                (T, 1),
                (0, i_block * BLOCK_T),
                (H, BLOCK_T),
                (1, 0),
            )
            ptr_o = tl.make_block_ptr(
                o + bos * H,
                (H, T),
                (T, 1),
                (0, i_block * BLOCK_T),
                (H, BLOCK_T),
                (1, 0),
            )
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
            b_s = tl.trans(b_s, (2, 0, 1))
            b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
            b_o = tl.trans(b_o, (2, 0, 1))
            b_o = tl.reshape(b_o, (H, BLOCK_T))
        else:
            ptr_s = tl.make_block_ptr(
                s + bos * H,
                (T, H),
                (H, 1),
                (i_block * BLOCK_T, 0),
                (BLOCK_T, H),
                (1, 0),
            )
            ptr_o = tl.make_block_ptr(
                o + bos * H,
                (T, H),
                (H, 1),
                (i_block * BLOCK_T, 0),
                (BLOCK_T, H),
                (1, 0),
            )
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
            b_s = tl.trans(b_s, (1, 0, 2))
            b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
            b_o = tl.trans(b_o, (1, 0, 2))
            b_o = tl.reshape(b_o, (BLOCK_T, H))

        tl.store(ptr_o, b_o.to(o.dtype.element_ty), boundary_check=(0,))
    return


def chunk_local_cumsum_scalar(
    g,
    chunk_size,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    block_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.Tensor | None = torch.float,
):
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"
    OPTIM_BLOCK_SIZE = triton.next_power_of_2((2**18) // (H * chunk_size))
    N = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else B
    num_blocks = (
        triton.cdiv(T, OPTIM_BLOCK_SIZE) + N - 1
        if cu_seqlens is not None
        else triton.cdiv(T, OPTIM_BLOCK_SIZE)
    )
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (num_blocks, B)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        N=N,
        H=H,
        BLOCK_T=OPTIM_BLOCK_SIZE,
        CHUNK_SIZE=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        num_warps=8,
        num_stages=3,
    )
    return g


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            block_indices=kwargs.get("block_indices"),
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )


# -----------------------------------------------------------------------------
# Stage 1: scaled KKT
# -----------------------------------------------------------------------------
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,  # [H, B, T]
    g_cumsum,  # [H, B, T]
    A,
    cu_seqlens,
    T,
    B,
    N,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    bt_stride = B * T
    i_t_i, _ = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        valid, bos, T, i_t = _map_flattened_varlen_block(
            i_t_i,
            cu_seqlens,
            N,
            BT,
        )
    else:
        valid, bos, i_t = True, 0, i_t_i

    if valid:
        for i_bh in range(B * H):
            i_b, i_h = i_bh // H, i_bh % H
            if not IS_VARLEN:
                bos = i_b * T
            o_t = tl.arange(0, BT)
            o_t_fp32 = o_t.to(tl.float32)

            p_beta = tl.make_block_ptr(
                beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,)
            )
            b_beta = tl.load(p_beta, boundary_check=(0,))

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k + (bos * Hg + i_h // (H // Hg)) * K,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_A += tl.dot(b_k, tl.trans(b_k))

            if USE_G:
                p_g = tl.make_block_ptr(
                    g_cumsum + i_h * bt_stride + bos,
                    (T,),
                    (1,),
                    (i_t * BT,),
                    (BT,),
                    (0,),
                )
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_diff = b_g[:, None] - b_g[None, :]
                b_A *= safe_exp(b_g_diff)

            b_A *= b_beta[:, None]
            b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
            p_A = tl.make_block_ptr(
                A + (bos * H + i_h) * BT,
                (T, BT),
                (BT * H, 1),
                (i_t * BT, 0),
                (BT, BT),
                (1, 0),
            )
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


# -----------------------------------------------------------------------------
# Stage 2-3: triangular inverse
# -----------------------------------------------------------------------------
@triton.jit
def _solve_tril_16x16_one_block(
    A,
    Ad,
    bos,
    T,
    i_t,
    i_h,
    H,
    BT: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    base_t = i_t * LARGE_BLOCK_T

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS

    for taskid in range(0, NTASKS):
        base_t += taskid * (LARGE_BLOCK_T // NTASKS)

        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)
        for blkid in range(0, N_BLOCKS):
            row_start_o = base_t + blkid * 16
            col_start_o = row_start_o % BT

            # 1 Create in-block offset
            offs_rows_in_block = tl.arange(0, 16)
            offs_cols_in_block = tl.arange(0, 16)

            # 2 Calculate the pointer of each element
            ptr_A_subrec16 = (
                A
                + row_start_o * H * BT
                + col_start_o
                + offs_rows_in_block[:, None] * H * BT
                + offs_cols_in_block[None, :]
            )

            # 3 Create a mask to prevent out-of-bounds access
            global_rows = row_start_o + offs_rows_in_block[:, None]
            global_cols = col_start_o + offs_cols_in_block[None, :]
            load_mask = (global_rows < T) & (global_cols < BT)

            # 4 Use mask to safely load data
            b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(
                tl.float32
            )
            b_A = insert_slice(
                ful=b_A,
                sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
                offsets=[blkid, 0, 0],
                sizes=[1, 16, 16],
                strides=[1, 1, 1],
            )

        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))

        # Convert mask into matrix multiplication to avoid for loops ub oom
        tmp = tl.arange(0, 16).to(tl.float32)
        rows = tmp[:, None]
        cols = tmp[None, :]
        is_lower = (rows > cols).to(b_A.dtype)
        b_A = -b_A * is_lower

        # for loop to update N_BLOCKS row vector
        for i in range(1, 16):
            nblks_vec16 = -extract_slice(
                local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1)
            )
            b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

            dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
            dot_product = tl.sum(dot_tmp, 0)
            b_a = b_a + dot_product

            b_a_new_expanded = b_a[:, None, :]
            b_A = insert_slice(
                ful=b_A,
                sub=b_a_new_expanded,
                offsets=[0, i, 0],
                sizes=[N_BLOCKS, 1, 16],
                strides=[1, 1, 1],
            )

        on_diagonal = rows == cols
        b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

        b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))

        # 1 Create in-block offset
        offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
        offs_cols_to_store = tl.arange(0, 16)

        # 2 Calculate the pointer of each element
        p_Ai = (
            Ad
            + base_t * H * 16
            + 0
            + offs_rows_to_store[:, None] * H * 16
            + offs_cols_to_store[None, :]
        )
        # 3 Create a mask to prevent out-of-bounds access, only check rows
        global_store_rows = base_t + offs_rows_to_store[:, None]
        store_mask = global_store_rows < T
        # 4 use mask to save data safely
        tl.store(
            p_Ai,
            b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=store_mask,
        )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "H", "N"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    T,
    H,
    N,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        valid, bos, T, i_t = _map_flattened_varlen_block(
            i_t,
            cu_seqlens,
            N,
            LARGE_BLOCK_T,
        )
    else:
        valid, bos = True, i_b * T

    if valid:
        _solve_tril_16x16_one_block(
            A,
            Ad,
            bos,
            T,
            i_t,
            i_h,
            H,
            BT,
            LARGE_BLOCK_T,
        )


@triton.jit(do_not_specialize=["T", "H"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    T,
    H,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(
        A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ad_11 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ad_22 = tl.make_block_ptr(
        Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )
    p_Ai_11 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0)
    )
    p_Ai_22 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0)
    )
    p_Ai_21 = tl.make_block_ptr(
        Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0)
    )

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"),
        Ai_11,
        input_precision="ieee",
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.jit(do_not_specialize=["T", "H"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    T,
    H,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    # Base pointers (already offset by batch and head)
    A += (bos * H + i_h) * 64
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 64

    # load Ai_22 (Ad block at row i_t * 64 + 16, col 0, 16 * 16)
    offs_m = i_t * 64 + 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_22 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # load A_21 (A block at row i_t * 64 + 16, col 0, 16 * 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

    # load Ai_11 (Ad block at row i_t * 64, col 0, 16 * 16)
    offs_m = i_t * 64 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_11 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

    # load Ai_44 (Ad block at row i_t * 64 + 48, col 0, 16 * 16)
    offs_m = i_t * 64 + 48 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_44 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    # load A_43 (Ad block at row i_t * 64 + 48, col 32, 16 * 16)
    offs_n = 32 + tl.arange(0, 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

    # load Ai_33 (Ad block at row i_t * 64 + 32, col 0, 16 * 16)
    offs_m = i_t * 64 + 32 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    mask_Ad = (offs_m[:, None] < T) & (offs_n[None, :] < 16)
    ptr_Ad = Ad + offs_m[:, None] * (H * 16) + offs_n[None, :]
    Ai_33 = tl.load(ptr_Ad, mask=mask_Ad, other=0.0).to(tl.float32)

    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

    # build Ai_22_32 (32 * 32)
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    # load A_21_32 (A block at row i_t * 64 + 32, col 0, 32 * 32)
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + offs_m[:, None] * (H * 64) + offs_n[None, :]
    A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

    # build Ai_11_32 (32 * 32)
    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

    # store Ai_11_32 to (i_t * 64, 0)
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_11_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # store Ai_22_32 to (i_t * 64 + 32, 32)
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_22_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # store Ai_21_32 to (i_t * 64 + 32, 32)
    offs_n = tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_Ai = Ai + offs_m[:, None] * (H * 64) + offs_n[None, :]
    tl.store(
        ptr_Ai,
        Ai_21_32.to(ptr_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_store,
    )

    # zero out the upper-right 32 * 32 block (rows 0 ~ 31, cols 32 ~ 63)
    offs_m = i_t * 64 + tl.arange(0, 32)
    offs_n = 32 + tl.arange(0, 32)
    mask_store = (offs_m[:, None] < T) & (offs_n[None, :] < BT)
    ptr_Ai = Ai + offs_m[:, None] * (H * BT) + offs_n[None, :]
    zero_block = tl.zeros((32, 32), dtype=ptr_Ai.dtype.element_ty)
    tl.store(ptr_Ai, zero_block, mask=mask_store)


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape

    Ad = torch.empty(
        B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype
    )

    LARGE_BLOCK_T = 608 * 2

    N = B
    NT = triton.cdiv(T, LARGE_BLOCK_T)

    solve_tril_16x16_kernel[NT, B * H](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        N=N,
        BT=BT,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        num_warps=1,
        num_stages=4,
    )

    if BT == 16:
        return Ad

    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = (
        merge_16x16_to_32x32_inverse_kernel
        if BT == 32
        else merge_16x16_to_64x64_inverse_kernel
    )
    NT = triton.cdiv(T, BT)

    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        T=T,
        H=H,
        BT=BT,
        num_warps=4,
        num_stages=3,
    )
    return Ai


# -----------------------------------------------------------------------------
# Stage 2-3: pair-merge triangular inverse
# -----------------------------------------------------------------------------
@triton.jit
def _load_ad16(ad, t_local, row, H):
    ptr = tl.make_block_ptr(
        ad,
        (t_local, 16),
        (H * 16, 1),
        (row, 0),
        (16, 16),
        (1, 0),
    )
    return tl.load(ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)


@triton.jit
def _load_a16(a, t_local, row, col, H, BT: tl.constexpr):
    ptr = tl.make_block_ptr(
        a,
        (t_local, BT),
        (H * BT, 1),
        (row, col),
        (16, 16),
        (1, 0),
    )
    return tl.load(ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)


@triton.jit
def _load_a32(a, t_local, row, H, BT: tl.constexpr):
    ptr = tl.make_block_ptr(
        a,
        (t_local, BT),
        (H * BT, 1),
        (row, 0),
        (32, 32),
        (1, 0),
    )
    return tl.load(ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)


@triton.jit
def _block_diag_16(x0, x1):
    result = tl.zeros((32, 32), dtype=tl.float32)
    result = insert_slice(result, x0, (0, 0), (16, 16), (1, 1))
    result = insert_slice(result, x1, (16, 16), (16, 16), (1, 1))
    return result


@triton.jit
def _block_diag_32(x0, x1):
    result = tl.zeros((64, 64), dtype=tl.float32)
    result = insert_slice(result, x0, (0, 0), (32, 32), (1, 1))
    result = insert_slice(result, x1, (32, 32), (32, 32), (1, 1))
    return result


@triton.jit
def _assemble_32(d0, d1, offdiag):
    result = tl.zeros((32, 32), dtype=tl.float32)
    result = insert_slice(result, d0, (0, 0), (16, 16), (1, 1))
    result = insert_slice(result, d1, (16, 16), (16, 16), (1, 1))
    result = insert_slice(result, offdiag, (16, 0), (16, 16), (1, 1))
    return result


@triton.jit
def _assemble_64(ai11, ai22, ai21):
    result = tl.zeros((64, 64), dtype=tl.float32)
    result = insert_slice(result, ai11, (0, 0), (32, 32), (1, 1))
    result = insert_slice(result, ai22, (32, 32), (32, 32), (1, 1))
    result = insert_slice(result, ai21, (32, 0), (32, 32), (1, 1))
    return result


@triton.jit
def _merge_pair_16x16_to_64x64_inverse_one(
    A,
    Ad,
    Ai,
    bos,
    t_local,
    i_pair,
    i_h,
    H,
    BT: tl.constexpr,
):
    A += (bos * H + i_h) * BT
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * BT
    row0 = i_pair * (2 * BT)
    row1 = row0 + BT

    d0 = _load_ad16(Ad, t_local, row0, H)
    d1 = _load_ad16(Ad, t_local, row0 + 16, H)
    d4 = _load_ad16(Ad, t_local, row1, H)
    d5 = _load_ad16(Ad, t_local, row1 + 16, H)
    a10_0 = _load_a16(A, t_local, row0 + 16, 0, H, BT)
    a10_1 = _load_a16(A, t_local, row1 + 16, 0, H, BT)
    ai10_pair = -tl.dot(
        tl.dot(
            _block_diag_16(d1, d5),
            _block_diag_16(a10_0, a10_1),
            allow_tf32=False,
        ),
        _block_diag_16(d0, d4),
        allow_tf32=False,
    )
    ai11_0 = _assemble_32(
        d0,
        d1,
        extract_slice(ai10_pair, (0, 0), (16, 16), (1, 1)),
    )
    ai11_1 = _assemble_32(
        d4,
        d5,
        extract_slice(ai10_pair, (16, 16), (16, 16), (1, 1)),
    )

    d2 = _load_ad16(Ad, t_local, row0 + 32, H)
    d3 = _load_ad16(Ad, t_local, row0 + 48, H)
    d6 = _load_ad16(Ad, t_local, row1 + 32, H)
    d7 = _load_ad16(Ad, t_local, row1 + 48, H)
    a32_0 = _load_a16(A, t_local, row0 + 48, 32, H, BT)
    a32_1 = _load_a16(A, t_local, row1 + 48, 32, H, BT)
    ai32_pair = -tl.dot(
        tl.dot(
            _block_diag_16(d3, d7),
            _block_diag_16(a32_0, a32_1),
            allow_tf32=False,
        ),
        _block_diag_16(d2, d6),
        allow_tf32=False,
    )
    ai22_0 = _assemble_32(
        d2,
        d3,
        extract_slice(ai32_pair, (0, 0), (16, 16), (1, 1)),
    )
    ai22_1 = _assemble_32(
        d6,
        d7,
        extract_slice(ai32_pair, (16, 16), (16, 16), (1, 1)),
    )

    a21_0 = _load_a32(A, t_local, row0 + 32, H, BT)
    a21_1 = _load_a32(A, t_local, row1 + 32, H, BT)
    ai21_pair = -tl.dot(
        tl.dot(
            _block_diag_32(ai22_0, ai22_1),
            _block_diag_32(a21_0, a21_1),
            allow_tf32=False,
        ),
        _block_diag_32(ai11_0, ai11_1),
        allow_tf32=False,
    )

    result0 = _assemble_64(
        ai11_0,
        ai22_0,
        extract_slice(ai21_pair, (0, 0), (32, 32), (1, 1)),
    )
    result1 = _assemble_64(
        ai11_1,
        ai22_1,
        extract_slice(ai21_pair, (32, 32), (32, 32), (1, 1)),
    )
    out0 = tl.make_block_ptr(
        Ai,
        (t_local, BT),
        (H * BT, 1),
        (row0, 0),
        (BT, BT),
        (1, 0),
    )
    out1 = tl.make_block_ptr(
        Ai,
        (t_local, BT),
        (H * BT, 1),
        (row1, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(
        out0,
        result0.to(out0.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        out1,
        result1.to(out1.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "H", "N"])
def merge_pair_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    T,
    H,
    N,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pair_pid = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        valid, bos, t_local, i_pair = _map_flattened_varlen_block(
            pair_pid,
            cu_seqlens,
            N,
            2 * BT,
        )
    else:
        valid, bos, t_local, i_pair = True, i_b * T, T, pair_pid

    if valid:
        _merge_pair_16x16_to_64x64_inverse_one(
            A,
            Ad,
            Ai,
            bos,
            t_local,
            i_pair,
            i_h,
            H,
            BT,
        )


def solve_tril_pair_merge(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    B, T, H, BT = A.shape
    assert BT == 64
    ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float32)
    large_block_t = 608 * 2
    N = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else B
    num_large_blocks = (
        triton.cdiv(T, large_block_t) + N - 1
        if cu_seqlens is not None
        else triton.cdiv(T, large_block_t)
    )
    solve_tril_16x16_kernel[(num_large_blocks, B * H)](
        A=A,
        Ad=ad,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        N=N,
        BT=BT,
        LARGE_BLOCK_T=large_block_t,
        num_warps=1,
        num_stages=4,
    )

    num_pairs = (
        triton.cdiv(T, 2 * BT) + N - 1
        if cu_seqlens is not None
        else triton.cdiv(T, 2 * BT)
    )
    result = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_pair_16x16_to_64x64_inverse_kernel[(num_pairs, B * H)](
        A,
        ad,
        result,
        cu_seqlens,
        T,
        H,
        N,
        BT=BT,
        num_warps=8,
        num_stages=1,
    )
    return result


# -----------------------------------------------------------------------------
# Stage 4-5: combined W/U recompute
# -----------------------------------------------------------------------------
@triton.jit
def _load_chunk_pair(A_inv, bos, t_local, row_start, i_h, H, BT: tl.constexpr):
    a0 = tl.make_block_ptr(
        A_inv + (bos * H + i_h) * BT,
        (t_local, BT),
        (H * BT, 1),
        (row_start, 0),
        (BT, BT),
        (1, 0),
    )
    block0 = tl.load(a0, boundary_check=(0, 1), padding_option="zero")
    block1 = tl.zeros([BT, BT], dtype=A_inv.dtype.element_ty)
    if row_start + BT < t_local:
        a1 = tl.make_block_ptr(
            A_inv + (bos * H + i_h) * BT,
            (t_local, BT),
            (H * BT, 1),
            (row_start + BT, 0),
            (BT, BT),
            (1, 0),
        )
        block1 = tl.load(a1, boundary_check=(0, 1), padding_option="zero")
    pair = tl.zeros([2 * BT, 2 * BT], dtype=A_inv.dtype.element_ty)
    pair = insert_slice(pair, block0, [0, 0], [BT, BT], [1, 1])
    pair = insert_slice(pair, block1, [BT, BT], [BT, BT], [1, 1])
    return pair


@triton.jit
def _stage4_uw_one_pair(
    k,
    v,
    beta,
    g_cumsum,
    A_inv,
    w,
    u,
    bos,
    t_local,
    i_pair,
    i_h,
    head_stride,
    H,
    Hg,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row_start = i_pair * BM
    rows = row_start + tl.arange(0, BM)
    row_mask = rows < t_local
    a_pair = _load_chunk_pair(A_inv, bos, t_local, row_start, i_h, H, BT)
    i_hg = i_h // (H // Hg)

    # Column-blocked two-phase form. Keep U and W in two source-level
    # lifetime regions: gate and K are not loaded until the U blocks have
    # been written back, so a [BM, BV] v tile and a [BM, BK] k tile never
    # coexist in UB. BK/BV <= 128 bounds the footprint independently of
    # K/V, which is what allows K or V beyond 128.
    beta_u = tl.load(
        beta + i_h * head_stride + bos + rows,
        mask=row_mask,
        other=0.0,
    ).to(v.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (t_local, V),
            (H * V, 1),
            (row_start, i_v * BV),
            (BM, BV),
            (1, 0),
        )
        v_tile = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.where(row_mask[:, None], v_tile, 0.0)
        u_tile = tl.dot(
            a_pair,
            (v_tile * beta_u[:, None]).to(v.dtype.element_ty),
            allow_tf32=False,
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (t_local, V),
            (H * V, 1),
            (row_start, i_v * BV),
            (BM, BV),
            (1, 0),
        )
        tl.store(p_u, u_tile.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    beta_w = tl.load(
        beta + i_h * head_stride + bos + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        g_cumsum + i_h * head_stride + bos + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    scale = (beta_w * tl.exp(gate)).to(k.dtype.element_ty)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_hg) * K,
            (t_local, K),
            (Hg * K, 1),
            (row_start, i_k * BK),
            (BM, BK),
            (1, 0),
        )
        k_tile = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        k_tile = tl.where(row_mask[:, None], k_tile, 0.0)
        w_tile = tl.dot(
            a_pair,
            (k_tile * scale[:, None]).to(k.dtype.element_ty),
            allow_tf32=False,
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (t_local, K),
            (H * K, 1),
            (row_start, i_k * BK),
            (BM, BK),
            (1, 0),
        )
        tl.store(p_w, w_tile.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_PAIR_INDICES": lambda args: args["pair_indices"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B", "H", "Hg", "N", "total_pairs"])
def stage4_uw_pair_persistent_kernel(
    k,
    v,
    beta,
    g_cumsum,
    A_inv,
    w,
    u,
    cu_seqlens,
    pair_indices,
    total_pairs,
    T,
    B,
    H,
    Hg,
    K: tl.constexpr,
    V: tl.constexpr,
    N,
    BT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_PAIR_INDICES: tl.constexpr,
):
    worker_id = tl.program_id(0)
    head_stride = B * T
    total_tasks = total_pairs * H
    for task_id in range(worker_id, total_tasks, NUM_WORKERS):
        pair_pid = task_id // H
        i_h = task_id % H
        if IS_VARLEN:
            if USE_PAIR_INDICES:
                i_n = tl.load(pair_indices + pair_pid * 2).to(tl.int32)
                i_pair = tl.load(pair_indices + pair_pid * 2 + 1).to(tl.int32)
                bos = tl.load(cu_seqlens + i_n).to(tl.int32)
                eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                _stage4_uw_one_pair(
                    k,
                    v,
                    beta,
                    g_cumsum,
                    A_inv,
                    w,
                    u,
                    bos,
                    eos - bos,
                    i_pair,
                    i_h,
                    head_stride,
                    H,
                    Hg,
                    K,
                    V,
                    BT,
                    BM,
                    BK,
                    BV,
                )
            else:
                pair_base = 0
                selected_bos = 0
                selected_t = 0
                selected_pair = 0
                for i_n in range(0, N):
                    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
                    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
                    t_local = eos - bos
                    pairs_local = (t_local + BM - 1) // BM
                    selected = (pair_pid >= pair_base) & (
                        pair_pid < pair_base + pairs_local
                    )
                    selected_bos = tl.where(selected, bos, selected_bos)
                    selected_t = tl.where(selected, t_local, selected_t)
                    selected_pair = tl.where(
                        selected, pair_pid - pair_base, selected_pair
                    )
                    pair_base += pairs_local
                if pair_pid < pair_base:
                    _stage4_uw_one_pair(
                        k,
                        v,
                        beta,
                        g_cumsum,
                        A_inv,
                        w,
                        u,
                        selected_bos,
                        selected_t,
                        selected_pair,
                        i_h,
                        head_stride,
                        H,
                        Hg,
                        K,
                        V,
                        BT,
                        BM,
                        BK,
                        BV,
                    )
        else:
            pairs_per_batch = (T + BM - 1) // BM
            i_b = pair_pid // pairs_per_batch
            i_pair = pair_pid % pairs_per_batch
            _stage4_uw_one_pair(
                k,
                v,
                beta,
                g_cumsum,
                A_inv,
                w,
                u,
                i_b * T,
                T,
                i_pair,
                i_h,
                head_stride,
                H,
                Hg,
                K,
                V,
                BT,
                BM,
                BK,
                BV,
            )


def stage4_chunk_pair_combined128(
    k: torch.Tensor,
    v: torch.Tensor,
    beta_permuted: torch.Tensor,
    g_cumsum_permuted: torch.Tensor,
    A_inv: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    pair_indices: torch.Tensor | None = None,
    *,
    num_cores: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = k.shape
    H, V = v.shape[-2:]
    BT = A_inv.shape[-1]
    BM = 2 * BT
    assert BT == 64
    N = B if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    if cu_seqlens is not None:
        assert B == 1
        if pair_indices is not None:
            assert pair_indices.ndim == 2 and pair_indices.shape[1] == 2
    else:
        assert pair_indices is None
    if pair_indices is not None:
        total_pairs = pair_indices.shape[0]
    elif cu_seqlens is None:
        total_pairs = N * triton.cdiv(T, BM)
    else:
        total_pairs = triton.cdiv(T, BM) + N - 1
    if num_cores is None:
        num_cores = _get_aic_num()
    num_workers = min(num_cores, total_pairs * H)
    # Column blocking alone: each dot stays [BM, <=128] wide, so the UB
    # footprint is bounded no matter how large K/V grow.
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)
    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)
    stage4_uw_pair_persistent_kernel[(num_workers,)](
        k,
        v,
        beta_permuted,
        g_cumsum_permuted,
        A_inv,
        w,
        u,
        cu_seqlens,
        pair_indices,
        total_pairs,
        T,
        B,
        H,
        Hg,
        K,
        V,
        N,
        BT=BT,
        BM=BM,
        BK=BK,
        BV=BV,
        NUM_WORKERS=num_workers,
        num_warps=8,
        num_stages=1,
    )
    return w, u


@triton.jit
def _transpose_gate_beta_kernel(
    g,
    beta,
    g_transposed,
    beta_transposed,
    T,
    B,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_block = tl.program_id(0)
    batch_idx = tl.program_id(1)
    token_start = token_block * BLOCK_T

    input_g = tl.make_block_ptr(
        g + batch_idx * T * H,
        (T, H),
        (H, 1),
        (token_start, 0),
        (BLOCK_T, H),
        (1, 0),
    )
    input_beta = tl.make_block_ptr(
        beta + batch_idx * T * H,
        (T, H),
        (H, 1),
        (token_start, 0),
        (BLOCK_T, H),
        (1, 0),
    )
    output_g = tl.make_block_ptr(
        g_transposed + batch_idx * T,
        (H, T),
        (B * T, 1),
        (0, token_start),
        (H, BLOCK_T),
        (1, 0),
    )
    output_beta = tl.make_block_ptr(
        beta_transposed + batch_idx * T,
        (H, T),
        (B * T, 1),
        (0, token_start),
        (H, BLOCK_T),
        (1, 0),
    )

    gate_tile = tl.load(input_g, boundary_check=(0,))
    beta_tile = tl.load(input_beta, boundary_check=(0,))
    tl.store(output_g, tl.trans(gate_tile), boundary_check=(1,))
    tl.store(output_beta, tl.trans(beta_tile), boundary_check=(1,))


@triton.jit
def _transpose_cumsum_output_kernel(
    g_transposed,
    g,
    T,
    B,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    token_block = tl.program_id(0)
    batch_idx = tl.program_id(1)
    token_start = token_block * BLOCK_T

    input_g = tl.make_block_ptr(
        g_transposed + batch_idx * T,
        (H, T),
        (B * T, 1),
        (0, token_start),
        (H, BLOCK_T),
        (1, 0),
    )
    output_g = tl.make_block_ptr(
        g + batch_idx * T * H,
        (T, H),
        (H, 1),
        (token_start, 0),
        (BLOCK_T, H),
        (1, 0),
    )
    gate_tile = tl.load(input_g, boundary_check=(1,))
    tl.store(output_g, tl.trans(gate_tile), boundary_check=(0,))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g"] is not None,
        # 是否把 Stage 1 的 chunk 内 cumsum 结果额外落盘（供下游复用，替代独立的
        # chunk_local_cumsum kernel）。
        "STORE_G_CUMSUM": lambda args: args["g_cumsum_out"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B", "H", "Hg", "K", "V", "N", "chunk_pid_offset"])
def chunk_gated_delta_rule_fwd_stage0_4_kernel(
    k,  # [B, T, Hg, K]
    v,  # [B, T, H,  V]
    g,  # [H, B, T]      raw log-gate (permuted, contiguous T axis)
    beta,  # [H, B, T]      permuted, contiguous T axis
    w,  # [B, T, H,  K]  out
    u,  # [B, T, H,  V]  out
    g_cumsum_out,  # [H, B, T]  out (permuted, contiguous T)；chunk 内 cumsum 后的 g
    cu_seqlens,
    chunk_pid_offset,
    T,
    B,
    H,
    Hg,
    K,
    V,
    N,  # 变长: 序列条数 (= len(cu_seqlens)-1)；定长: batch 数 B
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    JOINT_UW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    STORE_G_CUMSUM: tl.constexpr,
):
    chunk_pid = tl.program_id(0) + chunk_pid_offset
    i_h = tl.program_id(1)

    # head stride in the permuted [H, B, T] gate/beta buffers. MUST be computed
    # with the *total* T (the size of the packed T axis) and must NOT be
    # clobbered by the per-segment length inside the chunk loop below.
    bt_stride = B * T

    i_hg = i_h // (H // Hg)
    if IS_VARLEN:
        chunk_base = 0
        selected_bos = 0
        selected_T = 0
        selected_i_t = 0
        for i_n in range(0, N):
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int32),
                tl.load(cu_seqlens + i_n + 1).to(tl.int32),
            )
            T_local = eos - bos
            NT_local = (T_local + BT - 1) // BT
            is_selected = (chunk_pid >= chunk_base) & (
                chunk_pid < chunk_base + NT_local
            )
            selected_bos = tl.where(is_selected, bos, selected_bos)
            selected_T = tl.where(is_selected, T_local, selected_T)
            selected_i_t = tl.where(is_selected, chunk_pid - chunk_base, selected_i_t)
            chunk_base += NT_local

        # The packed grid uses an upper bound on total chunks. At most N-1
        # trailing programs are inactive when sequence tails share a BT window.
        if chunk_pid < chunk_base:
            _chunk_gated_delta_rule_fwd_stage0_4_one_tile(
                k,
                v,
                g,
                beta,
                w,
                u,
                g_cumsum_out,
                selected_bos,
                selected_T,
                i_h,
                i_hg,
                selected_i_t,
                bt_stride,
                H,
                Hg,
                K,
                V,
                BT,
                BK,
                BV,
                JOINT_UW,
                USE_G,
                STORE_G_CUMSUM,
            )
    else:
        NT_local = (T + BT - 1) // BT
        i_n = chunk_pid // NT_local
        i_t = chunk_pid % NT_local
        bos = i_n * T
        _chunk_gated_delta_rule_fwd_stage0_4_one_tile(
            k,
            v,
            g,
            beta,
            w,
            u,
            g_cumsum_out,
            bos,
            T,
            i_h,
            i_hg,
            i_t,
            bt_stride,
            H,
            Hg,
            K,
            V,
            BT,
            BK,
            BV,
            JOINT_UW,
            USE_G,
            STORE_G_CUMSUM,
        )


@triton.jit
def _chunk_gated_delta_rule_fwd_stage0_4_one_tile(
    k,
    v,
    g,
    beta,
    w,
    u,
    g_cumsum_out,
    bos,
    T_local,
    i_h,
    i_hg,
    i_t,
    bt_stride,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    JOINT_UW: tl.constexpr,
    USE_G: tl.constexpr,
    STORE_G_CUMSUM: tl.constexpr,
):
    # 处理单个 (序列, head, chunk) tile —— 与原实现逐 tile 的计算完全一致，只是抽成
    # helper 供 chunk-parallel grid 调用，且不改动内部数值逻辑。
    o_t = tl.arange(0, BT)
    o_t_f = o_t.to(tl.float32)
    global_t = i_t * BT + o_t
    mask_t = global_t < T_local

    # -------------------------------------------------------------- #
    # Stage 1: local cumsum of the log-gate over this chunk -> b_g
    #   g / beta are [H, B, T] contiguous, so g + i_h*bt_stride + bos +
    #   global_t is a stride-1 (contiguous) T-slice — no ×8 blow-up — and we
    #   still pad out-of-range tail positions with 0.0 explicitly (block_ptr
    #   boundary_check pads with garbage on this backend, which would inject
    #   NaN into the downstream triangular inverse via safe_exp on padded
    #   rows).
    # -------------------------------------------------------------- #
    if USE_G:
        p_g = g + i_h * bt_stride + bos + global_t
        b_g = tl.load(p_g, mask=mask_t, other=0.0).to(tl.float32)
        b_g = tl.cumsum(b_g, axis=0)  # [BT], prefix sum inside the chunk
        # 额外落盘 chunk 内 cumsum 后的 g，供下游 chunk_gated_delta_rule_fwd_h
        # 复用（等价于独立的 chunk_local_cumsum，output_dtype=fp32）。写回用与
        # 输入 g 相同的 [H,B,T] 连续布局（stride-1），不引入跨 T 的跳读/跳写，
        # 避免复发 ×8 AIV layout 膨胀。
        if STORE_G_CUMSUM:
            p_gc = g_cumsum_out + i_h * bt_stride + bos + global_t
            # cast 到输出 buffer 的 dtype（与输入 g 一致），再写回。
            tl.store(p_gc, b_g.to(g_cumsum_out.dtype.element_ty), mask=mask_t)
    else:
        b_g = tl.zeros([BT], dtype=tl.float32)

    p_beta = beta + i_h * bt_stride + bos + global_t
    b_beta = tl.load(p_beta, mask=mask_t, other=0.0).to(tl.float32)

    # -------------------------------------------------------------- #
    # Stage 3: A[i,j] = beta_i * (k_i . k_j) * safe_exp(g_i - g_j), i > j
    # -------------------------------------------------------------- #
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_hg) * K,
            (T_local, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        b_A *= safe_exp(b_g[:, None] - b_g[None, :])
    b_A *= b_beta[:, None]
    # Strictly-lower AND both indices in-range. The validity mask (mask_t) is
    # essential: on tail chunks the b_k boundary padding can be garbage/NaN,
    # so dot(b_k, b_kᵀ) leaves NaN in padded rows; without zeroing them here
    # the NaN would propagate through the triangular inverse into valid rows.
    b_A = tl.where(
        (o_t_f[:, None] > o_t_f[None, :]) & mask_t[:, None] & mask_t[None, :],
        b_A,
        0.0,
    )

    # -------------------------------------------------------------- #
    # Stage 4: A_inv = (I + A)^-1 via 16x16-blocked triangular inversion.
    # -------------------------------------------------------------- #
    BC: tl.constexpr = 16
    NB: tl.constexpr = BT // BC

    # (a) gather the NB strictly-lower 16x16 diagonal blocks of A
    b_D = tl.zeros([NB, BC, BC], dtype=tl.float32)
    for bi in range(NB):
        d_blk = extract_slice(b_A, (bi * BC, bi * BC), (BC, BC), (1, 1))
        b_D = insert_slice(
            ful=b_D,
            sub=d_blk[None, :, :],
            offsets=[bi, 0, 0],
            sizes=[1, BC, BC],
            strides=[1, 1, 1],
        )

    # batched forward substitution: T = (I + A_bb)^-1 for every diagonal block
    o_c = tl.arange(0, BC).to(tl.float32)
    is_lower = (o_c[:, None] > o_c[None, :]).to(tl.float32)
    local_ori = tl.reshape(tl.trans(b_D, (1, 0, 2)), (BC, BC * NB))
    b_D = -b_D * is_lower[None, :, :]
    for i in range(1, BC):
        row_i = -extract_slice(local_ori, (i, 0), (1, BC * NB), (BC * NB, 1))
        b_a = tl.reshape(row_i, (NB, BC))
        dot_i = tl.sum(tl.trans(b_a[:, :, None] * b_D, (1, 0, 2)), 0)
        b_a = b_a + dot_i
        b_D = insert_slice(
            ful=b_D,
            sub=b_a[:, None, :],
            offsets=[0, i, 0],
            sizes=[NB, 1, BC],
            strides=[1, 1, 1],
        )
    b_D = tl.where((o_c[:, None] == o_c[None, :])[None, :, :], b_D + 1.0, b_D)

    # (b) Hierarchical block-triangular inversion 16 -> 32 -> 64, mirroring
    # solve_tril's merge_16x16_to_64x64 scheme. Replaces the old O(NB^3)
    # serial Schur loop (6 off-diagonal blocks, each with an inner kb-accum
    # of 16x16 dots + a long extract/insert_slice chain) with a fixed, short
    # dependency chain that also uses larger 32x32 dots for better cube
    # utilization. Uses the block-lower-triangular inverse identity
    #   [[M11, 0], [M21, M22]]^-1 = [[M11^-1, 0],
    #                                [-M22^-1 M21 M11^-1, M22^-1]]
    # applied twice (16->32 then 32->64). Specialized for BT=64 (NB=4),
    # which the wrapper asserts.
    #
    # the four 16x16 diagonal-block inverses D_ii = (I + A_ii)^-1
    D0 = tl.reshape(extract_slice(b_D, (0, 0, 0), (1, BC, BC), (1, 1, 1)), (BC, BC))
    D1 = tl.reshape(extract_slice(b_D, (1, 0, 0), (1, BC, BC), (1, 1, 1)), (BC, BC))
    D2 = tl.reshape(extract_slice(b_D, (2, 0, 0), (1, BC, BC), (1, 1, 1)), (BC, BC))
    D3 = tl.reshape(extract_slice(b_D, (3, 0, 0), (1, BC, BC), (1, 1, 1)), (BC, BC))

    # --- level 2: two 32x32 inverses. M21 is the strictly-lower off-diagonal
    # block of the original A (still held in b_A here). X21 = -D_hi @ A21 @ D_lo.
    A_10 = extract_slice(b_A, (16, 0), (BC, BC), (1, 1))  # A[16:32, 0:16]
    Ai_10 = -tl.dot(tl.dot(D1, A_10, allow_tf32=False), D0, allow_tf32=False)
    A_32 = extract_slice(b_A, (48, 32), (BC, BC), (1, 1))  # A[48:64, 32:48]
    Ai_32 = -tl.dot(tl.dot(D3, A_32, allow_tf32=False), D2, allow_tf32=False)

    Ai_11_32 = tl.zeros([32, 32], dtype=tl.float32)
    Ai_11_32 = insert_slice(
        ful=Ai_11_32, sub=D0, offsets=[0, 0], sizes=[BC, BC], strides=[1, 1]
    )
    Ai_11_32 = insert_slice(
        ful=Ai_11_32, sub=D1, offsets=[16, 16], sizes=[BC, BC], strides=[1, 1]
    )
    Ai_11_32 = insert_slice(
        ful=Ai_11_32, sub=Ai_10, offsets=[16, 0], sizes=[BC, BC], strides=[1, 1]
    )

    Ai_22_32 = tl.zeros([32, 32], dtype=tl.float32)
    Ai_22_32 = insert_slice(
        ful=Ai_22_32, sub=D2, offsets=[0, 0], sizes=[BC, BC], strides=[1, 1]
    )
    Ai_22_32 = insert_slice(
        ful=Ai_22_32, sub=D3, offsets=[16, 16], sizes=[BC, BC], strides=[1, 1]
    )
    Ai_22_32 = insert_slice(
        ful=Ai_22_32, sub=Ai_32, offsets=[16, 0], sizes=[BC, BC], strides=[1, 1]
    )

    # --- level 3: merge the two 32x32 inverses into the full 64x64.
    # X_bl = -Ai_22_32 @ A[32:64, 0:32] @ Ai_11_32
    A_21_32 = extract_slice(b_A, (32, 0), (32, 32), (1, 1))  # A[32:64, 0:32]
    Ai_21_32 = -tl.dot(
        tl.dot(Ai_22_32, A_21_32, allow_tf32=False), Ai_11_32, allow_tf32=False
    )

    b_Ainv = tl.zeros([BT, BT], dtype=tl.float32)
    b_Ainv = insert_slice(
        ful=b_Ainv, sub=Ai_11_32, offsets=[0, 0], sizes=[32, 32], strides=[1, 1]
    )
    b_Ainv = insert_slice(
        ful=b_Ainv, sub=Ai_22_32, offsets=[32, 32], sizes=[32, 32], strides=[1, 1]
    )
    b_Ainv = insert_slice(
        ful=b_Ainv, sub=Ai_21_32, offsets=[32, 0], sizes=[32, 32], strides=[1, 1]
    )
    b_A = b_Ainv

    # -------------------------------------------------------------- #
    # Stage 5: u = A_inv @ (beta * v),  w = A_inv @ (beta * exp(g) * k)
    # -------------------------------------------------------------- #
    if USE_G:
        b_eg = tl.exp(b_g)
    else:
        b_eg = tl.full([BT], 1.0, dtype=tl.float32)

    if JOINT_UW:
        # K=V=128: concatenate the two right operands and issue one wide-N
        # Cube operation. This lets the backend reuse the same A_inv tile
        # across U and W instead of loading it for two independent dots.
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T_local, V),
            (H * V, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_hg) * K,
            (T_local, K),
            (Hg * K, 1),
            (i_t * BT, 0),
            (BT, BK),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.where(mask_t[:, None], b_v, 0.0)
        b_k = tl.where(mask_t[:, None], b_k, 0.0)
        b_vb = b_v * b_beta[:, None]
        b_kb = b_k * (b_beta * b_eg)[:, None]

        b_rhs = tl.zeros([BT, 256], dtype=v.dtype.element_ty)
        b_rhs = insert_slice(
            ful=b_rhs,
            sub=b_vb.to(v.dtype.element_ty),
            offsets=[0, 0],
            sizes=[BT, BV],
            strides=[1, 1],
        )
        b_rhs = insert_slice(
            ful=b_rhs,
            sub=b_kb.to(k.dtype.element_ty),
            offsets=[0, 128],
            sizes=[BT, BK],
            strides=[1, 1],
        )
        b_uw = tl.dot(
            b_A.to(v.dtype.element_ty),
            b_rhs,
            allow_tf32=False,
        )
        b_u = extract_slice(b_uw, (0, 0), (BT, BV), (1, 1))
        b_w = extract_slice(b_uw, (0, 128), (BT, BK), (1, 1))

        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T_local, V),
            (H * V, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T_local, K),
            (H * K, 1),
            (i_t * BT, 0),
            (BT, BK),
            (1, 0),
        )
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
    else:
        # Generic dimensions retain the original two-dot implementation.
        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (T_local, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            b_v = tl.where(mask_t[:, None], b_v, 0.0)
            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(
                b_A.to(v.dtype.element_ty),
                b_vb.to(v.dtype.element_ty),
                allow_tf32=False,
            )
            p_u = tl.make_block_ptr(
                u + (bos * H + i_h) * V,
                (T_local, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(
                p_u,
                b_u.to(p_u.dtype.element_ty),
                boundary_check=(0, 1),
            )

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(
                k + (bos * Hg + i_hg) * K,
                (T_local, K),
                (Hg * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.where(mask_t[:, None], b_k, 0.0)
            b_kb = b_k * (b_beta * b_eg)[:, None]
            b_w = tl.dot(
                b_A.to(k.dtype.element_ty),
                b_kb.to(k.dtype.element_ty),
                allow_tf32=False,
            )
            p_w = tl.make_block_ptr(
                w + (bos * H + i_h) * K,
                (T_local, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            tl.store(
                p_w,
                b_w.to(p_w.dtype.element_ty),
                boundary_check=(0, 1),
            )


def chunk_gated_delta_rule_fwd_stage0_4(
    k: torch.Tensor,  # [B, T, Hg, K]
    v: torch.Tensor,  # [B, T, H,  V]
    g: torch.Tensor,  # [B, T, H]  raw log-gate
    beta: torch.Tensor,  # [B, T, H]
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    num_cores: int | None = None,  # Legacy compatibility; unused.
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""×8-free fully fused chunk_gated_delta_rule_fwd preprocess. Same contract as fused_preprocess.

    Launches one program per independent ``chunk × head`` tile. The runtime
    schedules the grid over the available cores, while packed inputs recover
    the flattened chunk mapping from ``cu_seqlens`` on device.

    Returns (w, u) of shapes [B, T, H, K] and [B, T, H, V].

    The launcher also returns ``g_cumsum`` of shape [B, T, H] with dtype
    **fp32** — the chunk-local cumsum of the raw log-gate, numerically
    equivalent to ``chunk_local_cumsum(g, chunk_size)`` (whose default
    ``output_dtype`` is ``torch.float``). This lets callers drop the standalone
    ``chunk_local_cumsum`` kernel and reuse the cumsum the fused kernel already
    computes internally.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = chunk_size
    if BT != 64:
        raise NotImplementedError(
            "chunk_gated_delta_rule_fwd_stage0_4 supports only chunk_size=64"
        )
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("variable-length mode expects packed inputs with B=1")
        N = int(cu_seqlens.numel() - 1)
        max_total_chunks = triton.cdiv(T, BT) + N - 1
    else:
        N = B
        max_total_chunks = N * triton.cdiv(T, BT)

    # The single MIX kernel avoids staged launch overhead on very small packed
    # inputs. The metadata-free upper bound avoids device-to-host synchronization.
    use_single_kernel = (
        cu_seqlens is not None and max_total_chunks <= SINGLE_KERNEL_MAX_TOTAL_CHUNKS
    )
    # The staged pair pipeline handles any K/V now that stage4 column-blocks
    # its dots; route all B=1 prefill shapes through it.
    if B == 1 and not use_single_kernel:
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size=BT,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        g_perm = torch.empty((H, B, T), device=g.device, dtype=g_cumsum.dtype)
        beta_perm = torch.empty((H, B, T), device=beta.device, dtype=beta.dtype)
        transpose_block_t = 128
        _transpose_gate_beta_kernel[(triton.cdiv(T, transpose_block_t), B)](
            g_cumsum,
            beta,
            g_perm,
            beta_perm,
            T,
            B,
            H=H,
            BLOCK_T=transpose_block_t,
            num_warps=4,
            num_stages=1,
        )

        num_chunks = max_total_chunks
        a = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
        chunk_scaled_dot_kkt_fwd_kernel[(num_chunks, 1)](
            k=k,
            beta=beta_perm,
            g_cumsum=g_perm,
            A=a,
            cu_seqlens=cu_seqlens,
            T=T,
            B=B,
            N=N,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
            BK=128,
            num_warps=8,
            num_stages=3,
            multibuffer=True,
        )

        num_pairs = (
            triton.cdiv(T, 2 * BT)
            if cu_seqlens is None
            else triton.cdiv(T, 2 * BT) + N - 1
        )
        if cu_seqlens is not None or num_pairs * H >= 128:
            a_inv = solve_tril_pair_merge(
                A=a,
                cu_seqlens=cu_seqlens,
                output_dtype=k.dtype,
            )
        else:
            a_inv = solve_tril(
                A=a,
                cu_seqlens=cu_seqlens,
                output_dtype=k.dtype,
            )
        w, u = stage4_chunk_pair_combined128(
            k,
            v,
            beta_perm,
            g_perm,
            a_inv,
            cu_seqlens,
        )
        return w, u, g_cumsum

    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)

    # Produce both contiguous [H, B, T] inputs in one vector-kernel launch.
    g_perm = torch.empty((H, B, T), device=g.device, dtype=g.dtype)
    beta_perm = torch.empty((H, B, T), device=beta.device, dtype=beta.dtype)
    transpose_block_t = 128
    _transpose_gate_beta_kernel[(triton.cdiv(T, transpose_block_t), B)](
        g,
        beta,
        g_perm,
        beta_perm,
        T,
        B,
        H=H,
        BLOCK_T=transpose_block_t,
        num_warps=4,
        num_stages=1,
    )

    # 可选：chunk 内 cumsum 后的 g 输出。dtype 固定 fp32，对齐 chunk_local_cumsum 的
    # 默认输出（chunk.py 调用时未指定 output_dtype，即 torch.float），保证替换后下游
    # 拿到的 g 精度不变。用与 g_perm 相同的 [H,B,T] 连续布局，kernel 内 stride-1 写回，
    # 事后 permute 回 [B,T,H] 交给下游。
    g_cumsum_out = torch.empty_like(g_perm, dtype=torch.float32)

    # BK=128 让 K=128 的 KKT / w 走单次大 dot（cube 利用率更高），对齐独立
    # chunk_scaled_dot_kkt 的 BK=128；若因 UB 压力（[BT,128] fp32 tile）复发溢出
    # 可回退到 64。BV 同理但 V 常见 64/128。
    BK = min(triton.next_power_of_2(K), 128)
    BV = min(triton.next_power_of_2(V), 128)

    # Fixed-length mode has an exact tile count. For packed mode,
    # ceil(total_T / BT) + N - 1 is an upper bound on the sum of per-sequence
    # chunk counts; an overestimate only creates inactive programs.
    # The Ascend compiler allocates hidden MIX workspace per logical program.
    # Bound one launch to avoid overflowing its GM addressing on long sequences.
    max_chunks_per_launch = max(1, MAX_PROGRAMS_PER_LAUNCH // H)
    for chunk_pid_offset in range(0, max_total_chunks, max_chunks_per_launch):
        chunks_this_launch = min(
            max_chunks_per_launch, max_total_chunks - chunk_pid_offset
        )
        grid = (chunks_this_launch, H)
        chunk_gated_delta_rule_fwd_stage0_4_kernel[grid](
            k=k,
            v=v,
            g=g_perm,
            beta=beta_perm,
            w=w,
            u=u,
            g_cumsum_out=g_cumsum_out,
            cu_seqlens=cu_seqlens,
            chunk_pid_offset=chunk_pid_offset,
            T=T,
            B=B,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            N=N,
            BT=BT,
            BK=BK,
            BV=BV,
            JOINT_UW=K == 128 and V == 128,
            num_warps=8,
            num_stages=1,
        )
    g_cumsum = torch.empty_like(g, dtype=torch.float32)
    _transpose_cumsum_output_kernel[(triton.cdiv(T, transpose_block_t), B)](
        g_cumsum_out,
        g_cumsum,
        T,
        B,
        H=H,
        BLOCK_T=transpose_block_t,
        num_warps=4,
        num_stages=1,
    )
    return w, u, g_cumsum


@triton.jit
def _schedule_grouped_varlen_stage5_task(
    cu_seqlens, sequence_head_idx, total_tasks, H: tl.constexpr
):
    group_sequences: tl.constexpr = 3
    group_tasks: tl.constexpr = group_sequences * H
    num_sequences = total_tasks // H
    group_idx = sequence_head_idx // group_tasks
    group_sequence_start = group_idx * group_sequences
    local_task_idx = sequence_head_idx - group_idx * group_tasks
    group_size = min(group_sequences, num_sequences - group_sequence_start)
    tail_four_sequence_start = num_sequences - 4
    is_tail_pair_task = (num_sequences % group_sequences == 1) & (
        sequence_head_idx >= tail_four_sequence_start * H
    )
    if is_tail_pair_task:
        pair_tasks: tl.constexpr = 2 * H
        tail_task_idx = sequence_head_idx - tail_four_sequence_start * H
        pair_idx = tail_task_idx // pair_tasks
        group_sequence_start = tail_four_sequence_start + 2 * pair_idx
        local_task_idx = tail_task_idx - pair_idx * pair_tasks
        group_size = 2
    scheduled_local_task_idx = local_task_idx
    sequence_tokens0 = tl.load(cu_seqlens + group_sequence_start + 1) - tl.load(
        cu_seqlens + group_sequence_start
    )
    if group_size == 3:
        sequence_tokens1 = tl.load(cu_seqlens + group_sequence_start + 2) - tl.load(
            cu_seqlens + group_sequence_start + 1
        )
        sequence_tokens2 = tl.load(cu_seqlens + group_sequence_start + 3) - tl.load(
            cu_seqlens + group_sequence_start + 2
        )
        swap01 = sequence_tokens1 > sequence_tokens0
        high_tokens = tl.where(swap01, sequence_tokens1, sequence_tokens0)
        high_idx = tl.where(swap01, 1, 0)
        low_tokens = tl.where(swap01, sequence_tokens0, sequence_tokens1)
        low_idx = tl.where(swap01, 0, 1)
        swap12 = sequence_tokens2 > low_tokens
        middle_tokens = tl.where(swap12, sequence_tokens2, low_tokens)
        middle_idx = tl.where(swap12, 2, low_idx)
        short_idx = tl.where(swap12, low_idx, 2)
        swap_top = middle_tokens > high_tokens
        long_idx = tl.where(swap_top, middle_idx, high_idx)
        middle_idx = tl.where(swap_top, high_idx, middle_idx)
        long_tokens = tl.where(swap_top, middle_tokens, high_tokens)
        middle_tokens = tl.where(swap_top, high_tokens, middle_tokens)
        short_tokens = min(low_tokens, sequence_tokens2)
        flat_critical_tokens = long_tokens + middle_tokens
        balanced_critical_tokens = max(long_tokens + short_tokens, 2 * middle_tokens)
        should_balance = 100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        if should_balance:
            template_task_idx = local_task_idx
            head_group_base = 0
            if H == 32:
                head_group_idx = local_task_idx // 48
                template_task_idx = local_task_idx - head_group_idx * 48
                head_group_base = head_group_idx * 16
            is_long_task = template_task_idx < 16
            is_middle_task = (template_task_idx >= 16) & (template_task_idx < 24) | (
                template_task_idx >= 40
            )
            middle_head_idx = tl.where(
                template_task_idx < 24,
                template_task_idx - 16,
                template_task_idx - 32,
            )
            short_head_idx = template_task_idx - 24
            scheduled_local_task_idx = tl.where(
                is_long_task,
                long_idx * H + head_group_base + template_task_idx,
                tl.where(
                    is_middle_task,
                    middle_idx * H + head_group_base + middle_head_idx,
                    short_idx * H + head_group_base + short_head_idx,
                ),
            )
    elif group_size == 2:
        sequence_tokens1 = tl.load(cu_seqlens + group_sequence_start + 2) - tl.load(
            cu_seqlens + group_sequence_start + 1
        )
        sequence1_is_long = sequence_tokens1 > sequence_tokens0
        long_idx = tl.where(sequence1_is_long, 1, 0)
        short_idx = 1 - long_idx
        long_tokens = tl.where(sequence1_is_long, sequence_tokens1, sequence_tokens0)
        short_tokens = tl.where(sequence1_is_long, sequence_tokens0, sequence_tokens1)
        flat_critical_tokens = long_tokens + short_tokens
        balanced_critical_tokens = max(long_tokens, 2 * short_tokens)
        should_balance = 100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        if should_balance:
            template_task_idx = local_task_idx
            head_group_base = 0
            if H == 32:
                head_group_idx = local_task_idx // 32
                template_task_idx = local_task_idx - head_group_idx * 32
                head_group_base = head_group_idx * 16
            is_long_task = (template_task_idx >= 8) & (template_task_idx < 24)
            long_head_idx = template_task_idx - 8
            short_head_idx = tl.where(
                template_task_idx < 8,
                template_task_idx,
                template_task_idx - 16,
            )
            scheduled_local_task_idx = tl.where(
                is_long_task,
                long_idx * H + head_group_base + long_head_idx,
                short_idx * H + head_group_base + short_head_idx,
            )
    return group_sequence_start * H + scheduled_local_task_idx


@triton.jit
def _stage6_loop_pipeline_produce_qk_qh(
    q,
    k,
    h,
    qk_workspace,
    qh_workspace,
    bos,
    sequence_tokens,
    chunk_base,
    chunk_idx,
    head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    kv_head_idx = head_idx // (H // Hg)
    token_start = chunk_idx * BT
    stride_qk = Hg * K
    q_base = q + bos * Hg * K + kv_head_idx * K
    k_base = k + bos * Hg * K + kv_head_idx * K
    h_base = h + ((chunk_base + chunk_idx) * H + head_idx) * K * V
    p_q = tl.make_block_ptr(
        q_base, (sequence_tokens, K), (stride_qk, 1), (token_start, 0), (BT, K), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_base, (K, sequence_tokens), (1, stride_qk), (0, token_start), (K, BT), (0, 1)
    )
    p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    qk_worker_base = qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * K * V + buffer_id * BT * V
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_h = tl.load(p_h, volatile=True)
    b_qk = tl.dot(b_q, b_k)
    b_qh = tl.dot(b_q, b_h)
    al.sync_block_wait(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    tl.store(p_qk, b_qk.to(p_qk.dtype.element_ty))
    tl.store(p_qh, b_qh.to(p_qh.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage6_loop_pipeline_preprocess(
    g,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    bos,
    sequence_tokens,
    token_start,
    sequence_idx,
    head_idx,
    worker_idx,
    buffer_id,
    total_tokens,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    qk_worker_base = qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * K * V + buffer_id * BT * V
    gated_qk_worker_base = (
        gated_qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    )
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_gated_qk = tl.make_block_ptr(
        gated_qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qk = tl.load(p_qk).to(tl.float32)
    b_qh = tl.load(p_qh).to(tl.float32)
    al.sync_block_set(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    token_offsets = token_start + tl.arange(0, BT)
    valid_tokens = token_offsets < sequence_tokens
    if USE_G:
        if IS_VARLEN:
            g_base = g + bos + head_idx * total_tokens
        else:
            g_base = g + (sequence_idx * H + head_idx) * total_tokens
        b_g = tl.load(g_base + token_offsets, mask=valid_tokens, other=0.0)
        b_qh_scaled = b_qh * tl.exp(b_g)[:, None]
        b_qk *= safe_exp(b_g[:, None] - b_g[None, :])
    else:
        b_qh_scaled = b_qh
    causal_idx = tl.arange(0, BT)
    causal_mask = causal_idx[:, None] >= causal_idx[None, :]
    b_gated_qk = tl.where(causal_mask, b_qk, 0.0)
    tl.store(p_gated_qk, b_gated_qk.to(p_gated_qk.dtype.element_ty))
    al.sync_block_set(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    return b_qh_scaled


@triton.jit
def _stage6_loop_pipeline_qkv(
    v_new,
    gated_qk_workspace,
    qkv_workspace,
    bos,
    sequence_tokens,
    chunk_idx,
    head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    token_start = chunk_idx * BT
    stride_v = H * V
    v_base = v_new + bos * H * V + head_idx * V
    p_v = tl.make_block_ptr(
        v_base, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    gated_qk_worker_base = (
        gated_qk_workspace + worker_idx * BT * V + buffer_id * BT * BT
    )
    qkv_worker_base = qkv_workspace + worker_idx * V * V + buffer_id * BT * V
    p_gated_qk = tl.make_block_ptr(
        gated_qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qkv = tl.make_block_ptr(qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    al.sync_block_wait(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_gated_qk = tl.load(p_gated_qk)
    b_v = tl.load(p_v, boundary_check=(0, 1), volatile=True)
    b_qkv = tl.dot(b_gated_qk, b_v)
    tl.store(p_qkv, b_qkv.to(p_qkv.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage6_loop_pipeline_finalize(
    o,
    qkv_workspace,
    b_qh_scaled,
    scale,
    bos,
    sequence_tokens,
    chunk_idx,
    head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    token_start = chunk_idx * BT
    stride_v = H * V
    o_base = o + bos * H * V + head_idx * V
    qkv_worker_base = qkv_workspace + worker_idx * V * V + buffer_id * BT * V
    p_qkv = tl.make_block_ptr(qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_o = tl.make_block_ptr(
        o_base, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qkv = tl.load(p_qkv).to(tl.float32)
    b_o = (b_qh_scaled + b_qkv) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _stage6_head_pair_produce_qk_qh(
    q,
    k,
    h,
    qk_workspace,
    qh_workspace,
    bos,
    sequence_tokens,
    chunk_base,
    chunk_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    token_start = chunk_idx * BT
    stride_qk = Hg * K
    q_base = q + bos * Hg * K + kv_head_idx * K
    k_base = k + bos * Hg * K + kv_head_idx * K
    h_base0 = h + ((chunk_base + chunk_idx) * H + head_idx0) * K * V
    h_base1 = h + ((chunk_base + chunk_idx) * H + head_idx1) * K * V
    p_q = tl.make_block_ptr(
        q_base, (sequence_tokens, K), (stride_qk, 1), (token_start, 0), (BT, K), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_base, (K, sequence_tokens), (1, stride_qk), (0, token_start), (K, BT), (0, 1)
    )
    p_h0 = tl.make_block_ptr(h_base0, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    p_h1 = tl.make_block_ptr(h_base1, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    qk_worker_base = qk_workspace + worker_idx * 2 * BT * BT + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh0 = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_qh1 = tl.make_block_ptr(
        qh_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_h0 = tl.load(p_h0, volatile=True)
    b_h1 = tl.load(p_h1, volatile=True)
    b_qk = tl.dot(b_q, b_k)
    b_qh0 = tl.dot(b_q, b_h0)
    b_qh1 = tl.dot(b_q, b_h1)
    al.sync_block_wait(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    tl.store(p_qk, b_qk.to(p_qk.dtype.element_ty))
    tl.store(p_qh0, b_qh0.to(p_qh0.dtype.element_ty))
    tl.store(p_qh1, b_qh1.to(p_qh1.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage6_head_pair_preprocess(
    g,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    bos,
    sequence_tokens,
    token_start,
    sequence_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    total_tokens,
    H: tl.constexpr,
    Hg: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    qk_worker_base = qk_workspace + worker_idx * 2 * BT * BT + buffer_id * BT * BT
    qh_worker_base = qh_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    gated_worker_base = (
        gated_qk_workspace + worker_idx * 4 * BT * BT + buffer_id * 2 * BT * BT
    )
    p_qk = tl.make_block_ptr(
        qk_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qh0 = tl.make_block_ptr(qh_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0))
    p_qh1 = tl.make_block_ptr(
        qh_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_gated0 = tl.make_block_ptr(
        gated_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_gated1 = tl.make_block_ptr(
        gated_worker_base + BT * BT, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        2 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qk = tl.load(p_qk).to(tl.float32)
    b_qh0 = tl.load(p_qh0).to(tl.float32)
    b_qh1 = tl.load(p_qh1).to(tl.float32)
    al.sync_block_set(
        "vector",
        "cube",
        buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    token_offsets = token_start + tl.arange(0, BT)
    valid_tokens = token_offsets < sequence_tokens
    if USE_G:
        if IS_VARLEN:
            g_base0 = g + bos + head_idx0 * total_tokens
            g_base1 = g + bos + head_idx1 * total_tokens
        else:
            g_base0 = g + (sequence_idx * H + head_idx0) * total_tokens
            g_base1 = g + (sequence_idx * H + head_idx1) * total_tokens
        b_g0 = tl.load(g_base0 + token_offsets, mask=valid_tokens, other=0.0)
        b_g1 = tl.load(g_base1 + token_offsets, mask=valid_tokens, other=0.0)
        b_qh0 *= tl.exp(b_g0)[:, None]
        b_qh1 *= tl.exp(b_g1)[:, None]
        b_gated0 = b_qk * safe_exp(b_g0[:, None] - b_g0[None, :])
        b_gated1 = b_qk * safe_exp(b_g1[:, None] - b_g1[None, :])
    else:
        b_gated0 = b_qk
        b_gated1 = b_qk
    causal_idx = tl.arange(0, BT)
    causal_mask = causal_idx[:, None] >= causal_idx[None, :]
    b_gated0 = tl.where(causal_mask, b_gated0, 0.0)
    b_gated1 = tl.where(causal_mask, b_gated1, 0.0)
    tl.store(p_gated0, b_gated0.to(p_gated0.dtype.element_ty))
    tl.store(p_gated1, b_gated1.to(p_gated1.dtype.element_ty))
    al.sync_block_set(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    return (b_qh0, b_qh1)


@triton.jit
def _stage6_head_pair_qkv(
    v_new,
    gated_qk_workspace,
    qkv_workspace,
    bos,
    sequence_tokens,
    chunk_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    token_start = chunk_idx * BT
    stride_v = H * V
    v_base0 = v_new + bos * H * V + head_idx0 * V
    v_base1 = v_new + bos * H * V + head_idx1 * V
    p_v0 = tl.make_block_ptr(
        v_base0, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    p_v1 = tl.make_block_ptr(
        v_base1, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    gated_worker_base = (
        gated_qk_workspace + worker_idx * 4 * BT * BT + buffer_id * 2 * BT * BT
    )
    qkv_worker_base = qkv_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    p_gated0 = tl.make_block_ptr(
        gated_worker_base, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_gated1 = tl.make_block_ptr(
        gated_worker_base + BT * BT, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0)
    )
    p_qkv0 = tl.make_block_ptr(
        qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_qkv1 = tl.make_block_ptr(
        qkv_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    al.sync_block_wait(
        "vector",
        "cube",
        4 + buffer_id,
        sender_pipe=al.PIPE.PIPE_MTE3,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_gated0 = tl.load(p_gated0)
    b_gated1 = tl.load(p_gated1)
    b_v0 = tl.load(p_v0, boundary_check=(0, 1), volatile=True)
    b_v1 = tl.load(p_v1, boundary_check=(0, 1), volatile=True)
    b_qkv0 = tl.dot(b_gated0, b_v0)
    b_qkv1 = tl.dot(b_gated1, b_v1)
    tl.store(p_qkv0, b_qkv0.to(p_qkv0.dtype.element_ty))
    tl.store(p_qkv1, b_qkv1.to(p_qkv1.dtype.element_ty))
    al.sync_block_set(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )


@triton.jit
def _stage6_head_pair_finalize(
    o,
    qkv_workspace,
    b_qh0,
    b_qh1,
    scale,
    bos,
    sequence_tokens,
    chunk_idx,
    kv_head_idx,
    worker_idx,
    buffer_id,
    H: tl.constexpr,
    Hg: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    GROUP_SIZE: tl.constexpr = H // Hg
    head_idx0 = kv_head_idx * GROUP_SIZE
    head_idx1 = head_idx0 + 1
    token_start = chunk_idx * BT
    stride_v = H * V
    o_base0 = o + bos * H * V + head_idx0 * V
    o_base1 = o + bos * H * V + head_idx1 * V
    qkv_worker_base = qkv_workspace + worker_idx * 4 * BT * V + buffer_id * 2 * BT * V
    p_qkv0 = tl.make_block_ptr(
        qkv_worker_base, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_qkv1 = tl.make_block_ptr(
        qkv_worker_base + BT * V, (BT, V), (V, 1), (0, 0), (BT, V), (1, 0)
    )
    p_o0 = tl.make_block_ptr(
        o_base0, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    p_o1 = tl.make_block_ptr(
        o_base1, (sequence_tokens, V), (stride_v, 1), (token_start, 0), (BT, V), (1, 0)
    )
    al.sync_block_wait(
        "cube",
        "vector",
        6 + buffer_id,
        sender_pipe=al.PIPE.PIPE_FIX,
        receiver_pipe=al.PIPE.PIPE_MTE2,
    )
    b_qkv0 = tl.load(p_qkv0).to(tl.float32)
    b_qkv1 = tl.load(p_qkv1).to(tl.float32)
    b_o0 = (b_qh0 + b_qkv0) * scale
    b_o1 = (b_qh1 + b_qkv1) * scale
    tl.store(p_o0, b_o0.to(p_o0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_o1, b_o1.to(p_o1.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _run_stage6_head_pair_loop_pipeline(
    q,
    k,
    v_new,
    h,
    g,
    o,
    cu_seqlens,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    qkv_workspace,
    scale,
    T,
    TOTAL_TASKS,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    total_tokens = 1 * T
    num_sequences = TOTAL_TASKS // H
    al.sync_block_set(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_set(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    db_flag = 0
    global_task_base = 0
    for sequence_idx in range(num_sequences):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
            eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
            sequence_tokens = eos - bos
            chunk_base = bos // BT + sequence_idx
        else:
            bos = sequence_idx * T
            sequence_tokens = T
            chunk_base = sequence_idx * tl.cdiv(T, BT)
        num_chunks = tl.cdiv(sequence_tokens, BT)
        sequence_tasks = num_chunks * Hg
        local_start = (
            worker_idx - global_task_base % num_workers + num_workers
        ) % num_workers
        if local_start < sequence_tasks:
            first_chunk_idx = local_start // Hg
            first_kv_head_idx = local_start % Hg
            _stage6_head_pair_produce_qk_qh(
                q,
                k,
                h,
                qk_workspace,
                qh_workspace,
                bos,
                sequence_tokens,
                chunk_base,
                first_chunk_idx,
                first_kv_head_idx,
                worker_idx,
                db_flag % 2,
                H=H,
                Hg=Hg,
                K=K,
                V=V,
                BT=BT,
            )
            for local_task_idx in range(local_start, sequence_tasks, num_workers):
                chunk_idx = local_task_idx // Hg
                kv_head_idx = local_task_idx % Hg
                buffer_id = db_flag % 2
                b_qh0, b_qh1 = _stage6_head_pair_preprocess(
                    g,
                    qk_workspace,
                    qh_workspace,
                    gated_qk_workspace,
                    bos,
                    sequence_tokens,
                    chunk_idx * BT,
                    sequence_idx,
                    kv_head_idx,
                    worker_idx,
                    buffer_id,
                    total_tokens,
                    H=H,
                    Hg=Hg,
                    V=V,
                    BT=BT,
                    USE_G=USE_G,
                    IS_VARLEN=IS_VARLEN,
                )
                next_task_idx = local_task_idx + num_workers
                if next_task_idx < sequence_tasks:
                    _stage6_head_pair_produce_qk_qh(
                        q,
                        k,
                        h,
                        qk_workspace,
                        qh_workspace,
                        bos,
                        sequence_tokens,
                        chunk_base,
                        next_task_idx // Hg,
                        next_task_idx % Hg,
                        worker_idx,
                        (db_flag + 1) % 2,
                        H=H,
                        Hg=Hg,
                        K=K,
                        V=V,
                        BT=BT,
                    )
                _stage6_head_pair_qkv(
                    v_new,
                    gated_qk_workspace,
                    qkv_workspace,
                    bos,
                    sequence_tokens,
                    chunk_idx,
                    kv_head_idx,
                    worker_idx,
                    buffer_id,
                    H=H,
                    Hg=Hg,
                    V=V,
                    BT=BT,
                )
                _stage6_head_pair_finalize(
                    o,
                    qkv_workspace,
                    b_qh0,
                    b_qh1,
                    scale,
                    bos,
                    sequence_tokens,
                    chunk_idx,
                    kv_head_idx,
                    worker_idx,
                    buffer_id,
                    H=H,
                    Hg=Hg,
                    V=V,
                    BT=BT,
                )
                db_flag += 1
        global_task_base += sequence_tasks
    al.sync_block_wait(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_wait(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )


@triton.jit
def _run_stage6_loop_pipeline(
    q,
    k,
    v_new,
    h,
    g,
    o,
    cu_seqlens,
    qk_workspace,
    qh_workspace,
    gated_qk_workspace,
    qkv_workspace,
    scale,
    T,
    TOTAL_TASKS,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    total_tokens = 1 * T
    num_sequences = TOTAL_TASKS // H
    al.sync_block_set(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_set(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    db_flag = 0
    global_task_base = 0
    for sequence_idx in range(num_sequences):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
            eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
            sequence_tokens = eos - bos
            chunk_base = bos // BT + sequence_idx
        else:
            bos = sequence_idx * T
            sequence_tokens = T
            chunk_base = sequence_idx * tl.cdiv(T, BT)
        num_chunks = tl.cdiv(sequence_tokens, BT)
        sequence_tasks = num_chunks * H
        local_start = (
            worker_idx - global_task_base % num_workers + num_workers
        ) % num_workers
        if local_start < sequence_tasks:
            first_chunk_idx = local_start // H
            first_head_idx = local_start % H
            _stage6_loop_pipeline_produce_qk_qh(
                q=q,
                k=k,
                h=h,
                qk_workspace=qk_workspace,
                qh_workspace=qh_workspace,
                bos=bos,
                sequence_tokens=sequence_tokens,
                chunk_base=chunk_base,
                chunk_idx=first_chunk_idx,
                head_idx=first_head_idx,
                worker_idx=worker_idx,
                buffer_id=db_flag % 2,
                H=H,
                Hg=Hg,
                K=K,
                V=V,
                BT=BT,
            )
            for local_task_idx in range(local_start, sequence_tasks, num_workers):
                chunk_idx = local_task_idx // H
                head_idx = local_task_idx % H
                buffer_id = db_flag % 2
                token_start = chunk_idx * BT
                b_qh_scaled = _stage6_loop_pipeline_preprocess(
                    g=g,
                    qk_workspace=qk_workspace,
                    qh_workspace=qh_workspace,
                    gated_qk_workspace=gated_qk_workspace,
                    bos=bos,
                    sequence_tokens=sequence_tokens,
                    token_start=token_start,
                    sequence_idx=sequence_idx,
                    head_idx=head_idx,
                    worker_idx=worker_idx,
                    buffer_id=buffer_id,
                    total_tokens=total_tokens,
                    H=H,
                    K=K,
                    V=V,
                    BT=BT,
                    USE_G=USE_G,
                    IS_VARLEN=IS_VARLEN,
                )
                next_task_idx = local_task_idx + num_workers
                if next_task_idx < sequence_tasks:
                    next_chunk_idx = next_task_idx // H
                    next_head_idx = next_task_idx % H
                    _stage6_loop_pipeline_produce_qk_qh(
                        q=q,
                        k=k,
                        h=h,
                        qk_workspace=qk_workspace,
                        qh_workspace=qh_workspace,
                        bos=bos,
                        sequence_tokens=sequence_tokens,
                        chunk_base=chunk_base,
                        chunk_idx=next_chunk_idx,
                        head_idx=next_head_idx,
                        worker_idx=worker_idx,
                        buffer_id=(db_flag + 1) % 2,
                        H=H,
                        Hg=Hg,
                        K=K,
                        V=V,
                        BT=BT,
                    )
                _stage6_loop_pipeline_qkv(
                    v_new=v_new,
                    gated_qk_workspace=gated_qk_workspace,
                    qkv_workspace=qkv_workspace,
                    bos=bos,
                    sequence_tokens=sequence_tokens,
                    chunk_idx=chunk_idx,
                    head_idx=head_idx,
                    worker_idx=worker_idx,
                    buffer_id=buffer_id,
                    H=H,
                    V=V,
                    BT=BT,
                )
                _stage6_loop_pipeline_finalize(
                    o=o,
                    qkv_workspace=qkv_workspace,
                    b_qh_scaled=b_qh_scaled,
                    scale=scale,
                    bos=bos,
                    sequence_tokens=sequence_tokens,
                    chunk_idx=chunk_idx,
                    head_idx=head_idx,
                    worker_idx=worker_idx,
                    buffer_id=buffer_id,
                    H=H,
                    V=V,
                    BT=BT,
                )
                db_flag += 1
        global_task_base += sequence_tasks
    al.sync_block_wait(
        "vector",
        "cube",
        0,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )
    al.sync_block_wait(
        "vector",
        "cube",
        1,
        sender_pipe=al.PIPE.PIPE_MTE2,
        receiver_pipe=al.PIPE.PIPE_FIX,
    )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "TOTAL_TASKS", "scale"])
def chunk_gated_delta_rule_fwd_stage5_6_kernel(
    q,
    k,
    u,
    w,
    g,
    h,
    v_new,
    h0,
    ht,
    o,
    cu_seqlens,
    state_workspace,
    qh_workspace,
    wh_workspace,
    gated_v_workspace,
    kv_workspace,
    scale,
    T,
    TOTAL_TASKS,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr = False,
    STORE_FINAL_STATE: tl.constexpr = False,
    IS_VARLEN: tl.constexpr = False,
    STAGE6_HEAD_PAIR: tl.constexpr = False,
    STAGE5_BALANCE_VARLEN_N: tl.constexpr = 0,
):
    BK: tl.constexpr = 64
    BV: tl.constexpr = 128
    worker_idx = tl.program_id(1)
    num_workers = tl.num_programs(1)
    total_tokens = 1 * T
    offs_k = tl.arange(0, BK)
    offs_k_row = offs_k[:, None]
    offs_v = tl.arange(0, BV)[None, :]
    offs_t_row = tl.arange(0, BT)[:, None]
    state_worker_base = state_workspace + worker_idx * K * V
    wh_worker_base = wh_workspace + worker_idx * BT * V
    gated_v_worker_base = gated_v_workspace + worker_idx * BT * V
    kv_worker_base = kv_workspace + worker_idx * K * V
    if STAGE5_BALANCE_VARLEN_N == 2:
        sequence_tokens0 = tl.load(cu_seqlens + 1) - tl.load(cu_seqlens)
        sequence_tokens1 = tl.load(cu_seqlens + 2) - tl.load(cu_seqlens + 1)
        sequence1_is_long = sequence_tokens1 > sequence_tokens0
        long_sequence_idx = tl.where(sequence1_is_long, 1, 0)
        short_sequence_idx = 1 - long_sequence_idx
        long_sequence_tokens = tl.where(
            sequence1_is_long, sequence_tokens1, sequence_tokens0
        )
        short_sequence_tokens = tl.where(
            sequence1_is_long, sequence_tokens0, sequence_tokens1
        )
        flat_critical_tokens = long_sequence_tokens + short_sequence_tokens
        balanced_critical_tokens = max(long_sequence_tokens, 2 * short_sequence_tokens)
        should_balance_stage5 = (
            100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        )
    elif STAGE5_BALANCE_VARLEN_N == 3:
        sequence_tokens0 = tl.load(cu_seqlens + 1) - tl.load(cu_seqlens)
        sequence_tokens1 = tl.load(cu_seqlens + 2) - tl.load(cu_seqlens + 1)
        sequence_tokens2 = tl.load(cu_seqlens + 3) - tl.load(cu_seqlens + 2)
        swap01 = sequence_tokens1 > sequence_tokens0
        high_tokens = tl.where(swap01, sequence_tokens1, sequence_tokens0)
        high_idx = tl.where(swap01, 1, 0)
        low_tokens = tl.where(swap01, sequence_tokens0, sequence_tokens1)
        low_idx = tl.where(swap01, 0, 1)
        swap12 = sequence_tokens2 > low_tokens
        middle_tokens = tl.where(swap12, sequence_tokens2, low_tokens)
        middle_idx = tl.where(swap12, 2, low_idx)
        short_sequence_idx = tl.where(swap12, low_idx, 2)
        swap_top = middle_tokens > high_tokens
        long_sequence_idx = tl.where(swap_top, middle_idx, high_idx)
        middle_sequence_idx = tl.where(swap_top, high_idx, middle_idx)
        long_sequence_tokens = tl.where(swap_top, middle_tokens, high_tokens)
        middle_sequence_tokens = tl.where(swap_top, high_tokens, middle_tokens)
        short_sequence_tokens = min(low_tokens, sequence_tokens2)
        flat_critical_tokens = long_sequence_tokens + middle_sequence_tokens
        balanced_critical_tokens = max(
            long_sequence_tokens + short_sequence_tokens, 2 * middle_sequence_tokens
        )
        should_balance_stage5 = (
            100 * flat_critical_tokens >= 105 * balanced_critical_tokens
        )
    for sequence_head_idx in range(worker_idx, TOTAL_TASKS, num_workers):
        scheduled_sequence_head_idx = sequence_head_idx
        if STAGE5_BALANCE_VARLEN_N == -1:
            scheduled_sequence_head_idx = _schedule_grouped_varlen_stage5_task(
                cu_seqlens, sequence_head_idx, TOTAL_TASKS, H=H
            )
        elif STAGE5_BALANCE_VARLEN_N == 2:
            if should_balance_stage5:
                template_task_idx = sequence_head_idx
                head_group_base = 0
                if H == 32:
                    head_group_idx = sequence_head_idx // 32
                    template_task_idx = sequence_head_idx - head_group_idx * 32
                    head_group_base = head_group_idx * 16
                is_long_task = (template_task_idx >= 8) & (template_task_idx < 24)
                long_head_idx = template_task_idx - 8
                short_head_idx = tl.where(
                    template_task_idx < 8,
                    template_task_idx,
                    template_task_idx - 16,
                )
                scheduled_sequence_head_idx = tl.where(
                    is_long_task,
                    long_sequence_idx * H + head_group_base + long_head_idx,
                    short_sequence_idx * H + head_group_base + short_head_idx,
                )
        elif STAGE5_BALANCE_VARLEN_N == 3:
            if should_balance_stage5:
                template_task_idx = sequence_head_idx
                head_group_base = 0
                if H == 32:
                    head_group_idx = sequence_head_idx // 48
                    template_task_idx = sequence_head_idx - head_group_idx * 48
                    head_group_base = head_group_idx * 16
                is_long_task = template_task_idx < 16
                is_middle_task = (template_task_idx >= 16) & (
                    template_task_idx < 24
                ) | (template_task_idx >= 40)
                middle_head_idx = tl.where(
                    template_task_idx < 24,
                    template_task_idx - 16,
                    template_task_idx - 32,
                )
                short_head_idx = template_task_idx - 24
                scheduled_sequence_head_idx = tl.where(
                    is_long_task,
                    long_sequence_idx * H + head_group_base + template_task_idx,
                    tl.where(
                        is_middle_task,
                        middle_sequence_idx * H + head_group_base + middle_head_idx,
                        short_sequence_idx * H + head_group_base + short_head_idx,
                    ),
                )
        sequence_idx = scheduled_sequence_head_idx // H
        head_idx = scheduled_sequence_head_idx % H
        kv_head_idx = head_idx // (H // Hg)
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + sequence_idx).to(tl.int32)
            eos = tl.load(cu_seqlens + sequence_idx + 1).to(tl.int32)
            sequence_tokens = eos - bos
            num_chunks = tl.cdiv(sequence_tokens, BT)
            chunk_base = bos // BT + sequence_idx
        else:
            bos = sequence_idx * T
            sequence_tokens = T
            num_chunks = tl.cdiv(sequence_tokens, BT)
            chunk_base = sequence_idx * num_chunks
        stride_k = Hg * K
        stride_u = H * V
        stride_w = H * K
        k_base = k + bos * Hg * K + kv_head_idx * K
        u_base = u + bos * H * V + head_idx * V
        w_base = w + bos * H * K + head_idx * K
        v_new_base = v_new + bos * H * V + head_idx * V
        if USE_G:
            if IS_VARLEN:
                g_base = g + bos + head_idx * total_tokens
            else:
                g_base = g + (sequence_idx * H + head_idx) * total_tokens
        if USE_INITIAL_STATE:
            h0_base = h0 + scheduled_sequence_head_idx * K * V
            b_h_k0 = tl.load(h0_base + offs_k_row * V + offs_v).to(tl.float32)
            b_h_k1 = tl.load(h0_base + (BK + offs_k_row) * V + offs_v).to(tl.float32)
        else:
            b_h_k0 = tl.zeros([BK, BV], dtype=tl.float32)
            b_h_k1 = tl.zeros([BK, BV], dtype=tl.float32)
        p_state_k0 = tl.make_block_ptr(
            state_worker_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
        )
        p_state_k1 = tl.make_block_ptr(
            state_worker_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
        )
        tl.store(p_state_k0, b_h_k0.to(p_state_k0.dtype.element_ty))
        tl.store(p_state_k1, b_h_k1.to(p_state_k1.dtype.element_ty))
        first_h_base = h + (chunk_base * H + head_idx) * K * V
        p_first_h_k0 = tl.make_block_ptr(
            first_h_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
        )
        p_first_h_k1 = tl.make_block_ptr(
            first_h_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
        )
        tl.store(p_first_h_k0, b_h_k0.to(p_first_h_k0.dtype.element_ty))
        tl.store(p_first_h_k1, b_h_k1.to(p_first_h_k1.dtype.element_ty))
        al.sync_block_set(
            "vector",
            "cube",
            3,
            sender_pipe=al.PIPE.PIPE_MTE3,
            receiver_pipe=al.PIPE.PIPE_MTE2,
        )
        for chunk_idx in range(num_chunks):
            token_start = chunk_idx * BT
            valid_t_row = offs_t_row + token_start < sequence_tokens
            al.sync_block_wait(
                "vector",
                "cube",
                3,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            p_state_full = tl.make_block_ptr(
                state_worker_base, (K, V), (V, 1), (0, 0), (K, BV), (1, 0)
            )
            p_w_full = tl.make_block_ptr(
                w_base,
                (sequence_tokens, K),
                (stride_w, 1),
                (token_start, 0),
                (BT, K),
                (1, 0),
            )
            b_state_full = tl.load(p_state_full)
            b_w_full = tl.load(p_w_full, boundary_check=(0, 1))
            b_wh = tl.dot(b_w_full, b_state_full)
            p_wh = tl.make_block_ptr(
                wh_worker_base, (BT, V), (V, 1), (0, 0), (BT, BV), (1, 0)
            )
            tl.store(p_wh, b_wh.to(p_wh.dtype.element_ty))
            al.sync_block_set(
                "cube",
                "vector",
                0,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_u = tl.load(
                u_base + (token_start + offs_t_row) * stride_u + offs_v,
                mask=valid_t_row,
                other=0.0,
            ).to(tl.float32)
            if USE_G:
                token_offsets = token_start + tl.arange(0, BT)
                valid_tokens = token_offsets < sequence_tokens
                last_idx = min(token_start + BT, sequence_tokens) - 1
                b_g_last_raw = tl.load(g_base + last_idx)
                b_g = tl.load(g_base + token_offsets, mask=valid_tokens, other=0.0)
                b_decay = safe_exp(b_g_last_raw - b_g)
                b_state_decay = tl.exp(b_g_last_raw)
            al.sync_block_wait(
                "cube",
                "vector",
                0,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_wh = tl.load(p_wh).to(tl.float32)
            b_v_new = b_u - b_wh
            p_v_new = tl.make_block_ptr(
                v_new_base,
                (sequence_tokens, V),
                (stride_u, 1),
                (token_start, 0),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0,))
            if USE_G:
                b_h_k0 *= b_state_decay
                b_h_k1 *= b_state_decay
            p_k_full = tl.make_block_ptr(
                k_base,
                (K, sequence_tokens),
                (1, stride_k),
                (0, token_start),
                (K, BT),
                (0, 1),
            )
            b_k_full = tl.load(p_k_full, boundary_check=(0, 1))
            if USE_G:
                b_gated_v = b_v_new * b_decay[:, None]
            else:
                b_gated_v = b_v_new
            p_gated_v = tl.make_block_ptr(
                gated_v_worker_base, (BT, V), (V, 1), (0, 0), (BT, BV), (1, 0)
            )
            tl.store(p_gated_v, b_gated_v.to(p_gated_v.dtype.element_ty))
            al.sync_block_set(
                "vector",
                "cube",
                1,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            al.sync_block_wait(
                "vector",
                "cube",
                1,
                sender_pipe=al.PIPE.PIPE_MTE3,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_v_new_cube = tl.load(p_gated_v)
            p_kv_k0 = tl.make_block_ptr(
                kv_worker_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
            )
            p_kv_k1 = tl.make_block_ptr(
                kv_worker_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
            )
            p_kv_full = tl.make_block_ptr(
                kv_worker_base, (K, V), (V, 1), (0, 0), (K, BV), (1, 0)
            )
            b_kv_full = tl.dot(b_k_full, b_v_new_cube)
            tl.store(p_kv_full, b_kv_full.to(p_kv_full.dtype.element_ty))
            al.sync_block_set(
                "cube",
                "vector",
                2,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            al.sync_block_wait(
                "cube",
                "vector",
                2,
                sender_pipe=al.PIPE.PIPE_FIX,
                receiver_pipe=al.PIPE.PIPE_MTE2,
            )
            b_h_k0 += tl.load(p_kv_k0).to(tl.float32)
            b_h_k1 += tl.load(p_kv_k1).to(tl.float32)
            if chunk_idx + 1 < num_chunks:
                tl.store(p_state_k0, b_h_k0.to(p_state_k0.dtype.element_ty))
                tl.store(p_state_k1, b_h_k1.to(p_state_k1.dtype.element_ty))
                next_h_base = h + ((chunk_base + chunk_idx + 1) * H + head_idx) * K * V
                p_next_h_k0 = tl.make_block_ptr(
                    next_h_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
                )
                p_next_h_k1 = tl.make_block_ptr(
                    next_h_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
                )
                tl.store(p_next_h_k0, b_h_k0.to(p_next_h_k0.dtype.element_ty))
                tl.store(p_next_h_k1, b_h_k1.to(p_next_h_k1.dtype.element_ty))
            if chunk_idx + 1 < num_chunks:
                al.sync_block_set(
                    "vector",
                    "cube",
                    3,
                    sender_pipe=al.PIPE.PIPE_MTE3,
                    receiver_pipe=al.PIPE.PIPE_MTE2,
                )
        if STORE_FINAL_STATE:
            ht_base = ht + scheduled_sequence_head_idx * K * V
            p_ht_k0 = tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
            )
            p_ht_k1 = tl.make_block_ptr(
                ht_base, (K, V), (V, 1), (BK, 0), (BK, BV), (1, 0)
            )
            tl.store(p_ht_k0, b_h_k0)
            tl.store(p_ht_k1, b_h_k1)
    al.sync_block_all("all", 0)
    if STAGE6_HEAD_PAIR:
        _run_stage6_head_pair_loop_pipeline(
            q=q,
            k=k,
            v_new=v_new,
            h=h,
            g=g,
            o=o,
            cu_seqlens=cu_seqlens,
            qk_workspace=wh_workspace,
            qh_workspace=qh_workspace,
            gated_qk_workspace=gated_v_workspace,
            qkv_workspace=kv_workspace,
            scale=scale,
            T=T,
            TOTAL_TASKS=TOTAL_TASKS,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            USE_G=USE_G,
            IS_VARLEN=IS_VARLEN,
        )
    else:
        stage6_qkv_workspace = kv_workspace
        _run_stage6_loop_pipeline(
            q=q,
            k=k,
            v_new=v_new,
            h=h,
            g=g,
            o=o,
            cu_seqlens=cu_seqlens,
            qk_workspace=wh_workspace,
            qh_workspace=qh_workspace,
            gated_qk_workspace=gated_v_workspace,
            qkv_workspace=stage6_qkv_workspace,
            scale=scale,
            T=T,
            TOTAL_TASKS=TOTAL_TASKS,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            USE_G=USE_G,
            IS_VARLEN=IS_VARLEN,
        )


def _use_stage6_head_pair(
    *, total_tokens: int, num_sequences: int, num_heads: int, num_kv_heads: int
) -> bool:
    if num_heads != 2 * num_kv_heads:
        return False
    if num_sequences > 1:
        return total_tokens >= 512
    return total_tokens >= 1024 or (total_tokens >= 256 and total_tokens % 64 == 0)


def _chunk_gated_delta_rule_fwd_stage5_6_with_intermediates(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    B, T, Hg, K = q.shape
    H, V = u.shape[-2:]
    BT = chunk_size
    if K != 128 or V != 128 or BT != 64:
        raise ValueError(
            "chunk_gated_delta_rule_fwd Stage5+6 requires K=128, V=128, and chunk_size=64"
        )
    if H % Hg != 0:
        raise ValueError(
            "the number of query heads must be divisible by the number of KV heads"
        )
    if scale is None:
        scale = K**-0.5

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    num_tasks = N * H
    num_workers = _get_aic_num()
    use_head_pair = _use_stage6_head_pair(
        total_tokens=T,
        num_sequences=N,
        num_heads=H,
        num_kv_heads=Hg,
    )
    balance_varlen_n = 0
    if cu_seqlens is not None and H in (16, 32) and num_workers == 24 and N >= 2:
        balance_varlen_n = N if N in (2, 3) else -1

    chunk_slots = B * triton.cdiv(T, BT) if cu_seqlens is None else (T - 1) // BT + N
    h = k.new_empty(1, chunk_slots, H, K, V)
    v_new = torch.empty_like(u)
    o = torch.empty_like(u)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    g_head_first = g.transpose(1, 2).contiguous() if g is not None else None

    # NOTE: state_workspace (stage5 chunk state) and qh_workspace (stage6
    # double-buffered q@h slots) are deliberately kept as two separate kernel
    # arguments. Reusing one aliased tensor for both makes the CANN 9.1
    # TileAndBindSubBlock same-address check (areLoadAndStoreSameAddress)
    # bail out to limitAllAivToSubBlock0, which later overflows UB.
    state_workspace = k.new_empty(num_workers, K, V)
    qh_workspace = k.new_empty(num_workers, 4 * BT if use_head_pair else K, V)
    wh_workspace = k.new_empty(num_workers, BT, V)
    gated_v_workspace = k.new_empty(num_workers, K if use_head_pair else BT, V)
    kv_workspace = k.new_empty(num_workers, 2 * K if use_head_pair else K, V)

    def grid(meta):
        return (1, num_workers)

    chunk_gated_delta_rule_fwd_stage5_6_kernel[grid](
        q=q,
        k=k,
        u=u,
        w=w,
        g=g_head_first,
        h=h,
        v_new=v_new,
        h0=initial_state,
        ht=final_state,
        o=o,
        cu_seqlens=cu_seqlens,
        state_workspace=state_workspace,
        qh_workspace=qh_workspace,
        wh_workspace=wh_workspace,
        gated_v_workspace=gated_v_workspace,
        kv_workspace=kv_workspace,
        scale=scale,
        T=T,
        TOTAL_TASKS=num_tasks,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        STAGE6_HEAD_PAIR=use_head_pair,
        STAGE5_BALANCE_VARLEN_N=balance_varlen_n,
        num_warps=4,
        num_stages=2,
        enable_sync_block_lock=True,
        multibuffer=False,
        disable_auto_inject_block_sync=True,
    )
    return o, final_state, h, v_new


_GDR_CHUNK_SIZE = 64
_GDR_HEAD_DIM = 128
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.ndim != 4:
        raise ValueError(f"q must be 4D, got shape {tuple(q.shape)}")
    if q.device.type != "npu":
        raise NotImplementedError(
            "Ascend chunk_gated_delta_rule_fwd is available only on NPU tensors"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            "Ascend chunk_gated_delta_rule_fwd supports only float16 and bfloat16"
        )

    B, T, Hg, K = q.shape
    if T <= 0 or Hg <= 0:
        raise ValueError(
            "Ascend chunk_gated_delta_rule_fwd requires positive sequence and head counts"
        )
    if K != _GDR_HEAD_DIM:
        raise NotImplementedError("Ascend chunk_gated_delta_rule_fwd requires K=128")

    device, dtype = q.device, q.dtype
    _validate_tensor(k, name="k", ndim=4, device=device, dtype=dtype)
    _validate_tensor(v, name="v", ndim=4, device=device, dtype=dtype)
    # The scalar gate follows the serving convention and stays in float32
    # (vllm-ascend pass an fp32 gate); the compute dtype is also
    # accepted. Both flow into the fp32 chunk-local cumsum either way.
    if not isinstance(g, torch.Tensor):
        raise TypeError("g must be a torch.Tensor")
    if g.ndim != 3:
        raise ValueError(f"g must be 3D, got shape {tuple(g.shape)}")
    if g.device != device:
        raise ValueError(f"g must be on {device}, got {g.device}")
    if g.dtype not in (dtype, torch.float32):
        raise ValueError(f"g must have dtype {dtype} or float32, got {g.dtype}")
    _validate_tensor(beta, name="beta", ndim=3, device=device, dtype=dtype)

    if k.shape != q.shape:
        raise ValueError("q and k must have identical [B, T, Hg, K] shapes")
    if v.shape[:2] != (B, T):
        raise ValueError("v must have the same batch and sequence dimensions as q")
    H, V = v.shape[2:]
    if V != _GDR_HEAD_DIM:
        raise NotImplementedError("Ascend chunk_gated_delta_rule_fwd requires V=128")
    if H <= 0 or H % Hg != 0:
        raise ValueError("the q/k head count must divide the value head count")
    if g.shape != (B, T, H) or beta.shape != (B, T, H):
        raise ValueError(f"g and beta must have shape {(B, T, H)}")

    N = B
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a 1D tensor with at least two entries")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if not cu_seqlens.is_contiguous():
            raise NotImplementedError(
                "Ascend chunk_gated_delta_rule_fwd requires contiguous cu_seqlens"
            )
        if B != 1:
            raise ValueError("packed variable-length inputs require B=1")
        N = cu_seqlens.numel() - 1

    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor")
        if initial_state.device != device:
            raise ValueError("initial_state must be on the same device as q")
        if initial_state.dtype not in (dtype, torch.float32):
            raise ValueError("initial_state must use the input dtype or float32")
        if initial_state.shape != (N, H, K, V):
            raise ValueError(
                f"initial_state must have shape {(N, H, K, V)}, "
                f"got {tuple(initial_state.shape)}"
            )
        if not initial_state.is_contiguous():
            raise NotImplementedError(
                "Ascend chunk_gated_delta_rule_fwd requires contiguous initial_state"
            )

    return B, T, H, N


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """
    This backend is specialized for chunk size 64 and K=V=128. The source
    preprocessing kernel does not expose the KKT inverse ``A`` expected by the
    generic FLA seven-value protocol, so the third result is intentionally
    ``None`` rather than an unrelated intermediate.
    """

    _validate_inputs(q, k, v, g, beta, initial_state, cu_seqlens)
    if not isinstance(output_final_state, bool):
        raise TypeError("output_final_state must be a bool")
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise ValueError("scale must be a finite scalar")

    w, u, g_cumsum = chunk_gated_delta_rule_fwd_stage0_4(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=_GDR_CHUNK_SIZE,
    )
    o, final_state, h, v_new = _chunk_gated_delta_rule_fwd_stage5_6_with_intermediates(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        scale=float(scale),
        chunk_size=_GDR_CHUNK_SIZE,
    )

    if SUPPRESS_LEVEL < 3:
        return g_cumsum, o, None, final_state, None, None, None
    return g_cumsum, o, None, final_state, w, h, v_new
