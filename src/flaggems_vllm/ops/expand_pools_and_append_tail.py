# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand selected pool indices and append the incomplete token tail."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_pools_and_append_tail_kernel(
    pool_ids_ptr,  # [rows, n_groups], int (any int dtype)
    seq_lens_ptr,  # [rows], int32 (token-granular seq_len)
    out_ptr,  # [rows, out_cols], int32
    topk,  # n_groups * pool_size
    out_cols,  # topk + pool_size - 1
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    pid_s0,
    out_s0,
):
    # Fuses expand_pools_to_tokens + append_tail_to_topk (identity path) into a
    # single kernel. Each program writes one (row, column-tile) of the output.
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < out_cols

    seq_len = tl.load(seq_lens_ptr + row)
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start  # in [0, POOL_SIZE)

    # History region [0, topk): expand selected pool g = cols // POOL_SIZE.
    is_history = cols < topk
    g = cols // POOL_SIZE
    o = cols % POOL_SIZE
    pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
    hist_val = (pid * POOL_SIZE + o).to(tl.int32)
    hist_out = tl.where(pid >= 0, hist_val, -1)

    # Tail region [topk, out_cols): the request's trailing incomplete pool.
    tail_off = cols - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_val = (tail_start + tail_off).to(tl.int32)
    tail_out = tl.where(is_tail, tail_val, -1)

    result = tl.where(is_history, hist_out, tail_out)
    tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Fuse ``expand_pools_to_tokens`` + ``append_tail_to_topk`` (identity path).

    Produces the same ``[rows, topk + pool_size - 1]`` int32 output as calling
    the two functions in sequence when neither ``page_table`` nor
    ``topk_offsets`` is passed — the only path the GLM5Next indexer exercises.
    The kernel derives ``pool_len = seq_len // pool_size`` internally, so the
    caller no longer needs to precompute it. Replaces ~25 elementwise kernels
    with one Triton launch.
    """
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    if pool_ids.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("pool_ids must be rank 2 and seq_lens rank 1")
    if seq_lens.shape[0] != pool_ids.shape[0]:
        raise ValueError("pool_ids and seq_lens must have the same number of rows")
    if pool_ids.dtype not in (torch.int32, torch.int64) or seq_lens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("pool_ids and seq_lens must have int32 or int64 dtype")
    if pool_ids.device != seq_lens.device or pool_ids.device.type != "cuda":
        raise NotImplementedError(
            "pool expansion requires inputs on the same CUDA device"
        )
    if pool_ids.stride(1) != 1 or seq_lens.stride(0) != 1:
        raise ValueError("pool_ids columns and seq_lens must have unit stride")
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)
    if rows == 0 or out_cols == 0:
        return out
    BLOCK_COLS = 128
    n_tiles = triton.cdiv(out_cols, BLOCK_COLS)
    _expand_pools_and_append_tail_kernel[(rows, n_tiles)](
        pool_ids,
        seq_lens,
        out,
        topk,
        out_cols,
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
        pid_s0=pool_ids.stride(0),
        out_s0=out.stride(0),
    )
    return out
