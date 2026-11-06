# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand selected pool indices and append the incomplete token tail."""

import logging
import threading

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)
_INT32_MAX = 2**31 - 1
_INTEGER_DTYPES = (torch.int32, torch.int64)
_LAUNCH_CACHE_LIMIT = 128
_launch_cache = {}
_launch_cache_lock = threading.Lock()
_HCU_COMPILER = None
if runtime.device.vendor_name == "hygon" and hasattr(triton.backends, "hcu"):
    from triton.backends.hcu import compiler as _HCU_COMPILER


def _expand_grid(meta):
    block_cols = meta["BLOCK_COLS"]
    return (meta["rows"], (meta["out_cols"] + block_cols - 1) // block_cols)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("expand_pools_and_append_tail"),
    key=["rows", "topk", "out_cols", "POOL_SIZE", "pid_s0"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
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
    rows,
):
    # Fuses expand_pools_to_tokens + append_tail_to_topk (identity path) into a
    # single kernel. Each program writes one (row, column-tile) of the output.
    row = tl.program_id(0).to(tl.int64)
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


# Match LibEntry's inspected Triton 3.6 compiled-kernel argument convention.
_POINTER_ALIGNMENT = _expand_pools_and_append_tail_kernel.divisibility
_CACHE_LAUNCH = (
    runtime.device.vendor_name in ("nvidia", "metax", "hygon")
    and triton.__version__.split(".")[:2] == ["3", "6"]
    and (
        runtime.device.vendor_name != "hygon"
        or callable(
            getattr(
                getattr(_HCU_COMPILER, "HIPBackend", None),
                "get_tensor_specialization",
                None,
            )
        )
    )
    and not _expand_pools_and_append_tail_kernel._has_flagtune_tuner
    and all(
        config.pre_hook is None
        for config in _expand_pools_and_append_tail_kernel.fn.configs
    )
)


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

    Both inputs may be int32 or int64. Columns and sequence lengths must have
    unit stride; rows may have padding. Values must satisfy ``seq_len >= 0``
    and all generated token indices must fit int32. Negative pool IDs denote
    missing pools. Values outside this domain are not checked on the host.
    """
    logger.debug("GEMS EXPAND_POOLS_AND_APPEND_TAIL")
    if not isinstance(pool_size, int) or isinstance(pool_size, bool) or pool_size < 1:
        raise ValueError("pool_size must be a positive integer")
    if pool_size > _INT32_MAX:
        raise ValueError("pool_size must fit int32")
    if pool_ids.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("pool_ids must be rank 2 and seq_lens rank 1")
    rows, n_groups = pool_ids.shape
    if seq_lens.shape[0] != rows:
        raise ValueError("pool_ids and seq_lens must have the same number of rows")
    if pool_ids.dtype not in _INTEGER_DTYPES or seq_lens.dtype not in _INTEGER_DTYPES:
        raise TypeError("pool_ids and seq_lens must have int32 or int64 dtype")
    device = pool_ids.device
    if (
        device != seq_lens.device
        or device.type != runtime.device.name
        or device.type == "cpu"
    ):
        raise NotImplementedError(
            "pool expansion requires inputs on the same active accelerator device"
        )
    if pool_ids.stride(1) != 1 or seq_lens.stride(0) != 1:
        raise ValueError("pool_ids columns and seq_lens must have unit stride")
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    if out_cols > _INT32_MAX:
        raise ValueError("expanded output width must fit int32")
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=device)
    if rows == 0 or out_cols == 0:
        return out
    row_stride = pool_ids.stride(0)
    launch = None
    if _CACHE_LAUNCH:
        key = (
            device,
            rows,
            topk,
            out_cols,
            pool_size,
            row_stride,
            pool_ids.dtype,
            seq_lens.dtype,
            pool_ids.data_ptr() % _POINTER_ALIGNMENT == 0,
            seq_lens.data_ptr() % _POINTER_ALIGNMENT == 0,
            out.data_ptr() % _POINTER_ALIGNMENT == 0,
        )
        if _HCU_COMPILER is not None:
            # Match LibEntry's storage-range specialization, including views.
            specialize = _HCU_COMPILER.HIPBackend.get_tensor_specialization
            key += tuple(specialize(value) for value in (pool_ids, seq_lens, out))
        launch = _launch_cache.get(key)
    with runtime.torch_device_fn.device(device):
        if launch is not None:
            compiled, grid, block_cols = launch
            compiled[grid](
                pool_ids,
                seq_lens,
                out,
                topk,
                out_cols,
                pool_size,
                block_cols,
                row_stride,
                out_cols,
                rows,
            )
        else:
            compiled, meta = _expand_pools_and_append_tail_kernel.run(
                pool_ids,
                seq_lens,
                out,
                topk,
                out_cols,
                POOL_SIZE=pool_size,
                pid_s0=row_stride,
                out_s0=out_cols,
                rows=rows,
                grid=_expand_grid,
                warmup=False,
            )
            if _CACHE_LAUNCH:
                block_cols = meta["BLOCK_COLS"]
                grid = (rows, triton.cdiv(out_cols, block_cols), 1)
                with _launch_cache_lock:
                    if len(_launch_cache) >= _LAUNCH_CACHE_LIMIT:
                        _launch_cache.pop(next(iter(_launch_cache)))
                    _launch_cache[key] = (compiled, grid, block_cols)
    return out
