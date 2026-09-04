# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License IS DISTRIBUTED ON AN "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend-optimized add + RMSNorm kernel, ported from the tutorial kernel.

The tutorial NPU strategy is kept as-is: core splitting (one program per
vector core, each looping over its rows), weight preload into registers and
multibuffer launch options. Only the contract is adapted to the generic
``add_rms_norm``: the bias is dropped and the result goes to a newly
allocated output tensor instead of being written back in-place.


A single block covers a whole row, which overflows the Ascend unified buffer
for wide rows (N=65536 fails to compile), so N > 4096 falls back to a
column-loop kernel. Like the tutorial, M is capped at 65535.
"""

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Rows wider than this overflow the Ascend unified buffer as a single block;
# they take the column-loop kernel instead.
_BLOCK_N = 4096

_CACHED_CORE_NUM = None


def _get_core_num():
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        try:
            import torch_npu  # noqa: F401

            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            _CACHED_CORE_NUM = cores_dict["vector_core_num"]
        except (ImportError, AttributeError, KeyError, TypeError):
            _CACHED_CORE_NUM = None
    return _CACHED_CORE_NUM


def _get_rows_and_cores(M, cores):
    if M <= cores:
        return 1, M

    num_rows = (M + cores - 1) // cores
    use_cores = (M + num_rows - 1) // num_rows
    return num_rows, use_cores


@triton.jit(do_not_specialize=["eps"])
def add_rms_norm_kernel(
    Y,  # pointer to the output
    X1,  # pointer to the first input
    X2,  # pointer to the second input
    W,  # pointer to the weight
    MAX_ROWS,  # maximum number of rows to process
    eps,  # epsilon to avoid division by zero
    N_ROWS,  # number of rows to process in one core
    N,  # real number of columns in X1/X2
    NUM_COLUMNS: tl.constexpr,  # padded power-of-two column count
):
    pid = tl.program_id(0)
    Y += pid * N * N_ROWS
    X1 += pid * N * N_ROWS
    X2 += pid * N * N_ROWS

    # preload the weight once per core; it is reused by every row below
    col_off = tl.arange(0, NUM_COLUMNS)
    mask = col_off < N
    w = tl.load(W + col_off, mask, other=0.0)

    base_row = pid * N_ROWS
    rows = min(base_row + N_ROWS, MAX_ROWS) - base_row
    for row_off in tl.range(0, rows, 1):
        cols = row_off * N + col_off
        x1 = tl.load(X1 + cols, mask, other=0.0)
        x2 = tl.load(X2 + cols, mask, other=0.0)
        x = (x1 + x2).to(tl.float32)

        _var_base = (x * x) / N
        var = tl.sum(_var_base)

        rrms = 1 / tl.sqrt(var + eps)

        y = (x * rrms * w).to(Y.dtype.element_ty)
        # write back to the output
        tl.store(Y + cols, y, mask)


@triton.jit(do_not_specialize=["eps"])
def add_rms_norm_loop_kernel(
    Y,  # pointer to the output
    X1,  # pointer to the first input
    X2,  # pointer to the second input
    W,  # pointer to the weight
    N,  # number of columns in X1/X2
    eps,  # epsilon to avoid division by zero
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * N

    # Pass 1: accumulate the sum of squares over the whole row
    var = 0.0
    for start in tl.range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x1 = tl.load(X1 + base + offs, mask, other=0.0)
        x2 = tl.load(X2 + base + offs, mask, other=0.0)
        x = (x1 + x2).to(tl.float32)
        var += tl.sum(x * x)

    rrms = 1 / tl.sqrt(var / N + eps)

    # Pass 2: reload and write the normalized output
    for start in tl.range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x1 = tl.load(X1 + base + offs, mask, other=0.0)
        x2 = tl.load(X2 + base + offs, mask, other=0.0)
        x = (x1 + x2).to(tl.float32)
        w = tl.load(W + offs, mask, other=0.0)
        y = (x * rrms * w).to(Y.dtype.element_ty)
        tl.store(Y + base + offs, y, mask)


def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
    """
    Add_RMSNorm: Add two inputs element-wise and apply RMS normalization.

    The output is written to a newly allocated tensor; the inputs are not
    modified (same contract as the generic ``flaggems_vllm.add_rms_norm``).
    """
    logger.debug(
        "GEMS_ASCEND ADD_RMS_NORM FORWARD, [input1 shape]: %s, [input2 shape]: %s, [weight shape]: %s",
        x1.size(),
        x2.size(),
        weight.size(),
    )
    dim = x1.ndim - len(normalized_shape)
    M = min(math.prod(x1.shape[:dim]), 65535)
    N = math.prod(normalized_shape)

    # Verify shapes match
    assert x1.shape == x2.shape, f"Input shapes must match: {x1.shape} vs {x2.shape}"

    x1 = x1.contiguous()
    x2 = x2.contiguous()
    weight = weight.contiguous()
    y = torch.empty_like(x1)

    cores = _get_core_num()
    CORES = 24 if cores is None else cores

    if N <= _BLOCK_N:
        NUM_COLUMNS = triton.next_power_of_2(N)

        N_ROWS, USE_CORES = _get_rows_and_cores(M, CORES)

        add_rms_norm_kernel[(USE_CORES,)](
            y,
            x1,
            x2,
            weight,
            M,
            eps=eps,
            N_ROWS=N_ROWS,
            N=N,
            NUM_COLUMNS=NUM_COLUMNS,
            multibuffer=True,
            limit_auto_multi_buffer_only_for_local_buffer=False,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        )
    else:
        add_rms_norm_loop_kernel[(M,)](
            y,
            x1,
            x2,
            weight,
            N,
            eps=eps,
            BLOCK=_BLOCK_N,
        )
    return y
