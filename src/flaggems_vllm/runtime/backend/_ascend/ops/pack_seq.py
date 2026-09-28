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

import logging

import torch
import torch_npu
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
    PER_CORE_ROWS: tl.constexpr,
    LAST_CORE_ROWS: tl.constexpr,
    GRID_T: tl.constexpr,
    GRID_D: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_d = pid % GRID_D  # block over feature dimension
    pid_t = (pid // GRID_D) % GRID_T  # block over time dimension
    pid_b = pid // (GRID_T * GRID_D)  # batch id
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)

    row_len = LAST_CORE_ROWS if pid == tl.num_programs(0) - 1 else PER_CORE_ROWS
    for row_idx in tl.range(row_len):
        off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

        # Compute start index and sequence length from cumulative lengths.
        seq_len = tl.load(lengths_ptr + pid_b)

        # valid time positions for this block
        t_mask = off_t < Lmax

        # compute input row indices for valid (b, t)
        in_row = in_start + off_t
        valid_row = off_t < seq_len

        # Pointers
        # x_ptr: row-major [N, D]
        x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

        # out_ptr: row-major [B, Lmax, D]
        out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

        # Initialize with PAD. PAD_IS_UINT8 selects the pad tensor's dtype so
        # integer-typed outputs (e.g. MXFP4 packed nibbles, ue8m0 scale bytes)
        # get an exact-byte pad rather than going through an fp32->uint8 cast
        # that's implementation-defined outside of value 0.
        d_mask = off_d[None, :] < D
        if PAD_IS_UINT8:
            pad_val = tl.full((), PAD_VALUE, tl.uint8)
        else:
            pad_val = tl.full((), PAD_VALUE, tl.float32)

        x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask, other=pad_val)
        tl.store(out_row_ptr, x_vals, mask=t_mask[:, None] & d_mask)

        carry_d = pid_d == GRID_D - 1
        carry_dt = carry_d & (pid_t == GRID_T - 1)
        new_pid_d = tl.where(carry_d, 0, pid_d + 1)
        new_pid_t = tl.where(carry_dt, 0, tl.where(carry_d, pid_t + 1, pid_t))
        new_pid_b = tl.where(carry_dt, pid_b + 1, pid_b)
        new_in_start = tl.where(carry_dt, in_start + seq_len, in_start)
        pid_d = new_pid_d
        pid_t = new_pid_t
        pid_b = new_pid_b
        in_start = new_in_start


def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    logger.debug("GEMS PACK_SEQ_TRITON")
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert (
            isinstance(pad_value, int) and 0 <= pad_value <= 255
        ), f"uint8 pack requires an integer pad in [0, 255], got {pad_value!r}"
        pad_constexpr: int | float = int(pad_value)
    else:
        pad_constexpr = float(pad_value)

    original_shape = x.shape
    if len(original_shape) > 2:
        N = original_shape[0]
        x_reshaped = x.reshape(N, -1)
        D = x_reshaped.shape[1]
    else:
        N, D = x.shape
        x_reshaped = x

    B = lengths.numel()
    Lmax = int(lengths.max().item())

    out = torch.empty((B, Lmax, D), device=x.device, dtype=x.dtype)

    grid_t = triton.cdiv(Lmax, block_t)
    grid_d = triton.cdiv(D, block_d)
    grid = B * grid_t * grid_d
    vector_core_num = torch_npu.npu.get_device_properties(x.device).vector_core_num
    per_core_rows = (grid + vector_core_num - 1) // vector_core_num
    need_core_num = (grid + per_core_rows - 1) // per_core_rows
    last_core_rows = (
        per_core_rows if grid % per_core_rows == 0 else grid % per_core_rows
    )

    _pack_seq_kernel[(need_core_num,)](
        x_reshaped,
        out,
        lengths.int(),
        N,
        D,
        Lmax,
        PAD_VALUE=pad_constexpr,
        PAD_IS_UINT8=is_uint8,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        PER_CORE_ROWS=per_core_rows,
        LAST_CORE_ROWS=last_core_rows,
        GRID_T=grid_t,
        GRID_D=grid_d,
    )

    if len(original_shape) > 2:
        out = out.reshape((B, Lmax) + original_shape[1:])

    return out
