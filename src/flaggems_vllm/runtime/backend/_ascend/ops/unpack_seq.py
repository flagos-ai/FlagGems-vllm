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
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
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

        # bounds: compute start from cumulative lengths
        seq_len = tl.load(lengths_ptr + pid_b)

        # valid time positions for this block
        valid_row = off_t < seq_len

        # compute output row indices for valid (b, t)
        out_row = in_start + off_t

        # Pointers
        # packed_ptr: row-major [B, Lmax, D]
        packed_row_ptr = (
            packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]
        )

        # out_ptr: row-major [N, D]
        out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

        # Load from packed tensor and store to output
        d_mask = off_d[None, :] < D
        packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
        tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)

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


def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor:
    logger.debug("GEMS UNPACK_SEQ_TRITON")
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        packed_reshaped = packed_tensor.reshape(B, Lmax, -1)
        D = packed_reshaped.shape[2]
    else:
        B, Lmax, D = packed_tensor.shape
        packed_reshaped = packed_tensor

    N = int(lengths.sum().item())

    out = torch.empty((N, D), device=packed_tensor.device, dtype=packed_tensor.dtype)

    grid_t = triton.cdiv(Lmax, block_t)
    grid_d = triton.cdiv(D, block_d)
    grid = B * grid_t * grid_d
    vector_core_num = torch_npu.npu.get_device_properties(
        packed_tensor.device
    ).vector_core_num
    per_core_rows = (grid + vector_core_num - 1) // vector_core_num
    need_core_num = (grid + per_core_rows - 1) // per_core_rows
    last_core_rows = (
        per_core_rows if grid % per_core_rows == 0 else grid % per_core_rows
    )

    _unpack_seq_triton_kernel[(need_core_num,)](
        packed_reshaped,
        out,
        lengths.int(),
        B,
        Lmax,
        D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        PER_CORE_ROWS=per_core_rows,
        LAST_CORE_ROWS=last_core_rows,
        GRID_T=grid_t,
        GRID_D=grid_d,
    )

    if len(original_shape) > 3:
        output_shape = (N,) + original_shape[2:]
        out = out.reshape(output_shape)

    return out
