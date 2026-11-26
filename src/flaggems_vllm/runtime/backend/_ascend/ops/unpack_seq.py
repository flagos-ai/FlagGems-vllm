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

from flaggems_vllm.ops.data_movement import contiguous_copy
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle_exp
    from triton.experimental.tle import tile_load, tile_store


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
    first_tile = pid * PER_CORE_ROWS
    pid_d = first_tile % GRID_D  # block over feature dimension
    pid_t = (first_tile // GRID_D) % GRID_T  # block over time dimension
    pid_b = first_tile // (GRID_T * GRID_D)  # batch id
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
    total_tokens: int | None = None,
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

    N = int(lengths.sum().item()) if total_tokens is None else total_tokens

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
        contiguous_copy(lengths, torch.int32),
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


_unpack_seq_generic = unpack_seq_triton

if HAS_TENSORVIEW:
    _TV_MAX_ROWS = 128

    @triton.jit
    def _unpack_seq_tv_kernel(
        out,
        packed,
        lengths,
        max_len,
        total_tokens,
        FEATURES: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        seq = tl.program_id(0)
        tile = tl.program_id(1)
        start = 0
        for i in range(seq):
            start += tl.load(lengths + i)
        length = tl.minimum(tl.load(lengths + seq), max_len)

        packed_view = tle_exp.make_strided_view(
            packed,
            shape=[(seq + 1) * max_len, FEATURES],
            strides=[FEATURES, 1],
            tile=[BLOCK_T, BLOCK_D],
            traversal_strides=[1, BLOCK_D],
        )
        rows = tile_load(
            packed_view,
            index=(seq * max_len + tile * BLOCK_T, 0),
            dst_space=tle_exp.language.dsa.ascend.UB,
        )
        # Limit writes both to this sequence and to the caller's output size.
        out_view = tle_exp.make_strided_view(
            out,
            shape=[tl.minimum(start + length, total_tokens), FEATURES],
            strides=[FEATURES, 1],
            tile=[BLOCK_T, BLOCK_D],
            traversal_strides=[1, BLOCK_D],
        )
        tile_store(
            out_view,
            value=rows,
            index=(start + tile * BLOCK_T, 0),
            src_space=tle_exp.language.dsa.ascend.UB,
        )

    def _unpack_tv_rows(features, element_size):
        return tile_rows_for_ub(
            triton.next_power_of_2(features) * element_size, 1, _TV_MAX_ROWS
        )

    def _unpack_tv_eligible(packed_tensor, lengths, total_tokens):
        # The inherited fallback synchronizes lengths if metadata is absent;
        # TensorView only runs with an explicit caller-provided token count.
        if total_tokens is None:
            return False
        if packed_tensor.ndim < 3 or packed_tensor.numel() == 0:
            return False
        if lengths.ndim != 1 or lengths.numel() != packed_tensor.shape[0]:
            return False
        if packed_tensor.device != lengths.device or lengths.device.type == "cpu":
            return False
        if lengths.dtype not in (torch.int32, torch.int64):
            return False
        num_seqs, max_len = packed_tensor.shape[:2]
        if num_seqs > 1024 or max_len == 0:
            return False
        if total_tokens <= 0:
            return False
        features = packed_tensor.numel() // (num_seqs * max_len)
        return (
            features > 0
            and _unpack_tv_rows(features, packed_tensor.element_size()) >= MIN_TILE_ROWS
        )

    def _unpack_seq_tensorview(packed_tensor, lengths, block_t, block_d, total_tokens):
        logger.debug("GEMS_ASCEND UNPACK_SEQ_TRITON(TENSORVIEW)")
        original_shape = packed_tensor.shape
        num_seqs, max_len = original_shape[:2]
        packed = packed_tensor.reshape(num_seqs, max_len, -1).contiguous()
        features = packed.shape[2]
        total = total_tokens
        lengths = contiguous_copy(lengths, torch.int32)
        rows = _unpack_tv_rows(features, packed.element_size())
        rows = min(rows, max(triton.next_power_of_2(max_len), MIN_TILE_ROWS))
        out = torch.empty((total, features), dtype=packed.dtype, device=packed.device)
        with torch_device_fn.device(packed.device):
            _unpack_seq_tv_kernel[(num_seqs, triton.cdiv(max_len, rows))](
                out,
                packed,
                lengths,
                max_len,
                total,
                FEATURES=features,
                BLOCK_T=rows,
                BLOCK_D=triton.next_power_of_2(features),
            )
        return out.reshape((total,) + tuple(original_shape[2:]))

    def unpack_seq_triton(
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
        block_t: int = 64,
        block_d: int = 64,
        *,
        total_tokens: int | None = None,
    ) -> torch.Tensor:
        if not _unpack_tv_eligible(packed_tensor, lengths, total_tokens):
            return _unpack_seq_generic(
                packed_tensor, lengths, block_t, block_d, total_tokens
            )
        return tensorview_call(
            "UNPACK_SEQ_TRITON",
            _unpack_seq_tensorview,
            _unpack_seq_generic,
            packed_tensor,
            lengths,
            block_t,
            block_d,
            total_tokens,
        )
