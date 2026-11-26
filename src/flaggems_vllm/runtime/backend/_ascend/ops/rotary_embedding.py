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

"""Restricted TensorView fast path for explicit-position, non-interleaved RoPE."""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.rotary_embedding import apply_rotary_pos_emb as _generic
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.tensorview import (
    HAS_TENSORVIEW,
    MIN_TILE_ROWS,
    tensorview_call,
    tile_rows_for_ub,
)

logger = logging.getLogger(__name__)

if HAS_TENSORVIEW:
    import triton.experimental.tle as tle
    from triton.experimental.tle import tile_load, tile_store

    @triton.jit
    def _rotary_tv_kernel(
        q_out,
        k_out,
        q,
        k,
        cos,
        sin,
        positions,
        tokens,
        cos_stride,
        sin_stride,
        NUM_Q: tl.constexpr,
        NUM_K: tl.constexpr,
        Q_PAD: tl.constexpr,
        K_PAD: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_POS: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        tile = tl.program_id(0)
        half: tl.constexpr = HEAD_DIM // 2
        rows = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        live = rows < tokens
        pos = tl.load(positions + rows, mask=live, other=0)
        tl.device_assert(
            tl.min(tl.where(live, pos, 0), axis=0) >= 0, "negative position id"
        )
        tl.device_assert(
            tl.max(tl.where(live, pos, 0), axis=0) < MAX_POS,
            "position id out of bound",
        )
        columns = tl.arange(0, half)[None, :]
        safe_pos = tl.minimum(tl.maximum(pos, 0), MAX_POS - 1)
        c = tl.load(
            cos + safe_pos[:, None] * cos_stride + columns,
            mask=live[:, None],
            other=0,
        ).to(tl.float32)[:, None, :]
        s = tl.load(
            sin + safe_pos[:, None] * sin_stride + columns,
            mask=live[:, None],
            other=0,
        ).to(tl.float32)[:, None, :]

        q_view = tle.make_partition_view(
            q,
            shape=[tokens, NUM_Q, HEAD_DIM],
            strides=[NUM_Q * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, Q_PAD, half],
        )
        qo_view = tle.make_partition_view(
            q_out,
            shape=[tokens, NUM_Q, HEAD_DIM],
            strides=[NUM_Q * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, Q_PAD, half],
        )
        q1 = tile_load(
            q_view, index=(tile, 0, 0), dst_space=tle.language.dsa.ascend.UB
        ).to(tl.float32)
        q2 = tile_load(
            q_view, index=(tile, 0, 1), dst_space=tle.language.dsa.ascend.UB
        ).to(tl.float32)
        tile_store(
            qo_view,
            value=(q1 * c - q2 * s).to(q_out.dtype.element_ty),
            index=(tile, 0, 0),
            src_space=tle.language.dsa.ascend.UB,
        )
        tile_store(
            qo_view,
            value=(q2 * c + q1 * s).to(q_out.dtype.element_ty),
            index=(tile, 0, 1),
            src_space=tle.language.dsa.ascend.UB,
        )

        k_view = tle.make_partition_view(
            k,
            shape=[tokens, NUM_K, HEAD_DIM],
            strides=[NUM_K * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, K_PAD, half],
        )
        ko_view = tle.make_partition_view(
            k_out,
            shape=[tokens, NUM_K, HEAD_DIM],
            strides=[NUM_K * HEAD_DIM, HEAD_DIM, 1],
            tile=[BLOCK_T, K_PAD, half],
        )
        k1 = tile_load(
            k_view, index=(tile, 0, 0), dst_space=tle.language.dsa.ascend.UB
        ).to(tl.float32)
        k2 = tile_load(
            k_view, index=(tile, 0, 1), dst_space=tle.language.dsa.ascend.UB
        ).to(tl.float32)
        tile_store(
            ko_view,
            value=(k1 * c - k2 * s).to(k_out.dtype.element_ty),
            index=(tile, 0, 0),
            src_space=tle.language.dsa.ascend.UB,
        )
        tile_store(
            ko_view,
            value=(k2 * c + k1 * s).to(k_out.dtype.element_ty),
            index=(tile, 0, 1),
            src_space=tle.language.dsa.ascend.UB,
        )

    def _tensorview(q, k, cos, sin, position_ids, rotary_interleaved, inplace):
        q_shape, k_shape = q.shape, k.shape
        q = q.view(-1, q.shape[-2], q.shape[-1])
        k = k.view(-1, k.shape[-2], k.shape[-1])
        tokens, num_q, head_dim = q.shape
        num_k = k.shape[1]
        q_pad = triton.next_power_of_2(num_q)
        k_pad = triton.next_power_of_2(num_k)
        block_t = tile_rows_for_ub(max(q_pad, k_pad) * (head_dim // 2) * 4, 4, 16)
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        with torch_device_fn.device(q.device):
            _rotary_tv_kernel[(triton.cdiv(tokens, block_t),)](
                q_out,
                k_out,
                q,
                k,
                cos,
                sin,
                position_ids.view(-1),
                tokens,
                cos.stride(0),
                sin.stride(0),
                NUM_Q=num_q,
                NUM_K=num_k,
                Q_PAD=q_pad,
                K_PAD=k_pad,
                HEAD_DIM=head_dim,
                MAX_POS=cos.shape[0],
                BLOCK_T=block_t,
                debug=True,
            )
        return q_out.view(q_shape), k_out.view(k_shape)


def _eligible(q, k, cos, sin, position_ids, rotary_interleaved, inplace):
    if not HAS_TENSORVIEW or position_ids is None or rotary_interleaved or inplace:
        return False
    if q.ndim < 3 or k.ndim != q.ndim or q.numel() == 0 or k.numel() == 0:
        return False
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        return False
    if position_ids.shape != q.shape[:-2] or not position_ids.is_contiguous():
        return False
    if position_ids.dtype not in (torch.int32, torch.int64):
        return False
    if not q.is_contiguous() or not k.is_contiguous():
        return False
    dim = q.shape[-1]
    if dim < 2 or triton.next_power_of_2(dim // 2) != dim // 2:
        return False
    if cos.ndim != 2 or cos.shape != sin.shape or cos.shape[1] != dim // 2:
        return False
    if cos.stride(1) != 1 or sin.stride(1) != 1 or cos.shape[0] == 0:
        return False
    if q.dtype != k.dtype or cos.dtype != sin.dtype:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if cos.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    q_pad = triton.next_power_of_2(q.shape[-2])
    k_pad = triton.next_power_of_2(k.shape[-2])
    return tile_rows_for_ub(max(q_pad, k_pad) * (dim // 2) * 4, 4, 16) >= MIN_TILE_ROWS


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
    inplace: bool = False,
):
    if not _eligible(q, k, cos, sin, position_ids, rotary_interleaved, inplace):
        return _generic(q, k, cos, sin, position_ids, rotary_interleaved, inplace)
    return tensorview_call(
        "ROTARY_POS_EMBEDDING",
        _tensorview,
        _generic,
        q,
        k,
        cos,
        sin,
        position_ids,
        rotary_interleaved,
        inplace,
    )
