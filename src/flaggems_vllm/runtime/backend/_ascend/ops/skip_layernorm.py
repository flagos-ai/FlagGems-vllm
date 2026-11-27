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

"""Ascend TensorView specialization of the public skip_layer_norm operator."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.skip_layernorm import skip_layer_norm as _generic
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

    @triton.jit(do_not_specialize=["eps"])
    def _skip_layer_norm_tv_kernel(
        y,
        x,
        residual,
        weight,
        bias,
        rows,
        eps,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        tile = tl.program_id(0)
        x_view = tle.make_partition_view(
            x, shape=[rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, WIDTH]
        )
        r_view = tle.make_partition_view(
            residual, shape=[rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, WIDTH]
        )
        w_view = tle.make_partition_view(
            weight, shape=[WIDTH], strides=[1], tile=[WIDTH]
        )
        b_view = tle.make_partition_view(bias, shape=[WIDTH], strides=[1], tile=[WIDTH])
        values = tile_load(
            x_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB
        ).to(tl.float32)
        values += tile_load(
            r_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB
        ).to(tl.float32)
        scale = tile_load(w_view, index=(0,), dst_space=tle.language.dsa.ascend.UB).to(
            tl.float32
        )
        shift = tile_load(b_view, index=(0,), dst_space=tle.language.dsa.ascend.UB).to(
            tl.float32
        )
        mean = tl.sum(values, axis=1) / WIDTH
        centered = values - mean[:, None]
        variance = tl.sum(centered * centered, axis=1) / WIDTH
        result = centered * (1.0 / tl.sqrt(variance + eps))[:, None]
        result = result * scale[None, :] + shift[None, :]
        y_view = tle.make_partition_view(
            y, shape=[rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, WIDTH]
        )
        tile_store(
            y_view,
            value=result.to(y.dtype.element_ty),
            index=(tile, 0),
            src_space=tle.language.dsa.ascend.UB,
        )

    def _tensorview(x, residual, normalized_shape, weight, bias, eps):
        width = normalized_shape[0]
        rows = x.numel() // width
        block_m = tile_rows_for_ub(width * 4, 4, 32)
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        output = torch.empty_like(x)
        with torch_device_fn.device(x.device):
            _skip_layer_norm_tv_kernel[(triton.cdiv(rows, block_m),)](
                output,
                x,
                residual,
                weight,
                bias,
                rows,
                eps,
                WIDTH=width,
                BLOCK_M=block_m,
            )
        return output


def _eligible(x, residual, normalized_shape, weight, bias):
    if not HAS_TENSORVIEW or len(normalized_shape) != 1:
        return False
    width = normalized_shape[0]
    if (
        width <= 0
        or x.ndim < 1
        or x.shape[-1] != width
        or x.numel() == 0
        or x.shape != residual.shape
        or weight is None
        or bias is None
        or weight.numel() != width
        or bias.numel() != width
        or x.dtype != residual.dtype
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or triton.next_power_of_2(width) != width
    ):
        return False
    return tile_rows_for_ub(width * 4, 4, 32) >= MIN_TILE_ROWS


def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=1e-5):
    if not _eligible(x, residual, normalized_shape, weight, bias):
        return _generic(x, residual, normalized_shape, weight, bias, eps)
    return tensorview_call(
        "SKIP_LAYERNORM",
        _tensorview,
        _generic,
        x,
        residual,
        normalized_shape,
        weight,
        bias,
        eps,
    )
