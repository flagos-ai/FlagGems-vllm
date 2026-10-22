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

"""NVIDIA schedules for the existing per_token_group_quant_fp8 operator."""

from typing import Optional, Tuple

import torch

from flaggems_vllm.ops.per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE
from flaggems_vllm.ops.per_token_group_quant_fp8 import (
    per_token_group_quant_fp8 as _generic_per_token_group_quant_fp8,
)

from ._per_token_group_quant_fp8_block import get_h20_block_config, quantize_group_block


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = SUPPORTED_FP8_DTYPE if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    block_config = get_h20_block_config(
        x, group_size, fp8_dtype, eps, column_major_scales, scale_ue8m0
    )
    if block_config is not None:
        rows, block_warps = block_config
        return quantize_group_block(
            x,
            group_size,
            scale_ue8m0=scale_ue8m0,
            rows=rows,
            num_warps=block_warps,
        )

    # Block admission does not import or require the optional TLE extension.
    from ._per_token_group_quant_fp8_reorder import (
        can_use_group_reorder,
        quantize_group_tiles,
    )

    if group_size == 512 and can_use_group_reorder(
        x, group_size, fp8_dtype, eps, column_major_scales, scale_ue8m0
    ):
        return quantize_group_tiles(x, group_size, eps=eps)

    return _generic_per_token_group_quant_fp8(
        x,
        group_size,
        eps=eps,
        dtype=dtype,
        column_major_scales=column_major_scales,
        scale_ue8m0=scale_ue8m0,
    )
