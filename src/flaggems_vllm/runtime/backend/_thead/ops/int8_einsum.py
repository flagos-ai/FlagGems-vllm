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

from flaggems_vllm.runtime.backend._thead.ops.w8a8_block_int8_bmm import (
    w8a8_block_int8_bmm,
)

logger = logging.getLogger(__name__)


def int8_einsum(
    equation: str,
    x: torch.Tensor,
    xs: torch.Tensor | None,
    y: torch.Tensor,
    ys: torch.Tensor | None,
    block_size=(128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Map the upstream bhr,hdr->bhd contraction onto PPU W8A8 BMM.

    INT8 inputs use block scales; floating inputs use xs=ys=None.
    The head-to-batch permutations are views, without input copies.
    """
    logger.debug("GEMS_THEAD INT8_EINSUM")
    if equation != "bhr,hdr->bhd":
        raise ValueError("int8_einsum only supports 'bhr,hdr->bhd'")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("int8_einsum inputs must have three dimensions")
    b, h, r = x.shape
    if y.shape[0] != h or y.shape[2] != r or x.device != y.device:
        raise ValueError("int8_einsum input shape or device mismatch")
    if xs is not None and xs.ndim != 3:
        raise ValueError("int8_einsum activation scale must have three dimensions")
    z = torch.empty((b, h, y.shape[1]), device=x.device, dtype=output_dtype)
    w8a8_block_int8_bmm(
        x.permute(1, 0, 2),
        y,
        xs.permute(1, 0, 2) if xs is not None else None,
        ys,
        block_size=block_size,
        z=z.permute(1, 0, 2),
        output_dtype=output_dtype,
    )
    return z
