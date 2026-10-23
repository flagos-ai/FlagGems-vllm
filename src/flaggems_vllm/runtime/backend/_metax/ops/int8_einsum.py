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

from flaggems_vllm.runtime.backend._metax.ops.w8a8_block_int8_bmm import (
    w8a8_block_int8_bmm,
)

logger = logging.getLogger(__name__)


def int8_einsum(
    equation: str,
    x: torch.Tensor,
    xs: torch.Tensor | None,
    y: torch.Tensor,
    ys: torch.Tensor | None,
    block_size: tuple[int, int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
    per_channel_quant: bool = False,
) -> torch.Tensor:
    """Compute scaled INT8 bhr,hdr->bhd on MetaX.

    Per-channel scales: xs[B,H] or [B,H,1], ys[H,D] or [H,D,1].
    """
    logger.debug("GEMS_METAX INT8_EINSUM")
    if equation != "bhr,hdr->bhd":
        raise ValueError("int8_einsum only supports 'bhr,hdr->bhd'")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("int8_einsum inputs must have three dimensions")
    b, h, r = x.shape
    if y.shape[0] != h or y.shape[2] != r or x.device != y.device:
        raise ValueError("int8_einsum input shape or device mismatch")
    if per_channel_quant:
        if xs is not None and xs.ndim == 2:
            xs = xs.unsqueeze(-1)
        if ys is not None and ys.ndim == 2:
            ys = ys.unsqueeze(-1)
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
        per_channel_quant=per_channel_quant,
    )
    return z
