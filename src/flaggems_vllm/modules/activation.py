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
import torch.nn as nn

from flaggems_vllm.ops.silu_and_mul import silu_and_mul

logger = logging.getLogger(__name__)

__all__ = [
    "gems_silu_and_mul",
    "GemsSiluAndMul",
]


def gems_silu_and_mul(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute ``silu(x) * y`` with the FlagGems-vLLM fused operator."""
    if x.device != y.device:
        raise ValueError(
            f"x and y must be on the same device, got {x.device} and {y.device}"
        )

    logger.debug("GEMS CUSTOM SILU_AND_MUL FORWARD")
    return silu_and_mul(x, y)


class GemsSiluAndMul(nn.Module):
    """Fused SiLU activation followed by elementwise multiplication."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return gems_silu_and_mul(x, y)
