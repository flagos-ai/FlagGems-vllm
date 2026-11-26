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

"""iHC pre with a pipeline depth supported by the Ascend compiler."""

import logging

import torch

from flaggems_vllm.ops.ihc.ihc_pre import _ihc_pre

logger = logging.getLogger(__name__)


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels using the shared FP32 kernels on Ascend."""
    logger.debug("GEMS_ASCEND TRITON_IHC_PRE")
    # Ascend's tl.range lowering rejects the generic three-stage pipeline.
    return _ihc_pre(
        x, weight, scale, base, magnitude, hc_eps, norm_eps, _pipeline_stages=1
    )


def ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility entry for :func:`triton_ihc_pre`."""
    logger.debug("GEMS_ASCEND IHC_PRE")
    return _ihc_pre(
        x, weight, scale, base, magnitude, hc_eps, norm_eps, _pipeline_stages=1
    )
