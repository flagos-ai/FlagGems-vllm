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

"""RMSNorm modules compatible with the FlagGems and vLLM call conventions."""

import logging
import numbers
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Size
from torch.nn import Parameter, init

from flaggems_vllm.ops.fused_add_rms_norm import fused_add_rms_norm
from flaggems_vllm.ops.rms_norm import rms_norm

logger = logging.getLogger(__name__)

__all__ = [
    "gems_rms_forward",
    "GemsRMSNorm",
]


def _validate_rms_inputs(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
) -> None:
    if weight.ndim == 0:
        raise ValueError("weight must have at least one dimension")
    if x.ndim < weight.ndim or tuple(x.shape[-weight.ndim :]) != tuple(weight.shape):
        raise ValueError(
            "the trailing input dimensions must match weight, "
            f"got input shape {tuple(x.shape)} and weight shape {tuple(weight.shape)}"
        )
    if x.device != weight.device:
        raise ValueError(
            "x and weight must be on the same device, "
            f"got {x.device} and {weight.device}"
        )
    if residual is not None:
        if residual.shape != x.shape:
            raise ValueError(
                "residual must have the same shape as x, "
                f"got {tuple(residual.shape)} and {tuple(x.shape)}"
            )
        if residual.device != x.device:
            raise ValueError(
                "residual and x must be on the same device, "
                f"got {residual.device} and {x.device}"
            )


def gems_rms_forward(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Apply RMSNorm, optionally fusing an in-place residual addition.

    Without ``residual``, this returns the normalized output tensor. With a
    residual, the underlying vLLM-style operator updates ``x`` and ``residual``
    in place and returns the pair ``(x, residual)``.
    """
    _validate_rms_inputs(x, residual, weight)
    normalized_shape = list(weight.size())

    if residual is not None:
        logger.debug("GEMS CUSTOM FUSED_ADD_RMS_NORM")
        return fused_add_rms_norm(x, residual, normalized_shape, weight, eps)

    logger.debug("GEMS CUSTOM RMS_NORM")
    return rms_norm(x, normalized_shape, weight, eps)


class GemsRMSNorm(nn.Module):
    """RMSNorm with the residual-aware forward convention used by vLLM.

    ``forward(x)`` returns a tensor. ``forward(x, residual)`` applies fused
    residual addition and returns ``(x, residual)`` after both tensors have
    been updated by the operator. The residual path is inference-only and must
    run with gradient recording disabled.
    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Union[int, List[int], Size]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        if not self.normalized_shape or any(dim <= 0 for dim in self.normalized_shape):
            raise ValueError(
                "normalized_shape must contain at least one positive dimension, "
                f"got {self.normalized_shape}"
            )

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            # Parameter initialization is control-plane setup; all forward
            # numerical work stays in the local Triton operators.
            init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.elementwise_affine or self.weight is None:
            raise NotImplementedError(
                "GemsRMSNorm does not support elementwise_affine=False"
            )
        return gems_rms_forward(x, residual, self.weight, self.eps)

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
