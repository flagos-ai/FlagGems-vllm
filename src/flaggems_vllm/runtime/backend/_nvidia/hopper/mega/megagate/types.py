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

"""Model-agnostic MegaGate dispatch types.

The routing semantics are part of a kernel's ABI.  Two tensors with the same
matrix dimensions are not interchangeable when their scoring, grouping, or
normalization contracts differ, so every such property is represented in the
hashable :class:`ShapeKey` used by production dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RoutingMode(str, Enum):
    """Supported MegaGate scoring and expert-selection contracts."""

    SOFTMAX = "softmax"
    SQRT_SOFTPLUS_BIAS = "sqrt_softplus_bias"
    GROUPED_SIGMOID_BIAS = "grouped_sigmoid_bias"


@dataclass(frozen=True)
class ShapeKey:
    """Complete, model-independent key for one exact MegaGate kernel."""

    tokens: int
    hidden: int
    experts: int
    topk: int
    routing_mode: RoutingMode
    uses_correction_bias: bool
    expert_group_count: int | None
    topk_group: int | None
    renormalize: bool
    routed_scaling_factor: float
    input_dtype: str = "bfloat16"
    compute_capability: tuple[int, int] = (9, 0)

    def __post_init__(self) -> None:
        if min(self.tokens, self.hidden, self.experts, self.topk) <= 0:
            raise ValueError("MegaGate shape dimensions must be positive")
        if self.topk > self.experts:
            raise ValueError("MegaGate topk cannot exceed the expert count")
        grouped = self.expert_group_count is not None or self.topk_group is not None
        if grouped and (self.expert_group_count is None or self.topk_group is None):
            raise ValueError(
                "expert_group_count and topk_group must be specified together"
            )
        if self.routing_mode is RoutingMode.GROUPED_SIGMOID_BIAS and not grouped:
            raise ValueError(
                "grouped sigmoid routing requires an expert-group contract"
            )
        if self.routing_mode is not RoutingMode.GROUPED_SIGMOID_BIAS and grouped:
            raise ValueError("expert-group fields are only valid for grouped routing")


@dataclass(frozen=True)
class KernelPlan:
    """One exact-shape implementation selected by :class:`ShapeKey` only."""

    shape: ShapeKey
    implementation_module: str
    source_leaf: str


__all__ = ["KernelPlan", "RoutingMode", "ShapeKey"]
