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

import torch


def v2_apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """Scale logits in place by the temperature of each row's request slot.

    Logits have shape [tokens, vocabulary] with contiguous columns and optional
    row padding. The integer mapping selects entries from a contiguous FP32
    temperature vector on the same Ascend device. Temperatures 0 and 1 are
    identity cases. Empty logits need no launch. Computation uses FP32 and
    writes back to the input dtype (FP16, BF16, or FP32).
    """
    raise NotImplementedError("v2_apply_temperature is supported on Ascend")
