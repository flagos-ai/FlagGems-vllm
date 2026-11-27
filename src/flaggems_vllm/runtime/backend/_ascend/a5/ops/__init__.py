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

from flaggems_vllm.runtime.backend._ascend.a5.ops.causal_conv1d_fn import (
    causal_conv1d_fn,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.causal_conv1d_update import (
    causal_conv1d_update,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.flash_attention_bnsd import (
    flash_attention_fwd,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.grouped_matmul_swiglu_quant_v2 import (
    grouped_matmul_swiglu_quant_v2,
    grouped_matmul_swiglu_quant_v2_ref,
)
from flaggems_vllm.runtime.backend._ascend.a5.ops.swiglu_group_quant import (
    swiglu_group_quant,
)

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "flash_attention_fwd",
    "swiglu_group_quant",
    "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_v2_ref",
]
