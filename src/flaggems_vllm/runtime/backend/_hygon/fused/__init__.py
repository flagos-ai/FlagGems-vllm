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

from flaggems_vllm.runtime.backend._hygon.fused.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)
from flaggems_vllm.runtime.backend._hygon.fused.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from flaggems_vllm.runtime.backend._hygon.fused.fused_marlin_moe import (  # noqa: F401
    fused_marlin_moe,
)
from flaggems_vllm.runtime.backend._hygon.fused.moe_sum import moe_sum  # noqa: F401
from flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode import (
    top_k_per_row_decode,
)
from flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill import (
    top_k_per_row_prefill,
)

__all__ = [
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "fused_inv_rope_fp8_quant",
    "fused_marlin_moe",
    "moe_sum",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
]
