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

# tx81 specialized vLLM operators, migrated from FlagGems
# runtime/backend/_tsingmicro/tx81/fused/. Functions re-exported here override
# the generic implementations through SpecOpRegistrar (matched by function
# name), so every public entry must keep the generic-op signature.
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.fused_add_rms_norm import (
    fused_add_rms_norm,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.grouped_topk import grouped_topk
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.reshape_and_cache_flash import (
    reshape_and_cache_flash,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.silu_and_mul import (
    silu_and_mul,
    silu_and_mul_out,
)
from flaggems_vllm.runtime.backend._tsingmicro.tx81.ops.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp,
    silu_and_mul_with_clamp_out,
)

__all__ = [
    "fused_add_rms_norm",
    "grouped_topk",
    "reshape_and_cache_flash",
    "silu_and_mul",
    "silu_and_mul_out",
    "silu_and_mul_with_clamp",
    "silu_and_mul_with_clamp_out",
]
