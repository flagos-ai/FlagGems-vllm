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

"""
flaggems_vllm - DNN operations implemented with Triton
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

from flaggems_vllm import testing  # noqa: F401
from flaggems_vllm import runtime
from flaggems_vllm.ops import *  # noqa: F401,F403
from flaggems_vllm.runtime import flagtune  # noqa: F401

device = runtime.device.name
vendor_name = runtime.device.vendor_name
runtime.replace_customized_ops(globals())

try:
    __version__ = _metadata_version("flaggems_vllm")
except PackageNotFoundError:
    __version__ = "0.0.0"

_COMPAT_OP_ALIASES = {
    "ops_add_rms_norm": "fused_add_rms_norm",
    "ops_deepseek_v4_qnorm_rope_kv_rope_quant_insert": "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "ops_experts_impl": "fused_experts_impl",
    "ops_inv_rope_fp8_quant": "fused_inv_rope_fp8_quant",
    "ops_recurrent_gated_delta_rule_fwd": "fused_recurrent_gated_delta_rule_fwd",
}
for _alias, _target in _COMPAT_OP_ALIASES.items():
    if _target in globals():
        globals()[_alias] = globals()[_target]

__all__ = [
    "SUPPORTED_FP8_DTYPE",
]
