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

from functools import wraps

from flaggems_vllm.ops.flash_attn_varlen_func_w8a8_fp8 import (
    _flash_attn_varlen_func_w8a8_fp8,
)
from flaggems_vllm.runtime import torch_device_fn


@wraps(_flash_attn_varlen_func_w8a8_fp8, assigned=("__annotations__",))
def flash_attn_varlen_func_w8a8_fp8(*args, **kwargs):
    """Run block-scaled FP8 variable-length attention on MUSA.

    Reuse the shared FP8 varlen kernel and scale validation. The NVIDIA dense
    autotuner is bypassed; MUSA uses the varlen launch configuration.
    """
    if torch_device_fn.get_device_capability()[0] < 3:
        raise NotImplementedError(
            "W8A8 FP8 attention requires MUSA capability 3 or newer"
        )
    return _flash_attn_varlen_func_w8a8_fp8(*args, **kwargs)
