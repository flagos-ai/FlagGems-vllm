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

import triton

# Hopper FP8 einsum / w8a8_block_fp8_bmm require Triton >= 3.6.0.
if triton.__version__ >= "3.6.0":
    try:
        from .fp8_einsum import fp8_einsum  # noqa: F401
        from .w8a8_block_fp8_bmm import w8a8_block_fp8_bmm  # noqa: F401
    except (AttributeError, ImportError):
        pass

__all__ = ["*"]
