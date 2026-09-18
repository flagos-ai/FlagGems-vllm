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

# TN10 Direct defaults; paged D128/D256 resource overrides live in launch_direct.
HEURISTICS_CONFIGS = {
    "mha_block_128": {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 16,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
    },
    "mha_block_64": {
        "BLOCK_M": lambda args: 64,
        "BLOCK_N": lambda args: 32,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
    },
    "mha_block_32": {
        "BLOCK_M": lambda args: 32,
        "BLOCK_N": lambda args: 16,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
    },
    "mha_block_16": {
        "BLOCK_M": lambda args: 16,
        "BLOCK_N": lambda args: 16,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
    },
}
