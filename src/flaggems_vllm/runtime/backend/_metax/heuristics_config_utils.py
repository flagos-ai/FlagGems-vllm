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

# FP32 self-attention uses the existing block-64 profile after the short-context
# region. Keep this policy next to the launch hyperparameters, not in the kernel.
FP32_ENCODER_MIN_CONTEXT = 128
FP32_ENCODER_HEAD_SIZE = 64
FP32_ENCODER_LONG_CONTEXT = 1024


def select_fp32_varlen_config(default, *, head_size, max_query_len, self_attention):
    if (
        self_attention
        and head_size == FP32_ENCODER_HEAD_SIZE
        and max_query_len >= FP32_ENCODER_MIN_CONTEXT
    ):
        if max_query_len > FP32_ENCODER_LONG_CONTEXT:
            # Only replace the large native N16 profile. Other head/batch
            # regimes may already use N32 and should retain their own profile.
            return "mha_fp32_encoder_long" if default == "mha_block_128" else default
        return "mha_block_64"
    return default


# Direct defaults; paged D128/D256 resource overrides live in launch_direct.
HEURISTICS_CONFIGS = {
    "mha_fp32_encoder_long": {
        # Keep the native N=16 reduction grouping for sensitive long-context
        # token embeddings, while reducing M and overlapping two stages.
        "BLOCK_M": lambda args: 64,
        "BLOCK_N": lambda args: 16,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 2,
    },
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
