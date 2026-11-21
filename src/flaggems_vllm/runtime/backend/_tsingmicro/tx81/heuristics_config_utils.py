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


def mha_varlen_heur_block_m(params):
    # seqlen_q is the kernel's actual Q length (after swap for decode).
    # Prefill (sq >= 1024): large tiles for throughput -> BLOCK_M=512
    # Decode swapped  (sq=8):  small tiles for SM utilization -> BLOCK_M=8
    sq = params.seqlen_q
    if sq >= 1024:
        return 512
    elif sq >= 512:
        return 256
    elif sq >= 128:
        return 128
    elif sq >= 64:
        return 64
    elif sq >= 32:
        return 32
    else:
        return 8  # decode swap: sq=q_groups=8


def mha_varlen_heur_block_n(params):
    # BLOCK_N must be divide by block_size, so BLOCK_N must be <= block_size,
    # as TsingMicro DSA acrh, the BLOCK_N equal to block_size works well
    return params.block_size


HEURISTICS_CONFIGS = {
    # Dynamically chosen by mha_varlen_heur_block_m based on seqlen_q:
    "mha_varlen_fwd": {
        "BLOCK_M": mha_varlen_heur_block_m,
        "BLOCK_N": mha_varlen_heur_block_n,
        "num_warps": lambda args: 1,
        "num_stages": lambda args: 2,
    },
}
