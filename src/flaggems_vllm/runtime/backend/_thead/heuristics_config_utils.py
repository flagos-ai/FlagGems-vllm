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

HEURISTICS_CONFIGS = {
    # An entry replaces the NVIDIA one as a whole (ConfigLoader falls back per
    # op name, not per field), so the tile parameters are repeated unchanged.
    "mha_block_128": {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 32,
        "num_warps": lambda args: 4,
        "num_stages": lambda args: 3,
        # FlagTree: the NVIDIA entry switches the load merge off on this tile,
        # where the two large matmuls dominate the K/V loop on H20. They do not
        # dominate it on a PPU, whose dot is the synchronous
        # ppu.mma.sync.aligned.m16n16k16 rather than a warpgroup wgmma, so the
        # merge pays for itself and the tile keeps the default
        # (measured on PPU-ZW810E, page size 256).
        "pipeline_merge_loads": lambda args: "all",
    },
}
