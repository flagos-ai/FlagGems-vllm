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

"""top_k_per_row_prefill on MetaX: the generic operator, on its TLE path when
the installed FlagTree passes top_k_per_row_tle's self-test. Otherwise exactly
the generic non-TLE path.

On the TLE path the tile (NUM_THREADS_PER_BLOCK, generic default 512) is chosen
per call. Swept on the C550 (kernel mode, benchmark shapes,
drift <= 0.7%), 1024 against 512, ratio vs vLLM:

    (64,129280)  0.878 -> 0.981     (4,16385)   0.892 -> 0.908
    (4,8193)     0.920 -> 0.927     (4100,1025) 1.031 -> 0.912
    (16383,4095) 1.839 -> 1.542     (12961,4100) 1.279 -> 1.025
    (16380,5115) 1.919 -> 1.603

The wide tile pays on long rows and costs 12-20% on the many-row, short-vocab
shapes, so it is keyed on vocabulary size: >= 16384 takes 1024 (geomean 1.189
-> 1.211, nearly all of it the DeepSeek-V4 shape). The benchmark has no shape
with both many rows and a large vocabulary, so the cut is placed between the
two groups it does have, not fitted inside either.

1024 lanes on 8 warps is 2 elements per thread. That regime once corrupted a
masked shared atomic, but on a FlagTree with the Alias.cpp fix it measured
clean with and without the local_ptr shim: the earlier corruption was the allocator overlap, not the plugin.
"""

from importlib import import_module

from flaggems_vllm.runtime.backend._metax.fused import top_k_per_row_tle as _tle

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

WIDE_TILE_VOCAB = 16384


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    if _tle.ensure_tle(logits.device):
        # Read by the generic host dispatch on this very call.
        _generic.NUM_THREADS_PER_BLOCK = (
            1024 if logits.shape[1] >= WIDE_TILE_VOCAB else 512
        )
    return _generic.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
