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

"""top_k_per_row_decode on MetaX: the generic operator, on its TLE path when
the installed FlagTree passes top_k_per_row_tle's self-test. Otherwise exactly
the generic non-TLE path.

On the TLE path the number of blocks each row is split into is chosen per call.
The generic default, MULTIPLE_BLOCKS_PER_ROW_CONFIG = 10, was set for NVIDIA.
Swept on the C550 (kernel mode, benchmark shapes, drift
<= 0.8%), the best count falls as rows rise, and "keep rows x blocks at or
under 256 programs, as a power of two in [4, 16]" lands on the measured best at
all 11 benchmark row counts:

    rows    1    4    8   16   24   32   40   48   56  496  512
    blocks 16   16   16   16    8    8    4    4    4    4    4
    10 ->  1.35 1.37 1.37 1.78 2.09 1.98 1.92 2.10 2.22 2.59 2.58
    rule   1.53 1.58 1.50 1.95 2.21 2.46 1.96 2.30 2.40 3.17 3.23

(ratio vs vLLM; geomean 1.890 -> 2.137; confirmed by the benchmark at
2.117). The floor of 4 is measured too: at 128/256/496/512 rows, 2 blocks
was within 0.2-1.7% of 4 and 3 blocks was slower. One block per row returns WRONG
results at every BLOCK_SIZE (a generic merge-path bug), so 4 is also a floor
for correctness headroom. BLOCK_SIZE stays 512: 128/256 were slower everywhere
and 1024 only helped 1-8 rows, by 3-4%."""

from importlib import import_module

from flaggems_vllm.runtime.backend._metax.fused import top_k_per_row_tle as _tle

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")

MAX_PROGRAMS = 256
MIN_BLOCKS_PER_ROW = 4
MAX_BLOCKS_PER_ROW = 16


def _blocks_per_row(num_rows):
    target = max(1, MAX_PROGRAMS // max(1, num_rows))
    pow2 = 1 << (target.bit_length() - 1)
    return max(MIN_BLOCKS_PER_ROW, min(MAX_BLOCKS_PER_ROW, pow2))


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    if _tle.ensure_tle(logits.device):
        # Read by the generic host dispatch on this very call.
        _generic.MULTIPLE_BLOCKS_PER_ROW_CONFIG = _blocks_per_row(num_rows)
    return _generic.top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    )
