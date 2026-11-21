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

"""Full allocating API: MetaX fast path versus the generic FGV non-TLE path."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle

from . import base


class MoeAlignAllocatingBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "moe_align_block_size",
            moe_align_block_size_no_tle,
            [torch.int32],
            gems_op=flaggems_vllm.moe_align_block_size_no_tle,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 6, 16, 256),
            (16, 6, 16, 256),
            (64, 6, 16, 256),
            (64, 6, 16, 8),
            (86, 6, 64, 256),
            (1024, 6, 64, 8),
            (4096, 6, 128, 256),
            (4096, 6, 128, 8),
        ]

    def get_input_iter(self, dtype):
        for tokens, topk, block, active in self.shapes:
            ids = torch.randint(
                0, active, (tokens, topk), dtype=dtype, device=flaggems_vllm.device
            )
            yield ids, block, 256


@pytest.mark.moe_align_block_size
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX fast path")
def test_moe_align_allocating():
    MoeAlignAllocatingBenchmark().run()
