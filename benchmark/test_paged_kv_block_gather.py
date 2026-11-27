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

import pytest
import torch

import flaggems_vllm

from . import base


class PagedKVBlockGatherBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (512, 16, 64, 64, 8),
            (1024, 16, 64, 128, 8),
        ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.set_more_shapes()


@pytest.mark.paged_kv_block_gather
def test_paged_kv_block_gather():
    def input_kwargs(shape, dtype, device):
        num_blocks, block_size, head_dim, num_seqs, blocks_per_seq = shape
        cache = torch.randn(
            (num_blocks, block_size, head_dim), dtype=dtype, device=device
        )
        table = torch.randint(
            num_blocks,
            (num_seqs, blocks_per_seq),
            dtype=torch.int32,
            device=device,
        )
        yield cache, table

    bench = PagedKVBlockGatherBenchmark(
        op_name="paged_kv_block_gather",
        input_fn=input_kwargs,
        torch_op=lambda cache, table: cache[table.long()],
        gems_op=flaggems_vllm.paged_kv_block_gather,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
