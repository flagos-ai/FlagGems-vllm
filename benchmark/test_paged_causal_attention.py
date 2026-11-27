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
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

import flaggems_vllm

from . import base


def native(q, k, v, cu, lengths, table, max_q, max_k, out):
    unified_attention(
        q,
        k,
        v,
        out,
        cu,
        max_q,
        lengths,
        max_k,
        q.shape[-1] ** -0.5,
        True,
        (-1, -1),
        table,
        0.0,
        1.0,
        1.0,
        1.0,
    )
    return out


def candidate(q, k, v, cu, lengths, table, max_q, max_k, out):
    return flaggems_vllm.paged_causal_attention(
        q, k, v, cu, lengths, table, max_q, max_k, out=out
    )


class PagedCausalBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 2937, 0), (2, 2937, 0), (2, 619, 2318), (3, 1025, 273)]
        self.shape_desc = "B, query length, previous context"

    def get_input_iter(self, dtype):
        torch.manual_seed(83124)
        for batch, query, past in self.shapes:
            heads, dim, block = 10, 128, 16
            length = query + past
            pages = (length + block - 1) // block
            owner = torch.randn(
                (1 + batch * pages, heads, block, 2 * dim),
                device=self.device,
                dtype=dtype,
            )
            ids = torch.randperm(batch * pages, device=self.device) + 1
            table = torch.zeros((batch, 1280), device=self.device, dtype=torch.int32)
            table[:, :pages].copy_(ids.reshape(batch, pages))
            q = torch.randn(
                (batch * query, heads, dim), device=self.device, dtype=dtype
            )
            cu = torch.arange(batch + 1, device=self.device, dtype=torch.int32) * query
            lengths = torch.full(
                (batch,), length, device=self.device, dtype=torch.int32
            )
            k, v = owner[..., :dim].transpose(1, 2), owner[..., dim:].transpose(1, 2)
            out = torch.empty_like(q)
            args = (q, k, v, cu, lengths, table, query, length, out)
            expected = native(*args).clone()
            actual = candidate(*args)
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.02)
            yield args


@pytest.mark.paged_causal_attention
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax", reason="MetaX MACA runtime required"
)
def test_paged_causal_attention():
    bench = PagedCausalBenchmark(
        op_name="paged_causal_attention",
        torch_op=native,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(candidate)
    bench.run()
