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

"""Full prefix/window operator versus the native paged R-SWA baseline."""

import pytest
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

import flaggems_vllm

from . import base


def native_reference(
    q, k, v, lengths, table, prefix, cut, window, starts, maxlen, out, workspace
):
    extra = {}
    if workspace is not None:
        extra = dict(
            seq_threshold_3D=128 // q.shape[1],
            num_par_softmax_segments=16,
            softmax_segm_output=workspace[0],
            softmax_segm_max=workspace[1],
            softmax_segm_expsum=workspace[2],
        )
    unified_attention(
        q,
        k,
        v,
        out,
        starts,
        1,
        lengths,
        maxlen,
        q.shape[-1] ** -0.5,
        True,
        (-1, -1),
        table,
        0.0,
        1.0,
        1.0,
        1.0,
        rswa_prefix_lens=prefix,
        rswa_window=window,
        **extra,
    )
    return out


def candidate(
    q, k, v, lengths, table, prefix, cut, window, starts, maxlen, out, workspace
):
    return flaggems_vllm.paged_prefix_window_attention(
        q,
        k,
        v,
        lengths,
        table,
        cut,
        window,
        out=out,
    )


class PagedPrefixWindowBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(batch, 2937, 10, 128) for batch in [1, 8, 32, 64]]
        self.shape_desc = "B, prefix, heads, D"

    def get_input_iter(self, dtype):
        torch.manual_seed(420731)
        for batch, prefix, heads, dim in self.shapes:
            block, window, length = 16, 128, prefix + 257
            pages = (length + block - 1) // block
            owner = torch.randn(
                (1 + batch * pages, heads, block, 2 * dim),
                device=self.device,
                dtype=dtype,
            )
            owner[0].fill_(30)
            permutation = (
                torch.randperm(batch * pages, device=self.device, dtype=torch.int64) + 1
            )
            table = torch.zeros(
                (batch, 20480 // block), device=self.device, dtype=torch.int32
            )
            table[:, :pages].copy_(permutation.reshape(batch, pages))
            table[:, (prefix + block - 1) // block : (length - window) // block] = 0
            q = torch.randn((batch, heads, dim), device=self.device, dtype=dtype)
            lengths = torch.full(
                (batch,), length, device=self.device, dtype=torch.int32
            )
            protected = torch.full(
                (batch,), prefix, device=self.device, dtype=torch.int32
            )
            cuts = protected.clone()
            starts = torch.arange(batch + 1, device=self.device, dtype=torch.int32)
            out = torch.empty_like(q)
            workspace = None
            if batch <= 128 // heads:
                workspace = (
                    torch.empty((batch, heads, 16, dim), device=self.device),
                    torch.empty((batch, heads, 16), device=self.device),
                    torch.empty((batch, heads, 16), device=self.device),
                )
            args = (
                q,
                owner[..., :dim].transpose(1, 2),
                owner[..., dim:].transpose(1, 2),
                lengths,
                table,
                protected,
                cuts,
                window,
                starts,
                length,
                out,
                workspace,
            )
            expected = native_reference(*args).clone()
            actual = candidate(*args)
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
            yield args


@pytest.mark.paged_prefix_window_attention
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax", reason="MetaX MACA runtime required"
)
def test_paged_prefix_window_attention():
    bench = PagedPrefixWindowBenchmark(
        op_name="paged_prefix_window_attention",
        torch_op=native_reference,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(candidate)
    bench.run()
