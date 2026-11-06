# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete prefix TopK and serving epilogue with the same stable-key baseline."""

import pytest
import torch

import flaggems_vllm

from . import base


def reference(scores, lengths, count, out):
    bits = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = torch.where(scores == 0, 0, bits)
    mask = torch.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    order = torch.argsort(bits ^ mask, dim=-1, descending=True, stable=True)[:, :count]
    selected = scores.gather(1, order)
    out.copy_(torch.where(torch.isfinite(selected), order, -1).to(torch.int32))
    return out


class PrefixTopKBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(4, 16384), (256, 32768), (4, 204800)]

    def get_input_iter(self, dtype):
        for rows, width in self.shapes:
            scores = torch.randn((rows, width), dtype=dtype, device="cuda")
            scores[:, 4096:] = -torch.inf
            lengths = torch.full((rows,), 4096, dtype=torch.int32, device="cuda")
            out = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")
            yield scores, lengths, 2048, out


@pytest.mark.indexer_prefix_topk
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="CUDA required",
)
@torch.inference_mode()
def test_indexer_prefix_topk_perf():
    from flag_gems.utils.triton_version_utils import HAS_TLE

    if not HAS_TLE:
        pytest.skip("optional TLE backend required")
    bench = PrefixTopKBenchmark(
        op_name="indexer_prefix_topk", torch_op=reference, dtypes=[torch.float32]
    )
    bench.set_gems(flaggems_vllm.indexer_prefix_topk)
    bench.run()
