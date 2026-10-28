# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complete concat/head-pad and sink-pad chain; no FlashMLA or KV access."""
import pytest
import torch

from flaggems_vllm.ops.concat_pad_mla_q_and_sink import concat_pad_mla_q_and_sink

from . import base


def reference(nope, rope, sink, out_q, out_sink):
    query = torch.cat(
        (nope, rope.expand(nope.shape[0], nope.shape[1], rope.shape[2])), -1
    )
    query = torch.nn.functional.pad(query, (0, 0, 0, out_q.shape[1] - nope.shape[1]))
    padded_sink = torch.nn.functional.pad(
        sink, (0, out_sink.shape[0] - sink.shape[0]), value=float("-inf")
    )
    return query, padded_sink


class ConcatPadBenchmark(base.Benchmark):
    def set_shapes(self, shape_file=None):
        self.shapes = [(m, 4, 64, 512, 64) for m in (1, 2, 4, 8, 16, 32, 64, 1024)]

    def get_input_iter(self, dtype):
        for m, h, hp, n, r in self.shapes:
            yield (
                torch.randn(m, h, n, dtype=dtype, device="cuda"),
                torch.randn(m, 1, r, dtype=dtype, device="cuda"),
                torch.randn(h, device="cuda"),
                torch.empty(m, hp, n + r, dtype=dtype, device="cuda"),
                torch.empty(hp, device="cuda"),
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_concat_pad_mla_q_and_sink_perf():
    bench = ConcatPadBenchmark(
        op_name="concat_pad_mla_q_and_sink", torch_op=reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(concat_pad_mla_q_and_sink)
    bench.run()
