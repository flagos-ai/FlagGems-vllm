# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Padding and endpoint transport; both arms overwrite all output buffers."""

import pytest
import torch

import flaggems_vllm

from . import base


def reference(q, sink, indices, out_q, out_sink, out_bound):
    out_q.zero_()
    out_q[:, :4].copy_(q)
    out_sink.fill_(-torch.inf)
    out_sink[:4].copy_(sink)
    slots = torch.arange(
        1, indices.shape[1] + 1, dtype=torch.int32, device=indices.device
    )
    out_bound.copy_(torch.where(indices >= 0, slots[None, :], 0).amax(1))
    return out_q, out_sink, out_bound


class PadEndpointBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(tokens, 4, 576) for tokens in (1, 256, 1024)]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            tokens = shape[0]
            yield (
                torch.randn(shape, dtype=dtype, device="cuda"),
                torch.randn(4, device="cuda"),
                torch.randint(
                    -1, 204800, (tokens, 2048), dtype=torch.int32, device="cuda"
                ),
                torch.empty((tokens, 64, 576), dtype=dtype, device="cuda"),
                torch.empty(64, device="cuda"),
                torch.empty(tokens, dtype=torch.int32, device="cuda"),
            )


@pytest.mark.pad_mla_q_sink_slot_bound
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="CUDA required",
)
@torch.inference_mode()
def test_pad_mla_q_sink_slot_bound_perf():
    bench = PadEndpointBenchmark(
        op_name="pad_mla_q_sink_slot_bound", torch_op=reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(flaggems_vllm.pad_mla_q_sink_slot_bound)
    bench.run()
