# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import statistics

import pytest
import torch

import flaggems_vllm as gems
from flaggems_vllm.ops.mhc.mhc_post import MixLayout
from tests.test_megamhc import EPS, candidate, make_inputs

pytestmark = [
    pytest.mark.megamhc,
    pytest.mark.skipif(
        gems.vendor_name != "thead" or not torch.cuda.is_available(),
        reason="requires T-Head PPU",
    ),
]


def split(args):
    x, residual, post, comb, fn, scale, base, norm = args
    residual = gems.mhc_post(x, residual, post, comb, mix_layout=MixLayout.TRANSPOSED)
    post, comb, y = gems.mhc_pre(residual, fn, scale, base, *EPS)
    return residual, post, comb, gems.mhc_rms_norm(y, norm, 1e-6)


def _capture(call, args, warmup):
    for _ in range(warmup):
        call(args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = call(args)
    return graph, outputs


def _paired_graph_medians(args, fused_op):
    runs = [_capture(op, args, 20) for op in (split, fused_op)]
    graphs = [graph for graph, _ in runs]
    samples = [[], []]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(100):
        for j in (i % 2, 1 - i % 2):
            start.record()
            graphs[j].replay()
            end.record()
            end.synchronize()
            samples[j].append(start.elapsed_time(end) * 1000)
    return tuple(map(statistics.median, samples))


@pytest.mark.parametrize(
    "tokens", [1, 24, 64, 128, 512, 1028, 2048, 3076, 4096, 8192, 16384, 32768]
)
def test_mega_mhc(tokens):
    args = make_inputs(tokens, 4096, 4)
    baseline, fused = _paired_graph_medians(args, candidate)
    print(
        f"MegaMHC CUDA Graph T={tokens} split={baseline:.3f} us fused={fused:.3f} us {baseline / fused:.4f}x"
    )


@pytest.mark.parametrize("implementation", ["auto", "triton"])
@pytest.mark.parametrize("tokens", [4096, 8192, 16384, 32768])
def test_mega_mhc_prepacked(tokens, implementation):
    args = make_inputs(tokens, 4096, 4)
    # Both operators prepare their weights before capture. Keep the packed
    # tensor alive until all replays have finished.
    packed = gems.pack_mega_mhc_weights(args[4])

    def fused_op(inputs):
        return candidate(inputs, packed_fn=packed, implementation=implementation)

    baseline, fused = _paired_graph_medians(args, fused_op)
    print(
        f"MegaMHC prepacked CUDA Graph T={tokens} implementation={implementation} "
        f"split={baseline:.3f} us fused={fused:.3f} us {baseline / fused:.4f}x"
    )
