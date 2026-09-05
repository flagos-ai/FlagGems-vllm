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

"""PPU MXFP4 warm operator benchmark against dequantized Torch MoE.

Run with pytest -s benchmark/test_fused_marlin_moe_w4a16_mxfp4_ppu.py.
Weights are quantized/dequantized before timing. Both implementations perform
routing on every call. Packing is warmed before capture, as in model serving.
The baseline is Torch BF16/FP16, not vLLM's quantized Marlin implementation.
"""

import json
import statistics

import pytest
import torch

import flaggems_vllm
from tests.test_fused_marlin_moe_w4a16_mxfp4_ppu import make_inputs

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU specialization"
)


def torch_moe(args, refs):
    a = args["hidden_states"]
    ids = args["topk_ids"]
    tw = args["topk_weights"]
    m, topk = ids.shape
    a_routes = a[:, None, :].expand(m, topk, a.shape[1]).reshape(m * topk, 1, -1)
    w1 = refs[0][ids.flatten()]
    w2 = refs[1][ids.flatten()]
    gu = torch.bmm(a_routes, w1.transpose(1, 2))
    gate, up = gu.chunk(2, dim=-1)
    inter = torch.nn.functional.silu(gate) * up
    down = torch.bmm(inter, w2.transpose(1, 2)).reshape(m, topk, -1)
    return (down.float() * tw[:, :, None]).sum(dim=1).to(a.dtype)


def graph_us(fn):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Capture several calls to suppress graph replay overhead.
        for _ in range(20):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / 20)
    return statistics.median(times)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 256, 2),
        (4, 4, 256, 128, 2),
        (33, 8, 128, 256, 4),
        (128, 4, 512, 256, 2),
    ],
)
def test_benchmark_mxfp4(shape, dtype):
    args, refs = make_inputs(*shape, dtype=dtype)
    gems = lambda: flaggems_vllm.fused_marlin_moe(**args)
    baseline = lambda: torch_moe(args, refs)
    actual, expected = gems(), baseline()
    error = (
        actual.float() - expected.float()
    ).abs().mean() / expected.float().abs().mean()
    assert error < 0.04
    gems_us = graph_us(gems)
    torch_us = graph_us(baseline)
    print(
        "MXFP4_BENCH "
        + json.dumps(
            dict(
                shape=shape,
                dtype=str(dtype),
                gems_us=gems_us,
                torch_us=torch_us,
                speedup=torch_us / gems_us,
                error=error.item(),
            )
        )
    )
