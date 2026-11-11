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

# SPDX-License-Identifier: Apache-2.0
# Derived from vllm-plugin-FL sparse_attn_sharedkv.py and its tests (Apache-2.0).
"""Independent Torch device reference versus public API graph replay latency.

Compilation, warmup, capture and CPU correctness are excluded from latency.
The metric is median microseconds per replay; lower is better. SpeedUp is
reference/candidate, with geometric mean across the frozen active set.
The Torch reference is a test-only algorithmic baseline, not old-FL parity.
"""

import math
import statistics

import pytest
import torch

import flaggems_vllm
from tests.test_deepseek_v4_sparse_attention import (
    _case,
    _model_prefill_case,
    _oracle,
    _packed_device_inputs,
)

pytestmark = [
    pytest.mark.deepseek_v4_sparse_attention,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend"),
]


def _torch_reference(*args):
    q, swa, compact, st, ct, requests, positions, sl, cl, indices, window = args[:11]
    ratio, scale, sink, value_dim = args[11:]
    dim = q.shape[-1]

    def rows(cache, table, lengths, logical, limit):
        valid_request = (requests >= 0) & (requests < lengths.numel())
        request = requests.clamp(0, lengths.numel() - 1).long()
        blocks = torch.div(logical, cache.shape[1], rounding_mode="floor")
        valid = valid_request[:, None] & (positions[:, None] >= 0)
        valid = valid & (logical >= 0) & (logical < lengths[request, None])
        valid = valid & (logical < limit[:, None]) & (blocks < table.shape[1])
        page = table[request[:, None], blocks.clamp(0, table.shape[1] - 1).long()]
        valid = valid & (page >= 0) & (page < cache.shape[0])
        values = cache[
            page.clamp(0, cache.shape[0] - 1).long(),
            logical.remainder(cache.shape[1]).long(),
        ]
        return torch.where(valid[:, :, None], values.float(), 0.0), valid

    logical = positions[:, None] - window + 1 + torch.arange(window, device=q.device)
    values, valid = rows(swa, st, sl, logical, positions + 1)
    if compact is not None:
        compressed, compressed_valid = rows(
            compact, ct, cl, indices, (positions + 1) // ratio
        )
        values = torch.cat((values, compressed), dim=1)
        valid = torch.cat((valid, compressed_valid), dim=1)
    logits = torch.bmm(q.float(), values.transpose(1, 2)) * (
        dim**-0.5 if scale is None else scale
    )
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    denominator = torch.logsumexp(logits, dim=-1)
    if sink is not None:
        denominator = torch.logaddexp(denominator, sink.float()[None, :])
    weights = torch.where(
        valid[:, None, :], torch.exp(logits - denominator[:, :, None]), 0.0
    )
    return torch.bmm(weights, values[:, :, :value_dim]).to(q.dtype)


def _graph_latency(function, args, expected):
    for _ in range(3):
        output = function(*args)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().double(), expected, rtol=8e-3, atol=8e-3)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = function(*args)
    output.fill_(float("nan"))
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().double(), expected, rtol=8e-3, atol=8e-3)
    samples = []
    for _ in range(9):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return statistics.median(samples)


def test_deepseek_v4_sparse_attention(record_property):
    baseline = _torch_reference
    results = []
    # Tiny strided vector, contiguous Cube decode, packed chunk-tail prefill.
    for path in ("vector", "cube", "prefill"):
        for ratio in (0, 4, 128):
            if path == "prefill":
                cpu = _model_prefill_case(ratio, torch.bfloat16, tokens=129)
                args = _packed_device_inputs(cpu)
            else:
                cpu, args = _case(
                    ratio, torch.bfloat16, dim=64 if path == "vector" else 512
                )
                if path == "cube":
                    args = [
                        x.contiguous() if isinstance(x, torch.Tensor) else x
                        for x in args
                    ]
            expected = _oracle(cpu)
            base_us = _graph_latency(baseline, args, expected)
            candidate_us = _graph_latency(
                flaggems_vllm.deepseek_v4_sparse_attention, args, expected
            )
            speedup = base_us / candidate_us
            results.append(speedup)
            record_property(
                f"{path}_c{ratio}",
                {
                    "baseline_us": base_us,
                    "candidate_us": candidate_us,
                    "speedup": speedup,
                },
            )
            print(
                f"{path} C{ratio}: baseline={base_us:.3f}us candidate={candidate_us:.3f}us SpeedUp={speedup:.4f}"
            )
            assert (
                speedup >= 0.9
            ), f"Graph replay regression for {path} C{ratio}: {speedup:.4f}"
    record_property(
        "geomean_speedup", math.exp(sum(map(math.log, results)) / len(results))
    )
