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


import statistics
import time

import pytest
import torch

import flaggems_vllm

from . import base

try:
    from vllm.v1.worker.gpu.input_batch import (
        get_num_sampled_and_rejected as vllm_op,
    )
except Exception as exc:
    # A baseline is optional, including when its native extensions cannot load.
    vllm_op = None
    _VLLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _VLLM_IMPORT_ERROR = None


class GetNumSampledAndRejectedBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(n,) for n in (1, 16, 32, 64, 128, 256, 512)]

    def get_input_iter(self, dtype):
        for (n,) in self.shapes:
            index = torch.arange(n, device=self.device, dtype=dtype)
            prefill = index + 32
            seq = prefill + 1
            sampled = torch.full((n,), 2, device=self.device, dtype=dtype)
            cu = torch.arange(n + 1, device=self.device, dtype=dtype) * 4
            self._graph_latencies = None
            yield sampled, seq, cu, index, prefill

    def get_latency(self, op, sampled, *args, **kwargs):
        if base.Config.mode == base.consts.BenchMode.CUDAGRAPH:
            if self._graph_latencies is None:
                self._graph_latencies = self._measure_graph_pair(
                    sampled, *args, **kwargs
                )
            return self._graph_latencies[op]
        # Each provider owns a stable buffer. Copying is outside the timer.
        return super().get_latency(op, sampled.clone(), *args, **kwargs)

    def _measure_graph_pair(self, sampled, *args, **kwargs):
        # Equal graph sizes avoid provider-specific runtime estimates.
        calls_per_graph = 1024
        providers = tuple(op for op in (self.torch_op, self.gems_op) if op is not None)
        graphs = {}
        buffers = {}
        outputs = {}
        references = {}
        for op in providers:
            buffers[op] = sampled.clone()
            for _ in range(20):
                op(buffers[op], *args, **kwargs)
            references[op] = op(sampled.clone(), *args, **kwargs)
        torch.cuda.synchronize()
        for op in providers:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(calls_per_graph):
                    outputs[op] = op(buffers[op], *args, **kwargs)
            graphs[op] = graph

        # Warm both graphs together, then verify repeated replay is stable.
        for _ in range(10):
            for op in providers:
                graphs[op].replay()
        torch.cuda.synchronize()
        for op in providers:
            for actual, expected in zip(outputs[op], references[op]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(buffers[op], sampled, rtol=0, atol=0)
        if self.torch_op is not None:
            for native, gems in zip(outputs[self.torch_op], outputs[self.gems_op]):
                torch.testing.assert_close(gems, native, rtol=0, atol=0)

        samples = {op: [] for op in providers}
        # --iter is a measurement budget in ms per provider.
        budget_seconds = len(providers) * base.Config.repetition / 1000
        deadline = time.perf_counter() + budget_seconds
        round_id = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        while round_id < 10 or time.perf_counter() < deadline:
            order = providers if round_id % 2 == 0 else providers[::-1]
            for op in order:
                start.record()
                graphs[op].replay()
                end.record()
                end.synchronize()
                samples[op].append(start.elapsed_time(end) / calls_per_graph)
            round_id += 1
        return {op: statistics.median(values) for op, values in samples.items()}


@pytest.mark.get_num_sampled_and_rejected
def test_get_num_sampled_and_rejected_perf():
    if vllm_op is None:
        print(
            "vLLM baseline unavailable; baseline latency and speedup are N/A. "
            f"({_VLLM_IMPORT_ERROR})"
        )
    bench = GetNumSampledAndRejectedBenchmark(
        op_name="get_num_sampled_and_rejected",
        torch_op=vllm_op,
        dtypes=[torch.int32],
    )
    bench.set_gems(flaggems_vllm.get_num_sampled_and_rejected)
    bench.run()
