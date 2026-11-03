# SPDX-License-Identifier: Apache-2.0
"""Benchmark trace replay against the vLLM implementation."""

import contextlib
import io

import pytest
import torch

from flaggems_vllm import apply_trace_tokens as flaggems_apply_trace_tokens

try:
    from vllm.v1.worker.gpu.sample.trace_replay import (
        apply_trace_tokens as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None

from . import base
from .conftest import Config


_kernel_mark = pytest.MarkDecorator(
    pytest.Mark("_trace_replay_kernel", (), {}, _ispytest=True),
    _ispytest=True,
)
pytestmark = [
    pytest.mark.apply_trace_tokens,
    _kernel_mark,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(vllm_reference is None, reason="vLLM trace replay unavailable"),
]


class TraceReplayBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(1, 2048), (2, 2048), (3, 2048), (32, 2048), (128, 2048)]
    DEFAULT_SHAPE_DESC = "num_reqs, max_model_len"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latency_ms = {}

    def get_latency(self, op, *args, **kwargs):
        latency_ms = super().get_latency(op, *args, **kwargs)
        shape = (args[0].shape[0], args[2].shape[1])
        measured = self._latency_ms.setdefault(shape, {})
        if op is self.torch_op:
            measured["vllm"] = latency_ms
        elif op is self.gems_op:
            measured["flaggems"] = latency_ms
        return latency_ms

    def run(self):
        if Config.query:
            return super().run()
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                super().run()
        except BaseException:
            print(output.getvalue(), end="")
            raise
        self.print_summary()

    def print_summary(self):
        if not self._latency_ms:
            return
        print("\nTrace replay latency (CUDA event, microseconds)")
        print("Requests  Trace slots  vLLM ref  FlagGems  Speedup")
        print("-" * 52)
        for num_reqs, max_model_len in self.shapes:
            measured = self._latency_ms.get((num_reqs, max_model_len), {})
            if "vllm" not in measured or "flaggems" not in measured:
                continue
            baseline_us = measured["vllm"] * 1000
            candidate_us = measured["flaggems"] * 1000
            print(
                f"{num_reqs:>8}  {max_model_len:>11}  "
                f"{baseline_us:>8.3f}  {candidate_us:>8.3f}  "
                f"{baseline_us / candidate_us:>7.3f}x"
            )

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for num_reqs, max_model_len in self.shapes:
            sampled = torch.full((num_reqs,), -7, dtype=dtype, device=self.device)
            mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
            traces = torch.arange(
                num_reqs * max_model_len, dtype=torch.int32, device=self.device
            ).reshape(num_reqs, max_model_len)
            lengths = torch.full((num_reqs,), 3, dtype=torch.int32, device=self.device)
            prompt = torch.full((num_reqs,), 5, dtype=torch.int32, device=self.device)
            total = prompt + 1
            if num_reqs >= 3:
                # Exercise replayed, skipped and untraced requests.
                mapping[0] = 1
                mapping[1] = -1
                mapping[2] = 0
                lengths[0] = 0

            arguments = (mapping, traces, lengths, total, prompt)
            reference_sampled = sampled.clone()
            candidate_sampled = sampled.clone()
            assert vllm_reference(reference_sampled, *arguments) is None
            assert (
                flaggems_apply_trace_tokens(candidate_sampled, *arguments) is None
            )
            torch.testing.assert_close(
                candidate_sampled, reference_sampled, rtol=0, atol=0
            )
            yield (sampled, *arguments)


def test_trace_replay_perf():
    bench = TraceReplayBenchmark(
        op_name="apply_trace_tokens",
        torch_op=vllm_reference,
        dtypes=[torch.int64],
        is_inplace=True,
    )
    bench.set_gems(flaggems_apply_trace_tokens)
    bench.run()
