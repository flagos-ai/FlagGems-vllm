# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
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

from benchmark import base
from benchmark.conftest import Config


_kernel_mark = pytest.MarkDecorator(
    pytest.Mark("_trace_replay_kernel", (), {}, _ispytest=True),
    _ispytest=True,
)
pytestmark = [
    pytest.mark.apply_trace_tokens,
    _kernel_mark,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


class TraceReplayBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (n, 2048) for n in (1, 2, 3, 5, 32, 33, 127, 128, 129, 256, 257, 4096)
    ]
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
        if vllm_reference is None:
            self.init_user_config()
            for dtype in self.to_bench_dtypes:
                for inputs in self.get_input_iter(dtype):
                    args, kwargs = self.unpack_to_args_kwargs(inputs)
                    self.get_latency(self.gems_op, *args, **kwargs)
            self.print_summary()
            return
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
        mode_label = {
            "kernel": "CUDA event",
            "cudagraph": "CUDA Graph",
        }.get(Config.mode.value, Config.mode.value)
        print(f"\nTrace replay latency ({mode_label}, microseconds)")
        if vllm_reference is None:
            print("Requests  Trace slots  FlagGems")
            print("-" * 34)
        else:
            print("Requests  Trace slots  vLLM ref  FlagGems  Speedup")
            print("-" * 52)
        for num_reqs, max_model_len in self.shapes:
            measured = self._latency_ms.get((num_reqs, max_model_len), {})
            if "flaggems" not in measured:
                continue
            candidate_us = measured["flaggems"] * 1000
            if vllm_reference is None:
                print(f"{num_reqs:>8}  {max_model_len:>11}  {candidate_us:>8.3f}")
                continue
            if "vllm" not in measured:
                continue
            baseline_us = measured["vllm"] * 1000
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
            candidate_sampled = sampled.clone()
            assert (
                flaggems_apply_trace_tokens(candidate_sampled, *arguments) is None
            )
            if vllm_reference is not None:
                reference_sampled = sampled.clone()
                assert vllm_reference(reference_sampled, *arguments) is None
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
