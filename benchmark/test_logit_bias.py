# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import statistics
import sys
import time
from contextlib import redirect_stdout

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

from . import base
from .conftest import Config
from .consts import BenchMode

try:
    from vllm.v1.worker.gpu.sample.logit_bias import (
        apply_logit_bias as vllm_apply_logit_bias,
    )
except ImportError:
    vllm_apply_logit_bias = None


class _VllmTableOutput:
    """Forward output immediately, renaming only the generic baseline header."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        formatted = text.replace(
            "Torch Latency (ms)".rjust(20),
            "vLLM Latency (ms)".rjust(20),
        )
        self.stream.write(formatted)
        return len(text)

    def flush(self):
        self.stream.flush()


class _LogitBiasShapes(list):
    """Compact terminal display, retaining tensor shapes for JSON and baselines."""

    def __init__(self, shapes, description=None):
        super().__init__(shapes)
        self.description = description

    def __str__(self):
        return self.description or super().__str__()


class LogitBiasBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch, vocab, allowed_width, bias_width, scenario"
    DEFAULT_SHAPES = [
        (1, 32000, 0, 8, "bias_only"),
        (16, 32000, 16, 8, "allowlist"),
        (64, 128256, 0, 32, "bias_only"),
        (1, 32000, 1, 8, "restore"),
    ]

    def run(self):
        # Keep the shared benchmark and structured records unchanged.
        with redirect_stdout(_VllmTableOutput(sys.stdout)):
            return super().run()

    def set_shapes(self, shape_file_path=None):
        # Operator-specific scenarios are maintained here, like grouped_topk.
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_dtypes(self, user_desired_dtypes):
        super().set_dtypes(user_desired_dtypes)
        if not flaggems_vllm.runtime.device.support_bf16:
            self.to_bench_dtypes = [
                dtype for dtype in self.to_bench_dtypes if dtype != torch.bfloat16
            ]
        if not self.to_bench_dtypes:
            pytest.skip("backend does not support the requested dtypes")

    def get_input_iter(self, dtype):
        for batch, vocab, allowed_width, bias_width, scenario in self.shapes:
            yield self.make_inputs(
                batch, vocab, allowed_width, bias_width, scenario, dtype
            )

    def record_shapes(self, *args, **kwargs):
        return _LogitBiasShapes(
            super().record_shapes(*args, **kwargs), self._shape_description
        )

    def make_inputs(self, batch, vocab, allowed_width, bias_width, scenario, dtype):
        self._shape_description = (
            f"({batch}, {vocab}) allow={allowed_width} " f"bias={bias_width} {scenario}"
        )
        original = torch.randn((batch, vocab), device=self.device, dtype=dtype)
        mapping = torch.arange(batch, device=self.device, dtype=torch.int32)
        pos = torch.zeros(batch, device=self.device, dtype=torch.int32)
        allowed_count = torch.full(
            (batch,), allowed_width, device=self.device, dtype=torch.int32
        )
        allowed = torch.arange(
            max(1, allowed_width), device=self.device, dtype=torch.int32
        ).repeat(batch, 1)
        if scenario == "restore":
            allowed.fill_(vocab - 1)
        bias_count = torch.full(
            (batch,), bias_width, device=self.device, dtype=torch.int32
        )
        bias_ids = torch.arange(bias_width, device=self.device, dtype=torch.int32)
        bias_ids = (bias_ids + 64).repeat(batch, 1)
        bias = torch.full((batch, bias_width), 0.1, device=self.device)
        min_lens = torch.full((batch,), 10, device=self.device, dtype=torch.int32)
        stop_count = torch.full(
            (batch,),
            int(scenario != "bias_only"),
            device=self.device,
            dtype=torch.int32,
        )
        stop_ids = torch.full(
            (batch, 1), vocab - 1, device=self.device, dtype=torch.int32
        )
        restore = torch.full(
            (batch,), int(scenario == "restore"), device=self.device, dtype=torch.int32
        )
        shared = (
            mapping,
            pos,
            allowed_count,
            allowed,
            bias_count,
            bias_ids,
            bias,
            min_lens,
            stop_count,
        )

        return (original, *shared, restore, stop_ids, scenario == "restore")

    def get_latency(self, op, original, *args, **kwargs):
        # base.Benchmark does not restore in-place inputs. Reset outside timing.
        working = original.clone()
        event_cls = getattr(torch_device_fn, "Event", None)
        if Config.mode == BenchMode.CUDAGRAPH:
            raise NotImplementedError(
                "Untimed input reset requires kernel, operator, or wrapper mode"
            )
        if Config.mode == BenchMode.KERNEL and event_cls is None:
            raise NotImplementedError("kernel timing requires backend device events")

        def sample():
            working.copy_(original)
            torch_device_fn.synchronize()
            if Config.mode == BenchMode.KERNEL:
                start = event_cls(enable_timing=True)
                end = event_cls(enable_timing=True)
                start.record()
                op(working, *args, **kwargs)
                end.record()
                end.synchronize()
                return start.elapsed_time(end)
            start = time.perf_counter()
            op(working, *args, **kwargs)
            if Config.mode == BenchMode.OPERATOR:
                torch_device_fn.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if Config.mode == BenchMode.WRAPPER:
                torch_device_fn.synchronize()
            return elapsed

        sample()  # Compile before estimating duration-based iteration counts.
        estimate = max(sample(), 1e-6)
        for _ in range(max(0, int(Config.warm_up / estimate))):
            sample()
        return statistics.median(
            sample() for _ in range(max(1, int(Config.repetition / estimate)))
        )


@pytest.mark.apply_logit_bias
def test_apply_logit_bias_latency():
    probe = getattr(torch_device_fn, "is_available", None)
    if probe is None:
        probe = getattr(torch_device_fn, "device_count", None)
    if not Config.query and (
        flaggems_vllm.device == "cpu" or (probe is not None and not probe())
    ):
        pytest.skip("requires an available accelerator")
    bench = LogitBiasBenchmark(
        op_name="apply_logit_bias",
        torch_op=vllm_apply_logit_bias,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
        is_inplace=True,
    )
    bench.set_gems(flaggems_vllm.apply_logit_bias)
    bench.run()
