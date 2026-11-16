# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Public-entry workloads adapted from vLLM's thinking-budget benchmark.

Source: benchmarks/kernels/benchmark_thinking_budget.py at
947dd62c0700851a79b0a8f32cb823e25a003b31.
"""

import statistics
import time
from pathlib import Path

import pytest
import torch

from baseline.thinking_budget import apply_thinking_budget as vllm_reference
from benchmark import base, consts
from benchmark.conftest import Config
from flaggems_vllm import apply_thinking_budget, runtime

pytestmark = [
    pytest.mark.apply_thinking_budget,
    pytest.mark.skipif(
        runtime.device.vendor_name != "nvidia"
        or not runtime.torch_device_fn.is_available(),
        reason="NVIDIA event-timing benchmark",
    ),
]


class ThinkingBudgetBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "mode, history_len, batch_size"
    DEFAULT_SHAPES = [("cached", 128, 1)]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]

    def __init__(self):
        super().__init__(
            op_name="apply_thinking_budget",
            torch_op=vllm_reference,
            gems_op=apply_thinking_budget,
            dtypes=[torch.float32],
            is_inplace=True,
        )

    def init_user_config(self):
        if Config.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "apply_thinking_budget requires --mode kernel; CUDA Graph is disabled"
            )
        super().init_user_config()

    def get_input_iter(self, dtype):
        for self._case in self.shapes:
            mode, history_len, batch_size = self._case
            extra_tokens = int(mode == "incremental-decode")
            history = torch.full(
                (batch_size, history_len + extra_tokens + 1),
                10,
                dtype=torch.int32,
                device=self.device,
            )
            start_pos = history_len - 1 if mode == "cold-prefill" else 1
            history[:, start_pos] = 90
            # RequestState allocates request slots in reverse order.
            mapping = torch.arange(
                batch_size - 1, -1, -1, dtype=torch.int32, device=self.device
            )
            budget = (
                max(1, history_len - 2)
                if mode == "forced-end"
                else history_len + extra_tokens + 1
            )
            budgets = torch.full(
                (batch_size,), budget, dtype=torch.int32, device=self.device
            )
            if mode == "batched-mixed":
                budgets[mapping[1::2].long()] = -1
            args = (
                torch.zeros((batch_size, 128), dtype=dtype, device=self.device),
                mapping,
                mapping,
                budgets,
                history,
                torch.full(
                    (batch_size,), history_len, dtype=torch.int32, device=self.device
                ),
                torch.full((batch_size,), 10, dtype=torch.int32, device=self.device),
                torch.zeros(batch_size, dtype=torch.int32, device=self.device),
                torch.full((batch_size,), -1, dtype=torch.int32, device=self.device),
                torch.full((batch_size,), -1, dtype=torch.int32, device=self.device),
                torch.zeros(batch_size, dtype=torch.int32, device=self.device),
                torch.tensor([90], dtype=torch.int32, device=self.device),
                torch.tensor([91], dtype=torch.int32, device=self.device),
                torch.tensor([91], dtype=torch.int32, device=self.device),
            )
            if not mode.startswith("cold-"):
                vllm_reference(*args)
            if mode == "incremental-decode":
                # Replay the same next-token step for both implementations.
                args[5].fill_(history_len + 1)
            self._original = [args[index].clone() for index in (0, 8, 9, 10)]
            yield args

    def record_shapes(self, *args, **kwargs):
        return self._case

    def get_latency(self, op, *args, **kwargs):
        device_api = runtime.torch_device_fn

        def restore():
            for index, original in zip((0, 8, 9, 10), self._original):
                args[index].copy_(original)

        with device_api.device(args[0].device):
            restore()
            op(*args, **kwargs)
            device_api.synchronize()
            deadline = time.perf_counter() + Config.warm_up / 1000.0
            while time.perf_counter() < deadline:
                restore()
                op(*args, **kwargs)
                device_api.synchronize()
            start = device_api.Event(enable_timing=True)
            end = device_api.Event(enable_timing=True)
            samples = []
            deadline = time.perf_counter() + Config.repetition / 1000.0
            while not samples or time.perf_counter() < deadline:
                restore()
                start.record()
                op(*args, **kwargs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return statistics.median(samples)


def test_thinking_budget_perf():
    ThinkingBudgetBenchmark().run()
