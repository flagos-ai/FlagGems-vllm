# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Compare public thinking-budget entry points on identical workloads.

Cold/incremental workloads use stepwise kernel event timing with 10 warmup
and 100 measured steps. Other timing modes are not implemented for these cases.
The standard --warmup/--iter millisecond budgets apply to steady-state modes.
"""

import statistics

import pytest
import torch

try:
    from vllm.v1.worker.gpu.sample.thinking_budget import (
        apply_thinking_budget as vllm_reference,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import apply_thinking_budget from vLLM. "
        "Check that the installed vLLM version provides this API "
        "and its dependencies are installed correctly."
    ) from exc

from benchmark import base, consts
from benchmark.conftest import Config
from flaggems_vllm import apply_thinking_budget, runtime

pytestmark = [
    pytest.mark.apply_thinking_budget,
    pytest.mark.skipif(
        not runtime.torch_device_fn.is_available(),
        reason="requires the active accelerator",
    ),
]


class ThinkingBudgetBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "mode, history_len, batch_size"
    DEFAULT_SHAPES = [
        (mode, history_len, batch_size)
        for mode, batch_sizes in (
            ("cached", (1,)),
            ("incremental-decode", (1,)),
            ("cold-prefill", (1,)),
            ("cold-resume-worst-case", (1,)),
            ("forced-end", (1,)),
            ("batched-budgeted", (1, 8, 32)),
            ("batched-mixed", (8, 32)),
        )
        for batch_size in batch_sizes
        for history_len in (128, 16384, 32768)
    ]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    WARMUP_STEPS = 10
    ITERATIONS = 100

    def __init__(self):
        super().__init__(
            op_name="apply_thinking_budget",
            torch_op=vllm_reference,
            gems_op=apply_thinking_budget,
            dtypes=[torch.float32],
            is_inplace=True,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for self._case in self.shapes:
            mode, history_len, batch_size = self._case
            extra_tokens = (
                self.WARMUP_STEPS + self.ITERATIONS
                if mode == "incremental-decode"
                else 0
            )
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
            self._original = [args[index].clone() for index in (0, 5, 8, 9, 10)]
            yield args

    def record_shapes(self, *args, **kwargs):
        return self._case

    def get_latency(self, op, *args, **kwargs):
        device_api = runtime.torch_device_fn
        mode, history_len, _ = self._case
        cold_scan = mode.startswith("cold-")
        incremental = mode == "incremental-decode"
        if (cold_scan or incremental) and Config.mode != consts.BenchMode.KERNEL:
            raise NotImplementedError(
                f"{mode} requires --mode kernel for stepwise event timing"
            )

        def restore():
            for index, original in zip((0, 5, 8, 9, 10), self._original):
                args[index].copy_(original)

        with device_api.device(args[0].device):
            # Finish compilation/tuning before replaying the workload.
            restore()
            if incremental:
                args[5].fill_(history_len + 1)
            op(*args, **kwargs)
            device_api.synchronize()
            restore()
            if not cold_scan and not incremental:
                # These upstream cases reach a fixed state before measurement.
                return super().get_latency(op, *args, **kwargs)

            start = device_api.Event(enable_timing=True)
            end = device_api.Event(enable_timing=True)
            samples = []
            for step in range(self.WARMUP_STEPS + self.ITERATIONS):
                if cold_scan:
                    args[8].fill_(-1)
                    args[9].fill_(-1)
                    args[10].zero_()
                else:
                    # The history tail is already filled with token 10.
                    args[5].fill_(history_len + step + 1)
                device_api.synchronize()
                start.record()
                op(*args, **kwargs)
                end.record()
                end.synchronize()
                if step >= self.WARMUP_STEPS:
                    samples.append(start.elapsed_time(end))
            return statistics.median(samples)


def test_thinking_budget_perf():
    ThinkingBudgetBenchmark().run()
