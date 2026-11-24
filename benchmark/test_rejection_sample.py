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

"""Ordinary-chain candidate latency via Benchmark, with external per-call reset.

--warmup and --iter are millisecond budgets (not iteration counts). Each case
uses at least 5 warmups and 25 samples. kernel uses synchronized device events;
operator uses synchronized wall time. CUDA Graph/wrapper modes are rejected.
The installed vLLM ordinary-chain callable supplies the same-device baseline
when available. Both implementations use the same inputs, timer and reset.
Each timed case first passes independent point-mass output/state assertions.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from baseline.rejection_sample import native
from benchmark import base
from benchmark import conftest as benchmark_config
from benchmark import consts
from flaggems_vllm.runtime import torch_device_fn

SOURCE_COMMIT = "vllm-project/vllm@428c371b32c18a788def35abea1cb9f579278381"


def _make_case(batch, steps, vocab, path, full_cache, greedy, device):
    """Literal point-mass laws give exact independent per-case qualification."""
    rows = steps + 1
    target = torch.full((batch, rows, vocab), float("-inf"), dtype=torch.float32)
    target[:, :, 1] = 0
    draft = torch.full((batch, steps, vocab), float("-inf"), dtype=torch.float32)
    if steps:
        draft[:, :, 1] = 0
    proposed = torch.ones((batch, steps), dtype=torch.int64)
    rejection = None
    if path == "first":
        rejection = 0
    elif path == "middle":
        rejection = steps // 2
    elif path == "last":
        rejection = steps - 1
    elif path == "placeholder":
        rejection = 0
    if rejection is not None:
        proposed[:, rejection] = -1 if path == "placeholder" else 0
        draft[:, rejection, :] = float("-inf")
        draft[:, rejection, 0] = 0
    packed = torch.zeros((batch, rows), dtype=torch.int64)
    packed[:, 1:] = proposed
    values = draft.to(device) if full_cache else None
    inputs = dict(
        target_logits=target.reshape(batch * rows, vocab).to(device),
        draft_logits_cache=(
            None if values is None else SimpleNamespace(values=values, row_mapping=None)
        ),
        draft_sampled=packed.flatten().to(device),
        cu_num_logits=(torch.arange(batch + 1, dtype=torch.int32) * rows).to(device),
        pos=(torch.arange(batch * rows, dtype=torch.int64) + 97).to(device),
        idx_mapping=torch.arange(batch, dtype=torch.int32, device=device),
        expanded_idx_mapping=torch.arange(
            batch, dtype=torch.int32, device=device
        ).repeat_interleave(rows),
        expanded_local_pos=torch.arange(rows, dtype=torch.int32, device=device).repeat(
            batch
        ),
        temperature=torch.full(
            (batch,), 0.0 if greedy else 1.0, dtype=torch.float32, device=device
        ),
        seed=(torch.arange(batch, dtype=torch.int64) + 20261119).to(device),
        num_speculative_steps=steps,
    )
    expected_count = rows if rejection is None else rejection + 1
    return inputs, expected_count


class RejectionSampleBenchmark(base.Benchmark):
    DEFAULT_DTYPES = (torch.float32,)
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    # Local synthetic contract matrix; no production workload/performance claim.
    DEFAULT_SHAPES = (
        (1, 0, 128, "bonus", False, False),
        (1, 1, 128, "all", True, False),
        (8, 3, 257, "all", True, False),
        (8, 3, 257, "first", True, False),
        (32, 5, 1025, "middle", True, False),
        (32, 5, 1025, "last", True, False),
        (128, 3, 257, "placeholder", True, False),
        (8, 3, 257, "first", False, False),
        (8, 3, 257, "all", False, True),
    )
    DEFAULT_SHAPE_DESC = "batch, steps, vocab, path, full_cache, greedy"

    def __init__(self):
        self._candidate = flaggems_vllm.rejection_sample
        self._native, self._native_identity = native()
        super().__init__(
            op_name="rejection_sample", torch_op=self._native, gems_op=self._candidate
        )

    def set_shapes(self, shape_file):
        # This initial matrix is frozen locally rather than taking unrelated
        # two-dimensional fallback shapes from the common shape YAML.
        if shape_file and Path(shape_file).resolve() not in (
            Path(self.DEFAULT_SHAPE_FILES).resolve(),
            Path(__file__).with_name("core_shapes.yaml").resolve(),
        ):
            raise ValueError("rejection_sample uses its explicit local contract matrix")
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        super().init_user_config()
        cfg = benchmark_config.Config
        if self.mode not in (consts.BenchMode.KERNEL, consts.BenchMode.OPERATOR):
            raise ValueError(
                "rejection_sample requires kernel or operator mode, no Graph"
            )
        if cfg.parallel:
            raise ValueError(
                "rejection_sample measurements must run serially on one device"
            )
        if cfg.base_data is not None:
            raise ValueError(
                "rejection_sample uses a same-device installed vLLM baseline; "
                "offline baseline mapping is unsupported"
            )
        if cfg.warm_up < 0 or cfg.repetition <= 0:
            raise ValueError(
                "--warmup must be nonnegative and --iter positive (milliseconds)"
            )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self.case = shape
            self.torch_op = self._native
            self._case_baseline_reason = self._native_identity.get("reason")
            if (
                self._native
                and shape[3] == "placeholder"
                and not self._native_identity.get("supports_placeholder", False)
            ):
                self.torch_op = None
                self._case_baseline_reason = (
                    "Installed vLLM draft_logits interface has no negative-token guard; "
                    "the -1 placeholder case is unsupported by this baseline"
                )
            inputs, self.expected_count = _make_case(*shape, torch.device(self.device))
            # Base.unpack_to_args_kwargs iterates a tuple. A bare dict would
            # iterate its keys and silently produce no arguments.
            yield (inputs,)

    def record_shapes(self, *args, **kwargs):
        cfg = benchmark_config.Config
        module_path = Path(inspect.getfile(self._candidate))
        common = importlib.import_module("flaggems_vllm.ops.rejection_sampler")
        source_paths = {module_path, Path(common.__file__)}
        candidate_sources = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_paths)
        }
        self.detail = dict(
            shape=list(self.case),
            target_stride=list(kwargs["target_logits"].stride()),
            source=SOURCE_COMMIT,
            candidate_file=str(module_path),
            candidate_sources=candidate_sources,
            candidate_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
            baseline_kind="installed_vllm" if self.torch_op else "missing",
            baseline_status="unqualified" if self.torch_op else "N/A",
            baseline_reason=self._case_baseline_reason,
            native_identity=self._native_identity,
            graph=False,
            reset="all input tensors, seed, pos and cache before every call",
            output="private allocation per call; only returned valid prefix is defined",
            timer=(
                "device-events"
                if self.mode == consts.BenchMode.KERNEL
                else "synchronized-wall"
            ),
            warmup_budget_ms=cfg.warm_up,
            timing_budget_ms=cfg.repetition,
            minimum_warmups=5,
            minimum_samples=25,
        )
        return self.detail

    def get_latency(self, op, *args, **kwargs):
        cfg = benchmark_config.Config
        is_native = op is self._native
        if is_native:
            for entry in self._native_identity["files"].values():
                actual = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
                if actual != entry["sha256"]:
                    raise RuntimeError("installed vLLM source changed before timing")
        tensors = [value for value in kwargs.values() if torch.is_tensor(value)]
        cache = kwargs["draft_logits_cache"]
        if cache is not None:
            tensors.append(cache.values)
            if cache.row_mapping is not None:
                tensors.append(cache.row_mapping)
        snapshots = [value.clone() for value in tensors]
        cpu_rng = torch.random.get_rng_state().clone()
        device_rng = torch_device_fn.get_rng_state().clone()

        def reset():
            for tensor, snapshot in zip(tensors, snapshots):
                tensor.copy_(snapshot)
            torch_device_fn.synchronize()

        def assert_state():
            for tensor, snapshot in zip(tensors, snapshots):
                assert torch.equal(tensor, snapshot), "candidate mutated input"
            assert torch.equal(torch.random.get_rng_state(), cpu_rng)
            assert torch.equal(torch_device_fn.get_rng_state(), device_rng)

        def qualify():
            reset()
            sampled, counts = op(*args, **kwargs)
            torch_device_fn.synchronize()
            assert sampled.dtype == torch.int64 and counts.dtype == torch.int32
            assert tuple(sampled.shape) == (self.case[0], self.case[1] + 1)
            assert tuple(counts.shape) == (self.case[0],)
            assert sampled.device == kwargs["target_logits"].device
            assert counts.device == kwargs["target_logits"].device
            assert bool(torch.all(counts == self.expected_count))
            assert bool(torch.all(sampled[:, : self.expected_count] == 1))
            assert_state()

        qualify()
        warmups, warmed_ms = 0, 0.0
        while warmups < 5 or warmed_ms < cfg.warm_up:
            reset()
            begin = time.perf_counter()
            op(*args, **kwargs)
            torch_device_fn.synchronize()
            warmed_ms += (time.perf_counter() - begin) * 1000
            warmups += 1
        samples = []
        elapsed_ms = 0.0
        if self.mode == consts.BenchMode.KERNEL:
            begin_event = torch_device_fn.Event(enable_timing=True)
            end_event = torch_device_fn.Event(enable_timing=True)
        while len(samples) < 25 or elapsed_ms < cfg.repetition:
            reset()
            if self.mode == consts.BenchMode.KERNEL:
                begin_event.record()
                op(*args, **kwargs)
                end_event.record()
                end_event.synchronize()
                latency = float(begin_event.elapsed_time(end_event))
            else:
                begin = time.perf_counter()
                op(*args, **kwargs)
                torch_device_fn.synchronize()
                latency = (time.perf_counter() - begin) * 1000
            if latency <= 0:
                raise RuntimeError("timer returned a nonpositive duration")
            samples.append(latency)
            elapsed_ms += latency
        assert_state()
        qualify()
        ordered = sorted(samples)
        timing = dict(
            qualification="PASS",
            actual_warmups=warmups,
            actual_samples=len(samples),
            samples_ms=samples,
            p10_ms=ordered[int(0.1 * (len(ordered) - 1))],
            p90_ms=ordered[int(0.9 * (len(ordered) - 1))],
            median_ms=statistics.median(samples),
        )
        if is_native:
            for entry in self._native_identity["files"].values():
                actual = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
                if actual != entry["sha256"]:
                    raise RuntimeError("installed vLLM source changed during timing")
            self.detail["native_timing"] = timing
            self.detail["baseline_status"] = "PASS"
        else:
            self.detail.update(timing)
        return statistics.median(samples)


@pytest.mark.rejection_sample
def test_rejection_sample_benchmark():
    RejectionSampleBenchmark().run()
