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

"""Benchmark the complete compute_topk_scores call."""

import importlib
import statistics
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import triton.testing

import flaggems_vllm
from flaggems_vllm import runtime

try:
    from vllm.v1.worker.gpu.sample import logprob as baseline_logprob
except ImportError:
    baseline_logprob = None


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    vocab_size: int
    k: int
    custom_count: int = 0
    logits_mode: bool = False
    dtype: torch.dtype = torch.float32


CASES = (
    Case("decode-b1", 1, 151936, 5),
    Case("decode-b16", 16, 151936, 5),
    Case("decode-b64", 64, 151936, 5),
    Case("decode-b256", 256, 151936, 5),
    Case("decode-b16-fp16", 16, 151936, 5, dtype=torch.float16),
    Case("decode-b16-bf16", 16, 151936, 5, dtype=torch.bfloat16),
    Case("medium-k-b1", 1, 151936, 20),
    Case("medium-k-b64", 64, 151936, 20),
    Case("sample-only-b1", 1, 151936, 0),
    Case("sample-only-b64", 64, 151936, 0),
    Case("custom-b16", 16, 151936, 5, custom_count=8),
    Case("custom-b64", 64, 151936, 5, custom_count=8),
    Case("logits-b1", 1, 151936, 5, logits_mode=True),
    Case("logits-b64", 64, 151936, 5, logits_mode=True),
    Case("logits-k0-b1", 1, 151936, 0, logits_mode=True),
    Case("small-vocab-b1", 1, 32000, 5),
    Case("small-vocab-b64", 64, 32000, 5),
    Case("wide-k", 1, 32000, 1025),
    Case("full-vocab-logprob", 1, 32000, 32000),
    Case("full-vocab-logprob-b64", 64, 32000, 32000),
    Case("full-vocab-logits", 1, 32000, 32000, logits_mode=True),
    Case("full-vocab-logits-b64", 64, 32000, 32000, logits_mode=True),
)
WARMUP_MS = 50
REPETITION_MS = 300
REPEATS = 5
RESULTS = []


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(flaggems_vllm.device)


@pytest.fixture(scope="module")
def candidate_logprob():
    return importlib.import_module("flaggems_vllm.ops.logprob")


def _make_inputs(candidate, case: Case, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(2026)
    logits = torch.randn(
        (case.batch_size, case.vocab_size), device=device, dtype=torch.float32
    ).to(dtype)
    sampled = torch.tensor(
        [(7 + row * 31) % case.vocab_size for row in range(case.batch_size)],
        dtype=torch.int64,
        device=device,
    )
    if not case.custom_count:
        return logits, sampled, None, None

    state = candidate.LogprobTokenIdsState(case.batch_size, device)
    requested_ids = list(range(11, 11 + case.custom_count))
    for row in range(case.batch_size):
        ids = requested_ids if row % 2 else []
        state.add_request(row, SimpleNamespace(logprob_token_ids=ids))
    state.apply_staged_writes()
    mapping = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    return logits, sampled, state, mapping


class LogprobBenchmark:
    def __init__(self, calls):
        self.calls = calls

    def run(self):
        samples = {name: [] for name in self.calls}
        names = list(self.calls)
        for repeat in range(REPEATS):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                if flaggems_vllm.vendor_name == "ascend":
                    elapsed_ms = triton.backends.ascend.testing.do_bench_npu(
                        self.calls[name]
                    )
                else:
                    do_bench = (
                        triton.musa_testing.do_bench
                        if flaggems_vllm.device == "musa"
                        else triton.testing.do_bench
                    )
                    elapsed_ms = do_bench(
                        self.calls[name],
                        warmup=WARMUP_MS,
                        rep=REPETITION_MS,
                        return_mode="median",
                    )
                samples[name].append(elapsed_ms)
        return {name: statistics.median(values) for name, values in samples.items()}


def _print_summary():
    if not RESULTS:
        return
    headers = (
        "Case",
        "B",
        "V",
        "K",
        "Custom",
        "Mode",
        "Dtype",
        "FlagGems (us)",
    )
    if baseline_logprob is not None:
        headers += ("vLLM (us)", "Speedup")
    rows = [headers]
    for case, dtype, timings in RESULTS:
        candidate_us = timings["candidate"] * 1000
        row = (
            case.name,
            str(case.batch_size),
            str(case.vocab_size),
            str(case.k),
            str(case.custom_count),
            "logits" if case.logits_mode else "logprob",
            str(dtype).split(".", 1)[-1],
            f"{candidate_us:.2f}",
        )
        if "vllm" in timings:
            baseline_us = timings["vllm"] * 1000
            row += (f"{baseline_us:.2f}", f"{baseline_us / candidate_us:.3f}x")
        rows.append(row)

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row):
        return (
            "| "
            + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))
            + " |"
        )

    print("\ncompute_topk_scores performance (do_bench GPU time)")
    print(f"Device: {flaggems_vllm.device}")
    timer = (
        "Ascend do_bench_npu defaults"
        if flaggems_vllm.vendor_name == "ascend"
        else f"Warmup: {WARMUP_MS} ms; rep: {REPETITION_MS} ms"
    )
    print(
        f"{timer}; repeats: {REPEATS}"
        + ("; speedup = vLLM / FlagGems" if baseline_logprob is not None else "")
    )
    print(separator)
    print(format_row(rows[0]))
    print(separator)
    for row in rows[1:]:
        print(format_row(row))
    print(separator)


@pytest.fixture(scope="module", autouse=True)
def report_performance_results():
    yield
    _print_summary()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.compute_topk_scores
def test_compute_topk_scores_perf(candidate_logprob, device, case):
    dtype = case.dtype
    if dtype == torch.bfloat16 and not runtime.device.support_bf16:
        pytest.skip("bfloat16 is unavailable on this backend")
    logits, sampled, state, mapping = _make_inputs(
        candidate_logprob, case, device, dtype
    )
    kwargs = dict(
        logprob_token_ids_state=state,
        expanded_idx_mapping=mapping,
        max_per_req_token_ids=case.custom_count,
        logits_mode=case.logits_mode,
    )
    calls = {
        "candidate": lambda: candidate_logprob.compute_topk_scores(
            logits, case.k, sampled, **kwargs
        ),
    }
    if baseline_logprob is not None:
        calls["vllm"] = lambda: baseline_logprob.compute_topk_scores(
            logits, case.k, sampled, **kwargs
        )
    for call in calls.values():
        call()
    runtime.torch_device_fn.synchronize()
    RESULTS.append((case, dtype, LogprobBenchmark(calls).run()))
