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
from dataclasses import dataclass
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("TLE bad-words benchmarks require CUDA", allow_module_level=True)

flashinfer_testing = pytest.importorskip("flashinfer.testing")
bench_gpu_time_with_cupti = flashinfer_testing.bench_gpu_time_with_cupti

from flaggems_vllm.ops.bad_words import apply_bad_words as apply_bad_words_impl

try:
    from vllm.v1.worker.gpu.sample.bad_words import (
        apply_bad_words as apply_bad_words_reference,
    )
except (ImportError, OSError):
    apply_bad_words_reference = None

DEVICE = torch.device("cuda")
DTYPE = torch.float32
VOCAB_SIZE = 128_000
PROMPT_LEN = 128
DEFAULT_OUTPUT_LEN = 32
MAX_BAD_WORDS = 128
MAX_BAD_WORD_TOKENS = 1024
MATCH_PATTERNS = ("miss", "hit", "mixed")
BENCH_DRY_RUN = 50
BENCH_ITERS = 300
BENCH_REPEAT = 5
BENCH_GRAPH = True
BENCH_COLD_L2 = True
PERF_RESULTS = []


@dataclass(frozen=True)
class Workload:
    name: str
    num_reqs: int
    num_draft_tokens: int
    bad_words_per_req: int
    bad_word_len: int
    distribution: str = "dense"

    @property
    def num_tokens(self) -> int:
        return self.num_reqs * (self.num_draft_tokens + 1)


@dataclass(frozen=True)
class Inputs:
    logits: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    bad_word_token_ids: torch.Tensor
    bad_word_offsets: torch.Tensor
    num_bad_words: torch.Tensor
    all_token_ids: torch.Tensor
    prompt_len: torch.Tensor
    total_len: torch.Tensor
    input_ids: torch.Tensor
    expanded_local_pos: torch.Tensor
    max_num_bad_words: int


WORKLOADS = (
    Workload("decode_latency", 1, 0, 1, 1),
    Workload("decode_small", 8, 0, 4, 2),
    Workload("decode_medium", 32, 0, 8, 4),
    Workload("decode_large", 128, 0, 16, 4),
    Workload("decode_max_batch", 512, 0, 8, 4),
    Workload("spec_latency", 1, 4, 8, 3),
    Workload("spec_small", 8, 4, 16, 4),
    Workload("spec_medium", 32, 4, 32, 4),
    Workload("spec_large", 64, 8, 16, 8),
    Workload("max_bad_words", 32, 0, 128, 1),
    Workload("max_total_tokens", 32, 0, 64, 16),
    Workload("long_prefix", 8, 0, 16, 64),
    Workload("one_heavy", 128, 0, 128, 4, "one_heavy"),
    Workload("mostly_empty", 128, 0, 64, 4, "mostly_empty"),
    Workload("mixed_counts", 128, 4, 128, 4, "mixed"),
)


def _bad_word_counts(workload: Workload) -> list[int]:
    if workload.distribution == "dense":
        return [workload.bad_words_per_req] * workload.num_reqs
    if workload.distribution == "one_heavy":
        return [workload.bad_words_per_req] + [1] * (workload.num_reqs - 1)
    if workload.distribution == "mostly_empty":
        active = max(1, workload.num_reqs // 10)
        return [workload.bad_words_per_req] * active + [0] * (
            workload.num_reqs - active
        )
    if workload.distribution == "mixed":
        counts = (0, 1, 8, 32, 128)
        return [counts[index % len(counts)] for index in range(workload.num_reqs)]
    raise ValueError(f"unknown distribution: {workload.distribution}")


def _make_bad_word(
    visible_tokens: list[int],
    length: int,
    word_index: int,
    should_match: bool,
) -> list[int]:
    prefix_len = length - 1
    if prefix_len:
        prefix = visible_tokens[-prefix_len:].copy()
        if not should_match:
            prefix[0] = (prefix[0] + 1) % VOCAB_SIZE
    else:
        prefix = []
    last_token = (10_000 + word_index) % VOCAB_SIZE
    return prefix + [last_token]


def make_inputs(workload: Workload, match_pattern: str) -> Inputs:
    torch.manual_seed(0)
    output_len = max(DEFAULT_OUTPUT_LEN, workload.bad_word_len - 1)
    sequence_len = PROMPT_LEN + output_len
    rows_per_req = workload.num_draft_tokens + 1

    all_token_ids = torch.arange(
        workload.num_reqs * sequence_len,
        dtype=torch.int32,
    ).reshape(workload.num_reqs, sequence_len)
    all_token_ids %= VOCAB_SIZE - 20_000
    draft_tokens = (
        torch.arange(
            workload.num_reqs * workload.num_draft_tokens,
            dtype=torch.int32,
        ).reshape(workload.num_reqs, workload.num_draft_tokens)
        + 20_000
    )
    draft_tokens %= VOCAB_SIZE

    counts = _bad_word_counts(workload)
    bad_word_token_ids = torch.zeros(
        (workload.num_reqs, MAX_BAD_WORD_TOKENS),
        dtype=torch.int32,
    )
    bad_word_offsets = torch.zeros(
        (workload.num_reqs, MAX_BAD_WORDS + 1),
        dtype=torch.int32,
    )

    for req_idx, count in enumerate(counts):
        output = all_token_ids[req_idx, PROMPT_LEN:].tolist()
        drafts = draft_tokens[req_idx].tolist()
        visible_tokens = output + drafts
        cursor = 0
        for word_idx in range(count):
            should_match = match_pattern == "hit" or (
                match_pattern == "mixed" and word_idx % 2 == 0
            )
            word = _make_bad_word(
                visible_tokens,
                workload.bad_word_len,
                word_idx,
                should_match,
            )
            next_cursor = cursor + len(word)
            bad_word_token_ids[req_idx, cursor:next_cursor] = torch.tensor(
                word,
                dtype=torch.int32,
            )
            cursor = next_cursor
            bad_word_offsets[req_idx, word_idx + 1] = cursor

    input_ids = torch.empty(
        (workload.num_reqs, rows_per_req),
        dtype=torch.int32,
    )
    input_ids[:, 0] = all_token_ids[:, -1]
    if workload.num_draft_tokens:
        input_ids[:, 1:] = draft_tokens

    expanded_idx_mapping = torch.arange(
        workload.num_reqs,
        dtype=torch.int32,
    ).repeat_interleave(rows_per_req)
    expanded_local_pos = torch.arange(
        rows_per_req,
        dtype=torch.int32,
    ).repeat(workload.num_reqs)

    return Inputs(
        logits=torch.zeros(
            (workload.num_tokens, VOCAB_SIZE),
            dtype=DTYPE,
            device=DEVICE,
        ),
        expanded_idx_mapping=expanded_idx_mapping.to(DEVICE),
        bad_word_token_ids=bad_word_token_ids.to(DEVICE),
        bad_word_offsets=bad_word_offsets.to(DEVICE),
        num_bad_words=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        all_token_ids=all_token_ids.to(DEVICE),
        prompt_len=torch.full(
            (workload.num_reqs,),
            PROMPT_LEN,
            dtype=torch.int32,
            device=DEVICE,
        ),
        total_len=torch.full(
            (workload.num_reqs,),
            sequence_len,
            dtype=torch.int32,
            device=DEVICE,
        ),
        input_ids=input_ids.flatten().to(DEVICE),
        expanded_local_pos=expanded_local_pos.to(DEVICE),
        max_num_bad_words=max(counts),
    )


def _kernel_args(logits: torch.Tensor, inputs: Inputs) -> tuple:
    return (
        logits,
        inputs.expanded_idx_mapping,
        inputs.bad_word_token_ids,
        inputs.bad_word_offsets,
        inputs.num_bad_words,
        inputs.all_token_ids,
        inputs.prompt_len,
        inputs.total_len,
        inputs.input_ids,
        inputs.expanded_local_pos,
        inputs.max_num_bad_words,
    )


def _run_triton(*args) -> None:
    if apply_bad_words_reference is None:
        raise RuntimeError("vLLM bad-words reference is unavailable")
    apply_bad_words_reference(*args)


def _run_tle(*args) -> None:
    apply_bad_words_impl(*args)


def _time_us(fn, input_args: tuple) -> float:
    samples_ms = bench_gpu_time_with_cupti(
        fn,
        dry_run_iters=BENCH_DRY_RUN,
        repeat_iters=BENCH_ITERS,
        use_cuda_graph=BENCH_GRAPH,
        input_args=input_args,
        cold_l2_cache=BENCH_COLD_L2,
    )
    return statistics.median(samples_ms) * 1000


def _measure(calls: dict[str, tuple]) -> dict[str, tuple[float, float, float]]:
    names = list(calls)
    samples = {name: [] for name in names}
    for index in range(BENCH_REPEAT):
        order = names if index % 2 == 0 else list(reversed(names))
        for name in order:
            fn, input_args = calls[name]
            samples[name].append(_time_us(fn, input_args))
    return {
        name: (statistics.median(values), min(values), max(values))
        for name, values in samples.items()
    }


def _print_performance_table() -> None:
    if not PERF_RESULTS:
        return

    headers = (
        "Workload",
        "Match",
        "Reqs",
        "Tokens",
        "Draft",
        "Max Bad Words",
        "Word Len",
        ("vLLM Triton (us)" if apply_bad_words_reference is not None else "TLE (us)"),
        ("TLE (us)" if apply_bad_words_reference is not None else "Mode"),
        ("Triton/TLE" if apply_bad_words_reference is not None else "Reference"),
    )
    rows = [headers]
    for result in PERF_RESULTS:
        row = (
                result["workload"],
                result["match_pattern"],
                str(result["num_reqs"]),
                str(result["num_tokens"]),
                str(result["num_draft_tokens"]),
                str(result["max_num_bad_words"]),
                str(result["bad_word_len"]),
                (
                    f'{result["triton_us"]:.3f}'
                    if result["triton_us"] is not None
                    else f'{result["tle_us"]:.3f}'
                ),
                (
                    f'{result["tle_us"]:.3f}'
                    if result["triton_us"] is not None
                    else "-"
                ),
                (
                    f'{result["speedup"]:.3f}x'
                    if result["speedup"] is not None
                    else "unavailable"
                ),
        )
        rows.append(row)

    widths = [
        max(len(row[index]) for row in rows) for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
            + " |"
        )

    print("\n\nvLLM Bad Words performance summary "
          "(CUDA Graph, cold L2, median)")
    print(
        f"Warmup replays: {BENCH_DRY_RUN}; "
        f"Timed samples: {BENCH_ITERS}; Repeats: {BENCH_REPEAT}"
    )
    print(separator)
    print(format_row(rows[0]))
    print(separator)
    for row in rows[1:]:
        print(format_row(row))
    print(separator)
    if apply_bad_words_reference is None:
        print("vLLM reference unavailable: reporting TLE latency only.")
    else:
        print(
            "Triton/TLE > 1: TLE is faster; "
            "Triton/TLE < 1: vLLM Triton is faster."
        )


@pytest.fixture(scope="module", autouse=True)
def report_performance_results():
    yield
    _print_performance_table()


@pytest.mark.parametrize("match_pattern", MATCH_PATTERNS)
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_benchmark_bad_words(workload: Workload, match_pattern: str):
    inputs = make_inputs(workload, match_pattern)
    tle_logits = inputs.logits.clone()
    tle_args = _kernel_args(tle_logits, inputs)

    if apply_bad_words_reference is not None:
        triton_logits = inputs.logits.clone()
        triton_args = _kernel_args(triton_logits, inputs)
        _run_triton(*triton_args)
        _run_tle(*tle_args)
        torch.testing.assert_close(tle_logits, triton_logits, rtol=0, atol=0)
        calls = {
            "triton": (_run_triton, triton_args),
            "tle": (_run_tle, tle_args),
        }
    else:
        calls = {"tle": (_run_tle, tle_args)}

    timing = _measure(calls)
    triton_us = timing["triton"][0] if "triton" in timing else None
    tle_us = timing["tle"][0]
    speedup = triton_us / tle_us if triton_us is not None else None

    PERF_RESULTS.append(
        {
            "workload": workload.name,
            "match_pattern": match_pattern,
            "num_reqs": workload.num_reqs,
            "num_tokens": workload.num_tokens,
            "num_draft_tokens": workload.num_draft_tokens,
            "max_num_bad_words": inputs.max_num_bad_words,
            "bad_word_len": workload.bad_word_len,
            "triton_us": triton_us,
            "tle_us": tle_us,
            "speedup": speedup,
        }
    )

