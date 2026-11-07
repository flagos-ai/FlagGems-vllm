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
import triton
import triton.language as tl

import flaggems_vllm

try:
    from vllm.v1.worker.gpu.sample.bad_words import (
        apply_bad_words as apply_bad_words_reference,
    )
except (ImportError, OSError):
    apply_bad_words_reference = None

# vLLM.
@triton.jit
def _upstream_bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    bw_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)
    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos
    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos

    offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len
    start = tl.load(offsets_base + bw_idx)
    end = tl.load(offsets_base + bw_idx + 1)
    prefix_len = end - start - 1
    if prefix_len > effective_len:
        return
    last_token = tl.load(tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i
        if actual_pos >= output_len:
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset + 1)
        else:
            actual = tl.load(output_base + actual_pos)
        match = match & (expected == actual)
    if match:
        tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))


def _upstream_reference(
    logits,
    expanded_idx_mapping,
    bad_word_token_ids,
    bad_word_offsets,
    num_bad_words,
    all_token_ids,
    prompt_len,
    total_len,
    input_ids,
    expanded_local_pos,
    max_num_bad_words,
):
    num_tokens = logits.shape[0]
    if num_tokens == 0 or max_num_bad_words == 0:
        return
    _upstream_bad_words_kernel[(num_tokens, max_num_bad_words)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_token_ids.stride(0),
        bad_word_offsets,
        bad_word_offsets.stride(0),
        num_bad_words,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
    )


REFERENCE_FN = apply_bad_words_reference or _upstream_reference
REFERENCE_NAME = "vLLM" if apply_bad_words_reference is not None else "vLLM Triton"


DEVICE = torch.device(flaggems_vllm.device)
DTYPE = torch.float32
VOCAB_SIZE = 128_000
PROMPT_LEN = 128
DEFAULT_OUTPUT_LEN = 32
MAX_BAD_WORDS = 128
MAX_BAD_WORD_TOKENS = 1024
MATCH_PATTERNS = ("miss", "hit", "mixed")
BENCH_WARMUP_MS = 25
BENCH_REP_MS = 100
BENCH_REPEAT = 3
PERF_RESULTS = []
TIMING_BACKEND = "backend-aware Triton do_bench"


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


def _run_reference(*args) -> None:
    REFERENCE_FN(*args)


def _run_flaggems(*args) -> None:
    flaggems_vllm.apply_bad_words(*args)


def _backend_do_bench(fn, warmup: int, rep: int, return_mode="median") -> float:
    """Use the timing API supported by the active accelerator backend."""
    if flaggems_vllm.vendor_name == "ascend":
        return triton.backends.ascend.testing.do_bench_npu(fn)
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode=return_mode,
    )


def _time_us(fn, input_args: tuple) -> float:
    latency_ms = _backend_do_bench(
        lambda: fn(*input_args),
        warmup=BENCH_WARMUP_MS,
        rep=BENCH_REP_MS,
        return_mode="median",
    )
    return latency_ms * 1000


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
        f"{REFERENCE_NAME} (us)",
        "FlagGems-vllm (us)",
        "Ref/FlagGems",
    )
    rows = [headers]
    for result in PERF_RESULTS:
        rows.append(
            (
                result["workload"],
                result["match_pattern"],
                str(result["num_reqs"]),
                str(result["num_tokens"]),
                str(result["num_draft_tokens"]),
                str(result["max_num_bad_words"]),
                str(result["bad_word_len"]),
                f'{result["reference_us"]:.3f}',
                f'{result["flaggems_us"]:.3f}',
                f'{result["speedup"]:.3f}x',
            )
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    print(f"\n\nBad Words performance summary ({TIMING_BACKEND}, median)")
    print(f"Active FlagGems backend: {flaggems_vllm.vendor_name} ({flaggems_vllm.device})")
    print(
        f"Warmup: {BENCH_WARMUP_MS} ms; Repetition: {BENCH_REP_MS} ms; "
        f"Outer repeats: {BENCH_REPEAT}"
    )
    print(separator)
    print(format_row(rows[0]))
    print(separator)
    for row in rows[1:]:
        print(format_row(row))
    print(separator)
    print("Ref/FlagGems > 1: FlagGems-vllm is faster.")


@pytest.fixture(scope="module", autouse=True)
def report_performance_results():
    yield
    _print_performance_table()


@pytest.mark.apply_bad_words
@pytest.mark.parametrize("match_pattern", MATCH_PATTERNS)
@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda item: item.name)
def test_apply_bad_words(workload: Workload, match_pattern: str):
    inputs = make_inputs(workload, match_pattern)
    flaggems_logits = inputs.logits.clone()
    flaggems_args = _kernel_args(flaggems_logits, inputs)

    reference_logits = inputs.logits.clone()
    reference_args = _kernel_args(reference_logits, inputs)
    _run_reference(*reference_args)
    _run_flaggems(*flaggems_args)
    torch.testing.assert_close(flaggems_logits, reference_logits, rtol=0, atol=0)

    timing = _measure(
        {
            "reference": (_run_reference, reference_args),
            "flaggems": (_run_flaggems, flaggems_args),
        }
    )
    reference_us = timing["reference"][0]
    flaggems_us = timing["flaggems"][0]
    speedup = reference_us / flaggems_us

    PERF_RESULTS.append(
        {
            "workload": workload.name,
            "match_pattern": match_pattern,
            "num_reqs": workload.num_reqs,
            "num_tokens": workload.num_tokens,
            "num_draft_tokens": workload.num_draft_tokens,
            "max_num_bad_words": inputs.max_num_bad_words,
            "bad_word_len": workload.bad_word_len,
            "reference_us": reference_us,
            "flaggems_us": flaggems_us,
            "speedup": speedup,
        }
    )
