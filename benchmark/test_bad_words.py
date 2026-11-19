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

from dataclasses import dataclass

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

from . import base
from .conftest import Config

try:
    from vllm.v1.worker.gpu.sample.bad_words import (
        apply_bad_words as apply_bad_words_reference,
    )
except (ImportError, OSError):
    apply_bad_words_reference = None


DEVICE = torch.device(flaggems_vllm.device)
DTYPE = torch.float32
VOCAB_SIZE = 128_000
PROMPT_LEN = 128
DEFAULT_OUTPUT_LEN = 32
MAX_BAD_WORDS = 128
MAX_BAD_WORD_TOKENS = 1024
MATCH_PATTERNS = ("miss", "hit", "mixed")


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
    expected_masks: tuple[list[int], list[int]]


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
    masked_rows = []
    masked_tokens = []

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
            prefix = word[:-1]
            for local_pos in range(rows_per_req):
                history = output + drafts[:local_pos]
                if not prefix or history[-len(prefix) :] == prefix:
                    masked_rows.append(req_idx * rows_per_req + local_pos)
                    masked_tokens.append(word[-1])
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
        expected_masks=(masked_rows, masked_tokens),
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


def _expected_logits(inputs: Inputs) -> torch.Tensor:
    expected = inputs.logits.clone()
    expected[inputs.expected_masks] = -float("inf")
    return expected


def _shift_reference_input_ids(inputs: Inputs) -> torch.Tensor:
    input_ids = inputs.input_ids.reshape(inputs.total_len.numel(), -1)
    shifted = torch.empty_like(input_ids)
    shifted[:, :-1] = input_ids[:, 1:]
    shifted[:, -1] = 0  # The trailing padding is never part of a draft prefix.
    return shifted.flatten()


def _reference_needs_draft_shift() -> bool:
    # vLLM fixed its repeated-committed-token offset in #52311 (v0.28).
    # Probe both recognized contracts instead of assuming a version's behavior.
    # https://github.com/vllm-project/vllm/pull/52311
    inputs = make_inputs(Workload("reference_probe", 1, 2, 1, 3), "hit")
    expected = _expected_logits(inputs)
    reference = inputs.logits.clone()
    apply_bad_words_reference(*_kernel_args(reference, inputs))
    if torch.equal(reference, expected):
        return False

    # The old offset misses this prefix entirely. Reject any other behavior.
    torch.testing.assert_close(reference, inputs.logits, rtol=0, atol=0)
    reference = inputs.logits.clone()
    arguments = list(_kernel_args(reference, inputs))
    arguments[8] = _shift_reference_input_ids(inputs)
    apply_bad_words_reference(*arguments)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    return True


class ApplyBadWordsBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "workload, match"
    DEFAULT_SHAPES = [
        (workload.name, match_pattern)
        for workload in WORKLOADS
        for match_pattern in MATCH_PATTERNS
    ]

    def __init__(self):
        super().__init__(
            op_name="apply_bad_words",
            torch_op=apply_bad_words_reference,
            gems_op=flaggems_vllm.apply_bad_words,
            dtypes=[DTYPE],
            is_inplace=True,
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)

    def get_input_iter(self, dtype):
        workloads = {workload.name: workload for workload in WORKLOADS}
        shift_reference_inputs = (
            _reference_needs_draft_shift()
            if apply_bad_words_reference is not None
            else False
        )
        for workload_name, match_pattern in self.shapes:
            workload = workloads[workload_name]
            inputs = make_inputs(workload, match_pattern)
            self._case = (
                workload.name,
                match_pattern,
                workload.num_reqs,
                workload.num_tokens,
                workload.num_draft_tokens,
                inputs.max_num_bad_words,
                workload.bad_word_len,
            )

            expected_logits = _expected_logits(inputs)
            flaggems_logits = inputs.logits.clone()
            flaggems_vllm.apply_bad_words(*_kernel_args(flaggems_logits, inputs))
            torch.testing.assert_close(flaggems_logits, expected_logits, rtol=0, atol=0)
            if apply_bad_words_reference is not None:
                self._reference_input_ids = (
                    _shift_reference_input_ids(inputs)
                    if shift_reference_inputs
                    else inputs.input_ids
                )
                reference_logits = inputs.logits.clone()
                reference_args = list(_kernel_args(reference_logits, inputs))
                reference_args[8] = self._reference_input_ids
                apply_bad_words_reference(*reference_args)
                torch.testing.assert_close(
                    reference_logits, expected_logits, rtol=0, atol=0
                )
            yield _kernel_args(inputs.logits, inputs)

    def record_shapes(self, *args, **kwargs):
        return self._case

    def get_latency(self, op, logits, *args, **kwargs):
        if op is apply_bad_words_reference:
            # Reuse the prepared native input without charging conversion time.
            args = (*args[:7], self._reference_input_ids, *args[8:])
        # Masking is idempotent; isolate each implementation's input, then
        # reuse the common benchmark timing without resetting every iteration.
        return super().get_latency(op, logits.clone(), *args, **kwargs)


@pytest.mark.apply_bad_words
def test_apply_bad_words():
    probe = getattr(torch_device_fn, "is_available", None)
    if probe is None:
        probe = getattr(torch_device_fn, "device_count", None)
    if not Config.query and (
        DEVICE.type == "cpu" or (probe is not None and not probe())
    ):
        pytest.skip("requires an available accelerator")
    ApplyBadWordsBenchmark().run()
