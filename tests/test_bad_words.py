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

import pytest
import torch

try:
    from vllm.platforms import current_platform
except (ImportError, OSError):
    pytest.skip("vLLM is required for the bad-words reference state", allow_module_level=True)

if not torch.cuda.is_available() or not current_platform.is_cuda():
    pytest.skip("bad-words tests require CUDA", allow_module_level=True)

from flaggems_vllm.ops.bad_words import apply_bad_words
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.states import RequestState

try:
    from vllm.v1.worker.gpu.sample.logits_processor import (
        LogitsProcRequestState,
    )
except ModuleNotFoundError:
    LogitsProcRequestState = None

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128

# Committed tokens: prompt [5], output [10, 11]. Draft tokens: [12, 13].
# Local position 0 holds the last committed token; drafts start at position 1.
PROMPT_LEN = 1
COMMITTED = [5, 10, 11]
INPUT_IDS = [11, 12, 13]
LOCAL_POS = [0, 1, 2]


def _make_state(bad_words_token_ids: list[list[int]]) -> tuple[BadWordsState, int]:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=PROMPT_LEN,
        all_token_ids=COMMITTED,
        num_computed_tokens=len(COMMITTED),
        max_tokens=32,
    )
    req_states.apply_staged_writes()

    req_idx = req_states.req_id_to_index["req"]
    if LogitsProcRequestState is None:
        state = BadWordsState(req_states)
    else:
        state = BadWordsState(
            None,
            LogitsProcRequestState.from_request_state(req_states),
        )
    state.add_request(
        req_idx,
        SamplingParams(_bad_words_token_ids=bad_words_token_ids),
    )
    state.apply_staged_writes()
    return state, req_idx


def _apply_state(
    state: BadWordsState,
    req_idx: int,
    max_num_bad_words: int,
) -> torch.Tensor:
    num_logits = len(INPUT_IDS)
    logits = torch.zeros((num_logits, VOCAB_SIZE), device=DEVICE)
    expanded_idx_mapping = torch.tensor(
        [req_idx] * num_logits,
        dtype=torch.int32,
        device=DEVICE,
    )
    input_ids = torch.tensor(INPUT_IDS, dtype=torch.int32, device=DEVICE)
    expanded_local_pos = torch.tensor(
        LOCAL_POS,
        dtype=torch.int32,
        device=DEVICE,
    )

    apply_bad_words(
        logits,
        expanded_idx_mapping,
        state.bad_word_token_ids.gpu,
        state.bad_word_offsets.gpu,
        state.num_bad_words.gpu,
        state.req_states.all_token_ids.gpu,
        state.req_states.prompt_len.gpu,
        state.req_states.total_len.gpu,
        input_ids,
        expanded_local_pos,
        max_num_bad_words,
    )
    return logits.cpu()


def _apply(bad_words_token_ids: list[list[int]]) -> torch.Tensor:
    state, req_idx = _make_state(bad_words_token_ids)
    return _apply_state(state, req_idx, len(bad_words_token_ids))


def test_bad_words_slot_reuse_clears_state():
    state, req_idx = _make_state([[10, 11, 60]])
    state.add_request(req_idx, SamplingParams())
    state.apply_staged_writes()

    assert int(state.num_bad_words.np[req_idx]) == 0
    out = _apply_state(state, req_idx, max_num_bad_words=0)
    assert not torch.isinf(out).any()


def test_bad_words_prefix_inside_draft_tokens():
    out = _apply([[12, 13, 40]])
    expected = torch.zeros_like(out)
    expected[2, 40] = -float("inf")
    torch.testing.assert_close(out, expected)


def test_bad_words_prefix_spanning_committed_and_draft_tokens():
    out = _apply([[11, 12, 30]])
    expected = torch.zeros_like(out)
    expected[1, 30] = -float("inf")
    torch.testing.assert_close(out, expected)


def test_bad_words_no_spurious_match_from_last_committed_token():
    out = _apply([[11, 11, 50]])
    expected = torch.zeros_like(out)
    torch.testing.assert_close(out, expected)


def test_bad_words_committed_prefix():
    out = _apply([[10, 11, 60]])
    expected = torch.zeros_like(out)
    expected[0, 60] = -float("inf")
    torch.testing.assert_close(out, expected)


