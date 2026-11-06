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

import flaggems_vllm

device = torch.device(flaggems_vllm.device)
VOCAB_SIZE = 128
PROMPT_LEN = 1

# The committed history is [prompt=5, output=10]. input_ids contains the
# current input token followed by speculative tokens: [11, 12, 13].
# A logit at local position p sees input_ids[0:p] as prior tokens.
ALL_TOKEN_IDS = [5, 10]
INPUT_IDS = [11, 12, 13]
LOCAL_POS = [0, 1, 2]


def _apply(bad_words: list[list[int]]) -> torch.Tensor:
    """Call the public operator using low-level tensors, without vLLM state."""
    num_bad_words = len(bad_words)
    packed_tokens: list[int] = []
    offsets = [0]
    for word in bad_words:
        packed_tokens.extend(word)
        offsets.append(len(packed_tokens))

    # Keep storage non-empty for zero-word cases; max_num_bad_words=0 makes
    # the operator return without launching a kernel.
    token_storage = packed_tokens or [0]
    offset_storage = offsets if num_bad_words else [0]

    logits = torch.zeros((len(INPUT_IDS), VOCAB_SIZE), device=device)
    flaggems_vllm.apply_bad_words(
        logits,
        torch.zeros(len(INPUT_IDS), dtype=torch.int32, device=device),
        torch.tensor([token_storage], dtype=torch.int32, device=device),
        torch.tensor([offset_storage], dtype=torch.int32, device=device),
        torch.tensor([num_bad_words], dtype=torch.int32, device=device),
        torch.tensor([ALL_TOKEN_IDS], dtype=torch.int32, device=device),
        torch.tensor([PROMPT_LEN], dtype=torch.int32, device=device),
        torch.tensor([len(ALL_TOKEN_IDS)], dtype=torch.int32, device=device),
        torch.tensor(INPUT_IDS, dtype=torch.int32, device=device),
        torch.tensor(LOCAL_POS, dtype=torch.int32, device=device),
        num_bad_words,
    )
    return logits.cpu()


@pytest.mark.apply_bad_words
def test_apply_bad_words_empty_list_is_noop():
    out = _apply([])
    assert not torch.isinf(out).any()


@pytest.mark.apply_bad_words
def test_apply_bad_words_prefix_inside_speculative_history():
    # At local position 2, the prior input tokens are [11, 12].
    out = _apply([[11, 12, 40]])
    expected = torch.zeros_like(out)
    expected[2, 40] = -float("inf")
    torch.testing.assert_close(out, expected)


@pytest.mark.apply_bad_words
def test_apply_bad_words_prefix_spanning_committed_and_speculative_history():
    # At local position 2, the visible history ends in [10, 11, 12].
    out = _apply([[10, 11, 12, 30]])
    expected = torch.zeros_like(out)
    expected[2, 30] = -float("inf")
    torch.testing.assert_close(out, expected)


@pytest.mark.apply_bad_words
def test_apply_bad_words_does_not_duplicate_last_committed_token():
    # The history at local position 2 ends in [11, 12], not [11, 11].
    out = _apply([[11, 11, 50]])
    expected = torch.zeros_like(out)
    torch.testing.assert_close(out, expected)


@pytest.mark.apply_bad_words
def test_apply_bad_words_committed_prefix_before_current_input():
    # At local position 1, the visible history is [10, 11].
    out = _apply([[10, 11, 60]])
    expected = torch.zeros_like(out)
    expected[1, 60] = -float("inf")
    torch.testing.assert_close(out, expected)


@pytest.mark.apply_bad_words
def test_apply_bad_words_single_token_word_masks_every_position():
    out = _apply([[70]])
    expected = torch.zeros_like(out)
    expected[:, 70] = -float("inf")
    torch.testing.assert_close(out, expected)
