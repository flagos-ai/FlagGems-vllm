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
"""TLE-tiled implementation of the vLLM GPU bad-words operator."""

import torch

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

try:
    import triton.experimental.tle.language as tle

    HAS_TLE = hasattr(tle, "load")
except ImportError:  # pragma: no cover - depends on the Triton distribution
    tle = None
    HAS_TLE = False


if HAS_TLE:

    @triton.jit
    def _bad_words_load(ptr, mask=None, other=None):
        return tle.load(ptr, mask=mask, other=other)

else:

    @triton.jit
    def _bad_words_load(ptr, mask=None, other=None):
        return tl.load(ptr, mask=mask, other=other)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("bad_words"),
    key=["NUM_TOKENS", "MAX_NUM_BAD_WORDS"],
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _bad_words_kernel(
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
    NUM_TOKENS: tl.constexpr,
    MAX_NUM_BAD_WORDS: tl.constexpr,
    BAD_WORDS_TILE: tl.constexpr,
    PREFIX_TILE: tl.constexpr,
    REVERSE_PREFIX: tl.constexpr,
    SINGLE_TOKEN_FAST: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    bad_word_tile_idx = tl.program_id(1).to(tl.int64)

    req_state_idx = _bad_words_load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = _bad_words_load(num_bad_words_ptr + req_state_idx)

    bad_word_idx = (
        bad_word_tile_idx * BAD_WORDS_TILE + tl.arange(0, BAD_WORDS_TILE)
    )
    active_word = bad_word_idx < num_bad_words

    offsets_base = (
        bad_word_offsets_ptr
        + req_state_idx.to(tl.int64) * bad_word_offsets_stride
    )
    tokens_base = (
        bad_word_token_ids_ptr
        + req_state_idx.to(tl.int64) * bad_word_token_ids_stride
    )
    start = _bad_words_load(
        offsets_base + bad_word_idx,
        mask=active_word,
        other=0,
    )
    end = _bad_words_load(
        offsets_base + bad_word_idx + 1,
        mask=active_word,
        other=0,
    )
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    valid_word = active_word & (bad_word_len > 0)
    last_token = _bad_words_load(
        tokens_base + end - 1,
        mask=valid_word,
        other=0,
    )
    if SINGLE_TOKEN_FAST:
        single_token_word = valid_word & (bad_word_len == 1)
        tl.store(
            logits_ptr + token_idx * logits_stride + last_token,
            -float("inf"),
            mask=single_token_word,
        )
        matches = valid_word & (bad_word_len > 1)
        if tl.sum(matches.to(tl.int32), axis=0) == 0:
            return
    else:
        matches = valid_word

    pos = _bad_words_load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos.to(tl.int64)
    prompt_len = _bad_words_load(prompt_len_ptr + req_state_idx)
    total_len = _bad_words_load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos
    output_base = (
        all_token_ids_ptr
        + req_state_idx.to(tl.int64) * all_token_ids_stride
        + prompt_len
    )

    matches &= prefix_len <= effective_len
    max_prefix_len = tl.max(tl.where(matches, prefix_len, 0), axis=0)

    prefix_lane = tl.arange(0, PREFIX_TILE)[None, :]
    for prefix_base in tl.range(
        0,
        max_prefix_len,
        PREFIX_TILE,
        num_stages=1,
    ):
        if REVERSE_PREFIX:
            prefix_offset = prefix_len[:, None] - 1 - prefix_base - prefix_lane
            compare_mask = matches[:, None] & (prefix_offset >= 0)
        else:
            prefix_offset = prefix_base + prefix_lane
            compare_mask = matches[:, None] & (prefix_offset < prefix_len[:, None])

        expected = _bad_words_load(
            tokens_base + start[:, None] + prefix_offset,
            mask=compare_mask,
            other=0,
        )

        actual_pos = effective_len - prefix_len[:, None] + prefix_offset
        from_spec_input = actual_pos >= output_len

        committed = _bad_words_load(
            output_base + actual_pos,
            mask=compare_mask & ~from_spec_input,
            other=0,
        )
        spec_offset = actual_pos - output_len
        speculative = _bad_words_load(
            input_ids_ptr + cur_req_first_pos + spec_offset + 1,
            mask=compare_mask & from_spec_input,
            other=0,
        )
        actual = tl.where(from_spec_input, speculative, committed)
        mismatch_count = tl.sum(
            (compare_mask & (expected != actual)).to(tl.int32),
            axis=1,
        )
        chunk_matches = mismatch_count == 0
        matches &= chunk_matches

    tl.store(
        logits_ptr + token_idx * logits_stride + last_token,
        -float("inf"),
        mask=matches,
    )


def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
) -> None:
    """Apply vLLM bad-word masking with a TLE-tiled Triton kernel."""
    num_tokens = logits.shape[0]
    if num_tokens == 0 or max_num_bad_words == 0:
        return

    if max_num_bad_words == 1:
        bad_words_tile = 1
        prefix_tile = 1
    elif max_num_bad_words <= 4:
        bad_words_tile = 4
        prefix_tile = 1
    elif max_num_bad_words <= 8:
        bad_words_tile = 8
        prefix_tile = 4
    elif max_num_bad_words <= 16 and num_tokens <= 8:
        bad_words_tile = 32
        prefix_tile = 16
    elif max_num_bad_words <= 16:
        bad_words_tile = 16
        prefix_tile = 4
    else:
        bad_words_tile = 32
        prefix_tile = 8

    grid = (num_tokens, triton.cdiv(max_num_bad_words, bad_words_tile))
    _bad_words_kernel[grid](
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
        NUM_TOKENS=num_tokens,
        MAX_NUM_BAD_WORDS=max_num_bad_words,
        BAD_WORDS_TILE=bad_words_tile,
        PREFIX_TILE=prefix_tile,
        REVERSE_PREFIX=max_num_bad_words >= 128,
        SINGLE_TOKEN_FAST=(
            max_num_bad_words == 1
            or (max_num_bad_words >= 128 and num_tokens <= 32)
        ),
    )


__all__ = ["apply_bad_words"]


