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
"""MThreads implementation of the vLLM GPU bad-words operator."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _bad_words_load(ptr, mask=None, other=None):
    return tl.load(ptr, mask=mask, other=other)


@triton.jit
def _bad_words_match_short_prefix(
    matches,
    prefix_len,
    effective_len,
    output_len,
    start,
    tokens_base,
    output_base,
    input_ids_ptr,
    cur_req_first_pos,
    PREFIX_WIDTH: tl.constexpr,
):
    prefix_lane = tl.arange(0, PREFIX_WIDTH)[None, :]
    prefix_mask = matches[:, None] & (prefix_lane < prefix_len[:, None])
    expected = _bad_words_load(
        tokens_base + start[:, None] + prefix_lane,
        mask=prefix_mask,
        other=0,
    )
    actual_pos = effective_len - prefix_len[:, None] + prefix_lane
    from_spec = actual_pos >= output_len
    committed = _bad_words_load(
        output_base + actual_pos,
        mask=prefix_mask & ~from_spec,
        other=0,
    )
    spec_offset = actual_pos - output_len
    speculative = _bad_words_load(
        input_ids_ptr + cur_req_first_pos + spec_offset + 1,
        mask=prefix_mask & from_spec,
        other=0,
    )
    actual = tl.where(from_spec, speculative, committed)
    mismatch_count = tl.sum(
        (prefix_mask & (expected != actual)).to(tl.int32),
        axis=1,
    )
    return matches & (mismatch_count == 0)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("apply_bad_words"),
    key=["NUM_TOKENS", "MAX_NUM_BAD_WORDS"],
    use_cuda_graph=False,
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
    # Token and bad-word indices fit in int32; widen only the vocab-strided
    # logits offset, which can exceed int32 for large batches.
    token_idx = tl.program_id(0)
    bad_word_tile_idx = tl.program_id(1)
    logits_row_offset = token_idx.to(tl.int64) * logits_stride

    req_state_idx = _bad_words_load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = _bad_words_load(num_bad_words_ptr + req_state_idx)

    bad_word_idx = bad_word_tile_idx * BAD_WORDS_TILE + tl.arange(0, BAD_WORDS_TILE)
    active_word = bad_word_idx < num_bad_words

    offsets_base = (
        bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    )
    tokens_base = (
        bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
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
            logits_ptr + logits_row_offset + last_token,
            -float("inf"),
            mask=single_token_word,
        )
        matches = valid_word & (bad_word_len > 1)
        if tl.sum(matches.to(tl.int32), axis=0) == 0:
            return
    else:
        matches = valid_word

    pos = _bad_words_load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos
    prompt_len = _bad_words_load(prompt_len_ptr + req_state_idx)
    total_len = _bad_words_load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos
    output_base = (
        all_token_ids_ptr
        + req_state_idx * all_token_ids_stride
        + prompt_len
    )

    matches &= prefix_len <= effective_len
    max_prefix_len = tl.max(tl.where(matches, prefix_len, 0), axis=0)

    if max_prefix_len == 0:
        pass
    elif max_prefix_len <= 1:
        matches = _bad_words_match_short_prefix(
            matches,
            prefix_len,
            effective_len,
            output_len,
            start,
            tokens_base,
            output_base,
            input_ids_ptr,
            cur_req_first_pos,
            1,
        )
    elif max_prefix_len <= 2:
        matches = _bad_words_match_short_prefix(
            matches,
            prefix_len,
            effective_len,
            output_len,
            start,
            tokens_base,
            output_base,
            input_ids_ptr,
            cur_req_first_pos,
            2,
        )
    elif max_prefix_len <= 4:
        matches = _bad_words_match_short_prefix(
            matches,
            prefix_len,
            effective_len,
            output_len,
            start,
            tokens_base,
            output_base,
            input_ids_ptr,
            cur_req_first_pos,
            4,
        )
    else:
        long_lane = tl.arange(0, PREFIX_TILE)[None, :]
        for prefix_base in tl.range(
            0,
            max_prefix_len,
            PREFIX_TILE,
            num_stages=1,
        ):
            if REVERSE_PREFIX:
                long_offset = prefix_len[:, None] - 1 - prefix_base - long_lane
                long_mask = matches[:, None] & (long_offset >= 0)
            else:
                long_offset = prefix_base + long_lane
                long_mask = matches[:, None] & (long_offset < prefix_len[:, None])

            long_expected = _bad_words_load(
                tokens_base + start[:, None] + long_offset,
                mask=long_mask,
                other=0,
            )

            long_actual_pos = effective_len - prefix_len[:, None] + long_offset
            long_from_spec = long_actual_pos >= output_len

            long_committed = _bad_words_load(
                output_base + long_actual_pos,
                mask=long_mask & ~long_from_spec,
                other=0,
            )
            long_spec_offset = long_actual_pos - output_len
            long_speculative = _bad_words_load(
                input_ids_ptr + cur_req_first_pos + long_spec_offset + 1,
                mask=long_mask & long_from_spec,
                other=0,
            )
            long_actual = tl.where(
                long_from_spec,
                long_speculative,
                long_committed,
            )
            long_mismatch_count = tl.sum(
                (long_mask & (long_expected != long_actual)).to(tl.int32),
                axis=1,
            )
            matches &= long_mismatch_count == 0

    tl.store(
        logits_ptr + logits_row_offset + last_token,
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
    """Apply vLLM bad-word masking with the MThreads Triton kernel."""
    num_tokens = logits.shape[0]
    if num_tokens == 0 or max_num_bad_words == 0:
        return

    if max_num_bad_words > 16:
        bad_words_tile = 32
    else:
        tile_cap = 4 if num_tokens <= 8 else 8
        bad_words_tile = min(tile_cap, triton.next_power_of_2(max_num_bad_words))

    prefix_tile = 64 if num_tokens <= 8 and max_num_bad_words == 16 else 8
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
            max_num_bad_words == 1 or (max_num_bad_words >= 128 and num_tokens <= 32)
        ),
    )


__all__ = ["apply_bad_words"]
