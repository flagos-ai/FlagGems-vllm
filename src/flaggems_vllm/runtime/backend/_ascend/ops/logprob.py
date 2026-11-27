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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend implementation of ``compute_topk_scores``.

The selection goes through the Ascend hardware sort because the packed
``uint64`` sort/topk kernels of the shared implementation cannot be lowered
by the Ascend toolchain. The remaining kernels are shared with the generic
module; this file is registered as the Ascend override of the operator.
"""

from __future__ import annotations

from copy import copy
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime.backend._ascend.ops.topk import topk as _ascend_topk
from flaggems_vllm.utils import libentry, libtuner


try:
    from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS, SamplingParams
except ImportError:
    MAX_LOGPROB_TOKEN_IDS = 128
    SamplingParams = object

try:
    from vllm.v1.outputs import LogprobsTensors
except ImportError:

    class LogprobsTensors(NamedTuple):
        logprob_token_ids: torch.Tensor
        logprobs: torch.Tensor
        selected_token_ranks: torch.Tensor
        cu_num_generated_tokens: list[int] | None
        cu_num_generated_tokens_tensor: torch.Tensor | None


class _DeviceBackedTensor:
    def __init__(self, shape, dtype, device):
        self.np = np.zeros(shape, dtype=np.int32)
        self.gpu = torch.empty(shape, dtype=dtype, device=device)

    def stage_write(self, row, offset, values):
        self.np[row, offset : offset + len(values)] = values

    def apply_write(self):
        self.gpu.copy_(torch.from_numpy(self.np))

    def copy_to_uva(self):
        self.apply_write()


# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024
ORIGINAL_RANK_BLOCK_SIZE = tl.constexpr(8192)


def _tuning_configs(name: str):
    configs = runtime.get_tuned_config(name)
    if configs:
        return configs
    meta = {} if name == "logprob_fill_ids" else {"BLOCK_SIZE": 1024}
    return [triton.Config(meta, num_warps=4, num_stages=3)]


@triton.jit
def _triton_scores_and_ranks_kernel(
    output_ptr,
    ranks_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    token_ids_stride,
    sampled_token_ids_ptr,
    valid_mask_ptr,
    valid_mask_stride,
    num_scores,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SCORES_BLOCK_SIZE: tl.constexpr,
    COMPUTE_RANK: tl.constexpr,
    COMPUTE_LOGPROBS: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    if COMPUTE_RANK:
        sampled_token_id = tl.load(sampled_token_ids_ptr + req_idx)
        sampled_logit = tl.load(row_ptr + sampled_token_id)
        rank = 0

    if COMPUTE_LOGPROBS:
        max_val = float("-inf")
        se = 0.0

    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        valid = block < vocab_size
        logits = tl.load(row_ptr + block, mask=valid, other=float("-inf"))
        if COMPUTE_RANK:
            rank += tl.sum((valid & (logits >= sampled_logit)).to(tl.int32))
        if COMPUTE_LOGPROBS:
            logits = logits.to(tl.float32)
            next_max = tl.maximum(max_val, tl.max(logits))
            safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
            se = se * tl.exp(max_val - safe_max) + tl.sum(
                tl.where(valid, tl.exp(logits - safe_max), 0.0)
            )
            max_val = next_max

    if COMPUTE_LOGPROBS:
        lse = tl.log(se)

    for j in range(0, num_scores, SCORES_BLOCK_SIZE):
        offsets = j + tl.arange(0, SCORES_BLOCK_SIZE)
        score_mask = offsets < num_scores
        ids = tl.load(
            token_ids_ptr + req_idx * token_ids_stride + offsets,
            mask=score_mask,
            other=0,
        )
        scores = tl.load(row_ptr + ids, mask=score_mask).to(tl.float32)
        if COMPUTE_LOGPROBS:
            scores = scores - max_val - lse
        if HAS_MASK:
            valid = tl.load(
                valid_mask_ptr + req_idx * valid_mask_stride + offsets,
                mask=score_mask,
                other=False,
            )
            scores = tl.where(valid, scores, float("-inf"))
        tl.store(output_ptr + req_idx * num_scores + offsets, scores, mask=score_mask)

    if COMPUTE_RANK:
        padded_tail = (
            vocab_size + ORIGINAL_RANK_BLOCK_SIZE - 1
        ) // ORIGINAL_RANK_BLOCK_SIZE * ORIGINAL_RANK_BLOCK_SIZE - vocab_size
        rank += tl.where(sampled_logit == float("-inf"), padded_tail, 0)
        tl.store(ranks_ptr + req_idx, rank)


def compute_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor | None = None,
    token_ranks: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # NOTE(woosuk): To save GPU memory, we do not materialize the full
    # [batch_size, vocab_size] logprobs tensor. The kernel computes
    # max + logsumexp per row and only emits logprobs at `token_ids`.
    batch_size, vocab_size = logits.shape
    assert (sampled_token_ids is None) == (token_ranks is None)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    _tuned_logprob_kernel("logprob_triton_scores_ranks", topk_block_size)[
        (batch_size,)
    ](
        logprobs,
        token_ranks if token_ranks is not None else logprobs,
        logits,
        logits.stride(0),
        token_ids,
        token_ids.stride(0),
        sampled_token_ids if sampled_token_ids is not None else token_ids,
        valid_mask if valid_mask is not None else token_ids,
        valid_mask.stride(0) if valid_mask is not None else 0,
        num_logprobs,
        vocab_size,
        COMPUTE_RANK=token_ranks is not None,
        COMPUTE_LOGPROBS=True,
        HAS_MASK=valid_mask is not None,
    )
    return logprobs


_SELECTION_BLOCK = 1024
_ASSEMBLE_BLOCK = 1024


@libentry()
@triton.jit
def _selection_logits_kernel(
    logits,
    stride,
    out,
    vocab,
    BLOCK: tl.constexpr,
):
    """Cast a logits row to FP32 and canonicalize NaN before sorting.

    The reference top-k ranks NaN above every finite value, while the hardware
    sort leaves NaN comparisons undefined; mapping NaN to ``inf`` makes the
    ordering total. The caller gathers the original values afterwards.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    value = tl.load(logits + row * stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    bits = value.to(tl.int32, bitcast=True)
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    value = tl.where(nan, float("inf"), value)
    tl.store(out + row * vocab + offs, value, mask=mask)


@libentry()
@triton.jit
def _assemble_indices_kernel(
    topk_indices,
    sampled_token_ids,
    out,
    k,
    out_stride,
    PACK_SAMPLED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < k
    indices = tl.load(topk_indices + row * k + cols, mask=mask, other=0)
    tl.store(
        out + row * out_stride + cols + PACK_SAMPLED,
        indices.to(tl.int64),
        mask=mask,
    )
    if PACK_SAMPLED:
        sampled = tl.load(sampled_token_ids + row)
        tl.store(out + row * out_stride, sampled.to(tl.int64))


def _launch_topk_indices(logits, k, sampled_token_ids=None):
    """Select top-k tokens with the Ascend hardware sort.

    The packed-uint64 kernels of the shared implementation cannot be compiled
    by the Ascend toolchain, so selection goes through the backend sort engine
    and the int64 indices are assembled by a Triton kernel.
    """
    batch, vocab = logits.shape
    pack_sampled = sampled_token_ids is not None
    selection = torch.empty((batch, vocab), dtype=torch.float32, device=logits.device)
    _selection_logits_kernel[(batch, triton.cdiv(vocab, _SELECTION_BLOCK))](
        logits,
        logits.stride(0),
        selection,
        vocab,
        BLOCK=_SELECTION_BLOCK,
    )
    _, topk_indices = _ascend_topk(selection, k)
    out = torch.empty(
        (batch, k + (1 if pack_sampled else 0)),
        dtype=torch.int64,
        device=logits.device,
    )
    _assemble_indices_kernel[(batch, triton.cdiv(k, _ASSEMBLE_BLOCK))](
        topk_indices,
        sampled_token_ids if pack_sampled else topk_indices,
        out,
        k,
        out.stride(0),
        PACK_SAMPLED=pack_sampled,
        BLOCK=_ASSEMBLE_BLOCK,
    )
    return out


def _topk_indices(
    logits: torch.Tensor,
    k: int,
    sampled_token_ids: torch.Tensor | None = None,
    fused_scores: torch.Tensor | None = None,
    fused_ranks: torch.Tensor | None = None,
    fused_logprobs: bool = False,
) -> torch.Tensor:
    if k > logits.shape[-1]:
        raise RuntimeError("selected index k out of range")
    return _launch_topk_indices(logits, k, sampled_token_ids)


def compute_topk_scores(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | torch.Tensor | None = None,
    logprob_token_ids_state: "LogprobTokenIdsState | None" = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    max_per_req_token_ids: int = 0,
    logits_mode: bool = False,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    batch_size, vocab_size = logits.shape
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    # The fused chunk-sort chain uses packed uint64 keys that Ascend cannot
    # compile; the plain indices path computes the same values.
    fused_full = False
    if fused_full:
        scores = logits.new_empty((batch_size, vocab_size + 1), dtype=torch.float32)

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            logprob_token_ids = _topk_indices(
                logits,
                num_logprobs,
                sampled_token_ids,
                scores if fused_full else None,
                token_ranks if fused_full else None,
                fused_logprobs=fused_full and not logits_mode,
            )
        if not logits_mode and not fused_full:
            scores = compute_token_logprobs(
                logits, logprob_token_ids, sampled_token_ids, token_ranks
            )
    else:
        # Some requests specified logprob_token_ids. Build the [batch_size,
        # 1 + max_cols] token_ids matrix and validity mask on the GPU via a
        # single triton kernel, overriding the topk columns with per-request
        # tokens where applicable.
        assert logprob_token_ids_state is not None
        assert expanded_idx_mapping is not None

        if num_logprobs > 0:
            topk_token_ids = _topk_indices(logits, num_logprobs)
        else:
            # This tensor just used as an int32 pointer, data not accessed.
            topk_token_ids = logprob_token_ids_state.token_ids.gpu

        num_cols = max(num_logprobs, max_per_req_token_ids)
        logprob_token_ids = sampled_token_ids.new_empty((batch_size, 1 + num_cols))
        valid_mask = torch.empty_like(logprob_token_ids, dtype=torch.bool)
        _tuned_logprob_kernel("logprob_fill_ids", triton.next_power_of_2(num_cols))[
            (batch_size,)
        ](
            logprob_token_ids,
            logprob_token_ids.stride(0),
            valid_mask,
            valid_mask.stride(0),
            sampled_token_ids,
            topk_token_ids,
            topk_token_ids.stride(0),
            expanded_idx_mapping,
            logprob_token_ids_state.num_token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu,
            logprob_token_ids_state.token_ids.gpu.stride(0),
            NUM_TOPK=num_logprobs,
            NUM_COLS=num_cols,
        )
        if not logits_mode:
            scores = compute_token_logprobs(
                logits,
                logprob_token_ids,
                sampled_token_ids,
                token_ranks,
                valid_mask,
            )
    if logits_mode and not fused_full:
        num_scores = logprob_token_ids.shape[1]
        scores = logits.new_empty((batch_size, num_scores), dtype=torch.float32)
        scores_block_size = min(triton.next_power_of_2(num_scores), _MAX_TOPK_BLOCK)
        _tuned_logprob_kernel("logprob_triton_scores_ranks", scores_block_size)[
            (batch_size,)
        ](
            scores,
            token_ranks,
            logits,
            logits.stride(0),
            logprob_token_ids,
            logprob_token_ids.stride(0),
            sampled_token_ids,
            valid_mask if max_per_req_token_ids else logprob_token_ids,
            valid_mask.stride(0) if max_per_req_token_ids else 0,
            num_scores,
            vocab_size,
            COMPUTE_RANK=True,
            COMPUTE_LOGPROBS=False,
            HAS_MASK=max_per_req_token_ids > 0,
        )
    is_tensor = isinstance(cu_num_logits, torch.Tensor)
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=scores,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=None if is_tensor else cu_num_logits,
        cu_num_generated_tokens_tensor=cu_num_logits if is_tensor else None,
    )


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    NUM_COLS: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        valid = col < num_custom
        tokens = tl.load(
            per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride + col,
            mask=valid,
            other=0,
        ).to(tl.int64)
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        valid = col < NUM_TOPK
        tokens = tl.load(
            topk_indices_ptr + batch_idx * topk_indices_stride + col,
            mask=valid,
            other=0,
        ).to(tl.int64)

    tl.store(tid_base + col, tokens, mask=col < NUM_COLS)
    tl.store(mask_base + col, valid, mask=col < NUM_COLS)


@lru_cache(maxsize=64)
def _tuned_logprob_kernel(name: str, tile_size: int):
    kernel, tile_name, key = {
        "logprob_triton_scores_ranks": (
            _triton_scores_and_ranks_kernel,
            "SCORES_BLOCK_SIZE",
            ["vocab_size", "num_scores", "COMPUTE_LOGPROBS"],
        ),
        "logprob_fill_ids": (
            _fill_logprob_token_ids_kernel,
            "PADDED_COLS",
            ["NUM_COLS", "NUM_TOPK"],
        ),
    }[name]
    configs = []
    for base_config in _tuning_configs(name):
        config = copy(base_config)
        config.kwargs = {**base_config.kwargs, tile_name: tile_size}
        configs.append(config)
    return libentry()(libtuner(configs=configs, key=key)(kernel))


class LogprobTokenIdsState:
    """Per-request override of which token ids' logprobs to return.

    See `SamplingParams.logprob_token_ids`.
    """

    def __init__(self, max_num_reqs: int, device: torch.device):
        self.max_num_reqs = max_num_reqs
        self.num_token_ids = _DeviceBackedTensor((max_num_reqs,), torch.int32, device)
        self.token_ids = _DeviceBackedTensor(
            (max_num_reqs, MAX_LOGPROB_TOKEN_IDS), torch.int32, device
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        token_ids = sampling_params.logprob_token_ids
        if not token_ids:
            self.num_token_ids.np[req_idx] = 0
            return
        n = len(token_ids)
        if n > MAX_LOGPROB_TOKEN_IDS:
            raise ValueError(
                f"Too many logprob_token_ids: {n}. The max is {MAX_LOGPROB_TOKEN_IDS}."
            )
        self.num_token_ids.np[req_idx] = n
        self.token_ids.stage_write(req_idx, 0, token_ids)

    def apply_staged_writes(self) -> None:
        self.num_token_ids.copy_to_uva()
        self.token_ids.apply_write()

    def max_num_token_ids(self, idx_mapping_np: np.ndarray) -> int:
        return int(self.num_token_ids.np[idx_mapping_np].max(initial=0))
