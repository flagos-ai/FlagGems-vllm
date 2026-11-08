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

from __future__ import annotations

from copy import copy
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.device_info import get_device_capability
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

_TLE_CLUSTER_SUPPORTED = (
    runtime.device.vendor_name == "nvidia"
    and bool(getattr(runtime.device.info, "tle_enabled", False))
    and get_device_capability() >= (9, 0)
    and has_triton_tle_attrs(
        (
            "device_mesh",
            "shard_id",
            "distributed_barrier",
            "remote",
            "gpu.alloc",
            "gpu.local_ptr",
            "gpu.smem",
        ),
        3,
        6,
        0,
    )
)
if _TLE_CLUSTER_SUPPORTED:
    import triton.experimental.tle.language as tle
else:
    tle = None

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


try:
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor

    HAS_VLLM_BUFFERS = True
except ImportError:
    HAS_VLLM_BUFFERS = False


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


HAS_TLE = _TLE_CLUSTER_SUPPORTED

# Upper bound on the topk kernel's per-iteration gather width.
_MAX_TOPK_BLOCK = 1024
_FULL_SORT_VOCAB_THRESHOLD = 1024
_BATCH_CLUSTER_THRESHOLD = 59
_SMALL_BATCH_CLUSTER_SIZE = 16
_LARGE_BATCH_CLUSTER_SIZE = 8
_SMALL_BATCH_MESH = (
    tle.device_mesh({"block_cluster": [("cluster_x", _SMALL_BATCH_CLUSTER_SIZE)]})
    if HAS_TLE
    else None
)
_LARGE_BATCH_MESH = (
    tle.device_mesh({"block_cluster": [("cluster_x", _LARGE_BATCH_CLUSTER_SIZE)]})
    if HAS_TLE
    else None
)


def _tuning_configs(name: str):
    configs = runtime.get_tuned_config(name)
    if configs:
        return configs
    meta = {} if name == "logprob_fill_ids" else {"BLOCK_SIZE": 1024}
    return [triton.Config(meta, num_warps=4, num_stages=3)]


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    ranks_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk_ids_stride,
    sampled_token_ids_ptr,
    valid_mask_ptr,
    valid_mask_stride,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
    COMPUTE_RANK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
):
    req_idx = (tl.program_id(0) // CLUSTER_SIZE).to(tl.int64)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    row_ptr = logits_ptr + req_idx * logits_stride

    if COMPUTE_RANK:
        sampled_token_id = tl.load(sampled_token_ids_ptr + req_idx)
        sampled_logit = tl.load(row_ptr + sampled_token_id)
        rank = 0

    max_val = float("-inf")
    se = 0.0
    for i in range(cluster_rank * BLOCK_SIZE, vocab_size, CLUSTER_SIZE * BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        valid = block < vocab_size
        logits = tl.load(row_ptr + block, mask=valid, other=float("-inf"))
        if COMPUTE_RANK:
            rank += tl.sum((valid & (logits >= sampled_logit)).to(tl.int32))
        logits = logits.to(tl.float32)
        next_max = tl.maximum(max_val, tl.max(logits))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        se = se * tl.exp(max_val - safe_max) + tl.sum(
            tl.where(valid, tl.exp(logits - safe_max), 0.0)
        )
        max_val = next_max

    partial_lse = tle.gpu.alloc([2], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
    tl.store(tle.gpu.local_ptr(partial_lse, (0,)), max_val)
    tl.store(tle.gpu.local_ptr(partial_lse, (1,)), se)
    if COMPUTE_RANK:
        partial_rank = tle.gpu.alloc(
            [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem
        )
        tl.store(tle.gpu.local_ptr(partial_rank, (0,)), rank)
    tle.distributed_barrier(mesh)

    if cluster_rank == 0:
        merged_max = float("-inf")
        for peer in tl.static_range(CLUSTER_SIZE):
            remote_lse = tle.remote(partial_lse, peer, scope=mesh)
            peer_max = tl.load(tle.gpu.local_ptr(remote_lse, (0,)))
            merged_max = tl.maximum(merged_max, peer_max)

        safe_max = tl.where(merged_max == float("-inf"), 0.0, merged_max)
        merged_se = 0.0
        merged_rank = 0
        for peer in tl.static_range(CLUSTER_SIZE):
            remote_lse = tle.remote(partial_lse, peer, scope=mesh)
            peer_max = tl.load(tle.gpu.local_ptr(remote_lse, (0,)))
            peer_se = tl.load(tle.gpu.local_ptr(remote_lse, (1,)))
            merged_se += peer_se * tl.exp(peer_max - safe_max)
            if COMPUTE_RANK:
                remote_rank = tle.remote(partial_rank, peer, scope=mesh)
                merged_rank += tl.load(tle.gpu.local_ptr(remote_rank, (0,)))
        lse = tl.log(merged_se)

        for j in range(0, topk, TOPK_BLOCK_SIZE):
            k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
            k_mask = k_offset < topk
            topk_ids = tl.load(
                topk_ids_ptr + req_idx * topk_ids_stride + k_offset,
                mask=k_mask,
                other=0,
            )
            logits = tl.load(row_ptr + topk_ids, mask=k_mask).to(tl.float32)
            o = logits - merged_max - lse
            if HAS_MASK:
                valid = tl.load(
                    valid_mask_ptr + req_idx * valid_mask_stride + k_offset,
                    mask=k_mask,
                    other=False,
                )
                o = tl.where(valid, o, float("-inf"))
            tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)

        if COMPUTE_RANK:
            # Match the original rank kernel's padded -inf comparisons.
            padded_tail = ((vocab_size + 8191) // 8192) * 8192 - vocab_size
            merged_rank += tl.where(sampled_logit == float("-inf"), padded_tail, 0)
            tl.store(ranks_ptr + req_idx, merged_rank)
    tle.distributed_barrier(mesh)


@triton.jit
def _logits_scores_and_ranks_kernel(
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
    HAS_MASK: tl.constexpr,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
):
    req_idx = (tl.program_id(0) // CLUSTER_SIZE).to(tl.int64)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    row_ptr = logits_ptr + req_idx * logits_stride
    sampled_token_id = tl.load(sampled_token_ids_ptr + req_idx)
    sampled_logit = tl.load(row_ptr + sampled_token_id)

    for j in range(
        cluster_rank * SCORES_BLOCK_SIZE,
        num_scores,
        CLUSTER_SIZE * SCORES_BLOCK_SIZE,
    ):
        col = j + tl.arange(0, SCORES_BLOCK_SIZE)
        col_mask = col < num_scores
        token_ids = tl.load(
            token_ids_ptr + req_idx * token_ids_stride + col,
            mask=col_mask,
            other=0,
        )
        scores = tl.load(row_ptr + token_ids, mask=col_mask).to(tl.float32)
        if HAS_MASK:
            valid = tl.load(
                valid_mask_ptr + req_idx * valid_mask_stride + col,
                mask=col_mask,
                other=False,
            )
            scores = tl.where(valid, scores, float("-inf"))
        tl.store(output_ptr + req_idx * num_scores + col, scores, mask=col_mask)

    rank = 0
    for i in range(cluster_rank * BLOCK_SIZE, vocab_size, CLUSTER_SIZE * BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        valid = block < vocab_size
        logits = tl.load(row_ptr + block, mask=valid, other=float("-inf"))
        rank += tl.sum((valid & (logits >= sampled_logit)).to(tl.int32))

    partial_rank = tle.gpu.alloc([1], dtype=tl.int32, layout=None, scope=tle.gpu.smem)
    tl.store(tle.gpu.local_ptr(partial_rank, (0,)), rank)
    tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        merged_rank = 0
        for peer in tl.static_range(CLUSTER_SIZE):
            remote_rank = tle.remote(partial_rank, peer, scope=mesh)
            merged_rank += tl.load(tle.gpu.local_ptr(remote_rank, (0,)))
        # Match the original rank kernel's padded -inf comparisons.
        padded_tail = ((vocab_size + 8191) // 8192) * 8192 - vocab_size
        merged_rank += tl.where(sampled_logit == float("-inf"), padded_tail, 0)
        tl.store(ranks_ptr + req_idx, merged_rank)
    tle.distributed_barrier(mesh)


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
        padded_tail = ((vocab_size + 8191) // 8192) * 8192 - vocab_size
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
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    # Cap the kernel's per-iteration width so very large num_logprobs requests
    # stream the gather in bounded-size chunks, avoiding excessive mem use.
    topk_block_size = min(triton.next_power_of_2(num_logprobs), _MAX_TOPK_BLOCK)
    if HAS_TLE:
        if batch_size < _BATCH_CLUSTER_THRESHOLD:
            cluster_size = _SMALL_BATCH_CLUSTER_SIZE
            mesh = _SMALL_BATCH_MESH
        else:
            cluster_size = _LARGE_BATCH_CLUSTER_SIZE
            mesh = _LARGE_BATCH_MESH
        _tuned_logprob_kernel("logprob_tle_log_softmax", topk_block_size)[
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
            HAS_MASK=valid_mask is not None,
            mesh=mesh,
            CLUSTER_SIZE=cluster_size,
        )
    else:
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


def _topk_indices(logits: torch.Tensor, k: int) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    if k == vocab_size and vocab_size >= _FULL_SORT_VOCAB_THRESHOLD:
        return torch.sort(logits, dim=-1, descending=True).indices
    return torch.topk(logits, k, dim=-1).indices


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

    if max_per_req_token_ids == 0:
        # Fast path: no request asked for custom logprob_token_ids.
        logprob_token_ids = sampled_token_ids.unsqueeze(-1)
        if num_logprobs > 0:
            topk_indices = _topk_indices(logits, num_logprobs)
            logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)
        if not logits_mode:
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
            topk_token_ids = topk_token_ids.to(torch.int32)
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
    if logits_mode:
        num_scores = logprob_token_ids.shape[1]
        scores = logits.new_empty((batch_size, num_scores), dtype=torch.float32)
        scores_block_size = min(triton.next_power_of_2(num_scores), _MAX_TOPK_BLOCK)
        if HAS_TLE:
            if batch_size < _BATCH_CLUSTER_THRESHOLD:
                cluster_size = _SMALL_BATCH_CLUSTER_SIZE
                mesh = _SMALL_BATCH_MESH
            else:
                cluster_size = _LARGE_BATCH_CLUSTER_SIZE
                mesh = _LARGE_BATCH_MESH
            _tuned_logprob_kernel("logprob_tle_logits_ranks", scores_block_size)[
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
                HAS_MASK=max_per_req_token_ids > 0,
                mesh=mesh,
                CLUSTER_SIZE=cluster_size,
            )
        else:
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
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=col < NUM_COLS)
    tl.store(mask_base + col, valid, mask=col < NUM_COLS)


@lru_cache(maxsize=64)
def _tuned_logprob_kernel(name: str, tile_size: int):
    kernel, tile_name, key = {
        "logprob_tle_log_softmax": (
            _topk_log_softmax_kernel,
            "TOPK_BLOCK_SIZE",
            ["vocab_size", "topk", "CLUSTER_SIZE"],
        ),
        "logprob_tle_logits_ranks": (
            _logits_scores_and_ranks_kernel,
            "SCORES_BLOCK_SIZE",
            ["vocab_size", "num_scores", "CLUSTER_SIZE"],
        ),
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
        if HAS_VLLM_BUFFERS and runtime.device.vendor_name == "nvidia":
            self.num_token_ids = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
            self.token_ids = StagedWriteTensor(
                (max_num_reqs, MAX_LOGPROB_TOKEN_IDS),
                dtype=torch.int32,
                device=device,
            )
        else:
            self.num_token_ids = _DeviceBackedTensor(
                (max_num_reqs,), torch.int32, device
            )
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
