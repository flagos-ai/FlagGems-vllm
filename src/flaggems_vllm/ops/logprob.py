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
from flaggems_vllm.ops.top_k_per_row_prefill import top_k_per_row_prefill
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


@triton.jit
def _topk_chunk_candidates_kernel(
    logits_ptr,
    logits_stride,
    candidates_ptr,
    num_chunks,
    vocab_size,
    TOPK_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=offsets < vocab_size,
        other=float("-inf"),
    ).to(tl.float32)
    bits = values.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    packed = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - offsets.to(tl.uint64))
    packed = tl.where(offsets < vocab_size, packed, 0)
    if TOPK_PAD == 1:
        selected = tl.max(packed)
        tl.store(candidates_ptr + row * num_chunks + chunk, selected)
    else:
        selected = tl.topk(packed, TOPK_PAD)
        positions = tl.arange(0, TOPK_PAD)
        tl.store(
            candidates_ptr + (row * num_chunks + chunk) * TOPK_PAD + positions,
            selected,
        )


@triton.jit
def _topk_merge_candidates_kernel(
    candidates_ptr,
    output_ptr,
    sampled_ptr,
    num_chunks,
    num_topk,
    TOPK_PAD: tl.constexpr,
    MERGE_SIZE: tl.constexpr,
    PACK_SAMPLED: tl.constexpr,
):
    row = tl.program_id(0)
    if PACK_SAMPLED:
        tl.store(
            output_ptr + row * (num_topk + 1),
            tl.load(sampled_ptr + row),
        )
    positions = tl.arange(0, MERGE_SIZE)
    candidates = tl.load(
        candidates_ptr + row * num_chunks * TOPK_PAD + positions,
        mask=positions < num_chunks * TOPK_PAD,
        other=0,
    )
    if TOPK_PAD == 1:
        selected = tl.max(candidates)
        token_id = 0xFFFFFFFF - (selected & 0xFFFFFFFF)
        tl.store(
            output_ptr + row * (num_topk + PACK_SAMPLED) + PACK_SAMPLED,
            token_id.to(tl.int64),
        )
    else:
        selected = tl.topk(candidates, TOPK_PAD)
        topk_positions = tl.arange(0, TOPK_PAD)
        token_ids = 0xFFFFFFFF - (selected & 0xFFFFFFFF)
        tl.store(
            output_ptr
            + row * (num_topk + PACK_SAMPLED)
            + PACK_SAMPLED
            + topk_positions,
            token_ids.to(tl.int64),
            mask=topk_positions < num_topk,
        )


@triton.jit
def _sort_token_chunks_kernel(
    logits_ptr,
    sampled_ptr,
    logits_stride,
    sorted_keys_ptr,
    partial_lse_ptr,
    sorted_stride,
    vocab_size,
    NUM_CHUNKS: tl.constexpr,
    SORT_CHUNK: tl.constexpr,
    COMPUTE_LSE: tl.constexpr,
    COMPUTE_RANK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * SORT_CHUNK + tl.arange(0, SORT_CHUNK)
    values = tl.load(
        logits_ptr + row * logits_stride + offsets,
        mask=offsets < vocab_size,
        other=float("-inf"),
    ).to(tl.float32)
    if COMPUTE_RANK:
        sampled_id = tl.load(sampled_ptr + row)
        sampled_value = tl.load(logits_ptr + row * logits_stride + sampled_id).to(
            tl.float32
        )
        chunk_rank = tl.sum(
            ((offsets < vocab_size) & (values >= sampled_value)).to(tl.int32)
        )
        tl.store(
            partial_lse_ptr + (row * 3 + 2) * NUM_CHUNKS + chunk,
            chunk_rank,
            mask=chunk < NUM_CHUNKS,
        )
    if COMPUTE_LSE:
        chunk_max = tl.max(values)
        chunk_sum = tl.sum(tl.exp(values - chunk_max))
        tl.store(
            partial_lse_ptr + row * 3 * NUM_CHUNKS + chunk,
            chunk_max,
            mask=chunk < NUM_CHUNKS,
        )
        tl.store(
            partial_lse_ptr + (row * 3 + 1) * NUM_CHUNKS + chunk,
            chunk_sum,
            mask=chunk < NUM_CHUNKS,
        )
    bits = values.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    packed = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - offsets.to(tl.uint64))
    packed = tl.where(offsets < vocab_size, packed, 0)
    ordered = tl.sort(packed, descending=True)
    tl.store(
        sorted_keys_ptr + row * sorted_stride + offsets,
        ordered,
    )


@triton.jit
def _merge_four_sorted_runs_kernel(
    source_ptr,
    destination_ptr,
    sorted_stride,
    RUN_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, 4 * RUN_SIZE)
    keys = tl.load(source_ptr + row * sorted_stride + group * 4 * RUN_SIZE + offsets)
    pair_local = offsets % (2 * RUN_SIZE)
    first_index = (offsets // (2 * RUN_SIZE)) * 2 * RUN_SIZE + tl.where(
        pair_local < RUN_SIZE,
        pair_local,
        3 * RUN_SIZE - 1 - pair_local,
    )
    pairs = tl.gather(keys, first_index, axis=0)
    pairs = tl.reshape(pairs, (2, 2 * RUN_SIZE))
    pairs = tl.bitonic_merge(pairs, descending=True)
    pairs = tl.reshape(pairs, (4 * RUN_SIZE,))
    second_index = tl.where(
        offsets < 2 * RUN_SIZE,
        offsets,
        6 * RUN_SIZE - 1 - offsets,
    )
    merged = tl.bitonic_merge(tl.gather(pairs, second_index, axis=0), descending=True)
    tl.store(
        destination_ptr + row * sorted_stride + group * 4 * RUN_SIZE + offsets,
        merged,
    )


@triton.jit
def _merge_sorted_token_runs_kernel(
    source_ptr,
    destination_ptr,
    output_ptr,
    sampled_ptr,
    logits_ptr,
    logits_stride,
    scores_ptr,
    ranks_ptr,
    partial_lse_ptr,
    sorted_stride,
    vocab_size,
    num_topk,
    RUN_SIZE: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PACK_SAMPLED: tl.constexpr,
    FINAL: tl.constexpr,
    FUSED_LOGITS: tl.constexpr,
    FUSED_LOGPROBS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    LSE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    positions = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = positions < vocab_size
    pair_start = (positions // (2 * RUN_SIZE)) * (2 * RUN_SIZE)
    first_run = positions - pair_start < RUN_SIZE
    own_start = tl.where(first_run, pair_start, pair_start + RUN_SIZE)
    other_start = tl.where(first_run, pair_start + RUN_SIZE, pair_start)
    own_position = positions - own_start
    other_count = tl.minimum(RUN_SIZE, tl.maximum(vocab_size - other_start, 0))
    key = tl.load(source_ptr + row * sorted_stride + positions, mask=valid, other=0)
    lower = tl.full([BLOCK_SIZE], 0, tl.int32)
    upper = other_count
    for _ in tl.static_range(SEARCH_STEPS):
        middle = (lower + upper) // 2
        other_key = tl.load(
            source_ptr + row * sorted_stride + other_start + middle,
            mask=valid & (middle < other_count),
            other=0,
        )
        greater = (middle < other_count) & (other_key > key)
        lower = tl.where(greater, middle + 1, lower)
        upper = tl.where(greater, upper, middle)
    result_position = pair_start + own_position + lower
    if FINAL:
        token_id = 0xFFFFFFFF - (key & 0xFFFFFFFF)
        tl.store(
            output_ptr
            + row * (num_topk + PACK_SAMPLED)
            + PACK_SAMPLED
            + result_position,
            token_id.to(tl.int64),
            mask=valid & (result_position < num_topk),
        )
        if FUSED_LOGITS or FUSED_LOGPROBS:
            ordered_bits = (key >> 32).to(tl.uint32)
            score_bits = tl.where(
                (ordered_bits & 0x80000000) != 0,
                ordered_bits ^ 0x80000000,
                ~ordered_bits,
            )
            score = score_bits.to(tl.float32, bitcast=True)
            if FUSED_LOGPROBS:
                chunks = tl.arange(0, LSE_BLOCK)
                maxima = tl.load(
                    partial_lse_ptr + row * 3 * NUM_CHUNKS + chunks,
                    mask=chunks < NUM_CHUNKS,
                    other=float("-inf"),
                )
                sums = tl.load(
                    partial_lse_ptr + (row * 3 + 1) * NUM_CHUNKS + chunks,
                    mask=chunks < NUM_CHUNKS,
                    other=0,
                )
                global_max = tl.max(maxima)
                logsumexp = global_max + tl.log(
                    tl.sum(sums * tl.exp(maxima - global_max))
                )
                score -= logsumexp
            tl.store(
                scores_ptr + row * (vocab_size + 1) + 1 + result_position,
                score,
                mask=valid,
            )
        if tl.program_id(1) == 0 and PACK_SAMPLED:
            sampled_id = tl.load(sampled_ptr + row)
            tl.store(output_ptr + row * (num_topk + 1), sampled_id)
            if FUSED_LOGITS or FUSED_LOGPROBS:
                sampled_value = tl.load(
                    logits_ptr + row * logits_stride + sampled_id
                ).to(tl.float32)
                if FUSED_LOGPROBS:
                    tl.store(
                        scores_ptr + row * (vocab_size + 1),
                        sampled_value - logsumexp,
                    )
                else:
                    tl.store(scores_ptr + row * (vocab_size + 1), sampled_value)
                chunks = tl.arange(0, LSE_BLOCK)
                partial_ranks = tl.load(
                    partial_lse_ptr + (row * 3 + 2) * NUM_CHUNKS + chunks,
                    mask=chunks < NUM_CHUNKS,
                    other=0,
                )
                rank = tl.sum(partial_ranks)
                rank += tl.where(
                    sampled_value == float("-inf"),
                    (8192 - vocab_size % 8192) % 8192,
                    0,
                )
                tl.store(ranks_ptr + row, rank.to(tl.int64))
    else:
        tl.store(
            destination_ptr + row * sorted_stride + result_position,
            key,
            mask=valid,
        )


@triton.jit
def _unpack_sorted_token_indices_kernel(
    sorted_keys_ptr,
    sorted_stride,
    output_ptr,
    sampled_ptr,
    vocab_size,
    num_topk,
    BLOCK_SIZE: tl.constexpr,
    PACK_SAMPLED: tl.constexpr,
):
    row = tl.program_id(0)
    if PACK_SAMPLED and tl.program_id(1) == 0:
        tl.store(
            output_ptr + row * (num_topk + 1),
            tl.load(sampled_ptr + row),
        )
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    keys = tl.load(
        sorted_keys_ptr + row * sorted_stride + offsets,
        mask=offsets < num_topk,
        other=0,
    )
    ids = 0xFFFFFFFF - (keys & 0xFFFFFFFF)
    tl.store(
        output_ptr + row * (num_topk + PACK_SAMPLED) + PACK_SAMPLED + offsets,
        ids.to(tl.int64),
        mask=offsets < num_topk,
    )


@triton.jit
def _init_topk_ranges_kernel(
    starts_ptr,
    ends_ptr,
    batch_size,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_SIZE)
    tl.store(starts_ptr + rows, 0, mask=rows < batch_size)
    tl.store(ends_ptr + rows, vocab_size, mask=rows < batch_size)


@triton.jit
def _cast_topk_logits_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        input_ptr + row * input_stride + offsets,
        mask=offsets < vocab_size,
        other=0,
    )
    tl.store(
        output_ptr + row * vocab_size + offsets,
        values.to(tl.float32),
        mask=offsets < vocab_size,
    )


@triton.jit
def _sort_selected_topk_indices_kernel(
    logits_ptr,
    logits_stride,
    selected_ptr,
    output_ptr,
    sampled_ptr,
    num_topk,
    SORT_SIZE: tl.constexpr,
    PACK_SAMPLED: tl.constexpr,
):
    row = tl.program_id(0)
    if PACK_SAMPLED:
        tl.store(
            output_ptr + row * (num_topk + 1),
            tl.load(sampled_ptr + row),
        )
    positions = tl.arange(0, SORT_SIZE)
    ids = tl.load(
        selected_ptr + row * num_topk + positions,
        mask=positions < num_topk,
        other=0,
    )
    values = tl.load(
        logits_ptr + row * logits_stride + ids,
        mask=positions < num_topk,
        other=float("-inf"),
    ).to(tl.float32)
    bits = values.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    packed = (key.to(tl.uint64) << 32) | (0xFFFFFFFF - ids.to(tl.uint64))
    packed = tl.where(positions < num_topk, packed, 0)
    ordered = tl.sort(packed, descending=True)
    token_ids = 0xFFFFFFFF - (ordered & 0xFFFFFFFF)
    tl.store(
        output_ptr + row * (num_topk + PACK_SAMPLED) + PACK_SAMPLED + positions,
        token_ids.to(tl.int64),
        mask=positions < num_topk,
    )


def _topk_indices(
    logits: torch.Tensor,
    k: int,
    sampled_token_ids: torch.Tensor | None = None,
    fused_scores: torch.Tensor | None = None,
    fused_ranks: torch.Tensor | None = None,
    fused_logprobs: bool = False,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    if k > vocab_size:
        raise RuntimeError("selected index k out of range")
    pack_sampled = sampled_token_ids is not None
    if k <= 32:
        batch_size = logits.shape[0]
        block_size = 1024
        topk_pad = triton.next_power_of_2(k)
        num_chunks = triton.cdiv(vocab_size, block_size)
        candidates = torch.empty(
            (batch_size, num_chunks * topk_pad),
            dtype=torch.uint64,
            device=logits.device,
        )
        indices = torch.empty(
            (batch_size, k + pack_sampled), dtype=torch.int64, device=logits.device
        )
        _topk_chunk_candidates_kernel[(batch_size, num_chunks)](
            logits,
            logits.stride(0),
            candidates,
            num_chunks,
            vocab_size,
            TOPK_PAD=topk_pad,
            BLOCK_SIZE=block_size,
        )
        _topk_merge_candidates_kernel[(batch_size,)](
            candidates,
            indices,
            sampled_token_ids if pack_sampled else indices,
            num_chunks,
            k,
            TOPK_PAD=topk_pad,
            MERGE_SIZE=triton.next_power_of_2(num_chunks * topk_pad),
            PACK_SAMPLED=pack_sampled,
        )
        return indices
    if k < vocab_size and k <= 16384:
        batch_size = logits.shape[0]
        if logits.dtype == torch.float32:
            selection_logits = logits
        else:
            selection_logits = torch.empty(
                (batch_size, vocab_size), dtype=torch.float32, device=logits.device
            )
            _cast_topk_logits_kernel[(batch_size, triton.cdiv(vocab_size, 1024))](
                logits,
                selection_logits,
                logits.stride(0),
                vocab_size,
                BLOCK_SIZE=1024,
            )
        starts = torch.empty((batch_size,), dtype=torch.int32, device=logits.device)
        ends = torch.empty_like(starts)
        _init_topk_ranges_kernel[(1,)](
            starts,
            ends,
            batch_size,
            vocab_size,
            BLOCK_SIZE=triton.next_power_of_2(batch_size),
        )
        selected = torch.empty((batch_size, k), dtype=torch.int32, device=logits.device)
        indices = torch.empty(
            (batch_size, k + pack_sampled), dtype=torch.int64, device=logits.device
        )
        top_k_per_row_prefill(
            selection_logits,
            starts,
            ends,
            selected,
            batch_size,
            selection_logits.stride(0),
            selection_logits.stride(1),
            k,
        )
        _sort_selected_topk_indices_kernel[(batch_size,)](
            selection_logits,
            selection_logits.stride(0),
            selected,
            indices,
            sampled_token_ids if pack_sampled else indices,
            k,
            SORT_SIZE=triton.next_power_of_2(k),
            PACK_SAMPLED=pack_sampled,
            num_warps=8,
        )
        return indices
    if k <= vocab_size:
        batch_size = logits.shape[0]
        sort_chunk = 1024 if batch_size < _BATCH_CLUSTER_THRESHOLD else 512
        num_chunks = triton.cdiv(vocab_size, sort_chunk)
        padded_chunks = triton.next_power_of_2(num_chunks)
        sorted_stride = padded_chunks * sort_chunk
        partial_lse = (
            torch.empty(
                (batch_size, 3, num_chunks),
                dtype=torch.float32,
                device=logits.device,
            )
            if fused_scores is not None
            else None
        )
        source = torch.empty(
            (batch_size, sorted_stride), dtype=torch.uint64, device=logits.device
        )
        destination = torch.empty_like(source)
        indices = torch.empty(
            (batch_size, k + pack_sampled), dtype=torch.int64, device=logits.device
        )
        _sort_token_chunks_kernel[(batch_size, padded_chunks)](
            logits,
            sampled_token_ids if pack_sampled else logits,
            logits.stride(0),
            source,
            partial_lse if fused_scores is not None else source,
            sorted_stride,
            vocab_size,
            NUM_CHUNKS=num_chunks,
            SORT_CHUNK=sort_chunk,
            COMPUTE_LSE=fused_logprobs,
            COMPUTE_RANK=fused_scores is not None,
            num_warps=8,
        )
        run_size = sort_chunk
        if sorted_stride >= 8 * sort_chunk:
            _merge_four_sorted_runs_kernel[
                (batch_size, sorted_stride // (4 * sort_chunk))
            ](source, destination, sorted_stride, RUN_SIZE=sort_chunk, num_warps=8)
            source, destination = destination, source
            run_size = 4 * sort_chunk
        while run_size < sorted_stride:
            final = run_size * 2 >= sorted_stride
            _merge_sorted_token_runs_kernel[(batch_size, triton.cdiv(vocab_size, 128))](
                source,
                destination,
                indices,
                sampled_token_ids if pack_sampled else indices,
                logits,
                logits.stride(0),
                fused_scores if fused_scores is not None else indices,
                fused_ranks if fused_ranks is not None else indices,
                partial_lse if fused_scores is not None else source,
                sorted_stride,
                vocab_size,
                k,
                RUN_SIZE=run_size,
                SEARCH_STEPS=run_size.bit_length() + 1,
                BLOCK_SIZE=128,
                PACK_SAMPLED=pack_sampled,
                FINAL=final,
                FUSED_LOGITS=fused_scores is not None and not fused_logprobs,
                FUSED_LOGPROBS=fused_logprobs,
                NUM_CHUNKS=num_chunks,
                LSE_BLOCK=triton.next_power_of_2(num_chunks),
            )
            if not final:
                source, destination = destination, source
            run_size *= 2
        if sorted_stride == sort_chunk:
            _unpack_sorted_token_indices_kernel[(batch_size, 1)](
                source,
                sorted_stride,
                indices,
                sampled_token_ids if pack_sampled else indices,
                vocab_size,
                k,
                BLOCK_SIZE=sort_chunk,
                PACK_SAMPLED=pack_sampled,
            )
        return indices


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
    fused_full = (
        max_per_req_token_ids == 0
        and num_logprobs == vocab_size
        and vocab_size > (1024 if batch_size < _BATCH_CLUSTER_THRESHOLD else 512)
    )
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
