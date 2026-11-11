# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Utilities for vLLM multi-module MTP speculative decoding."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

# vLLM v1/attention/backends/utils.py at 947dd62c.
PAD_SLOT_ID = -1


def _prune_request_tiles(configs, named_args, **kwargs):
    n = named_args.get("NUM_REQS", kwargs.get("NUM_REQS"))
    # Preserve roughly one CTA per SM even for a single KV-cache group.
    # Over-grouping short batches reduces cold-memory parallelism.
    sm_count = 128
    if runtime.device.vendor_name == "nvidia":
        sm_count = runtime.torch_device_fn.get_device_properties(
            runtime.torch_device_fn.current_device()
        ).multi_processor_count
    max_rows = max(1, triton.cdiv(n, sm_count))
    return [c for c in configs if c.kwargs["BLOCK_ROWS"] <= max_rows]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("pad_trailing_draft_slots"),
    key=["NUM_REQS"],
    prune_configs_by={"early_config_prune": _prune_request_tiles},
    use_cuda_graph=runtime.device.name == "cuda",
)
@triton.jit
def _pad_trailing_draft_slots_kernel(
    slot_mappings_ptr,
    slot_mappings_stride0,
    query_start_loc_ptr,
    last_token_indices_ptr,
    PAD_ID,
    BLOCK_SIZE: tl.constexpr,
    NUM_REQS,
    BLOCK_ROWS: tl.constexpr,
):
    group_idx = tl.program_id(0)
    base = slot_mappings_ptr + group_idx * slot_mappings_stride0
    if BLOCK_ROWS == 1:
        req_idx = tl.program_id(1)
        start = tl.load(last_token_indices_ptr + req_idx) + 1
        end = tl.load(query_start_loc_ptr + req_idx + 1)
        if BLOCK_SIZE < 256 and end - start > 256:
            for i in range(start, end, 256):
                offs = i + tl.arange(0, 256)
                tl.store(base + offs, PAD_ID, offs < end)
        else:
            for i in range(start, end, BLOCK_SIZE):
                offs = i + tl.arange(0, BLOCK_SIZE)
                tl.store(base + offs, PAD_ID, offs < end)
    else:
        req = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid = req < NUM_REQS
        start = tl.load(last_token_indices_ptr + req, valid, other=-1) + 1
        end = tl.load(query_start_loc_ptr + req + 1, valid, other=0)
        limit = tl.max(tl.maximum(end - start, 0), axis=0)
        if BLOCK_SIZE < 256 and limit > 256:
            for i in range(0, limit, 256):
                offs = start[:, None] + i + tl.arange(0, 256)[None, :]
                tl.store(base + offs, PAD_ID, valid[:, None] & (offs < end[:, None]))
        else:
            for i in range(0, limit, BLOCK_SIZE):
                offs = start[:, None] + i + tl.arange(0, BLOCK_SIZE)[None, :]
                tl.store(base + offs, PAD_ID, valid[:, None] & (offs < end[:, None]))


def pad_trailing_draft_slots(
    slot_mappings: torch.Tensor,
    query_start_loc: torch.Tensor,
    last_token_indices: torch.Tensor,
    num_reqs: int,
) -> None:
    """Pad trailing stale slots in place, preserving all other mapping entries."""
    if slot_mappings.device.type == "cpu":
        raise NotImplementedError("pad_trailing_draft_slots requires an accelerator")
    if any(
        t.stride(-1) != 1 for t in (slot_mappings, query_start_loc, last_token_indices)
    ):
        raise NotImplementedError(
            "pad_trailing_draft_slots requires unit inner strides"
        )
    num_groups = slot_mappings.shape[0]
    if num_reqs == 0 or num_groups == 0:
        return
    _pad_trailing_draft_slots_kernel[
        lambda meta: (num_groups, triton.cdiv(num_reqs, meta["BLOCK_ROWS"]))
    ](
        slot_mappings,
        slot_mappings.stride(0),
        query_start_loc,
        last_token_indices,
        PAD_SLOT_ID,
        NUM_REQS=num_reqs,
    )
