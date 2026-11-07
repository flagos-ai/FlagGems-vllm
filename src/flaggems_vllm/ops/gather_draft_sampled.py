# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Gather logits-aligned input tokens and positions for draft verification."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import has_triton_tle_attrs, libentry, libtuner

_HAS_LAYOUT = runtime.device.vendor_name == "nvidia" and has_triton_tle_attrs(
    ("gpu.set_layout", "gpu.BlockEncoding")
)
if _HAS_LAYOUT:
    import triton.experimental.tle.language as tle


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("gather_draft_sampled"),
    key=["num_logits"],
    warmup=5,
    rep=30,
    use_cuda_graph=runtime.device.name == "cuda",
)
@triton.jit
def _gather_draft_sampled_kernel(
    draft_sampled_ptr,
    pos_ptr,
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    prefill_len_ptr,
    num_logits,
    USE_LAYOUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WARPS: tl.constexpr,
    ELEMENTS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_LAYOUT:
        # Adjacent per-thread elements permit vectorized metadata loads/stores.
        rows = tle.gpu.set_layout(
            rows, tle.gpu.BlockEncoding([ELEMENTS], [32], [WARPS], [0])
        )
    mask = rows < num_logits
    token_idx = tl.load(logits_indices_ptr + rows, mask=mask, other=0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + rows, mask=mask, other=0)
    local_pos = tl.load(expanded_local_pos_ptr + rows, mask=mask, other=0)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask, other=0)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask, other=0)
    pos = tl.load(positions_ptr + token_idx, mask=mask, other=0)
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    tl.store(
        draft_sampled_ptr + rows, tl.where(is_placeholder, -1, token_id), mask=mask
    )
    tl.store(pos_ptr + rows, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather each logits row's token/position and mark prefill draft placeholders.

    Inputs are one-dimensional unit-stride int32/int64 arrays on the active
    accelerator device, including device-addressable UVA state when available.
    Token and request indices must be in bounds. Output dtypes follow input_ids
    and positions.
    """
    tensors = (
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
    )
    if input_ids.device.type == "cpu" or input_ids.device.type != runtime.device.name:
        raise NotImplementedError(
            "gather_draft_sampled requires the active accelerator"
        )
    if any(t.device != input_ids.device for t in tensors):
        raise ValueError("all gather_draft_sampled inputs must be on the same device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise TypeError("gather_draft_sampled inputs must have int32 or int64 dtype")
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("gather_draft_sampled inputs must be one-dimensional")
    if any(t.stride(0) != 1 for t in tensors):
        raise NotImplementedError("gather_draft_sampled requires unit-stride inputs")
    num_logits = logits_indices.shape[0]
    if positions.shape != input_ids.shape:
        raise ValueError("positions and input_ids must have the same shape")
    if (
        expanded_idx_mapping.numel() != num_logits
        or expanded_local_pos.numel() != num_logits
    ):
        raise ValueError("expanded mappings must have one element per logits row")
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    if num_logits:
        _gather_draft_sampled_kernel[
            lambda meta: (triton.cdiv(num_logits, meta["BLOCK_SIZE"]),)
        ](
            draft_sampled,
            pos,
            input_ids,
            positions,
            logits_indices,
            expanded_idx_mapping,
            expanded_local_pos,
            prefill_len,
            num_logits,
            USE_LAYOUT=_HAS_LAYOUT,
        )
    return draft_sampled, pos
