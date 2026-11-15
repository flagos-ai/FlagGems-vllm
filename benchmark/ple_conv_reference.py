# SPDX-License-Identifier: Apache-2.0
# Frozen e5141d9 native batched pipeline, used only for performance comparison.
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from flaggems_vllm import ple_state_gather, ple_state_scatter_

NULL_BLOCK_ID = -1


def _short_conv_dilated_decode_batched(
    self,
    x_d: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_d: torch.Tensor,
    has_initial_states_d: torch.Tensor | None,
) -> torch.Tensor:
    state_indices = state_indices_tensor_d.to(
        device=conv_state.device, dtype=torch.int64
    )
    # TODO: need double-check
    # FULL cudagraph padded decode rows use NULL_BLOCK_ID. Remap them to
    # slot 0 for a safe gather, then zero output and skip write-back.
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    if has_initial_states_d is None:
        has_initial_state = valid_state
    else:
        if has_initial_states_d.numel() < state_indices_tensor_d.numel():
            raise ValueError(
                "has_initial_states_d size mismatch: "
                f"got {has_initial_states_d.numel()}, "
                f"need >= {state_indices_tensor_d.numel()}."
            )
        has_initial_state = has_initial_states_d[: state_indices_tensor_d.numel()].to(
            device=conv_state.device, dtype=torch.bool
        )
        has_initial_state = has_initial_state & valid_state

    cached_state = ple_state_gather(conv_state, state_indices, indices_are_safe=True)
    state = cached_state[..., : self.conv_state_len].to(x_d.dtype)
    if self.conv_state_len > 0:
        initial_state = torch.where(
            has_initial_state.view(-1, 1, 1),
            state,
            torch.zeros_like(state),
        )
        history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
    else:
        history = x_d.unsqueeze(-1)

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=self.short_conv_dilation,
    ).squeeze(-1)
    output = F.silu(conv_output)
    output = output * valid_state.view(-1, 1).to(output.dtype)

    if self.conv_state_len > 0:
        next_state = history[..., -self.conv_state_len :]
        # Padded rows are remapped to the reserved null slot. Preserve its
        # existing value while writing the new states for valid rows.
        existing_base_state = cached_state[..., : self.conv_state_len]
        safe_next_state = torch.where(
            valid_state.view(-1, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        cached_state[..., : self.conv_state_len] = safe_next_state
        ple_state_scatter_(
            conv_state,
            state_indices,
            cached_state,
            write_mask=valid_state,
            indices_are_safe=True,
        )

    return output


def _short_conv_dilated_prefill_batched(
    self,
    x_p: torch.Tensor,
    metadata: SimpleNamespace,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    num_prefills: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor | None:
    # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
    # requests and equals ``query_start_loc`` when spec-decode is inactive.
    non_spec_query_start_loc = metadata.non_spec_query_start_loc
    if non_spec_query_start_loc is None:
        raise ValueError("query_start_loc is required for prefill short-conv")
    query_start_loc_p = non_spec_query_start_loc[-num_prefills - 1 :]
    if num_decode_tokens:
        query_start_loc_p = query_start_loc_p - num_decode_tokens
    # The metadata builder guarantees that the prefill query offsets start
    # at 0 and end at num_prefill_tokens. Avoid reading those values here,
    # since doing so would force a device-to-host synchronization.
    has_initial_states_p = metadata.has_initial_states_p
    if has_initial_states_p is None:
        raise ValueError("has_initial_states_p is required for prefill short-conv")

    q_starts = query_start_loc_p.to(torch.int64)
    if state_indices_tensor_p.numel() < num_prefills:
        raise ValueError(
            "state_indices_tensor_p size mismatch: "
            f"got {state_indices_tensor_p.numel()}, "
            f"need >= {num_prefills}."
        )
    if has_initial_states_p.numel() < num_prefills:
        raise ValueError(
            "has_initial_states_p size mismatch: "
            f"got {has_initial_states_p.numel()}, "
            f"need >= {num_prefills}."
        )
    if num_prefills == 0 or x_p.numel() == 0:
        return None if output is not None else torch.empty_like(x_p)
    lengths = q_starts[1:] - q_starts[:-1]
    # Use the CPU-computed packing width from the metadata builder instead
    # of synchronizing on lengths.max().
    max_len = metadata.max_prefill_query_len
    if max_len <= 0:
        return None if output is not None else torch.empty_like(x_p)

    fallback_output = output if output is not None else torch.empty_like(x_p)
    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    has_initial = has_initial_states_p[:num_prefills].to(
        device=conv_state.device, dtype=torch.bool
    )
    if self.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros(
                (num_prefills, hidden_size, self.conv_state_len),
                dtype=x_p.dtype,
            )
        else:
            state = ple_state_gather(conv_state, state_indices, indices_are_safe=True)[
                ..., : self.conv_state_len
            ].to(x_p.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(
            use_initial_mask,
            state,
            torch.zeros_like(state),
        )
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=self.short_conv_dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
        num_prefills, 1
    )
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    fallback_output.copy_(conv_output[req_indices, col_indices])

    if self.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
            num_prefills, 1, 1
        )
        state_offsets = torch.arange(
            self.conv_state_len, device=history.device, dtype=torch.int64
        ).view(1, 1, self.conv_state_len)
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        # Write back without a host synchronization. Valid, non-empty rows
        # receive their new state; padding and zero-length rows keep the
        # current cache value.
        existing_state = ple_state_gather(
            conv_state, state_indices, indices_are_safe=True
        )
        existing_base_state = existing_state[..., : self.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., : self.conv_state_len] = safe_next_state
        ple_state_scatter_(
            conv_state,
            state_indices,
            existing_state,
            write_mask=update_mask,
            indices_are_safe=True,
        )
    return None if output is not None else fallback_output


def _short_conv_dilated_spec_batched(
    self,
    x_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    spec_query_len: int,
) -> torch.Tensor:
    """Dilated short-conv for speculative-decode (MTP) requests.

    Each spec request feeds multiple (draft + 1) query tokens. The conv
    outputs are computed causally after rolling back the previous draft
    state by ``num_accepted_tokens - 1``. The current candidate inputs stay
    in the extended cache for the next forward, matching
    ``causal_conv1d_update``.

    ``spec_query_len`` (== num_speculative_tokens + 1) is the maximum query
    length and is a Python int, so no host synchronization is needed; this
    keeps the path safe for full CUDA-graph capture/replay where the buffers
    are padded at the request level.
    """
    num_reqs = spec_state_indices_tensor.numel()
    hidden_size = x_spec.size(-1)
    # Use a fixed packing width instead of synchronizing on lengths.max().
    max_len = spec_query_len
    # Full CUDA graphs can pad these buffers. Only the first num_reqs
    # accepted-token counts belong to actual speculative requests.
    num_accepted_tokens = num_accepted_tokens[:num_reqs]
    q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
    # Keep the number of real speculative tokens on the device.
    total_real_tokens = q_starts[num_reqs]

    state_indices = spec_state_indices_tensor.to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    positions = torch.arange(x_spec.size(0), device=x_spec.device, dtype=torch.int64)
    # Route graph-padded token rows to the discarded dummy request so that
    # they cannot overwrite real packed data.
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    valid_requests = (req_indices >= 0) & (req_indices < num_reqs)
    safe_req_indices = torch.where(
        valid_requests, req_indices, torch.zeros_like(req_indices)
    )
    col_indices = positions - q_starts[safe_req_indices]
    valid_columns = (col_indices >= 0) & (col_indices < max_len)
    valid_tokens = (positions < total_real_tokens) & valid_requests & valid_columns
    safe_col_indices = torch.where(
        valid_tokens, col_indices, torch.zeros_like(col_indices)
    )
    pack_req_indices = torch.where(
        valid_tokens,
        safe_req_indices,
        torch.full_like(req_indices, num_reqs),
    )
    pack_col_indices = torch.where(
        valid_tokens, safe_col_indices, torch.zeros_like(col_indices)
    )

    # The last request row is the dummy sink for graph padding.
    packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
    packed[pack_req_indices, pack_col_indices] = x_spec
    packed = packed.transpose(1, 2).contiguous()

    if self.conv_state_len > 0:
        cached_state = ple_state_gather(
            conv_state, state_indices, indices_are_safe=True
        )
        rollback_offsets = num_accepted_tokens.to(
            device=conv_state.device, dtype=torch.int64
        ).sub(1)
        valid_rollback = (
            valid_state & (rollback_offsets >= 0) & (rollback_offsets < max_len)
        )
        safe_rollback_offsets = torch.where(
            valid_rollback,
            rollback_offsets,
            torch.zeros_like(rollback_offsets),
        )
        rollback_offsets = torch.where(
            valid_rollback,
            safe_rollback_offsets,
            torch.zeros_like(safe_rollback_offsets),
        )
        state_offsets = torch.arange(
            self.conv_state_len, device=conv_state.device, dtype=torch.int64
        ).view(1, 1, self.conv_state_len)
        rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
        state = cached_state.gather(2, rollback_indices.expand(-1, hidden_size, -1)).to(
            x_spec.dtype
        )
        state = torch.where(
            valid_rollback.view(num_reqs, 1, 1),
            state,
            torch.zeros_like(state),
        )
        # Append a zeroed dummy-row state to match the [num_reqs + 1] pack.
        dummy_state = state.new_zeros((1, hidden_size, self.conv_state_len))
        state_full = torch.cat((state, dummy_state), dim=0)
        history = torch.cat((state_full, packed), dim=-1)
    else:
        history = packed

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=self.short_conv_dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    output = conv_output[pack_req_indices, pack_col_indices]
    output = output * valid_tokens.view(-1, 1).to(output.dtype)

    # Keep all current candidate inputs in the extended state. On the next
    # target forward, ``num_accepted_tokens - 1`` selects the rollback
    # window before processing the newly scheduled tokens.
    if self.conv_state_len > 0:
        state_capacity = self.conv_state_len + max_len - 1
        if conv_state.size(-1) < state_capacity:
            raise RuntimeError(
                "PLE short-conv cache cannot retain speculative tokens: "
                f"got {conv_state.size(-1)}, need {state_capacity}."
            )
        candidate_state = history[:num_reqs, :, 1 : state_capacity + 1]
        query_lengths = q_starts[1:] - q_starts[:-1]
        state_positions = torch.arange(
            state_capacity, device=history.device, dtype=torch.int64
        ).view(1, 1, state_capacity)
        update_lengths = (self.conv_state_len + query_lengths - 1).view(num_reqs, 1, 1)
        update_mask = valid_state.view(num_reqs, 1, 1) & (
            state_positions < update_lengths
        )
        existing_state = cached_state[..., :state_capacity]
        next_state = torch.where(
            update_mask,
            candidate_state.to(conv_state.dtype),
            existing_state,
        )
        cached_state[..., :state_capacity] = next_state
        ple_state_scatter_(
            conv_state,
            state_indices,
            cached_state,
            write_mask=valid_state & (query_lengths > 0),
            indices_are_safe=True,
        )

    return output


def ple_conv_reference(
    x,
    output,
    state,
    weight,
    indices,
    *,
    dilation,
    query_start_loc=None,
    has_initial_state=None,
    accepted_tokens=None,
    max_query_len=1,
    null_block_id=-1,
):
    self = SimpleNamespace(
        conv_state_len=(weight.shape[1] - 1) * dilation, short_conv_dilation=dilation
    )
    if accepted_tokens is not None:
        result = _short_conv_dilated_spec_batched(
            self,
            x,
            state,
            weight,
            indices,
            query_start_loc,
            accepted_tokens,
            max_query_len,
        )
    elif query_start_loc is None:
        result = _short_conv_dilated_decode_batched(
            self, x, state, weight, indices, has_initial_state
        )
    else:
        metadata = SimpleNamespace(
            non_spec_query_start_loc=query_start_loc,
            has_initial_states_p=torch.ones_like(indices, dtype=torch.bool),
            max_prefill_query_len=max_query_len,
        )
        result = _short_conv_dilated_prefill_batched(
            self, x, metadata, state, weight, indices, indices.numel(), 0, x.shape[0]
        )
    output.copy_(result)
