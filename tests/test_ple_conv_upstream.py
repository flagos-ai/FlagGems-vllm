# SPDX-License-Identifier: Apache-2.0
"""Accuracy checks for upstream-compatible ``ple_conv`` (decode/prefill/spec)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from flaggems_vllm import NULL_BLOCK_ID, ple_conv

pytestmark = pytest.mark.ple_fusion

_KERNEL_STATE_BLOCKS = 64


def _short_conv_dilated_decode_pytorch(
    x_d: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_d: torch.Tensor,
    has_initial_states_d: torch.Tensor | None,
    *,
    conv_state_len: int,
    dilation: int,
) -> torch.Tensor:
    state_indices = state_indices_tensor_d.to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    if has_initial_states_d is None:
        has_initial_state = valid_state
    else:
        has_initial_state = has_initial_states_d[: state_indices_tensor_d.numel()].to(
            device=conv_state.device, dtype=torch.bool
        )
        has_initial_state = has_initial_state & valid_state

    cached_state = conv_state.index_select(0, state_indices)
    state = cached_state[..., :conv_state_len].to(x_d.dtype)
    if conv_state_len > 0:
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
        dilation=dilation,
    ).squeeze(-1)
    output = F.silu(conv_output)
    output = output * valid_state.view(-1, 1).to(output.dtype)

    if conv_state_len > 0:
        next_state = history[..., -conv_state_len:]
        existing_base_state = cached_state[..., :conv_state_len]
        safe_next_state = torch.where(
            valid_state.view(-1, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        cached_state[..., :conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, cached_state)
    return output


def _short_conv_dilated_prefill_pytorch(
    x_p: torch.Tensor,
    query_start_loc_p: torch.Tensor,
    has_initial_states_p: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    *,
    conv_state_len: int,
    dilation: int,
    max_prefill_query_len: int,
) -> torch.Tensor:
    output = torch.empty_like(x_p)
    q_starts = query_start_loc_p.to(torch.int64)
    num_prefills = state_indices_tensor_p.numel()
    num_prefill_tokens = x_p.shape[0]
    if num_prefills == 0 or x_p.numel() == 0:
        return output
    lengths = q_starts[1:] - q_starts[:-1]
    max_len = max_prefill_query_len
    if max_len <= 0:
        return output

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
    if conv_state_len > 0:
        state = conv_state.index_select(0, state_indices)[..., :conv_state_len].to(
            x_p.dtype
        )
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
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

    token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
    valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
    valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
        num_prefills, 1
    )
    conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
    output.copy_(conv_output[req_indices, col_indices])

    if conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
            num_prefills, 1, 1
        )
        state_offsets = torch.arange(
            conv_state_len, device=history.device, dtype=torch.int64
        ).view(1, 1, conv_state_len)
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., :conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., :conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def _short_conv_dilated_spec_pytorch(
    x_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    spec_query_len: int,
    *,
    conv_state_len: int,
    dilation: int,
) -> torch.Tensor:
    num_reqs = spec_state_indices_tensor.numel()
    hidden_size = x_spec.size(-1)
    max_len = spec_query_len
    num_accepted_tokens = num_accepted_tokens[:num_reqs]
    q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
    total_real_tokens = q_starts[num_reqs]

    state_indices = spec_state_indices_tensor.to(
        device=conv_state.device, dtype=torch.int64
    )
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(
        valid_state, state_indices, torch.zeros_like(state_indices)
    )
    positions = torch.arange(x_spec.size(0), device=x_spec.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    valid_tokens = (positions < total_real_tokens) & (req_indices < num_reqs)
    clamped_req_indices = req_indices.clamp_max(max(num_reqs - 1, 0))
    col_indices = (positions - q_starts[clamped_req_indices]).clamp_(0, max_len - 1)
    pack_req_indices = torch.where(
        valid_tokens,
        clamped_req_indices,
        torch.full_like(req_indices, num_reqs),
    )
    pack_col_indices = torch.where(
        valid_tokens, col_indices, torch.zeros_like(col_indices)
    )

    packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
    packed[pack_req_indices, pack_col_indices] = x_spec
    packed = packed.transpose(1, 2).contiguous()

    if conv_state_len > 0:
        cached_state = conv_state.index_select(0, state_indices)
        rollback_offsets = num_accepted_tokens.to(
            device=conv_state.device, dtype=torch.int64
        ).sub(1)
        rollback_offsets = torch.where(
            valid_state,
            rollback_offsets.clamp_(0, max_len - 1),
            torch.zeros_like(rollback_offsets),
        )
        state_offsets = torch.arange(
            conv_state_len, device=conv_state.device, dtype=torch.int64
        ).view(1, 1, conv_state_len)
        rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
        state = cached_state.gather(2, rollback_indices.expand(-1, hidden_size, -1)).to(
            x_spec.dtype
        )
        state = torch.where(
            valid_state.view(num_reqs, 1, 1),
            state,
            torch.zeros_like(state),
        )
        dummy_state = state.new_zeros((1, hidden_size, conv_state_len))
        state_full = torch.cat((state, dummy_state), dim=0)
        history = torch.cat((state_full, packed), dim=-1)
    else:
        history = packed
        cached_state = None

    conv_output = F.conv1d(
        history,
        conv_weights.unsqueeze(1).contiguous(),
        groups=history.size(1),
        dilation=dilation,
    )
    conv_output = F.silu(conv_output).transpose(1, 2).contiguous()[:num_reqs]
    token_positions = torch.arange(max_len, device=x_spec.device, dtype=torch.int64)
    lengths = q_starts[1:] - q_starts[:-1]
    valid_packed = token_positions.view(1, max_len) < lengths.view(num_reqs, 1)
    # Spec null slots still emit activations from the current chunk.
    conv_output.masked_fill_(~valid_packed.unsqueeze(-1), 0)
    output = x_spec.new_zeros(x_spec.shape)
    output[valid_tokens] = conv_output[clamped_req_indices[valid_tokens], col_indices[valid_tokens]]

    if conv_state_len > 0 and cached_state is not None:
        # Keep candidate tokens; roll by one like causal_conv1d_update.
        state_capacity = conv_state_len + max_len - 1
        next_state = history[:num_reqs, :, 1 : 1 + state_capacity]
        existing_state = cached_state
        write_counts = (conv_state_len + lengths - 1).clamp_min(0)
        for req in range(num_reqs):
            if not bool(valid_state[req]):
                continue
            count = int(write_counts[req])
            existing_state[req, :, :count] = next_state[req, :, :count].to(
                conv_state.dtype
            )
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def _make_conv_case(
    device: torch.device,
    seed: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    state_layout: str,
    spec_query_len: int = 1,
):
    rng = torch.Generator(device=device).manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    state_shape = (
        (_KERNEL_STATE_BLOCKS, state_width, channels)
        if state_layout == "SD"
        else (_KERNEL_STATE_BLOCKS, channels, state_width)
    )
    state_kernel = torch.randn(
        state_shape, device=device, dtype=torch.bfloat16, generator=rng
    )
    conv_state = (
        state_kernel.transpose(-1, -2) if state_layout == "SD" else state_kernel
    )
    state_reference = conv_state.clone()
    weights = torch.randn(
        channels, kernel_size, device=device, dtype=torch.bfloat16, generator=rng
    )
    return rng, state_reference, conv_state, weights


@dataclass(frozen=True)
class _ConvBatchCase:
    spec_query_lens: tuple[int, ...] = ()
    num_accepted: tuple[int, ...] = ()
    num_decodes: int = 0
    prefill_query_lens: tuple[int, ...] = ()
    channels: int = 512
    kernel_size: int = 4
    dilation: int = 3
    spec_query_len: int = 1
    graph_padding: int = 0
    state_index_stride: int = 1
    include_null_state: bool = True


def _make_conv_metadata(case: _ConvBatchCase, device: torch.device):
    request_groups: dict[str, list[int]] = {
        "spec": [],
        "decode": [],
        "prefill": [],
    }
    token_offset = 0
    max_num_reqs = max(
        len(case.spec_query_lens),
        case.num_decodes,
        len(case.prefill_query_lens),
    )
    request_kinds = []
    for req_idx in range(max_num_reqs):
        if req_idx < len(case.spec_query_lens):
            request_kinds.append(("spec", case.spec_query_lens[req_idx]))
        if req_idx < case.num_decodes:
            request_kinds.append(("decode", 1))
        if req_idx < len(case.prefill_query_lens):
            request_kinds.append(("prefill", case.prefill_query_lens[req_idx]))
    for kind, query_len in request_kinds:
        request_groups[kind].extend(range(token_offset, token_offset + query_len))
        token_offset += query_len

    num_spec_reqs = len(case.spec_query_lens)
    num_prefills = len(case.prefill_query_lens)
    has_spec = num_spec_reqs > 0
    has_non_spec = case.num_decodes > 0 or num_prefills > 0
    mixed_batch = has_spec and has_non_spec

    spec_state_indices = torch.arange(
        1, num_spec_reqs + 1, dtype=torch.int32, device=device
    )
    if num_spec_reqs >= 3:
        spec_state_indices[1] = NULL_BLOCK_ID
    non_spec_state_indices = torch.arange(
        num_spec_reqs + 1,
        num_spec_reqs + case.num_decodes + num_prefills + 1,
        dtype=torch.int32,
        device=device,
    )
    if case.state_index_stride > 1:
        strided_indices = torch.full(
            (non_spec_state_indices.numel(), case.state_index_stride),
            NULL_BLOCK_ID,
            dtype=torch.int32,
            device=device,
        )
        strided_indices[:, 0] = non_spec_state_indices
        non_spec_state_indices = strided_indices[:, 0]
    if case.num_decodes > 1:
        non_spec_state_indices[case.num_decodes - 1] = NULL_BLOCK_ID
    if num_prefills > 1 and case.include_null_state:
        non_spec_state_indices[case.num_decodes + 1] = NULL_BLOCK_ID

    spec_query_start_loc = torch.tensor(
        [0, *accumulate(case.spec_query_lens)], dtype=torch.int32, device=device
    )
    non_spec_query_lens = (1,) * case.num_decodes + case.prefill_query_lens
    non_spec_query_start_loc = torch.tensor(
        [0, *accumulate(non_spec_query_lens)], dtype=torch.int32, device=device
    )
    spec_token_indices = torch.tensor(
        request_groups["spec"], dtype=torch.int64, device=device
    )
    non_spec_token_indices = torch.tensor(
        request_groups["decode"] + request_groups["prefill"],
        dtype=torch.int64,
        device=device,
    )

    metadata = SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=case.num_decodes,
        num_decode_tokens=case.num_decodes,
        num_prefill_tokens=sum(case.prefill_query_lens),
        spec_sequence_masks=(
            torch.tensor([kind == "spec" for kind, _ in request_kinds], device=device)
            if has_spec
            else None
        ),
        spec_token_indx=spec_token_indices if mixed_batch else None,
        non_spec_token_indx=non_spec_token_indices if mixed_batch else None,
        num_actual_tokens=token_offset + case.graph_padding,
        spec_state_indices_tensor=spec_state_indices if has_spec else None,
        spec_query_start_loc=spec_query_start_loc if has_spec else None,
        num_accepted_tokens=(
            torch.tensor(case.num_accepted, dtype=torch.int32, device=device)
            if has_spec
            else None
        ),
        spec_query_len=case.spec_query_len,
        state_indices_tensor=non_spec_state_indices if has_non_spec else None,
        has_initial_states_d=(
            torch.arange(case.num_decodes, device=device) % 2 == 0
            if case.num_decodes
            else None
        ),
        non_spec_query_start_loc=(non_spec_query_start_loc if has_non_spec else None),
        query_start_loc_p=(
            non_spec_query_start_loc[case.num_decodes :] - case.num_decodes
            if num_prefills
            else None
        ),
        has_initial_states_p=(
            torch.arange(num_prefills, device=device) % 2 == 0 if num_prefills else None
        ),
        max_prefill_query_len=max(case.prefill_query_lens, default=0),
        num_spec_decodes=num_spec_reqs,
    )
    return metadata, token_offset


def _dispatch_ple_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    outer_residual: torch.Tensor | None,
    metadata,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    dilation: int,
) -> None:
    """Mirror vLLM ``_short_conv_dilated_dispatch`` against FlagGems ``ple_conv``."""
    num_prefills = metadata.num_prefills
    num_decodes = metadata.num_decodes
    num_decode_tokens = metadata.num_decode_tokens
    num_prefill_tokens = metadata.num_prefill_tokens
    has_prefill = num_prefills > 0
    has_decode = num_decodes > 0
    has_spec = metadata.spec_sequence_masks is not None
    has_non_spec = has_prefill or has_decode
    inputs = inputs[: metadata.num_actual_tokens]
    residual = residual[: metadata.num_actual_tokens]
    if outer_residual is not None:
        outer_residual = outer_residual[: metadata.num_actual_tokens]

    spec_token_indices = None
    non_spec_token_indices = None
    if has_spec and has_non_spec:
        spec_token_indices = metadata.spec_token_indx
        non_spec_token_indices = metadata.non_spec_token_indx

    if has_spec:
        ple_conv(
            inputs=inputs,
            residual=residual,
            conv_state=conv_state,
            conv_weights=conv_weights,
            state_indices=metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ],
            outer_residual=outer_residual,
            mode="spec",
            dilation=dilation,
            query_start_loc=metadata.spec_query_start_loc,
            num_accepted_tokens=metadata.num_accepted_tokens,
            spec_query_len=metadata.spec_query_len,
            token_indices=spec_token_indices,
        )

    if not has_non_spec:
        return

    state_indices = metadata.state_indices_tensor
    if has_prefill:
        state_indices_d, state_indices_p = torch.split(
            state_indices, [num_decodes, num_prefills], dim=0
        )
        if non_spec_token_indices is None:
            inputs_d, inputs_p = torch.split(
                inputs, [num_decode_tokens, num_prefill_tokens], dim=0
            )
            residual_d, residual_p = torch.split(
                residual,
                [num_decode_tokens, num_prefill_tokens],
                dim=0,
            )
            outer_residual_d = outer_residual_p = None
            if outer_residual is not None:
                outer_residual_d, outer_residual_p = torch.split(
                    outer_residual,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
            token_indices_d = token_indices_p = None
        else:
            inputs_d = inputs_p = inputs
            residual_d = residual_p = residual
            outer_residual_d = outer_residual_p = outer_residual
            token_indices_d, token_indices_p = torch.split(
                non_spec_token_indices,
                [num_decode_tokens, num_prefill_tokens],
                dim=0,
            )

        if has_decode:
            ple_conv(
                inputs=inputs_d,
                residual=residual_d,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices=state_indices_d,
                outer_residual=outer_residual_d,
                mode="decode",
                dilation=dilation,
                has_initial_states=metadata.has_initial_states_d,
                token_indices=token_indices_d,
            )

        ple_conv(
            inputs=inputs_p,
            residual=residual_p,
            conv_state=conv_state,
            conv_weights=conv_weights,
            state_indices=state_indices_p,
            outer_residual=outer_residual_p,
            mode="prefill",
            dilation=dilation,
            query_start_loc=metadata.query_start_loc_p,
            has_initial_states=metadata.has_initial_states_p,
            token_indices=token_indices_p,
        )
    else:
        num_decode_rows = (
            non_spec_token_indices.numel()
            if non_spec_token_indices is not None
            else inputs.size(0)
        )
        ple_conv(
            inputs=inputs,
            residual=residual,
            conv_state=conv_state,
            conv_weights=conv_weights,
            state_indices=state_indices[:num_decode_rows],
            outer_residual=outer_residual,
            mode="decode",
            dilation=dilation,
            has_initial_states=metadata.has_initial_states_d,
            token_indices=non_spec_token_indices,
        )


def _dispatch_pytorch(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    metadata,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    *,
    conv_state_len: int,
    dilation: int,
) -> None:
    num_prefills = metadata.num_prefills
    num_decodes = metadata.num_decodes
    num_decode_tokens = metadata.num_decode_tokens
    num_prefill_tokens = metadata.num_prefill_tokens
    has_prefill = num_prefills > 0
    has_decode = num_decodes > 0
    has_spec = metadata.spec_sequence_masks is not None
    x = inputs[: metadata.num_actual_tokens]
    residual = residual[: metadata.num_actual_tokens]

    if has_spec:
        if has_prefill or has_decode:
            spec_token_indices = metadata.spec_token_indx.long()
            non_spec_token_indices = metadata.non_spec_token_indx.long()
            x_spec = x.index_select(0, spec_token_indices)
            x_non_spec = x.index_select(0, non_spec_token_indices)
            residual_spec = residual.index_select(0, spec_token_indices)
            residual_non_spec = residual.index_select(0, non_spec_token_indices)
        else:
            spec_token_indices = non_spec_token_indices = None
            x_spec = x
            x_non_spec = None
            residual_spec = residual
            residual_non_spec = None
    else:
        spec_token_indices = non_spec_token_indices = None
        x_spec = None
        x_non_spec = x
        residual_spec = None
        residual_non_spec = residual

    if has_spec:
        conv_output = _short_conv_dilated_spec_pytorch(
            x_spec=x_spec,
            conv_state=conv_state,
            conv_weights=conv_weights,
            spec_state_indices_tensor=metadata.spec_state_indices_tensor[
                : metadata.num_spec_decodes
            ],
            spec_query_start_loc=metadata.spec_query_start_loc,
            num_accepted_tokens=metadata.num_accepted_tokens,
            spec_query_len=metadata.spec_query_len,
            conv_state_len=conv_state_len,
            dilation=dilation,
        )
        residual_spec.add_(conv_output)

    state_indices = metadata.state_indices_tensor
    if x_non_spec is not None:
        if has_prefill:
            state_indices_d, state_indices_p = torch.split(
                state_indices, [num_decodes, num_prefills], dim=0
            )
            x_d, x_p = torch.split(
                x_non_spec, [num_decode_tokens, num_prefill_tokens], dim=0
            )
            residual_d, residual_p = torch.split(
                residual_non_spec,
                [num_decode_tokens, num_prefill_tokens],
                dim=0,
            )
            if has_decode:
                residual_d.add_(
                    _short_conv_dilated_decode_pytorch(
                        x_d=x_d,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_d=state_indices_d,
                        has_initial_states_d=metadata.has_initial_states_d,
                        conv_state_len=conv_state_len,
                        dilation=dilation,
                    )
                )
            residual_p.add_(
                _short_conv_dilated_prefill_pytorch(
                    x_p=x_p,
                    query_start_loc_p=metadata.query_start_loc_p,
                    has_initial_states_p=metadata.has_initial_states_p,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_p=state_indices_p,
                    conv_state_len=conv_state_len,
                    dilation=dilation,
                    max_prefill_query_len=metadata.max_prefill_query_len,
                )
            )
        else:
            residual_non_spec.add_(
                _short_conv_dilated_decode_pytorch(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                    conv_state_len=conv_state_len,
                    dilation=dilation,
                )
            )

    if has_spec and residual_non_spec is not None:
        residual.index_copy_(0, spec_token_indices, residual_spec)
        residual.index_copy_(0, non_spec_token_indices, residual_non_spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("state_layout", ["SD", "DS"])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_ConvBatchCase(num_decodes=1, channels=512), id="decode-single"),
        pytest.param(
            _ConvBatchCase(
                num_decodes=33,
                channels=2048,
                kernel_size=5,
                dilation=1,
            ),
            id="decode-padded",
        ),
        pytest.param(
            _ConvBatchCase(prefill_query_lens=(37, 0, 5, 128)),
            id="prefill",
        ),
        pytest.param(
            _ConvBatchCase(
                prefill_query_lens=(16, 64),
                channels=1024,
                state_index_stride=4,
                include_null_state=False,
            ),
            id="prefill-strided-state-indices",
        ),
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(1, 4, 4, 0),
                num_accepted=(1, 2, 4, 1),
                spec_query_len=4,
                graph_padding=4,
            ),
            id="spec-graph-padded",
        ),
        pytest.param(
            _ConvBatchCase(
                spec_query_lens=(3, 4),
                num_accepted=(2, 4),
                num_decodes=1,
                prefill_query_lens=(5,),
                spec_query_len=4,
            ),
            id="mixed",
        ),
    ],
)
@pytest.mark.parametrize("fuse_residual", [False, True])
def test_ple_conv_matches_eager(case, state_layout, fuse_residual):
    device = torch.device("cuda")
    metadata, num_real_tokens = _make_conv_metadata(case, device)
    conv_state_len = (case.kernel_size - 1) * case.dilation
    dilation = case.dilation

    rng, state_reference, conv_state, weights = _make_conv_case(
        device,
        seed=num_real_tokens + case.channels,
        channels=case.channels,
        kernel_size=case.kernel_size,
        dilation=case.dilation,
        state_layout=state_layout,
        spec_query_len=case.spec_query_len,
    )
    inputs = torch.randn(
        metadata.num_actual_tokens,
        case.channels,
        device=device,
        dtype=torch.bfloat16,
        generator=rng,
    )
    residual = torch.randn(
        inputs.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=rng,
    )
    outer_residual = torch.randn(
        inputs.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=rng,
    )
    null_state = conv_state[NULL_BLOCK_ID].clone()
    residual_kernel = residual.clone()
    residual_reference = (
        residual.clone() if fuse_residual else torch.zeros_like(residual)
    )

    _dispatch_ple_conv(
        inputs=inputs,
        residual=residual_kernel,
        outer_residual=outer_residual if fuse_residual else None,
        metadata=metadata,
        conv_state=conv_state,
        conv_weights=weights,
        dilation=dilation,
    )
    _dispatch_pytorch(
        inputs=inputs,
        residual=residual_reference,
        metadata=metadata,
        conv_state=state_reference,
        conv_weights=weights,
        conv_state_len=conv_state_len,
        dilation=dilation,
    )
    if fuse_residual:
        residual_reference = outer_residual + residual_reference

    assert torch.equal(
        residual_kernel[:num_real_tokens], residual_reference[:num_real_tokens]
    )
    assert torch.equal(conv_state, state_reference)
    assert torch.equal(conv_state[NULL_BLOCK_ID], null_state)
    if case.graph_padding:
        assert torch.equal(
            residual_kernel[num_real_tokens:],
            residual_reference[num_real_tokens:],
        )
