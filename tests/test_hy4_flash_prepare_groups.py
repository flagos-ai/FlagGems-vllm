# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash IndexPool requires NVIDIA CUDA",
    ),
]

op = flaggems_vllm.hy4_flash_prepare_groups


def make_inputs(shape):
    batch, query = shape
    ratio = 4
    tokens = batch * query
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query
    lengths = torch.full((batch,), query + ratio - 1, device="cuda", dtype=torch.int32)
    keys = torch.randn((tokens, 128), dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(3 * tokens, dtype=torch.int64, device="cuda").reshape(
        3, tokens
    )
    tail = torch.randn((batch, ratio, 131), dtype=torch.float32, device="cuda")
    tail[..., 128:] = torch.arange(3, device="cuda")
    table = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
    return keys, positions, tail, table, starts, lengths, tokens, ratio


def reference(keys, positions, tail, table, starts, lengths, tokens, ratio):
    context = lengths - starts.diff()
    counts = lengths // ratio - context // ratio
    prefix = counts.cumsum(0)
    slots = (tokens + len(lengths) * (ratio - 1)) // ratio
    slot = torch.arange(slots, device=keys.device)
    request = torch.searchsorted(prefix, slot, right=True)
    group_live = request < len(lengths)
    request = request.clamp(max=len(lengths) - 1)
    prior = (prefix - counts)[request]
    group = context[request] // ratio + slot - prior
    member = group[:, None] * ratio + torch.arange(ratio, device=keys.device)
    old = member < context[request, None]
    source = (starts[request, None] + member - context[request, None]).clamp(
        0, tokens - 1
    )
    page = table[request, 0].long()
    retained = tail[page[:, None], member % tail.shape[1]]
    raw = torch.where(old[..., None], retained[..., :128], keys[source].float())
    raw = raw.masked_fill(~group_live[:, None, None], 0).to(keys.dtype)
    xyz = (
        positions[:, source[:, 0]].transpose(0, 1)
        if positions.ndim == 2
        else positions[source[:, 0], None]
    )
    xyz = torch.where(old[:, :1], retained[:, 0, 128:].long(), xyz).transpose(0, 1)
    xyz = xyz.masked_fill(~group_live[None, :], 0)
    if positions.ndim == 1:
        xyz = xyz[0]
    token = torch.arange(tokens, device=keys.device)
    owner = torch.searchsorted(starts[1:].contiguous().long(), token, right=True)
    safe_owner = owner.clamp(max=len(lengths) - 1)
    token_live = owner < len(lengths)
    logical = context[safe_owner].long() + token - starts[safe_owner]
    upper = torch.where(token_live, logical // ratio, 0).to(torch.int32)
    return (
        raw,
        xyz,
        request.to(torch.int32),
        group,
        group_live,
        safe_owner,
        logical,
        upper,
        token_live,
    )


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (4, 7)])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.hy4_flash_prepare_groups
def test_prepare_groups_reads_committed_carry_and_pads_unused_slots(shape, mrope):
    args = list(make_inputs(shape))
    if not mrope:
        args[1] = args[1][0]
    got = op(*args)
    expected = reference(*args)
    for actual_tensor, expected_tensor in zip(got, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
