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

op = flaggems_vllm.hy4_flash_write_tails


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
    return keys, positions, tail, table, starts, lengths


def reference(keys, positions, tail, table, starts, lengths):
    ring = tail.shape[1]
    used = starts.diff()
    end = lengths[:, None]
    logical = end - ring + torch.arange(ring, device=keys.device)
    context = end - used[:, None]
    live = (used[:, None] > 0) & (logical >= context) & (context >= 0)
    index = (starts[:-1, None] + logical - context).long().clamp(0, len(keys) - 1)
    page = table[:, :1].long()
    slot = logical.long() % ring
    if positions.ndim == 2:
        xyz = positions[:, index].movedim(0, -1)
    else:
        xyz = positions[index, None].expand(-1, -1, 3)
    value = torch.cat((keys[index].float(), xyz.float()), dim=-1)
    tail[page, slot] = torch.where(live[..., None], value, tail[page, slot])


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (4, 7)])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.hy4_flash_write_tails
def test_write_tails_retains_recent_keys_and_preserves_unused_slots(shape, mrope):
    keys, positions, actual, table, starts, lengths = make_inputs(shape)
    if not mrope:
        positions = positions[0]
    expected = actual.clone()
    reference(keys, positions, expected, table, starts, lengths)
    assert op(keys, positions, actual, table, starts, lengths) is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
