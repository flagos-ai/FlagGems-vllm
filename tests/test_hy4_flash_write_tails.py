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


def make_inputs(shape, strided=False):
    batch, query = shape
    ring = 4
    tokens = batch * query
    keys = torch.randn((tokens, 128), dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(3 * tokens, dtype=torch.int64, device="cuda").reshape(
        3, tokens
    )
    tail = torch.full((batch, ring, 131 * (1 + strided)), -7.0, device="cuda")
    tail = tail[..., :: 1 + strided]
    local = torch.arange(ring, device="cuda")[None, :]
    base = torch.arange(batch, device="cuda")[:, None]
    rows = base * query + query - ring + local
    slots = base * ring + (query + ring - 1 - ring + local) % ring
    live = local >= ring - query
    return (
        keys,
        positions,
        tail,
        torch.where(live, rows, -1).flatten(),
        torch.where(live, slots, -1).flatten(),
    )


def reference(keys, positions, tail, key_rows, slots):
    ring = tail.shape[1]
    valid = (
        (key_rows >= 0)
        & (key_rows < min(len(keys), positions.shape[-1]))
        & (slots >= 0)
        & (slots < tail.shape[0] * ring)
    )
    rows, slot = key_rows[valid], slots[valid]
    xyz = (
        positions[:, rows].T
        if positions.ndim == 2
        else positions[rows, None].expand(-1, 3)
    )
    value = torch.cat((keys[rows].float(), xyz.float()), dim=-1)
    tail[slot // ring, slot % ring] = value


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (4, 7)])
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.hy4_flash_write_tails
def test_write_tails_retains_explicit_rows_and_preserves_stride_padding(shape, mrope):
    keys, positions, actual, rows, slots = make_inputs(shape, strided=True)
    if not mrope:
        positions = positions[0]
    expected = actual.clone()
    reference(keys, positions, expected, rows, slots)
    assert op(keys, positions, actual, rows, slots) is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    base = actual.as_strided((*actual.shape[:-1], 262), (*actual.stride()[:-1], 1))
    assert torch.all(base[..., 1::2] == -7)


@pytest.mark.hy4_flash_write_tails
def test_write_tails_replays_new_slots_and_leaves_invalid_rows_unchanged():
    keys, positions, tail, rows, slots = make_inputs((2, 4))
    op(keys, positions, tail, rows, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(keys, positions, tail, rows, slots)
    rows.fill_(-1)
    slots.fill_(-1)
    rows[:2] = torch.tensor([0, 1], device="cuda")
    slots[:2] = torch.tensor([tail.shape[1] + 2, 10000], device="cuda")
    tail.fill_(-7)
    expected = tail.clone()
    reference(keys, positions, expected, rows, slots)
    graph.replay()
    torch.testing.assert_close(tail, expected, rtol=0, atol=0)
