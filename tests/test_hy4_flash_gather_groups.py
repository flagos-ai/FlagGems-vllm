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

op = flaggems_vllm.hy4_flash_gather_groups


def make_inputs(shape, strided=False):
    batch, query = shape
    ratio = 4
    tokens = batch * query
    keys = torch.randn((tokens, 128), dtype=torch.bfloat16, device="cuda")
    tail = torch.randn((batch, ratio, 131 * (1 + strided)), device="cuda")
    tail = tail[..., :: 1 + strided]
    slots = (tokens + batch * (ratio - 1)) // ratio
    source = torch.arange(slots * ratio, device="cuda").reshape(slots, ratio)
    return (
        keys,
        tail,
        torch.where((source % 3 == 0) & (source < tokens), source, -1),
        torch.where(source % 3 == 1, source % (batch * ratio), -1),
    )


def reference(keys, tail, key_rows, tail_slots):
    has_key = (key_rows >= 0) & (key_rows < len(keys))
    has_tail = (tail_slots >= 0) & (tail_slots < tail.shape[0] * tail.shape[1])
    source = tail_slots.clamp(0, tail.shape[0] * tail.shape[1] - 1)
    retained = tail[source // tail.shape[1], source % tail.shape[1], :128]
    return torch.where(
        has_key[..., None],
        keys[key_rows.clamp(0, len(keys) - 1)].float(),
        retained.masked_fill(~has_tail[..., None], 0),
    ).to(keys.dtype)


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (4, 7)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.hy4_flash_gather_groups
def test_gather_uses_prepared_rows_and_honors_tail_stride(shape, strided):
    args = make_inputs(shape, strided=strided)
    torch.testing.assert_close(op(*args), reference(*args), rtol=0, atol=0)


@pytest.mark.hy4_flash_gather_groups
def test_gather_replays_updated_sources():
    keys, tail, rows, slots = make_inputs((2, 4), strided=True)
    op(keys, tail, rows, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(keys, tail, rows, slots)
    rows.fill_(-1)
    slots.fill_(3)
    tail[0, 3, :128].fill_(19)
    graph.replay()
    torch.testing.assert_close(
        output, reference(keys, tail, rows, slots), rtol=0, atol=0
    )
