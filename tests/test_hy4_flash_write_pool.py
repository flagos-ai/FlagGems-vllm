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

op = flaggems_vllm.hy4_flash_write_pool


def make_inputs(shape, strided=False):
    rows, page_size = shape
    pages = max(1, (rows + page_size - 1) // page_size)
    storage = torch.full(
        (pages, page_size + int(strided), 132), 7, device="cuda", dtype=torch.uint8
    )
    cache = storage[:, :page_size]
    groups = torch.arange(rows, device="cuda", dtype=torch.int64)
    slots = groups.masked_fill(groups % 3 == 1, -1)
    q = torch.randn(rows, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(rows, device="cuda")
    return cache, slots, q, scales


def reference(cache, slots, quant, scales):
    page = cache.shape[1]
    bytes = cache.as_strided((cache.shape[0], page * 132), (cache.stride(0), 1))
    active = (slots >= 0) & (slots < cache.shape[0] * page)
    selected = slots[active]
    ptr = selected // page
    slot = selected % page
    i = slot[:, None] * 128 + torch.arange(128, device=cache.device)
    j = page * 128 + slot[:, None] * 4 + torch.arange(4, device=cache.device)
    # The model supplies distinct slots. Preserve the bytes of inactive slots.
    bytes[ptr[:, None], i] = quant.view(torch.uint8)[active]
    bytes[ptr[:, None], j] = scales.view(torch.uint8).reshape(-1, 4)[active]


@pytest.mark.parametrize("shape", [(0, 4), (1, 4), (17, 4), (129, 64)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.hy4_flash_write_pool
def test_write_pool_respects_inactive_slots_and_page_stride(shape, strided):
    actual, *args = make_inputs(shape, strided)
    expected, *_ = make_inputs(shape, strided)
    reference(expected, *args)
    assert op(actual, *args) is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.hy4_flash_write_pool
def test_write_pool_rejects_noninteger_indices():
    cache, slots, *args = make_inputs((1, 4))
    with pytest.raises(ValueError, match="metadata"):
        op(cache, slots.float(), *args)


@pytest.mark.hy4_flash_write_pool
def test_write_pool_graph_updates_slots_and_does_not_touch_padding():
    cache, slots, quant, scales = make_inputs((9, 4), strided=True)
    op(cache, slots, quant, scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(cache, slots, quant, scales)

    slots.copy_(torch.tensor([-1, 11, 0, 5, 99, 7, 1, 10, -1], device="cuda"))
    quant.view(torch.uint8).copy_(
        torch.full_like(quant, 1.5, dtype=torch.float32)
        .to(quant.dtype)
        .view(torch.uint8)
    )
    scales.fill_(2.5)
    cache.fill_(7)
    expected, *_ = make_inputs((9, 4), strided=True)
    reference(expected, slots, quant, scales)
    graph.replay()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    raw_pages = cache.as_strided((3, 5, 132), cache.stride())
    assert torch.all(raw_pages[:, -1] == 7)
