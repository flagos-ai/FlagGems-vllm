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
    table = torch.arange(pages, dtype=torch.int32, device="cuda")[None, :]
    groups = torch.arange(rows, device="cuda", dtype=torch.int64)
    owners = torch.zeros_like(groups)
    active = groups % 3 != 1
    q = torch.randn(rows, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(rows, device="cuda")
    return cache, table, owners, groups, active, q, scales


def reference(cache, table, owners, groups, active, quant, scales):
    page = cache.shape[1]
    bytes = cache.as_strided((cache.shape[0], page * 132), (cache.stride(0), 1))
    ptr = table[owners, (groups // page).clamp(0, table.shape[1] - 1)].long()
    slot = groups % page
    i = slot[:, None] * 128 + torch.arange(128, device=cache.device)
    j = page * 128 + slot[:, None] * 4 + torch.arange(4, device=cache.device)
    # The model supplies distinct slots. Preserve the bytes of inactive slots.
    bytes[ptr[:, None], i] = torch.where(
        active[:, None], quant.view(torch.uint8), bytes[ptr[:, None], i]
    )
    bytes[ptr[:, None], j] = torch.where(
        active[:, None], scales.view(torch.uint8).reshape(-1, 4), bytes[ptr[:, None], j]
    )


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
    cache, table, owners, *args = make_inputs((1, 4))
    with pytest.raises(ValueError, match="metadata"):
        op(cache, table, owners.float(), *args)
