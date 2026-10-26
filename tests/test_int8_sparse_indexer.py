# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Independent CPU references for the complete INT8 indexer data contract."""

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or gems.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)


def quant_reference(x, eps=1e-10, ue8m0=True):
    values = x.cpu().float().reshape(-1, 128)
    scale = values.abs().amax(-1, keepdim=True).clamp_min(eps) / 127
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (values / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.reshape(x.shape), scale


def cache_reference(k, slots, shape, marker, ue8m0=True):
    cache = torch.full(shape, marker, dtype=torch.uint8)
    page = shape[1]
    flat = cache.view(shape[0], -1)
    values = flat[:, : page * 128].view(torch.int8)
    scales = flat[:, page * 128 :].view(torch.float32)
    quantized, scale = quant_reference(k, 1e-4, ue8m0)
    for row, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < shape[0] * page:
            p, offset = divmod(slot, page)
            values[p, offset * 128 : (offset + 1) * 128] = quantized[row]
            scales[p, offset] = scale[row, 0]
    return cache


@pytest.mark.per_token_group_quant_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,groups", [(0, 1), (1, 1), (7, 2), (33, 1)])
@pytest.mark.parametrize("ue8m0", [False, True])
def test_query_quantization(dtype, rows, groups, ue8m0):
    torch.manual_seed(101)
    x = torch.randn(rows, groups * 128 + 3, device="cuda", dtype=dtype)[
        :, : groups * 128
    ]
    before = x.clone()
    q, scales = gems.per_token_group_quant_int8(x, 128, use_ue8m0=ue8m0)
    assert q.dtype == torch.int8 and scales.dtype == torch.float32
    assert scales.shape == (rows, groups)
    if rows:
        expected, expected_scales = quant_reference(x, ue8m0=ue8m0)
        torch.testing.assert_close(q.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(
            scales.cpu(), expected_scales.reshape(rows, groups), rtol=1e-6, atol=0
        )
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.per_token_group_quant_int8
def test_round_to_nearest_even_and_zero():
    x = torch.zeros(2, 128, device="cuda")
    x[0, :9] = torch.tensor(
        [-127, -0.5, 0.5, 1.5, 2.5, -1.5, -2.5, 3.5, 127], device="cuda"
    )
    q, scale = gems.per_token_group_quant_int8(x, 128, use_ue8m0=False)
    expected, expected_scale = quant_reference(x, ue8m0=False)
    torch.testing.assert_close(q.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=1e-6, atol=0)


@pytest.mark.indexer_k_quant_and_cache_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 7, 33])
def test_cache_capture_replay(dtype, rows):
    torch.manual_seed(103)
    k = torch.randn(rows + 2, 128, device="cuda", dtype=dtype)
    slots = torch.cat(
        [
            (torch.arange(rows, device="cuda") * 3 + 7) % 256,
            torch.tensor([-1, 256], device="cuda"),
        ]
    )
    cache = torch.full((4, 64, 132), 0x5A, device="cuda", dtype=torch.uint8)
    before = k.clone()

    def call():
        gems.indexer_k_quant_and_cache_int8(k, cache, slots, 128, "ue8m0")

    for _ in range(3):
        call()
    torch.testing.assert_close(
        cache.cpu(), cache_reference(k, slots, cache.shape, 0x5A), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for shift in (11, 19):
        k.normal_()
        slots[:rows].copy_((slots[:rows] + shift) % 256)
        cache.fill_(0xA5)
        before.copy_(k)
        expected = cache_reference(k, slots, cache.shape, 0xA5)
        graph.replay()
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(k, before, rtol=0, atol=0)


def logits_reference(q, k, scales, weights):
    # INT32 dot is exact for D=128; CPU FP64 gives an independent reference.
    dots = torch.einsum("mhd,nd->mhn", q.cpu().double(), k.cpu().double())
    scaled = (dots * scales.cpu().double()[None, None, :]).clamp_min(0)
    return (scaled * weights.cpu().double()[:, :, None]).sum(1).float()


@pytest.mark.int8_mqa_logits
@pytest.mark.parametrize(
    "m,n,h",
    [
        (0, 65, 32),
        (1, 1, 16),
        (7, 63, 32),
        (16, 64, 64),
        (33, 65, 32),
        (7, 257, 32),
        (32, 129, 16),
        (33, 127, 64),
        (128, 513, 32),
    ],
)
@pytest.mark.parametrize("clean", [False, True])
def test_prefill_integer_logits(m, n, h, clean):
    torch.manual_seed(107)
    q = torch.randint(-127, 128, (m, h, 128), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, (n, 128), device="cuda", dtype=torch.int8)
    scales = torch.rand(n, device="cuda") * 0.01 + 0.001
    weights = torch.randn(m, h, device="cuda")
    starts = torch.arange(m, device="cuda", dtype=torch.int32) % n
    ends = torch.full((m,), n, device="cuda", dtype=torch.int32)
    actual = gems.int8_mqa_logits((q, None), (k, scales), weights, starts, ends, clean)
    expected = logits_reference(q, k, scales, weights)
    valid = torch.arange(n)[None, :] >= starts.cpu()[:, None]
    expected.masked_fill_(~valid, -float("inf") if clean else 0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=2e-3)


def paged_reference(q, cache, weights, lengths, tables, capacity, clean):
    batch, next_n, h, _ = q.shape
    page = cache.shape[1]
    flat = cache.cpu().view(cache.shape[0], -1)
    values = flat[:, : page * 128].view(torch.int8).reshape(-1, page, 128)
    scales = flat[:, page * 128 :].view(torch.float32)
    expected = torch.full(
        (batch * next_n, capacity), -float("inf") if clean else 0, dtype=torch.float32
    )
    for b in range(batch):
        for token in range(next_n):
            row = b * next_n + token
            length = int(lengths[b, token]) if lengths.ndim == 2 else int(lengths[b])
            table = tables[b, token] if tables.ndim == 3 else tables[b]
            keys, ss = [], []
            for pos in range(min(length, capacity)):
                physical, offset = int(table[pos // page]), pos % page
                keys.append(values[physical, offset])
                ss.append(scales[physical, offset])
            if keys:
                ref = logits_reference(
                    q[b, token].unsqueeze(0),
                    torch.stack(keys),
                    torch.stack(ss),
                    weights[row : row + 1],
                )
                expected[row, : len(keys)] = ref[0]
    return expected


@pytest.mark.int8_paged_mqa_logits
@pytest.mark.parametrize("page", [16, 64, 128])
@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("clean", [False, True])
def test_paged_graph_refreshed_metadata(page, heads, clean):
    torch.manual_seed(109)
    q = torch.randint(-127, 128, (2, 2, heads, 128), device="cuda", dtype=torch.int8)
    weights = torch.randn(4, heads, device="cuda")
    k = torch.randn(4 * page, 128, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(4 * page, device="cuda")
    cache = torch.empty(4, page, 132, device="cuda", dtype=torch.uint8)
    gems.indexer_k_quant_and_cache_int8(k, cache, slots, 128, "ue8m0")
    tables = torch.tensor(
        [[[2, 0], [1, 3]], [[3, 1], [0, 2]]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor(
        [[0, page + 1], [1, page - 1]], device="cuda", dtype=torch.int32
    )
    capacity = 2 * page + 1

    def call():
        return gems.int8_paged_mqa_logits(
            (q, None),
            cache.unsqueeze(2),
            weights,
            lengths,
            tables,
            None,
            capacity,
            clean,
        )

    for _ in range(3):
        out = call()
    expected = paged_reference(
        q, cache, weights, lengths.cpu(), tables.cpu(), capacity, clean
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=2e-3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for _ in range(2):
        q.random_(-127, 128)
        weights.normal_()
        lengths.copy_(lengths.roll(1, 1))
        tables.copy_(tables.roll(1, -1))
        out.fill_(float("nan"))
        expected = paged_reference(
            q, cache, weights, lengths.cpu(), tables.cpu(), capacity, clean
        )
        graph.replay()
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=2e-3)


@pytest.mark.per_token_group_quant_int8
def test_invalid_query_layout_rejected():
    with pytest.raises(NotImplementedError, match="feature stride"):
        gems.per_token_group_quant_int8(torch.empty(2, 256, device="cuda")[:, ::2], 128)


@pytest.mark.per_token_group_quant_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,groups", [(1023, 1), (1024, 1), (1025, 2)])
@pytest.mark.parametrize("ue8m0", [False, True])
def test_batched_quantization_boundary(dtype, rows, groups, ue8m0):
    x = torch.randn(rows, groups * 128 + 3, device="cuda", dtype=dtype)[
        :, : groups * 128
    ]
    q, scale = gems.per_token_group_quant_int8(x, 128, use_ue8m0=ue8m0)
    ref, ref_scale = quant_reference(x, ue8m0=ue8m0)
    torch.testing.assert_close(q.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(
        scale.cpu(), ref_scale.view(rows, groups), rtol=1e-6, atol=0
    )


@pytest.mark.per_token_group_quant_int8
def test_batched_quantization_graph_refresh():
    x = torch.randn(1025, 256, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        gems.per_token_group_quant_int8(x, 128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, scale = gems.per_token_group_quant_int8(x, 128)
    x.normal_()
    q.fill_(-128)
    scale.fill_(float("nan"))
    graph.replay()
    ref, ref_scale = quant_reference(x)
    torch.testing.assert_close(q.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), ref_scale.view(1025, 2), rtol=1e-6, atol=0)
