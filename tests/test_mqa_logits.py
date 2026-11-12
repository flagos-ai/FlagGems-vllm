# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
import pytest
import torch

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(
    ops.vendor_name != "metax", reason="MetaX specialization"
)


def check_replays(call, update, check, outputs=()):
    # No compilation, allocation growth or tuning is permitted during capture.
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        update(value)
        for output in outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.parametrize("row_shard", [False, True])
@pytest.mark.parametrize("m,n", [(1, 64), (65, 129)])
@pytest.mark.mqa_logits
def test_public_mqa_logits_independent_reference_and_replay(row_shard, m, n):
    torch.manual_seed(1234)
    q = (torch.randn(m, 64, 128, device="cuda") * 0.1).bfloat16()
    kv = (torch.randn(n, 128, device="cuda") * 0.1).bfloat16()
    weights = torch.randn(m, 64, device="cuda") * 0.1
    ks = torch.zeros(m, dtype=torch.int32, device="cuda")
    ke = torch.full((m,), n, dtype=torch.int32, device="cuda")
    ke[0] = 0
    out = torch.empty(m + 3, n, device="cuda", dtype=torch.float32).fill_(123)

    def call():
        ops.mqa_logits(q, kv, weights, ks, ke, row_shard=row_shard, out=out)

    def check():
        # CPU FP32 reference is independent of MetaX TF32 overrides/tiled code.
        score = torch.einsum("mhd,nd->mhn", q.cpu().float(), kv.cpu().float()).relu()
        expected = (score * weights.cpu()[:, :, None]).sum(1)
        expected[0].fill_(-torch.inf)
        torch.testing.assert_close(out[:m].cpu(), expected, atol=2e-4, rtol=2e-4)
        assert (out[m:] == 123).all()

    call()
    check()
    check_replays(call, lambda value: q.fill_(value), check, outputs=(out[:m],))


@pytest.mark.mqa_logits
@pytest.mark.parametrize("row_shard", [False, True])
@pytest.mark.parametrize("m,n", [(1, 1), (65, 129)])
@pytest.mark.parametrize("pitched_output", [False, True])
def test_shared_mqa_partial_bounds_output_padding_and_replay(
    row_shard, m, n, pitched_output
):
    import importlib

    common = importlib.import_module("flaggems_vllm.ops.fp8_fp4_mqa_logits")
    vendor = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.mqa_logits"
    )
    assert (
        vendor._row_tile_logits.jit_function
        is common._fp8_fp4_mqa_logits_kernel.jit_function
    )
    assert (
        vendor._row_shard_logits.jit_function
        is common._fp8_fp4_mqa_logits_kernel.jit_function
    )
    generator = torch.Generator(device="cuda").manual_seed(104)
    q = (torch.randn(m, 64, 128, device="cuda", generator=generator) * 0.1).bfloat16()
    kv = (torch.randn(n, 128, device="cuda", generator=generator) * 0.1).bfloat16()
    weights = torch.randn(m, 64, device="cuda", generator=generator) * 0.1
    starts = (torch.arange(m, device="cuda", dtype=torch.int32) * 13) % (n + 1) - 1
    ends = torch.minimum(starts + n // 2 + 1, torch.full_like(starts, n + 3))
    ends[::3] = starts[::3]
    storage = torch.full(
        (m + 3, n + (7 if pitched_output else 0)), 123.0, device="cuda"
    )
    out = storage[:, :n]

    def call():
        ops.mqa_logits(q, kv, weights, starts, ends, row_shard=row_shard, out=out)

    def update(value):
        q.fill_(value)
        weights.mul_(-1)
        starts.add_(1)
        ends.add_(1)

    def check():
        expected = torch.einsum("mhd,nd->mhn", q.cpu().float(), kv.cpu().float()).relu()
        expected = (expected * weights.cpu()[:, :, None]).sum(1)
        columns = torch.arange(n)[None, :]
        valid = (columns >= starts.cpu()[:, None]) & (columns < ends.cpu()[:, None])
        expected.masked_fill_(~valid, -torch.inf)
        torch.testing.assert_close(out[:m].cpu(), expected, atol=2e-4, rtol=2e-4)
        assert (storage[m:] == 123).all()
        assert (storage[:m, n:] == 123).all()

    call()
    check()
    check_replays(call, update, check, outputs=(out[:m],))


@pytest.mark.mqa_logits
@pytest.mark.parametrize("row_shard", [False, True])
@pytest.mark.parametrize("m", [3, 17])
def test_shared_mqa_first_call_graph_capture(row_shard, m):
    generator = torch.Generator(device="cuda").manual_seed(104)
    n = 97
    q = (torch.randn(m, 64, 128, device="cuda", generator=generator) * 0.1).bfloat16()
    kv = (torch.randn(n, 128, device="cuda", generator=generator) * 0.1).bfloat16()
    weights = torch.randn(m, 64, device="cuda", generator=generator) * 0.1
    starts = torch.arange(m, device="cuda", dtype=torch.int32)
    ends = starts + 65
    storage = torch.full((m + 2, n + 5), 123.0, device="cuda")
    out = storage[:, :n]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Intentionally no target call before capture: a fixed one-config adapter
    # must not run a tuner benchmark while recording this new invocation.
    with torch.cuda.graph(graph):
        ops.mqa_logits(q, kv, weights, starts, ends, row_shard=row_shard, out=out)
    for value in (0.125, -0.25):
        q.fill_(value)
        out[:m].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.einsum("mhd,nd->mhn", q.cpu().float(), kv.cpu().float()).relu()
        expected = (expected * weights.cpu()[:, :, None]).sum(1)
        columns = torch.arange(n)[None, :]
        valid = (columns >= starts.cpu()[:, None]) & (columns < ends.cpu()[:, None])
        expected.masked_fill_(~valid, -torch.inf)
        torch.testing.assert_close(out[:m].cpu(), expected, atol=2e-4, rtol=2e-4)
        assert (storage[m:] == 123).all()
        assert (storage[:m, n:] == 123).all()


@pytest.mark.fp8_fp4_mqa_logits
@pytest.mark.parametrize("m,h,n", [(1, 1, 17), (17, 4, 65), (65, 64, 129), (2, 1, 5)])
@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("d", [128, 512])
def test_shared_mqa_legacy_fp8_public_api_and_replay(m, h, n, clean, d):
    generator = torch.Generator(device="cuda").manual_seed(104)
    q = (torch.randn(m, h, d, device="cuda", generator=generator) * 0.1).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(n, d, device="cuda", generator=generator) * 0.1).to(
        torch.float8_e4m3fn
    )
    if (m, h, n) == (2, 1, 5):
        # All 254 finite E4M3 encodings, including signed zero/subnormals/max.
        raw = torch.arange(256, device="cuda", dtype=torch.int32)
        raw[(raw & 127) == 127] = (
            0  # omit the two NaN encodings from this finite-value case
        )
        q = (
            raw.repeat((2 * d + 255) // 256)[: 2 * d]
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(2, 1, d)
        )
        k = (
            raw.repeat((n * d + 255) // 256)[: n * d]
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(n, d)
        )
    scales = torch.linspace(0.125, 1.125, n, device="cuda")
    weights = torch.randn(m, h, device="cuda", generator=generator) * 0.1
    starts = torch.arange(m, device="cuda", dtype=torch.int32) % n
    ends = torch.minimum(starts + max(1, n // 2), torch.full_like(starts, n))
    result = []

    def call():
        result[:] = [
            ops.fp8_fp4_mqa_logits(
                (q, None), (k, scales), weights, starts, ends, clean_logits=clean
            )
        ]

    def update(value):
        weights.fill_(value)
        scales.add_(0.125)

    def check():
        expected = torch.einsum("mhd,nd->mhn", q.cpu().float(), k.cpu().float())
        expected = (expected * scales.cpu()[None, None, :]).relu()
        expected = (expected * weights.cpu()[:, :, None]).sum(1)
        if clean:
            columns = torch.arange(n)[None, :]
            valid = (columns >= starts.cpu()[:, None]) & (columns < ends.cpu()[:, None])
            expected.masked_fill_(~valid, -torch.inf)
        torch.testing.assert_close(result[0].cpu(), expected, atol=2e-4, rtol=2e-4)

    call()
    check()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        update(value)
        result[0].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.fixture(autouse=True)
def restore_rng_state():
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    yield
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(gpu)


def bf16_cache_reference(q, keys, scales, weights, lengths):
    dequant = (keys.cpu().bfloat16() * scales.cpu().bfloat16()[..., None]).bfloat16()
    scores = torch.einsum("bhd,bnd->bhn", q.cpu().float(), dequant.float()).relu()
    expected = (scores * weights.cpu()[:, :, None]).sum(1)
    columns = torch.arange(keys.shape[1])[None, :]
    return expected.masked_fill(columns >= lengths.cpu()[:, None], -torch.inf)


@pytest.mark.mqa_logits
@pytest.mark.parametrize("batch,slots", [(1, 128), (3, 256)])
def test_gathered_bf16_query_reference_graph_and_padding(batch, slots):
    generator = torch.Generator(device="cuda").manual_seed(713)
    q = (
        torch.randn(batch, 64, 128, device="cuda", generator=generator) * 0.1
    ).bfloat16()
    keys = (
        torch.randn(batch, slots, 128, device="cuda", generator=generator) * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.12345, 1.12345, batch * slots, device="cuda").view(
        batch, slots
    )
    weights = torch.randn(batch, 64, device="cuda", generator=generator) * 0.1
    lengths = torch.full((batch,), slots - 7, device="cuda", dtype=torch.int32)
    lengths[0] = 0
    backing = torch.full((batch, slots + 8), 123.0, device="cuda")
    out = backing[:, :slots]

    def call():
        assert ops.mqa_logits(q, (keys, scales), weights, None, lengths, out=out) is out

    def check():
        expected = bf16_cache_reference(q, keys, scales, weights, lengths)
        torch.testing.assert_close(out.cpu(), expected, atol=2e-4, rtol=2e-4)
        assert bool((backing[:, slots:] == 123).all())

    def update(value):
        q.fill_(value)
        scales.add_(0.0625)
        lengths.fill_(slots - 13)

    call()
    check()
    check_replays(call, update, check, outputs=(out,))


@pytest.mark.paged_mqa_logits
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("rank4", [False, True])
def test_paged_bf16_query_pitched_shared_cache_graph(batch, rank4):
    generator = torch.Generator(device="cuda").manual_seed(714)
    pages, block, slots = 8, 64, 256
    pitch = block * 132 + 64
    raw = torch.full((pages * pitch + 64,), 93, device="cuda", dtype=torch.uint8)
    cache = raw.as_strided(
        (pages, block, 1, 132), (pitch, 132, 132, 1), storage_offset=64
    )
    flat = cache.view(pages, -1)
    keys = (
        torch.randn(pages, block, 128, device="cuda", generator=generator) * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.12345, 1.12345, pages * block, device="cuda").view(
        pages, block
    )
    flat[:, : block * 128].copy_(keys.view(torch.uint8).reshape(pages, -1))
    flat[:, block * 128 :].view(torch.float32).copy_(scales)
    q = (
        torch.randn(batch, 64, 128, device="cuda", generator=generator) * 0.1
    ).bfloat16()
    weights = torch.randn(batch, 64, device="cuda", generator=generator) * 0.1
    table_storage = torch.full((batch, 6), -1, device="cuda", dtype=torch.int32)
    tables = table_storage[:, 1:5]
    tables.copy_(torch.arange(4, device="cuda", dtype=torch.int32).repeat(batch, 1))
    lengths = torch.full((batch,), slots - 7, device="cuda", dtype=torch.int32)
    lengths[0] = 0
    backing = torch.full((batch, slots + 8), 123.0, device="cuda")
    out = backing[:, :slots]
    original_cache = raw.clone()

    def call():
        query = q[:, None] if rank4 else q
        returned = ops.paged_mqa_logits(
            query,
            cache,
            weights,
            lengths[:, None] if rank4 else lengths,
            tables,
            None,
            slots,
            True,
            out=out,
        )
        assert returned is out

    def check():
        table = tables.cpu().long()
        gathered_keys = keys.cpu()[table].reshape(batch, slots, 128)
        gathered_scales = scales.cpu()[table].reshape(batch, slots)
        expected = bf16_cache_reference(
            q, gathered_keys, gathered_scales, weights, lengths
        )
        torch.testing.assert_close(out.cpu(), expected, atol=2e-4, rtol=2e-4)
        assert bool((backing[:, slots:] == 123).all())
        assert torch.equal(raw, original_cache)
        assert bool((table_storage[:, [0, -1]] == -1).all())

    def update(value):
        q.fill_(value)
        tables.copy_(
            (
                torch.arange(4, device="cuda", dtype=torch.int32)[None, :]
                + (1 if value > 0 else 3)
            ).repeat(batch, 1)
            % pages
        )
        lengths.fill_(slots - 13)

    call()
    check()
    check_replays(call, update, check, outputs=(out,))
