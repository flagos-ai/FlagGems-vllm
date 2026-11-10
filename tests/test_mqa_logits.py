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
