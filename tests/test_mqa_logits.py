# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
import pytest
import torch

import flaggems_vllm as ops

from .mqa_utils import restore_rng_state  # noqa: F401 (pytest fixture)
from .mqa_utils import bf16_cache_reference, check_replays, shared_mqa_reference

pytestmark = [
    pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX specialization"),
    pytest.mark.usefixtures("restore_rng_state"),
]


@pytest.mark.mqa_logits
@pytest.mark.parametrize("row_shard", [False, True])
@pytest.mark.parametrize(
    "m,n,bounds,pitched_output",
    [
        (1, 64, "full", False),
        (65, 129, "full", False),
        (1, 1, "partial", False),
        (1, 1, "partial", True),
        (65, 129, "partial", False),
        (65, 129, "partial", True),
    ],
)
def test_shared_mqa_bounds_output_padding_and_replay(
    row_shard, m, n, bounds, pitched_output
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
    generator = torch.Generator(device="cuda").manual_seed(
        1234 if bounds == "full" else 104
    )
    q = (torch.randn(m, 64, 128, device="cuda", generator=generator) * 0.1).bfloat16()
    kv = (torch.randn(n, 128, device="cuda", generator=generator) * 0.1).bfloat16()
    weights = torch.randn(m, 64, device="cuda", generator=generator) * 0.1
    if bounds == "full":
        starts = torch.zeros(m, dtype=torch.int32, device="cuda")
        ends = torch.full((m,), n, dtype=torch.int32, device="cuda")
        ends[0] = 0
    else:
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
        if bounds == "partial":
            weights.mul_(-1)
            starts.add_(1)
            ends.add_(1)

    def check():
        expected = shared_mqa_reference(q, kv, weights, starts, ends)
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

    def call():
        ops.mqa_logits(q, kv, weights, starts, ends, row_shard=row_shard, out=out)

    def check():
        expected = shared_mqa_reference(q, kv, weights, starts, ends)
        torch.testing.assert_close(out[:m].cpu(), expected, atol=2e-4, rtol=2e-4)
        assert (storage[m:] == 123).all()
        assert (storage[:m, n:] == 123).all()

    # No target call before capture: the adapter must not tune while recording.
    check_replays(
        call,
        lambda value: q.fill_(value),
        check,
        outputs=(out[:m],),
        values=(0.125, -0.25),
        warmup=0,
    )


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
