# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm import runtime
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_compact

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name != "metax", reason="MetaX compact alignment contract"
)


def oracle(ids, block):
    n = ids.numel()
    sorted_ids = []
    experts = []
    for e in sorted(set(ids.reshape(-1).tolist())):
        matches = torch.nonzero(ids.reshape(-1) == e).flatten().tolist()
        pad = (-len(matches)) % block
        sorted_ids.extend(matches + [n] * pad)
        experts.extend([e] * ((len(matches) + pad) // block))
    return sorted_ids, experts


@pytest.mark.parametrize(
    "m,e",
    [
        (2, 896),
        (3, 896),
        (5, 896),
        (32, 896),
        (57, 896),
        (64, 896),
        (3, 1024),
        (64, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_prefix_and_graph(m, e, dtype):
    torch.manual_seed(5)
    x = torch.zeros((m, 16), dtype=dtype, device="cuda")
    allocated = moe_align_block_size_compact(x, 16, e)
    out = tuple(
        torch.full((t.numel() + 7,), -77, device="cuda", dtype=torch.int32)
        for t in allocated
    )
    for _ in range(2):
        moe_align_block_size_compact(x, 16, e, out=out)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        moe_align_block_size_compact(x, 16, e, out=out)
    for cpu in [
        torch.randint(0, e, (m, 16), dtype=dtype),
        torch.full((m, 16), e - 1, dtype=dtype),
    ]:
        x.copy_(cpu)
        for t in out:
            t.fill_(-77)
        g.replay()
        a, b = oracle(cpu, 16)
        assert out[0][: len(a)].tolist() == a and out[1][: len(b)].tolist() == b
        assert out[2][0].item() == len(a)
        assert all(
            t[end:].eq(-77).all().item() for t, end in zip(out, [len(a), len(b), 1])
        )
        prod = moe_align_block_size_compact(x, 16, e)
        assert (
            prod[0][: len(a)].tolist() == a
            and prod[1][: len(b)].tolist() == b
            and prod[2].item() == len(a)
        )


def test_invalid_buffers():
    x = torch.zeros((2, 16), device="cuda", dtype=torch.int32)
    for e in (1, 5, 256, 1025):
        with pytest.raises(ValueError):
            moe_align_block_size_compact(x, 16, e)
    a, b, c = moe_align_block_size_compact(x, 16, 896)
    for out in [
        (a[:-1], b, c),
        (a, b[:-1], c),
        (a, b, a[:1]),
        (a, b, x.flatten()[:1]),
        (a, b.cpu(), c),
    ]:
        with pytest.raises(ValueError):
            moe_align_block_size_compact(x, 16, 896, out=out)
