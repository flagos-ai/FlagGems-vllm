# SPDX-License-Identifier: Apache-2.0
"""FP8 page addressing, numerical parity and changing-input graph replay."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import paged_mqa_logits

from .mqa_utils import restore_rng_state  # noqa: F401 (pytest fixture)
from .mqa_utils import bf16_cache_reference, check_replays, pack_fp8_paged_cache

pytestmark = pytest.mark.gpu


def make_case(page_size, padding=0, pool_pages=128, batch=3):
    torch.manual_seed(11)
    device = "cuda"
    width, heads, max_len = 128, 32, 2051
    keys = torch.randn(pool_pages, page_size, width, device=device).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pool_pages, page_size, device=device) * 0.05 + 0.01
    cache, _ = pack_fp8_paged_cache(keys, scales, padding=padding)
    query = torch.randn(batch, 1, heads, width, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, device=device)
    table = torch.stack(
        [torch.randperm(pool_pages, device=device)[:65] for _ in range(batch)]
    ).int()
    lens = torch.full((batch,), 65, dtype=torch.int32, device=device)
    return query, cache, weights, lens, table, keys, scales, max_len


def candidate(case):
    q, cache, weights, lens, table, _, _, max_len = case
    return paged_mqa_logits(
        (q, None),
        cache,
        weights,
        lens,
        table,
        None,
        max_len,
        True,
    )


def reference(case):
    q, _, weights, lens, table, keys, scales, max_len = case
    result = torch.full((len(lens), max_len), -torch.inf, device=q.device)
    for row, length in enumerate(lens.tolist()):
        idx = table[row].long()
        k = keys.float()[idx].reshape(-1, 128)[:length]
        s = scales[idx].reshape(-1)[:length]
        dots = q[row, 0].float() @ k.T
        result[row, :length] = (torch.relu(dots * s) * weights[row, :, None]).sum(0)
    return result


@pytest.mark.paged_mqa_logits
@pytest.mark.parametrize("page_size,padding", [(32, 0), (32, 256), (64, 0), (64, 256)])
def test_page_stride_eager_and_changing_graph_inputs(page_size, padding):
    case = make_case(page_size, padding)
    for length in (0, 1, page_size - 1, page_size, page_size + 1, 2049):
        case[3].copy_(torch.tensor([length, length // 2, 0], device="cuda"))
        torch.testing.assert_close(
            candidate(case), reference(case), atol=2e-5, rtol=2e-5
        )
    captured = []

    def call():
        captured[:] = [candidate(case)]

    def update(iteration):
        # Rewrite metadata after capture: empty rows become live and logical
        # pages resolve to different physical pages.
        if iteration == 0:
            case[3].copy_(torch.tensor([33, 2051, 64], device="cuda"))
            case[4].copy_(case[4].flip(1))
        case[0].copy_(
            (case[0].float() * (-1 if iteration % 2 else 0.5)).to(case[0].dtype)
        )

    def check():
        torch.testing.assert_close(captured[0], reference(case), atol=2e-5, rtol=2e-5)

    check_replays(
        call, update, check, outputs=lambda: captured, values=range(10), warmup=0
    )


@pytest.mark.paged_mqa_logits
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax", reason="MetaX BF16-query extension"
)
@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("rank4", [False, True])
def test_paged_bf16_query_pitched_shared_cache_graph(batch, rank4):
    generator = torch.Generator(device="cuda").manual_seed(714)
    pages, block, slots = 8, 64, 256
    keys = (
        torch.randn(pages, block, 128, device="cuda", generator=generator) * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.12345, 1.12345, pages * block, device="cuda").view(
        pages, block
    )
    cache, raw = pack_fp8_paged_cache(
        keys, scales, padding=64, storage_offset=64, fill_value=93
    )
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
        returned = paged_mqa_logits(
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
