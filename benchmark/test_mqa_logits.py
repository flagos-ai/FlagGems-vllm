# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Public API benchmarks; component timings are not end-to-end model gains."""
import pytest
import torch

import flaggems_vllm

from . import base


def torch_mqa_into(q, kv, weights, ks, ke, out, row_shard=False):
    scores = torch.einsum("mhd,nd->mhn", q.float(), kv.float()).relu()
    result = (scores * weights[:, :, None]).sum(1)
    columns = torch.arange(kv.shape[0], device=q.device)
    result.masked_fill_(
        (columns[None, :] < ks[:, None]) | (columns[None, :] >= ke[:, None]), -torch.inf
    )
    out.copy_(result)
    return out


class MQAIndexBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "query_rows, key_rows, heads, head_dim"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(16, 1024, 64, 128), (64, 4096, 64, 128), (64, 16384, 64, 128)]

    def get_input_iter(self, dtype):
        for m, n, h, d in self.shapes:
            q = torch.randn(m, h, d, dtype=dtype, device=self.device)
            kv = torch.randn(n, d, dtype=dtype, device=self.device)
            weights = torch.randn(m, h, device=self.device)
            ks = torch.zeros(m, dtype=torch.int32, device=self.device)
            ke = torch.full((m,), n, dtype=torch.int32, device=self.device)
            out = torch.empty(m, n, device=self.device)
            yield q, kv, weights, ks, ke, {"out": out}


@pytest.mark.mqa_logits
def test_mqa_public_api():
    benchmark = MQAIndexBenchmark(
        "mqa_logits",
        torch_mqa_into,
        [torch.bfloat16],
        gems_op=flaggems_vllm.mqa_logits,
    )
    benchmark.run()


def torch_bf16_cache_logits(q, keys, scales, weights, lengths, out):
    dequant = (keys.bfloat16() * scales.bfloat16()[..., None]).bfloat16()
    score = torch.einsum("bhd,bnd->bhn", q.float(), dequant.float()).relu()
    value = (score * weights[:, :, None]).sum(1)
    columns = torch.arange(keys.shape[1], device=q.device)[None, :]
    out.copy_(value.masked_fill(columns >= lengths[:, None], -torch.inf))
    return out


def torch_gathered(q, cache, weights, starts, ends, *, out):
    keys, scales = cache
    return torch_bf16_cache_logits(q, keys, scales, weights, ends, out)


def torch_paged(q, cache, weights, lengths, tables, metadata, slots, clean, *, out):
    pages, block = cache.shape[:2]
    flat = cache.view(pages, -1)
    values = flat[:, : block * 128].view(pages, block, 128)
    scales = flat[:, block * 128 :].view(torch.float32)
    keys = (
        values[tables.long()].reshape(q.shape[0], slots, 128).view(torch.float8_e4m3fn)
    )
    gathered_scales = scales[tables.long()].reshape(q.shape[0], slots)
    return torch_bf16_cache_logits(q, keys, gathered_scales, weights, lengths, out)


class MQAStoredCacheBenchmark(base.Benchmark):
    def __init__(self, paged):
        self.paged = paged
        super().__init__(
            "paged_mqa_logits" if paged else "mqa_logits",
            torch_paged if paged else torch_gathered,
            [torch.bfloat16],
            gems_op=(
                flaggems_vllm.paged_mqa_logits if paged else flaggems_vllm.mqa_logits
            ),
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 128), (16, 1024), (64, 4096)]

    def get_input_iter(self, dtype):
        for rows, slots in self.shapes:
            q = (torch.randn(rows, 64, 128, device=self.device) * 0.1).bfloat16()
            weights = torch.randn(rows, 64, device=self.device) * 0.1
            lengths = torch.full(
                (rows,), slots - 7, device=self.device, dtype=torch.int32
            )
            out = torch.empty(rows, slots, device=self.device)
            if self.paged:
                block = 64
                pages = slots // block
                values = (torch.randn(pages, block, 128, device=self.device) * 0.1).to(
                    torch.float8_e4m3fn
                )
                scales = torch.rand(pages, block, device=self.device) + 0.125
                cache = torch.empty(
                    pages, block, 1, 132, dtype=torch.uint8, device=self.device
                )
                flat = cache.view(pages, -1)
                flat[:, : block * 128].copy_(values.view(torch.uint8).view(pages, -1))
                flat[:, block * 128 :].view(torch.float32).copy_(scales)
                tables = torch.arange(
                    pages, device=self.device, dtype=torch.int32
                ).repeat(rows, 1)
                args = (q, cache, weights, lengths, tables, None, slots, True)
            else:
                keys = (torch.randn(rows, slots, 128, device=self.device) * 0.1).to(
                    torch.float8_e4m3fn
                )
                scales = torch.rand(rows, slots, device=self.device) + 0.125
                args = (q, (keys, scales), weights, None, lengths)
            self.torch_op(*args, out=out)
            expected = out.clone()
            self.gems_op(*args, out=out)
            torch.testing.assert_close(out, expected, atol=2e-4, rtol=2e-4)
            yield (*args, {"out": out})


@pytest.mark.mqa_logits
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX cache contract")
def test_mqa_gathered_cache_public_api():
    MQAStoredCacheBenchmark(False).run()


@pytest.mark.paged_mqa_logits
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax", reason="MetaX BF16-query extension"
)
def test_mqa_paged_cache_public_api():
    MQAStoredCacheBenchmark(True).run()
