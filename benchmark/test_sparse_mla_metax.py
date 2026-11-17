# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Latency coverage for the physical H8/H16 MetaX sparse MLA paths."""

import pytest
import torch

import flaggems_vllm as ops

from . import base


def model1_cache(pages, block=256):
    values = (torch.randn(pages * block, 448, device=ops.device) * 0.1).to(
        torch.float8_e4m3fn
    )
    rope = (torch.randn(pages * block, 64, device=ops.device) * 0.1).bfloat16()
    cache = torch.zeros(pages, block, 1, 584, device=ops.device, dtype=torch.uint8)
    flat = cache.view(pages, -1)
    flat[:, : block * 576] = torch.cat(
        (values.view(torch.uint8), rope.view(torch.uint8)), dim=1
    ).reshape(pages, -1)
    flat[:, block * 576 :] = 125  # power-of-two NoPE scales, 2**(125-127)
    return cache


class SparseMLAMetaXBenchmark(base.Benchmark):
    # Measure only the actual public call; no in-tree comparator is executed.
    DEFAULT_METRICS = ["latency"]

    def __init__(self, mode, heads, head_dim=512):
        self.path = mode
        self.heads = heads
        self.head_dim = head_dim
        fn = (
            ops.flash_mla_with_kvcache if mode == "decode" else ops.flash_mla_sparse_fwd
        )
        super().__init__(fn.__name__, fn, [torch.bfloat16])

    def set_shapes(self, shape_file_path=None):
        self.shapes = (
            [(1, 512), (16, 512), (64, 512), (16, 1024)]
            if self.path == "decode"
            else [(1, 64), (2, 64), (17, 512), (64, 512), (256, 512)]
        )

    def record_shapes(self, *args, **kwargs):
        return {
            "heads": self.heads,
            "path": self.path,
            "head_dim": self.head_dim,
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_input_iter(self, dtype):
        heads, head_dim = self.heads, self.head_dim
        for rows, topk in self.shapes:
            if self.path == "decode":
                q = (torch.randn(rows, 1, heads, 512, device=self.device) * 0.1).to(
                    dtype
                )
                indices = (
                    torch.arange(topk, device=self.device, dtype=torch.int32)
                    .view(1, 1, topk)
                    .repeat(rows, 1, 1)
                )
                extra_indices = (
                    torch.arange(128, device=self.device, dtype=torch.int32)
                    .view(1, 1, 128)
                    .repeat(rows, 1, 1)
                )
                yield (
                    dict(
                        q=q,
                        k_cache=model1_cache(8),
                        block_table=None,
                        cache_seqlens=None,
                        head_dim_v=512,
                        tile_scheduler_metadata=ops.FlashMLASchedMeta(),
                        is_fp8_kvcache=True,
                        indices=indices,
                        topk_length=torch.full(
                            (rows,), topk, device=self.device, dtype=torch.int32
                        ),
                        softmax_scale=512**-0.5,
                        attn_sink=torch.randn(heads, device=self.device) * 0.1,
                        extra_k_cache=model1_cache(2),
                        extra_indices_in_kvcache=extra_indices,
                        extra_topk_length=torch.full(
                            (rows,), 64, device=self.device, dtype=torch.int32
                        ),
                        out=torch.empty_like(q),
                        split_workspace=tuple(
                            torch.empty(shape, device=self.device)
                            for shape in [
                                (64, 4, heads),
                                (64, 4, heads),
                                (64, 4, heads, 512),
                            ]
                        ),
                    ),
                )
            else:
                q = (torch.randn(rows, heads, head_dim, device=self.device) * 0.1).to(
                    dtype
                )
                kv = (torch.randn(1024, 1, head_dim, device=self.device) * 0.1).to(
                    dtype
                )
                indices = torch.arange(
                    topk, device=self.device, dtype=torch.int32
                ).repeat(rows, 1)
                yield (
                    q,
                    kv,
                    indices,
                    head_dim**-0.5,
                    dict(
                        attn_sink=torch.randn(heads, device=self.device) * 0.1,
                        topk_length=torch.full(
                            (rows,), topk, device=self.device, dtype=torch.int32
                        ),
                        out=torch.empty(
                            rows, heads, 512, device=self.device, dtype=dtype
                        ),
                        max_logits_out=torch.empty(rows, heads, device=self.device),
                        lse_out=torch.empty(rows, heads, device=self.device),
                    ),
                )


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend latency")
@pytest.mark.parametrize("heads", [8, 16], ids=["h8", "h16"])
def test_sparse_decode_latency(heads):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(104)
        SparseMLAMetaXBenchmark("decode", heads).run()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend latency")
@pytest.mark.parametrize("heads", [8, 16], ids=["h8", "h16"])
@pytest.mark.parametrize("head_dim", [512, 576])
def test_sparse_prefill_latency(heads, head_dim):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(104)
        SparseMLAMetaXBenchmark("prefill", heads, head_dim).run()
