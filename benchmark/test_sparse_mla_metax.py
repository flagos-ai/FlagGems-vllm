# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare staged/split paths with the retained serial MetaX implementation."""
import dataclasses

import pytest
import torch

import flaggems_vllm as ops
from flaggems_vllm.runtime.backend._metax.ops import sparse_mla_policy as policy

from . import base


def call_with_policy(enabled, mode, *args, **kwargs):
    previous = policy.DEFAULT_POLICY
    policy.DEFAULT_POLICY = dataclasses.replace(
        previous,
        decode_split_topk=enabled,
        prefill_staged=enabled,
        prefill_blockdiag=enabled,
    )
    try:
        fn = (
            ops.flash_mla_with_kvcache if mode == "decode" else ops.flash_mla_sparse_fwd
        )
        return fn(*args, **kwargs)
    finally:
        policy.DEFAULT_POLICY = previous


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
    def __init__(self, mode, real_heads):
        self.path = mode
        self.real_heads = real_heads
        super().__init__(
            "flash_mla_with_kvcache" if mode == "decode" else "flash_mla_sparse_fwd",
            lambda *a, **k: call_with_policy(False, mode, *a, **k),
            [torch.bfloat16],
            gems_op=lambda *a, **k: call_with_policy(True, mode, *a, **k),
        )

    def set_shapes(self, shape_file_path=None):
        self.shapes = (
            [(1, 512), (16, 512), (64, 512), (16, 1024)]
            if self.path == "decode"
            else [
                (2 if self.real_heads == 8 else 1, 64),
                (17, 512),
                (64, 512),
                (256, 512),
            ]
        )

    def record_shapes(self, *args, **kwargs):
        return {
            "real_heads": self.real_heads,
            "path": self.path,
            "comparator": "retained serial MetaX implementation",
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_input_iter(self, dtype):
        for rows, topk in self.shapes:
            if self.path == "decode":
                q = (torch.randn(rows, 1, 16, 512, device=self.device) * 0.1).bfloat16()
                if self.real_heads == 8:
                    q[:, :, 8:] = 0
                cache = model1_cache(8)
                extra = model1_cache(2)
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
                kwargs = dict(
                    q=q,
                    k_cache=cache,
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
                    attn_sink=torch.randn(16, device=self.device) * 0.1,
                    extra_k_cache=extra,
                    extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=torch.full(
                        (rows,), 64, device=self.device, dtype=torch.int32
                    ),
                    out=torch.empty_like(q),
                    real_num_heads=8 if self.real_heads == 8 else None,
                    split_workspace=tuple(
                        torch.empty(shape, device=self.device)
                        for shape in [(64, 4, 16), (64, 4, 16), (64, 4, 16, 512)]
                    ),
                )
                args = ()
            else:
                q = (torch.randn(rows, 16, 512, device=self.device) * 0.1).bfloat16()
                kv = (torch.randn(1024, 1, 512, device=self.device) * 0.1).bfloat16()
                indices = torch.arange(
                    topk, device=self.device, dtype=torch.int32
                ).repeat(rows, 1)
                if self.real_heads == 8:
                    q[:, 8:] = 0
                    indices = indices[:, None]
                args = (q, kv, indices, 512**-0.5)
                kwargs = dict(
                    attn_sink=torch.randn(16, device=self.device) * 0.1,
                    topk_length=torch.full(
                        (rows,), topk, device=self.device, dtype=torch.int32
                    ),
                    out=torch.empty_like(q),
                    max_logits_out=torch.empty(rows, 16, device=self.device),
                    lse_out=torch.empty(rows, 16, device=self.device),
                    real_num_heads=8 if self.real_heads == 8 else None,
                )
            # Check the two retained arithmetic paths before timing; clone the
            # reference because caller-owned output is reused by both calls.
            baseline = [
                x.clone() for x in call_with_policy(False, self.path, *args, **kwargs)
            ]
            current = call_with_policy(True, self.path, *args, **kwargs)
            for actual, expected in zip(current, baseline):
                if self.real_heads == 8:
                    actual, expected = (
                        actual[..., :8, :] if actual.ndim == 4 else actual[:, :8]
                    ), (expected[..., :8, :] if expected.ndim == 4 else expected[:, :8])
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
            yield (*args, kwargs)


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend comparison")
@pytest.mark.parametrize("real_heads", [8, 16], ids=["h8", "h16"])
def test_sparse_decode_split_vs_serial(real_heads):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(104)
        SparseMLAMetaXBenchmark("decode", real_heads).run()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend comparison")
@pytest.mark.parametrize("real_heads", [8, 16], ids=["h8", "h16"])
def test_sparse_prefill_optimized_vs_serial(real_heads):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(104)
        SparseMLAMetaXBenchmark("prefill", real_heads).run()
