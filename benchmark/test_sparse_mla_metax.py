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
                head_dim = getattr(self, "head_dim", 512)
                q = (
                    torch.randn(rows, 16, head_dim, device=self.device) * 0.1
                ).bfloat16()
                kv = (
                    torch.randn(1024, 1, head_dim, device=self.device) * 0.1
                ).bfloat16()
                indices = torch.arange(
                    topk, device=self.device, dtype=torch.int32
                ).repeat(rows, 1)
                if self.real_heads == 8:
                    q[:, 8:] = 0
                    indices = indices[:, None]
                args = (q, kv, indices, head_dim**-0.5)
                kwargs = dict(
                    attn_sink=torch.randn(16, device=self.device) * 0.1,
                    topk_length=torch.full(
                        (rows,), topk, device=self.device, dtype=torch.int32
                    ),
                    out=torch.empty(rows, 16, 512, device=self.device, dtype=dtype),
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


class NativeH8Benchmark(SparseMLAMetaXBenchmark):
    """Compare physical H8 with the optimized padded-H8 route, not serial."""

    def __init__(self, mode, include_padding, head_dim=512):
        self.head_dim = head_dim
        self.include_padding = include_padding
        super().__init__(mode, 8)
        self.torch_op = self.padded_call
        self.gems_op = self.native_call

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if self.path == "prefill":
            self.shapes.insert(0, (1, 64))

    def record_shapes(self, *args, **kwargs):
        bundle = args[0]
        q = bundle["native_kwargs"].get("q")
        if q is None:
            q = bundle["native_args"][0]
        return {
            "real_heads": 8,
            "physical_heads": 8,
            "path": self.path,
            "comparator": "optimized padded-H8 (physical H16)",
            "includes_q_padding": self.include_padding,
            "q_shape": list(q.shape),
            "topk": (
                bundle["padded_kwargs"]["indices"].shape[-1]
                if self.path == "decode"
                else bundle["padded_args"][2].shape[-1]
            ),
        }

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple[0]], {}

    def padded_call(self, bundle):
        args, kwargs = bundle["padded_args"], bundle["padded_kwargs"]
        if self.include_padding:
            q = kwargs["q"] if self.path == "decode" else args[0]
            source = bundle["native_kwargs"].get("q")
            if source is None:
                source = bundle["native_args"][0]
            # Explicitly time the two operations used to materialize H16 Q.
            q[..., :8, :].copy_(source)
            q[..., 8:, :].zero_()
        return call_with_policy(True, self.path, *args, **kwargs)

    def native_call(self, bundle):
        return call_with_policy(
            True, self.path, *bundle["native_args"], **bundle["native_kwargs"]
        )

    def get_input_iter(self, dtype):
        for packed in super().get_input_iter(dtype):
            args, kwargs = packed[:-1], packed[-1]
            native_kwargs = kwargs.copy()
            native_kwargs["real_num_heads"] = None
            q = kwargs["q"] if self.path == "decode" else args[0]
            native_q = q[..., :8, :].contiguous().clone()
            native_kwargs["out"] = torch.empty(
                (*native_q.shape[:-1], 512), device=self.device, dtype=dtype
            )
            native_kwargs["attn_sink"] = kwargs["attn_sink"][:8].clone()
            if self.path == "decode":
                native_args = ()
                native_kwargs["q"] = native_q
                native_kwargs["tile_scheduler_metadata"] = ops.FlashMLASchedMeta()
                native_kwargs["split_workspace"] = tuple(
                    torch.empty(shape, device=self.device)
                    for shape in [(64, 4, 8), (64, 4, 8), (64, 4, 8, 512)]
                )
            else:
                native_args = (native_q, *args[1:])
                native_kwargs["max_logits_out"] = torch.empty(
                    native_q.shape[:2], device=self.device
                )
                native_kwargs["lse_out"] = torch.empty(
                    native_q.shape[:2], device=self.device
                )
            bundle = dict(
                padded_args=args,
                padded_kwargs=kwargs,
                native_args=native_args,
                native_kwargs=native_kwargs,
            )
            expected = [x.clone() for x in self.padded_call(bundle)]
            actual = self.native_call(bundle)
            for i, (value, reference) in enumerate(zip(actual, expected)):
                if self.path == "decode" and i == 0:
                    reference = reference[:, :, :8]
                else:
                    reference = reference[:, :8]
                torch.testing.assert_close(value, reference, atol=2e-3, rtol=2e-2)
            yield (bundle,)


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend comparison")
@pytest.mark.parametrize("include_padding", [False, True], ids=["kernel", "q_padding"])
def test_native_h8_decode_vs_optimized_padded(include_padding):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(104)
        NativeH8Benchmark("decode", include_padding).run()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX backend comparison")
@pytest.mark.parametrize("include_padding", [False, True], ids=["kernel", "q_padding"])
@pytest.mark.parametrize("head_dim", [512, 576])
def test_native_h8_prefill_vs_optimized_padded(include_padding, head_dim):
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(104)
        NativeH8Benchmark("prefill", include_padding, head_dim).run()
