# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close, gems_assert_equal
from tests.flash_mla_with_kvcache_w8a8_utils import get_backend_adapter
from tests.test_flash_mla_sparse_fwd import (
    Flashmla_Sparse_Test_Param,
    FlashmlaSparseTestKit,
)

OP_NAME = "flash_mla_with_kvcache_sparse_w8a8_int8"
OVERRIDES = dict(flaggems_vllm.runtime.backend.get_customized_ops())
SUPPORTED = OP_NAME in OVERRIDES
pytestmark = [
    pytest.mark.flash_mla_with_kvcache_sparse_w8a8_int8,
    pytest.mark.skipif(not SUPPORTED, reason="No registered sparse INT8 KVCache MLA"),
]


def make_inputs(
    batch, queries, heads, topk, pages=32, dtype=torch.bfloat16, rope_dim=64
):
    param = Flashmla_Sparse_Test_Param(
        batch * queries, pages * 64, topk, heads, d_qk=576
    )
    query, cache, indices, _, _ = FlashmlaSparseTestKit.make_input_flashmla(
        param, seed=42
    )
    adapter = get_backend_adapter()
    q, qr, qs, native_q = adapter.quantize(
        query.view(batch, queries, heads, 576), benchmark=False
    )
    kv, kr, ks, native_kv = adapter.quantize(
        cache.view(pages, 64, 1, 576), benchmark=False
    )
    if rope_dim == 0:
        qr, kr = qr[..., :0].contiguous(), kr[..., :0].contiguous()
        native_q, native_kv = (
            native_q[..., :512].contiguous(),
            native_kv[..., :512].contiguous(),
        )
    inputs = [
        q,
        qr.to(dtype),
        kv,
        kr.to(dtype),
        qs,
        ks,
        indices.view(batch, queries, topk),
    ]
    return inputs, native_q, native_kv


def reference(inputs, *, attn_sink=None, topk_length=None, softmax_scale=None):
    q, qr, kv, kr, qs, ks, indices = inputs
    batch, queries, heads, _ = q.shape
    dims = 512 + qr.shape[-1]
    query = (torch.cat((q.float(), qr.float()), -1) * qs).view(-1, heads, dims)
    cache = (torch.cat((kv.float(), kr.float()), -1) * ks).view(-1, 1, dims)
    output, _, lse = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        batch * queries,
        cache.shape[0],
        heads,
        1,
        dims,
        indices.shape[-1],
        query,
        cache,
        indices.view(batch * queries, 1, -1),
        dims**-0.5 if softmax_scale is None else softmax_scale,
        512,
        attn_sink,
        None if topk_length is None else topk_length.flatten(),
        output_dtype=torch.float32,
    )
    return output.view(batch, queries, heads, 512), lse.view(
        batch, queries, heads
    ).transpose(1, 2)


def assert_accuracy(actual, expected):
    output, lse = actual
    ref_output, ref_lse = expected
    gems_assert_close(output, ref_output, output.dtype, atol=2e-3)
    gems_assert_equal(torch.isposinf(lse), torch.isposinf(ref_lse))
    finite = torch.isfinite(ref_lse)
    gems_assert_close(lse[finite], ref_lse[finite], torch.float32, atol=2e-4)


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("queries", [1, 2])
@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_decode(queries, heads, dtype, rope_dim):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, _, _ = make_inputs(4, queries, heads, 129, dtype=dtype, rope_dim=rope_dim)
    inputs[4].mul_(1.3)
    inputs[4][2].mul_(256)
    inputs[5].mul_(0.7)
    indices = inputs[-1]
    indices[0].fill_(-1)
    indices[1, :, ::3] = inputs[2].shape[0] * 64
    indices[2, :, 1] = indices[2, :, 0]
    lengths = torch.full((4, queries), 97, dtype=torch.int32, device=inputs[0].device)
    lengths[-1] = 0
    sink = torch.randn(heads, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    out = torch.empty((4, queries, heads, 512), dtype=dtype, device=inputs[0].device)
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_int8(
        *inputs,
        topk_length=lengths,
        attn_sink=sink,
        out=out,
        softmax_scale=0.125,
    )
    assert actual[0] is out
    assert_accuracy(
        actual,
        reference(inputs, topk_length=lengths, attn_sink=sink, softmax_scale=0.125),
    )


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("topk", [0, 1, 64, 65, 512, 2048, 8320])
def test_sparse_decode_lengths(topk, rope_dim):
    inputs, _, _ = make_inputs(2, 1, 64, topk, rope_dim=rope_dim)
    if topk > 0:
        inputs[-1][0].fill_(-1)
        inputs[-1][1, :, ::4] = 2147483647
    if topk == 8320:
        inputs[4].mul_(256)
    handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_sparse_w8a8_int8(*inputs)
    actual = handle()
    if topk == 0:
        gems_assert_equal(actual[0], torch.zeros_like(actual[0]))
        gems_assert_equal(actual[1], torch.full_like(actual[1], float("inf")))
    else:
        assert_accuracy(actual, reference(inputs))


@pytest.mark.parametrize("empty", ["batch", "heads", "cache"])
def test_sparse_decode_empty(empty):
    inputs, _, _ = make_inputs(2, 1, 64, 65)
    if empty == "batch":
        for index in (0, 1, 4, 6):
            inputs[index] = inputs[index][:0]
    elif empty == "heads":
        for index in (0, 1, 4):
            inputs[index] = inputs[index][:, :, :0]
    else:
        for index in (2, 3, 5):
            inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_int8(*inputs)
    gems_assert_equal(output, torch.zeros_like(output))
    gems_assert_equal(lse, torch.full_like(lse, float("inf")))
