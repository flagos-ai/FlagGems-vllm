# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
from importlib import import_module

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn
from tests.accuracy_utils import gems_assert_close, gems_assert_equal
from tests.test_flash_mla_sparse_fwd import (
    Flashmla_Sparse_Test_Param,
    FlashmlaSparseTestKit,
)

OP_NAME = "flash_mla_with_kvcache_sparse_w8a8_fp8"
OVERRIDES = dict(flaggems_vllm.runtime.backend.get_customized_ops())
SUPPORTED = OP_NAME in OVERRIDES
if SUPPORTED:
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_mla_with_kvcache_sparse_w8a8_fp8 import (
        prepare_flash_mla_with_kvcache_sparse_w8a8_fp8,
    )
else:
    prepare_flash_mla_with_kvcache_sparse_w8a8_fp8 = None
pytestmark = [
    pytest.mark.skipif(not SUPPORTED, reason="No registered sparse FP8 KVCache MLA"),
]


def quantize(tensor):
    scale = (
        tensor[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    )
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    content = (tensor[..., :512].float() / scale).to(torch.float8_e4m3fn)
    rope = (tensor[..., 512:].float() / scale).to(torch.bfloat16)
    native = torch.empty(tensor.shape, dtype=torch.bfloat16, device=tensor.device)
    row_elements = math.prod(tensor.shape[1:-1]) * 512
    chunk = max(1, (1 << 23) // max(1, row_elements))
    for first in range(0, tensor.shape[0], chunk):
        last = min(first + chunk, tensor.shape[0])
        values = content[first:last].float()
        if rope.shape[-1]:
            values = torch.cat((values, rope[first:last].float()), -1)
        native[first:last] = (values * scale[first:last]).to(torch.bfloat16)
    return content, rope, scale, native


def make_inputs(
    batch, queries, heads, topk, pages=32, dtype=torch.bfloat16, rope_dim=64
):
    param = Flashmla_Sparse_Test_Param(
        batch * queries, pages * 64, topk, heads, d_qk=576
    )
    query, cache, indices, _, _ = FlashmlaSparseTestKit.make_input_flashmla(
        param, seed=42
    )
    q, qr, qs, native_q = quantize(query.view(batch, queries, heads, 576))
    kv, kr, ks, native_kv = quantize(cache.view(pages, 64, 1, 576))
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


def reference(
    inputs,
    *,
    attn_sink=None,
    topk_length=None,
    softmax_scale=None,
    reference_device="cpu",
    chunk_rows=16,
):
    q, qr, kv, kr, qs, ks, indices = (tensor.to(reference_device) for tensor in inputs)
    attn_sink = None if attn_sink is None else attn_sink.to(reference_device)
    topk_length = (
        None if topk_length is None else topk_length.to(reference_device).flatten()
    )
    batch, queries, heads, _ = q.shape
    dims = 512 + qr.shape[-1]
    rows = batch * queries
    q = q.view(rows, heads, 512)
    qr = qr.view(rows, heads, qr.shape[-1])
    qs = qs.view(rows, heads, 1)
    tokens = kv.shape[0] * 64
    kv = kv.view(tokens, 1, 512)
    kr = kr.view(tokens, 1, kr.shape[-1])
    ks = ks.view(tokens, 1, 1)
    cache = torch.empty((tokens, 1, dims), dtype=torch.float32, device=reference_device)
    # Keep FP8 casts bounded: large unary datacast launches fail in muDNN.
    for first in range(0, tokens, 16384):
        last = min(first + 16384, tokens)
        values = kv[first:last].float()
        if dims > 512:
            values = torch.cat((values, kr[first:last].float()), -1)
        cache[first:last] = values * ks[first:last]
    selected = indices.view(rows, 1, -1)
    output = torch.empty(
        (batch * queries, heads, 512), dtype=torch.float32, device=reference_device
    )
    lse = torch.empty(
        (batch * queries, heads), dtype=torch.float32, device=reference_device
    )
    for begin in range(0, batch * queries, chunk_rows):
        end = min(begin + chunk_rows, batch * queries)
        query = q[begin:end].float()
        if dims > 512:
            query = torch.cat((query, qr[begin:end].float()), -1)
        query = query * qs[begin:end]
        output[begin:end], _, lse[begin:end] = (
            FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
                end - begin,
                cache.shape[0],
                heads,
                1,
                dims,
                indices.shape[-1],
                query,
                cache,
                selected[begin:end],
                dims**-0.5 if softmax_scale is None else softmax_scale,
                512,
                attn_sink,
                None if topk_length is None else topk_length[begin:end],
                output_dtype=torch.float32,
            )
        )
    return output.view(batch, queries, heads, 512), lse.view(
        batch, queries, heads
    ).transpose(1, 2)


def assert_accuracy(actual, expected, *, output_atol=4e-3):
    ref_output, ref_lse = expected
    output, lse = actual
    assert output.shape == ref_output.shape and lse.shape == ref_lse.shape
    # Compare every row while bounding casts and copies to the same small size.
    for first in range(0, output.shape[0], 16):
        last = min(first + 16, output.shape[0])
        value = output[first:last].to(ref_output.device)
        logsum = lse[first:last].to(ref_lse.device)
        reference_value = ref_output[first:last]
        reference_logsum = ref_lse[first:last]
        gems_assert_close(value, reference_value, output.dtype, atol=output_atol)
        gems_assert_equal(torch.isposinf(logsum), torch.isposinf(reference_logsum))
        finite = torch.isfinite(reference_logsum)
        gems_assert_close(
            logsum[finite], reference_logsum[finite], torch.float32, atol=2e-4
        )


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("topk", [129, 4097])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("queries", [1, 2])
@pytest.mark.parametrize("heads", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_decode(queries, heads, dtype, rope_dim, topk):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, _, _ = make_inputs(4, queries, heads, topk, dtype=dtype, rope_dim=rope_dim)
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
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_fp8(
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


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("queries", [1, 2])
@pytest.mark.parametrize("topk", [128, 256, 512, 640])
@pytest.mark.parametrize("small_batch", [False, True])
def test_sparse_decode_large_short_topk(dtype, queries, topk, small_batch):
    batch = 4 if small_batch else 128 // queries + 1
    inputs, _, _ = make_inputs(
        batch, queries, 64, topk, pages=16, dtype=dtype, rope_dim=0
    )
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [16, 64, 128])
@pytest.mark.parametrize("queries", [1, 2])
def test_sparse_decode_split_merge(dtype, heads, queries):
    inputs, _, _ = make_inputs(4, queries, heads, 640, pages=16, dtype=dtype)
    inputs[4].mul_(1.3)
    inputs[5].mul_(0.7)
    inputs[6][1, :, ::3] = -1
    lengths = (
        torch.tensor([640, 511, 97, 0], dtype=torch.int32, device=inputs[0].device)[
            :, None
        ]
        .expand(4, queries)
        .contiguous()
    )
    sink = torch.randn(heads, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_fp8(
        *inputs, topk_length=lengths, attn_sink=sink
    )
    assert_accuracy(actual, reference(inputs, topk_length=lengths, attn_sink=sink))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("topk", [0, 1, 64, 65, 512, 2048, 8320])
def test_sparse_decode_lengths(topk, rope_dim):
    inputs, _, _ = make_inputs(2, 1, 64, topk, rope_dim=rope_dim)
    if topk > 0:
        inputs[-1][0].fill_(-1)
        inputs[-1][1, :, ::4] = 2147483647
    if topk == 8320:
        inputs[4].mul_(256)
    handle, _ = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    actual = handle()
    if topk == 0:
        gems_assert_equal(actual[0], torch.zeros_like(actual[0]))
        gems_assert_equal(actual[1], torch.full_like(actual[1], float("inf")))
    else:
        assert_accuracy(actual, reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
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
    output, lse = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    gems_assert_equal(output, torch.zeros_like(output))
    gems_assert_equal(lse, torch.full_like(lse, float("inf")))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("queries", [1, 2])
def test_sparse_decode_batch_boundary(queries):
    batch = 512 // queries + 1
    inputs, _, _ = make_inputs(batch, queries, 16, 4097, pages=16, rope_dim=0)
    lengths = (
        torch.arange(batch * queries, dtype=torch.int32, device=inputs[0].device).view(
            batch, queries
        )
        % 129
    )
    sink = torch.randn(16, dtype=torch.float32, device=inputs[0].device)
    sink[0], sink[1] = float("inf"), -float("inf")
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_fp8(
        *inputs, topk_length=lengths, attn_sink=sink
    )
    assert_accuracy(actual, reference(inputs, topk_length=lengths, attn_sink=sink))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize(
    "bh,bn,warps", [(32, 64, 4), (32, 64, 8), (64, 128, 8), (64, 128, 16)]
)
def test_sparse_fp8_split_tail_configs(bh, bn, warps):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_mla_with_kvcache_sparse_w8a8_fp8 import (
        combine_sparse_fp8,
        sparse_fp8_mla,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, _, _ = make_inputs(2, 1, 64, 8320, pages=1544, rope_dim=0)
    device = inputs[0].device
    partial = torch.empty((2, 32, 64, 512), dtype=torch.float32, device=device)
    stats = torch.empty((2, 32, 64, 2), dtype=torch.float32, device=device)
    output = torch.empty((2, 1, 64, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((2, 64, 1), dtype=torch.float32, device=device)
    sparse_fp8_mla.jit_function[(2, (64 + bh - 1) // bh, 32)](
        *inputs[:6],
        inputs[6],
        None,
        partial,
        stats,
        output,
        lse,
        None,
        2,
        64,
        1,
        8320,
        1544 * 64,
        0,
        512**-0.5,
        False,
        False,
        32,
        320,
        bh,
        bn,
        num_warps=warps,
        num_stages=1,
    )
    combine_sparse_fp8.jit_function[(2, 64, 4)](
        partial,
        stats,
        None,
        output,
        lse,
        2,
        64,
        1,
        32,
        False,
        32,
        BV=128,
        num_warps=4,
        num_stages=1,
    )
    assert_accuracy((output, lse), reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("topk", [128, 257, 640, 1024, 1025, 8320])
def test_sparse_decode_mixed_pv_updates(dtype, topk):
    inputs, _, _ = make_inputs(4, 1, 64, topk, pages=16, dtype=dtype, rope_dim=0)
    handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs))
    # Reuse the prepared storage with changed FP8 content, scales and indices.
    inputs[0].copy_(inputs[0].float().flip(-1).to(torch.float8_e4m3fn))
    inputs[2].copy_((inputs[2].float() * 0.5).to(torch.float8_e4m3fn))
    inputs[4][1, 0, 0].mul_(256)
    inputs[5].mul_(0.7)
    inputs[6][0].fill_(-1)
    inputs[6][2, :, 1] = inputs[6][2, :, 0]
    inputs[6][3, :, ::4] = 2147483647
    # Multiplication by 512 would wrap these invalid indices to zero.
    inputs[6][3, :, 1::4] = 1 << 23
    inputs[6][3, :, 2::4] = -(1 << 23)
    assert_accuracy(handle(), reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pages", [16, 512])
@pytest.mark.parametrize("topk", [128, 129])
def test_sparse_decode_selected_cache_updates(dtype, pages, topk):
    pytest.importorskip("triton.experimental.tle.language")
    # Public dispatch must not hide missing dependencies of the TLE paths.
    import_module("flaggems_vllm.runtime.backend._mthreads._sparse_fp8_pipeline")
    inputs, _, _ = make_inputs(129, 1, 64, topk, pages=pages, dtype=dtype, rope_dim=0)
    handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs))
    inputs[2].copy_((inputs[2].float() * 0.5).to(torch.float8_e4m3fn))
    inputs[4][1, 0, 0].mul_(256)
    inputs[4][31:34, 0, 0].mul_(256)
    inputs[4][-1, 0, 0].mul_(256)
    inputs[5].mul_(0.7)
    inputs[6][0].fill_(-1)
    inputs[6][2, :, 1] = inputs[6][2, :, 0]
    # Both invalid indices would wrap to byte offset zero with 32-bit indexing.
    inputs[6][4, :, ::2] = 1 << 23
    inputs[6][4, :, 1::2] = -(1 << 23)
    inputs[6][3, :, ::3] = 2147483647
    assert_accuracy(handle(), reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("topk", [640, 2049])
def test_sparse_decode_pv_scale_updates(dtype, topk):
    inputs, _, _ = make_inputs(4, 1, 64, topk, pages=64, dtype=dtype, rope_dim=0)
    handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs))
    # Keep QK scores fixed while magnifying the values and PV rounding error.
    inputs[4].div_(64)
    inputs[5].mul_(64)
    assert_accuracy(handle(), reference(inputs))
    inputs[5][0].mul_(2**-16)
    inputs[6][..., 0] = 0
    assert_accuracy(handle(), reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("alternating", [False, True])
def test_sparse_decode_probability_rounding(dtype, alternating):
    inputs, _, _ = make_inputs(4, 1, 64, 2049, pages=40, dtype=dtype, rope_dim=0)
    query = torch.full(inputs[0].shape, 320.0, dtype=torch.float32, device="cpu")
    values = torch.full(inputs[2].shape, 256.0, dtype=torch.float32, device="cpu")
    values.view(-1, 512)[1::2].mul_(-1 if alternating else 1)
    inputs[0].copy_(query.to(torch.float8_e4m3fn).to(inputs[0].device))
    inputs[2].copy_(values.to(torch.float8_e4m3fn).to(inputs[2].device))
    for index, scale in ((4, 2**-15), (5, 2**-10)):
        inputs[index].copy_(
            torch.full(
                inputs[index].shape, scale, dtype=torch.float32, device="cpu"
            ).to(inputs[index].device)
        )
    actual = flaggems_vllm.flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,topk",
    [(4, 2049), (16, 640), (16, 8320), (128, 129), (128, 640), (128, 8193), (129, 128)],
)
def test_sparse_decode_exponential_guard_updates(dtype, batch, topk):
    inputs, _, _ = make_inputs(batch, 1, 64, topk, pages=40, dtype=dtype, rope_dim=0)
    query = torch.full(inputs[0].shape, 448.0, dtype=torch.float32, device="cpu")
    values = torch.full(inputs[2].shape, 448.0, dtype=torch.float32, device="cpu")
    values.view(-1, 512)[1::2].neg_()
    inputs[0].copy_(query.to(torch.float8_e4m3fn).to(inputs[0].device))
    inputs[2].copy_(values.to(torch.float8_e4m3fn).to(inputs[2].device))
    cache_scale = 2**-11
    inputs[5].fill_(cache_scale)
    score_per_query_scale = 512 * 448**2 * cache_scale / math.sqrt(512)
    inputs[4].fill_(1.99 / score_per_query_scale)
    handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs), output_atol=0.002)
    # The two scale bands put fragments on opposite sides of the guard.
    half_cache = inputs[5].numel() // 2
    indices = torch.arange(topk, dtype=torch.int32, device=inputs[0].device)
    indices.remainder_(half_cache)
    indices[topk // 2 :].add_(half_cache)
    inputs[6].copy_(indices.view(1, 1, -1).expand_as(inputs[6]))
    inputs[5].view(-1)[half_cache:].mul_(1.0625)
    assert_accuracy(handle(), reference(inputs), output_atol=0.002)
    inputs[4].fill_(2.01 / score_per_query_scale)
    assert_accuracy(handle(), reference(inputs), output_atol=0.002)


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,topk",
    [
        (1, 2049),
        (3, 2049),
        (8, 2049),
        (9, 3073),
        (16, 2049),
        (8, 513),
        (16, 640),
        (16, 8320),
        (129, 128),
        (181, 128),
    ],
)
def test_sparse_decode_query_partitions(dtype, batch, topk):
    inputs, _, _ = make_inputs(batch, 1, 64, topk, pages=40, dtype=dtype, rope_dim=0)
    # Distinct head/batch responses expose incorrect partial-slot assignments.
    positions = torch.arange(64)
    parity = torch.zeros((64, 64), dtype=torch.int32)
    for bit in range(6):
        parity ^= (positions[:, None] >> bit) & (positions[None, :] >> bit) & 1
    pattern = (1 - 2 * parity).float().repeat(1, 8) * 256
    pattern *= torch.tensor([1, 0.5, 0.25, -1]).repeat_interleave(128)
    cache = pattern.to(torch.float8_e4m3fn).view(torch.uint8).repeat(40, 1)
    query = torch.zeros(inputs[0].shape, dtype=torch.float32, device="cpu")
    for row in range(batch):
        query[row, 0, positions, (positions + row) % 64] = 448
    inputs[0].copy_(query.to(torch.float8_e4m3fn).to(inputs[0].device))
    inputs[2].copy_(
        cache.view(torch.float8_e4m3fn).view(inputs[2].shape).to(inputs[2].device)
    )
    inputs[4].fill_(2**-3)
    inputs[5].fill_(2**-9)
    selected = (
        torch.arange(topk)[None, None, :] + torch.arange(batch)[:, None, None] * 17
    ) % 2560
    inputs[6].copy_(selected.to(device=inputs[6].device, dtype=torch.int32))
    handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    assert_accuracy(actual, reference(inputs))
    inputs[4][-1].mul_(256)
    inputs[6][0, :, ::3] = -1
    assert_accuracy(handle(), reference(inputs))


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sparse_decode_metadata_format_replay(dtype):
    inputs, _, _ = make_inputs(128, 1, 64, 8320, pages=17, dtype=dtype, rope_dim=0)
    inputs[6][0].fill_(-1)
    inputs[6][1, :, ::3] = 2147483647
    inputs[6][2, :, ::3] = -(1 << 23)
    inputs[4][-1].mul_(256)
    scales = inputs[5].clone()
    handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(*inputs)
    handle()
    torch_device_fn.synchronize()
    graph = torch_device_fn.MUSAGraph()
    with torch_device_fn.graph(graph):
        handle()
    # Reuse the same workspace and captured graph across both metadata formats,
    # an all-repair scale range, and a return to the initial compressed format.
    for multiplier in (1.0, 1.125, 5.0, 1.0):
        inputs[5].copy_(scales * multiplier)
        expected = reference(inputs)
        for run in (handle, graph.replay):
            for tensor in actual:
                tensor.fill_(float("nan"))
            run()
            assert_accuracy(actual, expected, output_atol=0.002)
