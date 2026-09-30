# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib

import pytest
import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE

from . import conftest as cfg

CONTENT_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
HEAD_DIM = CONTENT_DIM + ROPE_DIM
FP8_MAX = 448.0


def quantize_ckv_per_token(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    content = tensor[..., :CONTENT_DIM]
    rope = tensor[..., CONTENT_DIM:]
    content_amax = content.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.where(content_amax == 0, 1.0, content_amax / FP8_MAX)
    return (
        (content.float() / scale).to(torch.float8_e4m3fn),
        (rope.float() / scale).to(tensor.dtype),
        scale.float(),
    )


def make_sparse_fp8_inputs(
    batch: int, heads: int, topk: int, seed: int = 42, magnitude: float = 0.1
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    pages = (topk + PAGE_SIZE - 1) // PAGE_SIZE + 4
    query = (
        torch.randn(batch, 1, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        * magnitude
    )
    cache = (
        torch.randn(pages, PAGE_SIZE, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        * magnitude
    )
    q_nope, q_rope, q_scale = quantize_ckv_per_token(query)
    k_nope, k_rope, k_scale = quantize_ckv_per_token(cache)
    indices = torch.randint(
        pages * PAGE_SIZE, (batch, 1, topk), device="cuda", dtype=torch.int32
    )
    return [q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices], query, cache


def sparse_fp8_reference(
    inputs: list[torch.Tensor],
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices = inputs
    query = torch.cat((q_nope.float(), q_rope.float()), -1) * q_scale
    cache = (torch.cat((k_nope.float(), k_rope.float()), -1) * k_scale).reshape(-1, 576)
    batch, _, heads, _ = query.shape
    output = torch.zeros((batch, 1, heads, 512), device="cuda", dtype=torch.float32)
    lse = torch.full((batch, heads, 1), float("inf"), device="cuda")
    for row in range(batch):
        length = (
            indices.shape[-1] if topk_length is None else max(0, int(topk_length[row]))
        )
        selected = indices[row, 0, :length].long()
        selected = selected[(selected >= 0) & (selected < cache.shape[0])]
        if selected.numel() == 0:
            continue
        keys = cache[selected]
        scale = 576**-0.5 if softmax_scale is None else softmax_scale
        logits = query[row, 0] @ keys.T * scale
        row_lse = torch.logsumexp(logits, -1)
        value = logits.softmax(-1) @ keys[:, :512]
        if attn_sink is not None:
            value *= torch.sigmoid(row_lse - attn_sink)[:, None]
        output[row, 0] = value
        lse[row, :, 0] = row_lse
    return output, lse


def assert_sparse_fp8_accuracy(
    output: torch.Tensor,
    lse: torch.Tensor,
    expected: torch.Tensor,
    expected_lse: torch.Tensor,
) -> None:
    relative_l2 = (
        output.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative_l2.item() < 0.05, relative_l2.item()
    torch.testing.assert_close(lse, expected_lse, atol=0.025, rtol=0.002)


def pack_cuda_sparse_fp8_cache(
    k_nope: torch.Tensor, k_scale: torch.Tensor, k_rope: torch.Tensor
) -> torch.Tensor:
    """Pack the CUDA sparse layout without changing the per-token NoPE scale."""
    packed = torch.empty(
        (*k_nope.shape[:2], 1, 656), device=k_nope.device, dtype=torch.uint8
    )
    token_bytes = packed[:, :, 0]
    token_bytes[..., :512].copy_(k_nope.view(torch.uint8))
    scales = k_scale.expand(*k_scale.shape[:2], 4).contiguous()
    token_bytes[..., 512:528].copy_(scales.view(torch.uint8))
    token_bytes[..., 528:].copy_(k_rope.contiguous().view(torch.uint8))
    return packed


pytestmark = [
    pytest.mark.flash_mla_sparse_fwd_w8a8_fp8,
    pytest.mark.skipif(
        not HAS_TLE
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9,
        reason="requires Hopper and FlagTree GPU extensions",
    ),
]


CASES = (
    [(2, 64, 129), (4, 128, 512)]
    if cfg.QUICK_MODE
    else [
        (1, 64, 0),
        (2, 64, 1),
        (3, 128, 65),
        (4, 64, 129),
        (2, 128, 2048),
        (64, 128, 256),
        (128, 128, 128),
    ]
)


@pytest.mark.parametrize("batch,heads,topk", CASES)
@pytest.mark.parametrize("magnitude", [0.1, 1.0])
def test_sparse_fp8_accuracy(batch, heads, topk, magnitude):
    inputs, _, _ = make_sparse_fp8_inputs(batch, heads, topk, magnitude=magnitude)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    reference, reference_lse = sparse_fp8_reference(inputs, attn_sink=sink)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_masks_lengths_and_sink():
    inputs, _, _ = make_sparse_fp8_inputs(4, 64, 1025)
    inputs[-1][0].fill_(-1)
    inputs[-1][2, :, ::2] = inputs[2].shape[0] * 64 + 10
    inputs[-1][3, :, :64] = -1
    lengths = torch.tensor([1025, 0, 129, 1025], device="cuda", dtype=torch.int32)
    sink = torch.zeros(64, device="cuda", dtype=torch.float32)
    sink[0], sink[1] = float("inf"), -float("inf")
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, attn_sink=sink, topk_length=lengths
    )
    reference, reference_lse = sparse_fp8_reference(
        inputs, attn_sink=sink, topk_length=lengths
    )
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)
    assert output[:2].count_nonzero().item() == 0
    assert output[:, :, 0].count_nonzero().item() == 0


@pytest.mark.parametrize("batch", [4, 16])
def test_sparse_fp8_strides_and_graph_replay(batch):
    inputs, _, _ = make_sparse_fp8_inputs(batch, 128, 513)
    # Noncontiguous outer strides must not change physical token addressing.
    for index in range(6):
        tensor = inputs[index]
        storage = torch.empty(
            (tensor.shape[0] * 2,) + tensor.shape[1:], device="cuda", dtype=tensor.dtype
        )
        storage[::2].copy_(tensor)
        inputs[index] = storage[::2]
    reference, reference_lse = sparse_fp8_reference(inputs)
    flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    graph.replay()
    expected = output.clone()
    graph.replay()
    assert torch.equal(output, expected)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_empty_batch_and_cache():
    inputs, _, _ = make_sparse_fp8_inputs(0, 64, 128)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    assert output.shape == (0, 1, 64, 512) and lse.shape == (0, 64, 1)
    inputs, _, _ = make_sparse_fp8_inputs(1, 64, 128)
    for index in (2, 3, 5):
        inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    assert output.count_nonzero().item() == 0 and lse.isposinf().all().item()


def test_sparse_fp8_rejects_wrong_rope_dtype():
    inputs, _, _ = make_sparse_fp8_inputs(1, 64, 128)
    inputs[1] = inputs[1].to(torch.float8_e4m3fn)
    with pytest.raises(TypeError, match="RoPE"):
        flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)


def test_sparse_fp8_scaled_rope_and_custom_softmax():
    inputs, _, _ = make_sparse_fp8_inputs(2, 64, 129)
    inputs[1].mul_(3)
    inputs[3].mul_(2)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, softmax_scale=0.5
    )
    reference, reference_lse = sparse_fp8_reference(inputs, softmax_scale=0.5)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("topk", [1, 65, 129, 513])
def test_sparse_fp8_staged_masks_and_lengths(topk):
    inputs, _, _ = make_sparse_fp8_inputs(16, 64, topk)
    inputs[-1][0].fill_(-1)
    inputs[-1][3, :, ::2] = inputs[2].shape[0] * 64
    lengths = torch.arange(16, device="cuda", dtype=torch.int32) * topk // 15
    sink = torch.randn(64, device="cuda", dtype=torch.float32)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, attn_sink=sink, topk_length=lengths
    )
    reference, reference_lse = sparse_fp8_reference(
        inputs, attn_sink=sink, topk_length=lengths
    )
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_cuda_bf16_reference():
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    inputs, query, cache = make_sparse_fp8_inputs(4, 128, 512)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, _, reference_lse = flash_mla_sparse_fwd(
        query[:, 0],
        cache.reshape(-1, 1, 576),
        inputs[-1],
        576**-0.5,
        512,
    )
    assert_sparse_fp8_accuracy(
        output, lse, reference[:, None], reference_lse[:, :, None]
    )


@pytest.mark.parametrize(
    "batch,heads,topk", [(1, 64, 129), (4, 128, 512), (16, 64, 1025)]
)
@pytest.mark.parametrize("magnitude", [0.1, 1.0])
def test_sparse_fp8_cuda_fp8_cache_reference(batch, heads, topk, magnitude):
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata

    inputs, query, cache = make_sparse_fp8_inputs(
        batch, heads, topk, magnitude=magnitude
    )
    packed = pack_cuda_sparse_fp8_cache(inputs[2], inputs[5], cache[..., 512:])
    # vLLM's Hopper decoder requires a multiple-of-64 index storage width.
    cuda_indices = torch.full(
        (batch, 1, (topk + 63) // 64 * 64), -1, device="cuda", dtype=torch.int32
    )
    cuda_indices[..., :topk].copy_(inputs[-1])
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    metadata, _ = get_mla_metadata()
    reference, reference_lse = flash_mla_with_kvcache(
        query,
        packed,
        None,
        None,
        512,
        metadata,
        softmax_scale=576**-0.5,
        is_fp8_kvcache=True,
        indices=cuda_indices,
        attn_sink=sink,
    )
    # The oracle uses the shared quantized NoPE and the CUDA path's BF16 Q/RoPE.
    oracle_inputs = [
        query[..., :512],
        query[..., 512:],
        inputs[2].float() * inputs[5],
        cache[..., 512:],
        torch.ones_like(inputs[4]),
        torch.ones_like(inputs[5]),
        inputs[-1],
    ]
    expected, expected_lse = sparse_fp8_reference(oracle_inputs, attn_sink=sink)
    assert_sparse_fp8_accuracy(reference, reference_lse, expected, expected_lse)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize(
    "batch,heads,topk",
    [(1, 64, 512), (1, 64, 2048), (4, 128, 1025), (16, 64, 4097), (1, 64, 8193)],
)
@pytest.mark.parametrize("magnitude", [0.1, 1.0])
def test_sparse_fp8_split_accuracy(batch, heads, topk, magnitude):
    inputs, _, _ = make_sparse_fp8_inputs(
        batch, heads, topk, seed=123, magnitude=magnitude
    )
    sink = torch.randn(heads, device="cuda")
    sink[0], sink[1] = float("inf"), -float("inf")
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    reference, reference_lse = sparse_fp8_reference(inputs, attn_sink=sink)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("length", [0, 1, 63, 65, 257, 1025])
def test_sparse_fp8_split_empty_partitions_and_replay(length):
    inputs, _, _ = make_sparse_fp8_inputs(2, 64, 1025, seed=123)
    inputs[-1][1].fill_(-1)
    lengths = torch.full((2,), length, device="cuda", dtype=torch.int32)
    reference, reference_lse = sparse_fp8_reference(inputs, topk_length=lengths)
    flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, topk_length=lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
            *inputs, topk_length=lengths
        )
    graph.replay()
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)
    expected = output.clone()
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("page", [0, 4, 8, 15])
def test_sparse_fp8_split_repairs_any_partition(page):
    inputs, _, _ = make_sparse_fp8_inputs(4, 64, 1024, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(1024, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][page].fill_(2.0**40)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = sparse_fp8_reference(inputs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_split_repairs_variable_lengths():
    inputs, _, _ = make_sparse_fp8_inputs(4, 64, 1025, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(1025, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][4].fill_(2.0**40)
    lengths = torch.tensor([0, 1, 513, 999], device="cuda", dtype=torch.int32)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        *inputs, topk_length=lengths
    )
    reference, reference_lse = sparse_fp8_reference(inputs, topk_length=lengths)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,heads,topk", [(2, 128, 1025), (8, 64, 4096)])
def test_sparse_fp8_unaligned_padded_strides(batch, heads, topk):
    inputs, _, _ = make_sparse_fp8_inputs(batch, heads, topk, seed=123)
    for index in (0, 1, 2, 3):
        tensor = inputs[index]
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] + 1),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        inputs[index] = storage[..., 1:]
        inputs[index].copy_(tensor)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = sparse_fp8_reference(inputs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,topk", [(1, 512), (8, 4096)])
def test_sparse_fp8_subnormal_probability_scale(batch, topk):
    inputs, _, _ = make_sparse_fp8_inputs(batch, 64, topk, seed=123)
    inputs[0].view(torch.uint8).zero_()
    inputs[1].zero_()
    inputs[2].fill_(128.0)
    inputs[3].zero_()
    inputs[4].fill_(1.0)
    inputs[5].fill_(2.0**-122)
    output, _ = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    # Normalize before comparison so the tiny output cannot pass via an absolute tolerance.
    torch.testing.assert_close(
        output.float() * (2.0**115),
        torch.ones_like(output, dtype=torch.float32),
        atol=0,
        rtol=1 / 128,
    )


@pytest.mark.parametrize("batch,topk", [(4, 1024), (8, 4096)])
def test_sparse_fp8_small_path_empty_cache(batch, topk):
    inputs, _, _ = make_sparse_fp8_inputs(batch, 64, topk, seed=123)
    for index in (2, 3, 5):
        inputs[index] = inputs[index][:0]
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    assert output.count_nonzero().item() == 0
    assert lse.isposinf().all().item()


@pytest.mark.parametrize("head", [16, 63, 127])
def test_sparse_fp8_exact_qk_late_heads(head):
    inputs, _, _ = make_sparse_fp8_inputs(8, 128, 1024, seed=123, magnitude=1.0)
    inputs[4][:, :, head].mul_(2.0**20)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = sparse_fp8_reference(inputs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,topk", [(1, 512), (8, 4096)])
def test_sparse_fp8_graph_switches_to_precise_qk(batch, topk):
    inputs, _, _ = make_sparse_fp8_inputs(batch, 64, topk, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(topk, device="cuda", dtype=torch.int32)[None, None])
    flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    inputs[5][1].fill_(2.0**40)
    graph.replay()
    reference, reference_lse = sparse_fp8_reference(inputs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize(
    "block_k,follower_regs", [(64, 160), (64, 168), (128, 224), (128, 232)]
)
def test_sparse_fp8_compact_config_boundaries(block_k, follower_regs, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.flash_mla_sparse_fwd_w8a8_fp8")
    kernel = module.sparse_fp8_compact
    config = triton.Config(
        {"BLOCK_K": block_k, "FOLLOWER_REGS": follower_regs},
        num_warps=4,
        num_stages=1,
    )
    monkeypatch.setattr(kernel.fn, "configs", [config])
    for cache in kernel.kernel_cache:
        cache.clear()
    try:
        inputs, _, _ = make_sparse_fp8_inputs(8, 128, 1025, seed=123, magnitude=1.0)
        inputs[-1][0].fill_(-1)
        lengths = torch.tensor(
            [0, 1, 63, 64, 127, 128, 129, 1025], device="cuda", dtype=torch.int32
        )
        sink = torch.randn(128, device="cuda")
        sink[0], sink[1] = float("inf"), -float("inf")
        kwargs = dict(attn_sink=sink, topk_length=lengths)
        flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, **kwargs)
        for replay in range(2):
            if replay:
                lengths.copy_(lengths.flip(0))
            graph.replay()
            reference, reference_lse = sparse_fp8_reference(inputs, **kwargs)
            assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)
    finally:
        # Libentry caches launches, so do not retain a forced configuration for later tests.
        for cache in kernel.kernel_cache:
            cache.clear()


@pytest.mark.parametrize("start,stop", [(0, 16), (8, 12), (9, 10)])
def test_sparse_fp8_large_scale_winner(start, stop):
    inputs, _, _ = make_sparse_fp8_inputs(16, 64, 192, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(192, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][1].fill_(2.0**40)
    lengths = torch.arange(16, device="cuda", dtype=torch.int32) * 192 // 15
    sink = torch.randn(64, device="cuda")
    inputs = [
        tensor[start:stop] if index in (0, 1, 4, 6) else tensor
        for index, tensor in enumerate(inputs)
    ]
    kwargs = dict(topk_length=lengths[start:stop], attn_sink=sink)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, **kwargs)
    reference, reference_lse = sparse_fp8_reference(inputs, **kwargs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_small_block_scale_keeps_accumulator_finite():
    inputs, _, _ = make_sparse_fp8_inputs(16, 64, 192, seed=123, magnitude=1.0)
    inputs[-1].copy_(torch.arange(192, device="cuda", dtype=torch.int32)[None, None])
    inputs[5].fill_(1.0)
    sink = torch.randn(64, device="cuda")
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)
    reference, reference_lse = sparse_fp8_reference(inputs, attn_sink=sink)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_preserves_small_contribution_after_cancellation():
    inputs, _, _ = make_sparse_fp8_inputs(16, 64, 128, seed=123, magnitude=1.0)
    inputs[0].view(torch.uint8).zero_()
    inputs[1].zero_()
    inputs[1][..., 0] = 1.0
    inputs[2].fill_(1.0)
    inputs[2][0, 32:].fill_(-1.0)
    inputs[3].zero_()
    inputs[3][1, :, 0] = -864.0
    inputs[4].fill_(1.0)
    inputs[5].fill_(1.0)
    inputs[-1].copy_(torch.arange(128, device="cuda", dtype=torch.int32)[None, None])
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = sparse_fp8_reference(inputs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("magnitude", [0.1, 1.0, 3.0])
def test_sparse_fp8_single_request_mixed_scales(magnitude):
    inputs, _, _ = make_sparse_fp8_inputs(1, 64, 192, seed=123, magnitude=magnitude)
    inputs[-1].copy_(torch.arange(192, device="cuda", dtype=torch.int32)[None, None])
    inputs[5][0, ::2].fill_(2.0**-40)
    inputs[5][0, 1::2].fill_(2.0**40)
    length = torch.full((1,), 192, device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda")
    kwargs = dict(topk_length=length, attn_sink=sink)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, **kwargs)
    reference, reference_lse = sparse_fp8_reference(inputs, **kwargs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


@pytest.mark.parametrize("batch,heads", [(1, 64), (8, 128), (32, 64)])
@pytest.mark.parametrize("topk", [0, 1, 63, 64, 65, 127, 128, 129])
def test_sparse_fp8_unified_small_shapes(batch, heads, topk):
    inputs, _, _ = make_sparse_fp8_inputs(batch, heads, topk, seed=123, magnitude=1.0)
    output, lse = flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)
    reference, reference_lse = sparse_fp8_reference(inputs)
    assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)


def test_sparse_fp8_requires_compiler_support(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.flash_mla_sparse_fwd_w8a8_fp8")
    inputs, _, _ = make_sparse_fp8_inputs(1, 64, 1)
    monkeypatch.setattr(module, "HAS_TLE", False)
    with pytest.raises(NotImplementedError, match="FlagTree GPU extensions"):
        flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)


def test_sparse_fp8_public_export():
    from flaggems_vllm import ops
    from flaggems_vllm.ops.flash_mla_sparse_fwd_w8a8_fp8 import (
        flash_mla_sparse_fwd_w8a8_fp8,
    )

    assert flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8 is flash_mla_sparse_fwd_w8a8_fp8
    assert "flash_mla_sparse_fwd_w8a8_fp8" in ops.__all__
