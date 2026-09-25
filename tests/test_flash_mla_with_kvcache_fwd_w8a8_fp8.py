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
import math

import pytest
import torch

from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    flash_mla_with_kvcache_fwd_w8a8_fp8,
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
)

from . import conftest as cfg

FP8_MAX = 448.0
CONTENT_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 64
HEAD_DIM = CONTENT_DIM + ROPE_DIM


def quantize_ckv_per_token(
    content_and_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    content = content_and_rope[..., :CONTENT_DIM]
    rope = content_and_rope[..., CONTENT_DIM:]
    content_amax = content.float().abs().amax(dim=-1, keepdim=True)
    scale = content_amax / FP8_MAX
    scale = torch.where(content_amax == 0, torch.ones_like(scale), scale)
    return (
        (content.float() / scale).to(torch.float8_e4m3fn),
        (rope.float() / scale).to(content_and_rope.dtype),
        scale.float(),
    )


def make_dense_mla_inputs(
    batch,
    heads,
    seqlen,
    *,
    page_multiple=1,
    seed=42,
    dtype=torch.bfloat16,
    device="cuda",
    magnitude=0.1,
):
    pages_per_row = math.ceil(seqlen / (PAGE_SIZE * page_multiple)) * page_multiple
    total_pages = batch * pages_per_row
    if seed is not None:
        torch.manual_seed(seed)
    query = (
        torch.randn(batch, 1, heads, HEAD_DIM, dtype=dtype, device=device) * magnitude
    )
    cache = (
        torch.randn(total_pages, PAGE_SIZE, HEAD_DIM, dtype=dtype, device=device)
        * magnitude
    )
    q_nope, q_rope, q_scale = quantize_ckv_per_token(query)
    k_lora, k_rope, k_scale = quantize_ckv_per_token(cache)
    block_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, pages_per_row
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    return dict(
        q=query,
        blocked_k=cache,
        q_nope=q_nope,
        q_rope=q_rope,
        q_scale=q_scale,
        k_lora=k_lora,
        k_rope=k_rope,
        k_scale=k_scale,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        lengths=(seqlen,) * batch,
    )


def dense_mla_reference(inputs):
    q = inputs["q"].float()
    blocked_k = inputs["blocked_k"].float()
    block_table = inputs["block_table"]
    cache_seqlens = inputs["cache_seqlens"]
    batch, _, h_q, _ = q.shape
    out = torch.empty(batch, 1, h_q, 512, dtype=torch.float32, device=q.device)
    lse = torch.empty(batch, h_q, 1, dtype=torch.float32, device=q.device)
    softmax_scale = 576**-0.5

    for batch_idx in range(batch):
        seqlen = int(cache_seqlens[batch_idx].item())
        page_count = math.ceil(seqlen / 64)
        page_ids = block_table[batch_idx, :page_count].long()
        kv = blocked_k.index_select(0, page_ids).reshape(-1, 576)[:seqlen]
        scores = torch.matmul(q[batch_idx, 0], kv.transpose(0, 1))
        scores *= softmax_scale
        probabilities = torch.softmax(scores, dim=-1)
        out[batch_idx, 0] = torch.matmul(probabilities, kv[:, :512])
        lse[batch_idx, :, 0] = torch.logsumexp(scores, dim=-1)
    return out, lse


def assert_dense_mla_accuracy(out, lse, ref_out, ref_lse):
    out_f32 = out.float()
    rel_l2 = torch.linalg.vector_norm(out_f32 - ref_out) / torch.linalg.vector_norm(
        ref_out
    ).clamp_min(1e-12)
    cosine_distance = 1.0 - torch.nn.functional.cosine_similarity(
        out_f32.flatten(), ref_out.flatten(), dim=0
    )
    lse_max_abs = (lse.float() - ref_lse).abs().max()
    assert rel_l2.item() <= 5e-2
    assert cosine_distance.item() <= 1e-3
    assert lse_max_abs.item() <= 2e-2


pytestmark = [
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_fp8,
    pytest.mark.skipif(
        not HAS_TLE
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9,
        reason="requires Hopper and FlagTree GPU extensions",
    ),
]


CASES = [(1, 64, 128)] if cfg.QUICK_MODE else [(1, 64, 128), (2, 64, 640)]


@pytest.mark.parametrize("batch,h_q,seqlen", CASES)
def test_flash_mla_with_kvcache_fwd_w8a8_fp8_accuracy(batch, h_q, seqlen):
    inputs = make_dense_mla_inputs(batch, h_q, seqlen)
    out, lse = flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
    )
    ref_out, ref_lse = dense_mla_reference(inputs)
    assert_dense_mla_accuracy(out, lse, ref_out, ref_lse)


def test_flash_mla_with_kvcache_fwd_w8a8_fp8_prepared_outputs_are_deterministic():
    inputs = make_dense_mla_inputs(1, 64, 128)
    handle, (fresh_out, fresh_lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    out = torch.empty_like(fresh_out)
    lse = torch.empty_like(fresh_lse)
    caller_out, caller_lse = handle(out=out, lse=lse)
    expected_out = caller_out.clone()
    expected_lse = caller_lse.clone()
    replay_out, replay_lse = handle(out=out, lse=lse)

    assert torch.equal(fresh_out, expected_out)
    assert torch.equal(fresh_lse, expected_lse)
    assert torch.equal(replay_out, expected_out)
    assert torch.equal(replay_lse, expected_lse)


def test_dense_fp8_requires_compiler_support(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8"
    )
    inputs = make_dense_mla_inputs(1, 64, 128)
    monkeypatch.setattr(module, "HAS_TLE", False)
    with pytest.raises(NotImplementedError, match="FlagTree GPU extensions"):
        flash_mla_with_kvcache_fwd_w8a8_fp8(
            inputs["q_nope"],
            inputs["q_rope"],
            inputs["k_lora"],
            inputs["k_rope"],
            inputs["q_scale"],
            inputs["k_scale"],
            inputs["block_table"],
            inputs["cache_seqlens"],
            512,
        )


@pytest.mark.parametrize("batch", [1, 2])
def test_bf16_and_fp8_prepared_execution_interleave(batch):
    bf16 = importlib.import_module("flaggems_vllm.ops.flash_mla")
    inputs = make_dense_mla_inputs(batch, 64, 128)
    expected, expected_lse = dense_mla_reference(inputs)
    plan = bf16.get_flash_mla_tle_decode_plan(
        b=batch,
        s_q=1,
        h_q=64,
        h_kv=1,
        d=576,
        dv=512,
        block_size=64,
        dtype=torch.bfloat16,
        device=inputs["q"].device,
        causal=False,
    )
    bf16_before = plan.run(
        inputs["q"],
        inputs["blocked_k"].unsqueeze(2),
        inputs["block_table"],
        inputs["cache_seqlens"],
    )
    torch.testing.assert_close(bf16_before.float(), expected, atol=1e-3, rtol=1e-2)
    handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    assert_dense_mla_accuracy(output, lse, expected, expected_lse)
    saved_output, saved_lse = output.clone(), lse.clone()
    bf16_after = plan.run(
        inputs["q"],
        inputs["blocked_k"].unsqueeze(2),
        inputs["block_table"],
        update_metadata=False,
    )
    torch.testing.assert_close(bf16_after, bf16_before, atol=0, rtol=0)
    output, lse = handle()
    torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
    torch.testing.assert_close(lse, saved_lse, atol=0, rtol=0)


@pytest.mark.parametrize(
    "batch,heads,use_pdl,pretranspose",
    [
        (4, 64, False, False),
        (4, 64, True, False),
        (16, 128, False, True),
        (4, 64, True, True),
    ],
)
def test_dense_fp8_compile_time_schedules(
    batch, heads, use_pdl, pretranspose, monkeypatch
):
    module = importlib.import_module(
        "flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8"
    )
    handle_type = module.FlashMLAFp8PreparedHandle
    monkeypatch.setattr(
        handle_type, "_use_programmatic_dependent_launch", lambda self: use_pdl
    )
    monkeypatch.setattr(handle_type, "_use_pretranspose_v1", lambda self: pretranspose)
    inputs = make_dense_mla_inputs(batch, heads, 640)
    expected, expected_lse = dense_mla_reference(inputs)
    handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    assert_dense_mla_accuracy(output, lse, expected, expected_lse)
    saved_output, saved_lse = output.clone(), lse.clone()
    output, lse = handle()
    torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
    torch.testing.assert_close(lse, saved_lse, atol=0, rtol=0)


def test_dense_fp8_public_export():
    from flaggems_vllm import ops

    assert (
        ops.flash_mla_with_kvcache_fwd_w8a8_fp8 is flash_mla_with_kvcache_fwd_w8a8_fp8
    )
    assert "flash_mla_with_kvcache_fwd_w8a8_fp8" in ops.__all__
