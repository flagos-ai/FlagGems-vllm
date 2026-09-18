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

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops import fp8_fp4_paged_mqa_logits

from .accuracy_utils import calc_diff
from .fp8_fp4_quant import quantize_to_mxfp4
from .test_fp8_fp4_mqa_logits import reference_fp4_mqa_logits

_vendor = flaggems_vllm.vendor_name
_skip_arch = False
_skip_ref = False

if _vendor == "mthreads":
    try:
        from deep_gemm import fp8_paged_mqa_logits as _ref_fp8_paged_mqa_logits
        from deep_gemm import get_paged_mqa_logits_metadata as _ref_get_metadata
    except ImportError:
        _skip_ref = "requires deep_gemm with fp8_mqa_logits support"

    def _ref_paged_mqa(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits,
    ):
        return _ref_fp8_paged_mqa_logits(
            q=q[0] if isinstance(q, tuple) else q,
            fused_kv_cache=kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_tables,
            schedule_meta=schedule_metadata,
            max_context_len=max_model_len,
            clean_logits=clean_logits,
        )

elif _vendor == "nvidia":
    try:
        from vllm.utils.deep_gemm import (
            fp8_fp4_paged_mqa_logits as _ref_fp8_fp4_paged_mqa_logits,
        )
        from vllm.utils.deep_gemm import (
            get_paged_mqa_logits_metadata as _ref_get_metadata,
        )
    except ImportError:
        _skip_ref = "requires vLLM with DeepGEMM and FP8 quantization support"

    _ref_paged_mqa = _ref_fp8_fp4_paged_mqa_logits

else:
    _skip_arch = f"unsupported vendor: {_vendor}"
    _skip_ref = f"unsupported vendor: {_vendor}"


device = flaggems_vllm.device

# DeepSeek-V4 model parameters
NUM_HEADS = 64
HEAD_DIM = 128
BLOCK_KV = 64  # KV cache page size
MAX_MODEL_LEN = 111 * 1024

# Test shapes: (batch_size, next_n, avg_context_len)
TEST_SHAPES = [
    (4, 1, 512),
    (4, 1, 1024),
    (4, 1, 2048),
    (8, 1, 1024),
    (8, 2, 1024),
    (16, 1, 2048),
    (32, 1, 4096),
    (64, 1, 4096),
    (128, 1, 2048),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _kv_cache_cast_to_fp8(x):
    """Cast bf16 KV cache to FP8 format matching DeepGEMM layout.

    Layout: [num_blocks, block_size, 1, head_dim + 4] where trailing 4 bytes
    store per-token float32 scale factors.
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def _make_inputs(batch_size, next_n, avg_kv, use_fp4=False):
    """Generate test inputs for FP8/FP4 paged MQA logits."""
    num_total_blocks = max(MAX_MODEL_LEN * 3 // BLOCK_KV, 1000)

    q_bf16 = torch.randn(
        (batch_size, next_n, NUM_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache_bf16 = torch.randn(
        (num_total_blocks, BLOCK_KV, 1, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, NUM_HEADS), device=device, dtype=torch.float
    )

    base_ctx = torch.randint(
        max(1, int(0.7 * avg_kv)),
        int(1.3 * avg_kv) + 1,
        (batch_size,),
        device=device,
        dtype=torch.int32,
    )
    base_ctx = base_ctx.clamp(max=MAX_MODEL_LEN)
    context_lens = base_ctx.unsqueeze(1).expand(-1, next_n).contiguous()

    if use_fp4:
        q_packed, q_scale = quantize_to_mxfp4(q_bf16)
    else:
        q_packed = q_bf16.to(torch.float8_e4m3fn)
        q_scale = None

    kv_fp8 = _kv_cache_cast_to_fp8(kv_cache_bf16)

    max_ctx = int(base_ctx.max().item())
    num_blocks_per_query = _ceil_div(max_ctx, BLOCK_KV)
    block_table = torch.zeros(
        (batch_size, num_blocks_per_query), device=device, dtype=torch.int32
    )
    block_idx_pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i in range(batch_size):
        n_blocks = _ceil_div(base_ctx[i].item(), BLOCK_KV)
        if offset + n_blocks > num_total_blocks:
            block_idx_pool = torch.randperm(
                num_total_blocks, device=device, dtype=torch.int32
            )
            offset = 0
        block_table[i, :n_blocks] = block_idx_pool[offset : offset + n_blocks]
        offset += n_blocks

    return q_packed, q_scale, kv_fp8, weights, context_lens, block_table


def _reference_fn(q_fp8, kv_fp8, weights, context_lens, block_table):
    """Reference: call platform-specific DeepGEMM kernel (FP8 only)."""
    from flaggems_vllm.utils.device_info import get_sm_count

    num_sms = get_sm_count()
    schedule_meta = _ref_get_metadata(context_lens, BLOCK_KV, num_sms)
    return _ref_paged_mqa(
        q=(q_fp8, None),
        kv_cache=kv_fp8,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_table,
        schedule_metadata=schedule_meta,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
    )


@pytest.mark.fp8_fp4_paged_mqa_logits
@pytest.mark.parametrize(
    "batch_size, next_n, avg_kv",
    TEST_SHAPES,
    ids=[f"B{b}_N{n}_L{l}" for b, n, l in TEST_SHAPES],
)
@pytest.mark.parametrize("use_fp4", [False, True])
@pytest.mark.skipif(_skip_arch, reason=_skip_arch or "")
@pytest.mark.skipif(_skip_ref, reason=_skip_ref or "")
def test_fp8_fp4_paged_mqa_logits(batch_size, next_n, avg_kv, use_fp4):
    torch.manual_seed(0)
    q_packed, q_scale, kv_fp8, weights, context_lens, block_table = _make_inputs(
        batch_size, next_n, avg_kv, use_fp4=use_fp4
    )

    # Run Triton kernel
    from flaggems_vllm.utils.device_info import get_sm_count

    num_sms = get_sm_count()
    schedule_meta = _ref_get_metadata(context_lens, BLOCK_KV, num_sms)
    q_input = (q_packed, q_scale)
    triton_out = fp8_fp4_paged_mqa_logits(
        q=q_input,
        kv_cache=kv_fp8,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_table,
        schedule_metadata=schedule_meta,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
    )

    total_rows = batch_size * next_n
    ctx_flat = context_lens.reshape(-1)[:total_rows]

    if use_fp4:
        # For FP4, verify paged output against dense float32 reference.
        # Extract flat K from the paged cache and call reference_fp4_mqa_logits.
        assert not torch.isnan(triton_out).any(), "NaN detected in Triton FP4 output"
        assert torch.isfinite(triton_out).all(), "Inf detected in Triton FP4 output"

        # Extract flat K and scales from paged cache (same as _preprocess_kv_cache)
        num_phys = kv_fp8.shape[0]
        flat_size = num_phys * BLOCK_KV
        kv_flat = kv_fp8.reshape(num_phys, BLOCK_KV * (HEAD_DIM + 4))
        kv_data = kv_flat[:, : BLOCK_KV * HEAD_DIM].reshape(flat_size, HEAD_DIM)
        kv_scales = kv_flat[:, BLOCK_KV * HEAD_DIM :].reshape(-1).view(torch.float32)

        ref_out = torch.zeros(
            total_rows, MAX_MODEL_LEN, device=device, dtype=torch.float32
        )
        for row in range(total_rows):
            ctx = ctx_flat[row].item()
            if ctx == 0:
                continue
            b_idx = row // next_n
            positions = torch.arange(ctx, device=device)
            logical_pages = positions // BLOCK_KV
            page_offsets = positions % BLOCK_KV
            phys_pages = block_table[b_idx, logical_pages]
            flat_indices = phys_pages * BLOCK_KV + page_offsets
            k_flat = kv_data[flat_indices].contiguous().view(torch.float8_e4m3fn)
            k_scale_flat = kv_scales[flat_indices]
            ref_row = reference_fp4_mqa_logits(
                q_packed.reshape(total_rows, NUM_HEADS, HEAD_DIM // 2)[row : row + 1],
                q_scale.reshape(total_rows, NUM_HEADS, 1)[row : row + 1],
                k_flat,
                k_scale_flat,
                weights[row : row + 1],
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor([ctx], dtype=torch.int32, device=device),
            )
            ref_out[row, :ctx] = ref_row.squeeze(0)

        _valid = torch.zeros_like(triton_out, dtype=torch.bool)
        for _r in range(total_rows):
            _c = ctx_flat[_r].item()
            if _c > 0:
                _valid[_r, :_c] = True
        diff = calc_diff(
            triton_out.masked_fill(~_valid, 0).float(),
            ref_out.masked_fill(~_valid, 0).float(),
        )
        assert diff < 1e-3, f"calc_diff={diff}"

    else:
        # FP8 path: compare against vLLM reference
        ref_out = _reference_fn(q_packed, kv_fp8, weights, context_lens, block_table)

        _valid = torch.zeros_like(triton_out, dtype=torch.bool)
        for _r in range(total_rows):
            _c = ctx_flat[_r].item()
            if _c > 0:
                _valid[_r, :_c] = True
        diff = calc_diff(
            triton_out.masked_fill(~_valid, 0).float(),
            ref_out.masked_fill(~_valid, 0).float(),
        )
        assert diff < 1e-3, f"calc_diff={diff}"
        assert not torch.isnan(triton_out).any(), "NaN detected in Triton output"
