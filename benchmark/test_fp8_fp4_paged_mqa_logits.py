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

from flaggems_vllm.ops import fp8_fp4_paged_mqa_logits
from tests.test_fp8_fp4_mqa_logits import quantize_to_mxfp4

from . import base

# DeepSeek-V4 model parameters
NUM_HEADS = 64
HEAD_DIM = 128
BLOCK_KV = 64  # KV cache page size
MAX_MODEL_LEN = 111 * 1024

# Benchmark shapes: (batch_size, next_n, avg_context_len)
# Representing production decode workloads at various context lengths.
BENCH_SHAPES = [
    (256, 1, 1024),
    (256, 1, 2048),
    (256, 1, 4096),
    (256, 1, 8192),
    (256, 2, 8192),
    (128, 1, 16384),
    (64, 1, 32768),
    (32, 1, 65536),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _kv_cache_cast_to_fp8(x):
    """Cast bf16 KV cache to FP8 format matching DeepGEMM layout."""
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


def _make_inputs(batch_size, next_n, avg_kv, device, use_fp4=False):
    """Generate benchmark inputs."""
    num_total_blocks = MAX_MODEL_LEN * 3 // BLOCK_KV

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

    base_ctx = torch.full((batch_size,), avg_kv, device=device, dtype=torch.int32)
    base_ctx = base_ctx.clamp(max=MAX_MODEL_LEN)
    context_lens = base_ctx.unsqueeze(1).expand(-1, next_n).contiguous()

    if use_fp4:
        q_packed, q_scale = quantize_to_mxfp4(q_bf16)
    else:
        q_packed = q_bf16.to(torch.float8_e4m3fn)
        q_scale = None

    kv_fp8 = _kv_cache_cast_to_fp8(kv_cache_bf16)

    num_blocks_per_query = _ceil_div(avg_kv, BLOCK_KV)
    block_table = torch.zeros(
        (batch_size, num_blocks_per_query), device=device, dtype=torch.int32
    )
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i in range(batch_size):
        n = num_blocks_per_query
        if offset + n > num_total_blocks:
            pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
            offset = 0
        block_table[i, :n] = pool[offset : offset + n]
        offset += n

    return q_packed, q_scale, kv_fp8, weights, context_lens, block_table


try:
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits as _vllm_fp8_fp4_paged_mqa_logits,
    )
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata as _vllm_get_metadata

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _baseline_fn(
    q_packed, q_scale, kv_fp8, weights, context_lens, block_table, schedule_meta
):
    """Baseline: vLLM DeepGEMM CUDA kernel (FP8 only)."""
    q_input = (q_packed, q_scale)
    return _vllm_fp8_fp4_paged_mqa_logits(
        q=q_input,
        kv_cache=kv_fp8,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_table,
        schedule_metadata=schedule_meta,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
    )


def _gems_fn(
    q_packed, q_scale, kv_fp8, weights, context_lens, block_table, schedule_meta
):
    q_input = (q_packed, q_scale)
    return fp8_fp4_paged_mqa_logits(
        q=q_input,
        kv_cache=kv_fp8,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_table,
        schedule_metadata=schedule_meta,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
    )


class Fp8Fp4PagedMqaLogitsBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch_size, next_n, avg_context_len, use_fp4"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (bs, nn, kv, fp4) for fp4 in (False, True) for (bs, nn, kv) in BENCH_SHAPES
        ]

    def get_input_iter(self, dtype):
        device = self.device
        for batch_size, next_n, avg_kv, use_fp4 in self.shapes:
            q_packed, q_scale, kv_fp8, weights, context_lens, block_table = (
                _make_inputs(batch_size, next_n, avg_kv, device, use_fp4=use_fp4)
            )
            num_sms = torch.cuda.get_device_properties(0).multi_processor_count
            schedule_meta = _vllm_get_metadata(context_lens, BLOCK_KV, num_sms)
            yield (
                q_packed,
                q_scale,
                kv_fp8,
                weights,
                context_lens,
                block_table,
                schedule_meta,
            )


@pytest.mark.fp8_fp4_paged_mqa_logits
@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not available")
def test_fp8_fp4_paged_mqa_logits():
    bench = Fp8Fp4PagedMqaLogitsBenchmark(
        op_name="fp8_fp4_paged_mqa_logits",
        torch_op=_baseline_fn,
        gems_op=_gems_fn,
        dtypes=[torch.float32],
    )
    bench.run()
