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

"""Benchmark for bf16_paged_mqa_logits (ported from FlagGems).

Baseline is the pure-PyTorch reference; the DeepGEMM CUDA baseline used by the
FlagGems benchmark is not available in this checkout.
"""

import pytest
import torch

from flaggems_vllm.ops import bf16_paged_mqa_logits

from . import base

# Model parameters (hardcoded to match kernel specializations)
HEAD_DIM = 128  # hardcoded in kernel specializations
BLOCK_SIZE = 64  # hardcoded KV cache page size

# Benchmark shapes: (batch_size, next_n, num_heads, context_len)
# Representing production decode workloads at various scales.
BENCH_SHAPES = [
    (1, 1, 32, 1024),
    (4, 1, 32, 1024),
    (8, 1, 32, 2048),
    (16, 1, 32, 4096),
    (4, 2, 32, 2048),
    (1, 1, 32, 4096),
    (4, 1, 64, 2048),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _pytorch_ref(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_metadata,
    max_context_len,
    clean_logits=False,
    logits_dtype=torch.float32,
):
    """Pure PyTorch reference implementation (baseline)."""
    B, next_n, H, D = q.shape
    total_tokens = B * next_n
    logits = torch.zeros(
        total_tokens, max_context_len, device=q.device, dtype=torch.float32
    )

    for b in range(B):
        for n in range(next_n):
            row = b * next_n + n
            ctx_len = int(context_lens[b, n].item())
            q_row = q[b, n]  # [H, D]

            for blk_idx in range(_ceil_div(ctx_len, BLOCK_SIZE)):
                phys_blk = int(block_table[b, blk_idx].item())
                start_pos = blk_idx * BLOCK_SIZE
                end_pos = min(start_pos + BLOCK_SIZE, ctx_len)
                num_pos = end_pos - start_pos

                k_block = kv_cache[phys_blk, :num_pos, 0, :]
                scores = torch.mm(k_block.float(), q_row.float().t())
                scores = torch.relu(scores)
                w = weights[row]
                logits_val = (scores * w.unsqueeze(0)).sum(dim=1)
                logits[row, start_pos:end_pos] = logits_val

    return logits


def _make_inputs(B, next_n, H, ctx_len, device):
    """Generate benchmark inputs."""
    torch.manual_seed(0)
    q = torch.randn(B, next_n, H, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(B * next_n, H, device=device, dtype=torch.float32).abs()
    context_lens = torch.full((B, next_n), ctx_len, device=device, dtype=torch.int32)

    num_blocks_per_seq = _ceil_div(ctx_len, BLOCK_SIZE)
    total_blocks = B * num_blocks_per_seq + 10
    kv_cache = torch.randn(
        total_blocks, BLOCK_SIZE, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )

    block_table = torch.zeros(B, num_blocks_per_seq, device=device, dtype=torch.int32)
    for b in range(B):
        for i in range(num_blocks_per_seq):
            block_table[b, i] = b * num_blocks_per_seq + i

    schedule_meta = torch.zeros(2, 2, device=device, dtype=torch.int32)
    return q, kv_cache, weights, context_lens, block_table, schedule_meta


class Bf16PagedMqaLogitsBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch_size, next_n, num_heads, context_len"

    def set_shapes(self, shape_file_path=None):
        self.shapes = BENCH_SHAPES

    def get_input_iter(self, dtype):
        for B, next_n, H, ctx_len in self.shapes:
            (
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                schedule_meta,
            ) = _make_inputs(B, next_n, H, ctx_len, self.device)

            yield (
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                ctx_len,
                {"clean_logits": False},
            )


@pytest.mark.bf16_paged_mqa_logits
def test_bf16_paged_mqa_logits():
    bench = Bf16PagedMqaLogitsBenchmark(
        op_name="bf16_paged_mqa_logits",
        torch_op=_pytorch_ref,
        gems_op=bf16_paged_mqa_logits,
        dtypes=[torch.bfloat16],
    )
    bench.run()
