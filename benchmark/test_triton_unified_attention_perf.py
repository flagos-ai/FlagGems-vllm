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

"""Performance benchmark for triton_unified_attention.

Follows the repo benchmark convention (``base.Benchmark`` with a torch
baseline and a gems op) so it participates in ``--level`` / ``--mode`` /
``--record`` and reports ``SpeedUp = latency_torch / latency_gems``.

The baseline is a pure-torch paged-attention reference (no vLLM op), matching
the correctness test's ``ref_paged_attn``.
"""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.triton_unified_attention import triton_unified_attention

from . import base

vendor_name = flaggems_vllm.vendor_name


def _ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    soft_cap,
):
    """Pure-torch paged attention reference (causal + softcap + GQA)."""
    num_seqs = len(query_lens)
    _, block_size, num_kv_heads, head_size = key_cache.shape
    block_tables_cpu = block_tables.cpu().numpy()

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float() * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size).float()[:kv_len]
        v = (
            value_cache[block_indices]
            .view(-1, num_kv_heads, head_size)
            .float()[:kv_len]
        )

        if q.shape[1] != k.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, repeats, dim=1)
            v = torch.repeat_interleave(v, repeats, dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k)
        if soft_cap and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)

        context_len = kv_len - query_len
        q_pos = torch.arange(query_len, device=query.device) + context_len
        kv_pos = torch.arange(kv_len, device=query.device)
        attn.masked_fill_(
            (kv_pos[None, :] > q_pos[:, None]).unsqueeze(0), float("-inf")
        )

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v).to(query.dtype))
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# Unified positional signature shared by torch_op and gems_op so the
# base.Benchmark harness can call both with the same argument tuple.
def _torch_op(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    scale,
    block_table,
    softcap,
    query_lens,
    kv_lens,
):
    return _ref_paged_attn(q, k, v, query_lens, kv_lens, block_table, scale, softcap)


def _gems_op(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    scale,
    block_table,
    softcap,
    query_lens,
    kv_lens,
):
    return triton_unified_attention(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        scale,
        True,
        (-1, -1),
        block_table,
        softcap,
        None,
        None,
        None,
    )


class TritonUnifiedAttentionBenchmark(base.Benchmark):
    """Benchmark for triton_unified_attention (paged, causal, GQA)."""

    # (query_lens, kv_lens, num_query_heads, num_kv_heads, head_size,
    #  block_size, softcap)
    CONFIGS = [
        # Decode: B=1, KV=1024, GQA 32:8
        ([1], [1024], 32, 8, 128, 16, 0.0),
        # Decode: B=8, KV=4096, GQA 32:8
        ([1] * 8, [4096] * 8, 32, 8, 128, 16, 0.0),
        # Decode: B=64, KV=4096, GQA 32:8
        ([1] * 64, [4096] * 64, 32, 8, 128, 16, 0.0),
        # Prefill: B=4, Q=128, GQA 32:8
        ([128] * 4, [128] * 4, 32, 8, 128, 16, 0.0),
        # Prefill: B=1, Q=512, KV=4096, GQA 32:8
        ([512], [4096], 32, 8, 128, 16, 0.0),
        # Decode with softcap
        ([1] * 8, [2048] * 8, 32, 8, 128, 16, 30.0),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.CONFIGS

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield self._build_inputs(config, dtype, self.device)

    def _build_inputs(self, config, dtype, device):
        (
            query_lens,
            kv_lens,
            num_query_heads,
            num_kv_heads,
            head_size,
            block_size,
            softcap,
        ) = config
        num_seqs = len(query_lens)
        total_q = sum(query_lens)
        max_q = max(query_lens)
        max_k = max(kv_lens)

        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(query_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=device,
        )
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

        max_blocks = max((kv + block_size - 1) // block_size for kv in kv_lens)
        block_table = torch.zeros(
            (num_seqs, max_blocks), dtype=torch.int32, device=device
        )
        phys = 0
        for i, kv in enumerate(kv_lens):
            for b in range((kv + block_size - 1) // block_size):
                block_table[i, b] = phys
                phys += 1

        q = torch.randn(total_q, num_query_heads, head_size, dtype=dtype, device=device)
        k = torch.randn(
            phys, block_size, num_kv_heads, head_size, dtype=dtype, device=device
        )
        v = torch.randn(
            phys, block_size, num_kv_heads, head_size, dtype=dtype, device=device
        )
        out = torch.empty_like(q)
        scale = head_size**-0.5

        return (
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            max_q,
            seqused_k,
            max_k,
            scale,
            block_table,
            float(softcap),
            query_lens,
            kv_lens,
        )


@pytest.mark.skipif(vendor_name == "cambricon", reason="tl.dot layout unsupported")
@pytest.mark.triton_unified_attention
def test_triton_unified_attention_perf():
    bench = TritonUnifiedAttentionBenchmark(
        op_name="triton_unified_attention",
        torch_op=_torch_op,
        gems_op=_gems_op,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.run()
