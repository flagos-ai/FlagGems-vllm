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

Uses the repository's own benchmark framework (benchmark/base.Benchmark,
--level/--warmup/--iter/--record options) instead of the external
pytest-benchmark plugin, so the file runs in any environment that can run
the other benchmark/ files. The five BENCH_CONFIGS below keep the exact
decode/prefill cases of the original pytest-benchmark parametrization.
"""

import pytest
import torch

import flaggems_vllm

from . import base

BENCH_CONFIGS = [
    # Decode: B=1, KV=1024, GQA 32:8, head_size=128
    {
        "num_seqs": 1,
        "query_lens": [1],
        "context_lens": [1023],
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # Decode: B=8, KV=4096, GQA 32:8
    {
        "num_seqs": 8,
        "query_lens": [1] * 8,
        "context_lens": [4095] * 8,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # Decode: B=64, KV=4096, GQA 32:8
    {
        "num_seqs": 64,
        "query_lens": [1] * 64,
        "context_lens": [4095] * 64,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # Prefill: B=4, Q=128, GQA 32:8
    {
        "num_seqs": 4,
        "query_lens": [128] * 4,
        "context_lens": [0] * 4,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # Prefill: B=1, Q=512, KV=3584, GQA 32:8
    {
        "num_seqs": 1,
        "query_lens": [512],
        "context_lens": [3584],
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
]


def _make_inputs(config, dtype, device):
    """Create inputs matching the triton_unified_attention signature."""
    num_seqs = config["num_seqs"]
    query_lens = config["query_lens"]
    context_lens = config["context_lens"]
    num_query_heads = config["num_query_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_size = config["head_size"]
    block_size = config["block_size"]

    total_q_tokens = sum(query_lens)
    max_seqlen_q = max(query_lens)
    seqused_k_list = [c + ql for c, ql in zip(context_lens, query_lens)]
    max_seqlen_k = max(seqused_k_list)

    cu_seqlens_q_list = [0]
    for ql in query_lens:
        cu_seqlens_q_list.append(cu_seqlens_q_list[-1] + ql)
    cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, device=device)
    seqused_k = torch.tensor(seqused_k_list, dtype=torch.int32, device=device)

    max_blocks_per_seq = max(
        (sl + block_size - 1) // block_size for sl in seqused_k_list
    )
    block_table = torch.zeros(
        (num_seqs, max_blocks_per_seq), dtype=torch.int32, device=device
    )
    physical_block_id = 0
    for i, sl in enumerate(seqused_k_list):
        num_blocks_for_seq = (sl + block_size - 1) // block_size
        for b in range(num_blocks_for_seq):
            block_table[i, b] = physical_block_id
            physical_block_id += 1
    total_physical_blocks = physical_block_id

    k_cache = torch.randn(
        total_physical_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn(
        total_physical_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    q = torch.randn(
        total_q_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    out = torch.empty(
        total_q_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )

    softmax_scale = 1.0 / (head_size**0.5)
    return (
        q,
        k_cache,
        v_cache,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        config["causal"],
        (-1, -1),
        block_table,
        config["softcap"],
        None,
        None,
        None,
    )


class TritonUnifiedAttentionBenchmark(base.Benchmark):
    """Benchmark for triton_unified_attention (vLLM paged attention)."""

    def __init__(self, config, **kwargs):
        self.config = config
        super().__init__(**kwargs)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [self.config]
        self.shape_desc = "num_seqs, query_len, kv_len, Hq, Hkv, head_size"

    def get_input_iter(self, dtype):
        yield _make_inputs(self.config, dtype, self.device)


@pytest.mark.triton_unified_attention
@pytest.mark.parametrize("config", BENCH_CONFIGS)
def test_triton_unified_attention_perf(config):
    bench = TritonUnifiedAttentionBenchmark(
        config,
        op_name="triton_unified_attention",
        torch_op=None,
        dtypes=[torch.bfloat16],
        gems_op=flaggems_vllm.triton_unified_attention,
    )
    bench.run()
