"""Performance benchmark for triton_unified_attention."""

import pytest
import torch

from flaggems_vllm.ops.triton_unified_attention import triton_unified_attention

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


def _make_inputs(config, device="cuda"):
    dtype = torch.bfloat16
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


@pytest.mark.parametrize("config", BENCH_CONFIGS)
def test_triton_unified_attention_perf(config, benchmark):
    inputs = _make_inputs(config)

    # warmup
    triton_unified_attention(*inputs)
    torch.cuda.synchronize()

    def run():
        triton_unified_attention(*inputs)
        torch.cuda.synchronize()

    benchmark(run)
