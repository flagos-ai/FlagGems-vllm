"""Correctness tests for triton_unified_attention."""

import pytest
import torch

from flaggems_vllm.ops.triton_unified_attention import triton_unified_attention

CONFIGS = [
    # single-seq decode (query_len=1)
    {
        "num_seqs": 1,
        "query_lens": [1],
        "context_lens": [15],
        "num_query_heads": 8,
        "num_kv_heads": 8,
        "head_size": 64,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # single-seq prefill
    {
        "num_seqs": 1,
        "query_lens": [32],
        "context_lens": [0],
        "num_query_heads": 8,
        "num_kv_heads": 8,
        "head_size": 64,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # GQA: 32 query heads / 4 kv heads
    {
        "num_seqs": 1,
        "query_lens": [16],
        "context_lens": [48],
        "num_query_heads": 32,
        "num_kv_heads": 4,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # multi-seq batch decode
    {
        "num_seqs": 4,
        "query_lens": [1, 1, 1, 1],
        "context_lens": [63, 31, 127, 7],
        "num_query_heads": 16,
        "num_kv_heads": 4,
        "head_size": 64,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # multi-seq mixed prefill + decode
    {
        "num_seqs": 3,
        "query_lens": [8, 1, 4],
        "context_lens": [0, 64, 32],
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_size": 128,
        "block_size": 16,
        "causal": True,
        "softcap": 0.0,
    },
    # softcap enabled
    {
        "num_seqs": 2,
        "query_lens": [1, 8],
        "context_lens": [50, 0],
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_size": 64,
        "block_size": 16,
        "causal": True,
        "softcap": 30.0,
    },
    # realistic decode: B=8, long KV=4096, GQA 32:8
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
]

DTYPES = [torch.bfloat16, torch.float16]


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
    window_size = (-1, -1)
    causal = config["causal"]
    softcap = config["softcap"]

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
        causal,
        window_size,
        block_table,
        softcap,
        None,
        None,
        None,
    )


def _reference_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
):
    """Pure-torch reference for paged attention."""
    num_seqs = len(seqused_k)
    block_size = k.shape[1]
    num_kv_heads = k.shape[2]
    head_size = q.shape[2]
    num_query_heads = q.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads

    k_flat = k.reshape(-1, num_kv_heads, head_size)
    v_flat = v.reshape(-1, num_kv_heads, head_size)

    for i in range(num_seqs):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        query_len = q_end - q_start
        seq_len = seqused_k[i].item()
        context_len = seq_len - query_len

        token_positions = torch.arange(seq_len, device=q.device)
        logical_blocks = token_positions // block_size
        offsets_in_block = token_positions % block_size
        physical_blocks = block_table[i, logical_blocks].long()
        flat_indices = physical_blocks * block_size + offsets_in_block

        k_seq = k_flat[flat_indices].to(torch.float32)
        v_seq = v_flat[flat_indices].to(torch.float32)

        k_expanded = k_seq.repeat_interleave(num_queries_per_kv, dim=1)
        v_expanded = v_seq.repeat_interleave(num_queries_per_kv, dim=1)

        q_seq = q[q_start:q_end].to(torch.float32)

        q_t = q_seq.permute(1, 0, 2)
        k_t = k_expanded.permute(1, 2, 0)
        v_t = v_expanded.permute(1, 0, 2)

        scores = torch.bmm(q_t, k_t) * softmax_scale

        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)

        if causal:
            q_positions = torch.arange(query_len, device=q.device) + context_len
            kv_positions = torch.arange(seq_len, device=q.device)
            causal_mask = kv_positions[None, :] <= q_positions[:, None]
            scores.masked_fill_(~causal_mask.unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.bmm(attn_weights, v_t)

        out[q_start:q_end] = attn_out.permute(1, 0, 2).to(out.dtype)

    return out


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_unified_attention(config, dtype):
    device = "cuda"
    inputs = _make_inputs(config, dtype, device)

    ref_out_tensor = inputs[3].clone()
    ref_inputs = list(inputs)
    ref_inputs[3] = ref_out_tensor
    _reference_attention(*ref_inputs)

    test_out = triton_unified_attention(*inputs)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(test_out, ref_out_tensor, atol=atol, rtol=1e-2)
