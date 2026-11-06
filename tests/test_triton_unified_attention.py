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

"""Correctness tests for triton_unified_attention."""

import pytest
import torch

from flaggems_vllm.ops.triton_unified_attention import triton_unified_attention

pytestmark = pytest.mark.triton_unified_attention

# ---------------------------------------------------------------------------
# Reference implementation (pure torch, aligned with vLLM's ref_paged_attn)
# ---------------------------------------------------------------------------


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list,
    kv_lens: list,
    block_tables: torch.Tensor,
    scale: float,
    soft_cap: float | None = None,
) -> torch.Tensor:
    """Pure-torch reference for paged multi-head attention with GQA.

    Mirrors the vLLM reference implementation. Supports softcap but not
    sliding window (matches the supported subset of triton_unified_attention).
    """
    num_seqs = len(query_lens)
    _, block_size, num_kv_heads, head_size = key_cache.shape
    block_tables_cpu = block_tables.cpu().numpy()

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float()
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size).float()
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size).float()
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, repeats, dim=1)
            v = torch.repeat_interleave(v, repeats, dim=1)

        # attn: [num_query_heads, query_len, kv_len]
        attn = torch.einsum("qhd,khd->hqk", q, k)

        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)

        # causal mask
        context_len = kv_len - query_len
        q_positions = torch.arange(query_len, device=query.device) + context_len
        kv_positions = torch.arange(kv_len, device=query.device)
        causal_mask = kv_positions[None, :] > q_positions[:, None]
        attn.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out.to(query.dtype))
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def make_inputs(
    query_lens,
    kv_lens,
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    softcap,
    dtype,
    device,
):
    num_seqs = len(query_lens)
    total_q_tokens = sum(query_lens)
    max_seqlen_q = max(query_lens)
    max_seqlen_k = max(kv_lens)

    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    max_blocks = max((kv + block_size - 1) // block_size for kv in kv_lens)
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=device)
    phys_id = 0
    for i, kv in enumerate(kv_lens):
        nblocks = (kv + block_size - 1) // block_size
        for b in range(nblocks):
            block_table[i, b] = phys_id
            phys_id += 1
    total_blocks = phys_id

    q = torch.randn(
        total_q_tokens, num_query_heads, head_size, dtype=dtype, device=device
    )
    k = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    v = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    out = torch.empty_like(q)

    scale = head_size**-0.5
    return dict(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=float(softcap) if softcap else 0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        # for reference
        _query_lens=query_lens,
        _kv_lens=kv_lens,
        _block_table_cpu=block_table,
    )


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16, torch.float16]
SOFT_CAPS = [None, 50.0]

SEQ_LENS = [
    # (query_len, kv_len) per seq; kv_len >= query_len
    [(1, 1328), (5, 18), (129, 463)],
    [(1, 523), (1, 37), (1, 2011)],
    # pure decode
    [(1, 4095)] * 4,
    # pure prefill
    [(128, 128), (64, 64)],
]


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_triton_unified_attention(
    seq_lens,
    num_heads,
    head_size,
    block_size,
    soft_cap,
    dtype,
):
    device = "cuda"
    num_query_heads, num_kv_heads = num_heads
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]

    inputs = make_inputs(
        query_lens=query_lens,
        kv_lens=kv_lens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        softcap=soft_cap,
        dtype=dtype,
        device=device,
    )

    q, k, v, out = inputs["q"], inputs["k"], inputs["v"], inputs["out"]
    block_table = inputs["block_table"]
    scale = inputs["softmax_scale"]

    ref = ref_paged_attn(
        query=q,
        key_cache=k,
        value_cache=v,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_table,
        scale=scale,
        soft_cap=soft_cap,
    )

    triton_unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        max_seqlen_q=inputs["max_seqlen_q"],
        seqused_k=inputs["seqused_k"],
        max_seqlen_k=inputs["max_seqlen_k"],
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=inputs["softcap"],
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=1e-2)


# ---------------------------------------------------------------------------
# Unsupported-feature guard
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_unsupported_raises():
    """Each unsupported feature must raise NotImplementedError, not silently fail."""
    device = "cuda"
    inputs = make_inputs(
        query_lens=[1],
        kv_lens=[16],
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        block_size=16,
        softcap=None,
        dtype=torch.bfloat16,
        device=device,
    )
    base = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        out=inputs["out"],
        cu_seqlens_q=inputs["cu_seqlens_q"],
        max_seqlen_q=inputs["max_seqlen_q"],
        seqused_k=inputs["seqused_k"],
        max_seqlen_k=inputs["max_seqlen_k"],
        softmax_scale=inputs["softmax_scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=inputs["block_table"],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    cases = [
        ("sliding window", dict(window_size=(63, 0))),
        ("fp8 q_descale", dict(q_descale=torch.tensor(1.0, device=device))),
        ("fp8 k_descale", dict(k_descale=torch.tensor(1.0, device=device))),
        ("fp8 v_descale", dict(v_descale=torch.tensor(1.0, device=device))),
        ("alibi_slopes", dict(alibi_slopes=torch.ones(4, device=device))),
        ("output_scale", dict(output_scale=1.0)),
        ("sinks", dict(sinks=torch.zeros(4, 1, device=device))),
        ("seq_threshold_3D", dict(seq_threshold_3D=8)),
        ("use_td", dict(use_td=True)),
        ("chunk_lookback", dict(chunk_lookback=2)),
        ("per_seq causal tensor", dict(causal=torch.tensor([True], device=device))),
    ]

    for name, override in cases:
        kwargs = {**base, **override}
        with pytest.raises(NotImplementedError, match=""):
            triton_unified_attention(**kwargs)
