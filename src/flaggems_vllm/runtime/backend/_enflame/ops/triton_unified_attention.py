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

"""GCU-specialized Triton paged multi-head attention kernel for LLM inference.

Inherits the device-agnostic structure (GQA batching, 3D grid with direct
indexing, K loaded transposed, separate decode/prefill tiles, lean kernel,
causal early exit) and adds GCU-specific optimizations, all measured on
GCU400 (48 SPC, triton-gcu 3.6.0, bf16, GQA 32:8, head 128, page 16):

1. uint16 view channel switch for K/V: bf16/fp16 paged-gather tiles feeding
   tl.dot lower through a slow dot-consumer path on the GCU backend
   (baseline 2.8-340 ms across core shapes, effective 1.4-6 GB/s). Passing
   the contiguous KV cache through a same-width ``torch.uint16`` view and
   bit-casting inside the kernel routes the loads through the integer path
   with bit-identical results (10.9x/24.7x on B64-decode/Q512-prefill).
   Non-contiguous caches or non-2-byte dtypes keep the direct-load arm.
2. num_warps=1 for the small dot tiles ([16,128]x[128,64] decode,
   [128,128]x[128,64] prefill): -31% (B1 decode) to -44% (prefill) on top
   of the channel switch.
3. Prefill BLOCK_M raised 64 -> 128 (32 query positions per CTA at GQA 8:1):
   -42/-44% on the prefill shapes.
4. V tile issued before the Q@K dot so the V DMA overlaps score compute
   (-6/-8% prefill, neutral decode).

Constraint notes: uint16-view kernels must keep TILE_KV <= 128 and
num_warps in {1, 4} (larger tiles / num_warps=2 hit an ld.lld relocation
failure, see compiler_issues/ld_lld_relocation.md).
"""

import sys

import torch  # noqa: F401
import triton
import triton.language as tl

from flaggems_vllm.ops.triton_unified_attention import (
    triton_unified_attention as _general_triton_unified_attention,
)


@triton.jit
def _paged_attn_fwd(
    # Pointers
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    # Block table
    block_table_ptr,
    # Sequence info
    cu_seqlens_q_ptr,
    seqused_k_ptr,
    # Dimensions
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    head_size: tl.constexpr,
    kv_block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
    # Scalars
    softmax_scale,
    # Constexpr controls
    USE_SOFTCAP: tl.constexpr,
    softcap,
    causal: tl.constexpr,
    # Strides for Q: [total_tokens, num_query_heads, head_size]
    stride_qt,
    stride_qh,
    stride_qd,
    # Strides for K: [num_blocks, block_size, num_kv_heads, head_size]
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    # Strides for V: [num_blocks, block_size, num_kv_heads, head_size]
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    # Strides for Out: [total_tokens, num_query_heads, head_size]
    stride_ot,
    stride_oh,
    stride_od,
    # Stride for block_table: [num_seqs, max_blocks_per_seq]
    stride_bt_seq,
    stride_bt_blk,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_Q_POS: tl.constexpr,
    TILE_KV: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    # GCU: K/V arrive as a uint16 view of the bf16/fp16 cache
    V_U16: tl.constexpr,
):
    """GQA-batched paged attention: one CTA covers query heads sharing one KV head."""
    # Program IDs
    pid_q_block = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_seq = tl.program_id(2)

    # Load sequence boundaries
    q_start = tl.load(cu_seqlens_q_ptr + pid_seq)
    q_end = tl.load(cu_seqlens_q_ptr + pid_seq + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seqused_k_ptr + pid_seq)
    context_len = seq_len - query_len

    # Early exit if this Q block is out of range
    q_block_start_pos = pid_q_block * BLOCK_Q_POS
    if q_block_start_pos >= query_len:
        return

    # Build BLOCK_M-dimensional index packing q_positions and heads together
    offs_m = tl.arange(0, BLOCK_M)
    query_pos_local = offs_m // num_queries_per_kv
    head_in_group = offs_m % num_queries_per_kv

    query_pos = q_block_start_pos + query_pos_local
    query_head_idx = pid_kv_head * num_queries_per_kv + head_in_group

    q_mask = (query_pos < query_len) & (query_head_idx < num_query_heads)

    # Load Q: [BLOCK_M, HEAD_SIZE_PADDED]
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    d_mask = offs_d < head_size

    q_ptrs = (
        Q_ptr
        + (q_start + query_pos)[:, None] * stride_qt
        + query_head_idx[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    Q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    q_dtype = Q_ptr.dtype.element_ty

    # Initialize online softmax accumulators
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # Compute loop bounds with causal early exit
    if causal:
        max_q_abs = context_len + tl.minimum(
            q_block_start_pos + BLOCK_Q_POS - 1, query_len - 1
        )
        kv_upper = max_q_abs + 1
        num_kv_tiles = (kv_upper + TILE_KV - 1) // TILE_KV
    else:
        num_kv_tiles = (seq_len + TILE_KV - 1) // TILE_KV

    query_abs_pos = context_len + query_pos
    offs_t = tl.arange(0, TILE_KV)

    # Iterate over KV tiles
    for tile_idx in range(num_kv_tiles):
        kv_start = tile_idx * TILE_KV
        kv_indices = kv_start + offs_t
        tile_mask = kv_indices < seq_len

        # Look up physical blocks
        logical_blocks = kv_indices // kv_block_size
        slot_offsets = kv_indices % kv_block_size

        bt_ptrs = (
            block_table_ptr + pid_seq * stride_bt_seq + logical_blocks * stride_bt_blk
        )
        phys_blocks = tl.load(bt_ptrs, mask=tile_mask, other=0)

        # Load K transposed: [HEAD_SIZE_PADDED, TILE_KV] for direct Q @ K
        k_ptrs = (
            K_ptr
            + phys_blocks[None, :] * stride_kb
            + slot_offsets[None, :] * stride_ks
            + pid_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        if V_U16:
            K = tl.load(k_ptrs, mask=d_mask[:, None] & tile_mask[None, :], other=0).to(
                q_dtype, bitcast=True
            )
        else:
            K = tl.load(k_ptrs, mask=d_mask[:, None] & tile_mask[None, :], other=0.0)

        # Issue the V load before the Q@K dot so its DMA overlaps score compute
        v_ptrs = (
            V_ptr
            + phys_blocks[:, None] * stride_vb
            + slot_offsets[:, None] * stride_vs
            + pid_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        if V_U16:
            V = tl.load(v_ptrs, mask=tile_mask[:, None] & d_mask[None, :], other=0).to(
                q_dtype, bitcast=True
            )
        else:
            V = tl.load(v_ptrs, mask=tile_mask[:, None] & d_mask[None, :], other=0.0)

        # Compute scores: [BLOCK_M, TILE_KV]
        S = tl.dot(Q, K) * softmax_scale

        # Apply softcap
        if USE_SOFTCAP:
            t = S / softcap
            S = softcap * (2.0 / (1.0 + tl.exp(-2.0 * t)) - 1.0)

        # Apply causal mask
        if causal:
            causal_mask = kv_indices[None, :] <= query_abs_pos[:, None]
            S = tl.where(causal_mask, S, float("-inf"))

        # Mask invalid KV and Q positions
        S = tl.where(tile_mask[None, :], S, float("-inf"))
        S = tl.where(q_mask[:, None], S, float("-inf"))

        # Online softmax
        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(M, m_j)
        m_safe = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(M - m_safe)
        P = tl.exp(S - m_safe[:, None])
        L_new = alpha * L + tl.sum(P, axis=1)

        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(P.to(V.dtype), V)

        M = m_safe
        L = L_new

    # Final normalization
    L_safe = tl.where(L == 0.0, 1.0, L)
    acc = acc / L_safe[:, None]

    # Write output
    out_ptrs = (
        Out_ptr
        + (q_start + query_pos)[:, None] * stride_ot
        + query_head_idx[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        acc.to(Out_ptr.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )


def triton_unified_attention(
    q,  # [total_tokens, num_query_heads, head_size]
    k,  # [num_blocks, block_size, num_kv_heads, head_size]
    v,  # [num_blocks, block_size, num_kv_heads, head_size]
    out,  # [total_tokens, num_query_heads, head_size]
    cu_seqlens_q,  # [num_seqs + 1]
    max_seqlen_q,  # int
    seqused_k,  # [num_seqs]
    max_seqlen_k,  # int
    softmax_scale,  # float
    causal,  # bool
    window_size,  # (int, int)
    block_table,  # [num_seqs, max_blocks_per_seq]
    softcap,  # float
    q_descale,  # None
    k_descale,  # None
    v_descale,  # None
):
    """Paged attention with GQA, causal mask, and optional softcap."""
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    num_kv_heads = k.shape[2]
    kv_block_size = k.shape[1]
    max_blocks_per_seq = block_table.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads

    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)

    # GCU: uint16 view channel switch for bf16/fp16 contiguous KV caches
    # (see module docstring). Strides are preserved by the same-itemsize
    # view, so the kernel addressing is unchanged.
    use_u16 = (
        q.dtype == k.dtype == v.dtype
        and k.dtype in (torch.bfloat16, torch.float16)
        and k.is_contiguous()
        and v.is_contiguous()
    )
    if use_u16:
        k_arg = k.view(torch.uint16)
        v_arg = v.view(torch.uint16)
    else:
        k_arg = k
        v_arg = v

    # ---------- Tile size selection ----------
    if max_seqlen_q == 1:
        # === Decode path ===
        if num_queries_per_kv <= 16:
            BLOCK_M = 16
        else:
            BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q_POS = BLOCK_M // num_queries_per_kv

        if HEAD_SIZE_PADDED <= 128:
            TILE_KV = 64
            num_warps = 1
            num_stages = 3
        else:
            TILE_KV = 32
            num_warps = 1
            num_stages = 2
    else:
        # === Prefill path ===
        if HEAD_SIZE_PADDED <= 128:
            BLOCK_M = max(
                16,
                min(
                    128,
                    triton.next_power_of_2(num_queries_per_kv * min(32, max_seqlen_q)),
                ),
            )
            if BLOCK_M < 16:
                BLOCK_M = 16
            BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
            TILE_KV = 64
            num_warps = 1
            num_stages = 2
        else:
            if num_queries_per_kv <= 16:
                BLOCK_M = 16
            else:
                BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
            BLOCK_Q_POS = BLOCK_M // num_queries_per_kv
            TILE_KV = 32
            num_warps = 1
            num_stages = 2

    # Softcap
    USE_SOFTCAP = softcap is not None and softcap > 0.0
    softcap_val = float(softcap) if USE_SOFTCAP else 0.0

    # Grid: (max_q_blocks_per_seq, num_kv_heads, num_seqs)
    max_q_blocks = (max_seqlen_q + BLOCK_Q_POS - 1) // BLOCK_Q_POS
    grid = (max_q_blocks, num_kv_heads, num_seqs)

    _paged_attn_fwd[grid](
        q,
        k_arg,
        v_arg,
        out,
        block_table,
        cu_seqlens_q,
        seqused_k,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        softmax_scale=softmax_scale,
        USE_SOFTCAP=USE_SOFTCAP,
        softcap=softcap_val,
        causal=causal,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kb=k.stride(0),
        stride_ks=k.stride(1),
        stride_kh=k.stride(2),
        stride_kd=k.stride(3),
        stride_vb=v.stride(0),
        stride_vs=v.stride(1),
        stride_vh=v.stride(2),
        stride_vd=v.stride(3),
        stride_ot=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        stride_bt_seq=block_table.stride(0),
        stride_bt_blk=block_table.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_Q_POS=BLOCK_Q_POS,
        TILE_KV=TILE_KV,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        V_U16=use_u16,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out


def _install():
    """Wire the GCU implementation into the direct-import entrypoints.

    Tests and benchmarks import ``from flaggems_vllm.ops.<module> import
    <op>``, which resolves against the already-imported generic modules, so
    the SpecOpRegistrar namespace swap alone can not reach them. Replace the
    attributes on those modules in place, mirroring the kunlunxin /
    bf16_paged_mqa_logits backend overrides.
    """
    for module_name in (
        "flaggems_vllm.ops.triton_unified_attention",
        "flaggems_vllm.ops",
    ):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        cur = getattr(mod, "triton_unified_attention", None)
        if cur is _general_triton_unified_attention:
            mod.triton_unified_attention = triton_unified_attention


_install()
