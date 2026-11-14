# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is adapted from flash-linear-attention under the MIT License.
# Original source:
# fla/ops/gdn2/chunk_intra_token_parallel.py
#
# Token-parallel intra-chunk kernel for GDN-2.
#
# Builds:
#   Aqk: causal, decay-weighted query-key scores for output computation.
#   Akk: diagonal BC x BC strictly-lower key-key blocks used by WY solving.
#
# GDN-2 difference from KDA:
# the channel-wise erase gate b is folded into the current-row key tile.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.triton_ops_helper import autotune_cache_kwargs, exp2

__all__ = [
    "chunk_gdn2_fwd_kernel_intra_token_parallel",
    "chunk_gdn2_fwd_intra_token_parallel",
]


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BH": bh}, num_warps=num_warps)
        for bh in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_gdn2_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    b,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t_global = tl.program_id(0)
    i_h_group = tl.program_id(1)

    if IS_VARLEN:
        # Find the packed sequence containing token i_t_global.
        left = 0
        right = N

        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                mid_end = tl.load(cu_seqlens + mid + 1).to(tl.int32)

                if i_t_global < mid_end:
                    right = mid
                else:
                    left = mid + 1

        i_seq = left
        bos = tl.load(cu_seqlens + i_seq).to(tl.int32)
        eos = tl.load(cu_seqlens + i_seq + 1).to(tl.int32)

        local_t = i_t_global - bos
        T = eos - bos
    else:
        bos = (i_t_global // T) * T
        local_t = i_t_global % T

    if local_t >= T:
        return

    i_chunk = local_t // BT
    i_subchunk = (local_t % BT) // BC

    chunk_start = i_chunk * BT
    subchunk_start = chunk_start + i_subchunk * BC

    # Move all pointers to the current sequence start.
    q += bos * H * K
    k += bos * H * K
    g += bos * H * K
    b += bos * H * K

    Aqk += bos * H * BT
    Akk += bos * H * BC

    BK: tl.constexpr = triton.next_power_of_2(K)

    offsets_h = tl.arange(0, BH)
    offsets_k = tl.arange(0, BK)

    mask_h = i_h_group * BH + offsets_h < H
    mask_k = offsets_k < K

    q_ptr = tl.make_block_ptr(
        base=q + local_t * H * K,
        shape=(H, K),
        strides=(K, 1),
        offsets=(i_h_group * BH, 0),
        block_shape=(BH, BK),
        order=(1, 0),
    )

    k_ptr = tl.make_block_ptr(
        base=k + local_t * H * K,
        shape=(H, K),
        strides=(K, 1),
        offsets=(i_h_group * BH, 0),
        block_shape=(BH, BK),
        order=(1, 0),
    )

    g_ptr = tl.make_block_ptr(
        base=g + local_t * H * K,
        shape=(H, K),
        strides=(K, 1),
        offsets=(i_h_group * BH, 0),
        block_shape=(BH, BK),
        order=(1, 0),
    )

    b_ptr = tl.make_block_ptr(
        base=b + local_t * H * K,
        shape=(H, K),
        strides=(K, 1),
        offsets=(i_h_group * BH, 0),
        block_shape=(BH, BK),
        order=(1, 0),
    )

    q_row = tl.load(q_ptr, boundary_check=(0, 1)).to(tl.float32)
    k_row = tl.load(k_ptr, boundary_check=(0, 1)).to(tl.float32)
    g_row = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)
    erase_row = tl.load(b_ptr, boundary_check=(0, 1)).to(tl.float32)

    # GDN-2: erase gate is channel-wise, so it multiplies the row key.
    k_row = k_row * erase_row

    subchunk_end = min(local_t + 1, min(T, subchunk_start + BC))

    for j in range(subchunk_start, subchunk_end):
        k_j_ptr = tl.make_block_ptr(
            base=k + j * H * K,
            shape=(H, K),
            strides=(K, 1),
            offsets=(i_h_group * BH, 0),
            block_shape=(BH, BK),
            order=(1, 0),
        )

        g_j_ptr = tl.make_block_ptr(
            base=g + j * H * K,
            shape=(H, K),
            strides=(K, 1),
            offsets=(i_h_group * BH, 0),
            block_shape=(BH, BK),
            order=(1, 0),
        )

        k_j = tl.load(k_j_ptr, boundary_check=(0, 1)).to(tl.float32)
        g_j = tl.load(g_j_ptr, boundary_check=(0, 1)).to(tl.float32)

        # k_j * exp2(g_t - g_j)
        k_decay_j = k_j * exp2(g_row - g_j)
        k_decay_j = tl.where(mask_k[None, :], k_decay_j, 0.0)

        aqk = tl.sum(q_row * k_decay_j, axis=1) * scale

        # Strictly lower triangular only: diagonal is zero.
        akk = tl.sum(k_row * k_decay_j, axis=1)
        akk *= tl.where(j < local_t, 1.0, 0.0)

        tl.store(
            Aqk + local_t * H * BT + (i_h_group * BH + offsets_h) * BT + j % BT,
            aqk.to(Aqk.dtype.element_ty),
            mask=mask_h,
        )

        tl.store(
            Akk
            + local_t * H * BC
            + (i_h_group * BH + offsets_h) * BC
            + j
            - subchunk_start,
            akk.to(Akk.dtype.element_ty),
            mask=mask_h,
        )


def chunk_gdn2_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    b: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build token-parallel diagonal blocks for native GDN2 forward.

    Args:
        q:   [B, T, H, K]
        k:   [B, T, H, K]
        gk:  [B, T, H, K], base-2 local cumulative gate
        b:   [B, T, H, K], activated erase gate
        Aqk: [B, T, H, BT], output buffer
        Akk: [B, T, H, BC], diagonal-block output buffer, preferably fp32
    """
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    BT = chunk_size
    BC = sub_chunk_size

    if BT != 64:
        raise ValueError(f"Native GDN2 token-parallel path requires BT=64, got {BT}.")

    if BC != 16:
        raise ValueError(f"Native GDN2 token-parallel path requires BC=16, got {BC}.")

    if BT % BC != 0:
        raise ValueError(f"chunk_size={BT} must be divisible by sub_chunk_size={BC}.")

    if Aqk.shape != (B, T, H, BT):
        raise ValueError(
            f"Aqk must have shape {(B, T, H, BT)}, got {tuple(Aqk.shape)}."
        )

    if Akk.shape != (B, T, H, BC):
        raise ValueError(
            f"Akk must have shape {(B, T, H, BC)}, got {tuple(Akk.shape)}."
        )

    def grid(meta):
        return (B * T, triton.cdiv(H, meta["BH"]))

    chunk_gdn2_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        b=b,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )

    return Aqk, Akk
