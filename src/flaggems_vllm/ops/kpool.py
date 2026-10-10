# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 K-pool compression and ordered tail-cache updates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.data_movement import contiguous_copy
from flaggems_vllm.ops.fp8_storage import encode_e4m3fn
from flaggems_vllm.utils import libentry, libtuner

INDEX_HEAD_DIM = 128


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845  # 1/sqrt(128)


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    # One FWHT butterfly stage on a flat tensor of N = GROUPS*2*STRIDE elems;
    # same construction as _hadamard128_stage but with a parametric N so it can
    # process BLOCK_R rows at once.
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@libentry()
@libtuner(configs=runtime.get_tuned_config("kpool_fwht"), key=["n_rows"])
@triton.jit
def _fwht_quant_kernel(
    q_ptr,
    qout_ptr,
    sout_ptr,
    n_rows,
    BLOCK_R: tl.constexpr,
):
    """Fused Hadamard-128 rotation + per-row absmax FP8 (ue8m0) quant.

    Per row of 128 elements (one indexer head): load bf16 -> fp32 butterflies
    (exact adds/subs; the 1/sqrt(128) scale stays fp32) -> round to bf16 ->
    absmax quant with power-of-2 scale. Numerically identical to the unfused
    sglang chain ``rotate_activation(q)`` (fast FWHT, bf16 out) followed by
    ``act_quant`` (absmax clamp 1e-4, exp2(ceil(log2)) scale, clamp +-448).
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    offs = tl.arange(0, 128)
    x = tl.load(
        q_ptr + rows[:, None] * 128 + offs[None, :], mask=rmask[:, None], other=0.0
    ).to(tl.float32)

    # Flatten so each row's 128 lanes stay contiguous: every stage's
    # (GROUPS, 2, STRIDE) tiling has 2*STRIDE dividing 128, so pairs never
    # straddle a row boundary. GROUPS of each stage scales by BLOCK_R.
    N: tl.constexpr = BLOCK_R * 128
    x = tl.reshape(x, (N,))
    x = _fwht_stage(x, N, BLOCK_R * 64, 1)
    x = _fwht_stage(x, N, BLOCK_R * 32, 2)
    x = _fwht_stage(x, N, BLOCK_R * 16, 4)
    x = _fwht_stage(x, N, BLOCK_R * 8, 8)
    x = _fwht_stage(x, N, BLOCK_R * 4, 16)
    x = _fwht_stage(x, N, BLOCK_R * 2, 32)
    x = _fwht_stage(x, N, BLOCK_R, 64)
    x = x * 0.08838834764831845  # 1/sqrt(128), exact in fp32

    # Match the unfused path's bf16 materialization before quantizing.
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, 128))

    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)

    tl.store(
        qout_ptr + rows[:, None] * 128 + offs[None, :],
        encode_e4m3fn(y),
        mask=rmask[:, None],
    )
    tl.store(sout_ptr + rows, scale, mask=rmask)


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each 128-wide row by the Hadamard-128 transform, then FP8-quant.

    Replaces ``q @ H`` (bf16 GEMM whose H entries are bf16-rounded) plus
    ``per_token_group_quant_fp8`` with one kernel: the rotation runs in fp32
    with the exact 1/sqrt(128) constant (matching sglang's fast FWHT), and the
    quant replicates sglang's ``act_quant`` (block 128, ue8m0 scale).

    Args:
        q: ``[rows, 128]`` bf16 — one head vector per row.

    Returns:
        (q_fp8 ``[rows, 128]`` float8_e4m3fn, scale ``[rows, 1]`` float32).
    """
    assert q.ndim == 2 and q.shape[1] == 128, q.shape
    assert q.dtype == torch.bfloat16
    assert q.is_contiguous()
    n_rows = q.shape[0]
    q_fp8 = torch.empty((n_rows, 128), dtype=torch.float8_e4m3fn, device=q.device)
    q_scale = torch.empty((n_rows, 1), dtype=torch.float32, device=q.device)
    if n_rows == 0:
        return q_fp8, q_scale
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_R"]),)
    _fwht_quant_kernel[grid](q, q_fp8.view(torch.uint8), q_scale, n_rows)
    return q_fp8, q_scale


@triton.jit
def _kpool_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per pool. softmax(slot_score+ape)-weighted sum of slot_k ->
    Hadamard-128 -> per-vector fp8 absmax quant -> write to cache at ``loc``."""
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    # --- Pass 1: per-dim max over the pool (softmax numerical stability) ---
    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        max_score = tl.maximum(max_score, score)

    # --- Pass 2: softmax-weighted sum of K ---
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        prob = tl.exp(score - max_score)
        denom += prob
        k = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += k * prob

    x = acc / denom
    x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)

    # Hadamard-128 rotation (spreads energy for uniform fp8 quant error).
    # Match sglang: bf16 round-trip after the Hadamard so the fp8 absmax/scale
    # sees the same precision as the unfused (bf16-stored) pooled-K path.
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    # --- per-vector absmax fp8 quant ---
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    absmax = tl.max(tl.abs(x), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv
    quantized = x / scale
    quantized = tl.minimum(tl.maximum(quantized, -fp8_max), fp8_max)

    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc % PAGE_SIZE
        out_k_offsets = (
            loc_page_index * BUF_NUMEL_PER_PAGE
            + loc_token_offset_in_page * HEAD_DIM
            + offs
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )
        tl.store(buf_fp8_ptr + out_k_offsets, encode_e4m3fn(quantized), mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)

    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            encode_e4m3fn(quantized),
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, scale)


def kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """Compress ``pool_size`` tokens into one fp8 K and write at ``loc``.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        slot_k: ``[n_pools, pool_size, head_dim]`` bf16 — raw per-token K.
        slot_score: ``[n_pools, pool_size, head_dim]`` — per-token gate score.
        ape: ``[pool_size, head_dim]`` fp32 — per-slot position bias.
        loc: ``[n_pools]`` int64 — flat physical slot per pool.
    """
    assert slot_k.ndim == 3
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == head_dim
    assert slot_k.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert write_cache or return_compressed

    page_size = kv_cache.shape[1]
    buf = kv_cache
    slot_k = contiguous_copy(slot_k)
    slot_score = contiguous_copy(slot_score)
    ape = contiguous_copy(ape)
    loc = contiguous_copy(loc)
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        write_mask = contiguous_copy(write_mask)
        has_write_mask = True
        assert not return_compressed

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf.view(torch.uint8)
    buf_fp32 = buf.view(torch.float32)
    # bytes per page (last dim of kv_cache) viewed as uint8
    buf_numel_per_page = buf.stride(0)
    s_offset_nbytes_in_page = page_size * head_dim

    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], head_dim),
            dtype=torch.float8_e4m3fn,
            device=slot_k.device,
        )
        compressed_scale = torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32

    _kpool_softmax_rotate_write_cache_kernel[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k.view(torch.uint8),
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    if return_compressed:
        return compressed_k, compressed_scale
    return None


@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    if (
        tail_kv_cache.dtype not in (torch.bfloat16, torch.float32)
        or key.dtype != tail_kv_cache.dtype
        or gate_score.dtype != key.dtype
    ):
        raise ValueError("Tail seed inputs must share bf16/fp32 dtype")
    if (
        tail_kv_cache.shape[1:] != (2, kpool, head_dim)
        or key.shape != gate_score.shape
        or key.shape[1:] != (head_dim,)
    ):
        raise ValueError("Tail seed shape mismatch")
    if any(t.device != tail_kv_cache.device for t in (key, gate_score, tslot)):
        raise ValueError("Tail seed inputs must share device")
    if not tail_kv_cache.is_contiguous():
        raise ValueError("Tail seed cache must be contiguous")
    key, gate_score, tslot = map(contiguous_copy, (key, gate_score, tslot))
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _kpool_decode_update_batched_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,  # [B, NEXT_N] int32
    key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    key_stride_b,
    key_stride_t,
    slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,  # [B, NEXT_N] int32
    positions_ptr,  # [B, NEXT_N] int32
    NEXT_N,  # runtime token count per request (no .item() needed)
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per request; iterates its NEXT_N verify tokens in order.

    Replaces the caller's per-token sequential launch loop. The intra-request
    iteration MUST stay in position order: a pool-completion at token t* reads
    the tail-ring slots that tokens t < t* (same request) just stashed in this
    same invocation. ``tl.range`` iterates sequentially within the program, so
    those stashes are visible to the later completion read. Cross-request
    programs are independent (distinct tail blocks). With NEXT_N < POOL_SIZE
    (the spec-verify case: NEXT_N ~= num_spec+1, POOL_SIZE=16) at most one
    completion can occur per request per call, but the ordered loop is correct
    for any NEXT_N.
    """
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    for t in tl.range(0, NEXT_N):
        idx = req * NEXT_N + t
        cache_loc = tl.load(slot_mapping_ptr + idx)
        pos = tl.load(positions_ptr + idx)
        safe_pos = tl.maximum(pos, 0)
        pos_valid = (cache_loc >= 0) & (pos >= 0)

        slot = safe_pos % POOL_SIZE
        phys_slot = safe_pos % POOL_SIZE

        # Derive the tail block from THIS token's tail_slot (the request's block
        # is constant across a pool, but a padded / invalid entry carries a
        # negative sentinel -- reading it from token 0 would poison every
        # token's base address). Clamp so an invalid entry can never form an
        # out-of-bounds base; the accesses below are gated on pos_valid anyway.
        tail_slot = tl.load(tail_slot_mapping_ptr + idx)
        block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE
        block_base = block * TAIL_BLOCK_ELEMS

        # The tail-ring stash must run for EVERY real token, so it is gated on
        # the token-granular tail slot -- not on `pos_valid`, which keys off the
        # POOL-granular `slot_mapping` and is therefore only true on the pool's
        # last token. Gating the stash on pos_valid dropped every intra-pool
        # token, so a decode-built pool compressed 3 stale ring entries (the
        # prefill-seeded prompt tail, frozen forever) plus the current token.
        stash_valid = (pos >= 0) & (tail_slot >= 0)

        key = tl.load(
            key_ptr + req * key_stride_b + t * key_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score_current = tl.load(
            slot_score_ptr + req * ss_stride_b + t * ss_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if pos_valid & (slot == POOL_SIZE - 1):
            pool_logical_start = safe_pos - slot

            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                prob = tl.exp(score - max_score)
                denom += prob
                k_buf = tl.load(
                    tail_kv_ptr + block_base + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                k = tl.where(is_current, key, k_buf)
                acc += k * prob

            x = (acc / denom).to(tl.bfloat16).to(tl.float32)
            x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

            fp8_max = 448.0
            fp8_max_inv = 1.0 / fp8_max
            absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
            else:
                scale = absmax * fp8_max_inv
            quantized = tl.minimum(tl.maximum(x / scale, -fp8_max), fp8_max)

            loc = cache_loc.to(tl.int64)
            loc_page_index = loc // PAGE_SIZE
            loc_token_offset_in_page = loc % PAGE_SIZE
            out_k_offsets = (
                loc_page_index * BUF_NUMEL_PER_PAGE
                + loc_token_offset_in_page * HEAD_DIM
                + offs
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(
                buf_fp8_ptr + out_k_offsets, encode_e4m3fn(quantized), mask=dim_mask
            )
            tl.store(buf_fp32_ptr + out_s_offset, scale)

        # Stash the current token AFTER any completion read so the completion
        # uses prior stashes (and the current token's own key/score via
        # is_current), then leaves this token for future pools. Order matches
        # the per-token kernel: completion read first, stash second.
        update_mask = dim_mask & stash_valid
        tl.store(
            tail_kv_ptr + block_base + phys_slot * HEAD_DIM + offs,
            key,
            mask=update_mask,
        )
        tl.store(
            tail_kv_ptr + block_base + KPOOL_HEAD + phys_slot * HEAD_DIM + offs,
            score_current,
            mask=update_mask,
        )


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Batched decode-step kpool update for spec verify (``next_n > 1``).

    One launch replaces the caller's per-token loop. Inputs are grouped per
    request: ``[num_requests, next_n, ...]``. Each program handles one
    request's ``next_n`` tokens in position order (see the kernel docstring for
    why ordering is required for pool-completion correctness).

    Plain decode (``next_n == 1``) is handled here too — the kernel collapses
    to a single-iteration loop.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        tail_kv_cache: paged tail cache ``[num_blocks, 2, pool_size, head_dim]``
            bf16 (K at half 0, gate score at half 1).
        tail_slot_mapping: ``[num_requests, next_n]`` int32.
        key / slot_score: ``[num_requests, next_n, head_dim]`` bf16.
        ape: ``[pool_size, head_dim]`` fp32.
        slot_mapping / positions: ``[num_requests, next_n]`` int32.
    """
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    assert tail_kv_cache.ndim == 4
    assert tail_kv_cache.shape[1] == 2
    assert tail_kv_cache.shape[2] == pool_size
    assert tail_kv_cache.shape[3] == head_dim
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.ndim == 3 and key.shape[2] == head_dim
    assert slot_score.shape == key.shape
    assert ape.shape == (pool_size, head_dim)
    assert tail_slot_mapping.shape == (num_requests, next_n)
    assert slot_mapping.shape == (num_requests, next_n)
    assert positions.shape == (num_requests, next_n)
    assert key.dtype == torch.bfloat16
    assert slot_score.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8

    page_size = kv_cache.shape[1]
    buf = kv_cache
    buf_fp8 = buf.view(torch.uint8)
    buf_fp32 = buf.view(torch.float32)

    # The kernel indexes the int tensors as ``req * next_n + t`` (row-major),
    # so they must be contiguous. Callers pass either a view of a contiguous
    # slice or a freshly scattered tensor, making these no-ops; the calls guard
    # against a future caller handing over a strided view.
    tail_slot_mapping = contiguous_copy(tail_slot_mapping)
    slot_mapping = contiguous_copy(slot_mapping)
    positions = contiguous_copy(positions)

    _kpool_decode_update_batched_kernel[(num_requests,)](
        buf_fp8,
        buf_fp32,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf.stride(0),
        POOL_SIZE=pool_size,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@libentry()
@libtuner(configs=runtime.get_tuned_config("kpool_hadamard"), key=["ROWS"])
@triton.jit
def _hadamard_kernel(X, Y, ROWS: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, 128)
    x = tl.load(X + row * 128 + offs).to(tl.float32)
    tl.store(Y + row * 128 + offs, _hadamard128(x))


def hadamard128(x):
    if x.shape[-1] != 128:
        raise ValueError("Hadamard transform requires last dimension 128")
    x = contiguous_copy(x)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.numel() // 128
    if rows:
        _hadamard_kernel[(rows,)](x, out, rows)
    return out
