# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consolidated H20 fused_q with the baseline public input/output contract.

Only the validated full H20 BF16 shapes use round-three scheduling: compact
CTAs for H=16/32, the original grid for H=64, and PDL disabled. Other inputs
retain the original grid and architecture PDL policy. No experiment selector,
latency table, scale probe or arithmetic change is included.
"""

from functools import cache

import torch
import triton
import triton.language as tl

_FP8_DTYPE = torch.float8_e4m3fn
_USE_FNUZ = False
_FP8_MAX = 448.0
_DUMMY_CACHE: dict[tuple, torch.Tensor] = {}


@cache
def _device_policy(device_index: int) -> tuple[bool, bool]:
    """Cache per device; exact H20 matching deliberately excludes H200 and MIG."""
    props = torch.cuda.get_device_properties(device_index)
    full_h20 = (
        props.name.strip().upper() == "NVIDIA H20"
        and (props.major, props.minor) == (9, 0)
        and props.multi_processor_count == 78
    )
    return full_h20, props.major >= 9


def _validated_h20_inputs(
    positions,
    q_pe,
    q_pe_cos_sin_cache,
    index_q,
    index_q_cos_sin_cache,
    ql_nope,
    q_scale,
    index_weights,
    has_indexer,
    index_rope_interleave,
    quantize_mqa,
):
    """Metadata-only scope guard; never read a tensor value or synchronize."""
    if not has_indexer or index_rope_interleave or not quantize_mqa:
        return False
    n, h, r = q_pe.shape
    if n not in (1, 6, 24, 96, 384, 1536, 4096) or h not in (16, 32, 64) or r != 64:
        return False
    if (
        positions.shape != (n,)
        or ql_nope.shape != (n, h, 512)
        or index_q.shape != (n, 32, 128)
        or index_weights.shape != (n, 32)
        or q_pe_cos_sin_cache.shape != (16384, 64)
        or index_q_cos_sin_cache.shape != (16384, 64)
    ):
        return False
    bf16 = (q_pe, q_pe_cos_sin_cache, index_q, index_q_cos_sin_cache, ql_nope)
    if any(t.dtype != torch.bfloat16 for t in bf16):
        return False
    if index_weights.dtype != torch.float32:
        return False
    return all(
        t.device == q_pe.device and t.is_contiguous()
        for t in (*bf16, positions, q_scale, index_weights)
    )


@triton.jit
def _compact_task(task, NUM_Q_HEADS: tl.constexpr, NUM_INDEX_Q_HEADS: tl.constexpr):
    q_heads = (NUM_Q_HEADS + 1) // 2
    pid = tl.where(
        task < q_heads, 0, tl.where(task < q_heads + NUM_INDEX_Q_HEADS, 1, 2)
    )
    head = tl.where(
        pid == 0,
        task,
        tl.where(pid == 1, task - q_heads, task - q_heads - NUM_INDEX_Q_HEADS),
    )
    return pid, head


def _dummy(shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (shape, dtype, device)
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _DUMMY_CACHE[key] = t
    return t


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fp8_ue8m0_quantize(vals, FP8_MAX: tl.constexpr, USE_FNUZ: tl.constexpr):
    """Quantize float32 values to FP8 E4M3 with a ue8m0 (power-of-2) scale.

    Returns (fp8_vals, scale) so the caller can store them or reuse the scale.
    """
    vals = vals.to(tl.float32)
    amax = tl.max(tl.abs(vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), FP8_MAX)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv
    fp8_vals = tl.div_rn(vals, scale).to(fp8_dtype)
    return fp8_vals, scale


@triton.jit
def _fused_q_kernel(
    pos_ptr,
    # MQA query PE: RoPE + FP8 pack into output tail
    q_pe_ptr,
    q_pe_stride0,
    q_pe_stride1,
    NUM_Q_HEADS: tl.constexpr,
    q_pe_cos_sin_ptr,
    q_pe_cos_sin_stride,
    Q_PE_HALF_ROT_DIM: tl.constexpr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    NUM_INDEX_Q_HEADS: tl.constexpr,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # MQA query pack: quantize ql_nope and RoPE+quantize q_pe into mqa_q_fp8
    ql_nope_ptr,
    ql_nope_stride0,
    ql_nope_stride1,
    mqa_q_fp8_ptr,
    mqa_q_fp8_stride0,
    mqa_q_fp8_stride1,
    q_scale_ptr,
    QL_NOPE_DIM: tl.constexpr,
    QL_NOPE_BLOCK: tl.constexpr,
    # bf16 MQA query RoPE output (when QUANTIZE_MQA is False); the NoPE part is
    # consumed directly from ql_nope, so only the RoPE'd q_pe is written here.
    q_pe_out_ptr,
    q_pe_out_stride0,
    q_pe_out_stride1,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    HAS_INDEXER: tl.constexpr,
    INDEX_ROPE_INTERLEAVE: tl.constexpr,
    QUANTIZE_MQA: tl.constexpr,
    USE_PDL: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_FNUZ: tl.constexpr,
    COMPACT_GRID: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    if COMPACT_GRID:
        pid, head_idx = _compact_task(tl.program_id(1), NUM_Q_HEADS, NUM_INDEX_Q_HEADS)
    else:
        pid = tl.program_id(1)
        head_idx = tl.program_id(2)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    fp8_dtype = tl.float8e4b8 if USE_FNUZ else tl.float8e4nv

    if pid == 2:
        # ql_nope quantize + pack into the front of mqa_q_fp8. On the bf16
        # query path ql_nope is consumed as-is (no pack), so skip entirely.
        if not QUANTIZE_MQA:
            return
        if 2 * head_idx >= NUM_Q_HEADS:
            return

        scale = tl.load(q_scale_ptr)
        for local_head in range(2):
            q_head_idx = head_idx * 2 + local_head
            if q_head_idx < NUM_Q_HEADS:
                ql_nope_off = tl.arange(0, QL_NOPE_BLOCK)
                ql_nope_mask = ql_nope_off < QL_NOPE_DIM
                ql_nope = tl.load(
                    ql_nope_ptr
                    + tok_idx * ql_nope_stride0
                    + q_head_idx * ql_nope_stride1
                    + ql_nope_off,
                    mask=ql_nope_mask,
                ).to(tl.float32)
                ql_nope_fp8 = (ql_nope / scale).to(fp8_dtype)
                tl.store(
                    mqa_q_fp8_ptr
                    + tok_idx * mqa_q_fp8_stride0
                    + q_head_idx * mqa_q_fp8_stride1
                    + ql_nope_off,
                    ql_nope_fp8,
                    mask=ql_nope_mask,
                )
        return
    elif pid == 0:
        # q_pe RoPE + quantize + pack into the tail of mqa_q_fp8.
        if 2 * head_idx >= NUM_Q_HEADS:
            return

        pos = tl.load(pos_ptr + tok_idx)
        cos, sin = _get_cos_sin(
            q_pe_cos_sin_ptr,
            q_pe_cos_sin_stride,
            pos,
            Q_PE_HALF_ROT_DIM,
        )

        scale = tl.load(q_scale_ptr)
        for local_head in range(2):
            q_head_idx = head_idx * 2 + local_head
            if q_head_idx < NUM_Q_HEADS:
                rot_off = tl.arange(0, Q_PE_HALF_ROT_DIM)
                x1 = tl.load(
                    q_pe_ptr
                    + tok_idx * q_pe_stride0
                    + q_head_idx * q_pe_stride1
                    + rot_off * 2,
                ).to(tl.float32)
                x2 = tl.load(
                    q_pe_ptr
                    + tok_idx * q_pe_stride0
                    + q_head_idx * q_pe_stride1
                    + rot_off * 2
                    + 1
                ).to(tl.float32)
                r1 = x1 * cos - x2 * sin
                r2 = x2 * cos + x1 * sin
                if QUANTIZE_MQA:
                    tl.store(
                        mqa_q_fp8_ptr
                        + tok_idx * mqa_q_fp8_stride0
                        + q_head_idx * mqa_q_fp8_stride1
                        + QL_NOPE_DIM
                        + rot_off * 2,
                        (r1 / scale).to(fp8_dtype),
                    )
                    tl.store(
                        mqa_q_fp8_ptr
                        + tok_idx * mqa_q_fp8_stride0
                        + q_head_idx * mqa_q_fp8_stride1
                        + QL_NOPE_DIM
                        + rot_off * 2
                        + 1,
                        (r2 / scale).to(fp8_dtype),
                    )
                else:
                    # bf16 query: write the RoPE'd q_pe unquantized.
                    out_ty = q_pe_out_ptr.dtype.element_ty
                    q_pe_dst = (
                        q_pe_out_ptr
                        + tok_idx * q_pe_out_stride0
                        + q_head_idx * q_pe_out_stride1
                    )
                    tl.store(q_pe_dst + rot_off * 2, r1.to(out_ty))
                    tl.store(q_pe_dst + rot_off * 2 + 1, r2.to(out_ty))
        return
    elif pid == 1:
        # Index Q RoPE + fp8 quant, all in registers. The roped bf16 index_q is
        # never consumed (only the fp8 below is), so we avoid an in-place
        # store-then-reload round-trip.
        if not HAS_INDEXER:
            return
        if head_idx >= NUM_INDEX_Q_HEADS:
            return

        pos = tl.load(pos_ptr + tok_idx)
        index_q_block = tl.arange(0, INDEX_Q_HEAD_DIM)
        iq_base = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
        index_q = tl.load(iq_base + index_q_block).to(tl.float32)

        # RoPE in registers (interleaved for GLM-5.2, NeoX for DeepSeek-V3.2),
        # gathering the rotation partner from the read-only input.
        in_rope = index_q_block < 2 * INDEX_Q_HALF_ROT_DIM
        if INDEX_ROPE_INTERLEAVE:
            cos_idx = index_q_block // 2
            partner_offs = tl.where(in_rope, index_q_block ^ 1, index_q_block)
            sign = tl.where(index_q_block % 2 == 0, -1.0, 1.0)
        else:
            cos_idx = index_q_block % INDEX_Q_HALF_ROT_DIM
            partner_offs = tl.where(
                in_rope, index_q_block ^ INDEX_Q_HALF_ROT_DIM, index_q_block
            )
            sign = tl.where(index_q_block < INDEX_Q_HALF_ROT_DIM, -1.0, 1.0)
        cos_full = tl.load(
            index_q_cos_sin_ptr + pos * index_q_cos_sin_stride + cos_idx,
            mask=in_rope,
            other=1.0,
        ).to(tl.float32)
        sin_full = tl.load(
            index_q_cos_sin_ptr
            + pos * index_q_cos_sin_stride
            + INDEX_Q_HALF_ROT_DIM
            + cos_idx,
            mask=in_rope,
            other=0.0,
        ).to(tl.float32)
        partner = tl.load(iq_base + partner_offs).to(tl.float32)
        roped = index_q * cos_full + sign * partner * sin_full
        index_q = tl.where(in_rope, roped, index_q)

        # Index Q Quantize (from registers)
        index_q_fp8, index_q_scale = _fp8_ue8m0_quantize(index_q, FP8_MAX, USE_FNUZ)
        tl.store(
            index_q_fp8_ptr
            + tok_idx * index_q_fp8_stride0
            + head_idx * index_q_fp8_stride1
            + index_q_block,
            index_q_fp8,
        )

        # Index weights update
        index_weights = tl.load(
            index_weights_ptr + tok_idx * index_weights_stride + head_idx
        )
        index_weights = index_weights.to(tl.float32)
        index_weights *= index_q_scale
        index_weights *= index_weights_softmax_scale
        index_weights *= index_weights_head_scale
        tl.store(
            index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
            index_weights,
        )


def fused_q(
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    q_pe_cos_sin_cache: torch.Tensor,
    index_q: torch.Tensor | None,
    index_q_cos_sin_cache: torch.Tensor | None,
    ql_nope: torch.Tensor,
    q_scale: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor | None,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    has_indexer: bool = True,
    index_rope_interleave: bool = False,
    quantize_mqa: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the MQA-query and indexer-query RoPE/quantization.

    Returns ``(index_q_fp8, index_weights_out, mqa_q)``. When ``quantize_mqa``
    is True (FlashInfer sparse, fp8 query) ``mqa_q`` is a single fp8 tensor
    packing ``[ql_nope; q_pe]``. When False (FlashMLA sparse, bf16 query) it is
    the RoPE'd ``q_pe`` in bf16; the caller pairs it with ``ql_nope`` as the
    ``(ql_nope, q_pe)`` tuple the backend expects.
    """
    assert positions.ndim == 1
    assert positions.dtype == torch.int64
    assert q_pe.ndim == 3
    assert q_pe_cos_sin_cache.ndim == 2
    assert ql_nope.ndim == 3
    assert ql_nope.shape[:2] == q_pe.shape[:2]
    assert q_scale.dtype == torch.float32 and q_scale.numel() == 1
    num_tokens = positions.shape[0]
    num_q_heads = q_pe.shape[1]
    # Grid's 3rd dim must cover the MQA-pack heads (pid 0/2 iterate 2 heads
    # each) and, when present, the indexer heads (pid 1).
    mqa_grid_heads = (num_q_heads + 1) // 2
    if not has_indexer:
        # Shared layer: cached 1-element dummies; pid 1 skipped by HAS_INDEXER
        # and never dereferences them.
        index_q = _dummy((1, 1, 1), q_pe.dtype, q_pe.device)
        index_q_cos_sin_cache = q_pe_cos_sin_cache
        index_weights = _dummy((1, 1), torch.float32, q_pe.device)
    assert index_q is not None and index_q.ndim == 3
    assert index_q_cos_sin_cache is not None
    assert index_weights is not None
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]
    grid_heads = max(mqa_grid_heads, num_index_q_heads)
    if quantize_mqa:
        # fp8 path: pack [ql_nope; q_pe] into a single fp8 tensor.
        mqa_q_fp8 = torch.empty(
            q_pe.shape[0],
            q_pe.shape[1],
            ql_nope.shape[2] + q_pe.shape[2],
            dtype=_FP8_DTYPE,
            device=q_pe.device,
        )
        # Placeholder; pid 0 packs q_pe into mqa_q_fp8 instead.
        q_pe_out = mqa_q_fp8
        mqa_q = mqa_q_fp8
    else:
        # bf16 path: only the RoPE'd q_pe is produced; ql_nope used directly.
        q_pe_out = torch.empty_like(q_pe)
        mqa_q_fp8 = q_pe_out  # unused placeholder for the fp8 pack pointer
        mqa_q = q_pe_out

    index_q_fp8 = torch.empty_like(index_q, dtype=_FP8_DTYPE)
    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    full_h20, supports_pdl = _device_policy(q_pe.device.index)
    optimized = full_h20 and _validated_h20_inputs(
        positions,
        q_pe,
        q_pe_cos_sin_cache,
        index_q,
        index_q_cos_sin_cache,
        ql_nope,
        q_scale,
        index_weights,
        has_indexer,
        index_rope_interleave,
        quantize_mqa,
    )
    compact_grid = optimized and num_q_heads in (16, 32)
    use_pdl = supports_pdl and not optimized
    grid = (
        (num_tokens, 2 * mqa_grid_heads + num_index_q_heads)
        if compact_grid
        else (num_tokens, 3, grid_heads)
    )
    pdl_kwargs = (
        {"USE_PDL": use_pdl, "launch_pdl": True} if use_pdl else {"USE_PDL": use_pdl}
    )
    _fused_q_kernel[grid](
        positions,
        q_pe,
        q_pe.stride(0),
        q_pe.stride(1),
        num_q_heads,
        q_pe_cos_sin_cache,
        q_pe_cos_sin_cache.stride(0),
        q_pe_cos_sin_cache.shape[-1] // 2,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        num_index_q_heads,
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        index_q_head_dim,
        ql_nope,
        ql_nope.stride(0),
        ql_nope.stride(1),
        mqa_q_fp8,
        mqa_q_fp8.stride(0),
        mqa_q_fp8.stride(1),
        q_scale,
        ql_nope.shape[2],
        triton.next_power_of_2(ql_nope.shape[2]),
        q_pe_out,
        q_pe_out.stride(0),
        q_pe_out.stride(1),
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        HAS_INDEXER=has_indexer,
        INDEX_ROPE_INTERLEAVE=index_rope_interleave,
        QUANTIZE_MQA=quantize_mqa,
        FP8_MAX=_FP8_MAX,
        USE_FNUZ=_USE_FNUZ,
        COMPACT_GRID=compact_grid,
        **pdl_kwargs,
        # num_warps=1 is optimal here: each program is a single 128-element
        # rope+quant, so the kernel is program-count/occupancy bound, not
        # per-program compute bound (swept 1/2/4/8 — 1 wins or ties everywhere).
        num_warps=1,
    )
    return index_q_fp8, index_weights_out, mqa_q


__all__ = ["fused_q"]
