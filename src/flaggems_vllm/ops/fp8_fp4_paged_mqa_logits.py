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

"""Triton kernel for FP8/FP4 Paged Multi-Query Attention Logits.

Computes weighted attention logits from FP8-quantized queries against a paged
KV cache in FP8/FP4 format. Used in DeepSeek-V4 decode-phase inference.

The kernel uses tensor-core MMA (dot product) and an adaptive BLOCK_KV
selection based on context length to balance SM utilization across
different workload distributions.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fp8_fp4_mqa_logits import _e2m1_to_f32
from flaggems_vllm.utils.device_info import get_device_capability
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

logger = logging.getLogger(__name__)

# =============================================================================
# TLE (TMA + WGMMA) fast path — purely additive. The baseline below is left
# byte-for-byte untouched; the fast path only fires inside its measured win
# band and falls back to the baseline everywhere else (no regression).
# =============================================================================

HAS_TLE = has_triton_tle(3, 6, 0) and TensorDescriptor is not None
if HAS_TLE:
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = hasattr(tle.gpu, "wgmma")
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None


def _tle_enabled() -> bool:
    """Master switch for the TMA fast path (mirrors FLAGGEMS_FLASHMLA_DECODE_TLE)."""
    value = os.environ.get("FLAGGEMS_FP8_FP4_PAGED_MQA_LOGITS_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}


# =========================================================================
# Platform-specialized FP8 dot subkernels.
# On nvidia or other platforms, batched tl.dot (Q[H,D] @ KV[N,D]^T) is ~6-10x faster.
# On mthreads, SQMMA requires per-head tl.dot ([1,D] @ [N,D]^T).
# =========================================================================


@triton.jit
def _fp8_dot_batched(
    q_fp8,
    kv_u8,
    w_all,
    scale_tile,
    h_ids,
    d_ids,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
    w_row_base=0,
):
    """Batched FP8 dot: Q[H,D] @ KV[N,D]^T -> [H,N], then weight-reduce."""
    kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)
    dots = tl.dot(q_fp8, tl.trans(kv_fp8))
    scores = tl.maximum(dots * scale_tile[None, :], 0.0)
    weighted = scores * w_all[:, None]
    return tl.sum(weighted, axis=0)


@triton.jit
def _fp8_dot_per_head(
    q_row_base,
    kv_u8,
    w_all,
    scale_tile,
    h_ids,
    d_ids,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
    w_row_base=0,
):
    """Per-head FP8 dot: each head [1,D] @ KV[N,D]^T -> [N], accumulated."""
    output_tile = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for h_idx in tl.static_range(num_heads):
        q_u8_h = tl.load(q_row_base + h_idx * head_dim + d_ids)
        q_fp8_h = tl.reshape(q_u8_h.to(tl.float8e4nv, bitcast=True), [1, head_dim])
        kv_fp8 = kv_u8.to(tl.float8e4nv, bitcast=True)
        dot_h = tl.reshape(tl.dot(q_fp8_h, tl.trans(kv_fp8)), [BLOCK_SIZE])
        w_h = tl.load(w_row_base + h_idx)
        score_h = tl.maximum(dot_h * scale_tile, 0.0)
        output_tile += score_h * w_h
    return output_tile


_use_fp8_batched_dot = not hasattr(torch, "musa")
_fp8_dot = _fp8_dot_batched if _use_fp8_batched_dot else _fp8_dot_per_head


@triton.jit
def _mqa_logits_kernel(
    Q_ptr,  # [total_rows, H * D] uint8 (FP8 bitcast) or [total_rows, H * D//2] uint8 (packed E2M1)
    Q_scale_ptr,  # [total_rows, H] int32 (MXFP4 ue8m0 bytes; unused when not IS_MXFP4)
    KV_data_ptr,  # [num_phys_blocks * BLOCK_SIZE, D] uint8 (FP8, flat paged)
    KV_scales_ptr,  # [num_phys_blocks * BLOCK_SIZE] float32
    Weights_ptr,  # [total_rows, H] float32
    Block_tables_ptr,  # [total_rows, max_blocks_per_seq] int32
    Output_ptr,  # [total_rows, max_model_len] float32
    Ctx_lens_ptr,  # [total_rows] int32
    total_rows,
    max_ctx,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_model_len,
    block_size: tl.constexpr,
    max_blocks_per_seq,
    num_phys_blocks,
    stride_q_row,
    stride_qs_row,
    stride_kv_flat,
    stride_bt_row,
    stride_out_row,
    stride_w_row,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    IS_MXFP4: tl.constexpr,
    USE_FP8_BATCHED_DOT: tl.constexpr,
):
    """Per-tile kernel: each program processes one BLOCK_KV tile for one row."""
    kv_block = tl.program_id(0)
    row_idx = tl.program_id(1)

    if row_idx >= total_rows:
        return

    ctx_len = tl.load(Ctx_lens_ptr + row_idx)
    kv_start = kv_block * BLOCK_KV
    if kv_start >= ctx_len:
        return

    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    bt_row_base = Block_tables_ptr + row_idx * stride_bt_row
    out_row_base = Output_ptr + row_idx * stride_out_row

    h_ids = tl.arange(0, num_heads)
    d_ids = tl.arange(0, BLOCK_D)

    # Pre-load Q as FP8 [num_heads, head_dim], or dequantize packed MXFP4 Q.
    if IS_MXFP4:
        d2_ids = tl.arange(0, BLOCK_D // 2)
        q_offsets = h_ids[:, None] * (head_dim // 2) + d2_ids[None, :]
        q_packed = tl.load(q_row_base + q_offsets)  # [num_heads, head_dim//2] uint8

        lo = q_packed & 0xF
        hi = (q_packed >> 4) & 0xF
        nibble = tl.reshape(tl.join(lo, hi), [num_heads, head_dim])

        q_f32 = _e2m1_to_f32(nibble)  # [num_heads, head_dim] fp32

        block_id = (d_ids // 32)[None, :]  # [1, head_dim]
        q_scale_val = tl.load(
            Q_scale_ptr + row_idx * stride_qs_row + h_ids
        )  # [num_heads] int32
        byte = ((q_scale_val[:, None] >> (8 * block_id)) & 0xFF).to(tl.float32)
        scale = tl.exp2(byte - 127.0)  # [num_heads, head_dim]

        q_param = (q_f32 * scale).to(tl.float16)  # [num_heads, head_dim]
    elif USE_FP8_BATCHED_DOT:
        q_offsets = h_ids[:, None] * head_dim + d_ids[None, :]
        q_u8 = tl.load(q_row_base + q_offsets)
        q_param = q_u8.to(tl.float8e4nv, bitcast=True)
    else:
        # FP8 Q per-head dots load inside the loop, thus pass the base addr
        q_param = q_row_base

    # Pre-load weights: [num_heads] float32
    w_all = tl.load(w_row_base + tl.arange(0, num_heads))

    end_pos = tl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size

    p_ids = tl.arange(0, block_size)

    for blk_idx in range(NUM_BLOCKS):
        lb = first_lb + blk_idx
        logical_base = lb * block_size
        if logical_base < end_pos:
            # Block-table lookup for physical block index
            phys_block = tl.load(bt_row_base + lb)
            phys_block = tl.maximum(phys_block, 0)
            phys_block = tl.minimum(phys_block, num_phys_blocks - 1)
            flat_base = phys_block * block_size

            # Coalesced KV load: [block_size, head_dim] as FP8
            kv_offsets = (flat_base + p_ids[:, None]) * stride_kv_flat + d_ids[None, :]
            kv_u8 = tl.load(KV_data_ptr + kv_offsets)

            # Coalesced scale load: [block_size] float32
            scale_tile = tl.load(KV_scales_ptr + flat_base + p_ids)

            if IS_MXFP4:
                # Tensor-core MMA: Q[H, D] @ KV[block_size, D]^T -> [H, block_size]
                kv_fp16 = kv_u8.to(tl.float8e4nv, bitcast=True).to(tl.float16)
                dots = tl.dot(q_param, tl.trans(kv_fp16))
                scores = tl.maximum(dots * scale_tile[None, :], 0.0)
                weighted = scores * w_all[:, None]
                output_tile = tl.sum(weighted, axis=0)
            else:
                output_tile = _fp8_dot(
                    q_param,
                    kv_u8,
                    w_all,
                    scale_tile,
                    h_ids,
                    d_ids,
                    num_heads,
                    head_dim,
                    block_size,
                    w_row_base,
                )

            pos_ids = logical_base + p_ids
            valid_mask = pos_ids < end_pos
            tl.store(out_row_base + pos_ids, output_tile, mask=valid_mask)


def _preprocess_kv_cache(kv_cache, block_tables, context_lens, total_rows, next_n_val):
    """Reshape paged KV cache from [num_blocks, block_size, 1, D+4] uint8
    into flat data [flat_size, D] and scales [flat_size] arrays.
    """
    num_phys_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    D = kv_cache.shape[3] - 4

    flat_size = num_phys_blocks * block_size
    block_stride = block_size * (D + 4)

    kv_flat = kv_cache.reshape(num_phys_blocks, block_stride)

    kv_data = kv_flat[:, : block_size * D].reshape(num_phys_blocks, block_size, D)
    kv_data = kv_data.reshape(flat_size, D).contiguous()

    # Extract per-token FP32 scales from the trailing 4 bytes per token
    scale_bytes = kv_flat[:, block_size * D :].reshape(num_phys_blocks, block_size, 4)
    kv_scales = (
        scale_bytes.contiguous()
        .reshape(flat_size, 4)
        .view(torch.float32)
        .reshape(flat_size)
        .contiguous()
    )

    if block_tables.dim() == 2:
        B = block_tables.shape[0]
        block_tables_expanded = (
            block_tables.unsqueeze(1)
            .expand(B, next_n_val, -1)
            .reshape(total_rows, -1)
            .contiguous()
            .to(torch.int32)
        )
    else:
        block_tables_expanded = block_tables.contiguous().to(torch.int32)

    max_ctx = int(context_lens.max().item())

    return kv_data, kv_scales, block_tables_expanded, max_ctx


def _select_block_kv(max_ctx, block_size):
    """Adaptive BLOCK_KV selection based on context length.

    Uses 4 levels to balance SM utilization across production context-length
    distribution (1k through 64k).
    """
    if max_ctx <= 2048:
        block_kv = 256
    elif max_ctx <= 4096:
        block_kv = 512
    elif max_ctx <= 8192:
        block_kv = 1024
    else:
        block_kv = 2048
    num_blocks = block_kv // block_size
    return block_kv, num_blocks


# =============================================================================
# TLE fast path — TMA + WGMMA
# =============================================================================

_TLE_NUM_STAGES = 2
_TLE_NUM_WARPS = 4


@triton.jit
def _mqa_logits_kernel_tle(
    Q_ptr,  # [total_rows, num_heads * head_dim] uint8 (FP8)
    KV_scales_ptr,  # [num_phys_blocks * BLOCK_SIZE] float32
    Weights_ptr,  # [total_rows, num_heads] float32
    Block_tables_ptr,  # [total_rows, max_blocks_per_seq] int32
    Output_ptr,  # [total_rows, max_model_len] float32
    Ctx_lens_ptr,  # [total_rows] int32
    kv_desc,  # TMA descriptor over [num_phys_blocks, block_size, head_dim] fp8
    total_rows,
    max_ctx,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_model_len,
    block_size: tl.constexpr,
    num_phys_blocks,
    stride_q_row,
    stride_bt_row,
    stride_out_row,
    stride_w_row,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    KV_BYTES: tl.constexpr,
):
    """TLE fast path: each KV page lands in SMEM via the TMA engine (bulk
    async copy, no per-lane register round-trip); WGMMA reads both operands
    directly from shared memory. The page loop is pipelined with
    tl.range(num_stages=NUM_STAGES) so the next page's TMA fill overlaps the
    current page's WGMMA + epilogue.

    The fast path fires only inside the measured win band and shares the
    operator's BLOCK_KV selection, so it is bit-identical to the baseline
    where it runs (see _can_use_tle).
    """
    kv_block = tl.program_id(0)
    row_idx = tl.program_id(1)
    if row_idx >= total_rows:
        return

    ctx_len = tl.load(Ctx_lens_ptr + row_idx)
    kv_start = kv_block * BLOCK_KV
    if kv_start >= ctx_len:
        return

    q_row_base = Q_ptr + row_idx * stride_q_row
    w_row_base = Weights_ptr + row_idx * stride_w_row
    bt_row_base = Block_tables_ptr + row_idx * stride_bt_row
    out_row_base = Output_ptr + row_idx * stride_out_row

    h_ids = tl.arange(0, num_heads)
    d_ids = tl.arange(0, BLOCK_D)
    p_ids = tl.arange(0, block_size)

    # Q into SMEM once — WGMMA operand A
    q_u8 = tl.load(q_row_base + h_ids[:, None] * head_dim + d_ids[None, :])
    q_fp8 = q_u8.to(tl.float8e4nv, bitcast=True)
    w_all = tl.load(w_row_base + tl.arange(0, num_heads))

    q_smem = tle.gpu.alloc(
        [num_heads, BLOCK_D], dtype=tl.float8e4nv, scope=tle.gpu.smem, layout=None
    )
    kv_smem = tle.gpu.alloc(
        [block_size, BLOCK_D], dtype=tl.float8e4nv, scope=tle.gpu.smem, layout=None
    )
    qr = tl.broadcast_to(h_ids[:, None], (num_heads, BLOCK_D))
    qc = tl.broadcast_to(d_ids[None, :], (num_heads, BLOCK_D))
    tl.store(tle.gpu.local_ptr(q_smem, (qr, qc)), q_fp8)

    kv_full = tle.gpu.alloc_barrier(expect_bytes=KV_BYTES)

    end_pos = tl.minimum(kv_start + BLOCK_KV, ctx_len)
    first_lb = kv_start // block_size

    for blk_idx in tl.range(0, NUM_BLOCKS, num_stages=NUM_STAGES):
        lb = first_lb + blk_idx
        # Clamp the block-table read so a partial trailing tile never goes OOB;
        # out-of-range pages compute garbage but are masked out of the store.
        phys_block = tl.load(bt_row_base + tl.minimum(lb, stride_bt_row - 1))
        phys_block = tl.maximum(phys_block, 0)
        phys_block = tl.minimum(phys_block, num_phys_blocks - 1)

        tle.gpu.copy(
            kv_desc,
            kv_smem,
            [1, block_size, BLOCK_D],
            [phys_block, 0, 0],
            barrier=kv_full,
        )
        tle.gpu.barrier_wait(kv_full, phaseIdx=blk_idx)

        # WGMMA: Q[heads, D] @ KV[page, D]^T — both read from SMEM
        dots = tle.gpu.wgmma(q_smem, kv_smem, out_dtype=tl.float32, trans_b=True)
        dots = tle.gpu.wgmma_wait(0, dots)

        scale_tile = tl.load(KV_scales_ptr + phys_block * block_size + p_ids)
        scores = tl.maximum(dots * scale_tile[None, :], 0.0)
        weighted = scores * w_all[:, None]
        output_tile = tl.sum(weighted, axis=0)

        pos_ids = lb * block_size + p_ids
        valid_mask = pos_ids < end_pos
        tl.store(out_row_base + pos_ids, output_tile, mask=valid_mask)


def _build_kv_descriptor(kv_data, num_phys_blocks, block_size, head_dim):
    """TMA descriptor over the paged KV cache [num_phys_blocks, block_size, D] fp8.

    kv_data is [num_phys_blocks * block_size, head_dim] uint8 (fp8 bit-cast);
    reinterpret as the 3D fp8 view and wrap it in a TMA descriptor.
    """
    kv3d = kv_data.view(num_phys_blocks, block_size, head_dim)
    kv3d = kv3d.view(torch.float8_e4m3fn)
    return TensorDescriptor.from_tensor(kv3d, block_shape=[1, block_size, head_dim])


def _can_use_tle(max_ctx, block_size, D) -> bool:
    """Deterministic TMA fast-path guard (fixed, empirical — NOT autotune).

    TMA + WGMMA pays off when the KV page is 256 tokens (32 KB fp8 tile) and
    the page loop is long enough for pipelining to reach steady state. Measured
    win band (H800, block_size=256, D=128): max_ctx in [12288, 16384], i.e.
    BLOCK_KV=2048 with 6-8 pages per CTA. Everything else keeps the baseline.
    The fast path shares the operator's BLOCK_KV selection, so it is
    bit-identical to the baseline wherever it fires.
    """
    if not (HAS_TLE and _tle_enabled()):
        return False
    if get_device_capability() < (9, 0):
        return False
    if D != 128 or block_size != 256:
        return False
    return 12288 <= max_ctx <= 16384


def _launch_tle_kernel(
    q_u8,
    kv_data,
    kv_scales,
    weights,
    block_tables,
    logits,
    ctx_lens,
    total_rows,
    max_ctx,
    H,
    D,
    max_model_len,
    block_size,
    num_phys_blocks,
    max_blocks_per_seq,
    block_kv,
    num_blocks,
):
    """Launch the TMA fast path with the operator's BLOCK_KV geometry."""
    kv_desc = _build_kv_descriptor(kv_data, num_phys_blocks, block_size, D)
    grid = (triton.cdiv(max_ctx, block_kv), total_rows)
    _mqa_logits_kernel_tle[grid](
        q_u8,
        kv_scales,
        weights,
        block_tables,
        logits,
        ctx_lens,
        kv_desc,
        total_rows=total_rows,
        max_ctx=max_ctx,
        num_heads=H,
        head_dim=D,
        max_model_len=max_model_len,
        block_size=block_size,
        num_phys_blocks=num_phys_blocks,
        stride_q_row=H * D,
        stride_bt_row=max_blocks_per_seq,
        stride_out_row=max_model_len,
        stride_w_row=H,
        BLOCK_KV=block_kv,
        BLOCK_D=128,
        NUM_BLOCKS=num_blocks,
        NUM_STAGES=_TLE_NUM_STAGES,
        KV_BYTES=block_size * D,
        num_warps=_TLE_NUM_WARPS,
    )
    return logits


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
):
    """Compute paged MQA logits from FP8/MXFP4 queries against FP8 KV cache.

    Args:
        q: Tuple of (q_values, q_scale).
            FP8 path: q_values [B, next_n, H, D] float8_e4m3fn, q_scale is None.
            FP4 path: q_values [B, next_n, H, D//2] uint8 (packed E2M1),
                q_scale [B, next_n, H] int32 (D//32 ue8m0 bytes per token-head).
        kv_cache: [num_blocks, block_size, 1, D+4] uint8 paged KV cache (fp8 K
            + trailing per-token fp32 scale; K stays fp8 in both paths).
        weights: [B*next_n, H] float32 per-head weights.
        context_lens: [B] or [B, next_n] int32 context lengths.
        block_tables: [B, max_blocks] int32 block table mapping.
        schedule_metadata: Metadata from get_paged_mqa_logits_metadata (unused
            by Triton kernel, kept for API compatibility with vLLM).
        max_model_len: Maximum model sequence length.
        clean_logits: If True, initialize output with -inf instead of 0.

    Returns:
        Logits tensor [total_rows, max_model_len] float32.
    """
    logger.debug("GEMS FP8_FP4_PAGED_MQA_LOGITS")

    q_values, q_scale = q
    is_fp4 = q_scale is not None

    if q_values.dim() == 3:
        q_values = q_values.unsqueeze(1)

    B, next_n_val, H, q_last_dim = q_values.shape
    total_rows = B * next_n_val

    block_size = kv_cache.shape[1]
    head_dim = kv_cache.shape[3] - 4
    if is_fp4:
        assert q_last_dim == head_dim // 2
    else:
        assert q_last_dim == head_dim

    if context_lens.dim() == 2:
        ctx_lens_flat = (
            context_lens.reshape(-1)[:total_rows].contiguous().to(torch.int32)
        )
    else:
        ctx_lens_flat = (
            context_lens.repeat_interleave(next_n_val).contiguous().to(torch.int32)
        )

    kv_data, kv_scales, block_tables_expanded, max_ctx = _preprocess_kv_cache(
        kv_cache, block_tables, ctx_lens_flat, total_rows, next_n_val
    )

    q_flat = q_values.reshape(total_rows, H, q_last_dim).contiguous()
    q_u8 = q_flat.view(torch.uint8).reshape(total_rows, H * q_last_dim)
    stride_q_row = H * q_last_dim

    if is_fp4:
        q_scale_flat = q_scale.reshape(total_rows, H).contiguous().to(torch.int32)
    else:
        # Dummy scale tensor; the kernel never dereferences it when not IS_MXFP4.
        q_scale_flat = torch.empty(
            (total_rows, H), dtype=torch.int32, device=q_values.device
        )

    logits = torch.full(
        (total_rows, max_model_len),
        float("-inf") if clean_logits else 0.0,
        device=q_values.device,
        dtype=torch.float32,
    )

    BLOCK_D = 128
    BLOCK_KV, NUM_BLOCKS = _select_block_kv(max_ctx, block_size)

    num_phys_blocks = kv_cache.shape[0]
    max_blocks_per_seq = block_tables_expanded.shape[1]

    grid = (triton.cdiv(max_ctx, BLOCK_KV), total_rows)
    use_tle = _can_use_tle(max_ctx, block_size, head_dim)
    if use_tle:
        _launch_tle_kernel(
            q_u8,
            kv_data,
            kv_scales,
            weights,
            block_tables_expanded,
            logits,
            ctx_lens_flat,
            total_rows,
            max_ctx,
            H,
            head_dim,
            max_model_len,
            block_size,
            num_phys_blocks,
            max_blocks_per_seq,
            BLOCK_KV,
            NUM_BLOCKS,
        )
    else:
        _mqa_logits_kernel[grid](
            q_u8,
            q_scale_flat,
            kv_data,
            kv_scales,
            weights,
            block_tables_expanded,
            logits,
            ctx_lens_flat,
            total_rows=total_rows,
            max_ctx=max_ctx,
            num_heads=H,
            head_dim=head_dim,
            max_model_len=max_model_len,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
            num_phys_blocks=num_phys_blocks,
            stride_q_row=stride_q_row,
            stride_qs_row=H,
            stride_kv_flat=head_dim,
            stride_bt_row=max_blocks_per_seq,
            stride_out_row=max_model_len,
            stride_w_row=H,
            BLOCK_KV=BLOCK_KV,
            BLOCK_D=BLOCK_D,
            NUM_BLOCKS=NUM_BLOCKS,
            IS_MXFP4=is_fp4,
            USE_FP8_BATCHED_DOT=_use_fp8_batched_dot,
        )

    return logits
