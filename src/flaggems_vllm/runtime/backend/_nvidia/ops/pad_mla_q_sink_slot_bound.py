# SPDX-License-Identifier: Apache-2.0
"""Bit-preserving MLA head padding and last-valid-slot endpoint transport."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


def _validate(q, sink, indices, out_q, out_sink, out_bound):
    tensors = (q, sink, indices, out_q, out_sink, out_bound)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("tensor inputs and caller-allocated outputs are required")
    if any(t.layout != torch.strided for t in tensors):
        raise NotImplementedError("dense strided tensors are required")
    if q.ndim != 3 or sink.ndim != 1 or indices.ndim != 2:
        raise ValueError("expected Q rank3, sink rank1, indices rank2")
    tokens = q.shape[0]
    if not isinstance(tokens, int) or not 1 <= tokens <= 2048:
        raise NotImplementedError("supported token range is 1..2048")
    if q.shape != (tokens, 4, 576) or sink.shape != (4,):
        raise ValueError("expected Q[T,4,576] and sink[4]")
    if indices.shape != (tokens, 2048):
        raise ValueError("expected indices[T,2048]")
    if (
        out_q.shape != (tokens, 64, 576)
        or out_sink.shape != (64,)
        or out_bound.shape != (tokens,)
    ):
        raise ValueError("caller output geometry does not match the input")
    expected = (
        torch.bfloat16,
        torch.float32,
        torch.int32,
        torch.bfloat16,
        torch.float32,
        torch.int32,
    )
    if any(t.dtype != dtype for t, dtype in zip(tensors, expected)):
        raise NotImplementedError("BF16 Q, FP32 sink, and I32 indices/bound only")
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise NotImplementedError("all tensors must share one CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise NotImplementedError("contiguous input and output tensors required")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise NotImplementedError("autograd is unsupported")
    if any(t.is_neg() or t.is_conj() for t in tensors):
        raise NotImplementedError("ordinary inference tensors required")
    outputs = (out_q, out_sink, out_bound)
    for i, value in enumerate(outputs):
        if any(
            torch._C._overlaps(value, other)
            for other in (q, sink, indices, *outputs[:i])
        ):
            raise ValueError("outputs must be disjoint from inputs and one another")
    return tokens


# Fixed launch preserves the qualified raw-bit transport and endpoint reduction.
# Retuning this migration requires separate numerical and performance checks.
@libentry()
@triton.jit
def _pad_mla_q_sink_slot_bound_kernel(
    Q,
    SINK,
    INDICES,
    OUT_Q,
    OUT_SINK,
    OUT_BOUND,
    Q_BLOCKS,
    BLOCK_Q: tl.constexpr,
    SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < Q_BLOCKS:
        offsets = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
        row_offset = (pid * BLOCK_Q) % (64 * 576)
        # The fixed 36864-element output row is exactly 36 blocks.
        if row_offset >= 4 * 576:
            bits = tl.full((BLOCK_Q,), 0, tl.uint16)
        else:
            token = offsets // (64 * 576)
            head_col = offsets % (64 * 576)
            bits = tl.load(
                Q.to(tl.pointer_type(tl.uint16)) + token * (4 * 576) + head_col,
                mask=head_col < 4 * 576,
                other=0,
            )
        tl.store(OUT_Q.to(tl.pointer_type(tl.uint16)) + offsets, bits)
        if pid == 0:
            heads = tl.arange(0, 64)
            sink_bits = tl.load(
                SINK.to(tl.pointer_type(tl.uint32)) + heads,
                mask=heads < 4,
                other=0xFF800000,
            )
            tl.store(OUT_SINK.to(tl.pointer_type(tl.uint32)) + heads, sink_bits)
    else:
        row = pid - Q_BLOCKS
        slots = tl.arange(0, SLOTS)
        indices = tl.load(INDICES + row * SLOTS + slots)
        endpoint = tl.max(tl.where(indices >= 0, slots + 1, 0), axis=0)
        tl.store(OUT_BOUND + row, endpoint)


def pad_mla_q_sink_slot_bound(q, sink, indices, out_q, out_sink, out_bound):
    """Write Q/sink padding and the final nonnegative index slot plus one.

    Inputs: contiguous BF16 Q[T,4,576], FP32 sink[4], I32 indices[T,2048].
    Outputs: contiguous BF16[T,64,576], FP32[64], I32[T] on the same CUDA device.
    T is 1..2048. The caller allocates ordinary independent output tensors and
    keeps inputs live and unchanged through completion. Outputs are overwritten.

    Real Q/sink lanes preserve raw bits. Padded Q lanes are exact +0 and padded
    sinks exact -Inf. Empty index rows give 0; holes never change slot order.
    No float arithmetic, device-value inspection, or Torch compute fallback.
    """
    logger.debug("GEMS_NVIDIA PAD_MLA_Q_SINK_SLOT_BOUND")
    tokens = _validate(q, sink, indices, out_q, out_sink, out_bound)
    q_blocks = 36 * tokens
    with torch.cuda.device(q.device):
        _pad_mla_q_sink_slot_bound_kernel[(q_blocks + tokens,)](
            q,
            sink,
            indices,
            out_q,
            out_sink,
            out_bound,
            q_blocks,
            BLOCK_Q=1024,
            SLOTS=2048,
            num_warps=4,
            num_stages=1,
            num_ctas=1,
        )
    return out_q, out_sink, out_bound


__all__ = ["pad_mla_q_sink_slot_bound"]
