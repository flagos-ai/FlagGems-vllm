# SPDX-License-Identifier: Apache-2.0
"""Raw-bit MLA query concat/head padding and sink padding into owned outputs."""
import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

_BLOCK = 1024
_I32_MAX = (1 << 31) - 1


def _storage_bounds(tensor):
    """Metadata-only allocation bounds and a conservative non-overlap proof."""
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    if any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("pad_mla_q_and_sink requires positive concrete dimensions")
    if any(type(v) is not int or v < 0 or v > _I32_MAX for v in strides):
        raise NotImplementedError("negative or unsafe strides are unsupported")
    span = 1
    for stride, size in sorted(zip(strides, shape)):
        if size <= 1:
            continue
        if stride < span:
            raise NotImplementedError(
                "overlapping/broadcast tensor strides unsupported"
            )
        span += (size - 1) * stride
    offset = tensor.storage_offset()
    last = sum((size - 1) * stride for size, stride in zip(shape, strides))
    if type(offset) is not int or offset < 0 or offset + last > _I32_MAX:
        raise NotImplementedError("tensor storage index span exceeds signed i32")
    storage = tensor.untyped_storage()
    base, nbytes = storage.data_ptr(), storage.nbytes()
    itemsize = tensor.element_size()
    if (
        base <= 0
        or nbytes <= 0
        or base + nbytes >= 1 << 64
        or (offset + last + 1) * itemsize > nbytes
        or tensor.data_ptr() != base + offset * itemsize
        or tensor.data_ptr() % itemsize
    ):
        raise ValueError("invalid tensor storage bounds/alignment")
    return base, base + nbytes


def _source_geometry(q_nope, q_rope, sink):
    values = (q_nope, q_rope, sink)
    if not all(isinstance(t, torch.Tensor) for t in values):
        raise TypeError("concat/pad requires Tensor inputs")
    if any(t.layout != torch.strided for t in values):
        raise NotImplementedError("dense strided inputs required")
    if q_nope.ndim != 3 or q_rope.ndim != 3 or sink.ndim != 1:
        raise ValueError("expected Q_nope/Q_rope rank3 and sink rank1")
    if any(type(dim) is not int or dim <= 0 for t in values for dim in t.shape):
        raise NotImplementedError("positive concrete input dimensions required")
    if (
        q_nope.dtype != torch.bfloat16
        or q_rope.dtype != torch.bfloat16
        or sink.dtype != torch.float32
    ):
        raise NotImplementedError("BF16 Q and FP32 sink only")
    if q_nope.device.type != "cuda" or any(t.device != q_nope.device for t in values):
        raise NotImplementedError("inputs must share one CUDA device")
    if any(t.is_neg() or t.is_conj() for t in values):
        raise NotImplementedError("lazy neg/conj inputs unsupported")
    if torch.is_grad_enabled() and any(t.requires_grad for t in values):
        raise NotImplementedError("inference-only transport")
    m, h, n = q_nope.shape
    r = q_rope.shape[2]
    if q_rope.shape[:2] not in ((m, h), (m, 1)) or sink.shape != (h,):
        raise ValueError("input token/head dimensions disagree")
    for t in values:
        _storage_bounds(t)
    return m, h, n, r


def _validate(q_nope, q_rope, sink, out_q, out_sink):
    m, h, n, r = _source_geometry(q_nope, q_rope, sink)
    outputs = (out_q, out_sink)
    if not all(isinstance(t, torch.Tensor) for t in outputs):
        raise TypeError("caller-owned output Tensors required")
    if any(t.layout != torch.strided or t.device != q_nope.device for t in outputs):
        raise NotImplementedError("dense outputs on the input CUDA device required")
    if out_q.dtype != torch.bfloat16 or out_sink.dtype != torch.float32:
        raise NotImplementedError("output dtypes must match source contract")
    if out_sink.ndim != 1 or out_q.ndim != 3:
        raise ValueError("output rank mismatch")
    if any(type(dim) is not int or dim <= 0 for t in outputs for dim in t.shape):
        raise NotImplementedError("positive concrete output dimensions required")
    hp = out_sink.shape[0]
    if hp < h or out_q.shape != (m, hp, n + r):
        raise ValueError("output geometry mismatch")
    if not out_q.is_contiguous() or not out_sink.is_contiguous():
        raise NotImplementedError("contiguous outputs required")
    if any(t.requires_grad or t.is_neg() or t.is_conj() for t in outputs):
        raise NotImplementedError("ordinary inference output owners required")
    if m * hp * (n + r) > _I32_MAX:
        raise NotImplementedError("output exceeds signed-i32 element span")
    inputs_bounds = [_storage_bounds(t) for t in (q_nope, q_rope, sink)]
    output_bounds = [_storage_bounds(t) for t in outputs]
    for i, (lo, hi) in enumerate(output_bounds):
        for other_lo, other_hi in inputs_bounds + output_bounds[:i]:
            if lo < other_hi and other_lo < hi:
                raise ValueError(
                    "output allocations must be disjoint from inputs and each other"
                )
    return m, h, hp, n, r


# Exploratory baseline schedule exemption is frozen before coding in workflow-design.md.
@libentry()
@triton.jit
def _concat_pad_mla_q_and_sink_kernel(
    Q_NOPE,
    Q_ROPE,
    SINK,
    OUT_Q,
    OUT_SINK,
    NQ,
    H: tl.constexpr,
    HP: tl.constexpr,
    NOPE: tl.constexpr,
    WIDTH: tl.constexpr,
    NS0,
    NS1,
    NS2,
    RS0,
    RS1,
    RS2,
    SS0,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int64) * BLOCK
    i = base + tl.arange(0, BLOCK)
    in_q = i < NQ
    row = HP * WIDTH
    first_row = base // row
    last_row = (base + BLOCK - 1) // row
    whole_padding = (first_row == last_row) & ((base % row) >= H * WIDTH)
    if whole_padding:
        tl.store(
            OUT_Q.to(tl.pointer_type(tl.uint16)) + tl.where(in_q, i, 0),
            tl.full((BLOCK,), 0, tl.uint16),
            mask=in_q,
        )
    else:
        col = i % WIDTH
        head = (i // WIDTH) % HP
        token = tl.where(in_q, i // (HP * WIDTH), 0)
        src_head = tl.where(head < H, head, 0)
        n_offset = token * NS0 + src_head * NS1 + tl.minimum(col, NOPE - 1) * NS2
        r_offset = token * RS0 + src_head * RS1 + tl.maximum(col - NOPE, 0) * RS2
        n_bits = tl.load(
            Q_NOPE.to(tl.pointer_type(tl.uint16)) + n_offset,
            mask=in_q & (head < H) & (col < NOPE),
            other=0,
        )
        r_bits = tl.load(
            Q_ROPE.to(tl.pointer_type(tl.uint16)) + r_offset,
            mask=in_q & (head < H) & (col >= NOPE),
            other=0,
        )
        q_bits = tl.where(col < NOPE, n_bits, r_bits)
        tl.store(
            OUT_Q.to(tl.pointer_type(tl.uint16)) + tl.where(in_q, i, 0),
            q_bits,
            mask=in_q,
        )
    if base < HP:
        sink_bits = tl.load(
            SINK.to(tl.pointer_type(tl.uint32)) + tl.where(i < H, i, 0) * SS0,
            mask=i < H,
            other=0xFF800000,
        )
        tl.store(
            OUT_SINK.to(tl.pointer_type(tl.uint32)) + tl.where(i < HP, i, 0),
            sink_bits,
            mask=i < HP,
        )


def concat_pad_mla_q_and_sink(q_nope, q_rope, sink, out_q, out_sink):
    """Write [Q_nope,Q_rope] real heads and exact +0/-Inf padded lanes.

    All five tensors are read/validated on metadata only. Read-read inputs may
    share backing storage; the two output allocations must be disjoint from all
    inputs and each other. No Torch compute fallback, allocation or float math.
    """
    m, h, hp, n, r = _validate(q_nope, q_rope, sink, out_q, out_sink)
    size = m * hp * (n + r)
    with torch.cuda.device(q_nope.device):
        _concat_pad_mla_q_and_sink_kernel[(triton.cdiv(size, _BLOCK),)](
            q_nope,
            q_rope,
            sink,
            out_q,
            out_sink,
            size,
            h,
            hp,
            n,
            n + r,
            *q_nope.stride(),
            q_rope.stride(0),
            0 if q_rope.shape[1] == 1 else q_rope.stride(1),
            q_rope.stride(2),
            sink.stride(0),
            BLOCK=_BLOCK,
            num_warps=4,
            num_stages=1,
            num_ctas=1,
        )
    return out_q, out_sink


__all__ = ["concat_pad_mla_q_and_sink"]
