# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact FP32 top-k with vLLM's in-place cooperative_topk interface.

Contract reference: vllm-project/vllm@90309f6b742cdb1d6305778b81063918cf61be81,
csrc/cooperative_topk.cu. Kernel boundaries synchronize the wide-row radix
passes, so this implementation does not require CUDA thread-block clusters.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

# The 8-partition path stores uint32 histograms and independent per-pass
# states in the supplied 16 KiB/row workspace. The 64-partition path fills that
# workspace with uint16 histograms and allocates a small separate uint32 state.
# Separate pass states make every autotuned launch safe to repeat.
_WORKSPACE_BYTES = 16384


@triton.jit
def _ordered_key(values):
    bits = values.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    ordered = tl.where((bits & 0x80000000) != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)
    # Integer comparisons preserve subnormals even on hardware using FTZ.
    # Canonicalize both signs of zero and NaN to match torch.topk ordering.
    ordered = tl.where(magnitude == 0, 0x80000000, ordered)
    return tl.where(magnitude > 0x7F800000, 0xFFFFFFFF, ordered).to(tl.uint32)


@triton.jit
def _radix_digit(histogram, remaining, BINS: tl.constexpr = 256):
    bins = tl.arange(0, BINS)
    suffix = tl.cumsum(histogram, reverse=True)
    digit = tl.max(tl.where(suffix >= remaining, bins, 0), 0)
    above = tl.sum(tl.where(bins > digit, histogram, 0), 0)
    return digit.to(tl.uint32), remaining - above


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_small"),
    key=["WIDTH", "K", "ROWS"],
    strategy=["default", "default", "default"],
)
@triton.jit
def _cooperative_topk_small(
    Logits,
    Lengths,
    Output,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    if length <= K:
        short_indices = tl.arange(0, K)
        tl.store(
            Output + row * K + short_indices,
            tl.where(short_indices < length, short_indices, -1),
        )
    else:
        offsets = tl.arange(0, BLOCK)
        valid = offsets < length
        values = tl.load(Logits + row * STRIDE + offsets, valid, other=0)
        keys = _ordered_key(values)
        prefix = tl.full((), 0, tl.uint32)
        remaining = tl.full((), K, tl.int32)
        candidates = valid
        for shift in tl.static_range(24, -1, -8):
            digits = ((keys >> shift) & 255).to(tl.int32)
            histogram = tl.histogram(digits, 256, mask=candidates)
            digit, remaining = _radix_digit(histogram, remaining)
            prefix = prefix | (digit << shift)
            candidates = candidates & (digits == digit)
        greater = valid & (keys > prefix)
        equal = valid & (keys == prefix)
        tie_rank = tl.cumsum(equal.to(tl.int32))
        selected = greater | (equal & (tie_rank <= remaining))
        positions = tl.cumsum(selected.to(tl.int32)) - 1
        tl.store(Output + row * K + positions, offsets, selected)


@triton.jit
def _histogram_pointer(Workspace, row, BITS: tl.constexpr):
    if BITS == 7:
        return Workspace.to(tl.pointer_type(tl.uint16)) + row * 8192
    else:
        return Workspace.to(tl.pointer_type(tl.uint32)) + row * 4096


@triton.jit
def _state_pointer(State, row, PARTS: tl.constexpr, BITS: tl.constexpr):
    PASSES: tl.constexpr = triton.cdiv(32, BITS)
    STATE_STRIDE: tl.constexpr = 4096 if BITS == 8 else PASSES * (PARTS + 2) + PARTS
    STATE_OFFSET: tl.constexpr = PARTS * (1 << BITS) if BITS == 8 else 0
    return State.to(tl.pointer_type(tl.uint32)) + row * STATE_STRIDE + STATE_OFFSET


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_histogram"),
    key=["WIDTH", "K", "ROWS", "PARTS", "BITS", "PASS"],
    strategy=["default"] * 6,
)
@triton.jit
def _cooperative_topk_histogram(
    Logits,
    Lengths,
    Workspace,
    State,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
    BITS: tl.constexpr,
    PASS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    BINS: tl.constexpr = 1 << BITS
    PASSES: tl.constexpr = triton.cdiv(32, BITS)
    SHIFT: tl.constexpr = (PASSES - 1 - PASS) * BITS
    STATE_SIZE: tl.constexpr = PARTS + 2
    part = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    scratch = _histogram_pointer(Workspace, row, BITS)
    state = _state_pointer(State, row, PARTS, BITS)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    histogram = tl.full((BINS,), 0, tl.int32)
    if length > K:
        prefix = tl.full((), 0, tl.uint32)
        if PASS > 0:
            prefix = tl.load(state + (PASS - 1) * STATE_SIZE)
        chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((part + 1) * chunk, length)
        for start in range(part * chunk, end, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            valid = offsets < end
            values = tl.load(Logits + row * STRIDE + offsets, valid, other=0)
            keys = _ordered_key(values)
            if PASS > 0:
                valid = valid & ((keys >> (SHIFT + BITS)) == (prefix >> (SHIFT + BITS)))
            digits = ((keys >> SHIFT) & (BINS - 1)).to(tl.int32)
            histogram += tl.histogram(digits, BINS, mask=valid)
    tl.store(scratch + part * BINS + tl.arange(0, BINS), histogram)


# Fixed per-row reduction, exempt from autotuning; the geometry follows the
# workspace protocol. Each pass reads only the preceding immutable state.
@triton.jit
def _cooperative_topk_threshold(
    Workspace,
    State,
    K: tl.constexpr,
    PARTS: tl.constexpr,
    BITS: tl.constexpr,
    PASS: tl.constexpr,
):
    BINS: tl.constexpr = 1 << BITS
    PASSES: tl.constexpr = triton.cdiv(32, BITS)
    STATE_SIZE: tl.constexpr = PARTS + 2
    row = tl.program_id(0)
    scratch = _histogram_pointer(Workspace, row, BITS)
    state_base = _state_pointer(State, row, PARTS, BITS)
    parts = tl.arange(0, PARTS)
    bins = tl.arange(0, BINS)
    histogram = tl.load(scratch + parts[:, None] * BINS + bins[None, :]).to(tl.int32)
    prefix = tl.full((), 0, tl.uint32)
    remaining = tl.full((), K, tl.int32)
    greater = tl.full((PARTS,), 0, tl.int32)
    if PASS > 0:
        previous = state_base + (PASS - 1) * STATE_SIZE
        prefix = tl.load(previous)
        remaining = tl.load(previous + 1).to(tl.int32)
        greater = tl.load(previous + 2 + parts).to(tl.int32)
    digit, remaining = _radix_digit(tl.sum(histogram, 0), remaining, BINS)
    greater += tl.sum(tl.where(bins[None, :] > digit, histogram, 0), 1)
    prefix = prefix | (digit << ((PASSES - 1 - PASS) * BITS))
    state = state_base + PASS * STATE_SIZE
    tl.store(state, prefix)
    tl.store(state + 1, remaining)
    tl.store(state + 2 + parts, greater)
    if PASS == PASSES - 1:
        ties = tl.sum(tl.where(bins[None, :] == digit, histogram, 0), 1)
        tl.store(state_base + PASSES * STATE_SIZE + parts, ties)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cooperative_topk_scatter"),
    key=["WIDTH", "K", "ROWS", "PARTS", "BITS"],
    strategy=["default"] * 5,
)
@triton.jit
def _cooperative_topk_scatter(
    Logits,
    Lengths,
    Output,
    State,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    PASSES: tl.constexpr = triton.cdiv(32, BITS)
    STATE_SIZE: tl.constexpr = PARTS + 2
    part = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), WIDTH)
    if length <= K:
        if part == 0:
            short_indices = tl.arange(0, K)
            tl.store(
                Output + row * K + short_indices,
                tl.where(short_indices < length, short_indices, -1),
            )
    else:
        state_base = _state_pointer(State, row, PARTS, BITS)
        state = state_base + (PASSES - 1) * STATE_SIZE
        threshold = tl.load(state)
        remaining = tl.load(state + 1).to(tl.int32)
        parts = tl.arange(0, PARTS)
        greater_counts = tl.load(state + 2 + parts).to(tl.int32)
        tie_counts = tl.load(state_base + PASSES * STATE_SIZE + parts).to(tl.int32)
        greater_offset = tl.sum(tl.where(parts < part, greater_counts, 0), 0)
        tie_offset = tl.sum(tl.where(parts < part, tie_counts, 0), 0)
        chunk: tl.constexpr = triton.cdiv(WIDTH, PARTS)
        end = tl.minimum((part + 1) * chunk, length)
        for start in range(part * chunk, end, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            valid = offsets < end
            values = tl.load(Logits + row * STRIDE + offsets, valid, other=0)
            keys = _ordered_key(values)
            greater = valid & (keys > threshold)
            equal = valid & (keys == threshold)
            greater_rank = tl.cumsum(greater.to(tl.int32))
            tie_rank = tie_offset + tl.cumsum(equal.to(tl.int32))
            positions = tl.where(
                greater, greater_offset + greater_rank - 1, K - remaining + tie_rank - 1
            )
            selected = greater | (equal & (tie_rank <= remaining))
            tl.store(Output + row * K + positions, offsets, selected)
            greater_offset += tl.sum(greater.to(tl.int32), 0)
            tie_offset += tl.sum(equal.to(tl.int32), 0)


def _memory_interval(tensor):
    if tensor.numel() == 0:
        return 0, 0
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    start = tensor.data_ptr()
    return start, start + span * tensor.element_size()


def cooperative_topk(logits, lengths, output, workspace, k, max_seq_len):
    """Write unsorted top-k indices into ``output`` and return ``None``.

    ``logits`` is FP32 [rows, width] with contiguous columns; ``lengths``
    is contiguous INT32 with one entry per row (rank 1 or 2). Callers must
    supply lengths <= width. Negative lengths are treated as zero. Rows with
    length <= k produce ascending indices followed by -1 padding.

    ``output`` is contiguous INT32 [rows, k]. ``workspace`` is contiguous
    UINT8 with at least rows * 16384 bytes and four-byte alignment. Neither
    mutable buffer may overlap another argument. The 64-partition wide-row
    path additionally allocates 1576 bytes per row of internal state; callers
    still provide the same 16384-byte-per-row workspace. Supports rows <= 64 and
    k in {512, 1024, 2048}. ``max_seq_len`` is ignored, as in vLLM. NaNs
    outrank +inf; tie indices and output order are unspecified. Forward only.
    """
    if logits.device.type != "cuda":
        raise NotImplementedError("cooperative_topk requires a CUDA device")
    if logits.dtype != torch.float32:
        raise NotImplementedError("cooperative_topk supports only float32 logits")
    if logits.layout != torch.strided:
        raise NotImplementedError("logits must use strided layout")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [rows, width]")
    rows, width = logits.shape
    if not isinstance(k, int) or k not in (512, 1024, 2048) or rows > 64:
        raise NotImplementedError(
            "cooperative_topk requires k in {512, 1024, 2048} and rows <= 64"
        )
    if width > 2147483647:
        raise NotImplementedError(
            "cooperative_topk requires int32-representable row lengths"
        )
    if logits.stride(1) != 1 or logits.stride(0) < width:
        raise NotImplementedError(
            "logits must have contiguous columns and non-overlapping rows"
        )
    tensors = {
        "logits": logits,
        "lengths": lengths,
        "output": output,
        "workspace": workspace,
    }
    for name, tensor in tensors.items():
        if tensor.device != logits.device:
            raise ValueError(f"{name} must be on the same device as logits")
        if tensor.layout != torch.strided:
            raise NotImplementedError(f"{name} must use strided layout")
    if (
        lengths.dtype != torch.int32
        or lengths.ndim not in (1, 2)
        or lengths.numel() != rows
    ):
        raise ValueError("lengths must be int32, rank 1 or 2, with one entry per row")
    if output.dtype != torch.int32 or output.shape != (rows, k):
        raise ValueError("output must be int32 with shape [rows, k]")
    if workspace.dtype != torch.uint8 or workspace.numel() < rows * _WORKSPACE_BYTES:
        raise ValueError("workspace must be uint8 with at least rows * 16384 bytes")
    for name in ("lengths", "output", "workspace"):
        if not tensors[name].is_contiguous():
            raise NotImplementedError(f"{name} must be contiguous")
    if workspace.data_ptr() % 4:
        raise ValueError("workspace must be aligned to four bytes")
    intervals = {name: _memory_interval(tensor) for name, tensor in tensors.items()}
    for name in ("output", "workspace"):
        start, end = intervals[name]
        for other, (other_start, other_end) in intervals.items():
            if name != other and start < other_end and other_start < end:
                raise ValueError(f"{name} must not overlap {other}")
    if rows == 0:
        return

    with runtime.torch_device_fn.device(logits.device):
        # Larger register tiles spill heavily on SM120; partition those rows
        # instead of keeping a full 16K-element row live through the scans.
        if width <= 8192:
            _cooperative_topk_small[(rows,)](
                logits,
                lengths,
                output,
                width,
                logits.stride(0),
                k,
                rows,
                BLOCK=triton.next_power_of_2(max(width, 1)),
            )
        else:
            # More partitions expose parallelism for small batches of long rows.
            # Larger batches retain eight partitions to avoid excess CTA overhead.
            use_many_parts = 65536 <= width <= 1048576 and rows <= 16
            parts, bits = (64, 7) if use_many_parts else (8, 8)
            passes = triton.cdiv(32, bits)
            if bits == 7:
                # Bound counts for every distribution, including all-equal rows.
                if triton.cdiv(width, parts) > 65535:
                    raise NotImplementedError(
                        "histogram counters would overflow uint16"
                    )
                state = torch.empty(
                    (rows, passes * (parts + 2) + parts),
                    dtype=torch.int32,
                    device=logits.device,
                )
            else:
                state = workspace
            for radix_pass in range(passes):
                _cooperative_topk_histogram[(parts, rows)](
                    logits,
                    lengths,
                    workspace,
                    state,
                    width,
                    logits.stride(0),
                    k,
                    rows,
                    parts,
                    bits,
                    radix_pass,
                )
                _cooperative_topk_threshold[(rows,)](
                    workspace,
                    state,
                    k,
                    parts,
                    bits,
                    radix_pass,
                    num_warps=4,
                    num_stages=1,
                )
            _cooperative_topk_scatter[(parts, rows)](
                logits,
                lengths,
                output,
                state,
                width,
                logits.stride(0),
                k,
                rows,
                parts,
                bits,
            )
