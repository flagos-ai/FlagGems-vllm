# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops.flash_mla_with_kvcache_w8a8 import (
    FlashMLAInt8PreparedHandle,
    NativeCachedHandle,
    NativeSmallHandle,
    SmallKernel,
)
from flaggems_vllm.utils import libentry

MAX_SPARSE_SPLITS = 24


@libentry()
@triton.jit
def initialize_sparse_lengths(
    Counts, ROWS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    tl.store(Counts + offsets, K, offsets < ROWS)


@libentry()
@triton.jit
def compact_sparse_indices(
    Indices,
    Lengths,
    Table,
    Counts,
    K: tl.constexpr,
    CAP: tl.constexpr,
    TOKENS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), K) if HAS_LENGTH else K
    selected = tl.load(Indices + row * K + offsets, offsets < K, -1)
    valid = (offsets < length) & (selected >= 0) & (selected < TOKENS)
    ranks = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    tl.store(Table + row * CAP + offsets, 0, (offsets >= count) & (offsets < CAP))
    tl.store(Table + row * CAP + ranks, selected, valid)
    tl.store(Counts + row, count)


@libentry()
@triton.jit
def sparse_output_epilogue(
    Source,
    LSE,
    Sink,
    Output,
    H: tl.constexpr,
    SQ: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, 512)
    values = tl.load(Source + (row * H + head) * 512 + dims)
    if HAS_SINK:
        lse = tl.load(LSE + row * H + head)
        sink = tl.load(Sink + head)
        # Empty rows use +inf LSE and must remain zero even for an infinite sink.
        weight = tl.where(lse == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink - lse)))
        values *= weight
    tl.store(Output + (row * H + head) * 512 + dims, values)


class SparseMLAInt8PreparedHandle:
    """Reuse workspace; calls must be serialized on one stream. Indices may change."""

    def __init__(self, tensors: tuple[torch.Tensor, ...], lengths, sink, out, scale):
        q, qr, kv, kr, qs, ks, indices = tensors
        self.batch, self.queries, self.heads, _ = q.shape
        self.indices, self.lengths, self.sink = indices, lengths, sink
        self.capacity = max(64, triton.cdiv(indices.shape[-1], 64) * 64)
        rows = self.batch * self.queries
        self.tokens = kv.shape[0] * 64
        self.uses_raw_indices = (
            indices.shape[-1] > 0
            and indices.shape[-1] % 64 == 0
            and lengths is None
            and self.tokens > 0
        )
        self.counts = torch.empty((rows,), dtype=torch.int32, device=q.device)
        if self.uses_raw_indices:
            self.table = indices.view(rows, self.capacity // 64, 64)
            if rows > 0 and self.heads > 0:
                with torch_device_fn.device(q.device):
                    initialize_sparse_lengths[(1,)](
                        self.counts,
                        rows,
                        indices.shape[-1],
                        triton.next_power_of_2(rows),
                        num_warps=4,
                    )
        else:
            self.table = torch.empty(
                (rows, self.capacity // 64, 64), dtype=torch.int32, device=q.device
            )
        self.needs_epilogue = sink is not None or out is not None
        self.native = None
        if rows > 0 and self.heads > 0:
            native_tensors = (
                q.view(rows, 1, self.heads, 512),
                qr.view(rows, 1, self.heads, qr.shape[-1]),
                kv,
                kr,
                qs.view(rows, 1, self.heads, 1),
                ks,
                self.table,
                self.counts,
            )
            if self.heads <= 32:
                self.native = NativeSmallHandle(
                    native_tensors,
                    scale,
                    False,
                    is_sparse=True,
                    uses_raw_indices=self.uses_raw_indices,
                )
            elif rows <= 32:
                if self.heads == 64 and self.capacity >= 4096:
                    tile = 32
                    # Fill two CTA slots per SM without excessive partial-buffer traffic.
                    resident_splits = max(
                        1,
                        2
                        * _get_num_sms(q.device)
                        // (rows * triton.cdiv(self.heads, tile)),
                    )
                    splits = min(
                        MAX_SPARSE_SPLITS, self.capacity // 64, resident_splits
                    )
                elif self.heads == 64 and self.capacity <= 1024:
                    tile = 16
                    resident_splits = max(
                        1,
                        2
                        * _get_num_sms(q.device)
                        // (rows * triton.cdiv(self.heads, tile)),
                    )
                    splits = (
                        1
                        if self.capacity <= 128
                        else min(self.capacity // 64, resident_splits)
                    )
                else:
                    tile = 16 if rows <= 8 and self.capacity <= 1024 else 32
                    target_programs = (
                        256 if self.capacity >= 4096 else min(128, 16 * rows)
                    )
                    splits = min(
                        16,
                        triton.next_power_of_2(self.capacity // 64),
                        triton.next_power_of_2(
                            max(
                                1,
                                triton.cdiv(
                                    target_programs,
                                    rows * triton.cdiv(self.heads, tile),
                                ),
                            )
                        ),
                    )
                self.native = FlashMLAInt8PreparedHandle(
                    native_tensors,
                    splits,
                    scale,
                    False,
                    is_sparse=True,
                    uses_raw_indices=self.uses_raw_indices,
                )
                self.native.head_tile = tile
                self.native.normalized_partials = True
                self.native.grid = (triton.cdiv(self.heads, tile), rows, splits)
                self.native.kernel = SmallKernel(tile, self.native)
            else:
                self.native = NativeCachedHandle(
                    native_tensors,
                    scale,
                    False,
                    is_sparse=True,
                    uses_raw_indices=self.uses_raw_indices,
                )
            self.output = (
                self.native.output.view(self.batch, self.queries, self.heads, 512)
                if out is None
                else out
            )
            self.lse = self.native.lse.view(
                self.batch, self.queries, self.heads
            ).transpose(1, 2)
        else:
            self.output = (
                out
                if out is not None
                else torch.empty(
                    (self.batch, self.queries, self.heads, 512),
                    dtype=qr.dtype,
                    device=q.device,
                )
            )
            self.lse = torch.empty(
                (self.batch, self.heads, self.queries),
                dtype=torch.float32,
                device=q.device,
            )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.native is None:
            return self.output, self.lse
        with torch_device_fn.device(self.indices.device):
            if not self.uses_raw_indices:
                compact_sparse_indices[(self.batch * self.queries,)](
                    self.indices,
                    self.lengths,
                    self.table,
                    self.counts,
                    self.indices.shape[-1],
                    self.capacity,
                    self.tokens,
                    self.lengths is not None,
                    triton.next_power_of_2(self.capacity),
                    num_warps=4,
                )
            native_output, _ = self.native()
            if self.needs_epilogue:
                sparse_output_epilogue[(self.batch * self.queries, self.heads)](
                    native_output,
                    self.native.lse,
                    self.sink,
                    self.output,
                    self.heads,
                    self.queries,
                    self.sink is not None,
                    num_warps=4,
                )
        return self.output, self.lse


def prepare_flash_mla_with_kvcache_sparse_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[SparseMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Sparse INT8 MLA over physical page64 cache indices.

    RoPE is divided by the corresponding content scale, as in dense W8A8 MLA.
    Negative/out-of-range indices are ignored; duplicates remain independent.
    Scales must be finite and positive. LSE excludes the optional attention sink.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if heads not in (0, 16, 32) and (heads < 64 or heads % 64 != 0):
        raise NotImplementedError("requires 16, 32, or a multiple of 64 query heads")
    if (
        k_nope.ndim not in (3, 4)
        or k_nope.shape[1] != 64
        or k_nope.shape[-1] != 512
        or (k_nope.ndim == 4 and k_nope.shape[2] != 1)
    ):
        raise NotImplementedError("requires KV [pages, 64, 512] or [pages, 64, 1, 512]")
    if q_nope.dtype != torch.int8 or k_nope.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE must have matching BF16 or FP16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if (
        indices.ndim != 3
        or indices.shape[:2] != (batch, queries)
        or indices.shape[-1] > 65536
    ):
        raise ValueError("indices must have shape [batch, queries, topk <= 65536]")
    if indices.dtype != torch.int32:
        raise TypeError("indices must be INT32")
    if (
        q_rope.ndim != 4
        or q_rope.shape[-1] not in (0, 64)
        or q_rope.shape != (batch, queries, heads, q_rope.shape[-1])
        or k_rope.shape != (*k_nope.shape[:-1], q_rope.shape[-1])
        or q_scale.shape != (batch, queries, heads, 1)
        or k_scale.shape != (*k_nope.shape[:-1], 1)
    ):
        raise ValueError(
            "RoPE/scales must match content rows with trailing dimension 0 or 64 / 1"
        )
    tensors = (q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices)
    optional = tuple(t for t in (topk_length, attn_sink, out) if t is not None)
    if q_nope.device.type != "cuda" or any(
        t.device != q_nope.device for t in tensors + optional
    ):
        raise ValueError("all inputs must share a MetaX device")
    if any(not t.is_contiguous() for t in tensors + optional):
        raise NotImplementedError("requires contiguous inputs and output")
    if any(t.numel() and t.data_ptr() % 16 for t in tensors[:6]):
        raise NotImplementedError("requires 16-byte aligned content, RoPE and scales")
    if topk_length is not None and (
        topk_length.shape != (batch, queries) or topk_length.dtype != torch.int32
    ):
        raise ValueError("topk_length must be INT32 [batch, queries]")
    if attn_sink is not None and (
        attn_sink.shape != (heads,) or attn_sink.dtype != torch.float32
    ):
        raise ValueError("attn_sink must be FP32 [heads]")
    if out is not None and (
        out.shape != (batch, queries, heads, 512) or out.dtype != q_rope.dtype
    ):
        raise ValueError("out must match output shape and RoPE dtype")
    if (
        out is not None
        and out.numel()
        and any(
            out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in tensors + optional[:-1]
        )
    ):
        raise ValueError("out must not alias an input")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float))
    ):
        raise TypeError("softmax_scale must be a Python number")
    scale = (
        (512 + q_rope.shape[-1]) ** -0.5
        if softmax_scale is None
        else float(softmax_scale)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if "C550" not in torch.cuda.get_device_name(q_nope.device):
        raise NotImplementedError("native sparse INT8 MLA requires MetaX C550")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare sparse MLA before graph capture, then call the handle"
        )
    rows = batch * queries
    capacity = max(64, triton.cdiv(indices.shape[-1], 64) * 64)
    # Native intermediate indexing is signed INT32; cap transient storage too.
    workspace = rows * capacity * (3 * heads + 4) + rows * heads * 8192
    if (
        rows >= 2**14
        or rows * capacity * max(heads, 1) >= 2**30
        or k_nope.shape[0] * 64 >= 2**31
        or workspace > 8 * 1024**3
    ):
        raise NotImplementedError(
            "native sparse workspace exceeds its indexing or 8 GiB budget"
        )
    handle = SparseMLAInt8PreparedHandle(tensors, topk_length, attn_sink, out, scale)
    return handle, handle()


def flash_mla_with_kvcache_sparse_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, output = prepare_flash_mla_with_kvcache_sparse_w8a8_int8(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=out,
    )
    return output
