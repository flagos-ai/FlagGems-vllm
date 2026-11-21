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

"""MACA KV-cache attention composition; metadata ownership stays with vLLM."""

import functools
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@functools.lru_cache(maxsize=1)
def _kv_api():
    from flash_attn.flash_attn_interface import flash_attn_cuda

    fn = getattr(flash_attn_cuda, "fwd_kvcache", None)
    doc = getattr(fn, "__doc__", "") or ""
    if not callable(fn) or "arg20:" not in doc or "arg21:" in doc:
        raise RuntimeError(
            "Unsupported MACA FlashAttention KV-cache ABI; expected 21 arguments"
        )
    return fn


def is_paged_prefix_window_attention_available():
    try:
        _kv_api()
    except (ImportError, OSError, RuntimeError):
        return False
    return True


@libentry()
@triton.jit
def _merge_prefix_window(
    A,
    B,
    LA,
    LB,
    O,
    PREFIX,
    LENGTHS,
    H: tl.constexpr,
    D: tl.constexpr,
    a0: tl.constexpr,
    a1: tl.constexpr,
    a2: tl.constexpr,
    b0: tl.constexpr,
    b1: tl.constexpr,
    b2: tl.constexpr,
    la0: tl.constexpr,
    la1: tl.constexpr,
    lb0: tl.constexpr,
    lb1: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    o2: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // H, row % H
    d = tl.arange(0, BD)
    has_prefix = tl.load(PREFIX + batch) > 0
    has_tail = tl.load(LENGTHS + batch) > 0
    la = tl.load(LA + batch * la0 + head * la1, has_prefix, -float("inf"))
    lb = tl.load(LB + batch * lb0 + head * lb1, has_tail, -float("inf"))
    la = tl.where(has_prefix, la, -float("inf"))
    lb = tl.where(has_tail, lb, -float("inf"))
    top = tl.maximum(la, lb)
    top = tl.where(has_prefix | has_tail, top, 0.0)
    wa = tl.where(has_prefix, tl.exp(la - top), 0.0)
    wb = tl.where(has_tail, tl.exp(lb - top), 0.0)
    a = tl.load(A + batch * a0 + head * a1 + d * a2, (d < D) & has_prefix, 0).to(
        tl.float32
    )
    b = tl.load(B + batch * b0 + head * b1 + d * b2, (d < D) & has_tail, 0).to(
        tl.float32
    )
    a = tl.where(has_prefix, a, 0.0)
    b = tl.where(has_tail, b, 0.0)
    total = wa + wb
    value = tl.where(total > 0, (wa * a + wb * b) / total, 0.0)
    tl.store(O + batch * o0 + head * o1 + d * o2, value, d < D)


def _validate(q, k, v, lengths, table, prefix, window, out, splits):
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected q[B,H,D] and matching K/V[pages,block,Hkv,D]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("Paged prefix/window attention supports FP16/BF16")
    if q.shape[-1] != 128 or k.shape[-1] != q.shape[-1]:
        raise NotImplementedError(
            "Paged prefix/window attention currently supports D128"
        )
    if not q.is_cuda or any(
        t.device != q.device for t in (k, v, lengths, table, prefix)
    ):
        raise ValueError("All tensors must be on the same CUDA device")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    if k.shape[2] <= 0 or q.shape[1] <= 0 or q.shape[1] % k.shape[2]:
        raise ValueError("The query-head count must be divisible by the KV-head count")
    if k.shape[1] <= 0 or k.shape[1] % 16:
        raise NotImplementedError(
            "The MACA KV page size must be a positive multiple of 16"
        )
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise NotImplementedError("Q/K/V must be contiguous in head_dim")
    batch = q.shape[0]
    for t in (lengths, prefix):
        if t.dtype != torch.int32 or t.shape != (batch,) or not t.is_contiguous():
            raise ValueError("Lengths must be contiguous int32 vectors with B entries")
    if (
        table.dtype != torch.int32
        or table.ndim != 2
        or table.shape[0] != batch
        or table.stride(-1) != 1
    ):
        raise ValueError(
            "The block table must be an int32 matrix with B rows and compact columns"
        )
    if batch and (k.shape[0] == 0 or table.shape[1] == 0):
        raise ValueError("A nonempty batch requires a nonempty cache and block table")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window_size must be a positive integer")
    if (
        isinstance(splits, bool)
        or not isinstance(splits, int)
        or not 0 <= splits <= 128
    ):
        raise ValueError("prefix_splits must be an integer in [0,128]")
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match q shape, dtype and device")
        if any(torch._C._overlaps(out, t) for t in (q, k, v, lengths, table, prefix)):
            raise ValueError("out must not overlap an input tensor")


def paged_prefix_window_attention(
    q,
    k_cache,
    v_cache,
    seq_lens,
    block_table,
    prefix_lens,
    window_size,
    softmax_scale=None,
    *,
    out=None,
    prefix_splits=0,
):
    _validate(
        q,
        k_cache,
        v_cache,
        seq_lens,
        block_table,
        prefix_lens,
        window_size,
        out,
        prefix_splits,
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, (float, int))
        or not math.isfinite(softmax_scale)
    ):
        raise ValueError("softmax_scale must be a finite Python scalar")
    if out is None:
        out = torch.empty_like(q)
    batch, heads, dim = q.shape
    if batch == 0:
        return out
    api = _kv_api()
    with torch_device_fn.device(q.device):
        query = q.unsqueeze(1)

        def attend(lengths, left, right, splits):
            return api(
                query,
                k_cache,
                v_cache,
                None,
                None,
                lengths,
                None,
                None,
                None,
                None,
                block_table,
                None,
                None,
                float(softmax_scale),
                True,
                left,
                right,
                0.0,
                True,
                splits,
                None,
            )

        prefix_out, prefix_lse = attend(prefix_lens, -1, -1, prefix_splits)
        tail_out, tail_lse = attend(seq_lens, window_size - 1, 0, 1)
        if prefix_out.shape != query.shape or tail_out.shape != query.shape:
            raise RuntimeError("Unexpected MACA attention output shape")
        if prefix_lse.shape != (batch, heads, 1) or tail_lse.shape != (batch, heads, 1):
            raise RuntimeError("Unexpected MACA attention LSE shape")
        a, b = prefix_out.squeeze(1), tail_out.squeeze(1)
        la, lb = prefix_lse.squeeze(-1), tail_lse.squeeze(-1)
        # D128 fixes the structural vector width. This MetaX-only composition
        # delegates attention tuning to the SDK; NVIDIA is unsupported, so no
        # NVIDIA autotune path is claimed for this small LSE merge.
        _merge_prefix_window[(batch * heads,)](
            a,
            b,
            la,
            lb,
            out,
            prefix_lens,
            seq_lens,
            heads,
            dim,
            *a.stride(),
            *b.stride(),
            *la.stride(),
            *lb.stride(),
            *out.stride(),
            triton.next_power_of_2(dim),
            num_warps=4,
        )
    return out
