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

"""Thin MACA variable-length paged causal attention adapter."""

import functools
import math

import torch

from flaggems_vllm.runtime import torch_device_fn


@functools.lru_cache(maxsize=1)
def _varlen_api():
    from flash_attn.flash_attn_interface import flash_attn_cuda

    fn = getattr(flash_attn_cuda, "varlen_fwd", None)
    doc = getattr(fn, "__doc__", "") or ""
    if not callable(fn) or "arg22:" not in doc or "arg23:" in doc:
        raise RuntimeError("Unsupported MACA varlen ABI; expected 23 arguments")
    return fn


def is_paged_causal_attention_available():
    try:
        _varlen_api()
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def _validate(q, k, v, starts, lengths, table, max_q, max_k, out):
    if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected Q[T,H,D] and matching K/V[pages,block,Hkv,D]")
    if q.dtype not in (torch.float16, torch.bfloat16) or q.shape[-1] != 128:
        raise NotImplementedError("Paged causal attention supports FP16/BF16 D128")
    if any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError("Paged causal attention is inference-only")
    if k.dtype != q.dtype or v.dtype != q.dtype or k.shape[-1] != 128:
        raise TypeError("Q/K/V must have the same dtype and head dimension")
    if not q.is_cuda or any(
        t.device != q.device for t in (k, v, starts, lengths, table)
    ):
        raise ValueError("All inputs must be on the same CUDA device")
    if k.shape[2] <= 0 or q.shape[1] <= 0 or q.shape[1] % k.shape[2]:
        raise ValueError("Q heads must be divisible by the KV-head count")
    if k.shape[1] <= 0 or k.shape[1] % 16:
        raise NotImplementedError("The KV page size must be a positive multiple of 16")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise NotImplementedError("Q/K/V must be contiguous in head_dim")
    if lengths.ndim != 1 or lengths.dtype != torch.int32 or not lengths.is_contiguous():
        raise ValueError("seq_lens must be a contiguous int32 vector")
    batch = lengths.numel()
    if (
        starts.shape != (batch + 1,)
        or starts.dtype != torch.int32
        or not starts.is_contiguous()
    ):
        raise ValueError(
            "Query offsets must be a contiguous int32 vector with B+1 entries"
        )
    if (
        table.ndim != 2
        or table.shape[0] != batch
        or table.dtype != torch.int32
        or table.stride(-1) != 1
    ):
        raise ValueError("The block table must be an int32 matrix with B compact rows")
    for name, value in (("max_query_len", max_q), ("max_seq_len", max_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(name + " must be a nonnegative host integer")
    if q.shape[0] and (
        not batch or not max_q or not max_k or not k.shape[0] or not table.shape[1]
    ):
        raise ValueError(
            "Nonempty queries require nonempty cache, metadata and positive maxima"
        )
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match Q shape, dtype and device")
        if out.stride(-1) != 1:
            raise NotImplementedError("out must be contiguous in head_dim")
        if any(torch._C._overlaps(out, t) for t in (q, k, v, starts, lengths, table)):
            raise ValueError("out must not overlap any input")


def paged_causal_attention(
    q,
    k_cache,
    v_cache,
    query_start_loc,
    seq_lens,
    block_table,
    max_query_len,
    max_seq_len,
    softmax_scale=None,
    *,
    out=None,
):
    _validate(
        q,
        k_cache,
        v_cache,
        query_start_loc,
        seq_lens,
        block_table,
        max_query_len,
        max_seq_len,
        out,
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if (
        isinstance(softmax_scale, bool)
        or not isinstance(softmax_scale, (int, float))
        or not math.isfinite(softmax_scale)
    ):
        raise ValueError("softmax_scale must be a finite host scalar")
    if out is None:
        out = torch.empty_like(q)
    if q.shape[0] == 0:
        return out
    with torch_device_fn.device(q.device):
        # In paged mode cache lengths determine the K ranges. Reuse the
        # immutable query offsets for the required, unused cumulative-K arg.
        result = _varlen_api()(
            q,
            k_cache,
            v_cache,
            out,
            query_start_loc,
            query_start_loc,
            seq_lens,
            None,
            block_table,
            None,
            max_query_len,
            max_seq_len,
            0.0,
            float(softmax_scale),
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            None,
            None,
            False,
        )
    value = result[0]
    if value.shape != q.shape or value.dtype != q.dtype or value.device != q.device:
        raise RuntimeError("Unexpected MACA paged causal attention output")
    if value.data_ptr() != out.data_ptr():
        raise RuntimeError("MACA paged causal attention did not honor output storage")
    return out
