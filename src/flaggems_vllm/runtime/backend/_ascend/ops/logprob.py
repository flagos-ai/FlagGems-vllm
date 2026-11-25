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

"""Ascend top-k selection for ``compute_topk_scores``.

The shared kernels pack ``(value_bits, token_id)`` into ``uint64`` keys and run
``tl.topk`` / ``tl.sort`` over them.  Ascend cannot lower integer sort/topk:
``bishengir-compile`` aborts with
``mapCompareModeHFusionToHiVM: unsupported hfusion::CompareFn``.  This module
selects with the backend's float32 hardware sort and assembles the int64 token
indices with Triton kernels, so the shared operator file stays vendor-agnostic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._ascend.ops import topk as _ascend_topk
from flaggems_vllm.utils import libentry

_SELECTION_BLOCK = 1024
_ASSEMBLE_BLOCK = 1024


@libentry()
@triton.jit
def _selection_logits_kernel(
    logits,
    stride,
    out,
    vocab,
    BLOCK: tl.constexpr,
):
    """Cast a logits row to FP32 and canonicalize NaN before sorting.

    The shared implementation bit-packs the float value, which ranks a positive
    NaN above every finite value and a negative NaN below every finite value.
    The hardware sort leaves NaN comparisons undefined, so NaNs are mapped to
    ``inf`` / ``-inf`` here; the caller gathers the original values afterwards.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    value = tl.load(logits + row * stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    bits = value.to(tl.int32, bitcast=True)
    negative = bits < 0
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    value = tl.where(nan & ~negative, float("inf"), value)
    value = tl.where(nan & negative, float("-inf"), value)
    tl.store(out + row * vocab + offs, value, mask=mask)


@libentry()
@triton.jit
def _assemble_indices_kernel(
    topk_indices,
    sampled_token_ids,
    out,
    k,
    out_stride,
    PACK_SAMPLED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < k
    indices = tl.load(topk_indices + row * k + cols, mask=mask, other=0)
    tl.store(out + row * out_stride + cols + PACK_SAMPLED, indices.to(tl.int64), mask=mask)
    if PACK_SAMPLED:
        sampled = tl.load(sampled_token_ids + row)
        tl.store(out + row * out_stride, sampled.to(tl.int64))


def _launch_topk_indices(
    logits: torch.Tensor,
    k: int,
    sampled_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[batch, k (+ 1)]`` int64 token indices for the top-k selection."""
    batch, vocab = logits.shape
    pack_sampled = sampled_token_ids is not None
    selection = torch.empty(
        (batch, vocab), dtype=torch.float32, device=logits.device
    )
    _selection_logits_kernel[(batch, triton.cdiv(vocab, _SELECTION_BLOCK))](
        logits,
        logits.stride(0),
        selection,
        vocab,
        BLOCK=_SELECTION_BLOCK,
    )
    _, topk_indices = _ascend_topk.topk(selection, k)
    out = torch.empty(
        (batch, k + (1 if pack_sampled else 0)),
        dtype=torch.int64,
        device=logits.device,
    )
    _assemble_indices_kernel[(batch, triton.cdiv(k, _ASSEMBLE_BLOCK))](
        topk_indices,
        sampled_token_ids if pack_sampled else topk_indices,
        out,
        k,
        out.stride(0),
        PACK_SAMPLED=pack_sampled,
        BLOCK=_ASSEMBLE_BLOCK,
    )
    return out
