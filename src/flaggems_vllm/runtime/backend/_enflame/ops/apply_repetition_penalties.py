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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# GCU400/Libra has 48 SPs and the CTA dispatch wave grows linearly with the
# grid beyond that (~50ns/CTA, see the `main_temperature_kernel` case in the
# knowledge base). The generic (num_seqs, cdiv(vocab, 1024)) grid therefore
# degrades into a micro-CTA tsunami: (64, 32000) alone spawns 2048 tiny CTAs
# and measures ~141us while the whole tensor only needs ~6us of HBM traffic.
#
# Structure (measured on (64, 32000), all dtypes, do_bench w1000/r100 median):
#   * flat arm, total tiles <= 48: one tile per CTA, no loop at all;
#   * row-persistent arm: grid = min(48, num_seqs), each CTA streams whole
#     rows (penalty load + reciprocal hoisted once per row, main_temperature
#     "per-token metadata hoisting" precedent) -> 141 -> ~13.4us;
#   * BLOCK = min(8192, next_pow2(vocab)) beats 4096 (16.2), 16384 (17.2) and
#     32768 (19.3); num_warps=4.
# The penalty chain runs in fp32 (x * (1/p) selects instead of a per-element
# division); 216 probe cases (3 dtypes x 8 shapes x 3 penalties x 3 mask
# modes) are bitwise identical to both the old fp16/fp32-division kernel and
# the fp64 reference rounded to the output dtype.

_NUM_SPC_CAP = 48
_BLOCK_CAP = 8192


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


@triton.jit
def _repetition_penalty_flat_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    penalties_ptr,
    vocab_size,
    tiles_per_seq,
    BLOCK_SIZE: tl.constexpr,
):
    item = tl.program_id(0)
    row = item // tiles_per_seq
    tile = item - row * tiles_per_seq

    penalty = tl.load(penalties_ptr + row).to(tl.float32)
    inv_penalty = 1.0 / penalty

    base = tile * BLOCK_SIZE
    vocab_idx = base + tl.arange(0, BLOCK_SIZE)
    row_base = row * vocab_size
    logits_idx = row_base + vocab_idx

    if base + BLOCK_SIZE <= vocab_size:
        prompt_mask = tl.load(prompt_mask_ptr + logits_idx)
        output_mask = tl.load(output_mask_ptr + logits_idx)
        logits = tl.load(logits_ptr + logits_idx)
        is_repeated = prompt_mask | output_mask
        x = logits.to(tl.float32)
        factor = tl.where(is_repeated, tl.where(x > 0.0, inv_penalty, penalty), 1.0)
        out = x * factor
        tl.store(logits_ptr + logits_idx, out.to(logits_ptr.dtype.element_ty))
    else:
        valid_vocab = vocab_idx < vocab_size
        prompt_mask = tl.load(
            prompt_mask_ptr + logits_idx, mask=valid_vocab, other=False
        )
        output_mask = tl.load(
            output_mask_ptr + logits_idx, mask=valid_vocab, other=False
        )
        logits = tl.load(logits_ptr + logits_idx, mask=valid_vocab, other=0.0)
        is_repeated = prompt_mask | output_mask
        x = logits.to(tl.float32)
        factor = tl.where(is_repeated, tl.where(x > 0.0, inv_penalty, penalty), 1.0)
        out = x * factor
        tl.store(
            logits_ptr + logits_idx,
            out.to(logits_ptr.dtype.element_ty),
            mask=valid_vocab,
        )


@triton.jit
def _repetition_penalty_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    penalties_ptr,
    num_seqs,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)

    for row in range(pid, num_seqs, NUM_SPC):
        penalty = tl.load(penalties_ptr + row).to(tl.float32)
        inv_penalty = 1.0 / penalty
        row_base = row * vocab_size

        for base in range(0, vocab_size, BLOCK_SIZE):
            vocab_idx = base + tl.arange(0, BLOCK_SIZE)
            logits_idx = row_base + vocab_idx
            if base + BLOCK_SIZE <= vocab_size:
                prompt_mask = tl.load(prompt_mask_ptr + logits_idx)
                output_mask = tl.load(output_mask_ptr + logits_idx)
                logits = tl.load(logits_ptr + logits_idx)
                is_repeated = prompt_mask | output_mask
                x = logits.to(tl.float32)
                factor = tl.where(
                    is_repeated, tl.where(x > 0.0, inv_penalty, penalty), 1.0
                )
                out = x * factor
                tl.store(logits_ptr + logits_idx, out.to(logits_ptr.dtype.element_ty))
            else:
                valid_vocab = vocab_idx < vocab_size
                prompt_mask = tl.load(
                    prompt_mask_ptr + logits_idx, mask=valid_vocab, other=False
                )
                output_mask = tl.load(
                    output_mask_ptr + logits_idx, mask=valid_vocab, other=False
                )
                logits = tl.load(logits_ptr + logits_idx, mask=valid_vocab, other=0.0)
                is_repeated = prompt_mask | output_mask
                x = logits.to(tl.float32)
                factor = tl.where(
                    is_repeated, tl.where(x > 0.0, inv_penalty, penalty), 1.0
                )
                out = x * factor
                tl.store(
                    logits_ptr + logits_idx,
                    out.to(logits_ptr.dtype.element_ty),
                    mask=valid_vocab,
                )


def apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties):
    logger.debug("GEMS APPLY REPETITION PENALTIES")
    assert logits.is_contiguous(), "logits must be contiguous"
    assert (
        prompt_mask.is_contiguous() and prompt_mask.dtype == torch.bool
    ), "prompt_mask must be contiguous bool tensor"
    assert (
        output_mask.is_contiguous() and output_mask.dtype == torch.bool
    ), "output_mask must be contiguous bool tensor"
    assert (
        repetition_penalties.is_contiguous()
    ), "repetition_penalties must be contiguous"
    assert logits.dim() == 2, f"logits must be 2D, got {logits.dim()}D"
    assert (
        logits.shape == prompt_mask.shape == output_mask.shape
    ), "shape mismatch between logits and masks"
    assert (
        repetition_penalties.dim() == 1
        and repetition_penalties.numel() == logits.shape[0]
    ), "repetition_penalties must be 1D with length equal to num_seqs"

    num_seqs, vocab_size = logits.shape
    if num_seqs == 0 or vocab_size == 0:
        return None

    BLOCK_SIZE = min(_BLOCK_CAP, _next_pow2(vocab_size))
    tiles_per_seq = triton.cdiv(vocab_size, BLOCK_SIZE)
    total_tiles = num_seqs * tiles_per_seq

    if total_tiles <= _NUM_SPC_CAP:
        grid = (total_tiles,)
        _repetition_penalty_flat_kernel[grid](
            logits,
            prompt_mask,
            output_mask,
            repetition_penalties,
            vocab_size,
            tiles_per_seq,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    else:
        grid = (min(_NUM_SPC_CAP, num_seqs),)
        _repetition_penalty_kernel[grid](
            logits,
            prompt_mask,
            output_mask,
            repetition_penalties,
            num_seqs,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_SPC=grid[0],
            num_warps=4,
        )
    return None


# ---------------------------------------------------------------------------
# vLLM torch.ops registration.
#
# On the GCU there is no CUDA driver, so vLLM's `_C` extension never loads
# and `torch.ops._C.apply_repetition_penalties_` stays unregistered: vLLM's
# `apply_repetition_penalties` wrapper (used by the sampler path and by this
# repo's vs-vLLM tests/benchmarks) then fails with
# `AttributeError: '_OpNamespace' '_C' object has no attribute
# 'apply_repetition_penalties_'`. Register this backend's Triton
# implementation under the same qualified name so the wrapper resolves on
# GCU. Hosts where the native CUDA op is already registered (real CUDA
# boxes) skip the override.
# ---------------------------------------------------------------------------
if not hasattr(torch.ops._C, "apply_repetition_penalties_"):
    _c_lib = torch.library.Library("_C", "FRAGMENT")
    _c_lib.define(
        "apply_repetition_penalties_(Tensor(a!) logits, Tensor prompt_mask, "
        "Tensor output_mask, Tensor repetition_penalties) -> ()"
    )
    _c_lib.impl("apply_repetition_penalties_", apply_repetition_penalties)
