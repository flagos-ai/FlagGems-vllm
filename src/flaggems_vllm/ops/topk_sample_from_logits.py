# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

_MUSA_LAUNCH_OPTIONS = {"num_warps": 8}
try:
    from triton.backends.mthreads.compiler import MUSAOptions

    if "enable_llc_opt" in MUSAOptions.__dataclass_fields__:
        # MUSA's burst combine can fail to compile this kernel's FP32 selects.
        # Override it at launch when another operator enables it process-wide.
        _MUSA_LAUNCH_OPTIONS["enable_llc_opt"] = False
except (ImportError, AttributeError):
    pass


@triton.jit
def fused_topk_sample_kernel(
    Logits, TopK, Noise, Output, VOCAB: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    logits = tl.load(Logits + row * VOCAB + index, index < VOCAB, other=-float("inf"))
    k = tl.load(TopK + row)
    kth = tl.full((), -float("inf"), tl.float32)
    if k < VOCAB:
        SMALL_K: tl.constexpr = min(BLOCK, 32)
        if k <= SMALL_K:
            small_ranked = tl.topk(logits, SMALL_K)
            ranks = tl.arange(0, SMALL_K)
            kth = tl.max(tl.where(ranks == k - 1, small_ranked, -float("inf")), 0)
        else:
            ranked = tl.sort(logits, descending=True)
            kth = tl.max(tl.where(index == k - 1, ranked, -float("inf")), 0)
    # Keep every logit tied at the kth value, as the vLLM mask does.
    logits = tl.where(logits < kth, -float("inf"), logits)
    has_nan = tl.sum((logits != logits).to(tl.int32), 0) > 0
    peak = tl.max(logits, 0)
    noise = tl.load(Noise + row * VOCAB + index, index < VOCAB, other=1.0)
    weight = tl.exp(logits - peak)
    prob = tl.div_rn(weight, tl.sum(weight, 0))
    score = tl.where(index < VOCAB, tl.div_rn(prob, noise), -float("inf"))
    # vLLM's softmax produces NaNs in these rows and argmax returns index zero.
    result = tl.where(~has_nan & (tl.abs(peak) < float("inf")), tl.argmax(score, 0), 0)
    tl.store(Output + row, result.to(tl.int64))


def topk_sample_from_logits(
    logits: torch.Tensor, k: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Return one int64 token index per row of FP32 MUSA logits.

    All inputs must be contiguous and on the same device. The caller supplies
    normalized integer k values in [1, vocab] and positive FP32 exponential
    draws with the same shape as logits. The supported vocabulary is 1–8192.
    """
    if logits.device.type != "musa":
        raise NotImplementedError("Fused Top-K sampling is supported on MUSA")
    if (
        logits.ndim != 2
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or not 1 <= logits.shape[1] <= 8192
    ):
        raise ValueError("Expected contiguous FP32 logits with vocabulary up to 8192")
    batch, vocab = logits.shape
    if (
        k.shape != (batch,)
        or k.dtype not in (torch.int32, torch.int64)
        or k.device != logits.device
        or not k.is_contiguous()
    ):
        raise ValueError("Expected one contiguous integer k per logit row")
    if (
        noise.shape != logits.shape
        or noise.dtype != logits.dtype
        or noise.device != logits.device
        or not noise.is_contiguous()
    ):
        raise ValueError("Exponential noise must match the logits")

    output = torch.empty((batch,), dtype=torch.int64, device=logits.device)
    if batch:
        with torch.musa.device(logits.device):
            fused_topk_sample_kernel[(batch,)](
                logits,
                k,
                noise,
                output,
                vocab,
                triton.next_power_of_2(vocab),
                **_MUSA_LAUNCH_OPTIONS,
            )
    return output
