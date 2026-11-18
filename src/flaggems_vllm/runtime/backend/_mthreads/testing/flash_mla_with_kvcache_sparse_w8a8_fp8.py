# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def prepare_baseline(inputs, query):
    """Prepare native MODEL1 D512 or V3.2 D576 MATE FP8 cache.

    Power-of-two fixture scales preserve the dequantized FP8 mathematical input
    in BF16 query/tail storage. MODEL1 pages place all576-byte token payloads
    before8-byte token exponent records; the seven64-element groups use the
    original token scale. Packing and metadata are outside prepared timings.
    """
    from mate.flashmla import flash_mla_with_kvcache, get_mla_metadata

    _, qrope, kv, krope, _, ks, indices = inputs
    batch, queries, heads, _ = query.shape
    pages = kv.shape[0]
    raw = kv.view(pages, 64, 1, 512)
    token_scale = ks.view(pages, 64, 1, 1)
    if qrope.shape[-1] == 0:
        packed = torch.empty((pages, 64, 1, 584), dtype=torch.uint8, device=kv.device)
        page = packed.view(pages, -1)
        payload = page[:, : 64 * 576].view(pages, 64, 1, 576)
        exponents = page[:, 64 * 576 :].view(pages, 64, 1, 8)
        for first in range(0, pages, 256):
            last = min(first + 256, pages)
            values = raw[first:last]
            scales = token_scale[first:last]
            payload[first:last, ..., :448].copy_(values[..., :448].view(torch.uint8))
            payload[first:last, ..., 448:].copy_(
                (values[..., 448:].float() * scales)
                .to(torch.bfloat16)
                .view(torch.uint8)
            )
            exponent = ((scales.view(torch.int32) >> 23) & 255).to(torch.uint8)
            exponents[first:last, ..., :7].copy_(
                exponent.expand(last - first, 64, 1, 7)
            )
            exponents[first:last, ..., 7:].zero_()
    else:
        rope = krope.view(pages, 64, 1, 64)
        packed = torch.empty((pages, 64, 1, 656), dtype=torch.uint8, device=kv.device)
        for first in range(0, pages, 256):
            last = min(first + 256, pages)
            scales = token_scale[first:last]
            packed[first:last, ..., :512].copy_(raw[first:last].view(torch.uint8))
            packed[first:last, ..., 512:528].copy_(
                scales.expand(last - first, 64, 1, 4).contiguous().view(torch.uint8)
            )
            packed[first:last, ..., 528:].copy_(
                (rope[first:last].float() * scales).to(torch.bfloat16).view(torch.uint8)
            )
    scale = (512 + qrope.shape[-1]) ** -0.5
    # muDNN FillOp fails for some real batch sizes. Prepare the equivalent
    # explicit lengths by host-to-device copy before either path is timed.
    topk_length = torch.full(
        (batch,), indices.shape[-1], dtype=torch.int32, device="cpu"
    ).to(query.device)
    if qrope.shape[-1] == 0:
        extra_topk_length = torch.zeros((batch,), dtype=torch.int32, device="cpu").to(
            query.device
        )
        extra_cache, extra_indices = packed, indices[..., :0]
    else:
        extra_topk_length = None
        extra_cache, extra_indices = None, None
    plans = []
    # The installed MODEL1 fails at batch 16384. Time every original query
    # through bounded native calls, including copies of the complete outputs.
    for first in range(0, batch, 8192):
        last = min(first + 8192, batch)
        metadata, splits = get_mla_metadata(
            None,
            queries * heads,
            1,
            num_heads_q=heads,
            is_fp8_kvcache=True,
            topk=indices.shape[-1],
            q=query[first:last],
            bs=last - first,
            topk_length=topk_length[first:last],
            extra_topk_length=(
                None if extra_topk_length is None else extra_topk_length[first:last]
            ),
        )
        plans.append((first, last, metadata, splits))
    if len(plans) == 1:
        output, lse = None, None
    else:
        output = torch.empty(
            (batch, queries, heads, 512), dtype=query.dtype, device=query.device
        )
        lse = torch.empty(
            (batch, heads, queries), dtype=torch.float32, device=query.device
        )

    def run():
        for first, last, metadata, splits in plans:
            values, logsum = flash_mla_with_kvcache(
                query[first:last],
                packed,
                None,
                None,
                512,
                metadata,
                splits,
                softmax_scale=scale,
                is_fp8_kvcache=True,
                indices=indices[first:last],
                topk_length=topk_length[first:last],
                extra_topk_length=(
                    None if extra_topk_length is None else extra_topk_length[first:last]
                ),
                extra_k_cache=extra_cache,
                extra_indices_in_kvcache=(
                    None if extra_indices is None else extra_indices[first:last]
                ),
            )
            if len(plans) == 1:
                return values, logsum
            output[first:last].copy_(values)
            lse[first:last].copy_(logsum)
        return output, lse

    return run
