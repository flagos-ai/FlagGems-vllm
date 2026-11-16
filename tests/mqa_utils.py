# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test-only MQA references, paged-cache storage and graph replay helpers."""

import pytest
import torch


@pytest.fixture
def restore_rng_state():
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)


def check_replays(call, update, check, outputs=(), *, values=(0.5, -0.75), warmup=3):
    """Capture once, mutate inputs and poison outputs before each replay.

    ``outputs`` may be a callable when ``call`` allocates its result. Set
    ``warmup=0`` to exercise a first invocation inside graph capture.
    """
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in values:
        update(value)
        for output in outputs() if callable(outputs) else outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


def pack_fp8_paged_cache(keys, scales, *, padding=0, storage_offset=0, fill_value=0):
    """Pack each page as FP8 values followed by FP32 per-token scales.

    Return both the cache view and its backing bytes so tests can check page
    padding and storage-offset sentinels. This does not quantize the inputs.
    """
    pages, block, width = keys.shape
    pitch = block * (width + 4) + padding
    raw = torch.full(
        (storage_offset + pages * pitch,),
        fill_value,
        device=keys.device,
        dtype=torch.uint8,
    )
    cache = raw.as_strided(
        (pages, block, 1, width + 4),
        (pitch, width + 4, width + 4, 1),
        storage_offset=storage_offset,
    )
    flat = cache.view(pages, -1)
    flat[:, : block * width].copy_(keys.view(torch.uint8).reshape(pages, -1))
    flat[:, block * width :].view(torch.float32).copy_(scales)
    return cache, raw


def shared_mqa_reference(q, keys, weights, starts, ends, *, scales=None, clean=True):
    """CPU FP32 oracle; the legacy FP8 API applies scales after the dot."""
    scores = torch.einsum("mhd,nd->mhn", q.cpu().float(), keys.cpu().float())
    if scales is not None:
        scores = scores * scales.cpu()[None, None, :]
    result = (scores.relu() * weights.cpu()[:, :, None]).sum(1)
    if clean:
        columns = torch.arange(keys.shape[0])[None, :]
        valid = (columns >= starts.cpu()[:, None]) & (columns < ends.cpu()[:, None])
        result.masked_fill_(~valid, -torch.inf)
    return result


def bf16_cache_reference(q, keys, scales, weights, lengths):
    """CPU oracle with BF16 key/scale multiplication rounded before the dot.

    Keep this distinct from the FP8-query reference: moving the scale after
    the dot would change the BF16-query contract's rounding semantics.
    """
    dequant = (keys.cpu().bfloat16() * scales.cpu().bfloat16()[..., None]).bfloat16()
    scores = torch.einsum("bhd,bnd->bhn", q.cpu().float(), dequant.float()).relu()
    expected = (scores * weights.cpu()[:, :, None]).sum(1)
    columns = torch.arange(keys.shape[1])[None, :]
    return expected.masked_fill(columns >= lengths.cpu()[:, None], -torch.inf)
