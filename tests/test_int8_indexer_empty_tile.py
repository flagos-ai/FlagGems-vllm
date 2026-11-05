# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged score math with independent INT64/FP64 accumulation."""

import importlib

import pytest
import torch

import flaggems_vllm


def fixture(rows, width):
    page_size, heads, pages = 64, 32, 4
    generator = torch.Generator().manual_seed(3026 + rows)
    q = torch.randint(
        -127, 128, (rows, heads, 128), generator=generator, dtype=torch.int8
    )
    values = torch.randint(
        -127, 128, (pages, page_size, 128), generator=generator, dtype=torch.int8
    )
    scales = (
        torch.tensor([0.125, 0.25, 0.5, 1.0])[:, None].expand(pages, page_size).clone()
    )
    cache = torch.empty((pages, page_size, 132), dtype=torch.uint8)
    flat = cache.view(pages, -1)
    flat[:, : page_size * 128].copy_(values.view(torch.uint8).reshape(pages, -1))
    flat[:, page_size * 128 :].copy_(scales.view(torch.uint8))
    table = torch.full((2, (width + page_size - 1) // page_size), -1, dtype=torch.int32)
    table[:, :3] = torch.tensor([[0, -1, 2], [3, 1, 0]], dtype=torch.int32)
    row_map = torch.arange(rows, dtype=torch.int64) % 2
    lengths = torch.full((rows,), 191, dtype=torch.int32)
    if rows:
        lengths[0] = 0
    if rows > 1:
        row_map[1] = -1
    weights = torch.randn((rows, heads), generator=generator)
    return q, cache, weights, table, row_map, lengths, width, page_size


def reference(args):
    q, cache, weights, table, row_map, lengths, width, page_size = args
    flat = cache.view(cache.shape[0], -1)
    keys = flat[:, : page_size * 128].view(torch.int8).reshape(-1, page_size, 128)
    scales = flat[:, page_size * 128 :].view(torch.float32)
    gold = torch.full((q.shape[0], width), -torch.inf, dtype=torch.float64)
    bounds = torch.zeros_like(gold)
    for row, batch in enumerate(row_map.tolist()):
        if batch < 0:
            continue
        for col in range(min(int(lengths[row]), width)):
            page = int(table[batch, col // page_size])
            if page < 0:
                continue
            dot = (
                q[row].to(torch.int64) * keys[page, col % page_size].to(torch.int64)
            ).sum(-1)
            terms = dot.clamp_min(0).double() * weights[row].double()
            scale = float(scales[page, col % page_size])
            gold[row, col] = terms.sum() * scale
            # Multiplications and a H-head FP32 reduction bound, independent
            # of the implementation's grouping and dot launch schedule.
            bounds[row, col] = (
                (q.shape[1] + 2)
                * torch.finfo(torch.float32).eps
                * terms.abs().sum()
                * abs(scale)
            )
    return gold, bounds


@pytest.mark.int8_indexer_paged_logits_empty_tile
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize(
    "rows,width,pure_prefill",
    [(0, 16384, False), (4, 32768, False), (4, 204800, False), (256, 16384, True)],
)
@torch.inference_mode()
def test_empty_tiles_and_replayed_inputs(rows, width, pure_prefill, monkeypatch):
    cpu = list(fixture(rows, width))
    coarse_calls = []
    if pure_prefill:
        cpu[3] = cpu[3][:1].contiguous()
        cpu[4] = torch.zeros_like(cpu[4])
        cpu[4][1] = -1
        module = importlib.import_module(
            "flaggems_vllm.runtime.backend._nvidia.ops.int8_indexer_empty_tile"
        )
        original = module._paged_logits_coarse32

        class RecordingKernel:
            def __getitem__(self, grid):
                selected = original[grid]

                def launch(*args, **kwargs):
                    coarse_calls.append(grid)
                    return selected(*args, **kwargs)

                return launch

        monkeypatch.setattr(module, "_paged_logits_coarse32", RecordingKernel())
    cpu = tuple(cpu)
    device = tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value for value in cpu
    )

    def run():
        return flaggems_vllm.int8_indexer_paged_logits_empty_tile(
            *device, pure_prefill=pure_prefill
        )

    def check(actual, expected_cpu):
        gold, bound = reference(expected_cpu)
        observed = actual.cpu().double()
        assert torch.equal(torch.isneginf(observed), torch.isneginf(gold))
        valid = torch.isfinite(gold)
        assert torch.all((observed[valid] - gold[valid]).abs() <= bound[valid])

    output = run()
    check(output, cpu)
    if pure_prefill and torch.cuda.get_device_capability() == (9, 0):
        assert coarse_calls, "qualified pure prefill must execute the coarse kernel"
    if not rows:
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
    torch.cuda.current_stream().wait_stream(stream)
    changed = list(cpu)
    changed[0] = -cpu[0]
    device[0].copy_(changed[0])
    captured.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    check(captured, tuple(changed))
