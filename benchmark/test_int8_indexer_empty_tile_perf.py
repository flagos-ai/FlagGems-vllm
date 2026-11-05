# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged scores, including complete output writes and caller-owned cache."""

import pytest
import torch

import flaggems_vllm

from . import base


def reference(q, cache, weights, table, rows, lengths, width, page_size, pure_prefill):
    return flaggems_vllm.int8_indexer_paged_logits(
        q, cache, weights, table, rows, lengths, width, page_size
    )


def target(q, cache, weights, table, rows, lengths, width, page_size, pure_prefill):
    return flaggems_vllm.int8_indexer_paged_logits_empty_tile(
        q,
        cache,
        weights,
        table,
        rows,
        lengths,
        width,
        page_size,
        pure_prefill=pure_prefill,
    )


class EmptyTileBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(256, 16384, True), (4, 32768, False), (4, 204800, False)]

    def get_input_iter(self, dtype):
        for tokens, width, pure_prefill in self.shapes:
            q = torch.randint(
                -127, 128, (tokens, 32, 128), dtype=torch.int8, device="cuda"
            )
            cache = torch.zeros((4, 64, 132), dtype=torch.uint8, device="cuda")
            weights = torch.randn((tokens, 32), device="cuda")
            table = torch.full(
                (1, (width + 63) // 64), -1, dtype=torch.int32, device="cuda"
            )
            table[:, :4] = torch.arange(4, dtype=torch.int32, device="cuda")
            rows = torch.zeros((tokens,), dtype=torch.int64, device="cuda")
            lengths = torch.full((tokens,), 256, dtype=torch.int32, device="cuda")
            yield q, cache, weights, table, rows, lengths, width, 64, pure_prefill


@pytest.mark.int8_indexer_paged_logits_empty_tile
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="CUDA required",
)
@torch.inference_mode()
def test_int8_indexer_empty_tile_perf():
    bench = EmptyTileBenchmark(
        op_name="int8_indexer_paged_logits_empty_tile",
        torch_op=reference,
        dtypes=[torch.int8],
    )
    bench.set_gems(target)
    bench.run()
