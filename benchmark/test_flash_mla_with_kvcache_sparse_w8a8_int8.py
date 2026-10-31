# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_with_kvcache_sparse_w8a8_int8 import (
    SUPPORTED,
    assert_accuracy,
    make_inputs,
    reference,
)

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


def run_native(inputs, query, cache, table, lengths, metadata, splits, valid, handle):
    from flash_mla import flash_mla_with_kvcache

    return flash_mla_with_kvcache(
        query,
        cache,
        table,
        lengths,
        512,
        metadata,
        splits,
        softmax_scale=(512 + inputs[1].shape[-1]) ** -0.5,
        indices=inputs[-1],
        indices_all_valid_per_q=valid,
    )


def run_gems(inputs, query, cache, table, lengths, metadata, splits, valid, handle):
    return handle()


class SparseMLAInt8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_sparse_w8a8_int8", run_native, [torch.bfloat16]
        )
        self.set_gems(run_gems)

    def get_input_iter(self, dtype):
        from flash_mla import get_mla_metadata

        for batch, queries, heads, topk, pages, rope_dim in self.shapes:
            inputs, query, cache = make_inputs(
                batch, queries, heads, topk, pages, dtype, rope_dim
            )
            if rope_dim == 0:
                # User-approved D512 baseline: zero-pad before timing; preserve D512 scale.
                query = torch.nn.functional.pad(query, (0, 64))
                cache = torch.nn.functional.pad(cache, (0, 64))
            device = query.device
            table = torch.arange(pages, dtype=torch.int32, device=device).repeat(
                batch, 1
            )
            lengths = torch.full((batch,), pages * 64, dtype=torch.int32, device=device)
            metadata, splits = get_mla_metadata(
                lengths, queries * heads, 1, heads, False, topk
            )
            valid = torch.zeros((batch, queries, 1), dtype=torch.bool, device=device)
            handle, actual = (
                flaggems_vllm.prepare_flash_mla_with_kvcache_sparse_w8a8_int8(*inputs)
            )
            args = (
                inputs,
                query,
                cache,
                table,
                lengths,
                metadata,
                splits,
                valid,
                handle,
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            expected = reference(inputs)
            assert_accuracy(actual, expected)
            assert_accuracy(run_native(*args), expected)
            yield (*args[:-1], {"handle": handle})

    def record_shapes(
        self, inputs, query, cache, table, lengths, metadata, splits, valid, handle
    ):
        return (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            inputs[-1].shape[-1],
            cache.shape[0],
            inputs[1].shape[-1],
        )


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_int8
@pytest.mark.skipif(not SUPPORTED, reason="No registered sparse INT8 KVCache MLA")
def test_flash_mla_with_kvcache_sparse_w8a8_int8():
    SparseMLAInt8Benchmark().run()
