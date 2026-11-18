# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_with_kvcache_sparse_w8a8_fp8 import (
    SUPPORTED,
    assert_accuracy,
    make_inputs,
    prepare_flash_mla_with_kvcache_sparse_w8a8_fp8,
    reference,
)

from . import base, conftest, consts
from .test_blas_perf_parallel import ParallelBenchmarkMixin


# Both paths receive the same frozen fixture and prepared-call boundary.
def run_native(inputs, query, cache, handle):
    return handle["baseline"]()


def run_gems(inputs, query, cache, handle):
    return handle["gems"]()


class SparseMLAFP8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_sparse_w8a8_fp8", run_native, [torch.bfloat16]
        )
        self.set_gems(run_gems)

    def get_input_iter(self, dtype):
        fixture_shapes = self.shapes
        if conftest.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            with (
                Path(__file__)
                .with_name("flash_mla_sparse_fp8_shapes.json")
                .open() as source
            ):
                fixture_shapes = self.shapes + [
                    tuple(record["shape"]) for record in json.load(source)["records"]
                ]

        @lru_cache(maxsize=3)
        def fixture(key):
            queries, heads, topk, pages, rope_dim = key
            max_batch = max(
                shape[0] for shape in fixture_shapes if tuple(shape[1:]) == key
            )
            return make_inputs(max_batch, queries, heads, topk, pages, dtype, rope_dim)

        for batch, queries, heads, topk, pages, rope_dim in self.shapes:
            key = queries, heads, topk, pages, rope_dim
            full_inputs, full_query, cache = fixture(key)
            inputs = [
                tensor[:batch] if index in (0, 1, 4, 6) else tensor
                for index, tensor in enumerate(full_inputs)
            ]
            query = full_query[:batch]
            from importlib import import_module

            adapter = import_module(
                "flaggems_vllm.runtime.backend."
                f"_{flaggems_vllm.vendor_name}.testing."
                "flash_mla_with_kvcache_sparse_w8a8_fp8"
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            expected = reference(inputs, reference_device=query.device)
            gems_handle, actual = prepare_flash_mla_with_kvcache_sparse_w8a8_fp8(
                *inputs
            )
            assert_accuracy(actual, expected)
            handle = {
                "gems": gems_handle,
                "baseline": adapter.prepare_baseline(inputs, query),
            }
            assert_accuracy(
                run_native(inputs, query, cache, handle), expected, output_atol=2e-3
            )
            yield inputs, query, cache, {"handle": handle}

    def set_more_shapes(self):
        if (
            Path(conftest.Config.shape_file).resolve()
            != Path(__file__).with_name("core_shapes.yaml").resolve()
        ):
            return []
        with (
            Path(__file__)
            .with_name("flash_mla_sparse_fp8_shapes.json")
            .open() as source
        ):
            return [tuple(record["shape"]) for record in json.load(source)["records"]]

    def record_shapes(self, inputs, query, cache, handle):
        return (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            inputs[-1].shape[-1],
            cache.shape[0],
            inputs[1].shape[-1],
        )


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_fp8
@pytest.mark.skipif(not SUPPORTED, reason="No registered sparse FP8 KVCache MLA")
def test_flash_mla_with_kvcache_sparse_w8a8_fp8():
    SparseMLAFP8Benchmark().run()
