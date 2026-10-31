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

import math
import random

import pytest
import torch

import flaggems_vllm
from tests.flash_mla_with_kvcache_w8a8_utils import (
    DenseW8A8Inputs,
    assert_dense_fp8_close,
    assert_dense_fp8_replay,
    get_backend_adapter,
)

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin

ADAPTER = get_backend_adapter()
SUPPORTED = ADAPTER is not None and ADAPTER.supported()
PREPARED_HANDLES = {}
UPSTREAM_SHAPES = [
    (batch, query_length, length, heads, varlen)
    for batch in [128]
    for length in [4096, 8192, 16384]
    for heads in [16, 32, 64, 128]
    for query_length in [1, 2]
    for varlen in [False, True]
]


def make_input(shape: tuple) -> DenseW8A8Inputs:
    batch, query_length, mean_length, heads, varlen = shape
    torch.manual_seed(0)
    lengths = [mean_length] * batch
    if varlen:
        # Replay the upstream main's single seeded length stream for independent cases.
        generator = random.Random(0)
        for upstream_shape in UPSTREAM_SHAPES:
            source_batch, source_queries, source_mean, _, source_varlen = upstream_shape
            if not source_varlen:
                continue
            lengths = [
                max(
                    int(generator.normalvariate(source_mean, source_mean / 2)),
                    source_queries,
                )
                for _ in range(source_batch)
            ]
            if upstream_shape == tuple(shape):
                break
        else:
            raise ValueError(
                "FP8 benchmark shape must be from the upstream test matrix"
            )
    padded_length = math.ceil(max(lengths) / 256) * 256
    device = flaggems_vllm.device
    gpu_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
    query = torch.randn(
        batch, query_length, heads, 576, dtype=torch.bfloat16, device=device
    )
    table = torch.arange(
        batch * padded_length // 64, dtype=torch.int32, device=device
    ).view(batch, padded_length // 64)
    cache = torch.randn(table.numel(), 64, 1, 576, dtype=torch.bfloat16, device=device)
    for row, length in enumerate(lengths):
        cache.view(batch, padded_length, 1, 576)[row, length:] = float("nan")
    q_nope, q_rope, q_scale, _ = ADAPTER.quantize(query, benchmark=True)
    k_nope, k_rope, k_scale, _ = ADAPTER.quantize(cache, benchmark=True)
    inputs = DenseW8A8Inputs(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        table,
        gpu_lengths,
        512,
        None,
        None,
        tuple(lengths),
        benchmark_shape=tuple(shape),
    )
    return ADAPTER.prepare_baseline(inputs)


def run_baseline(inputs):
    return ADAPTER.run_baseline(inputs)


def run_gems(inputs):
    key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr(), inputs.q_nope.shape)
    prepared = PREPARED_HANDLES.get(key)
    if prepared is None:
        PREPARED_HANDLES.clear()
        prepared = ADAPTER.prepare_gems(inputs)
        PREPARED_HANDLES[key] = prepared
    return prepared.run()


class FlashMLAWithKVCacheW8A8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__("flash_mla_with_kvcache_w8a8", run_baseline, [torch.bfloat16])
        self.set_gems(run_gems)

    def get_input_iter(self, dtype):
        _ = dtype
        for shape in self.shapes:
            inputs = make_input(shape)
            self.reference = ADAPTER.benchmark_reference(inputs)
            assert_dense_fp8_replay(
                lambda: run_gems(inputs),
                self.reference,
                output_atol=ADAPTER.OUTPUT_ATOL,
                capture=ADAPTER.capture,
            )
            yield (inputs,)

    def get_latency(self, op, *args, **kwargs):
        latency = super().get_latency(op, *args, **kwargs)
        if op is self.gems_op:
            inputs = args[0]
            key = (
                inputs.q_nope.data_ptr(),
                inputs.k_nope.data_ptr(),
                inputs.q_nope.shape,
            )
            prepared = PREPARED_HANDLES[key]
            assert_dense_fp8_close(
                (prepared.output, prepared.lse),
                self.reference,
                output_atol=ADAPTER.OUTPUT_ATOL,
            )
        return latency

    def record_shapes(self, inputs):
        return inputs.benchmark_shape


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(
    not SUPPORTED, reason="No registered quantized dense MLA implementation"
)
def test_flash_mla_with_kvcache_w8a8():
    FlashMLAWithKVCacheW8A8Benchmark().run()
