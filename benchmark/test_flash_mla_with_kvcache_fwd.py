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
from typing import NamedTuple, Optional

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin

HAS_INT8_MLA = "flash_mla_with_kvcache_fwd_w8a8_int8" in dict(
    flaggems_vllm.runtime.backend.get_customized_ops()
)
if HAS_INT8_MLA:
    try:
        from flash_mla import flash_mla_with_kvcache as native_bf16_flash_mla
        from flash_mla import get_mla_metadata as native_bf16_metadata

        HAS_NATIVE_BF16_FLASHMLA = True
    except ImportError:
        HAS_NATIVE_BF16_FLASHMLA = False
else:
    HAS_NATIVE_BF16_FLASHMLA = False
UPSTREAM_SHAPES = [
    (batch, query_length, length, heads, varlen)
    for batch in [128]
    for length in [4096, 8192, 16384]
    for heads in [16, 32, 64, 128]
    for query_length in [1, 2]
    for varlen in [False, True]
]
PREPARED_HANDLES = {}


class DenseInt8Inputs(NamedTuple):
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    value_dim: int
    cuda_metadata: torch.Tensor
    cuda_splits: torch.Tensor
    host_lengths: tuple[int, ...]
    bf16_query: Optional[torch.Tensor]
    bf16_cache: Optional[torch.Tensor]
    benchmark_shape: tuple[int, int, int, int, bool]


def make_input(shape: tuple) -> DenseInt8Inputs:
    batch, query_length, mean_length, heads, varlen = shape
    torch.manual_seed(0)
    lengths = [mean_length] * batch
    if varlen:
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
                "INT8 benchmark shape must be from the upstream test matrix"
            )
    padded_length = math.ceil(max(lengths) / 256) * 256
    gpu_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    query = torch.randn(
        batch, query_length, heads, 576, dtype=torch.bfloat16, device="cuda"
    )
    table = torch.arange(
        batch * padded_length // 64, dtype=torch.int32, device="cuda"
    ).view(batch, padded_length // 64)
    cache = torch.randn(table.numel(), 64, 1, 576, dtype=torch.bfloat16, device="cuda")
    for row, length in enumerate(lengths):
        cache.view(batch, padded_length, 1, 576)[row, length:] = float("nan")
    quant_max = 127.0
    quant_dtype = torch.int8
    query_scale = query[..., :512].float().abs().amax(-1, keepdim=True) / quant_max
    cache_scale = cache[..., :512].float().abs().amax(-1, keepdim=True) / quant_max
    query_scale = torch.exp2(torch.ceil(torch.log2(query_scale)))
    cache_scale = torch.exp2(torch.ceil(torch.log2(cache_scale)))
    query_nope = query[..., :512].float() / query_scale
    cache_nope = cache[..., :512].float() / cache_scale
    query_nope = query_nope.round().to(quant_dtype)
    cache_nope = cache_nope.round().to(quant_dtype)
    query_rope = (query[..., 512:].float() / query_scale).to(torch.bfloat16)
    cache_rope = (cache[..., 512:].float() / cache_scale).to(torch.bfloat16)
    metadata = splits = None
    query = (torch.cat((query_nope.float(), query_rope.float()), -1) * query_scale).to(
        torch.bfloat16
    )
    cache = (torch.cat((cache_nope.float(), cache_rope.float()), -1) * cache_scale).to(
        torch.bfloat16
    )
    metadata, splits = native_bf16_metadata(gpu_lengths, query_length * heads, 1)
    return DenseInt8Inputs(
        query_nope,
        query_rope,
        cache_nope,
        cache_rope,
        query_scale,
        cache_scale,
        table,
        gpu_lengths,
        512,
        metadata,
        splits,
        tuple(lengths),
        query,
        cache,
        tuple(shape),
    )


def native_bf16_flash_mla_with_kvcache_fwd(inputs: DenseInt8Inputs):
    return native_bf16_flash_mla(
        inputs.bf16_query,
        inputs.bf16_cache,
        inputs.block_table,
        inputs.lengths,
        inputs.value_dim,
        inputs.cuda_metadata,
        inputs.cuda_splits,
        causal=True,
    )


def gems_flash_mla_with_kvcache_fwd_w8a8_int8(inputs: DenseInt8Inputs):
    key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr())
    prepared = PREPARED_HANDLES.get(key)
    if prepared is None:
        PREPARED_HANDLES.clear()
        handle, (output, lse) = (
            flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
                *inputs[:9], causal=True
            )
        )
        prepared = (handle, output, lse)
        PREPARED_HANDLES[key] = prepared
    handle, output, lse = prepared
    return handle()


def assert_dense_int8_close(actual, reference):
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=0.002)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=0.0002)


def assert_dense_int8_replay(run, reference):
    assert_dense_int8_close(run(), reference)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = run()
    torch.cuda.current_stream().wait_stream(stream)
    output.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    assert_dense_int8_close((output, lse), reference)


class FlashMLAWithKVCacheFwdW8A8Int8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_fwd",
            native_bf16_flash_mla_with_kvcache_fwd,
            [torch.bfloat16],
        )
        self.set_gems(gems_flash_mla_with_kvcache_fwd_w8a8_int8)
        self.reference = None

    def get_input_iter(self, dtype):
        _ = dtype
        for shape in self.shapes:
            inputs = make_input(shape)
            from tests.test_flash_mla import _scaled_dot_product_attention

            batch, queries, _, heads, _ = inputs.benchmark_shape
            reference_out = torch.empty(
                (batch, queries, heads, 512),
                dtype=torch.float32,
                device=inputs.q_nope.device,
            )
            reference_lse = torch.empty(
                (batch, heads, queries),
                dtype=torch.float32,
                device=inputs.q_nope.device,
            )
            cache_rows = inputs.bf16_cache.view(batch, -1, 1, 576)
            for row, length in enumerate(inputs.host_lengths):
                for query_index in range(queries):
                    bound = length - queries + query_index + 1
                    cache = cache_rows[row, :bound].transpose(0, 1)
                    output, logsum = _scaled_dot_product_attention(
                        inputs.bf16_query[row, query_index].unsqueeze(0),
                        cache,
                        cache[..., :512],
                        1,
                        1,
                        False,
                    )
                    reference_out[row, query_index] = output[0]
                    reference_lse[row, :, query_index] = logsum[0]
            self.reference = (reference_out, reference_lse)
            native_bf16_flash_mla_with_kvcache_fwd(inputs)
            assert_dense_int8_replay(
                lambda: gems_flash_mla_with_kvcache_fwd_w8a8_int8(inputs),
                self.reference,
            )
            yield (inputs,)

    def record_shapes(self, inputs):
        return inputs.benchmark_shape

    def get_latency(self, op, *args, **kwargs):
        latency = super().get_latency(op, *args, **kwargs)
        if op is self.gems_op and self.reference is not None:
            inputs = args[0]
            key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr())
            _, output, lse = PREPARED_HANDLES[key]
            assert_dense_int8_close((output, lse), self.reference)
        return latency


@pytest.mark.flash_mla_with_kvcache_fwd
@pytest.mark.skipif(not HAS_INT8_MLA, reason="Requires registered dense MLA")
def test_flash_mla_with_kvcache_fwd():
    if not HAS_NATIVE_BF16_FLASHMLA:
        raise RuntimeError(
            "The INT8 benchmark requires the device-native BF16 dense FlashMLA package used by vLLM (flash_mla)"
        )
    bench = FlashMLAWithKVCacheFwdW8A8Int8Benchmark()
    bench.run()
