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
from typing import NamedTuple

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import HAS_TLE
from tests.accuracy_utils import gems_assert_close

from . import base

# Baseline: native CUDA FP8 FlashMLA from Meituan FlashMLA
# feature/ckv_fp8_per_token (a29b228).
try:
    from flash_mla_fp8 import flash_mla_ckv_fp8_per_token, get_mla_metadata

    HAS_CUDA_FP8_FLASHMLA = True
except ImportError:
    HAS_CUDA_FP8_FLASHMLA = False

HAS_FP8_MLA = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)

# BF16-default combinations in Meituan's unchanged upstream test main.
UPSTREAM_SHAPES = [
    (batch, query_length, length, heads, varlen)
    for batch in [128]
    for length in [4096, 8192, 16384]
    for heads in [16, 32, 64, 128]
    for query_length in [1, 2]
    for varlen in [False, True]
]
PREPARED_HANDLES = {}


class DenseFP8Inputs(NamedTuple):
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


def make_input(shape: tuple) -> DenseFP8Inputs:
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
    query_scale = query[..., :512].float().abs().amax(-1, keepdim=True) / 448.0
    query_nope = (query[..., :512].float() / query_scale).to(torch.float8_e4m3fn)
    query_rope = (query[..., 512:].float() / query_scale).to(torch.bfloat16)
    cache_scale = cache[..., :512].float().abs().amax(-1, keepdim=True) / 448.0
    cache_nope = (cache[..., :512].float() / cache_scale).to(torch.float8_e4m3fn)
    cache_rope = (cache[..., 512:].float() / cache_scale).to(torch.bfloat16)
    metadata, splits = get_mla_metadata(gpu_lengths, query_length * heads, 1)
    return DenseFP8Inputs(
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
    )


def cuda_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs: DenseFP8Inputs):
    return flash_mla_ckv_fp8_per_token(*inputs[:11], causal=True)


def gems_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs: DenseFP8Inputs):
    key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr())
    prepared = PREPARED_HANDLES.get(key)
    if prepared is None:
        PREPARED_HANDLES.clear()
        handle, (output, lse) = (
            flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
                *inputs[:9],
                causal=True,
                initial_cache_seqlens=inputs.host_lengths,
                max_cache_seqlens=inputs.host_lengths,
            )
        )
        prepared = (handle, output, lse)
        PREPARED_HANDLES[key] = prepared
    handle, output, lse = prepared
    return handle(out=output, lse=lse)


def assert_dense_fp8_close(actual, reference):
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=2e-3)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=2e-4)


def assert_dense_fp8_replay(run, reference):
    assert_dense_fp8_close(run(), reference)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = run()
    torch.cuda.current_stream().wait_stream(stream)
    # A replay must overwrite every output, rather than retain the warmup result.
    output.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    assert_dense_fp8_close((output, lse), reference)


class FlashMLAWithKVCacheFwdW8A8FP8Benchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_fwd_w8a8_fp8",
            cuda_flash_mla_with_kvcache_fwd_w8a8_fp8,
            [torch.bfloat16],
        )
        self.set_gems(gems_flash_mla_with_kvcache_fwd_w8a8_fp8)

    def get_input_iter(self, dtype):
        _ = dtype
        for shape in self.shapes:
            inputs = make_input(shape)
            self.reference = cuda_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs)
            # Validate eager and captured execution before either route is timed.
            assert_dense_fp8_replay(
                lambda: gems_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs), self.reference
            )
            yield (inputs,)

    def get_latency(self, op, *args, **kwargs):
        latency = super().get_latency(op, *args, **kwargs)
        if op is self.gems_op:
            inputs = args[0]
            key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr())
            _, output, lse = PREPARED_HANDLES[key]
            # Inspect the timed execution's buffers without launching another call.
            assert_dense_fp8_close((output, lse), self.reference)
        return latency


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_fp8
@pytest.mark.skipif(
    not HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and Triton TLE"
)
def test_flash_mla_with_kvcache_fwd_w8a8_fp8():
    if not HAS_CUDA_FP8_FLASHMLA:
        raise RuntimeError(
            "Meituan FP8 FlashMLA (feature/ckv_fp8_per_token, a29b228) "
            "is required for the dense FP8 CUDA benchmark"
        )
    bench = FlashMLAWithKVCacheFwdW8A8FP8Benchmark()
    bench.run()
