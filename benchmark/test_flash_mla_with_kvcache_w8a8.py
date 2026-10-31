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
from dataclasses import dataclass
from itertools import product

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
    FlashMLAFp8PackedHandle,
    FlashMLAFp8PreparedHandle,
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
)
from tests.test_flash_mla_with_kvcache_w8a8 import SUPPORTED, quantize_dense_fp8

from . import base
from .consts import BenchmarkMetrics
from .test_blas_perf_parallel import ParallelBenchmarkMixin

try:
    from flash_mla_fp8 import flash_mla_ckv_fp8_per_token as cuda_flash_mla
    from flash_mla_fp8 import get_mla_metadata as cuda_get_mla_metadata

    HAS_CUDA_FP8_FLASHMLA = True
except ImportError:
    HAS_CUDA_FP8_FLASHMLA = False


@dataclass
class TestParam:
    __test__ = False

    batch: int
    seq_q: int
    seqlen: int
    h_q: int
    varlen: bool


def run_prepared(
    handle: FlashMLAFp8PreparedHandle | FlashMLAFp8PackedHandle,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return handle(out=output, lse=lse)


class FlashMLAWithKVCacheW8A8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    """Measure steady-state decode after both implementations are prepared."""

    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        if not HAS_CUDA_FP8_FLASHMLA:
            raise RuntimeError("The native CUDA flash_mla_fp8 baseline is required")
        super().__init__(
            "flash_mla_with_kvcache_w8a8", cuda_flash_mla, [torch.bfloat16]
        )
        self.set_gems(run_prepared)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            for inputs, host_lengths in self.make_input(TestParam(*shape), dtype):
                args, kwargs = self.unpack_to_args_kwargs(inputs)
                # Match the native preplanned API: prepare both routes before timing decode.
                metadata, splits = cuda_get_mla_metadata(
                    args[7], args[0].shape[1] * args[0].shape[2], 1
                )
                cuda_input = (*args, metadata, splits, kwargs)
                handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
                    *args,
                    **kwargs,
                    initial_cache_seqlens=host_lengths,
                    max_cache_seqlens=host_lengths,
                )
                self.torch_op(*args, metadata, splits, **kwargs)
                self.gems_op(handle, output, lse)
                yield cuda_input, (handle, output, lse)

    def _build_metric_from_input(self, input_item):
        cuda_input, gems_input = input_item
        cuda_args, cuda_kwargs = self.unpack_to_args_kwargs(cuda_input)
        metric = BenchmarkMetrics()
        metric.shape_detail = self.record_shapes(*cuda_args[:9], **cuda_kwargs)
        if "latency_base" in self.to_bench_metrics:
            metric.latency_base = self.get_latency(
                self.torch_op, *cuda_args, **cuda_kwargs
            )
        if "latency" in self.to_bench_metrics:
            metric.latency = self.get_latency(self.gems_op, *gems_input)
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        return metric

    @staticmethod
    def make_input(param: TestParam, dtype):
        torch.manual_seed(0)
        lengths = [param.seqlen] * param.batch
        if param.varlen:
            # Preserve the native reference's seeded length stream when selecting a shard.
            generator = random.Random(0)
            for mean, heads, queries in product(
                (4096, 8192, 16384), (16, 32, 64, 128), (1, 2)
            ):
                lengths = [
                    max(int(generator.normalvariate(mean, mean / 2)), queries)
                    for _ in range(128)
                ]
                if (param.batch, param.seqlen, param.h_q, param.seq_q) == (
                    128,
                    mean,
                    heads,
                    queries,
                ):
                    break
            else:
                raise ValueError("Variable-length shape must be from the native matrix")
        else:
            pass
        padded_length = math.ceil(max(lengths) / 256) * 256
        device = flaggems_vllm.device
        cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)
        query = torch.randn(
            param.batch, param.seq_q, param.h_q, 576, dtype=dtype, device=device
        )
        table = torch.arange(
            param.batch * padded_length // 64, dtype=torch.int32, device=device
        ).view(param.batch, padded_length // 64)
        cache = torch.randn(table.numel(), 64, 1, 576, dtype=dtype, device=device)
        for row, length in enumerate(lengths):
            cache.view(param.batch, padded_length, 1, 576)[row, length:] = float("nan")
        q_nope, q_rope, q_scale, _ = quantize_dense_fp8(query, dequantize=False)
        k_nope, k_rope, k_scale, _ = quantize_dense_fp8(cache, dequantize=False)
        inputs = (
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            q_scale,
            k_scale,
            table,
            cache_seqlens,
            512,
            {"causal": True},
        )
        yield inputs, tuple(lengths)


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not SUPPORTED, reason="FP8 MLA requires NVIDIA Hopper and TLE")
def test_flash_mla_with_kvcache_w8a8():
    FlashMLAWithKVCacheW8A8Benchmark().run()
