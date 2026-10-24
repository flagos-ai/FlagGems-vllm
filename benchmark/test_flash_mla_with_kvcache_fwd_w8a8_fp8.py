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
from .test_blas_perf_parallel import ParallelBenchmarkMixin

HAS_MUSA_FP8_MLA = "flash_mla_with_kvcache_w8a8" in dict(
    flaggems_vllm.runtime.backend.get_customized_ops()
)


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
    bf16_query: torch.Tensor | None = None
    bf16_cache: torch.Tensor | None = None
    benchmark_shape: tuple = ()


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
    quant_max = 448.0
    query_scale = (
        query[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
        / quant_max
    )
    if HAS_MUSA_FP8_MLA:
        # Power-of-two scales keep FP8 dequantization exactly representable in BF16.
        query_scale = torch.exp2(torch.ceil(torch.log2(query_scale)))
    query_nope = (query[..., :512].float() / query_scale).to(torch.float8_e4m3fn)
    query_rope = (query[..., 512:].float() / query_scale).to(torch.bfloat16)
    cache_scale = (
        cache[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
        / quant_max
    )
    if HAS_MUSA_FP8_MLA:
        cache_scale = torch.exp2(torch.ceil(torch.log2(cache_scale)))
    cache_nope = (cache[..., :512].float() / cache_scale).to(torch.float8_e4m3fn)
    cache_rope = (cache[..., 512:].float() / cache_scale).to(torch.bfloat16)
    if HAS_MUSA_FP8_MLA:
        from mate.flashmla import get_mla_metadata as get_bf16_metadata

        bf16_query = (
            torch.cat((query_nope.float(), query_rope.float()), -1) * query_scale
        ).to(torch.bfloat16)
        bf16_cache = (
            torch.cat((cache_nope.float(), cache_rope.float()), -1) * cache_scale
        ).to(torch.bfloat16)
        # Keep ignored BF16 padding finite for MATE; valid tokens are identical.
        for row, length in enumerate(lengths):
            bf16_cache.view(batch, padded_length, 1, 576)[row, length:] = 0
        metadata, splits = get_bf16_metadata(
            gpu_lengths, query_length * heads, 1, num_heads_q=heads, q=bf16_query
        )
    else:
        metadata, splits = get_mla_metadata(gpu_lengths, query_length * heads, 1)
        bf16_query, bf16_cache = None, None
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
        bf16_query,
        bf16_cache,
        tuple(shape),
    )


def dense_fp8_paged_reference(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    q_scale,
    k_scale,
    block_table,
    lengths,
    *,
    softmax_scale=None,
    causal=False,
):
    batch, queries, heads, _ = q_nope.shape
    pages = block_table.shape[1]
    capacity = pages * 64
    scale = 576**-0.5 if softmax_scale is None else softmax_scale
    output = torch.empty(
        (*q_nope.shape[:-1], 512), dtype=q_rope.dtype, device=q_nope.device
    )
    lse = torch.empty(
        (batch, heads, queries), dtype=torch.float32, device=q_nope.device
    )
    # Convert before indexing because MUSA does not support FP8 advanced indexing.
    cache = torch.cat((k_nope.float(), k_rope.float()), -1).reshape(-1, 64, 576)
    cache_scale = k_scale.reshape(-1, 64)
    tokens = torch.arange(capacity, device=q_nope.device)
    query_tokens = torch.arange(queries, device=q_nope.device)
    for begin in range(0, batch, 4):
        end = min(begin + 4, batch)
        page_ids = block_table[begin:end].long()
        values = cache[page_ids].reshape(end - begin, capacity, 576)
        valid_token = tokens[None, :] < lengths[begin:end, None]
        values = values.masked_fill(~valid_token[:, :, None], 0)
        ks = cache_scale[page_ids].reshape(end - begin, capacity)
        ks = ks.masked_fill(~valid_token, 0)
        query = torch.cat((q_nope[begin:end].float(), q_rope[begin:end].float()), -1)
        scores = torch.einsum("bqhd,bnd->bqhn", query, values)
        scores = scores * q_scale[begin:end] * ks[:, None, None, :] * scale
        valid = valid_token[:, None, None, :]
        if causal:
            valid = valid & (
                tokens[None, None, None, :]
                < (
                    lengths[begin:end, None, None, None]
                    - queries
                    + query_tokens[None, :, None, None]
                    + 1
                )
            )
        scores = scores.masked_fill(~valid, -torch.inf)
        logsum = torch.logsumexp(scores, -1)
        has_tokens = valid.any(-1, keepdim=True)
        probabilities = torch.where(has_tokens, scores.softmax(-1), 0)
        weights = (probabilities * ks[:, None, None, :]).reshape(
            end - begin, queries, heads, pages, 64
        )
        p_scale = (weights.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
        # Per-page P quantization is invariant to the online softmax row scale.
        quantized = (weights / p_scale).to(torch.float8_e4m3fn).float() * p_scale
        result = torch.einsum(
            "bqhn,bnd->bqhd", quantized.flatten(-2), values[..., :512]
        )
        output[begin:end] = result.to(q_rope.dtype)
        lse[begin:end] = logsum.transpose(1, 2)
    return output, lse


def dense_musa_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    *,
    metadata: torch.Tensor | None = None,
    num_splits: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mate.flashmla import get_mla_metadata as get_bf16_metadata
    from mate.jit.mla_ops import get_mla_ops_module

    if metadata is None:
        metadata, num_splits = get_bf16_metadata(
            cache_seqlens,
            q.shape[1] * q.shape[2] // cache.shape[2],
            cache.shape[2],
            num_heads_q=q.shape[2],
            q=q,
        )
    output = torch.empty((*q.shape[:-1], head_dim_v), dtype=q.dtype, device=q.device)
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    # MATE's CUDA-derived C++ path avoids the intermittent H=128 MUBIN errors.
    get_mla_ops_module().get_function("mla_with_kvcache")(
        q[..., :head_dim_v],
        q[..., head_dim_v:],
        cache[..., :head_dim_v],
        cache[..., head_dim_v:],
        cache_seqlens,
        None,
        None,
        block_table,
        metadata,
        num_splits,
        output,
        lse,
        q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
        causal,
    )
    return output, lse.transpose(1, 2)


def cuda_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs: DenseFP8Inputs):
    if HAS_MUSA_FP8_MLA:
        return dense_musa_reference(
            inputs.bf16_query,
            inputs.bf16_cache,
            inputs.block_table,
            inputs.lengths,
            inputs.value_dim,
            metadata=inputs.cuda_metadata,
            num_splits=inputs.cuda_splits,
            causal=True,
        )
    return flash_mla_ckv_fp8_per_token(*inputs[:11], causal=True)


def gems_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs: DenseFP8Inputs):
    key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr(), inputs.q_nope.shape)
    prepared = PREPARED_HANDLES.get(key)
    if prepared is None:
        PREPARED_HANDLES.clear()
        if HAS_MUSA_FP8_MLA:
            output, lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(
                *inputs[:9], causal=True
            )
            handle = None
        else:
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
    if HAS_MUSA_FP8_MLA:
        return flaggems_vllm.flash_mla_with_kvcache_w8a8(
            *inputs[:9], causal=True, out=output, lse=lse
        )
    return handle(out=output, lse=lse)


def assert_dense_fp8_close(actual, reference):
    if reference[0].device.type == "cpu":
        actual = tuple(tensor.cpu() for tensor in actual)
    # FP32 QK rounding can put P on opposite sides of an FP8 rounding midpoint.
    output_atol = 1e-2 if HAS_MUSA_FP8_MLA else 2e-3
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=output_atol)
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
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    assert_dense_fp8_close((output, lse), reference)


class FlashMLAWithKVCacheFwdW8A8FP8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_w8a8",
            cuda_flash_mla_with_kvcache_fwd_w8a8_fp8,
            [torch.bfloat16],
        )
        self.set_gems(gems_flash_mla_with_kvcache_fwd_w8a8_fp8)

    def get_input_iter(self, dtype):
        _ = dtype
        for shape in self.shapes:
            inputs = make_input(shape)
            reference = cuda_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs)
            if HAS_MUSA_FP8_MLA:
                reference = dense_fp8_paged_reference(*inputs[:8], causal=True)
                self.reference = tuple(tensor.cpu() for tensor in reference)
            else:
                self.reference = reference
            # Validate eager and captured execution before either route is timed.
            assert_dense_fp8_replay(
                lambda: gems_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs), self.reference
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
            _, output, lse = PREPARED_HANDLES[key]
            # Inspect the timed execution's buffers without launching another call.
            assert_dense_fp8_close((output, lse), self.reference)
        return latency

    def record_shapes(self, inputs):
        return inputs.benchmark_shape


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(
    not (HAS_FP8_MLA or HAS_MUSA_FP8_MLA),
    reason="No registered quantized dense MLA implementation",
)
def test_flash_mla_with_kvcache_w8a8():
    if not HAS_CUDA_FP8_FLASHMLA and not HAS_MUSA_FP8_MLA:
        raise RuntimeError(
            "Meituan FP8 FlashMLA (feature/ckv_fp8_per_token, a29b228) "
            "is required for the dense FP8 CUDA benchmark"
        )
    bench = FlashMLAWithKVCacheFwdW8A8FP8Benchmark()
    bench.run()
