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
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import HAS_TLE
from tests.accuracy_utils import gems_assert_close

from . import base
from . import conftest as benchmark_config

# Baseline: native CUDA FP8 FlashMLA from Meituan FlashMLA
# feature/ckv_fp8_per_token (a29b228).
try:
    from flash_mla_fp8 import flash_mla_ckv_fp8_per_token, get_mla_metadata

    HAS_CUDA_FP8_FLASHMLA = True
except ImportError:
    HAS_CUDA_FP8_FLASHMLA = False

HAS_INT8_MLA = "flash_mla_with_kvcache_fwd_w8a8_int8" in dict(
    flaggems_vllm.runtime.backend.get_customized_ops()
)
if HAS_INT8_MLA:
    from flash_mla import flash_mla_with_kvcache as cuda_bf16_flash_mla
    from flash_mla import get_mla_metadata as cuda_bf16_metadata

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
    bf16_query: Optional[torch.Tensor]
    bf16_cache: Optional[torch.Tensor]
    shape: tuple


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
    quant_max = 127.0 if HAS_INT8_MLA else 448.0
    quant_dtype = torch.int8 if HAS_INT8_MLA else torch.float8_e4m3fn
    query_scale = query[..., :512].float().abs().amax(-1, keepdim=True) / quant_max
    if HAS_INT8_MLA:
        # Power-of-two scales make the BF16 dequantization exact for INT8 content.
        query_scale = torch.exp2(torch.ceil(torch.log2(query_scale)))
    query_nope = query[..., :512].float() / query_scale
    query_nope = (query_nope.round() if HAS_INT8_MLA else query_nope).to(quant_dtype)
    query_rope = (query[..., 512:].float() / query_scale).to(torch.bfloat16)
    cache_scale = cache[..., :512].float().abs().amax(-1, keepdim=True) / quant_max
    if HAS_INT8_MLA:
        cache_scale = torch.exp2(torch.ceil(torch.log2(cache_scale)))
    cache_nope = cache[..., :512].float() / cache_scale
    cache_nope = (cache_nope.round() if HAS_INT8_MLA else cache_nope).to(quant_dtype)
    cache_rope = (cache[..., 512:].float() / cache_scale).to(torch.bfloat16)
    if HAS_INT8_MLA:
        # Materialize the dequantized BF16 inputs outside both timed paths.
        query = (
            torch.cat((query_nope.float(), query_rope.float()), -1) * query_scale
        ).to(torch.bfloat16)
        cache = (
            torch.cat((cache_nope.float(), cache_rope.float()), -1) * cache_scale
        ).to(torch.bfloat16)
        metadata, splits = cuda_bf16_metadata(gpu_lengths, query_length * heads, 1)
    elif HAS_CUDA_FP8_FLASHMLA:
        metadata, splits = get_mla_metadata(gpu_lengths, query_length * heads, 1)
    else:
        metadata = splits = None
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
        query if HAS_INT8_MLA else None,
        cache if HAS_INT8_MLA else None,
        tuple(shape),
    )


def cuda_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs: DenseFP8Inputs):
    if HAS_INT8_MLA:
        return cuda_bf16_flash_mla(
            inputs.bf16_query,
            inputs.bf16_cache,
            inputs.block_table,
            inputs.lengths,
            inputs.value_dim,
            inputs.cuda_metadata,
            inputs.cuda_splits,
            causal=True,
        )
    return flash_mla_ckv_fp8_per_token(*inputs[:11], causal=True)


def gems_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs: DenseFP8Inputs):
    key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr())
    prepared = PREPARED_HANDLES.get(key)
    if prepared is None:
        PREPARED_HANDLES.clear()
        if HAS_INT8_MLA:
            handle, (output, lse) = (
                flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
                    *inputs[:9],
                    causal=True,
                )
            )
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
    return handle() if HAS_INT8_MLA else handle(out=output, lse=lse)


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
            (
                "flash_mla_with_kvcache_fwd_w8a8_int8"
                if HAS_INT8_MLA
                else "flash_mla_with_kvcache_fwd"
            ),
            cuda_flash_mla_with_kvcache_fwd_w8a8_fp8,
            [torch.bfloat16],
        )
        self.set_gems(gems_flash_mla_with_kvcache_fwd_w8a8_fp8)
        self.baseline_accuracy = []
        self.reference = None
        if not (HAS_INT8_MLA or HAS_CUDA_FP8_FLASHMLA):
            # Preserve the FP8 latency-only path when its CUDA reference is absent.
            self.metrics = ["latency"]

    def get_input_iter(self, dtype):
        _ = dtype
        for shape in self.shapes:
            inputs = make_input(shape)
            if HAS_INT8_MLA or HAS_CUDA_FP8_FLASHMLA:
                cuda_reference = cuda_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs)
                if HAS_INT8_MLA:
                    from tests.test_flash_mla import _scaled_dot_product_attention

                    batch, queries, _, heads, _ = shape
                    reference_out = torch.empty(
                        (batch, queries, heads, 512),
                        dtype=torch.bfloat16,
                        device=inputs.q_nope.device,
                    )
                    reference_lse = torch.empty(
                        (batch, heads, queries),
                        dtype=torch.float32,
                        device=inputs.q_nope.device,
                    )
                    cache_rows = inputs.bf16_cache.view(batch, -1, 1, 576)
                    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = False
                    try:
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
                    finally:
                        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                    self.reference = (reference_out, reference_lse)
                    # Native BF16 decoding can differ from FP32 on very short rows.
                    try:
                        assert_dense_fp8_close(cuda_reference, self.reference)
                        baseline_accuracy = {"shape": tuple(shape), "passed": True}
                    except AssertionError as error:
                        baseline_accuracy = {
                            "shape": tuple(shape),
                            "passed": False,
                            "error": str(error),
                        }
                    self.baseline_accuracy.append(baseline_accuracy)
                else:
                    self.reference = cuda_reference
                # Validate eager and captured execution before either route is timed.
                assert_dense_fp8_replay(
                    lambda: gems_flash_mla_with_kvcache_fwd_w8a8_fp8(inputs),
                    self.reference,
                )
            yield (inputs,)

    def record_shapes(self, inputs):
        return inputs.shape

    def get_latency(self, op, *args, **kwargs):
        latency = super().get_latency(op, *args, **kwargs)
        if op is self.gems_op and self.reference is not None:
            inputs = args[0]
            key = (inputs.q_nope.data_ptr(), inputs.k_nope.data_ptr())
            _, output, lse = PREPARED_HANDLES[key]
            # Inspect the timed execution's buffers without launching another call.
            assert_dense_fp8_close((output, lse), self.reference)
        return latency


@(
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
    if HAS_INT8_MLA
    else pytest.mark.flash_mla_with_kvcache_fwd
)
@pytest.mark.skipif(
    not (HAS_FP8_MLA or HAS_INT8_MLA),
    reason="FP8 MLA requires NVIDIA Hopper and Triton TLE",
)
def test_flash_mla_with_kvcache_fwd():
    if not HAS_INT8_MLA and not HAS_CUDA_FP8_FLASHMLA:
        print(
            "Meituan FP8 FlashMLA (feature/ckv_fp8_per_token, a29b228) is "
            "unavailable; recording FlagGems latency only"
        )
    bench = FlashMLAWithKVCacheFwdW8A8FP8Benchmark()
    bench.run()
    if not HAS_INT8_MLA or not benchmark_config.Config.record_json:
        return
    record = benchmark_config.TEST_RESULTS[bench.op_name]
    record["accuracy_reference"] = "existing FP32 scaled dot product attention"
    record["native_bf16_accuracy"] = bench.baseline_accuracy
