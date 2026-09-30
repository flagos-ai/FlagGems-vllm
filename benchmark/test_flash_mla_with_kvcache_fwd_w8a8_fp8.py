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

from collections.abc import Iterator
from typing import NamedTuple

import pytest
import torch

from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
)
from tests.test_flash_mla_with_kvcache_fwd_w8a8_fp8 import quantize_ckv_per_token

from . import base
from .test_flash_mla_with_kvcache import (
    HAS_CUDA_FLASHMLA,
    FlashMLAWithKVCacheBenchmark,
    TestParam,
    _cuda_wrapper,
)

CONTENT_DIM = 512
ROPE_DIM = 64
SCALE_GROUPS = 4
SCALE_BYTES = SCALE_GROUPS * 4
TOKEN_BYTES = CONTENT_DIM + SCALE_BYTES + ROPE_DIM * 2
MAX_OUTPUT_RELATIVE_L2 = 0.05
LSE_ATOL = 0.025
LSE_RTOL = 0.002
PREPARED_HANDLES = {}


class DenseFp8BenchmarkInputs(NamedTuple):
    query_bf16: torch.Tensor
    packed_kv_cache: torch.Tensor
    query_nope_fp8: torch.Tensor
    query_rope_bf16: torch.Tensor
    query_scale: torch.Tensor
    kv_nope_fp8: torch.Tensor
    kv_rope_bf16: torch.Tensor
    kv_scale: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    token_indices: torch.Tensor
    sequence_length: int
    value_dim: int
    is_causal: bool


def run_vllm_bf16_query_fp8_cache(
    inputs: DenseFp8BenchmarkInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert inputs.is_causal
    # vLLM's dense FP8 kernel requires FP8 Q; its indexed kernel supports BF16 Q.
    return _cuda_wrapper(
        inputs.query_bf16,
        inputs.packed_kv_cache,
        None,
        None,
        inputs.value_dim,
        causal=False,
        is_fp8_kvcache=True,
        indices=inputs.token_indices,
    )


def run_dense_fp8_mla(
    inputs: DenseFp8BenchmarkInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        int(inputs.query_nope_fp8.data_ptr()),
        int(inputs.kv_nope_fp8.data_ptr()),
        inputs.sequence_length,
    )
    prepared = PREPARED_HANDLES.get(cache_key)
    if prepared is None:
        # A single retained shape bounds the descriptor and workspace lifetime.
        PREPARED_HANDLES.clear()
        lengths = (inputs.sequence_length,) * int(inputs.query_nope_fp8.shape[0])
        handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
            inputs.query_nope_fp8,
            inputs.query_rope_bf16,
            inputs.kv_nope_fp8,
            inputs.kv_rope_bf16,
            inputs.query_scale,
            inputs.kv_scale,
            inputs.block_table,
            inputs.cache_seqlens,
            inputs.value_dim,
            causal=inputs.is_causal,
            initial_cache_seqlens=lengths,
            max_cache_seqlens=lengths,
        )
        prepared = (handle, output, lse)
        PREPARED_HANDLES[cache_key] = prepared
    handle, output, lse = prepared
    return handle(out=output, lse=lse)


class FlashMLAWithKVCacheFP8Benchmark(FlashMLAWithKVCacheBenchmark):
    def __init__(self) -> None:
        base.Benchmark.__init__(
            self,
            "flash_mla_with_kvcache_fwd_w8a8_fp8",
            run_vllm_bf16_query_fp8_cache,
            [torch.bfloat16],
        )
        self.set_gems(run_dense_fp8_mla)

    @staticmethod
    def get_performance_test_params() -> list[TestParam]:
        return [
            param
            for param in FlashMLAWithKVCacheBenchmark.get_performance_test_params()
            if param.topk == 0 and param.d_qk == CONTENT_DIM + ROPE_DIM
        ]

    def get_input_iter(
        self, dtype: torch.dtype
    ) -> Iterator[tuple[DenseFp8BenchmarkInputs]]:
        for (inputs,) in super().get_input_iter(dtype):
            reference, reference_lse = run_vllm_bf16_query_fp8_cache(inputs)
            output, lse = run_dense_fp8_mla(inputs)
            relative_l2 = (output.float() - reference.float()).norm() / (
                reference.float().norm().clamp_min(1e-12)
            )
            assert relative_l2.item() < MAX_OUTPUT_RELATIVE_L2
            torch.testing.assert_close(lse, reference_lse, atol=LSE_ATOL, rtol=LSE_RTOL)
            yield (inputs,)

    @staticmethod
    def make_input(param: TestParam) -> Iterator[tuple[DenseFp8BenchmarkInputs]]:
        for (
            query,
            cache,
            block_table,
            cache_seqlens,
            value_dim,
            kwargs,
        ) in FlashMLAWithKVCacheBenchmark.make_input(param):
            # The BF16-Q/FP8-KV CUDA path has a fixed top-k for every request.
            cache_seqlens.fill_(param.seqlen)
            query_nope, query_rope, query_scale = quantize_ckv_per_token(query)
            kv_nope, kv_rope, kv_scale = quantize_ckv_per_token(cache)
            packed_cache = torch.empty(
                (*cache.shape[:-1], TOKEN_BYTES),
                device=cache.device,
                dtype=torch.uint8,
            )
            packed_cache[..., :CONTENT_DIM].copy_(kv_nope.view(torch.uint8))
            packed_cache[..., CONTENT_DIM : CONTENT_DIM + SCALE_BYTES].copy_(
                kv_scale.expand(*kv_scale.shape[:-1], SCALE_GROUPS)
                .contiguous()
                .view(torch.uint8)
            )
            packed_cache[..., CONTENT_DIM + SCALE_BYTES :].copy_(
                cache[..., CONTENT_DIM:].contiguous().view(torch.uint8)
            )
            page_offsets = torch.arange(
                cache.shape[1], device=cache.device, dtype=torch.int32
            )
            pages_per_request = param.seqlen // cache.shape[1]
            indices = (
                block_table[:, :pages_per_request, None] * cache.shape[1] + page_offsets
            ).reshape(query.shape[0], 1, -1)
            yield (
                DenseFp8BenchmarkInputs(
                    query,
                    packed_cache,
                    query_nope,
                    query_rope,
                    query_scale,
                    kv_nope,
                    kv_rope,
                    kv_scale,
                    block_table,
                    cache_seqlens,
                    indices,
                    param.seqlen,
                    value_dim,
                    kwargs["causal"],
                ),
            )


@pytest.mark.skipif(
    not (HAS_TLE and HAS_CUDA_FLASHMLA and torch.cuda.is_available()),
    reason="requires Hopper, FlagTree TLE and vLLM FlashMLA CUDA",
)
@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_fp8
def test_flash_mla_with_kvcache_fwd_w8a8_fp8() -> None:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an NVIDIA Hopper GPU")
    FlashMLAWithKVCacheFP8Benchmark().run()
