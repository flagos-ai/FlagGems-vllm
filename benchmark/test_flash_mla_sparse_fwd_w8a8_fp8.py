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

import flaggems_vllm
from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from tests.test_flash_mla_sparse_fwd_w8a8_fp8 import (
    assert_sparse_fp8_accuracy,
    make_sparse_fp8_inputs,
    pack_cuda_sparse_fp8_cache,
)

from . import base
from .test_flash_mla_with_kvcache import (
    HAS_CUDA_FLASHMLA,
    FlashMLAWithKVCacheBenchmark,
    TestParam,
    _cuda_wrapper,
)

CONTENT_DIM = 512
ROPE_DIM = 64
HEAD_DIM = CONTENT_DIM + ROPE_DIM


class SparseFp8BenchmarkInputs(NamedTuple):
    query_nope_fp8: torch.Tensor
    query_rope_bf16: torch.Tensor
    kv_nope_fp8: torch.Tensor
    kv_rope_bf16: torch.Tensor
    query_scale: torch.Tensor
    kv_scale: torch.Tensor
    indices: torch.Tensor
    query_bf16: torch.Tensor
    packed_kv_cache: torch.Tensor
    attention_sink: torch.Tensor


def run_vllm_bf16_query_fp8_cache(
    inputs: SparseFp8BenchmarkInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _cuda_wrapper(
        inputs.query_bf16,
        inputs.packed_kv_cache,
        None,
        None,
        CONTENT_DIM,
        causal=False,
        is_fp8_kvcache=True,
        indices=inputs.indices,
        attn_sink=inputs.attention_sink,
        softmax_scale=HEAD_DIM**-0.5,
    )


def run_sparse_fp8_mla(
    inputs: SparseFp8BenchmarkInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(
        inputs.query_nope_fp8,
        inputs.query_rope_bf16,
        inputs.kv_nope_fp8,
        inputs.kv_rope_bf16,
        inputs.query_scale,
        inputs.kv_scale,
        inputs.indices,
        attn_sink=inputs.attention_sink,
    )


class FlashMLASparseFP8Benchmark(FlashMLAWithKVCacheBenchmark):
    def __init__(self) -> None:
        base.Benchmark.__init__(
            self,
            "flash_mla_sparse_fwd_w8a8_fp8",
            run_vllm_bf16_query_fp8_cache,
            [torch.bfloat16],
        )
        self.set_gems(run_sparse_fp8_mla)

    @staticmethod
    def get_performance_test_params() -> list[TestParam]:
        return [
            param
            for param in FlashMLAWithKVCacheBenchmark.get_performance_test_params()
            if param.topk > 0 and param.d_qk == HEAD_DIM and param.have_attn_sink
        ]

    def get_input_iter(
        self, dtype: torch.dtype
    ) -> Iterator[tuple[SparseFp8BenchmarkInputs]]:
        for (inputs,) in super().get_input_iter(dtype):
            reference, reference_lse = run_vllm_bf16_query_fp8_cache(inputs)
            output, lse = run_sparse_fp8_mla(inputs)
            assert_sparse_fp8_accuracy(output, lse, reference, reference_lse)
            yield (inputs,)

    @staticmethod
    def make_input(param: TestParam) -> Iterator[tuple[SparseFp8BenchmarkInputs]]:
        tensors, query, cache = make_sparse_fp8_inputs(
            param.batch, param.h_q, param.topk
        )
        query_nope, query_rope, kv_nope, kv_rope, query_scale, kv_scale, indices = (
            tensors
        )
        packed_cache = pack_cuda_sparse_fp8_cache(
            kv_nope, kv_scale, cache[..., CONTENT_DIM:]
        )
        attention_sink = torch.randn(param.h_q, device=query.device)
        yield (
            SparseFp8BenchmarkInputs(
                query_nope,
                query_rope,
                kv_nope,
                kv_rope,
                query_scale,
                kv_scale,
                indices,
                query,
                packed_cache,
                attention_sink,
            ),
        )


@pytest.mark.flash_mla_sparse_fwd_w8a8_fp8
@pytest.mark.skipif(
    not (HAS_TLE and HAS_CUDA_FLASHMLA and torch.cuda.is_available()),
    reason="requires Hopper, FlagTree TLE and vLLM FlashMLA CUDA",
)
def test_flash_mla_sparse_fwd_w8a8_fp8() -> None:
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an NVIDIA Hopper GPU")
    FlashMLASparseFP8Benchmark().run()
