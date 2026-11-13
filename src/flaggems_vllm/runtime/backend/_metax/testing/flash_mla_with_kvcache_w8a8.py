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

from importlib import import_module

import torch

import flaggems_vllm
from tests.flash_mla_with_kvcache_w8a8_utils import (
    CapturedCall,
    DenseW8A8Inputs,
    PreparedCall,
    assert_dense_fp8_close,
    assert_dense_fp8_replay,
)

OUTPUT_ATOL = 2e-3


def supported() -> bool:
    return flaggems_vllm.runtime.torch_device_fn.is_available()


def quantize(x: torch.Tensor, *, benchmark: bool):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True) / 127.0
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    nope = (x[..., :512].float() / scale).round().to(torch.int8)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    native = None
    if not benchmark:
        native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(
            torch.bfloat16
        )
    return nope, rope, scale, native


def prepare_baseline(inputs: DenseW8A8Inputs) -> DenseW8A8Inputs:
    native = import_module("flash_mla")

    query = inputs.native_query
    cache = inputs.native_cache
    if query is None:
        query = (
            torch.cat((inputs.q_nope.float(), inputs.q_rope.float()), -1)
            * inputs.q_scale
        ).to(torch.bfloat16)
    if cache is None:
        cache = (
            torch.cat((inputs.k_nope.float(), inputs.k_rope.float()), -1)
            * inputs.k_scale
        ).to(torch.bfloat16)
    metadata, splits = native.get_mla_metadata(
        inputs.lengths, inputs.q_nope.shape[1] * inputs.q_nope.shape[2], 1
    )
    return inputs._replace(
        native_query=query,
        native_cache=cache,
        native_metadata=metadata,
        native_splits=splits,
    )


def run_baseline(inputs: DenseW8A8Inputs):
    native = import_module("flash_mla")

    return native.flash_mla_with_kvcache(
        inputs.native_query,
        inputs.native_cache,
        inputs.block_table,
        inputs.lengths,
        inputs.value_dim,
        inputs.native_metadata,
        inputs.native_splits,
        causal=True,
    )


def functional_reference(inputs: DenseW8A8Inputs):
    return run_baseline(prepare_baseline(inputs))


def benchmark_reference(inputs: DenseW8A8Inputs):
    from benchmark import conftest as benchmark_config
    from tests.test_flash_mla import _scaled_dot_product_attention

    native = run_baseline(inputs)
    batch, queries, heads, _ = inputs.q_nope.shape
    output = torch.empty(
        (batch, queries, heads, 512), dtype=torch.bfloat16, device=inputs.q_nope.device
    )
    lse = torch.empty(
        (batch, heads, queries), dtype=torch.float32, device=inputs.q_nope.device
    )
    cache_rows = inputs.native_cache.view(batch, -1, 1, 576)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row, length in enumerate(inputs.host_lengths):
            for query_index in range(queries):
                bound = length - queries + query_index + 1
                cache = cache_rows[row, :bound].transpose(0, 1)
                result, logsum = _scaled_dot_product_attention(
                    inputs.native_query[row, query_index].unsqueeze(0),
                    cache,
                    cache[..., :512],
                    1,
                    1,
                    False,
                )
                output[row, query_index] = result[0]
                lse[row, :, query_index] = logsum[0]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    reference = output, lse
    try:
        assert_dense_fp8_close(native, reference, output_atol=OUTPUT_ATOL)
        accuracy = {"shape": inputs.benchmark_shape, "passed": True}
    except AssertionError as error:
        # Preserve the original informational check of the unchanged BF16 baseline.
        accuracy = {
            "shape": inputs.benchmark_shape,
            "passed": False,
            "error": str(error),
        }
    if benchmark_config.Config.record_json:
        record = benchmark_config.TEST_RESULTS.setdefault(
            "flash_mla_with_kvcache_w8a8", {}
        )
        record["accuracy_reference"] = "existing FP32 scaled dot product attention"
        record.setdefault("native_bf16_accuracy", []).append(accuracy)
    return reference


def prepare_gems(inputs: DenseW8A8Inputs) -> PreparedCall:
    handle, (output, lse) = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        *inputs[:9], causal=True
    )
    return PreparedCall(handle, output, lse)


def capture(run) -> CapturedCall:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = run()
    torch.cuda.current_stream().wait_stream(stream)
    return CapturedCall(graph.replay, output, lse)


def validate_functional(inputs, actual, reference) -> None:
    prepared = prepare_gems(inputs)
    assert_dense_fp8_replay(
        prepared.run, reference, output_atol=OUTPUT_ATOL, capture=capture
    )
