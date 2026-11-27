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

import torch

import flaggems_vllm
from flaggems_vllm.utils import has_triton_tle_attrs
from tests.flash_mla_with_kvcache_w8a8_utils import (
    CapturedCall,
    DenseW8A8Inputs,
    PreparedCall,
    assert_dense_fp8_replay,
    dense_fp8_paged_reference,
    quantize_dense_fp8,
)

OUTPUT_ATOL = 1e-2


def supported() -> bool:
    return (
        flaggems_vllm.runtime.torch_device_fn.is_available()
        and has_triton_tle_attrs(
            ("gpu.wgmma", "gpu.warp_specialize", "gpu.copy"), 3, 6, 0
        )
    )


def quantize(
    x: torch.Tensor, *, benchmark: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return quantize_dense_fp8(x, power_of_two=benchmark, dequantize=not benchmark)


def functional_reference(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    return dense_fp8_paged_reference(*inputs[:8], causal=True)


def prepare_baseline(inputs: DenseW8A8Inputs) -> DenseW8A8Inputs:
    from mate.flashmla import get_mla_metadata

    query = (
        torch.cat((inputs.q_nope.float(), inputs.q_rope.float()), -1) * inputs.q_scale
    ).to(torch.bfloat16)
    cache = (
        torch.cat((inputs.k_nope.float(), inputs.k_rope.float()), -1) * inputs.k_scale
    ).to(torch.bfloat16)
    capacity = inputs.block_table.shape[1] * 64
    for row, length in enumerate(inputs.host_lengths):
        cache.view(len(inputs.host_lengths), capacity, 1, 576)[row, length:] = 0
    metadata, splits = get_mla_metadata(
        inputs.lengths,
        inputs.q_nope.shape[1] * inputs.q_nope.shape[2],
        1,
        num_heads_q=inputs.q_nope.shape[2],
        q=query,
    )
    return inputs._replace(
        native_metadata=metadata,
        native_splits=splits,
        native_query=query,
        native_cache=cache,
    )


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


def run_baseline(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    return dense_musa_reference(
        inputs.native_query,
        inputs.native_cache,
        inputs.block_table,
        inputs.lengths,
        inputs.value_dim,
        metadata=inputs.native_metadata,
        num_splits=inputs.native_splits,
        causal=True,
    )


def benchmark_reference(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    run_baseline(inputs)
    return tuple(t.cpu() for t in functional_reference(inputs))


def prepare_gems(inputs: DenseW8A8Inputs) -> PreparedCall:
    output, lse = flaggems_vllm.flash_mla_with_kvcache_w8a8(*inputs[:9], causal=True)
    return PreparedCall(
        lambda: flaggems_vllm.flash_mla_with_kvcache_w8a8(
            *inputs[:9], causal=True, out=output, lse=lse
        ),
        output,
        lse,
    )


def capture(run) -> CapturedCall:
    device = flaggems_vllm.runtime.torch_device_fn
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    graph = device.MUSAGraph()
    with device.stream(stream):
        with device.graph(graph):
            output, lse = run()
    device.current_stream().wait_stream(stream)
    return CapturedCall(graph.replay, output, lse)


def validate_functional(
    inputs: DenseW8A8Inputs,
    actual: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output, lse = actual
    assert_dense_fp8_replay(
        lambda: flaggems_vllm.flash_mla_with_kvcache_w8a8(
            *inputs[:9], causal=True, num_splits=1, out=output, lse=lse
        ),
        reference,
        output_atol=OUTPUT_ATOL,
        capture=capture,
    )
