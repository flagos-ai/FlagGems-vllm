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

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops.flash_mla_with_kvcache_w8a8 import (
    FlashMLAInt8PreparedHandle,
    prepare_flash_mla_with_kvcache_fwd_w8a8_int8,
)


def supported() -> bool:
    return torch_device_fn.is_available()


def quantize(x: torch.Tensor, *, dequantize: bool = True):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    nope = (x[..., :512].float() / scale).round().to(torch.int8)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    if dequantize:
        native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(
            torch.bfloat16
        )
    else:
        native = None
    return nope, rope, scale, native


def native_reference(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    native = import_module("flash_mla")
    metadata, splits = native.get_mla_metadata(
        cache_seqlens, q.shape[1] * q.shape[2], 1
    )
    return native.flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, splits, **kwargs
    )


def run_baseline(*args, **kwargs):
    return import_module("flash_mla").flash_mla_with_kvcache(*args, **kwargs)


def prepare_benchmark(args: tuple, kwargs: dict) -> tuple[tuple, tuple]:
    native = import_module("flash_mla")
    q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, table, lengths, value_dim = args
    query = (torch.cat((q_nope.float(), q_rope.float()), -1) * q_scale).to(
        torch.bfloat16
    )
    cache = (torch.cat((k_nope.float(), k_rope.float()), -1) * k_scale).to(
        torch.bfloat16
    )
    metadata, splits = native.get_mla_metadata(
        lengths, q_nope.shape[1] * q_nope.shape[2], 1
    )
    handle, _ = prepare_flash_mla_with_kvcache_fwd_w8a8_int8(*args, **kwargs)
    baseline = query, cache, table, lengths, value_dim, metadata, splits, kwargs
    return baseline, (handle,)


def run_prepared(
    handle: FlashMLAInt8PreparedHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    return handle()


def benchmark_reference(inputs: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    from tests.test_flash_mla import _scaled_dot_product_attention

    query, cache, table, lengths, _, _, _, _ = inputs
    batch, queries, heads, _ = query.shape
    output = torch.empty(
        (batch, queries, heads, 512), dtype=query.dtype, device=query.device
    )
    lse = torch.empty((batch, heads, queries), dtype=torch.float32, device=query.device)
    cache_rows = cache.view(batch, -1, 1, 576)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for row, length in enumerate(lengths.tolist()):
            for query_index in range(queries):
                bound = length - queries + query_index + 1
                keys = cache_rows[row, :bound].transpose(0, 1)
                result, logsum = _scaled_dot_product_attention(
                    query[row, query_index].unsqueeze(0),
                    keys,
                    keys[..., :512],
                    1,
                    1,
                    False,
                )
                output[row, query_index] = result[0]
                lse[row, :, query_index] = logsum[0]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return output, lse
