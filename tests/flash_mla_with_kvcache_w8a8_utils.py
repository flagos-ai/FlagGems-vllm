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

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Callable, NamedTuple

import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close


class DenseW8A8Inputs(NamedTuple):
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    value_dim: int
    native_metadata: torch.Tensor | None
    native_splits: torch.Tensor | None
    host_lengths: tuple[int, ...]
    native_query: torch.Tensor | None = None
    native_cache: torch.Tensor | None = None
    benchmark_shape: tuple = ()


class PreparedCall(NamedTuple):
    run: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    output: torch.Tensor
    lse: torch.Tensor


class CapturedCall(NamedTuple):
    replay: Callable[[], None]
    output: torch.Tensor
    lse: torch.Tensor


@lru_cache(maxsize=1)
def get_backend_adapter() -> ModuleType | None:
    """Load validation only for an implemented backend; other devices skip."""
    vendor = flaggems_vllm.vendor_name
    overrides = dict(flaggems_vllm.runtime.backend.get_customized_ops())
    if vendor != "nvidia" and "flash_mla_with_kvcache_w8a8" not in overrides:
        return None
    return import_module(
        f"flaggems_vllm.runtime.backend._{vendor}.testing.flash_mla_with_kvcache_w8a8"
    )


def quantize_dense_fp8(x, *, power_of_two=False, dequantize=True):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    nope = (x[..., :512].float() / scale).to(torch.float8_e4m3fn)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    if dequantize:
        native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(
            torch.bfloat16
        )
    else:
        native = None
    return nope, rope, scale, native


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
    # Convert before indexing so the reference also works without FP8 gather support.
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


def assert_dense_fp8_close(actual, reference, *, output_atol):
    if reference[0].device.type == "cpu":
        actual = tuple(tensor.cpu() for tensor in actual)
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=output_atol)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=2e-4)


def assert_dense_fp8_replay(run, reference, *, output_atol, capture):
    assert_dense_fp8_close(run(), reference, output_atol=output_atol)
    captured = capture(run)
    captured.output.fill_(float("nan"))
    captured.lse.fill_(float("nan"))
    captured.replay()
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    assert_dense_fp8_close(
        (captured.output, captured.lse), reference, output_atol=output_atol
    )
