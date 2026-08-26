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

"""Correctness and graph-capture tests for the optimized mHC complete path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flaggems_vllm.ops.mhc.mhc_fused_post_pre import mhc_fused_post_pre
from flaggems_vllm.ops.mhc.mhc_prenorm import mhc_prenorm_gemm

HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_COUNT = 24


@dataclass
class Inputs:
    x: torch.Tensor
    residual: torch.Tensor
    post_mix: torch.Tensor
    comb_mix: torch.Tensor
    fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    norm_weight: torch.Tensor


def _require_gpu_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA and Triton are required")


def _make_inputs(num_tokens: int, seed: int = 42) -> Inputs:
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn((num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    residual = torch.randn(
        (num_tokens, HC_MULT, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    post_mix = 2.0 * torch.sigmoid(
        torch.randn((num_tokens, HC_MULT), device=device, dtype=torch.float32) * 0.1
    )
    comb_mix = torch.softmax(
        torch.randn((num_tokens, HC_MULT, HC_MULT), device=device, dtype=torch.float32),
        dim=-1,
    )
    for _ in range(3):
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + 1e-6)
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + 1e-6)
    fn = (
        torch.randn(
            (MIX_COUNT, HC_MULT * HIDDEN_SIZE),
            device=device,
            dtype=torch.float32,
        )
        * 1e-4
    )
    hc_scale = torch.randn((3,), device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn((MIX_COUNT,), device=device, dtype=torch.float32) * 0.1
    norm_weight = (
        1.0 + torch.randn((HIDDEN_SIZE,), device=device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    return Inputs(
        x=x,
        residual=residual,
        post_mix=post_mix,
        comb_mix=comb_mix,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
    )


def _candidate(inputs: Inputs) -> tuple[torch.Tensor, ...]:
    return mhc_fused_post_pre(
        inputs.x,
        inputs.residual,
        inputs.post_mix.unsqueeze(-1),
        inputs.comb_mix,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        n_splits=1,
        tile_n=1,
        norm_weight=inputs.norm_weight,
        norm_eps=1e-6,
    )


def _reference(inputs: Inputs) -> tuple[torch.Tensor, ...]:
    # Keep the reference independent from every candidate GPU implementation.
    # Running it on CPU also prevents TF32/library-specific behavior from
    # becoming part of the expected result.
    x = inputs.x.cpu().float()
    residual = inputs.residual.cpu().float()
    post_mix = inputs.post_mix.cpu()
    comb_mix = inputs.comb_mix.cpu()
    fn = inputs.fn.cpu()
    hc_scale = inputs.hc_scale.cpu()
    hc_base = inputs.hc_base.cpu()
    norm_weight = inputs.norm_weight.cpu().float()

    residual_cur = (
        x.unsqueeze(1) * post_mix.unsqueeze(-1) + torch.bmm(comb_mix.mT, residual)
    ).bfloat16()
    residual_fp32 = residual_cur.float()
    residual_2d = residual_fp32.flatten(1)
    mixes = residual_2d @ fn.mT
    residual_inv_rms = torch.rsqrt(
        residual_2d.square().sum(-1, keepdim=True) / (HC_MULT * HIDDEN_SIZE) + 1e-6
    )
    mixes = mixes * residual_inv_rms

    pre = torch.sigmoid(mixes[:, :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + 1e-6
    post = (
        torch.sigmoid(
            mixes[:, HC_MULT : 2 * HC_MULT] * hc_scale[1]
            + hc_base[HC_MULT : 2 * HC_MULT]
        )
        * 2.0
    )
    comb = (mixes[:, 2 * HC_MULT :] * hc_scale[2] + hc_base[2 * HC_MULT :]).view(
        -1, HC_MULT, HC_MULT
    )
    comb = torch.softmax(comb, dim=-1) + 1e-6
    comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + 1e-6)

    weighted_fp32 = (pre.unsqueeze(-1) * residual_fp32).sum(dim=1)
    weighted_bf16 = weighted_fp32.bfloat16()
    layer_inv_rms = torch.rsqrt(
        weighted_fp32.square().sum(-1, keepdim=True) / HIDDEN_SIZE + 1e-6
    )
    layer = (weighted_bf16.float() * layer_inv_rms * norm_weight).bfloat16()
    device = inputs.residual.device
    return (
        residual_cur.to(device),
        post.unsqueeze(-1).to(device),
        comb.to(device),
        layer.to(device),
    )


def _assert_outputs_close(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    tolerances = ((1e-2, 1e-2), (1e-3, 1e-3), (1e-3, 1e-3), (2e-2, 2e-2))
    for actual_tensor, expected_tensor, (rtol, atol) in zip(
        actual, expected, tolerances
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


@pytest.mark.mhc_fused_post_pre
@pytest.mark.parametrize("num_tokens", [64, 96, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mhc_fused_post_pre_matches_reference(num_tokens: int, seed: int) -> None:
    _require_gpu_runtime()
    inputs = _make_inputs(num_tokens, seed)
    expected = _reference(inputs)
    actual = _candidate(inputs)
    torch.cuda.synchronize()
    _assert_outputs_close(actual, expected)


@pytest.mark.mhc_fused_post_pre
def test_mhc_fused_post_pre_cuda_graph_replay() -> None:
    _require_gpu_runtime()
    inputs = _make_inputs(96)
    expected = _reference(inputs)
    _candidate(inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _candidate(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs_close(actual, expected)


@pytest.mark.mhc_fused_post_pre
def test_mhc_fused_post_pre_cold_cache_is_safe_across_streams() -> None:
    _require_gpu_runtime()
    inputs = _make_inputs(96, seed=17)
    expected = _reference(inputs)
    torch.cuda.synchronize()

    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    with torch.cuda.stream(first_stream):
        first = _candidate(inputs)
    with torch.cuda.stream(second_stream):
        second = _candidate(inputs)
    torch.cuda.synchronize()

    _assert_outputs_close(first, expected)
    _assert_outputs_close(second, expected)


@pytest.mark.mhc_fused_post_pre
def test_mhc_fused_post_pre_accepts_immutable_inference_weights() -> None:
    _require_gpu_runtime()
    with torch.inference_mode():
        inputs = _make_inputs(64, seed=23)
        expected = _reference(inputs)
        actual = _candidate(inputs)
        torch.cuda.synchronize()
        _assert_outputs_close(actual, expected)


@pytest.mark.mhc_fused_post_pre
def test_mhc_prenorm_rejects_cold_cuda_graph_capture() -> None:
    _require_gpu_runtime()
    residual = torch.randn(
        (64, HC_MULT * HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    fn = torch.randn(
        (MIX_COUNT, HC_MULT * HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="warm-up call"):
        with torch.cuda.graph(graph):
            mhc_prenorm_gemm(residual, fn)


@pytest.mark.mhc_fused_post_pre
def test_mhc_fused_post_pre_rejects_unvalidated_token_count() -> None:
    num_tokens = 32
    x = torch.empty((num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16)
    residual = torch.empty((num_tokens, HC_MULT, HIDDEN_SIZE), dtype=torch.bfloat16)
    post_mix = torch.empty((num_tokens, HC_MULT, 1), dtype=torch.float32)
    comb_mix = torch.empty((num_tokens, HC_MULT, HC_MULT), dtype=torch.float32)
    fn = torch.empty((MIX_COUNT, HC_MULT * HIDDEN_SIZE), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((MIX_COUNT,), dtype=torch.float32)
    norm_weight = torch.empty((HIDDEN_SIZE,), dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="token counts"):
        mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            comb_mix,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            norm_weight=norm_weight,
        )
