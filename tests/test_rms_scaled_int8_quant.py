# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Direct Hy4-preview fused RMS API; GPU cases are explicit and bounded."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.rms_scaled_int8_quant import (
    rms_scaled_int8_quant,
    supports_rms_scaled_int8_quant,
)


def test_direct_api_is_not_an_aten_registration():
    registered = dict(flaggems_vllm._FULL_CONFIG)
    assert "rms_scaled_int8_quant" not in registered
    assert "supports_rms_scaled_int8_quant" not in registered
    assert registered["add_rms_norm"] is flaggems_vllm.add_rms_norm


def test_cpu_inputs_decline():
    with torch.inference_mode():
        x = torch.ones((4, 2048), dtype=torch.bfloat16)
        weight = torch.ones(2048, dtype=torch.bfloat16)
        assert not supports_rms_scaled_int8_quant(x, weight)
        with pytest.raises(NotImplementedError, match="CUDA inference"):
            rms_scaled_int8_quant(x, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.rms_scaled_int8_quant
@pytest.mark.parametrize("rows", [0, 4, 208, 256])
@torch.inference_mode()
def test_finite_strided_math_and_ordinary_graph(rows):
    # Fixed CPU construction; no GPU RNG or checkpoint, tail row is zero.
    n = 2048
    cpu = ((torch.arange(rows * 2624) % 63 - 31).float() / 32).bfloat16()
    cpu = cpu.reshape(rows, 2624)
    if rows:
        cpu[-1, :n] = 0
    weight_cpu = (0.5 + (torch.arange(n) % 11).float() / 16).bfloat16()
    storage = cpu.cuda()
    x = storage[:, :n]
    weight = weight_cpu.cuda()
    y, q, scale = rms_scaled_int8_quant(x, weight)
    assert y.shape == q.shape == (rows, n)
    assert scale.shape == (rows, 1)
    assert (y.dtype, q.dtype, scale.dtype) == (
        torch.bfloat16,
        torch.int8,
        torch.float32,
    )
    if not rows:
        return
    xf = cpu[:, :n].float()
    norm = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-5)
    norm = (norm.bfloat16() * weight_cpu).bfloat16()
    torch.testing.assert_close(y.cpu(), norm, atol=0.015625, rtol=0.015625)
    # Independently quantize the materialized BF16 value returned by the op.
    z = y.cpu().float()
    absmax = z.abs().amax(-1, keepdim=True)
    expected_scale = absmax / 127
    inverse = torch.where(absmax == 0, 0.0, 127 / absmax)
    expected_q = (z * inverse).round().clamp(-128, 127).to(torch.int8)
    torch.testing.assert_close(q.cpu(), expected_q, atol=0, rtol=0)
    torch.testing.assert_close(scale.cpu(), expected_scale, atol=1e-10, rtol=2e-6)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = rms_scaled_int8_quant(x, weight)
    # Refresh captured inputs and poison every output to prove replay writes
    # current values, including the zero-row quantization contract.
    updated_cpu = cpu.clone()
    updated_cpu[:, :n] = -updated_cpu[:, :n]
    storage.copy_(updated_cpu)
    for value in captured:
        value.fill_(17)
    graph.replay()
    torch.cuda.synchronize()
    updated_xf = updated_cpu[:, :n].float()
    updated_norm = updated_xf / torch.sqrt(
        (updated_xf * updated_xf).mean(-1, keepdim=True) + 1e-5
    )
    updated_norm = (updated_norm.bfloat16() * weight_cpu).bfloat16()
    replay_y, replay_q, replay_scale = (value.cpu() for value in captured)
    torch.testing.assert_close(replay_y, updated_norm, atol=0.015625, rtol=0.015625)
    replay_z = replay_y.float()
    replay_absmax = replay_z.abs().amax(-1, keepdim=True)
    replay_inv = torch.where(replay_absmax == 0, 0.0, 127 / replay_absmax)
    replay_expected_q = (replay_z * replay_inv).round().clamp(-128, 127).to(torch.int8)
    torch.testing.assert_close(replay_q, replay_expected_q, atol=0, rtol=0)
    torch.testing.assert_close(replay_scale, replay_absmax / 127, atol=1e-10, rtol=2e-6)
    torch.testing.assert_close(storage.cpu(), updated_cpu, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), weight_cpu, atol=0, rtol=0)
