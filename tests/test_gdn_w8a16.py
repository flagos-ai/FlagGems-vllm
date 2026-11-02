# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.gdn_w8a16 import dequantize_gdn_state_fp8, quantize_gdn_state_fp8
from tests import conftest as cfg

pytestmark = [
    pytest.mark.gdn_w8a16,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia"
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 9,
        reason="FP8 GDN requires NVIDIA SM90 or newer",
    ),
]

DECODE_N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384]
PREFILL_NT = [
    (1, 2),
    (1, 8),
    (1, 32),
    (4, 4),
    (4, 16),
    (16, 4),
    (1, 256),
    (1, 512),
    (2, 192),
    (4, 256),
    (8, 128),
    (16, 128),
    (32, 64),
    (64, 32),
    (128, 24),
]
SHAPES = [(n, 1) for n in DECODE_N] + PREFILL_NT
if cfg.QUICK_MODE:
    SHAPES = [(1, 1), (32, 1), (128, 1), (1, 8), (2, 192)]


def test_gdn_w8a16_registration():
    import flaggems_vllm.ops as ops

    assert "gdn_w8a16" in ops.__all__
    assert flaggems_vllm.gdn_w8a16 is ops.gdn_w8a16
    entries = [entry for entry in flaggems_vllm._FULL_CONFIG if entry[0] == "gdn_w8a16"]
    assert entries == [("gdn_w8a16", flaggems_vllm.gdn_w8a16)]
    assert flaggems_vllm.FULL_CONFIG_BY_FUNC["gdn_w8a16"] == entries
    registered_names = {entry[0] for entry in flaggems_vllm._FULL_CONFIG}
    assert "quantize_gdn_state_fp8" not in registered_names
    assert "dequantize_gdn_state_fp8" not in registered_names


def make_inputs(n, t, h, hv, dtype, packed=True):
    shape = (1, n * t) if packed else (n, t)
    q = torch.randn(*shape, h, 128, device=flaggems_vllm.device, dtype=dtype)
    k = torch.randn_like(q)
    v = 0.125 * torch.randn(*shape, hv, 128, device=q.device, dtype=dtype)
    g = torch.empty(*shape, hv, device=q.device, dtype=dtype).uniform_(
        math.log(0.98), math.log(0.995)
    )
    beta = torch.rand(*shape, hv, device=q.device, dtype=dtype)
    cu = torch.arange(0, n * t + 1, t, device=q.device, dtype=torch.int32)
    indices = torch.arange(n, device=q.device, dtype=torch.int32)
    return q, k, v, g, beta, cu if packed else None, indices


def reference(q, k, v, g, beta, state, n, t):
    h, hv = q.shape[2], v.shape[2]
    q = q.float().reshape(n, t, h, 128)
    k = k.float().reshape(n, t, h, 128)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(hv // h, dim=2)
    k = k.repeat_interleave(hv // h, dim=2)
    v_view = v.reshape(n, t, hv, 128)
    g = g.reshape(n, t, hv)
    beta = beta.reshape(n, t, hv)
    output = torch.empty_like(v_view)
    state = state.float().clone()
    # Bound reference temporary memory for the largest decode batches.
    for start in range(0, n, 128):
        end = min(start + 128, n)
        acc = state[start:end]
        for token in range(t):
            acc = acc * g[start:end, token].float().exp()[..., None, None]
            key = k[start:end, token]
            prediction = torch.einsum("nhk,nhkv->nhv", key, acc)
            residual = (v_view[start:end, token].float() - prediction) * beta[
                start:end, token
            ].float()[..., None]
            acc = acc + key[..., None] * residual[..., None, :]
            output[start:end, token] = torch.einsum(
                "nhk,nhkv->nhv", q[start:end, token], acc
            ).to(v.dtype)
        state[start:end] = acc
    return output.reshape_as(v), state


@pytest.mark.parametrize("h,hv", [(2, 4), (2, 6)])
@pytest.mark.parametrize("n,t", SHAPES)
def test_gdn_w8a16_accuracy(h, hv, n, t):
    torch.manual_seed(42)
    q, k, v, g, beta, cu, indices = make_inputs(n, t, h, hv, torch.bfloat16)
    initial = torch.zeros(n, hv, 128, 128, device=q.device, dtype=q.dtype)
    fp8, scales = quantize_gdn_state_fp8(initial)
    expected, expected_state = reference(q, k, v, g, beta, initial, n, t)
    actual, returned_fp8, returned_scale = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )
    assert returned_fp8 is fp8 and returned_scale is scales
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("n,t", [(3, 4), (2, 192)])
def test_gdn_w8a16_nonzero_state(dtype, packed, n, t):
    torch.manual_seed(43)
    h, hv = 2, 6
    q, k, v, g, beta, cu, _ = make_inputs(n, t, h, hv, dtype, packed)
    # Strided q/k exercise recurrent address calculations.
    q = torch.stack((q, q), dim=-1)[..., 0]
    k = torch.stack((k, k), dim=-1)[..., 0]
    initial = 0.02 * torch.randn(2 * n, hv, 128, 128, device=q.device, dtype=dtype)
    # Permuted, strided slots must select state and scale together in both paths.
    indices = torch.arange(2 * n - 1, -1, -1, device=q.device, dtype=torch.int64)[::2]
    fp8, scales = quantize_gdn_state_fp8(initial)
    initial = dequantize_gdn_state_fp8(fp8, scales, output_dtype=dtype)
    expected, updated_state = reference(q, k, v, g, beta, initial[indices], n, t)
    expected_state = initial.float().clone()
    expected_state[indices] = updated_state
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)


@pytest.mark.parametrize("n,t", [(4, 1), (32, 1), (3, 3)])
def test_gdn_w8a16_speculative(n, t):
    torch.manual_seed(44)
    h, hv, slots = 4, 8, 3
    q, k, v, g, beta, cu, _ = make_inputs(n, t, h, hv, torch.bfloat16)
    indices = torch.randperm(n * slots, device=q.device).reshape(n, slots)
    accepted = torch.arange(n, device=q.device) % slots + 1
    initial = 0.02 * torch.randn(
        n * slots, hv, 128, 128, device=q.device, dtype=q.dtype
    )
    fp8, scales = quantize_gdn_state_fp8(initial)
    before_fp8, before_scale = fp8.view(torch.uint8).clone(), scales.clone()
    initial = dequantize_gdn_state_fp8(fp8, scales, output_dtype=q.dtype)
    expected = torch.empty_like(v)
    expected_state = initial.float().clone()
    for seq in range(n):
        acc = initial[indices[seq, accepted[seq] - 1]][None]
        for token in range(t):
            offset = seq * t + token
            out, acc = reference(
                q[:, offset : offset + 1],
                k[:, offset : offset + 1],
                v[:, offset : offset + 1],
                g[:, offset : offset + 1],
                beta[:, offset : offset + 1],
                acc,
                1,
                1,
            )
            expected[:, offset : offset + 1] = out
            expected_state[indices[seq, token]] = acc[0]
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)
    written = torch.zeros(n * slots, device=q.device, dtype=torch.bool)
    written[indices[:, :t].reshape(-1)] = True
    assert torch.equal(fp8.view(torch.uint8)[~written], before_fp8[~written])
    assert torch.equal(scales[~written], before_scale[~written])


def test_gdn_w8a16_empty_and_invalid():
    q, k, v, g, beta, cu, indices = make_inputs(2, 1, 2, 4, torch.bfloat16)
    state = torch.zeros(2, 4, 128, 128, device=q.device, dtype=q.dtype)
    fp8, scales = quantize_gdn_state_fp8(state)
    with pytest.raises(ValueError, match="2D per-token"):
        flaggems_vllm.gdn_w8a16(
            q,
            k,
            v,
            g,
            beta,
            128**-0.5,
            fp8,
            scales,
            cu_seqlens=cu,
            ssm_state_indices=indices,
            num_accepted_tokens=indices,
        )
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q[:, :0],
        k[:, :0],
        v[:, :0],
        g[:, :0],
        beta[:, :0],
        128**-0.5,
        fp8,
        scales,
    )
    assert actual.numel() == 0


def test_gdn_w8a16_variable_lengths_and_padding():
    torch.manual_seed(45)
    q, k, v, g, beta, _, _ = make_inputs(1, 5, 2, 4, torch.bfloat16)
    initial = 0.02 * torch.randn(3, 4, 128, 128, device=q.device, dtype=q.dtype)
    fp8, scales = quantize_gdn_state_fp8(initial)
    initial = dequantize_gdn_state_fp8(fp8, scales, output_dtype=q.dtype)
    cu = torch.tensor([0, 0, 1, 5, 5], device=q.device, dtype=torch.int32)
    indices = torch.tensor([2, 1, 0, -1], device=q.device, dtype=torch.int32)
    expected = torch.empty_like(v)
    expected_state = initial.float().clone()
    for start, end, slot in [(0, 1, 1), (1, 5, 0)]:
        out, state = reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            initial[slot : slot + 1],
            1,
            end - start,
        )
        expected[:, start:end] = out
        expected_state[slot] = state[0]
    actual, _, _ = flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        fp8,
        scales,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=4,
    )
    restored = dequantize_gdn_state_fp8(fp8, scales)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=5e-2)
    torch.testing.assert_close(restored, expected_state, atol=2e-2, rtol=0.25)


def test_gdn_w8a16_dynamic_scale_range():
    torch.manual_seed(46)
    amplitude = torch.logspace(-2, 3, 128, device=flaggems_vllm.device)
    state = (
        torch.randn(2, 4, 128, 128, device=amplitude.device) * amplitude
    ).bfloat16()
    fp8, scale = quantize_gdn_state_fp8(state)
    restored = dequantize_gdn_state_fp8(fp8, scale)
    error = (restored - state.float()).abs() / state.float().abs().amax(2, keepdim=True)
    assert error.max().item() < 0.08
    assert error.mean().item() < 0.02
    assert restored.abs().max().item() > 448
