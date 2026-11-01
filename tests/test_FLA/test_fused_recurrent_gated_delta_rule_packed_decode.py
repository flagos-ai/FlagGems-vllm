# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
import os

import pytest
import torch

import flaggems_vllm

OP = "fused_recurrent_gated_delta_rule_packed_decode"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    (1, 1, 1, 1, 1),
    (3, 2, 4, 17, 23),
    (9, 4, 12, 64, 32),
    (1, 16, 64, 128, 128),
    (32, 4, 8, 128, 128),
]


def _make_inputs(shape, dtype, state_dtype, device, strided=False, invalid=True):
    B, H, HV, K, V = shape
    gen = torch.Generator().manual_seed(20261026)

    def rand(size, dt=dtype):
        return (torch.randn(size, generator=gen) * 0.2).to(dtype=dt, device=device)

    width = 2 * H * K + HV * V
    pad = 7 if strided else 0
    mixed = rand((B, width + pad))[:, :width]
    a = rand((B, HV + pad))[:, :HV]
    b = rand((B, HV + pad))[:, :HV]
    state = rand((B + 3, HV * V * K + pad), state_dtype)
    state = state[:, : HV * V * K].view(B + 3, HV, V, K)
    indices = torch.arange(1, B + 1, dtype=torch.int64).flip(0)
    if invalid and B >= 3:
        indices[-2:] = torch.tensor([0, -1])
    storage = torch.zeros((B * 2,), dtype=torch.int64)
    storage[::2] = indices
    return dict(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=rand((HV,)),
        dt_bias=rand((HV,)),
        scale=K**-0.5,
        initial_state=state,
        out=torch.empty((B, 1, HV, V), dtype=dtype, device=device),
        ssm_state_indices=storage.to(device)[::2],
    )


def _reference(inputs, normalize=False):
    """Independent FP32 CPU math, including the old beta rounding boundary."""
    mixed = inputs["mixed_qkv"].detach().cpu().float()
    state = inputs["initial_state"].detach().cpu().clone()
    B = mixed.shape[0]
    HV, V, K = state.shape[1:]
    H = (mixed.shape[1] - HV * V) // (2 * K)
    q = mixed[:, : H * K].reshape(B, H, K).repeat_interleave(HV // H, 1)
    k = mixed[:, H * K : 2 * H * K].reshape(B, H, K).repeat_interleave(HV // H, 1)
    v = mixed[:, 2 * H * K :].reshape(B, HV, V)
    if normalize:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * inputs["scale"]
    a = inputs["a"].detach().cpu().float()
    bias = inputs["dt_bias"].detach().cpu().float()
    x = a + bias
    sp = torch.where(x <= 20, torch.log1p(torch.exp(x.clamp(max=20))), x)
    decay = torch.exp(-torch.exp(inputs["A_log"].detach().cpu().float()) * sp)
    beta = torch.sigmoid(inputs["b"].detach().cpu().float())
    beta = beta.to(inputs["b"].dtype).float()
    out = torch.zeros((B, 1, HV, V), dtype=inputs["out"].dtype)
    for n, slot in enumerate(inputs["ssm_state_indices"].cpu().tolist()):
        if slot <= 0:
            continue
        h = state[slot].float() * decay[n, :, None, None]
        residual = (v[n] - (h * k[n, :, None, :]).sum(-1)) * beta[n, :, None]
        h = h + residual[:, :, None] * k[n, :, None, :]
        out[n, 0] = (h * q[n, :, None, :]).sum(-1).to(out.dtype)
        state[slot] = h.to(state.dtype)
    return out, state


def _assert_result(inputs, expected, dtype):
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 1e-2)
    torch.testing.assert_close(inputs["out"].cpu(), expected[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(
        inputs["initial_state"].cpu(), expected[1], atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("entry", ["generic", "top"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fp32_state", [False, True])
@pytest.mark.parametrize("normalize,strided", [(False, False), (True, True)])
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_packed_decode(entry, shape, dtype, fp32_state, normalize, strided):
    data = _make_inputs(
        shape,
        dtype,
        torch.float32 if fp32_state else dtype,
        flaggems_vllm.device,
        strided,
    )
    expected = _reference(data, normalize)
    fn = getattr(flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP)
    result = fn(**data, use_qk_l2norm_in_kernel=normalize)
    assert result[0] is data["out"] and result[1] is data["initial_state"]
    _assert_result(data, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_multistep_and_beta_rounding(dtype):
    data = _make_inputs((3, 1, 2, 17, 23), dtype, torch.float32, flaggems_vllm.device)
    ref_data = dict(data)
    ref_data["initial_state"] = data["initial_state"].cpu().clone()
    for _ in range(8):
        expected = _reference(ref_data, True)
        ref_data["initial_state"] = expected[1]
        getattr(flaggems_vllm.ops, OP)(**data, use_qk_l2norm_in_kernel=True)
        _assert_result(data, expected, dtype)
    tiny = _make_inputs((1, 1, 1, 1, 1), dtype, torch.float32, flaggems_vllm.device)
    tiny["mixed_qkv"].fill_(1)
    tiny["initial_state"].zero_()
    tiny["b"].fill_(0.5)
    getattr(flaggems_vllm.ops, OP)(**tiny)
    beta = torch.sigmoid(torch.tensor(0.5)).to(dtype).float()
    torch.testing.assert_close(
        tiny["initial_state"][1, 0, 0, 0].cpu(), beta, atol=1e-6, rtol=1e-6
    )

    # Check low-precision beta rounding at saturation and FP16 subnormal values.
    values = torch.tensor(
        [-30.0, -18.0, -17.0, -16.0, -10.0, -1.0, 0.0, 0.5, 1.0, 10.0, 30.0]
    )
    wide = _make_inputs(
        (len(values), 1, 1, 1, 1),
        dtype,
        torch.float32,
        flaggems_vllm.device,
        invalid=False,
    )
    wide["mixed_qkv"].fill_(1)
    wide["initial_state"].zero_()
    wide["b"].copy_(values.to(dtype).to(flaggems_vllm.device)[:, None])
    getattr(flaggems_vllm.ops, OP)(**wide)
    slots = wide["ssm_state_indices"].cpu()
    actual = wide["initial_state"].cpu()[slots, 0, 0, 0]
    expected = torch.sigmoid(values.to(dtype).float()).to(dtype).float()
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_empty_and_metadata_errors():
    fn = getattr(flaggems_vllm.ops, OP)
    data = _make_inputs(
        (0, 1, 1, 8, 8), torch.float32, torch.float32, flaggems_vllm.device
    )
    result = fn(**data)
    assert result[0] is data["out"] and result[1] is data["initial_state"]
    data = _make_inputs(
        (1, 1, 1, 8, 8), torch.float32, torch.float32, flaggems_vllm.device
    )
    for key, value in [
        ("mixed_qkv", data["mixed_qkv"].unsqueeze(0)),
        ("out", data["out"].squeeze(0)),
        ("ssm_state_indices", data["ssm_state_indices"].float()),
        ("initial_state", data["initial_state"].transpose(1, 2)),
    ]:
        bad = dict(data)
        bad[key] = value
        with pytest.raises((ValueError, NotImplementedError)):
            fn(**bad)


@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia", reason="NVIDIA native vLLM comparison"
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_native_vllm(dtype, strided):
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode as native,
    )

    data = _make_inputs(
        (9, 4, 12, 128, 128), dtype, torch.float32, flaggems_vllm.device, strided
    )
    state0 = data["initial_state"].clone()
    expected = _reference(data, True)
    native(**data, use_qk_l2norm_in_kernel=True)
    _assert_result(data, expected, dtype)
    native_out = data["out"].clone()
    native_state = data["initial_state"].clone()
    data["initial_state"].copy_(state0)
    getattr(flaggems_vllm.ops, OP)(**data, use_qk_l2norm_in_kernel=True)
    _assert_result(data, (native_out.cpu(), native_state.cpu()), dtype)


@pytest.mark.parametrize("entry", ["generic", "top"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(9, 4, 12, 64, 32), (1, 16, 64, 128, 128)])
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_parallel_multistep_state_updated_once(entry, dtype, shape):
    # 108/256 logical tasks exceed Ascend 910B1's 48 Vector cores. Task zero
    # must remain valid: invalid first rows can hide duplicate execution.
    before = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    data = _make_inputs(
        shape, dtype, torch.float32, flaggems_vllm.device, strided=True, invalid=False
    )
    ref_data = dict(data)
    ref_data["initial_state"] = data["initial_state"].cpu().clone()
    fn = getattr(flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP)
    if flaggems_vllm.vendor_name == "ascend":
        from flaggems_vllm.runtime.backend._ascend.ops.fused_recurrent_gated_delta_rule_packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode as specialized,
        )

        assert getattr(flaggems_vllm, OP) is specialized
    for _ in range(4):
        expected = _reference(ref_data, True)
        ref_data["initial_state"] = expected[1]
        result = fn(**data, use_qk_l2norm_in_kernel=True)
        assert result[0] is data["out"] and result[1] is data["initial_state"]
        _assert_result(data, expected, dtype)
        assert os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") == before


@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("nvidia", "hygon", "thead"),
    reason="GPU metadata fast path",
)
@pytest.mark.parametrize("entry", ["generic", "top"])
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
def test_gpu_metadata_revalidated(entry):
    # Warm calls must revalidate metadata, rather than cache tensor identity.
    data = _make_inputs(
        (3, 2, 4, 17, 23), torch.float32, torch.float32, flaggems_vllm.device, True
    )
    fn = getattr(flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP)
    initial = data["initial_state"].clone()
    expected = _reference(data, True)
    fn(**data, use_qk_l2norm_in_kernel=True)
    _assert_result(data, expected, torch.float32)
    for key, value in [
        ("mixed_qkv", data["mixed_qkv"][:, :-1]),
        ("mixed_qkv", data["mixed_qkv"].double()),
        ("a", data["a"][:2]),
        ("b", data["b"][:, :2]),
        ("A_log", data["A_log"][:2]),
        ("dt_bias", data["dt_bias"].unsqueeze(0)),
        ("ssm_state_indices", data["ssm_state_indices"][:2]),
        ("b", data["b"].cpu()),
    ]:
        bad = dict(data)
        bad[key] = value
        with pytest.raises((ValueError, NotImplementedError)):
            fn(**bad, use_qk_l2norm_in_kernel=True)
    # Same output object, mutated shape: an identity-based validation cache
    # would incorrectly accept this call after the previous successful launch.
    data["out"].squeeze_(1)
    with pytest.raises(ValueError):
        fn(**data, use_qk_l2norm_in_kernel=True)
    data["out"].unsqueeze_(1)
    data["initial_state"].copy_(initial)
    fn(**data, use_qk_l2norm_in_kernel=True)
    _assert_result(data, expected, torch.float32)
