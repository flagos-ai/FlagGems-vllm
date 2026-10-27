# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Test-only data and independent CPU oracle for the vLLM 0.24.0 contract."""

import torch


def make_inputs(shape, dtype, state_dtype, device, strided=False, invalid=True):
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


def reference(inputs, normalize=False):
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


def assert_result(inputs, expected, dtype):
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 1e-2)
    torch.testing.assert_close(inputs["out"].cpu(), expected[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(
        inputs["initial_state"].cpu(), expected[1], atol=atol, rtol=rtol
    )


def native_vllm024():
    """Fail closed if a different vLLM contract is selected by PYTHONPATH."""
    import hashlib
    import importlib.metadata
    from pathlib import Path

    import vllm.model_executor.layers.fla.ops.fused_recurrent as module

    version = importlib.metadata.version("vllm").split("+")[0]
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if version != "0.24.0" or digest != (
        "3a2a3c5245acba127af326f0b3f287b352776a8eb1e1f6e40c24a767fae3c2ef"
    ):
        raise RuntimeError(
            f"Expected frozen vLLM 0.24.0 source; got {version}, {digest}"
        )
    return module.fused_recurrent_gated_delta_rule_packed_decode
