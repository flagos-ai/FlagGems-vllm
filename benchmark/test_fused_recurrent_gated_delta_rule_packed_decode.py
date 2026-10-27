# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph event timing, with state reset outside every timed region."""
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

OP = "fused_recurrent_gated_delta_rule_packed_decode"
SHAPES = [(b, 8, 16, 128, 128) for b in (1, 4, 16, 64, 256)] + [
    (9, 4, 12, 64, 32),
    (3, 2, 4, 17, 23),
    (1, 16, 64, 128, 128),
]


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


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize("entry", ["generic", "top"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", SHAPES)
def test_packed_decode_performance(entry, dtype, shape):
    data = make_inputs(
        shape, dtype, torch.float32, flaggems_vllm.device, strided=True, invalid=False
    )
    state0 = data["initial_state"].clone()
    expected = reference(data, True)
    candidate = getattr(flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP)
    functions = {"gems": candidate}
    native_reason = "No validated equivalent same-device native packed interface"
    if flaggems_vllm.vendor_name == "nvidia":
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode as native,
        )

        functions["native"] = native
        native_reason = None
    for fn in functions.values():
        data["initial_state"].copy_(state0)
        fn(**data, use_qk_l2norm_in_kernel=True)
        assert_result(data, expected, dtype)
    # Compilation/warmup and every reset are explicitly outside timing events.
    for fn in functions.values():
        for _ in range(5):
            data["initial_state"].copy_(state0)
            fn(**data, use_qk_l2norm_in_kernel=True)
    torch_device_fn.synchronize()
    samples = {name: [] for name in functions}
    for i in range(25):
        names = list(functions) if i % 2 == 0 else list(reversed(functions))
        for name in names:
            data["initial_state"].copy_(state0)
            torch_device_fn.synchronize()
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            start.record()
            functions[name](**data, use_qk_l2norm_in_kernel=True)
            end.record()
            torch_device_fn.synchronize()
            samples[name].append(start.elapsed_time(end))
    assert_result(data, expected, dtype)
    config = {
        k: (
            {"shape": list(v.shape), "stride": list(v.stride()), "dtype": str(v.dtype)}
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in data.items()
    }
    row = dict(
        op=OP,
        entry=entry,
        backend=flaggems_vllm.vendor_name,
        dtype=str(dtype),
        shape=shape,
        config=config,
        normalize=True,
        accuracy="PASS",
        performance="PASS",
        graph=False,
        unit="ms",
        protocol="device-event-reset-outside-alternating-median-v1",
        warmup=5,
        repetitions=25,
        samples=samples,
        median_ms={k: statistics.median(v) for k, v in samples.items()},
        native_reason=native_reason,
        candidate_module=candidate.__module__,
    )
    destination = os.environ.get("PACKED_DECODE_RESULTS")
    if not destination:
        raise RuntimeError("Set PACKED_DECODE_RESULTS to a fresh JSONL evidence path")
    with Path(destination).open("a") as f:
        f.write(json.dumps(row) + "\n")
