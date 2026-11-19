# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Packed GDN decode: native updates both the full state pool and out buffer."""

import math

import torch

OP_NAME = "fused_recurrent_gated_delta_rule_packed_decode"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = ("initial_state", "out")
NATIVE_UNAVAILABLE_REASON = None

SHAPES = [
    (1, 8, 16, 128, 128),
    (4, 8, 16, 128, 128),
    (16, 8, 16, 128, 128),
    (64, 8, 16, 128, 128),
    (256, 8, 16, 128, 128),
    (9, 4, 12, 64, 32),
    (3, 2, 4, 17, 23),
    (1, 16, 64, 128, 128),
]


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    op = fused_recurrent_gated_delta_rule_packed_decode
    NATIVE_UNAVAILABLE_REASON = (
        None if callable(op) else "Native packed GDN entry is not callable"
    )
    return op if callable(op) else None


def grid():
    return [dict(zip(("B", "H", "HV", "K", "V"), shape)) for shape in SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding[key] for key in ("B", "H", "HV", "K", "V"))
    data = make_inputs(shape, dtype, torch.float32, device, strided=True, invalid=False)
    data["use_qk_l2norm_in_kernel"] = True
    return (), data


def key_shape(binding):
    return [binding[key] for key in ("B", "H", "HV", "K", "V")]


def config(binding, dtype):
    b, h, hv, k, v = key_shape(binding)
    width = 2 * h * k + hv * v

    def tensor(shape, dt, stride=None):
        return {
            "shape": shape,
            "dtype": str(dt),
            "stride": stride or [math.prod(shape[i + 1 :]) for i in range(len(shape))],
        }

    state = tensor([b + 3, hv, v, k], torch.float32, [hv * v * k + 7, v * k, k, 1])
    out = tensor([b, 1, hv, v], dtype)
    return {
        "inputs": {
            "mixed_qkv": tensor([b, width], dtype, [width + 7, 1]),
            "a": tensor([b, hv], dtype, [hv + 7, 1]),
            "b": tensor([b, hv], dtype, [hv + 7, 1]),
            "A_log": tensor([hv], dtype),
            "dt_bias": tensor([hv], dtype),
            "scale": {"scalar": k**-0.5},
            "initial_state": state,
            "out": out,
            "ssm_state_indices": tensor([b], torch.int64, [2]),
            "use_qk_l2norm_in_kernel": {"scalar": True},
        },
        "outputs": {
            "out": {**out, "aliases": "out"},
            "final_state": {**state, "aliases": "initial_state"},
        },
        "dims": dict(binding),
        "accumulate_dtype": "torch.float32",
        "beta_rounding_dtype": str(dtype),
        "input_seed": 20261026,
        "indices": "positive unique slots B..1; stride=2",
        "mutation": ["initial_state (entire pool)", "out"],
        "graph": False,
    }


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
