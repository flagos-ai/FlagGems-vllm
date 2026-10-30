# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Native recurrent GDN workloads; preserve the benchmark's strided inputs."""

import math

import torch

OP_NAME = "fused_recurrent_gated_delta_rule_fwd"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = ("initial_state",)
NATIVE_UNAVAILABLE_REASON = None

PERF_CASES = [
    ("decode1", [1], 4, 8, 128, 128, "index", True, False),
    ("decode16", [1] * 16, 4, 8, 128, 128, "index", True, False),
    ("decode64", [1] * 64, 4, 8, 128, 128, "index", True, False),
    ("decode256", [1] * 256, 4, 8, 128, 128, "index", True, False),
    ("strided", [1] * 16, 4, 8, 128, 128, "index", True, True),
    ("mtp", [4] * 8, 4, 8, 128, 128, "spec", True, False),
    ("ragged", [1, 3, 2, 4], 2, 4, 64, 32, "table", True, True),
    ("tail", [1] * 3, 2, 4, 17, 23, "index", True, True),
]


def native():
    """The native name differs from this repository's public `_fwd` export."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    op = fused_recurrent_gated_delta_rule
    NATIVE_UNAVAILABLE_REASON = (
        None if callable(op) else "Native recurrent GDN entry is not callable"
    )
    return op if callable(op) else None


def grid():
    return [{"case": case} for case in PERF_CASES]


def build_inputs(binding, dtype, device):
    return (), build(binding["case"], dtype, device)


def key_shape(binding):
    # The native workload is shared by the generic and top-level Gems entries.
    return binding["case"][0]


def unsupported_reason(binding, dtype):
    case = binding["case"]
    if case[6] in ("table", "spec") and case[8]:
        return (
            "Native vLLM table indexing assumes token stride=1; this workload "
            "has stride=2 (previously caused illegal device access). "
            "Retained as an unsupported row pending native stride support."
        )
    return None


def config(binding, dtype):
    name, lengths, h, hv, k, v, mode, inplace, strided = binding["case"]
    n, t = len(lengths), sum(lengths)
    width = max(max(lengths), 4)
    pool = n * width + 3

    def tensor(shape, dt, step=1):
        return {
            "shape": shape,
            "dtype": str(dt),
            "stride": [math.prod(shape[i + 1 :]) * step for i in range(len(shape) - 1)]
            + [step],
        }

    step = 2 if strided else 1
    state = tensor([pool, hv, v, k], torch.float32, step)
    inputs = {
        "q": tensor([1, t, h, k], dtype, step),
        "k": tensor([1, t, h, k], dtype, step),
        "v": tensor([1, t, hv, v], dtype, step),
        "g": tensor([1, t, hv], torch.float32, step),
        "beta": tensor([1, t, hv], dtype, step),
        "initial_state": state,
        "cu_seqlens": tensor([n + 1], torch.int32),
        "ssm_state_indices": tensor(
            [n] if mode == "index" else [n, width],
            torch.int32 if mode == "index" else torch.int64,
            step,
        ),
        "num_accepted_tokens": (
            tensor([n], torch.int32) if mode == "spec" else {"scalar": None}
        ),
        "scale": {"scalar": k**-0.5},
        "inplace_final_state": {"scalar": inplace},
        "use_qk_l2norm_in_kernel": {"scalar": True},
    }
    return {
        "inputs": inputs,
        "outputs": {
            "out": tensor([1, t, hv, v], dtype),
            "final_state": {**state, "aliases": "initial_state"},
        },
        "dims": {"case": name, "lengths": lengths, "H": h, "HV": hv, "K": k, "V": v},
        "accumulate_dtype": "torch.float32",
        "input_seed": 2701,
        "mutation": ["initial_state (entire pool restored outside timing)"],
        "native_unsupported_reason": unsupported_reason(binding, dtype),
        "graph": False,
    }


def build(case, dtype, device, normalize=True):
    name, lengths, H, HV, K, V, mode, inplace, strided = case
    torch.manual_seed(2701)
    N = len(lengths)
    plain = mode in ("plain", "none")
    B, T = (N, lengths[0]) if plain else (1, sum(lengths))

    def rand(shape, dt=dtype):
        if strided:
            x = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dt)[..., ::2]
        else:
            x = torch.randn(shape, dtype=dt)
        # Preserve the tested strides after device transfer.
        out = torch.empty_strided(x.shape, x.stride(), dtype=dt, device=device)
        out.copy_(x)
        return out

    q, k, v = rand((B, T, H, K)), rand((B, T, H, K)), rand((B, T, HV, V))
    k.mul_(0.05)
    g = rand((B, T, HV), torch.float32)
    g.copy_(-g.abs() * 0.1)
    beta = rand((B, T, HV, V) if name == "vector_beta" else (B, T, HV))
    beta.copy_(beta.sigmoid())
    width = max(max(lengths), 4)
    pool = N * width + 3
    state_dtype = dtype if name == "low_state" else torch.float32
    state = rand((pool, HV, V, K), state_dtype)
    state.mul_(0.05)
    indices = None
    if mode in ("index", "table", "spec"):
        if mode == "index":
            indices = torch.arange(1, N + 1, dtype=torch.int32)
            if name == "padding":
                indices[:2] = torch.tensor([0, -1], dtype=torch.int32)
        else:
            indices = torch.arange(1, N * width + 1, dtype=torch.int64).view(N, width)
        if strided:
            buf = torch.empty(
                (*indices.shape[:-1], indices.shape[-1] * 2), dtype=indices.dtype
            )
            buf[..., ::2] = indices
            indices = buf[..., ::2]
        d = torch.empty_strided(
            indices.shape, indices.stride(), dtype=indices.dtype, device=device
        )
        d.copy_(indices)
        indices = d
    cu = (
        None
        if plain
        else torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
    )
    accepted = (
        torch.tensor([3] * N, dtype=torch.int32, device=device)
        if mode == "spec"
        else None
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=None if mode == "none" else state,
        inplace_final_state=inplace,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=normalize,
    )
