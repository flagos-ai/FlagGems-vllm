# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Native vLLM KDA decode workloads and full state/reset signatures."""

import math
from typing import NamedTuple

import torch

OP_NAME = "fused_recurrent_kda"
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
IS_INPLACE = True
MUTATED_INPUTS = ("initial_state",)
NATIVE_UNAVAILABLE_REASON = None
INPUT_SEED = 20261102


class Case(NamedTuple):
    name: str
    lengths: list
    heads: int = 4
    key_dim: int = 128
    value_dim: int = 128
    mode: str = "index"
    inplace: bool = True
    strided: bool = False
    state_strided: bool = False
    vector_beta: bool = False


PERF_CASES = [
    Case("decode1", [1]),
    Case("decode16", [1] * 16),
    Case("decode64", [1] * 64),
    Case("decode256", [1] * 256),
    Case("mtp4", [4] * 8, mode="table"),
    Case("ragged", [1, 3, 0, 2], 2, 64, 64, "table"),
    Case("strided_qkv", [1] * 8, 2, 64, 64, strided=True),
    Case("outofplace", [3, 2], 2, 64, 64, "table", False),
]


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        from vllm.model_executor.layers.fla.ops.kda import fused_recurrent_kda
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return fused_recurrent_kda


def grid():
    return [{"case": case} for case in PERF_CASES]


def key_shape(binding):
    return binding["case"][0]


def build_inputs(binding, dtype, device):
    return (), build(Case(*binding["case"]), dtype, device)


def unsupported_reason(binding, dtype):
    case = Case(*binding["case"])
    if case.key_dim != case.value_dim:
        return "Native vLLM 0.24.0 allocates output as empty_like(k), requiring K=V"
    if case.key_dim & (case.key_dim - 1):
        return "Native KDA loads gate without a tail mask; non-power-of-two K is unsafe"
    if case.state_strided:
        return "Native assumes dense H,V,K inside each state slot"
    if case.mode == "strided_table":
        return "Native ignores stride_indices_tok when reading/writing state slots"
    if case.mode == "invalid":
        return "Native returns uninitialized output for invalid initial state slots"
    return None


def build(case, dtype, device, normalize=True):
    case = Case(*case)
    gen = torch.Generator().manual_seed(INPUT_SEED)
    n, total = len(case.lengths), sum(case.lengths)
    plain = case.mode == "plain"
    b, t = (n, case.lengths[0]) if plain else (1, total)
    h, dk, dv = case.heads, case.key_dim, case.value_dim

    def transfer(host, step=1):
        if step == 2:
            storage = torch.empty(
                (*host.shape[:-1], host.shape[-1] * 2), dtype=host.dtype
            )
            storage[..., ::2] = host
            host = storage[..., ::2]
        out = torch.empty_strided(
            host.shape, host.stride(), dtype=host.dtype, device=device
        )
        out.copy_(host)
        return out

    def rand(shape, dt=dtype, kind="normal", strided=case.strided):
        if kind == "beta":
            value = torch.rand(shape, generator=gen)
        else:
            value = torch.randn(shape, generator=gen) * 0.2
            if kind == "gate":
                value = -value.abs()
        return transfer(value.to(dt), 2 if strided else 1)

    q, k = rand((b, t, h, dk)), rand((b, t, h, dk))
    v, g = rand((b, t, h, dv)), rand((b, t, h, dk), torch.float32, "gate")
    beta = rand((b, t, h, dv) if case.vector_beta else (b, t, h), kind="beta")
    width = max(max(case.lengths, default=0), 1)
    pool = (b * t if case.mode in ("plain", "unindexed") else n * width) + 3
    state = rand((pool, h, dv, dk), torch.float32, strided=case.state_strided)
    indices = None
    if case.mode not in ("plain", "unindexed"):
        if case.mode in ("index", "invalid"):
            indices = torch.arange(n, 0, -1, dtype=torch.int64)
            if case.mode == "invalid":
                indices[:3] = torch.tensor([0, -1, pool], dtype=torch.int64)
        else:
            indices = torch.arange(1, n * width + 1, dtype=torch.int64).reshape(
                n, width
            )
            if case.mode == "invalid_write":
                indices[:, -1] = 0
        indices = transfer(indices, 2 if case.mode == "strided_table" else 1)
    cu = (
        None
        if plain
        else torch.tensor(
            [0] + list(torch.tensor(case.lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=dk**-0.5,
        initial_state=state,
        inplace_final_state=case.inplace,
        use_qk_l2norm_in_kernel=normalize,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )


def config(binding, dtype):
    case = Case(*binding["case"])
    n, t = len(case.lengths), sum(case.lengths)
    b = 1
    if case.mode == "plain":
        b, t = n, case.lengths[0]
    h, dk, dv = case.heads, case.key_dim, case.value_dim
    width = max(max(case.lengths, default=0), 1)
    pool = (b * t if case.mode in ("plain", "unindexed") else n * width) + 3

    def tensor(shape, dt, step=1):
        return dict(
            shape=shape,
            dtype=str(dt),
            stride=[math.prod(shape[i + 1 :]) * step for i in range(len(shape))],
        )

    data_step, state_step = 2 if case.strided else 1, 2 if case.state_strided else 1
    state = tensor([pool, h, dv, dk], torch.float32, state_step)
    inputs = {
        "q": tensor([b, t, h, dk], dtype, data_step),
        "k": tensor([b, t, h, dk], dtype, data_step),
        "v": tensor([b, t, h, dv], dtype, data_step),
        "g": tensor([b, t, h, dk], torch.float32, data_step),
        "beta": tensor(
            [b, t, h, dv] if case.vector_beta else [b, t, h], dtype, data_step
        ),
        "initial_state": state,
        "cu_seqlens": (
            {"scalar": None} if case.mode == "plain" else tensor([n + 1], torch.int32)
        ),
        "ssm_state_indices": (
            {"scalar": None}
            if case.mode in ("plain", "unindexed")
            else tensor(
                [n] if case.mode in ("index", "invalid") else [n, width],
                torch.int64,
                2 if case.mode == "strided_table" else 1,
            )
        ),
        "scale": {"scalar": dk**-0.5},
        "inplace_final_state": {"scalar": case.inplace},
        "use_qk_l2norm_in_kernel": {"scalar": True},
    }
    return dict(
        inputs=inputs,
        outputs={
            "out": tensor([b, t, h, dv], dtype),
            "final_state": (
                {**state, "aliases": "initial_state"}
                if case.inplace
                else tensor([t, h, dv, dk], torch.float32)
            ),
        },
        dims=case._asdict(),
        accumulate_dtype="torch.float32",
        input_seed=INPUT_SEED,
        mutation=(
            ["initial_state (entire pool reset outside each event)"]
            if case.inplace
            else []
        ),
        native_unsupported_reason=unsupported_reason(binding, dtype),
        graph=False,
    )
