# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent FP64 oracle and recurrent state contracts."""

import pytest
import torch

import flaggems_vllm as gems
from baseline.fused_recurrent_kda import PERF_CASES, Case, build

CASES = PERF_CASES + [
    Case("min_dims", [1, 2], 1, 1, 1, "table"),
    Case("max_dims", [1, 2], 1, 256, 256, "table"),
    Case("rectangular", [1, 1, 1], 2, 17, 23, strided=True, state_strided=True),
    Case(
        "table_strides",
        [2, 0, 5],
        2,
        32,
        19,
        "strided_table",
        strided=True,
        state_strided=True,
    ),
    Case("vector_beta", [4, 2], 2, 32, 23, "table", vector_beta=True),
    Case("invalid_initial", [1] * 4, 2, 17, 23, "invalid"),
    Case("invalid_write", [3, 2], 2, 17, 23, "invalid_write"),
    Case("unindexed", [2, 0, 3], 2, 32, 32, "unindexed", False),
    Case("plain", [4], 2, 32, 32, "plain", False),
    Case("long", [65], 1, 32, 19, "table", False, True, True),
    Case("empty", [0], 2, 17, 23),
    Case("empty_requests", [], 2, 17, 23),
]


def reference(data):
    q, k, v, gate, beta = (
        data[key].detach().cpu().double() for key in ("q", "k", "v", "g", "beta")
    )
    initial = data["initial_state"].detach().cpu().clone()
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    cu, indices = data["cu_seqlens"], data["ssm_state_indices"]
    if cu is None:
        bounds = [(0, t)] * b
    else:
        values = cu.cpu().tolist()
        bounds = list(zip(values[:-1], values[1:]))
    indices = indices.cpu() if indices is not None else None
    output = torch.zeros(v.shape, dtype=torch.float64)
    final = (
        initial.clone()
        if data["inplace_final_state"]
        else torch.zeros((t, h, dv, dk), dtype=torch.float32)
    )
    for n, (start, end) in enumerate(bounds):
        batch = 0 if cu is not None else n
        slot = batch * t + start
        if indices is not None:
            slot = int(indices[n, 0] if indices.ndim == 2 else indices[n])
            if slot <= 0 or slot >= len(initial):
                continue
        if start == end:
            continue
        acc = initial[slot].double().clone()
        for token in range(start, end):
            qv, kv = q[batch, token], k[batch, token]
            if data["use_qk_l2norm_in_kernel"]:
                qv = qv / (qv.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                kv = kv / (kv.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            qv = qv * data["scale"]
            acc = acc * gate[batch, token, :, None, :].exp()
            delta = v[batch, token] - (acc * kv[:, None, :]).sum(-1)
            bv = beta[batch, token]
            delta *= bv[:, None] if bv.ndim == 1 else bv
            acc += delta[:, :, None] * kv[:, None, :]
            output[batch, token] = (acc * qv[:, None, :]).sum(-1)
            if data["inplace_final_state"]:
                dest = int(indices[n, token - start]) if indices.ndim == 2 else slot
                if 0 < dest < len(final):
                    final[dest] = acc
            else:
                final[batch * t + token] = acc
    return output.to(data["v"].dtype), final


def assert_result(result, expected):
    tolerances = {
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (3e-5, 3e-5),
    }
    for got, ref in zip(result, expected):
        rtol, atol = tolerances[got.dtype]
        torch.testing.assert_close(
            got.detach().cpu(), ref, rtol=rtol, atol=atol, equal_nan=True
        )


@pytest.mark.fused_recurrent_kda
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("normalize", [False, True])
def test_fused_recurrent_kda(case, dtype, normalize):
    data = build(case, dtype, gems.device, normalize)
    before = {
        key: data[key].clone() for key in ("q", "k", "v", "g", "beta", "initial_state")
    }
    expected = reference(data)
    result = gems.fused_recurrent_kda(**data)
    assert result[0].shape == data["v"].shape
    assert result[0].dtype == dtype
    assert (result[1] is data["initial_state"]) == case.inplace
    assert_result(result, expected)
    for key in ("q", "k", "v", "g", "beta"):
        torch.testing.assert_close(data[key], before[key], rtol=0, atol=0)
    torch.testing.assert_close(
        data["initial_state"].cpu(),
        expected[1] if case.inplace else before["initial_state"].cpu(),
        rtol=3e-5,
        atol=3e-5,
    )


@pytest.mark.fused_recurrent_kda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("layout", ["padded", "offset"])
def test_fused_recurrent_kda_state_layout(dtype, inplace, layout):
    data = build(
        Case("state-layout", [1, 2], 2, 32, 23, "table", inplace), dtype, gems.device
    )
    original = data["initial_state"]
    pool, heads, dv, dk = original.shape
    head_stride = dv * dk + (3 if layout == "padded" else 0)
    pool_stride = heads * head_stride + (5 if layout == "padded" else 0)
    offset = int(layout == "offset")
    flat = torch.full(
        (pool * pool_stride + 8,), 9.25, dtype=torch.float32, device=gems.device
    )
    state = flat.as_strided(original.shape, (pool_stride, head_stride, dk, 1), offset)
    state.copy_(original)
    data["initial_state"] = state
    before = flat.cpu().clone()
    padding = torch.ones(flat.shape, dtype=torch.bool)
    padding.as_strided(state.shape, state.stride(), offset).fill_(False)
    expected = reference(data)
    assert_result(gems.fused_recurrent_kda(**data), expected)
    actual = flat.cpu()
    if inplace:
        wanted = before.clone()
        wanted.as_strided(state.shape, state.stride(), offset).copy_(expected[1])
        torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=3e-5)
    else:
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    torch.testing.assert_close(actual[padding], before[padding], rtol=0, atol=0)


@pytest.mark.fused_recurrent_kda
def test_fused_recurrent_kda_live_metadata_and_values():
    data = build(
        Case("live", [1, 2], 1, 16, 16, "table", False), torch.float32, gems.device
    )
    for i in range(3):
        data["initial_state"].fill_(0.1 * i)
        data["cu_seqlens"].copy_(
            torch.tensor([0, 1 + i % 2, 3], dtype=torch.int32, device=gems.device)
        )
        expected = reference(data)
        assert_result(gems.fused_recurrent_kda(**data), expected)
    data["q"].requires_grad_(True)
    with pytest.raises(NotImplementedError, match="inference-only"):
        gems.fused_recurrent_kda(**data)
    data["q"].requires_grad_(False)
    assert_result(gems.fused_recurrent_kda(**data), reference(data))
    shape = data["initial_state"].shape
    data["initial_state"].as_strided_(shape, (0, 256, 16, 1))
    with pytest.raises(ValueError, match="non-overlapping"):
        gems.fused_recurrent_kda(**data)


@pytest.mark.fused_recurrent_kda
@pytest.mark.parametrize(
    "invalid",
    [
        "beta_missing",
        "q_shape",
        "value_shape",
        "dtype",
        "gate_dtype",
        "beta_shape",
        "state_shape",
        "state_dtype",
        "state_overlap",
        "cu_batch",
        "cu_dtype",
        "indices_dtype",
        "multi_token_1d",
        "scale_zero",
        "scale_nan",
        "grad",
        "extra_args",
    ],
)
def test_fused_recurrent_kda_invalid(invalid):
    data = build(PERF_CASES[0], torch.float32, gems.device)
    error = ValueError
    if invalid == "beta_missing":
        data["beta"] = None
    elif invalid == "q_shape":
        data["q"] = data["q"].squeeze(0)
    elif invalid == "value_shape":
        data["v"] = data["v"].squeeze(0)
    elif invalid == "dtype":
        data["k"] = data["k"].half()
    elif invalid == "gate_dtype":
        data["g"] = data["g"].int()
    elif invalid == "beta_shape":
        data["beta"] = data["beta"].reshape(-1)
    elif invalid == "state_shape":
        data["initial_state"] = data["initial_state"][..., :-1]
    elif invalid == "state_dtype":
        data["initial_state"] = data["initial_state"].half()
    elif invalid == "state_overlap":
        data["initial_state"] = data["initial_state"][:1].expand(4, -1, -1, -1)
    elif invalid == "cu_batch":
        data["q"] = data["q"].expand(2, -1, -1, -1)
        data["k"], data["g"] = data["q"], data["q"]
        data["v"] = data["q"]
        data["beta"] = data["beta"].expand(2, -1, -1)
    elif invalid == "cu_dtype":
        data["cu_seqlens"] = data["cu_seqlens"].float()
    elif invalid == "indices_dtype":
        data["ssm_state_indices"] = data["ssm_state_indices"].float()
    elif invalid == "multi_token_1d":
        data = build(PERF_CASES[4], torch.float32, gems.device)
        data["ssm_state_indices"] = data["ssm_state_indices"][:, 0]
    elif invalid == "scale_zero":
        data["scale"] = 0
    elif invalid == "scale_nan":
        data["scale"] = float("nan")
    elif invalid == "grad":
        data["q"].requires_grad_(True)
        error = NotImplementedError
    elif invalid == "extra_args":
        data["unknown"] = True
        error = NotImplementedError
    with pytest.raises(error):
        gems.fused_recurrent_kda(**data)
