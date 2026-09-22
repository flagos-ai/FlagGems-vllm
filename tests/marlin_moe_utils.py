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

"""Reference data shared by the dedicated weight-only Marlin MoE tests."""

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3, QUANT_TYPE_UINT4B8
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as generic_fused_marlin_moe,
)
from flaggems_vllm.runtime import torch_device_fn

QUICK_CONFIGS = [
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
]
PRODUCTION_GEOMETRIES = [
    (8, 4096, 14336, 2),
    (256, 7168, 2048, 8),
    (512, 4096, 1024, 10),
    (256, 4096, 2048, 6),
]
INT4_ORIGINAL_CONFIGS = (
    QUICK_CONFIGS
    + [
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
    ]
    + [
        (tokens, *geometry)
        for geometry in PRODUCTION_GEOMETRIES
        for tokens in (1, 16, 64)
    ]
)
FP8_ORIGINAL_CONFIGS = QUICK_CONFIGS + [
    (tokens, *geometry)
    for geometry in PRODUCTION_GEOMETRIES
    for tokens in (1, 16, 64, 256)
]
# Preserve all ordinary-precision configurations, including its opt-in stress set.
ORDINARY_CONFIGS = (
    [
        (1, 8, 128, 256, 2),
        (4, 8, 128, 256, 2),
        (8, 4, 64, 128, 2),
        (16, 8, 256, 512, 2),
        (32, 8, 128, 256, 4),
        (64, 8, 256, 512, 2),
        (128, 16, 128, 256, 4),
        (4, 16, 512, 1024, 2),
        (10, 256, 2048, 128, 8),
        (256, 256, 2048, 128, 8),
    ]
    + [(tokens, 8, 4096, 14336, 2) for tokens in (1, 4, 16, 64, 128, 256, 512)]
    + [(tokens, 256, 7168, 2048, 8) for tokens in (1, 4, 16, 64, 128, 256)]
)
INT4_CONFIGS = list(dict.fromkeys(INT4_ORIGINAL_CONFIGS + ORDINARY_CONFIGS))
FP8_CONFIGS = list(dict.fromkeys(FP8_ORIGINAL_CONFIGS + ORDINARY_CONFIGS))


def supported_device():
    if flaggems_vllm.fused_marlin_moe is not generic_fused_marlin_moe:
        return True
    if flaggems_vllm.vendor_name != "nvidia" or flaggems_vllm.device != "cuda":
        return False
    major, _ = torch_device_fn.get_device_capability()
    return major == 9


def decode_weight(weight, scale, precision, group_size, dtype):
    """Decode stored codes and stored scales, one expert at a time."""
    experts, rows, packed_k = weight.shape
    width = packed_k * 2 if precision == "int4" else packed_k
    group = width if group_size == -1 else group_size
    result = torch.empty((experts, rows, width), device=weight.device, dtype=dtype)
    for expert in range(experts):
        if precision == "int4":
            codes = weight[expert].to(torch.int32)
            values = torch.stack((codes & 15, codes >> 4), dim=-1)
            values = values.reshape(rows, width).float() - 8.0
        else:
            values = weight[expert].view(torch.float8_e4m3fn).float()
        result[expert] = (
            values * scale[expert].float().repeat_interleave(group, dim=-1)
        ).to(dtype)
    return result


def make_inputs(config, dtype, precision, group_size=128):
    """Quantize one expert at a time to bound production-shape scratch memory."""
    tokens, experts, hidden, intermediate, topk = config
    device = flaggems_vllm.device
    torch.manual_seed(0)
    hidden_states = torch.randn((tokens, hidden), device=device, dtype=dtype)
    if precision == "int4":
        hidden_states = hidden_states / 10.0

    def make_weight(rows, width):
        group = width if group_size == -1 else group_size
        assert width % group == 0
        qdtype = torch.uint8 if precision == "int4" else torch.float8_e4m3fn
        storage_width = width // 2 if precision == "int4" else width
        codes = torch.empty((experts, rows, storage_width), device=device, dtype=qdtype)
        scales = torch.empty(
            (experts, rows, width // group), device=device, dtype=dtype
        )
        for expert in range(experts):
            weight = torch.randn((rows, width), device=device, dtype=dtype) / 10.0
            grouped = weight.float().reshape(rows, -1, group)
            bound = 7.0 if precision == "int4" else 448.0
            scale = (grouped.abs().amax(-1) / bound).clamp_min(1e-8)
            normalized = grouped / scale.unsqueeze(-1)
            if precision == "int4":
                unpacked = (normalized.round().clamp(-7, 7) + 8).to(torch.uint8)
                unpacked = unpacked.reshape(rows, width)
                codes[expert] = unpacked[:, 0::2] | (unpacked[:, 1::2] << 4)
            else:
                codes[expert] = (
                    normalized.clamp(-448, 448).to(qdtype).reshape(rows, width)
                )
            scales[expert] = scale
        return codes, scales, decode_weight(codes, scales, precision, group_size, dtype)

    w1, s1, r1 = make_weight(2 * intermediate, hidden)
    w2, s2, r2 = make_weight(hidden, intermediate)
    gating = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, -1), topk, -1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(dtype)
    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "bias1": None,
        "bias2": None,
        "w1_scale": s1,
        "w2_scale": s2,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "quant_type_id": (
            QUANT_TYPE_UINT4B8 if precision == "int4" else QUANT_TYPE_FP8_E4M3
        ),
        "group_size": group_size,
    }, (r1, r2)


def reference(args, weights):
    """Independent FP32 SwiGLU reference over exactly decoded weights."""
    hs = args["hidden_states"]
    ids, routing = args["topk_ids"], args["topk_weights"]
    w1, w2 = weights
    result = torch.zeros(hs.shape, device=hs.device, dtype=torch.float32)
    on_input = args.get("apply_router_weight_on_input", False)
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        x = hs[tokens].float()
        weight = routing[tokens, slots, None].float()
        if on_input:
            x = x * weight
        gate, up = (x @ w1[expert].float().T).chunk(2, -1)
        values = (torch.nn.functional.silu(gate) * up) @ w2[expert].float().T
        # torch_musa 2.7.1 index_add_ ignores the stride-2 indices from where.
        result.index_add_(
            0, tokens.contiguous(), values if on_input else values * weight
        )
    return result


def check_result(result, expected, hidden_states):
    assert result.shape == hidden_states.shape
    assert result.dtype == hidden_states.dtype
    assert result.device == hidden_states.device
    assert torch.isfinite(result).all()
    torch_device_fn.synchronize()
    # Preserve the legacy vLLM Marlin mean-relative-error limit.
    denominator = expected.abs().mean().clamp_min(1e-12)
    error = (result.float() - expected).abs().mean() / denominator
    assert error < 0.04, f"mean relative error={error.item():.6f}"


def check_public(args, weights):
    expected = reference(args, weights)
    result = flaggems_vllm.fused_marlin_moe(**args)
    check_result(result, expected, args["hidden_states"])
    return result


def check_output(precision, dtype, shape, mode):
    tokens, hidden, intermediate = shape
    args, weights = make_inputs((tokens, 4, hidden, intermediate, 2), dtype, precision)
    expected = reference(args, weights)
    hs = args["hidden_states"]
    destination = torch.empty_like(hs) if mode == "out" else hs
    if mode == "inplace":
        args["inplace"] = True
    else:
        args["output"] = destination
    result = flaggems_vllm.fused_marlin_moe(**args)
    assert result is destination
    check_result(result, expected, hs)


def check_mutation(precision, dtype, field):
    args, weights = make_inputs((17, 4, 256, 512, 2), dtype, precision)
    check_public(args, weights)
    value = args[field]
    address = value.data_ptr()
    if field.endswith("scale"):
        value.mul_(1.5)
    else:
        value.view(torch.uint8).bitwise_xor_(255 if precision == "int4" else 128)
    assert value.data_ptr() == address
    weights = tuple(
        decode_weight(args[key], args[f"{key}_scale"], precision, 128, dtype)
        for key in ("w1", "w2")
    )
    check_public(args, weights)
    check_public(args, weights)


def check_decode_edges(precision, dtype, group_size):
    args, _ = make_inputs((3, 2, 256, 512, 2), dtype, precision, group_size)
    for key in ("w1", "w2"):
        raw = args[key].view(torch.uint8)
        codes = torch.arange(raw.numel(), device=raw.device) % 256
        if precision == "fp8":
            # 0x7f and 0xff are NaNs, outside the finite-weight contract.
            codes = torch.where((codes & 127) == 127, 0, codes)
        raw.copy_(codes.to(torch.uint8).reshape(raw.shape))
        scale = args[f"{key}_scale"]
        rows = torch.arange(scale.shape[1], device=raw.device).float().view(1, -1, 1)
        # Unequal rows avoid exact signed-weight cancellation in the down projection.
        scale.copy_((0.001 + (rows % 17) * 0.0001).expand_as(scale))
    weights = tuple(
        decode_weight(args[key], args[f"{key}_scale"], precision, group_size, dtype)
        for key in ("w1", "w2")
    )
    check_public(args, weights)


def check_large_expert_stride(precision, dtype):
    args, weights = make_inputs((1, 2, 128, 128, 1), dtype, precision)
    for key in ("w1", "w2"):
        original = args[key]
        wide = torch.empty_strided(
            original.shape,
            (2**31 + 256, original.shape[2], 1),
            dtype=original.dtype,
            device=original.device,
        )
        wide.copy_(original)
        assert wide.stride(0) * wide.element_size() > 2**31
        args[key] = wide
    args["topk_ids"].fill_(1)
    check_public(args, weights)


def check_large_route_reduction(topk, hidden):
    from flaggems_vllm.runtime.backend._mthreads.fused.fused_marlin_moe import _moe_sum

    # Keep each stride below int32 while the final token offset exceeds it.
    tokens = 2**31 // (topk * hidden) + 1
    source = torch.ones(
        (tokens, topk, hidden), device=flaggems_vllm.device, dtype=torch.bfloat16
    )
    output = torch.empty((tokens, hidden), device=source.device, dtype=source.dtype)
    _moe_sum(source, output)
    for row in (0, tokens - 2, tokens - 1):
        torch.testing.assert_close(
            output[row], torch.full_like(output[row], topk), rtol=0, atol=0
        )


@triton.jit
def _decode_test_kernel(
    W,
    S,
    OUT,
    DECODE: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.full((), 0, tl.int32)
    values = DECODE(
        W,
        S,
        expert,
        offsets // K,
        offsets % K,
        N,
        K,
        WE,
        WN,
        WK,
        SE,
        SN,
        SG,
        Q,
        GROUP,
        OUT.dtype.element_ty,
    )
    tl.store(OUT + offsets, values, offsets < N * K)


def check_exact_decode(precision, dtype):
    from flaggems_vllm.runtime.backend._mthreads.fused.fused_marlin_moe import (
        _decode_weight,
    )

    raw_cpu = torch.arange(256, dtype=torch.uint8)
    if precision == "int4":
        weight_cpu = raw_cpu.reshape(1, 2, 128)
        expected = torch.stack((weight_cpu & 15, weight_cpu >> 4), -1)
        expected = (expected.reshape(2, 256).float() - 8).to(dtype)
    else:
        weight_cpu = raw_cpu.reshape(1, 1, 256)
        expected = weight_cpu.view(torch.float8_e4m3fn).float().to(dtype).squeeze(0)
    weight = weight_cpu.to(flaggems_vllm.device)
    scale = torch.ones((1, expected.shape[0], 2), device=weight.device, dtype=dtype)
    actual = torch.empty(expected.shape, device=weight.device, dtype=dtype)
    _decode_test_kernel[(triton.cdiv(actual.numel(), 256),)](
        weight,
        scale,
        actual,
        _decode_weight,
        *expected.shape,
        *weight.stride(),
        *scale.stride(),
        0 if precision == "int4" else 1,
        128,
        256,
    )
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


def check_scale_strides(precision, dtype, group_size):
    args, weights = make_inputs(
        (17, 4, 2 * group_size, 4 * group_size, 2), dtype, precision, group_size
    )
    for key in ("w1_scale", "w2_scale"):
        original = args[key]
        experts, rows, groups = original.shape
        backing = torch.empty(
            (2 * experts, 2 * rows, 3 * groups),
            device=original.device,
            dtype=original.dtype,
        )
        strided = backing[::2, 1::2, ::3]
        strided.copy_(original)
        assert strided.shape == original.shape and strided.stride(-1) == 3
        args[key] = strided
    check_public(args, weights)


def check_graph_mutation(precision):
    graph_type = getattr(torch_device_fn, "MUSAGraph", None)
    graph_context = getattr(torch_device_fn, "graph", None)
    if graph_type is None or graph_context is None:
        pytest.skip("This torch_musa runtime does not expose MUSA graph capture")
    probe = torch.ones((1,), device=flaggems_vllm.device)
    probe.add_(1)
    torch_device_fn.synchronize()
    try:
        probe_graph = graph_type()
        with graph_context(probe_graph):
            probe.add_(1)
        probe_graph.replay()
        torch_device_fn.synchronize()
    except RuntimeError as error:
        if (
            "not supported" in str(error).lower()
            or "not compiled" in str(error).lower()
        ):
            pytest.skip(f"The MUSA runtime cannot capture a basic add: {error}")
        raise

    dtype = torch.bfloat16
    args, weights = make_inputs((4, 4, 128, 256, 2), dtype, precision)
    for _ in range(3):
        flaggems_vllm.fused_marlin_moe(**args)
    torch_device_fn.synchronize()
    graph = graph_type()
    with graph_context(graph):
        result = flaggems_vllm.fused_marlin_moe(**args)
    graph.replay()
    check_result(result, reference(args, weights), args["hidden_states"])
    for field in ("w1", "w2", "w1_scale", "w2_scale"):
        value = args[field]
        address = value.data_ptr()
        if field.endswith("scale"):
            value.mul_(1.5)
        else:
            value.view(torch.uint8).bitwise_xor_(255 if precision == "int4" else 128)
        assert value.data_ptr() == address
        weights = tuple(
            decode_weight(args[key], args[f"{key}_scale"], precision, 128, dtype)
            for key in ("w1", "w2")
        )
        expected = reference(args, weights)
        graph.replay()
        check_result(result, expected, args["hidden_states"])
