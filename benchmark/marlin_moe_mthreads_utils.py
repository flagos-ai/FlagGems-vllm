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

"""MUSA benchmark inputs and independent references for weight-only MoE."""

import importlib
import json

import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

GROUP_SIZE = 128
PRODUCTION_GEOMETRIES = (
    (8, 4096, 14336, 2),
    (256, 7168, 2048, 8),
    (512, 4096, 1024, 10),
    (256, 4096, 2048, 6),
)
# The complete ordinary-precision benchmark/test_fused_moe.py shape set.
ORDINARY_SHAPES = (
    [(tokens, 8, 4096, 14336, 2) for tokens in (1, 4, 16, 64, 128, 256, 512)]
    + [(tokens, 256, 7168, 2048, 8) for tokens in (1, 4, 16, 64, 128, 256)]
    + [(tokens, 256, 2048, 128, 8) for tokens in (1, 16, 64, 512, 1035)]
)


def mthreads_shapes(original, precision):
    # Add medium prefill to each INT4 model; FP8 already includes large prefill.
    extra = (
        [(1024, *geometry) for geometry in PRODUCTION_GEOMETRIES]
        if precision == "int4"
        else []
    )
    return list(dict.fromkeys(list(original) + ORDINARY_SHAPES + extra))


def _decode_expert(weight, scale, precision, dtype):
    if precision == "int4":
        codes = weight.to(torch.int32)
        values = torch.stack((codes & 15, codes >> 4), dim=-1)
        values = values.reshape(weight.shape[0], weight.shape[1] * 2).float() - 8
    else:
        values = weight.float()
    # The GEMMs consume scaled weights rounded to the activation dtype.
    return (values * scale.float().repeat_interleave(GROUP_SIZE, -1)).to(dtype)


def make_weight_bank(geometry, dtype, precision):
    """Quantize one expert at a time; never allocate a full FP32 weight bank."""
    experts, hidden, intermediate = geometry[:3]
    device = flaggems_vllm.device
    torch.manual_seed(17)

    def make_weight(rows, width):
        storage_width = width // 2 if precision == "int4" else width
        storage_dtype = torch.uint8 if precision == "int4" else torch.float8_e4m3fn
        weights = torch.empty(
            (experts, rows, storage_width), device=device, dtype=storage_dtype
        )
        scales = torch.empty(
            (experts, rows, width // GROUP_SIZE), device=device, dtype=dtype
        )
        # Only FP8 needs a decoded bank: vLLM has no matching FP8-W8A16 kernel.
        decoded = (
            torch.empty((experts, rows, width), device=device, dtype=dtype)
            if precision == "fp8"
            else None
        )
        for expert in range(experts):
            source = torch.randn((rows, width), device=device, dtype=dtype)
            grouped = (source.float() / 10.0).reshape(rows, -1, GROUP_SIZE)
            bound = 7.0 if precision == "int4" else 448.0
            scale = (grouped.abs().amax(-1) / bound).clamp_min(1e-8).to(dtype)
            normalized = grouped / scale.float().unsqueeze(-1)
            if precision == "int4":
                codes = (normalized.round().clamp(-7, 7) + 8).to(torch.uint8)
                codes = codes.reshape(rows, width)
                weights[expert] = codes[:, 0::2] | (codes[:, 1::2] << 4)
            else:
                weights[expert] = (
                    normalized.clamp(-448, 448).to(storage_dtype).reshape(rows, width)
                )
            scales[expert] = scale
            if decoded is not None:
                decoded[expert] = _decode_expert(
                    weights[expert], scale, precision, dtype
                )
        return weights, scales, decoded

    w1, s1, r1 = make_weight(2 * intermediate, hidden)
    w2, s2, r2 = make_weight(hidden, intermediate)
    return w1, w2, s1, s2, r1, r2


def make_inputs(config, dtype, precision, bank):
    tokens, experts, hidden, _, topk = config
    w1, w2, s1, s2, r1, r2 = bank
    torch.manual_seed(17 + tokens)
    hs = torch.randn((tokens, hidden), device=flaggems_vllm.device, dtype=dtype)
    if precision == "int4":
        hs = hs * 0.1
    gates = torch.randn((tokens, experts), device=hs.device, dtype=torch.float32)
    routing, ids = torch.topk(torch.softmax(gates, -1), topk, -1)
    routing = routing / routing.sum(-1, keepdim=True)
    if precision == "int4":
        return hs, w1, w2, s1, s2, w1, w2, s1, s2, routing, ids
    return hs, r1, r2, None, None, w1, w2, s1, s2, routing, ids


def _reference(inputs, precision):
    hs, *_, routing, ids = inputs
    w1, w2, s1, s2 = inputs[5:9]
    result = torch.zeros(hs.shape, device=hs.device, dtype=torch.float32)
    # Decode independently from production and accumulate both GEMMs in FP32.
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        r1 = _decode_expert(w1[expert], s1[expert], precision, hs.dtype).float()
        r2 = _decode_expert(w2[expert], s2[expert], precision, hs.dtype).float()
        gate, up = (hs[tokens].float() @ r1.T).chunk(2, -1)
        out = (torch.nn.functional.silu(gate) * up) @ r2.T
        # torch.where returns strided indices; torch_musa 2.7.1 index_add_
        # requires contiguous indices for correct accumulation.
        result.index_add_(0, tokens.contiguous(), out * routing[tokens, slots, None])
    return result


def verify_inputs(op_name, config, inputs, precision, candidate, baseline):
    expected = _reference(inputs, precision)
    denominator = expected.abs().mean().clamp_min(1e-12)
    errors = {}
    for name, operation in (("flaggems", candidate), ("vllm", baseline)):
        actual = operation(*inputs)
        torch_device_fn.synchronize()
        assert actual.shape == expected.shape and actual.dtype == inputs[0].dtype
        assert torch.isfinite(actual).all(), f"{name}: nonfinite output for {config}"
        error = ((actual.float() - expected).abs().mean() / denominator).item()
        errors[name] = error
        assert error < 0.04, f"{name}: mean relative error={error:.6f} for {config}"
    print(
        "MTHREADS_MARLIN_ACCURACY",
        json.dumps(dict(op_name=op_name, shape=config, mean_relative_error=errors)),
        flush=True,
    )


def mthreads_input_iter(bench, dtype, precision, candidate, baseline):
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    if not getattr(module, "ENABLE_TRITON_MOE", True):
        raise RuntimeError(
            "The installed MUSA vLLM requires VLLM_MUSA_ENABLE_MOE_TRITON=1"
        )
    print(
        "MTHREADS_MARLIN_BASELINE",
        json.dumps(
            dict(
                op_name=bench.op_name,
                baseline=(
                    "vLLM Triton INT4 W4A16" if precision == "int4" else "vLLM BF16"
                ),
                source=module.__file__,
                VLLM_MUSA_ENABLE_MOE_TRITON=getattr(module, "ENABLE_TRITON_MOE", None),
                shapes=len(bench.shapes),
                weight_preparation="offline; candidate dequantization remains timed",
            )
        ),
        flush=True,
    )
    # Group by geometry even if the benchmark framework reorders self.shapes.
    by_geometry = {}
    for config in bench.shapes:
        by_geometry.setdefault(tuple(config[1:]), []).append(config)
    for geometry, configs in by_geometry.items():
        bank = inputs = None
        torch_device_fn.empty_cache()
        bank = make_weight_bank(geometry, dtype, precision)
        for config in configs:
            inputs = make_inputs(config, dtype, precision, bank)
            verify_inputs(bench.op_name, config, inputs, precision, candidate, baseline)
            yield inputs
