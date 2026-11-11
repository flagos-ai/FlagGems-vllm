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

import os
import random
from math import ceil

import pytest
import torch
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts_impl as vllm_fused_experts_impl,
)

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

from .conftest import QUICK_MODE

random.seed(42)

# Vendor-dispatched entry points: flaggems_vllm/__init__.py calls
# runtime.replace_customized_ops() at import time, so on backends with a
# specialized implementation (ascend/hygon/metax/mthreads/thead) these names
# already point at the vendor version; on NVIDIA they are the generic ones.
fused_experts_impl = flaggems_vllm.ops_experts_impl

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name

if vendor_name in ("thead", "metax", "mthreads"):
    # These vendors' production MoE is not the stock fused_experts_impl.
    from benchmark.test_fused_moe import production_moe_op

    _vllm_moe_reference = production_moe_op(vendor_name)
    _VLLM_REF_DTYPES = (
        [torch.bfloat16] if vendor_name == "thead" else [torch.bfloat16, torch.float16]
    )
else:
    _vllm_moe_reference = vllm_fused_experts_impl
    _VLLM_REF_DTYPES = [torch.bfloat16, torch.float16]

# Per-vendor native precision, per vendor documentation. Kept in sync with
# benchmark/test_fused_moe_int8.py::_OFFICIAL_PRECISION; vendors that are not
# listed are treated as unsupported and their quantized cases are skipped.
_OFFICIAL_PRECISION = {
    "nvidia": (True, True),
    "mthreads": (True, True),
    "hygon": (False, True),
    "metax": (False, True),
    "thead": (False, True),
    "ascend": (False, True),
}
_SUPPORTS_FP8, _SUPPORTS_INT8 = _OFFICIAL_PRECISION.get(vendor_name, (False, False))


def _is_hopper_class():
    if device != "cuda" or not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    sm = major * 10 + minor
    return 90 <= sm < 100


# FP8 additionally requires a real Hopper-class device on NVIDIA: the vendor
# table records what the vendor can do, not what this particular card has.
FP8_AVAILABLE = _SUPPORTS_FP8 and (vendor_name != "nvidia" or _is_hopper_class())

# Tolerance per quantization mode: (rtol, atol floor, ratio of |ref|max).
TOLERANCE = {
    "plain": (1e-1, 1e-2, 1e-5),
    "fp8": (5e-1, 2e-1, 1e-1),
    "fp8_blockwise": (2e-1, 5e-2, 5e-2),
    "int8": (2e-1, 5e-2, 2e-2),
    "int8_w8a16": (2e-1, 5e-2, 2e-2),
    "int4_w4a16": (3e-1, 1e-1, 5e-2),
}


def assert_moe_close(result, ref, mode="plain", msg=""):
    """Compare a fused MoE result with its reference under the mode's tolerance."""
    rtol, atol_floor, ratio = TOLERANCE[mode]
    atol = max(atol_floor, float(ref.abs().max()) * ratio)
    torch.testing.assert_close(result, ref, rtol=rtol, atol=atol, msg=msg)


# Large production-like MoE shapes can allocate tens of GiB for weights alone.
# Keep the default per-op run bounded; opt in explicitly for stress coverage.
RUN_LARGE_MOE_TESTS = not QUICK_MODE and os.environ.get(
    "FLAGGEMS_VLLM_RUN_LARGE_MOE_TESTS", "0"
).lower() in {"1", "true", "yes", "on"}

FUSED_MOE_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (8, 4, 64, 128, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
    (64, 8, 256, 512, 2),
    (128, 16, 128, 256, 4),
    (4, 16, 512, 1024, 2),
    # Qwen3.5 shapes (TP=4)
    (10, 256, 2048, 128, 8),
    (256, 256, 2048, 128, 8),
]

if RUN_LARGE_MOE_TESTS:
    FUSED_MOE_CONFIGS += [
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (4, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
        (128, 8, 4096, 14336, 2),
        (256, 8, 4096, 14336, 2),
        (512, 8, 4096, 14336, 2),
        # DeepSeek-V3-like shapes (TP=8 shard)
        (1, 256, 7168, 2048, 8),
        (4, 256, 7168, 2048, 8),
        (16, 256, 7168, 2048, 8),
        (64, 256, 7168, 2048, 8),
        (128, 256, 7168, 2048, 8),
        (256, 256, 7168, 2048, 8),
    ]


FUSED_MOE_QUANT_CONFIGS = [
    # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
    (1, 8, 128, 256, 2),
    (4, 8, 128, 256, 2),
    (16, 8, 256, 512, 2),
    (32, 8, 128, 256, 4),
    (64, 8, 256, 512, 2),
    (128, 16, 128, 256, 4),
]

if RUN_LARGE_MOE_TESTS:
    FUSED_MOE_QUANT_CONFIGS += [
        # Mixtral-like shapes
        (1, 8, 4096, 14336, 2),
        (16, 8, 4096, 14336, 2),
        (64, 8, 4096, 14336, 2),
    ]

FUSED_MOE_FP8_BLOCKWISE_CONFIGS = list(FUSED_MOE_QUANT_CONFIGS)

if RUN_LARGE_MOE_TESTS:
    FUSED_MOE_FP8_BLOCKWISE_CONFIGS += [
        # Qwen3.5-397B-A17B
        (1, 512, 4096, 1024, 10),
        (4, 512, 4096, 1024, 10),
        (16, 512, 4096, 1024, 10),
        (64, 512, 4096, 1024, 10),
        (128, 512, 4096, 1024, 10),
        (256, 512, 4096, 1024, 10),
    ]


def make_topk_routing(gating, topk, dtype):
    """Build routing tensors using the production int32 expert-id contract."""
    topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(dtype), topk_ids.to(torch.int32)


def torch_fused_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation of fused MoE (no vLLM dependency).

    Computes:
        Y_m = sum_j  A_mj * W2[e_mj] @ SiLU(W1[e_mj] @ H_m)_{:D} ) * (W1[e_mj] @ H_m)_{D:})

    Args:
        hidden_states: (M, K)
        w1: (E, 2D, K)  -- gate + up projection concatenated
        w2: (E, K, D)   -- down projection
        topk_weights: (M, topk)
        topk_ids: (M, topk)

    Returns:
        output: (M, K)
    """
    M, K = hidden_states.shape
    num_experts = w1.shape[0]
    topk = topk_ids.shape[1]
    D = w1.shape[1] // 2

    # Expand each token once per routed slot, then run one dense matmul per
    # expert. This replaces the per-(token, slot) Python loop (M*topk device
    # syncs) with E loops, independent of the token count.
    x = hidden_states.to(torch.float32).repeat_interleave(topk, dim=0)
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1).to(torch.float32)

    z = torch.empty(M * topk, 2 * D, device=hidden_states.device, dtype=torch.float32)
    for e in range(num_experts):
        sel = flat_ids == e
        if sel.any():
            z[sel] = x[sel] @ w1[e].to(torch.float32).T

    gate, up = z[:, :D], z[:, D:]
    act = (gate * torch.sigmoid(gate)) * up  # SiLU(gate) * up

    r = torch.empty(M * topk, K, device=hidden_states.device, dtype=torch.float32)
    for e in range(num_experts):
        sel = flat_ids == e
        if sel.any():
            r[sel] = act[sel] @ w2[e].to(torch.float32).T

    output = (r * flat_weights[:, None]).view(M, topk, K).sum(dim=1)
    return output.to(hidden_states.dtype)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_vs_ref(config, dtype):
    """Test FlagGems fused_moe against a pure PyTorch reference."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # FlagGems result
    result = fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    # Pure PyTorch reference (no vLLM dependency)
    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    torch_device_fn.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    assert_moe_close(result, ref)


VLLM_SMALL_E_CRASH = vendor_name == "hygon"


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_CONFIGS)
@pytest.mark.parametrize("dtype", _VLLM_REF_DTYPES)
def test_fused_moe_vs_vllm(config, dtype):
    """Test FlagGems fused_moe against the vLLM reference implementation."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    if VLLM_SMALL_E_CRASH and num_experts < 256:
        pytest.skip(
            "vLLM fused_moe segfaults/errs for small-E shapes on Hygon HCU "
            "(upstream kernel, no HCU adaptation)"
        )
    device = flaggems_vllm.device

    torch.manual_seed(0)

    # Generate inputs with controlled magnitude to avoid numerical blow-up
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # FlagGems result
    result = fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)

    # Reference result
    ref = _vllm_moe_reference(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )

    torch_device_fn.synchronize()

    # Fused bf16/fp16 kernels accumulate rounding errors across two GEMMs
    # and an activation; use tolerances proportional to output magnitude.
    assert_moe_close(result, ref)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    not FP8_AVAILABLE,
    reason="FP8 quantization requires NVIDIA Hopper or an FP8-capable accelerator",
)
def test_accuracy_fused_moe_fp8(config):
    """Test FlagGems fused_moe with FP8 W8A8 quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create FP8 weights: quantize and store scale
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / intermediate_size**0.5)

    # Per-tensor quantization of weights
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10

    # Quantize w1 per-expert
    w1_scales = []
    w1_fp8_list = []
    for e in range(num_experts):
        amax = w1_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w1_q = (w1_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w1_fp8_list.append(w1_q)
        w1_scales.append(scale)
    w1_fp8 = torch.stack(w1_fp8_list)
    w1_scale = torch.tensor(w1_scales, device=device, dtype=torch.float32)

    # Quantize w2 per-expert
    w2_scales = []
    w2_fp8_list = []
    for e in range(num_experts):
        amax = w2_fp32[e].abs().amax().clamp(min=eps)
        scale = amax / fp8_max
        w2_q = (w2_fp32[e] / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
        w2_fp8_list.append(w2_q)
        w2_scales.append(scale)
    w2_fp8 = torch.stack(w2_fp8_list)
    w2_scale = torch.tensor(w2_scales, device=device, dtype=torch.float32)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # FlagGems FP8 result
    result = fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use the dequantized weights (fp8 → float) for reference
    w1_deq = torch.zeros_like(w1_fp32).to(dtype)
    for e in range(num_experts):
        w1_deq[e] = (w1_fp8[e].float() * w1_scales[e]).to(dtype)
    w2_deq = torch.zeros_like(w2_fp32).to(dtype)
    for e in range(num_experts):
        w2_deq[e] = (w2_fp8[e].float() * w2_scales[e]).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states, w1_deq, w2_deq, topk_weights, topk_ids, quant_mode="fp8"
    )

    torch_device_fn.synchronize()

    # FP8 quantization introduces more error than bf16: two quantized GEMMs
    # plus the activation accumulate rounding error.
    assert_moe_close(result, ref, "fp8")


def _fake_quantize_fp8(tensor: torch.Tensor):
    """Simulate FP8 E4M3 quantization round-trip for reference computation."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    eps = 1e-10
    # Per-token quantization (matches the fused Triton quant path in
    # flaggems_vllm.ops.fused_moe._fp8_quantize: one scale per row)
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=eps).float()
    scale = amax / fp8_max
    q = (tensor.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    return q.float() * scale  # dequantized


def _fake_quantize_int8(tensor: torch.Tensor):
    """Simulate INT8 quantization round-trip for reference computation."""
    eps = 1e-10
    # Per-token quantization
    amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=eps).float()
    scale = amax / 127.0
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)

    return q.float() * scale  # dequantized


def torch_fused_moe_quantized_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_mode: str = "fp8",
) -> torch.Tensor:
    """Reference fused MoE with simulated quantization noise.

    Simulates the quantization → dequantization round-trip on activations
    to model the same numerical behavior as the quantized kernel path.
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    if quant_mode == "fp8":
        fake_quant = _fake_quantize_fp8
    else:
        fake_quant = _fake_quantize_int8

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Quantize activation before GEMM1
            h_q = fake_quant(hidden_states[m].unsqueeze(0)).squeeze(0)
            # GEMM1
            z = h_q.float() @ w1[e].T.float()
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # Quantize intermediate before GEMM2
            s_q = fake_quant(s.unsqueeze(0)).squeeze(0)
            # GEMM2
            r = s_q.float() @ w2[e].T.float()
            output[m] += (weight.float() * r).to(output.dtype)

    return output


def torch_w8a8_block_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    a = a.to(compute_type)
    b = b.to(compute_type)
    assert a.shape[-1] == b.shape[-1]
    assert b.ndim == 2 and b.is_contiguous() and b_scales.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size
    assert (a.shape[-1] + block_k - 1) // block_k == a_scales.shape[-1]
    assert a.shape[:-1] == a_scales.shape[:-1]

    m = a.numel() // a.shape[-1]
    n, k = b.shape
    origin_c_shape = a.shape[:-1] + (n,)
    a = a.reshape(m, a.shape[-1])
    a_scales = a_scales.reshape(m, a_scales.shape[-1])
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == b_scales.shape[0]
    assert k_tiles == b_scales.shape[1]

    c = torch.zeros((m, n), dtype=compute_type, device=a.device)
    a_tiles = [a[:, i * block_k : min((i + 1) * block_k, k)] for i in range(k_tiles)]
    b_tiles = [
        [
            b[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    c_tiles = [c[:, j * block_n : min((j + 1) * block_n, n)] for j in range(n_tiles)]
    a_scale_tiles = [a_scales[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            scale = a_scale_tiles[i] * b_scales[j][i]
            c_tiles[j][:, :] += torch.matmul(a_tiles[i], b_tiles[j][i].t()) * scale

    return c.reshape(origin_c_shape).to(output_dtype)


def torch_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    assert x.shape[-1] % group_size == 0
    assert x.is_contiguous()

    finfo = torch.finfo(dtype)
    x_reshaped = x.reshape(x.numel() // group_size, group_size)
    amax = (
        x_reshaped.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    )
    x_scales = amax / finfo.max
    x_quant = (x_reshaped / x_scales).clamp(min=finfo.min, max=finfo.max).to(dtype)
    x_quant = x_quant.reshape(x.shape)
    x_scales = x_scales.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_quant, x_scales


def torch_w8a8_block_fp8_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_shape: list[int],
):
    batch_size, hidden_size = hidden_states.shape
    topk = topk_ids.size(1)
    expanded_hidden = hidden_states.view(batch_size, -1, hidden_size).repeat(1, topk, 1)
    expanded_hidden = expanded_hidden.reshape(-1, hidden_size)
    out = torch.zeros(
        batch_size * topk,
        w2.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    flat_weights = topk_weights.view(-1)
    flat_ids = topk_ids.view(-1)
    _, block_k = block_shape
    hidden_q, hidden_scale = torch_per_token_group_quant_fp8(expanded_hidden, block_k)
    hidden_q = hidden_q.to(torch.float32)

    def silu_and_mul(x):
        import torch.nn.functional as F

        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    for expert_idx in range(w1.shape[0]):
        mask = flat_ids == expert_idx
        if mask.sum():
            inter = torch_w8a8_block_matmul(
                hidden_q[mask],
                w1[expert_idx],
                hidden_scale[mask],
                w1_scale[expert_idx],
                block_shape,
                output_dtype=hidden_states.dtype,
            )
            act = silu_and_mul(inter)
            act_q, act_scale = torch_per_token_group_quant_fp8(act, block_k)
            out[mask] = torch_w8a8_block_matmul(
                act_q,
                w2[expert_idx],
                act_scale,
                w2_scale[expert_idx],
                block_shape,
                output_dtype=hidden_states.dtype,
            )

    return (
        out.view(batch_size, -1, w2.shape[1])
        * flat_weights.view(batch_size, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_FP8_BLOCKWISE_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.skipif(
    not FP8_AVAILABLE,
    reason="FP8 blockwise quantization requires NVIDIA Hopper or an FP8-capable accelerator",
)
def test_fused_moe_fp8_blockwise(config, block_shape):
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    if hidden_size % block_shape[1] != 0:
        # Invalid shape for block-wise quantization
        return
    if intermediate_size % block_shape[0] != 0:
        # Invalid shape for block-wise quantization
        return

    device = flaggems_vllm.device
    dtype = torch.bfloat16
    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1_fp8 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )
        * (1.0 / hidden_size**0.5)
    ).to(torch.float8_e4m3fn)
    w2_fp8 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.float32,
        )
        * (1.0 / intermediate_size**0.5)
    ).to(torch.float8_e4m3fn)

    w1_scale = torch.randn(
        num_experts,
        ceil(intermediate_size * 2 / block_shape[0]),
        ceil(hidden_size / block_shape[1]),
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.randn(
        num_experts,
        ceil(hidden_size / block_shape[0]),
        ceil(intermediate_size / block_shape[1]),
        device=device,
        dtype=torch.float32,
    )

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    result = fused_experts_impl(
        hidden_states,
        w1_fp8,
        w2_fp8,
        topk_weights,
        topk_ids,
        use_fp8_w8a8=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )

    ref = torch_w8a8_block_fp8_moe(
        hidden_states,
        w1_fp8,
        w2_fp8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        block_shape,
    )

    torch_device_fn.synchronize()

    assert_moe_close(result, ref, "fp8_blockwise")


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize(
    "config",
    FUSED_MOE_QUANT_CONFIGS
    + (
        [
            (64, 128, 6144, 384, 4),
            (1024, 128, 6144, 384, 4),
            (8192, 128, 6144, 384, 4),
        ]
        if RUN_LARGE_MOE_TESTS
        else []
    ),
)
@pytest.mark.skipif(
    not _SUPPORTS_INT8,
    reason="no native INT8 support on this vendor (see _OFFICIAL_PRECISION)",
)
def test_fused_moe_int8(config):
    """Test FlagGems fused_moe with INT8 W8A8 per-channel quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights: quantize per-channel (per output column of each expert)
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10

    # Per-channel quantization of weights: scale per [expert, output_dim]
    # w1 shape: [E, 2D, K] → scale shape: [E, 2D, 1]
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    # For the kernel: w1_scale shape [E, 2D] (per-channel: one scale per output dim)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # FlagGems INT8 result
    result = fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a8=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference: use dequantized weights
    w1_deq = (w1_int8.float() * w1_scale_full).to(dtype)
    w2_deq = (w2_int8.float() * w2_scale_full).to(dtype)

    ref = torch_fused_moe_quantized_reference(
        hidden_states,
        w1_deq,
        w2_deq,
        topk_weights,
        topk_ids,
        quant_mode="int8",
    )

    torch_device_fn.synchronize()

    assert_moe_close(result, ref, "int8")


def torch_fused_moe_weight_only_reference(
    hidden_states: torch.Tensor,
    w1_int: torch.Tensor,
    w2_int: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference fused MoE for weight-only quantization.

    Weights are dequantized (w_int * scale) then used in FP computation.
    Activations remain in original precision (no activation quantization).
    """
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=hidden_states.device, dtype=hidden_states.dtype)

    for m in range(M):
        for j in range(topk):
            e = topk_ids[m, j].item()
            weight = topk_weights[m, j]
            # Dequantize weights
            w1_deq = w1_int[e].float() * w1_scale[e].unsqueeze(-1).float()
            w2_deq = w2_int[e].float() * w2_scale[e].unsqueeze(-1).float()
            # GEMM1
            z = hidden_states[m].float() @ w1_deq.T
            # SiLU-and-Mul
            D = z.shape[-1] // 2
            gate, up = z[:D], z[D:]
            s = (gate * torch.sigmoid(gate)) * up
            # GEMM2
            r = s @ w2_deq.T
            output[m] += (weight.float() * r).to(output.dtype)

    return output


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    not _SUPPORTS_INT8,
    reason="no native INT8 support on this vendor (see _OFFICIAL_PRECISION)",
)
def test_fused_moe_int8_w8a16(config):
    """Test FlagGems fused_moe with INT8 W8A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT8 weights per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    # Per-channel quantization
    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / 127.0
    w1_int8 = (w1_fp32 / w1_scale_full).round().clamp(-128, 127).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)  # [E, 2D]

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / 127.0
    w2_int8 = (w2_fp32 / w2_scale_full).round().clamp(-128, 127).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)  # [E, K]

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # FlagGems INT8 W8A16 result
    result = fused_experts_impl(
        hidden_states,
        w1_int8,
        w2_int8,
        topk_weights,
        topk_ids,
        use_int8_w8a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int8,
        w2_int8,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch_device_fn.synchronize()

    # Weight-only quantization keeps activations in full precision, so only
    # the weight rounding error remains.
    assert_moe_close(result, ref, "int8_w8a16")


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize("config", FUSED_MOE_QUANT_CONFIGS)
@pytest.mark.skipif(
    not _SUPPORTS_INT8,
    reason="no native INT8 support on this vendor (see _OFFICIAL_PRECISION)",
)
def test_fused_moe_int4_w4a16(config):
    """Test FlagGems fused_moe with INT4 W4A16 (weight-only) quantization."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device
    dtype = torch.bfloat16

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)

    # Create INT4 weights stored in INT8 containers, per-channel
    w1_fp32 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / hidden_size**0.5)
    w2_fp32 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.float32,
    ) * (1.0 / intermediate_size**0.5)

    eps = 1e-10
    int4_max = 7
    int4_min = -8

    w1_amax = w1_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w1_scale_full = w1_amax / int4_max
    w1_int4 = (w1_fp32 / w1_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w1_scale = w1_scale_full.squeeze(-1)

    w2_amax = w2_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    w2_scale_full = w2_amax / int4_max
    w2_int4 = (w2_fp32 / w2_scale_full).round().clamp(int4_min, int4_max).to(torch.int8)
    w2_scale = w2_scale_full.squeeze(-1)

    # Generate routing
    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # FlagGems INT4 W4A16 result
    result = fused_experts_impl(
        hidden_states,
        w1_int4,
        w2_int4,
        topk_weights,
        topk_ids,
        use_int4_w4a16=True,
        per_channel_quant=True,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    # Reference
    ref = torch_fused_moe_weight_only_reference(
        hidden_states,
        w1_int4,
        w2_int4,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
    )

    torch_device_fn.synchronize()

    # INT4 has a coarser quantum, so it needs the widest tolerance.
    assert_moe_close(result, ref, "int4_w4a16")


@pytest.mark.fused_experts_impl
@pytest.mark.parametrize(
    "config",
    [
        (4, 8, 128, 256, 2),
        (16, 8, 256, 512, 2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_moe_apply_router_weight_on_input(config, dtype):
    """Test apply_router_weight_on_input vs default (weight on output)."""
    num_tokens, num_experts, hidden_size, intermediate_size, topk = config
    device = flaggems_vllm.device

    torch.manual_seed(0)

    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=dtype,
    ) * (1.0 / hidden_size**0.5)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * (1.0 / intermediate_size**0.5)

    gating = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    topk_weights, topk_ids = make_topk_routing(gating, topk, dtype)

    # Default (weight on GEMM2 output)
    result_default = fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
    )

    # Weight on GEMM1 input
    result_on_input = fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=True,
    )

    torch_device_fn.synchronize()

    # Due to SiLU nonlinearity, these will differ, but both should be
    # close to the reference with weight on the respective path.
    ref = torch_fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    # The default (weight on output) should match our standard reference.
    assert_moe_close(result_default, ref)

    # The apply_on_input result will differ but should be finite and nonzero
    assert torch.isfinite(
        result_on_input
    ).all(), "result_on_input has non-finite values"

    assert result_on_input.abs().sum() > 0, "result_on_input is all zeros"
