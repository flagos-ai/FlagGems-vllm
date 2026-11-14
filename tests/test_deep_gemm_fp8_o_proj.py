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

"""Functional tests for ``deep_gemm_fp8_o_proj``.

The operator composes ``fused_inv_rope_fp8_quant``, ``fp8_einsum`` and the
``wo_b`` projection, so every case is checked twice: against a dequantized
PyTorch reference, and (when the vLLM build exposes the DeepGEMM path) against
vLLM's own ``deep_gemm_fp8_o_proj``.
"""

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
from flaggems_vllm import deep_gemm_fp8_o_proj

from . import accuracy_utils as utils

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
BLOCK = 128

# (num_tokens, num_heads, n_groups, o_lora_rank, hidden_size)
CONFIGS = utils.DEEP_GEMM_FP8_O_PROJ_SHAPES["CONFIGS"]
SEEDS = utils.DEEP_GEMM_FP8_O_PROJ_SHAPES["SEEDS"]


def is_hopper_available():
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


def is_supported_backend():
    """Hopper runs the FP8 einsum path; MetaX runs the INT8 block-scaled path."""
    if flaggems_vllm.vendor_name == "metax":
        return torch.cuda.is_available()
    return is_hopper_available()


CUDA_AVAILABLE = is_supported_backend()
pytestmark = [
    pytest.mark.deep_gemm_fp8_o_proj,
    pytest.mark.skipif(
        not CUDA_AVAILABLE, reason="requires NVIDIA Hopper or MetaX (fp8 einsum)"
    ),
]

# The MetaX path transports the FP8 grid values as INT8 (extra error <= half an
# INT8 step, ~1.2% peak-relative measured); Hopper stays on exact FP8.
COMPOSED_ATOL = 0.02 if flaggems_vllm.vendor_name == "metax" else 0.01


class _QuantizedActivation:
    """Attribute-only stand-in for vLLM >= 0.32 ``QuantizedActivation``."""

    def __init__(self, data, scale):
        self.data = data
        self.scale = scale


class _QuantWoA(torch.nn.Module):
    """FP8 block-wise ``wo_a`` holder, mirroring vLLM's module layout."""

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight
        self.weight_scale_inv = scale


class _LinearBf16(torch.nn.Module):
    """Test-only stand-in for vLLM's ``wo_b`` row-parallel linear layer."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return F.linear(x, self.weight)


def _make_cos_sin_cache(max_pos, rope_dim, device):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _inv_rope_reference(o, positions, cos_sin_cache):
    """Undo RoPE on the trailing ``rope_dim`` elements of every head, in FP32.

    Mirrors the kernel math for a rotated pair ``(a, b)``: ``a' = a*cos + b*sin``
    and ``b' = b*cos - a*sin``.
    """
    x = o.float().clone()
    half = ROPE_DIM // 2
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin[:, :half], cos_sin[:, half:]
    rope = x[..., NOPE_DIM:]
    even, odd = rope[..., 0::2], rope[..., 1::2]
    rope_out = torch.stack(
        (
            even * cos.unsqueeze(1) + odd * sin.unsqueeze(1),
            odd * cos.unsqueeze(1) - even * sin.unsqueeze(1),
        ),
        dim=-1,
    ).flatten(-2)
    x[..., NOPE_DIM:] = rope_out
    return x


def _quantize_block_weight(weight):
    """Quantize ``[g, d, r]`` bf16 weights to FP8 with 128x128 block scales."""
    g, d, r = weight.shape
    view = weight.float().reshape(g, d // BLOCK, BLOCK, r // BLOCK, BLOCK)
    scale = view.abs().amax(dim=(2, 4)).clamp(1e-4) / 448.0
    quant = (view / scale.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return quant.reshape(g, d, r), scale


def _dequant_block_weight(weight, scale):
    """Dequantize FP8 block-wise weights back to FP32."""
    g, d, r = weight.shape
    view = weight.float().reshape(g, d // BLOCK, BLOCK, r // BLOCK, BLOCK)
    return (view * scale.unsqueeze(-1).unsqueeze(-3)).reshape(g, d, r)


def _build_inputs(config, seed, device):
    num_tokens, num_heads, n_groups, o_lora_rank, hidden = config
    torch.manual_seed(seed)
    heads_per_group = num_heads // n_groups
    reduction = heads_per_group * HEAD_DIM
    max_pos = max(4096, num_tokens * 2)

    o = torch.randn(
        (num_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, device)

    wa_bf16 = (
        torch.randn(
            (n_groups, o_lora_rank, reduction), dtype=torch.bfloat16, device=device
        )
        / reduction**0.5
    )
    wa_fp8, wa_scale = _quantize_block_weight(wa_bf16)

    wb = (
        torch.randn(
            (hidden, n_groups * o_lora_rank), dtype=torch.bfloat16, device=device
        )
        / (n_groups * o_lora_rank) ** 0.5
    )

    return {
        "o": o,
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "wo_a": _QuantWoA(wa_fp8, wa_scale),
        "wo_b": _LinearBf16(wb),
        "n_groups": n_groups,
        "heads_per_group": heads_per_group,
        "o_lora_rank": o_lora_rank,
        "wa_dequant": _dequant_block_weight(wa_fp8, wa_scale),
    }


def _reference(inp):
    """Dequantized FP32 reference: inverse RoPE -> FP8 einsum -> wo_b."""
    o = _inv_rope_reference(inp["o"], inp["positions"], inp["cos_sin_cache"])
    num_tokens = o.shape[0]
    n_groups = inp["n_groups"]
    grouped = o.reshape(num_tokens, n_groups, -1)
    z = torch.einsum("tgr,gdr->tgd", grouped, inp["wa_dequant"])
    return F.linear(z.flatten(1), inp["wo_b"].weight.float())


def _call_op(inp, recipe=(1, 128, 128), tma_aligned_scales=False):
    return deep_gemm_fp8_o_proj(
        inp["o"],
        inp["positions"],
        inp["cos_sin_cache"],
        inp["wo_a"],
        inp["wo_b"],
        n_groups=inp["n_groups"],
        heads_per_group=inp["heads_per_group"],
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=inp["o_lora_rank"],
        einsum_recipe=recipe,
        tma_aligned_scales=tma_aligned_scales,
    )


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", SEEDS)
def test_accuracy_deep_gemm_fp8_o_proj(config, seed):
    """Validate the composed operator against a dequantized PyTorch reference."""
    device = flaggems_vllm.device
    inp = _build_inputs(config, seed, device)

    out = _call_op(inp)
    ref = utils.to_reference(_reference(inp), upcast=True)
    torch.cuda.synchronize()

    assert out.shape == (config[0], config[4])
    # FP8 block-wise quantization plus bf16 output rounding.
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    utils.gems_assert_close(out, ref, torch.bfloat16, atol=atol)


def _quantize_activation(inp):
    """Inverse RoPE + per-128 FP8 quant, as vLLM 0.32's attention kernel emits."""
    o = _inv_rope_reference(inp["o"], inp["positions"], inp["cos_sin_cache"])
    num_tokens, n_groups = o.shape[0], inp["n_groups"]
    grouped = o.reshape(num_tokens, n_groups, -1)
    view = grouped.reshape(num_tokens, n_groups, -1, BLOCK)
    amax = view.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-10)
    scale = amax / float(torch.finfo(torch.float8_e4m3fn).max)
    data = (view / scale).to(torch.float8_e4m3fn)
    dequant = (data.float() * scale).reshape(num_tokens, n_groups, -1)
    return (
        data.reshape(num_tokens, n_groups, -1),
        scale.squeeze(-1),
        dequant,
    )


# Decode / small-batch / prefill points and both group counts, so the MetaX
# dequantized-BF16 path is checked against every GEMM tile bucket.
TORCH_COMPOSED_CONFIGS = (
    (1, 8, 1, 1024, 7168),
    (16, 64, 8, 1024, 7168),
    (128, 64, 8, 1024, 7168),
    (4096, 64, 8, 1024, 7168),
    (16, 128, 16, 1024, 7168),
)


def _reference_torch_from_dequant(inp, dequant):
    num_tokens = dequant.shape[0]
    z = torch.einsum("tgr,gdr->tgd", dequant, inp["wa_dequant"])
    return F.linear(z.reshape(num_tokens, -1), inp["wo_b"].weight.float())


def _reference_torch(inp):
    """Assemble the operator's exact math from torch ops.

    Inverse RoPE, the per-128 FP8 quant/dequant of the activation, the
    dequantized einsum over ``wo_a`` and the ``wo_b`` linear. MetaX has no vLLM
    DeepGEMM to cross-check against, so this composition is the reference.
    """
    _, _, dequant = _quantize_activation(inp)
    return _reference_torch_from_dequant(inp, dequant)


@pytest.mark.parametrize("config", TORCH_COMPOSED_CONFIGS)
def test_accuracy_deep_gemm_fp8_o_proj_torch_composed(config):
    """Match a torch-composed reference across decode, small batch and prefill."""
    device = flaggems_vllm.device
    inp = _build_inputs(config, SEEDS[0], device)

    out = _call_op(inp)
    ref = utils.to_reference(_reference_torch(inp), upcast=True)
    torch.cuda.synchronize()

    assert out.shape == (config[0], config[4])
    # Both sides quantize the activation identically, so only the GEMM rounding
    # separates them (measured <= 0.6% bf16 / <= 1.2% INT8 of the peak).
    atol = max(1e-2, ref.abs().max().item() * COMPOSED_ATOL)
    utils.gems_assert_close(out, ref, torch.bfloat16, atol=atol)


@pytest.mark.parametrize("config", TORCH_COMPOSED_CONFIGS)
def test_accuracy_deep_gemm_fp8_o_proj_prequantized(config):
    """A pre-quantized activation skips RoPE/quant and keeps the same math."""
    device = flaggems_vllm.device
    inp = _build_inputs(config, SEEDS[0], device)
    data, scale, dequant = _quantize_activation(inp)

    out = deep_gemm_fp8_o_proj(
        _QuantizedActivation(data, scale),
        inp["positions"],
        inp["cos_sin_cache"],
        inp["wo_a"],
        inp["wo_b"],
        n_groups=inp["n_groups"],
        heads_per_group=inp["heads_per_group"],
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=inp["o_lora_rank"],
    )

    ref = utils.to_reference(_reference_torch_from_dequant(inp, dequant), upcast=True)
    torch.cuda.synchronize()

    assert out.shape == (config[0], config[4])
    atol = max(1e-2, ref.abs().max().item() * COMPOSED_ATOL)
    utils.gems_assert_close(out, ref, torch.bfloat16, atol=atol)


def _vllm_native_op():
    try:
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import compute_fp8_einsum_recipe
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
            deep_gemm_fp8_o_proj as native_op,
        )
    except ImportError:
        return None, None
    return native_op, compute_fp8_einsum_recipe


@pytest.mark.parametrize("config", CONFIGS)
def test_deep_gemm_fp8_o_proj_matches_vllm_native(config):
    """Cross-check against vLLM's DeepGEMM implementation when it is runnable."""
    native_op, recipe_fn = _vllm_native_op()
    if native_op is None:
        pytest.skip("vLLM build has no deepseek_v4 deep_gemm_fp8_o_proj")

    device = flaggems_vllm.device
    inp = _build_inputs(config, SEEDS[0], device)
    recipe, tma_aligned = recipe_fn()
    try:
        ref = native_op(
            inp["o"],
            inp["positions"],
            inp["cos_sin_cache"],
            inp["wo_a"],
            inp["wo_b"],
            n_groups=inp["n_groups"],
            heads_per_group=inp["heads_per_group"],
            nope_dim=NOPE_DIM,
            rope_dim=ROPE_DIM,
            o_lora_rank=inp["o_lora_rank"],
            einsum_recipe=recipe,
            tma_aligned_scales=tma_aligned,
        )
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"vLLM native path unavailable: {exc}")

    out = _call_op(inp, recipe=recipe, tma_aligned_scales=tma_aligned)
    torch.cuda.synchronize()

    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    utils.gems_assert_close(
        out, utils.to_reference(ref, upcast=True), torch.bfloat16, atol=atol
    )


def test_deep_gemm_fp8_o_proj_invalid_recipe():
    """The generic path only implements the Hopper FP32 block-scale recipe."""
    device = flaggems_vllm.device
    inp = _build_inputs(CONFIGS[0], SEEDS[0], device)
    with pytest.raises(NotImplementedError):
        _call_op(inp, recipe=(1, 1, 128), tma_aligned_scales=True)
