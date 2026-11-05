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

"""Functional tests for flaggems_vllm.dsv41_o_proj.

Two references:
  * ``ref_dsv41_o_proj``: pure torch, written step by step so the math is easy
    to read. Mirrors the kernel's quantization exactly -> tight comparison.
  * vLLM's ``deep_gemm_fp8_o_proj`` (if vLLM + DeepGEMM are installed). Looser,
    because the two sides round activation scales differently.
"""

import math
import types

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg
from .accuracy_utils import gems_assert_close, to_reference

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
QUANT_BLOCK = 128
NOPE_DIM, ROPE_DIM = 448, 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
MAX_POSITION = 8192


def is_hopper():
    if flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


HOPPER = is_hopper()
pytestmark = [
    pytest.mark.dsv41_o_proj,
    pytest.mark.skipif(not HOPPER, reason="requires NVIDIA Hopper GPU"),
]

try:
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    HAS_VLLM = is_deep_gemm_supported()
except ImportError:
    HAS_VLLM = False


# --------------------------------------------------------------------------
# Torch reference (tests only; never imported by production code)
# --------------------------------------------------------------------------
def ref_inverse_rope(o, positions, cos_sin_cache, nope_dim, rope_dim):
    """Inverse GPT-J (interleaved) RoPE on the last ``rope_dim`` dims of each head.

    For each pair (x0, x1) = (x[2i], x[2i+1]) with angle theta_i(pos):
        forward RoPE : y0 = x0*c - x1*s,  y1 = x1*c + x0*s
        inverse RoPE : x0 = y0*c + y1*s,  x1 = y1*c - y0*s   (rotate by -theta)
    cos_sin_cache[pos] = [cos_0..cos_{d/2-1} | sin_0..sin_{d/2-1}].
    """
    x = o.float()
    nope, rope = x[..., :nope_dim], x[..., nope_dim:]
    half = rope_dim // 2
    cs = cos_sin_cache[positions]  # [T, rope_dim]
    cos = cs[:, :half].unsqueeze(1)  # [T, 1, half]
    sin = cs[:, half:].unsqueeze(1)
    y0, y1 = rope[..., 0::2], rope[..., 1::2]
    x0 = y0 * cos + y1 * sin
    x1 = y1 * cos - y0 * sin
    rope_out = torch.stack((x0, x1), dim=-1).flatten(-2)
    return torch.cat((nope, rope_out), dim=-1)


def ref_quant_per_block(x, block=QUANT_BLOCK, eps=1e-10):
    """Per-``block`` absmax FP8 quant along the last dim, same as the Triton kernel."""
    shape = x.shape
    xb = x.view(*shape[:-1], shape[-1] // block, block)
    scale = xb.abs().amax(dim=-1).clamp_min(eps) / FP8_MAX  # [..., K/128]
    q = (xb / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(shape), scale


def ref_dequant(q, scale, block=QUANT_BLOCK):
    shape = q.shape
    qb = q.float().view(*shape[:-1], shape[-1] // block, block)
    return (qb * scale.unsqueeze(-1)).view(shape)


def ref_dequant_weight(w3, ws3, block=QUANT_BLOCK):
    """w3: [G, N, K] fp8, ws3: [G, N/128, K/128] fp32 -> [G, N, K] fp32."""
    s = ws3.repeat_interleave(block, dim=1).repeat_interleave(block, dim=2)
    return w3.float() * s


def ref_dsv41_o_proj(
    attn_out,
    positions,
    cos_sin_cache,
    w3,
    ws3,
    n_groups,
    heads_per_group,
    o_lora_rank,
    quantize_activation=True,
):
    """Step-by-step reference. Returns z [T, G * o_lora_rank] in fp32."""
    T = attn_out.shape[0]
    n_heads = n_groups * heads_per_group
    o = attn_out[:, :n_heads]  # drop padded heads
    x = ref_inverse_rope(o, positions, cos_sin_cache, NOPE_DIM, ROPE_DIM)
    x = x.reshape(T, n_groups, heads_per_group * HEAD_DIM)  # [T, G, K]
    if quantize_activation:
        q, s = ref_quant_per_block(x)
        x = ref_dequant(q, s)
    w = ref_dequant_weight(w3, ws3)  # [G, N, K]
    z = torch.einsum("bhr,hdr->bhd", x, w)  # [T, G, N]
    return z.reshape(T, n_groups * o_lora_rank)


# --------------------------------------------------------------------------
# Input construction
# --------------------------------------------------------------------------
def make_cos_sin_cache(max_position, rope_dim, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device)


def per_block_cast_to_fp8(w, block=QUANT_BLOCK):
    """[N, K] -> fp8 [N, K], fp32 scale [N/128, K/128] (128x128 blocks)."""
    n, k = w.shape
    wb = w.float().view(n // block, block, k // block, block)
    scale = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4) / FP8_MAX
    q = (wb / scale).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.view(n // block, k // block)


def make_inputs(
    num_tokens, n_groups, heads_per_group, o_lora_rank, pad_heads=0, seed=0
):
    device = flaggems_vllm.device
    torch.manual_seed(seed)
    n_heads = n_groups * heads_per_group
    k = heads_per_group * HEAD_DIM
    attn_out = torch.randn(
        num_tokens, n_heads + pad_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(
        0, MAX_POSITION, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin_cache = make_cos_sin_cache(MAX_POSITION, ROPE_DIM, device=device)
    w = torch.randn(n_groups * o_lora_rank, k, device=device) / math.sqrt(k)
    w_fp8, w_scale = per_block_cast_to_fp8(w)  # 2D, as loaded by vLLM
    return attn_out, positions, cos_sin_cache, w_fp8, w_scale


def call_gems(attn_out, positions, cos_sin_cache, w, ws, n_groups, hpg, rank, **kw):
    return flaggems_vllm.dsv41_o_proj(
        attn_out,
        positions,
        cos_sin_cache,
        w,
        ws,
        n_groups=n_groups,
        heads_per_group=hpg,
        o_lora_rank=rank,
        **kw,
    )


# gems_assert_close(reduce_dim=K) allows an absolute error of 1e-4 * K (about 0.4
# here) on values of order 1, so also bound the error of every (token, group) row.
# BF16 output rounding alone is about 2e-3; FP8 cast differences can reach 1.2e-2.
MAX_REL_ERR = 1.5e-2
# vLLM rounds activation scales up to a power of two on Hopper and this kernel does
# not: two independent FP8 roundings of the same values differ by about 3.7%.
MAX_VLLM_REL_ERR = 5e-2


def assert_relative_close(res, ref, n_groups, rank, max_rel=MAX_REL_ERR):
    res = res.float().to(ref.device).view(-1, n_groups, rank)
    ref = ref.float().view(-1, n_groups, rank)
    rel = (res - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert rel.max().item() < max_rel, f"worst relative error {rel.max().item():.4e}"


# (n_groups, heads_per_group, o_lora_rank)
#   flash TP1: 64 heads  -> 8 groups;  pro TP1: 128 heads -> 16 groups
#   TP splits keep 8 heads per group and shrink n_groups.
GROUP_CONFIGS = [(8, 8, 1024), (16, 8, 1024), (2, 8, 1024)]
TOKENS = [1, 7, 32, 96, 257] if cfg.QUICK_MODE else [1, 2, 7, 32, 64, 96, 257, 2048]
if cfg.QUICK_MODE:
    GROUP_CONFIGS = GROUP_CONFIGS[:1]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("n_groups, hpg, rank", GROUP_CONFIGS)
def test_dsv41_o_proj_accuracy(num_tokens, n_groups, hpg, rank):
    attn_out, pos, csc, w, ws = make_inputs(num_tokens, n_groups, hpg, rank)
    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)

    ref = ref_dsv41_o_proj(
        to_reference(attn_out),
        to_reference(pos),
        to_reference(csc),
        to_reference(w.view(n_groups, rank, -1)),
        to_reference(ws.view(n_groups, rank // QUANT_BLOCK, -1)),
        n_groups,
        hpg,
        rank,
    )
    gems_assert_close(
        res, ref.to(torch.bfloat16), torch.bfloat16, reduce_dim=hpg * HEAD_DIM
    )
    assert_relative_close(res, ref, n_groups, rank)


@pytest.mark.parametrize("num_tokens", [1, 64])
def test_dsv41_o_proj_quant_error_sanity(num_tokens):
    """Against an un-quantized activation: only FP8 activation error remains."""
    n_groups, hpg, rank = GROUP_CONFIGS[0]
    attn_out, pos, csc, w, ws = make_inputs(num_tokens, n_groups, hpg, rank)
    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank).float()
    ref = ref_dsv41_o_proj(
        attn_out,
        pos,
        csc,
        w.view(n_groups, rank, -1),
        ws.view(n_groups, rank // QUANT_BLOCK, -1),
        n_groups,
        hpg,
        rank,
        quantize_activation=False,
    )
    rel = (res - ref).norm() / ref.norm()
    assert rel < 5e-2, f"relative L2 error too large: {rel:.4f}"


def test_dsv41_o_proj_padded_heads():
    """vLLM pads query heads; only the first G*hpg heads must be read."""
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(32, n_groups, hpg, rank, pad_heads=16)
    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    attn_out[:, n_groups * hpg :] = float("nan")  # garbage in padding
    res2 = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    torch.testing.assert_close(res, res2, atol=0, rtol=0)


def test_dsv41_o_proj_3d_weight_and_out():
    """DeepGEMM post-processing stores wo_a as 3D; also check ``out=``."""
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(16, n_groups, hpg, rank)
    res2d = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    out = torch.empty_like(res2d)
    res3d = call_gems(
        attn_out,
        pos,
        csc,
        w.view(n_groups, rank, -1),
        ws.view(n_groups, rank // QUANT_BLOCK, -1),
        n_groups,
        hpg,
        rank,
        out=out,
    )
    assert res3d.data_ptr() == out.data_ptr()
    torch.testing.assert_close(res2d, res3d, atol=0, rtol=0)


def test_dsv41_o_proj_empty():
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(0, n_groups, hpg, rank)
    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    assert res.shape == (0, n_groups * rank)


def test_dsv41_o_proj_unsupported():
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(4, n_groups, hpg, rank)
    with pytest.raises(NotImplementedError):  # BF16 wo_a
        call_gems(attn_out, pos, csc, w.to(torch.bfloat16), ws, n_groups, hpg, rank)
    with pytest.raises(NotImplementedError):  # packed UE8M0 scales (SM100 recipe)
        call_gems(attn_out, pos, csc, w, ws.view(torch.int32), n_groups, hpg, rank)
    with pytest.raises(NotImplementedError):  # non-bf16 activation
        call_gems(attn_out.half(), pos, csc, w, ws, n_groups, hpg, rank)


def test_dsv41_o_proj_rejects_non_hopper(monkeypatch):
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(4, n_groups, hpg, rank)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    with pytest.raises(NotImplementedError, match="Hopper"):
        call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)


def test_dsv41_o_proj_rejects_weight_on_other_device():
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(4, n_groups, hpg, rank)
    with pytest.raises(ValueError, match="same device"):
        call_gems(attn_out, pos, csc, w, ws.cpu(), n_groups, hpg, rank)


def test_dsv41_o_proj_rejects_autograd():
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(4, n_groups, hpg, rank)
    attn_out.requires_grad_()
    with pytest.raises(NotImplementedError, match="autograd"):
        call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)


def test_dsv41_o_proj_is_library_only():
    """No ATen schema: importable, but never registered into PyTorch dispatch."""
    assert callable(flaggems_vllm.dsv41_o_proj)
    assert "dsv41_o_proj" not in [name for name, _ in flaggems_vllm._FULL_CONFIG]


def test_dsv41_o_proj_wo_b_proj():
    """wo_b gets the flattened z once, and its result is what the op returns."""
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(16, n_groups, hpg, rank)
    z = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    seen = []

    def wo_b(flat):
        seen.append(flat)
        return flat[:, :128] * 2

    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank, wo_b_proj=wo_b)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], z, atol=0, rtol=0)
    torch.testing.assert_close(res, z[:, :128] * 2, atol=0, rtol=0)


def test_dsv41_o_proj_wo_b_proj_with_out():
    """``out`` holds z; the callback's result is still what the op returns."""
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(16, n_groups, hpg, rank)
    z = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    out = torch.empty_like(z)
    res = call_gems(
        attn_out,
        pos,
        csc,
        w,
        ws,
        n_groups,
        hpg,
        rank,
        out=out,
        wo_b_proj=lambda flat: flat[:, :128],
    )
    torch.testing.assert_close(out, z, atol=0, rtol=0)
    assert res.shape == (16, 128)


def test_dsv41_o_proj_wo_b_proj_empty():
    """Zero tokens still go through wo_b, as they do in vLLM."""
    n_groups, hpg, rank = 2, 8, 1024
    attn_out, pos, csc, w, ws = make_inputs(0, n_groups, hpg, rank)
    seen = []

    def wo_b(flat):
        seen.append(flat.shape)
        return flat[:, :128]

    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank, wo_b_proj=wo_b)
    assert seen == [(0, n_groups * rank)]
    assert res.shape == (0, 128)


@pytest.mark.skipif(not HAS_VLLM, reason="requires vLLM with DeepGEMM")
@pytest.mark.parametrize("num_tokens", [1, 7, 32, 63, 65, 257, 1000])
@pytest.mark.parametrize("n_groups, hpg, rank", [*GROUP_CONFIGS, (1, 8, 1024)])
def test_dsv41_o_proj_vs_vllm(num_tokens, n_groups, hpg, rank):
    attn_out, pos, csc, w, ws = make_inputs(num_tokens, n_groups, hpg, rank)
    w3 = w.view(n_groups, rank, -1)
    ws3 = ws.view(n_groups, rank // QUANT_BLOCK, -1)
    wo_a = types.SimpleNamespace(weight=w3, weight_scale_inv=ws3)

    ref = deep_gemm_fp8_o_proj(
        attn_out[:, : n_groups * hpg],
        pos,
        csc,
        wo_a,
        lambda flat: flat,  # identity wo_b: compare z only
        n_groups=n_groups,
        heads_per_group=hpg,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=rank,
        einsum_recipe=(1, 128, 128),  # Hopper recipe
        tma_aligned_scales=False,
    )
    res = call_gems(attn_out, pos, csc, w, ws, n_groups, hpg, rank)
    assert_relative_close(
        res, to_reference(ref), n_groups, rank, max_rel=MAX_VLLM_REL_ERR
    )
