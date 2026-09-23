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

import pytest
import torch

import flaggems_vllm

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128
INT8_ABS_MAX = 127.0

pytestmark = pytest.mark.skipif(
    flaggems_vllm.device != "npu", reason="ascend int8 inverse-RoPE quant"
)


def _make_cache(max_pos, rope_dim, device):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    freqs = torch.outer(
        torch.arange(max_pos, device=device, dtype=torch.float32), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _rotate_gptj(values):
    even = values[..., ::2]
    odd = values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _quantize(blocks):
    absmax = blocks.abs().amax(dim=-1)
    scale = absmax / INT8_ABS_MAX
    scaled = torch.where(absmax.unsqueeze(-1) == 0, 0.0, blocks / scale.unsqueeze(-1))
    codes = torch.floor(scaled + 0.5).clamp(-128, 127).to(torch.int8)
    return codes, scale


def _reference(values, positions, cache, n_groups, heads_per_group):
    half = ROPE_DIM // 2
    cos_sin = cache.index_select(0, positions)
    cos = cos_sin[:, :half].repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = -cos_sin[:, half:].repeat_interleave(2, dim=-1).unsqueeze(1)
    passed = values[..., :NOPE_DIM].float()
    rotated = values[..., NOPE_DIM:].float()
    rotated = rotated * cos + _rotate_gptj(rotated) * sin
    merged = torch.cat((passed, rotated), dim=-1)
    num_tokens, _, head_dim = merged.shape
    width = heads_per_group * head_dim
    blocks = merged.view(num_tokens, n_groups, width)
    blocks = blocks.reshape(
        num_tokens, n_groups, width // QUANT_GROUP_SIZE, QUANT_GROUP_SIZE
    )
    codes, scale = _quantize(blocks)
    return (
        codes.reshape(num_tokens, n_groups, width),
        scale,
        merged.view(num_tokens, n_groups, width),
    )


def _run(num_tokens, num_heads, n_groups, seed=0, scale=1.0):
    heads_per_group = num_heads // n_groups
    torch.manual_seed(seed)
    device = flaggems_vllm.device
    values = scale * torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.randint(0, 128, (num_tokens,), device=device)
    cache = _make_cache(256, ROPE_DIM, device)
    quantized, quant_scale = flaggems_vllm.fused_inv_rope_int8_quant(
        values, positions, cache, n_groups, heads_per_group
    )
    ref_q, ref_scale, merged = _reference(
        values.cpu().float(), positions.cpu(), cache.cpu(), n_groups, heads_per_group
    )
    return quantized.cpu(), quant_scale.cpu(), ref_q, ref_scale, merged


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128])
@pytest.mark.parametrize("num_heads,n_groups", [(32, 4), (64, 8), (128, 8)])
@pytest.mark.parametrize("seed", [0, 42])
def test_fused_inv_rope_int8_quant(num_tokens, num_heads, n_groups, seed):
    quantized, quant_scale, _ref_q, _ref_scale, merged = _run(
        num_tokens, num_heads, n_groups, seed=seed
    )
    dequant = quantized.float() * quant_scale.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )
    abs_err = (dequant - merged).abs()
    limit = (0.5 * quant_scale).repeat_interleave(QUANT_GROUP_SIZE, dim=-1) + 1e-4
    assert torch.le(abs_err, limit).all()


def test_output_layout():
    num_tokens, num_heads, n_groups = 7, 64, 8
    heads_per_group = num_heads // n_groups
    device = flaggems_vllm.device
    values = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(num_tokens, device=device)
    cache = _make_cache(32, ROPE_DIM, device)
    quantized, quant_scale = flaggems_vllm.fused_inv_rope_int8_quant(
        values, positions, cache, n_groups, heads_per_group
    )
    width = heads_per_group * HEAD_DIM
    assert quantized.dtype == torch.int8
    assert quant_scale.dtype == torch.float32
    assert quantized.shape == (num_tokens, n_groups, width)
    assert quant_scale.shape[-1] == width // QUANT_GROUP_SIZE
    # Check strides before a host copy; .cpu() packs the view into contiguous memory.
    assert quantized.stride() == (width, num_tokens * width, 1)
    assert quantized[:, 0, :].is_contiguous()


def test_identity_rope_matches_passthrough():
    num_tokens, num_heads, n_groups = 8, 64, 8
    heads_per_group = num_heads // n_groups
    device = flaggems_vllm.device
    torch.manual_seed(0)
    values = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(num_tokens, device=device)
    cache = torch.zeros(32, ROPE_DIM, device=device)
    cache[:, : ROPE_DIM // 2] = 1
    quantized, quant_scale = flaggems_vllm.fused_inv_rope_int8_quant(
        values, positions, cache, n_groups, heads_per_group
    )
    ref_q, ref_scale, _ = _reference(
        values.cpu().float(), positions.cpu(), cache.cpu(), n_groups, heads_per_group
    )
    got = quantized.cpu()
    # Division rounding can move a boundary value by one int8 code.
    assert (got.int() - ref_q.int()).abs().max() <= 1
    torch.testing.assert_close(quant_scale.cpu(), ref_scale, atol=1e-5, rtol=1e-5)


def test_large_values_stay_in_int8_range():
    quantized, quant_scale, _, _, merged = _run(8, 64, 8, scale=1000.0)
    assert quantized.min() >= -128 and quantized.max() <= 127
    dequant = quantized.float() * quant_scale.repeat_interleave(
        QUANT_GROUP_SIZE, dim=-1
    )
    abs_err = (dequant - merged).abs()
    limit = (0.5 * quant_scale).repeat_interleave(QUANT_GROUP_SIZE, dim=-1) + 1e-2
    assert torch.le(abs_err, limit).all()
