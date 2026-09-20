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

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops import per_token_group_quant_fp8
from flaggems_vllm.utils.device_info import get_device_capability

from . import accuracy_utils as utils

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128
EPS = 1e-10

HAS_NATIVE_FP8 = hasattr(torch, "float8_e4m3fn") and (
    flaggems_vllm.SUPPORTED_FP8_DTYPE == torch.float8_e4m3fn
    or (flaggems_vllm.vendor_name == "mthreads" and get_device_capability() >= (3, 1))
)


def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
        max_val - min_val
    )
    return torch.clamp(linear_func, 0, 1)


def _make_cos_sin_cache(max_pos, rope_dim, device):
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _rotate_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _pack_ue8m0_scales(scales):
    scale_bits = scales.contiguous().view(torch.int32)
    ue8m0_bytes = (scale_bits >> 23) & 0xFF
    packed = torch.zeros(
        ue8m0_bytes.shape[:-1], dtype=torch.int32, device=scales.device
    )
    for idx in range(ue8m0_bytes.shape[-1]):
        packed |= ue8m0_bytes[..., idx] << (idx * 8)
    return packed


def _unpack_ue8m0_scales(scale_packed, chunks_per_head):
    shifts = (
        torch.arange(chunks_per_head, device=scale_packed.device, dtype=torch.int32) * 8
    )
    ue8m0_bytes = (scale_packed.unsqueeze(-1) >> shifts) & 0xFF
    scale_bits = (ue8m0_bytes << 23).contiguous().view(torch.float32)
    return scale_bits.reshape(
        *scale_packed.shape[:-1], scale_packed.shape[-1] * chunks_per_head
    )


def _dequantize(o_fp8, scale, heads_per_group, quant_group_size):
    chunks_per_head = o_fp8.shape[-1] // (heads_per_group * quant_group_size)
    if scale.dtype == torch.int32:
        scale = _unpack_ue8m0_scales(scale, chunks_per_head)
    scales_expanded = scale.unsqueeze(-1).expand(*scale.shape, quant_group_size)
    return o_fp8.float() * scales_expanded.reshape_as(o_fp8)


def _assert_dequant_close(out, scale, ref_out, ref_scale, heads_per_group, msg=""):
    out_dq = (
        _dequantize(out, scale, heads_per_group, QUANT_GROUP_SIZE).flatten().float()
    )
    ref_dq = (
        _dequantize(ref_out, ref_scale, heads_per_group, QUANT_GROUP_SIZE)
        .flatten()
        .float()
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        out_dq.unsqueeze(0), ref_dq.unsqueeze(0)
    ).item()
    diff = 1.0 - cos_sim
    assert diff < 1e-4, f"Dequant diff too large: {diff:.8f} (expected < 1e-4). {msg}"


def _head_group_ids(values):
    return [f"H{num_heads}_G{n_groups}" for num_heads, n_groups in values]


def _reference_inv_rope(
    o,
    positions,
    cos_sin_cache,
    nope_dim=NOPE_DIM,
    rope_dim=ROPE_DIM,
):
    half_rope = rope_dim // 2
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos = cos_sin[:, :half_rope].repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = -cos_sin[:, half_rope:].repeat_interleave(2, dim=-1).unsqueeze(1)

    o_pass = o[..., :nope_dim]
    o_rot_f32 = o[..., nope_dim:].float()
    o_rot_f32 = o_rot_f32 * cos + _rotate_gptj(o_rot_f32) * sin
    return torch.cat((o_pass, o_rot_f32.to(o.dtype)), dim=-1)


def native_fused_inv_rope_fp8_quant(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim=NOPE_DIM,
    rope_dim=ROPE_DIM,
    quant_group_size=QUANT_GROUP_SIZE,
    tma_aligned_scales=False,
):
    del rope_dim
    half_rope = ROPE_DIM // 2
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos = cos_sin[:, :half_rope].repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = -cos_sin[:, half_rope:].repeat_interleave(2, dim=-1).unsqueeze(1)

    o_pass = o[..., :nope_dim]
    o_rot_f32 = o[..., nope_dim:].float()
    o_rot_f32 = o_rot_f32 * cos + _rotate_gptj(o_rot_f32) * sin
    o_rot = torch.cat((o_pass, o_rot_f32.to(o.dtype)), dim=-1)

    num_tokens, _num_heads, head_dim = o_rot.shape
    d = heads_per_group * head_dim
    chunks_per_head = head_dim // quant_group_size
    num_scale_blocks = d // quant_group_size

    o_grouped = o_rot.view(num_tokens, n_groups, d).float()
    o_blocks = o_grouped.reshape(
        num_tokens, n_groups, num_scale_blocks, quant_group_size
    )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_absmax = o_blocks.abs().amax(dim=-1).clamp(min=EPS)
    scales = block_absmax * (1.0 / fp8_max)
    if tma_aligned_scales:
        scales = torch.exp2(torch.ceil(torch.log2(scales.clamp(min=EPS))))
    o_fp8 = (
        (o_blocks / scales.unsqueeze(-1))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    o_fp8 = o_fp8.reshape(num_tokens, n_groups, d)

    if not tma_aligned_scales:
        return o_fp8, scales

    scales = scales.reshape(num_tokens, n_groups, heads_per_group, chunks_per_head)
    return o_fp8, _pack_ue8m0_scales(scales)


def _unfused_inv_rope_fp8_quant(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim=NOPE_DIM,
    rope_dim=ROPE_DIM,
    quant_group_size=QUANT_GROUP_SIZE,
    tma_aligned_scales=False,
):
    cos = cos_sin_cache[:, : rope_dim // 2]
    sin = -cos_sin_cache[:, rope_dim // 2 :]

    o_nope = o[..., :nope_dim]
    o_rope = o[..., nope_dim:]
    o_rope_rot, _ = flaggems_vllm.apply_rotary_pos_emb(
        o_rope,
        o_rope,
        cos,
        sin,
        position_ids=positions,
        rotary_interleaved=True,
    )
    o_rot = torch.cat((o_nope, o_rope_rot), dim=-1)

    num_tokens = o.shape[0]
    d = heads_per_group * o.shape[-1]
    o_grouped = o_rot.view(num_tokens, n_groups, d)
    o_flat = o_grouped.permute(1, 0, 2).contiguous().reshape(-1, d)
    o_fp8, o_scale = per_token_group_quant_fp8(
        o_flat,
        group_size=quant_group_size,
        dtype=torch.float8_e4m3fn,
        scale_ue8m0=tma_aligned_scales,
    )
    o_fp8 = o_fp8.view(n_groups, num_tokens, d).transpose(0, 1)
    if tma_aligned_scales:
        chunks_per_head = o.shape[-1] // quant_group_size
        o_scale = o_scale.view(n_groups, num_tokens, heads_per_group, chunks_per_head)
        o_scale = _pack_ue8m0_scales(o_scale.transpose(0, 1))
    else:
        o_scale = o_scale.view(n_groups, num_tokens, -1).transpose(0, 1)
    return o_fp8, o_scale


def _make_real_deepseek_v4_cache(
    max_pos, rope_dim, device, scaling_factor=16, base=10000.0
):
    beta_fast, beta_slow = 32, 1
    pos_freqs = base ** (
        torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim
    )
    inv_freq_extra = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (scaling_factor * pos_freqs)
    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, rope_dim, base, max_pos
    )
    mask = 1.0 - yarn_linear_ramp_mask(low, high, rope_dim // 2).to(
        device=device, dtype=torch.float32
    )
    inv_freq = inv_freq_interp * (1 - mask) + inv_freq_extra * mask
    t = torch.arange(max_pos * scaling_factor, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _run_case(
    num_tokens,
    num_heads,
    n_groups,
    tma_aligned_scales,
    seed=0,
    scale=1.0,
    positions=None,
    cos_sin_cache=None,
):
    heads_per_group = num_heads // n_groups

    torch.manual_seed(seed)
    device = flaggems_vllm.device
    max_pos = max(4096, num_tokens * 2)

    o = scale * torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    if positions is None:
        positions = torch.randint(
            0, max_pos, (num_tokens,), dtype=torch.long, device=device
        )
    if cos_sin_cache is None:
        cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, torch.device(device))

    ref_out, ref_scale = native_fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        tma_aligned_scales=tma_aligned_scales,
    )

    with flaggems_vllm.use_gems():
        out, scale_out = flaggems_vllm.fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups,
            heads_per_group,
            dtype=torch.float8_e4m3fn,
            tma_aligned_scales=tma_aligned_scales,
        )

    return {
        "o": o,
        "positions": positions,
        "cos_sin_cache": cos_sin_cache,
        "out": out,
        "scale": scale_out,
        "ref_out": ref_out,
        "ref_scale": ref_scale,
        "heads_per_group": heads_per_group,
    }


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("seed", utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
@pytest.mark.parametrize(
    "num_heads,n_groups",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["NUM_HEADS_AND_GROUPS"],
    ids=_head_group_ids(utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["NUM_HEADS_AND_GROUPS"]),
)
@pytest.mark.parametrize(
    "num_tokens", utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["NUM_TOKENS"]
)
def test_fused_inv_rope_fp8_quant(
    num_tokens, num_heads, n_groups, tma_aligned_scales, seed
):
    result = _run_case(num_tokens, num_heads, n_groups, tma_aligned_scales, seed=seed)
    out = result["out"]
    scale = result["scale"]
    ref_out = result["ref_out"]
    ref_scale = result["ref_scale"]
    heads_per_group = result["heads_per_group"]

    out_scale_fp32 = scale
    ref_scale_fp32 = ref_scale
    if tma_aligned_scales:
        chunks_per_head = HEAD_DIM // QUANT_GROUP_SIZE
        out_scale_fp32 = _unpack_ue8m0_scales(scale, chunks_per_head)
        ref_scale_fp32 = _unpack_ue8m0_scales(ref_scale, chunks_per_head)

    scale_ratio = out_scale_fp32 / ref_scale_fp32.clamp(min=1e-30)
    assert scale_ratio.max() <= 2.0 and scale_ratio.min() >= 0.5, (
        f"Scale ratio out of [0.5, 2]: min={scale_ratio.min():.4f} "
        f"max={scale_ratio.max():.4f}"
    )
    _assert_dequant_close(out, scale, ref_out, ref_scale, heads_per_group)


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "num_tokens",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["OUTPUT_LAYOUT_NUM_TOKENS"],
)
@pytest.mark.parametrize(
    "num_heads,n_groups",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["OUTPUT_LAYOUT_NUM_HEADS_AND_GROUPS"],
    ids=_head_group_ids(
        utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["OUTPUT_LAYOUT_NUM_HEADS_AND_GROUPS"]
    ),
)
def test_output_layout(num_tokens, num_heads, n_groups):
    heads_per_group = num_heads // n_groups
    d = heads_per_group * HEAD_DIM

    fp32_case = _run_case(num_tokens, num_heads, n_groups, False)
    assert fp32_case["out"].stride() == (d, num_tokens * d, 1)
    assert fp32_case["scale"].shape[-1] == d // QUANT_GROUP_SIZE
    assert fp32_case["scale"].permute(1, 0, 2).stride(1) == 1 or num_tokens == 1

    packed_case = _run_case(num_tokens, num_heads, n_groups, True)
    packed_k = (d // QUANT_GROUP_SIZE + 3) // 4
    assert packed_case["out"].stride() == (d, num_tokens * d, 1)
    assert packed_case["scale"].dtype == torch.int32
    assert packed_case["scale"].shape[-1] == packed_k
    assert packed_case["scale"].permute(1, 0, 2).stride(1) == 1 or num_tokens == 1


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "num_tokens",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["PER_GROUP_CONTIGUITY_NUM_TOKENS"],
)
def test_per_group_contiguity(num_tokens):
    result = _run_case(num_tokens, 64, 8, False, seed=0)

    for g in range(8):
        fp8_slice = result["out"][:, g, :]
        assert fp8_slice.is_contiguous(), (
            f"o_fp8[:, {g}, :] is not contiguous: "
            f"shape={list(fp8_slice.shape)}, stride={list(fp8_slice.stride())}"
        )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
def test_scales_are_power_of_two(tma_aligned_scales):
    result = _run_case(32, 64, 8, tma_aligned_scales, seed=0)
    scales = result["scale"]
    if tma_aligned_scales:
        scales = _unpack_ue8m0_scales(scales, HEAD_DIM // QUANT_GROUP_SIZE)

    log2_scales = torch.log2(scales)
    residual = (log2_scales - log2_scales.round()).abs()
    if tma_aligned_scales:
        assert (
            residual.max() < 1e-5
        ), f"Not all scales are powers of 2: max log2 residual = {residual.max().item()}"
    else:
        assert residual.max() > 1e-5, "Unexpected power-of-two scales in non-UE8M0 mode"


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
def test_large_values(tma_aligned_scales):
    result = _run_case(8, 64, 8, tma_aligned_scales, seed=0, scale=1000.0)
    _assert_dequant_close(
        result["out"],
        result["scale"],
        result["ref_out"],
        result["ref_scale"],
        result["heads_per_group"],
        msg="large-value saturation case",
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
def test_nope_dims_unchanged(tma_aligned_scales):
    num_tokens, num_heads, n_groups = 16, 64, 8
    heads_per_group = num_heads // n_groups

    result = _run_case(num_tokens, num_heads, n_groups, tma_aligned_scales, seed=0)

    zero_cache = torch.zeros_like(result["cos_sin_cache"])
    half = ROPE_DIM // 2
    zero_cache[:, :half] = 1.0
    norope = _run_case(
        num_tokens,
        num_heads,
        n_groups,
        tma_aligned_scales,
        seed=0,
        positions=result["positions"],
        cos_sin_cache=zero_cache,
    )

    chunks_per_head = HEAD_DIM // QUANT_GROUP_SIZE
    fused_scale = result["scale"]
    norope_scale = norope["scale"]
    if tma_aligned_scales:
        fused_scale = _unpack_ue8m0_scales(fused_scale, chunks_per_head)
        norope_scale = _unpack_ue8m0_scales(norope_scale, chunks_per_head)

    for h in range(heads_per_group):
        for c in range(chunks_per_head - 1):
            qb = h * chunks_per_head + c
            start = qb * QUANT_GROUP_SIZE
            end = start + QUANT_GROUP_SIZE

            fused_nope = result["out"][:, :, start:end].view(torch.uint8)
            norope_nope = norope["out"][:, :, start:end].view(torch.uint8)
            assert torch.equal(
                fused_nope, norope_nope
            ), f"Nope block (head={h}, chunk={c}) differs between fused and no-rope reference"
            assert torch.equal(
                fused_scale[:, :, qb], norope_scale[:, :, qb]
            ), f"Nope scale (head={h}, chunk={c}) differs"


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
def test_zero_positions(tma_aligned_scales):
    positions = torch.zeros(16, device=flaggems_vllm.device, dtype=torch.long)
    result = _run_case(16, 64, 8, tma_aligned_scales, seed=0, positions=positions)
    _assert_dequant_close(
        result["out"],
        result["scale"],
        result["ref_out"],
        result["ref_scale"],
        result["heads_per_group"],
        msg="all-zero positions",
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
def test_dequant_numerical_accuracy(tma_aligned_scales):
    result = _run_case(32, 64, 8, tma_aligned_scales, seed=0)
    o = result["o"]
    positions = result["positions"]
    cos_sin_cache = result["cos_sin_cache"]
    heads_per_group = result["heads_per_group"]

    o_after_rope = _reference_inv_rope(
        o,
        positions,
        cos_sin_cache,
    ).view(32, 8, heads_per_group * HEAD_DIM)
    dequant = _dequantize(
        result["out"], result["scale"], heads_per_group, QUANT_GROUP_SIZE
    )

    abs_err = (dequant.float() - o_after_rope.float()).abs()
    rel_err = abs_err / o_after_rope.float().abs().clamp(min=1e-6)
    mean_rel_err = rel_err.mean().item()
    assert (
        mean_rel_err < 0.15
    ), f"Mean relative error too high: {mean_rel_err:.4f} (expected < 0.15)"


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("seed", utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
@pytest.mark.parametrize(
    "num_heads,n_groups",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["NUM_HEADS_AND_GROUPS"],
    ids=_head_group_ids(utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["NUM_HEADS_AND_GROUPS"]),
)
@pytest.mark.parametrize(
    "num_tokens", utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["NUM_TOKENS"]
)
def test_unfused_path(num_tokens, num_heads, n_groups, tma_aligned_scales, seed):
    result = _run_case(num_tokens, num_heads, n_groups, tma_aligned_scales, seed=seed)
    unfused_out, unfused_scale = _unfused_inv_rope_fp8_quant(
        result["o"].clone(),
        result["positions"],
        result["cos_sin_cache"],
        n_groups,
        result["heads_per_group"],
        tma_aligned_scales=tma_aligned_scales,
    )

    _assert_dequant_close(
        result["out"],
        result["scale"],
        unfused_out,
        unfused_scale,
        result["heads_per_group"],
        msg="fused vs unfused Triton path",
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "num_tokens",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["REAL_ROPE_NUM_TOKENS"],
)
@pytest.mark.parametrize(
    "tma_aligned_scales",
    utils.FUSED_INV_ROPE_FP8_QUANT_SHAPES["TMA_ALIGNED_SCALES"],
)
def test_with_real_deepseek_v4_rope(num_tokens, tma_aligned_scales):
    num_heads, n_groups = 64, 8
    positions = torch.randint(
        0, 4096, (num_tokens,), device=flaggems_vllm.device, dtype=torch.long
    )
    cos_sin_cache = _make_real_deepseek_v4_cache(
        65536, ROPE_DIM, torch.device(flaggems_vllm.device)
    )
    result = _run_case(
        num_tokens,
        num_heads,
        n_groups,
        tma_aligned_scales,
        seed=0,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
    )
    _assert_dequant_close(
        result["out"],
        result["scale"],
        result["ref_out"],
        result["ref_scale"],
        result["heads_per_group"],
        msg="Real DeepSeek V4 rope",
    )


def _edge_inputs(num_tokens=7, num_heads=8):
    device = flaggems_vllm.device
    o = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cache = _make_cos_sin_cache(max(16, num_tokens), ROPE_DIM, torch.device(device))
    return o, positions, cache


def _call_edge(o, positions, cache, tma_aligned_scales, **kwargs):
    kwargs.setdefault("dtype", torch.float8_e4m3fn)
    return flaggems_vllm.fused_inv_rope_fp8_quant(
        o,
        positions,
        cache,
        1,
        o.shape[1],
        tma_aligned_scales=tma_aligned_scales,
        **kwargs,
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize(
    "num_tokens", [512] if utils.QUICK_MODE else [512, 1024, 2048, 4096]
)
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_prefill_shapes(num_tokens, tma_aligned_scales):
    result = _run_case(num_tokens, 64, 8, tma_aligned_scales)
    _assert_dequant_close(
        result["out"],
        result["scale"],
        result["ref_out"],
        result["ref_scale"],
        result["heads_per_group"],
        msg="medium and long prefill",
    )
    o_after_rope = _reference_inv_rope(
        result["o"], result["positions"], result["cos_sin_cache"]
    ).view(num_tokens, 8, result["heads_per_group"] * HEAD_DIM)
    dequant = _dequantize(
        result["out"], result["scale"], result["heads_per_group"], QUANT_GROUP_SIZE
    )
    abs_err = (dequant.float() - o_after_rope.float()).abs()
    rel_err = abs_err / o_after_rope.float().abs().clamp(min=1e-6)
    mean_rel_err = rel_err.mean().item()
    assert (
        mean_rel_err < 0.15
    ), f"Mean relative error too high: {mean_rel_err:.4f} (expected < 0.15)"


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("value", [0.0, 2.0**-40])
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_zero_and_tiny_values(value, tma_aligned_scales):
    o, positions, cache = _edge_inputs()
    o.fill_(value)
    positions.zero_()
    out, scale = _call_edge(o, positions, cache, tma_aligned_scales)
    ref_out, ref_scale = native_fused_inv_rope_fp8_quant(
        o, positions, cache, 1, o.shape[1], tma_aligned_scales=tma_aligned_scales
    )
    assert out.dtype == torch.float8_e4m3fn
    assert torch.equal(out.view(torch.uint8), ref_out.view(torch.uint8))
    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=0)


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_custom_epsilon(tma_aligned_scales):
    o, positions, cache = _edge_inputs()
    o.zero_()
    eps = 1e-6
    out, scale = _call_edge(o, positions, cache, tma_aligned_scales, eps=eps)
    if tma_aligned_scales:
        scale = _unpack_ue8m0_scales(scale, HEAD_DIM // QUANT_GROUP_SIZE)
    expected_scale = eps / torch.finfo(torch.float8_e4m3fn).max
    if tma_aligned_scales:
        expected_scale = 2.0 ** math.ceil(math.log2(max(expected_scale, EPS)))
    assert torch.count_nonzero(out.float()).item() == 0
    torch.testing.assert_close(
        scale, torch.full_like(scale, expected_scale), rtol=1e-6, atol=0
    )


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_empty_tokens(tma_aligned_scales):
    o, positions, cache = _edge_inputs(num_tokens=0)
    out, scale = _call_edge(o, positions, cache, tma_aligned_scales)
    assert out.shape == (0, 1, 8 * HEAD_DIM)
    assert out.dtype == torch.float8_e4m3fn
    assert out.device == o.device
    assert scale.shape == (0, 1, 8 if tma_aligned_scales else 32)
    assert scale.dtype == (torch.int32 if tma_aligned_scales else torch.float32)
    assert scale.device == o.device


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("num_tokens", [1, 7])
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_scale_padding_is_zero(num_tokens, tma_aligned_scales):
    result = _run_case(num_tokens, 32, 4, tma_aligned_scales)
    scale = result["scale"]
    aligned_tokens = (num_tokens + 3) // 4 * 4
    assert scale.stride() == (1, scale.shape[2] * aligned_tokens, aligned_tokens)
    storage_view = scale.as_strided(
        (aligned_tokens, scale.shape[1], scale.shape[2]), scale.stride()
    )
    assert torch.count_nonzero(storage_view[num_tokens:]).item() == 0


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("layout", ["token", "head", "cache"])
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_strided_outer_dimensions(layout, tma_aligned_scales):
    o, positions, cache = _edge_inputs()
    if layout == "token":
        storage = torch.empty(
            o.shape[0] * 2, o.shape[1], HEAD_DIM, dtype=o.dtype, device=o.device
        )
        storage[::2].copy_(o)
        o = storage[::2]
    elif layout == "head":
        storage = torch.empty(
            o.shape[0], o.shape[1] * 2, HEAD_DIM, dtype=o.dtype, device=o.device
        )
        storage[:, ::2].copy_(o)
        o = storage[:, ::2]
    else:
        storage = torch.empty(
            cache.shape[0] * 2, ROPE_DIM, dtype=cache.dtype, device=cache.device
        )
        storage[::2].copy_(cache)
        cache = storage[::2]
    out, scale = _call_edge(o, positions, cache, tma_aligned_scales)
    ref_out, ref_scale = native_fused_inv_rope_fp8_quant(
        o, positions, cache, 1, o.shape[1], tma_aligned_scales=tma_aligned_scales
    )
    _assert_dequant_close(out, scale, ref_out, ref_scale, o.shape[1])


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_repeated_calls_observe_mutation(tma_aligned_scales):
    o, positions, cache = _edge_inputs()
    addresses = [x.data_ptr() for x in (o, positions, cache)]
    previous = None
    for iteration in range(2):
        if iteration:
            o.add_(0.75)
            positions.add_(1)
            cache.mul_(0.5)
        snapshots = [x.clone() for x in (o, positions, cache)]
        out, scale = _call_edge(o, positions, cache, tma_aligned_scales)
        for actual, saved, address in zip((o, positions, cache), snapshots, addresses):
            assert actual.data_ptr() == address
            assert torch.equal(actual, saved), "operator mutated an input"
        ref_out, ref_scale = native_fused_inv_rope_fp8_quant(
            o, positions, cache, 1, o.shape[1], tma_aligned_scales=tma_aligned_scales
        )
        _assert_dequant_close(out, scale, ref_out, ref_scale, o.shape[1])
        assert out.data_ptr() != o.data_ptr()
        if previous is not None:
            assert not torch.equal(out.view(torch.uint8), previous.view(torch.uint8))
        previous = out


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "mthreads", reason="MUSA dispatch contract"
)
def test_musa_public_dispatch():
    from flaggems_vllm.runtime.backend._mthreads.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    assert flaggems_vllm.fused_inv_rope_fp8_quant is fused_inv_rope_fp8_quant
    assert flaggems_vllm.ops_inv_rope_fp8_quant is fused_inv_rope_fp8_quant
    assert any(
        name == "fused_inv_rope_fp8_quant" and function is fused_inv_rope_fp8_quant
        for name, function in flaggems_vllm._FULL_CONFIG
    )
    o, positions, cache = _edge_inputs()
    out, _ = flaggems_vllm.fused_inv_rope_fp8_quant(o, positions, cache, 1, o.shape[1])
    assert out.dtype == torch.float8_e4m3fn


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "mthreads", reason="MUSA input contract"
)
@pytest.mark.parametrize(
    "invalid",
    [
        "positions_stride",
        "cache_stride",
        "input_stride",
        "input_dtype",
        "positions_dtype",
        "cache_dtype",
        "head_dim",
        "output_dtype",
        "quant_group_size",
    ],
)
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_musa_rejects_unsupported_inputs(invalid, tma_aligned_scales):
    o, positions, cache = _edge_inputs()
    kwargs = {}
    if invalid == "positions_stride":
        positions = torch.zeros(2 * o.shape[0], dtype=positions.dtype, device=o.device)[
            ::2
        ]
    elif invalid == "cache_stride":
        cache = torch.zeros(cache.shape[0], 2 * ROPE_DIM, device=o.device)[:, ::2]
    elif invalid == "input_stride":
        o = torch.zeros(*o.shape[:-1], 2 * HEAD_DIM, dtype=o.dtype, device=o.device)[
            ..., ::2
        ]
    elif invalid == "input_dtype":
        o = o.float()
    elif invalid == "positions_dtype":
        positions = positions.float()
    elif invalid == "cache_dtype":
        cache = cache.bfloat16()
    elif invalid == "head_dim":
        o = o[..., :256]
        kwargs["nope_dim"] = 192
    elif invalid == "output_dtype":
        kwargs["dtype"] = torch.float32
    else:
        kwargs["quant_group_size"] = 64
    with pytest.raises((AssertionError, ValueError, NotImplementedError)):
        _call_edge(o, positions, cache, tma_aligned_scales, **kwargs)


@pytest.mark.fused_inv_rope_fp8_quant
@pytest.mark.skipif(not HAS_NATIVE_FP8, reason="requires native float8_e4m3fn support")
@pytest.mark.parametrize("tma_aligned_scales", [False, True])
def test_prefill_head_tail_and_strides(tma_aligned_scales):
    torch.manual_seed(42)
    device = flaggems_vllm.device
    o = torch.randn(258, 20, HEAD_DIM, dtype=torch.bfloat16, device=device)[::2, ::2]
    positions = torch.arange(129, dtype=torch.int64, device=device)
    cache = _make_cos_sin_cache(512, ROPE_DIM, torch.device(device))[::2]
    out, scale = flaggems_vllm.fused_inv_rope_fp8_quant(
        o,
        positions,
        cache,
        2,
        5,
        dtype=torch.float8_e4m3fn,
        tma_aligned_scales=tma_aligned_scales,
    )
    ref_out, ref_scale = native_fused_inv_rope_fp8_quant(
        o, positions, cache, 2, 5, tma_aligned_scales=tma_aligned_scales
    )
    _assert_dequant_close(out, scale, ref_out, ref_scale, 5)
    dequant = _dequantize(out, scale, 5, QUANT_GROUP_SIZE)
    reference = _reference_inv_rope(o, positions, cache).reshape(129, 2, 5 * HEAD_DIM)
    relative_error = (
        dequant - reference.float()
    ).abs() / reference.float().abs().clamp(min=1e-6)
    assert relative_error.mean().item() < 0.15
    padded_scale = scale.as_strided((132, 2, scale.shape[-1]), scale.stride())
    assert torch.count_nonzero(padded_scale[129:]).item() == 0
