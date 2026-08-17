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
from flaggems_vllm.modules.rotary_embedding import (
    GemsDeepseekYarnRoPE,
    GemsRope,
    gems_rope_forward,
)

from . import accuracy_utils as utils
from . import conftest as cfg

DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
FUNCTION_CASES = [
    pytest.param(False, False, False, id="neox-sequential-outplace"),
    pytest.param(True, True, False, id="interleaved-position-ids-outplace"),
    pytest.param(False, True, True, id="neox-position-ids-inplace"),
    pytest.param(True, False, True, id="interleaved-sequential-inplace"),
]


def _reference_device():
    return "cpu" if cfg.TO_CPU else flaggems_vllm.device


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _torch_apply_rope(
    query, key, cos, sin, position_ids=None, rotary_interleaved=False
):
    query = query.float()
    key = key.float()
    if position_ids is None:
        cos = cos[None, : query.shape[-3], None, :]
        sin = sin[None, : query.shape[-3], None, :]
    else:
        cos = cos[position_ids].unsqueeze(-2)
        sin = sin[position_ids].unsqueeze(-2)

    if rotary_interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        rotate = _rotate_interleaved
    else:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        rotate = _rotate_half

    return query * cos + rotate(query) * sin, key * cos + rotate(key) * sin


def _standard_cache(max_seq_len, rotary_dim, base, dtype, device):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _yarn_mscale(scale, mscale):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_correction_dim(
    num_rotations, rotary_dim, base, original_max_position_embeddings
):
    return (
        rotary_dim
        * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_cache(
    max_seq_len,
    rotary_dim,
    base,
    dtype,
    device,
    scaling_factor,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    mscale,
    mscale_all_dim,
):
    freq_indices = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    freq_extra = 1.0 / (base ** (freq_indices / rotary_dim))
    freq_inter = 1.0 / (scaling_factor * base ** (freq_indices / rotary_dim))

    low = math.floor(
        _yarn_correction_dim(
            beta_fast, rotary_dim, base, original_max_position_embeddings
        )
    )
    high = math.ceil(
        _yarn_correction_dim(
            beta_slow, rotary_dim, base, original_max_position_embeddings
        )
    )
    low = max(low, 0)
    high = min(high, rotary_dim - 1)
    if low == high:
        high += 0.001
    ramp = (torch.arange(rotary_dim // 2, dtype=torch.float32, device=device) - low) / (
        high - low
    )
    inv_freq_mask = 1.0 - ramp.clamp(0, 1)
    inv_freq = freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask

    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    scale = _yarn_mscale(scaling_factor, mscale) / _yarn_mscale(
        scaling_factor, mscale_all_dim
    )
    return (freqs.cos() * scale).to(dtype), (freqs.sin() * scale).to(dtype)


def _make_inputs(dtype, max_seq_len=37):
    query = torch.randn((2, 7, 3, 32), dtype=dtype, device=flaggems_vllm.device)
    key = torch.randn((2, 7, 1, 32), dtype=dtype, device=flaggems_vllm.device)
    position_ids = torch.randint(0, max_seq_len, (2, 7), device=flaggems_vllm.device)
    return query, key, position_ids


@pytest.mark.gems_rotary_embedding
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rotary_interleaved,has_position_ids,inplace", FUNCTION_CASES)
def test_gems_rope_forward_modes(dtype, rotary_interleaved, has_position_ids, inplace):
    max_seq_len = 37
    query, key, position_ids = _make_inputs(dtype, max_seq_len)
    if has_position_ids and rotary_interleaved:
        position_ids = position_ids.to(torch.int32)
    query_before = query.clone()
    key_before = key.clone()
    cos, sin = _standard_cache(max_seq_len, 32, 10000.0, dtype, flaggems_vllm.device)
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_cos = utils.to_reference(cos, True)
    ref_sin = utils.to_reference(sin, True)
    ref_position_ids = utils.to_reference(position_ids)
    ref_out_query, ref_out_key = _torch_apply_rope(
        ref_query,
        ref_key,
        ref_cos,
        ref_sin,
        ref_position_ids if has_position_ids else None,
        rotary_interleaved,
    )

    out_query, out_key = gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids if has_position_ids else None,
        rotary_interleaved,
        inplace,
    )

    utils.gems_assert_close(out_query, ref_out_query, dtype, reduce_dim=32)
    utils.gems_assert_close(out_key, ref_out_key, dtype, reduce_dim=32)
    if inplace:
        assert out_query.data_ptr() == query.data_ptr()
        assert out_key.data_ptr() == key.data_ptr()
    else:
        assert out_query.data_ptr() != query.data_ptr()
        assert out_key.data_ptr() != key.data_ptr()
        utils.gems_assert_equal(query, query_before)
        utils.gems_assert_equal(key, key_before)


@pytest.mark.gems_rotary_embedding
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [False, True])
def test_gems_rope_module_cache_and_forward(dtype, rotary_interleaved):
    max_seq_len = 37
    base = 10000.0
    query, key, position_ids = _make_inputs(dtype, max_seq_len)
    module = GemsRope(
        rotary_dim=32,
        max_position_embeddings=max_seq_len,
        base=base,
        rotary_interleaved=rotary_interleaved,
        dtype=dtype,
        device=flaggems_vllm.device,
    )
    ref_cos, ref_sin = _standard_cache(
        max_seq_len, 32, base, dtype, _reference_device()
    )

    assert module.cos_cached.shape == (max_seq_len, 16)
    assert module.sin_cached.shape == (max_seq_len, 16)
    assert "cos_cached" not in module.state_dict()
    assert "sin_cached" not in module.state_dict()
    utils.gems_assert_close(module.cos_cached, ref_cos, dtype)
    utils.gems_assert_close(module.sin_cached, ref_sin, dtype)

    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_position_ids = utils.to_reference(position_ids)
    ref_out_query, ref_out_key = _torch_apply_rope(
        ref_query,
        ref_key,
        utils.to_reference(ref_cos, True),
        utils.to_reference(ref_sin, True),
        ref_position_ids,
        rotary_interleaved,
    )
    out_query, out_key = module(query, key, position_ids=position_ids, inplace=False)

    utils.gems_assert_close(out_query, ref_out_query, dtype, reduce_dim=32)
    utils.gems_assert_close(out_key, ref_out_key, dtype, reduce_dim=32)


@pytest.mark.gems_rotary_embedding
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("scaling_factor", [1.0, 40.0])
def test_gems_deepseek_yarn_rope_cache_and_forward(
    dtype, rotary_interleaved, scaling_factor
):
    max_seq_len = 37
    base = 10000.0
    original_max_position_embeddings = 4096
    beta_fast = 32.0
    beta_slow = 1.0
    mscale = 1.0
    mscale_all_dim = 1.0
    query, key, position_ids = _make_inputs(dtype, max_seq_len)
    module = GemsDeepseekYarnRoPE(
        rotary_dim=32,
        max_position_embeddings=max_seq_len,
        base=base,
        rotary_interleaved=rotary_interleaved,
        dtype=dtype,
        device=flaggems_vllm.device,
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max_position_embeddings,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
    )
    ref_cos, ref_sin = _yarn_cache(
        max_seq_len,
        32,
        base,
        dtype,
        _reference_device(),
        scaling_factor,
        original_max_position_embeddings,
        beta_fast,
        beta_slow,
        mscale,
        mscale_all_dim,
    )

    utils.gems_assert_close(module.cos_cached, ref_cos, dtype)
    utils.gems_assert_close(module.sin_cached, ref_sin, dtype)
    ref_query = utils.to_reference(query, True)
    ref_key = utils.to_reference(key, True)
    ref_position_ids = utils.to_reference(position_ids)
    ref_out_query, ref_out_key = _torch_apply_rope(
        ref_query,
        ref_key,
        utils.to_reference(ref_cos, True),
        utils.to_reference(ref_sin, True),
        ref_position_ids,
        rotary_interleaved,
    )
    out_query, out_key = module(query, key, position_ids=position_ids, inplace=False)

    utils.gems_assert_close(out_query, ref_out_query, dtype, reduce_dim=32)
    utils.gems_assert_close(out_key, ref_out_key, dtype, reduce_dim=32)
