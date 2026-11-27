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

DEVICE = flaggems_vllm.device
EPS = 1e-6


def _fma32(x, y, addend):
    # Float64 represents an FP32 product exactly. Retain its sum's rounding
    # direction so the conversion to FP32 handles midpoint cases as fmaf does.
    product = x.double() * y.double()
    addend = addend.double()
    total = product + addend
    part = total - product
    error = (product - (total - part)) + (addend - part)
    bits = total.view(torch.int64)
    round_to_odd = torch.isfinite(total) & (error != 0) & ((bits & 1) == 0)
    step = torch.where((total < 0) == (error < 0), 1, -1)
    return torch.where(round_to_odd, bits + step, bits).view(torch.float64).float()


def _rotate(x, positions, table):
    cos, sin = table[positions].chunk(2, dim=-1)
    if x.ndim == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    even, odd = x[..., 448::2].float(), x[..., 449::2].float()
    pairs = torch.stack(
        (_fma32(even, cos, -(odd * sin)), _fma32(even, sin, odd * cos)), dim=-1
    )
    return pairs.flatten(-2).bfloat16()


def reference(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size,
    apply_q_norm=True,
):
    q = q.detach().cpu().clone()
    kv = kv.detach().cpu().clone()
    cache = k_cache.detach().cpu().clone()
    positions, table = position_ids.cpu(), cos_sin_cache.cpu()
    x = q.float()
    if apply_q_norm:
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    q_expected = x.bfloat16() if apply_q_norm else q
    q_expected[..., 448:] = _rotate(x, positions, table)
    slots = slot_mapping.cpu()
    valid = slots >= 0
    tokens = torch.arange(slots.numel())[valid]
    blocks, rows = slots[valid] // cache_block_size, slots[valid] % cache_block_size
    cache[blocks, rows, :448] = kv[tokens, :448]
    cache[blocks, rows, 448:] = _rotate(kv[tokens], positions[tokens], table)
    return q_expected, cache


def _assert_bits(actual, expected):
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


def assert_cache(actual_backing, expected_backing, cache_view, slots, block_size):
    # RoPE results use the upstream exact numeric comparison (+0 == -0).
    # Compare all remaining storage, including copied NaNs, by representation.
    actual = actual_backing.detach().cpu().clone()
    expected = expected_backing.detach().cpu()
    layout = (cache_view.shape, cache_view.stride(), cache_view.storage_offset())
    actual_view, expected_view = actual.as_strided(*layout), expected.as_strided(
        *layout
    )
    slots = slots.cpu()
    slots = slots[slots >= 0]
    pages, rows = slots // block_size, slots % block_size
    tail = expected_view[pages, rows, 448:]
    torch.testing.assert_close(actual_view[pages, rows, 448:], tail, rtol=0, atol=0)
    actual_view[pages, rows, 448:] = tail
    _assert_bits(actual, expected)


def _case(n, heads, slots, norm, *, row_pad=0, page_pad=0, offset=0, zeros=False):
    generator = torch.Generator().manual_seed(971)
    q = (torch.randn(n, heads, 512, generator=generator) * 3.5).bfloat16()
    kv = torch.randn(n, 512, generator=generator).bfloat16()
    if zeros:
        q.zero_()
        kv.zero_()
    elif n:
        # Signed zero and distinct NaN payloads exercise verbatim BF16 copies.
        kv.view(torch.int16)[:, :3] = torch.tensor([-32768, 32705, 32706])
        if norm is False:
            q.view(torch.int16)[:, :, :3] = kv.view(torch.int16)[:, None, :3]
    q = q.to(DEVICE)
    kv = kv.to(DEVICE)
    block = 4
    pages = max(2, 1 + max(slots, default=0) // block)
    row_stride = 512 + row_pad
    page_stride = block * row_stride + page_pad
    storage = torch.full(
        (offset + pages * page_stride + 8,), -93.5, dtype=torch.bfloat16
    )
    backing = storage.to(DEVICE)
    cache = backing.as_strided(
        (pages, block, 512), (page_stride, row_stride, 1), offset
    )
    positions = (torch.arange(n) * 3 + 4) % 7 + 1
    angle = torch.arange(8)[:, None] * (torch.arange(32)[None, :] + 1) * 0.079
    table = torch.cat((angle.cos(), angle.sin()), 1)
    q_before, kv_before = q.cpu().clone(), kv.cpu().clone()
    expected = backing.cpu().clone()
    expected_cache = expected.as_strided(cache.shape, cache.stride(), offset)
    slot_mapping = torch.tensor(slots, dtype=torch.int64)
    q_expected, cache_expected = reference(
        q_before,
        kv_before,
        cache,
        slot_mapping,
        positions,
        table,
        EPS,
        block,
        norm is not False,
    )
    expected_cache.copy_(cache_expected)

    args = (
        q,
        kv,
        cache,
        slot_mapping.to(DEVICE),
        positions.to(DEVICE),
        table.to(DEVICE),
        EPS,
        block,
    )
    fn = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
    result = fn(*args) if norm is None else fn(*args, apply_q_norm=norm)
    assert result is None
    q_actual = q.cpu()
    if norm is False:
        _assert_bits(q_actual[..., :448], q_before[..., :448])
        torch.testing.assert_close(
            q_actual[..., 448:], q_expected[..., 448:], rtol=1e-2, atol=1e-2
        )
    else:
        torch.testing.assert_close(q_actual, q_expected, rtol=1e-2, atol=1e-2)
    _assert_bits(kv, kv_before)
    # Includes unassigned slots, padding between rows/pages, and both outer guards.
    assert_cache(backing, expected, cache, slot_mapping, block)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
@pytest.mark.parametrize("n", [4, 17])
@pytest.mark.parametrize("heads", [8, 17])
@pytest.mark.parametrize("norm", [None, True, False])
def test_full_cache(n, heads, norm):
    _case(n, heads, list(range(n)), norm)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
@pytest.mark.parametrize("offset", [0, 1])
def test_large_partial_query_tile(offset):
    # 1037 Q head rows span token boundaries and leave an incomplete final tile.
    _case(61, 17, [4, -1, 3], True, row_pad=3, page_pad=7, offset=offset)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
@pytest.mark.parametrize("heads", [1, 8, 17])
@pytest.mark.parametrize("norm", [None, True, False])
@pytest.mark.parametrize("slots", [[], [4, -1, 3], [-1] * 5, [7, 3, -1, 4, 0]])
def test_partial_insert_and_strides(heads, norm, slots):
    # Q rows 1, 3, 4 rotate even for [S, -1, S-1]. Page 1 begins at an odd offset.
    _case(5, heads, slots, norm, row_pad=3, page_pad=7, offset=1)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
@pytest.mark.parametrize("n,slots", [(0, []), (5, [4, -1, 3])])
def test_empty_and_zero(n, slots):
    _case(n, 1, slots, True, page_pad=16, zeros=True)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
@pytest.mark.parametrize("n", [0, 2])
def test_too_many_slots(n):
    q = torch.zeros(n, 1, 512, dtype=torch.bfloat16, device=DEVICE)
    kv = torch.zeros(n, 512, dtype=q.dtype, device=DEVICE)
    cache = torch.full((2, 4, 512), 7.0, dtype=q.dtype, device=DEVICE)
    q_before, cache_before = q.cpu().clone(), cache.cpu().clone()
    slots = torch.zeros(n + 1, dtype=torch.int64, device=DEVICE)
    positions = torch.zeros(n, dtype=torch.int64, device=DEVICE)
    table = torch.ones(1, 64, dtype=torch.float32, device=DEVICE)
    with pytest.raises((ValueError, RuntimeError), match="slot|insert"):
        flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            q, kv, cache, slots, positions, table, EPS, 4
        )
    _assert_bits(q, q_before)
    _assert_bits(cache, cache_before)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
def test_norm_keeps_fp32_until_rope():
    q = torch.full((1, 1, 512), 0.25, dtype=torch.bfloat16)
    q[0, 0, 448:450] = torch.tensor([8, 24])
    # cos:sin = 3:1 cancels this pair. Rounding both normalized inputs to
    # BF16 first leaves a residual near -0.030, exceeding the upstream atol.
    table = torch.ones(1, 64)
    table[:, :32] *= 3 / 10**0.5
    table[:, 32:] *= 1 / 10**0.5
    q = q.to(DEVICE)
    kv = torch.zeros(1, 512, dtype=q.dtype, device=DEVICE)
    cache = torch.zeros(1, 4, 512, dtype=q.dtype, device=DEVICE)
    slots = torch.empty(0, dtype=torch.int64, device=DEVICE)
    positions = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    args = (q, kv, cache, slots, positions, table.to(DEVICE), EPS, 4)
    expected, _ = reference(*args)
    assert (
        flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(*args)
        is None
    )
    torch.testing.assert_close(q.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert
def test_kv_rotary_rounding_boundary():
    q = torch.zeros(1, 1, 512, dtype=torch.bfloat16, device=DEVICE)
    kv = torch.zeros(1, 512, dtype=torch.bfloat16)
    kv[0, 448:450] = torch.tensor([-0.5078125, 0.54296875])
    kv[0, 450:452] = torch.tensor([1.5, -(2**-90)])
    table = torch.zeros(1, 64, dtype=torch.float32)
    table[0, 1] = torch.tensor(0x3F21AAAB, dtype=torch.int32).view(torch.float32)
    table[0, 33] = 1.0
    table[0, 0], table[0, 32] = -0.749736487865448, 0.6617364883422852
    slots = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    cache = torch.zeros(1, 1, 512, dtype=q.dtype, device=DEVICE)
    args = (q, kv.to(DEVICE), cache, slots, slots, table.to(DEVICE), EPS, 1)
    expected = reference(*args, False)[1]
    assert (
        flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
            *args, False
        )
        is None
    )
    # The fused even result is FP32 0x3caf7fff. Two rounded products would
    # land on 0x3caf8000 and round up by one BF16 ULP.
    _assert_bits(
        cache[0, 0, 448:450],
        torch.tensor([0x3CAF, -16578], dtype=torch.int16).view(torch.bfloat16),
    )
    # The tiny positive addend nudges an FP32 tie up; FP64 alone loses it.
    _assert_bits(
        cache[0, 0, 450:451],
        torch.tensor([0x3F73], dtype=torch.int16).view(torch.bfloat16),
    )
    _assert_bits(cache[0, 0, 448:452], expected[0, 0, 448:452])
