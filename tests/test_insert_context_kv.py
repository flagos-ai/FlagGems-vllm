# SPDX-License-Identifier: Apache-2.0
"""Independent CPU reference and accelerator tests for DSpark context insertion."""

from dataclasses import dataclass, replace
from typing import Optional

import pytest
import torch

import flaggems_vllm

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM


@dataclass
class InsertCase:
    kv: torch.Tensor
    backing: torch.Tensor
    slot_mapping: torch.Tensor
    position_ids: torch.Tensor
    cos_sin_cache: torch.Tensor
    cache_block_size: int
    fp8_scale: Optional[torch.Tensor]
    cache_format: str

    @property
    def kv_mxfp8(self):
        return self.cache_format == "v41"

    @property
    def k_cache(self):
        dtype, width = {
            "bf16": (torch.bfloat16, 512),
            "fp8": (torch.float8_e4m3fn, 512),
            "v4": (torch.uint8, 584),
            "v41": (torch.uint8, 528),
        }[self.cache_format]
        typed_backing = self.backing.view(dtype)
        return typed_backing.as_strided(
            (self.backing.shape[0], self.cache_block_size, width),
            (typed_backing.stride(0), width, 1),
        )

    def to(self, device):
        return replace(
            self,
            kv=self.kv.to(device),
            backing=self.backing.to(device),
            slot_mapping=self.slot_mapping.to(device),
            position_ids=self.position_ids.to(device),
            cos_sin_cache=self.cos_sin_cache.to(device),
            fp8_scale=None if self.fp8_scale is None else self.fp8_scale.to(device),
        )

    def arguments(self):
        return (
            self.kv,
            self.k_cache,
            self.slot_mapping,
            self.position_ids,
            self.cos_sin_cache,
            self.cache_block_size,
            self.fp8_scale,
            self.kv_mxfp8,
        )


def rope_reference(kv, position_ids, cos_sin_cache):
    """Interleaved rotation with one rounded product followed by fused arithmetic."""
    original = kv.float()
    cos, sin = cos_sin_cache[position_ids].split(32, dim=-1)
    tail = original[:, NOPE_DIM:].reshape(-1, 32, 2)
    even, odd = tail.unbind(-1)
    # Use FP64 to model an FP32 fused multiply-add without relying on host ISA.
    new_even = (even.double() * cos.double() - (odd * sin).double()).float()
    new_odd = (even.double() * sin.double() + (odd * cos).double()).float()
    return torch.cat(
        (original[:, :NOPE_DIM], torch.stack((new_even, new_odd), -1).flatten(1)),
        dim=-1,
    )


def _e4m3_bytes(values):
    clamped = torch.where(values.isnan(), -448.0, values).clamp(-448, 448)
    return clamped.to(torch.float8_e4m3fn).view(torch.uint8)


def _quant_groups(values, group_size):
    groups = values.unflatten(-1, (-1, group_size))
    magnitudes = groups.abs().nan_to_num(posinf=float("inf"))
    amax = magnitudes.amax(dim=-1).double().clamp_min(1.0e-4)
    exponent = (amax / 448.0).log2().ceil().float()
    inv = torch.pow(2.0, -exponent).unsqueeze(-1)
    return _e4m3_bytes(groups * inv).flatten(1), (exponent + 127).clamp(0, 255).byte()


def cpu_insert_context_kv(case):
    """Return complete expected backing storage, including inter-page padding."""
    backing = case.backing.clone()
    live = case.slot_mapping >= 0
    slots = case.slot_mapping[live]
    if not slots.numel():
        return backing
    rows = case.slot_mapping.numel()
    transformed = rope_reference(
        case.kv[:rows][live], case.position_ids[:rows][live], case.cos_sin_cache
    )

    if case.cache_format == "v4":
        quant, scales = _quant_groups(transformed[:, :NOPE_DIM], 64)
        rope = transformed[:, NOPE_DIM:].to(torch.bfloat16).view(torch.uint8)
        data = torch.cat((quant, rope), dim=-1)
        scales = torch.cat((scales, torch.zeros_like(scales[:, :1])), dim=-1)
    elif case.cache_format == "v41":
        quant, scales = _quant_groups(transformed.to(torch.bfloat16).float(), 32)
        data = torch.cat((quant, scales), dim=-1)
    elif case.cache_format == "fp8":
        data = _e4m3_bytes(transformed * case.fp8_scale.reciprocal())
    else:
        data = transformed.to(torch.bfloat16).view(torch.uint8)

    for index, slot in enumerate(slots.tolist()):
        page, row = divmod(slot, case.cache_block_size)
        offset = row * data.shape[1]
        backing[page, offset : offset + data.shape[1]] = data[index]
        if case.cache_format == "v4":
            scale_offset = case.cache_block_size * 576 + row * 8
            backing[page, scale_offset : scale_offset + 8] = scales[index]
    return backing


def make_case(
    cache_format,
    *,
    tokens=17,
    pages=3,
    block_size=8,
    padding=16,
    skip_slots=False,
    quadrant=False,
):
    """Build deterministic sparse or dense insertions into shuffled cache pages."""
    generator = torch.Generator().manual_seed(20261124)
    kv = torch.randn((tokens, HEAD_DIM), generator=generator).bfloat16()
    if quadrant:
        angles = torch.arange(32).remainder(4)
        cos = torch.tensor([1.0, 0.0, -1.0, 0.0])[angles].expand(tokens + 1, -1)
        sin = torch.tensor([0.0, 1.0, 0.0, -1.0])[angles].expand(tokens + 1, -1)
    else:
        angles = torch.randn((tokens + 1, 32), generator=generator)
        cos, sin = angles.cos(), angles.sin()
    cos_sin = torch.cat((cos, sin), dim=-1)
    positions = torch.randperm(tokens + 1, generator=generator)[:tokens]
    pages = max(pages, (tokens + block_size - 1) // block_size + 1)
    slots = torch.randperm(pages * block_size, generator=generator)[:tokens]
    if skip_slots:
        slots = slots[:-2]
        slots[::4] = -1
        positions[: slots.numel()][slots < 0] = -999
    width = {"bf16": 1024, "fp8": 512, "v4": 584, "v41": 528}[cache_format]
    backing = torch.full((pages, block_size * width + padding), 0x5A, dtype=torch.uint8)
    scale = torch.tensor(0.7, dtype=torch.float32) if cache_format == "fp8" else None
    return InsertCase(
        kv, backing, slots, positions, cos_sin, block_size, scale, cache_format
    )


@pytest.mark.insert_context_kv
@pytest.mark.parametrize("cache_format", ["bf16", "fp8", "v4", "v41"])
@pytest.mark.parametrize(
    "tokens,quadrant,padding", [(0, True, 16), (17, True, 0), (65, False, 16)]
)
def test_insert_context_kv_matches_full_backing(
    cache_format, tokens, quadrant, padding
):
    case = make_case(
        cache_format, tokens=tokens, quadrant=quadrant, padding=padding, skip_slots=True
    )
    expected = cpu_insert_context_kv(case)
    actual = case.to(flaggems_vllm.device)
    cache_pointer = actual.k_cache.data_ptr()
    for _ in range(2):
        assert flaggems_vllm.insert_context_kv(*actual.arguments()) is None
        assert actual.k_cache.data_ptr() == cache_pointer
        torch.testing.assert_close(actual.backing.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.kv.cpu(), case.kv, rtol=0, atol=0)


@pytest.mark.insert_context_kv
@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "cache_width", "scale"])
def test_insert_context_kv_rejects_bad_metadata(bad):
    case = make_case("fp8" if bad == "scale" else "bf16")
    args = list(case.to(flaggems_vllm.device).arguments())
    if bad == "dtype":
        args[2] = args[2].int()
    elif bad == "shape":
        args[3] = args[3][:1]
    elif bad == "stride":
        args[4] = args[4].T.contiguous().T
    elif bad == "cache_width":
        args[1] = args[1][..., :-1]
    else:
        args[6] = None
    with pytest.raises(NotImplementedError):
        flaggems_vllm.insert_context_kv(*args)


@pytest.mark.insert_context_kv
def test_insert_context_kv_all_negative_slots_do_not_read_positions():
    case = make_case("v4")
    case.slot_mapping.fill_(-1)
    case.position_ids.fill_(-999)
    actual = case.to(flaggems_vllm.device)
    flaggems_vllm.insert_context_kv(*actual.arguments())
    torch.testing.assert_close(actual.backing.cpu(), case.backing, rtol=0, atol=0)


@pytest.mark.insert_context_kv
@pytest.mark.parametrize("cache_format", ["v4", "v41"])
def test_insert_context_kv_ue8m0_boundary(cache_format):
    case = make_case(cache_format, tokens=1, quadrant=True)
    case.slot_mapping.zero_()
    case.kv.zero_()
    group_size = 64 if cache_format == "v4" else 32
    case.kv[0, 0] = 448
    case.kv[0, group_size] = 450
    expected = cpu_insert_context_kv(case)
    scales = case.cache_block_size * 576 if cache_format == "v4" else 512
    assert expected[0, scales : scales + 2].tolist() == [127, 128]
    actual = case.to(flaggems_vllm.device)
    flaggems_vllm.insert_context_kv(*actual.arguments())
    torch.testing.assert_close(actual.backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.insert_context_kv
def test_insert_context_kv_fp8_clamps_nan_from_reciprocal_overflow():
    case = make_case("fp8", tokens=1, quadrant=True)
    case.kv.zero_()
    case.slot_mapping.zero_()
    case.fp8_scale.fill_(1.0e-40)
    actual = case.to(flaggems_vllm.device)
    flaggems_vllm.insert_context_kv(*actual.arguments())
    expected = case.backing.clone()
    expected[0, :HEAD_DIM] = 0xFE
    torch.testing.assert_close(actual.backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.insert_context_kv
def test_insert_context_kv_fp8_reciprocal_rounds_before_quantization():
    case = make_case("fp8", tokens=1, quadrant=True)
    case.kv.zero_()
    case.kv[0, 448] = 1
    case.slot_mapping.zero_()
    midpoint_input = torch.tensor(0x3A333333, dtype=torch.int32).view(torch.float32)
    case.cos_sin_cache[:, 0] = midpoint_input
    expected = cpu_insert_context_kv(case)
    assert expected[0, 448].item() == 0
    actual = case.to(flaggems_vllm.device)
    flaggems_vllm.insert_context_kv(*actual.arguments())
    torch.testing.assert_close(actual.backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.insert_context_kv
def test_insert_context_kv_fp8_subnormal_reciprocal():
    case = make_case("fp8", tokens=1, quadrant=True)
    case.kv.zero_()
    case.kv[0, 0] = 2.0**127
    case.slot_mapping.zero_()
    case.fp8_scale.fill_(2.0**127)
    expected = cpu_insert_context_kv(case)
    assert expected[0, 0].item() == 0x38
    actual = case.to(flaggems_vllm.device)
    flaggems_vllm.insert_context_kv(*actual.arguments())
    torch.testing.assert_close(actual.backing.cpu(), expected, rtol=0, atol=0)


@pytest.mark.insert_context_kv
def test_insert_context_kv_mxfp8_rotary_overflow():
    case = make_case("v41", tokens=1)
    case.kv.fill_(1)
    case.kv[0, 448:450] = torch.finfo(torch.bfloat16).max
    case.cos_sin_cache.fill_(1)
    case.slot_mapping.zero_()
    expected = cpu_insert_context_kv(case)
    assert expected[0, 449] == 0xFE
    assert expected[0, 526] == 255
    actual = case.to(flaggems_vllm.device)
    flaggems_vllm.insert_context_kv(*actual.arguments())
    torch.testing.assert_close(actual.backing.cpu(), expected, rtol=0, atol=0)
