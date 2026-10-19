# SPDX-License-Identifier: Apache-2.0
"""Verify the cache codec against independent CPU PyTorch E4M3FN conversion."""

import pytest
import torch
import triton
import triton.language as tl

from flaggems_vllm import device
from flaggems_vllm.ops.fp8_storage import decode_e4m3fn, encode_e4m3fn

pytestmark = pytest.mark.gpu


@triton.jit
def _encode_kernel(x, output, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, mask=offsets < count, other=0)
    tl.store(output + offsets, encode_e4m3fn(values), mask=offsets < count)


@triton.jit
def _decode_kernel(x, output, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, mask=offsets < count, other=0)
    tl.store(output + offsets, decode_e4m3fn(values), mask=offsets < count)


def test_encoding_matches_cpu_cast_at_every_rounding_boundary():
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (positive[:-1] + positive[1:]) / 2
    boundaries = torch.cat(
        [
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
            torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
        ]
    )
    # Exhaust every finite BF16 input in range, including both signed zeros.
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    bf16 = bf16.float()
    bf16 = bf16[torch.isfinite(bf16) & (bf16.abs() <= 448)]
    values = torch.cat(
        [
            boundaries,
            -boundaries,
            bf16,
            torch.tensor([-1e4, 1e4, -torch.inf, torch.inf, torch.nan, -torch.nan]),
        ]
    )
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    inputs = values.to(device)
    output = torch.empty_like(inputs, dtype=torch.uint8)
    _encode_kernel[(triton.cdiv(inputs.numel(), 256),)](
        inputs, output, inputs.numel(), BLOCK=256
    )
    assert torch.equal(output.cpu(), expected)


def test_decoding_matches_cpu_cast_for_every_byte():
    values = torch.arange(256, dtype=torch.uint8)
    expected = values.view(torch.float8_e4m3fn).float()
    inputs = values.to(device)
    output = torch.empty_like(inputs, dtype=torch.float32)
    _decode_kernel[(1,)](inputs, output, inputs.numel(), BLOCK=256)
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    assert torch.equal(
        torch.signbit(actual[[0, 128]]), torch.signbit(expected[[0, 128]])
    )
