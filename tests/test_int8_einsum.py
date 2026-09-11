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
from benchmark.test_int8_einsum import (
    EINSUM_LOW_PRECISION_DTYPE,
    _einsum_low_precision_available,
    _gems_einsum_bf16_wrapper,
    _make_block_einsum_inputs,
)

from .conftest import QUICK_MODE

_EINSUM_BATCHES = (1, 4, 8, 16, 32, 64, 128)
if not QUICK_MODE:
    _EINSUM_BATCHES += (4096, 8192, 16384, 32768)
_EINSUM_BLOCK_SHAPES = [
    (b, h, r, 1024) for h, r in [(8, 4096), (16, 7168)] for b in _EINSUM_BATCHES
]


@pytest.mark.int8_einsum
@pytest.mark.skipif(not _einsum_low_precision_available(), reason="requires PPU INT8")
@pytest.mark.parametrize("shape", _EINSUM_BLOCK_SHAPES)
def test_accuracy_int8_einsum(shape):
    x, xs, y, ys, xf, yf = _make_block_einsum_inputs(
        *shape, (128, 128), flaggems_vllm.device, EINSUM_LOW_PRECISION_DTYPE
    )
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    b, h, r, d = shape
    assert out.shape == (b, h, d) and out.is_contiguous()
    assert torch.isfinite(out).all()
    rows = torch.linspace(0, b - 1, min(b, 32), device=x.device).long()
    cols = torch.linspace(0, d - 1, min(d, 32), device=x.device).long()
    kk = torch.arange(r, device=x.device) // 128
    xd = x[rows].float() * xs[rows][:, :, kk]
    yd = y[:, cols].float() * ys[:, cols // 128, :][:, :, kk]
    ref = torch.einsum("bhr,hdr->bhd", xd, yd)
    original = torch.einsum("bhr,hdr->bhd", xf[rows].float(), yf[:, cols].float())
    sampled = out[rows][:, :, cols].float()
    nrms = ((sampled - ref).square().mean() / ref.square().mean()).sqrt().item()
    total = (
        ((sampled - original).square().mean() / original.square().mean()).sqrt().item()
    )
    print(f"shape={shape} dequant_nrms={nrms:.6f} total_nrms={total:.6f}")
    limit = 0.10 if flaggems_vllm.vendor_name == "thead" else 0.20
    assert nrms < limit and total < limit
    # Validate the floating precision route for the same layouts, including
    # the largest interleaved input whose element offsets exceed int32.
    floating = _gems_einsum_bf16_wrapper(xf, None, yf, None, xf, yf)
    floating_sample = floating[rows][:, :, cols].float()
    floating_nrms = (
        ((floating_sample - original).square().mean() / original.square().mean())
        .sqrt()
        .item()
    )
    assert floating_nrms < 0.01


@pytest.mark.einsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU precision dispatch"
)
@pytest.mark.parametrize("shape", [(3, 2, 129, 33), (16, 4, 256, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_einsum_precision_route(shape, dtype):
    b, h, r, d = shape
    x = torch.randn((b, h, r), dtype=dtype, device=flaggems_vllm.device)
    y = torch.randn((h, d, r), dtype=dtype, device=x.device)
    out = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, None, y, None, output_dtype=dtype
    )
    ref = torch.einsum("bhr,hdr->bhd", x.float(), y.float())
    error = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert error.item() < 0.01


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU INT8")
@pytest.mark.parametrize("layout", ["contiguous", "offset", "padded", "broadcast"])
@pytest.mark.parametrize("shape", [(16, 2, 64, 32), (32, 2, 128, 128), (3, 2, 129, 33)])
def test_int8_einsum_layouts(shape, layout):
    b, h, r, d = shape
    if layout == "offset":
        x = torch.randint(
            -128, 128, (b * h * r + 1,), dtype=torch.int8, device=flaggems_vllm.device
        )[1:].view(b, h, r)
    elif layout == "padded":
        x = torch.randint(
            -128, 128, (b, h, r + 1), dtype=torch.int8, device=flaggems_vllm.device
        )[:, :, :r]
    elif layout == "broadcast":
        x = torch.randint(
            -128, 128, (1, h, r), dtype=torch.int8, device=flaggems_vllm.device
        ).expand(b, -1, -1)
    else:
        x = torch.randint(
            -128, 128, (b, h, r), dtype=torch.int8, device=flaggems_vllm.device
        )
    y = torch.randint(-128, 128, (h, d, r), dtype=torch.int8, device=x.device)
    xs = torch.rand((b, h, (r + 127) // 128), device=x.device) * 0.01
    ys = torch.rand((h, (d + 127) // 128, (r + 127) // 128), device=x.device) * 0.01
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    kk = torch.arange(r, device=x.device) // 128
    nn = torch.arange(d, device=x.device) // 128
    ref = torch.einsum(
        "bhr,hdr->bhd", x.float() * xs[:, :, kk], y.float() * ys[:, nn, :][:, :, kk]
    )
    nrms = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert torch.isfinite(out).all() and nrms.item() < 0.10


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU INT8")
@pytest.mark.parametrize("shape", [(0, 2, 128, 32), (3, 2, 0, 32), (3, 2, 128, 0)])
def test_int8_einsum_empty(shape):
    b, h, r, d = shape
    x = torch.empty((b, h, r), dtype=torch.int8, device=flaggems_vllm.device)
    y = torch.empty((h, d, r), dtype=torch.int8, device=x.device)
    xs = torch.ones((b, h, (r + 127) // 128), device=x.device)
    ys = torch.ones((h, (d + 127) // 128, (r + 127) // 128), device=x.device)
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    assert out.shape == (b, h, d)
    assert torch.count_nonzero(out) == 0


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU INT8")
def test_int8_einsum_validation_and_extremes():
    x = torch.full((16, 2, 64), -128, dtype=torch.int8, device=flaggems_vllm.device)
    y = torch.full((2, 32, 64), 127, dtype=torch.int8, device=x.device)
    y[:, ::2, :] = -128
    xs = torch.full((16, 2, 1), 0.5, device=x.device)
    ys = torch.full((2, 1, 1), 0.25, device=x.device)
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    ref = (torch.einsum("bhr,hdr->bhd", x.float(), y.float()) * 0.125).bfloat16()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    with pytest.raises(ValueError, match="equation|supports"):
        flaggems_vllm.int8_einsum("bij,bjk->bik", x, xs, y, ys)
    with pytest.raises(ValueError, match="scale"):
        flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, None, y, ys)
    with pytest.raises(TypeError, match="matching"):
        flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y.float(), ys)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU W8A8 interface")
@pytest.mark.parametrize("shape", [(3, 2, 129, 33), (128, 2, 256, 128)])
@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("provide_output", [False, True])
def test_w8a8_block_int8_bmm_interface(shape, dtype, provide_output):
    b, h, k, n = shape
    x, xs, y, ys, xf, yf = _make_block_einsum_inputs(
        b,
        h,
        k,
        n,
        (128, 128),
        flaggems_vllm.device,
        torch.int8 if dtype == torch.int8 else torch.bfloat16,
    )
    if dtype != torch.int8:
        x, y = x.to(dtype), y.to(dtype)
    # Match PR #3297: y/ys retain [B,N,K]/[B,N-block,K-block].
    x = x.permute(1, 0, 2)
    xs = xs.permute(1, 0, 2) if xs is not None else None
    z = (
        torch.empty((b, h, n), dtype=torch.float32, device=x.device).permute(1, 0, 2)
        if provide_output
        else None
    )
    result = flaggems_vllm.w8a8_block_int8_bmm(
        x, y, xs, ys, block_size=[128, 128], z=z, output_dtype=torch.float32
    )
    if provide_output:
        assert result is z
    assert result.shape == (h, b, n) and result.dtype == torch.float32
    if dtype == torch.int8:
        kk = torch.arange(k, device=x.device) // 128
        nn = torch.arange(n, device=x.device) // 128
        ref = torch.bmm(
            x.float() * xs[:, :, kk],
            (y.float() * ys[:, nn, :][:, :, kk]).transpose(1, 2),
        )
    else:
        ref = torch.bmm(x.float(), y.float().transpose(1, 2))
    nrms = ((result - ref).square().mean() / ref.square().mean()).sqrt()
    assert torch.isfinite(result).all() and nrms.item() < (
        0.10 if dtype == torch.int8 else 0.01
    )


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU W8A8 interface")
def test_w8a8_block_int8_bmm_output_validation():
    x = torch.ones((2, 3, 128), dtype=torch.int8, device=flaggems_vllm.device)
    y = torch.ones((2, 128, 128), dtype=torch.int8, device=x.device)
    xs = torch.ones((2, 3, 1), device=x.device)
    ys = torch.ones((2, 1, 1), device=x.device)
    with pytest.raises(ValueError, match="shape"):
        flaggems_vllm.w8a8_block_int8_bmm(x, y, xs.transpose(1, 2), ys)
    z = torch.empty((2, 3, 128), dtype=torch.float32, device=x.device)
    with pytest.raises(ValueError, match="dtype"):
        flaggems_vllm.w8a8_block_int8_bmm(x, y, xs, ys, z=z)
    with pytest.raises(ValueError, match="scale"):
        flaggems_vllm.w8a8_block_int8_bmm(x.bfloat16(), y.bfloat16(), xs, ys)
