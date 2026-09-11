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

from . import base, conftest

# The upstream FP8 einsum shape grid and quantization structure are shared by
# the floating and low-precision routes. PPU quantizes to signed INT8.
IS_PPU = flaggems_vllm.vendor_name == "thead"
pytestmark = pytest.mark.skipif(not IS_PPU, reason="PPU int8_einsum backend")
EINSUM_LOW_PRECISION_DTYPE = torch.int8
DEFAULT_BLOCK_SHAPE = (128, 128)


def _einsum_low_precision_available():
    if IS_PPU:
        return torch.cuda.is_available()
    return False


def _cast_einsum_low_precision(x):
    return x.round().clamp(-128, 127).to(torch.int8)


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the nearest power-of-two (UE8M0 grid)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def per_token_cast_to_int8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 127.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_int8 = (
        _cast_einsum_low_precision(x_view * (1.0 / sf.unsqueeze(2)))
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_int8, sf


def per_block_cast_to_int8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_m = math.ceil(m / gran_k) * gran_k
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 127.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = _cast_einsum_low_precision(x_view * (1.0 / sf))
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


def _make_block_einsum_inputs(b, h, r, d, block_shape, device, dtype, seed=0):
    """Build upstream per-token x and per-block y inputs for bhr,hdr->bhd.

    Return x, xs, y, ys and the two original BF16 tensors used by baselines.
    PPU quantized inputs are signed INT8.
    """
    block_n, block_k = block_shape
    torch.manual_seed(seed)
    x = torch.randn((b, h, r), device=device, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=device, dtype=torch.bfloat16)

    if dtype == torch.bfloat16:
        return x, None, y, None, x, y
    x_int8 = per_token_cast_to_int8(x.reshape(b * h, r), use_ue8m0=True, gran_k=block_k)
    x_data = x_int8[0].view(b, h, r)
    x_scale = x_int8[1].view(b, h, math.ceil(r / block_k))

    y_data = torch.empty_like(y, dtype=EINSUM_LOW_PRECISION_DTYPE)
    y_scale = torch.empty(
        (h, math.ceil(d / block_n), math.ceil(r / block_k)),
        device=device,
        dtype=torch.float32,
    )
    for i in range(h):
        y_data[i], y_scale[i] = per_block_cast_to_int8(y[i], use_ue8m0=True)

    return x_data, x_scale, y_data, y_scale, x, y


class INT8EinsumBenchmark(base.Benchmark):
    """Benchmark for block-wise INT8 ``bhr,hdr->bhd`` einsum on PPU."""

    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.block_shape = DEFAULT_BLOCK_SHAPE

    def set_shapes(self, shape_file_path=None):
        # (b, h, r, d)
        batches = (1, 4, 8, 16, 32, 64, 128, 4096, 8192, 16384, 32768)
        hrd_groups = {
            "flash": (8, 4096, 1024),
            "pro": (16, 7168, 1024),
        }
        self.shapes = [
            (b, h, r, d) for (h, r, d) in hrd_groups.values() for b in batches
        ]

    def get_input_iter(self, cur_dtype):
        device = flaggems_vllm.device
        for b, h, r, d in self.shapes:
            yield _make_block_einsum_inputs(
                b, h, r, d, self.block_shape, device, cur_dtype
            )

    def get_tflops(self, op, *args, **kwargs):
        x_data, _, y_data, _, _, _ = args
        b, h, r = x_data.shape
        d = y_data.shape[1]
        return 2.0 * b * h * r * d


def _gems_einsum_precision_wrapper(x, xs, y, ys, x_bf16, y_bf16):
    if xs is None:
        return _gems_einsum_bf16_wrapper(x, xs, y, ys, x_bf16, y_bf16)
    return flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, block_size=DEFAULT_BLOCK_SHAPE
    )


def _torch_einsum_bf16_wrapper(x, xs, y, ys, x_bf16, y_bf16):
    return torch.einsum("bhr,hdr->bhd", x_bf16, y_bf16)


def _gems_einsum_bf16_wrapper(x, xs, y, ys, x_bf16, y_bf16):
    return flaggems_vllm.int8_einsum("bhr,hdr->bhd", x_bf16, None, y_bf16, None)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16", marks=pytest.mark.einsum),
        pytest.param(
            EINSUM_LOW_PRECISION_DTYPE,
            id="int8",
            marks=pytest.mark.int8_einsum,
        ),
    ],
)
def test_perf_int8_einsum(dtype):
    low_precision = dtype == EINSUM_LOW_PRECISION_DTYPE
    if low_precision and not _einsum_low_precision_available():
        pytest.skip("requires PPU INT8 support")
    op_name = "int8_einsum" if low_precision else "einsum"
    baselines = [("torch_bf16", _torch_einsum_bf16_wrapper)]
    if low_precision:
        baselines.append(("flaggems_bf16", _gems_einsum_bf16_wrapper))
    for baseline_name, baseline in baselines:
        previous = len(conftest.TEST_RESULTS.get(op_name, {}).get("details", []))
        bench = INT8EinsumBenchmark(op_name=op_name, torch_op=baseline, dtypes=[dtype])
        bench.set_gems(_gems_einsum_precision_wrapper)
        bench.run()
        # Keep the public operator ID exact; distinguish references as metadata.
        for detail in conftest.TEST_RESULTS.get(op_name, {}).get("details", [])[
            previous:
        ]:
            detail["baseline"] = baseline_name
