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

import importlib.util
from pathlib import Path

import pytest
import torch

import flaggems_vllm

from . import base, conftest
from .test_blas_perf_parallel import ParallelBenchmarkMixin
from .test_int8_einsum import per_token_cast_to_int8


def make_channel_einsum_inputs(b, h, r, d, device, seed=0):
    """Build per-token activations and per-output-channel weights, zero point 0."""
    torch.manual_seed(seed)
    x = torch.randn((b, h, r), device=device, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=device, dtype=torch.bfloat16)
    x_data, x_scale = per_token_cast_to_int8(
        x.reshape(b * h, r), use_ue8m0=False, gran_k=max(1, r)
    )
    y_data, y_scale = per_token_cast_to_int8(
        y.reshape(h * d, r), use_ue8m0=False, gran_k=max(1, r)
    )
    return (
        x_data.view(b, h, r),
        x_scale.view(b, h),
        y_data.view(h, d, r),
        y_scale.view(h, d),
        x,
        y,
    )


class INT8EinsumChannelwiseBenchmark(base.Benchmark):
    """Benchmark for per-channel INT8 ``bhr,hdr->bhd`` einsum."""

    DEFAULT_METRICS = base.consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        default_shape_file = Path(__file__).with_name(self.DEFAULT_SHAPE_FILES)
        shape_file = (
            default_shape_file
            if shape_file_path in (None, self.DEFAULT_SHAPE_FILES)
            else Path(shape_file_path)
        )
        if shape_file.resolve() != default_shape_file.resolve():
            return super().set_shapes(str(shape_file))
        batches = (
            1,
            2,
            4,
            *range(8, 257, 8),
            *range(272, 513, 16),
            1028,
            2048,
            3076,
            3805,
            4100,
            12991,
            13005,
            13021,
            16384,
        )
        self.shapes = [(batch, 1, 4096, 1024) for batch in batches]
        self.shape_desc = "(b, h, r, d)"

    def get_input_iter(self, cur_dtype):
        for b, h, r, d in self.shapes:
            yield make_channel_einsum_inputs(b, h, r, d, flaggems_vllm.device)

    def get_tflops(self, op, *args, **kwargs):
        x_data, _, y_data, _, _, _ = args
        b, h, r = x_data.shape
        d = y_data.shape[1]
        return 2.0 * b * h * r * d


class ParallelINT8EinsumChannelwiseBenchmark(
    ParallelBenchmarkMixin, INT8EinsumChannelwiseBenchmark
):
    RECORD_PARALLEL_RESULTS = True


def metax_vllm_einsum_bf16(
    x: torch.Tensor,
    xs: torch.Tensor | None,
    y: torch.Tensor,
    ys: torch.Tensor | None,
    x_bf16: torch.Tensor,
    y_bf16: torch.Tensor,
) -> torch.Tensor:
    # vllm_metax.utils.deep_gemm.bf16_einsum invokes this native entry point.
    import deep_gemm

    output = torch.empty(
        (x_bf16.shape[0], x_bf16.shape[1], y_bf16.shape[1]),
        dtype=torch.bfloat16,
        device=x_bf16.device,
    )
    deep_gemm.einsum("bhr,hdr->bhd", x_bf16, y_bf16, output)
    return output


def gems_einsum_channelwise_wrapper(x, xs, y, ys, x_bf16, y_bf16):
    return flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, per_channel_quant=True
    )


@pytest.mark.fp8_einsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "metax" or not torch.cuda.is_available(),
    reason="requires MetaX per-channel INT8 einsum",
)
def test_perf_int8_einsum_channelwise():
    if importlib.util.find_spec("deep_gemm") is None:
        pytest.skip("MetaX vLLM BF16 baseline requires the native deep_gemm package")
    op_name = "fp8_einsum"
    previous = len(conftest.TEST_RESULTS.get(op_name, {}).get("details", []))
    bench = ParallelINT8EinsumChannelwiseBenchmark(
        op_name=op_name,
        torch_op=metax_vllm_einsum_bf16,
        dtypes=[torch.int8],
    )
    bench.set_gems(gems_einsum_channelwise_wrapper)
    bench.run()
    for detail in conftest.TEST_RESULTS.get(op_name, {}).get("details", [])[previous:]:
        detail["quantization"] = "per_channel"
        detail["baseline"] = "metax_vllm_bf16"
        detail["input_source"] = "synthetic_per_channel_quantization"
