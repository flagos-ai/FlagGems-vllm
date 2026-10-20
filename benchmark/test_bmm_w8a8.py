# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import flaggems_vllm

from . import base
from .conftest import Config
from .test_blas_perf_parallel import ParallelBenchmarkMixin

BMM_WORKLOAD_SHAPES = [
    # Real BMM M/N/K; missing batches are supplemented with B=1,2,4.
    # Counts are aggregated separately for each supplemented batch (sum=26).
    (1, 16384, 7168, 1024),  # count: 12
    (1, 16384, 2112, 7168),  # count: 5
    (1, 448, 7168, 256),  # count: 4
    (1, 14429, 7168, 1024),  # count: 4
    (1, 14429, 2112, 7168),  # count: 1
    (2, 16384, 7168, 1024),  # count: 12
    (2, 16384, 2112, 7168),  # count: 5
    (2, 448, 7168, 256),  # count: 4
    (2, 14429, 7168, 1024),  # count: 4
    (2, 14429, 2112, 7168),  # count: 1
    (4, 16384, 7168, 1024),  # count: 12
    (4, 16384, 2112, 7168),  # count: 5
    (4, 448, 7168, 256),  # count: 4
    (4, 14429, 7168, 1024),  # count: 4
    (4, 14429, 2112, 7168),  # count: 1
    # FlagOSTune DeepSeek-V4-Flash bhr,hdr->bhd calls (sum(count)=41462).
    # Contraction (b,h,r,d) maps to BMM (B,M,N,K)=(h,b,d,r).
    # Shapes are reused; scales follow this operator's channel-wise contract.
    (1, 1, 1024, 4096),  # count: 688
    (1, 2, 1024, 4096),  # count: 688
    (1, 4, 1024, 4096),  # count: 688
    (1, 8, 1024, 4096),  # count: 688
    (1, 16, 1024, 4096),  # count: 688
    (1, 24, 1024, 4096),  # count: 688
    (1, 32, 1024, 4096),  # count: 688
    (1, 40, 1024, 4096),  # count: 688
    (1, 48, 1024, 4096),  # count: 688
    (1, 56, 1024, 4096),  # count: 688
    (1, 64, 1024, 4096),  # count: 688
    (1, 72, 1024, 4096),  # count: 688
    (1, 80, 1024, 4096),  # count: 688
    (1, 88, 1024, 4096),  # count: 688
    (1, 96, 1024, 4096),  # count: 688
    (1, 104, 1024, 4096),  # count: 688
    (1, 112, 1024, 4096),  # count: 688
    (1, 120, 1024, 4096),  # count: 688
    (1, 128, 1024, 4096),  # count: 688
    (1, 136, 1024, 4096),  # count: 688
    (1, 144, 1024, 4096),  # count: 688
    (1, 152, 1024, 4096),  # count: 688
    (1, 160, 1024, 4096),  # count: 688
    (1, 168, 1024, 4096),  # count: 688
    (1, 176, 1024, 4096),  # count: 688
    (1, 184, 1024, 4096),  # count: 688
    (1, 192, 1024, 4096),  # count: 688
    (1, 200, 1024, 4096),  # count: 688
    (1, 208, 1024, 4096),  # count: 688
    (1, 216, 1024, 4096),  # count: 688
    (1, 224, 1024, 4096),  # count: 688
    (1, 232, 1024, 4096),  # count: 688
    (1, 240, 1024, 4096),  # count: 688
    (1, 248, 1024, 4096),  # count: 688
    (1, 256, 1024, 4096),  # count: 688
    (1, 272, 1024, 4096),  # count: 688
    (1, 288, 1024, 4096),  # count: 688
    (1, 304, 1024, 4096),  # count: 688
    (1, 320, 1024, 4096),  # count: 688
    (1, 336, 1024, 4096),  # count: 688
    (1, 352, 1024, 4096),  # count: 688
    (1, 368, 1024, 4096),  # count: 688
    (1, 384, 1024, 4096),  # count: 688
    (1, 400, 1024, 4096),  # count: 688
    (1, 416, 1024, 4096),  # count: 688
    (1, 432, 1024, 4096),  # count: 688
    (1, 448, 1024, 4096),  # count: 688
    (1, 464, 1024, 4096),  # count: 688
    (1, 480, 1024, 4096),  # count: 688
    (1, 496, 1024, 4096),  # count: 1376
    (1, 512, 1024, 4096),  # count: 1376
    (1, 1028, 1024, 4096),  # count: 43
    (1, 2048, 1024, 4096),  # count: 172
    (1, 3076, 1024, 4096),  # count: 43
    (1, 3805, 1024, 4096),  # count: 43
    (1, 4100, 1024, 4096),  # count: 172
    (1, 12991, 1024, 4096),  # count: 43
    (1, 13005, 1024, 4096),  # count: 43
    (1, 13021, 1024, 4096),  # count: 129
    (1, 16384, 1024, 4096),  # count: 4310
    # Supplementary BMM grid; no real-call count is available.
    (8, 1, 1024, 4096),
    (8, 4, 1024, 4096),
    (8, 8, 1024, 4096),
    (8, 16, 1024, 4096),
    (8, 32, 1024, 4096),
    (8, 64, 1024, 4096),
    (8, 128, 1024, 4096),
    (8, 4096, 1024, 4096),
    (8, 8192, 1024, 4096),
    (8, 16384, 1024, 4096),
    (8, 32768, 1024, 4096),
    (16, 1, 1024, 7168),
    (16, 4, 1024, 7168),
    (16, 8, 1024, 7168),
    (16, 16, 1024, 7168),
    (16, 32, 1024, 7168),
    (16, 64, 1024, 7168),
    (16, 128, 1024, 7168),
    (16, 4096, 1024, 7168),
    (16, 8192, 1024, 7168),
    (16, 16384, 1024, 7168),
    (16, 32768, 1024, 7168),
]


def make_channel_int8_bmm_inputs(
    shape: tuple[int, int, int, int], device: str, dtype: torch.dtype = torch.int8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, rows, columns, reduction = shape
    torch.manual_seed(0)
    activation = torch.randint(
        -128, 128, (batch, rows, reduction), device=device, dtype=dtype
    )
    weight = torch.randint(
        -128, 128, (batch, columns, reduction), device=device, dtype=dtype
    ).transpose(1, 2)
    activation_scale = torch.rand((batch, rows), device=device) / 127
    weight_scale = torch.rand((batch, columns), device=device) / 127
    return activation, weight, activation_scale, weight_scale


def bf16_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(baseline_activation, baseline_weight)


def channel_int8_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    return flaggems_vllm.bmm_w8a8_int8(
        activation, weight, activation_scale, weight_scale
    )


class ChannelInt8BmmBenchmark(base.Benchmark):
    """Time BMM on prepared INT8 and BF16 inputs with the shared timer."""

    DEFAULT_SHAPE_DESC = "B, M, N, K"
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def set_more_shapes(self):
        if (
            Path(Config.shape_file).resolve()
            != Path(self.DEFAULT_SHAPE_FILES).resolve()
        ):
            return []
        return BMM_WORKLOAD_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            activation, weight, activation_scale, weight_scale = (
                make_channel_int8_bmm_inputs(shape, self.device, dtype)
            )
            # Both timed APIs receive prepared inputs; conversion stays outside timing.
            baseline_activation = (
                activation.float() * activation_scale[:, :, None]
            ).to(torch.bfloat16)
            baseline_weight = (
                weight.transpose(1, 2).float() * weight_scale[:, :, None]
            ).to(torch.bfloat16)
            yield (
                activation,
                weight,
                activation_scale,
                weight_scale,
                baseline_activation,
                baseline_weight.transpose(1, 2),
            )


class ParallelChannelInt8BmmBenchmark(ParallelBenchmarkMixin, ChannelInt8BmmBenchmark):
    RECORD_PARALLEL_RESULTS = True

    def should_forward_parallel_dtype(self, dtype_name):
        # Workers select their INT8 default without extending the shared CLI choices.
        return False


@pytest.mark.bmm_w8a8
@pytest.mark.skipif(
    "bmm_w8a8_int8" not in vars(flaggems_vllm),
    reason="backend does not export channel-wise INT8 BMM",
)
def test_perf_bmm_w8a8_int8():
    bench = ParallelChannelInt8BmmBenchmark(
        op_name="bmm_w8a8", torch_op=bf16_bmm, dtypes=[torch.int8]
    )
    bench.set_gems(channel_int8_bmm)
    bench.run()
