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
from flaggems_vllm.ops.triton_scaled_mm import triton_scaled_mm

from . import base

device = flaggems_vllm.device

# Check if vLLM triton_scaled_mm is available
try:
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm as vllm_triton_scaled_mm,
    )

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    vllm_triton_scaled_mm = None


def get_8bit_types():
    """Get available 8-bit types for testing."""
    types = [torch.int8]
    # try:
    #     _ = torch.empty(1, dtype=torch.float8_e4m3fn, device=device)
    #     types.append(torch.float8_e4m3fn)
    # except (RuntimeError, AttributeError):
    #     pass
    return types


# Shape configurations: (M, N, K)
MNK_FACTORS = [
    (1, 256, 128),
    (33, 256, 496),
    (64, 971, 1024),
    (64, 20486, 128),
    (512, 256, 496),
    (512, 20486, 1024),
]


class TritonScaledMMBenchmark(base.Benchmark):
    """Benchmark for triton_scaled_mm operation."""

    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.out_dtype = torch.bfloat16

    def set_shapes(self, shape_file_path=None):
        self.shapes = MNK_FACTORS

    def get_input_iter(self, dtype):
        """Generate inputs for all shape and scale configurations."""
        is_floating_point_type = lambda t: torch.tensor(
            [1, 1], dtype=t
        ).is_floating_point()

        scale_configs = [
            (True, True, False),  # scalar_a, scalar_b, no bias
            (True, True, True),  # scalar_a, scalar_b, with bias
            (False, False, False),  # vector_a, vector_b, no bias
            (False, False, True),  # vector_a, vector_b, with bias
            (True, False, False),  # scalar_a, vector_b, no bias
            (False, True, False),  # vector_a, scalar_b, no bias
        ]

        for M, N, K in self.shapes:
            for use_scalar_scale_a, use_scalar_scale_b, use_bias in scale_configs:
                torch.manual_seed(0)
                torch.cuda.manual_seed(0)

                # Generate input tensors
                if is_floating_point_type(dtype):
                    a = (
                        0.25 * torch.rand((M, K), dtype=torch.float32, device=device)
                    ).to(dtype)
                    b = (
                        0.25 * torch.rand((K, N), dtype=torch.float32, device=device)
                    ).to(dtype)
                else:
                    a = torch.randint(-32, 32, (M, K), dtype=dtype, device=device)
                    b = torch.randint(-32, 32, (K, N), dtype=dtype, device=device)

                # Generate scale tensors
                if use_scalar_scale_a:
                    scale_a = torch.rand((1, 1), device=device)
                else:
                    scale_a = 0.25 * torch.rand((M, 1), device=device)

                if use_scalar_scale_b:
                    scale_b = torch.rand((1, 1), device=device)
                else:
                    scale_b = 0.25 * torch.rand((N, 1), device=device)

                # Generate bias if needed
                bias = None
                if use_bias:
                    bias = torch.rand((N,), device=device, dtype=self.out_dtype)

                yield (a, b, scale_a, scale_b, self.out_dtype, bias)


@pytest.mark.triton_scaled_mm
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
def test_triton_scaled_mm_benchmark():
    """Benchmark triton_scaled_mm with all configurations."""
    bench = TritonScaledMMBenchmark(
        op_name="triton_scaled_mm",
        torch_op=vllm_triton_scaled_mm,
        dtypes=get_8bit_types(),
    )
    bench.set_gems(triton_scaled_mm)
    bench.run()
