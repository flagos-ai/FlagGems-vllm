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

from typing import Generator

import pytest
import torch

import flaggems_vllm

from . import base, consts

try:
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        w8a8_block_int8_matmul as vllm_w8a8_block_int8_matmul,
    )

    VLLM_W8A8_BLOCK_INT8_AVAILABLE = True
except Exception:
    vllm_w8a8_block_int8_matmul = None
    VLLM_W8A8_BLOCK_INT8_AVAILABLE = False


try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm as vllm_w8a8_triton_block_scaled_mm,
    )

    VLLM_W8A8_BLOCK_FP8_AVAILABLE = True
except Exception:
    vllm_w8a8_triton_block_scaled_mm = None
    VLLM_W8A8_BLOCK_FP8_AVAILABLE = False


W8A8_BLOCK_MNK_SHAPES = [
    (64, 128, 128),
    (128, 256, 512),
    (1, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (83, 7748, 3884),
    (84, 7168, 3884),
]

W8A8_BLOCK_SIZE = [128, 128]


def rand_quantized_tensor(shape, device, dtype):
    if dtype == torch.int8:
        iinfo = torch.iinfo(dtype)
        return torch.randint(
            iinfo.min, iinfo.max + 1, shape, device=device, dtype=dtype
        )
    finfo = torch.finfo(dtype)
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(min=finfo.min, max=finfo.max)
        .to(dtype)
    )


class W8A8BlockMatmulBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)

        if block_size is None:
            self.block_size = W8A8_BLOCK_SIZE[:]
        else:
            self.block_size = list(block_size)

        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = W8A8_BLOCK_MNK_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype) -> Generator:
        if dtype is None:
            raise RuntimeError(
                "W8A8 block matmul benchmark requires a supported input dtype"
            )

        block_n, block_k = self.block_size
        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_quantized_tensor((m, k), self.device, dtype).contiguous()
            B = rand_quantized_tensor((n, k), self.device, dtype).contiguous()
            As = (
                0.01
                * torch.rand((m, num_k_groups), dtype=torch.float32, device=self.device)
                + 0.005
            ).contiguous()
            Bs = (
                0.01
                * torch.rand(
                    (num_n_groups, num_k_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                + 0.005
            ).contiguous()

            yield A, B, As, Bs, self.block_size[:], torch.bfloat16

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        m, k = A.shape
        n = B.shape[0]
        return 2 * m * n * k


@pytest.mark.w8a8_block_matmul
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("thead", "nvidia", "mthreads"),
    reason="No W8A8 block matmul backend",
)
def test_perf_w8a8_block_matmul():
    if not hasattr(flaggems_vllm, "w8a8_block_matmul"):
        pytest.skip("W8A8 block matmul is unavailable on this architecture")
    if flaggems_vllm.vendor_name == "thead":
        if not VLLM_W8A8_BLOCK_INT8_AVAILABLE:
            pytest.skip("W8A8 block matmul benchmark requires the vLLM INT8 baseline")
        torch_op = vllm_w8a8_block_int8_matmul
        dtypes = [torch.int8]
    else:
        if not VLLM_W8A8_BLOCK_FP8_AVAILABLE:
            pytest.skip("W8A8 block matmul benchmark requires the vLLM FP8 baseline")
        if not consts.FP8_DTYPES:
            pytest.skip("W8A8 block matmul benchmark requires FP8 support")
        torch_op = vllm_w8a8_triton_block_scaled_mm
        dtypes = consts.FP8_DTYPES

    bench = W8A8BlockMatmulBenchmark(
        op_name="w8a8_block_matmul",
        torch_op=torch_op,
        gems_op=flaggems_vllm.w8a8_block_matmul,
        dtypes=dtypes,
    )
    bench.run()
