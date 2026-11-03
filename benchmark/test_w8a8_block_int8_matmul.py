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


W8A8_BLOCK_INT8_MNK_SHAPES = [
    (64, 128, 128),
    (128, 256, 512),
    (1, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (83, 7748, 3884),
    (84, 7168, 3884),
]

W8A8_BLOCK_INT8_BLOCK_SIZE = [128, 128]


def rand_int8_tensor(shape, device, dtype):
    iinfo = torch.iinfo(dtype)
    return torch.randint(iinfo.min, iinfo.max + 1, shape, device=device, dtype=dtype)


class W8A8BlockINT8MatmulBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)

        if block_size is None:
            self.block_size = W8A8_BLOCK_INT8_BLOCK_SIZE[:]
        else:
            self.block_size = list(block_size)

        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = W8A8_BLOCK_INT8_MNK_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype) -> Generator:
        if dtype is None:
            raise RuntimeError("w8a8_block_int8_matmul benchmark requires INT8 inputs")

        block_n, block_k = self.block_size
        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_int8_tensor((m, k), self.device, dtype).contiguous()
            B = rand_int8_tensor((n, k), self.device, dtype).contiguous()
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

            yield A, B, As, Bs, self.block_size[:], torch.float16

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        m, k = A.shape
        n = B.shape[0]
        return 2 * m * n * k


@pytest.mark.w8a8_block_int8_matmul
@pytest.mark.skipif(
    not VLLM_W8A8_BLOCK_INT8_AVAILABLE,
    reason="w8a8_block_int8_matmul benchmark requires vLLM baseline operator",
)
def test_perf_w8a8_block_int8_matmul():
    if not hasattr(flaggems_vllm, "w8a8_block_int8_matmul"):
        pytest.skip("w8a8_block_int8_matmul is not included on this backend")

    bench = W8A8BlockINT8MatmulBenchmark(
        op_name="w8a8_block_int8_matmul",
        torch_op=vllm_w8a8_block_int8_matmul,
        gems_op=flaggems_vllm.w8a8_block_int8_matmul,
        dtypes=[torch.int8],
    )
    bench.run()
