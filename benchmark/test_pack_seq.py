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
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm import pack_seq_triton

from . import base

# =============================================================================
# vLLM availability check
# =============================================================================

try:
    from vllm.v1.attention.ops.common import pack_seq_triton as vllm_pack_seq

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# =============================================================================
# FP8 availability check
# =============================================================================


@triton.jit
def _fp8_check_kernel(x, y):
    val = tl.load(x)
    tl.store(y, val)


try:
    FP8_DTYPE = torch.float8_e4m3fn
    x1 = torch.randn(1, dtype=torch.float32, device=flaggems_vllm.device).to(FP8_DTYPE)
    y1 = torch.empty([1], dtype=FP8_DTYPE, device=flaggems_vllm.device)
    _fp8_check_kernel[(1,)](x1, y1)
    FP8_AVAILABLE = True
except Exception:
    try:
        FP8_DTYPE = torch.float8_e5m2
        x2 = torch.randn(1, dtype=torch.float32, device=flaggems_vllm.device).to(
            FP8_DTYPE
        )
        y2 = torch.empty([1], dtype=FP8_DTYPE, device=flaggems_vllm.device)
        _fp8_check_kernel[(1,)](x2, y2)
        FP8_AVAILABLE = True
    except Exception:
        FP8_DTYPE = None
        FP8_AVAILABLE = False


# =============================================================================
# Benchmark shapes: (N, D, B, lengths_list)
# =============================================================================

PACK_BENCH_SHAPES = [
    (512, 64, 5, [64, 128, 64, 128, 128]),
    (4096, 128, 4, [1024, 1024, 1024, 1024]),
    (8192, 256, 5, [1024, 2048, 1024, 2048, 2048]),
    (2048, 512, 4, [512, 512, 512, 512]),
    (16384, 64, 8, [2048] * 8),
    (1024, 1024, 4, [256] * 4),
    (2048, 2048, 512, [4] * 512),
    (4094, 1024, 1024, [4] * 1024),
    (8192, 1024, 1024, [8] * 1024),
]

FP8_BENCH_SHAPES = [
    (512, 64, 5, [64, 128, 64, 128, 128]),
    (4096, 128, 4, [1024, 1024, 1024, 1024]),
    (2048, 512, 4, [512, 512, 512, 512]),
    (2048, 2048, 512, [4] * 512),
    (4094, 1024, 1024, [4] * 1024),
    (8192, 1024, 1024, [8] * 1024),
]


# =============================================================================
# Custom Benchmark class — pack_seq (float dtypes)
# =============================================================================


class PackSeqBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = PACK_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._pack_input_fn(config, cur_dtype)

    def _pack_input_fn(self, config, dtype):
        N, D, B, lengths_list = config
        device = flaggems_vllm.device
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
        x = torch.randn(N, D, dtype=dtype, device=device)
        yield x, lengths


# =============================================================================
# Custom Benchmark class — pack_seq (FP8)
# =============================================================================


class PackSeqFP8Benchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = FP8_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for config in self.shapes:
            yield from self._fp8_input_fn(config)

    def _fp8_input_fn(self, config):
        N, D, B, lengths_list = config
        device = flaggems_vllm.device
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
        x = torch.randn(N, D, dtype=torch.float32, device=device) * 0.1
        x_fp8 = x.to(FP8_DTYPE)
        yield x_fp8, lengths


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not HAS_VLLM,
    reason="requires vLLM to be installed for reference comparison",
)
def test_pack_seq():
    bench = PackSeqBenchmark(
        op_name="pack_seq_triton",
        torch_op=vllm_pack_seq,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.set_gems(pack_seq_triton)
    bench.run()


# =============================================================================
# Custom Benchmark class — pack_seq (INT8)
# =============================================================================


class PackSeqINT8Benchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = PACK_BENCH_SHAPES

    def get_input_iter(self, cur_dtype):
        del cur_dtype
        for config in self.shapes:
            yield from self._int8_input_fn(config)

    def _int8_input_fn(self, config):
        N, D, B, lengths_list = config
        device = flaggems_vllm.device
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
        x = torch.randint(-128, 128, (N, D), dtype=torch.int8, device=device)
        # Explicit int pad_value (0) instead of the float default (-inf):
        # a quantization-style int8 pad, and representative of the
        # performance-relevant case (correctness edge cases are covered
        # by tests/test_pack_seq.py instead).
        yield x, lengths, 0


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not HAS_VLLM,
    reason="requires vLLM to be installed for reference comparison",
)
@pytest.mark.skipif(
    not FP8_AVAILABLE,
    reason="FP8 is not supported on the current device",
)
def test_pack_seq_fp8():
    bench = PackSeqFP8Benchmark(
        op_name="pack_seq_triton",
        torch_op=vllm_pack_seq,
        dtypes=[FP8_DTYPE],
    )
    bench.set_gems(pack_seq_triton)
    bench.run()


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not HAS_VLLM,
    reason="requires vLLM to be installed for reference comparison",
)
def test_pack_seq_int8():
    bench = PackSeqINT8Benchmark(
        op_name="pack_seq_triton",
        torch_op=vllm_pack_seq,
        dtypes=[torch.int8],
    )
    bench.set_gems(pack_seq_triton)
    bench.run()
