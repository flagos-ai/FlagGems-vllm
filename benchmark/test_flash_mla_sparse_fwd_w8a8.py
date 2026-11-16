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

from importlib import import_module

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_SPARSE_W8A8,
    SPARSE_INT8_OP,
    assert_sparse_fixture_close,
    assert_sparse_int8_close,
    dequantize_sparse_int8_fixture,
    flashmla_reference,
    make_sparse_inputs,
    quantize_sparse_fp8_fixture,
    quantize_sparse_int8_fixture,
)

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


class FlashmlaSparseBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            flashmla_reference().flash_mla_sparse_fwd,
            [torch.bfloat16],
        )
        self.set_gems(SPARSE_INT8_OP or flaggems_vllm.flash_mla_sparse_fwd_w8a8)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if len(shape) not in (7, 10):
                raise ValueError(
                    "Sparse MLA shapes need SQ,HQ,DQK,SKV,HKV,TOPK,DV, "
                    "optionally followed by sink,topk_length,seed"
                )
            inputs = make_sparse_inputs(shape, dtype=dtype)
            if SPARSE_INT8_OP is not None:
                packed = quantize_sparse_int8_fixture(*inputs)
                inputs = dequantize_sparse_int8_fixture(
                    packed, inputs[1].shape[0], dtype
                )
                implementation = import_module(SPARSE_INT8_OP.__module__)
                prepared = implementation.prepare_flash_mla_sparse_fwd_w8a8_int8(
                    *packed
                )
                reference, _, reference_lse = self.torch_op(*inputs)
                output, lse = prepared()
                assert_sparse_int8_close(
                    output[:, 0],
                    lse[:, :, 0],
                    reference,
                    reference_lse,
                    lse_atol=2e-3,  # The native comparison includes BF16 rounding.
                )
                yield inputs, prepared, shape
            else:
                packed = quantize_sparse_fp8_fixture(*inputs)
                reference, _, reference_lse = self.torch_op(*inputs)
                output, lse = self.gems_op(*packed)
                assert_sparse_fixture_close(
                    output[:, 0], lse[:, :, 0], reference, reference_lse
                )
                yield inputs, packed, shape

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple], {}

    def record_shapes(self, input_tuple):
        return input_tuple[2]

    def get_latency(self, op, *args, **kwargs):
        original, packed, _ = args[0]
        if SPARSE_INT8_OP is not None and op is self.gems_op:
            return super().get_latency(packed)
        selected = packed if op is self.gems_op else original
        return super().get_latency(op, *selected)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_SPARSE_W8A8, reason="Sparse W8A8 MLA is not registered")
def test_flash_mla_sparse_fwd_w8a8():
    FlashmlaSparseBenchmark().run()
