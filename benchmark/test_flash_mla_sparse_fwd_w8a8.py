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

from functools import partial
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
    quantize_sparse_fixture,
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
        if SPARSE_INT8_OP is not None:
            implementation = import_module(SPARSE_INT8_OP.__module__)
            self.prepare_int8 = implementation.prepare_flash_mla_sparse_fwd_w8a8_int8
            # The native comparison includes BF16 input rounding.
            self.assert_output = partial(assert_sparse_int8_close, lse_atol=2e-3)
            self.set_gems(SPARSE_INT8_OP)
        else:
            self.prepare_int8 = None
            self.assert_output = assert_sparse_fixture_close
            self.set_gems(flaggems_vllm.flash_mla_sparse_fwd_w8a8)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if len(shape) not in (7, 10):
                raise ValueError(
                    "Sparse MLA shapes need SQ,HQ,DQK,SKV,HKV,TOPK,DV, "
                    "optionally followed by sink,topk_length,seed"
                )
            inputs = make_sparse_inputs(shape, dtype=dtype)
            int8 = self.prepare_int8 is not None
            packed = quantize_sparse_fixture(*inputs, int8=int8)
            if int8:
                q, kv, indices, *options = dequantize_sparse_int8_fixture(
                    packed, inputs[1].shape[0], dtype
                )
                # Keep reference indices separate from the prepared INT8 input.
                inputs = (q, kv, indices.clone(), *options)
                run = self.prepare_int8(*packed)
            else:
                run = partial(self.gems_op, *packed)
            reference, _, reference_lse = self.torch_op(*inputs)
            output, lse = run()
            self.assert_output(output[:, 0], lse[:, :, 0], reference, reference_lse)
            yield inputs, run, shape

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple], {}

    def record_shapes(self, input_tuple):
        return input_tuple[2]

    def get_latency(self, op, *args, **kwargs):
        original, run, _ = args[0]
        if op is self.gems_op:
            return super().get_latency(run)
        return super().get_latency(op, *original)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_SPARSE_W8A8, reason="Sparse W8A8 MLA is not registered")
def test_flash_mla_sparse_fwd_w8a8():
    FlashmlaSparseBenchmark().run()
