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

import os
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_FP8_MLA,
    SparseMLAParam,
    assert_sparse_fixture_close,
    flashmla_reference,
    make_sparse_inputs,
    quantize_sparse_fp8_fixture,
    run_sparse_fp8_fixture,
)

from . import base
from .test_blas_perf_parallel import PARALLEL_WORKER_ENV, ParallelBenchmarkMixin


class FlashmlaSparseBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            flashmla_reference().flash_mla_sparse_fwd,
            [torch.bfloat16],
        )
        self.set_gems(run_sparse_fp8_fixture)

    def set_more_shapes(self):
        if os.environ.get(PARALLEL_WORKER_ENV):
            return []
        path = Path(__file__).with_name("flash_mla_sparse_fwd_w8a8_shapes.yaml")
        with path.open() as stream:
            return [
                tuple(shape) for shape in yaml.safe_load(stream)[self.op_name]["shapes"]
            ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if len(shape) not in (7, 10):
                raise ValueError(
                    "Sparse MLA shapes need SQ,HQ,DQK,SKV,HKV,TOPK,DV, "
                    "optionally followed by sink,topk_length,seed"
                )
            s_q, h_q, d_qk, s_kv, h_kv, topk, d_v = shape[:7]
            sink, length, seed = shape[7:] if len(shape) == 10 else (False, False, 0)
            param = SparseMLAParam(
                s_q,
                s_kv,
                topk,
                h_q,
                h_kv,
                d_qk,
                d_v,
                have_attn_sink=sink,
                have_topk_length=length,
                dtype=dtype,
                seed=seed,
            )
            inputs = make_sparse_inputs(param, uniform_indices=False)
            packed = quantize_sparse_fp8_fixture(*inputs)
            reference, _, reference_lse = self.torch_op(*inputs)
            output, lse = run_sparse_fp8_fixture(*packed)
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
        selected = packed if op is self.gems_op else original
        return super().get_latency(op, *selected)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE")
def test_flash_mla_sparse_fwd_w8a8():
    FlashmlaSparseBenchmark().run()
