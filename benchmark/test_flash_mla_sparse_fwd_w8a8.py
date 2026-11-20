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

from itertools import product
from pathlib import Path

import pytest
import torch
import yaml

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_FP8_MLA,
    PPU_SPARSE_SUPPORTED,
    assert_sparse_fixture_close,
    check_sparse_decode_sample,
    flashmla_reference,
    make_sparse_decode_inputs,
    make_sparse_inputs,
    quantize_sparse_fp8_fixture,
)

from . import base, consts
from .conftest import Config
from .test_blas_perf_parallel import ParallelBenchmarkMixin
from .test_flash_mla_with_kvcache_w8a8 import TestParam


class FlashmlaSparseBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            flashmla_reference().flash_mla_sparse_fwd,
            [torch.bfloat16],
        )
        self.set_gems(flaggems_vllm.flash_mla_sparse_fwd_w8a8)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            if len(shape) not in (7, 10):
                raise ValueError(
                    "Sparse MLA shapes need SQ,HQ,DQK,SKV,HKV,TOPK,DV, "
                    "optionally followed by sink,topk_length,seed"
                )
            inputs = make_sparse_inputs(shape, dtype=dtype)
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
        selected = packed if op is self.gems_op else original
        return super().get_latency(op, *selected)


class PPUSparseBenchmark(FlashmlaSparseBenchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def __init__(self):
        from flash_mla import flash_mla_with_kvcache

        base.Benchmark.__init__(
            self, "flash_mla_sparse_fwd_w8a8", flash_mla_with_kvcache, [torch.bfloat16]
        )
        self.set_gems(flaggems_vllm.flash_mla_sparse_fwd_w8a8)

    def set_shapes(self, shape_file_path=None):
        if Path(shape_file_path).resolve() != Path(self.DEFAULT_SHAPE_FILES).resolve():
            return super().set_shapes(shape_file_path)
        with open(shape_file_path) as f:
            entry = yaml.safe_load(f)["flash_mla_with_kvcache_w8a8"]
        self.shape_desc = entry["shape_desc"]
        self.shapes = [tuple(shape) for shape in entry["shapes"]]
        if Config.bench_level == consts.BenchLevel.COMPREHENSIVE and not Config.query:
            self.shapes = [
                (128, q, length, h, ragged)
                for length, h, q, ragged in product(
                    (4096, 8192, 16384), (16, 32, 64, 128), (1, 2), (False, True)
                )
            ]

    def get_input_iter(self, dtype):
        from flash_mla import get_mla_metadata

        for shape in self.shapes:
            param = TestParam(*shape)
            # DeepSeek-V3.2 index_topk augments the dense PR #1241 matrix.
            inputs, q, kv = make_sparse_decode_inputs(
                param.cache_lengths(), param.seq_q, param.h_q, 2048, dtype
            )
            metadata, splits = get_mla_metadata()
            native = (
                q,
                kv,
                None,
                None,
                512,
                metadata,
                splits,
                None,
                False,
                False,
                inputs[-1],
            )
            reference, ref_lse = self.torch_op(*native)
            output, lse = self.gems_op(*inputs)
            assert_sparse_fixture_close(output, lse, reference, ref_lse)
            check_sparse_decode_sample(
                inputs, q, kv, (output, lse), (reference, ref_lse)
            )
            yield native, inputs, shape


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(
    not (HAS_FP8_MLA or PPU_SPARSE_SUPPORTED), reason="Requires a sparse W8A8 backend"
)
def test_flash_mla_sparse_fwd_w8a8():
    benchmark = PPUSparseBenchmark if PPU_SPARSE_SUPPORTED else FlashmlaSparseBenchmark
    benchmark().run()
