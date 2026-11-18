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

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_FP8_MLA,
    MTHREADS_FP8_MLA,
    assert_mthreads_sparse_accuracy,
    assert_sparse_fixture_close,
    flashmla_reference,
    make_mthreads_sparse_inputs,
    make_sparse_inputs,
    mthreads_sparse_reference,
    quantize_sparse_fp8_fixture,
)

from . import base, conftest, consts
from .test_blas_perf_parallel import ParallelBenchmarkMixin


def run_prepared_baseline(handle):
    return handle()


def run_prepared_gems(handle):
    return handle()


class FlashmlaSparseBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            (
                run_prepared_baseline
                if MTHREADS_FP8_MLA
                else flashmla_reference().flash_mla_sparse_fwd
            ),
            [torch.bfloat16],
        )
        self.set_gems(
            run_prepared_gems
            if MTHREADS_FP8_MLA
            else flaggems_vllm.flash_mla_sparse_fwd_w8a8
        )

    def get_input_iter(self, dtype):
        if MTHREADS_FP8_MLA:
            yield from self._get_mthreads_inputs(dtype)
            return
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

    @staticmethod
    def _workload_shapes():
        path = Path(__file__).with_name("flash_mla_sparse_fwd_w8a8_shapes.json")
        with path.open() as stream:
            return [tuple(row["shape"]) for row in json.load(stream)["records"]]

    def set_more_shapes(self):
        if (
            MTHREADS_FP8_MLA
            and Path(conftest.Config.shape_file).resolve()
            == Path(__file__).with_name("core_shapes.yaml").resolve()
        ):
            # Match the seven-field export keys and their seed-42 fixtures.
            self.shapes = [shape[:7] for shape in self.shapes]
            return self._workload_shapes()
        return []

    def _get_mthreads_inputs(self, dtype):
        from flaggems_vllm.runtime.backend._mthreads.ops.flash_mla_sparse_fwd_w8a8 import (
            prepare_flash_mla_sparse_fwd_w8a8,
        )
        from flaggems_vllm.runtime.backend._mthreads.testing.flash_mla_sparse_fwd_w8a8 import (
            prepare_baseline,
        )

        fixture_shapes = self.shapes
        if conftest.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            fixture_shapes = self.shapes + self._workload_shapes()

        def shape_key(shape):
            _, heads, dims, tokens, h_kv, topk, value_dim = shape[:7]
            if len(shape) not in (7, 10) or h_kv != 1 or value_dim != 512:
                raise ValueError(
                    "MThreads sparse FP8 MLA requires one KV head and DV=512"
                )
            if dims not in (512, 576):
                raise ValueError("MThreads sparse FP8 MLA requires DQK=512 or 576")
            return heads, dims, tokens, topk, shape[9] if len(shape) == 10 else 42

        @lru_cache(maxsize=3)
        def fixture(key):
            heads, dims, tokens, topk, seed = key
            batch = max(shape[0] for shape in fixture_shapes if shape_key(shape) == key)
            return make_mthreads_sparse_inputs(
                batch, heads, topk, tokens, dtype, dims - 512, seed
            )

        for shape in self.shapes:
            batch = shape[0]
            heads, _, _, topk, _ = key = shape_key(shape)
            full, query = fixture(key)
            packed = [t[:batch] if i in (0, 1, 4, 6) else t for i, t in enumerate(full)]
            sink, length = shape[7:9] if len(shape) == 10 else (False, False)
            attn_sink = (
                torch.randn(heads, device=query.device, dtype=torch.float32)
                if sink
                else None
            )
            topk_length = (
                torch.randint(
                    0, topk + 1, (batch,), device=query.device, dtype=torch.int32
                )
                if length
                else None
            )
            kwargs = dict(attn_sink=attn_sink, topk_length=topk_length)
            torch.backends.cuda.matmul.allow_tf32 = False
            expected = mthreads_sparse_reference(
                packed, reference_device=query.device, **kwargs
            )
            gems, actual = prepare_flash_mla_sparse_fwd_w8a8(*packed, **kwargs)
            assert_mthreads_sparse_accuracy(actual, expected, output_atol=2e-3)
            native = prepare_baseline(packed, query[:batch], **kwargs)
            assert_mthreads_sparse_accuracy(native(), expected, output_atol=2e-3)
            yield (native,), (gems,), shape

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple], {}

    def record_shapes(self, input_tuple):
        return input_tuple[2]

    def get_latency(self, op, *args, **kwargs):
        original, packed, _ = args[0]
        selected = packed if op is self.gems_op else original
        return super().get_latency(op, *selected)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="Sparse FP8 MLA backend is unavailable")
def test_flash_mla_sparse_fwd_w8a8():
    FlashmlaSparseBenchmark().run()
