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
    HAS_FP8_MLA,
    HAS_INT8_MLA,
    assert_sparse_fixture_close,
    check_sparse_int8,
    flashmla_reference,
    make_sparse_inputs,
    prepare_sparse_int8,
    quantize_sparse_fixture,
)

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


class FlashmlaSparseBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        reference = import_module("flash_mla") if HAS_INT8_MLA else flashmla_reference()
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            reference.flash_mla_sparse_fwd,
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
            packed, query, cache = quantize_sparse_fixture(*inputs)
            if HAS_INT8_MLA:
                native, handle = self.prepare_metax_inputs(packed, query, cache)
                yield native, handle, shape
                continue
            reference, _, reference_lse = self.torch_op(*inputs)
            output, lse = self.gems_op(*packed)
            assert_sparse_fixture_close(
                output[:, 0], lse[:, :, 0], reference, reference_lse
            )
            yield inputs, packed, shape

    def prepare_metax_inputs(self, packed, query, cache):
        import flash_mla

        scale, sink, lengths = packed[7:]
        handle, actual = prepare_sparse_int8(
            *packed[:7], softmax_scale=scale, attn_sink=sink, topk_length=lengths
        )
        indices = packed[6].clone()
        valid = torch.zeros_like(indices[..., :1], dtype=torch.bool)
        if query.shape[0] <= 1028 and sink is None and lengths is None:
            # The vendor paged scheduler exceeds C550 shared memory at larger SQ.
            metadata, splits = flash_mla.get_mla_metadata()
            native = partial(
                flash_mla.flash_mla_with_kvcache,
                query[:, None],
                cache.view(-1, 64, 1, cache.shape[-1]),
                None,
                None,
                512,
                metadata,
                splits,
                softmax_scale=scale,
                indices=indices,
                indices_all_valid_per_q=valid,
            )
            expected = native()
        else:
            native = partial(
                self.torch_op,
                query,
                cache,
                indices,
                scale,
                512,
                indices_all_valid_per_q=valid[:, 0],
                attn_sink=sink,
                topk_length=lengths,
            )
            output, _, lse = native()
            expected = output[:, None], lse[:, :, None]
        check_sparse_int8(packed, actual, expected)
        # The paged reference may rewrite invalid indices in place.
        indices.copy_(packed[6])
        return native, handle

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple], {}

    def record_shapes(self, input_tuple):
        return input_tuple[2]

    def get_latency(self, op, *args, **kwargs):
        original, packed, _ = args[0]
        selected = packed if op is self.gems_op else original
        if HAS_INT8_MLA:
            return super().get_latency(selected)
        return super().get_latency(op, *selected)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(
    not (HAS_FP8_MLA or HAS_INT8_MLA), reason="Sparse MLA requires Hopper or C550"
)
def test_flash_mla_sparse_fwd_w8a8():
    FlashmlaSparseBenchmark().run()
