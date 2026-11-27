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
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_FP8_MLA,
    PPU_SPARSE_SUPPORTED,
    assert_sparse_fixture_close,
    check_sparse_decode_sample,
    flashmla_reference,
    make_sparse_inputs,
    quantize_sparse_fp8_fixture,
    quantize_sparse_int8,
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
        base.Benchmark.__init__(
            self, "flash_mla_sparse_fwd_w8a8", lambda run: run(), [torch.bfloat16]
        )
        self.set_gems(flaggems_vllm.flash_mla_sparse_fwd_w8a8)

    def set_shapes(self, shape_file_path=None):
        default_path = Path(self.DEFAULT_SHAPE_FILES).resolve()
        if (
            shape_file_path is not None
            and Path(shape_file_path).resolve() != default_path
        ):
            return super().set_shapes(shape_file_path)
        # Top 10 by call count from the DeepSeek-V4-Flash and GLM-5.2 traces in #997.
        self.shape_desc = "SQ, HQ, DQK, SKV, HKV, TOPK, DV"
        self.shapes = [
            (16, 64, 512, 1114624, 1, 640, 512),
            (16, 64, 512, 98816, 1, 8320, 512),
            (64, 64, 576, 361344, 1, 2048, 512),
            (449, 64, 576, 361344, 1, 2048, 512),
            (4112, 64, 512, 1114624, 1, 640, 512),
            (4112, 64, 512, 98816, 1, 8320, 512),
            (4, 64, 512, 1114624, 1, 640, 512),
            (4, 64, 512, 98816, 1, 8320, 512),
            (193, 64, 576, 361344, 1, 2048, 512),
            (512, 64, 576, 361344, 1, 2048, 512),
        ]

    def get_input_iter(self, dtype):
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        for shape in self.shapes:
            query, cache, ids, _, dv, sink, lengths = make_sparse_inputs(shape, dtype)
            dim = query.shape[-1]
            scale = 0.5 if dim == 512 else dim**-0.5
            q, qr, qs = quantize_sparse_int8(query[:, None], power_of_two=True)
            if cache.shape[0] % 64:
                cache = torch.nn.functional.pad(
                    cache, (0, 0, 0, 0, 0, 64 - cache.shape[0] % 64)
                )
            kv, kr, ks = quantize_sparse_int8(
                cache.view(-1, 64, dim), power_of_two=True
            )
            inputs = (q, qr, kv, kr, qs, ks, ids, scale, sink, lengths)
            native_q = (torch.cat((q.float(), qr.float()), -1) * qs).to(dtype)
            native_k = (
                (torch.cat((kv.float(), kr.float()), -1) * ks).to(dtype).unsqueeze(2)
            )
            if query.shape[0] <= 1028:
                native = partial(
                    flash_mla_with_kvcache,
                    native_q,
                    native_k,
                    None,
                    None,
                    dv,
                    *get_mla_metadata(),
                    indices=ids,
                    softmax_scale=scale,
                    attn_sink=sink,
                    topk_length=lengths,
                )
                ref, ref_lse = native()
            else:
                native = partial(
                    flash_mla_sparse_fwd,
                    native_q[:, 0],
                    native_k.view(-1, 1, dim),
                    ids,
                    scale,
                    dv,
                    sink,
                    lengths,
                )
                ref, _, ref_lse = native()
                ref, ref_lse = ref[:, None], ref_lse[:, :, None]
            actual = self.gems_op(*inputs)
            check_sparse_decode_sample(
                native_q, native_k, ids, scale, sink, lengths, actual, (ref, ref_lse)
            )
            yield (native,), inputs, shape


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(
    not (HAS_FP8_MLA or PPU_SPARSE_SUPPORTED), reason="Requires a sparse W8A8 backend"
)
def test_flash_mla_sparse_fwd_w8a8():
    benchmark = PPUSparseBenchmark if PPU_SPARSE_SUPPORTED else FlashmlaSparseBenchmark
    benchmark().run()
