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

import flag_gems
import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark
from benchmark.conftest import Config
from benchmark.consts import BenchLevel
from tests.test_gdn_w8a16_int8 import (
    build_int8_gdn_decode_inputs,
    build_int8_gdn_prefill_inputs,
)


class Int8StateGdnBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [
        (1, 8192, 32, 128),
        (3, 8192, 32, 128),
        ("update", 1, 3, 16, 128, 32, 128, 3),
        (1, 17, 16, 128, 32, 128, 1),
        (1, 1826, 16, 128, 32, 128, 2),
    ]
    DEFAULT_SHAPE_DESC = (
        "decode: T,packed,HV,V; prefill: B,T,H,K,HV,V,N; update: update,B,T,H,K,HV,V,N"
    )

    def set_shapes(self, shape_file_path: str):
        super().set_shapes(shape_file_path)
        if (
            Config.bench_level == BenchLevel.CORE
            and Path(shape_file_path).resolve()
            == Path(__file__).with_name("core_shapes.yaml").resolve()
        ):
            # Keep the default core run small; explicit shape files run in full.
            self.shapes = [
                shape for shape in self.shapes if shape in self.DEFAULT_SHAPES
            ]

    def get_input_iter(self, dtype):
        # Resolve the compiled backends selected by vLLM-SAIL 0.31.0. Importing
        # upstream FLA functions alone would select its Triton implementation.
        from vllm_sail.attention.pla_decode import (
            get_sail_cuda_pla_k_last,
            get_sail_cuda_pla_k_last_packed,
        )
        from vllm_sail.attention.pla_prefill import get_sail_cuda_pla_prefill_fwd

        packed_decode = get_sail_cuda_pla_k_last_packed()
        update = get_sail_cuda_pla_k_last()
        prefill = get_sail_cuda_pla_prefill_fwd()
        if packed_decode is None or update is None or prefill is None:
            raise RuntimeError(
                "GDN benchmarks require the vLLM-SAIL PLA native backends"
            )

        for shape in self.shapes:
            torch.manual_seed(42)
            if len(shape) == 7:
                _, reference, args = build_int8_gdn_prefill_inputs(shape, dtype)
                yield ("prefill", prefill, reference), args
                continue

            separate_qkv = shape[0] == "update"
            if separate_qkv:
                _, batch, tokens, query_heads, reduction, heads, columns, sequences = (
                    shape
                )
                if batch != 1 or sequences != tokens:
                    raise ValueError(
                        "update shapes require B=1 and one token per sequence"
                    )
                packed = 2 * query_heads * reduction + heads * columns
            else:
                tokens, packed, heads, columns = shape
                reduction = 128
            _, reference, args = build_int8_gdn_decode_inputs(
                (tokens, packed, heads, columns), dtype, reduction
            )
            mixed, a, b, a_log, bias, scale, state, out, indices, normalize = reference
            native_bias = bias.to(dtype)
            native_indices = indices.to(torch.int32)
            cu_seqlens = torch.arange(
                tokens + 1, device=mixed.device, dtype=torch.int32
            )
            if separate_qkv:
                q, k, v = torch.split(
                    mixed,
                    (query_heads * reduction, query_heads * reduction, heads * columns),
                    dim=-1,
                )
                qkv = (
                    q.reshape(1, tokens, query_heads, reduction).contiguous(),
                    k.reshape(1, tokens, query_heads, reduction).contiguous(),
                    v.reshape(1, tokens, heads, columns).contiguous(),
                )
                native_call = partial(
                    update,
                    a_log,
                    a,
                    native_bias,
                    1.0,
                    20.0,
                    *qkv,
                    b,
                    state,
                    native_indices,
                    scale,
                    normalize,
                    cu_seqlens,
                    False,
                    False,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                )
                yield ("update", native_call, state), (qkv, *args[1:])
            else:
                native_call = partial(
                    packed_decode,
                    mixed,
                    a,
                    b,
                    a_log,
                    native_bias,
                    1.0,
                    20.0,
                    scale,
                    state,
                    native_indices,
                    out,
                    normalize,
                    cu_seqlens,
                    False,
                    False,
                )
                yield ("decode", native_call, state), args


def native_int8_state_gdn_baseline(reference, args):
    kind, native_call, data = reference
    if kind == "prefill":
        # Include normalization, beta conversion and the native state-layout
        # bridge in timing. Host sequence metadata is prepared for both routes.
        result = native_call(
            data["q"],
            data["k"],
            data["v"],
            data["g"],
            data["beta"].float(),
            initial_state=data["initial_state"].transpose(-1, -2).contiguous(),
            cu_seqlens=data["cu_seqlens"],
            output_final_state=True,
            normalize_qk=True,
            raw_cu_seqlens=data["raw_cu_seqlens"],
        )
        return result[2], result[4].transpose(-1, -2)
    else:
        return native_call(), data


def run_int8_state_gdn(reference, args):
    if reference[0] == "prefill":
        return flaggems_vllm.chunk_gated_delta_rule_w8a16_int8(*args)
    elif reference[0] == "update":
        tokens = args[0][0].shape[1]
        mixed = flag_gems.cat(
            tuple(tensor.view(tokens, -1) for tensor in args[0]), dim=1
        )
        return flaggems_vllm.gdn_w8a16_int8(mixed, *args[1:])
    else:
        return flaggems_vllm.gdn_w8a16_int8(*args)


@pytest.mark.gdn_w8a16_int8
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state GDN",
)
def test_perf_gdn_w8a16_int8():
    bench = Int8StateGdnBenchmark(
        op_name="gdn_w8a16_int8", torch_op=native_int8_state_gdn_baseline
    )
    bench.set_gems(run_int8_state_gdn)
    bench.run()
