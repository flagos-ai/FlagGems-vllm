# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path

import torch
import yaml

import flaggems_vllm
from benchmark.base import Benchmark
from benchmark.conftest import Config
from benchmark.consts import BenchLevel
from tests.accuracy_utils import gems_assert_close
from tests.test_gdn_w8a16 import (
    build_int8_gdn_decode_inputs,
    build_int8_gdn_prefill_inputs,
)


def native_gdn(reference, args):
    return reference[1]()


def int8_gdn(reference, args):
    kind = reference[0]
    if kind == "prefill":
        return flaggems_vllm.chunk_gated_delta_rule_w8a16(*args)
    else:
        return flaggems_vllm.gdn_w8a16(*args)


class HygonGdnW8A16Benchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
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
        default = Path(__file__).with_name("core_shapes.yaml").resolve()
        if Path(shape_file_path).resolve() != default:
            super().set_shapes(shape_file_path)
            return
        # The shared operator entry also carries NVIDIA's FP8-state schema.
        with open(shape_file_path) as file:
            config = yaml.safe_load(file)[self.op_name]["hygon"]
        self.shapes = [tuple(shape) for shape in config["shapes"]]
        self.shape_desc = config["shape_desc"]
        if Config.bench_level == BenchLevel.CORE:
            self.shapes = [
                shape for shape in self.shapes if shape in self.DEFAULT_SHAPES
            ]

    def get_input_iter(self, dtype):
        from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
        from vllm.model_executor.layers.fla.ops.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
            fused_sigmoid_gating_delta_rule_update,
        )

        for shape in self.shapes:
            torch.manual_seed(42)
            native_state = None
            if len(shape) == 7:
                _, data, args = build_int8_gdn_prefill_inputs(shape, dtype)
                # vLLM 0.24.0 prefill and decode both use V-major state.
                # Candidate de/quantization remains inside the timed call.
                native_call = partial(
                    chunk_gated_delta_rule,
                    q=data["q"],
                    k=data["k"],
                    v=data["v"],
                    g=data["g"],
                    beta=data["beta"],
                    initial_state=data["initial_state"],
                    output_final_state=True,
                    cu_seqlens=args[7],
                    chunk_indices=args[8],
                    chunk_offsets=args[9],
                    use_qk_l2norm_in_kernel=True,
                )
                kind = "prefill"
            else:
                separate = shape[0] == "update"
                if separate:
                    (
                        _,
                        batch,
                        tokens,
                        query_heads,
                        reduction,
                        heads,
                        columns,
                        sequences,
                    ) = shape
                    if batch != 1 or sequences != tokens:
                        raise ValueError("update shapes require one token per sequence")
                    packed = 2 * query_heads * reduction + heads * columns
                else:
                    tokens, packed, heads, columns = shape
                    reduction = 128
                    query_heads = (packed - heads * columns) // (2 * reduction)
                _, data, args = build_int8_gdn_decode_inputs(
                    (tokens, packed, heads, columns), dtype, reduction
                )
                mixed, a, b, a_log, bias, scale, state, out, indices, normalize = data
                native_state = state
                if separate:
                    values = mixed.split(
                        (
                            query_heads * reduction,
                            query_heads * reduction,
                            heads * columns,
                        ),
                        -1,
                    )
                    qkv = tuple(
                        value.view(1, tokens, width, feature).contiguous()
                        for value, width, feature in zip(
                            values,
                            (query_heads, query_heads, heads),
                            (reduction, reduction, columns),
                        )
                    )
                    cu = torch.arange(
                        tokens + 1, device=mixed.device, dtype=torch.int32
                    )
                    native_call = partial(
                        fused_sigmoid_gating_delta_rule_update,
                        A_log=a_log,
                        a=a,
                        b=b,
                        dt_bias=bias,
                        q=qkv[0],
                        k=qkv[1],
                        v=qkv[2],
                        scale=scale,
                        initial_state=state,
                        inplace_final_state=True,
                        cu_seqlens=cu,
                        ssm_state_indices=indices,
                        use_qk_l2norm_in_kernel=normalize,
                    )
                    args = (qkv, *args[1:])
                    kind = "update"
                else:
                    native_call = partial(
                        fused_recurrent_gated_delta_rule_packed_decode, *data
                    )
                    kind = "decode"

            reference = (kind, native_call)
            quantized_position, scale_position = (5, 6) if kind == "prefill" else (6, 7)
            saved_state = args[quantized_position].clone()
            saved_scale = args[scale_position].clone()
            saved_native_state = None if native_state is None else native_state.clone()
            expected, expected_state = native_gdn(reference, args)
            actual, state = int8_gdn(reference, args)
            gems_assert_close(actual.reshape_as(expected), expected, dtype, atol=2e-3)
            # Budget half an INT8 step per channel, plus the same floating
            # arithmetic tolerance used for the output/native comparison.
            restored = state.float() * args[scale_position][..., None]
            state_tolerance = 0.501 * args[scale_position][..., None] + 2e-3
            gems_assert_close(
                (restored - expected_state) / state_tolerance,
                torch.zeros_like(expected_state),
                torch.float32,
                atol=1.0,
            )
            args[quantized_position].copy_(saved_state)
            args[scale_position].copy_(saved_scale)
            if native_state is not None:
                native_state.copy_(saved_native_state)
            yield reference, args
