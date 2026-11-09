# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import statistics
from functools import partial
from pathlib import Path

import flag_gems
import pytest
import torch
import yaml

import flaggems_vllm
from benchmark.base import Benchmark
from benchmark.conftest import Config
from benchmark.consts import BenchLevel, BenchMode
from flaggems_vllm.ops.gdn_w8a16 import quantize_gdn_state_fp8
from flaggems_vllm.runtime import torch_device_fn
from tests.test_gdn_w8a16_int8 import (
    ascend_gdn_native_runtime,
    build_int8_gdn_decode_inputs,
    build_int8_gdn_prefill_inputs,
    prepare_ascend_gdn_decode_baseline,
    prepare_ascend_gdn_prefill_baseline,
)


def _vllm_bf16(q, k, v, g, beta, state_bf16, state_fp8, state_scale, cu, indices, t):
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=128**-0.5,
        initial_state=state_bf16,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def _fp8(q, k, v, g, beta, state_bf16, state_fp8, state_scale, cu, indices, t):
    return flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        state_fp8,
        state_scale,
        cu_seqlens=cu,
        ssm_state_indices=indices[:, 0],
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )


class GdnW8A16Benchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "N, T, H, HV, K, V"
    DEFAULT_SHAPES = [
        (n, t, 2, hv, 128, 128)
        for hv in (4, 6)
        for n, t in (
            [
                (n, 1)
                for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
            ]
            + [
                (1, 2),
                (1, 8),
                (1, 32),
                (4, 4),
                (4, 16),
                (16, 4),
                (1, 256),
                (1, 512),
                (2, 192),
                (4, 256),
                (8, 128),
                (16, 128),
                (32, 64),
                (64, 32),
                (128, 24),
            ]
        )
    ]

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, t, h, hv, key_dim, value_dim in self.shapes:
            if key_dim != 128 or value_dim != 128:
                raise ValueError("This benchmark covers K=V=128")
            torch.manual_seed(42)
            device = flaggems_vllm.device
            q = torch.randn(1, n * t, h, key_dim, device=device, dtype=cur_dtype)
            k = torch.randn_like(q)
            v = 0.125 * torch.randn(
                1, n * t, hv, value_dim, device=device, dtype=cur_dtype
            )
            g = torch.empty(1, n * t, hv, device=device, dtype=cur_dtype).uniform_(
                math.log(0.98), math.log(0.995)
            )
            beta = torch.rand(1, n * t, hv, device=device, dtype=cur_dtype)
            state = torch.zeros(
                n + 1, hv, key_dim, value_dim, device=device, dtype=cur_dtype
            )
            state_fp8, state_scale = quantize_gdn_state_fp8(state)
            cu = torch.arange(0, n * t + 1, t, device=device, dtype=torch.int32)
            indices = (
                torch.arange(1, n + 1, device=device, dtype=torch.int64)[:, None]
                .expand(n, t)
                .contiguous()
            )
            inputs = q, k, v, g, beta, state, state_fp8, state_scale, cu, indices, t
            baseline, _ = _vllm_bf16(*inputs)
            torch.cuda.synchronize()
            # A zero initial state gives a closed-form first-token output.
            query = q.reshape(n, t, h, key_dim)[:, 0].float()
            key = k.reshape(n, t, h, key_dim)[:, 0].float()
            query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
            key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            correlation = (query * key).sum(-1).repeat_interleave(hv // h, -1)
            expected = (
                correlation[..., None]
                * key_dim**-0.5
                * v.reshape(n, t, hv, value_dim)[:, 0].float()
                * beta.reshape(n, t, hv)[:, 0].float()[..., None]
            )
            first_output = baseline.reshape(n, t, hv, value_dim)[:, 0]
            torch.testing.assert_close(
                first_output.float(), expected, atol=2e-3, rtol=5e-2
            )
            state.zero_()
            yield inputs


HYGON_INT8_STATE = (
    "gdn_w8a16_int8" in vars(flaggems_vllm)
    and flaggems_vllm.gdn_w8a16_int8.__module__
    == "flaggems_vllm.runtime.backend._hygon.ops.gdn_w8a16"
)
ASCEND_INT8_STATE = (
    "gdn_w8a16_int8" in vars(flaggems_vllm)
    and flaggems_vllm.gdn_w8a16_int8.__module__
    == "flaggems_vllm.runtime.backend._ascend.ops.gdn_w8a16_int8"
)


class Int8StateGdnBenchmark(Benchmark):
    DEFAULT_DTYPES = (
        [torch.bfloat16, torch.float16]
        if HYGON_INT8_STATE or ASCEND_INT8_STATE
        else [torch.bfloat16]
    )
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
        # The shared operator ID also covers FP8, whose shape layout differs.
        is_default = (
            Path(shape_file_path).resolve()
            == Path(__file__).with_name("core_shapes.yaml").resolve()
        )
        if ASCEND_INT8_STATE and is_default:
            shape_source = (
                Path(flaggems_vllm.__file__).parent
                / "runtime/backend/_ascend/core_shapes.yaml"
            )
        else:
            shape_source = Path(shape_file_path)
        with open(shape_source) as shape_file:
            shape_config = yaml.safe_load(shape_file).get("gdn_w8a16_int8")
        if shape_config is None:
            super().set_shapes(shape_file_path)
        else:
            self.shapes = [
                tuple(shape)
                for shape in shape_config.get("shapes", self.DEFAULT_SHAPES)
            ]
            self.shape_desc = shape_config.get("shape_desc", self.DEFAULT_SHAPE_DESC)
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
        if ASCEND_INT8_STATE:
            yield from self.get_ascend_inputs(dtype)
            return
        if HYGON_INT8_STATE:
            from vllm.model_executor.layers.fla.ops.chunk import (
                chunk_gated_delta_rule as prefill,
            )
            from vllm.model_executor.layers.fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode as packed_decode,
            )
            from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
                fused_sigmoid_gating_delta_rule_update as update,
            )
        else:
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
                if HYGON_INT8_STATE:
                    native_call = partial(
                        update,
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
                        cu_seqlens=cu_seqlens,
                        ssm_state_indices=indices,
                        use_qk_l2norm_in_kernel=normalize,
                    )
                else:
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
                if HYGON_INT8_STATE:
                    native_call = partial(packed_decode, *reference)
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

    def get_ascend_inputs(self, dtype):
        # Register the custom OPP before input allocation creates an NPU context.
        ascend_gdn_native_runtime()
        for shape in self.shapes:
            torch.manual_seed(42)
            if len(shape) == 7:
                _, reference, args = build_int8_gdn_prefill_inputs(shape, dtype)
                native_call, buffers = prepare_ascend_gdn_prefill_baseline(reference)
                yield ("prefill", native_call, buffers), args
                continue
            if shape[0] == "update":
                _, batch, tokens, query_heads, reduction, heads, columns, sequences = (
                    shape
                )
                if batch != 1 or tokens != sequences:
                    raise ValueError("update requires one token per sequence")
                packed = 2 * query_heads * reduction + heads * columns
                kind = "update"
            else:
                tokens, packed, heads, columns = shape
                reduction = 128
                query_heads = (packed - heads * columns) // (2 * reduction)
                kind = "decode"
            _, reference, args = build_int8_gdn_decode_inputs(
                (tokens, packed, heads, columns),
                dtype,
                reduction,
            )
            if kind == "update":
                q, k, v = args[0].split(
                    (query_heads * reduction, query_heads * reduction, heads * columns),
                    dim=-1,
                )
                qkv = (
                    q.reshape(1, tokens, query_heads, reduction).contiguous(),
                    k.reshape(1, tokens, query_heads, reduction).contiguous(),
                    v.reshape(1, tokens, heads, columns).contiguous(),
                )
                args = (qkv, *args[1:])
                reference = (qkv, *reference[1:])
            native_call, buffers = prepare_ascend_gdn_decode_baseline(reference)
            yield (kind, native_call, buffers), args

    def get_latency(self, op, *args, **kwargs):
        if not ASCEND_INT8_STATE:
            return super().get_latency(op, *args, **kwargs)
        if Config.mode != BenchMode.KERNEL:
            raise ValueError("Ascend stateful GDN measurements require --mode kernel")
        reference, inputs = args
        if op is self.torch_op:
            buffers = reference[2]
        elif reference[0] == "prefill":
            buffers = inputs[5:7]
        else:
            buffers = inputs[6:9]
        saved = tuple(tensor.clone() for tensor in buffers)
        import tempfile

        import torch_npu
        from triton.backends.ascend.testing import KernelMonitor, _collect_prof_result

        samples = []
        # Match the NPU helper's bounded 5/30 sampling defaults.
        warmup, repetitions = min(Config.warm_up, 5), min(Config.repetition, 30)
        for index in range(warmup + repetitions):
            for tensor, original in zip(buffers, saved):
                tensor.copy_(original)
            torch_device_fn.synchronize()
            if index < warmup:
                op(*args, **kwargs)
                torch_device_fn.synchronize()
                continue
            # Match do_bench_npu's device-kernel sum, excluding state restoration.
            if KernelMonitor is not None:
                durations = []
                monitor = KernelMonitor()
                monitor.start(
                    lambda record: durations.append(record.end - record.start)
                )
                try:
                    op(*args, **kwargs)
                    torch_device_fn.synchronize()
                finally:
                    monitor.stop()
                if not durations:
                    raise RuntimeError("No NPU kernel records were captured")
                samples.append(sum(durations) / 1e6)
            else:
                with tempfile.TemporaryDirectory(prefix="gdn-native-profile-") as path:
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU],
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                            path
                        ),
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                            data_simplification=False,
                        ),
                    ):
                        op(*args, **kwargs)
                        torch_device_fn.synchronize()
                    samples.append(_collect_prof_result(path, [op], 0, 1))
        for tensor, original in zip(buffers, saved):
            tensor.copy_(original)
        if not samples or any(
            not math.isfinite(sample) or sample <= 0 for sample in samples
        ):
            raise RuntimeError("NPU profiling did not return valid device latencies")
        return statistics.median(samples)


def native_int8_state_gdn_baseline(reference, args):
    if ASCEND_INT8_STATE:
        return reference[1]()
    kind, native_call, data = reference
    if kind == "prefill":
        if HYGON_INT8_STATE:
            return native_call(
                q=data["q"],
                k=data["k"],
                v=data["v"],
                g=data["g"],
                beta=data["beta"],
                initial_state=data["initial_state"],
                cu_seqlens=data["cu_seqlens"],
                chunk_indices=args[8],
                chunk_offsets=args[9],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
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
    elif HYGON_INT8_STATE:
        return native_call()
    else:
        return native_call(), data


def run_int8_state_gdn(reference, args):
    if reference[0] == "prefill":
        return flaggems_vllm.chunk_gated_delta_rule_w8a16_int8(*args)
    elif reference[0] == "update" and not (HYGON_INT8_STATE or ASCEND_INT8_STATE):
        tokens = args[0][0].shape[1]
        mixed = flag_gems.cat(
            tuple(tensor.view(tokens, -1) for tensor in args[0]), dim=1
        )
        return flaggems_vllm.gdn_w8a16_int8(mixed, *args[1:])
    else:
        return flaggems_vllm.gdn_w8a16_int8(*args)


@pytest.mark.gdn_w8a16
def test_gdn_w8a16():
    if "gdn_w8a16_int8" in vars(flaggems_vllm):
        if ASCEND_INT8_STATE:
            print(
                "Native baseline: vLLM-Ascend BF16 activation / FP32 state; pure device kernel sum."
            )
        bench = Int8StateGdnBenchmark(
            op_name="gdn_w8a16", torch_op=native_int8_state_gdn_baseline
        )
        bench.base_variant = "int8"
        bench.set_gems(run_int8_state_gdn)
    else:
        if (
            flaggems_vllm.vendor_name != "nvidia"
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 9
        ):
            pytest.skip("FP8 GDN requires NVIDIA SM90 or newer")
        bench = GdnW8A16Benchmark(op_name="gdn_w8a16", torch_op=_vllm_bf16)
        bench.set_gems(_fp8)
    bench.run()
