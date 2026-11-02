# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Framework benchmark with no-Graph, reset-outside-event stateful timing."""

import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import fused_recurrent_gated_delta_rule_packed_decode as workload
from benchmark import base, consts
from benchmark.conftest import Config
from flaggems_vllm.runtime import torch_device_fn

OP = "fused_recurrent_gated_delta_rule_packed_decode"


class PackedDecodeBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
    DEFAULT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, entry, warmup=5, repetitions=25):
        self.entry = entry
        self.warmup = warmup
        self.repetitions = repetitions
        self._latencies = None
        self.native_reason = None
        super().__init__(
            op_name=OP,
            torch_op=None,
            gems_op=getattr(
                flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP
            ),
            is_inplace=True,
        )

    def init_user_config(self):
        if Config.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "Packed decode uses --mode kernel with no-Graph device events."
            )
        try:
            from vllm.model_executor.layers.fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode as native,
            )
        except ImportError as exc:
            self.torch_op = None
            self.native_reason = f"Native import unavailable: {exc}"
            self.metrics = ["latency"]
        else:
            self.torch_op = native
            self.native_reason = None
            self.metrics = self.DEFAULT_METRICS
        super().init_user_config()

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self._shape = shape
            self._dtype = dtype
            self._latencies = None
            yield (
                self.make_inputs(
                    shape,
                    dtype,
                    torch.float32,
                    self.device,
                    strided=True,
                    invalid=False,
                ),
            )

    def record_shapes(self, *args, **kwargs):
        return {
            "entry": self.entry,
            "normalize": True,
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_latency(self, op, *args, **kwargs):
        # Base.run requests native and Gems separately. Measure the pair once
        # per input so they still use alternating order and identical resets.
        # Keep the established fixed-count protocol (5 warmups / 25 samples);
        # the generic --warmup/--iter duration budgets do not apply here.
        if self._latencies is None:
            self._latencies = self._measure_pair(kwargs)
        return self._latencies["native" if op is self.torch_op else "gems"]

    def _measure_pair(self, data):
        state0 = data["initial_state"].clone()
        expected = self.reference(data, True)
        functions = {"gems": self.gems_op}
        if self.torch_op is not None:
            functions["native"] = self.torch_op
        for fn in functions.values():
            data["initial_state"].copy_(state0)
            fn(**data, use_qk_l2norm_in_kernel=True)
            self.assert_result(data, expected, self._dtype)
        for fn in functions.values():
            for _ in range(self.warmup):
                data["initial_state"].copy_(state0)
                fn(**data, use_qk_l2norm_in_kernel=True)
        torch_device_fn.synchronize()
        samples = {name: [] for name in functions}
        for i in range(self.repetitions):
            names = list(functions) if i % 2 == 0 else list(reversed(functions))
            for name in names:
                data["initial_state"].copy_(state0)
                torch_device_fn.synchronize()
                start = torch_device_fn.Event(enable_timing=True)
                end = torch_device_fn.Event(enable_timing=True)
                start.record()
                functions[name](**data, use_qk_l2norm_in_kernel=True)
                end.record()
                torch_device_fn.synchronize()
                samples[name].append(start.elapsed_time(end))
        self.assert_result(data, expected, self._dtype)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        self._record_samples(data, samples, medians)
        return medians

    def _record_samples(self, data, samples, medians):
        # Optional raw samples supplement the framework's --record/--output.
        # Ordinary benchmark runs do not require a custom environment variable.
        destination = os.environ.get("PACKED_DECODE_RESULTS")
        if not destination:
            return
        config = {
            k: (
                {
                    "shape": list(v.shape),
                    "stride": list(v.stride()),
                    "dtype": str(v.dtype),
                }
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in data.items()
        }
        row = dict(
            op=OP,
            entry=self.entry,
            backend=flaggems_vllm.vendor_name,
            dtype=str(self._dtype),
            shape=self._shape,
            config=config,
            normalize=True,
            accuracy="PASS",
            performance="PASS",
            graph=False,
            unit="ms",
            protocol="device-event-reset-outside-alternating-median-v1",
            warmup=self.warmup,
            repetitions=self.repetitions,
            samples=samples,
            median_ms=medians,
            native_reason=self.native_reason,
            candidate_module=self.gems_op.__module__,
        )
        with Path(destination).open("a") as f:
            f.write(json.dumps(row) + "\n")

    make_inputs = staticmethod(workload.make_inputs)

    @staticmethod
    def reference(inputs, normalize=False):
        """Independent FP32 CPU math, including the old beta rounding boundary."""
        mixed = inputs["mixed_qkv"].detach().cpu().float()
        state = inputs["initial_state"].detach().cpu().clone()
        B = mixed.shape[0]
        HV, V, K = state.shape[1:]
        H = (mixed.shape[1] - HV * V) // (2 * K)
        q = mixed[:, : H * K].reshape(B, H, K).repeat_interleave(HV // H, 1)
        k = mixed[:, H * K : 2 * H * K].reshape(B, H, K).repeat_interleave(HV // H, 1)
        v = mixed[:, 2 * H * K :].reshape(B, HV, V)
        if normalize:
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * inputs["scale"]
        a = inputs["a"].detach().cpu().float()
        bias = inputs["dt_bias"].detach().cpu().float()
        x = a + bias
        sp = torch.where(x <= 20, torch.log1p(torch.exp(x.clamp(max=20))), x)
        decay = torch.exp(-torch.exp(inputs["A_log"].detach().cpu().float()) * sp)
        beta = torch.sigmoid(inputs["b"].detach().cpu().float())
        beta = beta.to(inputs["b"].dtype).float()
        out = torch.zeros((B, 1, HV, V), dtype=inputs["out"].dtype)
        for n, slot in enumerate(inputs["ssm_state_indices"].cpu().tolist()):
            if slot <= 0:
                continue
            h = state[slot].float() * decay[n, :, None, None]
            residual = (v[n] - (h * k[n, :, None, :]).sum(-1)) * beta[n, :, None]
            h = h + residual[:, :, None] * k[n, :, None, :]
            out[n, 0] = (h * q[n, :, None, :]).sum(-1).to(out.dtype)
            state[slot] = h.to(state.dtype)
        return out, state

    @staticmethod
    def assert_result(inputs, expected, dtype):
        atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 1e-2)
        torch.testing.assert_close(
            inputs["out"].cpu(), expected[0], atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            inputs["initial_state"].cpu(), expected[1], atol=atol, rtol=rtol
        )


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize("entry", ["generic", "top"])
def test_packed_decode_performance(entry):
    PackedDecodeBenchmark(entry).run()
