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
from benchmark.conftest import Config, update_result
from flaggems_vllm.runtime import torch_device_fn

OP = "fused_recurrent_gated_delta_rule_packed_decode"


def _native_slot(*args, **kwargs):
    # Routing token for Base.run, never a callable performance baseline.
    raise RuntimeError("packed native slot must be handled by get_latency")


class PackedDecodeBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
    DEFAULT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]

    def __init__(self, warmup=5, repetitions=25):
        self.entry = None
        self.warmup = warmup
        self.repetitions = repetitions
        self._latencies = None
        self.native_op = None
        self.native_reason = None
        super().__init__(
            op_name=OP,
            torch_op=_native_slot,
            gems_op=getattr(flaggems_vllm, OP),
            is_inplace=True,
        )

    def init_user_config(self):
        if Config.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "Packed decode uses --mode kernel with no-Graph device events."
            )
        super().init_user_config()
        self.native_op = None
        self.native_reason = "Native timing disabled by --metrics"
        if "latency_base" not in self.to_bench_metrics:
            self._native_import_reason = self.native_reason
            return
        try:
            from vllm.model_executor.layers.fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode as native,
            )
        except ImportError as exc:
            self.native_reason = f"Native import unavailable: {exc}"
        else:
            if not callable(native):
                self.native_reason = "Native entry is not callable"
            elif native is getattr(flaggems_vllm.ops, OP) or native is getattr(
                flaggems_vllm, OP
            ):
                self.native_reason = "Native entry resolves to the Gems candidate"
            else:
                self.native_op = native
                self.native_reason = None
        self._native_import_reason = self.native_reason

    def run(self):
        # Explicit latency-only mode must not resolve or compare either baseline.
        # Restore Config even if a candidate/input/oracle raises.
        with pytest.MonkeyPatch.context() as patch:
            if Config.user_desired_metrics == ["latency"]:
                patch.setattr(Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        # One framework result block per dtype, containing both public entries.
        # Separate blocks with the same dtype overwrite each other downstream.
        misses = []
        for entry in ("generic", "top"):
            self.entry = entry
            self.gems_op = getattr(
                flaggems_vllm.ops if entry == "generic" else flaggems_vllm, OP
            )
            for shape in self.shapes:
                self._shape = shape
                self._dtype = dtype
                self._latencies = None
                self.native_reason = self._native_import_reason
                self._native_result = dict(
                    accuracy="NOT_RUN", performance="N/A", reason=self.native_reason
                )
                data = self.make_inputs(
                    shape,
                    dtype,
                    torch.float32,
                    self.device,
                    strided=True,
                    invalid=False,
                )
                yield (data,)
                if self.native_reason:
                    misses.append(
                        {
                            "shape": self.record_shapes(**data),
                            "reason": self.native_reason,
                        }
                    )
        if misses:
            # The existing consumer uses this metadata to retain null scores
            # rather than summing N/A as zero (or raising on None).
            update_result(
                self.op_name,
                {
                    "dtype": str(dtype),
                    "native_baseline_skip_reason": "; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    "native_baseline_misses": misses,
                },
            )

    def record_shapes(self, *args, **kwargs):
        return {
            "entry": self.entry,
            "normalize": True,
            "native": self._native_result,
            "inputs": super().record_shapes(*args, **kwargs),
        }

    def get_latency(self, op, *args, **kwargs):
        # Base.run requests native and Gems separately. Measure the pair once
        # per input so they still use alternating order and identical resets.
        # Keep the established fixed-count protocol (5 warmups / 25 samples);
        # the generic --warmup/--iter duration budgets do not apply here.
        if self._latencies is None:
            self._latencies = self._measure_pair(kwargs)
        return self._latencies.get("native" if op is self.torch_op else "gems")

    def _measure_pair(self, data):
        state0 = data["initial_state"].clone()
        data["out"].fill_(float("nan"))
        out0 = data["out"].clone()

        def restore():
            data["initial_state"].copy_(state0)
            data["out"].copy_(out0)

        expected = self.reference(data, True)
        functions = {"gems": self.gems_op}
        if self.native_op is not None:
            functions["native"] = self.native_op
        for name, fn in list(functions.items()):
            restore()
            try:
                fn(**data, use_qk_l2norm_in_kernel=True)
                self.assert_result(data, expected, self._dtype)
            except Exception as exc:
                if name == "gems":
                    raise
                self._native_result.update(accuracy="FAIL")
                self._exclude_native(exc)
                del functions[name]
            else:
                if name == "native":
                    self._native_result.update(accuracy="PASS")
        samples = {name: [] for name in functions}
        for name, fn in list(functions.items()):
            try:
                for _ in range(self.warmup):
                    restore()
                    fn(**data, use_qk_l2norm_in_kernel=True)
            except Exception as exc:
                if name == "gems":
                    raise
                self._exclude_native(exc)
                del functions[name]
                del samples[name]
        torch_device_fn.synchronize()
        for i in range(self.repetitions):
            names = list(functions) if i % 2 == 0 else list(reversed(functions))
            for name in names:
                restore()
                torch_device_fn.synchronize()
                try:
                    start = torch_device_fn.Event(enable_timing=True)
                    end = torch_device_fn.Event(enable_timing=True)
                    start.record()
                    functions[name](**data, use_qk_l2norm_in_kernel=True)
                    end.record()
                    torch_device_fn.synchronize()
                    samples[name].append(start.elapsed_time(end))
                except Exception as exc:
                    if name == "gems":
                        raise
                    self._exclude_native(exc)
                    del functions[name]
                    del samples[name]
        # Recheck each implementation separately; native must not invalidate a
        # correct Gems row if its state update becomes unstable after warmup.
        for name, fn in functions.items():
            restore()
            try:
                fn(**data, use_qk_l2norm_in_kernel=True)
                self.assert_result(data, expected, self._dtype)
            except Exception as exc:
                if name == "gems":
                    raise
                self._native_result.update(accuracy="FAIL")
                self._exclude_native(exc)
                del samples[name]
        if "native" in samples:
            self._native_result.update(performance="PASS", reason=None)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        self._record_samples(data, samples, medians)
        return medians

    def _exclude_native(self, exc):
        self.native_reason = f"{type(exc).__name__}: {exc}"
        self._native_result.update(performance="N/A", reason=self.native_reason)

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
            native=dict(self._native_result),
            native_speedup=(
                medians["native"] / medians["gems"] if "native" in medians else None
            ),
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
def test_packed_decode_performance():
    PackedDecodeBenchmark().run()
