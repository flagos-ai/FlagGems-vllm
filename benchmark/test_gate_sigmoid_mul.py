# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard kernel benchmark with full-output checks and explicit native N/A."""

import json
import logging
import os
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import gate_sigmoid_mul as workloads

from . import base, conftest, consts

LOGGER = logging.getLogger(__name__)
TOLERANCES = {
    torch.float16: (2e-3, 2e-5),
    torch.bfloat16: (1.6e-2, 2e-4),
    torch.float32: (2e-6, 2e-7),
}


def _record(role, binding, dtype, latency_ms, status, reason=None, active_counts=None):
    config = conftest.Config
    ascend = flaggems_vllm.vendor_name == "ascend"
    row = {
        "op_name": workloads.OP_NAME,
        "role": role,
        "status": status,
        "reason": reason,
        "shape_key": workloads.key_shape(binding),
        "dtype": str(dtype),
        "case_id": f"{workloads.key_shape(binding)}::{dtype}",
        "config": workloads.config(binding, dtype),
        "backend": flaggems_vllm.vendor_name,
        "unit": "ms",
        "native_provenance": workloads.NATIVE_ARTIFACT,
        "graph": False,
        "mode": "kernel",
        "timing_protocol": {
            "framework": "benchmark.base.Benchmark.get_latency",
            "timer": (
                "triton.backends.ascend.testing.do_bench_npu"
                if ascend
                else "triton.testing.do_bench"
            ),
            "measurement": "NPU kernel timer" if ascend else "device-event interval",
            "requested_warmup": config.warm_up,
            "requested_repetition": config.repetition,
            "effective_warmup_ms": None if ascend else config.warm_up,
            "effective_repetition_ms": None if ascend else config.repetition,
            "reduction": "backend default" if ascend else "median",
            "note": (
                "Standard Ascend path uses do_bench_npu defaults; Config durations "
                "are not forwarded."
                if ascend
                else "Standard kernel path forwards Config durations."
            ),
        },
        "latency_ms": latency_ms,
        "accuracy": "PASS" if status == "PASS" else "N/A",
        "accuracy_scope": (
            "full matrix output against CPU FP64 stable sigmoid and multiply, "
            "rounded once; separate pytest unit evidence"
        ),
        "accuracy_checks": "shape,dtype,device,independent storage,no input mutation",
        "level": config.bench_level.value,
        "expected_comprehensive_candidate_rows": 192,
        "active_counts": active_counts,
    }
    target = os.environ.get("GATE_RESULTS")
    if target:
        with Path(target).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
    LOGGER.info("%s", json.dumps(row))


def _check_full_output(output, args, snapshots):
    """One output transfer; 32-row CPU FP64 chunks bound temporary memory."""
    a, gate = args
    assert isinstance(output, torch.Tensor), "output must be a Tensor"
    assert output.shape == a.shape, "output shape mismatch"
    output_dtype = torch.result_type(a, gate)
    assert output.dtype == output_dtype, "output dtype mismatch"
    assert output.device == a.device, "output device mismatch"
    output_storage = output.untyped_storage().data_ptr()
    assert all(
        output_storage != value.untyped_storage().data_ptr() for value in args
    ), "output must have storage independent of both inputs"
    output_cpu = output.detach().cpu()
    rtol, atol = TOLERANCES[output_dtype]
    for start in range(0, a.shape[0], 32):
        stop = start + 32
        a64, g64 = (value[start:stop].double() for value in snapshots)
        exp_abs = torch.exp(-torch.abs(g64))
        sigmoid = torch.where(g64 >= 0, 1 / (1 + exp_abs), exp_abs / (1 + exp_abs))
        expected = (a64 * sigmoid).to(output_dtype)
        torch.testing.assert_close(
            output_cpu[start:stop], expected, rtol=rtol, atol=atol
        )


def _changed_inputs(args, snapshots, restore=False):
    changed = []
    for index, (value, saved) in enumerate(zip(args, snapshots)):
        if not torch.equal(value.detach().cpu(), saved):
            changed.append(index)
            if restore:
                value.copy_(saved)
    return changed


class GateSigmoidMulBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "tokens, width, gate_layout"

    def set_more_shapes(self):
        # Standard set_shapes merges/deduplicates these with the six core triples.
        return [
            (binding["tokens"], binding["width"], binding["layout"])
            for binding in workloads.grid()
        ]

    def record_shapes(self, *args, **kwargs):
        # Tensor.size alone loses the split-gate stride/offset distinction.
        return workloads.key_shape(self._gate_binding)

    def get_input_iter(self, dtype):
        native = workloads.native()
        unavailable_reason = workloads.NATIVE_UNAVAILABLE_REASON
        if native is self.gems_op:
            native = None
            unavailable_reason = "resolved native aliases the candidate callable"
        active_counts = {
            "bindings": len(self.shapes),
            "dtypes": len(self.to_bench_dtypes),
            "candidate_rows": len(self.shapes) * len(self.to_bench_dtypes),
        }
        for tokens, width, layout in self.shapes:
            binding = {"tokens": tokens, "width": width, "layout": layout}
            self._gate_binding, self._gate_dtype = binding, dtype
            self._gate_counts = active_counts
            self.torch_op = native
            args, _ = workloads.build_inputs(binding, dtype, self.device)
            snapshots = tuple(value.detach().cpu().clone() for value in args)
            native_error = unavailable_reason
            if native is not None:
                native_error = None
                native_out = None
                try:
                    native_out = native(*args)
                    _check_full_output(native_out, args, snapshots)
                except Exception as exc:
                    native_error = f"{type(exc).__name__}: {exc}"
                finally:
                    del native_out
                changed = _changed_inputs(args, snapshots, restore=True)
                if changed:
                    mutation_error = f"native mutated positional inputs {changed}"
                    native_error = (
                        f"{native_error}; {mutation_error}"
                        if native_error
                        else mutation_error
                    )
            if native is None or native_error is not None:
                self.torch_op = None
                _record(
                    "native",
                    binding,
                    dtype,
                    None,
                    "N/A",
                    reason=native_error or "exact native callable unavailable",
                    active_counts=active_counts,
                )
            # Candidate failures remain hard failures, outside native exception scope.
            candidate_out = self.gems_op(*args)
            _check_full_output(candidate_out, args, snapshots)
            changed = _changed_inputs(args, snapshots)
            assert not changed, f"candidate mutated positional inputs {changed}"
            self._gate_snapshots = snapshots
            del candidate_out
            yield args

    def get_latency(self, op, *args, **kwargs):
        candidate = op is self.gems_op
        try:
            latency = super().get_latency(op, *args, **kwargs)
            changed = _changed_inputs(args, self._gate_snapshots, restore=not candidate)
            if changed:
                raise AssertionError(
                    f"timed callable mutated positional inputs {changed}"
                )
        except Exception as exc:
            if candidate:
                raise
            # Base.run fails the whole iterator on an exception; invalid native
            # rows must instead retain N/A while later candidate cases continue.
            _changed_inputs(args, self._gate_snapshots, restore=True)
            self.torch_op = None
            _record(
                "native",
                self._gate_binding,
                self._gate_dtype,
                None,
                "N/A",
                reason=f"native timing invalid: {type(exc).__name__}: {exc}",
                active_counts=self._gate_counts,
            )
            return None
        _record(
            "candidate" if candidate else "native",
            self._gate_binding,
            self._gate_dtype,
            latency,
            "PASS",
            active_counts=self._gate_counts,
        )
        return latency


@pytest.mark.gate_sigmoid_mul
def test_gate_sigmoid_mul():
    if conftest.Config.mode != consts.BenchMode.KERNEL:
        raise ValueError("gate benchmark requires --mode kernel (no Graph)")
    bench = GateSigmoidMulBenchmark(
        op_name=workloads.OP_NAME,
        gems_op=flaggems_vllm.gate_sigmoid_mul,
        torch_op=workloads.native(),
        dtypes=workloads.DTYPES,
    )
    bench.run()
