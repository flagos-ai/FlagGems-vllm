# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Benchmark the public iHC post call with a qualified local native."""

import math
from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import triton_ihc_post as workloads
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result
from tests.test_triton_ihc_post import _assert_exact, _oracle


def _native_slot(*args, **kwargs):
    raise RuntimeError("native must be qualified before timing")


def _check_output(output, args, snapshots):
    x, residual, _ = args
    assert isinstance(output, torch.Tensor)
    assert output.shape == residual.shape
    assert output.dtype == x.dtype and output.device == x.device
    assert output.is_contiguous()
    if output.numel():
        assert all(
            output.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
            for value in args
        ), "output aliases input storage"
    # Bound CPU memory for the 4096-token cases.
    actual = output.detach().cpu()
    for start in range(0, x.shape[0], 32):
        rows = slice(start, start + 32)
        _assert_exact(actual[rows], _oracle(*(value[rows] for value in snapshots)))


def _check_inputs(args, snapshots):
    for index, (value, saved) in enumerate(zip(args, snapshots)):
        actual = value.detach().cpu().contiguous().view(torch.uint8)
        assert torch.equal(
            actual, saved.view(torch.uint8)
        ), f"input {index} was mutated"


def _check_native_arithmetic(native, x, channels):
    # These operands distinguish separate FP32 multiply/add from a fused FMA.
    value = 1025 if x.dtype == torch.float16 else 129
    snapshots = (
        torch.full((1, 1), value, dtype=x.dtype),
        torch.full((1, channels, 1), -value, dtype=x.dtype),
        torch.full((1, channels), 1 + 2**-23, dtype=torch.float32),
    )
    args = tuple(t.to(x.device, copy=True) for t in snapshots)
    _check_output(native(*args), args, snapshots)
    _check_inputs(args, snapshots)


class TritonIhcPostBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "tokens, channels, hidden, layout"
    DEFAULT_SHAPES = [
        (row["tokens"], row["channels"], row["hidden"], row["layout"])
        for row in workloads.grid()
    ]

    def set_shapes(self, shape_file_path):
        if shape_file_path and Path(shape_file_path).resolve() != Path(
            self.DEFAULT_SHAPE_FILES
        ):
            super().set_shapes(shape_file_path)
        else:
            self.shapes = self.DEFAULT_SHAPES
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def run(self):
        data = Config.base_data
        entry = (data or {}).get(self.op_name, {})
        # Dense and strided inputs share tensor shapes. Give the base lookup
        # one variant per full collector key so the layouts stay distinct.
        with pytest.MonkeyPatch.context() as patch:
            if Config.user_desired_metrics == ["latency"]:
                patch.setattr(Config, "base_data", None)
            elif "shapes" in entry and "variants" not in entry:
                variants = {
                    key: {"shapes": {key: rows}}
                    for key, rows in entry["shapes"].items()
                }
                patch.setattr(
                    Config, "base_data", {**data, self.op_name: {"variants": variants}}
                )
            super().run()

    def get_input_iter(self, dtype):
        self.native = None
        reason = "native timing disabled by --metrics"
        if "latency_base" in self.to_bench_metrics:
            try:
                self.native = workloads.native()
                reason = workloads.NATIVE_UNAVAILABLE_REASON
            except Exception as exc:
                reason = f"native resolution failed: {exc!r}"
            if self.native is self.gems_op:
                reason = "installed native resolves to Gems"
            elif not callable(self.native):
                reason = reason or "installed native unavailable"

        misses = []
        for n, hc, d, layout in self.shapes:
            binding = dict(tokens=n, channels=hc, hidden=d, layout=layout)
            args, kwargs = workloads.build_inputs(binding, dtype, self.device)
            config = workloads.config(binding, dtype)
            for value, name in zip(args, ("x", "residual", "post")):
                spec = config["inputs"][name]
                assert (
                    list(value.shape) == spec["shape"]
                    and str(value.dtype) == spec["dtype"]
                )
                assert (
                    list(value.stride()) == spec["stride"]
                    and value.storage_offset() == spec["storage_offset"]
                )
            self.base_variant = workloads.key_shape(binding)
            self.snapshots = tuple(value.detach().cpu().clone() for value in args)
            self.detail = dict(
                arguments=super().record_shapes(*args, **kwargs),
                shape_key=self.base_variant,
                config=config,
                native=dict(status="N/A", reason=reason),
            )
            yield (*args, kwargs)
            if self.detail["native"]["reason"]:
                misses.append(
                    dict(
                        shape_key=self.base_variant,
                        reason=self.detail["native"]["reason"],
                    )
                )
        if misses:
            update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(m["reason"] for m in misses)
                    ),
                    native_baseline_misses=misses,
                ),
            )

    def record_shapes(self, *args, **kwargs):
        return self.detail

    def _checked_latency(self, op, *args, **kwargs):
        _check_output(op(*args, **kwargs), args, self.snapshots)
        _check_inputs(args, self.snapshots)
        latency = super().get_latency(op, *args, **kwargs)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid latency for {self.base_variant}: {latency!r}")
        _check_inputs(args, self.snapshots)
        return latency

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            return self._checked_latency(op, *args, **kwargs)

        native = self.detail["native"]
        if native["reason"]:
            return None
        try:
            _check_native_arithmetic(self.native, args[0], args[1].shape[1])
            latency = self._checked_latency(self.native, *args, **kwargs)
        except Exception as exc:
            for value, saved in zip(args, self.snapshots):
                value.copy_(saved)
            native["reason"] = repr(exc)
            return None
        native["status"] = "PASS"
        return latency


@pytest.mark.triton_ihc_post
def test_triton_ihc_post():
    TritonIhcPostBenchmark(
        op_name=workloads.OP_NAME,
        torch_op=_native_slot,
        gems_op=flaggems_vllm.triton_ihc_post,
        dtypes=workloads.DTYPES,
    ).run()
