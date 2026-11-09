# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard no-Graph benchmark of the complete iHC post public callable."""

from pathlib import Path

import pytest
import torch

import flaggems_vllm
from baseline import triton_ihc_post as workloads

from . import base, conftest, consts


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
    actual = output.detach().cpu()
    for start in range(0, x.shape[0], 32):
        stop = start + 32
        x32, residual32, post32 = (value[start:stop].float() for value in snapshots)
        product = torch.mul(post32.unsqueeze(-1), x32.unsqueeze(1))
        expected = torch.add(product, residual32).to(x.dtype)
        part = actual[start:stop]
        torch.testing.assert_close(part, expected, rtol=0, atol=0, equal_nan=True)
        zeros = expected == 0
        assert torch.equal(torch.signbit(part[zeros]), torch.signbit(expected[zeros]))


def _changed_inputs(args, snapshots, restore=False):
    changed = []
    for index, (value, saved) in enumerate(zip(args, snapshots)):
        actual = value.detach().cpu().contiguous().view(torch.uint8)
        if not torch.equal(actual, saved.view(torch.uint8)):
            changed.append(index)
            if restore:
                value.copy_(saved)
    return changed


def _check_native_arithmetic(native, dtype, channels, device):
    """Qualify separate FP32 mul/add before accepting any native timing row."""
    value = 1025 if dtype == torch.float16 else 129
    snapshots = (
        torch.full((1, 1), value, dtype=dtype),
        torch.full((1, channels, 1), -value, dtype=dtype),
        torch.full((1, channels), 1 + 2**-23, dtype=torch.float32),
    )
    args = tuple(value.to(device) for value in snapshots)
    try:
        output = native(*args)
        _check_output(output, args, snapshots)
        changed = _changed_inputs(args, snapshots)
        if changed:
            raise AssertionError(f"native mutated arithmetic fixture inputs {changed}")
    except Exception as exc:
        raise AssertionError(
            "native arithmetic protocol requires FP32 multiply then FP32 add "
            f"without FMA: {type(exc).__name__}: {exc}"
        ) from exc


class TritonIhcPostBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPE_DESC = "tokens, channels, hidden, layout"
    DEFAULT_SHAPES = [
        (row["tokens"], row["channels"], row["hidden"], row["layout"])
        for row in workloads.grid()
    ]

    def set_shapes(self, shape_file_path=None):
        # The repository's generic Benchmark YAML must not replace this grid.
        # Explicit external shape files retain the normal base-class behavior.
        default = Path(__file__).with_name("core_shapes.yaml")
        if (
            shape_file_path is not None
            and Path(shape_file_path).resolve() == default.resolve()
        ):
            self.shapes = list(self.DEFAULT_SHAPES)
            self.shape_desc = self.DEFAULT_SHAPE_DESC
            return
        super().set_shapes(shape_file_path)

    def init_user_config(self):
        super().init_user_config()
        # OPERATOR uses synchronized wall-clock loops in the checked-in base;
        # vendor-specific kernel timers may themselves capture a device Graph.
        if conftest.Config.mode != consts.BenchMode.OPERATOR:
            raise ValueError("iHC post requires --mode operator (no Graph)")
        if conftest.Config.base_data is not None:
            raise ValueError(
                "iHC post --base-data is unsupported: the common lookup cannot "
                "distinguish layouts or retain missing-native N/A; use native "
                "comparison and the separate baseline collector"
            )

    def record_shapes(self, *args, **kwargs):
        # Preserve normal tensor shapes while retaining layout and per-row N/A.
        return {
            "arguments": super().record_shapes(*args, **kwargs),
            "shape_key": workloads.key_shape(self._binding),
            "config": workloads.config(self._binding, self._dtype),
            "native": self._native_status,
            "timing": {
                "scope": "complete public callable, allocation and contiguous copies",
                "framework": "Benchmark.get_latency / operator",
                "graph": False,
                "unit": "ms",
                "statistic": "synchronized wall-clock mean",
                "warmup_budget_ms": conftest.Config.warm_up,
                "repetition_budget_ms": conftest.Config.repetition,
            },
            "active_rows": len(self.shapes) * len(self.to_bench_dtypes),
        }

    def get_input_iter(self, dtype):
        native = workloads.native()
        if native is self.gems_op:
            raise RuntimeError("native resolution aliases the candidate callable")
        for n, hc, d, layout in self.shapes:
            binding = {"tokens": n, "channels": hc, "hidden": d, "layout": layout}
            args, kwargs = workloads.build_inputs(binding, dtype, self.device)
            snapshots = tuple(value.detach().cpu().clone() for value in args)
            self._binding, self._dtype = binding, dtype
            self._snapshots = snapshots
            self.torch_op = native
            self._native_status = {
                "status": "N/A" if native is None else "VALIDATING",
                "reason": workloads.NATIVE_UNAVAILABLE_REASON,
            }
            if native is not None:
                try:
                    _check_native_arithmetic(native, dtype, hc, self.device)
                    output = native(*args, **kwargs)
                    _check_output(output, args, snapshots)
                    del output
                    changed = _changed_inputs(args, snapshots, restore=True)
                    if changed:
                        raise AssertionError(
                            f"native mutated positional inputs {changed}"
                        )
                except Exception as exc:
                    _changed_inputs(args, snapshots, restore=True)
                    self.torch_op = None
                    self._native_status.update(
                        status="N/A",
                        reason=f"native qualification failed: {type(exc).__name__}: {exc}",
                    )
                else:
                    self._native_status.update(status="ACCURACY_PASS", reason=None)
            # Candidate failures remain outside the native exception scope.
            output = self.gems_op(*args, **kwargs)
            _check_output(output, args, snapshots)
            assert not _changed_inputs(args, snapshots), "candidate mutated inputs"
            del output
            yield (*args, kwargs)

    def get_latency(self, op, *args, **kwargs):
        # Base.run aborts the iterator for any timing failure. Limit recovery to
        # native failures so candidate rows and standard JSON remain available.
        candidate = op is self.gems_op
        try:
            latency = super().get_latency(op, *args, **kwargs)
            changed = _changed_inputs(args, self._snapshots, restore=not candidate)
            if changed:
                raise AssertionError(f"timed callable mutated inputs {changed}")
        except Exception as exc:
            if candidate:
                raise
            _changed_inputs(args, self._snapshots, restore=True)
            self.torch_op = None
            self._native_status.update(
                status="N/A",
                reason=f"native timing failed: {type(exc).__name__}: {exc}",
            )
            return None
        if not candidate:
            self._native_status.update(status="PASS", reason=None)
        return latency


@pytest.mark.triton_ihc_post
def test_triton_ihc_post():
    bench = TritonIhcPostBenchmark(
        op_name=workloads.OP_NAME,
        torch_op=workloads.native(),
        gems_op=flaggems_vllm.triton_ihc_post,
        dtypes=workloads.DTYPES,
    )
    bench.run()
