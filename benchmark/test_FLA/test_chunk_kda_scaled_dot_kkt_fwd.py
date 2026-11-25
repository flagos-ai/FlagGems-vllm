# SPDX-License-Identifier: Apache-2.0
import functools
import importlib.util
import inspect
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import flaggems_vllm as gems
from benchmark.base import Benchmark
from benchmark.conftest import Config, update_result

spec = importlib.util.spec_from_file_location(
    "kda_kkt_workload",
    Path(__file__).resolve().parents[2] / "baseline/chunk_kda_scaled_dot_kkt_fwd.py",
)
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


def _adapt_native_op(op):
    """Adapt the resolved native vLLM op to the workload's calling convention.

    The new-stack vLLM wrapper derives chunk indices from ``cu_seqlens``
    internally and dropped the explicit ``chunk_indices`` parameter that the
    flaggems op still accepts (the workload passes it for varlen shapes).
    Keep only kwargs the native signature still declares so the baseline leg
    stays measurable; on stacks that still accept them nothing is dropped.
    """
    native_params = set(inspect.signature(op).parameters)

    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        dropped = [name for name in kwargs if name not in native_params]
        if dropped:
            print(
                f"[native-adapt] {workload.OP_NAME}: dropping kwargs {dropped} "
                "absent from the native signature"
            )
            kwargs = {
                name: value for name, value in kwargs.items() if name in native_params
            }
        return op(*args, **kwargs)

    return wrapper


# Only these measured bindings use the bounded native search. Custom and
# comprehensive-only shapes retain the provider's original configuration list.
_NATIVE_PROFILE_SHAPES = {
    (1, 65, 4, 128, 64),
    (1, 513, 4, 128, 64, 17, 0, 64, 432),
}


def _native_slot(*args, **kwargs):
    raise RuntimeError("native slot must be resolved by get_latency")


@contextmanager
def _native_tuning_scope(native, enabled=False):
    """Limit Ascend cold search to measured provider configurations locally."""
    if not enabled or native is None or gems.vendor_name != "ascend":
        yield
        return
    from triton.runtime.autotuner import Autotuner

    module = sys.modules.get(native.__module__)
    profiles = {
        "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter": "kda_kkt_native_inter",
        "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra": "kda_kkt_native_intra",
    }
    with pytest.MonkeyPatch.context() as patch:
        for name, profile in profiles.items():
            tuner = getattr(module, name, None)
            while not isinstance(tuner, Autotuner) and hasattr(tuner, "fn"):
                tuner = tuner.fn
            if not isinstance(tuner, Autotuner):
                continue
            allowed = {
                (repr(sorted(c.kwargs.items())), c.num_warps, c.num_stages)
                for c in gems.runtime.get_tuned_config(profile)
            }
            selected = [
                c
                for c in tuner.configs
                if (repr(sorted(c.kwargs.items())), c.num_warps, c.num_stages)
                in allowed
            ]
            # An unrecognized provider retains its original search unchanged.
            if not selected:
                continue
            patch.setattr(tuner, "configs", selected)
            for attr in ("cache", "configs_timings", "seen_tuned_metas"):
                if hasattr(tuner, attr):
                    patch.setattr(tuner, attr, {})
            for attr in ("best_config", "bench_time"):
                patch.setattr(tuner, attr, getattr(tuner, attr, None), raising=False)
        yield


class ChunkKDAScaledDotKKTFwdBenchmark(Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).resolve().parents[1] / "core_shapes.yaml")
    DEFAULT_DTYPES = workload.DTYPES
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (
            binding["B"],
            binding["T"],
            binding["H"],
            binding["K"],
            binding["BT"],
            *(binding["lengths"] or []),
        )
        for binding in workload.grid()
    ]
    DEFAULT_SHAPE_DESC = "B, T, H, K, BT, seq_lens... (omitted for fixed length)"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if any(len(shape) < 5 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES
            self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def run(self):
        gems_only = Config.user_desired_metrics == ["latency"]
        self.native_op = None
        self.native_reason = (
            "native timing disabled by --metrics" if gems_only else None
        )
        if not gems_only:
            try:
                self.native_op = workload.native()
                self.native_reason = workload.NATIVE_UNAVAILABLE_REASON
                if self.native_op is self.gems_op:
                    self.native_op = None
                    self.native_reason = "native resolves to the Gems candidate"
            except Exception as error:
                self.native_reason = f"native resolution: {error!r}"
        with pytest.MonkeyPatch.context() as config:
            if gems_only:
                config.setattr(Config, "base_data", None)
            super().run()

    def get_input_iter(self, dtype):
        misses = []
        for b, t, h, k, bt, *lengths in self.shapes:
            binding = dict(B=b, T=t, H=h, K=k, BT=bt, lengths=lengths or None)
            args, kwargs = workload.build_inputs(binding, dtype, self.device)
            self.row_native_reason = self.native_reason
            self.row_native_tuning = (
                b,
                t,
                h,
                k,
                bt,
                *lengths,
            ) in _NATIVE_PROFILE_SHAPES and dtype in workload.DTYPES
            yield (*args, kwargs)
            if self.row_native_reason:
                misses.append(
                    dict(
                        shape=self.record_shapes(*args, **kwargs),
                        reason=self.row_native_reason,
                    )
                )
        if misses:
            update_result(
                self.op_name,
                dict(
                    dtype=str(dtype),
                    native_baseline_skip_reason="; ".join(
                        dict.fromkeys(row["reason"] for row in misses)
                    ),
                    native_baseline_misses=misses,
                ),
            )

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            return super().get_latency(op, *args, **kwargs)
        if self.native_op is None:
            return None

        expected = self.gems_op(*args, **kwargs)
        try:
            with _native_tuning_scope(self.native_op, self.row_native_tuning):
                actual = self.native_op(*args, **kwargs)
                for got, want in zip(actual, expected, strict=True):
                    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-5)
                latency = super().get_latency(self.native_op, *args, **kwargs)
                if not math.isfinite(latency) or latency <= 0:
                    raise ValueError(f"invalid native latency: {latency}")
        except Exception as error:
            self.row_native_reason = f"native validation or timing: {error!r}"
            return None
        return latency


@pytest.mark.chunk_kda_scaled_dot_kkt_fwd
def test_chunk_kda_scaled_dot_kkt_fwd_perf():
    # enflame-only gate (2026-11-20 定标): the vLLM import chain
    # (vllm.config -> transformers.audio_utils -> torchaudio) can raise OSError
    # from a broken torchaudio/torch c10 ABI pairing in addition to ImportError,
    # neither of which is an op defect. Degrade to latency-only metrics
    # (torch_op=None) instead of failing the whole benchmark run. When the
    # native op does resolve, adapt its signature (see _adapt_native_op).
    if gems.vendor_name == "enflame":
        try:
            torch_op = workload.native()
        except (ImportError, OSError) as exc:
            workload.NATIVE_UNAVAILABLE_REASON = repr(exc)
            torch_op = None
        if torch_op is not None:
            torch_op = _adapt_native_op(torch_op)
    else:
        torch_op = workload.native()
    bench = ChunkKDAScaledDotKKTFwdBenchmark(
        op_name=workload.OP_NAME,
        torch_op=torch_op,
    )
    bench.set_gems(gems.chunk_kda_scaled_dot_kkt_fwd)
    bench.run()
