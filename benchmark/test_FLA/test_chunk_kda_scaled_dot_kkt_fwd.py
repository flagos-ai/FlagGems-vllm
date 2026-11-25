# SPDX-License-Identifier: Apache-2.0
import functools
import importlib.util
import inspect
from pathlib import Path

import pytest

import flaggems_vllm as gems
from benchmark.base import Benchmark

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


class ChunkKDAScaledDotKKTFwdBenchmark(Benchmark):
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

    def get_input_iter(self, cur_dtype):
        for b, t, h, k, bt, *lengths in self.shapes:
            binding = dict(B=b, T=t, H=h, K=k, BT=bt, lengths=lengths or None)
            args, kwargs = workload.build_inputs(binding, cur_dtype, gems.device)
            yield (*args, kwargs)


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
