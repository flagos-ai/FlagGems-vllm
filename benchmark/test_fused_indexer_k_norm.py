# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Standard kernel-mode benchmark for fused_indexer_k_norm.

Uses base.Benchmark.get_latency, like the other operator benchmarks.
The shared framework selects the backend timer (do_bench_npu on Ascend).
G5 correctness is handled separately in
``tests/test_fused_indexer_k_norm.py``.
JSON records merge both output modes per dtype, with ``out_mode`` identifying
each detail. The summary's dtype-level speedup averages both output modes.

Run with::

    pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
    pytest -q -s benchmark/test_fused_indexer_k_norm.py::test_fused_indexer_k_norm_benchmark_out_provided --mode kernel
"""

import gc

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
from benchmark import base, consts
from benchmark.conftest import TEST_RESULTS, Config


class FusedIndexerKNormBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "core_shapes.yaml"
    DEFAULT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "M, D, row_gap"
    EPS = 1e-6
    SEED = 20261106

    def __init__(self, use_out_provided=False, **kwargs):
        self._use_out_provided = use_out_provided
        self._out_buffer = None
        _gems = flaggems_vllm.fused_indexer_k_norm
        if use_out_provided:

            def _gems_op(x, weight, bias, dim, eps):
                buf = self._out_buffer
                if (
                    buf is None
                    or buf.shape != x.shape
                    or buf.dtype != x.dtype
                    or buf.device != x.device
                ):
                    buf = torch.empty(x.shape, dtype=x.dtype, device=x.device)
                    self._out_buffer = buf
                return flaggems_vllm.fused_indexer_k_norm(
                    x, weight, bias, dim, eps, buf
                )

            _gems = _gems_op
        super().__init__(
            op_name="fused_indexer_k_norm",
            torch_op=self._torch_op,
            gems_op=_gems,
            **kwargs,
        )

    def init_user_config(self):
        # Use the project's kernel benchmark mode without CUDA Graph.
        if Config.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "fused_indexer_k_norm benchmark requires --mode kernel "
                "without CUDA Graph."
            )
        super().init_user_config()

    def run(self):
        super().run()
        if not Config.record_json or Config.query:
            return

        # The summary parser retains only the last record per dtype. Merge the
        # two tests' results so all out_mode-tagged details survive parsing.
        records = TEST_RESULTS.get(self.op_name, {}).get("details", [])
        merged = {}
        for record in records:
            key = (record["dtype"], record["mode"], record["level"])
            if key in merged:
                merged[key]["result"].extend(record["result"])
            else:
                merged[key] = record
        records[:] = merged.values()

    def _torch_op(self, x, weight, bias, dim, eps):
        return F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            m, d, gap = shape
            gen = torch.Generator(device="cpu").manual_seed(self.SEED)
            parent_cpu = (torch.rand((m, d + gap), generator=gen) * 6 - 3).to(dtype)
            weight_cpu = torch.rand((d,), generator=gen) * 3 - 1.5
            bias_cpu = torch.rand((d,), generator=gen) * 3 - 1.5
            parent = parent_cpu.to(device=self.device)
            weight = weight_cpu.to(device=self.device)
            bias = bias_cpu.to(device=self.device)
            del gen, parent_cpu, weight_cpu, bias_cpu
            gc.collect()
            yield parent[:, :d], weight, bias, d, self.EPS

    def record_shapes(self, *args, **kwargs):
        x, weight, bias, dim, eps = args
        return {
            "M": x.shape[0],
            "D": dim,
            # Inputs are column-contiguous slices of a padded parent tensor.
            "row_gap": x.stride(0) - dim,
            "dtype": str(x.dtype),
            "x_stride": list(x.stride()),
            "out_mode": "out_provided" if self._use_out_provided else "default",
        }


@pytest.mark.fused_indexer_k_norm
def test_fused_indexer_k_norm_benchmark():
    """Default mode: fresh output allocation per call (out=None)."""
    FusedIndexerKNormBenchmark().run()


@pytest.mark.fused_indexer_k_norm
def test_fused_indexer_k_norm_benchmark_out_provided():
    """Pre-allocated output buffer (vLLM production pattern)."""
    FusedIndexerKNormBenchmark(use_out_provided=True).run()
