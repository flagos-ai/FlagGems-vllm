# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""No-Graph device-event benchmark with independent oracle.

Inherits the base.Benchmark framework for weekly-CI aggregation while keeping
the 5-warmup/25-sample device-event protocol and per-case oracle qualification
from the dedicated indexer script. Run with:

    pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
"""

import gc
import statistics

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
from benchmark import base, consts
from benchmark.conftest import Config, emit_record_logger, update_result
from dataclasses import asdict
from flaggems_vllm.runtime import torch_device_fn


class FusedIndexerKNormBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "core_shapes.yaml"
    DEFAULT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPE_DESC = "M, D, row_gap"
    EPS = 1e-6
    SEED = 20261106
    WARMUP = 5
    SAMPLES = 25

    def __init__(self):
        super().__init__(
            op_name="fused_indexer_k_norm",
            torch_op=self._decomposed_op,
            gems_op=flaggems_vllm.fused_indexer_k_norm,
        )
        self._latencies = None

    def init_user_config(self):
        if Config.mode != consts.BenchMode.KERNEL:
            raise ValueError(
                "fused_indexer_k_norm benchmark requires --mode kernel "
                "with no-Graph device events."
            )
        super().init_user_config()

    def _decomposed_op(self, x, weight, bias, dim, eps):
        return F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x)

    def _oracle(self, x_cpu, weight_cpu, bias_cpu, dim, eps):
        values = x_cpu.float().double()
        centered = values - values.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered / torch.sqrt(variance + eps)
        return (normalized * weight_cpu.double() + bias_cpu.double()).to(x_cpu.dtype)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            # Begin a fresh measurement pair for every shape.
            self._latencies = None
            m, d, gap = shape
            gen = torch.Generator(device="cpu").manual_seed(self.SEED)
            parent_cpu = (torch.rand((m, d + gap), generator=gen) * 6 - 3).to(dtype)
            weight_cpu = torch.rand((d,), generator=gen) * 3 - 1.5
            bias_cpu = torch.rand((d,), generator=gen) * 3 - 1.5
            parent = parent_cpu.to(device=self.device)
            weight = weight_cpu.to(device=self.device)
            bias = bias_cpu.to(device=self.device)
            # The generator and construction must not carry into timing.
            del gen, parent_cpu, weight_cpu, bias_cpu
            gc.collect()
            yield parent[:, :d], weight, bias, d, self.EPS

    def get_latency(self, op, *args, **kwargs):
        if self._latencies is None:
            self._latencies = self._measure_pair(args)
        key = "torch_op" if op == self.torch_op else "gems"
        return self._latencies[key]

    def _measure_pair(self, args):
        x, weight, bias, dim, eps = args

        # Independent CPU oracle from the exact stored inputs.
        x_cpu = x.cpu()
        weight_cpu = weight.cpu()
        bias_cpu = bias.cpu()
        expected = self._oracle(x_cpu, weight_cpu, bias_cpu, dim, eps)

        _tolerances = {
            torch.float32: (2e-5, 2e-5),
            torch.float16: (2e-3, 2e-3),
            torch.bfloat16: (1.6e-2, 1.6e-2),
        }

        # Candidate must pass the oracle before it is timed.
        candidate_out = self.gems_op(*args)
        rtol, atol = _tolerances[x.dtype]
        torch.testing.assert_close(
            candidate_out.cpu(), expected, rtol=rtol, atol=atol
        )

        methods = {
            "torch_op": self.torch_op,
            "gems": self.gems_op,
        }
        results = {}
        for key, fn in methods.items():
            for _ in range(self.WARMUP):
                fn(*args)
            torch_device_fn.synchronize()
            samples = []
            for _ in range(self.SAMPLES):
                torch_device_fn.synchronize()
                start = torch_device_fn.Event(enable_timing=True)
                end = torch_device_fn.Event(enable_timing=True)
                start.record()
                output = fn(*args)
                end.record()
                torch_device_fn.synchronize()
                samples.append(start.elapsed_time(end))
                del output
            results[key] = statistics.median(samples)
        return results

    def record_shapes(self, *args, **kwargs):
        x, weight, bias, dim, eps = args
        return {
            "M": x.shape[0],
            "D": dim,
            "row_gap": x.shape[1] - dim if x.shape[1] > dim else 0,
            "dtype": str(x.dtype),
            "x_stride": list(x.stride()),
        }

    def run(self):
        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            input_iter = self.get_input_iter(dtype)
            done = False
            while not done:
                try:
                    inp = next(input_iter)
                except StopIteration:
                    done = True
                    continue
                metric = consts.BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(inp)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)
                    if (
                        "latency_base" in self.to_bench_metrics
                        and self.torch_op is not None
                    ):
                        metric.latency_base = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
                    if "latency" in self.to_bench_metrics:
                        metric.latency = self.get_latency(
                            self.gems_op, *args, **kwargs
                        )
                    if (
                        "speedup" in self.to_bench_metrics
                        and metric.latency_base is not None
                        and metric.latency
                    ):
                        metric.speedup = metric.latency_base / metric.latency
                except Exception as e:
                    metric.error_msg = str(e)
                    raise
                finally:
                    metrics.append(metric)
                    gc.collect()
            result = consts.BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            update_result(self.op_name, asdict(result))
            emit_record_logger(result.to_json())


@pytest.mark.fused_indexer_k_norm
def test_fused_indexer_k_norm_benchmark():
    FusedIndexerKNormBenchmark().run()
