# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Model-derived KDA geometry with an optional FLA native baseline.

The baseline is upstream FLA's exact suboperator, never the full chunk_kda op.
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from benchmark.base import Benchmark
from benchmark.conftest import Config
from benchmark.consts import BenchMode
from flaggems_vllm import vendor_name
from flaggems_vllm.ops.FLA import chunk_kda_fwd_intra_token_parallel
from flaggems_vllm.runtime import torch_device_fn

MODEL_INPUT_SEED = 20261027

# (label, heads, evidence): all profiles use K=128, HV=H, TP=1.
MODEL_PROFILES = (
    ("kimi-linear-48b-a3b", 32, "chunk-path-safe-gate-false"),
    ("kimi-k3", 96, "geometry-only-safe-gate-true-bypasses-target"),
    ("glm-5.3-flash", 64, "geometry-only-exact-call-unverified"),
)
# B, per-sequence lengths. Multiple lengths mean packed B=1, not padded batch.
# B/T are selected workloads, not observed production traces. No decode cases:
# the inspected Kimi model switches to fused_recurrent for q_len <= 64.
WORKLOADS = (
    (1, (128,)),
    (1, (512,)),
    (1, (2048,)),
    (1, (8192,)),
    (1, (32768,)),
    (4, (2048,)),
    (1, (1024,) * 8),
    (1, (1300, 547, 2048, 963, 271, 3063)),
)


def make_inputs(batch, lengths, heads, dtype, device):
    """Build post-normalization/post-cumsum inputs outside the measured region."""
    torch.manual_seed(MODEL_INPUT_SEED)
    lengths = tuple(lengths)
    if batch < 1 or not lengths or min(lengths) <= 64:
        raise ValueError("prefill workloads require positive batch and lengths > 64")
    packed = len(lengths) > 1
    if packed and batch != 1:
        raise ValueError("packed workloads require B=1")
    total = sum(lengths)
    shape = (batch, total, heads, 128)
    q = F.normalize(torch.randn(shape, device=device, dtype=torch.float32), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    # Synthetic negative per-token log2 gates, cumulative within each BT=64
    # chunk and each sequence. This is input preparation, not a timed baseline.
    gk = torch.empty_like(q)
    offset = 0
    for length in lengths:
        for start in range(0, length, 64):
            end = min(start + 64, length)
            part = gk[:, offset + start : offset + end]
            part.copy_((-torch.rand_like(part) * 0.1).cumsum(dim=1))
        offset += length
    beta = torch.randn(shape[:-1], device=device, dtype=torch.float32).sigmoid()
    aqk = torch.zeros((*shape[:-1], 64), device=device, dtype=dtype)
    akk = torch.zeros((*shape[:-1], 16), device=device, dtype=torch.float32)
    cu = None
    if packed:
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        cu = torch.tensor(boundaries, device=device, dtype=torch.int64)
    return (
        q.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        gk.contiguous(),
        beta.contiguous(),
        aqk,
        akk,
        128**-0.5,
        {"cu_seqlens": cu, "chunk_size": 64, "sub_chunk_size": 16, "use_graph": False},
    )


class ChunkKDAIntraTokenBenchmark(Benchmark):
    DEFAULT_SHAPES = WORKLOADS
    DEFAULT_SHAPE_DESC = "batch, sequence_lengths (packed when multiple)"
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]

    def __init__(self, native_op, profile, dtype):
        super().__init__(
            "chunk_kda_fwd_intra_token_parallel", native_op, dtypes=[dtype]
        )
        self.profile = profile
        self.set_gems(chunk_kda_fwd_intra_token_parallel)

    def get_input_iter(self, dtype):
        for batch, lengths in self.shapes:
            self.current_lengths = tuple(lengths)
            yield make_inputs(batch, lengths, self.profile[1], dtype, self.device)

    def record_shapes(self, *args, **kwargs):
        return {
            "model_profile": self.profile[0],
            "evidence": self.profile[2],
            "sequence_lengths": self.current_lengths,
            "tp": 1,
            "input_seed": MODEL_INPUT_SEED,
            "timer": "device-event-median-no-graph-output-reset-excluded",
            "tensors_and_scalars": super().record_shapes(*args, **kwargs),
            "tensor_dtypes": [str(arg.dtype) for arg in args if torch.is_tensor(arg)],
        }

    def get_latency(self, op, *args, **kwargs):
        # Device events implement the no-Graph protocol. Compilation
        # and autotuning finish before warmup. Reset output buffers before the
        # start event so both implementations see identical inputs each time.
        if Config.mode != BenchMode.KERNEL:
            raise ValueError("KDA benchmark requires --mode kernel")
        if kwargs.get("use_graph") is not False:
            raise ValueError("Graph must remain disabled")

        def reset():
            args[4].zero_()
            args[5].zero_()

        with torch_device_fn.device(args[0].device), torch.inference_mode():
            reset()
            op(*args, **kwargs)
            torch_device_fn.synchronize()
            deadline = time.perf_counter() + Config.warm_up / 1000
            while time.perf_counter() < deadline:
                reset()
                op(*args, **kwargs)
                torch_device_fn.synchronize()
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            samples = []
            deadline = time.perf_counter() + Config.repetition / 1000
            while len(samples) < 10 or time.perf_counter() < deadline:
                reset()
                start.record()
                op(*args, **kwargs)
                end.record()
                torch_device_fn.synchronize()
                samples.append(start.elapsed_time(end))
        # Event interval includes device-visible launch gaps; this is not a
        # pure-kernel profiler metric. Return milliseconds like Benchmark.
        median = statistics.median(samples)
        record = {
            "backend": vendor_name,
            "shape": self.record_shapes(*args, **kwargs),
            "callable": f"{op.__module__}.{op.__name__}",
            "samples_ms": samples,
            "median_ms": median,
            "graph": False,
        }
        if path := os.environ.get("KDA_PERFORMANCE_RAW"):
            with Path(path).open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return median


@pytest.mark.chunk_kda_fwd_intra_token_parallel
@pytest.mark.parametrize("profile", MODEL_PROFILES, ids=lambda p: p[0])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_chunk_kda_intra_token_parallel(profile, dtype):
    native = pytest.importorskip("fla.ops.kda.chunk_intra_token_parallel")
    bench = ChunkKDAIntraTokenBenchmark(
        native.chunk_kda_fwd_intra_token_parallel, profile, dtype
    )
    bench.run()
