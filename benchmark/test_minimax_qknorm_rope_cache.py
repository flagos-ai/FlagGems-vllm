# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch

import flaggems_vllm
from baseline.minimax_qknorm_rope_cache import (
    Q_HEADS,
    TOKENS,
    build_case,
    load_baseline,
)
from tools.minimax_qknorm_rope_cache.run_all import make_sampler

from . import base
from .conftest import Config
from .consts import BenchLevel, BenchMode

pytestmark = [
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="Ascend benchmark"
    ),
]


class MiniMaxBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        tokens = (16, 128, 256) if Config.bench_level == BenchLevel.CORE else TOKENS
        self.shapes = [(t, q) for q in Q_HEADS for t in tokens]
        self.shape_desc = "tokens, query heads"

    def get_input_iter(self, dtype):
        for tokens, heads in self.shapes:
            kwargs, _, _ = build_case(tokens, heads, self.device)
            yield (kwargs,)

    def get_latency(self, op, *args, **kwargs):
        sample = make_sampler(
            torch,
            lambda: op(*args, **kwargs),
            "do-bench-npu",
            max(Config.warm_up, 1),
            max(Config.repetition, 1),
        )
        latency = sample()
        if not math.isfinite(latency) or latency <= 0:
            pytest.fail(f"do_bench_npu returned invalid latency: {latency}")
        return latency


@pytest.mark.minimax_qknorm_rope_cache
@torch.inference_mode()
def test_perf_minimax_qknorm_rope_cache():
    if not torch.npu.get_device_name().startswith("Ascend910B"):
        pytest.skip("Ascend 910B benchmark")
    if Config.mode != BenchMode.KERNEL:
        pytest.skip("MiniMax performance benchmark uses kernel timing")
    baseline_kind = os.environ.get("MINIMAX_BASELINE", "native")
    if baseline_kind == "native":
        module = base.import_vllm_benchmark_baseline(
            "vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache"
        )
        baseline = module.minimax_qknorm_rope_cache
    elif baseline_kind == "source-replay":
        baseline = load_baseline(baseline_kind)
    else:
        raise ValueError("MINIMAX_BASELINE must be native or source-replay")
    print(f"MiniMax baseline={baseline_kind}, timer=do_bench_npu")
    MiniMaxBenchmark(
        op_name="minimax_qknorm_rope_cache",
        torch_op=baseline,
        gems_op=flaggems_vllm.minimax_qknorm_rope_cache,
        dtypes=[torch.bfloat16],
        is_inplace=True,
    ).run()
