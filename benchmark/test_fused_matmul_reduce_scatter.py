# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective benchmark; FLAGGEMS_GEMM_RS_TUNE=1 enables a coordinated sweep.

All ranks execute identical warmups/trials. Score = max(rank-local median ms).
Packing/copying and result allocation are included; context creation and compile
time are excluded. This reports composed Torch GEMM+RS as the baseline.
"""

import importlib
import json
import os
import statistics
from dataclasses import asdict, replace

import pytest
import torch
import torch.distributed as dist
import triton

from flaggems_vllm.ops.fused_matmul_reduce_scatter import (
    _default_config,
    _validate_device,
    clear_fused_matmul_reduce_scatter_cache,
    configure_fused_matmul_reduce_scatter,
)
from flaggems_vllm.runtime.symm_mem import use_symm_mem
from flaggems_vllm.testing.fused_matmul_reduce_scatter import (
    assert_close,
    make_reference,
)

from . import consts
from .conftest import Config, emit_record_logger, update_result

pytestmark = pytest.mark.fused_matmul_reduce_scatter


@pytest.fixture(scope="module")
def distributed_context():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch this benchmark with torchrun and at least two GPUs")
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    _validate_device(device)
    importlib.import_module("triton.experimental.tle.language")
    if Config.mode == consts.BenchMode.CUDAGRAPH:
        pytest.skip("the initial TLE implementation does not support graph capture")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", device_id=device)
    try:
        with use_symm_mem():
            yield dist.group.WORLD, device
    finally:
        try:
            clear_fused_matmul_reduce_scatter_cache()
        finally:
            if owns_group:
                dist.destroy_process_group()


def _score(call, group, device):
    for _ in range(max(1, Config.warm_up)):
        call()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(max(1, Config.repetition)):
        dist.barrier(group=group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    score = torch.tensor(statistics.median(samples), device=device, dtype=torch.float64)
    dist.all_reduce(score, op=dist.ReduceOp.MAX, group=group)
    return score.item()


def _candidates(group, device):
    default = _default_config()
    configs = [default]
    if os.environ.get("FLAGGEMS_GEMM_RS_TUNE", "0") == "1":
        configs += [
            replace(default, num_warps=4),
            replace(default, block_n=256),
            replace(default, num_stages=3, epilogue_subtile=True),
            replace(default, num_scatter_sms=8, num_reduction_sms=8),
        ]
    limit = triton.runtime.driver.active.utils.get_device_properties(device.index)[
        "max_shared_mem"
    ]
    eligible = torch.tensor(
        [int(config.estimated_shared_memory() <= limit) for config in configs],
        device=device,
    )
    dist.all_reduce(eligible, op=dist.ReduceOp.MIN, group=group)
    return [config for config, fits in zip(configs, eligible.tolist()) if fits]


@pytest.mark.parametrize(
    "rows,n,k", [(256, 256, 256), (384, 512, 256), (256, 4096, 2048)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_matmul_reduce_scatter(distributed_context, rows, n, k, dtype):
    group, device = distributed_context
    torch.manual_seed(8123 + group.rank())
    a = torch.randn((rows * group.size(), k), device=device, dtype=dtype) * 0.01
    b = torch.randn((k, n), device=device, dtype=dtype) * 0.01

    def baseline():
        partial = a @ b
        out = torch.empty((rows, n), device=device, dtype=dtype)
        dist.reduce_scatter_tensor(out, partial, group=group)
        return out

    expected, tolerance = make_reference(a, b, "sum", group)
    baseline_ms = _score(baseline, group, device)
    trials = []
    try:
        configs = _candidates(group, device)
        assert configs, "no candidate fits the device shared-memory budget"
        for config in configs:
            # Reset all buffers/barrier state between trials, in the same order.
            clear_fused_matmul_reduce_scatter_cache(group.group_name)
            configure_fused_matmul_reduce_scatter(group.group_name, config)

            def implementation():
                return torch.ops.symm_mem.fused_matmul_reduce_scatter(
                    a, b, "sum", 0, group.group_name
                )

            actual = implementation()
            try:
                assert_close(actual, expected, tolerance, group, dtype=dtype)
            except AssertionError as error:
                raise AssertionError(
                    f"candidate failed correctness: {config}"
                ) from error
            milliseconds = _score(implementation, group, device)
            trials.append((milliseconds, config))
        best_ms, best_config = min(trials, key=lambda item: item[0])
        if group.rank() == 0:
            print(
                json.dumps(
                    {
                        "op": "fused_matmul_reduce_scatter",
                        "shape": list(a.shape) + [n],
                        "dtype": str(dtype),
                        "world_size": group.size(),
                        "aggregation": "max_rank_median_ms",
                        "best_config": asdict(best_config),
                        "trials": [
                            {"ms": ms, "config": asdict(cfg)} for ms, cfg in trials
                        ],
                    }
                )
            )
            metric = consts.BenchmarkMetrics(
                shape_detail=f"M={a.shape[0]}, N={n}, local_K={k}, TP={group.size()}, packing included",
                latency_base=baseline_ms,
                latency=best_ms,
                speedup=baseline_ms / best_ms,
            )
            result = consts.BenchmarkResult(
                level=Config.bench_level.value,
                op_name="fused_matmul_reduce_scatter",
                dtype=str(dtype),
                mode=Config.mode.value,
                result=[metric],
            )
            print(result)
            update_result(result.op_name, asdict(result))
            emit_record_logger(result.to_json())
    finally:
        clear_fused_matmul_reduce_scatter_cache(group.group_name)
