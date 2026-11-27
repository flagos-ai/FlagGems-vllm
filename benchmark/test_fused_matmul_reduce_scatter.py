# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Collective benchmark; FLAGGEMS_GEMM_RS_TUNE=1 enables a coordinated sweep.

All ranks execute identical warmups/trials. Score = max(rank-local median ms).
Packing/copying and result allocation are included; context creation and compile
time are excluded. This reports composed Torch GEMM+RS as the baseline.
The measured API is selected from the installed PyTorch override capability and
reported with the results: symm_mem when supported, otherwise direct calls.
Shapes are global (M, N, K), with 1K = 1024. Each rank holds A[M, K/TP]
and B[K/TP, N], and receives an output shard [M/TP, N]. Unsupported shapes
are reported as skips on every rank before allocating inputs or timing.
"""

import json
import os
import statistics
from dataclasses import asdict, replace

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_matmul_reduce_scatter import (
    _default_config,
    clear_fused_matmul_reduce_scatter_cache,
    configure_fused_matmul_reduce_scatter,
)
from flaggems_vllm.testing.fused_matmul_reduce_scatter import (
    assert_close,
    make_reference,
)

from . import consts
from .conftest import Config, emit_record_logger, update_result

pytestmark = [
    pytest.mark.fused_matmul_reduce_scatter,
    pytest.mark.skipif(
        Config.mode == consts.BenchMode.CUDAGRAPH,
        reason="the initial TLE implementation does not support graph capture",
    ),
]


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
    configs = list(dict.fromkeys(configs))
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
    "m,n,k",
    [
        pytest.param(2048, 4096, 16384, id="M2K-N4K-K16K"),
        pytest.param(4096, 4096, 16384, id="M4K-N4K-K16K"),
        pytest.param(8192, 8192, 32768, id="M8K-N8K-K32K"),
        pytest.param(16384, 12288, 49152, id="M16K-N12K-K48K"),
        pytest.param(16384, 16384, 65536, id="M16K-N16K-K64K"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_matmul_reduce_scatter(
    gemm_rs_context, gemm_rs_op, gemm_rs_api, m, n, k, dtype
):
    group, device = gemm_rs_context
    world_size = group.size()
    if m % world_size or k % world_size:
        pytest.skip(f"global M={m} and K={k} must be divisible by TP={world_size}")
    rows, local_k = m // world_size, k // world_size
    if rows < 256:
        pytest.skip(
            f"M/TP={m}/{world_size}={rows}; the TLE path requires at least 256 rows"
        )

    # Match the runtime's cross-node transfer limit. A collective decision keeps
    # every rank in the same trial sequence even when compiler limits differ.
    transfer_elements = triton.next_power_of_2(rows * n)
    multi_node = world_size > int(os.environ["LOCAL_WORLD_SIZE"])
    eligible = torch.tensor(
        int(not multi_node or transfer_elements <= tl.TRITON_MAX_TENSOR_NUMEL),
        device=device,
    )
    dist.all_reduce(eligible, op=dist.ReduceOp.MIN, group=group)
    if not eligible.item():
        pytest.skip(
            f"cross-node transfer requires {transfer_elements} tensor elements; "
            "at least one rank's Triton tensor limit is smaller "
            f"(local limit: {tl.TRITON_MAX_TENSOR_NUMEL})"
        )

    torch.manual_seed(8123 + group.rank())
    a = torch.randn((m, local_k), device=device, dtype=dtype) * 0.01
    b = torch.randn((local_k, n), device=device, dtype=dtype) * 0.01

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
                return gemm_rs_op(a, b, "sum", 0, group.group_name)

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
                        "api": gemm_rs_api,
                        "shape": [m, n, k],
                        "local_k": local_k,
                        "rows_per_rank": rows,
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
                shape_detail=(
                    f"M={m}, N={n}, K={k}, local_K={local_k}, TP={world_size}, "
                    f"api={gemm_rs_api}, packing included"
                ),
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
