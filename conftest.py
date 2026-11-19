# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared GEMM reduce-scatter fixtures for tests/ and benchmark/."""

import atexit
import importlib
import os
import sys
import traceback
from pathlib import Path

import pytest

_TLE_BOOTSTRAP = None


def _cleanup_tle():
    global _TLE_BOOTSTRAP
    tle, _TLE_BOOTSTRAP = _TLE_BOOTSTRAP, None
    if tle is not None:
        import torch.distributed as dist

        try:
            tle.cleanup_communicator()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _init_tle():
    global _TLE_BOOTSTRAP
    tle = importlib.import_module("triton.experimental.tle.language")
    if not tle.communication.enabled:
        raise RuntimeError("GEMM reduce-scatter requires a matching FlagCX runtime")
    _TLE_BOOTSTRAP = tle
    # Also covers failures while importing a child conftest, before configure.
    atexit.register(_cleanup_tle)
    try:
        if tle.get_mem_pool() is None:
            raise RuntimeError("TLE bootstrap did not provide a memory pool")
    except BaseException:
        # Report before collective teardown or torchrun termination can hide it.
        print(
            f"[rank {os.getenv('RANK', '?')}] TLE bootstrap failed:",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        _cleanup_tle()
        raise


# Root conftest loads before tests/benchmark import FlagGems-vllm. Preserve the
# BW tutorial's TLE-before-operator import order for explicit distributed runs.
# Ordinary tests and collection-only commands never initialize communication.
if (
    int(os.environ.get("WORLD_SIZE", "1")) > 1
    and not {"--collect-only", "--collectonly", "--co", "--help", "-h"}.intersection(
        sys.argv[1:]
    )
    and any(
        Path(arg.split("::", 1)[0]).name == "test_fused_matmul_reduce_scatter.py"
        for arg in sys.argv[1:]
    )
):
    import torch
    import torch.distributed as dist

    if torch.version.hip is not None and not dist.is_initialized():
        _init_tle()


def pytest_unconfigure(config):
    _cleanup_tle()


@pytest.fixture(scope="session")
def _gemm_rs_group():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("launch with torchrun and at least two ranks")
    import torch
    import torch.distributed as dist

    owns_group = not dist.is_initialized()
    if owns_group and torch.version.hip is not None:
        _init_tle()
    if not torch.cuda.is_available():
        pytest.fail("GEMM reduce-scatter requires CUDA/HIP GPUs")
    if torch.cuda.device_count() < int(os.environ["LOCAL_WORLD_SIZE"]):
        pytest.fail("not enough visible GPUs for the local workers")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    from flaggems_vllm.ops.fused_matmul_reduce_scatter import _validate_device

    _validate_device(device)
    tle = importlib.import_module("triton.experimental.tle.language")
    if not tle.communication.enabled:
        pytest.fail("GEMM reduce-scatter requires a matching FlagCX runtime")
    if owns_group and _TLE_BOOTSTRAP is None:
        dist.init_process_group("nccl", device_id=device)
    try:
        yield dist.group.WORLD, device
    finally:
        if owns_group and _TLE_BOOTSTRAP is None:
            dist.destroy_process_group()


@pytest.fixture
def gemm_rs_context(_gemm_rs_group):
    from flaggems_vllm import clear_fused_matmul_reduce_scatter_cache, use_symm_mem

    # Function scope lets registration tests verify the original dispatcher
    # between collective cases, even when both live in the same test module.
    with use_symm_mem():
        try:
            yield _gemm_rs_group
        finally:
            clear_fused_matmul_reduce_scatter_cache()
