# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standalone correctness entry point, without pytest.

    torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 \
        --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
        tests/run_fused_matmul_reduce_scatter.py --dtype bf16 --reduce-op sum

Each worker calls torch.ops.symm_mem.fused_matmul_reduce_scatter and compares it
against an FP32 mathematical reference with a low-precision roundoff budget.
Any rank's mismatch fails every worker.
Use --comm-only to check Torch NCCL/RCCL without importing TLE or FlagGems-vllm.
"""

import argparse
import importlib
import os
from contextlib import contextmanager


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _init_nccl_group(device):
    import torch.distributed as dist

    rank = os.environ["RANK"]
    print(
        f"[rank {rank}] NCCL init begin pid={os.getpid()} device={device} "
        f"NCCL_IB_DISABLE={os.getenv('NCCL_IB_DISABLE', '<unset>')} "
        f"HSA_FORCE_FINE_GRAIN_PCIE={os.getenv('HSA_FORCE_FINE_GRAIN_PCIE', '<unset>')}",
        flush=True,
    )
    dist.init_process_group("nccl", device_id=device)
    print(f"[rank {rank}] NCCL init complete", flush=True)
    return dist.group.WORLD


def _check_communication(device, repeat):
    import torch
    import torch.distributed as dist

    group = _init_nccl_group(device)
    try:
        value = torch.empty((1,), dtype=torch.float32, device=device)
        expected = group.size() * (group.size() + 1) // 2
        for iteration in range(repeat):
            value.fill_(group.rank() + 1)
            dist.all_reduce(value, group=group)
            actual = value.item()
            if actual != expected:
                raise AssertionError(
                    f"all-reduce returned {actual}, expected {expected}"
                )
            print(
                f"[rank {group.rank()}] COMM PASS iteration={iteration + 1}/{repeat} "
                f"all_reduce={actual}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def _load_tle(parser):
    try:
        tle = importlib.import_module("triton.experimental.tle.language")
    except ImportError:
        parser.error("requires FlagTree with TLE; ordinary Triton is unsupported")
    if not tle.communication.enabled:
        parser.error("requires a matching FlagCX host/device runtime")
    return tle


def _select_device(parser, local_rank):
    import torch

    if not torch.cuda.is_available():
        parser.error("requires GPUs available through torch.cuda (CUDA or HCU/HIP)")
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if torch.cuda.device_count() < local_size:
        parser.error(
            f"this node needs {local_size} visible GPUs, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return device


@contextmanager
def _operator_group(parser, local_rank, bootstrap):
    import torch.distributed as dist

    if dist.is_initialized():
        parser.error("start this standalone runner in fresh torchrun workers")
    if bootstrap == "torch":
        device = _select_device(parser, local_rank)
        group = _init_nccl_group(device)
        try:
            yield group, device
        finally:
            dist.destroy_process_group()
        return

    rank = os.environ["RANK"]
    print(f"[rank {rank}] TLE tutorial bootstrap begin", flush=True)
    # Import TLE before touching CUDA, exactly as the tutorial does. Its
    # get_mem_pool owns lazy NCCL setup, GPU ID broadcast, and FlagCX bootstrap.
    tle = _load_tle(parser)
    try:
        if tle.get_mem_pool() is None:
            parser.error("TLE tutorial bootstrap did not provide a memory pool")
        device = _select_device(parser, local_rank)
        print(f"[rank {rank}] TLE tutorial bootstrap complete", flush=True)
        yield dist.group.WORLD, device
    finally:
        try:
            tle.cleanup_communicator()
        finally:
            # Covers partial bootstrap failures. Normal TLE cleanup already
            # destroys its group, so do not destroy it twice.
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--reduce-op", choices=("sum", "avg"), default="sum")
    parser.add_argument(
        "--m", type=_positive_int, help="global output rows; default: 256 * group size"
    )
    parser.add_argument("--n", type=_positive_int, default=256)
    parser.add_argument(
        "--local-k",
        type=_positive_int,
        default=256,
        help="K already local to each rank",
    )
    parser.add_argument("--repeat", type=_positive_int, default=3)
    parser.add_argument(
        "--bootstrap",
        choices=("torch", "tle"),
        default="torch",
        help="tle reuses the original tutorial's get_mem_pool initialization",
    )
    parser.add_argument(
        "--comm-only",
        action="store_true",
        help="only check Torch NCCL/RCCL init and all-reduce; skip TLE and FlagGems-vllm",
    )
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        parser.error(
            "launch with torchrun --nproc-per-node=2 (or more); at least two GPUs are required"
        )
    local_rank = int(os.environ["LOCAL_RANK"])

    if args.comm_only:
        device = _select_device(parser, local_rank)
        _check_communication(device, args.repeat)
        return
    with _operator_group(parser, local_rank, args.bootstrap) as (group, device):
        _check_operator(args, parser, group, device)


def _check_operator(args, parser, group, device):
    import torch
    import torch.distributed as dist

    world_size = group.size()
    print(f"[rank {group.rank()}] importing TLE", flush=True)
    _load_tle(parser)

    print(f"[rank {group.rank()}] importing FlagGems-vllm", flush=True)
    import flaggems_vllm
    from flaggems_vllm.ops.fused_matmul_reduce_scatter import (
        _default_config,
        _validate_device,
        _validate_shape,
    )
    from flaggems_vllm.testing.fused_matmul_reduce_scatter import (
        assert_close,
        error_metrics,
        make_exact_case,
        make_reference,
    )

    print(f"[rank {group.rank()}] operator imports complete", flush=True)
    m = args.m if args.m is not None else 256 * world_size
    config = _default_config()
    try:
        backend = _validate_device(device)
        _validate_shape(m, args.n, args.local_k, world_size, config)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    try:
        if group.rank() == 0 and backend == "hcu":
            keys = (
                "FLAGCX_USE_HETERO_COMM",
                "FLAGCX_MEM_ENABLE",
                "FLAGCX_VMM_ENABLE",
                "FLAGCX_P2P_DISABLE",
                "FLAGCX_IB_HCA",
                "NCCL_SOCKET_IFNAME",
                "AMDGCN_USE_BUFFER_OPS",
            )
            print(
                "HCU environment: "
                + " ".join(f"{key}={os.getenv(key, '<unset>')}" for key in keys),
                flush=True,
            )
        print(
            f"[rank {group.rank()}] backend={backend} device={torch.cuda.get_device_name(device)} config={config}",
            flush=True,
        )
        torch.manual_seed(1234 + group.rank())
        a = torch.randn((m, args.local_k), device=device, dtype=dtype) * 0.1
        b = torch.randn((args.local_k, args.n), device=device, dtype=dtype) * 0.1
        with flaggems_vllm.use_symm_mem():
            if group.size() in (2, 4, 8):
                exact_a, exact_b, exact_expected = make_exact_case(
                    group, device, dtype, args.reduce_op
                )
                exact_output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                    exact_a, exact_b, args.reduce_op, 0, group.group_name
                )
                assert_close(
                    exact_output,
                    exact_expected,
                    torch.zeros_like(exact_expected),
                    group,
                    dtype=dtype,
                )
                print(f"[rank {group.rank()}] EXACT PASS tolerance=0", flush=True)
                flaggems_vllm.clear_fused_matmul_reduce_scatter_cache(group.group_name)
            for iteration in range(args.repeat):
                a.add_(0.001)
                output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                    a, b, args.reduce_op, 0, group.group_name
                )

                reference, tolerance = make_reference(a, b, args.reduce_op, group)
                try:
                    assert_close(output, reference, tolerance, group, dtype=dtype)
                except AssertionError:
                    # All ranks fail together, so this diagnostic collective
                    # cannot leave peers waiting in a different code path.
                    baseline = torch.empty_like(output)
                    dist.reduce_scatter_tensor(baseline, a @ b, group=group)
                    if args.reduce_op == "avg":
                        baseline.div_(group.size())
                    print(
                        f"[rank {group.rank()}] FP32 diagnostic: "
                        f"implementation={error_metrics(output, reference, tolerance)}; "
                        f"torch_baseline={error_metrics(baseline, reference, tolerance)}",
                        flush=True,
                    )
                    raise
                metrics = error_metrics(output, reference, tolerance)
                print(
                    f"[rank {group.rank()}] PASS iteration={iteration + 1}/{args.repeat} "
                    f"dtype={args.dtype} reduce_op={args.reduce_op} "
                    f"A={tuple(a.shape)} B={tuple(b.shape)} output={tuple(output.shape)} "
                    f"reference=fp32 metrics={metrics}",
                    flush=True,
                )
    finally:
        flaggems_vllm.clear_fused_matmul_reduce_scatter_cache(group.group_name)


if __name__ == "__main__":
    main()
