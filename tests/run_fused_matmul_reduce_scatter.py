# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Standalone correctness entry point, without pytest.

    torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 \
        --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
        tests/run_fused_matmul_reduce_scatter.py --dtype bf16 --reduce-op sum

Each worker calls torch.ops.symm_mem.fused_matmul_reduce_scatter and compares it
against Torch GEMM + NCCL reduce-scatter. Any rank's mismatch fails every worker.
"""

import argparse
import importlib
import os


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        parser.error(
            "launch with torchrun --nproc-per-node=2 (or more); at least two GPUs are required"
        )
    local_rank = int(os.environ["LOCAL_RANK"])

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        parser.error("requires GPUs available through torch.cuda (CUDA or HCU/HIP)")
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if torch.cuda.device_count() < local_size:
        parser.error(
            f"this node needs {local_size} visible GPUs, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    try:
        tle = importlib.import_module("triton.experimental.tle.language")
    except ImportError:
        parser.error(
            "requires FlagTree with TLE; ordinary Triton does not provide this backend"
        )
    if not tle.communication.enabled:
        parser.error("requires a matching FlagCX host/device runtime")

    import flaggems_vllm
    from flaggems_vllm.ops.fused_matmul_reduce_scatter import (
        _default_config,
        _validate_device,
        _validate_shape,
    )

    m = args.m if args.m is not None else 256 * world_size
    config = _default_config()
    try:
        backend = _validate_device(device)
        _validate_shape(m, args.n, args.local_k, world_size, config)
    except (ValueError, NotImplementedError) as exc:
        parser.error(str(exc))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
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
            for iteration in range(args.repeat):
                a.add_(0.001)
                output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                    a, b, args.reduce_op, 0, group.group_name
                )

                # Reference computation is confined to this test entry point.
                reference = torch.empty_like(output)
                dist.reduce_scatter_tensor(reference, a @ b, group=group)
                if args.reduce_op == "avg":
                    reference.div_(group.size())
                error = None
                try:
                    torch.testing.assert_close(
                        output,
                        reference,
                        rtol=0.04 if dtype == torch.bfloat16 else 0.01,
                        atol=0.005 if dtype == torch.bfloat16 else 0.001,
                    )
                except AssertionError as exc:
                    error = str(exc)
                failed = torch.tensor(int(error is not None), device=device)
                dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
                if failed.item():
                    raise AssertionError(
                        error or "a peer rank failed correctness validation"
                    )
                print(
                    f"[rank {group.rank()}] PASS iteration={iteration + 1}/{args.repeat} "
                    f"dtype={args.dtype} reduce_op={args.reduce_op} "
                    f"A={tuple(a.shape)} B={tuple(b.shape)} output={tuple(output.shape)}",
                    flush=True,
                )
    finally:
        try:
            flaggems_vllm.clear_fused_matmul_reduce_scatter_cache(group.group_name)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
