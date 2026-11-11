# TLE fused_matmul_reduce_scatter (initial implementation)

This ports the persistent GEMM producer and hierarchical reduce-scatter from
FlagTree tutorials 08/09 into the installed `flaggems_vllm` package. No tutorial
directory is needed at runtime. The adapted kernel module retains FlagTree's MIT
notice. Source: local FlagTree revision
`ff67c2fb6be1d4651258605f6a5edde533d3a61c` (tutorials numbered 08/09).

## Quick end-to-end correctness check

From the repository root, in an environment with FlagTree/TLE and matching
FlagCX installed, run this standalone test (pytest is not required):

```bash
PYTHONPATH=src:../FlagGems/src torchrun \
  --nnodes=1 --node-rank=0 --nproc-per-node=2 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
  tests/run_fused_matmul_reduce_scatter.py --dtype bf16 --reduce-op sum
```

The script creates the process group and inputs, enables the CUDA dispatcher
override, calls `torch.ops.symm_mem.fused_matmul_reduce_scatter`, compares against
Torch GEMM + NCCL reduce-scatter, and clears resources on exit. Each rank prints
`PASS` and its output shape for three iterations. A mismatch exits with failure.
By default A is `[256 * group_size, 256]`, B is `[256, 256]`, and every rank's
output is `[256, 256]`. Use `--dtype fp16 --reduce-op avg` for the other variants,
or `--help` to see shape and repetition options. `--local-k` is already per-rank.
Full communication requires at least two GPUs; a one-GPU or non-TLE launch
reports the missing requirement instead of skipping or falling back.

The public call has the same argument names and order as
`torch.ops.symm_mem.fused_matmul_reduce_scatter`:

```python
import flaggems_vllm

# torch.distributed and the TP group have already been initialized.
# A: [M, local_K], B: [local_K, N]
out = flaggems_vllm.fused_matmul_reduce_scatter(
    A, B, "sum", 0, tp_group.group_name
)
# out: [M / tp_group.size(), N]

# Collectively, before the application destroys its TP process group:
flaggems_vllm.clear_fused_matmul_reduce_scatter_cache(tp_group.group_name)
```

Each rank computes its local `A @ B`; contributions are reduced and the output
rows are divided among group ranks. The API does not split local K again.
`reduce_op="avg"` scales once in the final reduction. Accumulation/reduction order
follows the tutorial and is not bitwise identical to Torch GEMM + NCCL.

## Supported scope

- NVIDIA CUDA SM90+, or an experimental HCU/HIP path using FlagTree's tensor
  descriptor-to-pointer lowering. Both require TLE and matching FlagCX
  host/device runtime. HCU execution is pending validation on BW hardware.
- Two-dimensional A/B, matching FP16 or BF16 dtype/device, inference only.
- `reduce_op` is `sum` or `avg`; `scatter_dim` is exactly 0.
- At least two group members. Existing groups and TP subgroups are used without
  initializing or destroying a PyTorch group.
- Group ranks must be node-contiguous, with the same number of ranks per node
  and a distinct GPU UUID per local rank. Shapes, dtype, device SM count and
  configuration must agree across ranks. Cross-node transfers retain the
  tutorial's `next_power_of_2((M/P)*N) <= TRITON_MAX_TENSOR_NUMEL` limit.
- `M % P == 0`, `M/P >= 256`, and `M/P` divisible by GEMM `block_m`.
  N is divisible by both GEMM `block_n` and RS `rs_block_n`; local K is divisible
  by GEMM `block_k`. Empty matrices and CUDA graph capture are unsupported.
- A is made contiguous if necessary. B is packed to continuous `[N, local_K]`.
  Misaligned contiguous views are cloned to preserve descriptor alignment.
  Packing/copies are part of the public call and benchmark timing.

Unsupported scope raises explicitly; there is no Torch compute fallback.
Importing the package exports the library function. To route existing
`torch.ops.symm_mem.fused_matmul_reduce_scatter` call sites, explicitly activate
the CUDA dispatcher override as described below. ATen enable/use_gems does not
activate this override.

## Calling through torch.ops.symm_mem

Enable the override in every participating worker before its first collective:

```python
import torch
import flaggems_vllm

# Existing distributed/TP initialization and input creation precede this code.
flaggems_vllm.enable_symm_mem()
try:
    out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        A, B, "sum", 0, tp_group.group_name
    )
finally:
    flaggems_vllm.disable_symm_mem()
    flaggems_vllm.clear_fused_matmul_reduce_scatter_cache(tp_group.group_name)
```

For a temporary override use a context manager:

```python
with flaggems_vllm.use_symm_mem():
    out = torch.ops.symm_mem.fused_matmul_reduce_scatter(
        A, B, "sum", 0, tp_group.group_name
    )
```

Both forms install a real CUDA dispatcher implementation via
`torch.library.Library("symm_mem", "IMPL")` and
`impl(..., "CUDA", allow_override=True)`. They first import PyTorch's symmetric
memory module to load its existing schema and kernels. They do not monkey-patch
`torch.ops` attributes, create a second schema, or alter Meta/other dispatch keys.
An overload captured before activation uses the replacement while it is active.

Registration is process-wide. Persistent enable is idempotent; nested context
managers and persistent activation share the registration until the last user
releases it. Exit, including exceptional exit, restores the underlying PyTorch
CUDA implementation. Disabling registration does not collectively free buffers;
clear the context cache on every group member before destroying its group.
The original Python registration's bookkeeping is retained during restoration.

HIP PyTorch still uses the `CUDA` dispatcher key and `torch.cuda` interfaces.
HCU activation checks FlagTree's selected backend instead of rejecting HIP builds.
This requires a PyTorch build exposing `Library.impl(allow_override=True)`.
An older API raises a clear error and leaves the direct library call available.
Registration itself does not initialize a process group, load TLE, or create a
FlagCX communicator. Current dtype/layout/shape limitations continue to apply
through torch.ops; unsupported inputs raise without a Torch compute fallback.

## Communication resources

The implementation uses FlagCX bindings shipped by FlagTree, but creates its own
communicator/device communicator/window for each cached workload. Bootstrap uses
the caller's process-group store, not global WORLD collectives. It does not call
TLE's global `get_mem_pool()` or `cleanup_communicator()`.

FlagCX communication memory is exposed through a zero-copy CUDA array interface
and dtype views. Outputs own separate storage. Completion events serialize
scratch-buffer reuse across caller streams. Cache keys include the process group,
device, dtype, M/N/local K, and configuration. Buffers persist until collective
cache clearing; workloads with many shapes should clear between phases to bound
memory use. Clearing releases this operator's handles and allocations, and keeps
the caller's process group alive. Call each group operation/configuration/clear
in matching order on every participating rank.

## Manual tuning

The tutorial's benchmark search does not run inside the production operator.
The first call allocates resources and JIT compiles; subsequent calls reuse
them. There is an explicit no-`libtuner` exemption: producer readiness flags,
barriers and multi-stream scheduling form a stateful collective protocol.
Independent rank-local trials can deadlock.

The benchmark offers a coordinated, opt-in sweep of GEMM warps, N tiles,
pipeline stages/epilogue subtiles, and scatter/reduction SM budgets. All ranks
try candidates in the same order. Candidates are filtered against all ranks'
shared-memory limits. Buffers are cleared between candidates. Every candidate
is checked against the reference before timing. The score is the maximum of the
rank-local median latencies; every rank selects the same best configuration.
The reduced sweep is intentionally smaller than the tutorial's exhaustive joint
search. No performance claim is made before multi-GPU benchmarking.

```bash
# Run from the repository root, with FlagTree/FlagCX and the package installed.
# Optional: configure FLAGCX_IB_HCA and other fabric settings for your cluster.
FLAGGEMS_GEMM_RS_TUNE=1 torchrun --standalone --nproc-per-node=2 \
  -m pytest -s -q benchmark/test_fused_matmul_reduce_scatter.py \
  --warmup 5 --iter 20
```

Rank zero prints every trial and `best_config`. Apply that configuration on every
rank before first use (or after collectively clearing cached contexts):

```python
config = flaggems_vllm.GemmReduceScatterConfig(
    # Substitute the full best_config from the benchmark for this workload.
    block_n=128, num_warps=8, num_stages=2,
    num_scatter_sms=16, num_reduction_sms=15,
)
flaggems_vllm.configure_fused_matmul_reduce_scatter(tp_group.group_name, config)
```

The default is a conservative fixed configuration, not an autotuned result.
NVIDIA retains a 128x128x64 GEMM tile with two stages. HCU defaults to a
64x64x32 tile, four warps, and one stage, avoiding the NVIDIA configuration's
shared-memory estimate exceeding BW's 64 KiB LDS budget. HCU skips installing a
CUDA TMA descriptor allocator because descriptor lowering removes that need.
There is no tune-config YAML entry because no independent kernel autotuner is
used. Future tuning integration must preserve collective candidate ordering and
reset the protocol state between trials.

## HCU/BW multi-GPU validation

Use a FlagTree HCU build with descriptor-to-pointer lowering and matching
gfx936 FlagCX device bitcode. Confirm the original import path works:
`triton.experimental.tle.language`, with `communication.enabled == True`.
The HCU compatibility changes do not establish a hardware correctness or
performance result; validate on the actual GPUs before acceptance.

First check whether the HIP device selection variable is honored by this DTK
build (CUDA_VISIBLE_DEVICES may be ignored):

```bash
HIP_VISIBLE_DEVICES=0,1 python -c \
  'import torch; print(torch.cuda.device_count())'
```

Then run the real two-rank correctness entry point. Static rendezvous avoids
using an unresolvable container hostname for TCPStore bootstrap:

```bash
FLAGTREE_BACKEND=hcu HIP_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run \
  --nnodes=1 --node-rank=0 --nproc-per-node=2 \
  --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29501 \
  tests/run_fused_matmul_reduce_scatter.py --dtype bf16 --reduce-op sum --repeat 3
```

The entry point reports its selected backend, GPU, and configuration before
testing. Change to `--dtype fp16 --reduce-op avg` for another combination. Native
Torch's `nccl` process-group name also selects RCCL in a HIP build. For multiple
nodes, use the same routable master address/port on every node and distinct node
ranks; the loopback address above is only for the single-node multi-GPU case.

## Validation and benchmark commands

```bash
# Host checks, zero-copy allocation contract, and optional single-GPU TMA tests:
PYTHONPATH=src pytest -q tests/test_fused_matmul_reduce_scatter.py --quick

# Single-GPU dispatcher routing, activation lifetime, and restoration checks:
PYTHONPATH=src pytest -q tests/test_symm_mem_registration.py

# Full communication tests (sum/avg, dtype/layout, repeated calls, stream reuse):
torchrun --standalone --nproc-per-node=2 \
  -m pytest -q -s tests/test_fused_matmul_reduce_scatter.py

# Four ranks additionally exercise independent two-rank TP subgroups:
torchrun --standalone --nproc-per-node=4 \
  -m pytest -q -s tests/test_fused_matmul_reduce_scatter.py

# On each of two nodes; set node_rank=0/1 and the same rendezvous endpoint:
torchrun --nnodes=2 --nproc-per-node=4 --node-rank=0 \
  --master-addr=<node0-address> --master-port=29501 \
  -m pytest -q -s tests/test_fused_matmul_reduce_scatter.py

# Fixed configuration through torch.ops, composed Torch GEMM+NCCL baseline:
torchrun --standalone --nproc-per-node=2 \
  -m pytest -q -s benchmark/test_fused_matmul_reduce_scatter.py \
  --warmup 5 --iter 20
```

The benchmark active set uses `(rows_per_rank, N, local_K)` of
`(256,256,256)`, `(384,512,256)`, `(256,4096,2048)` for FP16/BF16.
It measures the complete eager operator through the CUDA dispatcher, including
layout conversion and output allocation, with compilation/context bootstrap
excluded by warmup. Metric: ms,
lower is better. Aggregation: maximum rank-local median. Speedup is composed
Torch GEMM+RS latency divided by this implementation's latency. This is not a
measurement against PyTorch's native fused implementation. No CUDA graph, kernel
only, or multi-GPU performance acceptance result is claimed by the first version.

On a single GPU the communication tests skip; an explicit multi-rank launch fails
visibly if required dependencies are missing. Single-GPU producer/reduction tests
exercise real TMA computation and readiness counters, but do not validate remote
transfers, distributed barriers, or concurrent multi-rank execution.
