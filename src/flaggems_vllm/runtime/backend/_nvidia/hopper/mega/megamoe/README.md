# Hopper Triton/TLE MegaMoE

This directory contains the v234 production-shape multi-rank MegaMoE candidate
migrated from FlagTree PR #712. The operator belongs in FlagGems-vllm; the
separate FlagTree PR #1079 contains only the compiler/TLE capabilities it uses.

Source provenance:

```text
FlagTree PR #712 head: e1704575e6526069115b0247795ca9c4f8bfb46c
zhiyuan_megakernel source: fdb7f8e
pre-migration kernel SHA256: d57fb2252c63818a0058f936ff3ed46b9e7fadba29858f9f47eadb9526d6464b
```

## Scope

- one persistent Triton/TLE kernel launch per rank;
- SM90 FP8 L1 gate/up GEMM, SwiGLU and L2 down-projection;
- two math warp-groups and two TMA producer roles;
- two descriptorless TMA1D D8 token-pull streams;
- NVSHMEM symmetric allocation and NVLink peer access;
- remote L2 scatter and local top-k combine;
- synthetic and immutable Qwen3 FP8 input modes.

Triton/TLE retains metadata/layout handling, the GEMM pipe, WGMMA, scaling,
SwiGLU, L1/L2 math and combine. Raw CUDA is restricted to the D6-D9
descriptorless TMA1D dispatch/pull boundary and the remote L2 scatter boundary.

This is currently a directly runnable integration candidate. It is not yet
registered as a generic vLLM fused-MoE API.

Consequently, the repository's normal public-operator integration requirements
are intentionally deferred: there is no `torch` API contract, top-level export,
runtime registration or generic autotune table in this migration. The launch
shape is the fixed, previously validated UserHopper-aligned configuration. The
co-located host harness uses PyTorch only to allocate inputs and perform
correctness/reference checks; it is not a compute fallback reachable from a
registered FlagGems-vllm operator.

## Compiler dependency

The kernel requires a FlagTree/TLE build containing:

- `buffered_tensor.subslice`;
- multiple pure-TMA writers on one `tle.pipe`;
- multi-writer TMA token `full_count` lowering.

These compiler changes are maintained separately in FlagTree PR #1079. Keep
the exact compiler SHA in every correctness or performance result.

The known H100 validation used the historical PR837-based compiler build. A
new compiler build must pass 8-rank correctness before its timings are compared
with historical data. When reproducing with the historical environment-gated
PR837 build, set `TLE_MULTI_TMA_WRITERS=1` externally; the operator no longer
sets that legacy compiler switch itself.

## Environment

The runner expects CUDA, OpenMPI and NVSHMEM. These variables select non-default
installations:

| Variable | Meaning |
| --- | --- |
| `TLE_PYTHON_OVERRIDE` | Python executable used by spawned MPI workers |
| `MEGAMOE_TORCH_SITE_PACKAGES` | Optional site-packages containing PyTorch/NVSHMEM |
| `NVSHMEM_HOME` | NVSHMEM root containing `include/` and `lib/` |
| `CUDA_HOME` | CUDA toolkit root; defaults to `/usr/local/cuda-12.8` |
| `CLANG` | clang used by TLE raw CUDA compilation |
| `MEGAMOE_MPIRUN` | OpenMPI launcher; defaults to `/usr/bin/mpirun` |
| `W_CACHE_ROOT` | Per-rank Triton cache root |
| `MEGAMOE_BUILD_DIR` | Cache for the host-side NVSHMEM wrapper |

Run the source entry directly with the intended FlagTree Python. Do not wrap
the command in another `mpirun`: the runner launches one worker per rank itself.

## Standard workload

```text
H=4096, I=1536, E=128, topk=8, tokens/rank=512, drop=0, stages=4
```

### Eight-rank synthetic correctness

```bash
MEGAMOE_NP=8 \
W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=0 W_TIMEOUT=900 \
python src/flaggems_vllm/runtime/backend/_nvidia/hopper/mega/megamoe/kernel.py
```

All eight ranks must report:

```text
partials=4096 scatter_bad=0 errors=0 ... -> PASS
```

### Immutable Qwen3 FP8 data

```bash
MEGAMOE_SHARED_DATA_DIR=/path/to/qwen3_fp8_shared \
MEGAMOE_NP=8 \
W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=0 W_TIMEOUT=900 \
python src/flaggems_vllm/runtime/backend/_nvidia/hopper/mega/megamoe/kernel.py
```

### CUDA Event benchmark

Only benchmark after correctness passes on the same compiler and node:

```bash
MEGAMOE_NP=8 \
W_NTOK=512 W_TOPK=8 W_NEXP=128 W_K=4096 W_INTER=1536 \
W_DROP=0 W_STAGES=4 W_BENCH=1 W_WARMUP=10 W_ITERS=30 \
W_BENCH_REDUCE=mean W_GPU_START_BARRIER=1 W_TIMEOUT=900 \
python src/flaggems_vllm/runtime/backend/_nvidia/hopper/mega/megamoe/kernel.py
```

Distributed latency is determined by the slowest rank. Do not compare results
collected with different inputs, compiler revisions, clocks or rank-reduction
rules.

## Validation status

- migrated FlagGems-vllm checkout, H100 synthetic NP2/T128/topk4/E16/K256/I128,
  stages2: 2/2 ranks passed, `ws=True`, `tma=9`, `scatter_bad=0`, `errors=0`;
- H100 synthetic NP2 correctness: 2/2 ranks passed with the historical
  PR837-based compiler;
- H100 Qwen3 NP8/T512 correctness: 8/8 ranks passed;
- independent PyTorch oracle on the validated Qwen3 case: relative L2
  `0.0032209039`, cosine `0.9999948372`.

The implementation is partial, not fully stage-equivalent to UserHopper CUDA:
the full D0-D10 workspace lifecycle, scheduler/cursor dispatch contract and
TMA/mbarrier combine path remain open gaps.
