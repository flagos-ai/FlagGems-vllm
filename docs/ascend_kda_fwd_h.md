# Ascend resident KDA H stage

The public `flaggems_vllm.chunk_kda_fwd_h` API implements the prepared H stage
of vLLM-Ascend's GLM5Next KDA prefill, not a full attention layer. It consumes
contiguous BF16 BNSD k/w/u `[1,H,T,128]`, FP32 base-2 chunk-local gk of the same
shape, FP32 initial VK state `[N,H,128,128]`, and a host sequence of cumulative
lengths. It returns BF16 snapshots `[1,sum(ceil(length/64)),H,V,K]`, BF16
corrected values in u's layout and FP32 final VK state. Inputs are read-only.

Each program keeps one sequence/head/value tile's FP32 state across 64-token
chunks. Snapshots and corrected values round to BF16 before the two matrix
products; decay is exp(gk_last * ln(2)), followed by separate multiply/add.
Empty sequences preserve their initial state; partial chunks mask padded rows.
Non-finite values propagate through arithmetic. No backward is provided.

Host dispatch rejects non-Ascend devices, non-BF16 operands, non-FP32 state/gk,
nonstandard or noncontiguous storage, B!=1, K/V!=128 and device-side metadata.
Malformed shapes and cumulative lengths raise ValueError; unsupported backend,
dtype/layout or training inputs raise NotImplementedError. There is no torch
compute fallback, native-op dependency or runtime D2H metadata read.

The API intentionally differs from generic FLA chunk_delta_h, which uses a
K-first state and another gate/layout contract. It is exported independently
rather than changing that operator's numerical semantics. The kernel is adapted
from the previously validated GLM5Next H implementation (Apache-2.0).

Autotune exemption: Ascend-only path, no NVIDIA implementation/configuration.
The default value tile 128 retains the measured large-prefill configuration;
32/64 are explicit alternatives covered by correctness tests. CHUNK=64 and
DIM=128 are semantic extents. No blind autotune trial changes the recurrent
rounding schedule. The backend CANN dot extension is imported lazily.

Validation uses the repository pytest entry, an independent CPU FP64 oracle
with explicit BF16 materialization, native FLA-NPU comparison when available,
ragged/empty/tail/long inputs, input immutability, invalid arguments and changed
input NPUGraph replay. Benchmark uses the matching native H stage and reports
SpeedUp=baseline/candidate, with output allocations in both paths. Neither
vLLM model serving nor an end-to-end benchmark is required for this change.

Development protocol: workflow.md is present; the referenced optimization.md
and deep_opt.md are absent from this main revision. Scope is a direct port of
a validated algorithm, with no new multi-round tuning. Build is Python source
via PYTHONPATH=src; validation and benchmarks use one 910C on the A3 image.

The FP64 oracle envelope follows the native H-stage validation (atol=1e-3,
rtol=1e-2), with additional NRMSE <= 1e-3. FP64 is not the native BF16/FP32
execution order: at 4096 tokens the native and candidate both differ by up to
7.292e-4 in final state against FP64, while all three native/candidate outputs
are bitwise equal. Separate native parity tests require bitwise equality and
cover short ragged, 4096-token and 8193-token inputs.

Compiler compatibility: use CANN extension.dot when it is provided; FlagTree
3.5 uses tl.dot with the same BF16 inputs/FP32 accumulator. The
`enable_graph_optimize=False` option is only passed when supported by the
active compiler. No torch compute path is used in either case.

Scalar options and cumulative-length elements must be host numbers. NPU scalar
tensors are rejected before conversion, avoiding implicit device-to-host reads.

## Validation on Ascend 910C (2026-11-15)

- Status: complete for the documented Ascend BF16 inference contract.
- Public entry: `flaggems_vllm.chunk_kda_fwd_h`; `ops.__all__` feeds the existing
  top-level wildcard export. The public entry is exercised by every test.
- Full unit suite: **46 passed**, zero skipped; quick mode:
  **37 passed**, zero skipped. Tests include scoped dispatch-log
  assertions, invalid contracts, numerical references and changed-input graph
  replay. The optional native reference was available and executed.
- Build: `pip wheel --no-deps --no-build-isolation .` passed. All configured
  pre-commit hooks and operator metadata/export/marker/reference CI checks pass.
- Production audit: no torch computation or native-op fallback. Torch calls
  allocate outputs/scratch; shape/device/dtype/format reads are metadata. Casts
  in kernel bodies are Triton operations. Unsupported paths raise explicitly.
- Environment: Ascend910_9382 (24 AIC/48 AIV), AArch64, Python 3.12.16,
  CANN 9.2.1, torch 2.10.0+cpu, torch_npu 2.10.0.post7,
  FlagTree 0.7.1a1+ascend3.5 / Triton 3.5.1, cann-shmem 1.6.0;
  native FLA-NPU A3 26.9.1. Customer 20 AIC/40 AIV was not tested.

```shell
PYTHONPATH=src python -m pytest -q tests/test_chunk_kda_fwd_h.py
PYTHONPATH=src python -m pytest -q tests/test_chunk_kda_fwd_h.py --quick
PYTHONPATH=src python -m pytest -q benchmark/test_chunk_kda_fwd_h.py --level core --iter 10 --warmup 5
```

The validation container's older system libc required an isolated official
Ubuntu 2.39 user-space loader/libstdc++ overlay for the published FlagTree
wheel. Python and the unchanged FlagTree triton-mlir-opt binary were launched
through that loader. System libraries and compiler sources were not modified.
The GLM5Next native reference additionally needs its installed custom OPP path
registered before Python starts; this is not a production-op dependency.

Benchmark: four core shapes, BF16, default fixed tile configuration, same
native H-stage or HcPre+RMSNorm boundary; unit is **ms**. These are repository
**kernel-mode** measurements (`do_bench_npu`, torch_npu.profiler fallback when
mspti is absent), not host-inclusive operator time or whole-model latency.
The native/candidate outputs are checked before each shape is timed.
SpeedUp is baseline/candidate; geometric mean over the frozen set is
**2.240x**, and every shape exceeds the repository's 0.9 threshold.

| Active shape | Native ms | Candidate ms | SpeedUp |
|---|---:|---:|---:|
| T=4096, H=16 | 0.829574 | 0.319260 | 2.598 |
| T=8192, H=16 | 1.792488 | 0.674827 | 2.656 |
| T=16384, H=32 | 3.920507 | 2.775195 | 1.413 |
| T=8193, H=16 | 1.782658 | 0.690108 | 2.583 |

Autotune: the Ascend-only fixed-tile exemption above applies; no NVIDIA path is
advertised. Deep optimization was not triggered: this is a direct integration
of the validated algorithm and no active shape regressed. Remaining scope is
limited to the documented dtype/layout/shape and forward inference. Other
chip configurations and end-to-end model integration were not revalidated in
this change. Large profiler artifacts remain on the test machine.
