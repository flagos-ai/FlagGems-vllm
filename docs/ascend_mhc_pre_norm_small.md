# Ascend small-row mHC Pre with RMSNorm

`flaggems_vllm.mhc_pre_norm_small` is a forward-only, three-kernel GLM5Next
decode mHC Pre + RMSNorm operator. Residual is contiguous BF16 `[N,4,4096]`,
0<=N<=4; fn is contiguous FP32 `[24,16384]`, scales/base are FP32 `[3]`/`[24]`,
and norm_weight is BF16 `[4096]`. All tensors use the same Ascend device and
standard ND storage. Outputs: FP32 post `[N,4,1]`, FP32 comb `[N,4,4]`, BF16
layer input `[N,4096]`. Empty N returns allocated empty outputs. No input is
modified. Other backends/dtypes/layouts/training are unsupported; malformed
shapes/device mismatches/scalar parameters raise ValueError.

The three kernels perform split-K FP32 projection and squared sum, small
coefficient/Sinkhorn mixing, and premix plus RMSNorm. Premix rounds to BF16
before RMSNorm. Projection does not round fn or the projection to BF16: this
matches native GLM5Next npu_hc_pre_v2, and is why this is a separate entry
rather than silently replacing generic mhc_pre_with_norm's BF16 projection.
Pre-gate epsilon and Sinkhorn epsilon remain independently configurable.
Non-finite inputs propagate through arithmetic. No backward or implicit dtype
conversion is provided. No cache, observer, model patch or native library is
needed by the production operator; all arithmetic is Triton.

Autotune exemption: Ascend-only, no NVIDIA implementation/config. The previous
validated tile BK=1024/BG=4 is retained; BS=16 and BD=4096 are semantic reduction
extents. This port deliberately preserves the three-kernel launch/rounding
schedule rather than introducing new tuning. No physical core count is fixed.

Pytest covers N=0/1/2/3/4, independent epsilons and iteration counts, zero/small/
large/random activations, FP64 CPU oracle, optional native GLM5Next comparison,
invalid dtype/layout/device/shape and changed-input NPUGraph replay. Every
output is checked separately. Benchmark compares native HcPre + RMSNorm and
reports baseline/candidate SpeedUp for N=1/2/3/4. No serving test is required.

workflow.md is followed; its referenced optimization.md/deep_opt.md are absent
from this main revision. Scope is integration of a previously validated kernel,
not a new multi-round optimization. Python-source build uses PYTHONPATH=src;
validation and benchmarks run on a single 910C using the A3 image.

Scalar epsilons, multipliers and iteration counts must be host numbers. NPU
scalar tensors are rejected before conversion, avoiding implicit host reads.

## Validation on Ascend 910C (2026-11-15)

- Status: complete for the documented Ascend BF16 inference contract.
- Public entry: `flaggems_vllm.mhc_pre_norm_small`; `ops.__all__` feeds the existing
  top-level wildcard export. The public entry is exercised by every test.
- Full unit suite: **42 passed**, zero skipped; quick mode:
  **32 passed**, zero skipped. Tests include scoped dispatch-log
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
PYTHONPATH=src python -m pytest -q tests/test_mhc_pre_norm_small.py
PYTHONPATH=src python -m pytest -q tests/test_mhc_pre_norm_small.py --quick
PYTHONPATH=src python -m pytest -q benchmark/test_mhc_pre_norm_small.py --level core --iter 10 --warmup 5
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
**1.704x**, and every shape exceeds the repository's 0.9 threshold.

| Active shape | Native ms | Candidate ms | SpeedUp |
|---|---:|---:|---:|
| N=1 | 0.030844 | 0.017078 | 1.806 |
| N=2 | 0.036061 | 0.018212 | 1.980 |
| N=3 | 0.036428 | 0.021640 | 1.683 |
| N=4 | 0.035945 | 0.025663 | 1.401 |

Autotune: the Ascend-only fixed-tile exemption above applies; no NVIDIA path is
advertised. Deep optimization was not triggered: this is a direct integration
of the validated algorithm and no active shape regressed. Remaining scope is
limited to the documented dtype/layout/shape and forward inference. Other
chip configurations and end-to-end model integration were not revalidated in
this change. Large profiler artifacts remain on the test machine.
