# Gemma 4 routing: portable kernels and Hopper TLE specialization

Branch: `opt/gemma4`, base upstream `af184e5`.
The vLLM reference is frozen at commit
`5d83998e927f3081c804932e6586bd7312a424e8`, `gemma4.py`.
It is used only in test/benchmark reference code.

## Current implementation

- `ops/gemma4_fused_routing_kernel_triton.py` contains portable Triton arithmetic,
  shared helpers, validation, and the generic public function.
- `runtime/backend/_nvidia/hopper/ops/gemma4_fused_routing_kernel_triton.py`
  contains the Hopper-specific TLE kernel and same-name public function.
  The existing backend registrar replaces the top-level package export on Hopper.
- The Hopper implementation checks both Python TLE APIs and compiler builder
  support. Missing TLE support delegates to the portable implementation.
- Small K (K <= 8) uses repeated integer max/min selection for every batch size.
  The former vLLM full-sort production path has been removed.
- Larger K uses a descending, stable lexicographic key and `tl.topk`, operating
  on only the next-power-of-two K selected elements during normalization. Older
  compilers without `tl.topk` use Triton's sort primitive in this independent
  kernel. Neither path calls a vLLM function or a torch compute fallback.

The key innovation is splitting argmax into an integer maximum and a masked
minimum expert ID. On H800 these become `redux.sync` operations instead of
shuffle-based paired reductions. An availability mask preserves index ordering
for all-ones negative NaN keys as well as ordinary ties and signed zero.

Hopper explicitly distributes tokens across warps and experts across the lanes
of one warp using TLE `BlockEncoding`, `SlicedEncoding`, and `set_layout`. This
keeps the selection reduction within one warp, without CTA-wide shared-memory
exchange. The ordinary Triton path shares the arithmetic but does not depend on
TLE or contain Hopper layout code.

## Source-informed design and experiments

- FlagGems `ops/topk.py`: separate value/index selection and an availability mask.
- FlagGems-vllm `ops/topk_sigmoid.py`: batch independent token rows, fuse routing
  and weight computation, and apply padding sentinels in-kernel.
- Hopper `ops/w8a8_block_fp8_bmm.py`: guard optional TLE capability and retain a
  portable fallback.
- FlagTree `python/test/tle/unit/test_tle.py`: explicit distributed block/slice
  encodings and compiler support detection.

The initial controlled experiment compared paired argmax versus max/min, and
implicit versus TLE layouts, holding shape and row/warp candidates fixed. For
`[8192,128], K=8`, the TLE max/min prototype measured 2.20x versus vLLM and 1.63x
versus the previous branch implementation. These preliminary numbers are kept
separate from the final fixed 100-shape qualification.

## Contract and dispatch

| Item | Contract |
|---|---|
| Inputs | CUDA FP16/BF16/FP32 logits [T,E], expert scales [E]; arbitrary positive/zero strides |
| Padding | Optional strided bool mask [T], same device |
| Bounds | T >= 0; 1 <= K <= E <= 1024 |
| Outputs | Fresh contiguous FP32 weights and INT32 IDs [T,K] |
| Weight | Softmax normalized over selected experts, multiplied by learned expert scale |
| Ordering | Logit bit-order descending; lower expert ID on equal keys; +0 before -0 |
| Nonfinite | Reference comparisons include standard NaN/Inf and all-ones NaN payload |
| Padding result | Weight 0, expert ID -1 |
| Empty T | Allocation only, no tuning/launch |
| Errors | ValueError for malformed shapes/devices/K; NotImplementedError for unsupported device/dtypes/width |
| Autograd | Inference forward only |

## Autotuning

All kernels use `@libentry() -> @libtuner() -> @triton.jit`.

| Config | Location | Parameters |
|---|---|---|
| gemma4_routing_native | NVIDIA tune_configs.yaml | (BLOCK_T,num_warps)=(1,1),(4,4),(8,4),(16,4) |
| gemma4_routing_partial | NVIDIA tune_configs.yaml | num_warps=1,4 |
| gemma4_routing_hopper | Hopper tune_configs.yaml | Same token/warp candidates as native |

Keys are exact T/E/K/padding/layout category; LibTuner additionally keys by tensor
dtypes. Token tiles above 4 are pruned when T <= 32. num_stages=1 and num_ctas=1
remain fixed because there is no tiled-load pipeline or cluster communication.
The expert block size is fixed by E, not treated as a free tuning parameter.
The public `num_warps` argument remains a compatibility hint; actual geometry
comes from tuning. Prewarm each shape before CUDA Graph capture. Warm calls use
LibEntry's compiled kernel cache and LibTuner's persistent best configuration.

## Frozen qualification protocol

The previous branch implementation and its measured per-shape best configurations
are frozen separately. Its benchmark harness removes only tuning decorators and
replays the exact old kernel with those configurations; kernel math is unchanged.

100 synthetic FP32 shapes use the original 25 token counts (1 through 32768)
crossed with (E,K)=(64,4),(128,4),(128,8),(256,8). Seed 20261117; half of cases
supply a mask with roughly 10% padding. Each result requires exact expert IDs,
FP32 weight tolerance atol=1e-6/rtol=1e-5, and independent FP64 checks.

vLLM, previous branch, and new Hopper implementation each have a 128-call CUDA
Graph. Five timing rounds rotate the order of all three implementations; each
round is the median of nine CUDA-event measurements. Results report median
microseconds, arithmetic/geometric mean speedup, minimum, all timing rounds,
and selected tuning configs. Compilation/tuning and Python overhead are excluded.
These are operator timings, not model throughput.

Validation commands:

```bash
PYTHONPATH=src python -m pytest -q tests/test_gemma4_fused_routing_kernel_triton.py
PYTHONPATH=src python -m pytest -q benchmark/test_gemma4_fused_routing_kernel_triton.py \
    --level core --mode cudagraph --iter 1 --warmup 1
```

Production audit: torch is used only for allocation, dtype declarations and
metadata. All arithmetic and casting is Triton. No external-library path or
ATen registration is needed; package exports and backend replacement are used.
The referenced optimization.md/deep_opt.md are absent in this checkout.

## Final measured results (2026-11-18)

Environment: H800 physical GPU 2, Docker wy_nv, FlagTree 0.7.0+git3693980e
(Triton API 3.6.0), PyTorch 2.10.0+cu128.

| Metric over all 100 shapes | Result |
|---|---:|
| Arithmetic mean versus vLLM | 1.7003x |
| Geometric mean versus vLLM | 1.5852x |
| Minimum versus vLLM | 1.0670x |
| Maximum versus vLLM | 4.0940x |
| Cases below 0.95x vLLM | 0 |
| Arithmetic mean versus previous branch | 1.2674x |
| Minimum versus previous branch | 1.0653x |

Functional tests: **68 passed**. Repository core benchmark: **1 passed**, five
representative shapes. Cross-process persistent cache: passed for Hopper,
generic selection and partial-top-k, with rebenchmarking disabled.

For [8192,128], K=8, BLOCK_T=4/num_warps=4, generated-code evidence:

| Implementation | Registers | Shared bytes | Redux | Shuffles | Barriers |
|---|---:|---:|---:|---:|---:|
| New Hopper | 32 | 0 | 16 | 8 | 0 |
| Previous branch | 32 | 0 | 0 | 86 | 0 |
| vLLM | 37 | 0 | 0 | 130 | 0 |

Nsight Compute collection succeeded. It reported no local/shared spilling,
zero static shared memory, 32 registers/thread, 100% theoretical occupancy
and 60.66% achieved occupancy for this representative configuration. Instruction
counts above are static PTX counts, not dynamic instruction counts.

A follow-up experiment tested 1/2/4/8 contiguous experts per thread with the
same token/warp schedules. Wider layouts gave no consistent cross-shape gain
and regressed several shapes, so this change was rejected and not added to the
production autotune space. The qualified implementation remains unchanged.

Static audit, Black/isort/flake8, YAML/operator/export checks and whitespace
checks passed. Scope remains inference forward on the validated H800 environment;
backward and other accelerator performance are not claimed.

## Observed model-shape validation, all input dtypes

The 47 unique shapes observed for Gemma 4 26B-A4B across TP=1/2/4/8
were tested separately against the vLLM v0.23.0 routing kernel. These shapes
were collected on H20; operator timing was measured on H800 / wy_nv GPU 2.
The three input dtype variants cover 141 cases. Logits and expert scales use
the same target dtype; output weights remain FP32 and IDs INT32. No padding.

| Input dtype | Shapes | Mean speedup | Minimum | Maximum |
|---|---:|---:|---:|---:|
| FP32 | 47 | 1.2043x | 1.0714x | 2.2288x |
| BF16 | 47 | 1.2228x | 1.0817x | 2.2519x |
| FP16 | 47 | 1.2208x | 1.0785x | 2.2406x |

All 141 cases passed exact expert-ID checks and FP32 weight tolerance
atol=1e-6, rtol=1e-5, including independent FP64 reference comparisons. The
previously documented CUDA Graph timing protocol is retained, alternating
baseline and candidate across five rounds. Aggregation is equal per unique
shape, not weighted by Python observation counts. TP membership is used for
subset summaries; these are not distributed end-to-end model benchmarks.
