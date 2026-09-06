# THead MXFP4 fused Marlin MoE

The THead backend exposes MXFP4 W4A16 inference through
`flaggems_vllm.fused_marlin_moe_w4a16_mxfp4`. Use `QUANT_TYPE_FP4_E2M1` from
`flaggems_vllm.ops.fused_marlin_moe` and `group_size=32`.

## Input and output contract

- Activations: contiguous FP16/BF16 `[M, K]`.
- Weights: output-major uint8, two E2M1 nibbles per byte, low nibble first.
  W1 is `[E, 2N, K/2]`; W2 is `[E, K, N/2]`.
- Scales: uint8 E8M0 codes or `torch.float8_e8m0fnu`.
  S1 is `[E, 2N, K/32]`; S2 is `[E, K, N/32]`.
- K and N are positive multiples of 32; partial 128-K tiles are padded.
  Strided weight and scale tensors are supported.
- Contiguous routing IDs `[M, topk]` are int32/int64 and must name valid
  local experts. Routing weights have matching shape and FP16/BF16/FP32 dtype.
- Output is `[M, K]` in the activation dtype. Empty inputs, validated output
  buffers and inplace output are supported.
- SiLU, both routing-weight placements, and forward inference are supported.
  Bias, expert maps, custom activation/reduction, additional global scales,
  zero points, caller-provided workspaces and autograd are unsupported.

## Implementation

The existing backend-dispatch mechanism selects the specialization in
`src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe_w4a16_mxfp4.py`.
The generic operator is unchanged.

- TLE AIU asynchronous block-pointer loads transfer activations and INT32
  containers holding packed 4-bit weights.
- A whole 128-K tile is decoded through broadcast/reshape with one 4xN
  scale load. E2M1 is decoded through its exact FP16 bit representation,
  rescaled in FP32, and converted to the activation dtype for the matrix product.
- Small routed batches use direct routing, fused GEMM1/SwiGLU, and a fused
  GEMM2/top-k reduction for M<=2. Larger batches use expert grouping,
  contiguous activation staging, and fused SwiGLU/GEMM2-input staging.
- GEMMs accumulate in FP32. Shape-specific policies select tile sizes,
  program grouping, warp counts and pipeline stages.
- Triton repacks weights and decodes/transposes E8M0 scales into an FP32 cache.
  The weights remain 4-bit; full floating-point expert weights are not materialized.
- Weak caches track ordinary tensor versions. Inference-mode weights/scales
  must be immutable after first use because they lack version counters.
  Cold CUDA Graph captures do not publish graph-owned packed buffers.
  Warm up before capture to reuse packing, and recapture after changing weights/scales.

The new production path contains no Torch compute fallback. Torch is used
only for uninitialized allocations, metadata/device queries and no-copy views.
Grouped alignment uses the existing native vLLM PPU extension. NVIDIA kernels
and tuning are unchanged; the explicit policies are specific to the PPU backend.

## Validation and performance

82 functional cases passed, including FP16/BF16, every E2M1/E8M0 code combination,
actual packed-tile decoding, signed zero, subnormals, overflow, NaN, K tails,
strided weights/scales, routing skew, output alias checks, cache invalidation,
and warm/cold CUDA Graph capture. Finite decoded output bits match a CPU FP64 oracle.

The production-trace benchmark follows [FlagGems PR #5140](https://github.com/flagos-ai/FlagGems/pull/5140):
E=256, K=4096, N=256, topk=6, BF16; 53 shapes with 10,105 total call-count weight.
Each shape is compared with native vLLM Marlin MXFP4 using the same packed values,
E8M0 scales, top-k IDs and FP32 router weights in the corresponding layouts.

- Kernel mode: weighted speedup 3.203x, range
  1.305x–4.502x.
- CUDA Graph mode: weighted speedup 3.216x, range
  1.304x–4.514x.
- The separate same-seed accuracy pass over all 53 shapes measured maximum
  absolute difference 0.0234375 and maximum mean
  relative error 0.0051507 (threshold 0.04).

Full timing results are in [ppu_mxfp4_trace.csv](ppu_mxfp4_trace.csv).
Weighted speedup is `sum(calls * native_latency) / sum(calls * optimized_latency)`.
Warm timings exclude first compilation and packing. These are operator results,
not full model-serving throughput.

## Reproduce

Validated environment: PPU-ZW810E, PyTorch 2.9.0, FlagTree/Triton 3.6.0 with
TLE/PPU B32 lowering, PPU SDK 2.1, and THead vLLM
`0.13.1.dev0+g72506c983.d20260218`.
Asynchronous INT32 AIU transfers require the compiler capability described in
[FlagTree issue #1050](https://github.com/flagos-ai/FlagTree/issues/1050).

Activate that environment and make the matching FlagGems package available, then run:

```bash
export GEMS_VENDOR=thead
export VLLM_PLUGINS=
export PYTHONPATH=src:$PYTHONPATH
python -m pytest -q tests/test_fused_marlin_moe_w4a16_mxfp4_ppu.py
python -m pytest -qs benchmark/test_fused_marlin_moe_w4a16_mxfp4_ppu_trace.py --mode kernel --warmup 1000 --iter 100
python -m pytest -qs benchmark/test_fused_marlin_moe_w4a16_mxfp4_ppu_trace.py --mode cudagraph --iter 100
```

The benchmark uses the repository `base.Benchmark` interface and median timings.
The source-to-test mapping selects both the functional test and performance targets.
No compiler or SDK source is changed by this implementation.
