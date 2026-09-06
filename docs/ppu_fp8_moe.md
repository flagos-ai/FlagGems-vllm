# THead FP8 W8A16 fused Marlin MoE

The THead specialization is `runtime/backend/_thead/fused/fused_marlin_moe_fp8.py`.
Call the public `flaggems_vllm.fused_marlin_moe` with `QUANT_TYPE_FP8_E4M3`
from `flaggems_vllm.ops.fused_marlin_moe`. This constant matches native
`vllm.scalar_type.scalar_types.float8_e4m3fn.id`.

## Contract

- Activations `[M, K]`: contiguous FP16/BF16.
- Weights: output-major E4M3FN tensors or raw uint8 codes, W1 `[E, 2N, K]`,
  W2 `[E, K, N]`. E5M2 and FNUZ encodings are unsupported.
- Scales: ordinary FP16/BF16/FP32 values, S1 `[E, 2N, G1]`, S2 `[E, K, G2]`.
  Group sizes 32/64/128 give G=input_size/group_size; group size -1 is
  channelwise scaling with G=1. Do not fold the native Marlin exponent bias
  into these public input scales.
- K/N are positive multiples of 32; grouped dimensions must be divisible
  by group_size. K tails within a 128-K tile and strided weights/scales are supported.
- Contiguous IDs `[M, T]` are int32/int64 and must name valid local experts.
  Matching router weights may be FP16/BF16/FP32, with 1<=T<=E.
- Output matches activation shape, dtype and device. Empty, validated output
  buffers and inplace operation are supported, with SiLU and both routing-weight placements.
- Forward inference only. W8A8/FP8 activations, expert maps, bias, additional
  global scales, zero points, custom callbacks/workspaces and autograd are unsupported.

## Implementation

Triton packs four FP8 bytes per INT32 word in four interleaved 32-K subtiles.
TLE AIU transfers move packed weights and activations. The whole 128-K tile
is decoded using an exact FP16 bit representation, with FP32 arithmetic and
FP16/BF16 matrix operands. GEMMs accumulate in FP32. Full floating-point
expert weights are not materialized.

Small routed batches use direct routing, fused GEMM1/SwiGLU, and fused
GEMM2/router-weight/top-k reduction for M<=2. Grouped paths stage activations
in contiguous expert order and fuse SwiGLU with GEMM2 input staging.
Shape-specific PPU policies select block sizes, grouping, warps and stages.

Packing also produces GPU safety flags. Weight flags detect E4M3FN NaNs;
scale flags detect finite scales that overflow FP32 when multiplied by 256.
The fast kernel applies this power-of-two correction to the scale vector
before K broadcasting and skips the weight-NaN predicate only when the flags
prove that transformation safe. The general Triton kernel preserves all
special-value semantics for other inputs. NaN/Inf scales, signed zero,
subnormals and scale-folding overflow are covered by tests.

The two variants use mutually exclusive guards on the same stream. Grouped
kernels use per-expert flags; direct kernels use conservative whole-tensor
flags because their reduction can span experts. No CPU readback selects the path.
An inactive variant returns before loading activation/weight tiles.

Weak weight/scale caches track ordinary tensor versions, including the safety
flags. Inference-mode tensors must be immutable after first use. Cold graph
captures recompute packing/flags and never publish graph-owned tensors to
global caches. Warm up before capture to reuse packing; recapture after
changing weights/scales used by a warm graph.

No Torch compute fallback is introduced. Host Torch use is limited to empty
allocations, metadata/device queries and no-copy views. Grouped alignment uses
the existing native vLLM PPU extension. NVIDIA compute and tuning are unchanged.

## Validation

143 passed, 4 skipped. The four skipped checks require native FP8 Marlin
configurations for group32/64 small shapes unavailable in the installed version.
Independent references cover those supported PPU paths. Coverage includes
all E4M3FN encodings, actual packed-tile decoding, FP16/BF16, all supported
groups, tails, layouts, mixed fast/general experts, cache updates, output
validation, warm/cold graphs and cold-replay safety-path changes.

Every one of the 53 production-trace shapes also passes a same-input native
Marlin comparison. A separate same-seed accuracy run measured maximum absolute
difference 0.046875 and maximum mean relative
error 0.005580, below the 0.04 criterion.

## Performance and reproduction

Validated environment: PPU-ZW810E, Torch 2.9.0, FlagTree/Triton 3.6.0 with
TLE B32 lowering, PPU SDK 2.1, and THead vLLM
`0.13.1.dev0+g72506c983.d20260218`. INT32 AIU transfers require the capability
described in [FlagTree issue #1050](https://github.com/flagos-ai/FlagTree/issues/1050).

The benchmark follows [FlagGems PR #5140](https://github.com/flagos-ai/FlagGems/pull/5140):
E=256, K=4096, N=256, top_k=6, BF16, group_size=128; 53 shapes with 10,105
total call-count weight. Both implementations consume corresponding layouts
of identical finite E4M3 values, fold-safe scales and routing data. Native scale
permutation and exponent-bias folding happen in benchmark setup.

- Kernel weighted speedup: 3.224x; range 1.176x–4.521x.
- CUDA Graph weighted speedup: 3.243x; range 1.166x–4.535x.

Full results are in [ppu_fp8_trace.csv](ppu_fp8_trace.csv). Weighted speedup
is `sum(calls*native_latency)/sum(calls*optimized_latency)`. Warm timings
exclude first JIT, packing and safety-flag generation. These are operator
measurements, not full model throughput; end-to-end plugin serving is not validated.

Activate the validated environment and matching FlagGems package, then run:

```bash
export GEMS_VENDOR=thead
export VLLM_PLUGINS=
export PYTHONPATH=src:$PYTHONPATH
python -m pytest -q tests/test_fused_marlin_moe_w8a16_fp8_ppu.py
python -m pytest -qs benchmark/test_fused_marlin_moe_w8a16_fp8.py --mode kernel --warmup 1000 --iter 100
python -m pytest -qs benchmark/test_fused_marlin_moe_w8a16_fp8.py --mode cudagraph --iter 100
```
