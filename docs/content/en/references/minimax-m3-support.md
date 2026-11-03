---
title: MiniMax M3 Support
weight: 15
---

# MiniMax M3 Support and Quantization Fusion

This document details operator support specifications, quantization fusion, validation, and performance benchmarks for MiniMax M3 model inference workloads.

## Operator Specifications

| Operator | Supported Specification |
|---|---|
| `fp32_router_gemm` | BF16/FP32 `[M,6144]`, FP32 `[128,6144]`, M<=32; fresh FP32 logits, fixed FMA/reduction, no TF32/atomics, 2D strides, empty batches. Unsupported dimensions are rejected. |
| `topk_sigmoid` | FP16/BF16/FP32 contiguous `[M,E]`, E<=256, K<=min(32,E); optional FP32 correction bias affects selection, not output probability; stable expert tie convention, exact source-row IDs, optional renormalization, NaN padding, mutable caller outputs. |
| `fused_minimax_m3_qknorm_rope_kv_insert` | Contiguous BF16 packed QKV, D128, Gemma RMSNorm, NeoX rotary prefix 8/16/32/64/128, optional Q/index-Q gather, strided NHD/HND paged main cache, BF16 index cache, BF16 or identity-scale E4M3 main cache. Negative slots skip writes. Caller must supply valid positions, cache bounds, and unique nonnegative write destinations. |
| `vit_flash_attn_wrapper` / `vision_flash_attn_varlen` | Noncausal FP16/BF16 D64/80 attention; 3D packed or 4D batch inputs, Triton strided packing/uniform cumulative lengths, finite scale including zero/negative, device sequence boundaries remain replay inputs. Uses existing varlen attention. |
| OAI MoE | Explicit `swigluoai_uninterleave` plus finite alpha/beta/clamp parameters, propagated through all expert wrappers; preserves the existing activation names. |
| Dynamic INT8 fusion | BF16/FP16 2D rows up to K8192; optional OAI+quant2 retains the rounded activation workspace. Plain Torch and complete FlagGems quantization registrations retain their respective arithmetic; mixed registrations retain the existing unfused path. |
| Short INT8 reductions | Nonblock W8A8, unfused activation and K<=1024 use INT32 dot accumulation. Maximum absolute sum 2^24 converts exactly to FP32. Larger K, block quantization, FP8, and fused-SiLU paths retain their previous schedule. |

### Environment Flags

`FLAG_GEMS_W8A8_QUANT_FUSION=quant|quant_oai` is opt-in and read before capture. Default is off. Set it before importing the package:

```bash
export FLAG_GEMS_W8A8_QUANT_FUSION=quant
```

Fixed launch exceptions preserve reduction order, row ownership, mutation safety, and the staged quantization boundaries; existing attention/MoE GEMM tuning remains unchanged.

New production paths use Triton and metadata/empty allocations, without Torch compute/copy/cast fallback. The port preserves the existing upstream MoE legacy paths, including their unfused quantization behavior. New APIs are forward inference only; autograd and unlisted layouts/dtypes are not promised.

## Validation

**Platform: NVIDIA H100**

- **214 targeted numerical tests**: passed, including:
  - 36 native CUDA cache-writer comparisons with exact visible output bytes for BF16/E4M3, NHD/HND, three rotary sizes, and three token counts.
  - 12 full expert OAI cases covering fused/unfused quantization, both half dtypes, and the K1024 accumulation boundary.
  - Mutable buffers, empty inputs, ties/NaN routing, dynamic CUDA Graph inputs/sequence boundaries, strided inputs, invalid parameter validation, and both expert wrappers.
- **3 fullgraph compilation boundary tests**: passed. Fullgraph tests use eager backend to validate fake kernels and mutation boundaries.
- **6 existing W8A8 expert regression tests**: passed.

## Synthetic Performance

CUDA Graph timing setup: 25 warmup calls, 100 captured calls, 5 event samples, 2 reversed-order rounds. Values below are mean round medians.

References are explicit in JSON:
- Router baseline uses literal FP32 `F.linear` with TF32 disabled.
- Quantization baseline uses the full Torch chain.
- ViT uses optimized Torch SDPA.
- Cache writer uses native M3 CUDA fused operator with identical clone/reset overhead on both sides.

| Operator / shape | Baseline (μs) | Candidate (μs) | Speedup |
|---|---:|---:|---:|
| `fp32_router_gemm` [1, 6144, 128] | 7.263 | 2.271 | **3.199×** |
| `fp32_router_gemm` [4, 6144, 128] | 14.172 | 3.489 | **4.061×** |
| `fp32_router_gemm` [8, 6144, 128] | 15.760 | 3.945 | **3.995×** |
| `fp32_router_gemm` [16, 6144, 128] | 16.264 | 5.068 | **3.209×** |
| `fp32_router_gemm` [32, 6144, 128] | 15.375 | 6.997 | **2.197×** |
| `per_token_int8_quant` [1, 6144] | 18.380 | 5.426 | **3.387×** |
| `per_token_int8_quant` [64, 6144] | 33.058 | 5.558 | **5.948×** |
| `per_token_int8_quant` [4096, 6144] | 446.016 | 60.878 | **7.326×** |
| `per_token_int8_quant` [5089, 768] | 66.883 | 12.298 | **5.438×** |
| `per_token_int8_quant` [8192, 768] | 107.506 | 18.474 | **5.819×** |
| `vit_attention` [1, 128, 16, 64] | 6.599 | 5.196 | **1.270×** |
| `vit_attention` [1, 512, 16, 80] | 12.282 | 14.998 | 0.819× |
| `vit_attention` [2, 256, 16, 80] | 9.345 | 10.783 | 0.867× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [1, 4, 2, 2, 128] | 3.042 | 3.107 | 0.979× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [4, 4, 2, 2, 128] | 3.224 | 3.175 | 1.016× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [64, 4, 2, 2, 128] | 3.573 | 3.410 | 1.048× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [4096, 4, 2, 2, 128] | 21.257 | 35.222 | 0.604× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [8192, 4, 2, 2, 128] | 45.699 | 72.405 | 0.631× |

## Limits and Scheduling Notes

- The router result above compares against PyTorch `F.linear`, not a native specialized router.
- The cache writer remains slower than native CUDA on large prefill batches.
- ViT attention provides a CUDA-graph-safe extension-free adapter; at D80 shapes, optimized SDPA remains faster.
- Further native prefill/attention tuning can be tracked in benchmark updates.

## Related Links

- [Operator List](operators.md)
- [Project Structure](../contribution/project-structure.md)
- [Benchmark Results](../performance/results.md)
