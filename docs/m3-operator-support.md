# M3 inference operator support and quantization fusion

## Contracts

| Operator | Supported contract |
|---|---|
| `fp32_router_gemm` | BF16/FP32 `[M,6144]`, FP32 `[128,6144]`, M<=32; fresh FP32 logits, fixed FMA/reduction, no TF32/atomics, 2D strides, empty batches. Unsupported dimensions are rejected. |
| `topk_sigmoid` | FP16/BF16/FP32 contiguous `[M,E]`, E<=256, K<=min(32,E); optional FP32 correction bias affects selection, not output probability; stable expert tie convention, exact source-row IDs, optional renormalization, NaN padding, mutable caller outputs. |
| `fused_minimax_m3_qknorm_rope_kv_insert` | Contiguous BF16 packed QKV, D128, Gemma RMSNorm, NeoX rotary prefix 8/16/32/64/128, optional Q/index-Q gather, strided NHD/HND paged main cache, BF16 index cache, BF16 or identity-scale E4M3 main cache. Negative slots skip writes. Caller must supply valid positions, cache bounds, and unique nonnegative write destinations. |
| `vit_flash_attn_wrapper` / `vision_flash_attn_varlen` | Noncausal FP16/BF16 D64/80 attention; 3D packed or 4D batch inputs, Triton strided packing/uniform cumulative lengths, finite scale including zero/negative, device sequence boundaries remain replay inputs. Uses existing varlen attention. |
| OAI MoE | Explicit `swigluoai_uninterleave` plus finite alpha/beta/clamp parameters, propagated through all expert wrappers; preserves the existing activation names. |
| Dynamic INT8 fusion | BF16/FP16 2D rows up to K8192; optional OAI+quant2 retains the rounded activation workspace. Plain Torch and complete FlagGems quantization registrations retain their respective arithmetic; mixed registrations retain the existing unfused path. |
| Short INT8 reductions | Nonblock W8A8, unfused activation and K<=1024 use INT32 dot accumulation. Maximum absolute sum 2^24 converts exactly to FP32. Larger K, block quantization, FP8, and fused-SiLU paths retain their previous schedule. |

`FLAG_GEMS_W8A8_QUANT_FUSION=quant|quant_oai` is opt-in and read before capture. Default is off. Set it before importing the package. Fixed launch exceptions preserve reduction order, row ownership, mutation safety, and the staged quantization boundaries; existing attention/MoE GEMM tuning remains unchanged.

New production paths use Triton and metadata/empty allocations, without Torch compute/copy/cast fallback. The port preserves the existing upstream MoE legacy paths, including their unfused quantization behavior. New APIs are forward inference only; autograd and unlisted layouts/dtypes are not promised.

## Validation

H100: 214 targeted numerical tests, three fullgraph compilation boundary tests, and six existing W8A8 expert regression tests passed. The 214 include 36 native CUDA cache-writer comparisons with exact visible output bytes for BF16/E4M3, NHD/HND, three rotary sizes and three token counts; 12 full expert OAI cases cover fused/unfused quantization, both half dtypes and the K1024 accumulation boundary. Also checked: mutable buffers, empties, ties/NaN routing, changing CUDA Graph inputs/sequence boundaries, strided inputs, invalid parameters, and both expert wrappers. Fullgraph tests use the eager backend to validate fake kernels and mutation boundaries; they do not claim Inductor coverage.

## Synthetic performance

CUDA Graph timing: 25 warmup calls, 100 captured calls, five event samples, two reversed-order rounds. Values below are mean round medians. References are explicit in JSON: router uses literal FP32 F.linear with TF32 disabled; quantization uses the full Torch chain; ViT uses optimized Torch SDPA; cache writer uses the native M3 CUDA fused operator with identical clone/reset cost on both sides. These are separate references and are not aggregated into model throughput.

| Operator / shape | Baseline us | Candidate us | Speedup |
|---|---:|---:|---:|
| `fp32_router_gemm` [1, 6144, 128] | 7.263 | 2.271 | 3.199x |
| `fp32_router_gemm` [4, 6144, 128] | 14.172 | 3.489 | 4.061x |
| `fp32_router_gemm` [8, 6144, 128] | 15.760 | 3.945 | 3.995x |
| `fp32_router_gemm` [16, 6144, 128] | 16.264 | 5.068 | 3.209x |
| `fp32_router_gemm` [32, 6144, 128] | 15.375 | 6.997 | 2.197x |
| `per_token_int8_quant` [1, 6144] | 18.380 | 5.426 | 3.387x |
| `per_token_int8_quant` [64, 6144] | 33.058 | 5.558 | 5.948x |
| `per_token_int8_quant` [4096, 6144] | 446.016 | 60.878 | 7.326x |
| `per_token_int8_quant` [5089, 768] | 66.883 | 12.298 | 5.438x |
| `per_token_int8_quant` [8192, 768] | 107.506 | 18.474 | 5.819x |
| `vit_attention` [1, 128, 16, 64] | 6.599 | 5.196 | 1.270x |
| `vit_attention` [1, 512, 16, 80] | 12.282 | 14.998 | 0.819x |
| `vit_attention` [2, 256, 16, 80] | 9.345 | 10.783 | 0.867x |
| `fused_minimax_m3_qknorm_rope_kv_insert` [1, 4, 2, 2, 128] | 3.042 | 3.107 | 0.979x |
| `fused_minimax_m3_qknorm_rope_kv_insert` [4, 4, 2, 2, 128] | 3.224 | 3.175 | 1.016x |
| `fused_minimax_m3_qknorm_rope_kv_insert` [64, 4, 2, 2, 128] | 3.573 | 3.410 | 1.048x |
| `fused_minimax_m3_qknorm_rope_kv_insert` [4096, 4, 2, 2, 128] | 21.257 | 35.222 | 0.604x |
| `fused_minimax_m3_qknorm_rope_kv_insert` [8192, 4, 2, 2, 128] | 45.699 | 72.405 | 0.631x |

## Limits and scheduling trial

The router result above compares against F.linear, not the native specialized router. The cache writer remains slower than native CUDA on large prefill batches. ViT remains slower than optimized SDPA at the D80 shapes; it provides a Graph-safe extension-free adapter. Neither result is advertised as a universal native-kernel speedup. Further native prefill/attention tuning is outside this operator support port.

A focused cache scheduling trial tested one and two warps per head, preserving the literal reduction. One warp was retained after the complete numerical suite and native bitwise checks. Compiled shared memory decreased from 512 bytes at four warps to 256 bytes at one warp. Large-batch Graph latency improved relative to the previous four-warp port, while the native comparison above still exposes the remaining gap. No NCU hardware-counter claim is made.

Reproduce targeted tests:

```sh
PYTHONPATH=src python -m pytest -q tests/test_fp32_router_gemm.py tests/test_topk_sigmoid.py tests/test_fused_minimax_m3_qknorm_rope_kv_insert.py tests/test_vit_attention.py tests/test_fused_moe.py
PYTHONPATH=src python -m pytest -q tests/test_m3_compile_boundaries.py
PYTHONPATH=src python -m pytest -q tests/test_fused_experts_impl.py -k 'test_fused_moe_int8 and not w8a16' --quick
```

The native reference test skips if the M3 stable-ABI vLLM extension is unavailable. Functional tests against literal formulas and changed Graph inputs remain active. Corresponding repository benchmarks are included; their default timing mode is separate from the explicit Graph JSON report.

The exact Graph timing protocol is runnable without serving artifacts:

```sh
VLLM_PLUGINS= PYTHONPATH=src python benchmark/m3_graph_benchmark.py --output m3-operator-graph-results.json
```
