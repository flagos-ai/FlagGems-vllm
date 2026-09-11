---
title: Benchmark Results
weight: 20
useTabulator: true
---

# Benchmark Results

Performance results for FlagGems-vllm operators across backends and architectures.

## Interactive Results

{{< benchmark-table >}}

## Result Interpretation

- **SpeedUp > 1.0**: FlagGems-vllm is faster than the baseline
- **SpeedUp = 1.0**: Performance parity
- **SpeedUp < 1.0**: Baseline is faster (optimization opportunity)

**Target**: SpeedUp ≥ 0.9 for production operators on core shapes.

## Benchmark Conditions

- **Device**: NVIDIA H100 (default CI runner)
- **Shapes**: Core shapes from `benchmark/core_shapes.yaml`
- **Iterations**: 100 (warmup: 20)
- **Baseline**: PyTorch reference implementations or vLLM native ops

## Per-Operator Highlights

### MoE Operators

| Operator | SpeedUp (H100) | Notes |
|----------|----------------|-------|
| `moe_align_block_size` | 1.2-1.5× | Pre-tuned for block_size=32,64,128 |
| `grouped_topk` | 1.1-1.3× | Optimized for n_group=4,8 |
| `fused_experts_impl` | 1.0-1.2× | Fused matmul + activation |

### Attention Operators

| Operator | SpeedUp (H100) | Notes |
|----------|----------------|-------|
| `flash_attention_forward` | 0.95-1.05× | Parity with Flash Attention 2 |
| `flash_attn_varlen_func` | 0.98-1.1× | Variable-length optimization |
| `triton_unified_attention` | 1.0-1.15× | Multi-head attention fusion |

### Quantization

| Operator | SpeedUp (H100) | Notes |
|----------|----------------|-------|
| `per_token_group_quant_fp8` | 1.3-1.6× | FP8 quantization with grouping |
| `scaled_int8_quant` | 1.2-1.4× | INT8 symmetric quantization |

## Backend Comparison

Performance varies by backend. NVIDIA has the most mature tuning:

- **NVIDIA**: Pre-tuned autotune configs for Ampere/Hopper
- **AMD**: Baseline tuning, ongoing optimization
- **Ascend**: Experimental, FlagTree compiler required

## Reproducing Results

Run the same benchmarks used in CI:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q benchmark \
    --level core --iter 100 --warmup 20 \
    --record --output my_results.json
```

## Next Steps

- [Run your own benchmarks](benchmark/)
- [Contribute optimizations](/FlagGems-vllm/contribution/overview/)
