---
title: Operator List
weight: 10
---

# Operator List

Complete list of FlagGems-vllm operators with metadata from `conf/operators.yaml`.

{{< operator-list >}}

## Operator Stages

- **alpha**: Experimental, API may change
- **beta**: Stable API, performance tuning in progress
- **stable**: Production-ready with validated performance

## Operator Kinds

- **NeuralNetwork**: Neural network layers and operations
- **Math**: Mathematical functions
- **LinearAlgebra**: Matrix operations
- **Quantization**: Quantization and dequantization

## Common Labels

- **vLLM**: vLLM-specific operators
- **MoE**: Mixture of Experts routing and computation
- **Attention**: Attention mechanisms (Flash, MLA, etc.)
- **Quantization**: FP8, INT8, FP4 quantization
- **RoPE**: Rotary Position Embedding
- **fused**: Fused multi-operation kernels
- **aten**: PyTorch ATen operator replacements

## Usage

FlagGems-vllm depends on FlagGems, and some operators call FlagGems operators internally. Operators can be called directly, and it is recommended to enable FlagGems for underlying operator support:

```python
import flag_gems
import flaggems_vllm

# 1. Enable FlagGems general operators
flag_gems.enable()

# 2. Direct call to FlagGems-vllm operator
result = flaggems_vllm.moe_align_block_size(topk_ids, block_size, num_experts)

# 3. Inspect all exported operators
ops = flaggems_vllm.ops.__all__
print(f"Available: {len(ops)} operators")
```

## Next Steps

- [Basic usage patterns](../usage/basic.md)
- [Operator source code](https://github.com/flagos-ai/FlagGems-vllm/tree/main/src/flaggems_vllm/ops)
