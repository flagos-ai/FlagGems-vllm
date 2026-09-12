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

All operators are exported from `flaggems_vllm`:

```python
import flaggems_vllm

# Direct call
result = flaggems_vllm.moe_align_block_size(topk_ids, block_size, num_experts)

# Check availability
ops = flaggems_vllm.all_registered_ops()
print(f"Available: {len(ops)} operators")
```

## Next Steps

- [Basic usage patterns](/FlagGems-vllm/usage/basic/)
- [Operator source code](https://github.com/flagos-ai/FlagGems-vllm/tree/main/src/flaggems_vllm/ops)
