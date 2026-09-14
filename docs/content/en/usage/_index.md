---
title: Usage
weight: 20
bookCollapseSection: true
---

# Usage Guide

Learn how to use FlagGems-vllm operators in your applications.

## Basic Patterns

FlagGems-vllm operators can be used in three ways:

1. **Direct invocation**: Call operators directly from `flaggems_vllm`
2. **With FlagGems enabled**: Use alongside `flag_gems.enable()` for general ops
3. **Through vllm-plugin-fl**: Automatic integration in vLLM inference

## Topics

- [Basic Usage](basic/) - Direct operator calls and API patterns
- [vLLM Plugin Integration](vllm-plugin/) - Using with vllm-plugin-fl
- [Selective Enablement](selective/) - Fine-grained operator control
- [Non-NVIDIA Backends](non-nvidia/) - Running on AMD, Ascend, and other hardware
- [Debugging](debugging/) - Logging and troubleshooting

## Quick Example

```python
import torch
import flaggems_vllm

# MoE grouped topk with bias and renormalization
scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
bias = torch.randn((16,), device="cuda", dtype=torch.float32)

topk_weights, topk_ids = flaggems_vllm.grouped_topk(
    scores,
    n_group=4,
    topk_group=2,
    topk=2,
    renormalize=True,
    routed_scaling_factor=1.0,
    bias=bias,
    scoring_func=0,
)

print(topk_weights.shape, topk_ids.shape)  # (8, 2), (8, 2)
```
