---
title: Usage
weight: 20
bookCollapseSection: true
---

# Usage Guide

Learn how to use FlagGems-vllm operators in your applications.

> [!IMPORTANT]
> **Dependency Note**: FlagGems-vllm depends on FlagGems. Several FlagGems-vllm operators invoke FlagGems operators internally during execution. Ensure `flag_gems` is installed, and call `flag_gems.enable()` before running workloads.

## Usage Patterns

FlagGems-vllm **only supports direct invocation; it does not support PyTorch dispatch registration, nor does it support the `use_gems` context manager**:

1. **Direct invocation (recommended with FlagGems enabled)**: Import `flag_gems`, call `flag_gems.enable()` to initialize the general operator foundation, and call fused operators directly via `flaggems_vllm.<op>()`;
2. **Through vllm-plugin-fl**: Automatic end-to-end integration within vLLM inference by invoking `flaggems_vllm.<op>()`.

## Topics

- [Basic Usage](basic.md) - Direct operator calls and FlagGems cooperation patterns
- [vLLM Plugin Integration](vllm-plugin.md) - Using with vllm-plugin-fl
- [Non-NVIDIA Backends](non-nvidia.md) - Running on AMD, Ascend, and other hardware
- [Debugging](debugging.md) - Logging and troubleshooting

## Quick Example

```python
import torch
import flag_gems
import flaggems_vllm

# Enable FlagGems general operators (required for underlying kernel dispatch)
flag_gems.enable()

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
