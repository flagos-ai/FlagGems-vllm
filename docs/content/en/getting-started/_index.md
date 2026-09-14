---
title: Getting Started
weight: 10
bookCollapseSection: true
---

# Getting Started with FlagGems-vllm

This section covers installation, setup, and basic usage of FlagGems-vllm.

## Prerequisites

- Python ≥ 3.8 (CI uses 3.11/3.12)
- PyTorch ≥ 2.6.0
- Triton or FlagTree compiler
- CUDA-capable GPU (for NVIDIA backend) or other supported accelerator
- vLLM (optional, for integration testing)

## Installation

See the [installation guide](install/) for detailed instructions.

## Quick Start

After installation, verify FlagGems-vllm is working:

```python
import torch
import flaggems_vllm

print(f"FlagGems-vllm version: {flaggems_vllm.__version__}")
print(f"Vendor: {flaggems_vllm.vendor_name}")
print(f"Device: {flaggems_vllm.device}")

# Test a simple operator
topk_ids = torch.randint(0, 16, (128, 2), device='cuda', dtype=torch.int32)
sorted_ids, expert_ids, num_tokens = flaggems_vllm.moe_align_block_size(
    topk_ids, block_size=32, num_experts=16
)
print("✓ moe_align_block_size works!")
```

## Next Steps

- [Installation details](install/)
- [Basic usage patterns](/FlagGems-vllm/usage/basic/)
- [Integration with vllm-plugin-fl](/FlagGems-vllm/usage/vllm-plugin/)
