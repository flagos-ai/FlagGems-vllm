---
title: 快速开始
weight: 10
bookCollapseSection: true
---

# FlagGems-vllm 快速开始

本节介绍 FlagGems-vllm 的安装、设置和基本使用。

## 前置要求

- Python ≥ 3.8(CI 使用 3.11/3.12)
- PyTorch ≥ 2.6.0
- Triton 或 FlagTree 编译器
- 支持 CUDA 的 GPU(NVIDIA 后端)或其他支持的加速器
- vLLM(可选,用于集成测试)

## 安装

详见[安装指南](install/)。

## 快速验证

安装后验证 FlagGems-vllm 是否正常工作:

```python
import torch
import flaggems_vllm

print(f"FlagGems-vllm 版本: {flaggems_vllm.__version__}")
print(f"硬件后端: {flaggems_vllm.vendor_name}")
print(f"设备: {flaggems_vllm.device}")

# 测试一个简单的算子
topk_ids = torch.randint(0, 16, (128, 2), device='cuda', dtype=torch.int32)
sorted_ids, expert_ids, num_tokens = flaggems_vllm.moe_align_block_size(
    topk_ids, block_size=32, num_experts=16
)
print("✓ moe_align_block_size 工作正常!")
```

## 下一步

- [安装详情](install/)
- [基础用法](/FlagGems-vllm/usage/basic/)
- [与 vllm-plugin-fl 集成](/FlagGems-vllm/usage/vllm-plugin/)
