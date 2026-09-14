---
title: 用法指南
weight: 20
bookCollapseSection: true
---

# 用法指南

学习如何在应用中使用 FlagGems-vllm 算子。

## 基础模式

FlagGems-vllm 算子有三种使用方式:

1. **直接调用**: 从 `flaggems_vllm` 直接调用算子
2. **与 FlagGems 一起使用**: 配合 `flag_gems.enable()` 使用通用算子
3. **通过 vllm-plugin-fl**: 在 vLLM 推理中自动集成

## 主题

- [基础用法](basic/) - 直接算子调用和 API 模式
- [vLLM 插件集成](vllm-plugin/) - 与 vllm-plugin-fl 一起使用
- [选择性启用](selective/) - 细粒度算子控制
- [非 NVIDIA 后端](non-nvidia/) - 在 AMD、昇腾等硬件上运行
- [调试](debugging/) - 日志和故障排除

## 快速示例

```python
import torch
import flaggems_vllm

# MoE grouped topk 带偏置和重归一化
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
