---
title: 用法指南
weight: 20
bookCollapseSection: true
---

# 用法指南

学习如何在应用中使用 FlagGems-vllm 算子。

> [!IMPORTANT]
> **依赖说明**：FlagGems-vllm 依赖 FlagGems。当前部分 FlagGems-vllm 算子在底层执行时会直接调用 FlagGems 中的算子。在安装和使用 FlagGems-vllm 时，需确保环境中已安装 `flag_gems`，并推荐在运行前调用 `flag_gems.enable()` 激活底层算子调度。

## 使用方式

FlagGems-vllm **只支持直接调用方式，不支持向 PyTorch 注册分发，亦不支持 `use_gems` 上下文管理器**：

1. **直接导入调用（推荐配合 FlagGems）**: 导入 `flag_gems` 并调用 `flag_gems.enable()` 激活通用算子底座，随后直接通过 `flaggems_vllm.<op>()` 调用特化/融合算子；
2. **通过 vllm-plugin-fl 自动集成**: 在 vLLM 推理服务中通过插件层自动挂接并调用 `flaggems_vllm.<op>()`。

## 主题

- [基础用法](basic.md) - 直接算子调用与 FlagGems 协同模式
- [vLLM 插件集成](vllm-plugin.md) - 与 vllm-plugin-fl 一起使用
- [非 NVIDIA 后端](non-nvidia.md) - 在 AMD、昇腾等硬件上运行
- [调试](debugging.md) - 日志和故障排除

## 快速示例

```python
import torch
import flag_gems
import flaggems_vllm

# 启用 FlagGems（FlagGems-vllm 部分算子依赖其底层算子支持）
flag_gems.enable()

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
