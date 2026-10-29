---
title: 算子列表
weight: 10
---

# 算子列表

FlagGems-vllm 算子的完整列表与元数据，来源于 `conf/operators.yaml`。

{{< operator-list >}}

## 算子阶段 (Stages)

- **alpha**: 实验阶段，API 可能变动
- **beta**: API 已稳定，正在进行深度性能调优
- **stable**: 生产就绪，性能已验证

## 算子类型 (Kinds)

- **NeuralNetwork**: 神经网络层与核心运算
- **Math**: 数学计算函数
- **LinearAlgebra**: 矩阵与线性代数运算
- **Quantization**: 量化与反量化运算

## 常见标签 (Labels)

- **vLLM**: vLLM 专用算子
- **MoE**: 混合专家路由与计算 (Mixture of Experts)
- **Attention**: 注意力机制 (Flash Attention, MLA 等)
- **Quantization**: FP8, INT8, FP4 等量化支持
- **RoPE**: 旋转位置编码 (Rotary Position Embedding)
- **fused**: 融合算子 (Fused multi-operation kernels)
- **aten**: PyTorch ATen 算子替换

## 使用方式

FlagGems-vllm 依赖 FlagGems，部分算子在底层执行时会调用 FlagGems 中的算子。算子可直接从 `flaggems_vllm` 导入调用，推荐配合 `flag_gems.enable()` 获得完整的算子运行时支持：

```python
import flag_gems
import flaggems_vllm

# 1. 启用 FlagGems 算子（提供底层基础算子与通用调度支持）
flag_gems.enable()

# 2. 直接调用 FlagGems-vllm 特化算子
result = flaggems_vllm.moe_align_block_size(topk_ids, block_size, num_experts)

# 3. 查看已导出的全部可用算子
ops = flaggems_vllm.ops.__all__
print(f"当前已导出: {len(ops)} 个算子")
```

## 下一步

- [基础用法](../usage/basic.md)
- [MiniMax M3 算子支持](minimax-m3-support.md)
- [算子源码仓库](https://github.com/flagos-ai/FlagGems-vllm/tree/main/src/flaggems_vllm/ops)
