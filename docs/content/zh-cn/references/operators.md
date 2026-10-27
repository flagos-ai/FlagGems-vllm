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

所有算子均可直接从 `flaggems_vllm` 导入调用：

```python
import flaggems_vllm

# 直接调用
result = flaggems_vllm.moe_align_block_size(topk_ids, block_size, num_experts)

# 启用算子后查看已注册的算子列表
flaggems_vllm.enable()
ops = flaggems_vllm.all_registered_ops()
print(f"当前已注册: {len(ops)} 个算子")
```

## 下一步

- [基础用法](/FlagGems-vllm/zh-cn/usage/basic/)
- [M3 算子支持](/FlagGems-vllm/zh-cn/references/m3-support/)
- [算子源码仓库](https://github.com/flagos-ai/FlagGems-vllm/tree/main/src/flaggems_vllm/ops)
