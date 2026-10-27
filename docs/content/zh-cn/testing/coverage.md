---
title: 测试覆盖率
weight: 20
---

# 测试覆盖率

FlagGems-vllm 对各算子实现的测试覆盖程度与验收标准。

## 算子覆盖分级

算子测试覆盖度按开发阶段划分：
- **核心算子 (Core Ops)**：90%+ 覆盖率（如 `grouped_topk`、`moe_align_block_size`、`flash_attention_forward` 等）
- **模型特化算子 (Model-specific Ops)**：70%–90% 覆盖率（如 `qwen4_*`、`deepseek_v4_*`）
- **实验性算子 (Experimental Ops)**：50%–70% 覆盖率

## 覆盖度考核维度

1. **形状维度覆盖**：涵盖 vLLM 推理典型形状（Prompt 阶段长序列 prefill、Decode 阶段小 batch 极短序列）；
2. **数据类型覆盖**：`float16`、`bfloat16`、`float32`、`int8`、`fp8` 等；
3. **边界情况覆盖**：空张量（batch 为 0）、单元素/单 token 维度、极限专家数、跨步不连续切片等。

## 生成覆盖率报告

```bash
# 生成覆盖率数据
PYTHONPATH=src pytest tests --cov=flaggems_vllm --cov-report=html

# 打开 HTML 查看详情
open htmlcov/index.html
```

## 拓展测试用例

为新算子或既有算子补齐覆盖率时，应包含：

```python
# 1. 扩充形状与边界
@pytest.mark.parametrize("shape", [
    (128, 256),     # 标准常规形状
    (1, 256),       # 单 batch 边界
    (0, 256),       # 空 batch 边界
])
# 2. 扩充数据类型
@pytest.mark.parametrize("dtype", [
    torch.float16,
    torch.bfloat16,
    torch.float32,
])
def test_my_op(shape, dtype):
    ...
```

## 各阶段覆盖率目标

- **alpha**: ≥ 50%（基础功能通路打通）
- **beta**: ≥ 70%（主流形状与主流数据类型完备）
- **stable**: ≥ 90%（覆盖全部边缘形状、动态 Graph 兼容与长时间稳定性）

## 下一步

- [单元测试指南](unittest.md)
- [贡献指南概述](../contribution/overview.md)
