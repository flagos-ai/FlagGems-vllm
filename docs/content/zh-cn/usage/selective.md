---
title: 选择性启用
weight: 30
---

# 选择性启用算子

精细化控制 FlagGems-vllm 启用的算子，便于调优性能、排查问题与渐进式上线。

## 典型使用场景

- **调试排错**：隔离特定算子以验证数值正确性与边界表现
- **性能调优**：仅启用在特定机型或特定形状上深度优化的算子
- **渐进上线**：在生产环境中分批、逐步引入新算子
- **基准比对**：将单个特化算子与 PyTorch 原生基线进行 A/B 测试

## `only_enable(include=...)`

仅启用列表中指定的算子：

```python
import flaggems_vllm

# 仅启用 MoE 相关算子
flaggems_vllm.only_enable(include=['moe_align_block_size', 'grouped_topk', 'fused_experts_impl'])
```

未包含在列表中的其他算子保持未启用状态。

## `use_gems(include=..., exclude=...)`

通过上下文管理器实现更细致的作用域控制：

```python
import flaggems_vllm

# 排除特定实验阶段算子
with flaggems_vllm.use_gems(exclude=['experimental_op']):
    result = computation()

# 仅在局部代码块中启用特定算子
with flaggems_vllm.use_gems(include=['flash_attention_forward', 'fused_add_rms_norm']):
    result = attention_computation()
```

## 查询已注册算子

通过以下 API 查看当前注册生效的算子信息：

```python
import flaggems_vllm

flaggems_vllm.enable()

ops = flaggems_vllm.all_registered_ops()
print(f"已注册算子数: {len(ops)}")
print(ops[:10])  # 前 10 个算子

keys = flaggems_vllm.all_registered_keys()
print(f"已注册调度键: {keys}")
```

## 示例：调试单算子精度与行为

```python
import flaggems_vllm
import torch

# 1. 运行 PyTorch 基准
input_tensor = torch.randn(128, 256, device='cuda')
baseline = torch_computation(input_tensor)

# 2. 仅启用待测算子
flaggems_vllm.only_enable(include=['my_operator'])
result = computation_using_my_operator(input_tensor)

# 3. 对比差值
diff = (result - baseline).abs().max()
print(f"最大绝对误差: {diff}")
```

## 示例：生产环境渐进式灰度

```python
import flaggems_vllm

# 第一阶段：仅启用最成熟的稳定算子
STABLE_OPS = [
    'moe_align_block_size',
    'grouped_topk',
    'flash_attention_forward',
    'fused_add_rms_norm',
]

flaggems_vllm.only_enable(include=STABLE_OPS)

# 运行业务负载
# ...
```

## 最佳实践

1. **稳健优先**：在生产环境中优先启用已验证加速比的稳定算子。
2. **逐步定位**：排查精度问题时，通过二分排除法缩小问题算子范围。
3. **性能监控**：结合性能基准测试与选择性启用，防止出现个别形状的性能回归。
4. **代码注释**：在配置 `include` 或 `exclude` 时，明确标注选择或跳过特定算子的原因。

## 下一步

- [基础用法](basic.md)
- [调试与日志](debugging.md)
- [运行性能基准评测](../performance/benchmark.md)
