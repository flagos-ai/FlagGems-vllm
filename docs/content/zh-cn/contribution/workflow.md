---
title: 算子开发协议
weight: 20
---

# 算子开发协议规范

FlagGems-vllm 算子开发必须严格遵循仓库根目录下 `workflow.md` 规定的开发流程与硬性门禁（G0–G7）。

## 门禁概览 (G0–G7)

- **G0**: 任务明确与协议知情
- **G1**: 严格执行 `NO_TORCH_COMPUTE_FALLBACK` 验证
- **G2**: 编码前三张设计表填写完毕（工程基线、Torch 对齐规范、实现路径设计）
- **G3**: Host 调度函数与边界校验设计
- **G4**: 自动调优（Autotune）集成接入
- **G5**: 内核代码与辅助工具完整实现
- **G6**: 功能单元测试全量通过
- **G7**: 性能验收达标（核心形状加速比 SpeedUp ≥ 0.9）

## 关键核心规则

### 1. 严禁 PyTorch 计算回退 (NO_TORCH_COMPUTE_FALLBACK)

**生产内核执行路径严禁调用任何 PyTorch 计算算子与内存拷贝算子。**

- **禁止项**：`torch.matmul`, `torch.sum`, `torch.softmax`, `tensor.contiguous()`, `tensor.to()`, `tensor.copy_()` 等；
- **允许项**：读取元数据（`.shape`, `.dtype`, `.device`）、未初始化分配（`torch.empty_like()`）、真正零拷贝的 view、以及 Triton 内核发射；
- **不支持路径**：必须显式 `raise NotImplementedError(...)`，绝不能静默调用 PyTorch 计算作为退路。

### 2. 编码前三张设计表

在编写任何实现代码之前，必须先梳理并记录以下三张设计表：

1. **工程基线表**：算子名称、函数签名、vLLM 上下文调用环境、输入输出规格、性能预期；
2. **Torch 对齐规范表**：对标基准行为、浮点容差（rtol/atol）、特殊值（空张量、NaN、Inf、边界维度）；
3. **实现路径设计表**：不同形状/类型的工作负载分类、调度分支策略、不支持情况的处理。

### 3. 自动调优 (Autotune) 规范

凡是暴露 `BLOCK_*`、`num_warps`、`num_stages` 等参数的算子，NVIDIA 后端必须接入自动调优：
- 在 `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` 配置参数空间；
- 在 Triton 内核上使用 `@libtuner(configs=..., key=[...], strategy=[...])` 装饰器。

### 4. 标准代码落点

| 职责 | 落点路径 |
|---------|------|
| 算子实现 | `src/flaggems_vllm/ops/<op>.py` |
| 功能测试 | `tests/test_<op>.py` |
| 性能测试 | `benchmark/test_<op>.py` |
| 元数据录入 | `conf/operators.yaml` |
| 模块导出 | `src/flaggems_vllm/ops/__init__.py` 与 `flaggems_vllm/__init__.py` |
| 自动调优配置 | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` |

## 标准开发流程

### 步骤 1：明确需求并阅读协议
阅读 `workflow.md`，参考相似算子的实现风格。

### 步骤 2：明确规范并填写三张设计表
确定输入输出规格、对齐容差与不支持边界。

### 步骤 3：编写 Host 调度函数与 Triton 内核
- 使用 `@libentry()` 修饰 Host 入口；
- 编写 `@triton.jit` 内核；
- 接入 `@libtuner()`。

### 步骤 4：功能测试验证
在 `tests/test_<op>.py` 中添加参数化测试，确保 `--quick` 和常规模式均通过。

### 步骤 5：性能基准评测
在 `benchmark/test_<op>.py` 中编写评测用例，并在 `benchmark/core_shapes.yaml` 录入典型形状。确保 NVIDIA H100 上的加速比达到 **SpeedUp ≥ 0.9**。

## 下一步

- [阅读根目录 workflow.md 原始规范](https://github.com/flagos-ai/FlagGems-vllm/blob/main/workflow.md)
- [贡献指南概述](overview.md)
- [单元测试指南](../testing/unittest.md)
- [基准测试指南](../performance/benchmark.md)
