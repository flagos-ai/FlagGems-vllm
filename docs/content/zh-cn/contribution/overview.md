---
title: 贡献概述
weight: 10
---

# 贡献概述

向 FlagGems-vllm 贡献代码与文档的指南与约定。

## 贡献前须知

1. **阅读 `workflow.md`**：算子开发协议规范是新增或修改算子的强制要求；
2. **检索既有 Issue**：确认是否已有相关 PR 或 issue 正在推进；
3. **重大改动提前沟通**：架构调整或重大改动建议先提交 Issue 讨论。

## 贡献类型

### 1. 新增算子

**必读**：仓库根目录的 `workflow.md`（算子开发协议）。

流程：
1. 确认该算子属于 vLLM 推理场景的特化或融合算子；
2. 编写编码前三张设计表（工程基线、Torch 对齐规范、实现路径设计）；
3. 实现 host 调度与 Triton 内核函数；
4. 接入自动调优（NVIDIA 后端必须提供调优配置）；
5. 编写功能测试用例（`tests/test_<op>.py`）；
6. 编写性能评测用例（`benchmark/test_<op>.py`）；
7. 在 `conf/operators.yaml` 补充算子元数据；
8. 在 `src/flaggems_vllm/ops/__init__.py` 及 `flaggems_vllm/__init__.py` 中导出。

**验收标准**：
- 单元测试在 `--quick` 和全量模式下均通过；
- NVIDIA H100 核心形状下 **SpeedUp ≥ 0.9**；
- 严格遵循 `NO_TORCH_COMPUTE_FALLBACK` 规则（允许调用 `torch.empty`、`torch.zeros` 等内存算子以及 `contiguous`、`clone`、`copy_` 等创建或拷贝算子，除此之外禁止调用 torch 算子实现计算功能，可以调用 FlagGems 算子）。

### 2. 优化既有算子

1. 使用 Profiler 工具分析瓶颈；
2. 提出优化方案并验证精度无回归；
3. 扩展或更新自动调优配置；
4. 提交前后性能对比数据。

### 3. 贡献异构硬件后端

详见[后端开发指南](backend.md)。

## 代码规范与格式化

提交代码前请运行 pre-commit：

```bash
pre-commit run --all-files
```

规范工具：
- **black** (行宽 88)
- **isort** (`--profile black`)
- **flake8** (最大行宽 120)

## 文件落点规范

新增算子 `<op>` 的标准落点如下：

| 目的 | 文件路径 |
|---------|------|
| 算子实现 | `src/flaggems_vllm/ops/<op>.py` |
| 功能测试 | `tests/test_<op>.py` |
| 性能测试 | `benchmark/test_<op>.py` |
| NVIDIA 自动调优配置 | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` |
| 算子元数据 | `conf/operators.yaml` |
| 接口导出 | `src/flaggems_vllm/ops/__init__.py` + `src/flaggems_vllm/__init__.py` |

## 严禁 PyTorch 计算回退规则 (NO_TORCH_COMPUTE_FALLBACK)

**生产内核执行路径允许调用 PyTorch 框架中的 `torch.empty`、`torch.zeros` 等内存算子以及 `contiguous`、`clone`、`copy_` 等创建或拷贝算子，除此之外禁止调用 torch 算子实现计算功能。可以调用 FlagGems 的算子。**

- **允许 PyTorch 内存与创建/拷贝算子**：允许调用 PyTorch 框架中的 `torch.empty`、`torch.empty_like`、`torch.zeros`、`torch.zeros_like`、`torch.empty_strided` 等内存分配算子，以及 `tensor.contiguous()`、`tensor.clone()`、`tensor.copy_()` 等创建或拷贝算子；
- **允许调用 FlagGems 算子**：可以调用 FlagGems (`flag_gems`) 库中已有的高性能算子（如通用矩阵乘法、逐元素计算等）；
- **元数据与视图操作**：允许读取元数据（`.shape`, `.stride`, `.dtype`, `.device` 等）以及真正零拷贝的 view 操作；
- **禁止 PyTorch 计算算子**：除上述内存/创建/拷贝算子、元数据与视图操作外，禁止调用任何 PyTorch 算子实现计算功能（如 `torch.matmul`、`torch.sum`、`torch.softmax` 等）；
- **不支持配置的处理**：遇到不支持的形状或类型应显式抛出 `NotImplementedError`，绝不能静默调用 PyTorch 算子进行回退计算。

### 允许调用

```python
# ✓ 允许的操作
import flag_gems
import torch

def my_operator(x, weight):
    # 元数据读取:
    shape = x.shape              # ✓
    dtype = x.dtype              # ✓
    device = x.device            # ✓

    # 允许调用 PyTorch 内存算子 (torch.empty, torch.zeros 等):
    output = torch.empty_like(x) # ✓
    buf = torch.zeros((shape[0], 64), dtype=dtype, device=device) # ✓

    # 允许调用创建或拷贝算子 (contiguous, clone, copy_ 等):
    x_contig = x.contiguous()    # ✓
    cloned = x.clone()           # ✓
    output.copy_(cloned)         # ✓

    # 零拷贝 view:
    view = x.view(new_shape)     # ✓

    # 允许调用 FlagGems 算子:
    gemm_out = flag_gems.mm(x, weight)  # ✓

    # Triton 内核发射:
    triton_kernel[grid](x, ...)  # ✓
```

### 严禁调用

```python
# ❌ 生产路径严禁调用 PyTorch 算子实现计算功能
def my_operator(x):
    return torch.matmul(x, x.T)  # ❌ 禁止调用 PyTorch 计算算子
    return x.sum(dim=-1)          # ❌ 禁止调用 PyTorch 计算算子
    return torch.softmax(x, dim=-1) # ❌ 禁止调用 PyTorch 计算算子
    return x.to(torch.float32)    # ❌ 触发类型转换内核
```

### 不支持配置的处理

遇到不受支持的形状或数据类型时，显式抛出 `NotImplementedError`：

```python
if x.dtype not in [torch.float16, torch.bfloat16]:
    raise NotImplementedError(f"Unsupported dtype: {x.dtype}")
```

PyTorch 计算仅允许在 `tests/`、`benchmark/` 以及显式声明的 Reference 参考实现中调用。

## 下一步

- [算子开发协议](workflow.md)
- [后端开发指南](backend.md)
- [单元测试指南](../testing/unittest.md)
