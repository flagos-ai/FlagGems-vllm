---
title: 算子开发协议
weight: 20
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# 算子开发协议规范

FlagGems-vllm 算子开发必须严格遵循仓库根目录下 `workflow.md` 规定的开发流程与硬性门禁（G0–G7）。有关新算子交付要求、文件落点、以及 `NO_TORCH_COMPUTE_FALLBACK` 规范，请参阅[贡献概述](overview.md)。

## 门禁概览 (G0–G7)

所有算子新增与修改任务必须通过以下硬性门禁：

- **G0（任务知情）**：明确算子任务范围，确认属于 vLLM 场景特化/融合算子，知情并承诺遵守开发协议；
- **G1（无 Torch 计算回退）**：严格执行 `NO_TORCH_COMPUTE_FALLBACK` 验证，生产路径除内存分配算子（`empty`、`zeros` 等）与创建/拷贝算子（`contiguous`、`clone`、`copy_` 等）外，严禁调用任何 PyTorch 计算算子；
- **G2（编码前三张设计表）**：在编写任何实现前，必须先填写并记录工程基线表、vLLM 对齐规范表与实现路径设计表；
- **G3（Host 调度与边界防御）**：设计 Host 调度函数，明确参数校验、数据类型分支、设备检查及不支持情况的 `NotImplementedError` 抛出；
- **G4（自动调优集成）**：凡包含可调参数的算子，必须在 NVIDIA 后端提供预调优配置并接入 `@libtuner`；
- **G5（完整实现）**：完成 Host 入口、Triton JIT 内核与公共导出（`ops/__init__.py` 及 `flaggems_vllm/__init__.py`）；
- **G6（功能测试通过）**：`tests/test_<op>.py` 在 `--quick` 与全量测试模式下均 100% 通过；
- **G7（性能验收达标）**：`benchmark/test_<op>.py` 评测核心形状加速比达到 **SpeedUp ≥ 0.9**。

## 编码前三张设计表

在编写任何算子实现代码之前，必须先梳理并记录以下三张设计表：

### 1. 工程基线表

| 字段 | 内容要求 |
| --- | --- |
| 算子名称与 ID | 算子 API 名称，与 `conf/operators.yaml` 中的 `id` 保持一致 |
| 函数签名 | 完整的入参与返回值定义 |
| vLLM 上下文 | 调用的模型架构与层（如 DeepSeek-V3 MoE、PagedAttention 等） |
| 依赖关系 | 是否依赖 FlagGems 算子或外部扩展 |
| 目标硬件后端 | 默认 NVIDIA，或特定异构硬件后端 |
| 性能验收目标 | 关键核心形状加速比默认 `SpeedUp >= 0.9` |

### 2. vLLM 对齐规范表

> [!NOTE]
> 本仓库存放 vLLM 仓库算子，不接受通用 PyTorch 算子，因此无需对齐 PyTorch 原生算子，对标对象为 vLLM 现有原生算子、C++/CUDA 扩展或模型推理参考实现。

| 字段 | 内容要求 |
| --- | --- |
| 对标基准实现 | 对齐的 vLLM 原生 C++/CUDA 扩展、Triton 实现或模型算法参考 |
| 浮点容差要求 | 明确 FP32/FP16/BF16/FP8 的 `rtol` 和 `atol` 容差阈值 |
| 边界与特殊值 | 空张量、极端大维度、NaN、Inf、边界维度的处理规则 |
| 布局与连续性 | 支持的连续性与步长要求（非连续张量可使用 `contiguous()`） |

### 3. 实现路径设计表

| 字段 | 内容要求 |
| --- | --- |
| 工作负载分类 | 区分 Prefill 阶段、Decode 阶段或特定批量大小的处理分支 |
| 内核架构策略 | 单内核完成融合计算，或组合调用 FlagGems 算子 |
| 自动调优参数 | 暴露的 `BLOCK_*`、`num_warps`、`num_stages` 参数空间候选值 |
| 不支持范围 | 不支持的形状、数据类型或后端，显式抛出 `NotImplementedError` |

## 静态审计规范

在提交 PR 之前，建议运行静态检查脚本审计生产代码路径：

```bash
python - <<'PY'
from pathlib import Path
op = '<op>'
p = Path(f'src/flaggems_vllm/ops/{op}.py')
if p.exists():
    text = p.read_text()
    suspicious = [pat for pat in ['torch.matmul', 'torch.sum', 'torch.softmax', 'torch.nn.functional', 'torch.ops.aten', '.to('] if pat in text]
    print(f"Audit {p}: suspicious calls = {suspicious}")
PY
```

## 相关链接

- [贡献概述](overview.md)：新算子交付要求、`conf/operators.yaml` 元数据管理、编码规则
- [根目录 workflow.md 协议完整定义](https://github.com/flagos-ai/FlagGems-vllm/blob/main/workflow.md)
- [硬件后端开发指南](backend.md)
