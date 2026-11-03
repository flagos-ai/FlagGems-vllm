---
title: 贡献概述
weight: 10
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

# 贡献概述

在拉取请求（Pull Request）中，贡献者应就所提议的变更进行描述，包括变更的原因。如有需要，请一并提交单元测试与性能测试用例。拉取请求被合入之前，需要项目成员的审批。此外，所有拉取请求都必须通过持续集成（CI）检查。

## 1. 拉取请求流程 {#pr-workflow}

1. **创建 Fork 与特性分支**：基于 `main` 分支拉取新分支进行开发；
2. **遵守开发协议**：在实现新算子或修改既有算子时，严格遵从协议规范；
3. **本地代码格式化与自测**：
   ```bash
   pre-commit run --all-files
   PYTHONPATH=src pytest -q tests --quick
   PYTHONPATH=src pytest -q benchmark/test_<op>.py --level core --iter 1
   ```
4. **提交 Pull Request**：
   - 提供清晰的变更描述与对标 Issue；
   - 附带功能测试与性能加速比测试数据；
   - 检查 CI 各项门禁（代码风格、`operators.yaml` 规范、单元测试、基准评测）。

## 2. 设备隔离与可见性 {#device-visibility}

在多人共享的加速卡节点上运行单元测试或性能测试时，建议指定特定设备运行。FlagGems-vllm 依赖各后端厂商的可见性环境变量来完成设备隔离，可见性变量必须在进程启动**之前**设置。

各硬件后端对应的可见性环境变量配置与详细说明，请参阅 FlagGems 官方文档：[设备隔离与可见性](https://flagos-ai.github.io/FlagGems/zh-cn/contribution/device-visibility/)。

## 3. 算子命名规范 {#operator-naming-conventions}

FlagGems-vllm 专为存放 vLLM 仓库的特化与融合算子而设立，**不接受通用 PyTorch 算子**（通用 PyTorch 算子请向 [FlagGems](https://github.com/flagos-ai/FlagGems) 提交）。因此，算子命名应直接对标 vLLM 仓库现有算子或相关模型实现，**不需要对齐 PyTorch**：

- 采用 `snake_case` 命名风格（例如 `grouped_topk`、`moe_align_block_size`、`fused_inv_rope_fp8_quant`）；
- 算子名称直接保持与 vLLM 现有算子 API 或模型层命名一致；
- 算子名称不得以下划线开头（算子名称需与测试采用的自定义 marker 严格对应，由于 pytest marker 规则的限制，不允许包含前缀下划线）；
- 算子名称中不得包含点号；
- 测试文件与基准测试文件命名必须与算子保持一致（`tests/test_<op>.py`、`benchmark/test_<op>.py`）。

## 4. 算子元信息管理 {#operator-metadata-management}

FlagGems-vllm 在 `conf/operators.yaml` 中维护所有算子的元数据清单（Operator Inventory）。仓库 CI 会对 `conf/operators.yaml` 进行严格校验（包含字段完整性、无重复 ID 以及按 `id.casefold()` 字典序排序等）。每个算子的字段说明如下：

| 字段 | 类型 | 用途 | 约束 |
| --- | --- | --- | --- |
| `id` | string | 唯一标识 FlagGems-vllm 仓库中的算子 | 必须与算子 API 名称一致，如 `grouped_topk`；采用 snake_case，不能以下划线开头 |
| `description` | string | 算子功能、算法、输入输出及推理场景的描述（支持多行 YAML 块文本） | 需清晰阐述在 vLLM 推理架构中的作用及支持的精度细节 |
| `for` | array of string | 标识该算子所替代或对标的 vLLM 算子或函数；若无直接对标则填 `None` | 例如 `_moe_C.grouped_topk`、`_C.apply_repetition_penalties_`，无需对标 PyTorch 算子 |
| `labels` | array of string | 在不同维度对算子进行标签分组（非空列表） | 常用标签：`vLLM`, `fused`, `MoE`, `Attention`, `NoCPU`, `KernelGen` 等 |
| `kind` | array of string 或 string | 算子的主要功能类别 | 例如 `MoE`, `NeuralNetwork`, `Quantization`, `Normalization` 等 |
| `stages` | array of key-value pairs | 记录算子的成熟度演化历史（非空列表） | 键为成熟度阶段（`alpha`, `beta`, `stable`），值为引入/晋升的版本号（如 `'5.0'`） |

算子成熟度阶段定义如下：
- 新开发或移植的融合算子通常以 `alpha` 或 `beta` 阶段起步；
- 当算子经过全面测试、接入自动调优并在真实推理负载与全量 CI 中稳定运行后，晋升至 `stable` 阶段。

配置示例（`conf/operators.yaml`）：

```yaml
  - id: grouped_topk
    description: |
      A specialized routing mechanism used in Mixture-of-Experts (MoE) models (like DeepSeek-V3/R1)
      to select top-k experts by first grouping them, rather than selecting globally.
    for:
      - _moe_C.grouped_topk
    labels:
      - fused
      - NoCPU
      - vLLM
    kind:
      - MoE
    stages:
      - stable: '5.0'
```

## 5. 新算子开发交付要求 {#operator-deliverables}

在着手开发新算子之前，首先应查阅 `conf/operators.yaml` 确认是否已有同类实现。

> [!IMPORTANT]
> **仅限 vLLM 算子，不接受 PyTorch 算子**：
> FlagGems-vllm 专为存放 vLLM 仓库特化与融合算子而设立，**不接受通用 PyTorch 算子**（通用 PyTorch 算子请向 [FlagGems](https://github.com/flagos-ai/FlagGems) 贡献）。因此，开发算子**不需要对齐 PyTorch** 原生算子语义，只需对齐 vLLM 仓库现有算子、原生 CUDA/C++ 扩展或模型推理参考实现。

**新算子开发交付必须满足以下完整要求**：

1. **填写 `conf/operators.yaml` 算子元数据信息**：
   - 必须在 `conf/operators.yaml` 中新增算子条目；
   - 必须完整填写 `id`, `description`, `for`, `labels`, `kind`, `stages` 六项必填字段；
   - 确保 `id` 唯一且按字母序排列，并通过 `python3 tools/ci_checks/check_operators_yaml.py` 静态检查。
2. **算子实现与 Host 调度 (`src/flaggems_vllm/ops/<op>.py`)**：
   - 编写 Host 调度函数与 Triton 内核函数；
   - 使用 `@libentry()` 修饰入口函数，做好输入张量形状、步长、数据类型与设备校验；
   - **严格遵守 NO_TORCH_COMPUTE_FALLBACK 规范**：允许调用 PyTorch 内存算子（`torch.empty`、`torch.zeros` 等）以及创建/拷贝算子（`contiguous`、`clone`、`copy_` 等），允许调用 FlagGems 算子，禁止调用 PyTorch 算子实现计算功能；
   - 不支持的形状或数据类型必须显式抛出 `NotImplementedError`，严禁静默回退。
3. **公共接口导出**：
   - 在 `src/flaggems_vllm/ops/__init__.py` 中导入该算子并加入 `__all__` 列表；
   - 在 `src/flaggems_vllm/__init__.py` 中完成顶层包导出。
4. **自动调优配置 (`src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml`)**：
   - 凡是暴露 `BLOCK_*`、`num_warps`、`num_stages` 等性能参数的算子，必须在 `tune_configs.yaml` 中提供候选参数空间；
   - 在 Triton 内核函数上使用 `@libtuner(configs=..., key=[...], strategy=[...])` 装饰器接入。
5. **功能单元测试 (`tests/test_<op>.py`)**：
   - 测试函数必须添加 `@pytest.mark.<op>` 装饰器；
   - 参数化覆盖典型形状、数据类型（如 FP16/BF16/FP8 等）与边界情况；
   - 单元测试在 `--quick` 快速模式与全量模式下均必须通过；
   - 精度对齐验证对标 vLLM 原生算子或参考实现，使用 `tests.accuracy_utils`（`gems_assert_close` / `gems_assert_equal`）。
6. **性能基准评测 (`benchmark/test_<op>.py` 及 `benchmark/core_shapes.yaml`)**：
   - 在 `benchmark/test_<op>.py` 中编写基准测试用例；
   - 在 `benchmark/core_shapes.yaml` 录入典型核心形状；
   - **性能验收标准**：在 NVIDIA H100 等目标设备上，核心形状的加速比达到 **SpeedUp ≥ 0.9**。

### 标准文件落点清单

| 交付项目 | 标准落点路径 | 说明 |
| --- | --- | --- |
| 算子元数据 | `conf/operators.yaml` | 填写算子元数据条目（强制要求） |
| 算子实现与调度 | `src/flaggems_vllm/ops/<op>.py` | Host 调度入口与 Triton 内核 |
| 自动调优配置 | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` | NVIDIA 候选参数空间与调优策略 |
| 功能单元测试 | `tests/test_<op>.py` | 正确性与边界测试（带 `@pytest.mark.<op>`） |
| 性能基准测试 | `benchmark/test_<op>.py` | 性能评测脚本 |
| 核心测试形状 | `benchmark/core_shapes.yaml` | 评测的核心输入形状列表 |
| 模块导出 | `src/flaggems_vllm/ops/__init__.py` | 模块内部导出并加入 `__all__` |
| 顶层导出 | `src/flaggems_vllm/__init__.py` | 包级别对外公开接口直接导出 |

## 6. 严禁 PyTorch 计算回退规则 (NO_TORCH_COMPUTE_FALLBACK) {#host-dispatch-conventions}

**生产内核执行路径允许调用 PyTorch 框架中的 `torch.empty`、`torch.zeros` 等内存算子以及 `contiguous`、`clone`、`copy_` 等创建或拷贝算子，除此之外禁止调用 torch 算子实现计算功能。可以调用 FlagGems 的算子。**

### 允许调用

- **PyTorch 内存分配算子**：允许调用 `torch.empty`、`torch.empty_like`、`torch.zeros`、`torch.zeros_like`、`torch.empty_strided` 等内存分配算子；
- **创建与拷贝算子**：允许调用 `tensor.contiguous()`、`tensor.clone()`、`tensor.copy_()` 等创建或拷贝算子；
- **FlagGems 算子**：可以调用 FlagGems (`flag_gems`) 库中已有的高性能算子（如通用矩阵乘法、逐元素计算等）；
- **元数据与视图操作**：允许读取元数据（`.shape`, `.stride`, `.dtype`, `.device` 等）以及真正零拷贝的 view 操作；
- **Triton 内核发射**：调用自研 Triton JIT 内核。

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

除上述内存分配算子、创建/拷贝算子与元数据/视图操作外，**禁止调用任何 PyTorch 算子实现计算功能**（包含计算核函数、隐式类型转换等）：

```python
# ❌ 生产路径严禁调用 PyTorch 算子实现计算功能
def my_operator(x):
    return torch.matmul(x, x.T)  # ❌ 禁止调用 PyTorch 计算算子
    return x.sum(dim=-1)          # ❌ 禁止调用 PyTorch 计算算子
    return torch.softmax(x, dim=-1) # ❌ 禁止调用 PyTorch 计算算子
    return x.to(torch.float32)    # ❌ 触发类型转换内核
```

### 不支持配置的处理

遇到不受支持的形状或数据类型时，必须显式抛出 `NotImplementedError`，绝不能静默调用 PyTorch 算子进行回退计算：

```python
if x.dtype not in [torch.float16, torch.bfloat16]:
    raise NotImplementedError(f"Unsupported dtype: {x.dtype}")
```

PyTorch 计算仅允许在 `tests/`、`benchmark/` 以及显式声明的 Reference 参考实现中调用。

## 7. 代码规范与格式化 {#code-style}

提交代码前请安装并运行 `pre-commit`：

```bash
pip install pre-commit
pre-commit run --all-files
```

规范工具：
- **black** (行宽 88)
- **isort** (`--profile black`)
- **flake8** (最大行宽 120，忽略 `F405,E731,W503,E203,E704`)

## 8. 算子单元测试 {#operator-unit-tests}

在 `tests/test_<op>.py` 中添加单元测试：
- 使用 `@pytest.mark.<op>` 标记测试用例；
- 覆盖典型推理形状与数据类型；
- 使用 `tests.accuracy_utils`（`gems_assert_close` / `gems_assert_equal`）对比参考实现（如 vLLM 原生算子或 PyTorch 参考实现）；
- 支持 `--quick` 快速测试与全量测试：
  ```bash
  pytest -v -s tests/test_<op>.py
  pytest -q tests/test_<op>.py --quick
  ```

## 9. 算子性能评测 {#operator-performance-benchmarking}

在 `benchmark/test_<op>.py` 中添加性能评测：
- 使用 `@pytest.mark.parametrize("shape", Config.get_shapes("<op>"))`；
- 在 `benchmark/core_shapes.yaml` 录入代表性推理形状；
- **加速比定义**：`SpeedUp = latency_vllm_baseline / latency_flaggems`；
- **验收目标**：关键核心形状下达到 **SpeedUp ≥ 0.9**；
- 运行测试命令：
  ```bash
  pytest -q benchmark/test_<op>.py --level core --iter 1 --warmup 1
  ```

## 10. 算子开发协议规范 {#development-protocol}

算子开发与优化必须遵从仓库根目录的 `workflow.md` 执行协议。详见[算子开发协议规范](workflow.md)。
