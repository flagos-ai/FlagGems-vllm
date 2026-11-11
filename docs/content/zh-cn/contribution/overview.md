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

在拉取请求（Pull Request）中，贡献者应就所提议的变更进行描述，包括变更的原因。如有需要，请一并提交单元测试与性能测试用例。拉取请求被合入之前，需要项目维护者的审批。此外，所有拉取请求都必须通过持续集成（CI）检查。

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

FlagGems-vllm 专为存放 vLLM 仓库的特化与融合算子而设立，**不接受通用 PyTorch 算子**（通用 PyTorch 算子请向 [FlagGems](https://github.com/flagos-ai/FlagGems) 提交）。因此，算子命名应直接保持与 vLLM 现有算子 Python API 命名一致，采用 `snake_case` 命名风格（例如 `grouped_topk`、`moe_align_block_size`、`fused_inv_rope_fp8_quant`），**不需要对齐 PyTorch**：

- 算子名称直接保持与 vLLM 现有算子 Python API 命名一致，采用 `snake_case` 命名风格（例如 `grouped_topk`、`moe_align_block_size`、`fused_inv_rope_fp8_quant`）；
- 算子名称不得以下划线开头（算子名称需与测试采用的自定义 marker 严格对应，由于 pytest marker 规则的限制，不允许包含前缀下划线）；
- 算子名称中不得包含点号；
- 测试文件与基准测试文件命名必须与算子保持一致（`tests/test_<op>.py`、`benchmark/test_<op>.py`），CI 工具 `tools/select_tests.py` 依赖此命名约定自动筛选测试目标。

## 4. 算子元信息管理 {#operator-metadata-management}

FlagGems-vllm 在 `conf/operators.yaml` 中维护所有算子的元数据清单（Operator Inventory）。仓库 CI 会对 `conf/operators.yaml` 进行严格校验（包含字段完整性、无重复 ID 以及按 `id.casefold()` 字典序排序等）。每个算子的字段说明如下：

| 字段 | 类型 | 用途 | 约束 |
| --- | --- | --- | --- |
| `id` | string | 唯一标识 FlagGems-vllm 仓库中的算子 | 必须与算子 API 名称一致，如 `grouped_topk`；采用 snake_case，不能以下划线开头 |
| `description` | string | 算子功能、算法、输入输出及推理场景的描述（支持多行 YAML 块文本） | 需清晰阐述在 vLLM 推理架构中的作用及支持的精度细节 |
| `for` | array of string | 标识该算子所替代或对标的 vLLM 算子或函数；若无直接对标则填 `None` | 例如 `_moe_C.grouped_topk`、`_C.apply_repetition_penalties_`，无需对标 PyTorch 算子 |
| `labels` | array of string | 在不同维度对算子进行标签分组（非空列表） | 常用标签：`vLLM`, `fused`, `MoE`, `Attention`, `NoCPU`, `KernelGen` 等 |
| `kind` | array of string 或 string | 算子的主要功能类别 | 例如 `MoE`, `NeuralNetwork`, `Quantization`, `Normalization` 等 |
| `stages` | array of key-value pairs | 记录算子的成熟度演化历史（非空列表） | 键为成熟度阶段（`alpha`, `beta`, `stable`, `removed`），值为引入/晋升的版本号（如 `'0.2'`、`'0.3'`） |

### 算子成熟度生命周期

算子成熟度的阶段定义如下：

- 新的手工编写融合算子通常以 `beta` 阶段起步；
- 初步原型或新生成的实验性算子通常以 `alpha` 阶段起步；
- **版本演化规则**：当前仓库演化版本为 0.2，**下一发布版本为 0.3**。当算子经过完整测试、接入自动调优，并在真实推理负载与全量 CI 中稳定运行至少一个发版周期后，可在下一个版本中晋升至更高阶段。例如：
  - 某算子在当前版本以 `beta: '0.2'` 引入；
  - 经持续测试与优化后，可在下一版本中晋升为 `stable: '0.3'`；
  - 若已达到 `stable` 的算子在后续演进中暴露严重缺陷，也可能被降级为 `beta` 或 `alpha`。

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
      - beta: '0.2'
      - stable: '0.3'
```

提交 PR 前可使用以下命令执行元数据静态检查：

```bash
python3 tools/ci_checks/check_operators_yaml.py
```

## 5. 新算子开发交付要求 {#operator-deliverables}

在着手开发新算子之前，首先应查阅 `conf/operators.yaml` 确认是否已有同类实现。

> [!IMPORTANT]
> **仅限 vLLM 算子，不接受通用 PyTorch 算子**：
> FlagGems-vllm 专为存放 vLLM 仓库特化与融合算子而设立，**不接受通用 PyTorch 算子**（通用 PyTorch 算子请向 [FlagGems](https://github.com/flagos-ai/FlagGems) 贡献）。因此，开发算子**不需要对齐 PyTorch** 原生算子语义，只需对齐 vLLM 仓库现有算子、原生 CUDA/C++ 扩展或模型推理参考实现。

> [!IMPORTANT]
> **公版算子定位与多后端架构要求**：
> `src/flaggems_vllm/ops/` 目录下的代码为 **公版算子实现**（Generic Baseline Implementations）。
> - **基准兼容性**：为确保能够平滑移植到各异构芯片，公版算子实现必须能够基于官方上游 Triton 编译器（或 FlagTree）在通用架构（如 NVIDIA Ampere / Hopper）上直接稳定运行；
> - **严禁厂商非标私有扩展**：公版算子中严禁使用仅特定厂商或私有编译器支持的非标语言扩展、私有指令或芯片内联汇编（统一的 `has_triton_tle` 优雅降级机制除外）；
> - **厂商特化路径**：若您的优化仅适用于特定硬件架构（如昇腾、海光、沐曦、摩尔线程、天数智芯等），请将其作为厂商特化算子放置于 `src/flaggems_vllm/runtime/backend/_<vendor>/` 目录下进行派发重载，以保持公版实现的高兼容性与纯粹性；
> - FlagGems-vllm 为设备端高性能算子库，不接受无直接或间接设备端计算调用的空算子。

**新算子开发交付必须满足以下完整要求**：

1. **填写 `conf/operators.yaml` 算子元数据信息**：
   - 在 `conf/operators.yaml` 中新增算子条目；
   - 完整填写 `id`, `description`, `for`, `labels`, `kind`, `stages` 六项必填字段；
   - 确保 `id` 唯一且按字母序排列，并通过 `python3 tools/ci_checks/check_operators_yaml.py` 静态检查。
2. **算子实现与 Host 调度 (`src/flaggems_vllm/ops/<op>.py`)**：
   - 编写 Host 调度函数与 Triton 内核函数；
   - 使用 `@libentry()` 修饰入口函数，做好输入张量形状、步长、数据类型与设备校验；
   - **遵守 Log 日志输出规范**（见后文第 6 节，logger 中采用大写下划线分隔方式）；
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
   - 覆盖典型推理形状与数据类型（如 FP16/BF16/FP8 等）以及边界情况；
   - **遵守测试算子日志断言规范**（见后文第 8 节）；
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
| 功能单元测试 | `tests/test_<op>.py` | 正确性与边界测试（带 `@pytest.mark.<op>` 与日志断言） |
| 性能基准测试 | `benchmark/test_<op>.py` | 性能评测脚本 |
| 核心测试形状 | `benchmark/core_shapes.yaml` | 评测的核心输入形状列表 |
| 模块导出 | `src/flaggems_vllm/ops/__init__.py` | 模块内部导出并加入 `__all__` |
| 顶层导出 | `src/flaggems_vllm/__init__.py` | 包级别对外公开接口直接导出 |

## 6. 算子 Host 函数与日志规范 {#operator-host-function-conventions}

### 日志规范 (Log Conventions)

为了便于运行时链路追踪、多后端分发调试以及单元测试中的派发存活性检验，每个算子函数必须严格遵守以下日志规范：

1. **Logger 初始化**：在算子源文件头部，必须通过标准 `logging` 模块初始化当前模块的 logger：
   ```python
   import logging
   logger = logging.getLogger(__name__)
   ```
2. **入口第一条语句输出 DEBUG 日志**：每个算子 Host 调度函数/入口函数的第一条有效执行语句，都必须输出 `DEBUG` 级别的派发日志。
3. **日志前缀规范**：
   - **公版通用实现**（`src/flaggems_vllm/ops/`）：统一使用 `GEMS` 前缀；
   - **厂商特化实现**（`src/flaggems_vllm/runtime/backend/_<vendor>/`）：必须使用 `GEMS_<VENDOR>` 前缀（其中 `<VENDOR>` 为大写的厂商名，如 `GEMS_NVIDIA`、`GEMS_ASCEND`、`GEMS_HYGON`、`GEMS_METAX`、`GEMS_MTHREADS` 等）。
4. **日志内容格式**：
   - 前缀后用**空格**分隔，紧接着在 logger 中采用**大写下划线分隔方式**（`UPPER_SNAKE_CASE`）输出算子名称（例如 `GEMS GROUPED_TOPK`、`GEMS MOE_ALIGN_BLOCK_SIZE`、`GEMS_ASCEND GROUPED_TOPK`）；
   - 可在算子名后根据需要附带输入形状或核心参数调试信息。

标准日志输出范例：

```python
# 1. 基础公版算子（大写下划线分隔）
def grouped_topk(scores, num_groups, topk_group, top_k, ...):
    logger.debug("GEMS GROUPED_TOPK")
    ...

# 2. 带有输入尺寸信息的算子
def add_rms_norm(x1, x2, normalized_shape, weight, eps=1e-5):
    logger.debug(
        "GEMS ADD_RMS_NORM FORWARD, [input1 shape]: %s, [input2 shape]: %s, [weight shape]: %s",
        x1.size(),
        x2.size(),
        weight.size() if weight is not None else None,
    )
    ...

# 3. 厂商特化算子（以 Ascend 后端为例，位于 runtime/backend/_ascend/ops/grouped_topk.py）
def grouped_topk(scores, num_groups, topk_group, top_k, ...):
    logger.debug("GEMS_ASCEND GROUPED_TOPK")
    ...
```

### 严禁 PyTorch 计算回退规则 (NO_TORCH_COMPUTE_FALLBACK)

**生产内核执行路径允许调用 PyTorch 框架中的 `torch.empty`、`torch.zeros` 等内存算子以及 `contiguous`、`clone`、`copy_` 等创建或拷贝算子，除此之外禁止调用 torch 算子实现计算功能。可以调用 FlagGems 的算子。**

#### 允许调用

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

#### 严禁调用

除上述内存分配算子、创建/拷贝算子与元数据/视图操作外，**禁止调用任何 PyTorch 算子实现计算功能**（包含计算核函数、隐式类型转换等）：

```python
# ❌ 生产路径严禁调用 PyTorch 算子实现计算功能
def my_operator(x):
    return torch.matmul(x, x.T)  # ❌ 禁止调用 PyTorch 计算算子
    return x.sum(dim=-1)          # ❌ 禁止调用 PyTorch 计算算子
    return torch.softmax(x, dim=-1) # ❌ 禁止调用 PyTorch 计算算子
    return x.to(torch.float32)    # ❌ 触发类型转换内核
```

#### 不支持配置的处理

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

规范工具与约束：
- **black** (行宽 88)
- **isort** (`--profile black`)
- **flake8** (最大行宽 120，忽略 `F405,E731,W503,E203,E704`)

## 8. 算子单元测试 {#operator-unit-tests}

单元测试用于检验算子实现的正确性与派发存活性。在添加新算子时，需要在 `tests/test_<op>.py` 中添加单元测试用例。

- 使用 `@pytest.mark.<op>` 标记测试用例，mark 名需与算子 API 名称保持一致；
- 覆盖典型推理形状与数据类型；
- 使用 `tests.accuracy_utils`（`gems_assert_close` / `gems_assert_equal`）对比参考实现（如 vLLM 原生算子或模型参考实现）；
- 支持 `--quick` 快速测试与全量测试：
  ```bash
  pytest -v -s tests/test_<op>.py
  pytest -q tests/test_<op>.py --quick
  ```

### 测试算子日志 {#testing-operator-logs}

在算子单元测试中，除了验证计算结果的数值精度（`gems_assert_close` / `gems_assert_equal`）外，**还必须验证运行时是否确实派发到了 FlagGems-vllm 的 Triton 实现（或目标后端的特化实现），而不是静默回退或走入未生效路径**。

根据算子日志规范，每个算子 Host 函数的第一条语句都输出带有规范前缀和大写下划线分隔算子名的 DEBUG 日志。单元测试通过 pytest 的 `caplog` fixture 捕获并断言该日志，以验证派发路径与算子存活性（liveness）。

#### 核心辅助工具

测试辅助模块 `tests.accuracy_utils`（通常简写为 `utils`）提供了针对日志验证的专用工具：

1. **`caplog` (pytest fixture)**：用于捕获测试用例执行期间的日志信息。测试用例入参需显式声明 `caplog`（例如 `def test_grouped_topk(shape, dtype, caplog):`）。如需在同一测试函数中进行多轮验证，可调用 `caplog.clear()` 清空此前捕获的日志。
2. **`utils.gems_log_logger(fn)`**：返回被测算子实现函数所使用的 logger 名称（即 `fn.__module__`）。算子源文件中统一通过 `logger = logging.getLogger(__name__)` 初始化 logger。使用该辅助函数可将 DEBUG 捕获范围严格限定在当前算子所在的模块（例如公版通用实现 `flaggems_vllm.ops.grouped_topk`，或厂商特化模块 `flaggems_vllm.runtime.backend._ascend.ops.grouped_topk`），避免无差别捕获 root logger 或全局其他模块的调试日志造成干扰。
3. **`utils.gems_log_prefix(fn)`**：根据最终解析出的算子实现函数自动推导期望的日志前缀。若运行时命中了厂商特化实现（模块路径中包含 `_{flaggems_vllm.vendor_name}`），则返回 `GEMS_<VENDOR>`（如 `GEMS_NVIDIA`、`GEMS_ASCEND`、`GEMS_HYGON` 等）；若使用的是通用公版实现，则返回 `GEMS`。这使得测试用例无需修改即可在不同硬件后端上保持通用性。

#### 标准测试代码范例

在编写单元测试时，标准做法是将算子调用放置在 `caplog.at_level` 上下文管理器中，并在调用后断言包含动态前缀与大写下划线分隔算子名称的日志：

```python
import pytest
import torch
import flaggems_vllm
import tests.accuracy_utils as utils

@pytest.mark.grouped_topk
@pytest.mark.parametrize("shape", [(16, 64), (64, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_grouped_topk(shape, dtype, caplog):
    scores = torch.randn(shape, device=flaggems_vllm.device, dtype=dtype)
    entry = flaggems_vllm.grouped_topk

    # 1. 限定 logger 作用域捕获 DEBUG 日志，并调用算子
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(entry)):
        out_weights, out_indices = entry(scores, num_groups=4, topk_group=2, top_k=4)

    # 2. 验证派发日志：动态前缀 + 大写下划线分隔算子名
    assert f"{utils.gems_log_prefix(entry)} GROUPED_TOPK" in caplog.text

    # 3. 验证计算数值精度
    ref_weights, ref_indices = ref_grouped_topk(scores, 4, 2, 4)
    utils.gems_assert_close(out_weights, ref_weights, dtype=dtype)
    utils.gems_assert_equal(out_indices, ref_indices)
```

#### 规范与避坑要点

- **禁止硬编码前缀**：严禁在断言中直接书写 `"GEMS GROUPED_TOPK"` 或 `"GEMS_NVIDIA GROUPED_TOPK"` 等固定字符串，推荐使用 `f"{utils.gems_log_prefix(entry)} <OP_NAME>"`，以保证测试用例在全硬件后端（NVIDIA、Ascend、Kunlunxin、Hygon、Cambricon 等）的兼容性。
- **严格限定 Logger 作用域**：切勿直接在不指定 `logger` 的情况下对 root logger 启用 DEBUG（如 `caplog.at_level("DEBUG")`），否则可能捕获到 PyTorch 内部或编译器的海量日志，甚至引起断言误判或内存溢出。
- **算子名称格式严格匹配**：日志断言中的算子名称须为大写下划线分隔格式（如 `GROUPED_TOPK`），并严格与 Host 函数中 `logger.debug(...)` 输出保持一致（包括后缀，如 `FORWARD` 或 `OUT`）。

### 模型端到端测试 {#model-test}

模型测试用于检验算子接入 vLLM 推理引擎后的端到端正确性与性能表现。新增或优化算子后，建议配合 `vllm-plugin-fl` 在代表性大语言模型（如 DeepSeek、Qwen、Llama 等）上执行端到端文本生成验证，确保融合算子在真实模型上下文中的数值稳定性。

### 测试覆盖率 {#test-coverage}

单元测试覆盖率用于检测算子测试用例的代码覆盖深度。执行单元测试时，可通过 `pytest-cov` / `coverage` 工具收集被覆盖的代码行：

```bash
PYTHONPATH=src pytest tests/test_<op>.py --cov=flaggems_vllm.ops.<op> --cov-report=term-missing
```

确保覆盖正常推理输入、各种数据类型矩阵、以及异常边界情况（如空输入、不支持类型抛错路径等）。

## 9. 算子性能评测 {#operator-performance-benchmarking}

在 `benchmark/test_<op>.py` 中添加性能评测：
- 使用 `@pytest.mark.parametrize("shape", Config.get_shapes("<op>"))`；
- 在 `benchmark/core_shapes.yaml` 录入代表性推理形状；
- **加速比定义**：
  $$\text{SpeedUp} = \frac{\text{latency}_{\text{vllm\_baseline}}}{\text{latency}_{\text{flaggems}}}$$
- **验收目标**：关键核心形状下达到 **SpeedUp ≥ 0.9**；
- 运行测试命令：
  ```bash
  pytest -q benchmark/test_<op>.py --level core --iter 1 --warmup 1
  ```
- 当有新的基准测试添加到 `benchmark/` 目录或已有测试被修改时，CI 流水线会自动检测到相关变更并触发性能评测。

## 10. 算子开发协议规范 {#development-protocol}

算子开发与优化必须严格遵从仓库根目录的 `workflow.md` 执行协议（涵盖 G0–G7 门禁）。详见[算子开发协议规范](workflow.md)。
