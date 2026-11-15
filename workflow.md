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

# Workflow: 仓库内 Triton 算子开发执行协议

## 0. 文档目的

本文档规定 AI 在本仓库中开发算子的执行顺序、代码落点、测试/benchmark 要求、deep optimization 触发条件和最终交付格式。

> **仓库定位说明**：
> 本仓库存放 vLLM 仓库的特化与融合算子，**不接受通用 PyTorch 算子**（通用 PyTorch 算子请向 [FlagGems](https://github.com/flagos-ai/FlagGems) 提交）。因此，在本仓库中开发算子**不需要对齐 PyTorch** 原生算子，对标对象为 vLLM 现有原生算子、C++/CUDA 扩展或模型推理参考实现。

本文档不解释每类算子的高级优化技巧；这些内容在 `optimization.md` 和 `deep_opt.md` 中。

阅读顺序固定：

1. `workflow.md`：确认当前任务、仓库路径、命令和 gate。
2. `optimization.md`：完成一阶段设计、路径拆分和 autotune 方案。
3. `deep_opt.md`：仅当实现正确但性能未达标时启用。

## 1. 全局硬 gate

AI 不能跳过以下 gate。

| Gate | 通过条件 | 失败处理 |
| --- | --- | --- |
| G0 任务输入 | 当前任务输入只需包含 vLLM 算子名、层定义或参考链接/版本 | 用户不再补其他字段；AI 自行补齐其余任务事实，不能直接写代码 |
| G1 工程基线 | editable/read-only、build、validation、benchmark、active set、autotune、profiler 已冻结 | 不允许比较性能 |
| G2 vLLM 规范 | dtype matrix、shape、layout、promotion、边界语义、unsupported scope 已明确（不要求对齐 PyTorch） | 不允许实现 |
| G3 路径设计 | fast/special/general/unsupported/external-library path 已定义；无 torch compute fallback | 不允许写 kernel |
| G4 实现集成 | operators.yaml、ops、tune config、test、benchmark、导出已完成或说明不需要 | 不允许验收 |
| G5 功能测试 | 功能测试通过，含 dtype、边界、unsupported path | 立即修复或回滚 |
| G6 性能验收 | benchmark active set 达标，或触发 deep_opt | 未达标必须进入 `deep_opt.md` |
| G7 交付 | 结果、风险、autotune、fallback/unsupported、deep opt 结论完整 | 不得宣称完成 |

## 2. 生产路径禁止 torch compute fallback

### 2.1 默认策略

本仓库默认 `NO_TORCH_COMPUTE_FALLBACK = true`。

正式实现、host dispatch、helper、fallback、unsupported path 中禁止用任何 `torch` 计算算子生成目标结果。禁止范围包括：

- `torch.<target_op>`。
- 可组合出目标语义的其他 `torch.<op>`。
- `torch.nn.functional.*`。
- `torch.ops.aten.*`。
- 会发射计算或类型转换 kernel 的 `Tensor` 方法，例如 `matmul`、`sum`、`max`、`to`。

允许范围：

- 元信息读取：shape、stride、dtype、device、layout。
- 内存算子分配：允许调用 PyTorch 框架中的 `torch.empty`、`torch.empty_like`、`torch.empty_strided`、`torch.zeros`、`torch.zeros_like` 等内存算子。
- 创建与拷贝算子：允许调用 `contiguous`、`clone`、`copy_` 等创建或拷贝算子。
- 确认可不触发数据移动的 view/metadata 操作。
- 允许调用 FlagGems (`flag_gems`) 的算子。

除此之外，禁止调用任何 torch 算子实现计算功能。

### 2.2 静态审计要求

提交前必须执行一次生产路径审计。至少检查：

```bash
python - <<'PY'
from pathlib import Path
op = '<op>'
paths = [Path(f'src/flaggems_vllm/ops/{op}.py')]
for p in paths:
    if not p.exists():
        continue
    text = p.read_text()
    suspicious = []
    patterns = ['torch.', 'torch.nn.functional', 'torch.ops.aten', '.contiguous(', '.clone(', '.copy_(', '.to(']
    for pat in patterns:
        if pat in text:
            suspicious.append(pat)
    print(p, suspicious)
PY
```

如果审计发现 `torch` 调用，必须逐项解释：

| 调用 | 文件/行 | 是否生产路径 | 是否计算/拷贝 | 是否允许 | 处理 |
| --- | --- | --- | --- | --- | --- |

测试、对拍、benchmark baseline 中允许使用 `torch` 参考实现，但不能被生产路径 import 或调用。

## 3. 仓库代码落点

开发一个新算子时，通常需要修改：

| 目的 | 路径 |
| --- | --- |
| 算子元数据 | `conf/operators.yaml` |
| 算子实现 | `src/flaggems_vllm/ops/<op>.py` |
| NVIDIA autotune 配置 | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` |
| 功能测试 | `tests/test_<op>.py` |
| 性能测试 | `benchmark/test_<op>.py` |
| ops 导出 | `src/flaggems_vllm/ops/__init__.py` |
| 顶层导出 | `src/flaggems_vllm/__init__.py` |

`tune_configs.yaml` 不是后置可选项。凡是 kernel 暴露 `BLOCK_*`、`TILE_*`、`GROUP_*`、`SPLIT_*`、`UNROLL`、`num_warps`、`num_stages`、`num_ctas` 等性能参数，NVIDIA backend 默认必须通过 `runtime.get_tuned_config("<op_or_path>")` + `@libtuner(...)` 接入。豁免必须在交付结论中写明。

## 4. 强制开发流程

### Step 1. 读取任务和文档

必须先完成：

1. 读取本文件当前任务输入。
2. 阅读 `optimization.md`。
3. 阅读 vLLM 对应算子源码、模型层实现或官方参考（无需查阅 PyTorch ATen）。
4. 检查仓库内相近算子、测试、benchmark 和 libtuner 接入方式。

### Step 2. 明确 vLLM 算子接口规范与约束

必须先明确 dtype matrix、shape/output rules、promotion、NaN/Inf、异常、unsupported scope。

如果无法确认某个事实，写“未知 + 假设 + 验证方式”。不能把未知语义留到实现后再补。

### Step 3. workload 分型和路径拆分

按 `optimization.md` 判断算子家族、主导维度、退化结构、fast/special/general/unsupported path。

如果不同 shape、dtype、layout 的最优调度明显不同，必须拆 path；不能让一个 general kernel 靠 autotune 解决算法选择。

### Step 4. 设计 host dispatch

host dispatch 必须先于 kernel 设计，至少负责：

- 参数规范化。
- dtype/layout/shape 路由。
- empty/identity early return。
- workspace 和外部库边界。
- autotune key。
- unsupported path。
- 禁止 torch compute fallback。

### Step 5. 设计 autotune 接入

每条 kernel path 必须写清：

- 配置名。
- `runtime.get_tuned_config("<op_or_path>")` 是否存在。
- `@libtuner(...)` key / strategy。
- `tune_configs.yaml` 的 `META` 字段与 kernel `tl.constexpr` 参数映射。
- 候选参数与 active shape。
- no-autotune 豁免。

NVIDIA backend 检查项：

| 检查 | 要求 |
| --- | --- |
| 配置名 | 每个 `get_tuned_config` 名称在 `tune_configs.yaml` 中存在 |
| 候选集 | 非空，除非明确豁免 |
| META | 与 kernel `tl.constexpr` 参数一一对应 |
| launch 参数 | `num_warps`、`num_stages`、`num_ctas` 放在配置条目或生成规则中 |
| key | 来自真实性能维度，不是纯语义参数 |
| active set | 覆盖每个 autotuned path 的代表 shape |

### Step 6. 实现代码

实现要求：

- 在 `src/flaggems_vllm/ops/<op>.py` 完成 host 逻辑、kernel launcher 和 Triton kernel。
- 不把大段候选配置硬编码在算子文件里，除非有明确局部模式和豁免。
- 不用 kernel 内大量运行时分支替代 host dispatch。
- 需要 `out` 时，显式检查 shape、dtype、device 和 alias 风险。
- 低频复杂路径可以 unsupported，但不能 torch fallback。
- 保持 helper 的生产路径和测试路径分离。

### Step 7. 完成集成

必须完成或明确不需要：

1. `src/flaggems_vllm/ops/__init__.py` 导出。
2. `src/flaggems_vllm/__init__.py` 顶层导出。
3. `tune_configs.yaml` 配置。
4. `tests/test_<op>.py`。
5. `benchmark/test_<op>.py`。

缺少导出，算子视为未接入完成。

## 5. 功能测试要求

测试文件：`tests/test_<op>.py`

基本命令：

```bash
python -m pytest -v -s tests/test_<op>.py
```

默认覆盖：

- 目标 device 上 `torch` 支持的全部 dtype。
- empty、single element、非 2 的幂、热点 shape、奇异 shape。
- contiguous、非连续、特殊 layout。
- broadcast，若支持。
- `out=` / alias / in-place，若接口暴露。
- integer、bool、complex，若 `torch` 支持。
- NaN / Inf / signed zero / tie-break / deterministic。
- autograd/backward，若承诺支持训练。
- unsupported path 报错。
- 生产路径无 torch compute fallback 的审计。

测试中允许 `torch` 作为参考结果；测试 helper 不得被生产实现 import。

## 6. 性能测试要求

benchmark 文件：`benchmark/test_<op>.py`

基本命令：

```bash
python -m pytest -v -s benchmark/test_<op>.py
```

benchmark 必须包含：

- vLLM 原生 baseline 或参考实现。
- FlagGems-vllm 实现。
- 小 shape、热点 shape、奇异 shape、大 shape。
- 算子关键维度 sweep，例如 M/N/K、reduce dim、embedding dim、window、index 分布。
- active set、metric、unit、direction、aggregation。

当前仓库 SpeedUp 定义：

```text
SpeedUp = latency_base / latency
```

其中 `latency_base` 为 baseline 耗时（如 vLLM 原生实现耗时），`latency` 为 FlagGems-vllm 实现耗时。

验收默认规则：

- 关键目标 shape 的 `SpeedUp < 0.9`：当前实现收益不足。
- active set 中关键 workload 明显回归：不能接受。
- 只改善少数非关键 shape：不能掩盖整体回归。
- benchmark 口径变化后必须重建 baseline。

## 7. Profiling 预检

进入多轮优化前，确认 profiler 可用。

NVIDIA backend 默认记录：

```bash
nsys profile --stats=true <benchmark_command>
ncu --set full <benchmark_command>
```

若 benchmark 发射大量 kernel，必须收窄到目标 kernel 再运行 `ncu`。

profiler 不可用时，必须写明限制，并至少提供替代证据：dispatch path、kernel launch 数、benchmark 分解、autotune 命中配置、shape crossing point。

## 8. 性能 trial loop

基础优化阶段按以下循环：

1. 跑当前 best，记录 baseline。
2. 明确本轮唯一主要假设。
3. 修改实现或配置。
4. build。
5. validation。
6. benchmark active set。
7. 记录结果。
8. keep / revert。

记录字段：

| 字段 | 内容 |
| --- | --- |
| Trial | 编号和时间 |
| Hypothesis | 本轮假设 |
| Change | 代码/config 变化 |
| Validation | 命令和结果 |
| Benchmark | active set 结果 |
| Autotune | 配置名、候选变化、best config |
| Decision | keep / revert / split path / deep_opt |

连续 3 到 5 轮没有稳定收益，或关键目标仍低于阈值，必须进入 `deep_opt.md`。

## 9. Deep Optimization 触发与执行

进入条件：

- 功能测试通过。
- 生产路径无 torch compute fallback。
- active benchmark set 已冻结。
- autotune 接入已验证，或豁免已写明。
- 关键目标 shape 未达标，例如 `SpeedUp < 0.9`，或关键 workload 回归。

进入后必须：

1. 阅读 `deep_opt.md`。
2. 填写 Deep Opt Board。
3. 先做 metric/dispatch/autotune/no-torch audit。
4. 使用 profiler 或替代证据定位瓶颈。
5. 按算子家族选择 1 到 3 个候选策略。
6. 每轮只改一个主要变化。
7. validate -> benchmark -> keep/revert。
8. 完成或停止时输出 deep optimization summary。

禁止：

- 不看 profiler 连续盲调 block size 或 num_warps。
- 同一轮同时改算法、dispatch、autotune key、候选空间和 benchmark 口径。
- 用 torch fallback 达成性能指标。
- 放松 dtype、边界语义、数值稳定或测试覆盖。

## 10. Keep / Revert 规则

| 情况 | 处理 |
| --- | --- |
| build 失败 | `build_error`，修复或回滚 |
| 功能测试失败 | `validation_error`，立即回滚 |
| benchmark 崩溃/超时 | `runtime_error`，修复或回滚 |
| 无稳定收益 | 默认回滚 |
| 单点收益但关键 active 回归 | 默认回滚或拆 path |
| 语义正确 + active 无明显回归 + 关键目标收益 | keep |
| benchmark/validation/profiler 命令变化 | 重建 baseline 后再比较 |

## 11. 最终交付格式

完成后必须给出：

```text
Operator delivery summary:
- Op: vLLM API / FlagGems API
- Status: complete / limited / unsupported / needs deep_opt
- Specification: dtype matrix, output rules, key semantics
- Paths: fast / special / general / unsupported / external-library
- No torch compute fallback audit: pass/fail, suspicious calls and resolution
- Autotune: config names, keys, candidate parameters, no-autotune exemptions
- Tests: commands and results
- Benchmark: active set, aggregation, SpeedUp, regressions
- Deep opt: trigger, tried strategies, keep/revert, final result, stop reason
- Residual risks: dtype/layout/shape/training/autograd/workspace
```

## 12. 当前任务输入

本节是用户每次更换待开发算子时唯一需要修改的地方。用户只填写两项：待开发算子接口和对应的官方 `torch` 文档链接。

```text
Current operator task:
- Op API: flaggems_vllm.grouped_topk(gating_output, num_experts, topk, ...)
- Reference docs: vllm/model_executor/layers/fused_moe/...
```

用户不需要填写 backend、性能目标、workload、dtype、autograd、fallback、external library 或 unsupported scope。它们全部由 AI 按本文档默认规则自主补齐。

AI 默认补齐规则：

| 项目 | 默认处理 |
| --- | --- |
| Target backend/device | 从仓库、测试环境和当前 backend 推断；若无额外信息，默认按 CUDA/NVIDIA 路径处理。 |
| Performance target | 使用本文档性能验收标准；关键 active shape 默认要求 `SpeedUp >= 0.9`。 |
| Workload hints | 用户未提供时，由 AI 根据 vLLM 语义、模型层需求、仓库 benchmark 风格和算子家族自动设计 active set。 |
| Training/autograd | 由 AI 查阅 vLLM 源码和目标推理任务语义后判断；本项目专注于前向推理，默认实现 forward。 |
| Fallback policy | 固定为 `NO_TORCH_COMPUTE_FALLBACK`。生产实现、dispatch、helper 和 fallback 中禁止使用 `torch` 计算路径补洞。 |
| Allowed external-library path | 默认 `none`；只有仓库已有机制或项目明确批准时，才允许非 torch 外部库路径。 |
| Unsupported scope | 由 AI 根据语义复杂度、设备能力和工程风险显式判定；不支持的路径必须报错或标注 unsupported，不能静默走 torch fallback。 |

AI 自主补齐要求：

- 查阅 vLLM 官方文档、源码与模型实现，恢复完整函数签名、默认参数和语义。
- 确认目标 device/backend 上 vLLM 对应算子或模型的实际 dtype 支持矩阵（无需对齐 PyTorch）。
- 梳理 shape、rank、stride/layout、promotion、alias、`out=`、in-place、autograd、deterministic。
- 梳理 empty、非法输入、NaN/Inf、signed zero、integer/complex/bool 等边界行为。
- 完成 workload 分型，识别 fast path、special path、general path、unsupported path 和必要的 external-library path。
- 设计 host dispatch、Triton kernel path、autotune 配置和 active benchmark set。
- 初版未达标时自行进入 `deep_opt.md`，不要要求用户继续提供优化方向。
- 所有额外假设必须写入最终交付结论；缺失信息不能成为向用户索要更多模板字段的理由。
