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
- **G1（工程基线与环境）**：冻结构建、测试命令、测试集与评测环境；
- **G2（vLLM 算子接口规范）**：明确 dtype matrix、shape、layout、promotion、边界语义与 unsupported scope（对标 vLLM 原生算子或模型参考，不要求对齐 PyTorch）；
- **G3（路径设计与 Host 调度）**：定义 fast/special/general/unsupported 路径，设计 Host 调度与边界防御，严格遵守 `NO_TORCH_COMPUTE_FALLBACK`；
- **G4（实现与集成）**：完成元数据登记（`conf/operators.yaml`）、Host 调度与 Triton JIT 内核实现、自动调优配置（`tune_configs.yaml`）、测试与公共导出；
- **G5（功能测试通过）**：`tests/test_<op>.py` 在 `--quick` 与全量测试模式下均 100% 通过；
- **G6（性能验收达标）**：`benchmark/test_<op>.py` 评测核心形状加速比达到 **SpeedUp ≥ 0.9**；
- **G7（完整交付）**：输出交付结论，明确实现路径、自动调优配置、测试结果与残留风险。

## 静态审计规范

在提交 PR 之前，必须运行静态检查脚本审计生产代码路径，确保不存在未经允许的 PyTorch 计算调用：

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
