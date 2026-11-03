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

[中文版|[English](./README.md)]

## 介绍

FlagGems-vllm 是 [FlagOS](https://flagos.io/) 的一部分，是一个使用 OpenAI 推出的 [Triton 编程语言](https://github.com/openai/triton) 实现、面向多种芯片后端的高性能算子库。它提供了常见 vLLM 算子的高性能实现，支持多种主流模型的高性能推理及部署。

## 特性

- 算子已经过深度性能调优
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 支持常见 vLLM 算子（如 `moe_align_block_size`、`grouped_topk` 等）

## 与 FlagGems 和 vllm-plugin-fl 的关系

三个仓库在 FlagOS 生态中协同配合，但分工不同：

- **FlagGems**：通用 FlagGems 算子库。提供通用的 PyTorch/Triton 算子替换，并通过 `flag_gems.enable()` / `flag_gems.use_gems()` 将算子注册到 PyTorch dispatch。**FlagGems-vllm 依赖 FlagGems**，当前部分 FlagGems-vllm 算子在底层执行时会直接调用 FlagGems 中的算子。
- **FlagGems-vllm**（本项目）：聚焦 vLLM 推理场景的特化与融合算子库。通过 `flaggems_vllm` Python 包导出算子，例如 `flaggems_vllm.grouped_topk`、`flaggems_vllm.fused_experts_impl` 和 `flaggems_vllm.moe_align_block_size`。
- **vllm-plugin-fl**：vLLM 插件层。通过导入 FlagGems 并调用 `flag_gems.enable()` 作为全局通用算子后端；对于未挂接在 PyTorch dispatch 上的 vLLM 专用融合内核，则显式导入并调用 FlagGems-vllm 提供的算子。

在典型的 vLLM 插件环境中，调用链路如下：

```text
vLLM
	-> vllm-plugin-fl
			-> flag_gems.enable() 注册通用 FlagGems 算子
			-> flaggems_vllm.<operator>() 调用 vLLM 专用融合算子 (部分内部调用 flag_gems 算子)
```

因此，`FlagGems-vllm` 建立在 `FlagGems` 之上：`FlagGems` 提供通用算子底座与部分底层核函数，而 `FlagGems-vllm` 提供 vLLM 特化内核以及供给 `vllm-plugin-fl` 使用的兼容性测试与性能评测。

## 快速上手

### 快速安装

```shell
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
pip install --no-build-isolation -e .
```

完整安装流程（包含测试依赖安装、环境引导脚本 `tools/setup.sh` 以及与 FlagGems、vllm-plugin-fl 协同安装等），请参阅[安装指南](docs/content/zh-cn/getting-started/install.md)。

### 使用示例

由于 FlagGems-vllm 依赖 FlagGems 且部分算子在底层调用 FlagGems 算子，推荐在使用 FlagGems-vllm 算子前先调用 `flag_gems.enable()`：

```python
import torch
import flag_gems
import flaggems_vllm

# 启用 FlagGems 通用算子
flag_gems.enable()

# 调用 FlagGems-vllm 融合算子
scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
bias = torch.randn((16,), device="cuda", dtype=torch.float32)

topk_weights, topk_ids = flaggems_vllm.grouped_topk(
	scores,
	n_group=4,
	topk_group=2,
	topk=2,
	renormalize=True,
	routed_scaling_factor=1.0,
	bias=bias,
	scoring_func=0,
)
print(topk_weights.shape, topk_ids.shape)
```

更多算子示例（如 `moe_align_block_size`、`flash_attention_forward` 和 `fused_add_rms_norm` 等），请参阅[基础用法](docs/content/zh-cn/usage/basic.md)。

## 文档指引

更多详细内容与操作指引已整理至 [docs](docs/) 目录：

- 📖 **起步与安装**: [安装指南](docs/content/zh-cn/getting-started/install.md)
- 🚀 **使用方法**: [基础用法](docs/content/zh-cn/usage/basic.md) | [vLLM 插件集成](docs/content/zh-cn/usage/vllm-plugin.md)
- 🔧 **硬件后端**: [非 NVIDIA 硬件支持](docs/content/zh-cn/usage/non-nvidia.md)
- 🧪 **测试验证**: [单元测试指南](docs/content/zh-cn/testing/unittest.md) | [覆盖率统计](docs/content/zh-cn/testing/coverage.md)
- ⚡ **基准测试**: [运行性能基准测试](docs/content/zh-cn/performance/benchmark.md) | [评测结果](docs/content/zh-cn/performance/results.md)
- 📋 **参考信息**: [支持算子列表](docs/content/zh-cn/references/operators.md) | [MiniMax M3 算子支持](docs/content/zh-cn/references/minimax-m3-support.md)
- 🤝 **贡献指南**: [贡献概述](docs/content/zh-cn/contribution/overview.md) | [开发协议规范](docs/content/zh-cn/contribution/workflow.md) | [项目结构规范](docs/content/zh-cn/contribution/project-structure.md)

## 开源协议

本项目采用 [Apache (Version 2.0) License](./LICENSE) 开源协议。
