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

FlagGems-vllm 是 [FlagOS](https://flagos.io/) 的一部分。
FlagGems-vllm 是一个面向多种芯片后端的高性能算子库，它提供了常见 vLLM 算子的高性能实现，支持多种主流模型的高性能推理及部署。

FlagGems-vllm 是一个使用 OpenAI 推出的 [Triton 编程语言](https://github.com/openai/triton) 实现的高性能深度学习算子库。

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

## 快速安装

### 安装依赖

```shell
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```

在安装 FlagGems-vllm 之前，请先安装与目标加速卡兼容的 PyTorch。与 FlagGems 一致，PyTorch 由底层环境管理，FlagGems-vllm 不会重复安装 PyTorch。

### 安装 FlagGems-vllm

```shell
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install .
```

开发模式下建议使用可编辑模式安装：

```shell
pip install --no-build-isolation -e .
```

若需运行测试套件，可一并安装测试依赖：

```shell
pip install -e '.[test]'
```

### 与 FlagGems 和 vllm-plugin-fl 协同安装

验证完整插件栈时，请按以下顺序安装：

```shell
# 1. 安装 FlagGems
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .

# 2. 安装 FlagGems-vllm
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install --no-build-isolation -e .

# 3. 安装 vllm-plugin-fl
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
pip install --no-build-isolation -e .
```

当环境中安装了多个 vLLM 插件时，显式指定启用 FlagOS 插件：

```shell
export VLLM_PLUGINS=fl
```

## 使用示例

由于 FlagGems-vllm 依赖 FlagGems 且部分算子在底层调用 FlagGems 算子，推荐在使用 FlagGems-vllm 算子时先调用 `flag_gems.enable()` 初始化全局算子分发支持：

```python
import torch
import flag_gems
import flaggems_vllm

# 推荐：启用 FlagGems 通用算子
flag_gems.enable()

# 构造 MoE 路由所需的 topk_ids
num_tokens = 128
topk = 2
num_experts = 16
block_size = 32

topk_ids = torch.randint(
	low=0,
	high=num_experts,
	size=(num_tokens, topk),
	device='cuda',
	dtype=torch.int32,
)

# 按 expert 和 block_size 对 token 做对齐
sorted_ids, expert_ids, num_tokens_post_pad = flaggems_vllm.ops.moe_align_block_size(
	topk_ids=topk_ids,
	block_size=block_size,
	num_experts=num_experts,
)

print(sorted_ids.shape, expert_ids.shape, num_tokens_post_pad)
```

### 与 FlagGems 配合使用

`vllm-plugin-fl` 会启用 FlagGems 通用后端，同时支持直接调用 FlagGems-vllm 算子。最小冒烟验证如下：

```python
import torch
import flag_gems
import flaggems_vllm

flag_gems.enable()

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

## 测试与基准测试快速上手

以下命令可用于安装后的快速功能验证。大多数测试与基准评测需要支持的加速卡运行环境、厂商定制 PyTorch、配套 Triton 编译器以及 vLLM 相关依赖。

### 导入冒烟测试

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src python - <<'PY'
import torch
import flaggems_vllm
from flaggems_vllm import runtime

print('torch:', torch.__version__)
print('vendor:', flaggems_vllm.vendor_name)
print('flaggems_vllm device:', flaggems_vllm.device)
print('device count:', runtime.device.device_count)
print('grouped_topk:', callable(flaggems_vllm.grouped_topk))
PY
```

### 运行功能测试

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests --collect-only
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests --quick
```

运行单个目标算子测试：

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests/test_grouped_topk.py
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests/test_fused_inv_rope_fp8_quant.py --quick
```

### 运行性能基准测试

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark --collect-only
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark/test_moe_align_block_size_triton.py::test_moe_align_block_size_triton --level core --iter 1 --warmup 1
```

运行 vLLM 特化算子的针对性评测：

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark/test_grouped_topk.py --level core --iter 1 --warmup 1
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark/test_fused_inv_rope_fp8_quant.py --level core --iter 1 --warmup 1
```

### 使用 vllm-plugin-fl 进行全链路验证

在完成 `FlagGems`、`FlagGems-vllm` 和 `vllm-plugin-fl` 安装后，验证插件能否正常导入且 FlagGems-vllm 算子可用：

```shell
export VLLM_PLUGINS=fl
python - <<'PY'
import flag_gems
import flaggems_vllm

flag_gems.enable()
print('FlagGems enabled')
print('FlagGems-vllm grouped_topk available:', callable(flaggems_vllm.grouped_topk))
PY
```

如果具备模型权重与 vLLM 运行环境，可直接运行 `vllm-plugin-fl` 仓库中的离线推理示例，如 `examples/offline_inference.py`。

### 注意事项

- 建议首先运行 `--collect-only`，快速确认模块导入与测试发现是否正常。
- 算子支持时优先使用 `--quick` 进行快速功能验证。
- 基准评测冒烟可使用 `--level core --iter 1 --warmup 1`。
- 全量基准测试耗时较长，建议仅在正式性能验证时执行。

本项目采用 [Apache (Version 2.0) License](./LICENSE) 开源协议。
