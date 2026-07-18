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
FlagGems-vllm是一个面向多种芯片后端的高性能算子库，它提供了常见vllm算子的高性能实现，支持多种常见模型的高性能推理及部署。

FlagGems-vllm 是一个使用 OpenAI 推出的[Triton 编程语言](https://github.com/openai/triton)实现的高性能深度学习算子库，

## 特性

- 算子已经过深度性能调优
- Triton kernel 调用优化
- 灵活的多后端支持机制
- 支持常见vllm算子（如 moe_align_block_size 等）

## 快速安装

### 安装依赖

```shell
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```
### 安装FlagGems-vllm
```shell
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install  .
```

## 使用示例

```python
import torch
import flaggems_vllm

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

## Tests 与 Benchmark 快速使用

下面命令已在当前仓库验证通过，可用于安装后的快速检查。

### 运行 tests

```shell
cd /workspace/FlagGems-vllm
pytest -q tests --collect-only
pytest -q tests/test_moe_align_block_size.py --quick
```

### 运行 benchmark

```shell
cd /workspace/FlagGems-vllm
pytest -q benchmark --collect-only
pytest -q benchmark/test_moe_align_block_size_triton.py::test_moe_align_block_size_triton --level core --iter 1 --warmup 1
```

### 说明

- 大多数 tests/benchmark 需要 CUDA GPU 环境。
- 建议先执行 `--collect-only`，快速确认导入与用例发现是否正常。


本项目采用 [Apache (Version 2.0) License](./LICENSE) 授权许可。
