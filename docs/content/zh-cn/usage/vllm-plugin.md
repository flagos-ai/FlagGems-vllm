---
title: vLLM 插件集成
weight: 20
---

# 与 vllm-plugin-fl 集成

FlagGems-vllm 专为配合 [vllm-plugin-fl](https://github.com/flagos-ai/vllm-plugin-fl)（FlagOS 的 vLLM 插件层）无缝运行而设计。

## 架构体系

三层协同架构：

```
vLLM (推理引擎)
    ↓
vllm-plugin-fl (插件层)
    ↓
    ├─→ flag_gems.enable()        (通用 PyTorch 算子替换)
    └─→ flaggems_vllm.<op>()      (vLLM 专用融合内核)
```

- **FlagGems**：提供通用的 PyTorch 基础算子替换，注册到 PyTorch dispatch。同时作为 FlagGems-vllm 的核心底层依赖，支撑其部分特化内核的内部算子调用。
- **FlagGems-vllm**：依赖 FlagGems，提供针对 vLLM 场景特化的融合算子，由插件层显式导入调用。
- **vllm-plugin-fl**：统筹调度两层算子，将它们挂接到 vLLM 对应的模型执行路径。

## 安装步骤

请按以下顺序安装全部三个组件：

```bash
# 1. 安装 FlagGems (通用基础算子，5.4.0 版本)
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/

# 2. 安装 FlagGems-vllm (vLLM 专用算子)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm && pip install --no-build-isolation -e . && cd ..

# 3. 安装 vllm-plugin-fl (插件层)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl && pip install --no-build-isolation -e .
```

## 启用插件

当环境中安装了多个 vLLM 插件时，请显式指定使用 FlagOS 插件：

```bash
export VLLM_PLUGINS=fl
```

当 vLLM 初始化时，该插件会自动被加载激活。

## 使用示例

安装插件并设置环境变量后，vLLM 会自动调用 FlagGems 和 FlagGems-vllm 的高性能算子：

```python
from vllm import LLM, SamplingParams

# 插件自动生效
llm = LLM(model="meta-llama/Llama-2-7b-hf")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

其中：
- **通用算子**（matmul, layernorm, softmax 等）由 FlagGems 负责。
- **vLLM 特化融合算子**（`grouped_topk`、`moe_align_block_size`、`flash_attention_forward` 等）由 FlagGems-vllm 负责。

## 验证插件生效

可以通过以下 Python 脚本检查 FlagGems-vllm 算子是否就绪：

```python
import os
os.environ["VLLM_PLUGINS"] = "fl"

import flag_gems
import flaggems_vllm

flag_gems.enable()
print("FlagGems 已启用")
print("FlagGems-vllm grouped_topk 可用:", callable(flaggems_vllm.grouped_topk))
```

## 硬件后端选择

插件支持通过 `DNN_VENDOR` 环境变量指定底层硬件后端：

```bash
# NVIDIA (默认)
export DNN_VENDOR=nvidia

# AMD
export DNN_VENDOR=amd

# 华为昇腾
export DNN_VENDOR=ascend
```

## 常见问题与排查

### 插件未加载

检查环境变量是否设置：
```bash
echo $VLLM_PLUGINS  # 应输出: fl
pip show vllm-plugin-fl
```

### 算子回退排查

如果未观测到预期的性能提升，请排查：
1. FlagGems-vllm 是否正常安装：`python -c "import flaggems_vllm; print(flaggems_vllm.__version__)"`
2. 后端是否与实际硬件匹配：`python -c "import flaggems_vllm; print(flaggems_vllm.vendor_name)"`
3. 检查是否有冲突的环境变量或缺少 Triton/FlagTree 编译依赖。

## 下一步

- [基础用法](basic.md)
- [非 NVIDIA 后端配置](non-nvidia.md)
- [运行性能基准评测](../performance/benchmark.md)
