---
title: 安装指南
weight: 20
---

# 安装指南

## 前置依赖

### 必需依赖

在安装 FlagGems-vllm 之前先安装构建工具:

```bash
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```

### PyTorch

在安装 FlagGems-vllm **之前**,先安装与目标加速器兼容的 PyTorch。FlagGems-vllm 不会自动重新安装 PyTorch。

对于使用 CUDA 12.1+ 的 NVIDIA GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

其他后端请遵循硬件厂商的 PyTorch 安装说明。

## 从源码安装

### 基础安装

克隆并安装 FlagGems-vllm:

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install .
```

### 开发安装

开发模式使用可编辑安装:

```bash
pip install --no-build-isolation -e .
```

### 包含测试依赖

运行测试和基准测试需要:

```bash
pip install --no-build-isolation -e '.[test]'
```

这会安装 pytest、numpy、scipy、cupy-cuda12x 等测试工具。

## 完整 GPU 环境搭建

使用提供的 `tools/setup.sh` 脚本进行完整环境引导:

```bash
source tools/setup.sh
```

此脚本会:
- 创建 `uv` 虚拟环境
- 安装 vLLM 和匹配的 PyTorch
- 安装 FlagGems-vllm
- 可选地将 Triton 替换为 FlagTree(如果设置 `USE_FLAGTREE=1`)
- 运行冒烟测试

### 环境变量

安装脚本支持以下变量:

- `DNN_VENDOR`: 后端硬件厂商(如 `nvidia`、`amd`、`ascend`)。默认:自动检测
- `VLLM_VERSION`: vLLM 版本。默认:最新版
- `USE_FLAGTREE`: 设置为 `1` 使用 FlagTree 编译器替代 Triton
- `CUDA_HOME`: CUDA 安装路径。默认:自动检测

AMD 后端示例:
```bash
DNN_VENDOR=amd source tools/setup.sh
```

## 与 FlagGems 和 vllm-plugin-fl 一起安装

完整的 FlagOS vLLM 插件栈需按以下顺序安装:

```bash
# 1. 安装 FlagGems (通用算子后端)
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .
cd ..

# 2. 安装 FlagGems-vllm (vLLM 专用算子)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install --no-build-isolation -e .
cd ..

# 3. 安装 vllm-plugin-fl (vLLM 插件集成)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
pip install --no-build-isolation -e .
```

如果安装了多个 vLLM 插件,选择 FlagOS 插件:

```bash
export VLLM_PLUGINS=fl
```

## 验证安装

### 导入冒烟测试

```bash
python -c "
import torch
import flaggems_vllm
from flaggems_vllm import runtime

print('torch:', torch.__version__)
print('硬件后端:', flaggems_vllm.vendor_name)
print('设备:', flaggems_vllm.device)
print('设备数量:', runtime.device.device_count)
print('grouped_topk:', callable(flaggems_vllm.grouped_topk))
"
```

### 运行快速测试

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
PYTHONPATH=src pytest -q tests --quick
```

### 运行基准测试冒烟

```bash
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py --level core --iter 1 --warmup 1
```

## 后端选择

FlagGems-vllm 自动检测硬件后端。要覆盖:

```bash
export DNN_VENDOR=nvidia  # 或 amd、ascend、hygon 等
```

可用后端: `nvidia`、`amd`、`ascend`、`cambricon`、`enflame`、`hygon`、`iluvatar`、`kunlunxin`、`metax`、`mthreads` 等(共 18 个)。

## C++ 扩展

FlagGems-vllm 包含通过 pybind11 编译的可选 C++ 算子。如果所需依赖存在,会自动构建。

运行时启用 C++ 算子:

```bash
export USE_C_EXTENSION=1
```

检查 C++ 扩展是否可用:

```python
import flaggems_vllm
print(flaggems_vllm.config.has_c_extension)
print(flaggems_vllm.config.use_c_extension)
```

## 故障排除

### CUDA 未找到

如果未找到 CUDA,设置 `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation -e .
```

### Triton 版本不匹配

FlagGems-vllm 需要 Triton ≥ 3.0。如果遇到导入错误,升级 Triton:

```bash
pip install -U triton
```

或切换到 FlagTree:

```bash
pip install flagtree
export USE_FLAGTREE=1
```

### 缺少测试依赖

如果测试失败并有导入错误,安装测试额外依赖:

```bash
pip install -e '.[test]'
```

## 下一步

- [基础用法](/FlagGems-vllm/usage/basic/)
- [与 vllm-plugin-fl 集成](/FlagGems-vllm/usage/vllm-plugin/)
- [运行测试](/FlagGems-vllm/testing/unittest/)
- [运行基准测试](/FlagGems-vllm/performance/benchmark/)
