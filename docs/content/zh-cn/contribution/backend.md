---
title: 后端开发
weight: 30
---

# 硬件后端开发指南

为 FlagGems-vllm 增加新的异构硬件加速器后端支持。

## 概述

FlagGems-vllm 具备统一的多后端运行时架构，目前已支持 17 个硬件后端。每个后端模块主要承担：
- 芯片与设备识别逻辑；
- 架构特化的自动调优配置（如 Ampere vs Hopper）；
- 硬件专属算子或定制实现覆盖（可选）。

## 后端目录布局

后端目录位于 `src/flaggems_vllm/runtime/backend/`：

```
backend/
├── device_finder.py          # DeviceDetector 统一检测单例
├── common.py                 # 厂商注册与能力特性
├── _nvidia/                  # NVIDIA 后端
│   ├── tune_configs.yaml     # 预调优配置
│   ├── ampere/               # Ampere 架构配置
│   └── hopper/               # Hopper 架构配置
├── _amd/                     # AMD 后端
├── _ascend/                  # 华为昇腾后端
└── _<vendor>/                # 待新增的厂商后端
    ├── __init__.py
    └── tune_configs.yaml     # 可选调优配置
```

## 新增后端的标准步骤

### 步骤 1：创建后端目录

```bash
mkdir src/flaggems_vllm/runtime/backend/_myvendor
touch src/flaggems_vllm/runtime/backend/_myvendor/__init__.py
```

### 步骤 2：注册厂商并配置设备检测

1. 在 `src/flaggems_vllm/runtime/common.py` 中注册厂商信息：
   - 添加 `VendorInfo`（厂商名称、设备名称、调度键、FP64/BF16 特性支持标志）；
   - 在 `_VENDOR_TORCH_ATTR` 中添加对应的 PyTorch 扩展属性或在 `tools/` 提供检测脚本。
2. 如需扩展环境检测，在 `src/flaggems_vllm/runtime/backend/device_finder.py` 中添加自定义探测逻辑。

### 步骤 3：实现后端模块与特性能力

在 `src/flaggems_vllm/runtime/backend/_myvendor/__init__.py` 中：

```python
"""MyVendor backend for FlagGems-vllm."""

DEVICE_NAME = "myvendor"

def initialize():
    """在后端加载时调用"""
    pass

SUPPORTS_FP8 = False
SUPPORTS_BF16 = True
```

### 步骤 4：提供自动调优配置 (可选)

创建 `src/flaggems_vllm/runtime/backend/_myvendor/tune_configs.yaml`：

```yaml
grouped_topk:
  gen: true
  param_map:
    num_warps: [4, 8]
    num_stages: [2, 3]
    BLOCK_M: [64, 128]
    BLOCK_N: [64, 128]
```

### 步骤 5：在目标硬件上运行验证

```bash
export DNN_VENDOR=myvendor
PYTHONPATH=src pytest -q tests --quick
```

### 步骤 6：运行基准测试并对比性能

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level core --iter 100 --warmup 20 \
    --record --output myvendor_results.json
```

## 编译器选择

- **Triton 原生编译器**：主要支持 NVIDIA 与 AMD ROCm；
- **FlagTree 统一编译器**：面向广大国产芯片，通过多后端统一中间表示生成芯片目标代码。

## 下一步

- [贡献指南概述](overview.md)
- [算子开发协议](workflow.md)
- [单元测试指南](../testing/unittest.md)
