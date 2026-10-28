---
title: 非 NVIDIA 后端
weight: 40
---

# 在非 NVIDIA 硬件后端上运行

FlagGems-vllm 具备出色的硬件异构扩展能力，共支持 **17 个硬件后端**（NVIDIA 加上 16 个非 NVIDIA 后端）。

## 支持的后端

- **AMD**: 基于 ROCm 的 GPU
- **昇腾 (Huawei Ascend)**: 昇腾 910B 及更新系列
- **寒武纪 (Cambricon)**: MLU 系列加速卡
- **燧原科技 (Enflame)**: GCU 系列加速卡
- **海光 (Hygon)**: DCU 系列加速卡
- **天数智芯 (Iluvatar)**: CoreX 系列 GPU
- **昆仑芯 (Kunlunxin)**: XPU 系列加速卡
- **沐曦 (MetaX)**: 曦云 C 系列 GPU
- **摩尔线程 (Moore Threads)**: MTT 系列 GPU
- **AIPU**, **ARM**, **壁仞 (Biren)**, **进迭时空 (SpacemiT)**, **曦望 (Sunrise)**, **平头哥 (T-Head)**, **清微智能 (Tsingmicro)**

完整支持列表参见 `src/flaggems_vllm/runtime/backend/`。

## 后端选择方式

### 自动检测

FlagGems-vllm 在初始化时会自动检测当前机器上的硬件加速卡类型：

```python
import flaggems_vllm

print(f"检测到的厂商: {flaggems_vllm.vendor_name}")
print(f"设备名称: {flaggems_vllm.device}")
```

### 手动指定

通过设置 `DNN_VENDOR` 环境变量强制覆盖自动检测结果：

```bash
export DNN_VENDOR=amd
python your_script.py
```

或者在 Python 脚本导入前设置：

```python
import os
os.environ['DNN_VENDOR'] = 'ascend'

import flaggems_vllm
```

## 典型后端配置说明

### AMD (ROCm)

安装 ROCm 对应版本的 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
export DNN_VENDOR=amd
```

ROCm 环境下可直接使用原生 Triton 编译器。

### 华为昇腾 (Ascend)

需要安装华为 CANN 工具包以及配套的 Torch-NPU：

```bash
# 安装配套的 Torch-NPU
pip install torch-npu

export DNN_VENDOR=ascend
```

在昇腾环境下推荐使用 FlagTree 多芯片统一编译器：

```bash
pip install flagtree
export USE_FLAGTREE=1
```

### 其他国产芯片与加速硬件

通常需要遵循芯片厂商提供的特定 PyTorch 构建包或适配插件，确保能够通过厂商的 Torch 拓展识别加速设备。

## 编译器选择：Triton 与 FlagTree

- **OpenAI Triton**：原生支持 NVIDIA 和 AMD (ROCm)。
- **FlagTree 编译器**：FlagOS 推出的多架构统一编译器，支持国产多种不同硬件后端。

启用 FlagTree 编译：

```bash
pip install flagtree
export USE_FLAGTREE=1
```

## 硬件后端成熟度一览

| 硬件后端 | 成熟度 | 自动调优配置 | 说明 |
|---------|----------|----------|-------|
| NVIDIA  | ★★★★★   | 完备 | 提供针对 Ampere 和 Hopper 架构的预调优配置 |
| AMD     | ★★★☆☆   | 部分 | 具备基线调优配置，持续优化中 |
| 华为昇腾 | ★★★☆☆   | 具备专用内核 | 支持专用算子扩展，推荐搭配 FlagTree |
| 摩尔线程/海光/天数等 | ★★★☆☆ | 部分 | 已提供部分融合算子调优与适配 |
| 其他厂商 | ★★☆☆☆   | 基础功能 | 基础功能适配与连通性验证 |

## 故障排查

### 后端未检测成功

检查对应的 PyTorch 设备拓展：

```python
import torch
print("CUDA 是否可用:", torch.cuda.is_available())

# 针对昇腾
try:
    import torch_npu
    print("NPU 是否可用:", torch_npu.npu.is_available())
except ImportError:
    pass
```

若自动检测未匹配，请显式 export `DNN_VENDOR=<vendor>`。

### 算子报错或未实现

某些算子可能需要特定的硬件指令或编译器功能扩展：
- 查阅 `conf/operators.yaml` 获取算子支持矩阵；
- 遇到不支持配置时算子会抛出 `NotImplementedError`；
- 欢迎向社区反馈问题或贡献后端适配代码。

## 下一步

- [后端开发指南](../contribution/backend.md)
- [单元测试指南](../testing/unittest.md)
- [性能基准测试](../performance/benchmark.md)
