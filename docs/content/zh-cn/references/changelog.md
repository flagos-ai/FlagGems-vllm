---
title: 更新日志
weight: 30
---

# 更新日志

FlagGems-vllm 的版本历史与发布说明。

## 未发布

### 工具链

- `tools/setup.sh` 改为由 `src/flaggems_vllm/backends.yaml` 注册表驱动，支持 21 个后端配置
  （NVIDIA、昇腾、海光、沐曦、天数智芯、摩尔线程、寒武纪、燧原、昆仑芯、
  平头哥、清微智能、壁仞、进迭时空、曦望以及 AMD ROCm），每个配置固定
  Python、torch、Triton/FlagTree 和 vLLM 版本。
- 新增 `--no-venv` 模式，用于已提供匹配 torch/triton/vLLM 栈的厂商镜像；
  同时新增 `--list`、`--dry-run`、`--skip-vllm`、`--skip-compiler`、
  `--skip-smoke`。
- 将仅 CUDA 需要的 `cupy-cuda12x` 从 `[test]` 拆分到 `[test-nvidia]`，
  避免非 NVIDIA 后端安装带 CUDA 的 wheel。
- 运行时在选择后端时新增对 `DNN_VENDOR` 的支持。

## v0.1.0 (当前版本)

FlagGems-vllm 作为独立 vLLM 算子库的首发版本。

### 特性

- **105 个算子**（157 个导出项），针对 vLLM 推理场景深度调优
- **17 个硬件后端**支持（NVIDIA, AMD, 华为昇腾, 海光, 天数智芯等）
- **MoE 核心算子**: `moe_align_block_size`、`grouped_topk`、`fused_experts_impl`、`moe_sum`
- **Flash Attention**: `flash_attention_forward`、`flash_attn_varlen_func`、`flash_mla`、`triton_unified_attention`
- **量化算子**: FP8、INT8、FP4 激活与权重分块量化算子
- **融合内核**: RMS norm、RoPE、注意力机制、激活函数融合
- **模型专用算子**: Qwen4、DeepSeek-v4、RWKV、M3 等

### 架构

- 基于 Triton 实现，支持 FlagTree 多芯片统一编译器
- 针对 NVIDIA Ampere/Hopper 架构预设调优配置
- 支持基于 pybind11 的 C++ 扩展（可选）
- 提供直接导入调用接口，与 vLLM 插件层深度集成

### 测试与持续集成 (CI)

- 功能测试套件，提供 `--quick` 模式进行快速验证
- 性能基准测试套件，覆盖 `benchmark/core_shapes.yaml` 核心形状
- 基于 GitHub Actions 的多后端 CI
- 基于 `tools/select_tests.py` 的改动代码自动关联测试筛选

### 文档

- 基于 Hugo 的中英双语文档站点
- 严格的算子开发规范协议 (`workflow.md`)
- MiniMax M3 算子支持规范与性能评测

### 已知问题

- 部分算子仍处于 alpha/beta 阶段（详见[算子列表](operators.md)）
- 非 NVIDIA 后端的自动调优配置仍在持续完善
- 编译 C++ 扩展需要显式开启 `USE_C_EXTENSION=1`

## 未来规划

### v0.2.0 (计划中)

- 更多 MoE 结构优化
- 拓展 Flash Attention 更多变体与上下文长度
- 扩展多后端自动调优覆盖面
- 接入性能跟踪数据库
- 增强调试与性能分析工具链

## 参与贡献

参考[贡献指南](../contribution/_index.md)了解如何向 FlagGems-vllm 提交代码。

## 版本规范

FlagGems-vllm 遵循语义化版本规范：
- **主版本号** (x.0.0): 不兼容的 API 修改
- **次版本号** (0.x.0): 向下兼容的功能性新增或新算子
- **修订号** (0.0.x): 向下兼容的问题修复与性能优化
