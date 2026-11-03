---
title: 调试与日志
weight: 50
---

# 调试与日志

针对 FlagGems-vllm 算子开发的调试工具、日志配置与常见问题定位技巧。

## 日志配置

FlagGems-vllm 使用 Python 标准 `logging` 模块输出信息。

### 开启调试日志

在脚本开头配置日志级别：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import flaggems_vllm
```

### 仅捕获 FlagGems-vllm 日志

如果需要过滤第三方库日志，可指定 logger：

```python
import logging
logger = logging.getLogger('flaggems_vllm')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
```

## 常见问题与定位

### 1. 遇到 `NotImplementedError`

- **现象**：调用算子时抛出 `NotImplementedError`。
- **原因**：传入的数据类型、张量形状、步长（stride）或参数组合不在算子支持的核函数路径中。
- **说明**：这是有意为之的架构设计（严格遵循 `NO_TORCH_COMPUTE_FALLBACK` 规则），禁止静默回退到低效的 PyTorch 算子。
- **排查与解决**：
  - 检查输入张量是否符合算子接口规范（如是否要求连续张量、指定的数据类型等）；
  - 查阅 `conf/operators.yaml` 确认算子支持的 stage 与配置；
  - 根据业务需要扩展该算子的 Triton 分支实现。

### 2. 数值精度差异

- **现象**：FlagGems-vllm 输出结果与 PyTorch 原生参考实现存在微小差异。
- **排查步骤**：
  1. 计算绝对误差与相对误差：
     ```python
     diff = (result - expected).abs().max()
     rel_diff = ((result - expected).abs() / (expected.abs() + 1e-7)).max()
     print(f"最大绝对误差: {diff}, 最大相对误差: {rel_diff}")
     ```
  2. 检查累加精度（如 Triton 内核中累加器是否使用了 FP32，以及浮点 FMA 顺序导致的精度微小波动）；
  3. 运行对应算子的单元测试：`pytest -v tests/test_<op>.py`。

### 3. 性能未达预期

- **现象**：算子实测耗时大于 PyTorch 原生基线。
- **排查建议**：
  1. 保证有充分的预热（Warmup）：Triton 首次运行时需要 JIT 编译内核；
  2. 检查是否加载了预调优配置：
     ```bash
     grep "my_operator" src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml
     ```
  3. 运行专用基准测试工具：
     ```bash
     pytest -q benchmark/test_<op>.py --level core --iter 100 --warmup 20
     ```

### 4. 导入错误 (`ImportError`)

- **现象**：`cannot import name 'my_operator'`。
- **排查建议**：
  - 检查算子是否已在 `src/flaggems_vllm/ops/__init__.py` 的 `__all__` 导出列表中；
  - 检查当前环境是否采用了可编辑模式安装：`pip install --no-build-isolation -e .`。

## 性能分析与 Profiling

### Triton 内置 Profiling

```python
import triton

with triton.profiler.Profile() as prof:
    result = flaggems_vllm.my_operator(x)

prof.print()
```

### NVIDIA Nsight Systems / Nsight Compute

使用系统级分析工具捕捉内核执行耗时与硬件指令流水瓶颈：

```bash
# Nsight Systems 系统级耗时追踪
nsys profile -t cuda,nvtx python test_script.py

# Nsight Compute 内核深度微架构分析
ncu --set full python test_script.py
```

## 下一步

- [单元测试指南](../testing/unittest.md)
- [性能基准测试](../performance/benchmark.md)
- [算子开发规范协议](../contribution/workflow.md)
