---
title: 单元测试
weight: 10
---

# 单元测试指南

FlagGems-vllm 的功能测试套件用于验证算子相比参考实现的计算精度与数值正确性。

## 执行测试

### 收集并查看所有测试

检查 Pytest 是否能正常识别所有测试：

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
```

### 快速验证模式 (`--quick`)

在开发调试期间，推荐使用 `--quick` 模式。该模式会缩小形状与数据类型测试组合，极大缩短测试耗时：

```bash
PYTHONPATH=src pytest -q tests --quick
```

### 完整回归测试

执行完整的全量参数化测试：

```bash
PYTHONPATH=src pytest -q tests
```

### 针对单个算子执行测试

```bash
# 测试 Grouped TopK 算子
PYTHONPATH=src pytest -v tests/test_grouped_topk.py

# 测试 MoE Token 对齐算子
PYTHONPATH=src pytest -v tests/test_moe_align_block_size.py

# 快速验证特定融合算子
PYTHONPATH=src pytest -q tests/test_fused_inv_rope_fp8_quant.py --quick
```

### 查看详细输出与日志

配合 `-s` 选项禁用输出捕获，实时查看标准输出与 print 内容：

```bash
PYTHONPATH=src pytest -v -s tests/test_flash_attention_forward.py
```

## 测试命令行选项

### `--quick`

精简测试矩阵，仅测试核心维度与主要数据类型（如仅测试 float16），大幅提升迭代效率。

### `--ref {device|cpu}`

指定运行基准参考实现的设备环境：
- `device` (默认): 在与待测算子相同的设备（如 CUDA）上运行基准实现。
- `cpu`: 在 CPU 上执行基准实现（在 GPU 显存紧张或需要隔离验证时非常有用）。

示例：
```bash
PYTHONPATH=src pytest -q tests --ref cpu
```

### `--record` 与 `--output`

将测试执行详情与数值最大差值记录到 JSON 文件中：

```bash
PYTHONPATH=src pytest -q tests --quick --record --output test_results.json
```

## 测试用例结构规范

典型测试文件结构如下：

```python
import pytest
import torch
from tests.accuracy_utils import to_reference, gems_assert_close

@pytest.mark.grouped_topk
@pytest.mark.parametrize("shape", [(8, 16), (32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_grouped_topk(shape, dtype):
    # 1. 准备输入
    scores = torch.randn(shape, device='cuda', dtype=dtype)

    # 2. 调用 FlagGems-vllm 算子
    result = flaggems_vllm.grouped_topk(scores, ...)

    # 3. 调用参考基准实现
    ref_result = reference_grouped_topk(to_reference(scores), ...)

    # 4. 精度断言比对
    gems_assert_close(result, ref_result, rtol=1e-3, atol=1e-4)
```

## 精度断言工具 (`tests/accuracy_utils.py`)

- `to_reference(tensor)`: 将张量转移至 `--ref` 指定的参考设备；
- `gems_assert_equal(result, expected)`: 严格逐位或整型完全相等断言；
- `gems_assert_close(result, expected, rtol, atol)`: 浮点误差范围内的数值接近性断言。

## 下一步

- [测试覆盖率](coverage.md)
- [基准测试指南](../performance/benchmark.md)
- [贡献指南概述](../contribution/overview.md)
