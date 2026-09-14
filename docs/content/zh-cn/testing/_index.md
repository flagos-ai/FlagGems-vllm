---
title: 测试
weight: 50
bookCollapseSection: true
---

# 测试

FlagGems-vllm 包含全面的功能测试来验证算子正确性。

## 主题

- [单元测试](unittest/) - 运行功能测试
- [测试覆盖率](coverage/) - 覆盖率报告

## 快速开始

使用快速模式运行所有测试:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --quick
```

运行特定算子测试:

```bash
PYTHONPATH=src pytest -v tests/test_grouped_topk.py
```

## 测试结构

测试遵循 `tests/test_<op>.py` 模式,匹配 `src/flaggems_vllm/ops/<op>.py`。

每个测试:
- 参数化覆盖形状和数据类型
- 与参考实现比较(vLLM 或 PyTorch)
- 使用 `accuracy_utils.py` 辅助函数(`gems_assert_equal`、`gems_assert_close`)
- 使用算子特定的 pytest 标记(如 `@pytest.mark.grouped_topk`)

## 测试选项

- `--quick`: 使用减少的形状/dtype 覆盖进行快速验证
- `--ref {device|cpu}`: 参考设备(默认: device)
- `--record`: 记录结果到 JSON
- `--output FILE`: 输出文件(默认: `accuracy_result.json`)

## 下一步

- [单元测试指南](unittest/)
- [覆盖率报告](coverage/)
