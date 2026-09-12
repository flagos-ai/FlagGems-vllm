---
title: 贡献指南
weight: 60
bookCollapseSection: true
---

# 为 FlagGems-vllm 做贡献

欢迎!我们非常感谢对 FlagGems-vllm 项目的贡献。

## 贡献方式

- **添加新算子** - 用于 vLLM 场景
- **优化现有算子** - 提升性能,添加自动调优配置
- **添加后端支持** - 非 NVIDIA 硬件
- **改进测试** - 扩展覆盖率,添加边界情况
- **编写文档** - 指南、示例
- **报告 bug** - 问题、性能回归

## 开始

1. 阅读[算子开发协议](workflow/)
2. 查看[贡献概述](overview/)
3. 搭建开发环境
4. 按照协议进行修改
5. 提交 pull request

## 算子开发协议

**必读**: 在添加或修改算子之前阅读 `workflow.md`。

关键要求:
- **NO_TORCH_COMPUTE_FALLBACK**: 生产代码不得使用 torch 计算算子
- **编码前表格**: 项目真相、torch 对齐契约、实现路径
- **自动调优必需**: NVIDIA 后端必须有调优配置
- **标准代码落点**: `src/flaggems_vllm/ops/<op>.py`、`tests/test_<op>.py`、`benchmark/test_<op>.py`
- **验收标准**: 核心形状的加速比 ≥ 0.9

协议定义了必须通过的硬 gate(G0–G7)。

## 代码风格

提交前运行 pre-commit:

```bash
pip install pre-commit
pre-commit run --all-files
```

配置的工具:
- **black**: 行长度 88
- **isort**: `--profile black`
- **flake8**: 最大行长度 120

## 测试要求

所有新算子必须包括:
- `tests/test_<op>.py` 中的功能测试
- `benchmark/test_<op>.py` 中的性能基准测试
- 测试标记: `@pytest.mark.<op>`

测试必须在 `--quick` 和完整模式下通过。

## 提交更改

1. Fork 仓库
2. 创建功能分支
3. 按照协议进行修改
4. 运行测试和 linting
5. 提交 pull request,包含:
   - 更改的清晰描述
   - 相关 issue 的链接(如有)
   - 测试结果(功能 + 性能)

## 沟通

- **GitHub Issues**: Bug 报告、功能请求
- **邮件**: flaggems@baai.ac.cn
- **微信**: 加入 FlagGems 社区群

## 主题

- [概述](overview/) - 贡献指南和约定
- [工作流协议](workflow/) - 算子开发协议
- [后端开发](backend/) - 添加新硬件后端

## 下一步

- [贡献概述](overview/)
- [算子开发协议](workflow/)
