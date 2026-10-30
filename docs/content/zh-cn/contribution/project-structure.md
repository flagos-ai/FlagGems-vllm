---
title: 项目结构
weight: 40
---

# 项目结构

FlagGems-vllm 仓库组织与模块架构。

## 顶层目录结构

```
FlagGems-vllm/
├── src/flaggems_vllm/      # Python 核心包
│   ├── __init__.py         # 顶层导出与公开算子暴露
│   ├── config.py           # 配置与 C 扩展检测
│   ├── ops/                # 算子实现
│   ├── runtime/            # 后端管理、设备检测、自动调优配置
│   └── utils/              # libentry、libtuner、辅助工具
├── tests/                  # 功能测试套件
├── benchmark/              # 性能基准评测
├── conf/                   # 算子元数据 (operators.yaml)
├── tools/                  # CI 脚本、环境配置、测试过滤
├── docs/                   # 文档站点 (Hugo)
├── workflow.md             # 算子开发协议规范
├── pyproject.toml          # 打包元数据与构建配置
└── pytest.ini              # Pytest 配置
```

## 源码目录 (`src/flaggems_vllm/`)

### `ops/` - 算子实现

```
ops/
├── __init__.py             # 重新导出所有算子与 __all__（共 157 个导出项）
├── <op>.py                 # 根目录下 80+ 个单文件算子（如 grouped_topk.py）
├── DSA/                    # 领域专用架构算子 (Domain-Specific Architectures)
├── FLA/                    # Flash Linear Attention 算子（含 gdn2_native）
├── mhc/                    # 多头计算算子 (Multi-Head Computation)
└── qwen4/                  # Qwen4 模型专用算子
```

每个算子文件通常包含：
- Host 侧调度函数（数据类型/形状校验、路由逻辑）
- 一个或多个 `@triton.jit` 内核函数
- 通过 `@libtuner()` 接入的自动调优集成

### `runtime/` - 后端与设备管理

```
runtime/
├── __init__.py             # 设备检测、get_tuned_config()、replace_customized_ops()
├── common.py               # 厂商注册表、VendorInfo、硬件特性能力集
├── register.py             # 调度注册器类 (PyTorch dispatch)
├── configloader.py         # 加载 tune_configs.yaml
├── flagtune.py             # FlagTune 自动调优集成
└── backend/
    ├── device_finder.py    # DeviceDetector 单例
    ├── device.py           # 设备辅助抽象
    ├── backend_utils.py    # 后端辅助工具与动态导入
    ├── _nvidia/            # NVIDIA 后端
    │   ├── tune_configs.yaml    # 预调优配置
    │   ├── ampere/              # Ampere 架构配置
    │   └── hopper/              # Hopper 架构配置
    ├── _amd/               # AMD 后端
    ├── _ascend/            # 华为昇腾后端
    └── ...                 # 共 17 个硬件后端支持
```

### `utils/` - 工具与辅助库

```
utils/
├── libentry.py             # @libentry() 与 @libtuner() 装饰器
├── pointwise_dynamic.py    # 动态逐元素代码生成
├── shape_utils.py          # 形状、广播与步长偏移量计算
├── type_utils.py           # 数据类型转换与辅助
├── triton_lang_helper.py   # Triton 语言兼容垫片
├── triton_driver_helper.py # Triton 设备驱动属性查询
├── triton_version_utils.py # Triton 及 TLE 特性版本检查
└── models/                 # 模型专用辅助模块
```

## 测试套件 (`tests/`)

```
tests/
├── conftest.py             # Pytest 钩子、--quick/--ref/--record 选项
├── accuracy_utils.py       # to_reference()、gems_assert_*() 精度比对
├── test_<op>.py            # 每个算子的独立功能测试
├── test_FLA/               # Flash Linear Attention 测试套件
└── test_DSA/               # DSA 测试套件
```

测试文件命名遵循 `test_<op>.py`，与 `ops/<op>.py` 对应，便于 CI 精准测试定位。

## 性能基准评测 (`benchmark/`)

```
benchmark/
├── conftest.py             # Pytest 钩子、--level/--iter/--warmup 选项
├── core_shapes.yaml        # 每个算子的核心负载形状
├── test_<op>.py            # 单算子性能基准测试
├── test_FLA/               # FLA 基准测试
└── base.py                 # 基准测试运行与统计工具
```

## 配置目录 (`conf/`)

```
conf/
└── operators.yaml          # 算子元数据 (105 个算子)
                            # - id, description, for, labels, kind, stages
```

## 工具脚本 (`tools/`)

```
tools/
├── ci_checks/              # CI 校验脚本（YAML 完整性、导出排序、标记等）
├── gpu_check_<vendor>.sh   # 13 款芯片厂商硬件环境检测脚本
├── setup.sh                # 完整 GPU 运行环境一键部署
├── select_tests.py         # CI：根据改动文件映射测试目标
├── select_backends.py      # CI：多后端测试目标过滤
└── run_ci_targets.py       # CI：执行筛选后的测试与 Benchmark
```

## 开发协议规范 (`workflow.md`)

强制性算子开发协议规范，设有严格的门禁（G0–G7）：
- NO_TORCH_COMPUTE_FALLBACK 无 PyTorch 计算回退规则
- 自动调优要求
- 测试与性能验收标准

**新增或修改算子前，必须阅读 `workflow.md`。**

## 打包元数据 (`pyproject.toml`)

- 包名：`flaggems_vllm`
- 构建系统：`scikit-build-core`（通过 pybind11 编译可选 C++ 扩展）
- 依赖项：`torch>=2.6.0`, `pyyaml`, `triton` 或 `flagtree`
- 可选测试依赖：`[test]` (pytest, numpy, scipy, cupy)

## 下一步

- [贡献指南概述](overview.md)
- [算子开发协议](workflow.md)
- [硬件后端开发指南](backend.md)
- [MiniMax M3 算子支持](../references/minimax-m3-support.md)
- [单元测试指南](../testing/unittest.md)
