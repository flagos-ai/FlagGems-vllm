---
title: 安装指南
weight: 20
---

# 安装指南

FlagGems-vllm 是一个纯 Python 的 `setuptools` 包，运行在厂商提供的
PyTorch/Triton/vLLM 环境之上。它**不会**声明或安装 PyTorch、Triton、
FlagTree 或 vLLM——这些依赖必须与目标加速卡匹配，由厂商镜像或
`tools/setup.sh` 安装。

## 前置依赖

### Python

包元数据支持 Python ≥ 3.8。`src/flaggems_vllm/backends.yaml` 中固定了各厂商工具链验证过的
版本（目前为 3.10、3.11 或 3.12）。

### 构建工具

本包使用 `setuptools`。可编辑安装前请先安装构建后端和辅助工具：

```bash
python -m pip install -U "setuptools>=64,<82" wheel packaging
```

只有在构建可选的原生扩展或从源码构建厂商 wheel 时才需要
`scikit-build-core`、`pybind11`、`cmake`、`ninja`；`tools/setup.sh` 会在虚拟环境中
自动安装它们。

### FlagGems 依赖

**FlagGems-vllm 依赖 FlagGems**。部分 FlagGems-vllm 算子在底层会直接调用
FlagGems 的通用算子。请从 FlagOS 索引安装匹配的 `flag-gems`（当前文档和 CI
使用 5.4.0）：

```bash
python -m pip install --no-deps "flag-gems==5.4.0" \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
```

当厂商镜像已经提供 PyTorch/NumPy 时请使用 `--no-deps`。在干净环境中，请按下文
手动安装小节显式安装 FlagGems 的少量纯 Python 依赖（`packaging`、`PyYAML`、
`sqlalchemy`）。

### PyTorch、Triton 与 vLLM

在安装 FlagGems-vllm **之前**，先安装与目标加速卡兼容的 PyTorch、编译器
和 vLLM。NVIDIA 示例：

```bash
# 仅为示例，请使用与驱动匹配的 CUDA/torch 组合。
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

其他后端请遵循厂商的 PyTorch/Triton 安装说明。经过验证的组合统一记录在
`src/flaggems_vllm/backends.yaml` 的 `deps`、`triton`、`flagtree`、`vllm` 字段中，
`tools/setup.sh` 以该文件为唯一数据源。

## 从源码安装

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm

# 开发 / 可编辑安装
python -m pip install --no-build-isolation -e .

# 安装基础测试依赖
python -m pip install --no-build-isolation -e '.[test]'
```

`[test]` 只包含 `pytest`、`scipy`、`einops`、`distro`。仅 CUDA 需要的
`cupy-cuda12x` 放在独立的 `[test-nvidia]` extra 中，避免非 NVIDIA 后端把
CUDA wheel 装进厂商环境：

```bash
# 仅 NVIDIA
python -m pip install --no-build-isolation -e '.[test,test-nvidia]'
```

## 使用 `tools/setup.sh` 进行多后端环境搭建

`tools/setup.sh` 仿照 FlagGems 的环境搭建流程：读取 `src/flaggems_vllm/backends.yaml`，
创建 `uv` 虚拟环境，依次安装厂商 torch 栈、vLLM、FlagGems、FlagGems-vllm，
可选地用 FlagTree 替换 Triton，并把厂商环境写入 `.venv/bin/activate`。

### 查看后端列表

```bash
bash tools/setup.sh --list
```

常用配置与别名：

| 别名 | 配置 | 厂商 | vLLM 来源 |
| --- | --- | --- | --- |
| `nvidia` | `nvidia-cuda133`（默认） | NVIDIA | PyPI `vllm==0.20.2` |
| `nvidia-cuda128` | `nvidia-cuda128` | NVIDIA | PyPI `vllm==0.20.2` |
| `ascend` | `ascend-cann900` | 华为昇腾 | `vllm-ascend==0.24.0rc1` |
| `ascend-cann850` | `ascend-cann850` | 华为昇腾 | `vllm-ascend==0.24.0rc1` |
| `hygon` | `hygon-dtk2604` | 海光 DCU | 厂商镜像（`--no-venv`） |
| `metax` | `metax-maca3810` | 沐曦 MetaX | 厂商 `vllm-metax` wheel |
| `iluvatar` | `iluvatar-corex450` | 天数智芯 | 厂商镜像（`--no-venv`） |
| `mthreads` | `mthreads-musa520` | 摩尔线程 | 厂商镜像（`--no-venv`） |
| `cambricon` | `cambricon-neuware472` | 寒武纪 MLU | 厂商镜像（`--no-venv`） |
| `enflame` | `enflame-tops1106` | 燧原 GCU | 厂商镜像（`--no-venv`） |
| `kunlunxin` | `kunlunxin-xre5371` | 昆仑芯 | 厂商镜像（`--no-venv`） |
| `thead` | `thead-ppu210` | 平头哥 PPU | 厂商 `vllm` wheel |
| `tsingmicro` | `tsingmicro-tsm260610` | 清微智能 | 厂商镜像（`--no-venv`） |
| `biren`、`spacemit`、`sunrise` | 对应配置 | 厂商 | 厂商镜像（`--no-venv`） |
| `amd` | `amd-rocm` | AMD ROCm | 先装 ROCm vLLM，再 `--no-venv` |

### 创建虚拟环境

```bash
# 直接执行：创建 .venv 并打印激活命令
bash tools/setup.sh nvidia

# 或 source：创建 .venv 并在当前 shell 中激活
source tools/setup.sh nvidia
```

支持的运行模式与选项：

```text
--no-venv        安装到当前 Python 环境，而不是创建虚拟环境
--skip-vllm      不安装 vLLM（镜像已提供时使用）
--skip-compiler  不替换/安装 Triton 或 FlagTree
--skip-smoke     跳过导入和 pytest 收集检查
--dry-run        只打印命令，不实际执行
```

### 环境变量

所有默认值来自 `src/flaggems_vllm/backends.yaml`，可通过环境变量覆盖：

| 变量 | 含义 |
| --- | --- |
| `PYTHON` | `--no-venv` 使用的解释器（默认 `python3`）。 |
| `PYTHON_VERSION` | 虚拟环境的 Python 版本。 |
| `VENV_DIR` | 虚拟环境目录（默认 `.venv`）。 |
| `PIP_INDEX_URL` | 主包索引（默认 PyPI；网络较慢时可改为国内镜像）。 |
| `FLAG_GEMS_VERSION` | 覆盖 registry 中的 `flag-gems` 版本（默认 5.4.0）。 |
| `FLAG_GEMS_INDEX` | 发布 `flag-gems` 的索引（默认 FlagOS hosted 索引）。 |
| `VLLM_VERSION` | 覆盖配置中 `vllm==...` 形式的版本（NVIDIA）。 |
| `VLLM_PACKAGE` | 完整的 vLLM 依赖或 wheel URL，优先级高于 `VLLM_VERSION`。 |
| `COMPILER` | `flagtree`、`triton` 或 `keep`。 |
| `USE_FLAGTREE` | `1` 强制 FlagTree，`0` 强制 Triton。 |
| `FLAGTREE_VERSION` / `TRITON_PACKAGE` | 覆盖编译器版本。 |
| `SKIP_BACKEND_DEPS` | `1` 表示不改动厂商 torch 栈。 |
| `SKIP_FLAGGEMS` / `SKIP_PROJECT` | 跳过 FlagGems / FlagGems-vllm 安装。 |
| `STRICT_TRITON_IMPORT` | `1` 表示 `import triton` 失败即报错。 |
| `STRICT_PIP_CHECK` | `1` 表示依赖冲突即报错。 |
| `UV_REFRESH` | `1` 强制刷新 uv 缓存。 |
| `CUDA_HOME` | NVIDIA 配置使用的 CUDA 路径。 |

示例：

```bash
# NVIDIA，指定 vLLM 版本
VLLM_PACKAGE="vllm==0.20.2" bash tools/setup.sh nvidia

# 昇腾，保留厂商编译器
COMPILER=keep bash tools/setup.sh ascend

# 沐曦厂商镜像内：只安装本仓库和测试依赖
bash tools/setup.sh --no-venv metax
```

在 `--no-venv` 模式下，`SKIP_BACKEND_DEPS=1` 且默认不安装 vLLM。镜像未提供时，
可显式开启：

```bash
SKIP_BACKEND_DEPS=0 VLLM_PACKAGE="vllm-ascend==0.24.0rc1" \
  bash tools/setup.sh --no-venv ascend
```

## 不创建虚拟环境的手动安装

当厂商镜像已经提供匹配的 Python、PyTorch、Triton/FlagTree 和 vLLM 时，推荐使用
该方式，只补充 FlagGems-vllm 和测试工具。它等价于
`bash tools/setup.sh --no-venv <backend>`，但每一步都可自行控制。

### 1. 激活厂商环境

使用厂商镜像预期的 Python：

```bash
# 示例
source /usr/local/Ascend/cann/set_env.sh          # 昇腾
source /opt/dtk-26.04/env.sh                      # 海光
source /usr/local/PPU_SDK/envsetup.sh             # 平头哥
# 或：conda activate <vendor-env>
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

除非你准备同时安装匹配的 torch/triton/vLLM，否则**不要**新建虚拟环境。

### 2. 安装镜像未提供的 Python 依赖

```bash
# FlagGems-vllm 自身需要
python -m pip install "setuptools>=64,<82" wheel packaging pybind11 PyYAML sqlalchemy

# FlagGems 运行时
python -m pip install --no-deps "flag-gems==5.4.0" \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
```

### 3. 安装 FlagGems-vllm，且不替换厂商栈

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm

# --no-deps 很重要：避免 pip 用通用 flag-gems（以及 NVIDIA 上的 CUDA 测试依赖）
# 覆盖厂商提供的包。
python -m pip install --no-build-isolation --no-deps -e .
```

在厂商镜像中请保留 `--no-deps`：本项目声明了 `flag-gems`，其元数据固定了
`numpy`、`packaging`、`sqlalchemy` 的版本，让 pip 解析项目依赖可能替换厂商
numpy。请按上面的方式显式安装运行时依赖。如果是在干净的 NVIDIA 环境中、不担心
这个问题，也可以去掉 `--no-deps`：

```bash
python -m pip install --no-build-isolation -e .
```

### 4. 安装测试依赖

与后端无关的基础测试集：

```bash
python -m pip install \
  "pytest>=7.1.0" \
  "scipy>=1.14,<1.18.0" \
  "einops>=0.6" \
  "distro>=1.9.0"
```

NVIDIA 额外安装 CUDA 专用依赖：

```bash
python -m pip install "cupy-cuda12x>=12.0.0,<14.0.0"
```

也可以直接安装 `pyproject.toml` 中声明的 extras：

```bash
# 与后端无关
python -m pip install --no-build-isolation -e '.[test]'

# NVIDIA
python -m pip install --no-build-isolation -e '.[test,test-nvidia]'
```

注意：这种模式会让 pip 同时解析项目的 `flag-gems` 依赖。`flag-gems` 固定了
`numpy`、`packaging`、`sqlalchemy` 的版本，可能与厂商固定的 numpy 冲突。在
厂商镜像中，优先使用上面的显式命令（或脚本的 `--no-venv` 模式），它们用
`--no-deps` 安装本项目：

依赖 vLLM 的测试（约 35 个测试文件）需要与厂商栈匹配的 vLLM。请使用厂商镜像
或配置中声明的包：

```bash
# NVIDIA（PyPI）
python -m pip install "vllm==0.20.2"

# 昇腾
python -m pip install "vllm-ascend==0.24.0rc1"

# 沐曦 / 平头哥：使用 src/flaggems_vllm/backends.yaml 中的 wheel URL
# 其他厂商：使用厂商镜像自带的 vLLM
```

未安装 vLLM 时，这些测试会自动 skip，其余测试仍可运行。

### 5. 指定后端

`flaggems_vllm` 会自动探测加速卡，但在多厂商镜像中显式指定更可靠：

```bash
export FLAGGEMS_VENDOR=ascend   # nvidia、ascend、hygon、metax ...
export DNN_VENDOR=ascend        # 兼容旧文档
```

### 6. 验证

```bash
# 设备、厂商与 float32 冒烟检查
python tools/check_backend_env.py \
  --expected-vendor "${FLAGGEMS_VENDOR}" \
  --require-flaggems

# 测试发现
python -m pytest tests --collect-only -q

# 快速功能测试
python -m pytest tests -q --quick
```

在仓库根目录运行测试并不强制要求可编辑安装：`pytest.ini` 已设置
`pythonpath = src`，因此仅设置 `PYTHONPATH=src` 也能完成导入。仍建议安装，
因为 `importlib.metadata.version` 需要 `flaggems_vllm` 的包元数据。

## 与 FlagGems 和 vllm-plugin-fl 一起安装

完整的 FlagOS vLLM 插件栈需按以下顺序安装：

```bash
# 1. 安装 FlagGems（通用算子后端）
python -m pip install --no-deps "flag-gems==5.4.0" \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/

# 2. 安装 FlagGems-vllm（vLLM 专用算子）
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
python -m pip install --no-build-isolation -e .
cd ..

# 3. 安装 vllm-plugin-fl（vLLM 插件集成）
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
python -m pip install --no-build-isolation -e .
```

如果安装了多个 vLLM 插件，选择 FlagOS 插件：

```bash
export VLLM_PLUGINS=fl
```

## 故障排查

### 未找到 CUDA

如果未找到 CUDA，设置 `CUDA_HOME`：

```bash
export CUDA_HOME=/usr/local/cuda
python -m pip install --no-build-isolation -e .
```

### Triton 版本不匹配

FlagGems-vllm 要求 Triton ≥ 3.0。请优先使用 `src/flaggems_vllm/backends.yaml` 中固定的
厂商 Triton/FlagTree；直接 `pip install -U triton` 可能替换厂商编译器并破坏
加速卡后端。

```bash
# 查看后端验证过的编译器
grep -A6 '^  ascend-cann900:' src/flaggems_vllm/backends.yaml

# 或交给安装脚本
bash tools/setup.sh ascend
```

### 缺少测试依赖

```bash
python -m pip install --no-build-isolation -e '.[test]'
# 仅 NVIDIA
python -m pip install --no-build-isolation -e '.[test-nvidia]'
```

### vLLM 导入错误

vLLM wheel 必须与厂商 torch 匹配。在厂商镜像中，不要用 PyPI vLLM 覆盖镜像自带
的版本；使用本仓库/测试安装路径：

```bash
bash tools/setup.sh --no-venv <backend>
```

## 下一步

- [基础用法](../usage/basic.md)
- [非 NVIDIA 后端](../usage/non-nvidia.md)
- [与 vllm-plugin-fl 集成](../usage/vllm-plugin.md)
- [运行测试](../testing/unittest.md)
- [运行基准测试](../performance/benchmark.md)
