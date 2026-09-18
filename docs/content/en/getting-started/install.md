---
title: Installation
weight: 20
---

# Installation Guide

## Prerequisites

### Required Dependencies

Install build tools before installing FlagGems-vllm:

```bash
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```

### PyTorch

Install a PyTorch build compatible with your target accelerator **before** installing FlagGems-vllm. FlagGems-vllm does not reinstall PyTorch automatically.

For NVIDIA GPUs with CUDA 12.1+:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For other backends, follow your vendor's PyTorch installation instructions.

## Install from Source

### Basic Installation

Clone and install FlagGems-vllm:

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install .
```

### Development Installation

For development with editable mode:

```bash
pip install --no-build-isolation -e .
```

### With Test Dependencies

To run tests and benchmarks:

```bash
pip install --no-build-isolation -e '.[test]'
```

This installs pytest, numpy, scipy, cupy-cuda12x, and other test utilities.

## Full GPU Environment Setup

Use the provided `tools/setup.sh` script for a complete environment bootstrap:

```bash
source tools/setup.sh
```

This script:
- Creates a `uv` virtual environment
- Installs vLLM and matching PyTorch
- Installs FlagGems-vllm
- Optionally swaps Triton for FlagTree (if `USE_FLAGTREE=1`)
- Runs smoke tests

### Environment Variables

The setup script honors these variables:

- `DNN_VENDOR`: Backend vendor (e.g., `nvidia`, `amd`, `ascend`). Default: auto-detect
- `VLLM_VERSION`: vLLM version to install. Default: latest
- `USE_FLAGTREE`: Set to `1` to use FlagTree compiler instead of Triton
- `CUDA_HOME`: CUDA installation path. Default: auto-detect

Example for AMD:
```bash
DNN_VENDOR=amd source tools/setup.sh
```

## Install with FlagGems and vllm-plugin-fl

For the full FlagOS vLLM plugin stack, install in this order:

```bash
# 1. Install FlagGems (general operator backend)
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .
cd ..

# 2. Install FlagGems-vllm (vLLM-specific operators)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install --no-build-isolation -e .
cd ..

# 3. Install vllm-plugin-fl (vLLM plugin integration)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
pip install --no-build-isolation -e .
```

If multiple vLLM plugins are installed, select the FlagOS plugin:

```bash
export VLLM_PLUGINS=fl
```

## Verify Installation

### Import Smoke Test

```bash
python -c "
import torch
import flaggems_vllm
from flaggems_vllm import runtime

print('torch:', torch.__version__)
print('vendor:', flaggems_vllm.vendor_name)
print('device:', flaggems_vllm.device)
print('device count:', runtime.device.device_count)
print('grouped_topk:', callable(flaggems_vllm.grouped_topk))
"
```

### Run Quick Tests

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
PYTHONPATH=src pytest -q tests --quick
```

### Run a Benchmark Smoke Test

```bash
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py --level core --iter 1 --warmup 1
```

## Backend Selection

FlagGems-vllm auto-detects the hardware backend. To override:

```bash
export DNN_VENDOR=nvidia  # or amd, ascend, hygon, etc.
```

Available backends: `nvidia`, `amd`, `ascend`, `cambricon`, `enflame`, `hygon`, `iluvatar`, `kunlunxin`, `metax`, `mthreads`, and more (18 total).

## C++ Extension

FlagGems-vllm includes optional C++ operators compiled via pybind11. These are built automatically if the required dependencies are present.

To enable C++ operators at runtime:

```bash
export USE_C_EXTENSION=1
```

Check if C++ extension is available:

```python
import flaggems_vllm
print(flaggems_vllm.config.has_c_extension)
print(flaggems_vllm.config.use_c_extension)
```

## Troubleshooting

### Missing CUDA

If CUDA is not found, set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
pip install --no-build-isolation -e .
```

### Triton Version Mismatch

FlagGems-vllm requires Triton ≥ 3.0. If you see import errors, upgrade Triton:

```bash
pip install -U triton
```

Or switch to FlagTree:

```bash
pip install flagtree
export USE_FLAGTREE=1
```

### Missing Test Dependencies

If tests fail with import errors, install test extras:

```bash
pip install -e '.[test]'
```

## Next Steps

- [Basic usage patterns](/FlagGems-vllm/usage/basic/)
- [Integration with vllm-plugin-fl](/FlagGems-vllm/usage/vllm-plugin/)
- [Running tests](/FlagGems-vllm/testing/unittest/)
- [Running benchmarks](/FlagGems-vllm/performance/benchmark/)
