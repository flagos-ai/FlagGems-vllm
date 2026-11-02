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

### FlagGems Dependency

**FlagGems-vllm depends on FlagGems**. Several FlagGems-vllm operators invoke general FlagGems operators internally. Before installing FlagGems-vllm, install `flag-gems==5.4.0` via pip (hosted at [FlagOS PyPI repository](https://resource.flagos.net/#browse/browse:flagos-pypi-hosted:flag-gems%2F5.4.0)):

```bash
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
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

### Native indexer baseline dependencies

The `fused_indexer_weight_scale` tests and benchmark directly import the native
GLM-5.3 function from vLLM 0.33.0. The QA image's vLLM 0.24+empty does not contain
this entry. The operator implementation itself does not import vLLM.

`pyproject.toml` records two opt-in dependency groups: `indexer-build` contains
source-build tools and `indexer-test` pins the official vLLM v0.33.0 release
commit and the matching torchvision 0.25.0 / torchaudio 2.10.0 imports.
The uv build configuration selects `VLLM_TARGET_DEVICE=empty` and reuses
the existing accelerator torch for the build. The install command excludes
vendor-provided torch, torch_npu, Triton, and FlagTree
from dependency resolution. This exclusion applies only to that command;
normal repository setup is unchanged. No CUDA vLLM wheel is used.

Use Python 3.11+ and uv (validated with 0.11.22). Activate this repository's
vendor test virtual environment (validated with torch 2.10.0+cpu and
torch_npu 2.10.0.post2), with FlagTree,
FlagGems, and test tools already installed, then run from the repository root:

```bash
source .venv/bin/activate
uv pip install --python "$VIRTUAL_ENV/bin/python" --group indexer-build
uv pip install --python "$VIRTUAL_ENV/bin/python" --group indexer-test \
  --excludes tools/accelerator-test-excludes.txt

PYTHONPATH=src python -m pytest -q tests/test_fused_indexer_weight_scale.py --quick
PYTHONPATH=src python -m pytest -q \
  benchmark/test_fused_indexer_weight_scale.py --level core --warmup 1 --iter 1
```

These commands install the newer empty baseline and its Python dependencies in
this repository's test environment. They do not add this dependency to the
normal package installation or change the shared image. QA/CI must run the
group installation explicitly before these tests; `pip install -e . --no-deps`
does not install the groups. vLLM 0.24 itself remains unsupported as this
operator's native reference. The source download requires access to GitHub's
official source archive endpoint.

If the archive URL is blocked, download the same pinned source through GitHub's
codeload endpoint and pass a local source override. The checksum below was
verified against the official release archive:

```bash
curl --fail --location \
  https://codeload.github.com/vllm-project/vllm/tar.gz/4ba787ec9959e566d3b587a796dff07673324da5 \
  --output /tmp/indexer-vllm-0.33.0.tar.gz
printf '%s  %s\n' \
  0293dfa0ffa7e1b657b60a3e383d2eaa1f14813ba55d3119faa542441a897670 \
  /tmp/indexer-vllm-0.33.0.tar.gz | sha256sum --check
printf '%s\n' 'vllm @ file:///tmp/indexer-vllm-0.33.0.tar.gz' \
  > /tmp/indexer-source-override.txt
uv pip install --python "$VIRTUAL_ENV/bin/python" --group indexer-test \
  --excludes tools/accelerator-test-excludes.txt \
  --overrides /tmp/indexer-source-override.txt
```

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
# 1. Install FlagGems (general operator backend, version 5.4.0)
pip install flag-gems==5.4.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/

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

### Run Smoke Tests

Verify test discovery and basic module import:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
PYTHONPATH=src pytest -q tests --quick
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

- [Basic usage patterns](../usage/basic.md)
- [Integration with vllm-plugin-fl](../usage/vllm-plugin.md)
- [Running tests](../testing/unittest.md)
- [Running benchmarks](../performance/benchmark.md)
