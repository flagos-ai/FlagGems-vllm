---
title: Installation
weight: 20
---

# Installation Guide

FlagGems-vllm is a pure-Python `setuptools` package that runs on top of a
vendor-provided PyTorch/Triton/vLLM stack. It does **not** declare or install
PyTorch, Triton, FlagTree or vLLM itself — those must match the target
accelerator and are installed by the backend image or by
`tools/setup.sh`.

## Prerequisites

### Python

Python ≥ 3.8 is supported by the package metadata. The backend registry in
`src/flaggems_vllm/backends.yaml` pins the version that each vendor toolchain is validated
against (currently 3.10, 3.11 or 3.12).

### Build tools

The package uses `setuptools`. Install the build backend and test/build
helpers before an editable install:

```bash
python -m pip install -U "setuptools>=64,<82" wheel packaging
```

`scikit-build-core`, `pybind11`, `cmake` and `ninja` are only needed when
building optional native extensions or when a vendor wheel is built from
source; `tools/setup.sh` installs them into its virtual environment.

### FlagGems dependency

**FlagGems-vllm depends on FlagGems.** Several FlagGems-vllm operators call
general FlagGems operators internally. Install a matching `flag-gems` release
(5.4.0 is the version used by the current docs and CI) from the FlagOS index:

```bash
python -m pip install --no-deps "flag-gems==5.4.0" \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
```

Use `--no-deps` when the vendor image already provides PyTorch/NumPy. In a
clean environment, install FlagGems' small pure-Python dependencies explicitly
(`packaging`, `PyYAML`, `sqlalchemy`) as shown in the manual installation
section below.

### PyTorch, Triton and vLLM

Install a PyTorch build, compiler and vLLM build compatible with your target
accelerator **before** installing FlagGems-vllm. For NVIDIA:

```bash
# Example only; use the CUDA/torch combination your driver supports.
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

For other backends, use the vendor's PyTorch/Triton instructions. The
validated combinations live in `src/flaggems_vllm/backends.yaml`; the `deps`, `triton`,
`flagtree` and `vllm` fields there are the single source of truth used by
`tools/setup.sh`.

## Install from Source

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm

# Development / editable install
python -m pip install --no-build-isolation -e .

# With the base test dependencies
python -m pip install --no-build-isolation -e '.[test]'
```

`[test]` installs `pytest`, `scipy`, `einops` and `distro`. The CUDA-only
`cupy-cuda12x` dependency is in the separate `[test-nvidia]` extra so that
non-NVIDIA backends do not pull a CUDA wheel into their vendor environment:

```bash
# NVIDIA only
python -m pip install --no-build-isolation -e '.[test,test-nvidia]'
```

## Multi-Backend Setup with `tools/setup.sh`

`tools/setup.sh` mirrors the FlagGems setup workflow. It reads
`src/flaggems_vllm/backends.yaml`, creates a `uv` virtual environment, installs the vendor
torch stack, vLLM, FlagGems and FlagGems-vllm, optionally replaces Triton with
FlagTree, and writes the vendor environment into `.venv/bin/activate`.

### List backends

```bash
bash tools/setup.sh --list
```

Common profiles and aliases:

| Alias | Profile | Vendor | vLLM source |
| --- | --- | --- | --- |
| `nvidia` | `nvidia-cuda133` (default) | NVIDIA | PyPI `vllm==0.20.2` |
| `nvidia-cuda128` | `nvidia-cuda128` | NVIDIA | PyPI `vllm==0.20.2` |
| `ascend` | `ascend-cann900` | Huawei Ascend | `vllm-ascend==0.24.0rc1` |
| `ascend-cann850` | `ascend-cann850` | Huawei Ascend | `vllm-ascend==0.24.0rc1` |
| `hygon` | `hygon-dtk2604` | Hygon DCU | vendor image (`--no-venv`) |
| `metax` | `metax-maca3810` | MetaX | vendor `vllm-metax` wheel |
| `iluvatar` | `iluvatar-corex450` | Iluvatar CoreX | vendor image (`--no-venv`) |
| `mthreads` | `mthreads-musa520` | Moore Threads | vendor image (`--no-venv`) |
| `cambricon` | `cambricon-neuware472` | Cambricon MLU | vendor image (`--no-venv`) |
| `enflame` | `enflame-tops1106` | Enflame GCU | vendor image (`--no-venv`) |
| `kunlunxin` | `kunlunxin-xre5371` | Kunlunxin XPU | vendor image (`--no-venv`) |
| `thead` | `thead-ppu210` | T-Head PPU | vendor `vllm` wheel |
| `tsingmicro` | `tsingmicro-tsm260610` | TsingMicro | vendor image (`--no-venv`) |
| `biren`, `spacemit`, `sunrise` | matching profile | vendor | vendor image (`--no-venv`) |
| `amd` | `amd-rocm` | AMD ROCm | install ROCm vLLM, then `--no-venv` |

### Create a virtual environment

```bash
# Execute: creates .venv and prints the activation command.
bash tools/setup.sh nvidia

# Or source it to create the venv and activate it in the current shell.
source tools/setup.sh nvidia
```

The script supports these modes and flags:

```text
--no-venv        install into the current Python environment
--skip-vllm      do not install vLLM (already provided by the image)
--skip-compiler  do not replace/install Triton or FlagTree
--skip-smoke     skip import and pytest collection checks
--dry-run        print commands without executing them
```

### Environment variables

All defaults come from `src/flaggems_vllm/backends.yaml` and can be overridden:

| Variable | Meaning |
| --- | --- |
| `PYTHON` | Interpreter used by `--no-venv` (default `python3`). |
| `PYTHON_VERSION` | Python version for the virtual environment. |
| `VENV_DIR` | Virtual environment directory (default `.venv`). |
| `PIP_INDEX_URL` | Primary package index (default: PyPI; use a regional mirror if needed). |
| `FLAG_GEMS_VERSION` | Override the registry's `flag-gems` pin (default 5.4.0). |
| `FLAG_GEMS_INDEX` | Index that publishes `flag-gems` (default: FlagOS hosted index). |
| `VLLM_VERSION` | Override a profile whose vLLM pin is `vllm==...` (NVIDIA). |
| `VLLM_PACKAGE` | Full vLLM requirement or wheel URL; wins over `VLLM_VERSION`. |
| `COMPILER` | `flagtree`, `triton` or `keep`. |
| `USE_FLAGTREE` | `1` = force FlagTree, `0` = force Triton. |
| `FLAGTREE_VERSION` / `TRITON_PACKAGE` | Override the compiler pin. |
| `SKIP_BACKEND_DEPS` | `1` = leave the vendor torch stack untouched. |
| `SKIP_FLAGGEMS` / `SKIP_PROJECT` | Skip FlagGems / FlagGems-vllm install. |
| `STRICT_TRITON_IMPORT` | `1` = fail when `import triton` fails. |
| `STRICT_PIP_CHECK` | `1` = fail when dependency resolution reports conflicts. |
| `UV_REFRESH` | `1` = force uv cache revalidation. |
| `CUDA_HOME` | CUDA toolkit path for NVIDIA profiles. |

Examples:

```bash
# NVIDIA with a different vLLM build
VLLM_PACKAGE="vllm==0.20.2" bash tools/setup.sh nvidia

# Ascend with the vendor compiler left untouched
COMPILER=keep bash tools/setup.sh ascend

# MetaX inside the vendor image: install only this repo + tests
bash tools/setup.sh --no-venv metax
```

In `--no-venv` mode, `SKIP_BACKEND_DEPS=1` and vLLM installation are the
defaults. Opt in explicitly when the image does not provide them:

```bash
SKIP_BACKEND_DEPS=0 VLLM_PACKAGE="vllm-ascend==0.24.0rc1" \
  bash tools/setup.sh --no-venv ascend
```

## Manual Installation Without a Venv

This is the supported path when the vendor image already contains a matched
Python, PyTorch, Triton/FlagTree and vLLM stack, and you only need to add
FlagGems-vllm and the test tooling. It is equivalent to
`bash tools/setup.sh --no-venv <backend>`, but lets you control every step.

### 1. Activate the vendor environment

Use the Python that the vendor image intends:

```bash
# Examples
source /usr/local/Ascend/cann/set_env.sh          # Ascend
source /opt/dtk-26.04/env.sh                      # Hygon
source /usr/local/PPU_SDK/envsetup.sh             # T-Head
# or: conda activate <vendor-env>
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Do **not** create a fresh venv unless you also plan to install a matching
torch/triton/vLLM stack into it.

### 2. Install the Python dependencies that are not provided by the image

```bash
# Required by FlagGems-vllm itself
python -m pip install "setuptools>=64,<82" wheel packaging pybind11 PyYAML sqlalchemy

# FlagGems runtime
python -m pip install --no-deps "flag-gems==5.4.0" \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/
```

### 3. Install FlagGems-vllm without replacing the vendor stack

```bash
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm

# --no-deps is important: it keeps pip from resolving the generic `flag-gems`
# and (on NVIDIA) CUDA test dependencies over the vendor packages.
python -m pip install --no-build-isolation --no-deps -e .
```

Keep `--no-deps` in vendor images: the project declares `flag-gems`, whose
metadata pins `numpy`, `packaging` and `sqlalchemy`, so letting pip resolve the
project can replace the vendor numpy. Install the runtime dependencies
explicitly as shown above. On a clean NVIDIA environment where that is not a
concern, dropping `--no-deps` is optional:

```bash
python -m pip install --no-build-isolation -e .
```

### 4. Install the test dependencies

The base test set is backend-neutral:

```bash
python -m pip install \
  "pytest>=7.1.0" \
  "scipy>=1.14,<1.18.0" \
  "einops>=0.6" \
  "distro>=1.9.0"
```

Add the CUDA-only test dependency on NVIDIA:

```bash
python -m pip install "cupy-cuda12x>=12.0.0,<14.0.0"
```

Alternatively, install the extras declared in `pyproject.toml`:

```bash
# Backend-neutral
python -m pip install --no-build-isolation -e '.[test]'

# NVIDIA
python -m pip install --no-build-isolation -e '.[test,test-nvidia]'
```

Note that pip will also resolve the project's `flag-gems` dependency in this
mode. `flag-gems` pins `numpy`, `packaging` and `sqlalchemy`, which can
conflict with a vendor-pinned numpy. In a vendor image, prefer the explicit
commands above (or the script's `--no-venv` mode), which install the project
with `--no-deps`:

vLLM-dependent tests (about 35 test files) need the vLLM build that matches
the vendor stack. Install it from the vendor or use the profile's package:

```bash
# NVIDIA (PyPI)
python -m pip install "vllm==0.20.2"

# Ascend
python -m pip install "vllm-ascend==0.24.0rc1"

# MetaX / T-Head: use the wheel URLs in src/flaggems_vllm/backends.yaml
# Other vendors: use the vLLM build shipped in the vendor image
```

If you do not install vLLM, those tests skip; the rest of the suite still
runs.

### 5. Point FlagGems-vllm at the right backend

`flaggems_vllm` auto-detects the accelerator, but explicit selection is more
predictable in multi-vendor images:

```bash
export FLAGGEMS_VENDOR=ascend   # nvidia, ascend, hygon, metax, ...
export DNN_VENDOR=ascend        # kept for compatibility with older docs
```

### 6. Verify

```bash
# Device, vendor and float32 smoke check
python tools/check_backend_env.py \
  --expected-vendor "${FLAGGEMS_VENDOR}" \
  --require-flaggems

# Test discovery
python -m pytest tests --collect-only -q

# Quick functional run
python -m pytest tests -q --quick
```

Running from the repository root does not require an editable install:
`pytest.ini` sets `pythonpath = src`, so `PYTHONPATH=src` is enough for the
imports. The install is still recommended for the `flaggems_vllm` metadata
used by `importlib.metadata.version`.

## Install with FlagGems and vllm-plugin-fl

For the full FlagOS vLLM plugin stack, install in this order:

```bash
# 1. FlagGems (general operator backend)
python -m pip install --no-deps "flag-gems==5.4.0" \
  --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple/

# 2. FlagGems-vllm (vLLM-specific operators)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
python -m pip install --no-build-isolation -e .
cd ..

# 3. vllm-plugin-fl (vLLM plugin integration)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
python -m pip install --no-build-isolation -e .
```

If multiple vLLM plugins are installed, select the FlagOS plugin:

```bash
export VLLM_PLUGINS=fl
```

## Troubleshooting

### Missing CUDA

If CUDA is not found, set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
python -m pip install --no-build-isolation -e .
```

### Triton Version Mismatch

FlagGems-vllm requires Triton ≥ 3.0. Prefer the vendor Triton/FlagTree build
pinned in `src/flaggems_vllm/backends.yaml`; a generic `pip install -U triton` can replace
the vendor compiler and break the accelerator backend.

```bash
# Inspect the validated compiler for your backend
grep -A6 '^  ascend-cann900:' src/flaggems_vllm/backends.yaml

# Or let the setup script install it
bash tools/setup.sh ascend
```

### Missing Test Dependencies

```bash
python -m pip install --no-build-isolation -e '.[test]'
# NVIDIA only
python -m pip install --no-build-isolation -e '.[test-nvidia]'
```

### vLLM import errors

The vLLM wheel must match the vendor torch build. Inside a vendor image, do
not install a PyPI vLLM over the image's build; use the repo/test install
path instead:

```bash
bash tools/setup.sh --no-venv <backend>
```

## Next Steps

- [Basic usage patterns](../usage/basic.md)
- [Non-NVIDIA backends](../usage/non-nvidia.md)
- [Integration with vllm-plugin-fl](../usage/vllm-plugin.md)
- [Running tests](../testing/unittest.md)
- [Running benchmarks](../performance/benchmark.md)
