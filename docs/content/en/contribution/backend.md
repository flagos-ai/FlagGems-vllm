---
title: Backend Development
weight: 30
---

# Backend Development Guide

Add support for new hardware backends in FlagGems-vllm.

## Overview

FlagGems-vllm supports 17 hardware backends. Each backend has:
- Device detection logic
- Optional autotune configuration
- Architecture-specific tuning (e.g., NVIDIA Ampere vs Hopper)

## Backend Directory Structure

Backends live under `src/flaggems_vllm/runtime/backend/`:

```
backend/
├── _nvidia/
│   ├── __init__.py
│   ├── tune_configs.yaml     # Pre-tuned autotune configs
│   ├── ampere/               # Ampere-specific configs
│   │   └── tune_configs.yaml
│   └── hopper/               # Hopper-specific configs
│       └── tune_configs.yaml
├── _amd/
│   ├── __init__.py
│   └── tune_configs.yaml
├── _ascend/
│   └── __init__.py
└── _<vendor>/                # Your new backend
    ├── __init__.py
    └── tune_configs.yaml     # Optional
```

## Adding a New Backend

### Step 0: Add a Setup Profile

Add an entry to `src/flaggems_vllm/backends.yaml` so `tools/setup.sh` can build the vendor
environment. The registry records the Python version, vendor torch/accelerator
packages, Triton/FlagTree builds, vLLM source, environment variables, vendor
env scripts and backend-specific test dependencies:

```yaml
backends:
  myvendor-myhw:
    vendor: myvendor
    python: "3.12"
    deps:
      - torch==2.10.0+myvendor
      - myvendor-torch==1.2.3
    triton: triton==3.6.0+myvendor
    flagtree: flagtree==0.7.0+myvendor
    # Omit vllm if the vendor image provides it; use --no-venv there.
    vllm:
      - vllm-myvendor==0.24.0
    env:
      MYVENDOR_HOME: /opt/myvendor
      LD_LIBRARY_PATH: $MYVENDOR_HOME/lib:$LD_LIBRARY_PATH
    env_source:
      - /opt/myvendor/set_env.sh
    # Optional backend-specific test packages (NVIDIA uses this for cupy).
    test_deps:
      - "some-vendor-test-helper>=1.0"
```

Add a short alias under `aliases:` if the vendor name alone should select this
profile. Then verify with:

```bash
bash tools/setup.sh --list
bash tools/setup.sh --dry-run myvendor
```

### Step 1: Create Backend Directory

```bash
mkdir src/flaggems_vllm/runtime/backend/_myvendor
touch src/flaggems_vllm/runtime/backend/_myvendor/__init__.py
```

### Step 2: Register Vendor and Add Device Detection

1. Register vendor information in `src/flaggems_vllm/runtime/common.py`:
   - Add `VendorInfo` (vendor name, device name, dispatch key, features like bf16/fp64 support)
   - Add entry to `_VENDOR_TORCH_ATTR` or shell check scripts in `tools/`

2. If needed, customize detection in `src/flaggems_vllm/runtime/backend/device_finder.py`:

```python
# DeviceDetector detects vendor from environment variable DNN_VENDOR,
# vendor check scripts, or PyTorch device modules.
```

### Step 3: Implement Backend Module

`src/flaggems_vllm/runtime/backend/_myvendor/__init__.py`:

```python
"""MyVendor backend for FlagGems-vllm."""

# Device name mapping
DEVICE_NAME = "myvendor"  # or "cuda" if compatible

# Optional: Backend-specific initialization
def initialize():
    """Called when backend is loaded."""
    pass

# Optional: Backend capabilities
SUPPORTS_FP8 = False
SUPPORTS_BF16 = True
MAX_THREADS_PER_BLOCK = 1024
```

### Step 4: Add Autotune Configs (Optional but Recommended)

Create `src/flaggems_vllm/runtime/backend/_myvendor/tune_configs.yaml`:

```yaml
# Operator-specific configs
grouped_topk:
  gen: true
  param_map:
    num_warps: [4, 8]
    num_stages: [2, 3]
    BLOCK_M: [64, 128]
    BLOCK_N: [64, 128]

moe_align_block_size:
  gen: true
  param_map:
    num_warps: [4, 8, 16]
    BLOCK_SIZE: [32, 64, 128, 256]
```

Format:
- `gen: true`: Enable config generation
- `param_map`: Tunable parameters and their candidate values

### Step 5: Test on Target Hardware

```bash
export DNN_VENDOR=myvendor
PYTHONPATH=src pytest -q tests --quick
```

Start with a few core operators:
```bash
PYTHONPATH=src pytest -v tests/test_grouped_topk.py
PYTHONPATH=src pytest -v tests/test_moe_align_block_size.py
```

### Step 6: Benchmark

Run benchmarks to establish baseline performance:

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level core --iter 100 --warmup 20 \
    --record --output myvendor_results.json
```

### Step 7: Document Limitations

Create `src/flaggems_vllm/runtime/backend/_myvendor/README.md`:

```markdown
# MyVendor Backend

## Setup

1. Install MyVendor PyTorch extension: `pip install myvendor-torch`
2. Set environment: `export DNN_VENDOR=myvendor`

## Supported Operators

- ✓ grouped_topk
- ✓ moe_align_block_size
- ✗ flash_attention_forward (not yet supported)

## Known Issues

- FP8 quantization not supported
- Large batch sizes may be slow

## Performance

See benchmark results in `benchmark_results.json`.
```

## Compiler Support

### OpenAI Triton

Works on NVIDIA and AMD (ROCm). If your backend has Triton support:

```python
# No special handling needed
# Triton will compile to your backend if supported
```

### FlagTree

Required for most non-NVIDIA/AMD backends:

```bash
pip install flagtree
export USE_FLAGTREE=1
```

FlagTree generates vendor-specific code via its compiler backend.

## Architecture-Specific Tuning

For backends with multiple architectures (like NVIDIA Ampere/Hopper):

```
_myvendor/
├── __init__.py
├── tune_configs.yaml          # Default configs
├── gen1/                      # Generation 1
│   └── tune_configs.yaml
└── gen2/                      # Generation 2
    └── tune_configs.yaml
```

Detect architecture in `__init__.py`:

```python
import myvendor_torch

def get_architecture():
    device = myvendor_torch.get_device_properties(0)
    if device.generation == "Gen1":
        return "gen1"
    elif device.generation == "Gen2":
        return "gen2"
    return None

# Load arch-specific config if available
```

## CI Integration

Backend CI is driven by data files, not by a per-vendor job in the workflow.
Do not add a job to `.github/workflows/unittest.yaml`. To enroll a backend:

1. Add a profile to `src/flaggems_vllm/backends.yaml` so `tools/setup.sh` can
   build its environment.
2. Add an entry (`backend`, `runner_label`, `label`, `gpu_check`, `enabled`) to
   `.github/backends.json` and a health script `tools/gpu_check_<vendor>.sh`.
3. List the tests proven on the backend in `.github/backend-capabilities.json`.
   Until `auto_select` is `true` there, only an explicit `vendor/*` label
   schedules the backend; a path change alone does not.

How a pull request reaches a backend:

- `unittest.yaml` starts with a `preprocess` job. It waits for the triage
  workflow, reads the PR labels, and selects tests (`tools/select_tests.py`)
  and backends (`tools/select_backends.py`, `tools/select_ci_backends.py`).
- A backend runs when the PR changes `src/flaggems_vllm/runtime/backend/_<vendor>/`
  (only if `auto_select` is `true`), or when a maintainer adds its `vendor/*`
  label. `ci/all-vendors` selects every enabled backend.
- The H20 job (`python-op`) runs for pull requests from this repository. A
  maintainer opts in a fork pull request with the `ci/nvidia` label.
- `unittest-required` is the only check to require in branch protection. It
  fails if a job that should have run was skipped, or if any job failed.

See `.github/MULTI_BACKEND_CI.md` for runner setup and the trust model.

## Example: Minimal Backend

`src/flaggems_vllm/runtime/backend/_example/__init__.py`:

```python
"""Example minimal backend."""

DEVICE_NAME = "cuda"  # Reuse CUDA device name if compatible

def initialize():
    print("Example backend initialized")
```

`src/flaggems_vllm/runtime/device_finder.py`:

```python
def detect_vendor():
    if os.environ.get("USE_EXAMPLE_BACKEND") == "1":
        return "example"
    # ... rest of detection
```

Usage:

```bash
export USE_EXAMPLE_BACKEND=1
export DNN_VENDOR=example
python script.py
```

## Performance Tuning

1. **Start with NVIDIA configs**: Copy and adapt from `_nvidia/tune_configs.yaml`
2. **Profile**: Use vendor profilers to identify bottlenecks
3. **Iterate**: Adjust `num_warps`, `num_stages`, block sizes
4. **Measure**: Run benchmarks after each change
5. **Document**: Record best configs in `tune_configs.yaml`

## Operator Compatibility

Not all operators may work on all backends:
- Some rely on hardware-specific features (e.g., tensor cores)
- Some use CUDA-specific intrinsics

Mark unsupported operators:

```python
# In operator implementation
if runtime.vendor_name == "myvendor":
    raise NotImplementedError("my_operator not supported on MyVendor")
```

Or contribute a backend-specific implementation!

## Submitting Your Backend

PR checklist:
- ✓ Backend directory created
- ✓ Device detection added
- ✓ Tested on target hardware (include test results)
- ✓ Benchmark results (at least core operators)
- ✓ README documenting setup and limitations
- ✓ CI integration (if runners available)

## Next Steps

- [Contribution overview](overview.md)
- [Operator development protocol](workflow.md)
- [Testing guide](../testing/unittest.md)
