---
title: Backend Development
weight: 30
---

# Backend Development Guide

Add support for new hardware backends in FlagGems-vllm.

## Overview

FlagGems-vllm supports 18+ hardware backends. Each backend has:
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

### Step 1: Create Backend Directory

```bash
mkdir src/flaggems_vllm/runtime/backend/_myvendor
touch src/flaggems_vllm/runtime/backend/_myvendor/__init__.py
```

### Step 2: Add Device Detection

Edit `src/flaggems_vllm/runtime/device_finder.py`:

```python
def detect_vendor():
    # Existing detection logic...

    # Add your backend
    try:
        import myvendor_torch  # Vendor's PyTorch extension
        if myvendor_torch.is_available():
            return "myvendor"
    except ImportError:
        pass

    return "nvidia"  # default
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

Add CI job for your backend (if runners available):

`.github/workflows/basic-ci.yml`:

```yaml
myvendor-tests:
  runs-on: myvendor-runner
  steps:
    - uses: actions/checkout@v4
    - name: Setup
      run: |
        pip install myvendor-torch
        pip install --no-build-isolation -e '.[test]'
    - name: Tests
      run: |
        export DNN_VENDOR=myvendor
        pytest -q tests --quick
```

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

- [Contribution overview](/FlagGems-vllm/contribution/overview/)
- [Operator development protocol](/FlagGems-vllm/contribution/workflow/)
- [Testing guide](/FlagGems-vllm/testing/unittest/)
