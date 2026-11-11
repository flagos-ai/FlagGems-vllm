---
title: Non-NVIDIA Backends
weight: 40
---

# Running on Non-NVIDIA Backends

FlagGems-vllm supports **17 hardware backends** (NVIDIA plus 16 non-NVIDIA backends).

## Supported Backends

- **AMD**: ROCm-based GPUs
- **Ascend** (Huawei): Ascend 910B and newer
- **Cambricon**: MLU series
- **Enflame**: GCU series
- **Hygon**: DCU series
- **Iluvatar**: CoreX series
- **Kunlunxin** (Baidu): XPU series
- **MetaX**: C-series accelerators
- **Moore Threads**: MTT GPUs
- **AIPU**, **ARM**, **Biren**, **SpacemiT**, **Sunrise**, **T-Head**, **Tsingmicro**

See `src/flaggems_vllm/runtime/backend/` for the complete list.

## Backend Selection

### Automatic Detection

FlagGems-vllm auto-detects the backend based on available hardware:

```python
import flaggems_vllm

print(f"Detected vendor: {flaggems_vllm.vendor_name}")
print(f"Device: {flaggems_vllm.device}")
```

### Manual Override

Set the `DNN_VENDOR` environment variable:

```bash
export DNN_VENDOR=amd
python your_script.py
```

Or in Python:

```python
import os
os.environ['DNN_VENDOR'] = 'ascend'

import flaggems_vllm
# Now uses Ascend backend
```

## Backend-Specific Setup

### AMD (ROCm)

Install ROCm-compatible PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
export DNN_VENDOR=amd
```

Triton should work out-of-the-box with ROCm.

### Huawei Ascend

Requires the Ascend CANN toolkit and Torch-NPU. The validated combination is
the `ascend-cann900` (or `ascend-cann850`) profile in `src/flaggems_vllm/backends.yaml`:

```bash
bash tools/setup.sh ascend
```

FlagTree is recommended for Ascend (OpenAI Triton has limited support); the
profile pins a matching `flagtree` build and `tools/setup.sh` installs it by
default.

#### Tests using vLLM-Ascend custom ops

Sparse flash attention tests and benchmarks use vLLM-Ascend as their reference.
Enable its pytest plugin explicitly when running these files from the repository
root:

```bash
python -m pytest -p tools.ascend.pytest_custom_ops -q tests/test_sparse_flash_attention.py
python -m pytest -p tools.ascend.pytest_custom_ops -q benchmark/test_sparse_flash_attention.py
```

The plugin calls vLLM-Ascend's `enable_custom_op()` before pytest loads the common
`conftest.py` files. CANN must discover the vendor implementations before device
initialization; loading them later in a test or fixture can select the built-in
implementation instead. The installed vLLM-Ascend package configures its own
custom-op paths; no installation path needs to be set manually.

The plugin lives under `tools/ascend/` so vendor initialization stays outside the
shared tests directory. Importing its package does not import FlagGems or
initialize the device runtime.

`tools/run_tests.py` enables this plugin automatically for Ascend
`sparse_flash_attention` workers. Other operator tests do not load it implicitly.
If running a larger Ascend test selection that includes sparse flash attention,
pass the same `-p tools.ascend.pytest_custom_ops` option for that session.

### Other Vendors

Each vendor typically requires:
1. Vendor-specific PyTorch build or plugin
2. Triton or FlagTree compiler
3. `DNN_VENDOR` environment variable set

Consult your hardware vendor's documentation for PyTorch installation.

## Compiler: Triton vs FlagTree

- **OpenAI Triton**: Works on NVIDIA, AMD (ROCm)
- **FlagTree**: Supports all 17 backends via vendor-specific codegen

To use FlagTree, prefer the backend profile's pinned build; a bare
`pip install flagtree` may not match the vendor torch stack:

```bash
# Install the compiler pinned for the backend
bash tools/setup.sh ascend

# Or force a compiler when the environment is already prepared
COMPILER=flagtree bash tools/setup.sh --no-venv ascend
COMPILER=triton   bash tools/setup.sh --no-venv ascend
```

## Backend Maturity

| Backend | Maturity | Autotune | Notes |
|---------|----------|----------|-------|
| NVIDIA  | ★★★★★   | Yes      | Pre-tuned configs for Ampere/Hopper |
| AMD     | ★★★☆☆   | Partial  | Baseline tuning, ongoing work |
| Ascend  | ★★☆☆☆   | No       | Experimental, FlagTree required |
| Others  | ★☆☆☆☆   | No       | Basic functionality, limited testing |

NVIDIA has the most mature support with pre-tuned autotune configs.

## Performance Expectations

Non-NVIDIA backends may have:
- **Lower absolute performance** (hardware differences)
- **Higher performance variance** (less mature tuning)
- **Missing operator support** (some ops may be NVIDIA-only)

Always benchmark on your target hardware.

## Setting Up a Backend

`tools/setup.sh` reads the backend registry in `src/flaggems_vllm/backends.yaml` and installs
the matching torch stack, vLLM package, compiler and test dependencies:

```bash
# List profiles and aliases
bash tools/setup.sh --list

# Create .venv and install a profile (prints the activation command)
bash tools/setup.sh ascend

# ...or create the venv and activate it in the current shell
source tools/setup.sh metax
```

Common aliases: `nvidia`, `ascend`, `hygon`, `metax`, `iluvatar`, `mthreads`,
`cambricon`, `enflame`, `kunlunxin`, `thead`, `tsingmicro`, `biren`,
`spacemit`, `sunrise`, `amd`. The exact profile name is accepted too
(e.g. `ascend-cann850`, `metax-maca3720`).

Vendor images already ship a matched torch/triton/vLLM stack. Inside such an
image, use `--no-venv` so the script only adds FlagGems-vllm and the test
dependencies without replacing the vendor packages:

```bash
bash tools/setup.sh --no-venv hygon
```

The same installation can be done step by step with pip; see
[Manual Installation Without a Venv](../getting-started/install.md#manual-installation-without-a-venv).

The registry is also the place to add a new backend profile. A profile records
the Python version, vendor torch dependencies, Triton/FlagTree packages, the
vLLM source, environment variables, vendor env scripts and backend-specific
test dependencies.

## Troubleshooting

### Backend Not Detected

Check device availability:

```python
import torch
print(torch.cuda.is_available())  # NVIDIA/AMD
print(torch.cuda.device_count())

# For Ascend
import torch_npu
print(torch_npu.npu.is_available())
```

Set `DNN_VENDOR` explicitly if auto-detection fails.

### Operator Failures

Some operators may not work on all backends:
- Check operator metadata in `conf/operators.yaml` for backend restrictions
- Operators raise `NotImplementedError` for unsupported configurations
- Report issues to help improve backend coverage

### Performance Issues

1. Verify PyTorch is using the correct device:
   ```python
   x = torch.randn(128, 256, device='cuda')
   print(x.device)  # Should show the vendor device
   ```

2. Check if autotune configs exist for your backend:
   ```bash
   ls src/flaggems_vllm/runtime/backend/_<vendor>/tune_configs.yaml
   ```

3. Profile to identify bottlenecks (use vendor-specific profilers)

## Contributing Backend Support

To add a new backend:

1. Create `src/flaggems_vllm/runtime/backend/_<vendor>/`
2. Add device detection in `src/flaggems_vllm/runtime/backend/device_finder.py`
3. Test core operators on target hardware
4. Submit PR with test results

See the [backend development guide](../contribution/backend.md) for details.

## Next Steps

- [Backend development guide](../contribution/backend.md)
- [Testing guide](../testing/unittest.md)
- [Performance benchmarking](../performance/benchmark.md)
