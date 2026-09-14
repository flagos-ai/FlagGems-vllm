---
title: Non-NVIDIA Backends
weight: 40
---

# Running on Non-NVIDIA Backends

FlagGems-vllm supports **18 hardware backends** beyond NVIDIA GPUs.

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
- And more...

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

Requires Ascend CANN toolkit and Torch-NPU:

```bash
# Install CANN (follow Huawei docs)
# Install Torch-NPU
pip install torch-npu

export DNN_VENDOR=ascend
```

FlagTree compiler recommended for Ascend (OpenAI Triton has limited support):

```bash
pip install flagtree
export USE_FLAGTREE=1
```

### Other Vendors

Each vendor typically requires:
1. Vendor-specific PyTorch build or plugin
2. Triton or FlagTree compiler
3. `DNN_VENDOR` environment variable set

Consult your hardware vendor's documentation for PyTorch installation.

## Compiler: Triton vs FlagTree

- **OpenAI Triton**: Works on NVIDIA, AMD (ROCm)
- **FlagTree**: Supports all 18+ backends via vendor-specific codegen

To use FlagTree:

```bash
pip install flagtree
export USE_FLAGTREE=1
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

## Testing on Multiple Backends

The `tools/setup.sh` script supports backend selection:

```bash
DNN_VENDOR=amd source tools/setup.sh
```

CI includes a multi-backend test matrix (when runners are available).

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
2. Add device detection in `runtime/device_finder.py`
3. Test core operators on target hardware
4. Submit PR with test results

See the [backend development guide](../contribution/backend/) for details.

## Next Steps

- [Backend development guide](/FlagGems-vllm/contribution/backend/)
- [Testing guide](/FlagGems-vllm/testing/unittest/)
- [Performance benchmarking](/FlagGems-vllm/performance/benchmark/)
