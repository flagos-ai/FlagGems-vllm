---
title: Debugging
weight: 50
---

# Debugging and Logging

Tools and techniques for debugging FlagGems-vllm operators.

## Logging

FlagGems-vllm uses Python's standard logging module.

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import flaggems_vllm
# Now see detailed logs
```

### Selective Logging

Log only FlagGems-vllm messages:

```python
import logging
logger = logging.getLogger('flaggems_vllm')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
```

## Common Issues

### 1. NotImplementedError

**Symptom**: Operator raises `NotImplementedError` for your input.

**Cause**: Unsupported dtype, shape, or configuration.

**Solution**:
- Check operator documentation for supported configurations
- This is by design (NO_TORCH_COMPUTE_FALLBACK rule)
- Implement support for your configuration or use a different operator

Example:
```python
try:
    result = flaggems_vllm.my_operator(x)
except NotImplementedError as e:
    print(f"Unsupported: {e}")
    # Handle explicitly (e.g., use torch fallback)
```

### 2. Numerical Differences

**Symptom**: Results differ from reference implementation.

**Cause**: Floating-point precision, different algorithms, or bugs.

**Solution**:
1. Check tolerance:
   ```python
   import torch
   diff = (result - expected).abs().max()
   rel_diff = ((result - expected).abs() / expected.abs()).max()
   print(f"Max abs diff: {diff}, max rel diff: {rel_diff}")
   ```

2. Verify input data:
   ```python
   print(f"Input shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
   print(f"Input range: [{x.min()}, {x.max()}]")
   ```

3. Test with simpler inputs:
   ```python
   # Identity case
   x = torch.eye(4, device='cuda')
   
   # Small random
   x = torch.randn(4, 4, device='cuda') * 0.1
   ```

### 3. Performance Slower Than Expected

**Symptom**: FlagGems-vllm slower than torch baseline.

**Cause**: Missing autotune, poor shape, or cold start.

**Solution**:
1. Warm up before benchmarking:
   ```python
   for _ in range(10):  # Warmup
       flaggems_vllm.my_operator(x)
   
   import time
   torch.cuda.synchronize()
   start = time.time()
   result = flaggems_vllm.my_operator(x)
   torch.cuda.synchronize()
   print(f"Time: {time.time() - start}")
   ```

2. Check autotune config exists:
   ```bash
   grep "my_operator" src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml
   ```

3. Profile (see below)

### 4. Import Errors

**Symptom**: `ImportError: cannot import name 'my_operator'`

**Cause**: Operator not exported or not installed.

**Solution**:
1. Check operator is exported:
   ```bash
   grep "my_operator" src/flaggems_vllm/ops/__init__.py
   grep "my_operator" src/flaggems_vllm/__init__.py
   ```

2. Reinstall in editable mode:
   ```bash
   pip install --no-build-isolation -e .
   ```

## Profiling

### Triton Profiler

Triton has built-in profiling:

```python
import triton

with triton.profiler.Profile() as prof:
    result = flaggems_vllm.my_operator(x)

prof.print()
```

### NVIDIA Nsight Compute

Profile CUDA kernels:

```bash
ncu --set full -o profile python script.py
```

Open `profile.ncu-rep` in Nsight Compute GUI.

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    result = flaggems_vllm.my_operator(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("trace.json")
```

View `trace.json` in Chrome at `chrome://tracing`.

## Debugging Triton Kernels

### Print Intermediate Values

Triton supports `tl.device_print()`:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, ...):
    pid = tl.program_id(0)
    tl.device_print("Block", pid, "processing")
    # ...
```

**Note**: Device prints can be slow. Remove in production.

### Check Kernel Compilation

Set `TRITON_DEBUG=1` to see compilation logs:

```bash
TRITON_DEBUG=1 python script.py
```

### Disable Autotune

Test with a single config:

```python
from flaggems_vllm.utils import libtuner

@libtuner(configs=[{"BLOCK_SIZE": 128}], key=[], strategy=[])
@triton.jit
def my_kernel(...):
    ...
```

## Testing Utilities

### Compare Against Reference

Use `tests/accuracy_utils.py`:

```python
from tests.accuracy_utils import to_reference, gems_assert_close

ref_input = to_reference(cuda_input)
ref_output = reference_impl(ref_input)
result = flaggems_vllm.my_operator(cuda_input)

gems_assert_close(result, ref_output, rtol=1e-3, atol=1e-5)
```

### Isolate Operator

Test operator in isolation:

```python
import torch
import flaggems_vllm

x = torch.randn(128, 256, device='cuda', dtype=torch.float16)

# Direct call (no other operators involved)
result = flaggems_vllm.my_operator(x)
print(result.shape, result.dtype)
```

## Getting Help

If you're stuck:

1. **Check documentation**: Operator may have known limitations
2. **Search issues**: Someone may have hit the same problem
3. **Create minimal reproducer**: Simplify to smallest failing case
4. **Open an issue**: Include:
   - FlagGems-vllm version
   - PyTorch version
   - Hardware (GPU model, driver version)
   - Minimal code to reproduce
   - Error message and stack trace

## Next Steps

- [Testing guide](/FlagGems-vllm/testing/unittest/)
- [Performance benchmarking](/FlagGems-vllm/performance/benchmark/)
- [Contribution guide](/FlagGems-vllm/contribution/overview/)
