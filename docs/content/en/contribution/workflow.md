---
title: Operator Development Protocol
weight: 20
---

# Operator Development Protocol

FlagGems-vllm follows a strict operator development protocol defined in `workflow.md` at the repository root.

## Protocol Overview

The protocol defines **hard gates (G0–G7)** that must be passed when adding or modifying operators:

- **G0**: Task clarity and protocol acknowledgment
- **G1**: NO_TORCH_COMPUTE_FALLBACK verification
- **G2**: Pre-coding tables completed
- **G3**: Host dispatch design
- **G4**: Autotune integration
- **G5**: Implementation complete
- **G6**: Functional tests pass
- **G7**: Performance acceptance (SpeedUp ≥ 0.9)

## Critical Rules

### 1. NO_TORCH_COMPUTE_FALLBACK

**Production code paths must NOT use torch compute operations.**

Forbidden in production code:
- `torch.matmul`, `torch.sum`, `torch.softmax` (compute ops)
- `tensor.contiguous()`, `tensor.to()`, `tensor.copy_()` (memory kernels)
- Any operation that launches a PyTorch/ATen kernel

Allowed:
- Metadata reads: `tensor.shape`, `tensor.dtype`, `tensor.device`
- Uninitialized allocation: `torch.empty_like()`
- True no-copy views: `tensor.view()`
- Triton kernel launches

Unsupported configurations must raise `NotImplementedError`, not fall back to torch.

### 2. Pre-Coding Tables

Before writing code, create three tables:

#### Project Truth Table
- Operator name, torch signature, vLLM usage context
- Input/output specs (shapes, dtypes, devices)
- Performance targets

#### Torch Contract Table
- What torch reference behavior to match
- Tolerance specifications (rtol, atol)
- Edge cases (empty, NaN, inf)

#### Implementation Path Table
- Workload classification (shape/dtype/device)
- Routing strategy
- Unsupported paths (raise NotImplementedError)

### 3. Autotune Required

NVIDIA backend operators must have autotune integration:
- Define tunable parameters (BLOCK_M, BLOCK_N, num_warps, num_stages)
- Add configs to `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml`
- Use `@libtuner()` decorator with `configs`, `key`, and `strategy`

### 4. Standard Code Drops

Each operator requires files at specific locations:

| Purpose | Path |
|---------|------|
| Implementation | `src/flaggems_vllm/ops/<op>.py` |
| Test | `tests/test_<op>.py` |
| Benchmark | `benchmark/test_<op>.py` |
| Metadata | Add entry to `conf/operators.yaml` |
| Export | `src/flaggems_vllm/ops/__init__.py` + `src/flaggems_vllm/__init__.py` |
| Autotune config | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` |

### 5. Testing Requirements

Functional tests must:
- Cover representative shapes and dtypes
- Compare against reference implementation
- Use `gems_assert_close()` with appropriate tolerances
- Pass with both `--quick` and full modes
- Include edge cases

Benchmarks must:
- Cover core shapes from `benchmark/core_shapes.yaml`
- Measure latency and throughput
- Report SpeedUp vs baseline

### 6. Performance Acceptance

**SpeedUp ≥ 0.9** on core shapes (NVIDIA H100 baseline).

SpeedUp = `latency_torch_baseline / latency_flaggems_vllm`

If SpeedUp < 0.9:
1. Profile to identify bottleneck
2. Iterate on kernel optimization
3. Update autotune configs
4. Re-measure

## Development Flow

### Step 1: Read Task and Protocol

- Understand the operator requirement (vLLM usage context)
- Read `workflow.md` in full
- Review similar operators in the codebase

### Step 2: Freeze Torch Contract

Define what torch behavior to match:
- Input/output signatures
- Numerical tolerances
- Edge case handling

### Step 3: Workload Classification

Classify inputs by:
- Shape patterns (batch size, sequence length, hidden dim)
- Data types (float16, bfloat16, float32, int8, fp8)
- Device (CUDA, ROCm, Ascend)

Design routing strategy.

### Step 4: Design Host Dispatch

Before writing kernels, design the host function:
- Argument validation
- Shape/dtype routing
- Empty tensor early returns
- Unsupported path exceptions

### Step 5: Autotune Integration

Define tunable parameters and configs:

```yaml
# tune_configs.yaml
my_operator:
  gen: true
  param_map:
    META:
      - TN
      - NT
    num_warps: [4, 8]
    num_stages: [2, 3]
    BLOCK_M: [64, 128]
    BLOCK_N: [64, 128]
```

Use in code:

```python
from flaggems_vllm.utils import libtuner

@libtuner(
    configs=runtime.get_tuned_config("my_operator"),
    key=["M", "N"],
    strategy=["layout"],
)
@triton.jit
def my_kernel(...):
    ...
```

### Step 6: Implement Code

Write:
- Host dispatch function with `@libentry()`
- Triton kernels with `@triton.jit`
- Helper functions (if needed)

Follow NO_TORCH_COMPUTE_FALLBACK rule.

### Step 7: Integration

- Export from `ops/__init__.py` (add to `__all__`)
- Register in top-level `__init__.py` (if using dispatch)
- Add metadata to `conf/operators.yaml`

## Functional Testing

Create `tests/test_<op>.py`:

```python
import pytest
import torch
import flaggems_vllm
from tests.accuracy_utils import gems_assert_close

@pytest.mark.my_operator
@pytest.mark.parametrize("shape", [(128, 256), (512, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_my_operator(shape, dtype):
    input_tensor = torch.randn(shape, device='cuda', dtype=dtype)
    result = flaggems_vllm.my_operator(input_tensor)
    expected = reference_impl(input_tensor)
    gems_assert_close(result, expected, rtol=1e-3, atol=1e-5)
```

Run:
```bash
PYTHONPATH=src pytest -v tests/test_my_operator.py
PYTHONPATH=src pytest -q tests/test_my_operator.py --quick
```

## Performance Testing

Create `benchmark/test_<op>.py`:

```python
import pytest
from benchmark.conftest import Config

@pytest.mark.parametrize("shape", Config.get_shapes("my_operator"))
def test_my_operator_perf(benchmark, shape):
    input_tensor = torch.randn(shape, device='cuda', dtype=torch.float16)
    result = benchmark(lambda: flaggems_vllm.my_operator(input_tensor))
    assert result is not None
```

Add shapes to `benchmark/core_shapes.yaml`:

```yaml
my_operator:
  - [128, 256]
  - [512, 1024]
  - [2048, 4096]
```

Run:
```bash
PYTHONPATH=src pytest -q benchmark/test_my_operator.py --level core --iter 100 --warmup 20
```

## Profiling

Profile before optimization:

```bash
# Nsight Compute
ncu --set full -o profile python -c "import torch; import flaggems_vllm; ..."

# PyTorch profiler
python -c "
import torch
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    flaggems_vllm.my_operator(...)
prof.export_chrome_trace('trace.json')
"
```

## Deep Optimization

Trigger deep optimization if:
- SpeedUp < 0.9 after initial implementation
- Performance regression detected
- New architecture requires tuning

Consult `deep_opt.md` (if present) or follow trial loop:
1. Profile → identify bottleneck
2. Hypothesize optimization
3. Implement and measure
4. Keep if improved, revert if not

## Delivery Format

PR must include:
- ✓ Implementation (`src/flaggems_vllm/ops/<op>.py`)
- ✓ Tests (`tests/test_<op>.py`)
- ✓ Benchmarks (`benchmark/test_<op>.py`)
- ✓ Metadata (`conf/operators.yaml`)
- ✓ Autotune config (`runtime/backend/_nvidia/tune_configs.yaml`)
- ✓ Export (`ops/__init__.py`, top-level `__init__.py`)
- ✓ Test results (pass with `--quick` and full)
- ✓ Performance results (SpeedUp ≥ 0.9 on core shapes)

## Full Protocol

For complete details, read **`workflow.md`** at the repository root.

The protocol is written in Chinese but is the authoritative source for all operator development.

## Next Steps

- [Read workflow.md](https://github.com/flagos-ai/FlagGems-vllm/blob/main/workflow.md)
- [Contribution overview](overview/)
- [Testing guide](/FlagGems-vllm/testing/unittest/)
- [Benchmark guide](/FlagGems-vllm/performance/benchmark/)
