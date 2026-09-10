---
title: Overview
weight: 10
---

# Contribution Overview

Guidelines for contributing to FlagGems-vllm.

## Before You Start

1. **Read `workflow.md`**: The operator development protocol is mandatory for all operator work
2. **Check existing issues**: See if someone is already working on your idea
3. **Discuss large changes**: Open an issue first for significant features or refactors

## Contribution Types

### Adding a New Operator

**Required reading**: `workflow.md` (operator development protocol)

Steps:
1. Verify the operator is needed for vLLM scenarios (not in FlagGems)
2. Create three pre-coding tables (project truth, torch contract, implementation paths)
3. Implement host dispatch + Triton kernels
4. Add autotune integration (NVIDIA backend required)
5. Write functional tests (`tests/test_<op>.py`)
6. Write benchmarks (`benchmark/test_<op>.py`)
7. Add metadata to `conf/operators.yaml`
8. Export from `src/flaggems_vllm/ops/__init__.py` and `src/flaggems_vllm/__init__.py`

**Acceptance criteria**:
- Tests pass in `--quick` and full modes
- SpeedUp ≥ 0.9 on core shapes (NVIDIA H100)
- NO_TORCH_COMPUTE_FALLBACK rule followed

### Optimizing an Existing Operator

Steps:
1. Identify bottleneck via profiling (see `workflow.md` section 8)
2. Propose optimization in an issue
3. Implement and measure performance improvement
4. Update autotune configs if needed
5. Ensure no regression in accuracy or other shapes

Required:
- Before/after performance comparison
- Accuracy validation (no regression)

### Adding Backend Support

See [backend development guide](backend/).

Steps:
1. Create `src/flaggems_vllm/runtime/backend/_<vendor>/`
2. Add device detection in `runtime/device_finder.py`
3. Test core operators on target hardware
4. Add CI configuration (if possible)

### Improving Tests

- Add edge cases (empty tensors, large shapes, dtypes)
- Improve reference implementations
- Add parametrized cases for better coverage

### Documentation

- Fix typos or unclear sections
- Add examples or usage patterns
- Translate content (English ↔ Chinese)

## Code Style and Conventions

### Formatting

Run before committing:

```bash
pre-commit run --all-files
```

Tools:
- **black** (line length 88)
- **isort** (`--profile black`)
- **flake8** (max line 120, ignores: F405, E731, W503, E203, E704)

### Naming Conventions

- **Operators**: Snake case (e.g., `moe_align_block_size`, `grouped_topk`)
- **Test files**: `test_<op>.py` matches operator name
- **Benchmark files**: `test_<op>.py` in `benchmark/` directory
- **Markers**: `@pytest.mark.<op>` lowercase with underscores

### File Organization

Standard drop points for operator `<op>`:

| Purpose | Path |
|---------|------|
| Implementation | `src/flaggems_vllm/ops/<op>.py` |
| Functional test | `tests/test_<op>.py` |
| Benchmark | `benchmark/test_<op>.py` |
| NVIDIA autotune config | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` |
| Metadata | `conf/operators.yaml` |
| Export | `src/flaggems_vllm/ops/__init__.py` + `src/flaggems_vllm/__init__.py` |

### Operator Metadata

Add entry to `conf/operators.yaml`:

```yaml
- id: moe_align_block_size
  name: moe_align_block_size
  description: Align MoE tokens to block boundaries
  for: null  # PyTorch op it replaces (null for vLLM-specific)
  kind: NeuralNetwork
  labels: [vLLM, MoE, fused]
  stages:
    - stage: stable
      version: v0.1.0
  source: src/flaggems_vllm/ops/moe_align_block_size.py
```

## Testing Requirements

### Functional Tests

Required for all operators:

```python
# tests/test_<op>.py
import pytest
import torch
import flaggems_vllm
from tests.accuracy_utils import gems_assert_close

@pytest.mark.<op>
@pytest.mark.parametrize("shape", [...])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_<op>(shape, dtype):
    # Setup
    inputs = create_inputs(shape, dtype)
    
    # Test
    result = flaggems_vllm.<op>(*inputs)
    expected = reference_<op>(*inputs)
    
    # Validate
    gems_assert_close(result, expected, rtol=1e-3, atol=1e-5)
```

Tests must:
- Cover representative shapes and dtypes
- Compare against a reference (vLLM, PyTorch, or custom)
- Use `gems_assert_close` with appropriate tolerances
- Pass in both `--quick` and full modes

### Performance Benchmarks

Required for all operators:

```python
# benchmark/test_<op>.py
import pytest
from benchmark.conftest import Config

@pytest.mark.parametrize("shape", Config.get_shapes("<op>"))
def test_<op>_perf(benchmark, shape):
    inputs = create_inputs(shape)
    result = benchmark(lambda: flaggems_vllm.<op>(*inputs))
    assert result is not None
```

Add core shapes to `benchmark/core_shapes.yaml`:

```yaml
<op>:
  - [128, 256]
  - [512, 1024]
  - [2048, 4096]
```

Acceptance: **SpeedUp ≥ 0.9** on core shapes (NVIDIA H100).

## NO_TORCH_COMPUTE_FALLBACK Rule

**Critical**: Production code paths must NOT use torch compute operations.

### Forbidden

```python
# ❌ FORBIDDEN in production paths
def my_operator(x):
    # These emit torch kernels:
    return torch.matmul(x, x.T)  # ❌
    return x.sum(dim=-1)          # ❌
    return x.contiguous()         # ❌ (memory copy)
    return x.to(torch.float32)    # ❌ (cast kernel)
```

### Allowed

```python
# ✓ ALLOWED
def my_operator(x):
    # Metadata reads:
    shape = x.shape              # ✓
    dtype = x.dtype              # ✓
    device = x.device            # ✓
    
    # Uninitialized allocation:
    output = torch.empty_like(x) # ✓
    
    # True no-copy views:
    view = x.view(new_shape)     # ✓
    
    # Triton kernel launch:
    triton_kernel[grid](x, ...)  # ✓
```

### Unsupported Paths

```python
# ✓ CORRECT: raise NotImplementedError
def my_operator(x):
    if x.dtype not in [torch.float16, torch.bfloat16]:
        raise NotImplementedError(f"Unsupported dtype: {x.dtype}")
    # ... operator implementation
```

Torch compute is allowed **only** in:
- Test files (`tests/`)
- Benchmark files (`benchmark/`)
- Reference implementations (marked clearly)

## Pull Request Process

1. **Create a fork** and feature branch
2. **Make your changes** following the protocol
3. **Run tests and linting**:
   ```bash
   pre-commit run --all-files
   PYTHONPATH=src pytest -q tests --quick
   PYTHONPATH=src pytest -q benchmark/test_<op>.py --level core --iter 1
   ```
4. **Commit with clear messages**:
   ```
   Add grouped_topk operator for MoE routing
   
   - Implements grouped topk with bias and renormalization
   - Adds autotune configs for NVIDIA H100
   - Tests pass with rtol=1e-3
   - SpeedUp 1.15x on core shapes
   ```
5. **Submit PR** with:
   - Description of changes
   - Link to issue (if applicable)
   - Test results (functional + performance)
   - Any breaking changes or migration notes

## Code Review

PRs are reviewed for:
- **Correctness**: Tests pass, accuracy validated
- **Performance**: Meets acceptance criteria (SpeedUp ≥ 0.9)
- **Style**: Follows code style and conventions
- **Protocol compliance**: `workflow.md` requirements met
- **Documentation**: Docstrings, comments (when needed), updated docs

## Communication Channels

- **GitHub Issues**: Bug reports, feature requests, design discussions
- **GitHub Discussions**: General questions, usage help
- **Email**: flaggems@baai.ac.cn (for security issues or private inquiries)
- **WeChat**: FlagGems community group (see [FlagGems homepage](https://github.com/flagos-ai/FlagGems))

## License

By contributing to FlagGems-vllm, you agree that your contributions will be licensed under the Apache License 2.0.

## Next Steps

- [Operator development protocol](workflow/)
- [Backend development guide](backend/)
- [Testing guide](/FlagGems-vllm/testing/unittest/)
