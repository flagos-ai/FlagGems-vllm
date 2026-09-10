---
title: Unit Tests
weight: 10
---

# Unit Tests

FlagGems-vllm functional tests validate operator correctness against reference implementations.

## Running Tests

### Collect Tests

Verify test discovery:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --collect-only
```

This should show all test files and parametrized cases.

### Quick Validation

Run all tests in quick mode (reduced shapes/dtypes):

```bash
PYTHONPATH=src pytest -q tests --quick
```

### Full Test Suite

Run all tests with comprehensive coverage:

```bash
PYTHONPATH=src pytest -q tests
```

### Single Operator

Test a specific operator:

```bash
PYTHONPATH=src pytest -v tests/test_grouped_topk.py
PYTHONPATH=src pytest -v tests/test_moe_align_block_size.py
```

### Verbose Output

Show detailed output with `-s` (no capture):

```bash
PYTHONPATH=src pytest -v -s tests/test_flash_attention_forward.py
```

## Test Options

### `--quick`

Fast validation mode with reduced coverage:
- Fewer shapes per operator
- Fewer dtypes (typically float16 only)
- Smaller tensor sizes

Use for rapid iteration during development.

### `--ref {device|cpu}`

Reference device for baseline computation:
- `device` (default): Run reference on same device as test (e.g., CUDA)
- `cpu`: Run reference on CPU (useful when GPU reference is unavailable)

Example:
```bash
PYTHONPATH=src pytest -q tests --ref cpu
```

### `--record` and `--output`

Record test results to JSON:

```bash
PYTHONPATH=src pytest -q tests --quick --record --output my_results.json
```

Output format:
```json
{
  "operator": "grouped_topk",
  "tests": [
    {
      "shape": [8, 16],
      "dtype": "float16",
      "passed": true,
      "max_diff": 1.5e-5
    }
  ]
}
```

## Test Structure

Typical test file structure:

```python
import pytest
import torch
from tests.accuracy_utils import to_reference, gems_assert_close

@pytest.mark.grouped_topk
@pytest.mark.parametrize("shape", [(8, 16), (32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_grouped_topk(shape, dtype):
    # Setup
    scores = torch.randn(shape, device='cuda', dtype=dtype)
    
    # FlagGems-vllm implementation
    result = flaggems_vllm.grouped_topk(scores, ...)
    
    # Reference implementation
    ref_result = reference_grouped_topk(to_reference(scores), ...)
    
    # Validation
    gems_assert_close(result, ref_result, rtol=1e-3, atol=1e-4)
```

## Accuracy Utilities

`tests/accuracy_utils.py` provides comparison helpers:

### `to_reference(tensor)`

Move tensor to reference device (respects `--ref` flag):

```python
ref_input = to_reference(cuda_input)
```

### `gems_assert_equal(result, expected)`

Assert exact equality:

```python
gems_assert_equal(result, expected)
```

### `gems_assert_close(result, expected, rtol, atol)`

Assert numerical closeness:

```python
gems_assert_close(result, expected, rtol=1e-3, atol=1e-5)
```

### `gems_assert_shape_equal(result, expected)`

Assert shape match only:

```python
gems_assert_shape_equal(result, expected)
```

## Test Markers

Each operator has a pytest marker for selective execution:

```bash
# Run only grouped_topk tests
PYTHONPATH=src pytest -m grouped_topk

# Run all MoE-related tests (if marked)
PYTHONPATH=src pytest -m moe
```

List all markers:

```bash
pytest --markers
```

## Skipping Tests

Tests that require CUDA skip automatically if CUDA is unavailable:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_operator():
    ...
```

## CI Integration

Tests run in CI for changed operators:
- `tools/select_tests.py` maps changed files to test targets
- Runs with `--quick` for fast feedback
- Full suite runs on scheduled builds

## Debugging Test Failures

### 1. Run with verbose output

```bash
PYTHONPATH=src pytest -v -s tests/test_<op>.py
```

### 2. Isolate a single case

Parametrized tests can be selected:

```bash
PYTHONPATH=src pytest -v tests/test_grouped_topk.py::test_grouped_topk[shape0-float16]
```

### 3. Add debug logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 4. Check reference implementation

Verify the reference produces expected results:

```python
ref_result = reference_impl(inputs)
print(ref_result)
```

## Writing New Tests

When adding a new operator:

1. Create `tests/test_<op>.py`
2. Add operator marker: `@pytest.mark.<op>`
3. Parametrize over representative shapes and dtypes
4. Compare against reference (vLLM, PyTorch, or custom)
5. Use `gems_assert_close` with appropriate tolerances
6. Test edge cases (empty tensors, identity, etc.)

Example template:

```python
import pytest
import torch
import flaggems_vllm
from tests.accuracy_utils import to_reference, gems_assert_close

@pytest.mark.my_new_op
@pytest.mark.parametrize("shape", [(128, 256), (512, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_my_new_op(shape, dtype):
    input_tensor = torch.randn(shape, device='cuda', dtype=dtype)
    
    result = flaggems_vllm.my_new_op(input_tensor)
    expected = reference_my_new_op(to_reference(input_tensor))
    
    gems_assert_close(result, expected, rtol=1e-3, atol=1e-5)
```

## Next Steps

- [Coverage reporting](../coverage/)
- [Benchmark guide](/FlagGems-vllm/performance/benchmark/)
- [Contribution guide](/FlagGems-vllm/contribution/overview/)
