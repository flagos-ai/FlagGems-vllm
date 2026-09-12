---
title: Testing
weight: 50
bookCollapseSection: true
---

# Testing

FlagGems-vllm includes comprehensive functional tests to validate operator correctness.

## Topics

- [Unit Tests](unittest/) - Running functional tests
- [Test Coverage](coverage/) - Coverage reporting

## Quick Start

Run all tests with quick mode:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q tests --quick
```

Run a specific operator test:

```bash
PYTHONPATH=src pytest -v tests/test_grouped_topk.py
```

## Test Structure

Tests follow the pattern `tests/test_<op>.py` matching `src/flaggems_vllm/ops/<op>.py`.

Each test:
- Parametrizes over shapes and dtypes
- Compares against a reference implementation (vLLM or PyTorch)
- Uses `accuracy_utils.py` helpers (`gems_assert_equal`, `gems_assert_close`)
- Marks with operator-specific pytest marker (e.g., `@pytest.mark.grouped_topk`)

## Test Options

- `--quick`: Fast validation with reduced shape/dtype coverage
- `--ref {device|cpu}`: Reference device (default: device)
- `--record`: Record results to JSON
- `--output FILE`: Output file (default: `accuracy_result.json`)

## Next Steps

- [Unit test guide](unittest/)
- [Coverage reporting](coverage/)
