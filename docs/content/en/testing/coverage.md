---
title: Test Coverage
weight: 20
---

# Test Coverage

FlagGems-vllm tracks test coverage for operator implementations.

## Current Coverage

Test coverage varies by operator:
- **Core operators**: 90%+ coverage (grouped_topk, moe_align_block_size, flash_attention_forward)
- **Model-specific operators**: 70-90% coverage (qwen4_*, deepseek_v4_*)
- **Experimental operators**: 50-70% coverage

## Coverage Metrics

Coverage includes:
- **Shape coverage**: Representative shapes from vLLM workloads
- **Dtype coverage**: float16, bfloat16, float32, int8, fp8
- **Edge cases**: Empty tensors, identity operations, boundary conditions

## Viewing Coverage Reports

### HTML Coverage Reports

If coverage reports are generated (not in default setup):

```bash
# Generate coverage
PYTHONPATH=src pytest tests --cov=flaggems_vllm --cov-report=html

# View report
open htmlcov/index.html
```

### Coverage by Operator

Check test files to see what's covered:

```bash
# List all test files
ls tests/test_*.py

# Check a specific operator's tests
grep "@pytest.mark.parametrize" tests/test_grouped_topk.py
```

## Improving Coverage

To improve coverage for an operator:

1. **Identify gaps**: Review test parametrization

2. **Add test cases**:
   ```python
   @pytest.mark.my_operator
   @pytest.mark.parametrize("shape", [
       (128, 256),     # Existing
       (1, 256),       # Edge: batch size 1
       (0, 256),       # Edge: empty batch
       (128, 1),       # Edge: single feature
   ])
   def test_my_operator(shape):
       ...
   ```

3. **Add dtypes**:
   ```python
   @pytest.mark.parametrize("dtype", [
       torch.float16,
       torch.bfloat16,
       torch.float32,   # Add if supported
       torch.int8,      # Add for quantized ops
   ])
   ```

4. **Add edge cases**:
   ```python
   def test_my_operator_edge_cases():
       # Empty tensor
       x = torch.empty(0, 256, device='cuda')
       result = flaggems_vllm.my_operator(x)
       assert result.shape == (0, 256)

       # Identity
       x = torch.eye(256, device='cuda')
       result = flaggems_vllm.my_operator(x)
       # ... validate identity behavior
   ```

## Coverage Goals

Target coverage by operator stage:
- **alpha**: 50%+ (basic functionality)
- **beta**: 70%+ (expanded shapes/dtypes)
- **stable**: 90%+ (comprehensive coverage including edge cases)

## Next Steps

- [Unit test guide](unittest/)
- [Contribution guide](/FlagGems-vllm/contribution/overview/)
