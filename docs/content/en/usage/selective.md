---
title: Selective Enablement
weight: 30
---

# Selective Operator Enablement

Control which FlagGems-vllm operators are enabled for fine-grained performance tuning.

## Use Cases

- **Debugging**: Isolate specific operators to test behavior
- **Performance tuning**: Enable only well-optimized operators
- **Gradual rollout**: Enable operators incrementally in production
- **Benchmarking**: Compare specific operators against baselines

## `only_enable(include=...)`

Enable only specified operators:

```python
import flaggems_vllm

# Enable only MoE operators
flaggems_vllm.only_enable(include=['moe_align_block_size', 'grouped_topk', 'fused_experts_impl'])
```

All other operators remain disabled.

## `use_gems(include=..., exclude=...)`

Context manager with fine-grained control:

```python
import flaggems_vllm

# Enable all except problematic operators
with flaggems_vllm.use_gems(exclude=['experimental_op']):
    result = computation()

# Enable only specific operators
with flaggems_vllm.use_gems(include=['flash_attention_forward', 'fused_add_rms_norm']):
    result = attention_computation()
```

## Discovering Operators

List all available operators:

```python
import flaggems_vllm

ops = flaggems_vllm.all_registered_ops()
print(f"Total operators: {len(ops)}")
print(ops[:10])  # First 10

# Get registration keys
keys = flaggems_vllm.all_registered_keys()
print(keys)
```

## Example: Debug a Specific Operator

```python
import flaggems_vllm
import torch

# Disable all operators
# (no global enable call)

# Test baseline (torch)
input_tensor = torch.randn(128, 256, device='cuda')
baseline = torch_computation(input_tensor)

# Enable only the operator under test
flaggems_vllm.only_enable(include=['my_operator'])
result = computation_using_my_operator(input_tensor)

# Compare
diff = (result - baseline).abs().max()
print(f"Max diff: {diff}")
```

## Example: Gradual Rollout

```python
import flaggems_vllm

# Stage 1: Enable only battle-tested operators
STABLE_OPS = [
    'moe_align_block_size',
    'grouped_topk',
    'flash_attention_forward',
    'fused_add_rms_norm',
]

flaggems_vllm.only_enable(include=STABLE_OPS)

# Run production workload
# ...

# Stage 2: Add more operators after validation
```

## Checking What's Enabled

Currently there's no direct API to query enabled status. As a workaround, check registration keys:

```python
import flaggems_vllm

all_ops = flaggems_vllm.all_registered_ops()
all_keys = flaggems_vllm.all_registered_keys()

print(f"Registered operators: {len(all_ops)}")
print(f"Registered dispatch keys: {len(all_keys)}")
```

## Best Practices

1. **Start conservative**: Enable only operators with proven performance
2. **Test incrementally**: Add operators one at a time when debugging
3. **Monitor performance**: Use selective enablement to isolate regressions
4. **Document choices**: Comment why specific operators are included/excluded

## Limitations

- `include` and `exclude` operate at operator name granularity
- No regex or wildcard matching (e.g., `"moe_*"` doesn't work)
- Must know exact operator names (use `all_registered_ops()` to discover)

## Next Steps

- [Basic usage patterns](basic/)
- [Debugging and logging](debugging/)
- [Performance benchmarking](/FlagGems-vllm/performance/benchmark/)
