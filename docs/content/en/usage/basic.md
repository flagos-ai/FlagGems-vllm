---
title: Basic Usage
weight: 10
---

# Basic Usage

FlagGems-vllm provides operators that can be called directly or used with FlagGems' global enablement API.

## Direct Operator Invocation

All operators are exported from the `flaggems_vllm` package:

```python
import torch
import flaggems_vllm

# MoE token alignment
topk_ids = torch.randint(0, 16, (128, 2), device='cuda', dtype=torch.int32)
sorted_ids, expert_ids, num_tokens = flaggems_vllm.moe_align_block_size(
    topk_ids=topk_ids,
    block_size=32,
    num_experts=16,
)

# Grouped topk for MoE routing
scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
topk_weights, topk_ids = flaggems_vllm.grouped_topk(
    scores,
    n_group=4,
    topk_group=2,
    topk=2,
    renormalize=True,
)

# Flash attention
q = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
k = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
v = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
output = flaggems_vllm.flash_attention_forward(q, k, v, causal=True)

# Fused add + RMS norm
hidden = torch.randn((128, 4096), device='cuda', dtype=torch.float16)
residual = torch.randn((128, 4096), device='cuda', dtype=torch.float16)
weight = torch.randn((4096,), device='cuda', dtype=torch.float16)
output = flaggems_vllm.fused_add_rms_norm(hidden, residual, weight, eps=1e-6)
```

## Operator Discovery

List all available operators:

```python
import flaggems_vllm

# Get all registered operator names
ops = flaggems_vllm.all_registered_ops()
print(f"Available operators: {len(ops)}")
print(ops[:10])  # First 10 operators

# Get all registration keys
keys = flaggems_vllm.all_registered_keys()
print(f"Registration keys: {len(keys)}")
```

## Global Enablement API

FlagGems-vllm provides the same enablement API as FlagGems for consistent usage patterns:

### `enable()`

Enable all FlagGems-vllm operators globally:

```python
import flaggems_vllm

flaggems_vllm.enable()
# Now FlagGems-vllm operators are registered for PyTorch dispatch
```

**Note**: This is rarely needed for FlagGems-vllm since most operators are called directly. Use `flag_gems.enable()` for general PyTorch operator replacement.

### `only_enable()`

Enable only specific operators:

```python
import flaggems_vllm

flaggems_vllm.only_enable(include=['grouped_topk', 'moe_align_block_size'])
```

### `use_gems()` (context manager)

Temporarily enable operators in a scope:

```python
import torch
import flaggems_vllm

with flaggems_vllm.use_gems():
    # FlagGems-vllm operators active here
    result = some_computation()
# Operators disabled outside the context
```

## Checking Device and Backend

```python
import flaggems_vllm
from flaggems_vllm import runtime

print(f"Vendor: {flaggems_vllm.vendor_name}")  # e.g., 'nvidia'
print(f"Device: {flaggems_vllm.device}")        # e.g., 'cuda'
print(f"Device count: {runtime.device.device_count}")
```

## Operator Categories

FlagGems-vllm operators fall into these categories:

### MoE (Mixture of Experts)
- `moe_align_block_size` - Align tokens to expert block boundaries
- `grouped_topk` - Grouped topk selection with bias and renormalization
- `fused_experts_impl` - Fused expert computation
- `dispatch_fused_moe_kernel` - MoE dispatch kernel
- `moe_sum` - Expert output aggregation

### Attention
- `flash_attention_forward` - Flash Attention forward pass
- `flash_attn_varlen_func` - Variable-length flash attention
- `flash_mla` - Multi-latent attention
- `triton_unified_attention` - Unified attention kernel
- `sparse_attn_triton` - Sparse attention

### Quantization
- `per_token_group_quant_fp8` - Per-token FP8 quantization
- `scaled_int8_quant` - Scaled INT8 quantization
- `fp8_fp4_mqa_logits` - Mixed FP8/FP4 MQA logits
- `act_quant_triton` - Activation quantization

### Fused Kernels
- `fused_add_rms_norm` - Fused add + RMS normalization
- `fused_inv_rope_fp8_quant` - Fused inverse RoPE + FP8 quantization
- `fused_q_kv_rmsnorm` - Fused Q/KV RMS normalization
- `gelu_and_mul`, `silu_and_mul` - Fused activation + multiply

### RoPE (Rotary Position Embedding)
- `apply_rotary_pos_emb` - Apply rotary position embeddings
- `mrope` - Multi-resolution RoPE

### Model-Specific
- `qwen4_*` - Qwen4 model operators
- `stage_deepseek_v4_*` - DeepSeek-v4 operators
- `rwkv_*` - RWKV model operators

See the [operator reference](/FlagGems-vllm/references/operators/) for the complete list.

## Error Handling

FlagGems-vllm operators raise `NotImplementedError` for unsupported configurations instead of silently falling back to PyTorch:

```python
try:
    result = flaggems_vllm.some_operator(unsupported_args)
except NotImplementedError as e:
    print(f"Unsupported configuration: {e}")
    # Handle fallback explicitly
```

This is by design (see `NO_TORCH_COMPUTE_FALLBACK` in the contribution guide).

## Next Steps

- [vLLM plugin integration](vllm-plugin/)
- [Selective operator enablement](selective/)
- [Debugging and logging](debugging/)
- [Performance benchmarking](/FlagGems-vllm/performance/benchmark/)
