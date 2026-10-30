---
title: Basic Usage
weight: 10
---

# Basic Usage

FlagGems-vllm provides optimized operators designed for vLLM inference workloads. **FlagGems-vllm depends on FlagGems**, and currently some FlagGems-vllm operators invoke FlagGems operators internally.

## Dependency on FlagGems and Execution Flow

- **Prerequisite**: Ensure `flag_gems` is installed in your environment before installing or running FlagGems-vllm.
- **Internal Operator Invocation**: Several FlagGems-vllm operators (such as multi-head computation `mhc_pre`, backend-specific quantized matrix multiplications `bmm_w8a8_int8`, and cache writers `reshape_and_cache_flash`) invoke FlagGems operators (e.g., `flag_gems.bmm_out`, `flag_gems.mm_w8a8_int8`, `torch.ops.flag_gems.*`) during execution.
- **Recommended Usage Pattern**: Call `flag_gems.enable()` prior to executing workloads with FlagGems-vllm to initialize FlagGems' operator registration and ensure proper dispatch.

## Direct Operator Invocation

All operators are exported from the `flaggems_vllm` package. The recommended invocation pattern is:

```python
import torch
import flag_gems
import flaggems_vllm

# Recommended: Enable FlagGems general operator replacement and internal dispatch
flag_gems.enable()

# 1. MoE token alignment
topk_ids = torch.randint(0, 16, (128, 2), device='cuda', dtype=torch.int32)
sorted_ids, expert_ids, num_tokens = flaggems_vllm.moe_align_block_size(
    topk_ids=topk_ids,
    block_size=32,
    num_experts=16,
)

# 2. Grouped topk for MoE routing
scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
topk_weights, topk_ids = flaggems_vllm.grouped_topk(
    scores,
    n_group=4,
    topk_group=2,
    topk=2,
    renormalize=True,
)

# 3. Flash attention
q = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
k = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
v = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
output = flaggems_vllm.flash_attention_forward(q, k, v, causal=True)

# 4. Fused add + RMS norm
hidden = torch.randn((128, 4096), device='cuda', dtype=torch.float16)
residual = torch.randn((128, 4096), device='cuda', dtype=torch.float16)
weight = torch.randn((4096,), device='cuda', dtype=torch.float16)
output = flaggems_vllm.fused_add_rms_norm(hidden, residual, weight, eps=1e-6)
```

## Operator Discovery

Inspect all exported specialized operators directly via `flaggems_vllm.ops.__all__`:

```python
import flaggems_vllm

ops = flaggems_vllm.ops.__all__
print(f"Available operators: {len(ops)}")
print(ops[:10])  # First 10 operators
```

## Direct Invocation Only

> [!IMPORTANT]
> **Design Note**: FlagGems-vllm is designed exclusively for vLLM specialized and fused operators. **It only supports direct invocation (`flaggems_vllm.<op>()`); it does NOT support PyTorch dispatch registration, nor does it support `flaggems_vllm.use_gems`**.
>
> PyTorch dispatch registration for general operators is handled by the underlying FlagGems dependency. Simply call `flag_gems.enable()` to initialize the general operator foundation:
>
> ```python
> import flag_gems
> import flaggems_vllm
>
> # Initialize FlagGems general operator foundation
> flag_gems.enable()
>
> # Call FlagGems-vllm specialized fused operators directly
> output = flaggems_vllm.grouped_topk(...)
> ```

## Checking Device and Backend

```python
import flaggems_vllm
from flaggems_vllm import runtime

print(f"Vendor: {flaggems_vllm.vendor_name}")  # e.g., 'nvidia'
print(f"Device: {flaggems_vllm.device}")        # e.g., 'cuda'
print(f"Device count: {runtime.device.device_count}")
```

For multi-backend selection and environment variables, see [Non-NVIDIA Backends](non-nvidia.md).

## C++ Native Extensions Configuration

FlagGems-vllm includes optional C++ operators compiled via pybind11. If dependencies are satisfied during build, they are compiled automatically.

Enable C++ operators at runtime via an environment variable:

```bash
export USE_C_EXTENSION=1
```

Check C++ extension status:

```python
import flaggems_vllm

print("C++ extension available:", flaggems_vllm.config.has_c_extension)
print("C++ extension enabled:", flaggems_vllm.config.use_c_extension)
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

See the [operator reference](../references/operators.md) for the complete list.

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

- [vLLM plugin integration](vllm-plugin.md)
- [Non-NVIDIA hardware platforms](non-nvidia.md)
- [Debugging and logging](debugging.md)
- [Performance benchmarking](../performance/benchmark.md)
