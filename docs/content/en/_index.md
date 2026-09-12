---
title: FlagGems-vllm
type: docs
---

# FlagGems-vllm Overview

## About FlagGems-vllm

**FlagGems-vllm** is a high-performance operator library designed for vLLM inference scenarios, part of [FlagOS](https://flagos.io/). It provides optimized implementations of vLLM-specific fused kernels written in the [Triton](https://github.com/openai/triton) programming language.

FlagGems-vllm focuses on operators that accelerate large language model (LLM) inference in vLLM, including:
- **MoE (Mixture of Experts) operators**: `moe_align_block_size`, `grouped_topk`, `fused_experts_impl`
- **Fused attention kernels**: `flash_attention_forward`, `flash_attn_varlen_func`, `triton_unified_attention`
- **Quantization operators**: `per_token_group_quant_fp8`, `scaled_int8_quant`, `fp8_fp4_mqa_logits`
- **RoPE and normalization fusions**: `fused_add_rms_norm`, `fused_inv_rope_fp8_quant`, `apply_rotary_pos_emb`
- **Model-specific kernels**: Qwen4, DeepSeek-v4, RWKV operators

FlagGems-vllm is supported by the OpenAI Triton compiler (for NVIDIA and AMD) and [FlagTree compiler](https://github.com/flagos-ai/flagtree/) for different AI hardware platforms (18 backend vendors supported).

## Relationship with FlagGems and vllm-plugin-fl

The three repositories work together in the FlagOS ecosystem:

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**: General-purpose operator library providing PyTorch operator replacements via `flag_gems.enable()` and `flag_gems.use_gems()`
- **FlagGems-vllm** (this project): vLLM-specific fused kernels exposed through the `flaggems_vllm` package
- **[vllm-plugin-fl](https://github.com/flagos-ai/vllm-plugin-fl)**: vLLM plugin layer that calls `flag_gems.enable()` for general ops and imports `flaggems_vllm.<operator>()` for vLLM fused kernels

**Call flow**: `vLLM → vllm-plugin-fl → flag_gems.enable() + flaggems_vllm.<op>()`

## Supported Operators

FlagGems-vllm provides **102 operators** optimized for vLLM inference, including:

- MoE routing and computation
- Flash Attention variants (standard, variable-length, MLA)
- Quantization (FP8, FP4, INT8)
- Fused normalization + RoPE
- Activation functions (GELU, SwiGLU, GeGLU)
- Model-specific operators (Qwen4, DeepSeek-v4, RWKV)

See the [operator reference](/FlagGems-vllm/references/operators/) for the complete list.

## Supported Backends

FlagGems-vllm supports **18 hardware backends**:

- NVIDIA (default, with Ampere/Hopper architecture-specific tuning)
- AMD
- Ascend (Huawei)
- Cambricon
- Enflame
- Hygon
- Iluvatar
- Kunlunxin (Baidu)
- MetaX
- Moore Threads
- And more...

Backend selection is automatic based on the detected hardware, or can be controlled via the `DNN_VENDOR` environment variable.

## Next Steps

- [Get started with installation](/FlagGems-vllm/getting-started/)
- [Learn basic usage patterns](/FlagGems-vllm/usage/basic/)
- [Review operator list](/FlagGems-vllm/references/operators/)
- [Check performance benchmarks](/FlagGems-vllm/performance/)
- [Contribute new operators](/FlagGems-vllm/contribution/)

## Quick Example

```python
import torch
import flaggems_vllm

# MoE token routing and alignment
num_tokens = 128
topk = 2
num_experts = 16
block_size = 32

topk_ids = torch.randint(
    low=0, high=num_experts,
    size=(num_tokens, topk),
    device='cuda', dtype=torch.int32
)

sorted_ids, expert_ids, num_tokens_post_pad = flaggems_vllm.moe_align_block_size(
    topk_ids=topk_ids,
    block_size=block_size,
    num_experts=num_experts,
)

print(f"Aligned: {sorted_ids.shape}, experts: {expert_ids.shape}, padded tokens: {num_tokens_post_pad}")
```

## Contact Us

If you have questions or want to contribute, please:
- Submit an issue on [GitHub](https://github.com/flagos-ai/FlagGems-vllm)
- Email us at <flaggems@baai.ac.cn>
- Join the FlagGems WeChat group (see [FlagGems homepage](https://github.com/flagos-ai/FlagGems))

## License

FlagGems-vllm is licensed under the Apache License 2.0.
