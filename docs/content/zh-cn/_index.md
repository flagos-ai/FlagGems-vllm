---
title: FlagGems-vllm
type: docs
---

# FlagGems-vllm 概述

## 关于 FlagGems-vllm

**FlagGems-vllm** 是专为 vLLM 推理场景设计的高性能算子库，是 [FlagOS](https://flagos.io/) 的一部分。它使用 [Triton 编程语言](https://github.com/openai/triton)实现了针对 vLLM 的融合算子优化。

FlagGems-vllm 专注于加速大语言模型(LLM)在 vLLM 中的推理，包括:
- **MoE(专家混合)算子**: `moe_align_block_size`、`grouped_topk`、`fused_experts_impl`
- **融合注意力算子**: `flash_attention_forward`、`flash_attn_varlen_func`、`triton_unified_attention`
- **量化算子**: `per_token_group_quant_fp8`、`scaled_int8_quant`、`fp8_fp4_mqa_logits`
- **RoPE 和归一化融合**: `fused_add_rms_norm`、`fused_inv_rope_fp8_quant`、`apply_rotary_pos_emb`
- **模型专用算子**: Qwen4、DeepSeek-v4、RWKV 算子

FlagGems-vllm 支持 OpenAI Triton 编译器(NVIDIA 和 AMD)以及 [FlagTree 编译器](https://github.com/flagos-ai/flagtree/)(支持 18 个硬件后端)。

## 与 FlagGems 和 vllm-plugin-fl 的关系

三个仓库在 FlagOS 生态系统中协同工作:

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**: 通用算子库,通过 `flag_gems.enable()` 和 `flag_gems.use_gems()` 提供 PyTorch 算子替换
- **FlagGems-vllm**(本项目): vLLM 专用融合算子,通过 `flaggems_vllm` 包导出
- **[vllm-plugin-fl](https://github.com/flagos-ai/vllm-plugin-fl)**: vLLM 插件层,调用 `flag_gems.enable()` 启用通用算子,显式导入 `flaggems_vllm.<operator>()` 使用 vLLM 融合算子

**调用流程**: `vLLM → vllm-plugin-fl → flag_gems.enable() + flaggems_vllm.<op>()`

## 支持的算子

FlagGems-vllm 提供 **102 个算子**,针对 vLLM 推理场景优化,包括:

- MoE 路由和计算
- Flash Attention 变体(标准、变长、MLA)
- 量化(FP8、FP4、INT8)
- 融合归一化 + RoPE
- 激活函数(GELU、SwiGLU、GeGLU)
- 模型专用算子(Qwen4、DeepSeek-v4、RWKV)

查看[算子参考](/FlagGems-vllm/references/operators/)获取完整列表。

## 支持的硬件后端

FlagGems-vllm 支持 **18 个硬件后端**:

- NVIDIA(默认,针对 Ampere/Hopper 架构优化)
- AMD
- 昇腾(华为)
- 寒武纪
- 燧原科技
- 海光
- 天数智芯
- 昆仑芯(百度)
- 墨芯
- 摩尔线程
- 等等...

根据检测到的硬件自动选择后端,也可以通过 `DNN_VENDOR` 环境变量控制。

## 快速开始

- [安装指南](/FlagGems-vllm/getting-started/)
- [基础用法](/FlagGems-vllm/usage/basic/)
- [算子列表](/FlagGems-vllm/references/operators/)
- [性能基准](/FlagGems-vllm/performance/)
- [贡献指南](/FlagGems-vllm/contribution/)

## 快速示例

```python
import torch
import flaggems_vllm

# MoE token 路由和对齐
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

print(f"对齐后: {sorted_ids.shape}, 专家: {expert_ids.shape}, 填充后 tokens: {num_tokens_post_pad}")
```

## 联系我们

如有问题或想要贡献,请:
- 在 [GitHub](https://github.com/flagos-ai/FlagGems-vllm) 提交 issue
- 发送邮件至 <flaggems@baai.ac.cn>
- 加入 FlagGems 微信群(见 [FlagGems 主页](https://github.com/flagos-ai/FlagGems))

## 开源协议

FlagGems-vllm 采用 Apache License 2.0 协议。
