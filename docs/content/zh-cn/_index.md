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
- **模型专用算子**: Qwen4、DeepSeek-v4、RWKV、M3 算子

FlagGems-vllm 支持 OpenAI Triton 编译器(NVIDIA 和 AMD)以及 [FlagTree 编译器](https://github.com/flagos-ai/flagtree/)(支持 17 个硬件后端)。

## 与 FlagGems 和 vllm-plugin-fl 的关系

三个仓库在 FlagOS 生态系统中协同工作:

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**: 通用算子库，通过 `flag_gems.enable()` 和 `flag_gems.use_gems()` 提供 PyTorch 算子替换。**FlagGems-vllm 依赖 FlagGems，部分特化算子在底层执行时会直接调用 FlagGems 中的通用算子**。
- **FlagGems-vllm**(本项目): 针对 vLLM 推理场景优化的特化与融合算子库，通过 `flaggems_vllm` 包导出。
- **[vllm-plugin-fl](https://github.com/flagos-ai/vllm-plugin-fl)**: vLLM 插件层，调用 `flag_gems.enable()` 启用通用算子，显式导入 `flaggems_vllm.<operator>()` 使用 vLLM 融合算子。

**调用流程**: `vLLM → vllm-plugin-fl → flag_gems.enable() + flaggems_vllm.<op>()` (其中部分 `flaggems_vllm.<op>()` 亦会调用 `flag_gems` 内置算子)。

## 支持的算子

FlagGems-vllm 提供 **105 个算子**（157 个导出项）,针对 vLLM 推理场景优化,包括:

- MoE 路由和计算
- Flash Attention 变体(标准、变长、MLA)
- 量化(FP8、FP4、INT8)
- 融合归一化 + RoPE
- 激活函数(GELU、SwiGLU、GeGLU)
- 模型专用算子(Qwen4、DeepSeek-v4、RWKV、M3)

查看[算子参考](references/operators.md)获取完整列表。

## 支持的硬件后端

FlagGems-vllm 支持 **17 个硬件后端**:

- NVIDIA(默认,针对 Ampere/Hopper 架构优化)
- AMD
- 昇腾(华为)
- 寒武纪
- 燧原科技
- 海光
- 天数智芯
- 昆仑芯(百度)
- 沐曦
- 摩尔线程
- AIPU、ARM、壁仞、进迭时空、曦望、平头哥、清微智能

根据检测到的硬件自动选择后端,也可以通过 `DNN_VENDOR` 环境变量控制。

## 快速开始

- [安装指南](getting-started/_index.md)
- [基础用法](usage/basic.md)
- [算子列表](references/operators.md)
- [MiniMax M3 算子支持](references/minimax-m3-support.md)
- [性能基准](performance/_index.md)
- [贡献指南](contribution/_index.md)

## 快速示例

```python
import torch
import flag_gems
import flaggems_vllm

# 启用 FlagGems 算子（FlagGems-vllm 部分算子依赖其底层算子支持）
flag_gems.enable()

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
