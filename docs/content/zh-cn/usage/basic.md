---
title: 基础用法
weight: 10
---

# 基础用法

FlagGems-vllm 提供了面向 vLLM 大模型推理的高性能特化算子。**FlagGems-vllm 依赖 FlagGems**，当前部分 FlagGems-vllm 算子在底层执行时会直接调用 FlagGems 中的算子。

## 依赖 FlagGems 与调用机制

- **前置依赖**：在使用 FlagGems-vllm 前，必须在环境中安装 `flag_gems`。
- **内部算子调用**：部分 FlagGems-vllm 算子（如多头计算 `mhc_pre`、特定后端矩阵乘 `bmm_w8a8_int8`、缓存写入 `reshape_and_cache_flash` 等）在实现中会直接调用 `flag_gems` 提供的算子（如 `flag_gems.bmm_out`、`flag_gems.mm_w8a8_int8`、`torch.ops.flag_gems.*`）。
- **推荐初始化方式**：在使用 FlagGems-vllm 算子前，建议同时导入 `flag_gems` 并调用 `flag_gems.enable()`，以确保底层调用的通用算子能够正常分发并发挥最佳性能。

## 直接调用算子

所有特化算子均可直接从 `flaggems_vllm` 包导入调用。推荐的使用模式如下：

```python
import torch
import flag_gems
import flaggems_vllm

# 推荐：启用 FlagGems 通用算子，为 FlagGems-vllm 底层调用的算子提供完整运行时支持
flag_gems.enable()

# 1. MoE Token 对齐
topk_ids = torch.randint(0, 16, (128, 2), device='cuda', dtype=torch.int32)
sorted_ids, expert_ids, num_tokens = flaggems_vllm.moe_align_block_size(
    topk_ids=topk_ids,
    block_size=32,
    num_experts=16,
)

# 2. MoE 路由 Grouped TopK
scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
topk_weights, topk_ids = flaggems_vllm.grouped_topk(
    scores,
    n_group=4,
    topk_group=2,
    topk=2,
    renormalize=True,
)

# 3. Flash Attention 前向计算
q = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
k = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
v = torch.randn((2, 8, 128, 64), device='cuda', dtype=torch.float16)
output = flaggems_vllm.flash_attention_forward(q, k, v, causal=True)

# 4. 融合 Add + RMSNorm
hidden = torch.randn((128, 4096), device='cuda', dtype=torch.float16)
residual = torch.randn((128, 4096), device='cuda', dtype=torch.float16)
weight = torch.randn((4096,), device='cuda', dtype=torch.float16)
output = flaggems_vllm.fused_add_rms_norm(hidden, residual, weight, eps=1e-6)
```

## 查询可用算子

直接从 `flaggems_vllm.ops.__all__` 获取已导出的全部可用特化算子列表：

```python
import flaggems_vllm

ops = flaggems_vllm.ops.__all__
print(f"可用算子总数: {len(ops)}")
print(ops[:10])  # 前 10 个算子
```

## 仅支持直接调用机制

> [!IMPORTANT]
> **设计说明**：FlagGems-vllm 专为 vLLM 专用特化与融合算子设立，**只支持直接调用方式（`flaggems_vllm.<op>()`），不支持向 PyTorch 注册分发**，亦不支持 `flaggems_vllm.use_gems`。
>
> 通用基础算子的 PyTorch 动态分发由底层依赖的 FlagGems 负责。在调用 FlagGems-vllm 算子前，只需调用 `flag_gems.enable()` 激活通用算子底座即可：
>
> ```python
> import flag_gems
> import flaggems_vllm
>
> # 激活 FlagGems 通用算子支持
> flag_gems.enable()
>
> # 直接调用 FlagGems-vllm 特化融合算子
> output = flaggems_vllm.grouped_topk(...)
> ```

## 检查设备与后端

```python
import flaggems_vllm
from flaggems_vllm import runtime

print(f"厂商名称: {flaggems_vllm.vendor_name}")  # 例如 'nvidia'
print(f"设备名称: {flaggems_vllm.device}")        # 例如 'cuda'
print(f"设备数量: {runtime.device.device_count}")
```

多后端详细配置与环境变量选择请参阅[非 NVIDIA 硬件平台](non-nvidia.md)。

## C++ 原生扩展配置

FlagGems-vllm 包含通过 pybind11 编译的可选 C++ 扩展算子。如果构建时所需依赖就绪，会自动编译生成。

运行时可通过环境变量控制是否启用 C++ 算子：

```bash
export USE_C_EXTENSION=1
```

检查 C++ 扩展状态：

```python
import flaggems_vllm

print("C++ 扩展可用:", flaggems_vllm.config.has_c_extension)
print("当前启用 C++ 扩展:", flaggems_vllm.config.use_c_extension)
```

## 算子分类概览

FlagGems-vllm 算子主要分为以下几类：

### MoE (混合专家)
- `moe_align_block_size` - 按专家块边界对齐 token
- `grouped_topk` - 带偏置和重归一化的分组 topk 路由
- `fused_experts_impl` - 融合专家前向计算
- `dispatch_fused_moe_kernel` - MoE 调度内核
- `moe_sum` - 专家输出聚合累加

### 注意力机制 (Attention)
- `flash_attention_forward` - Flash Attention 前向计算
- `flash_attn_varlen_func` - 变长序列 Flash Attention
- `flash_mla` - Multi-latent attention
- `triton_unified_attention` - 统一注意力内核
- `sparse_attn_triton` - 稀疏注意力

### 量化 (Quantization)
- `per_token_group_quant_fp8` - Per-token 分组 FP8 量化
- `scaled_int8_quant` - 比例 INT8 对称量化
- `fp8_fp4_mqa_logits` - 混合精度 FP8/FP4 MQA logits
- `act_quant_triton` - 激活值量化

### 融合算子 (Fused Kernels)
- `fused_add_rms_norm` - 融合残差相加与 RMSNorm
- `fused_inv_rope_fp8_quant` - 融合逆 RoPE 与 FP8 量化
- `fused_q_kv_rmsnorm` - 融合 Q/KV RMSNorm
- `gelu_and_mul`, `silu_and_mul` - 融合激活与元素级乘法

### 位置编码 (RoPE)
- `apply_rotary_pos_emb` - 旋转位置编码计算
- `mrope` - 多模态分辨率 RoPE

### 模型专用
- `qwen4_*` - Qwen4 模型专用内核
- `stage_deepseek_v4_*` - DeepSeek-v4 专用优化
- `rwkv_*` - RWKV 系列算子
- M3 系列算子（`fp32_router_gemm`, `topk_sigmoid` 等）

详见[算子列表](../references/operators.md)。

## 异常处理与无回退规则

FlagGems-vllm 严格遵循 `NO_TORCH_COMPUTE_FALLBACK` 规则。当遇到不支持的参数组合、数据类型或形状时，会明确抛出 `NotImplementedError`，而不是静默回退到 PyTorch 计算：

```python
try:
    result = flaggems_vllm.some_operator(unsupported_args)
except NotImplementedError as e:
    print(f"不支持的配置: {e}")
    # 显式处理降级逻辑
```

## 下一步

- [vLLM 插件集成](vllm-plugin.md)
- [非 NVIDIA 硬件平台](non-nvidia.md)
- [调试与日志](debugging.md)
- [运行基准测试](../performance/benchmark.md)
