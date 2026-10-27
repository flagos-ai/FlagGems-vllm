---
title: 基础用法
weight: 10
---

# 基础用法

FlagGems-vllm 提供了可直接调用的高性能算子，同时也支持通过类似 FlagGems 的全局启用 API 进行 PyTorch 调度分发。

## 直接调用算子

所有算子均可直接从 `flaggems_vllm` 包导入调用：

```python
import torch
import flaggems_vllm

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

列出所有已注册的算子（注意：需要在调用 `enable()` 后，注册器才能列出注册项）：

```python
import flaggems_vllm

# 启用注册
flaggems_vllm.enable()

# 获取所有注册算子名称
ops = flaggems_vllm.all_registered_ops()
print(f"可用算子总数: {len(ops)}")
print(ops[:10])  # 前 10 个算子

# 获取所有注册键值
keys = flaggems_vllm.all_registered_keys()
print(f"注册键数: {len(keys)}")
```

## 全局启用 API

FlagGems-vllm 提供了与 FlagGems 一致的启用接口：

### `enable()`

全局启用所有 FlagGems-vllm 算子：

```python
import flaggems_vllm

flaggems_vllm.enable()
# 此时 FlagGems-vllm 算子已注册进 PyTorch 调度机制
```

**提示**: 在 FlagGems-vllm 中通常直接显式调用算子即可；通用的 PyTorch 基础算子替换建议使用 `flag_gems.enable()`。

### `only_enable()`

仅启用指定算子：

```python
import flaggems_vllm

flaggems_vllm.only_enable(include=['grouped_topk', 'moe_align_block_size'])
```

### `use_gems()` (上下文管理器)

在指定代码块中临时启用：

```python
import torch
import flaggems_vllm

with flaggems_vllm.use_gems():
    # 作用域内算子生效
    result = some_computation()
# 退出作用域后自动还原
```

## 检查设备与后端

```python
import flaggems_vllm
from flaggems_vllm import runtime

print(f"厂商名称: {flaggems_vllm.vendor_name}")  # 例如 'nvidia'
print(f"设备名称: {flaggems_vllm.device}")        # 例如 'cuda'
print(f"设备数量: {runtime.device.device_count}")
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

详见[算子列表](/FlagGems-vllm/zh-cn/references/operators/)。

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

- [vLLM 插件集成](/FlagGems-vllm/zh-cn/usage/vllm-plugin/)
- [选择性启用](/FlagGems-vllm/zh-cn/usage/selective/)
- [调试与日志](/FlagGems-vllm/zh-cn/usage/debugging/)
- [运行基准测试](/FlagGems-vllm/zh-cn/performance/benchmark/)
