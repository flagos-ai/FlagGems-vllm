---
title: 基准测试结果
weight: 20
useTabulator: true
---

# 基准测试性能结果

FlagGems-vllm 算子在不同硬件后端与微架构上的实测性能与加速比。

## 交互式基准结果

{{< benchmark-table >}}

## 结果指标解读

- **SpeedUp > 1.0**: FlagGems-vllm 性能快于对比基线
- **SpeedUp = 1.0**: 性能持平
- **SpeedUp < 1.0**: 对比基线更快（优化空间）

**验收目标**: 生产级算子在核心形状上达到 **SpeedUp ≥ 0.9**。

## 评测基准环境

- **测试卡型**: NVIDIA H100 (默认 CI 环境)
- **输入形状**: 来自 `benchmark/core_shapes.yaml` 的核心形状
- **评测轮次**: 100 次（预热: 20 次）
- **对比基线**: PyTorch 原生参考实现或 vLLM 原生算子

## 重点算子性能亮点

### MoE 专家混合算子

| 算子 | 加速比 (H100) | 优化说明 |
|----------|----------------|-------|
| `moe_align_block_size` | 1.2–1.5× | 针对 block_size=32,64,128 预调优 |
| `grouped_topk` | 1.1–1.3× | 针对 n_group=4,8 深度优化 |
| `fused_experts_impl` | 1.0–1.2× | 融合矩阵乘法与激活函数计算 |

### 注意力机制算子

| 算子 | 加速比 (H100) | 优化说明 |
|----------|----------------|-------|
| `flash_attention_forward` | 0.95–1.05× | 达到 Flash Attention 2 相当水平 |
| `flash_attn_varlen_func` | 0.98–1.1× | 变长序列动态调度优化 |
| `triton_unified_attention` | 1.0–1.15× | 统一多头注意力融合内核 |

### 量化算子

| 算子 | 加速比 (H100) | 优化说明 |
|----------|----------------|-------|
| `per_token_group_quant_fp8` | 1.3–1.6× | 支持分组的 FP8 高效量化 |
| `scaled_int8_quant` | 1.2–1.4× | INT8 对称动态缩放量化 |

更多关于 MiniMax M3 推理量化融合（`fp32_router_gemm`, `fused_minimax_m3_qknorm_rope_kv_insert` 等）的详细评测，请查阅 [MiniMax M3 算子支持与评测](../references/minimax-m3-support.md)。

## 复现测试命令

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q benchmark \
    --level core --iter 100 --warmup 20 \
    --record --output my_results.json
```

## 下一步

- [运行自己的基准测试](benchmark.md)
- [MiniMax M3 算子支持与评测](../references/minimax-m3-support.md)
- [贡献性能优化代码](../contribution/overview.md)
