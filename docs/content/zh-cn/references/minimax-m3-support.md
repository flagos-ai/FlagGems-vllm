---
title: MiniMax M3 算子支持
weight: 15
---

# MiniMax M3 算子支持与量化融合

本文档详细说明了针对 MiniMax M3 模型推理场景的算子支持规范、量化融合、功能验证与基准测试性能表现。

## 算子接口规范 (Operator Specifications)

| 算子 | 支持规范与约束 |
|---|---|
| `fp32_router_gemm` | BF16/FP32 `[M,6144]`, FP32 `[128,6144]`, M<=32；原生 FP32 logits，固定 FMA/约简逻辑，禁用 TF32/原子操作，2D strides，支持空批次。拒绝不支持的维度。 |
| `topk_sigmoid` | FP16/BF16/FP32 连续 `[M,E]`, E<=256, K<=min(32,E)；可选 FP32 修正偏置仅影响选择不影响输出概率；稳定的 tie 打平规则，精确源行 ID，可选重归一化与 NaN 填充，支持就地修改调用方缓冲区。 |
| `fused_minimax_m3_qknorm_rope_kv_insert` | 连续 BF16 打包 QKV, D128, Gemma RMSNorm, NeoX rotary 前缀 8/16/32/64/128，可选 Q/index-Q gather，strided NHD/HND 分页主缓存，BF16 index 缓存，BF16 或单位比例 E4M3 主缓存。负 slot 自动跳过写入。调用方需提供有效位置、缓存边界和互不重复的非负目标位置。 |
| `vit_flash_attn_wrapper` / `vision_flash_attn_varlen` | 非因果 FP16/BF16 D64/80 注意力；3D packed 或 4D batch 输入，Triton 步长打包/均匀累计长度，有限缩放（含 0 与负数），设备序列边界保持重放输入。复用现有变长注意力内核。 |
| OAI MoE | 显式 `swigluoai_uninterleave` 配合有限 alpha/beta/clamp 参数，透传至所有专家包装器；保留原有激活函数命名。 |
| 动态 INT8 融合 | BF16/FP16 2D 行至 K8192；可选 OAI+quant2 保留取整激活工作区。纯 Torch 与完整 FlagGems 量化注册保留各自算术；混合注册保留原有未融合路径。 |
| 短序列 INT8 约简 | 非块 W8A8、未融合激活且 K<=1024 采用 INT32 点积累加。最大绝对和 2^24 精确转为 FP32。更大 K、分块量化、FP8 和融合 SiLU 路径保留原有调度。 |

### 环境变量开关

`FLAG_GEMS_W8A8_QUANT_FUSION=quant|quant_oai` 为可选项，在 capture 前读取，默认为关闭。请在导入包前设置：

```bash
export FLAG_GEMS_W8A8_QUANT_FUSION=quant
```

所有新生产路径均使用 Triton 与元数据/空分配实现，严格遵循无 PyTorch 计算回退（NO_TORCH_COMPUTE_FALLBACK）规则。新 API 均为前向推理专用，不提供 autograd 支持。

## 功能验证

**硬件平台: NVIDIA H100**

- **214 个针对性数值测试全部通过**：包含 36 组原生 CUDA 缓存写入对比（BF16/E4M3、NHD/HND 布局下精确比对输出字节），12 组完整专家 OAI 测试用例（覆盖融合/未融合量化及 K1024 累加边界），以及可变缓冲区、空输入、动态 CUDA Graph 边界、跨步输入等测试。
- **3 个 fullgraph 编译边界测试全部通过**。
- **6 个既有 W8A8 专家回归测试全部通过**。

## 综合性能评测

基于 CUDA Graph 测时（25 次预热，100 次捕获调用，5 次事件采样，2 轮反序测试取中位数均值）：

| 算子与形状 | 基线 (μs) | 候选实现 (μs) | 加速比 |
|---|---:|---:|---:|
| `fp32_router_gemm` [1, 6144, 128] | 7.263 | 2.271 | **3.199×** |
| `fp32_router_gemm` [4, 6144, 128] | 14.172 | 3.489 | **4.061×** |
| `fp32_router_gemm` [8, 6144, 128] | 15.760 | 3.945 | **3.995×** |
| `fp32_router_gemm` [16, 6144, 128] | 16.264 | 5.068 | **3.209×** |
| `fp32_router_gemm` [32, 6144, 128] | 15.375 | 6.997 | **2.197×** |
| `per_token_int8_quant` [1, 6144] | 18.380 | 5.426 | **3.387×** |
| `per_token_int8_quant` [64, 6144] | 33.058 | 5.558 | **5.948×** |
| `per_token_int8_quant` [4096, 6144] | 446.016 | 60.878 | **7.326×** |
| `per_token_int8_quant` [5089, 768] | 66.883 | 12.298 | **5.438×** |
| `per_token_int8_quant` [8192, 768] | 107.506 | 18.474 | **5.819×** |
| `vit_attention` [1, 128, 16, 64] | 6.599 | 5.196 | **1.270×** |
| `vit_attention` [1, 512, 16, 80] | 12.282 | 14.998 | 0.819× |
| `vit_attention` [2, 256, 16, 80] | 9.345 | 10.783 | 0.867× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [1, 4, 2, 2, 128] | 3.042 | 3.107 | 0.979× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [4, 4, 2, 2, 128] | 3.224 | 3.175 | 1.016× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [64, 4, 2, 2, 128] | 3.573 | 3.410 | 1.048× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [4096, 4, 2, 2, 128] | 21.257 | 35.222 | 0.604× |
| `fused_minimax_m3_qknorm_rope_kv_insert` [8192, 4, 2, 2, 128] | 45.699 | 72.405 | 0.631× |

## 说明与边界

- Router 结果对比基线为 PyTorch `F.linear`（未启用 TF32）。
- 大批次 prefill 下缓存写入器仍慢于原生专用 CUDA 实现。
- ViT attention 提供免 C++ 拓展且 CUDA Graph 安全的适配，D80 形状下针对性优化的 SDPA 更快。

## 相关链接

- [算子列表](operators.md)
- [项目结构](../contribution/project-structure.md)
- [基准测试结果](../performance/results.md)
