---
title: 运行基准测试
weight: 10
---

# 运行性能基准测试

FlagGems-vllm 包含一套完善的 Benchmark 性能评测套件，用于验证算子在不同形状、数据类型和硬件平台下的执行延迟与吞吐。

## 快速上手

### 发现所有基准测试用例

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q benchmark --collect-only
```

### 快速冒烟基准测试

使用最小迭代次数与预热轮次快速检查链路：

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level core --iter 1 --warmup 1
```

针对特定算子进行聚焦性能评测：

```bash
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py::test_moe_align_block_size_triton \
    --level core --iter 1 --warmup 1
PYTHONPATH=src pytest -q benchmark/test_fused_inv_rope_fp8_quant.py \
    --level core --iter 1 --warmup 1
```

## 基准测试常用选项

基准测试选项在 `benchmark/conftest.py` 中统一定义：

### 形状覆盖层级

- `--level core` (默认): 运行 `benchmark/core_shapes.yaml` 中定义的核心负载形状
- `--level comprehensive`: 扩展形状覆盖，包含更大规模的长短序列

### 迭代与预热控制

- `--warmup N`: 预热轮次（默认: 10）
- `--iter N`: 正式测时轮次（默认: 100）

### 数据类型筛选

- `--dtypes`: 逗号分隔的类型列表（如 `float16,bfloat16,float32`）

### 测时模式

- `--mode kernel`: 仅测内核执行耗时
- `--mode operator`: 测量算子调用全流程开销（含参数校验和 host 调度）

### 结果记录与保存

- `--record`: 开启测试结果记录
- `--output FILE`: 保存为 JSON 文件（默认: `benchmark_result.json`）

## 常用命令示例

### 核心形状常规测时

```bash
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py \
    --level core --iter 10 --warmup 5
```

### 生产级完整基准评测并导出结果

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level comprehensive --iter 100 --warmup 20 \
    --record --output grouped_topk_perf.json
```

### 指定数据类型运行

```bash
PYTHONPATH=src pytest -q benchmark/test_flash_attention_forward.py \
    --level core --dtypes float16,bfloat16 --iter 50
```

## 核心形状配置 (`benchmark/core_shapes.yaml`)

核心形状直接取材于 vLLM 线上主流模型（如 LLaMA, Mixtral, DeepSeek 等）：

```yaml
moe_align_block_size:
  - [128, 2, 32, 16]   # [num_tokens, topk, block_size, num_experts]
  - [256, 4, 64, 32]
  - [1024, 8, 128, 64]

grouped_topk:
  - [8, 16, 4, 2, 2]   # [batch, num_experts, n_group, topk_group, topk]
  - [32, 32, 8, 4, 4]
```

## 加速比计算标准

加速比计算公式：

```text
SpeedUp = 基线耗时 (latency_torch_baseline) / FlagGems-vllm 算子耗时 (latency_flaggems_vllm)
```

- **SpeedUp > 1.0**: FlagGems-vllm 性能优于基线
- **SpeedUp = 1.0**: 性能持平
- **SpeedUp < 1.0**: 原生基线更快（需持续调优）
- **准入基线**: 核心形状下 **SpeedUp ≥ 0.9**

## 下一步

- [查看基准测试结果](results.md)
- [MiniMax M3 算子支持评测](../references/minimax-m3-support.md)
- [贡献基准测试代码](../contribution/overview.md)
