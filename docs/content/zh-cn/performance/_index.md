---
title: 性能
weight: 30
bookCollapseSection: true
---

# 性能

了解如何对 FlagGems-vllm 算子进行基准测试并查看性能结果。

## 概述

FlagGems-vllm 算子针对 vLLM 推理场景优化:
- NVIDIA Ampere/Hopper 架构的预调优配置
- 多后端支持(NVIDIA、AMD、昇腾等 15+ 个后端)
- 针对 vLLM 工作负载的算子专用形状优化

## 主题

- [运行基准测试](benchmark/) - 如何运行性能基准测试
- [基准测试结果](results/) - 性能数据和加速指标

## 快速基准测试

运行快速冒烟基准测试:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py \
    --level core --iter 1 --warmup 1
```

运行算子的全面基准测试:

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level comprehensive --iter 100 --warmup 10
```

## 性能指标

基准测试报告:
- **延迟**(ms): 每次操作的时间
- **吞吐量**(tokens/s 或 GB/s): 处理速率
- **加速比**: `latency_torch_baseline / latency_flaggems_vllm`

验收目标: 核心形状的**加速比 ≥ 0.9**。

## 下一步

- [基准测试指南](benchmark/)
- [查看结果](results/)
