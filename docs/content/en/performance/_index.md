---
title: Performance
weight: 30
bookCollapseSection: true
---

# Performance

Learn how to benchmark FlagGems-vllm operators and review performance results.

## Overview

FlagGems-vllm operators are optimized for vLLM inference scenarios with:
- Pre-tuned autoconfigs for NVIDIA Ampere/Hopper architectures
- Multi-backend support (NVIDIA, AMD, Ascend, and 15+ more)
- Operator-specific shape optimizations from vLLM workloads

## Topics

- [Running Benchmarks](benchmark/) - How to run performance benchmarks
- [Benchmark Results](results/) - Performance data and speedup metrics

## Quick Benchmark

Run a fast smoke benchmark:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py \
    --level core --iter 1 --warmup 1
```

Run comprehensive benchmarks for an operator:

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level comprehensive --iter 100 --warmup 10
```

## Performance Metrics

Benchmarks report:
- **Latency** (ms): Time per operation
- **Throughput** (tokens/s or GB/s): Processing rate
- **SpeedUp**: `latency_torch_baseline / latency_flaggems_vllm`

Acceptance target: **SpeedUp ≥ 0.9** for core shapes.

## Next Steps

- [Benchmark guide](benchmark/)
- [View results](results/)
