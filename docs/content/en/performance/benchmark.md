---
title: Running Benchmarks
weight: 10
---

# Running Benchmarks

FlagGems-vllm includes a comprehensive benchmark suite for performance validation.

## Quick Start

Collect all benchmarks:

```bash
cd FlagGems-vllm
PYTHONPATH=src pytest -q benchmark --collect-only
```

Run a fast smoke test:

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level core --iter 1 --warmup 1
```

## Benchmark Options

Benchmarks are controlled via pytest options (defined in `benchmark/conftest.py`):

### Shape Coverage

- `--level core` (default): Core shapes from `benchmark/core_shapes.yaml`
- `--level comprehensive`: Extended shape coverage

### Iteration Control

- `--warmup N`: Warmup iterations (default: 10)
- `--iter N`: Measurement iterations (default: 100)

### Data Type Selection

- `--dtypes`: Comma-separated list (e.g., `float16,float32,bfloat16`)

### Measurement Mode

- `--mode kernel`: Measure kernel time only
- `--mode operator`: Measure operator call overhead
- `--mode wrapper`: Measure full wrapper overhead

### Metrics

- `--metrics latency,throughput,memory`: Select metrics to report

### Multi-GPU

- `--parallel N`: Run benchmarks across N GPUs in parallel

### Output

- `--record`: Enable result recording
- `--output FILE`: Output JSON file (default: `benchmark_result.json`)

## Example Commands

### Core shapes, quick validation

```bash
PYTHONPATH=src pytest -q benchmark/test_moe_align_block_size_triton.py \
    --level core --iter 10 --warmup 5
```

### Comprehensive shapes, production benchmarking

```bash
PYTHONPATH=src pytest -q benchmark/test_grouped_topk.py \
    --level comprehensive --iter 100 --warmup 20 \
    --record --output grouped_topk_perf.json
```

### Specific dtypes

```bash
PYTHONPATH=src pytest -q benchmark/test_flash_attention_forward.py \
    --level core --dtypes float16,bfloat16 --iter 50
```

### Multi-GPU benchmarking

```bash
PYTHONPATH=src pytest -q benchmark/test_fused_experts_impl.py \
    --level core --parallel 4
```

## Benchmark Structure

Benchmark files follow the pattern `benchmark/test_<op>.py` and typically include:

```python
import pytest
from benchmark.conftest import Config

@pytest.mark.parametrize("shape", Config.get_shapes("moe_align_block_size"))
@pytest.mark.parametrize("dtype", Config.dtypes())
def test_moe_align_block_size_perf(benchmark, shape, dtype):
    # Setup
    inputs = create_inputs(shape, dtype)

    # Benchmark
    result = benchmark(lambda: flaggems_vllm.moe_align_block_size(*inputs))

    # Validation
    assert result is not None
```

## Shape Configuration

Core shapes are defined in `benchmark/core_shapes.yaml`:

```yaml
moe_align_block_size:
  - [128, 2, 32, 16]   # [num_tokens, topk, block_size, num_experts]
  - [256, 4, 64, 32]
  - [1024, 8, 128, 64]

grouped_topk:
  - [8, 16, 4, 2, 2]   # [batch, num_experts, n_group, topk_group, topk]
  - [32, 32, 8, 4, 4]
```

## SpeedUp Calculation

Benchmarks compute speedup relative to a torch baseline:

```
SpeedUp = latency_torch_baseline / latency_flaggems_vllm
```

- **SpeedUp > 1.0**: FlagGems-vllm is faster
- **SpeedUp < 1.0**: Torch baseline is faster
- **Target**: SpeedUp ≥ 0.9 for core shapes

## Result Format

JSON output (`--record --output results.json`):

```json
{
  "operator": "moe_align_block_size",
  "backend": "nvidia",
  "device": "NVIDIA H100",
  "results": [
    {
      "shape": [128, 2, 32, 16],
      "dtype": "float16",
      "latency_ms": 0.045,
      "throughput_tokens_per_sec": 2844444,
      "speedup": 1.23
    }
  ]
}
```

## CI Integration

Benchmarks run in CI for changed operators:

- `tools/select_tests.py` maps changed files to benchmark targets
- Runs with `--level core --iter 10 --warmup 5` for speed
- Results are archived as artifacts

## Next Steps

- [View benchmark results](results/)
- [Contribute benchmarks](/FlagGems-vllm/contribution/overview/)
