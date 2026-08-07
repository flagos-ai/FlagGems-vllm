[English|[中文版](./README_cn.md)]

## Introduction

FlagGems-vllm is part of [FlagOS](https://flagos.io/).
FlagGems-vllm is a high-performance operator library designed for multiple hardware backends. It provides optimized implementations of common vLLM operators and supports high-performance inference and deployment for a variety of widely used models.

FlagGems-vllm is a high-performance deep learning operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common vllm operators (moe_align_block_size, etc.)

## Relationship with FlagGems and vllm-plugin-fl

The three repositories are used together but have different responsibilities:

- **FlagGems**: the general-purpose FlagGems operator library. It provides common PyTorch/Triton operator replacements and exposes `flag_gems.enable()` / `flag_gems.use_gems()` to register operators into PyTorch dispatch.
- **FlagGems-vllm**: this repository. It contains vLLM-scenario operator implementations and tests/benchmarks that are aligned with the corresponding FlagGems implementations where the same operator exists. It exposes operators through the `flaggems_vllm` Python package, for example `flaggems_vllm.grouped_topk`, `flaggems_vllm.fused_experts_impl`, and `flaggems_vllm.moe_align_block_size`.
- **vllm-plugin-fl**: the vLLM plugin layer. It uses FlagGems as the global operator backend by importing FlagGems and calling `flag_gems.enable()`. For vLLM-specific fused kernels that are not enabled through PyTorch dispatch, it explicitly imports and calls operators from FlagGems-vllm.

In a typical vLLM plugin environment, the call flow is:

```text
vLLM
	-> vllm-plugin-fl
			-> flag_gems.enable() for general FlagGems operator registration
			-> flaggems_vllm.<operator>() for vLLM-specific fused operators
```

This means `FlagGems` and `FlagGems-vllm` are complementary: `FlagGems` provides the common operator backend, while `FlagGems-vllm` provides vLLM-oriented kernels and compatibility tests/benchmarks used by `vllm-plugin-fl`.

## Quick Installation
### Install Dependencies
```shell
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
```
### Install FlagGems-vllm
```shell
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install  .
```

For development, use editable installation:

```shell
pip install --no-build-isolation -e .
```

If you want to run tests, install the test dependencies as well:

```shell
pip install -e '.[test]'
```

### Install with FlagGems and vllm-plugin-fl

When validating the full plugin stack, install the repositories in this order:

```shell
# 1. Install FlagGems
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems
pip install --no-build-isolation -e .

# 2. Install FlagGems-vllm
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm
pip install --no-build-isolation -e .

# 3. Install vllm-plugin-fl
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl
pip install --no-build-isolation -e .
```

If multiple vLLM plugins are installed, select the FlagOS plugin explicitly:

```shell
export VLLM_PLUGINS=fl
```

## Usage Example

```python
import torch
import flaggems_vllm

# Prepare a simple topk_ids tensor for MoE routing
num_tokens = 128
topk = 2
num_experts = 16
block_size = 32

topk_ids = torch.randint(
	low=0,
	high=num_experts,
	size=(num_tokens, topk),
	device='cuda',
	dtype=torch.int32,
)

# Align tokens by expert and block size
sorted_ids, expert_ids, num_tokens_post_pad = flaggems_vllm.ops.moe_align_block_size(
	topk_ids=topk_ids,
	block_size=block_size,
	num_experts=num_experts,
)

print(sorted_ids.shape, expert_ids.shape, num_tokens_post_pad)
```

### Usage with FlagGems enabled

`vllm-plugin-fl` enables the general FlagGems backend and can also call FlagGems-vllm operators directly. A minimal smoke test looks like this:

```python
import torch
import flag_gems
import flaggems_vllm

flag_gems.enable()

scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
bias = torch.randn((16,), device="cuda", dtype=torch.float32)

topk_weights, topk_ids = flaggems_vllm.grouped_topk(
	scores,
	n_group=4,
	topk_group=2,
	topk=2,
	renormalize=True,
	routed_scaling_factor=1.0,
	bias=bias,
	scoring_func=0,
)

print(topk_weights.shape, topk_ids.shape)
```

## Tests and Benchmark Quick Start

The following commands can be used for quick validation after installation.

Most tests and benchmarks require a supported accelerator runtime, its vendor
PyTorch build, a compatible Triton implementation, and vLLM dependencies.

### Import smoke tests

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src python - <<'PY'
import torch
import flaggems_vllm
from flaggems_vllm import runtime

print('torch:', torch.__version__)
print('vendor:', flaggems_vllm.vendor_name)
print('flaggems_vllm device:', flaggems_vllm.device)
print('device count:', runtime.device.device_count)
print('grouped_topk:', callable(flaggems_vllm.grouped_topk))
PY
```

### Run tests

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests --collect-only
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests --quick
```

Run a focused operator test:

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests/test_grouped_topk.py
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q tests/test_fused_inv_rope_fp8_quant.py --quick
```

### Run benchmark

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark --collect-only
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark/test_moe_align_block_size_triton.py::test_moe_align_block_size_triton --level core --iter 1 --warmup 1
```

Run focused benchmarks for vLLM-specific operators:

```shell
cd /workspace/FlagGems-vllm
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark/test_grouped_topk.py --level core --iter 1 --warmup 1
PYTHONPATH=/workspace/FlagGems-vllm/src pytest -q benchmark/test_fused_inv_rope_fp8_quant.py --level core --iter 1 --warmup 1
```

### Validate with vllm-plugin-fl

After installing `FlagGems`, `FlagGems-vllm`, and `vllm-plugin-fl`, validate that the plugin imports and that FlagGems-vllm operators are available:

```shell
export VLLM_PLUGINS=fl
python - <<'PY'
import flag_gems
import flaggems_vllm

flag_gems.enable()
print('FlagGems enabled')
print('FlagGems-vllm grouped_topk available:', callable(flaggems_vllm.grouped_topk))
PY
```

If a model and vLLM runtime are available, run a small vLLM offline inference from the `vllm-plugin-fl` repository examples, for example `examples/offline_inference.py`.

### Notes

- `--collect-only` is recommended first to quickly check import and test discovery.
- Use `--quick` for fast functional validation when supported by the test.
- Use `--level core --iter 1 --warmup 1` for fast benchmark smoke tests.
- Full benchmark runs can take a long time and should be reserved for performance validation.

This project is licensed under the [Apache (version 2.0) License](./LICENSE).
