[English|[中文版](./README_cn.md)]

## Introduction

FlagGems-vllm is part of [FlagOS](https://flagos.io/). It is a high-performance operator library implemented using the [Triton programming language](https://github.com/openai/triton) launched by OpenAI, designed for multiple hardware backends. It provides optimized implementations of common vLLM operators and supports high-performance inference and deployment for a variety of widely used models.

## Features

- Operators have undergone deep performance tuning
- Triton kernel call optimization
- Flexible multi-backend support mechanism
- Support for common vLLM operators (`moe_align_block_size`, `grouped_topk`, etc.)

## Relationship with FlagGems and vllm-plugin-fl

The three repositories work together in the FlagOS ecosystem but have different responsibilities:

- **FlagGems**: The general-purpose FlagGems operator library. It provides common PyTorch/Triton operator replacements and exposes `flag_gems.enable()` / `flag_gems.use_gems()` to register operators into PyTorch dispatch. **FlagGems-vllm depends on FlagGems**, and currently some FlagGems-vllm operators invoke operators from FlagGems internally.
- **FlagGems-vllm**: This repository. It contains vLLM-scenario operator implementations and tests/benchmarks that are aligned with the corresponding FlagGems implementations where the same operator exists. It exposes operators through the `flaggems_vllm` Python package, for example `flaggems_vllm.grouped_topk`, `flaggems_vllm.fused_experts_impl`, and `flaggems_vllm.moe_align_block_size`.
- **vllm-plugin-fl**: The vLLM plugin layer. It uses FlagGems as the global operator backend by importing FlagGems and calling `flag_gems.enable()`. For vLLM-specific fused kernels that are not enabled through PyTorch dispatch, it explicitly imports and calls operators from FlagGems-vllm.

In a typical vLLM plugin environment, the call flow is:

```text
vLLM
	-> vllm-plugin-fl
			-> flag_gems.enable() for general FlagGems operator registration
			-> flaggems_vllm.<operator>() for vLLM-specific fused operators (some calling flag_gems operators internally)
```

This means `FlagGems-vllm` builds on top of `FlagGems`: `FlagGems` provides the common operator foundation and internal utility operators, while `FlagGems-vllm` provides vLLM-oriented kernels and compatibility tests/benchmarks used by `vllm-plugin-fl`.

## Quick Start

### Quick Installation

```shell
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake
pip install --no-build-isolation -e .
```

For complete installation steps (including test dependencies, full environment bootstrapping via `tools/setup.sh`, and multi-repository setup with FlagGems and vllm-plugin-fl), please refer to the [Installation Guide](docs/content/en/getting-started/install.md).

### Quick Usage

Since FlagGems-vllm depends on FlagGems, remember to enable FlagGems before using FlagGems-vllm operators:

```python
import torch
import flag_gems
import flaggems_vllm

# Enable FlagGems general operators
flag_gems.enable()

# Call FlagGems-vllm fused operators
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

For more examples (such as `moe_align_block_size`, `flash_attention_forward`, and `fused_add_rms_norm`), see the [Basic Usage Guide](docs/content/en/usage/basic.md).

## Documentation

Comprehensive documentation is available in the [docs](docs/) directory:

- 📖 **Getting Started**: [Installation Guide](docs/content/en/getting-started/install.md)
- 🚀 **Usage**: [Basic Usage](docs/content/en/usage/basic.md) | [vLLM Plugin Integration](docs/content/en/usage/vllm-plugin.md)
- 🔧 **Hardware Backends**: [Non-NVIDIA Hardware Support](docs/content/en/usage/non-nvidia.md)
- 🧪 **Testing**: [Unit Testing Guide](docs/content/en/testing/unittest.md) | [Coverage](docs/content/en/testing/coverage.md)
- ⚡ **Benchmarking**: [Running Benchmarks](docs/content/en/performance/benchmark.md) | [Performance Results](docs/content/en/performance/results.md)
- 📋 **References**: [Supported Operators](docs/content/en/references/operators.md) | [MiniMax M3 Support](docs/content/en/references/minimax-m3-support.md)
- 🤝 **Contributing**: [Contribution Overview](docs/content/en/contribution/overview.md) | [Development Workflow](docs/content/en/contribution/workflow.md) | [Project Structure](docs/content/en/contribution/project-structure.md)

## License

This project is licensed under the [Apache (version 2.0) License](./LICENSE).

### Explicit Hy4 inference APIs

Use one combined `flaggems_vllm` revision for the Hy4 numerical API set. It
includes the RMSNorm/INT8 foundation; install the matching fixed FlagGems Core
dependency before using these APIs. This revision declares the public fixed
Core commit `36d8e98ba627d52963f1fd0d56567eabd3b79c53` in its package metadata;
ordinary installation resolves it automatically. If installing with `--no-deps`,
install the same Core revision explicitly first.

| API | Contract |
|---|---|
| `rms_scaled_int8_quant` | BF16 RMSNorm, INT8 data and FP32 row scales; retain both BF16 boundaries. |
| `int8_indexer_quantize`, `int8_indexer_cache_store`, `int8_indexer_paged_logits` | Identity exports of Core APIs; callers own mapping and cache lifetime. |
| `int8_indexer_paged_logits_empty_tile` | Packed D128 paged INT8 scores; invalid tiles become negative infinity and all output lanes are written. Width is caller-proven capacity. |
| `indexer_prefix_topk` | Stable descending FP32 TopK with GPU prefix lengths, int32 indices and -1 for nonfinite selections. Width up to 204800 uses 32-bit packing above 65535. |
| `can_ihc_pair`, `ihc_pre_pair`, `ihc_post_pre_pair` | SM90 paired-token weight reuse at rows 208/256/1024, hidden 6144 and HC4; preserve the BF16 residual boundary. |
| `can_ihc_post_pre`, `ihc_post_pre` | Metadata predicate and existing post-to-pre iHC contract. |
| `pad_mla_q_sink_slot_bound` | Raw-bit query/sink padding and final nonnegative TopK slot plus one; interior holes retain their positions. |
| `int8_quantize_per_token` | The selected per-token INT8 scale/division contract for MoE input and output quantization, isolated from other modes. |
| `silu_clamp`, `silu_clamp_int8_quant` | Clamped SiLU, optionally with INT8 data/scales, preserving the BF16/FP16 activation workspace. |
| `fused_mm_input_norm` | FP32 per-channel affine normalization of contiguous image patches into a caller-owned output. |
| `mul_scaled_int8_quant`, `int8_scaled_mm_decode`, `concat_pad_mla_q_and_sink`, `mrope_strided_cache` | Existing gate quantization, projection, query transport and rotary contracts. |

The standard backend registry selects NVIDIA implementations. Explicit fused
APIs stay outside ATen registration; unsupported metadata rejects before launch,
and execution failures propagate without retries. `fused_moe` retains the
existing `FLAG_GEMS_W8A8_QUANT_FUSION` setting: `quant_silu` selects the clamped
SiLU path and its two per-token quantizers; other modes preserve current-main quantization and its wide-row/NaN
handling. Request/token/slot mapping and staging remain framework responsibilities.

The existing `FLAG_GEMS_GROUPED_TOPK_SCORE_FUSION=1` setting fuses score
processing into group selection. Setting
`FLAG_GEMS_GROUPED_TOPK_SINGLE_GROUP_FUSION=1` as well enables the selected
single-group route. Both default off and are resolved before graph warmup;
current-main small-expert paths keep their precedence.

Fixed radix, paired-iHC, paged-dot and transport launch schedules preserve the
selected numerical order and are exempt from autotuning during this migration.
Retuning requires separate correctness and performance qualification. Callers
own persistent buffers, graph lifetimes, independent outputs and stream ordering.
Matching functional and `benchmark/test_<api>_perf.py` tests require the supported
CUDA runtime. CPU checks do not qualify integrated model quality or throughput.
