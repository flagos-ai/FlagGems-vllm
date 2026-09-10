---
title: Changelog
weight: 30
---

# Changelog

Version history and release notes for FlagGems-vllm.

## v0.1.0 (Current)

Initial release of FlagGems-vllm as a standalone vLLM operator library.

### Features

- **102 operators** optimized for vLLM inference scenarios
- **18 backend vendors** supported (NVIDIA, AMD, Ascend, Hygon, Iluvatar, and more)
- **MoE operators**: `moe_align_block_size`, `grouped_topk`, `fused_experts_impl`, `moe_sum`
- **Flash Attention**: `flash_attention_forward`, `flash_attn_varlen_func`, `flash_mla`, `triton_unified_attention`
- **Quantization**: FP8, INT8, FP4 operators for activation and weight quantization
- **Fused kernels**: RMS norm, RoPE, attention, activation functions
- **Model-specific operators**: Qwen4, DeepSeek-v4, RWKV

### Architecture

- Triton-based operator implementations with FlagTree compiler support
- Pre-tuned autotune configs for NVIDIA Ampere/Hopper architectures
- C++ extension support via pybind11 (optional)
- PyTorch dispatch integration via `enable()` / `use_gems()` API

### Testing & CI

- Functional test suite with `--quick` mode for fast validation
- Performance benchmark suite with core shapes from `benchmark/core_shapes.yaml`
- Multi-backend CI via GitHub Actions (H20 runner for NVIDIA)
- Automatic test selection via `tools/select_tests.py`

### Documentation

- Hugo-based documentation site (this site)
- Operator development protocol (`workflow.md`)
- Bilingual support (English/Chinese)

### Known Issues

- Some operators are in alpha/beta stage (see [operator list](/FlagGems-vllm/references/operators/))
- Non-NVIDIA backends have less mature autotune configs
- C++ extension build requires manual `USE_C_EXTENSION=1` flag

## Upcoming

### v0.2.0 (Planned)

- Additional MoE optimizations
- Expanded Flash Attention variants
- Improved multi-backend autotune coverage
- Performance database integration
- Enhanced debugging tools

## Contributing

See the [contribution guide](/FlagGems-vllm/contribution/) for how to contribute to FlagGems-vllm.

## Version Scheme

FlagGems-vllm follows semantic versioning:
- **Major** (x.0.0): Breaking API changes
- **Minor** (0.x.0): New operators, features, backward-compatible changes
- **Patch** (0.0.x): Bug fixes, performance improvements

## Release Coordination

FlagGems-vllm releases are coordinated with:
- **FlagGems**: General operator library (upstream dependency)
- **vllm-plugin-fl**: vLLM plugin layer (downstream consumer)

Compatible versions are documented in each repository's README.
