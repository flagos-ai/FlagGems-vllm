---
title: Getting Started
weight: 10
bookCollapseSection: true
---

# Getting Started with FlagGems-vllm

This section covers the environment prerequisites and installation options for FlagGems-vllm.

## Prerequisites

- **Python**: Python ≥ 3.8 (CI uses 3.11/3.12)
- **PyTorch**: PyTorch ≥ 2.6.0
- **FlagGems**: Required dependency; install via `pip install flag-gems==5.4.0` (some FlagGems-vllm operators invoke general FlagGems operators internally)
- **Compiler**: Triton (≥ 3.0) or FlagTree compiler
- **Hardware**: CUDA-capable GPU (NVIDIA) or other supported accelerators (Ascend, AMD, Cambricon, Hygon, etc.)
- **vLLM**: Optional, for end-to-end inference and plugin integration

## Installation

FlagGems-vllm supports source installation, editable development installation, automated bootstrapping scripts, and full-stack integration with FlagGems and vllm-plugin-fl.

See the [Installation Guide](install.md) for detailed instructions.

## Next Steps

- [Installation Guide](install.md) - Environment setup and installation options
- [Usage Guide](../usage/basic.md) - Operator invocation, global dispatch, and APIs
- [Integration with vllm-plugin-fl](../usage/vllm-plugin.md) - vLLM plugin orchestration
