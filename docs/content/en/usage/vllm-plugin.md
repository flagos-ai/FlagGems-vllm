---
title: vLLM Plugin Integration
weight: 20
---

# Integration with vllm-plugin-fl

FlagGems-vllm is designed to work seamlessly with [vllm-plugin-fl](https://github.com/flagos-ai/vllm-plugin-fl), the vLLM plugin layer for FlagOS.

## Architecture

The three-layer architecture:

```
vLLM (inference engine)
    ↓
vllm-plugin-fl (plugin layer)
    ↓
    ├─→ flag_gems.enable()        (general PyTorch ops)
    └─→ flaggems_vllm.<op>()      (vLLM-specific fused kernels)
```

**FlagGems** provides general operator replacements registered into PyTorch dispatch.
**FlagGems-vllm** provides vLLM-specific fused operators called explicitly by the plugin.
**vllm-plugin-fl** orchestrates both layers.

## Installation

Install all three components in order:

```bash
# 1. FlagGems (general operators)
git clone https://github.com/flagos-ai/FlagGems.git
cd FlagGems && pip install --no-build-isolation -e . && cd ..

# 2. FlagGems-vllm (vLLM operators)
git clone https://github.com/flagos-ai/FlagGems-vllm.git
cd FlagGems-vllm && pip install --no-build-isolation -e . && cd ..

# 3. vllm-plugin-fl (plugin layer)
git clone https://github.com/flagos-ai/vllm-plugin-fl.git
cd vllm-plugin-fl && pip install --no-build-isolation -e .
```

## Activating the Plugin

If multiple vLLM plugins are installed, select FlagOS:

```bash
export VLLM_PLUGINS=fl
```

The plugin is automatically activated when vLLM initializes.

## Usage Example

Once the plugin is installed, vLLM automatically uses FlagGems and FlagGems-vllm operators:

```python
from vllm import LLM, SamplingParams

# The plugin is activated automatically
llm = LLM(model="meta-llama/Llama-2-7b-hf")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

FlagGems operators handle general PyTorch operations, while FlagGems-vllm operators accelerate:
- MoE routing and expert dispatch
- Flash Attention variants
- Quantization (FP8, INT8)
- Fused normalization + RoPE
- Model-specific kernels

## Verifying Plugin Usage

Check that FlagGems-vllm operators are being called:

```python
import os
os.environ["VLLM_PLUGINS"] = "fl"

from vllm import LLM
import flaggems_vllm

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

llm = LLM(model="meta-llama/Llama-2-7b-hf")
# Check logs for FlagGems-vllm operator invocations
```

## Backend Selection

The plugin respects the `DNN_VENDOR` environment variable:

```bash
# NVIDIA (default)
export DNN_VENDOR=nvidia

# AMD
export DNN_VENDOR=amd

# Huawei Ascend
export DNN_VENDOR=ascend
```

## Performance

vllm-plugin-fl leverages both FlagGems and FlagGems-vllm for performance:

- **General ops** (matmul, layernorm, softmax): Handled by FlagGems
- **vLLM-specific fusions**: Handled by FlagGems-vllm (grouped_topk, moe_align_block_size, flash_attention_forward, etc.)

Expected speedups vary by model and hardware. See [performance benchmarks](/FlagGems-vllm/performance/) for details.

## Troubleshooting

### Plugin Not Loading

Check that `VLLM_PLUGINS` is set:

```bash
echo $VLLM_PLUGINS  # Should output: fl
```

Verify the plugin is installed:

```bash
pip show vllm-plugin-fl
```

### Operator Fallback

If FlagGems-vllm operators are not being used, check:

1. FlagGems-vllm is installed: `python -c "import flaggems_vllm; print(flaggems_vllm.__version__)"`
2. Backend matches your hardware: `python -c "import flaggems_vllm; print(flaggems_vllm.vendor_name)"`
3. No conflicting environment variables

### Performance Issues

If performance is slower than expected:

1. Verify autotune configs are loaded (NVIDIA backend has pre-tuned configs)
2. Check that Triton/FlagTree is correctly installed
3. Run benchmarks to isolate the issue: `pytest benchmark/test_<op>.py --level core`

## Next Steps

- [Selective operator enablement](../selective/)
- [Non-NVIDIA backend configuration](../non-nvidia/)
- [Performance benchmarking](/FlagGems-vllm/performance/benchmark/)
