---
title: Overview
weight: 10
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Overview

In pull requests, contributors should describe what changed and why. Please also provide test cases and benchmark results if applicable. Pull requests require approval from project maintainers and must pass continuous integration checks before merging.

## 1. Pull Request Workflow {#pr-workflow}

1. **Create a fork and feature branch** from `main`;
2. **Follow development guidelines**: Ensure all protocol requirements and coding standards are met;
3. **Format code and run local tests**:
   ```bash
   pre-commit run --all-files
   PYTHONPATH=src pytest -q tests --quick
   PYTHONPATH=src pytest -q benchmark/test_<op>.py --level core --iter 1
   ```
4. **Submit a pull request**:
   - Provide a clear description and link to relevant issues;
   - Include test results (accuracy and performance SpeedUp);
   - Ensure all CI gates pass (code style, `operators.yaml` validation, unit tests, benchmarks).

## 2. Device Isolation and Visibility {#device-visibility}

On a node shared by multiple users or GPUs, you usually want a test or benchmark run to use only a subset of the available devices. FlagGems-vllm relies on each backend vendor's visibility environment variable to do the isolation. Visibility variables must be set **before** the process starts.

For the list of visibility environment variables and details across different hardware backends, please refer to the FlagGems documentation: [Device Isolation and Visibility](https://flagos-ai.github.io/FlagGems/en/contribution/device-visibility/) (or Chinese: [设备隔离与可见性](https://flagos-ai.github.io/FlagGems/zh-cn/contribution/device-visibility/)).

## 3. Operator Naming Conventions {#operator-naming-conventions}

FlagGems-vllm specifically hosts specialized and fused operators for the vLLM repository and **does not accept general PyTorch operators** (general PyTorch operators should be contributed to [FlagGems](https://github.com/flagos-ai/FlagGems)). Therefore, operator names align directly with vLLM operators or model layer names, **without needing to align with PyTorch**:

- Use `snake_case` naming style (e.g., `grouped_topk`, `moe_align_block_size`, `fused_inv_rope_fp8_quant`);
- Keep names consistent with vLLM operator APIs or model layer naming;
- Operator names cannot start with an underscore (as operator names correspond directly to custom pytest markers, which cannot begin with an underscore due to pytest marker rules);
- Operator names must not contain dots;
- Functional test and benchmark filenames must strictly match the operator name (`tests/test_<op>.py`, `benchmark/test_<op>.py`).

## 4. Operator Metadata Management {#operator-metadata-management}

FlagGems-vllm maintains an operator inventory in `conf/operators.yaml`. CI validates this file for schema compliance, field completeness, absence of duplicates, and case-insensitive alphabetical sorting (`id.casefold()`). Each operator record has the following fields:

| Field | Type | Purpose | Constraints |
| --- | --- | --- | --- |
| `id` | string | Uniquely identifies an operator in FlagGems-vllm | Matches the operator API name, e.g., `grouped_topk`; snake_case, cannot start with underscore |
| `description` | string | Description of the operator purpose, algorithm, and inference role (supports multi-line YAML block) | Clearly explains function in vLLM architecture, supported precisions, etc. |
| `for` | array of string | Marks the external / vLLM operation or function this operator replaces; use `None` if not applicable | e.g., `_moe_C.grouped_topk`, `_C.apply_repetition_penalties_`; no need to align with PyTorch operators |
| `labels` | array of string | Categorizes operators along multiple dimensions (non-empty list) | Common labels: `vLLM`, `fused`, `MoE`, `Attention`, `NoCPU`, `KernelGen` |
| `kind` | array of string or string | Major category of the operator | e.g., `MoE`, `NeuralNetwork`, `Quantization`, `Normalization` |
| `stages` | array of key-value pairs | Records the maturity evolution history (non-empty list) | Keys are stage names (`alpha`, `beta`, `stable`), values are versions (e.g., `'5.0'`) |

Maturity stages are defined as follows:
- New or ported fused operators typically start at the `alpha` or `beta` stage;
- When an operator is thoroughly verified, integrated with autotuning, and continuously passing full CI and real-world workloads, it is promoted to `stable`.

Example entry (`conf/operators.yaml`):

```yaml
  - id: grouped_topk
    description: |
      A specialized routing mechanism used in Mixture-of-Experts (MoE) models (like DeepSeek-V3/R1)
      to select top-k experts by first grouping them, rather than selecting globally.
    for:
      - _moe_C.grouped_topk
    labels:
      - fused
      - NoCPU
      - vLLM
    kind:
      - MoE
    stages:
      - stable: '5.0'
```

## 5. Operator Deliverables {#operator-deliverables}

Before developing a new operator, check `conf/operators.yaml` to ensure it does not duplicate an existing operator.

> [!IMPORTANT]
> **vLLM Operators Only — No PyTorch Operators Accepted**:
> FlagGems-vllm is dedicated to vLLM-specialized and fused operators and **does not accept general PyTorch operators** (general PyTorch ops belong in [FlagGems](https://github.com/flagos-ai/FlagGems)). Consequently, new operators **do not need to align with PyTorch** ATen semantics; they only need to align with vLLM repository operators, native CUDA/C++ kernels, or model reference implementations.

**New operator contributions must fulfill the following complete deliverables**:

1. **Fill in operator metadata in `conf/operators.yaml`**:
   - Add a new entry to `conf/operators.yaml`;
   - Fill in all six required fields: `id`, `description`, `for`, `labels`, `kind`, and `stages`;
   - Ensure `id` is unique and sorted alphabetically, passing `python3 tools/ci_checks/check_operators_yaml.py`.
2. **Implement operator & host dispatch (`src/flaggems_vllm/ops/<op>.py`)**:
   - Implement host dispatch and `@triton.jit` kernel functions;
   - Decorate host dispatch with `@libentry()` and validate tensor shapes, strides, dtypes, and devices;
   - **Strictly adhere to the NO_TORCH_COMPUTE_FALLBACK rule** (see next section): memory operators (`torch.empty`, `torch.zeros`, etc.) and creation/copy operators (`contiguous`, `clone`, `copy_`, etc.) are allowed; calling FlagGems operators is allowed; calling PyTorch operators to implement computation is forbidden;
   - Raise explicit `NotImplementedError` for unsupported shapes or dtypes.
3. **Export public APIs**:
   - Import the operator and add it to `__all__` in `src/flaggems_vllm/ops/__init__.py`;
   - Export at top level in `src/flaggems_vllm/__init__.py`.
4. **Autotune configuration (`src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml`)**:
   - Any kernel exposing performance parameters (`BLOCK_*`, `num_warps`, `num_stages`, etc.) must provide tuned configurations in `tune_configs.yaml`;
   - Integrate autotuning with `@libtuner(configs=..., key=[...], strategy=[...])`.
5. **Functional unit tests (`tests/test_<op>.py`)**:
   - Test functions must be decorated with `@pytest.mark.<op>`;
   - Parametrize across representative shapes, dtypes (e.g., FP16, BF16, FP8), and edge cases;
   - Tests must pass in both `--quick` and full test modes;
   - Verify accuracy against vLLM native/reference implementations using `tests.accuracy_utils` (`gems_assert_close` / `gems_assert_equal`).
6. **Performance benchmarks (`benchmark/test_<op>.py` and `benchmark/core_shapes.yaml`)**:
   - Add benchmark test cases in `benchmark/test_<op>.py`;
   - Record representative core shapes in `benchmark/core_shapes.yaml`;
   - **Performance target**: Achieve **SpeedUp ≥ 0.9** on target devices (e.g., NVIDIA H100) for core shapes.

### Standard File Drop Points

| Deliverable | Path | Description |
| --- | --- | --- |
| Operator Metadata | `conf/operators.yaml` | Operator inventory entry (mandatory) |
| Operator Implementation | `src/flaggems_vllm/ops/<op>.py` | Host dispatch and Triton kernels |
| Autotune Configuration | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` | NVIDIA candidate parameters and strategy |
| Functional Unit Test | `tests/test_<op>.py` | Accuracy and edge tests with `@pytest.mark.<op>` |
| Performance Benchmark | `benchmark/test_<op>.py` | Benchmark evaluation script |
| Core Benchmark Shapes | `benchmark/core_shapes.yaml` | List of primary shapes for acceptance |
| Internal Module Export | `src/flaggems_vllm/ops/__init__.py` | Export and add to `__all__` |
| Package-level Export | `src/flaggems_vllm/__init__.py` | Top-level direct public package export |

## 6. NO_TORCH_COMPUTE_FALLBACK Rule {#host-dispatch-conventions}

**Critical**: Production code paths allow calling memory operators such as `torch.empty` and `torch.zeros` as well as creation or copy operators such as `contiguous`, `clone`, and `copy_` from the PyTorch framework, but otherwise forbid calling torch operators to implement computational functions. Calling FlagGems operators is allowed.

### Allowed

- **PyTorch Memory Allocation Operators**: Calling `torch.empty`, `torch.empty_like`, `torch.zeros`, `torch.zeros_like`, `torch.empty_strided`, etc. is allowed;
- **Creation & Copy Operators**: Calling `tensor.contiguous()`, `tensor.clone()`, `tensor.copy_()`, etc. is allowed;
- **FlagGems Operators**: Calling operators from FlagGems (`flag_gems`) is allowed;
- **Metadata and Views**: Reading tensor metadata (`.shape`, `.stride`, `.dtype`, `.device`) and true no-copy views are allowed;
- **Triton Kernel Launches**: Launching custom Triton JIT kernels.

```python
# ✓ ALLOWED
import flag_gems
import torch

def my_operator(x, weight):
    # Metadata reads:
    shape = x.shape              # ✓
    dtype = x.dtype              # ✓
    device = x.device            # ✓

    # Allowed PyTorch memory operators (torch.empty, torch.zeros, etc.):
    output = torch.empty_like(x) # ✓
    buf = torch.zeros((shape[0], 64), dtype=dtype, device=device) # ✓

    # Allowed creation / copy operators (contiguous, clone, copy_, etc.):
    x_contig = x.contiguous()    # ✓
    cloned = x.clone()           # ✓
    output.copy_(cloned)         # ✓

    # True no-copy views:
    view = x.view(new_shape)     # ✓

    # Calling FlagGems operators:
    gemm_out = flag_gems.mm(x, weight)  # ✓

    # Triton kernel launch:
    triton_kernel[grid](x, ...)  # ✓
```

### Forbidden

Except for memory allocation, creation/copy operators, and metadata/views, **calling PyTorch operators to implement computation is strictly prohibited**:

```python
# ❌ FORBIDDEN in production paths (torch compute ops)
def my_operator(x):
    # Calling PyTorch operators for computation is forbidden:
    return torch.matmul(x, x.T)  # ❌
    return x.sum(dim=-1)          # ❌
    return torch.softmax(x, dim=-1) # ❌
    return x.to(torch.float32)    # ❌ (cast kernel)
```

### Unsupported Configurations

When encountering unsupported shapes or dtypes, explicitly raise `NotImplementedError`, never silently fall back to torch compute:

```python
if x.dtype not in [torch.float16, torch.bfloat16]:
    raise NotImplementedError(f"Unsupported dtype: {x.dtype}")
```

Torch compute is allowed **only** in `tests/`, `benchmark/`, and clearly marked reference implementations.

## 7. Code Style and Formatting {#code-style}

Run `pre-commit` before submitting changes:

```bash
pip install pre-commit
pre-commit run --all-files
```

Configured tools:
- **black** (line length 88)
- **isort** (`--profile black`)
- **flake8** (max line length 120, ignores `F405,E731,W503,E203,E704`)

## 8. Operator Unit Tests {#operator-unit-tests}

Add functional unit tests in `tests/test_<op>.py`:
- Annotate test functions with `@pytest.mark.<op>`;
- Parametrize typical shapes and dtypes;
- Compare against references using `tests.accuracy_utils` (`gems_assert_close` / `gems_assert_equal`);
- Support both quick and full testing modes:
  ```bash
  pytest -v -s tests/test_<op>.py
  pytest -q tests/test_<op>.py --quick
  ```

## 9. Operator Performance Benchmarking {#operator-performance-benchmarking}

Add performance benchmarks in `benchmark/test_<op>.py`:
- Parametrize shapes using `Config.get_shapes("<op>")`;
- Record representative shapes in `benchmark/core_shapes.yaml`;
- **SpeedUp definition**: `SpeedUp = latency_vllm_baseline / latency_flaggems`;
- **Acceptance criteria**: Achieve **SpeedUp ≥ 0.9** on core shapes;
- Run benchmarks:
  ```bash
  pytest -q benchmark/test_<op>.py --level core --iter 1 --warmup 1
  ```

## 10. Operator Development Protocol {#development-protocol}

Operator implementation and modifications must adhere to the mandatory protocol defined in `workflow.md` (quality gates G0–G6). See [Operator Development Protocol](workflow.md).
