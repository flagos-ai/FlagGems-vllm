---
title: Operator Development Protocol
weight: 20
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

# Operator Development Protocol

FlagGems-vllm follows a strict operator development protocol defined in `workflow.md` at the repository root. For operator deliverables, metadata management in `conf/operators.yaml`, and coding rules, see the [Contribution Overview](overview.md).

## Hard Gates Overview (G0–G7)

All operator addition and modification tasks must satisfy the following hard gates:

- **G0 (Task Clarity & Protocol Acknowledgment)**: Clarify task scope, confirm the operator belongs to vLLM-specific fused kernels, and commit to following the protocol;
- **G1 (NO_TORCH_COMPUTE_FALLBACK Verification)**: Production paths allow calling memory operators (`empty`, `zeros`, etc.) and creation/copy operators (`contiguous`, `clone`, `copy_`, etc.), but strictly forbid calling PyTorch operators to implement computation;
- **G2 (Pre-coding Design Tables)**: Before writing code, complete the Engineering Baseline, vLLM Alignment Specification, and Implementation Path Design tables;
- **G3 (Host Dispatch & Boundary Checks)**: Implement host dispatch with strict shape, stride, dtype, and device validation, explicitly raising `NotImplementedError` for unsupported cases;
- **G4 (Autotune Integration)**: Any operator with performance parameters must provide tuned configurations in NVIDIA backend and integrate with `@libtuner`;
- **G5 (Implementation Complete)**: Implement host dispatch, `@triton.jit` kernels, and public exports (`ops/__init__.py` and `flaggems_vllm/__init__.py`);
- **G6 (Functional Tests Pass)**: `tests/test_<op>.py` passes 100% in both `--quick` and full test modes;
- **G7 (Performance Acceptance)**: Core shape benchmarks in `benchmark/test_<op>.py` achieve **SpeedUp ≥ 0.9**.

## Pre-Coding Design Tables

Before implementing an operator, contributors must complete and record three design tables:

### 1. Engineering Baseline Table

| Field | Requirements |
| --- | --- |
| Operator Name & ID | Operator API name, matching `id` in `conf/operators.yaml` |
| Function Signature | Complete input parameters and return type specification |
| vLLM Context | Target model architecture and layer (e.g. DeepSeek-V3 MoE, PagedAttention) |
| Dependencies | Internal dependencies (FlagGems operators) or external libraries |
| Target Backend | Default NVIDIA, or specific heterogeneous hardware backends |
| Performance Target | Acceptance criterion: core shape SpeedUp `SpeedUp >= 0.9` |

### 2. vLLM Alignment Specification Table

> [!NOTE]
> FlagGems-vllm stores vLLM operators and does not accept general PyTorch operators; therefore, alignment with PyTorch is not required. Operators align against vLLM native C++/CUDA extensions, Triton kernels, or model reference implementations.

| Field | Requirements |
| --- | --- |
| Reference Baseline | vLLM native C++/CUDA extension, Triton kernel, or model reference implementation |
| Floating-Point Tolerance | Explicit `rtol` and `atol` thresholds for FP32/FP16/BF16/FP8 |
| Edge Cases & Special Values | Handling for empty tensors, extreme dimensions, NaN, Inf |
| Layout & Contiguity | Supported strides and memory layout (contiguous can be ensured via `contiguous()`) |

### 3. Implementation Path Design Table

| Field | Requirements |
| --- | --- |
| Workload Classification | Branching logic for Prefill vs Decode phases or batch size tiers |
| Kernel Architecture Strategy | Single fused Triton kernel or composition with FlagGems operators |
| Autotune Search Space | Candidate space for `BLOCK_*`, `num_warps`, `num_stages` |
| Unsupported Scope | Explicit conditions raising `NotImplementedError` |

## Static Audit Requirements

Before opening a PR, audit production code paths for disallowed torch calls:

```bash
python - <<'PY'
from pathlib import Path
op = '<op>'
p = Path(f'src/flaggems_vllm/ops/{op}.py')
if p.exists():
    text = p.read_text()
    suspicious = [pat for pat in ['torch.matmul', 'torch.sum', 'torch.softmax', 'torch.nn.functional', 'torch.ops.aten', '.to('] if pat in text]
    print(f"Audit {p}: suspicious calls = {suspicious}")
PY
```

## Related Links

- [Contribution Overview](overview.md): Operator deliverables, `conf/operators.yaml` metadata management, and coding rules
- [Root workflow.md Protocol](https://github.com/flagos-ai/FlagGems-vllm/blob/main/workflow.md)
- [Backend Development Guide](backend.md)
