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
- **G1 (Engineering Baseline & Environment)**: Freeze build and test commands, validation set, and benchmark environment;
- **G2 (vLLM Operator Specification)**: Clarify dtype matrix, shape, layout, promotion, boundary semantics, and unsupported scope (aligning with vLLM native kernels or model references, no PyTorch alignment required);
- **G3 (Path Design & Host Dispatch)**: Define fast/special/general/unsupported paths, design host dispatch and defensive checks, strictly obeying `NO_TORCH_COMPUTE_FALLBACK`;
- **G4 (Implementation & Integration)**: Complete metadata registration (`conf/operators.yaml`), host dispatch and Triton JIT kernels, autotune configuration (`tune_configs.yaml`), tests, and public exports;
- **G5 (Functional Tests Pass)**: `tests/test_<op>.py` passes 100% in both `--quick` and full test modes;
- **G6 (Performance Acceptance)**: Core shape benchmarks in `benchmark/test_<op>.py` achieve **SpeedUp ≥ 0.9**;
- **G7 (Complete Delivery)**: Output delivery summary detailing paths, autotune configuration, test results, and residual risks.

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
