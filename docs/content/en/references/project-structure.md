---
title: Project Structure
weight: 20
---

# Project Structure

FlagGems-vllm repository organization.

## Top-Level Layout

```
FlagGems-vllm/
├── src/flaggems_vllm/      # Python package
│   ├── __init__.py         # Top-level exports, enable()/use_gems()
│   ├── config.py           # Configuration, C extension detection
│   ├── ops/                # Operator implementations
│   ├── runtime/            # Backend, device, autotune config
│   └── utils/              # libentry, libtuner, helpers
├── tests/                  # Functional tests
├── benchmark/              # Performance benchmarks
├── conf/                   # Operator metadata (operators.yaml)
├── tools/                  # CI scripts, setup, test selection
├── docs/                   # Documentation (Hugo site)
├── workflow.md             # Operator development protocol
├── pyproject.toml          # Package metadata, build config
└── pytest.ini              # Pytest configuration
```

## Source Code (`src/flaggems_vllm/`)

### `ops/` - Operator Implementations

```
ops/
├── __init__.py             # Re-exports all operators + __all__
├── <op>.py                 # Single-file operator (e.g., grouped_topk.py)
├── DSA/                    # Domain-specific architectures
├── FLA/                    # Flash Linear Attention
├── mhc/                    # Multi-head computation
└── qwen4/                  # Qwen4 model-specific ops
```

Each operator file typically contains:
- Host dispatch function (dtype/shape routing, validation)
- One or more `@triton.jit` kernels
- Autotune integration via `@libtuner()`

### `runtime/` - Backend and Device

```
runtime/
├── __init__.py             # Device detection, get_tuned_config()
├── register.py             # Register class (PyTorch dispatch)
├── configloader.py         # Load tune_configs.yaml
├── backend/
│   ├── _nvidia/            # NVIDIA backend
│   │   ├── tune_configs.yaml    # Pre-tuned autotune configs
│   │   ├── ampere/              # Ampere-specific configs
│   │   └── hopper/              # Hopper-specific configs
│   ├── _amd/               # AMD backend
│   ├── _ascend/            # Huawei Ascend
│   └── ...                 # 15+ more backends
└── device.py               # Device abstraction (device_count, etc.)
```

### `utils/` - Utilities

```
utils/
├── libentry.py             # @libentry() decorator (operator entry point)
├── libtuner.py             # @libtuner() decorator (autotune integration)
├── triton_helpers.py       # Triton kernel utilities
├── pointwise_dynamic.py    # Dynamic pointwise codegen
└── dtype_utils.py          # Data type helpers
```

## Tests (`tests/`)

```
tests/
├── conftest.py             # Pytest hooks, --quick/--ref/--record
├── accuracy_utils.py       # to_reference(), gems_assert_*()
├── test_<op>.py            # Per-operator functional tests
├── test_FLA/               # Flash Linear Attention tests
└── test_DSA/               # DSA tests
```

Test naming: `test_<op>.py` matches `ops/<op>.py` for CI test selection.

## Benchmarks (`benchmark/`)

```
benchmark/
├── conftest.py             # Pytest hooks, --level/--iter/--warmup
├── core_shapes.yaml        # Core shapes per operator
├── test_<op>.py            # Per-operator performance benchmarks
└── base.py                 # Benchmark utilities
```

## Configuration (`conf/`)

```
conf/
└── operators.yaml          # Operator metadata (102 ops)
                            # - id, name, description, kind, labels, stage
```

## Tools (`tools/`)

```
tools/
├── setup.sh                # Full GPU environment bootstrap
├── select_tests.py         # CI: map changed files → test targets
├── select_backends.py      # CI: multi-backend test selection
└── run_ci_targets.py       # CI: run selected tests/benchmarks
```

## Development Protocol (`workflow.md`)

Mandatory operator development protocol with hard gates (G0–G7):
- Pre-coding tables (project truth, torch contract, implementation paths)
- NO_TORCH_COMPUTE_FALLBACK rule
- Autotune requirements
- Testing and performance acceptance criteria

**Read `workflow.md` before adding or modifying operators.**

## Package Metadata (`pyproject.toml`)

- Package name: `flaggems_vllm`
- Build system: `scikit-build-core` (C++ extensions via pybind11)
- Dependencies: `torch>=2.6.0`, `pyyaml`, `triton` or `flagtree`
- Optional extras: `[test]` (pytest, numpy, scipy, cupy)

## Next Steps

- [Operator development protocol](/FlagGems-vllm/contribution/workflow/)
- [Contributing guide](/FlagGems-vllm/contribution/overview/)
- [Testing guide](/FlagGems-vllm/testing/unittest/)
