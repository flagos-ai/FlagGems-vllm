---
title: Project Structure
weight: 40
---

# Project Structure

FlagGems-vllm repository organization.

## Top-Level Layout

```
FlagGems-vllm/
├── src/flaggems_vllm/      # Python package
│   ├── __init__.py         # Top-level exports and public operator APIs
│   ├── backends.yaml       # Backend registry (python/torch/triton/vLLM pins)
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
├── __init__.py             # Re-exports all operators + __all__ (157 exported ops)
├── <op>.py                 # 80+ root operator implementations (e.g., grouped_topk.py)
├── DSA/                    # Domain-specific architectures
├── FLA/                    # Flash Linear Attention (includes gdn2_native)
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
├── __init__.py             # Device detection, get_tuned_config(), replace_customized_ops()
├── common.py               # Vendor registry, VendorInfo, feature capabilities
├── register.py             # Register class (PyTorch dispatch)
├── configloader.py         # Load tune_configs.yaml
├── flagtune.py             # FlagTune integration
└── backend/
    ├── device_finder.py    # DeviceDetector singleton
    ├── device.py           # Device abstraction helpers
    ├── backend_utils.py    # Backend helper utilities
    ├── _nvidia/            # NVIDIA backend
    │   ├── tune_configs.yaml    # Pre-tuned autotune configs
    │   ├── ampere/              # Ampere-specific configs
    │   └── hopper/              # Hopper-specific configs
    ├── _amd/               # AMD backend
    ├── _ascend/            # Huawei Ascend backend
    └── ...                 # 17 hardware backends total
```

### `utils/` - Utilities

```
utils/
├── libentry.py             # @libentry() and @libtuner() decorators
├── pointwise_dynamic.py    # Dynamic pointwise codegen
├── shape_utils.py          # Tensor shape, broadcast, and offset helpers
├── type_utils.py           # Data type utilities
├── triton_lang_helper.py   # Triton language shims and helpers
├── triton_driver_helper.py # Triton device driver properties
├── triton_version_utils.py # Triton and TLE version feature checks
└── models/                 # Model-specific helper modules
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
├── test_FLA/               # FLA benchmarks
└── base.py                 # Benchmark utilities
```

## Configuration (`conf/`)

```
conf/
└── operators.yaml          # Operator metadata (105 ops)
                            # - id, description, for, labels, kind, stages
```

## Tools (`tools/`)

```
tools/
├── ci_checks/              # CI verification scripts (YAML, exports, markers)
├── gpu_check_<vendor>.sh   # Hardware check scripts for 13 vendor chips
├── setup.sh                # Multi-backend environment bootstrap
├── source_env_bash.sh      # Re-export vendor env scripts for activate
├── select_tests.py         # CI: map changed files → test targets
├── select_backends.py      # CI: multi-backend test selection
└── run_ci_targets.py       # CI: run selected tests/benchmarks
```

## Development Protocol (`workflow.md`)

Mandatory operator development protocol with hard gates (G0–G7):
- NO_TORCH_COMPUTE_FALLBACK rule
- Autotune requirements
- Testing and performance acceptance criteria

**Read `workflow.md` before adding or modifying operators.**

## Package Metadata (`pyproject.toml`)

- Package name: `flaggems_vllm`
- Build system: `setuptools`
- Dependencies: `flag-gems`, `packaging`, `pybind11`, `PyYAML`, `sqlalchemy`
  (torch/triton/vLLM are provided by the vendor environment)
- Optional test extras: `[test]` (pytest, scipy, einops, distro) and
  `[test-nvidia]` (cupy-cuda12x)
- Backend extras with vendor vLLM packages: `[nvidia]`, `[metax]`, `[ascend]`,
  `[thead]`

## Next Steps

- [Contributing Guide](overview.md)
- [Operator Development Protocol](workflow.md)
- [Backend Development Guide](backend.md)
- [MiniMax M3 Support](../references/minimax-m3-support.md)
- [Testing Guide](../testing/unittest.md)
