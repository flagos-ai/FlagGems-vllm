# AGENTS.md

Guidance for AI coding agents (Claude Code, Gemini, Copilot, Codex, Cursor, etc.)
working in this repository. Human contributors may also find it useful.

## Project overview

**FlagGems-vllm** is a high-performance operator library for vLLM, part of
[FlagOS](https://flagos.io/). Operators are written in the
[Triton](https://github.com/openai/triton) programming language (FlagTree in
production) and target multiple hardware backends (NVIDIA by default).

The Python package is `flaggems_vllm` and lives under `src/`. It exposes
vLLM-scenario fused kernels — e.g. `flaggems_vllm.grouped_topk`,
`flaggems_vllm.fused_experts_impl`, `flaggems_vllm.ops.moe_align_block_size`.

### Relationship with sibling repos

- **FlagGems** — general-purpose operator backend; registers ops into PyTorch
  dispatch via `flag_gems.enable()` / `flag_gems.use_gems()`.
- **FlagGems-vllm** (this repo) — vLLM-specific fused kernels plus aligned
  tests/benchmarks. Exposes ops through the `flaggems_vllm` package.
- **vllm-plugin-fl** — the vLLM plugin layer; calls `flag_gems.enable()` for
  general ops and imports `flaggems_vllm.<operator>()` for vLLM fused kernels.

Call flow: `vLLM -> vllm-plugin-fl -> flag_gems.enable() + flaggems_vllm.<op>()`.

## Repository layout

```
src/flaggems_vllm/
  __init__.py              # top-level exports, enable()/only_enable()/use_gems()
  config.py                # enable/exclude config resolution, C-extension detection
  ops/                     # operator implementations (one file per op, mostly)
    __init__.py            # re-exports every public op + __all__
    <op>.py                # host dispatch + Triton kernels for <op>
    DSA/  FLA/  mhc/        # grouped op families (subpackages)
  runtime/
    __init__.py            # device detection, get_tuned_config(), torch_device_fn
    register.py            # Register class (PyTorch dispatch registration)
    configloader.py        # loads tune_configs.yaml / heuristics configs
    backend/_<vendor>/     # per-vendor config: tune_configs.yaml, heuristics, arch dirs
  utils/                   # libentry, libtuner, triton helpers, codegen, dtype utils
  testing/
tests/                     # functional tests: tests/test_<op>.py (+ test_FLA/, test_DSA/)
  conftest.py              # pytest hooks, --quick / --ref / --record options
  accuracy_utils.py        # to_reference(), gems_assert_* comparison helpers
benchmark/                 # perf tests: benchmark/test_<op>.py (+ conftest, base, consts)
  core_shapes.yaml         # recommended shapes per op
tools/                     # setup.sh (env bootstrap), select_tests.py (CI selection), etc.
workflow.md                # MANDATORY operator-development protocol (read before coding ops)
```

Supported backend vendor dirs live under `src/flaggems_vllm/runtime/backend/`:
`_nvidia` (default), `_amd`, `_metax`, `_iluvatar`, `_cambricon`, `_ascend`,
`_hygon`, `_mthreads`, `_kunlunxin`, and others. NVIDIA additionally has
`ampere/` and `hopper/` arch subdirs.

## Environment & installation

Requires Python ≥ 3.8 (CI uses 3.11/3.12), `torch>=2.6.0`, Triton/FlagTree, and
(for most tests/benchmarks) a CUDA-capable GPU plus vLLM.

```shell
# Build deps
pip install -U 'scikit-build-core>=0.11' pybind11 ninja cmake

# Editable install (dev)
pip install --no-build-isolation -e .
# With test extras (pytest, numpy, scipy, cupy-cuda12x)
pip install --no-build-isolation -e '.[test]'
```

Full GPU environment bootstrap (creates a `uv` venv, installs vLLM + matching
torch, installs this repo, swaps Triton for FlagTree, runs smoke tests):

```shell
source tools/setup.sh          # honors DNN_VENDOR, VLLM_VERSION, USE_FLAGTREE, CUDA_HOME, ...
```

`DNN_VENDOR` (e.g. `nvidia`) selects the backend. `VLLM_PLUGINS=fl` selects the
FlagOS plugin when multiple vLLM plugins are installed.

## Running tests

`pytest.ini` sets `pythonpath = src` and `testpaths = tests`, so a plain
`pytest` works from the repo root. CI and the README also pass `PYTHONPATH=src`
explicitly — do the same if invoking from elsewhere.

```shell
# Discover / import smoke check (do this first)
pytest -q tests --collect-only

# Fast functional validation
pytest -q tests --quick

# A single operator
pytest -v -s tests/test_grouped_topk.py
pytest -q tests/test_fused_inv_rope_fp8_quant.py --quick
```

Test options (from `tests/conftest.py`): `--quick` (small shape/dtype set),
`--ref {device|cpu}` (reference device), `--record {none|log|json}` +
`--output` (result capture; JSON default `accuracy_result.json`).

Most tests `skipif(not torch.cuda.is_available())` and compare against a
reference (often vLLM's own op) via `tests/accuracy_utils.py`
(`to_reference`, `gems_assert_equal`, `gems_assert_close`).

## Running benchmarks

```shell
pytest -q benchmark --collect-only
# Fast smoke: core shapes, 1 iter, 1 warmup
pytest -q benchmark/test_grouped_topk.py --level core --iter 1 --warmup 1
```

Benchmark options (from `benchmark/conftest.py`): `--level
{comprehensive|core}`, `--mode {kernel|operator|wrapper}`, `--warmup`, `--iter`,
`--dtypes`, `--metrics`, `--shape_file`, `--parallel N` (multi-GPU),
`--record`/`--output` (JSON default `benchmark_result.json`). Shapes come from
`benchmark/core_shapes.yaml`.

SpeedUp convention: `SpeedUp = latency_torch_baseline / latency_flaggems`.
Acceptance target for key shapes is `SpeedUp >= 0.9`.

## Code style & linting

`pre-commit` runs on all commits and in CI (`code-style` job, must pass first).

```shell
pip install pre-commit
pre-commit run --all-files
```

Configured hooks (`.pre-commit-config.yaml`):
- **black** — line length 88 (`pyproject.toml`).
- **isort** — `--profile black`.
- **flake8** — max line length **120**, ignores `F405,E731,W503,E203,E704`
  (`.flake8`).
- trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files.

Note the two line-length numbers: black formats to 88, flake8 only errors above
120. Match the style of surrounding code.

## Operator development protocol — READ `workflow.md` FIRST

`workflow.md` is the **mandatory execution protocol** for adding or modifying an
operator. It defines hard gates (G0–G7), the required pre-coding tables, and
acceptance/delivery format. Key rules an agent MUST honor:

1. **No torch compute fallback** (`NO_TORCH_COMPUTE_FALLBACK = true`).
   Production paths (impl, host dispatch, helpers, fallback, unsupported) must
   **not** use `torch` compute/copy/cast ops (`torch.<op>`,
   `torch.nn.functional.*`, `torch.ops.aten.*`, or `Tensor` methods that emit
   kernels like `matmul`, `sum`, `contiguous`, `clone`, `copy_`, `to`).
   Allowed: metadata reads (shape/stride/dtype/device), uninitialized allocation
   (`torch.empty*`), and true no-copy views. Unsupported paths must raise
   `NotImplementedError`, never silently fall back to torch.
   `torch` reference implementations are allowed **only** in tests/benchmarks,
   and must never be imported by production code.

2. **Standard code drop points** when adding an op `<op>`:

   | Purpose | Path |
   | --- | --- |
   | Operator implementation | `src/flaggems_vllm/ops/<op>.py` |
   | NVIDIA autotune config | `src/flaggems_vllm/runtime/backend/_nvidia/tune_configs.yaml` |
   | Functional test | `tests/test_<op>.py` |
   | Performance benchmark | `benchmark/test_<op>.py` |
   | Op export | `src/flaggems_vllm/ops/__init__.py` (import + `__all__`) |
   | Top-level registration | `src/flaggems_vllm/__init__.py` |

   An op is not "done" until it is both exported and registered.

3. **Autotune is not optional.** Any kernel exposing perf params (`BLOCK_*`,
   `TILE_*`, `num_warps`, `num_stages`, `num_ctas`, ...) should, on the NVIDIA
   backend, go through `runtime.get_tuned_config("<name>")` + `@libtuner(...)`,
   with the config present in `tune_configs.yaml`. Exemptions must be stated
   explicitly.

4. Design **host dispatch before kernels**: normalize args, route by
   dtype/layout/shape, handle empty/identity early returns, define the autotune
   key, and handle unsupported paths — all without torch compute.

Reading order for op work: `workflow.md` → `optimization.md` → `deep_opt.md`
(the latter two are referenced by the protocol; consult them when present).

### Existing patterns to follow

- Ops use `@libentry()` + `@libtuner(configs=..., key=[...], strategy=[...])` +
  `@triton.jit` (see `src/flaggems_vllm/ops/moe_align_block_size.py`).
- Optional TLE (Triton experimental language extensions) paths gate on
  `has_triton_tle(...)` and fall back gracefully.
- Tests mark each case with a per-op marker (e.g. `@pytest.mark.grouped_topk`),
  parametrize dtype/shape, honor `cfg.QUICK_MODE`, and compare against a
  reference through `accuracy_utils`.

## CI

`.github/workflows/basic-ci.yml`:
1. **code style** — `pre-commit run --all-files` (blocking).
2. **select targets** — `tools/select_tests.py` maps changed files to the
   relevant `tests/`/`benchmark/` targets (edits to `pyproject.toml`,
   `pytest.ini`, `tools/setup.sh`, or the CI file trigger an env smoke set).
3. **nvidia tests** — on an H20 runner: `pytest -q <selected> --quick` and
   `pytest -q <benchmarks> --level core`.

When you add an op, keep source↔test↔benchmark names aligned so
`select_tests.py` picks them up (it infers `test_<stem>.py` from the source
stem; non-standard names need an explicit entry in that script).

## Quick reference

| Task | Command |
| --- | --- |
| Install (dev) | `pip install --no-build-isolation -e '.[test]'` |
| Lint / format | `pre-commit run --all-files` |
| Collect tests | `pytest -q tests --collect-only` |
| Fast tests | `pytest -q tests --quick` |
| One test | `pytest -v -s tests/test_<op>.py` |
| Benchmark smoke | `pytest -q benchmark/test_<op>.py --level core --iter 1 --warmup 1` |
| Full GPU setup | `source tools/setup.sh` |

License: Apache 2.0 (`LICENSE`).
