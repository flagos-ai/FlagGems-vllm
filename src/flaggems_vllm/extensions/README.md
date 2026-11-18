# Extension operators

Opt-in fused operators that are **not** part of `flaggems_vllm._FULL_CONFIG`.
Importing this package never patches PyTorch dispatch; callers import an
operator explicitly:

```python
from flaggems_vllm.extensions import my_fused_op

out = my_fused_op(x, w)
```

## Directory layout

```
src/flaggems_vllm/extensions/
  __init__.py            # public entry points, one per operator
  _dispatch.py           # backend resolution: arch -> vendor -> generic
  _registry.py           # @register_extension / @register_backend_impl
  ops/                   # generic ("公版") implementation, one file per op
    __init__.py          # imports every op module + __all__
    <op>.py
  backends/              # optional per-backend overrides
    _nvidia/
      __init__.py
      ops/<op>.py        # vendor-wide override for nvidia
      hopper/ops/<op>.py # arch override (nvidia + hopper only)
      ampere/ops/<op>.py
    _ascend/ops/<op>.py
    _amd/ … _metax/ … _thead/ …    # one dir per supported vendor
```

Extension backends live here rather than under `runtime/backend/_<vendor>/`
because `BackendArchEvent._get_supported_archs()` treats every non-underscore
subdirectory of a vendor directory as an architecture. Adding `extensions/`
there would show up as a bogus arch.

## Resolution order

`resolve()` walks this list and returns the first hit:

1. `backends/_<vendor>/<arch>/ops` — e.g. `_nvidia/hopper/ops`
2. `backends/_<vendor>/ops` — vendor-wide
3. `ops` — generic implementation

Vendor directories are optional. A vendor without one uses the generic
implementation, so adding a backend never requires touching dispatch code.

## Adding an operator

1. **Generic implementation** — `ops/<op>.py`:

   ```python
   from flaggems_vllm.extensions import register_extension

   @register_extension("<op>")
   def <op>(x, w):
       ...
   ```

   This must run on the official upstream Triton compiler on NVIDIA Ampere
   (A100) — same rule as `ops/` in the main package.

2. **Export** — import it in `ops/__init__.py`, add it to that file's `__all__`,
   and expose a dispatching entry point in `extensions/__init__.py`:

   ```python
   from flaggems_vllm.extensions._registry import make_dispatcher

   <op> = make_dispatcher("<op>")
   ```

   The dispatcher resolves the backend per call, so an override added later is
   picked up without re-importing the caller.

3. **Backend override** (optional) — `backends/_<vendor>/ops/<op>.py`:

   ```python
   from flaggems_vllm.extensions import register_backend_impl

   @register_backend_impl("<op>", vendor="nvidia")
   def <op>_nvidia(x, w):
       ...
   ```

   Use `arch="hopper"` in a `backends/_nvidia/hopper/ops/` module for a single
   micro-architecture.

4. **Metadata** — add an entry to `conf/operators.yaml` with the `extension`
   label so it is distinguishable from the always-registered ops:

   ```yaml
     - id: <op>
       description: ...
       for:
         - None
       labels:
         - extension
         - fused
       kind:
         - NeuralNetwork
       stages:
         - beta: '5.4'
   ```

5. **Tests and benchmark** — `tests/test_extensions/test_<op>.py` decorated with
   `@pytest.mark.<op>`, and `benchmark/test_extensions/test_<op>.py`.

## Rules carried over from the main package

- **No torch compute fallback.** Host dispatch may allocate, copy and read
  metadata only. An unsupported dtype/layout/shape combination raises
  `NotImplementedError`, and `resolve()` raises it too when the current backend
  has no implementation.
- **Autotune is not optional.** Kernels exposing `BLOCK_*` / `TILE_*` /
  `num_warps` / `num_stages` go through `runtime.get_tuned_config("<op>")` +
  `@libtuner(...)` with the config in
  `runtime/backend/_nvidia/tune_configs.yaml`. Extension overrides that need
  their own tuning table use `runtime.get_tuned_config` with a distinct key
  (`"<op>_<vendor>"`).
- **Dual validation.** Accuracy against CPU / vLLM reference plus a
  `triton.testing.do_bench` speedup against the vLLM baseline.
