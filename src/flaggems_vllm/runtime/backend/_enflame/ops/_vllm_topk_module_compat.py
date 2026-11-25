# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCU module-level compat for ``flaggems_vllm.ops.top_k_per_row_*``.

``tests/test_tle_capabilities.py`` re-executes the top-K module files from
``import_module("flaggems_vllm.ops.top_k_per_row_<phase>").__file__`` to
exercise the TLE capability contract on a fresh module object.  The generic
files' non-TLE fallback is the seed radix-select kernel (CTA-scope
``tl.atomic_add`` plus constant zero-value histogram stores), which is
banned on GCU: it produced wrong results and device aborts on narrow-band
inputs, and its fresh compilation is rejected by the current toolchain
(``memref_ext.memset_start`` properties, see
compiler_issues/vllm_top_k_per_row_prefill.md).  Warm ``~/.triton/cache``
artifacts masked both symptoms locally, but cold environments (QA runs)
hit them — the post-11-09 ``Fatal Python error: Aborted`` regression.

The SpecOpRegistrar function swap cannot reach this path (it replaces
callables, not module files), so this module redirects the module identity
itself: ``install_vllm_topk_module_compat()`` (wired at the end of
``_enflame/ops/__init__.py``) replaces the ``sys.modules`` entries for
``flaggems_vllm.ops.top_k_per_row_prefill`` / ``..._decode`` with this file
executed under those names, and serves the same file through a meta-path
finder if the generic module has not been imported yet.  Consumers that
patched the module attributes keep their patches (only pristine generic
modules are swapped).

The served module preserves the contract the tests rely on:

- the TLE capability detection header (``HAS_TLE`` / ``tle``, evaluated
  against whatever ``triton.experimental.tle`` is importable at exec time,
  never mutating the language namespace);
- ``non_tle_top_k_per_row_prefill`` / ``non_tle_top_k_per_row_decode``
  exposing ``cache_key`` — here as GCU-safe binary-search top-K (no
  atomics, no zero-value stores, single launch).  The decode kernel mirrors
  the generic direct-launch signature; the prefill entry is a thin
  call-style adapter that accepts both the legacy benchmark call (without
  the ``NUM_ROWS``/``CANONICAL_OUTPUT`` constexprs) and the current
  generic keyword set;
- host entrypoints with the generic signatures, delegating to the
  dedicated GCU implementations in
  ``_enflame/ops/top_k_per_row_prefill.py`` / ``..._decode.py``.

Remove this redirect once the generic non-TLE kernels are GCU-safe.
"""

import os

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils.tle_capabilities import supports_topk_tle
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

# Marker checked by the installer: set before anything else so that an
# in-flight (finder-served) exec of this file is recognized as compat.
_FLAGGEMS_GCU_TOPK_MODULE_COMPAT = True


def _vendor_tle_enabled() -> bool:
    """Whether the backend declares TLE support in its VendorDescriptor.

    Mirrors the generic module header: ``has_triton_tle()`` only proves the
    module imports, not that the backend can lower ``tle.gpu.alloc``.
    ``FLAGGEMS_FORCE_TLE`` overrides.
    """
    override = os.environ.get("FLAGGEMS_FORCE_TLE")
    if override is not None:
        return override.lower() not in {"0", "false", "off", "no"}
    try:
        return bool(getattr(runtime.device.info, "tle_enabled", False))
    except Exception:  # noqa: BLE001 - never let detection break the import
        return False


if has_triton_tle(3, 6, 0) and _vendor_tle_enabled():
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = supports_topk_tle(tle)
        if not HAS_TLE:
            # Triton's dependency walk also visits disabled TLE branches.
            tle = None
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False


# Public constants of the generic module surface (values kept identical so
# tests and the benchmark fallback keep working unchanged).
NUM_THREADS_PER_BLOCK = 512
NUM_BINS = 2048
NUM_FILNAL_ITEMS = 2048
SORTING_ALGORITHM_THRESHOLD = 12288
CANONICAL_OUTPUT_VOCAB_THRESHOLD = 65536
RADIX_FINAL_PREFILL_VOCAB_THRESHOLD = CANONICAL_OUTPUT_VOCAB_THRESHOLD

# Launch geometry helper of the generic module surface: the benchmark
# fallback sizes the baseline launch with ``_num_warps(NUM_THREADS_PER_BLOCK)``
# instead of hardcoding a 32-lane division, so the served module must expose
# it too. GCU reports neither device property, so the generic 32/1024
# fallback applies and the result matches ``block_size // 32``.
_LAUNCH_GEOMETRY = None


def _launch_geometry():
    global _LAUNCH_GEOMETRY
    if _LAUNCH_GEOMETRY is None:
        warp, maxt = 32, 1024
        try:
            props = runtime.torch_device_fn.get_device_properties(0)
            warp = getattr(props, "warp_size", 0) or 32
            maxt = getattr(props, "max_threads_per_block", 0) or 1024
        except Exception:  # noqa: BLE001 - never let detection break dispatch
            pass
        _LAUNCH_GEOMETRY = (warp, maxt)
    return _LAUNCH_GEOMETRY


def _num_warps(block_size):
    """Warps covering a BLOCK_SIZE-wide tile, divided by the real warp size."""
    warp, maxt = _launch_geometry()
    return max(1, min(block_size // warp, maxt // warp))


# Binary-search passes: one probe per pass halves the int32 bracket, so 32
# rounds converge; the guard skips any extra round. Instantiated as a
# tl.constexpr because @jit kernels reference it as a module global.
_MAX_ITERS = tl.constexpr(40)


def _use_radix_final_for_prefill(vocab_size):
    """Same threshold semantics as the generic module (config query only)."""
    return vocab_size >= RADIX_FINAL_PREFILL_VOCAB_THRESHOLD


@triton.jit
def _monotone_key_i32(x):
    # Monotone map from the fp32 bit pattern to signed int32: larger key ==
    # larger float (total order for non-NaN inputs). Negative branch flips
    # the low 31 bits, non-negative branch keeps the bits; the sign bit
    # separates the two ranges, so plain signed compares order the floats.
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _gcu_topk_row(
    row_base,
    out_base,
    row_len,
    stride1,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_ITERS: tl.constexpr,
):
    """Atomic-free per-row top-K over ``[row_base, row_base + row_len)``.

    Proven GCU recipe (same as ``_enflame/ops/top_k_per_row_prefill.py``):
    a monotone int32 key, an overflow-safe binary search for the K-th
    largest key T, then one stable selection scan (forward ``tl.cumsum``
    slot assignment; ties resolve to the lowest columns). No scratch
    buffers, no zero-value stores, no atomics.
    """
    lane = tl.arange(0, BLOCK)

    if row_len <= TOPK:
        # Every valid element is selected; pad the rest with -1.
        n_chunks = tl.cdiv(TOPK, BLOCK)
        for chunk in tl.range(0, n_chunks):
            pos = chunk * BLOCK + lane
            tl.store(out_base + pos, pos, mask=pos < row_len)
            tl.store(out_base + pos, -1, mask=(pos >= row_len) & (pos < TOPK))
    else:
        n_tiles = tl.cdiv(row_len, BLOCK)

        # Binary search T = min{x : count(key > x) < TOPK} over the full
        # int32 range; floor midpoint is overflow-safe for signed bounds.
        lo = -2147483648
        hi = 2147483647
        for _iter in tl.range(0, MAX_ITERS):
            if lo < hi:
                mid = (lo & hi) + ((lo ^ hi) >> 1)
                count = 0
                for tile in tl.range(0, n_tiles):
                    offs = tile * BLOCK + lane
                    in_row = offs < row_len
                    x = tl.load(row_base + offs * stride1, mask=in_row, other=0.0)
                    key = _monotone_key_i32(x)
                    key = tl.where(in_row, key, -2147483648)
                    count += tl.sum((key > mid).to(tl.int32), axis=0)
                if count >= TOPK:
                    lo = mid + 1
                else:
                    hi = mid

        # count(key > T) for the tie budget (one explicit pass: robust to
        # wherever the search converged).
        n_gt = 0
        for tile in tl.range(0, n_tiles):
            offs = tile * BLOCK + lane
            in_row = offs < row_len
            x = tl.load(row_base + offs * stride1, mask=in_row, other=0.0)
            key = _monotone_key_i32(x)
            key = tl.where(in_row, key, -2147483648)
            n_gt += tl.sum((key > hi).to(tl.int32), axis=0)

        # Stable selection in one scan: keys > T fill slots [0, n_gt), the
        # first (TOPK - n_gt) tied keys (ascending column) fill the rest.
        need = TOPK - n_gt
        found = 0
        ties = 0
        for tile in tl.range(0, n_tiles):
            offs = tile * BLOCK + lane
            in_row = offs < row_len
            x = tl.load(row_base + offs * stride1, mask=in_row, other=0.0)
            key = _monotone_key_i32(x)
            key = tl.where(in_row, key, -2147483648)
            gt = (key > hi).to(tl.int32)
            eq = ((key == hi) & in_row).to(tl.int32)
            gt_ex = tl.cumsum(gt, axis=0) - gt
            eq_ex = tl.cumsum(eq, axis=0) - eq
            take_gt = gt > 0
            take_eq = (eq > 0) & ((ties + eq_ex) < need)
            take = take_gt | take_eq
            slot = tl.where(take_gt, found + gt_ex, n_gt + ties + eq_ex)
            tl.store(out_base + slot, offs, mask=take)
            found += tl.sum(gt, axis=0)
            ties += tl.sum(eq, axis=0)


@triton.jit
def _gcu_non_tle_top_k_per_row_prefill(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    stride1,
    vocab_size,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    NUM_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CANONICAL_OUTPUT: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
):
    """GCU-safe non-TLE prefill kernel (generic direct-launch signature).

    ``NUM_ROWS`` / ``CANONICAL_OUTPUT`` are accepted for call-site
    compatibility and unused, as are the scratch buffers: this kernel needs
    none of them (no histogram, no final-items staging).
    """
    row = tl.program_id(0) + ROW_OFFSET
    row_start = tl.load(row_starts + row)
    row_end = tl.load(row_ends + row)
    row_len = row_end - row_start
    row_base = logits_ptr + row.to(tl.int64) * stride0 + row_start
    out_base = out_indices_ptr + row.to(tl.int64) * TOPK
    _gcu_topk_row(
        row_base,
        out_base,
        row_len,
        stride1,
        TOPK=TOPK,
        BLOCK=BLOCK_SIZE,
        MAX_ITERS=_MAX_ITERS,
    )


class _NonTleTopkPrefillKernel:
    """Call-style adapter in front of the GCU-safe non-TLE prefill kernel.

    The benchmark fallback predating the ``NUM_ROWS`` / ``CANONICAL_OUTPUT``
    constexprs launches the kernel without them; newer callers pass the full
    generic keyword set. The adapter serves both: it forwards everything
    unchanged and only fills the two config constexprs the legacy call
    omits, exactly like the generic host dispatch does
    (``NUM_ROWS == grid[0]``, ``CANONICAL_OUTPUT = vocab >= 65536``).
    """

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if "NUM_ROWS" not in kwargs and isinstance(grid, tuple) and grid:
                kwargs["NUM_ROWS"] = grid[0]
            if "CANONICAL_OUTPUT" not in kwargs and len(args) > 6:
                kwargs["CANONICAL_OUTPUT"] = args[6] >= CANONICAL_OUTPUT_VOCAB_THRESHOLD
            return _gcu_non_tle_top_k_per_row_prefill[grid](*args, **kwargs)

        return launch

    @property
    def cache_key(self):
        return _gcu_non_tle_top_k_per_row_prefill.cache_key


non_tle_top_k_per_row_prefill = _NonTleTopkPrefillKernel()


from flaggems_vllm.runtime.backend._enflame.ops.top_k_per_row_decode import (  # noqa: E402
    top_k_per_row_decode as _gcu_top_k_per_row_decode,
)
from flaggems_vllm.runtime.backend._enflame.ops.top_k_per_row_prefill import (  # noqa: E402
    top_k_per_row_prefill as _gcu_top_k_per_row_prefill,
)


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Selects top_k indices from each row of logits. Only valid elements
    within [row_starts[i], row_ends[i]) are considered. Delegates to the
    dedicated GCU implementation (single kernel launch, atomic-free
    binary search); see ``_enflame/ops/top_k_per_row_prefill.py``.
    """
    return _gcu_top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for decode phase of DeepSeek V4.

    Selects top_k indices per row of ``logits``; only valid elements
    within the per-batch prefix are considered. Delegates to the dedicated
    GCU implementation; see ``_enflame/ops/top_k_per_row_decode.py``.
    """
    return _gcu_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    )


# ────────────────────────────── wiring ──────────────────────────────

import importlib.util  # noqa: E402
import logging
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

logger = logging.getLogger(__name__)

_COMPAT_FILE = Path(__file__).resolve()

_TARGETS = (
    "flaggems_vllm.ops.top_k_per_row_prefill",
    "flaggems_vllm.ops.top_k_per_row_decode",
)


def _load_compat_module(target_name):
    """Execute this file as a fresh module named ``target_name``."""
    spec = importlib.util.spec_from_file_location(target_name, _COMPAT_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[target_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[target_name]
        raise
    return module


def _is_pristine_generic(module, target_name):
    """True when ``module`` is the untouched generic top-K module."""
    if getattr(module, "_FLAGGEMS_GCU_TOPK_MODULE_COMPAT", False):
        return False
    file = getattr(module, "__file__", None)
    if not file:
        return False
    resolved = Path(file).resolve()
    generic_ops_dir = _COMPAT_FILE.parents[4] / "ops"
    return resolved.parent == generic_ops_dir and resolved.name == (
        target_name.rsplit(".", 1)[-1] + ".py"
    )


class _VllmTopkModuleCompatFinder:
    """Serve this file for the target names if they are not imported yet."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or fullname in sys.modules:
            return None
        return importlib.util.spec_from_file_location(fullname, _COMPAT_FILE)


def install_vllm_topk_module_compat():
    """Redirect the top-K module identities to the GCU compat (idempotent).

    Only pristine generic modules are swapped, so explicit user or test
    patches always win. The ``flaggems_vllm.ops`` package attributes are
    updated too (direct-import entrypoint, mirroring the mrope/bin_topk
    backend override precedents); the ``flaggems_vllm`` top-level names
    stay with the SpecOpRegistrar swap.
    """
    try:
        if runtime.device.vendor_name != "enflame":
            return
    except Exception:  # noqa: BLE001 - never break the backend import
        return

    for finder in sys.meta_path:
        if isinstance(finder, _VllmTopkModuleCompatFinder):
            break
    else:
        sys.meta_path.insert(0, _VllmTopkModuleCompatFinder())

    ops_pkg = sys.modules.get("flaggems_vllm.ops")
    for target_name in _TARGETS:
        fn_name = target_name.rsplit(".", 1)[-1]
        generic = sys.modules.get(target_name)
        if generic is None:
            continue  # the finder serves the compat on first import
        if not _is_pristine_generic(generic, target_name):
            continue
        try:
            compat = _load_compat_module(target_name)
        except Exception:  # noqa: BLE001 - keep the generic module on failure
            continue
        if ops_pkg is not None and getattr(ops_pkg, fn_name, None) is getattr(
            generic, fn_name, None
        ):
            setattr(ops_pkg, fn_name, getattr(compat, fn_name))


@triton.jit
def non_tle_top_k_per_row_decode(
    logits_ptr,
    out_indices_ptr,
    seq_lens_ptr,
    next_n,
    stride0,
    stride1,
    vocab_size,
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_found_topk_values_ptr,
    NUM_ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CANONICAL_OUTPUT: tl.constexpr,
):
    """GCU-safe non-TLE decode kernel (generic direct-launch signature).

    Rows belong to batch ``row // next_n`` with valid prefix
    ``seq_lens[b] - next_n + (row % next_n) + 1``, matching the generic
    kernel's semantics; scratch buffers are accepted and ignored.
    """
    row = tl.program_id(0)
    batch_id = row // next_n
    batch_offset = row % next_n
    seq_len = tl.load(seq_lens_ptr + batch_id)
    row_len = seq_len - next_n + batch_offset + 1
    row_base = logits_ptr + row.to(tl.int64) * stride0
    out_base = out_indices_ptr + row.to(tl.int64) * TOPK
    _gcu_topk_row(
        row_base,
        out_base,
        row_len,
        stride1,
        TOPK=TOPK,
        BLOCK=BLOCK_SIZE,
        MAX_ITERS=_MAX_ITERS,
    )
