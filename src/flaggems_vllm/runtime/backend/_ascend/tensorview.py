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

"""Private availability gate and Unified Buffer budget for Ascend TensorView."""

import logging

from flaggems_vllm.runtime import device

logger = logging.getLogger(__name__)

tle = None
if device.vendor_name == "ascend":
    try:
        import triton.experimental.tle as _tle
        import triton.experimental.tle.language  # noqa: F401
        from triton.experimental.tle import tile_load, tile_store  # noqa: F401
        from triton.experimental.tle.language.dsa.ascend import UB  # noqa: F401

        tle = _tle
    except ImportError:
        pass

HAS_TENSORVIEW = bool(
    tle is not None
    and all(
        hasattr(tle, name)
        for name in (
            "make_partition_view",
            "make_gather_scatter_view",
            "make_strided_view",
        )
    )
)
_UNKNOWN, _GOOD, _BAD = 0, 1, 2
_state = _UNKNOWN


def tensorview_available():
    return HAS_TENSORVIEW and _state != _BAD


def _is_compile_error(exc):
    return any(
        getattr(cls, "__module__", "").startswith("triton.")
        and cls.__name__ in ("MLIRCompilationError", "CompilationError")
        for cls in type(exc).__mro__
    )


def tensorview_call(op_name, fast_fn, generic_fn, *args, **kwargs):
    """Memoize a compiler incompatibility, never hide runtime/semantic errors."""
    global _state
    if not tensorview_available():
        return generic_fn(*args, **kwargs)
    try:
        result = fast_fn(*args, **kwargs)
    except Exception as exc:
        if not _is_compile_error(exc):
            raise
        _state = _BAD
        logger.warning(
            "GEMS_ASCEND %s TensorView compilation failed (%s); using generic path",
            op_name,
            exc,
        )
        return generic_fn(*args, **kwargs)
    _state = _GOOD
    return result


# DAV_2201 UB is 192 KiB. Keep half for compiler intermediates/buffering.
UB_BYTES = 192 * 1024
MIN_TILE_ROWS = 2


def tile_rows_for_ub(row_bytes, live_tiles, max_rows):
    budget = (UB_BYTES // 2) // (max(row_bytes, 1) * max(live_tiles, 1))
    if budget < 1:
        return 1
    return max(1, min(max_rows, 1 << (budget.bit_length() - 1)))


def tile_fits_ub(row_bytes, live_tiles):
    return (UB_BYTES // 2) // (max(row_bytes, 1) * max(live_tiles, 1)) >= MIN_TILE_ROWS
