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

import logging
import os
import threading
import types
from importlib import import_module

import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

logger = logging.getLogger(__name__)

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")


# BEGIN TLE GATE: identical in top_k_per_row_decode.py and top_k_per_row_prefill.py
# The TLE path keeps the histogram and counters in shared memory, where a
# single-address atomic is ~17x cheaper than in global scratch on a C550. It is
# taken whenever this Triton has TLE at all. What it needs beyond that is not
# checked: a FlagTree built with mctle, with __MCTLE__ reaching TableGen -- or
# the first kernel fails to compile -- and with metax's Alias.cpp aware of
# mctle.local_pointers, or it compiles and returns wrong answers.
# FLAGGEMS_METAX_TLE=0 forces the generic non-TLE path.

# Thread limit of the compiled merge kernel on a C550; torch reports more.
_MAX_THREADS = 512

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as _tle
    except ImportError:  # pragma: no cover - depends on the installed wheel
        _tle = None
else:
    _tle = None


@triton.jit
def _cumsum_exclusive(x, axis: tl.constexpr = 0, reverse: tl.constexpr = False):
    """tle.cumsum's (exclusive prefix, total) in plain tl."""
    tl.static_assert(not reverse, "reverse=True is not implemented")
    return tl.cumsum(x, axis=axis) - x, tl.sum(x, axis=axis)


@triton.jit
def _final_select_rank(
    s_histogram_ptr,
    s_final_logits_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_out_indices_ptr,
    s_out_logits_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MULTIPLE_BLOCKS_PER_ROW: tl.constexpr,
):
    """The generic non-radix final select, behind _final_select_radix's signature."""
    NUM_FINAL_ITEMS: tl.constexpr = 2048
    lane = tl.arange(0, BLOCK_SIZE)
    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), NUM_FINAL_ITEMS)
    sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
    for sort_chunk in tl.range(0, sort_chunks):
        pos = sort_chunk * BLOCK_SIZE + lane
        valid = pos < final_cnt
        logit_i = tl.load(s_final_logits_ptr + pos, mask=valid, other=0)
        out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        for j in tl.range(0, final_cnt):
            logit_j = tl.load(s_final_logits_ptr + j)
            better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
            out_rank = out_rank + (valid & better).to(tl.int32)
        dst_pos = base_idx + out_rank
        take = valid & (dst_pos < TOPK)
        idx_i = tl.load(s_histogram_ptr + pos, mask=take, other=0)
        tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
        if MULTIPLE_BLOCKS_PER_ROW:
            tl.store(s_out_logits_ptr + dst_pos, logit_i, mask=take)
    tl.debug_barrier()


def _build_shim():
    if _tle is None:
        return None
    real = _tle.gpu

    # Adds pid >> 31 (always 0, alignment unprovable): the metax plugin widens
    # shared loads from pointer alignment and asserts below 4 elements/thread.
    # A builtin, since a @jit function cannot take a TLE buffer on metax.
    @tl.core.builtin
    def local_ptr(buffer, indices=None, _semantic=None, _generator=None):
        p = real.local_ptr(buffer, indices, _semantic=_semantic, _generator=_generator)
        pid = tl.program_id(0, _semantic=_semantic)
        return p.__add__(pid.__rshift__(31, _semantic=_semantic), _semantic=_semantic)

    gpu = types.ModuleType("flaggems_metax_tle_gpu")
    for name, value in vars(real).items():
        if not name.startswith("__"):
            setattr(gpu, name, value)
    gpu.local_ptr = local_ptr
    shim = types.ModuleType("flaggems_metax_tle")
    shim.gpu = gpu
    shim.cumsum = _cumsum_exclusive
    return shim


_SHIM = _build_shim()


_lock = threading.Lock()
_state = {"done": False, "on": False, "why": "not checked yet"}


def _install():
    _generic.tle = _SHIM
    _generic.HAS_TLE = True
    # Radix final select is slower on a C550, and wrong in prefill.
    _generic._final_select_radix = _final_select_rank
    _generic.NUM_THREADS_PER_BLOCK_MERGE = _MAX_THREADS
    warp, maxt = _generic._launch_geometry()
    _generic._LAUNCH_GEOMETRY = (warp, min(maxt, _MAX_THREADS))


def ensure_tle():
    """Decides once per process whether this operator takes the TLE path."""
    if _state["done"]:
        return _state["on"]
    with _lock:
        if _state["done"]:
            return _state["on"]
        env = os.environ.get("FLAGGEMS_METAX_TLE", "").strip().lower()
        on, why = False, ""
        if env in ("0", "false", "off", "no"):
            why = "disabled by FLAGGEMS_METAX_TLE"
        elif _SHIM is None:
            why = "this Triton has no TLE"
        else:
            on, why = True, "TLE available"
        if on:
            _install()
            logger.info("%s: MetaX TLE path enabled (%s)", _generic.__name__, why)
        else:
            logger.info("%s: MetaX TLE path off (%s)", _generic.__name__, why)
        _state.update(done=True, on=on, why=why)
        return on


def status():
    """{'done', 'on', 'why'}."""
    return dict(_state)


# END TLE GATE

# A 1024-thread tile pays on long rows ((64, 129280) 0.88 -> 0.98 of vLLM on a
# C550) and costs 12-20% on many short rows, so it is keyed on vocabulary size.
WIDE_TILE_VOCAB = 16384


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    if ensure_tle():
        # Read by the generic host dispatch on this very call.
        _generic.NUM_THREADS_PER_BLOCK = (
            1024 if logits.shape[1] >= WIDE_TILE_VOCAB else 512
        )
    return _generic.top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )
