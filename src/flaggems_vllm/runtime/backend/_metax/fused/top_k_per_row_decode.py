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

_generic = import_module("flaggems_vllm.ops.top_k_per_row_decode")


# BEGIN TLE GATE: identical in top_k_per_row_decode.py and top_k_per_row_prefill.py
# The TLE path keeps the histogram and counters in shared memory, where a
# single-address atomic is ~17x cheaper than in global scratch on a C550. It
# needs a FlagTree built with mctle, __MCTLE__ passed to TableGen, and metax's
# Alias.cpp aware of mctle.local_pointers; without the last, kernels compile but
# share bytes, so this is a runtime self-test. FLAGGEMS_METAX_TLE=0 disables it.

# Thread limit of the compiled merge kernel on a C550; torch reports more.
_MAX_THREADS = 512

_PROBE_N = 512
_PROBE_WARPS = 8

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


@triton.jit
def _self_test_kernel(bad_ptr, N: tl.constexpr):
    """Counts values an overlapping shared-memory allocation corrupted."""
    lane = tl.arange(0, N)
    a = _SHIM.gpu.alloc(
        [N],
        dtype=tl.int32,
        layout=None,
        scope=_SHIM.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pa = _SHIM.gpu.local_ptr(a, (0,))
    tl.store(pa + lane, lane)
    # Allocated after a's last direct use: an allocator that cannot see the
    # pointer users gives b a's offset.
    b = _SHIM.gpu.alloc(
        [N],
        dtype=tl.int32,
        layout=None,
        scope=_SHIM.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pb = _SHIM.gpu.local_ptr(b, (0,))
    tl.store(pb + lane, lane + N)
    c = _SHIM.gpu.alloc(
        [64],
        dtype=tl.int32,
        layout=None,
        scope=_SHIM.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(_SHIM.gpu.local_ptr(c), tl.zeros([64], tl.int32))
    tl.debug_barrier()
    pc = _SHIM.gpu.local_ptr(c, (0,))
    tl.atomic_add(
        pc + lane * 0, lane * 0 + 1, mask=(lane % 2) == 0, sem="relaxed", scope="cta"
    )
    # Reductions whose scratch an unpatched allocator lays over a or b.
    total = tl.sum(lane, axis=0)
    prefix, total2 = _SHIM.cumsum(lane, axis=0)
    tl.debug_barrier()
    bad = tl.sum((tl.load(pa + lane) != lane).to(tl.int32), axis=0)
    bad += tl.sum((tl.load(pb + lane) != lane + N).to(tl.int32), axis=0)
    bad += (tl.load(pc) != N // 2).to(tl.int32)
    bad += (total != N * (N - 1) // 2).to(tl.int32)
    bad += (total2 != total).to(tl.int32)
    bad += tl.sum((prefix != (lane * (lane - 1)) // 2).to(tl.int32), axis=0)
    tl.store(bad_ptr, bad)


_lock = threading.Lock()
_state = {"done": False, "on": False, "why": "not checked yet"}


def _is_mctle_build():
    from triton._C import libtriton

    if not hasattr(libtriton.ir.builder, "make_swizzled_shared_encoding_attr"):
        return False
    try:
        compiler = import_module("triton.backends.metax.compiler")
    except ImportError:
        return False
    return getattr(compiler, "enable_mctle", False) is True


def _self_test(device):
    """(ok, reason)."""
    import torch

    bad = torch.empty((1,), dtype=torch.int32, device=device)
    compiled = _self_test_kernel[(1,)](bad, N=_PROBE_N, num_warps=_PROBE_WARPS)
    count = int(bad.item())
    if count != 0:
        return False, f"self-test read back {count} wrong values (smem aliasing)"
    need = 2 * _PROBE_N * 4 + 64 * 4
    shared = getattr(getattr(compiled, "metadata", None), "shared", need)
    if shared < need:
        return False, f"self-test kernel got {shared} B of smem for {need} B of buffers"
    return True, "self-test passed"


def _install():
    _generic.tle = _SHIM
    _generic.HAS_TLE = True
    # Radix final select is slower on a C550, and wrong in prefill.
    _generic._final_select_radix = _final_select_rank
    _generic.NUM_THREADS_PER_BLOCK_MERGE = _MAX_THREADS
    warp, maxt = _generic._launch_geometry()
    _generic._LAUNCH_GEOMETRY = (warp, min(maxt, _MAX_THREADS))


def ensure_tle(device):
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
        elif not _is_mctle_build():
            why = "FlagTree built without mctle"
        else:
            try:
                on, why = _self_test(device)
            except Exception as e:  # noqa: BLE001 - any failure means "not this build"
                on, why = False, f"self-test failed: {type(e).__name__}: {e}"[:300]
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

# Blocks per row: the largest power of two keeping rows x blocks <= 256
# programs, within [4, 16]. The measured best at every benchmark row count on a
# C550 (generic default 10: geomean 1.89 -> 2.12 of vLLM). One block per row
# returns wrong results, hence the floor.
MAX_PROGRAMS = 256
MIN_BLOCKS_PER_ROW = 4
MAX_BLOCKS_PER_ROW = 16


def _blocks_per_row(num_rows):
    target = max(1, MAX_PROGRAMS // max(1, num_rows))
    pow2 = 1 << (target.bit_length() - 1)
    return max(MIN_BLOCKS_PER_ROW, min(MAX_BLOCKS_PER_ROW, pow2))


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    if ensure_tle(logits.device):
        # Read by the generic host dispatch on this very call.
        _generic.MULTIPLE_BLOCKS_PER_ROW_CONFIG = _blocks_per_row(num_rows)
    return _generic.top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    )
