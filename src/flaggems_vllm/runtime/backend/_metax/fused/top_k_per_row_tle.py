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

"""Put top_k_per_row on its TLE (shared-memory) path on MetaX -- only when the
installed FlagTree can actually run it.

Measured on a C550 against vLLM's own kernels (geomean, kernel mode), with the
per-call launch geometry chosen in top_k_per_row_{decode,prefill}.py:

    path             decode   prefill
    generic non-TLE   0.582     0.640
    TLE               2.117     1.210

The difference is where the histogram and the write counters live:
global scratch on the non-TLE path, shared memory here, and a single-address
atomic costs ~17x less in shared memory on this card.

WHICH BUILDS QUALIFY. The TLE path needs three things that MetaX FlagTree
builds before flagos-ai/FlagTree#1164 do not all have, so this file never
assumes them -- it runs a
small self-test kernel on the first call and switches the generic modules over
only if that passes:

  1. mctle compiled in (BUILD_MCTLE=ON): the tle.gpu bindings exist and the
     metax backend reports enable_mctle.
  2. __MCTLE__ reaching TableGen: FlagTree defines it for C++ only, so
     metax's TritonOps.td keeps the #else tt.atomic_rmw constraint and an
     atomic on a shared pointer fails the TTIR verifier. Surfaces as a
     compile error, caught.
  3. metax's own lib/Analysis/Alias.cpp taught about mctle.local_pointers:
     without it every buffer reached only through local_ptr looks dead to the
     shared-memory allocator after local_pointers, and its bytes are handed to
     reduction scratch and to other buffers. Nothing fails to compile -- the
     answers are silently wrong. The self-test is built to expose exactly
     that: two buffers written through local_ptr across cross-warp
     reductions, then read back.

A build with (1) and (2) but not (3) is the dangerous one, and is why this is a
runtime check and not a version test. FLAGGEMS_METAX_TLE=0 forces the non-TLE
path.

WHAT GETS CHANGED, by rebinding the generic modules' globals (no generic diff):

  tle        a module whose cumsum is plain tl (metax has no
             create_exclusive_cumsum binding) and whose local_ptr adds an
             always-zero, unprovable offset: the plugin widens the vector
             width of an unmasked shared load from pointer alignment alone,
             unclamped by elements per thread, and asserts below 4
             elements/thread -- which BLOCK_SIZE=512 on 8 warps is. It is a
             builtin rather than @jit because passing a tle buffer into a jit
             function needs builder.get_memdesc_type, also absent on metax.
  HAS_TLE    True.
  _final_select_radix
             replaced by the rank-based final select the kernel uses when
             radix is off. radix final is both slower here (decode 1.558 vs
             1.760, prefill 1.134 vs 1.194) and, in prefill, still wrong.
             Replacing the function rather than the policy keeps
             _use_radix_final_for_prefill as it is, and also catches prefill's
             hard-coded USE_RADIX_FINAL=True launch for rows past 12288.
  NUM_THREADS_PER_BLOCK_MERGE / _LAUNCH_GEOMETRY
             512. The TLE-only multi-block merge asks for 1024 threads; the
             compiled kernel's own limit on this card is 512 ("Hardware
             limit: 512"), while torch's device properties say more. A 512
             merge tile on 8 warps also measured fastest against 256 / 1024
             tiles and 2 / 4 warps.

All of it happens before the first TLE kernel is compiled, so the kernels'
cache keys include the replacements and never collide with an unswitched
process's.
"""

import logging
import os
import threading
import types
from importlib import import_module

import triton
import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

logger = logging.getLogger(__name__)

_GENERIC_MODULES = (
    "flaggems_vllm.ops.top_k_per_row_decode",
    "flaggems_vllm.ops.top_k_per_row_prefill",
)

# The compiled merge kernel's thread limit on a C550, as reported by Triton's
# OutOfResources at 16 warps. Per-kernel, so it is a measured constant here
# rather than a device property.
_MAX_THREADS = 512

# The self-test runs at the operator's own geometry: 512 lanes on 8 warps.
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
    """tle.cumsum's contract -- (exclusive prefix, total) -- in plain tl."""
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
    """_final_select_radix's signature, the non-radix final select's body --
    the else-branch of _top_k_per_row_job, identical in both generic files."""
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

    @tl.core.builtin
    def local_ptr(buffer, indices=None, _semantic=None, _generator=None):
        p = real.local_ptr(buffer, indices, _semantic=_semantic, _generator=_generator)
        # pid >> 31 is 0 for every valid program id, and AxisInfo cannot prove
        # its divisibility, so the pointer's alignment becomes unprovable.
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
    """Write two buffers through local_ptr, run cross-warp reductions and a
    masked shared atomic, read everything back. Any overlap the allocator
    was not told about shows up as a nonzero count."""
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
    # Allocated only after a's last direct use: an allocator that cannot see
    # the pointer users gives b a's offset.
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
    """(ok, reason). One launch and one 4-byte read, once per process."""
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
    for name in _GENERIC_MODULES:
        m = import_module(name)
        m.tle = _SHIM
        m.HAS_TLE = True
        m._final_select_radix = _final_select_rank
        m.NUM_THREADS_PER_BLOCK_MERGE = _MAX_THREADS
        warp, maxt = m._launch_geometry()
        m._LAUNCH_GEOMETRY = (warp, min(maxt, _MAX_THREADS))


def ensure_tle(device):
    """Switch top_k_per_row onto the TLE path if this build can run it.
    Decided once per process, on the first call; returns whether it is on."""
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
            logger.info("top_k_per_row: MetaX TLE path enabled (%s)", why)
        else:
            logger.info("top_k_per_row: MetaX TLE path off (%s)", why)
        _state.update(done=True, on=on, why=why)
        return on


def status():
    """{'done', 'on', 'why'} -- for tools and tests."""
    return dict(_state)
