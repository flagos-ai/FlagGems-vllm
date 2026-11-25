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

import ctypes
import logging
import threading

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# GCU constraints (see knowledge/coding_rules.md + knowledge/compiler_traps.md):
#   * atomics are forbidden in GCU kernels (the generic implementation's
#     tl.atomic_add/tl.atomic_max on the int64 buffers abort the device), so
#     the histogram uses a bin-centric partition: each program owns a TILE_B
#     slice of bins and scans the input with a one-hot compare. The [BLOCK_N,
#     TILE_B] orientation (input rows / bin columns) is mandatory: the
#     transposed [TILE_B, BLOCK_N] form with a column-broadcast weight
#     silently miscompiles on triton_gcu 3.6.
#   * kernels must not touch int64 buffers: on this stack torch_gcu maps
#     torch.int64 to a packed 4-byte-per-element device layout ("GCU not
#     support Long use Int replace"), while Triton scales int64 pointer
#     arithmetic by 8 bytes, so any i64 tl.load/tl.store reads/writes the
#     wrong addresses (the previously documented "silent i64 corruption").
#     Inputs are therefore normalized to int32 on the host: because the
#     physical storage already holds the 32-bit values contiguously,
#     `input.view(torch.int32)[:numel]` is a zero-copy int32 alias of the
#     same bytes (validated bitwise against `.to(torch.int32)`); a one-time
#     layout probe falls back to the casting path if the storage layout
#     ever changes. The int64 output is staged through an int32 buffer
#     plus copy_.
#   * weighted counts accumulate in fp32 and round to the weights dtype
#     exactly once at the final store (the official generic semantics,
#     see ops/bincount.py bincount_weights_fp32_kernel): the GCU-native
#     torch.bincount accumulates half-precision weights in half precision
#     and therefore disagrees with the CPU/exact reference by ~1 half ULP
#     per addition, which the accuracy suite flags. The weighted kernels
#     below carry a 2D [BLOCK_N, TILE_B] fp32 accumulator across the input
#     loop and reduce it once (tl.sum(axis=0)) right before the single
#     dtype cast -- per-iteration tl.sum reductions measured ~12-45us
#     slower at the bench core shapes. Only bins below max_val+1 can ever
#     match an input value, so the kernels take a num_active bound and the
#     minlength-padded tail is zero-filled without scanning the input
#     (halves the work for the core minlength=8192 shape). For large n
#     with few bin tiles the single-CTA scan starves the device, so the
#     input is split into chunks: stage 1 computes per-(chunk, bin-tile)
#     fp32 partial histograms, stage 2 reduces the partials and applies
#     the single dtype cast (this two-stage form is what keeps the
#     n=100000 shapes within ~2.2x of the scatter-based vendor kernel
#     while the single-stage form needs 4-7x).
#   * fp32 weights above _WEIGHTED_TOPS_MIN_N route to the vendor
#     topsatenBincount (see _TopsatenBridge below): the vendor op
#     accumulates fp32 weights in fp32 (verified bitwise-identical to the
#     on-device native torch.bincount), which already satisfies the fp32
#     accumulation semantics, and its scatter-based kernel outruns the
#     one-hot Triton kernel whenever the input is long and the bin count
#     is large. fp16/bf16 weights never route there: the vendor op
#     accumulates them in half precision, which is exactly the deviation
#     this backend must not reproduce.
#   * num_warps=1 is a known silent-miscompile case for one-hot counting
#     kernels, keep num_warps=4.

_HIST_BLOCK_N = 1024
_HIST_TILE_B = 32
_HIST_SMALL_N = 256
_HIST_SMALL_BLOCK_N = 256
_HIST_SMALL_TILE_B = 64
_MAX_BLOCK_N = 1024
_MAX_PARTIAL_BLOCK_N = 1024
_MAX_PARTIAL_PROGS = 8192
_SINGLE_CTA_MAX_N = 65536

# Weighted one-hot histogram tile forms, selected by the active bin count
# (max(input)+1) and the input length (probe-measured on GCU400, bench core
# shapes + n=10000/100000 QA shapes): narrow histograms run best with a tall
# [BLOCK_N, TILE_B] tile (few wasted bin columns per program), wide
# histograms with [256, 64]. Large streaming work (n > _W2_MIN_N with more
# than 2M compare lanes) prefers [256, 64] in every measured case. The 2D
# fp32 accumulator peaks at ~3 tiles of [512, 32]/[256, 64] fp32 (~192KB on
# DSM), well inside the 448KB Libra budget.
_W_FORM_NARROW_TILE_B = 16
_W_FORM_NARROW_BLOCK_N = 512
_W_FORM_MID_TILE_B = 32
_W_FORM_MID_BLOCK_N = 512
_W_FORM_WIDE_TILE_B = 64
_W_FORM_WIDE_BLOCK_N = 256
_W_FORM_MID_ACTIVE_LIMIT = 128
_W_FORM_WIDE_ACTIVE_LIMIT = 512

# Two-stage (chunked partial histogram + reduce) thresholds: only bins below
# num_active are compared, so with few bin tiles a single-CTA scan starves
# the 48-SP device once n grows. Past _W2_MIN_N the input is split into
# num_chunks = min(_W2_TARGET_PROGS // num_tiles, cdiv(n, BLOCK_N)) chunks;
# stage 1 writes [num_chunks, num_active] fp32 partials, stage 2 reduces
# them and casts once. Below _W2_MIN_N the extra launches cost more (~15us
# device time each) than the parallelism gains (measured: two-stage loses
# 4-10us at every n <= 4096 bench shape, wins 20-460us at n >= 10000).
_W2_TARGET_PROGS = 96
_W2_MIN_N = 8192
_W2_WORK_BIG = 1 << 21
_W2_BLOCK_C = 128

# Above this input length the fp32-weighted path routes to the native
# topsatenBincount call: the vendor op accumulates fp32 weights in fp32
# (bitwise-identical to the on-device native result, i.e. the same
# single-rounding semantics), and its scatter kernel beats the one-hot
# Triton kernel for long inputs with many bins (~35us at n=4096 core
# shapes). fp16/bf16 weights never route there -- the vendor op would
# accumulate them in half precision.
_WEIGHTED_TOPS_MIN_N = 2048

_INT64_ALIAS_MODE = None


def _int32_view_of_int64(input_contig):
    """Zero-copy int32 alias of a GCU int64 tensor.

    torch_gcu stores int64 tensors in a packed 4-byte-per-element device
    layout (the low 32 bits of each value, contiguously), so the first
    `numel` words of `view(torch.int32)` are exactly the int32 values --
    identical to `.to(torch.int32)` without launching a cast kernel. A
    one-time per-process probe verifies the layout and falls back to the
    casting path if a future stack switches to a true 8-byte layout.
    """
    global _INT64_ALIAS_MODE
    if _INT64_ALIAS_MODE is None:
        try:
            probe = torch.arange(3, dtype=torch.int64, device=input_contig.device)
            words = probe.view(torch.int32)[:3]
            _INT64_ALIAS_MODE = (
                "packed" if words.cpu().tolist() == [0, 1, 2] else "wide"
            )
        except Exception:
            _INT64_ALIAS_MODE = "wide"
    if _INT64_ALIAS_MODE == "packed":
        return input_contig.view(torch.int32)[: input_contig.numel()]
    return input_contig.to(torch.int32)


# ---------------------------------------------------------------------------
# topsaten native route for long-input fp32-weighted bincount.
#
# torch_gcu's native bincount (aten -> topsaten::topsatenBincount) passes the
# weights straight into the vendor op, whose accumulation domain follows the
# weight dtype: fp32 weights are accumulated in fp32 (verified
# bitwise-identical to the on-device native torch.bincount), while fp16/bf16
# weights are accumulated in half precision (deterministic in-process, order
# opaque for large bins). Only the fp32 case routes here (n above
# _WEIGHTED_TOPS_MIN_N): it already satisfies the fp32-accumulation +
# single-rounding semantics and its scatter kernel beats the one-hot Triton
# kernel on long inputs with many bins. The half-precision dtypes stay on the
# local fp32-accumulating kernels -- delegating them would reintroduce the
# half-domain accumulation error (~1 half ULP per addition) that the accuracy
# suite flags against the CPU/exact reference.
#
# ABI facts (verified against libtopsaten.so on this stack, headers under
# /opt/tops/include/gcu/topsaten/):
#   * topsatenTensor holds only a std::shared_ptr<topsatenTensorImpl>
#     (16 bytes); its constructor and SetTensorData are exported under the
#     mangled names below, so a 16-byte buffer can be placement-constructed
#     from ctypes.
#   * topsatenBincount(topsatenTensor& output, const topsatenTensor& input,
#     const topsatenTensor& weights, int64_t minlength, topsStream_t stream)
#     -> topsatenStatus_t. The output must be caller-allocated, sized and
#     pre-zeroed (the kernel uses atomic adds).
#   * The topsatenTensor destructor is not exported: descriptors are pooled
#     and reused (bounded leak, one TensorImpl per distinct
#     shape/dtype/data_ptr signature; steady-state benchmark loops allocate
#     none). Descriptors may be rebound right after the call returns.
#   * Two tensors of one call must never share a descriptor (aliasing
#     corrupts outputs); the per-call claim set enforces that.
# ---------------------------------------------------------------------------

_TOPS_DTYPES = {
    torch.int32: 6,
    torch.float16: 4,
    torch.bfloat16: 5,
    torch.float32: 8,
}

_TENSOR_CTOR_SYM = (
    "_ZN14topsatenTensorC1ERK14topsatenSize_tS2_RK18topsatenDataType_tRKPv"
)
_TENSOR_SET_DATA_SYM = "_ZN14topsatenTensor13SetTensorDataERKPv"
_BINCOUNT_SYM = (
    "_ZN8topsaten16topsatenBincountER14topsatenTensorRKS0_S3_lP12EfiStream_st"
)

_STATUS_NAMES = {
    0: "SUCCESS",
    1: "ALLOC_FAILED",
    2: "BAD_PARAM",
    3: "NOT_SUPPORT",
    4: "INTERNAL_ERROR",
    5: "RUNTIME_ERROR",
    6: "EXECUTE_ERROR",
}


class _TopsatenSize(ctypes.Structure):
    """struct topsatenSize_t { const int64_t* data; int64_t len; }"""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int64)),
        ("len", ctypes.c_int64),
    ]


class _TopsatenTensorBuf(ctypes.Structure):
    """16-byte storage of topsatenTensor (shared_ptr control pair)."""

    _fields_ = [("ptr", ctypes.c_void_p), ("ctrl", ctypes.c_void_p)]


def _topsaten_strides(shape):
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= max(shape[i], 1)
    return tuple(strides)


class _TopsatenBridge:
    """Lazy ctypes binding of topsatenBincount for the long-input fp32 path."""

    def __init__(self):
        lib = ctypes.CDLL("libtopsaten.so")
        ctor = lib[_TENSOR_CTOR_SYM]
        ctor.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenSize),
            ctypes.POINTER(_TopsatenSize),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ctor.restype = None
        set_data = lib[_TENSOR_SET_DATA_SYM]
        set_data.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        set_data.restype = None
        bincount = lib[_BINCOUNT_SYM]
        bincount.argtypes = [
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.POINTER(_TopsatenTensorBuf),
            ctypes.c_long,
            ctypes.c_void_p,
        ]
        bincount.restype = ctypes.c_int
        self._lib = lib
        self._ctor = ctor
        self._set_data = set_data
        self._bincount = bincount
        self._pool = {}
        self._call_lock = threading.Lock()
        self._raw_stream_fn = self._load_raw_stream_fn()

    @staticmethod
    def _load_raw_stream_fn():
        try:
            from torch_gcu import _C
        except Exception as exc:  # pragma: no cover - torch_gcu always present
            raise RuntimeError(f"torch_gcu._C import failed: {exc}")
        fn = getattr(_C, "_gcu_getCurrentRawStream", None)
        if fn is None:
            raise RuntimeError(
                "torch_gcu._C._gcu_getCurrentRawStream is unavailable; cannot "
                "resolve the current GCU stream for the topsaten route"
            )
        return fn

    def _current_stream(self, index):
        return self._raw_stream_fn(index)

    def _descriptor(self, tensor, claim):
        dtype_code = _TOPS_DTYPES.get(tensor.dtype)
        if dtype_code is None:
            raise RuntimeError(
                f"topsaten bincount route: unsupported dtype {tensor.dtype}"
            )
        shape = tuple(tensor.shape)
        strides = _topsaten_strides(shape)
        data_ptr = tensor.data_ptr()
        pool = self._pool.setdefault((shape, strides, tensor.dtype), [])
        for entry in pool:
            handle = entry[0]
            if id(handle) not in claim and entry[3] == data_ptr:
                claim.add(id(handle))
                return handle
        for entry in pool:
            handle = entry[0]
            if id(handle) in claim:
                continue
            data = ctypes.c_void_p(data_ptr)
            self._set_data(handle, ctypes.byref(data))
            entry[3] = data_ptr
            claim.add(id(handle))
            return handle
        dims = (ctypes.c_int64 * len(shape))(*shape)
        strd = (ctypes.c_int64 * len(strides))(*strides)
        sizes = _TopsatenSize(dims, len(shape))
        strides_size = _TopsatenSize(strd, len(strides))
        dtype = ctypes.c_int(dtype_code)
        data = ctypes.c_void_p(data_ptr)
        buf = _TopsatenTensorBuf()
        self._ctor(
            ctypes.byref(buf),
            ctypes.byref(sizes),
            ctypes.byref(strides_size),
            ctypes.byref(dtype),
            ctypes.byref(data),
        )
        handle = ctypes.pointer(buf)
        pool.append([handle, dims, strd, data_ptr])
        claim.add(id(handle))
        return handle

    def bincount_weighted(self, output, input_contig, weights_contig, minlength):
        # ctypes releases the GIL inside the call; hold the bridge lock so a
        # concurrent call cannot rebind pooled descriptors mid-flight.
        with self._call_lock:
            claim = set()
            out_desc = self._descriptor(output, claim)
            in_desc = self._descriptor(input_contig, claim)
            w_desc = self._descriptor(weights_contig, claim)
            stream = self._current_stream(input_contig.device.index or 0)
            status = self._bincount(out_desc, in_desc, w_desc, int(minlength), stream)
        if status != 0:
            name = _STATUS_NAMES.get(status, "UNKNOWN")
            raise RuntimeError(f"topsatenBincount failed: status={status} ({name})")


_BRIDGE = None
_BRIDGE_LOCK = threading.Lock()
_BRIDGE_ERROR = None


def _get_topsaten_bridge():
    global _BRIDGE, _BRIDGE_ERROR
    if _BRIDGE is not None:
        return _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            if _BRIDGE_ERROR is None:
                try:
                    _BRIDGE = _TopsatenBridge()
                except Exception as exc:
                    _BRIDGE_ERROR = str(exc)
            if _BRIDGE is None:
                raise RuntimeError(f"topsaten bridge unavailable: {_BRIDGE_ERROR}")
    return _BRIDGE


@libentry()
@triton.jit
def _bincount_max_kernel(input_ptr, out_ptr, n_elements, BLOCK_N: tl.constexpr):
    acc = tl.full((BLOCK_N,), -1, dtype=tl.int32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(input_ptr + offs, mask=offs < n_elements, other=-1).to(tl.int32)
        acc = tl.maximum(acc, vals)
    tl.store(out_ptr, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_partial_max_kernel(
    input_ptr,
    partials_ptr,
    n_elements,
    num_progs,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.full((BLOCK_N,), -1, dtype=tl.int32)
    for base in range(pid * BLOCK_N, n_elements, num_progs * BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        vals = tl.load(input_ptr + offs, mask=offs < n_elements, other=-1).to(tl.int32)
        acc = tl.maximum(acc, vals)
    tl.store(partials_ptr + pid, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_reduce_max_kernel(
    partials_ptr,
    out_ptr,
    num_partials,
    BLOCK_R: tl.constexpr,
):
    acc = tl.full((BLOCK_R,), -1, dtype=tl.int32)
    for base in range(0, num_partials, BLOCK_R):
        offs = base + tl.arange(0, BLOCK_R)
        vals = tl.load(partials_ptr + offs, mask=offs < num_partials, other=-1)
        acc = tl.maximum(acc, vals)
    tl.store(out_ptr, tl.max(acc, axis=0))


@libentry()
@triton.jit
def _bincount_hist_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.int32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        match = (vals[:, None] == bins[None, :]).to(tl.int32)
        acc += tl.sum(match, axis=0)
    tl.store(output_ptr + bins, acc, mask=bin_mask)


@libentry()
@triton.jit
def _bincount_hist_weights_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
    num_bins,
    num_active,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    if HAS_TAIL:
        # Only bins below num_active (= max(input)+1) can match anything;
        # the minlength-padded tail is zero-filled without scanning.
        if pid * TILE_B < num_active:
            acc2d = tl.zeros((BLOCK_N, TILE_B), dtype=tl.float32)
            for base in range(0, n_elements, BLOCK_N):
                offs = base + tl.arange(0, BLOCK_N)
                mask = offs < n_elements
                vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
                w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                acc2d += (vals[:, None] == bins[None, :]).to(tl.float32) * w[:, None]
            acc = tl.sum(acc2d, axis=0)
        else:
            acc = tl.zeros((TILE_B,), dtype=tl.float32)
        tl.store(output_ptr + bins, acc.to(output_ptr.dtype.element_ty), mask=bin_mask)
    else:
        acc2d = tl.zeros((BLOCK_N, TILE_B), dtype=tl.float32)
        for base in range(0, n_elements, BLOCK_N):
            offs = base + tl.arange(0, BLOCK_N)
            mask = offs < n_elements
            vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
            w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            acc2d += (vals[:, None] == bins[None, :]).to(tl.float32) * w[:, None]
        acc = tl.sum(acc2d, axis=0)
        tl.store(output_ptr + bins, acc.to(output_ptr.dtype.element_ty), mask=bin_mask)


@libentry()
@triton.jit
def _bincount_weights_partial_kernel(
    input_ptr,
    weights_ptr,
    partials_ptr,
    n_elements,
    num_active,
    num_tiles,
    CHUNK: tl.constexpr,
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_id = pid % num_tiles
    chunk_id = pid // num_tiles
    bins = tile_id * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_active
    chunk_start = chunk_id * CHUNK
    chunk_end = tl.minimum(chunk_start + CHUNK, n_elements)
    acc2d = tl.zeros((BLOCK_N, TILE_B), dtype=tl.float32)
    for base in range(chunk_start, chunk_end, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < chunk_end
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc2d += (vals[:, None] == bins[None, :]).to(tl.float32) * w[:, None]
    acc = tl.sum(acc2d, axis=0)
    tl.store(partials_ptr + chunk_id * num_active + bins, acc, mask=bin_mask)


@libentry()
@triton.jit
def _bincount_weights_reduce_kernel(
    partials_ptr,
    output_ptr,
    num_chunks,
    num_active,
    num_bins,
    TILE_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    active_mask = bins < num_active
    acc = tl.zeros((TILE_B,), dtype=tl.float32)
    for base in range(0, num_chunks, BLOCK_C):
        rows = base + tl.arange(0, BLOCK_C)
        load_mask = (rows < num_chunks)[:, None] & active_mask[None, :]
        partials = tl.load(
            partials_ptr + rows[:, None] * num_active + bins[None, :],
            mask=load_mask,
            other=0.0,
        )
        acc += tl.sum(partials, axis=0)
    tl.store(output_ptr + bins, acc.to(output_ptr.dtype.element_ty), mask=bin_mask)


def _weighted_hist_form(active, n):
    """Probe-tuned (TILE_B, BLOCK_N) for the weighted one-hot kernels."""
    if active <= _W_FORM_MID_ACTIVE_LIMIT:
        tile_b, block_n = _W_FORM_NARROW_TILE_B, _W_FORM_NARROW_BLOCK_N
    elif active <= _W_FORM_WIDE_ACTIVE_LIMIT:
        tile_b, block_n = _W_FORM_MID_TILE_B, _W_FORM_MID_BLOCK_N
    else:
        tile_b, block_n = _W_FORM_WIDE_TILE_B, _W_FORM_WIDE_BLOCK_N
    if n > _W2_MIN_N and n * active > _W2_WORK_BIG:
        tile_b, block_n = _W_FORM_WIDE_TILE_B, _W_FORM_WIDE_BLOCK_N
    elif n < block_n:
        # Tiny inputs waste the masked lanes of a tall tile; shrinking
        # BLOCK_N to the (power-of-two) input length runs ~1-2us faster
        # (probe: [64, 16] vs [512, 16] at n=64).
        shifted = 64
        while shifted < n:
            shifted <<= 1
        block_n = min(block_n, shifted)
    return tile_b, block_n


def _weighted_hist_launch(input_contig, weights_contig, n, active, needed_size):
    """fp32-accumulated weighted histogram with a single final dtype cast.

    Every addition happens in fp32 (weights upcast at load); the fp32 sums
    are rounded to the weights dtype exactly once, at the final store --
    the official generic bincount semantics. Bin-centric one-hot partition,
    no atomics (GCU red line). Inputs longer than _W2_MIN_N with too few
    bin tiles to fill the device go through the two-stage chunked form.
    """
    device = input_contig.device
    if active <= 0:
        # No input value can fall into any bin (max(input) < 0): every count
        # is zero; needed_size > 0 is guaranteed by the caller.
        return torch.zeros(needed_size, dtype=weights_contig.dtype, device=device)
    tile_b, block_n = _weighted_hist_form(active, n)
    num_tiles = -(-active // tile_b)
    output = torch.empty(needed_size, dtype=weights_contig.dtype, device=device)

    num_chunks = _W2_TARGET_PROGS // num_tiles
    max_chunks = -(-n // block_n)
    if num_chunks > max_chunks:
        num_chunks = max_chunks
    if num_chunks < 1:
        num_chunks = 1
    if num_chunks > 1 and n > _W2_MIN_N:
        per_chunk = (n + num_chunks - 1) // num_chunks
        chunk_len = (per_chunk + block_n - 1) // block_n * block_n
        num_chunks = (n + chunk_len - 1) // chunk_len
        partials = torch.empty(num_chunks * active, dtype=torch.float32, device=device)
        _bincount_weights_partial_kernel[(num_chunks * num_tiles,)](
            input_contig,
            weights_contig,
            partials,
            n,
            active,
            num_tiles,
            CHUNK=chunk_len,
            TILE_B=tile_b,
            BLOCK_N=block_n,
            num_warps=4,
        )
        _bincount_weights_reduce_kernel[(-(-needed_size // tile_b),)](
            partials,
            output,
            num_chunks,
            active,
            needed_size,
            TILE_B=tile_b,
            BLOCK_C=_W2_BLOCK_C,
            num_warps=4,
        )
        return output

    _bincount_hist_weights_kernel[(-(-needed_size // tile_b),)](
        input_contig,
        weights_contig,
        output,
        n,
        needed_size,
        active,
        TILE_B=tile_b,
        BLOCK_N=block_n,
        HAS_TAIL=(needed_size > active),
        num_warps=4,
    )
    return output


def bincount(input, weights=None, minlength=0):
    logger.debug("GEMS_ENFLAME BINCOUNT")

    assert input.dim() == 1, "input must be a 1-D tensor"
    assert minlength >= 0, "minlength must be non-negative"

    if weights is not None:
        assert weights.shape == input.shape, "weights must have the same shape as input"

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    input_contig = input.contiguous()
    weights_contig = weights.contiguous() if weights is not None else None

    if input_contig.dtype == torch.int64:
        input_contig = _int32_view_of_int64(input_contig)

    device = input_contig.device
    max_out = torch.empty(1, dtype=torch.int32, device=device)
    if n <= _SINGLE_CTA_MAX_N:
        _bincount_max_kernel[(1,)](
            input_contig,
            max_out,
            n,
            BLOCK_N=_MAX_BLOCK_N,
            num_warps=4,
        )
    else:
        num_progs = min(triton.cdiv(n, _MAX_PARTIAL_BLOCK_N), _MAX_PARTIAL_PROGS)
        partials = torch.empty(num_progs, dtype=torch.int32, device=device)
        _bincount_partial_max_kernel[(num_progs,)](
            input_contig,
            partials,
            n,
            num_progs,
            BLOCK_N=_MAX_PARTIAL_BLOCK_N,
            num_warps=4,
        )
        _bincount_reduce_max_kernel[(1,)](
            partials,
            max_out,
            num_progs,
            BLOCK_R=_MAX_PARTIAL_BLOCK_N,
            num_warps=4,
        )

    # The output length is data-dependent (max(input) + 1 vs minlength), so a
    # single host sync is semantically required; torch's own bincount syncs
    # the same way on CUDA.
    max_val = int(max_out.item())
    needed_size = max(max_val + 1, minlength)

    if needed_size <= 0:
        if weights is not None:
            return torch.empty(0, dtype=weights.dtype, device=device)
        return torch.empty(0, dtype=torch.int64, device=device)

    if weights is None:
        if n <= _HIST_SMALL_N:
            hist_block_n = _HIST_SMALL_BLOCK_N
            hist_tile_b = _HIST_SMALL_TILE_B
        else:
            hist_block_n = _HIST_BLOCK_N
            hist_tile_b = _HIST_TILE_B
        staging = torch.empty(needed_size, dtype=torch.int32, device=device)
        _bincount_hist_kernel[(triton.cdiv(needed_size, hist_tile_b),)](
            input_contig,
            staging,
            n,
            needed_size,
            TILE_B=hist_tile_b,
            BLOCK_N=hist_block_n,
            num_warps=4,
        )
        output = torch.empty(needed_size, dtype=torch.int64, device=device)
        output.copy_(staging)
        return output

    if weights_contig.dtype == torch.float64:
        raise NotImplementedError(
            "bincount with float64 weights is not supported on the enflame GCU backend"
        )

    active = max_val + 1

    if weights_contig.dtype == torch.float32 and n > _WEIGHTED_TOPS_MIN_N:
        # Long-input fp32 route: the vendor op accumulates fp32 weights in
        # fp32 (verified bitwise-identical to the on-device native
        # torch.bincount), which already satisfies the fp32-accumulation +
        # single-rounding semantics, and its scatter kernel outruns the
        # one-hot Triton kernel here (~35us at the n=4096 core shapes).
        # The bridge only accepts int32 inputs, so narrow integer inputs
        # are normalized.
        if input_contig.dtype != torch.int32:
            input_contig = input_contig.to(torch.int32)
        bridge = _get_topsaten_bridge()
        output = torch.zeros(needed_size, dtype=weights_contig.dtype, device=device)
        bridge.bincount_weighted(output, input_contig, weights_contig, minlength)
        return output

    # fp16/bf16 weights at any length, and fp32 weights up to
    # _WEIGHTED_TOPS_MIN_N: local fp32-accumulated one-hot histogram with a
    # single final rounding to the weights dtype (the official generic
    # semantics). The vendor op is deliberately not used for the half
    # dtypes: it accumulates them in half precision, which deviates from
    # the CPU/exact reference by ~1 half ULP per addition.
    return _weighted_hist_launch(input_contig, weights_contig, n, active, needed_size)
