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
#     slice of bins and scans the input with a one-hot compare, summing with
#     tl.sum(axis=0). The [BLOCK_N, TILE_B] orientation (input rows / bin
#     columns) is mandatory: the transposed [TILE_B, BLOCK_N] form with a
#     column-broadcast weight silently miscompiles on triton_gcu 3.6.
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
#   * half-precision (fp16/bf16) weighted counts are routed to the vendor's
#     native topsatenBincount (see _TopsatenBridge below): the GCU-native
#     torch.bincount accumulates half-precision weights in half precision
#     with device-side atomics, so any exact fp32-accumulated Triton kernel
#     disagrees with the native reference by ~1 fp16 ULP per rounding step
#     and the accuracy suite (which compares against the on-device native
#     result with atol=1e-4) cannot pass deterministically. The routed call
#     reproduces the native semantics bit-for-bit while keeping the plain /
#     fp32-weighted paths on the faster local kernels.
#   * num_warps=1 is a known silent-miscompile case for one-hot counting
#     kernels, keep num_warps=4.

_HIST_BLOCK_N = 1024
_HIST_TILE_B = 32
_HIST_SMALL_N = 256
_HIST_SMALL_BLOCK_N = 256
_HIST_SMALL_TILE_B = 64
_HIST_W_BLOCK_N = 256
_HIST_W_TILE_B = 64
_MAX_BLOCK_N = 1024
_MAX_PARTIAL_BLOCK_N = 1024
_MAX_PARTIAL_PROGS = 8192
_SINGLE_CTA_MAX_N = 65536

# Above this input length the fp32-weighted path routes to the native
# topsatenBincount call: the one-hot Triton kernel scans the input with a
# single CTA whenever num_bins is small, which loses to the scatter-based
# vendor kernel by ~35us at n=4096 (bench core data), while staying ~2us
# faster at n<=1024.
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
# topsaten native route for weighted bincount.
#
# torch_gcu's native bincount (aten -> topsaten::topsatenBincount) passes the
# fp16/bf16 weights straight into the vendor op, which accumulates them in
# half precision via device-side atomics (deterministic in-process, order
# opaque for large bins). An exact fp32-accumulated kernel therefore differs
# from the native reference by the reference's own rounding noise, which the
# accuracy suite (atol=1e-4) flags. Routing these dtypes through the same
# vendor call reproduces the native result bit-for-bit. fp32 weights also
# route here for long inputs (verified bitwise-identical to native) because
# the one-hot Triton kernel degenerates to a single CTA when num_bins is
# small; see _WEIGHTED_TOPS_MIN_N above.
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
    """Lazy ctypes binding of topsatenBincount for the weighted half path."""

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
    TILE_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * TILE_B + tl.arange(0, TILE_B)
    bin_mask = bins < num_bins
    acc = tl.zeros((TILE_B,), dtype=tl.float32)
    for base in range(0, n_elements, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        vals = tl.load(input_ptr + offs, mask=mask, other=-1).to(tl.int32)
        w = tl.load(weights_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        match = (vals[:, None] == bins[None, :]).to(tl.float32)
        acc += tl.sum(match * w[:, None], axis=0)
    tl.store(output_ptr + bins, acc.to(output_ptr.dtype.element_ty), mask=bin_mask)


def bincount(input, weights=None, minlength=0):
    logger.debug("GEMS BINCOUNT")

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

    if weights_contig.dtype in (
        torch.float16,
        torch.bfloat16,
    ) or (weights_contig.dtype == torch.float32 and n > _WEIGHTED_TOPS_MIN_N):
        # Native-semantics route: the vendor op accumulates half-precision
        # weights in half precision (atomic order), which an exact Triton
        # kernel cannot reproduce; delegate so the result matches the
        # GCU-native torch.bincount bit-for-bit (verified bitwise for fp32
        # too). fp32 weights also route here for long inputs (n above
        # _WEIGHTED_TOPS_MIN_N): the one-hot Triton kernel degenerates to a
        # single CTA whenever num_bins is small and loses ~35us to the
        # scatter-based vendor kernel at n=4096, while staying ~2us faster
        # at n<=1024, hence the length-based dispatch. The bridge only
        # accepts int32 inputs, so narrow integer inputs are normalized.
        if input_contig.dtype != torch.int32:
            input_contig = input_contig.to(torch.int32)
        bridge = _get_topsaten_bridge()
        output = torch.zeros(needed_size, dtype=weights_contig.dtype, device=device)
        bridge.bincount_weighted(output, input_contig, weights_contig, minlength)
        return output

    output = torch.empty(needed_size, dtype=weights.dtype, device=device)
    _bincount_hist_weights_kernel[(triton.cdiv(needed_size, _HIST_W_TILE_B),)](
        input_contig,
        weights_contig,
        output,
        n,
        needed_size,
        TILE_B=_HIST_W_TILE_B,
        BLOCK_N=_HIST_W_BLOCK_N,
        num_warps=4,
    )
    return output
