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

"""SwiGLU fused with group quantization for Ascend 950DT (A5).

Replaces ``aclnnSwigluGroupQuant`` for the fp8 output modes, as called by
vllm-ascend ``ops/fused_moe/shared_experts.py`` and
``quantization/methods/w4a8/w4a8_mxfp4.py``.  Note the mode numbering in
vllm-ascend is shifted by one: its ``quant_mode=2`` is MX, i.e. ``quant_mode=1``
here and in the CANN interface.

Computation, per token row, with ``H = D / 2``::

    x0, x1   = x[..., :H], x[..., H:]
    x0, x1   = min(x0, c), clamp(x1, -c, c)           # if clamp_limit > 0
    y_origin = x0 / (1 + exp(-x0)) * x1               # SwiGLU
    y        = y_origin * weight[row]                  # if weight given
    amax     = max(|y|) over each block, >= 1e-4       # block = 128 or 32
    y_fp8    = round_to_fp8(y / scale),  scale ~ amax / fp8_max

Scale shape by ``(quant_mode, round_scale)``:

    mode 0, round_scale=False   fp32 scale = amax/fp8_max, block 128
    mode 0, round_scale=True    fp32 scale rounded up to a power of 2
    mode 1, round_scale=True    E8M0 exponent, block 32 (MX mode, vllm default)

Modes 2 and 3 (HiFloat8) are not implemented; use the vendor there.

Accuracy contract: verified byte-exact against ``aclnnSwigluGroupQuant`` for
both input dtypes and both fp8 output types, with and without weight / clamp /
group_index, and on inf / nan / zero / saturating edge blocks.

``tl.exp`` lowers to the same vector instruction as the vendor's ``VFSwiGlu``,
so mode-0 fp32 scales match at 0 ULP.  A torch-on-NPU reference may differ by
up to 3 ULP on that path because ``aclnnExp`` differs; the tolerances in the
test suite exist for that reason.

Constraints:

  * A column chunk narrower than 1024 costs 7-20x unless it covers the whole
    row (DMA descriptor cliff, not arithmetic).  ``tile_cols`` enforces this.
  * A chunk wider than H requires a column mask and costs ~2x; ``tile_cols``
    caps at the largest power-of-two <= H.
  * ``group_index`` is summed inside the kernel to avoid a host-side device
    sync, which would cost more than the whole kernel and break ACL graph
    capture.
  * The sign-of-zero fixup passes ``ZERO=0.0`` as a runtime argument rather
    than a literal, because a literal ``y + 0.0`` is folded away as an
    identity by the compiler.
"""

import torch
import triton
import triton.language as tl
from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream

__all__ = [
    "swiglu_group_quant",
    "SwigluGroupQuantPlan",
    "scale_shape",
    "pick_tile",
    "should_use_triton",
    "est_kernel_us",
    "vendor_est_us",
    "tile_cols",
]

# The kernel multiplies amax by a rounded fp32 1/fp8max (INV_FP8_E4M3_MAX_VALUE
# / INV_FP8_E5M2_MAX_VALUE in the vendor header).  Reproducing the exact bit
# pattern matters: it decides the scale exponent at the boundary, and a division
# would round differently.
_INV_FP8_MAX = {
    torch.float8_e4m3fn: 0x3B124925,
    torch.float8_e5m2: 0x37924925,
}
_FP8_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
_BLOCK = {0: 128, 1: 32}
_IN_DTYPES = (torch.bfloat16, torch.float16)
# Accept uint8 as the byte-compatible buffer type for E8M0 scales.
_E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
_EPS = 1e-4
_CORES = 64

# Largest tile in output elements (rows * cols).  A tile holds x0, x1 as fp32,
# the fp32 SwiGLU, and the fp8 output (~16 bytes per output element); 16384 is
# the largest that compiles at every (rows, cols) shape.
_UB_TILE_ELEMS = 16384
_ROWS = (1, 2, 4, 8, 16, 32)
# A column chunk narrower than this costs 7-20x unless it covers the whole row
# (DMA descriptor cliff).  See tile_cols.
_MIN_STRIDED_COLS = 1024
_DISPATCH_US = 16.0  # aclnn host dispatch cost in eager mode


@triton.jit
def _sgq_kernel(
    x_ptr,
    w_ptr,
    gi_ptr,
    y_ptr,
    s_ptr,
    yo_ptr,
    M,
    ZERO,
    D: tl.constexpr,
    H: tl.constexpr,
    NBLK: tl.constexpr,
    NSP: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    B: tl.constexpr,
    CB: tl.constexpr,
    INVB: tl.constexpr,
    FMAX: tl.constexpr,
    MX: tl.constexpr,
    RS: tl.constexpr,
    HASW: tl.constexpr,
    HASGI: tl.constexpr,
    CLAMP: tl.constexpr,
    CL: tl.constexpr,
    E5: tl.constexpr,
    ORIGIN: tl.constexpr,
    RMASK: tl.constexpr,
    STEPS: tl.constexpr,
    OVERLAP: tl.constexpr,
    PROGS: tl.constexpr,
):
    """One program walks ROWS rows at a time, striding by PROGS * ROWS.

    ``ZERO`` is always 0.0 and is an argument rather than a constant on
    purpose; see the sign-of-zero fixup in the inner loop.

    ``COLS`` columns of the output are handled per inner step, ``CB = COLS / B``
    blocks, ``STEPS`` steps to cover a row.  ``RMASK`` is set when the row
    count is not a multiple of ROWS or when ``group_index`` makes it dynamic.
    ``OVERLAP`` is set when COLS does not divide H, which happens whenever H
    is not a power of two (e.g. D = 2560 -> H = 1280); see the inner loop.
    """
    pid = tl.program_id(0)
    N: tl.constexpr = ROWS * CB
    # 1/fp8max as a bit pattern so the fp32 constant matches the vendor exactly.
    inv_fp8 = tl.full((), INVB, tl.int32).to(tl.float32, bitcast=True)

    real = M
    if HASGI:
        # Sum group_index inside the kernel to avoid a host-side device sync.
        g = tl.arange(0, GP)
        gv = tl.load(gi_ptr + g, mask=g < G, other=0).to(tl.int32)
        real = tl.minimum(tl.sum(gv), M)
        real = tl.maximum(real, 0)

    c = tl.arange(0, COLS)
    bl = tl.arange(0, CB)
    for r0 in range(pid * ROWS, real, PROGS * ROWS):
        r = r0 + tl.arange(0, ROWS)
        rm = (r < real)[:, None]
        xb = x_ptr + r[:, None] * D
        if HASW:
            # Load weight once per row group, not once per column chunk.
            wv = tl.load(w_ptr + r, mask=r < real, other=0.0)[:, None]
        for ci in range(0, STEPS):
            # When COLS does not divide H, the last chunk overlaps the previous
            # one instead of running past the row end with a column mask.
            # A column mask stops the DMA using its contiguous path and costs
            # ~20x (12.6 us at H=2048 becomes 274 us at H=1280).  Overlapping
            # recomputes some blocks, which is safe because each block's amax,
            # scale and payload depend only on that block.
            c0 = ci * COLS
            if OVERLAP:
                c0 = min(c0, H - COLS)
            off = (c0 + c)[None, :]
            if RMASK:
                x0 = tl.load(xb + off, mask=rm, other=0.0).to(tl.float32)
                x1 = tl.load(xb + H + off, mask=rm, other=0.0).to(tl.float32)
            else:
                x0 = tl.load(xb + off).to(tl.float32)
                x1 = tl.load(xb + H + off).to(tl.float32)
            if CLAMP:
                x0 = tl.minimum(x0, CL)
                x1 = tl.minimum(tl.maximum(x1, -CL), CL)
            # Spelled as the vendor's VFSwiGlu (neg, exp, add, div, mul);
            # x0 * sigmoid(x0) differs by up to 2 ULP and would lose byte parity.
            y = (x0 / (tl.exp(-x0) + 1.0)) * x1
            if ORIGIN:
                # y_origin is the SwiGLU before the per-row weight, in x's dtype.
                yov = y.to(yo_ptr.dtype.element_ty)
                if RMASK:
                    tl.store(yo_ptr + r[:, None] * H + off, yov, mask=rm)
                else:
                    tl.store(yo_ptr + r[:, None] * H + off, yov)
            if HASW:
                y = y * wv
            # Sign-of-zero fixup: -0.0 + 0.0 == +0.0.  ZERO must be a runtime
            # argument because the compiler folds a literal `y + 0.0` as an
            # identity (leaving 0x80 where the vendor gives 0x00).
            # Note: y_origin is stored before this, because the vendor's
            # y_origin keeps the -0.0 (bf16 bits 0x8000).
            y = y + ZERO

            yt = tl.reshape(y, (N, B))
            if MX:
                amax = tl.max(tl.abs(yt), 1)
            else:
                # mode 0: mask inf/nan out of the amax.
                fin = (yt * 0.0) == (yt * 0.0)
                amax = tl.max(tl.where(fin, tl.abs(yt), 0.0), 1)
            amax = tl.maximum(amax, 1e-4)
            sf = amax * inv_fp8
            if RS:
                sb = sf.to(tl.int32, bitcast=True)
                # ceil(log2(sf)) from fp32 fields: exponent - 127, plus one if
                # any mantissa bit is set.
                e = ((sb >> 23) & 0xFF) - 127 + ((sb & 0x7FFFFF) != 0).to(tl.int32)
                inv = (((127 - e) & 0xFF) << 23).to(tl.float32, bitcast=True)
            else:
                inv = FMAX / amax

            q = yt * inv[:, None]
            if not MX:
                # A scaled value that overflowed to inf falls back to unscaled.
                q = tl.where((q * 0.0) == (q * 0.0), q, yt)
            q = tl.reshape(q, (ROWS, COLS))
            if E5:
                q8 = q.to(tl.float8e5)
            else:
                q8 = q.to(tl.float8e4nv)

            yo = y_ptr + r[:, None] * H + off
            sp = (c0 // B + bl)[None, :]
            if MX:
                sv = tl.reshape((e + 127).to(tl.uint8), (ROWS, CB))
                so = s_ptr + r[:, None] * NSP + sp
            elif RS:
                sv = tl.reshape(
                    (((e + 127) & 0xFF) << 23).to(tl.float32, bitcast=True),
                    (ROWS, CB),
                )
                so = s_ptr + r[:, None] * NBLK + sp
            else:
                sv = tl.reshape(sf, (ROWS, CB))
                so = s_ptr + r[:, None] * NBLK + sp
            if RMASK:
                tl.store(yo, q8, mask=rm)
                tl.store(so, sv, mask=rm)
            else:
                tl.store(yo, q8)
                tl.store(so, sv)


def scale_shape(shape, quant_mode, round_scale=None):
    """Vendor scale layout for ``x`` of shape ``shape``.

    mode 1 (MX): uint8 viewed as E8M0, ``[..., ceil(nblk/2), 2]``.
    mode 0: fp32 ``[..., nblk]``.
    """
    H = shape[-1] // 2
    nblk = -(-H // _BLOCK[quant_mode])
    if quant_mode == 1:
        return tuple(shape[:-1]) + ((nblk + 1) // 2, 2)
    return tuple(shape[:-1]) + (nblk,)


def pick_tile(M, H, quant_mode=1):
    """Return (rows, cols) per tile that minimises the modelled kernel cost."""
    best, cost = None, None
    for rows in _ROWS:
        for cols in tile_cols(H, quant_mode):
            if rows * cols <= _UB_TILE_ELEMS:
                c = est_kernel_us(M, H, rows, cols)
                if cost is None or c < cost:
                    best, cost = (rows, cols), c
    if best is None:
        raise ValueError(f"no tile fits for H={H} (block {_BLOCK[quant_mode]})")
    return best


def tile_cols(H, quant_mode=1):
    """Legal column widths per inner step, widest first.

    A chunk must be a power-of-two and at most H (a wider chunk needs a column
    mask, which costs ~2x).  The floor is 1024: a strided chunk narrower than
    that costs 7-20x unless it covers the whole row (DMA descriptor cliff).
    ``top`` (the full row width, largest power-of-two <= H) is always offered
    even when it is below the floor, so small H values (128/256/512) still work.
    """
    B = _BLOCK[quant_mode]
    top = 1 << (H.bit_length() - 1)  # largest power of two <= H
    out = [top]
    c = top // 2
    while c >= _MIN_STRIDED_COLS and c >= B:
        out.append(c)
        c //= 2
    return out


def est_kernel_us(M, H, rows, cols):
    """Estimated kernel time (us): pipeline fill plus steady-state steps.

    Steps after the first pipeline against each other, so cost scales with
    steps per core rather than waves of launches.  The fill scales with
    occupied cores (concurrent starts contend).

    Max relative error is 12% over 63 measured cells; picks a tile within
    1.05x of the measured best in all 16 (T, H) cells tested.
    """
    groups = -(-M // rows)
    cores = min(groups, _CORES)
    waves = -(-groups // _CORES)
    # ceil so a cols that does not divide H prices its overlapping last chunk
    # as a full step.
    steps = waves * -(-H // cols)
    return 1.20 + 0.024 * cores + steps * (0.14 + 0.242e-3 * rows * cols)


def should_use_triton(M, H, quant_mode=1, graph_captured=False):
    """Return True when this kernel is at least as fast as the vendor.

    The one known loss is H=1280 (D=2560) at large T (from T=768): the widest
    legal chunk (1024) covers 2048 columns in two steps and 37.5% of the work
    is redundant.  At small T the tile is not the bottleneck (T=256 measures
    1.01x), so the cutoff is on T as well as H.

    ``graph_captured`` controls what the vendor is charged for: False adds the
    ~16 us of aclnn host dispatch it pays in eager mode; True compares kernel
    time only (what survives ACL graph capture).
    """
    rows, cols = pick_tile(M, H, quant_mode)
    est = est_kernel_us(M, H, rows, cols)
    vend = vendor_est_us(M, H)
    return est < 1.10 * (vend if graph_captured else vend + _DISPATCH_US)


def vendor_est_us(M, H):
    """Vendor kernel time estimate (us), fitted to the same measurements.

    Same shape as est_kernel_us: both sides are transcendental-bound and scale
    identically, differing only in the constants.
    """
    cores = min(M, _CORES)
    return 2.50 + 0.029 * cores + 0.270e-3 * M * H / _CORES


def _validate(
    x, weight, group_index, dst_type, quant_mode, block_size, round_scale, clamp_limit
):
    if not x.is_npu:
        raise ValueError("x must be an NPU tensor")
    if x.dtype not in _IN_DTYPES:
        raise ValueError(
            f"x dtype {x.dtype} not supported for quant_mode "
            f"{quant_mode}; expected float16 or bfloat16"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims, got {tuple(x.shape)}")
    if quant_mode not in _BLOCK:
        raise ValueError(
            f"quant_mode {quant_mode} not supported (2 and 3 are HiFloat8, "
            "which Triton cannot emit; use the vendor)"
        )
    if dst_type not in _FP8_MAX:
        raise ValueError(
            f"dst_type {dst_type} not supported; expected "
            "float8_e4m3fn or float8_e5m2"
        )
    if quant_mode == 1 and not round_scale:
        raise ValueError("quant_mode 1 (MX) requires round_scale=True")
    D = x.shape[-1]
    if D % 256:
        # Same constraint the vendor enforces (it fails at runtime otherwise).
        raise ValueError(f"last dim {D} must be a multiple of 256")
    if block_size not in (0, _BLOCK[quant_mode]):
        raise ValueError(
            f"block_size {block_size} invalid for quant_mode "
            f"{quant_mode}; expected 0 or {_BLOCK[quant_mode]}"
        )
    if (
        clamp_limit is not None
        and float(clamp_limit) != -1.0
        and float(clamp_limit) <= 0.0
    ):
        raise ValueError(
            f"clamp_limit must be -1.0 (off) or > 0, got " f"{clamp_limit}"
        )
    T = x.numel() // D
    if weight is not None:
        if weight.dtype != torch.float32:
            raise ValueError(f"weight must be float32, got {weight.dtype}")
        if weight.numel() != T:
            raise ValueError(f"weight has {weight.numel()} elements, expected " f"{T}")
    if group_index is not None:
        if group_index.dtype != torch.int64:
            raise ValueError(f"group_index must be int64, got " f"{group_index.dtype}")
        if group_index.dim() != 1:
            raise ValueError("group_index must be 1-D")


class SwigluGroupQuantPlan:
    """Prepare once, launch many: resolve the tile, compile, bind a raw launch.

    Holding the plan removes the Triton dispatch from the steady-state path,
    which matters because this operator runs for only a few microseconds.
    """

    __slots__ = (
        "T",
        "D",
        "H",
        "nblk",
        "nsp",
        "rows",
        "cols",
        "grid",
        "dtype",
        "dst_type",
        "quant_mode",
        "round_scale",
        "clamp_limit",
        "output_origin",
        "has_w",
        "has_gi",
        "g",
        "masked",
        "launch",
        "_shape",
    )

    def __init__(
        self,
        x,
        weight=None,
        group_index=None,
        dst_type=torch.float8_e4m3fn,
        quant_mode=1,
        block_size=0,
        round_scale=None,
        clamp_limit=-1.0,
        output_origin=False,
        rows=None,
        cols=None,
    ):
        if round_scale is None:
            round_scale = quant_mode == 1
        _validate(
            x,
            weight,
            group_index,
            dst_type,
            quant_mode,
            block_size,
            round_scale,
            clamp_limit,
        )
        self._shape = tuple(x.shape)
        self.D = x.shape[-1]
        self.H = self.D // 2
        self.T = x.numel() // self.D
        self.dtype = x.dtype
        self.dst_type = dst_type
        self.quant_mode = quant_mode
        self.round_scale = bool(round_scale)
        self.clamp_limit = float(clamp_limit if clamp_limit is not None else -1.0)
        self.output_origin = bool(output_origin)
        self.has_w = weight is not None
        self.has_gi = group_index is not None
        self.g = group_index.numel() if self.has_gi else 0
        B = _BLOCK[quant_mode]
        self.nblk = -(-self.H // B)
        self.nsp = (self.nblk + 1) // 2 * 2

        clamp_on = self.clamp_limit != -1.0
        cands = []
        if rows and cols:
            cands.append((rows, cols))
        tile = pick_tile(self.T, self.H, quant_mode)
        legal = [
            (r, c)
            for r in _ROWS
            for c in tile_cols(self.H, quant_mode)
            if r * c <= _UB_TILE_ELEMS
        ]
        legal.sort(key=lambda rc: est_kernel_us(self.T, self.H, *rc))
        for rc in [tile] + legal:
            if rc not in cands:
                cands.append(rc)

        err = None
        for r, c in cands:
            rmask = 1 if (self.T % r or self.has_gi) else 0
            overlap = 1 if self.H % c else 0
            steps = -(-self.H // c)
            grid = min(triton.cdiv(self.T, r), _CORES)
            gp = max(8, triton.next_power_of_2(max(self.g, 1)))
            # ZERO is a runtime argument so the sign-of-zero fixup in the kernel
            # survives the compiler (a constexpr 0.0 is folded away as an identity).
            args = (
                self.T,
                0.0,
                self.D,
                self.H,
                self.nblk,
                self.nsp,
                self.g,
                gp,
                r,
                c,
                B,
                c // B,
                _INV_FP8_MAX[dst_type],
                _FP8_MAX[dst_type],
                1 if quant_mode == 1 else 0,
                1 if round_scale else 0,
                1 if self.has_w else 0,
                1 if self.has_gi else 0,
                1 if clamp_on else 0,
                self.clamp_limit,
                1 if dst_type == torch.float8_e5m2 else 0,
                1 if self.output_origin else 0,
                rmask,
                steps,
                overlap,
                grid,
            )
            y = torch.empty((self.T, self.H), dtype=dst_type, device=x.device)
            s = self._new_scale(x.device)
            yo = (
                torch.empty((self.T, self.H), dtype=x.dtype, device=x.device)
                if self.output_origin
                else y
            )
            try:
                # JITFunction.run returns the CompiledKernel for this
                # specialization.  Do not fish in device_caches for the newest
                # entry: on a cache hit nothing is added, so that could bind a
                # kernel compiled for a different mode.
                ck = _sgq_kernel[(grid,)](
                    x.view(self.T, self.D),
                    weight,
                    group_index,
                    y,
                    self._scale_bytes(s, (self.T, self.nsp)) if quant_mode == 1 else s,
                    yo,
                    *args,
                )
                torch.npu.synchronize()
            except Exception as e:  # bisheng crash / UB overflow
                err = e
                continue
            self.rows, self.cols, self.grid = r, c, grid
            self.masked = (rmask, overlap)
            dev = triton.runtime.driver.active.get_current_device()
            ck = ck.result() if hasattr(ck, "result") else ck
            fn, meta = ck.function, ck.packed_metadata

            def launch(
                x,
                w,
                gi,
                y,
                s,
                yo,
                _ck=ck,
                _fn=fn,
                _meta=meta,
                _grid=grid,
                _dev=dev,
                _args=args,
            ):
                _ck.run(
                    _grid,
                    1,
                    1,
                    _raw_stream(_dev),
                    _fn,
                    _meta,
                    None,
                    None,
                    None,
                    x,
                    w,
                    gi,
                    y,
                    s,
                    yo,
                    *_args,
                )

            self.launch = launch
            return
        raise RuntimeError(
            f"no tile compiles for T={self.T} H={self.H} mode={quant_mode}: "
            f"{type(err).__name__}: {str(err)[:200] if err else 'none fit UB'}"
        )

    def _new_scale(self, device):
        if self.quant_mode == 1:
            # torch_npu cannot zero an E8M0 tensor directly; allocate as uint8
            # then reinterpret without a compute kernel.
            scale = torch.zeros((self.T, self.nsp), dtype=torch.uint8, device=device)
            return scale.view(_E8M0) if _E8M0 != torch.uint8 else scale
        return torch.empty((self.T, self.nblk), dtype=torch.float32, device=device)

    @staticmethod
    def _scale_bytes(scale, shape):
        # Triton does not know E8M0; pass a uint8 view and keep the public
        # dtype on the returned tensor only.
        if scale.numel() != shape[0] * shape[1]:
            raise ValueError(
                f"scale has {scale.numel()} elements, expected "
                f"{shape[0] * shape[1]} for the vendor layout"
            )
        if scale.dtype == _E8M0 and _E8M0 != torch.uint8:
            scale = scale.view(torch.uint8)
        return scale.view(*shape)

    def __call__(
        self, x, weight=None, group_index=None, out=None, scale=None, y_origin=None
    ):
        if tuple(x.shape) != self._shape or x.dtype != self.dtype:
            raise ValueError(
                f"plan is for {self._shape} {self.dtype}, got "
                f"{tuple(x.shape)} {x.dtype}"
            )
        if out is None:
            out = torch.empty((self.T, self.H), dtype=self.dst_type, device=x.device)
        if scale is None:
            scale = self._new_scale(x.device)
        if y_origin is None and self.output_origin:
            y_origin = torch.empty((self.T, self.H), dtype=self.dtype, device=x.device)
        if self.quant_mode == 1 and scale.dtype not in (
            torch.uint8,
            _E8M0,
        ):
            raise ValueError(f"MX scale must be {_E8M0} or uint8, got {scale.dtype}")
        if self.quant_mode == 0 and scale.dtype != torch.float32:
            raise ValueError(f"mode-0 scale must be float32, got {scale.dtype}")
        sv = self._scale_bytes(
            scale, (self.T, self.nsp if self.quant_mode == 1 else self.nblk)
        )
        result_scale_dtype = scale.dtype
        self.launch(
            x.view(self.T, self.D),
            weight,
            group_index,
            out.view(self.T, self.H),
            sv,
            (y_origin if self.output_origin else out).view(self.T, self.H),
        )
        shp = self._shape[:-1]
        return (
            out.view(*shp, self.H),
            sv.view(result_scale_dtype).view(scale_shape(self._shape, self.quant_mode)),
            y_origin.view(*shp, self.H) if self.output_origin else None,
        )

    def __repr__(self):
        return (
            f"SwigluGroupQuantPlan(T={self.T}, H={self.H}, "
            f"rows={self.rows}, cols={self.cols}, grid={self.grid}, "
            f"rowmask={bool(self.masked[0])}, "
            f"overlap={bool(self.masked[1])}, mode={self.quant_mode}, "
            f"round_scale={self.round_scale}, clamp={self.clamp_limit}, "
            f"{self.dtype} -> {self.dst_type})"
        )


def swiglu_group_quant(
    x,
    weight=None,
    group_index=None,
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    block_size=0,
    round_scale=None,
    clamp_limit=-1.0,
    output_origin=False,
    out=None,
    scale=None,
    y_origin=None,
):
    """One-shot: same result as ``aclnnSwigluGroupQuant``, as ``(y, scale,
    y_origin)``.

    ``y_origin`` is ``None`` unless ``output_origin=True`` (the vendor returns
    an empty tensor there; this returns None so the difference is visible).

    With ``group_index``, rows past ``sum(group_index)`` are left as they were
    in ``out`` / ``scale``, which is what the vendor does -- pass your own
    buffers if that tail matters.  MX scale defaults to ``float8_e8m0fnu``;
    a ``uint8`` ``scale=`` buffer is accepted as packed E8M0 bytes.
    """
    plan = SwigluGroupQuantPlan(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        block_size,
        round_scale,
        clamp_limit,
        output_origin,
    )
    return plan(x, weight, group_index, out=out, scale=scale, y_origin=y_origin)
