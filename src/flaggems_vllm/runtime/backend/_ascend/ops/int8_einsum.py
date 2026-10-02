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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

try:
    import triton.experimental.tle as tle
    from triton.experimental.tle.language.dsa.ascend.custom_ops import (
        data_copy_gm_to_l1_nd2nz_int8 as _nd2nz_primitive,
    )
except ImportError:
    _nd2nz_primitive = None
    tle = None


@triton.jit
def _ascend_float_einsum_kernel(
    X,
    Y,
    O,
    B: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    SX: tl.constexpr,
    SY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    h = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    m = (pid // tl.cdiv(D, BLOCK_N)).to(tl.int64) * BLOCK_M
    m += tl.arange(0, BLOCK_M).to(tl.int64)
    n = (pid % tl.cdiv(D, BLOCK_N)).to(tl.int64) * BLOCK_N
    n += tl.arange(0, BLOCK_N).to(tl.int64)
    k = tl.arange(0, 128).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kb in range(tl.cdiv(R, 128)):
        r = kb.to(tl.int64) * 128 + k
        x = tl.load(
            X + m[:, None] * SX[0] + h * SX[1] + r[None, :] * SX[2],
            (m[:, None] < B) & (r[None, :] < R),
            other=0,
        )
        y = tl.load(
            Y + h * SY[0] + n[None, :] * SY[1] + r[:, None] * SY[2],
            (n[None, :] < D) & (r[:, None] < R),
            other=0,
        )
        acc = tl.dot(x, y, acc, allow_tf32=False)
    tl.store(
        O + (m[:, None] * H + h) * D + n[None, :],
        acc,
        (m[:, None] < B) & (n[None, :] < D),
    )


@triton.jit
def _zero_int8_einsum_kernel(Output, N: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * 256 + tl.arange(0, 256)
    tl.store(Output + i, 0.0, i < N)


@triton.jit
def _pack_int8_x_kernel(
    Source,
    Target,
    B: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    ST: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Copy logical BHR input into HBR storage without changing its values."""
    pid = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    nr = tl.cdiv(R, BLOCK_R)
    b = (pid // nr).to(tl.int64) * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
    r = (pid % nr).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    source = Source + b[:, None] * SX[0] + h * SX[1] + r[None, :] * SX[2]
    target = Target + h * ST[0] + b[:, None] * ST[1] + r[None, :] * ST[2]
    if B % BLOCK_B == 0 and R % BLOCK_R == 0:
        value = tl.load(source)
        tl.store(target, value)
    else:
        mask = (b[:, None] < B) & (r[None, :] < R)
        value = tl.load(source, mask, other=0)
        tl.store(target, value, mask)


def _pack_int8_x(x):
    """Make a per-call INT8 copy, returning a logical BHR metadata view."""
    b, h, r = x.shape
    # Wide contiguous rows benefit from longer transfers with fewer row gaps.
    # Four rows keep the copy tile bounded even at the 8192-element cap.
    block_b, block_r = (
        (4, min(8192, triton.next_power_of_2(r))) if r >= 2048 else (64, 128)
    )
    nr = triton.cdiv(r, block_r)
    if h * nr > 32768:
        # Packing is optional; retain the direct path outside this copy grid.
        return x
    storage = torch.empty((h, b, r), dtype=x.dtype, device=x.device)
    chunk_b = (32768 // (h * nr)) * block_b
    for start in range(0, b, chunk_b):
        end = min(start + chunk_b, b)
        source = x[start:end]
        # This view preserves the full storage's head stride b*r. Its base
        # pointer already advances by start*r, including offsets above INT32.
        target = storage[:, start:end, :]
        _pack_int8_x_kernel[(triton.cdiv(end - start, block_b) * nr, h)](
            source,
            target,
            end - start,
            r,
            source.stride(),
            target.stride(),
            block_b,
            block_r,
            num_warps=4,
            num_stages=1,
        )
    return storage.permute(1, 0, 2)


@triton.jit
def _load_int8_nd2nz(P, ROWS: tl.constexpr, ROW_STRIDE: tl.constexpr):
    source = tl.make_block_ptr(
        P, (ROWS, 128), (ROW_STRIDE, 1), (0, 0), (ROWS, 128), (1, 0)
    )
    return tle.dsa.ascend.raw(
        "data_copy_gm_to_l1_nd2nz_int8",
        source,
        1,
        ROWS,
        128,
        0,
        ROW_STRIDE,
        ROWS,
        1,
        1,
        out=tl.full((ROWS, 128), 0, tl.int8),
    )


@triton.jit
def _ascend_block_int8_einsum_kernel(
    X,
    XS,
    Y,
    YS,
    O,
    B: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    SX: tl.constexpr,
    SXS: tl.constexpr,
    SY: tl.constexpr,
    SYS: tl.constexpr,
    SCALE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ALIGNED: tl.constexpr,
    USE_ND2NZ: tl.constexpr = False,
):
    h = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    nn = tl.cdiv(D, BLOCK_N)
    pm = (pid // nn).to(tl.int64)
    pn = (pid % nn).to(tl.int64)
    m = pm * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n = pn * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    k = tl.arange(0, 128).to(tl.int64)
    if BLOCK_N > SCALE_N:
        tl.static_assert(BLOCK_N % SCALE_N == 0)
        ng = pn * (BLOCK_N // SCALE_N)
        ng += tl.arange(0, BLOCK_N // SCALE_N).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N // SCALE_N, SCALE_N), tl.float32)
    else:
        tl.static_assert(SCALE_N % BLOCK_N == 0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kb in range(tl.cdiv(R, 128)):
        # Arbitrary input/scale strides can exceed INT32 even on a small tile.
        kb64 = kb.to(tl.int64)
        r = kb64 * 128 + k
        xp = X + m[:, None] * SX[0] + h * SX[1] + r[None, :] * SX[2]
        yp = Y + h * SY[0] + n[None, :] * SY[1] + r[:, None] * SY[2]
        xsp = XS + m * SXS[0] + h * SXS[1] + kb64 * SXS[2]
        if ALIGNED:
            if USE_ND2NZ:
                x = _load_int8_nd2nz(
                    X + pm * BLOCK_M * SX[0] + h * SX[1] + kb64 * 128,
                    BLOCK_M,
                    SX[0],
                )
                y = _load_int8_nd2nz(
                    Y + h * SY[0] + pn * BLOCK_N * SY[1] + kb64 * 128,
                    BLOCK_N,
                    SY[1],
                ).T
            else:
                x = tl.load(xp)
                y = tl.load(yp)
            xs = tl.load(xsp).to(tl.float32)
        else:
            x = tl.load(xp, (m[:, None] < B) & (r[None, :] < R), other=0)
            y = tl.load(yp, (n[None, :] < D) & (r[:, None] < R), other=0)
            xs = tl.load(xsp, m < B, other=0).to(tl.float32)
        if BLOCK_N > SCALE_N:
            # Wider tiles amortize the loop while retaining each group's scale.
            ys = tl.load(
                YS + h * SYS[0] + ng * SYS[1] + kb64 * SYS[2],
                ng < tl.cdiv(D, SCALE_N),
                other=0,
            ).to(tl.float32)
            row_scale = xs[:, None] * ys[None, :]
        else:
            ysp = YS + h * SYS[0] + (pn * BLOCK_N // SCALE_N) * SYS[1] + kb64 * SYS[2]
            ys = tl.load(ysp).to(tl.float32)
            row_scale = xs * ys
        # A K128 signed INT8 partial fits exactly in INT32. Apply this block's
        # FP32 scales before accumulating the next block in FP32.
        partial = tl.dot(x, y, out_dtype=tl.int32).to(tl.float32)
        if BLOCK_N > SCALE_N:
            partial = partial.reshape((BLOCK_M, BLOCK_N // SCALE_N, SCALE_N))
            acc = tl.fma(partial, row_scale[:, :, None], acc)
        else:
            acc = tl.fma(partial, row_scale[:, None], acc)
    if BLOCK_N > SCALE_N:
        acc = acc.reshape((BLOCK_M, BLOCK_N))
    output = O + (m[:, None] * H + h) * D + n[None, :]
    if ALIGNED:
        tl.store(output, acc)
    else:
        tl.store(output, acc, (m[:, None] < B) & (n[None, :] < D))


def _prepare_int8_einsum(
    equation: str,
    x: torch.Tensor,
    xs: torch.Tensor | None,
    y: torch.Tensor,
    ys: torch.Tensor | None,
    block_size=(128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Validate the Ascend inference contract and allocate its contiguous output."""
    if equation != "bhr,hdr->bhd":
        raise ValueError("int8_einsum only supports 'bhr,hdr->bhd'")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("int8_einsum inputs must have three dimensions")
    b, h, r = x.shape
    if y.shape[0] != h or y.shape[2] != r or x.device != y.device:
        raise ValueError("int8_einsum input shape or device mismatch")
    if x.dtype != y.dtype:
        raise TypeError("int8_einsum inputs must have matching dtypes")
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain block_n and block_k")
    floating_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if output_dtype not in floating_dtypes:
        raise TypeError("unsupported output dtype")
    if x.layout != torch.strided or y.layout != torch.strided:
        raise ValueError("int8_einsum inputs must use strided layouts")
    if x.requires_grad or y.requires_grad:
        raise NotImplementedError("int8_einsum does not support autograd")

    d = y.shape[1]
    is_int8 = x.dtype == torch.int8
    sn, sk = block_size
    if is_int8:
        if xs is None or ys is None:
            raise ValueError("INT8 int8_einsum requires both scale tensors")
        if sn < 16 or sn & (sn - 1) or sk != 128:
            raise NotImplementedError(
                "requires power-of-two block_n >= 16 and block_k=128"
            )
        if xs.shape != (b, h, triton.cdiv(r, sk)) or ys.shape != (
            h,
            triton.cdiv(d, sn),
            triton.cdiv(r, sk),
        ):
            raise ValueError("incorrect int8_einsum scale shape")
        if any(t.device != x.device or t.layout != torch.strided for t in (xs, ys)):
            raise ValueError(
                "scale tensors must have strided layouts on the input device"
            )
        if any(t.dtype not in floating_dtypes for t in (xs, ys)):
            raise TypeError("scales must be float32, bfloat16 or float16")
        if xs.requires_grad or ys.requires_grad:
            raise NotImplementedError("int8_einsum does not support autograd")
    elif x.dtype not in floating_dtypes:
        raise TypeError("unsupported int8_einsum input dtype")
    elif xs is not None or ys is not None:
        raise ValueError(
            "floating int8_einsum inputs must not have quantization scales"
        )

    return torch.empty((b, h, d), device=x.device, dtype=output_dtype)


def int8_einsum(
    equation: str,
    x: torch.Tensor,
    xs: torch.Tensor | None,
    y: torch.Tensor,
    ys: torch.Tensor | None,
    block_size=(128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute block-scaled INT8 or floating einsum with FP32 accumulation.

    INT8 tiles use up to 256 rows, or wider columns for small batches while
    preserving independent weight-scale groups. Selected input strides use a
    per-call INT8 copy into HBR storage; scales keep their original layout. Floating inputs use a local BF16/FP16/FP32
    compatibility kernel.
    No-copy batch slices bound the logical launch grid; all device offsets use
    INT64. This backend owns validation, output allocation and zero reduction.
    """
    if x.device.type != "npu":
        raise NotImplementedError("the Ascend int8_einsum path requires NPU inputs")
    out = _prepare_int8_einsum(equation, x, xs, y, ys, block_size, output_dtype)
    if out.numel() == 0:
        return out
    b, h, r = x.shape
    d = y.shape[1]
    is_int8 = x.dtype == torch.int8
    with torch_device_fn.device(x.device):
        if r == 0:
            flat = out.view(-1)
            for start in range(0, out.numel(), 32768 * 256):
                chunk = flat[start : start + 32768 * 256]
                _zero_int8_einsum_kernel[(triton.cdiv(chunk.numel(), 256),)](
                    chunk, chunk.numel(), num_warps=4
                )
            return out
        bm = (
            max(16, min(256 if b >= 256 else 128, triton.next_power_of_2(b)))
            if is_int8
            else (16 if b < 32 else 32)
        )
        bn = min(block_size[0], 128) if is_int8 else 128
        if (
            is_int8
            and block_size[0] == 128
            and b <= 32
            and d >= 512
            and x.stride(2) == 1
            and y.stride(2) == 1
        ):
            bn = 512
        nn = triton.cdiv(d, bn)
        if nn * h > 32768:
            raise NotImplementedError("too many head/column tiles for an Ascend launch")
        # Packing amortizes for wide row strides once at least 16 rows are used.
        if is_int8 and b >= 16 and x.stride(0) >= 65536 and x.stride(2) == 1:
            x = _pack_int8_x(x)
        # Native ND2NZ benefits the larger row tiles. The 32-byte layout gate
        # bounds the validated fast path; other layouts retain Triton loads.
        native_layout = (
            is_int8
            and _nd2nz_primitive is not None
            and bm >= 128
            and bn == 128
            and x.stride(2) == y.stride(2) == 1
            and 0 < x.stride(0) <= 65535
            and 0 < y.stride(1) <= 65535
            and x.data_ptr() % 32 == y.data_ptr() % 32 == 0
            and x.stride(0) % 32 == x.stride(1) % 32 == 0
            and y.stride(0) % 32 == y.stride(1) % 32 == 0
        )
        chunk_b = (32768 // (nn * h)) * bm
        for start in range(0, b, chunk_b):
            end = min(start + chunk_b, b)
            xc = x[start:end]
            oc = out[start:end]
            grid = (triton.cdiv(end - start, bm) * nn, h)
            if is_int8:
                xsc = xs[start:end]
                aligned = (end - start) % bm == 0 and d % bn == 0 and r % 128 == 0
                use_nd2nz = aligned and native_layout
                options = {}
                if aligned and bm == 256 and bn == 128:
                    options["unit_flag"] = True
                if use_nd2nz:
                    # CANN 9.1's newer mixed-core pass misinfers the memory
                    # space of MTE2 custom outputs; use its supported old pass.
                    options["enable_legacy_insert_load_store_for_mix_cv"] = True
                _ascend_block_int8_einsum_kernel[grid](
                    xc,
                    xsc,
                    y,
                    ys,
                    oc,
                    end - start,
                    h,
                    r,
                    d,
                    xc.stride(),
                    xsc.stride(),
                    y.stride(),
                    ys.stride(),
                    block_size[0],
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    ALIGNED=aligned,
                    USE_ND2NZ=use_nd2nz,
                    num_warps=4,
                    num_stages=2 if bm >= 128 else 1,
                    **options,
                )
            else:
                _ascend_float_einsum_kernel[grid](
                    xc,
                    y,
                    oc,
                    end - start,
                    h,
                    r,
                    d,
                    xc.stride(),
                    y.stride(),
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    num_stages=1,
                )
    return out
