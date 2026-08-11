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
from triton.language.extra import libdevice


@triton.jit
def _scaled_int8_quant_kernel(
    in_ptr,
    out_ptr,
    meta_ptr,
    meta2_ptr,
    n_cols,
    n_elem,
    DYN: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    if DYN:
        row = pid
        base = row * n_cols
        if n_cols <= BLOCK:
            # Single-load register-resident path: no second HBM read.
            idx = offs
            if MASKED:
                m = idx < n_cols
                x = tl.load(in_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
            else:
                x = tl.load(in_ptr + base + idx).to(tl.float32)
            if SYM:
                absmax = tl.max(tl.abs(x), axis=0)
                inv = tl.where(absmax > 0.0, 127.0 / absmax, 0.0)
                tl.store(meta_ptr + row, absmax / 127.0)
                q = libdevice.rint(x * inv)
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                if MASKED:
                    tl.store(out_ptr + base + idx, q.to(tl.int8), mask=m)
                else:
                    tl.store(out_ptr + base + idx, q.to(tl.int8))
            else:
                if MASKED:
                    xm = tl.where(m, x, -float("inf"))
                    xn = tl.where(m, x, float("inf"))
                else:
                    xm = x
                    xn = x
                rmax = tl.max(xm, axis=0)
                rmin = tl.min(xn, axis=0)
                scale = (rmax - rmin) / 255.0
                azp = libdevice.rint(-128.0 - rmin / scale)
                azp = tl.minimum(tl.maximum(azp, -2147483648.0), 2147483647.0)
                tl.store(meta_ptr + row, scale)
                tl.store(meta2_ptr + row, azp.to(tl.int32))
                q = libdevice.rint(x / scale) + azp
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                if MASKED:
                    tl.store(out_ptr + base + idx, q.to(tl.int8), mask=m)
                else:
                    tl.store(out_ptr + base + idx, q.to(tl.int8))
        elif SYM:
            acc = tl.full([BLOCK], -float("inf"), tl.float32)
            for c0 in range(0, tl.cdiv(n_cols, BLOCK)):
                idx = c0 * BLOCK + offs
                if MASKED:
                    m = idx < n_cols
                    x = tl.load(in_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
                else:
                    x = tl.load(in_ptr + base + idx).to(tl.float32)
                acc = tl.maximum(acc, tl.abs(x))
            absmax = tl.max(acc, axis=0)
            inv = tl.where(absmax > 0.0, 127.0 / absmax, 0.0)
            tl.store(meta_ptr + row, absmax / 127.0)
            for c0 in range(0, tl.cdiv(n_cols, BLOCK)):
                idx = c0 * BLOCK + offs
                if MASKED:
                    m = idx < n_cols
                    x = tl.load(in_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
                else:
                    x = tl.load(in_ptr + base + idx).to(tl.float32)
                q = libdevice.rint(x * inv)
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                if MASKED:
                    tl.store(out_ptr + base + idx, q.to(tl.int8), mask=m)
                else:
                    tl.store(out_ptr + base + idx, q.to(tl.int8))
        else:
            acc_max = tl.full([BLOCK], -float("inf"), tl.float32)
            acc_min = tl.full([BLOCK], float("inf"), tl.float32)
            for c0 in range(0, tl.cdiv(n_cols, BLOCK)):
                idx = c0 * BLOCK + offs
                if MASKED:
                    m = idx < n_cols
                    x = tl.load(in_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
                    xm = tl.where(m, x, -float("inf"))
                    xn = tl.where(m, x, float("inf"))
                else:
                    x = tl.load(in_ptr + base + idx).to(tl.float32)
                    xm = x
                    xn = x
                acc_max = tl.maximum(acc_max, xm)
                acc_min = tl.minimum(acc_min, xn)
            rmax = tl.max(acc_max, axis=0)
            rmin = tl.min(acc_min, axis=0)
            scale = (rmax - rmin) / 255.0
            azp = libdevice.rint(-128.0 - rmin / scale)
            azp = tl.minimum(tl.maximum(azp, -2147483648.0), 2147483647.0)
            tl.store(meta_ptr + row, scale)
            tl.store(meta2_ptr + row, azp.to(tl.int32))
            for c0 in range(0, tl.cdiv(n_cols, BLOCK)):
                idx = c0 * BLOCK + offs
                if MASKED:
                    m = idx < n_cols
                    x = tl.load(in_ptr + base + idx, mask=m, other=0.0).to(tl.float32)
                else:
                    x = tl.load(in_ptr + base + idx).to(tl.float32)
                q = libdevice.rint(x / scale) + azp
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                if MASKED:
                    tl.store(out_ptr + base + idx, q.to(tl.int8), mask=m)
                else:
                    tl.store(out_ptr + base + idx, q.to(tl.int8))
    else:
        idx = pid * BLOCK + offs
        if MASKED:
            m = idx < n_elem
            if SYM:
                s = tl.load(meta_ptr).to(tl.float32)
                x = tl.load(
                    in_ptr + idx, mask=m, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                q = libdevice.rint(x / s)
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                tl.store(
                    out_ptr + idx, q.to(tl.int8), mask=m, eviction_policy="evict_first"
                )
            else:
                s = tl.load(meta_ptr).to(tl.float32)
                azp = tl.load(meta2_ptr).to(tl.float32)
                x = tl.load(
                    in_ptr + idx, mask=m, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                q = libdevice.rint(x / s) + azp
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                tl.store(
                    out_ptr + idx, q.to(tl.int8), mask=m, eviction_policy="evict_first"
                )
        else:
            if SYM:
                s = tl.load(meta_ptr).to(tl.float32)
                x = tl.load(in_ptr + idx, eviction_policy="evict_first").to(tl.float32)
                q = libdevice.rint(x / s)
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                tl.store(out_ptr + idx, q.to(tl.int8), eviction_policy="evict_first")
            else:
                s = tl.load(meta_ptr).to(tl.float32)
                azp = tl.load(meta2_ptr).to(tl.float32)
                x = tl.load(in_ptr + idx, eviction_policy="evict_first").to(tl.float32)
                q = libdevice.rint(x / s) + azp
                q = tl.minimum(tl.maximum(q, -128.0), 127.0)
                tl.store(out_ptr + idx, q.to(tl.int8), eviction_policy="evict_first")


_BLOCK_DYN_WIDE = 1024
_NUM_WARPS_DYN_WIDE = 8
_BLOCK_DYN = 4096
_NUM_WARPS_DYN = 8
_BLOCK_DYN_ULTRA = 16384
_BLOCK = 2048
_NUM_WARPS = 4


def _pick_dyn_config(cols):
    # Rows up to 4096 wide use a single register-resident load (BLOCK=4096).
    # Rows 5120-8192 use the waste-free two-pass loop (BLOCK=1024); num_warps=8
    # halves the per-thread reduction chain (4 elems/thread) for the serialized
    # 5-iteration two-pass loop of 2048x5120. Very wide rows (e.g. 13824) fit
    # one BLOCK=16384 register-resident load.
    if cols <= 4096:
        return _BLOCK_DYN, _NUM_WARPS_DYN
    if cols <= 8192:
        return _BLOCK_DYN_WIDE, _NUM_WARPS_DYN_WIDE
    if cols <= 16384:
        return _BLOCK_DYN_ULTRA, _NUM_WARPS_DYN
    return _BLOCK_DYN, _NUM_WARPS_DYN


def scaled_int8_quant(input, scale, azp, symmetric):
    if isinstance(symmetric, torch.Tensor):
        symmetric = bool(symmetric.item())
    else:
        symmetric = bool(symmetric)

    if input.dim() == 1:
        rows, cols = 1, input.numel()
    else:
        rows, cols = input.shape[0], input.numel() // input.shape[0]

    out = torch.empty_like(input, dtype=torch.int8)
    n_elem = input.numel()

    dynamic = scale is None

    if dynamic:
        scale_out = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
        blk_dyn, wrp_dyn = _pick_dyn_config(cols)
        masked = cols % blk_dyn != 0
        if symmetric:
            _scaled_int8_quant_kernel[(rows,)](
                input,
                out,
                scale_out,
                out,
                cols,
                n_elem,
                DYN=True,
                SYM=True,
                BLOCK=blk_dyn,
                MASKED=masked,
                num_warps=wrp_dyn,
            )
            return out, scale_out, None
        azp_out = torch.empty((rows, 1), dtype=torch.int32, device=input.device)
        _scaled_int8_quant_kernel[(rows,)](
            input,
            out,
            scale_out,
            azp_out,
            cols,
            n_elem,
            DYN=True,
            SYM=False,
            BLOCK=blk_dyn,
            MASKED=masked,
            num_warps=wrp_dyn,
        )
        return out, scale_out, azp_out

    grid = (triton.cdiv(n_elem, _BLOCK),)
    _scaled_int8_quant_kernel[grid](
        input,
        out,
        scale,
        azp,
        cols,
        n_elem,
        DYN=False,
        SYM=symmetric,
        BLOCK=_BLOCK,
        MASKED=n_elem % _BLOCK != 0,
        num_warps=_NUM_WARPS,
    )
    return out, scale, azp
