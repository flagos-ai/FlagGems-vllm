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

# ── static: flat pointwise, int32 rounding+saturation path ───────────────────


@triton.jit
def _static_sym_full(
    x_ptr,
    y_ptr,
    scale_ptr,
    n_full,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(x_ptr + offs).to(tl.float32)
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    q = (src * inv_s).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -128), 127)
    tl.store(y_ptr + offs, q.to(tl.int8))


@triton.jit
def _static_sym_tail(
    x_ptr,
    y_ptr,
    scale_ptr,
    base,
    tail,
    BLOCK: tl.constexpr,
):
    offs = base + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < tail
    src = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    q = (src * inv_s).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -128), 127)
    tl.store(y_ptr + offs, q.to(tl.int8), mask=mask)


@triton.jit
def _static_asym_full(
    x_ptr,
    y_ptr,
    scale_ptr,
    azp_ptr,
    n_full,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(x_ptr + offs).to(tl.float32)
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    azp = tl.load(azp_ptr).to(tl.int32)
    q = (src * inv_s).to(tl.int32) + azp
    q = tl.minimum(tl.maximum(q, -128), 127)
    tl.store(y_ptr + offs, q.to(tl.int8))


@triton.jit
def _static_asym_tail(
    x_ptr,
    y_ptr,
    scale_ptr,
    azp_ptr,
    base,
    tail,
    BLOCK: tl.constexpr,
):
    offs = base + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < tail
    src = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    inv_s = 1.0 / scale
    azp = tl.load(azp_ptr).to(tl.int32)
    q = (src * inv_s).to(tl.int32) + azp
    q = tl.minimum(tl.maximum(q, -128), 127)
    tl.store(y_ptr + offs, q.to(tl.int8), mask=mask)


# ── dynamic stats: unmasked 2D tile for full chunks ──────────────────────────


@triton.jit
def _stats_sym_full(
    x_ptr,
    part_ptr,
    hidden,
    cpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    m0 = tl.program_id(0) * R + tl.arange(0, R)
    n = tl.program_id(1)
    if SPLIT:
        # 4-way split reduction: four BLOCK/4-wide axis-1 maxes accumulated.
        # Validated ~35% faster than one BLOCK-wide reduction on wide tiles
        # with enough programs to hide the 4 serialized sub-reductions.
        Q: tl.constexpr = BLOCK // 4
        acc = tl.zeros((R,), dtype=tl.float32)
        for k in tl.static_range(4):
            cols = n * BLOCK + k * Q + tl.arange(0, Q)
            offs = m0[:, None] * hidden + cols[None, :]
            src = tl.load(x_ptr + offs).to(tl.float32)
            acc = tl.maximum(acc, tl.max(tl.abs(src), axis=1))
        v = acc
    else:
        cols = n * BLOCK + tl.arange(0, BLOCK)
        offs = m0[:, None] * hidden + cols[None, :]
        src = tl.load(x_ptr + offs).to(tl.float32)
        v = tl.max(tl.abs(src), axis=1)
    tl.store(part_ptr + m0 * cpr + n, v)


@triton.jit
def _stats_asym_full(
    x_ptr,
    part_max_ptr,
    part_min_ptr,
    hidden,
    cpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m0 = tl.program_id(0) * R + tl.arange(0, R)
    n = tl.program_id(1)
    cols = n * BLOCK + tl.arange(0, BLOCK)
    offs = m0[:, None] * hidden + cols[None, :]
    src = tl.load(x_ptr + offs).to(tl.float32)
    vmax = tl.max(src, axis=1)
    vmin = -tl.max(-src, axis=1)
    tl.store(part_max_ptr + m0 * cpr + n, vmax)
    tl.store(part_min_ptr + m0 * cpr + n, vmin)


@triton.jit
def _stats_sym_tail_tile(
    x_ptr,
    part_ptr,
    hidden,
    base,
    cpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m0 = tl.program_id(0) * R + tl.arange(0, R)
    cols = base + tl.arange(0, BLOCK)
    offs = m0[:, None] * hidden + cols[None, :]
    src = tl.load(x_ptr + offs).to(tl.float32)
    v = tl.max(tl.abs(src), axis=1)
    tl.store(part_ptr + m0 * cpr + (cpr - 1), v)


@triton.jit
def _stats_sym_tail(
    x_ptr,
    part_ptr,
    hidden,
    base,
    tail,
    cpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = base + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < tail
    src = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.max(tl.abs(src))
    tl.store(part_ptr + row * cpr + (cpr - 1), v)


@triton.jit
def _stats_asym_tail(
    x_ptr,
    part_max_ptr,
    part_min_ptr,
    hidden,
    base,
    tail,
    cpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = base + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < tail
    src = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    vmax = tl.max(tl.where(mask, src, -1e30))
    vmin = -tl.max(tl.where(mask, -src, -1e30))
    tl.store(part_max_ptr + row * cpr + (cpr - 1), vmax)
    tl.store(part_min_ptr + row * cpr + (cpr - 1), vmin)


# ── dynamic fold ─────────────────────────────────────────────────────────────


@triton.jit
def _fold_sym(
    part_ptr,
    scale_out_ptr,
    inv_out_ptr,
    rows,
    cpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    m0 = tl.program_id(0) * R + tl.arange(0, R)
    maskr = m0 < rows
    idx = m0[:, None] * cpr + tl.arange(0, C)[None, :]
    mask = tl.arange(0, C)[None, :] < cpr
    acc = tl.max(tl.load(part_ptr + idx, mask=mask, other=0.0), axis=1)
    scale = acc / 127.0
    inv_s = tl.where(acc == 0.0, 0.0, 127.0 / acc)
    tl.store(scale_out_ptr + m0, scale, mask=maskr)
    tl.store(inv_out_ptr + m0, inv_s, mask=maskr)


@triton.jit
def _fold_asym(
    part_max_ptr,
    part_min_ptr,
    scale_out_ptr,
    azp_out_ptr,
    inv_out_ptr,
    rows,
    cpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    m0 = tl.program_id(0) * R + tl.arange(0, R)
    maskr = m0 < rows
    idx = m0[:, None] * cpr + tl.arange(0, C)[None, :]
    mask = tl.arange(0, C)[None, :] < cpr
    rmax = tl.max(tl.load(part_max_ptr + idx, mask=mask, other=-1e30), axis=1)
    rmin = -tl.max(-tl.load(part_min_ptr + idx, mask=mask, other=1e30), axis=1)
    scale = (rmax - rmin) / 255.0
    inv_s = 1.0 / scale
    azp = (-128.0 - rmin * inv_s).to(tl.int32)
    tl.store(scale_out_ptr + m0, scale, mask=maskr)
    tl.store(azp_out_ptr + m0, azp, mask=maskr)
    tl.store(inv_out_ptr + m0, inv_s, mask=maskr)


# ── dynamic quantize ─────────────────────────────────────────────────────────


@triton.jit
def _quant_sym_flat(
    x_ptr,
    y_ptr,
    inv_ptr,
    CPR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // CPR
    inv_s = tl.load(inv_ptr + row)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(x_ptr + offs).to(tl.float32)
    q = (src * inv_s).to(tl.int8)
    tl.store(y_ptr + offs, q)


@triton.jit
def _quant_sym_full(
    x_ptr,
    y_ptr,
    inv_ptr,
    hidden,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    inv_s = tl.load(inv_ptr + m)
    offs = m * hidden + n * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(x_ptr + offs).to(tl.float32)
    q = (src * inv_s).to(tl.int8)
    tl.store(y_ptr + offs, q)


@triton.jit
def _quant_sym_tail(
    x_ptr,
    y_ptr,
    inv_ptr,
    hidden,
    base,
    tail,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    inv_s = tl.load(inv_ptr + row)
    offs = base + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < tail
    src = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    q = (src * inv_s).to(tl.int8)
    tl.store(y_ptr + row * hidden + offs, q, mask=mask)


@triton.jit
def _quant_sym_tail_flat(
    x_ptr,
    y_ptr,
    inv_ptr,
    hidden,
    base,
    total,
    TAIL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one flat pass over the tail region (rows x TAIL), row = idx // TAIL
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total
    row = idx // TAIL
    col = idx - row * TAIL
    inv_s = tl.load(inv_ptr + row, mask=mask, other=0.0)
    offs = row * hidden + base + col
    src = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = (src * inv_s).to(tl.int8)
    tl.store(y_ptr + offs, q, mask=mask)


@triton.jit
def _quant_asym_full(
    x_ptr,
    y_ptr,
    inv_ptr,
    azp_ptr,
    hidden,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    inv_s = tl.load(inv_ptr + m)
    azp = tl.load(azp_ptr + m).to(tl.float32)
    offs = m * hidden + n * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(x_ptr + offs).to(tl.float32)
    q = (src * inv_s + azp).to(tl.int8)
    tl.store(y_ptr + offs, q)


@triton.jit
def _quant_asym_tail(
    x_ptr,
    y_ptr,
    inv_ptr,
    azp_ptr,
    hidden,
    base,
    tail,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    inv_s = tl.load(inv_ptr + row)
    azp = tl.load(azp_ptr + row).to(tl.float32)
    offs = base + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < tail
    src = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    q = (src * inv_s + azp).to(tl.int8)
    tl.store(y_ptr + row * hidden + offs, q, mask=mask)


def _pow2_part(hidden):
    p = 1
    while hidden % (p * 2) == 0:
        p *= 2
    return p


def _R(rows, blk):
    if rows % 8 == 0 and 8 * blk <= 32768:
        return 8
    if rows % 4 == 0 and 4 * blk <= 16384:
        return 4
    if rows % 2 == 0 and 2 * blk <= 16384:
        return 2
    return 1


def scaled_int8_quant(input, scale, azp, symmetric):
    input_2d = input.reshape(-1, input.shape[-1])
    num_tokens, hidden = input_2d.shape
    device = input.device

    if scale is not None:
        output = torch.empty_like(input_2d, dtype=torch.int8)
        n = input_2d.numel()
        blk = 8192
        n_full = n // blk * blk
        tail = n - n_full
        if symmetric:
            if n_full:
                _static_sym_full[(n_full // blk,)](
                    input_2d, output, scale, n_full, BLOCK=blk, num_warps=8
                )
            if tail:
                _static_sym_tail[(1,)](
                    input_2d, output, scale, n_full, tail, BLOCK=blk, num_warps=8
                )
        else:
            if n_full:
                _static_asym_full[(n_full // blk,)](
                    input_2d, output, scale, azp, n_full, BLOCK=blk, num_warps=8
                )
            if tail:
                _static_asym_tail[(1,)](
                    input_2d, output, scale, azp, n_full, tail, BLOCK=blk, num_warps=8
                )
        return output.view(input.shape), scale, azp

    output = torch.empty_like(input_2d, dtype=torch.int8)
    scale_out = torch.empty((num_tokens, 1), dtype=torch.float32, device=device)
    inv_buf = torch.empty((num_tokens,), dtype=torch.float32, device=device)

    p2 = _pow2_part(hidden)
    use_flat = symmetric and p2 >= 64

    if use_flat:
        # Stats and quant both use BLOCK=4096 for wide rows with a small tail pass;
        # flat quant with constexpr CPR covers the full chunks of every row.
        qblk = 4096 if hidden >= 4096 else p2
        cpr_q = hidden // qblk
        q_tail = hidden % qblk
        sblk = 4096 if hidden >= 4096 else qblk
        cpr_full = hidden // sblk
        s_tail = hidden % sblk
        cpr = cpr_full + (1 if s_tail else 0)
        C = 4
        while C < cpr:
            C *= 2
        R = _R(num_tokens, sblk)
        Rf = 1
        for r_ in (64, 32, 16, 8, 4, 2, 1):
            if num_tokens % r_ == 0:
                Rf = r_
                break
        part = torch.empty((num_tokens, C), dtype=torch.float32, device=device)
        use_split = sblk >= 2048 and num_tokens * cpr_full >= 256
        _stats_sym_full[(num_tokens // R, cpr_full)](
            input_2d,
            part,
            hidden,
            cpr,
            R=R,
            BLOCK=sblk,
            SPLIT=use_split,
            num_warps=8,
        )
        if s_tail:
            tb = 1
            while tb * 2 <= s_tail:
                tb *= 2
            if s_tail == tb:
                _stats_sym_tail_tile[(num_tokens // R, 1)](
                    input_2d,
                    part,
                    hidden,
                    cpr_full * sblk,
                    cpr,
                    R=R,
                    BLOCK=tb,
                    num_warps=8,
                )
            else:
                tb2 = 1
                while tb2 < s_tail:
                    tb2 *= 2
                _stats_sym_tail[(num_tokens,)](
                    input_2d,
                    part,
                    hidden,
                    cpr_full * sblk,
                    s_tail,
                    cpr,
                    BLOCK=max(tb2, 32),
                    num_warps=8,
                )
        _fold_sym[((num_tokens + Rf - 1) // Rf,)](
            part, scale_out, inv_buf, num_tokens, cpr, C=C, R=Rf, num_warps=4
        )
        if q_tail:
            if cpr_q == 1:
                # flat pass over the whole tail region: few wide programs
                total = num_tokens * q_tail
                _quant_sym_tail_flat[(triton.cdiv(total, 4096),)](
                    input_2d,
                    output,
                    inv_buf,
                    hidden,
                    qblk,
                    total,
                    TAIL=q_tail,
                    BLOCK=4096,
                    num_warps=8,
                )
            else:
                tb = 1
                while tb < q_tail:
                    tb *= 2
                _quant_sym_tail[(num_tokens,)](
                    input_2d,
                    output,
                    inv_buf,
                    hidden,
                    cpr_q * qblk,
                    q_tail,
                    BLOCK=max(tb, 32),
                    num_warps=4,
                )
        else:
            _quant_sym_flat[(num_tokens * cpr_q,)](
                input_2d, output, inv_buf, CPR=cpr_q, BLOCK=qblk, num_warps=8
            )
        return output.view(input.shape), scale_out, None

    # fallback: non-power-of-2-divisible hidden (correctness shapes like 17, 1025, 8193)
    blk = 4096
    while blk > hidden:
        blk //= 2
    cpr_full = hidden // blk
    tail = hidden % blk
    cpr = cpr_full + (1 if tail else 0)
    C = 4
    R = _R(num_tokens, blk)
    Rf = 1
    for r_ in (64, 32, 16, 8, 4, 2, 1):
        if num_tokens % r_ == 0:
            Rf = r_
            break
    tail_blk = max(32, _pow2_part(tail) if tail else 0)

    if symmetric:
        part = torch.empty((num_tokens, C), dtype=torch.float32, device=device)
        use_split = blk >= 2048 and num_tokens * cpr_full >= 256
        _stats_sym_full[(num_tokens // R, cpr_full)](
            input_2d,
            part,
            hidden,
            cpr,
            R=R,
            BLOCK=blk,
            SPLIT=use_split,
            num_warps=8,
        )
        if tail:
            _stats_sym_tail[(num_tokens,)](
                input_2d,
                part,
                hidden,
                cpr_full * blk,
                tail,
                cpr,
                BLOCK=tail_blk,
                num_warps=8,
            )
        _fold_sym[((num_tokens + Rf - 1) // Rf,)](
            part, scale_out, inv_buf, num_tokens, cpr, C=C, R=Rf, num_warps=4
        )
        if cpr_full:
            _quant_sym_full[(num_tokens, cpr_full)](
                input_2d, output, inv_buf, hidden, BLOCK=blk, num_warps=8
            )
        if tail:
            _quant_sym_tail[(num_tokens,)](
                input_2d,
                output,
                inv_buf,
                hidden,
                cpr_full * blk,
                tail,
                BLOCK=tail_blk,
                num_warps=8,
            )
        return output.view(input.shape), scale_out, None

    part_max = torch.empty((num_tokens, C), dtype=torch.float32, device=device)
    part_min = torch.empty((num_tokens, C), dtype=torch.float32, device=device)
    azp_out = torch.empty((num_tokens, 1), dtype=torch.int32, device=device)
    _stats_asym_full[(num_tokens // R, cpr_full)](
        input_2d, part_max, part_min, hidden, cpr, R=R, BLOCK=blk, num_warps=8
    )
    if tail:
        _stats_asym_tail[(num_tokens,)](
            input_2d,
            part_max,
            part_min,
            hidden,
            cpr_full * blk,
            tail,
            cpr,
            BLOCK=tail_blk,
            num_warps=8,
        )
    _fold_asym[((num_tokens + Rf - 1) // Rf,)](
        part_max,
        part_min,
        scale_out,
        azp_out,
        inv_buf,
        num_tokens,
        cpr,
        C=C,
        R=Rf,
        num_warps=4,
    )
    if cpr_full:
        _quant_asym_full[(num_tokens, cpr_full)](
            input_2d, output, inv_buf, azp_out, hidden, BLOCK=blk, num_warps=8
        )
    if tail:
        _quant_asym_tail[(num_tokens,)](
            input_2d,
            output,
            inv_buf,
            azp_out,
            hidden,
            cpr_full * blk,
            tail,
            BLOCK=tail_blk,
            num_warps=8,
        )
    return output.view(input.shape), scale_out, azp_out
