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


@triton.jit
def _cc1d_prefill_w4_kernel(
    x_ptr,
    w_ptr,
    st_ptr,
    idx_ptr,
    qsl_ptr,
    hs_ptr,
    out_ptr,
    SXT: tl.constexpr,
    SWW: tl.constexpr,
    SSC: tl.constexpr,
    SSJ: tl.constexpr,
    SOT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NTOK: tl.constexpr,
    POOL: tl.constexpr,
    COMMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    # Contiguous output => stride(0) == channel count, so the channel-block count
    # follows from constexpr metadata without being passed separately.
    pid_t = pid // (SOT // BLOCK_D)
    pid_d = pid % (SOT // BLOCK_D)

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_D), BLOCK_D)

    slot_raw = tl.load(idx_ptr)
    # Clamped so that every address stays in bounds even for the pad sentinel;
    # the state store is masked off in that case.
    slot = tl.maximum(slot_raw.to(tl.int64), 0)
    t0 = tl.load(qsl_ptr).to(tl.int64)
    t1 = tl.load(qsl_ptr + 1).to(tl.int64)
    n = (t1 - t0).to(tl.int32)
    has_init = tl.load(hs_ptr) != 0
    if slot_raw >= POOL or t0 < 0 or t1 > NTOK or t1 <= t0:
        return
    valid_slot = slot_raw >= 0
    keep = valid_slot & has_init

    start = pid_t * BLOCK_T

    # Base addresses; token-relative row arithmetic is added on top of them.  The
    # channel axis of x, weight, conv_states and out is unit-stride on the fast
    # path (guarded host-side), so it contributes no stride term.
    if start >= n:
        return

    xb = x_ptr + t0 * SXT + cols
    sb = st_ptr + slot * SSC + cols
    ob = out_ptr + t0 * SOT + cols
    wp = w_ptr + cols

    # Negated taps.  The convolution is accumulated as m = -y so that the SiLU
    # exponential needs no scale multiply: silu(y) = y / (1 + e^{-y}) and with
    # m = -y that is m / (-1 - e^{m}).  Negating the weights once here costs four
    # hoisted vector ops per program, and since IEEE negation is exact and both
    # fma and divide are sign-symmetric, z = m / (-1 - e^{m}) is bit-identical to
    # y / (1 + e^{-y}) while removing one vector ALU slot per output element.
    w0 = -tl.load(wp).to(tl.float32)
    w1 = -tl.load(wp + SWW).to(tl.float32)
    w2 = -tl.load(wp + 2 * SWW).to(tl.float32)
    w3 = -tl.load(wp + 3 * SWW).to(tl.float32)

    # Rolling-window prologue.  Relative rows start-3, start-2, start-1 come from
    # conv_states when negative (chunk 0 only) and from x otherwise; both probe
    # loads are index-clamped, so no masked or out-of-bounds access is needed and
    # the discarded side of each select is harmless.
    r0 = start - 3
    r1 = start - 2
    r2 = start - 1
    a0 = tl.load(sb + tl.minimum(tl.maximum(r0 + 3, 0), 2) * SSJ)
    a1 = tl.load(sb + tl.minimum(tl.maximum(r1 + 3, 0), 2) * SSJ)
    a2 = tl.load(sb + tl.minimum(tl.maximum(r2 + 3, 0), 2) * SSJ)
    b0 = tl.load(xb + tl.maximum(r0, 0) * SXT)
    b1 = tl.load(xb + tl.maximum(r1, 0) * SXT)
    b2 = tl.load(xb + tl.maximum(r2, 0) * SXT)

    a0 = tl.where(keep, a0, 0.0)
    a1 = tl.where(keep, a1, 0.0)
    a2 = tl.where(keep, a2, 0.0)

    s0 = tl.where(r0 < 0, a0.to(tl.float32), b0.to(tl.float32))
    s1 = tl.where(r1 < 0, a1.to(tl.float32), b1.to(tl.float32))
    s2 = tl.where(r2 < 0, a2.to(tl.float32), b2.to(tl.float32))

    ntok = tl.minimum(tl.maximum(n - start, 0), BLOCK_T)
    # Eight tokens per iteration.
    ng = ntok // 8
    for i in range(ng):
        row = start + i * 8
        x0 = tl.load(xb + row * SXT).to(tl.float32)
        x1 = tl.load(xb + (row + 1) * SXT).to(tl.float32)
        x2 = tl.load(xb + (row + 2) * SXT).to(tl.float32)
        x3 = tl.load(xb + (row + 3) * SXT).to(tl.float32)
        x4 = tl.load(xb + (row + 4) * SXT).to(tl.float32)
        x5 = tl.load(xb + (row + 5) * SXT).to(tl.float32)
        x6 = tl.load(xb + (row + 6) * SXT).to(tl.float32)
        x7 = tl.load(xb + (row + 7) * SXT).to(tl.float32)
        m0 = tl.math.fma(w3, x0, tl.math.fma(w2, s2, tl.math.fma(w1, s1, w0 * s0)))
        m1 = tl.math.fma(w3, x1, tl.math.fma(w2, x0, tl.math.fma(w1, s2, w0 * s1)))
        m2 = tl.math.fma(w3, x2, tl.math.fma(w2, x1, tl.math.fma(w1, x0, w0 * s2)))
        m3 = tl.math.fma(w3, x3, tl.math.fma(w2, x2, tl.math.fma(w1, x1, w0 * x0)))
        oty = out_ptr.dtype.element_ty
        z0 = m0 / tl.math.fma(tl.math.exp(m0), -1.0, -1.0)
        z1 = m1 / tl.math.fma(tl.math.exp(m1), -1.0, -1.0)
        z2 = m2 / tl.math.fma(tl.math.exp(m2), -1.0, -1.0)
        z3 = m3 / tl.math.fma(tl.math.exp(m3), -1.0, -1.0)
        tl.store(ob + row * SOT, z0.to(oty))
        tl.store(ob + (row + 1) * SOT, z1.to(oty))
        tl.store(ob + (row + 2) * SOT, z2.to(oty))
        tl.store(ob + (row + 3) * SOT, z3.to(oty))
        m4 = tl.math.fma(w3, x4, tl.math.fma(w2, x3, tl.math.fma(w1, x2, w0 * x1)))
        m5 = tl.math.fma(w3, x5, tl.math.fma(w2, x4, tl.math.fma(w1, x3, w0 * x2)))
        m6 = tl.math.fma(w3, x6, tl.math.fma(w2, x5, tl.math.fma(w1, x4, w0 * x3)))
        m7 = tl.math.fma(w3, x7, tl.math.fma(w2, x6, tl.math.fma(w1, x5, w0 * x4)))
        z4 = m4 / tl.math.fma(tl.math.exp(m4), -1.0, -1.0)
        z5 = m5 / tl.math.fma(tl.math.exp(m5), -1.0, -1.0)
        z6 = m6 / tl.math.fma(tl.math.exp(m6), -1.0, -1.0)
        z7 = m7 / tl.math.fma(tl.math.exp(m7), -1.0, -1.0)
        tl.store(ob + (row + 4) * SOT, z4.to(oty))
        tl.store(ob + (row + 5) * SOT, z5.to(oty))
        tl.store(ob + (row + 6) * SOT, z6.to(oty))
        tl.store(ob + (row + 7) * SOT, z7.to(oty))
        s0 = x5
        s1 = x6
        s2 = x7

    # case and for any token count that is a multiple of the group size).
    for i in range(ng * 8, ntok):
        row = start + i
        xv = tl.load(xb + row * SXT).to(tl.float32)
        mv = tl.math.fma(w3, xv, tl.math.fma(w2, s2, tl.math.fma(w1, s1, w0 * s0)))
        z = mv / tl.math.fma(tl.math.exp(mv), -1.0, -1.0)
        tl.store(ob + row * SOT, z.to(out_ptr.dtype.element_ty))
        s0 = s1
        s1 = s2
        s2 = xv

    if not COMMIT:
        return

    # Shift-register writeback: every channel block of the program that owns the
    # final token chunk commits its own disjoint slice of rows 0..2 of the
    # addressed cache slot.  The row source is resolved generically (state rows
    # when n < SL), and only the three real state rows are stored, never the
    # cache-line padding.
    is_last = (start + BLOCK_T >= n) & valid_slot
    k3 = tl.arange(0, 3)[:, None].to(tl.int32)
    c2 = cols[None, :]
    tgt = n - 3 + k3
    from_state = tgt < 0
    srow = tl.minimum(tl.maximum(tgt + 3, 0), 2)
    xoff = tl.maximum(tgt, 0)
    wa = tl.load(st_ptr + slot * SSC + srow * SSJ + c2)
    wb = tl.load(x_ptr + t0 * SXT + xoff * SXT + c2)
    # Missing-history rows (tgt < 0) hold the *history* of the extended stream.
    # A native probe on the target showed that with has_initial_state=False the
    # history counts as zeros (those rows become zeros), while with
    # has_initial_state=True the old state rows are shifted in - the same rule
    # that already gates the prologue loads a0..a2.
    wa = tl.where(keep, wa, 0.0)
    wv = tl.where(from_state, wa, wb)
    # Channel masking is unnecessary: BLOCK_D divides the channel count on the
    # fast path, so every column of this tile is a real state channel.
    msk = is_last & (k3 >= 0) & (c2 >= 0)
    tl.store(st_ptr + slot * SSC + k3 * SSJ + c2, wv, mask=msk)


@triton.jit
def _commit_tail(
    X,
    STATE,
    IDX,
    LENGTHS,
    SX: tl.constexpr,
    SP: tl.constexpr,
    ST: tl.constexpr,
    T: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(IDX).to(tl.int64)
    if slot < 0 or slot >= POOL:
        return
    end = tl.load(LENGTHS + 1)
    start = tl.load(LENGTHS)
    if start != 0 or end != T:
        return
    rows = tl.arange(0, 4)[:, None]
    x = tl.load(X + (T - 3 + rows) * SX + d[None, :], mask=rows < 3, other=0.0)
    tl.store(STATE + slot * SP + rows * ST + d[None, :], x, mask=rows < 3)


def fla_causal_conv1d_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    *,
    activation="silu",
    pad_slot_id=-1,
    null_block_id=0,
    head_num=32,
):
    """Single active sequence, FLA layouts X[T,D], W[4,D], S[slots,3,D].

    The prefix offsets are [0,T] and cache_indices selects one valid active slot.
    Returns a [32,T,D/32] view and writes the last three inputs into that slot.
    """
    state, cu, indices = conv_states, query_start_loc, cache_indices
    if (
        x.device.type != "npu"
        or x.ndim != 2
        or x.dtype != torch.bfloat16
        or x.stride(-1) != 1
        or not x.numel()
    ):
        raise ValueError("expected nonempty token-major BF16 activations on NPU")
    t, d = x.shape
    if (
        d != 4096
        or weight.shape != (4, d)
        or weight.dtype != x.dtype
        or not weight.is_contiguous()
    ):
        raise ValueError("expected a contiguous four-tap weight with 4096 channels")
    if (
        state.ndim != 3
        or state.shape[1:] != (3, d)
        or state.dtype != x.dtype
        or state.stride(-1) != 1
        or state.stride(1) < d
        or state.stride(0) < (2 * state.stride(1) + d)
        or state.shape[0] == 0
    ):
        raise ValueError("expected a non-overlapping [slots,3,4096] BF16 state")
    if (
        any(a.device != x.device for a in (state, weight, cu, indices))
        or cu.dtype != torch.int32
        or indices.dtype != torch.int32
        or not cu.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError(
            "expected contiguous int32 lengths and indices on the same NPU"
        )

    if (
        bias is not None
        or activation != "silu"
        or head_num != 32
        or pad_slot_id != -1
        or null_block_id != 0
    ):
        raise ValueError(
            "expected bias-free SiLU with 32 output heads and the default padding conventions"
        )
    if (
        cu.shape != (2,)
        or indices.shape != (1,)
        or has_initial_state.shape != (1,)
        or has_initial_state.dtype != torch.bool
        or has_initial_state.device != x.device
    ):
        raise ValueError("expected a single segment and a boolean history flag")
    out = torch.empty((t, d), device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        _cc1d_prefill_w4_kernel[(triton.cdiv(t, 32) * (d // 2048),)](
            x,
            weight,
            state,
            indices,
            cu,
            has_initial_state,
            out,
            x.stride(0),
            weight.stride(0),
            state.stride(0),
            state.stride(1),
            d,
            32,
            2048,
            t,
            state.shape[0],
            t <= 32,
            num_stages=2,
            num_warps=1,
        )
        if t > 32:
            _commit_tail[(d // 512,)](
                x,
                state,
                indices,
                cu,
                x.stride(0),
                state.stride(0),
                state.stride(1),
                t,
                state.shape[0],
                512,
            )
    return out.view(t, head_num, d // head_num).permute(1, 0, 2)
