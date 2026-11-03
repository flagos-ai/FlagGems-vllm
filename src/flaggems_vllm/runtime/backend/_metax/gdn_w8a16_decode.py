# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX fused decode. FP32 state, no state quantization."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _decode(
    Q,
    K,
    V,
    A,
    B,
    AL,
    Bias,
    S,
    O,
    Ids,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    FACTOR: tl.constexpr,
    FMA: tl.constexpr,
    ROUND_BETA: tl.constexpr,
):
    n, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(Ids + n).to(tl.int64)
    d = tl.arange(0, BK)
    vd = tile * BV + tl.arange(0, BV)
    if slot <= 0:
        tl.store(O + (n * HV + h) * DV + vd, 0, vd < DV)
        return
    kh = h // (HV // H)
    q = tl.load(Q + n * QS + kh * DK + d, d < DK, 0).to(tl.float32)
    k = tl.load(K + n * KS + kh * DK + d, d < DK, 0).to(tl.float32)
    v = tl.load(V + n * VS + h * DV + vd, vd < DV, 0).to(tl.float32)
    if NORMALIZE:
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    gate = tl.load(A + n * AS + h).to(tl.float32) + tl.load(Bias + h).to(tl.float32)
    sp = tl.where(gate <= 20, tl.log(1 + tl.exp(gate)), gate)
    decay = tl.exp(-tl.exp(tl.load(AL + h).to(tl.float32)) * sp)
    beta = tl.sigmoid(tl.load(B + n * BS + h).to(tl.float32))
    if ROUND_BETA:
        beta = beta.to(B.dtype.element_ty).to(tl.float32)
    off = ((slot * HV + h) * DV + vd[:, None]) * DK + d[None, :]
    mask = (vd[:, None] < DV) & (d[None, :] < DK)
    state = tl.load(S + off, mask, 0)
    if FACTOR:
        residual = (v - tl.sum(state * k[None, :], 1) * decay) * beta
        if FMA:
            state = tl.fma(residual[:, None], k[None, :], state * decay)
        else:
            state = state * decay + residual[:, None] * k[None, :]
    else:
        state *= decay
        residual = (v - tl.sum(state * k[None, :], 1)) * beta
        if FMA:
            state = tl.fma(residual[:, None], k[None, :], state)
        else:
            state += residual[:, None] * k[None, :]
    out = tl.sum(state * (q * SCALE)[None, :], 1)
    tl.store(O + (n * HV + h) * DV + vd, out, vd < DV)
    tl.store(S + off, state, mask)


def decode(
    qkv,
    a,
    b,
    alog,
    bias,
    state,
    ids,
    out=None,
    *,
    scale=None,
    normalize=True,
    bv=None,
    warps=None,
    factor=False,
    fma=False,
    round_beta=True,
):
    """One token per request, distinct positive state slots; <=0 denotes padding.

    Packed [T,2*H*K+HV*V] or separate [1,T,H,K]/[1,T,HV,V] no-copy views.
    Mutates FP32 state; caller owns out/cache. No data tensors are cached.
    """
    if state.ndim != 4 or state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous FP32 [slots,HV,V,K]")
    slots, hv, dv, dk = state.shape
    if isinstance(qkv, tuple):
        q, k, v = qkv
        if q.ndim != 4 or q.shape != k.shape or q.shape[0] != 1:
            raise ValueError("separate Q/K must have shape [1,T,H,K]")
        n, h = q.shape[1:3]
        if q.shape[3] != dk or v.shape != (1, n, hv, dv):
            raise ValueError("invalid QKV dimensions")
    else:
        if qkv.ndim != 2:
            raise ValueError("packed QKV must be rank two")
        n, packed = qkv.shape
        width = packed - hv * dv
        if width <= 0 or width % (2 * dk):
            raise ValueError("invalid packed QKV dimensions")
        h = width // (2 * dk)
        q = qkv[:, : h * dk].view(1, n, h, dk)
        k = qkv[:, h * dk : 2 * h * dk].view(1, n, h, dk)
        v = qkv[:, 2 * h * dk :].view(1, n, hv, dv)
    if h < 1 or hv % h or min(dk, dv) < 1 or max(dk, dv) > 256:
        raise NotImplementedError("unsupported head dimensions")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share BF16/FP16 dtype")
    if any(t.stride(-1) != 1 or t.stride(-2) != t.shape[-1] for t in (q, k, v)):
        raise ValueError("head dimensions must be contiguous")
    if (
        a.shape != (n, hv)
        or b.shape != a.shape
        or a.dtype != q.dtype
        or b.dtype != q.dtype
    ):
        raise ValueError("invalid gate activations")
    if (
        alog.shape != (hv,)
        or bias.shape != (hv,)
        or alog.dtype != torch.float32
        or bias.dtype != torch.float32
    ):
        raise ValueError("A_log and bias must be FP32 [HV]")
    if ids.shape != (n,) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("invalid state indices")
    if any(not t.is_contiguous() for t in (a, b, alog, bias, ids)):
        raise ValueError("gate and index buffers must be contiguous")
    if out is None:
        out = torch.empty((1, n, hv, dv), device=q.device, dtype=q.dtype)
    if (
        out.shape not in ((1, n, hv, dv), (n, 1, hv, dv))
        or out.dtype != q.dtype
        or not out.is_contiguous()
    ):
        raise ValueError("invalid output")
    tensors = (q, k, v, a, b, alog, bias, state, ids, out)
    if q.device.type != "cuda" or any(
        t.device != q.device or t.requires_grad for t in tensors
    ):
        raise ValueError("forward-only tensors on the same accelerator required")
    if n and any(out.data_ptr() == t.data_ptr() for t in tensors[:-1]):
        raise ValueError("output must not alias an input")
    if bv is None:
        bv = 8 if n < 8 else 64
    if warps is None:
        warps = 1 if n < 8 else 4
    if n:
        with torch.cuda.device(q.device):
            _decode[(n, hv, tr.cdiv(dv, bv))](
                q,
                k,
                v,
                a,
                b,
                alog,
                bias,
                state,
                out,
                ids,
                h,
                hv,
                dk,
                dv,
                q.stride(1),
                k.stride(1),
                v.stride(1),
                a.stride(0),
                b.stride(0),
                dk**-0.5 if scale is None else scale,
                normalize,
                tr.next_power_of_2(dk),
                bv,
                factor,
                fma,
                round_beta,
                num_warps=warps,
                num_stages=1,
                enable_fp_fusion=fma,
            )
    return out, state
