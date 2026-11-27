# SPDX-License-Identifier: Apache-2.0
"""Ascend-optimized PLE gate implementation."""

import os

import torch
import triton
import triton.language as tl


def _detect_target_programs() -> int:
    env = os.environ.get("PLE_GATE_TARGET_PROGRAMS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        import torch_npu  # noqa: F401

        props = torch.npu.get_device_properties(torch.npu.current_device())
        for attr in (
            "num_aicore",
            "aicore_num",
            "aivector_num",
            "num_vector_core",
            "multi_processor_count",
        ):
            if hasattr(props, attr):
                val = int(getattr(props, attr))
                if val > 0:
                    return val
    except Exception:
        pass
    return 32


def _pick_chunk_h(h: int) -> int:
    env = os.environ.get("PLE_GATE_CHUNK_H")
    if env is not None:
        return max(1, int(env))
    for cand in (256, 512, 1024, 128):
        if cand <= h and h % cand == 0:
            return cand
    return min(512, triton.next_power_of_2(h))


def _pick_block_t() -> int:
    env = os.environ.get("PLE_GATE_BLOCK_T")
    if env is not None:
        return max(1, int(env))
    return 2


_TARGET_PROGRAMS = _detect_target_programs()
_LAUNCH_KWARGS = {}
_env_warps = os.environ.get("PLE_GATE_NUM_WARPS")
if _env_warps is not None:
    _LAUNCH_KWARGS["num_warps"] = int(_env_warps)
_env_stages = os.environ.get("PLE_GATE_NUM_STAGES")
if _env_stages is not None:
    _LAUNCH_KWARGS["num_stages"] = int(_env_stages)


@triton.jit(do_not_specialize=["num_tokens", "num_groups", "num_token_groups"])
def _ple_gate_kernel(
    key_ptr,
    value_ptr,
    hidden_ptr,
    nk_ptr,
    nq_ptr,
    ncw_ptr,
    gated_ptr,
    normed_ptr,
    key_rs,
    value_rs,
    eps,
    num_tokens,
    num_groups,
    num_token_groups,
    H: tl.constexpr,
    HC: tl.constexpr,
    CHUNK_H: tl.constexpr,
    NCHUNKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    g0 = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    base = s * H
    dtype: tl.constexpr = key_ptr.dtype.element_ty
    inv_H: tl.constexpr = 1.0 / H
    inv_sqrt_H: tl.constexpr = 1.0 / (H**0.5)
    EXACT: tl.constexpr = H == NCHUNKS * CHUNK_H

    row = tl.arange(0, NCHUNKS)[:, None]
    col = tl.arange(0, CHUNK_H)[None, :]
    offs2d = row * CHUNK_H + col

    if EXACT:
        nk_f = tl.load(nk_ptr + base + offs2d).to(tl.float32)
        nq_f = tl.load(nq_ptr + base + offs2d).to(tl.float32)
        ncw_f = tl.load(ncw_ptr + base + offs2d).to(tl.float32)
    else:
        hmask = offs2d < H
        nk_f = tl.load(nk_ptr + base + offs2d, mask=hmask, other=0.0).to(tl.float32)
        nq_f = tl.load(nq_ptr + base + offs2d, mask=hmask, other=0.0).to(tl.float32)
        ncw_f = tl.load(ncw_ptr + base + offs2d, mask=hmask, other=0.0).to(tl.float32)

    num_tokens64 = num_tokens.to(tl.int64)
    num_groups64 = num_groups.to(tl.int64)
    num_token_groups64 = num_token_groups.to(tl.int64)
    h_offs = offs2d[None, :, :]

    for tg in tl.range(g0, num_token_groups64, num_groups64):
        t_idx = tg * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
        tmask = t_idx < num_tokens64
        tok3 = t_idx[:, None, None]

        if EXACT:
            lmask = tmask[:, None, None]
        else:
            lmask = tmask[:, None, None] & hmask[None, :, :]

        k = tl.load(key_ptr + tok3 * key_rs + base + h_offs, mask=lmask, other=0.0)
        q = tl.load(hidden_ptr + tok3 * HC * H + base + h_offs, mask=lmask, other=0.0)
        v = tl.load(value_ptr + tok3 * value_rs + h_offs, mask=lmask, other=0.0)

        k_f = k.to(tl.float32)
        q_f = q.to(tl.float32)

        sumsq_k = tl.sum(tl.sum(k_f * k_f, axis=2), axis=1)
        sumsq_q = tl.sum(tl.sum(q_f * q_f, axis=2), axis=1)
        rms_k = tl.rsqrt(sumsq_k * inv_H + eps)
        rms_q = tl.rsqrt(sumsq_q * inv_H + eps)

        k_n = (k_f * rms_k[:, None, None] * (1.0 + nk_f[None, :, :])).to(dtype)
        q_n = (q_f * rms_q[:, None, None] * (1.0 + nq_f[None, :, :])).to(dtype)

        products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(dtype)
        dot = tl.sum(tl.sum(products.to(tl.float32), axis=2), axis=1)
        d = (dot * inv_sqrt_H).to(dtype).to(tl.float32)

        sign = tl.where(d < 0, -1.0, 0.0)
        sign = tl.where(d > 0, 1.0, sign)
        magnitude = tl.sqrt(tl.maximum(tl.abs(d), 1e-6)).to(dtype)
        g = tl.sigmoid(sign * magnitude).to(dtype).to(tl.float32)

        gated = (g[:, None, None] * v.to(tl.float32)).to(dtype)
        gf = gated.to(tl.float32)
        sumsq_g = tl.sum(tl.sum(gf * gf, axis=2), axis=1)
        rms_g = tl.rsqrt(sumsq_g * inv_H + eps)
        normed = gf * rms_g[:, None, None] * (1.0 + ncw_f[None, :, :])

        out_offs = tok3 * HC * H + base + h_offs
        tl.store(gated_ptr + out_offs, gated, mask=lmask)
        tl.store(normed_ptr + out_offs, normed, mask=lmask)


def ple_gate(
    key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps, nprograms=None
):
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    gated = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)

    chunk_h = _pick_chunk_h(h)
    nchunks = (h + chunk_h - 1) // chunk_h
    block_t = _pick_block_t()
    num_token_groups = (num_tokens + block_t - 1) // block_t

    target = nprograms if nprograms is not None else _TARGET_PROGRAMS
    num_groups = max(1, target // hc)
    num_groups = min(num_groups, num_token_groups)

    _ple_gate_kernel[(num_groups, hc)](
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        gated,
        normed,
        key.stride(0),
        value.stride(0),
        eps,
        num_tokens,
        num_groups,
        num_token_groups,
        H=h,
        HC=hc,
        CHUNK_H=chunk_h,
        NCHUNKS=nchunks,
        BLOCK_T=block_t,
        **_LAUNCH_KWARGS,
    )
    return gated, normed
