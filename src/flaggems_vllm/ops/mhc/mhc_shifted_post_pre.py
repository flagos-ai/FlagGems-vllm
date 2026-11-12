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

"""Shifted (delayed) mHC post -> pre boundary for DSV4.1.

One call performs the mHC work of a sublayer boundary, with the coefficient
computation shifted one sublayer ahead:

1. ``post``      -- mix the residual streams with the coefficients computed by
   the *previous* boundary and add the sublayer output.
2. ``project``   -- ``residual_cur @ fn^T`` plus the row sum of squares, in the
   same pass (the RMS scaling needs both).
3. ``coefficients`` -- pre / post / comb coefficients for the *next* boundary.
   ``pre`` is returned instead of being applied, which is what makes the
   coefficient GEMM overlap-able with the sublayer in vLLM.
4. ``collapse``  -- this layer's input, using the ``pre_mix`` carried in from the
   previous boundary, plus the stream mean (``aux``) when requested.
5. ``norm``      -- optional RMSNorm of the collapsed input.

Layout contract (see ``mhc_post.MixLayout``): ``comb_res_mix`` and the returned
``comb_mix`` use the vLLM convention, i.e. ``out[t, j] = sum_i comb[t, i, j] *
residual[t, i]``.  In FlagGems terms that is ``MixLayout.TRANSPOSED``, the
``mhc_post`` default -- **not** ``NATURAL``.  The two differ silently: a wrong
layout produces plausible-looking but wrong values rather than an error.

Autotune note: every tile size is a tuner META field and no tile is passed as a
launch-only constexpr.  FlagTree's aabs rewrites tile constexprs while tuning,
and a rewritten tile that is not a config field cannot be persisted, which is
how ``mhc_rms_norm`` breaks for non-power-of-two hidden sizes.
"""

import weakref

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.mhc.mhc_post import MixLayout, mhc_post
from flaggems_vllm.utils import libentry, libtuner

_FN_BF16_CACHE: dict = {}


@triton.jit
def _cast_bf16_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    value = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, value, mask=mask)


def _fn_bf16(fn: torch.Tensor) -> torch.Tensor:
    """bfloat16 copy of ``fn``, cached until the weight moves or is edited.

    ``_version`` is skipped for inference tensors: they do not track one, and
    reading it raises.  Inference tensors cannot be edited in place either, so
    the cache stays valid.
    """
    key = id(fn)
    version = None if fn.is_inference() else fn._version
    ptr = fn.data_ptr()
    entry = _FN_BF16_CACHE.get(key)
    if entry is not None:
        ref, cached_version, cached_ptr, cached = entry
        if ref() is fn and cached_version == version and cached_ptr == ptr:
            return cached

    out = torch.empty(fn.shape, dtype=torch.bfloat16, device=fn.device)
    if fn.numel():
        _cast_bf16_kernel[lambda meta: (triton.cdiv(fn.numel(), meta["BLOCK"]),)](
            fn, out, fn.numel(), BLOCK=1024
        )
    _FN_BF16_CACHE[key] = (
        weakref.ref(fn, lambda ref, key=key: _FN_BF16_CACHE.pop(key, None)),
        version,
        ptr,
        out,
    )
    return out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mhc_shifted_project"),
    key=["num_tokens", "hc_hidden_size"],
)
@triton.jit
def _mhc_shifted_project_kernel(
    a_ptr,  # (num_tokens, hc_hidden_size) bfloat16
    bt_ptr,  # (hc_hidden_size, mix_size) bfloat16, fn^T
    out_ptr,  # (num_tokens, mix_size) float32
    ssq_ptr,  # (num_tokens,) float32
    num_tokens,
    hc_hidden_size,
    mix_size,
    a_stride_n,
    bt_stride_k,
    bt_stride_j,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``a @ fn^T`` with the row sum of squares folded into the same K loop."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < num_tokens
    n_mask = offs_n < mix_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ssq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, hc_hidden_size, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hc_hidden_size
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_n + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            bt_ptr + offs_k[:, None] * bt_stride_k + offs_n[None, :] * bt_stride_j,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)
        # Keep the reduction after the dot and num_stages at 1 (see the tuner
        # config).  A loop-carried reduction sharing the pipelined K loop with
        # tl.dot miscompiles under FlagTree 0.7: the product comes out tens of
        # percent off, and the error changes from run to run.  The plain dot is
        # exact at every num_stages, so only the fusion is affected.
        a_f32 = a.to(tl.float32)
        ssq += tl.sum(a_f32 * a_f32, axis=1)
    tl.store(
        out_ptr + offs_m[:, None] * mix_size + offs_n[None, :],
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )
    # The sum of squares does not depend on the N tile: only the tile at 0
    # writes it, otherwise blocks would race on the same addresses.
    if tl.program_id(1) == 0:
        tl.store(ssq_ptr + offs_m, ssq, mask=m_mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("mhc_shifted_coeff"), key=["num_tokens"])
@triton.jit
def _mhc_shifted_coeff_kernel(
    mix_ptr,  # (num_tokens, MIX_SIZE) float32 -- raw projection
    ssq_ptr,  # (num_tokens,) float32 -- row sum of squares of the projection input
    scale_ptr,  # (3,) float32
    base_ptr,  # (MIX_SIZE,) float32
    pre_out_ptr,  # (num_tokens, HC) float32 -- next boundary's pre-mix
    post_out_ptr,  # (num_tokens, HC) float32 -- next boundary's post-mix
    comb_out_ptr,  # (num_tokens, HC, HC) float32 -- next boundary's comb-mix
    num_tokens,
    hc_hidden_size,
    RSUM_EPS: tl.constexpr,
    HC: tl.constexpr,
    MIX_SIZE: tl.constexpr,
    HC_PAD: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    SINK_REPEAT: tl.constexpr,
):
    """Turn the raw projection into the three coefficient groups.

    Everything vLLM hard-codes in its generic sinkhorn helper is a parameter
    here: the pre epsilon, the sinkhorn epsilon and the post multiplier.
    Reusing a helper that bakes in ``2 * sigmoid`` would be correct only for
    ``hc_post_mult_value == 2``.
    """
    n = tl.program_id(0)
    scale_0 = tl.load(scale_ptr + 0)
    scale_1 = tl.load(scale_ptr + 1)
    scale_2 = tl.load(scale_ptr + 2)
    rms_inv = tl.rsqrt(tl.load(ssq_ptr + n) / hc_hidden_size + RSUM_EPS)
    mix_base = mix_ptr + n * MIX_SIZE

    idx = tl.arange(0, HC_PAD)
    valid = idx < HC

    pre_logits = tl.load(mix_base + idx, mask=valid, other=0.0) * rms_inv * scale_0
    pre_logits += tl.load(base_ptr + idx, mask=valid, other=0.0)
    tl.store(
        pre_out_ptr + n * HC + idx,
        tl.sigmoid(pre_logits) + PRE_EPS,
        mask=valid,
    )

    post_idx = HC + idx
    post_logits = tl.load(mix_base + post_idx, mask=valid, other=0.0) * rms_inv
    post_logits = post_logits * scale_1 + tl.load(
        base_ptr + post_idx, mask=valid, other=0.0
    )
    tl.store(
        post_out_ptr + n * HC + idx,
        tl.sigmoid(post_logits) * POST_MULT,
        mask=valid,
    )

    # comb: Sinkhorn-normalised (hc_mult, hc_mult) block, row-major [i, j] with
    # i the residual stream and j the output stream (vLLM's orientation).
    comb_idx = 2 * HC + idx[:, None] * HC + idx[None, :]
    comb_valid = valid[:, None] & valid[None, :]
    comb = tl.load(mix_base + comb_idx, mask=comb_valid, other=0.0) * rms_inv
    comb = comb * scale_2 + tl.load(base_ptr + comb_idx, mask=comb_valid, other=0.0)
    comb = tl.where(comb_valid, comb, float("-inf"))
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None]
    comb = tl.where(comb_valid, comb + SINK_EPS, 0.0)
    comb = comb / (tl.sum(comb, axis=0)[None, :] + SINK_EPS)
    for _ in range(SINK_REPEAT - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + SINK_EPS)
    tl.store(
        comb_out_ptr + n * HC * HC + idx[:, None] * HC + idx[None, :],
        comb,
        mask=comb_valid,
    )


@libentry()
@libtuner(configs=runtime.get_tuned_config("mhc_shifted_collapse"), key=["hidden_size"])
@triton.jit
def _mhc_shifted_collapse_kernel(
    res_ptr,  # (num_tokens, HC, hidden_size) bfloat16 -- post-mapped residual
    pre_ptr,  # (num_tokens, HC) float32, or a dummy when USE_PRE is off
    out_ptr,  # (num_tokens, hidden_size) bfloat16
    aux_ptr,  # (num_tokens, hidden_size) bfloat16, or a dummy
    hidden_size,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PRE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
):
    """Collapse the HC streams with the carried pre-mix and average them."""
    n = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden_size
    base = n * HC * hidden_size
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    total = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for j in tl.static_range(HC):
        value = tl.load(res_ptr + base + j * hidden_size + offs, mask=mask, other=0.0)
        value = value.to(tl.float32)
        total += value
        if USE_PRE:
            weight = tl.load(pre_ptr + n * HC + j)
            acc += weight * value
        elif j == 0:
            # Model entry: the collapse selects stream zero, it does not average.
            acc = value
    tl.store(out_ptr + n * hidden_size + offs, acc, mask=mask)
    if WRITE_AUX:
        tl.store(aux_ptr + n * hidden_size + offs, total / HC, mask=mask)


@libentry()
@libtuner(configs=runtime.get_tuned_config("mhc_shifted_norm"), key=["D"])
@triton.jit
def _mhc_shifted_norm_kernel(
    X,  # (num_tokens, D) bfloat16
    W,  # (D,) bfloat16, or a dummy when HAS_W is off
    Y,  # (num_tokens, D) bfloat16
    D,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
):
    """RMSNorm over a row, in two blocked passes."""
    row = tl.program_id(0)
    row_off = X + row * D
    ssq = tl.zeros((1,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        value = tl.load(row_off + offs, mask=mask, other=0.0).to(tl.float32)
        ssq += tl.sum(value * value, axis=0)
    inv = tl.rsqrt(ssq / D + EPS)
    for d0 in range(0, D, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        mask = offs < D
        value = tl.load(row_off + offs, mask=mask, other=0.0).to(tl.float32)
        result = value * inv
        if HAS_W:
            result = result * tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + row * D + offs, result, mask=mask)


def _check_common(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    pre_mix,
    norm_weight,
    sinkhorn_repeat,
    norm_eps,
):
    """Validate the ABI without touching tensor data."""
    if not isinstance(x, torch.Tensor):
        raise NotImplementedError(
            "mHC shifted post/pre operates on a plain tensor; the unfinalized "
            "MoE output path (all-reduce fused with mHC) belongs to the plugin "
            "layer, not to this operator"
        )
    if residual.dtype != torch.bfloat16:
        raise NotImplementedError("mHC shifted post/pre residual must be bfloat16")
    if residual.device.type not in ("cuda", "musa", "npu"):
        raise NotImplementedError("mHC shifted post/pre requires an accelerator tensor")
    if residual.dim() != 3:
        raise ValueError(
            f"residual must be 3-D (N, hc, H), got {tuple(residual.shape)}"
        )
    num_tokens, hc, hidden_size = residual.shape
    if x.shape != (num_tokens, hidden_size):
        raise ValueError(
            f"x must have shape ({num_tokens}, {hidden_size}), got {tuple(x.shape)}"
        )
    if not (x.is_contiguous() and residual.is_contiguous()):
        raise NotImplementedError(
            "mHC shifted post/pre requires contiguous x and residual"
        )
    if post_layer_mix.shape not in ((num_tokens, hc, 1), (num_tokens, hc)):
        raise ValueError(
            f"post_layer_mix must be (N, hc, 1) or (N, hc), got "
            f"{tuple(post_layer_mix.shape)}"
        )
    if comb_res_mix.shape != (num_tokens, hc, hc):
        raise ValueError(
            f"comb_res_mix must be ({num_tokens}, {hc}, {hc}), got "
            f"{tuple(comb_res_mix.shape)}"
        )
    if post_layer_mix.dtype != torch.float32 or comb_res_mix.dtype != torch.float32:
        raise NotImplementedError("mHC post/comb coefficients must be float32")
    if not (post_layer_mix.is_contiguous() and comb_res_mix.is_contiguous()):
        raise NotImplementedError("mHC post/comb coefficients must be contiguous")
    mix_size = hc * (hc + 2)
    if fn.dtype != torch.float32 or not fn.is_contiguous():
        raise ValueError("fn must be a contiguous float32 tensor")
    if fn.shape != (mix_size, hc * hidden_size):
        raise ValueError(
            f"fn must have shape ({mix_size}, {hc * hidden_size}), got "
            f"{tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,) or hc_scale.dtype != torch.float32:
        raise ValueError("hc_scale must be a float32 tensor of 3 values")
    if hc_base.shape != (mix_size,) or hc_base.dtype != torch.float32:
        raise ValueError(f"hc_base must be a float32 tensor of {mix_size} values")
    if not (hc_scale.is_contiguous() and hc_base.is_contiguous()):
        raise ValueError("hc_scale and hc_base must be contiguous")
    if pre_mix is not None:
        if pre_mix.shape != (num_tokens, hc) or pre_mix.dtype != torch.float32:
            raise ValueError(
                f"pre_mix must be a float32 ({num_tokens}, {hc}) tensor, got "
                f"{tuple(pre_mix.shape)} {pre_mix.dtype}"
            )
        if not pre_mix.is_contiguous():
            raise NotImplementedError("pre_mix must be contiguous")
    if norm_weight is not None and norm_weight.shape != (hidden_size,):
        raise ValueError(
            f"norm_weight must have shape ({hidden_size},), got "
            f"{tuple(norm_weight.shape)}"
        )
    if sinkhorn_repeat < 1:
        raise ValueError(f"sinkhorn_repeat must be >= 1, got {sinkhorn_repeat}")
    if norm_eps < 0:
        raise ValueError(f"norm_eps must be non-negative, got {norm_eps}")
    return num_tokens, hc, hidden_size, mix_size


def _mhc_shifted_project(a_flat, fn_bf16_t, num_tokens, mix_size):
    """``a @ fn^T`` (fp32) plus the row sum of squares, in one pass."""
    out = torch.empty(num_tokens, mix_size, dtype=torch.float32, device=a_flat.device)
    ssq = torch.empty(num_tokens, dtype=torch.float32, device=a_flat.device)
    hc_hidden_size = a_flat.shape[1]
    _mhc_shifted_project_kernel[
        lambda meta: (
            triton.cdiv(num_tokens, meta["BLOCK_M"]),
            triton.cdiv(mix_size, meta["BLOCK_N"]),
        )
    ](
        a_flat,
        fn_bf16_t,
        out,
        ssq,
        num_tokens,
        hc_hidden_size,
        mix_size,
        a_flat.stride(0),
        fn_bf16_t.stride(0),
        fn_bf16_t.stride(1),
    )
    return out, ssq


def mhc_shifted_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
    capture_aux=False,
    *,
    stream=None,
    reduce_results=False,
):
    """One shifted mHC post block followed by the next delayed pre block.

    The coefficients used here were computed by the previous boundary; the
    coefficients returned here are consumed by the next one.  See the module
    docstring for the pipeline and the ``comb`` layout contract.

    Args:
        x: bfloat16 sublayer output, ``(num_tokens, hidden_size)``.
        residual: bfloat16 residual streams, ``(num_tokens, hc, hidden_size)``.
        post_layer_mix: float32, ``(num_tokens, hc, 1)`` or ``(num_tokens, hc)``.
        comb_res_mix: float32 ``(num_tokens, hc, hc)``, vLLM orientation.
        fn: float32 projection, ``(hc * (hc + 2), hc * hidden_size)``.
        hc_scale: float32 ``(3,)``.
        hc_base: float32 ``(hc * (hc + 2),)``.
        rms_eps: RMS epsilon of the coefficient projection.
        hc_pre_eps: pre-mix epsilon.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: post-mix multiplier.
        sinkhorn_repeat: number of Sinkhorn iterations (>= 1).
        pre_mix: float32 ``(num_tokens, hc)`` carried from the previous
            boundary, or ``None`` at the model entry (collapse selects stream 0).
        norm_weight: optional bfloat16 RMSNorm weight of ``hidden_size``.
        norm_eps: RMSNorm epsilon, ignored when ``norm_weight`` is ``None``.
        capture_aux: also return the mean over the post-mapped streams.
        stream: accepted for ABI compatibility with vLLM's overlapped path.  This
            implementation runs on the caller's current stream, which keeps the
            result correct (the caller joins an unused stream as a no-op) but
            gives up the coefficient/compute overlap.
        reduce_results: tensor-parallel all-reduce is a collective and is not
            performed here; pass the reduced ``x`` instead.

    Returns:
        ``(residual_cur, post_mix, comb_mix, layer_input, next_pre, aux)`` with
        dtypes/shapes ``(N, hc, H) bf16``, ``(N, hc, 1) fp32``,
        ``(N, hc, hc) fp32``, ``(N, H) bf16``, ``(N, hc) fp32`` and either
        ``(N, H) bf16`` (``capture_aux``) or an empty ``(0, H)`` tensor.
    """
    if reduce_results:
        raise NotImplementedError(
            "mHC shifted post/pre does not perform the tensor-parallel "
            "all-reduce; reduce x before the call or keep vLLM's overlap path"
        )
    num_tokens, hc, hidden_size, mix_size = _check_common(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        pre_mix,
        norm_weight,
        sinkhorn_repeat,
        norm_eps,
    )
    device = residual.device

    if num_tokens == 0:
        empty_aux = residual.new_empty(0, hidden_size)
        return (
            torch.empty((0, hc, hidden_size), dtype=torch.bfloat16, device=device),
            torch.empty((0, hc, 1), dtype=torch.float32, device=device),
            torch.empty((0, hc, hc), dtype=torch.float32, device=device),
            torch.empty((0, hidden_size), dtype=torch.bfloat16, device=device),
            torch.empty((0, hc), dtype=torch.float32, device=device),
            empty_aux,
        )

    # (1) post, with the coefficients the previous boundary produced.  The comb
    # layout is vLLM's, which is mhc_post's TRANSPOSED default.
    residual_cur = mhc_post(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        mix_layout=MixLayout.TRANSPOSED,
    )

    # (2) projection + row sum of squares in one pass over the residual.
    a_flat = residual_cur.view(num_tokens, hc * hidden_size)
    mixes, sqsum = _mhc_shifted_project(a_flat, _fn_bf16(fn).t(), num_tokens, mix_size)

    # (3) the three coefficient groups for the next boundary.
    post_mix = torch.empty(num_tokens, hc, 1, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc, hc, dtype=torch.float32, device=device)
    next_pre = torch.empty(num_tokens, hc, dtype=torch.float32, device=device)
    _mhc_shifted_coeff_kernel[(num_tokens,)](
        mixes,
        sqsum,
        hc_scale,
        hc_base,
        next_pre,
        post_mix,
        comb_mix,
        num_tokens,
        hc * hidden_size,
        RSUM_EPS=rms_eps,
        HC=hc,
        MIX_SIZE=mix_size,
        HC_PAD=triton.next_power_of_2(hc),
        PRE_EPS=hc_pre_eps,
        SINK_EPS=hc_sinkhorn_eps,
        POST_MULT=hc_post_mult_value,
        SINK_REPEAT=sinkhorn_repeat,
    )

    # (4) collapse with the carried pre-mix, plus the stream mean.
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    aux = (
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        if capture_aux
        else residual_cur.new_empty(0, hidden_size)
    )
    _mhc_shifted_collapse_kernel[
        lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_H"]))
    ](
        residual_cur,
        pre_mix if pre_mix is not None else post_mix,
        layer_input,
        aux if capture_aux else layer_input,
        hidden_size,
        HC=hc,
        USE_PRE=pre_mix is not None,
        WRITE_AUX=capture_aux,
    )

    # (5) optional RMSNorm of the collapsed input.
    if norm_weight is not None:
        normed = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
        _mhc_shifted_norm_kernel[(num_tokens,)](
            layer_input,
            norm_weight,
            normed,
            hidden_size,
            EPS=norm_eps,
            HAS_W=True,
        )
        layer_input = normed

    return residual_cur, post_mix, comb_mix, layer_input, next_pre, aux


__all__ = ["mhc_shifted_post_pre"]
