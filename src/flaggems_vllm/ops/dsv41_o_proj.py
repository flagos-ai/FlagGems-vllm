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

"""DeepSeek-V4.1 attention output projection (``dsv41_o_proj``).

vLLM reference:
    vllm/models/deepseek_v41/nvidia/ops/o_proj.py::dsv41_o_proj
    vllm/models/deepseek_v41/common/grouped_wo_a.py::grouped_forward
    vllm/models/deepseek_v4/nvidia/ops/o_proj.py::deep_gemm_fp8_o_proj

Scope of this op (the "general path" of vLLM's ``dsv41_o_proj``):

    attn_out[:, :n_heads]                       [T, H_pad, 512] bf16
      -> inverse GPT-J RoPE on the last 64 dims of every head
      -> per-128 FP8 quant, grouped as [T, G, hpg * 512]
      -> grouped FP8 GEMM  "bhr,hdr->bhd" with wo_a [G, o_lora_rank, hpg*512]
      -> z  [T, G * o_lora_rank] bf16
      -> wo_b_proj(z), when a callback is given

``wo_b`` is a callback, as in vLLM (a regular quantized linear, optionally with
reduce-scatter): this op has no kernel for it and only forwards ``z``.

Two paths with the same results, except with n_groups=1 and fewer than 300 tokens,
where ``fp8_einsum`` takes its split-K path and rounds slightly differently:
  * Up to ``_SINGLE_LAUNCH_MAX_TOKENS_X_GROUPS``: one kernel launch
    (``dsv41_o_proj_single_launch``), so the GPU does not idle waiting for a second.
  * Above that: ``fused_inv_rope_fp8_quant`` + ``fp8_einsum``, whose quant kernel has
    more throughput once computation outweighs the launch time.

Results are close to vLLM but not bit-identical: vLLM rounds activation scales up
to a power of two on Hopper and ``fused_inv_rope_fp8_quant`` does not.

Like vLLM, the precision follows wo_a: with a BF16 wo_a (and no scale) nothing is
quantized. vLLM then rotates into BF16 and calls ``torch.bmm``; here two Triton
kernels do the same, a BF16 inverse-RoPE copy and a grouped BF16 GEMM.

Not covered (raise NotImplementedError, no torch fallback):
  * Backends other than NVIDIA Hopper.
  * vLLM's small-batch SM100/SM103 fused WO-A path (CuTe DSL, MXFP8 output).
  * Blackwell einsum recipes (1, 1, 32) / (1, 1, 128) with packed UE8M0 scales.
  * A pre-quantized ``QuantizedActivation`` input (attention kernel already
    did inverse RoPE + FP8 cast).
"""

import logging
from functools import lru_cache
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fp8_einsum import fp8_einsum
from flaggems_vllm.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

logger = logging.getLogger(__name__)

# Hopper recipe (1, 128, 128): activation scales per 128 elements along K,
# weight scales per 128x128 block. This is what fp8_einsum supports today.
_QUANT_BLOCK = 128
# Where the two paths cross on H20, in num_tokens * n_groups.
_SINGLE_LAUNCH_MAX_TOKENS_X_GROUPS = 2048


# Small batches are sensitive to CPU time here, so these per-call lookups are cached.
@lru_cache(maxsize=None)
def _is_hopper(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device)[0] == 9
    )


@lru_cache(maxsize=1)
def _single_launch_path():
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.dsv41_o_proj_single_launch import (
        dsv41_o_proj_single_launch,
        single_launch_supported,
    )

    return dsv41_o_proj_single_launch, single_launch_supported


@triton.jit(do_not_specialize=["T"])
def _inv_rope_bf16_kernel(
    o_ptr,
    pos_ptr,
    cache_ptr,
    x_ptr,
    T,
    o_sT,
    o_sH,
    cache_sP,
    NH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BT: tl.constexpr,
):
    # x[t, h, :] = inverse RoPE of o[t, h, :], rounded to BF16 as vLLM's
    # non-quantizing path does; x is a contiguous [T, NH, HEAD_DIM] copy.
    tb = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    rm = tb * BT + tl.arange(0, BT)
    m_mask = rm < T
    cols = tl.arange(0, HEAD_DIM)
    is_rope = cols >= NOPE
    rope_col = cols - NOPE
    cs = tl.maximum(rope_col >> 1, 0)
    even = (rope_col & 1) == 0
    rope_mask = m_mask[:, None] & is_rope[None, :]
    pos = tl.load(pos_ptr + rm, mask=m_mask, other=0)
    cache_row = cache_ptr + pos[:, None] * cache_sP
    cos = tl.load(cache_row + cs[None, :], mask=rope_mask, other=1.0)
    sin = tl.load(cache_row + HALF_ROPE + cs[None, :], mask=rope_mask, other=0.0)
    base = o_ptr + rm.to(tl.int64)[:, None] * o_sT + h * o_sH
    x = tl.load(base + cols[None, :], mask=m_mask[:, None], other=0.0)
    partner = tl.load(base + (cols ^ 1)[None, :], mask=rope_mask, other=0.0).to(
        tl.float32
    )
    xf = x.to(tl.float32)
    rotated = tl.where(
        even[None, :], xf * cos + partner * sin, xf * cos - partner * sin
    )
    x = tl.where(is_rope[None, :], rotated.to(tl.bfloat16), x)
    rows = rm.to(tl.int64)[:, None] * NH + h
    tl.store(x_ptr + rows * HEAD_DIM + cols[None, :], x, mask=m_mask[:, None])


@triton.jit(do_not_specialize=["T"])
def _bf16_grouped_gemm_kernel(
    x_ptr,
    w_ptr,
    z_ptr,
    T,
    z_sB,
    z_sM,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # z[g, t, :] = w[g] @ x[t, g, :]: x is [T, G, K], w is [G, N, K], both BF16.
    pid = tl.program_id(0)
    g = tl.program_id(1).to(tl.int64)
    num_m = tl.cdiv(T, BM)
    num_n: tl.constexpr = N // BN
    first_m = pid // (GROUP_M * num_n) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    local = pid % (GROUP_M * num_n)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BM + tl.arange(0, BM)
    m_mask = rm < T
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = x_ptr + rm.to(tl.int64)[:, None] * (G * K) + g * K + rk[None, :]
    b_ptrs = w_ptr + g * N * K + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(K // BK):
        a = tl.load(a_ptrs + kk * BK, mask=m_mask[:, None], other=0.0)
        acc = tl.dot(a, tl.load(b_ptrs + kk * BK), acc)
    z_ptrs = z_ptr + g * z_sB + rm[:, None] * z_sM + rn[None, :]
    tl.store(z_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])


def _bf16_gemm_config(T, G):
    """(BM, BN, BK, num_warps, num_stages), from sweeps on H20."""
    if T <= 16:
        return 16, 64, 128, 4, 4
    if T <= 32 or (G == 1 and T <= 128):
        return 32, 64, 128, 4 if G > 1 else 8, 4
    if G == 1:
        return 64, 128, 64, 4, 3
    if 128 < T <= 512:
        return 64, 64, 64, 8, 4
    return 64, 64, 64, 4, 3


def _bf16_o_proj(o, positions, cos_sin_cache, w3, z, nope_dim, rope_dim):
    """vLLM's non-quantizing branch: BF16 inverse RoPE, then a grouped BF16 GEMM."""
    G, T, N = z.shape
    n_heads, head_dim = o.shape[1], o.shape[2]
    K = w3.shape[2]
    x = torch.empty((T, n_heads, head_dim), dtype=torch.bfloat16, device=o.device)
    BT = 1 if T <= 1024 else 2
    _inv_rope_bf16_kernel[(triton.cdiv(T, BT), n_heads)](
        o,
        positions,
        cos_sin_cache,
        x,
        T,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        NH=n_heads,
        HEAD_DIM=head_dim,
        NOPE=nope_dim,
        HALF_ROPE=rope_dim // 2,
        BT=BT,
        num_warps=1 if T == 1 else 2,
    )
    BM, BN, BK, num_warps, num_stages = _bf16_gemm_config(T, G)
    if N % BN:
        BN = 64
    _bf16_grouped_gemm_kernel[(triton.cdiv(T, BM) * (N // BN), G)](
        x,
        w3,
        z,
        T,
        z.stride(0),
        z.stride(1),
        G=G,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _as_grouped_weight(
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
    n_groups: int,
    o_lora_rank: int,
    group_dim: int,
):
    """View wo_a weight/scale as [G, o_lora_rank, K] / [G, o_lora_rank/128, K/128].

    vLLM keeps wo_a as 2D [G * o_lora_rank, K] at load time; DeepGEMM's
    post-processing (``is_bmm=True``) reshapes it to 3D. Accept both, views only.
    A BF16 wo_a has no scale.
    """
    w_shape = (n_groups, o_lora_rank, group_dim)
    s_shape = (n_groups, o_lora_rank // _QUANT_BLOCK, group_dim // _QUANT_BLOCK)

    if weight.ndim == 2:
        if weight.shape != (n_groups * o_lora_rank, group_dim):
            raise ValueError(
                f"wo_a weight must be [{n_groups * o_lora_rank}, {group_dim}], "
                f"got {tuple(weight.shape)}"
            )
        weight = weight.view(w_shape)
    elif weight.shape != w_shape:
        raise ValueError(f"wo_a weight must be {w_shape}, got {tuple(weight.shape)}")

    if weight_scale is None:
        return weight, None
    if weight_scale.ndim == 2:
        if weight_scale.shape != (s_shape[0] * s_shape[1], s_shape[2]):
            raise ValueError(
                f"wo_a weight_scale must be [{s_shape[0] * s_shape[1]}, {s_shape[2]}], "
                f"got {tuple(weight_scale.shape)}"
            )
        weight_scale = weight_scale.view(s_shape)
    elif weight_scale.shape != s_shape:
        raise ValueError(
            f"wo_a weight_scale must be {s_shape}, got {tuple(weight_scale.shape)}"
        )
    return weight, weight_scale


def dsv41_o_proj(
    attn_out: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_a_weight_scale: Optional[torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    o_lora_rank: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    out: Optional[torch.Tensor] = None,
    wo_b_proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Inverse RoPE + FP8 quant + grouped wo_a + optional wo_b for DeepSeek-V4.1.

    Args:
        attn_out: [num_tokens, num_heads_padded, nope_dim + rope_dim], bf16.
            Only the first ``n_groups * heads_per_group`` heads are used
            (vLLM pads heads for the attention kernel).
        positions: [num_tokens], int64.
        cos_sin_cache: [max_position, rope_dim] float32, laid out cos || sin.
        wo_a_weight: FP8 E4M3 or BF16, [G * o_lora_rank, hpg * head_dim] or
            [G, o_lora_rank, hpg * head_dim].
        wo_a_weight_scale: for FP8, FP32 128x128 block scales, 2D or 3D matching the
            weight: vLLM's loaded ``weight_scale_inv``, not the checkpoint's E8M0
            ``scale``. None for a BF16 wo_a.
        out: optional [num_tokens, G * o_lora_rank] bf16 buffer for ``z``.
        wo_b_proj: optional callback applied to ``z``, like vLLM's ``wo_b_proj``.

    Returns:
        ``wo_b_proj(z)`` if a callback is given, otherwise
        z: [num_tokens, G * o_lora_rank] bf16.
    """
    logger.debug("GEMS DSV41 O PROJ")

    # ---- argument checks -------------------------------------------------
    if attn_out.ndim != 3:
        raise ValueError("attn_out must be [num_tokens, num_heads, head_dim]")
    if attn_out.dtype != torch.bfloat16:
        raise NotImplementedError("dsv41_o_proj: attn_out must be bfloat16")
    # Hopper only for now, checked first so no kernel runs on an unsupported GPU.
    device = attn_out.device
    if not _is_hopper(device):
        raise NotImplementedError("dsv41_o_proj requires an NVIDIA Hopper GPU")
    if any(
        t is not None and t.requires_grad
        for t in (attn_out, wo_a_weight, wo_a_weight_scale)
    ):
        raise NotImplementedError("dsv41_o_proj does not support autograd")
    head_dim = nope_dim + rope_dim
    num_tokens, num_heads_padded, attn_head_dim = attn_out.shape
    n_heads = n_groups * heads_per_group
    if attn_head_dim != head_dim:
        raise ValueError(f"head_dim must be {head_dim}, got {attn_head_dim}")
    if num_heads_padded < n_heads:
        raise ValueError(
            f"attn_out has {num_heads_padded} heads, need n_groups*heads_per_group={n_heads}"
        )
    if o_lora_rank % _QUANT_BLOCK:
        raise NotImplementedError("o_lora_rank must be a multiple of 128")

    use_fp8 = wo_a_weight.dtype == torch.float8_e4m3fn
    if wo_a_weight.dtype == torch.bfloat16:
        # vLLM's non-quantizing branch: BF16 inverse RoPE, then a BF16 GEMM.
        if wo_a_weight_scale is not None:
            raise ValueError("dsv41_o_proj: a BF16 wo_a takes no weight_scale")
        if head_dim & (head_dim - 1) or rope_dim % 2 or not 0 < rope_dim <= nope_dim:
            raise NotImplementedError(
                "dsv41_o_proj: BF16 wo_a needs a power-of-two head_dim and an even "
                "rope_dim"
            )
    elif not use_fp8:
        raise NotImplementedError("dsv41_o_proj: wo_a must be FP8 (e4m3fn) or BF16")
    elif wo_a_weight_scale is None or wo_a_weight_scale.dtype != torch.float32:
        # int32 = packed UE8M0 (SM100 recipe), not supported by fp8_einsum yet.
        raise NotImplementedError(
            "dsv41_o_proj: only FP32 128x128 block scales (Hopper recipe) are supported"
        )

    group_dim = heads_per_group * head_dim
    w3, ws3 = _as_grouped_weight(
        wo_a_weight, wo_a_weight_scale, n_groups, o_lora_rank, group_dim
    )

    out_shape = (num_tokens, n_groups * o_lora_rank)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=attn_out.device)
    else:
        if (
            tuple(out.shape) != out_shape
            or out.dtype != torch.bfloat16
            or out.device != attn_out.device
        ):
            raise ValueError(f"out must be {out_shape} bfloat16 on {attn_out.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    if num_tokens == 0:
        return out if wo_b_proj is None else wo_b_proj(out)

    # Slicing off padded heads is a view; the kernels read through strides.
    o = attn_out[:, :n_heads, :]

    if not use_fp8:
        if not w3.is_contiguous():
            raise ValueError("dsv41_o_proj: a BF16 wo_a must be contiguous")
        _bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            w3,
            out.view(num_tokens, n_groups, o_lora_rank).permute(1, 0, 2),
            nope_dim,
            rope_dim,
        )
        return out if wo_b_proj is None else wo_b_proj(out)

    # ---- small batches: one launch ---------------------------------------
    if num_tokens * n_groups <= _SINGLE_LAUNCH_MAX_TOKENS_X_GROUPS:
        dsv41_o_proj_single_launch, single_launch_supported = _single_launch_path()
        if single_launch_supported(
            o,
            positions,
            cos_sin_cache,
            w3,
            ws3,
            heads_per_group,
            rope_dim,
            _QUANT_BLOCK,
        ):
            dsv41_o_proj_single_launch(
                o,
                positions,
                cos_sin_cache,
                w3,
                ws3,
                out.view(num_tokens, n_groups, o_lora_rank).permute(1, 0, 2),
                heads_per_group,
                rope_dim,
                _QUANT_BLOCK,
            )
            return out if wo_b_proj is None else wo_b_proj(out)

    # ---- step 1: inverse RoPE + per-128 FP8 quant ------------------------
    x_fp8, x_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=_QUANT_BLOCK,
        tma_aligned_scales=False,
    )
    # x_fp8:   [T, G, group_dim]        (group-major storage, transposed view)
    # x_scale: [T, G, group_dim / 128]  FP32, token-stride 1

    # ---- step 2: grouped FP8 GEMM  z[t,g,:] = W_g @ x[t,g,:] -------------
    z = out.view(num_tokens, n_groups, o_lora_rank)
    fp8_einsum(
        "bhr,hdr->bhd",
        x_fp8,
        x_scale,
        w3,
        ws3,
        block_size=(_QUANT_BLOCK, _QUANT_BLOCK),
        out=z,
    )
    return out if wo_b_proj is None else wo_b_proj(out)
