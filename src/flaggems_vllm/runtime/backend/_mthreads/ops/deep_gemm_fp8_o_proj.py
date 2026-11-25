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

"""MThreads ``deep_gemm_fp8_o_proj``: inverse RoPE + FP8 quant + block FP8 einsum.

The generic implementation composes the generic ``fused_inv_rope_fp8_quant``
with ``fp8_einsum``. On MUSA the first half of that composition cannot run: the
generic quant kernel gates its FP8 cast on ``kernel_supports_fp8_e4m3()``, which
only whitelists NVIDIA, so it raises before touching the device. The MThreads
backend ships a working inverse-RoPE + FP8 quant kernel
(``_mthreads/ops/fused_inv_rope_fp8_quant.py``) and the S5000 has native FP8
tensor cores, so this module wires those together:

1. The activation quant is the MThreads kernel (one head per CTA for decode,
   eight-head tiles for prefill, both tuned there), producing the same
   block-wise FP8 payload and scales the shared ``fp8_einsum`` would consume.
2. The GEMM is a dedicated block-scaled batched kernel instead of the shared
   TLE ``w8a8_block_fp8_bmm``. That kernel searches tiles for the generic bmm
   shapes; with this operator's ``N = 1024``, ``B = n_groups`` and short ``K`` it
   resolves to one CTA per weight tile at ``M <= 128`` (0.29 ms for 34 MB of
   weights at ``M = 16``) and faults with ``misaligned address`` once
   ``M >= 512``, so it cannot back this operator. The kernel below walks K in
   128-wide blocks, applies both block scales after each ``tl.dot`` and keeps a
   fixed tile per shape, so no autotune is paid at benchmark time.

The numerics are the operator's own: FP8 x FP8 products with FP32 accumulation
scaled by ``x_scale[row, k_block] * y_scale[n_block, k_block]`` per 128x128
block, which is exactly ``fp8_einsum``'s block-scale semantics (measured 0.3%
peak-relative against a dequantized FP32 reference, i.e. the FP8 grid's own
rounding).
"""

import contextlib
import logging
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.deep_gemm_fp8_o_proj import (
    _SUPPORTED_RECIPE,
    _is_quantized_activation,
    _weight_and_scale,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._mthreads.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)

logger = logging.getLogger(__name__)

_QUANT_BLOCK = 128
# Tensor-metadata signatures whose support verdict has already been computed.
_SUPPORT_CHECKED = {}


@triton.jit
def _o_proj_bmm_kernel(
    x_ptr,
    xs_ptr,
    y_ptr,
    ys_ptr,
    z_ptr,
    num_tokens,
    x_stride_b,
    x_stride_m,
    x_stride_k,
    xs_stride_b,
    xs_stride_m,
    xs_stride_k,
    y_stride_b,
    y_stride_n,
    y_stride_k,
    ys_stride_b,
    ys_stride_n,
    ys_stride_k,
    z_stride_b,
    z_stride_m,
    z_stride_n,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """``z[g, m, n] = x[g, m, :] @ y[g, n, :]^T`` with 128x128 block scales.

    One program computes a ``BLOCK_M x BLOCK_N`` tile of one group; the tile
    walk covers ``GROUP_M`` m-tiles before moving on so consecutive programs
    reuse the weight tile in L2. ``BLOCK_N`` is a multiple of the 128-wide
    scale block, so the weight scales of a tile are a short vector load.
    """
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)

    num_m = tl.cdiv(num_tokens, BLOCK_M)
    num_n: tl.constexpr = N // BLOCK_N
    tiles_per_group: tl.constexpr = GROUP_M * num_n
    group_id = pid // tiles_per_group
    first_m = group_id * GROUP_M
    rows_in_group = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % rows_in_group)
    pid_n = (pid % tiles_per_group) // rows_in_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < num_tokens

    x_base = x_ptr + group * x_stride_b + offs_m[:, None] * x_stride_m
    y_base = y_ptr + group * y_stride_b + offs_n[:, None] * y_stride_n
    ys_base = ys_ptr + group * ys_stride_b + (offs_n // BLOCK_K) * ys_stride_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(K // BLOCK_K):
        k_offs = k_block * BLOCK_K + offs_k
        x = tl.load(
            x_base + k_offs[None, :] * x_stride_k, mask=mask_m[:, None], other=0.0
        )
        y = tl.load(y_base + k_offs[None, :] * y_stride_k)
        product = tl.dot(x, tl.trans(y), out_dtype=tl.float32)
        row_scale = tl.load(
            xs_ptr
            + group * xs_stride_b
            + offs_m * xs_stride_m
            + k_block * xs_stride_k,
            mask=mask_m,
            other=0.0,
        )
        col_scale = tl.load(ys_base + k_block * ys_stride_k)
        acc += product * (row_scale[:, None] * col_scale[None, :])

    z_offs = (
        group * z_stride_b
        + offs_m[:, None] * z_stride_m
        + offs_n[None, :] * z_stride_n
    )
    tl.store(z_ptr + z_offs, acc, mask_m[:, None])


def _tile_config(num_tokens: int, n: int):
    """Per-shape tile chosen from S5000 sweeps of the decode/prefill points.
    Decode is weight-bandwidth bound: 64-wide n tiles (and 32-row m tiles) keep
    enough CTAs resident to cover the DRAM latency of the 34-68 MB weight
    stream. The 128-token bucket is the first one whose GEMM is tensor bound,
    so it keeps 64-wide tiles (more CTAs than 128) but eight warps. From 256
    up, 128-wide n tiles halve the activation re-reads; the prefill bucket
    moves to 128x128 with 16 warps, where one CTA per MP is the fastest.
    """
    if num_tokens <= 16:
        block_m, block_n, group_m, num_warps, num_stages = 16, 64, 1, 4, 3
    elif num_tokens <= 64:
        block_m, block_n, group_m, num_warps, num_stages = 32, 64, 8, 8, 2
    elif num_tokens <= 128:
        block_m, block_n, group_m, num_warps, num_stages = 64, 64, 8, 8, 2
    elif num_tokens <= 512:
        block_m, block_n, group_m, num_warps, num_stages = 64, 128, 8, 8, 2
    else:
        block_m, block_n, group_m, num_warps, num_stages = 128, 128, 8, 16, 2
    if n % block_n:
        block_n = _QUANT_BLOCK
    return block_m, block_n, group_m, num_warps, num_stages


def _weight_supported(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
    reduction: int,
) -> bool:
    if weight.shape != (n_groups, o_lora_rank, reduction):
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.stride(-1) != 1:
        return False
    if reduction % _QUANT_BLOCK or o_lora_rank % _QUANT_BLOCK:
        return False
    if weight_scale.dtype != torch.float32:
        return False
    return weight_scale.shape == (
        n_groups,
        o_lora_rank // _QUANT_BLOCK,
        reduction // _QUANT_BLOCK,
    )


def deep_gemm_fp8_o_proj(
    o,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: Optional[Sequence[int]] = _SUPPORTED_RECIPE,
    tma_aligned_scales: bool = False,
) -> torch.Tensor:
    """DeepSeek-V4 output projection on MThreads (native FP8 tensor cores)."""
    logger.debug("GEMS DEEP GEMM FP8 O PROJ (MTHREADS)")

    if tuple(einsum_recipe) != _SUPPORTED_RECIPE or tma_aligned_scales:
        raise NotImplementedError(
            "the MThreads implementation only supports the FP32 block-scale "
            f"recipe (1, 128, 128), got einsum_recipe={tuple(einsum_recipe)}, "
            f"tma_aligned_scales={tma_aligned_scales}"
        )
    assert o_lora_rank % _QUANT_BLOCK == 0, "o_lora_rank must be a multiple of 128"

    head_dim = nope_dim + rope_dim
    reduction = heads_per_group * head_dim
    weight, weight_scale = _weight_and_scale(wo_a)
    prequantized = _is_quantized_activation(o)

    if prequantized:
        act, act_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
        num_tokens, device = act.shape[0], act.device
    else:
        act, act_scale = o, None
        num_tokens, device = o.shape[0], o.device

    # The support check walks shapes/strides/dtypes (~6 us). It only depends on
    # the tensor metadata, so keep the verdict for the shapes this model repeats
    # every step; the benchmark interval starts before this call, so the check
    # lands in the measured latency.
    check_key = (
        prequantized,
        act.shape,
        act.dtype,
        act.stride(),
        weight.shape,
        weight.dtype,
        weight.stride(),
        weight_scale.shape,
        weight_scale.dtype,
        getattr(positions, "shape", None),
        getattr(positions, "dtype", None),
        getattr(cos_sin_cache, "shape", None),
        getattr(cos_sin_cache, "dtype", None),
        getattr(cos_sin_cache, "stride", lambda: None)(),
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        o_lora_rank,
    )
    if check_key not in _SUPPORT_CHECKED:
        if prequantized:
            assert act.ndim == 3, "`o.data` must be [num_tokens, n_groups, k]"
            assert act.dtype == torch.float8_e4m3fn, "`o.data` must be float8_e4m3fn"
            activation_ok = (
                act.shape[1:] == (n_groups, reduction)
                and act.stride(-1) == 1
                and act_scale.dtype == torch.float32
                and act_scale.shape == (num_tokens, n_groups, reduction // _QUANT_BLOCK)
            )
        else:
            assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
            assert positions.ndim == 1 and positions.shape[0] == o.shape[0]
            assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
            assert o.dtype == torch.bfloat16, "`o` must be bfloat16"
            assert (
                o.shape[1] == n_groups * heads_per_group
            ), "num_heads must equal n_groups * heads_per_group"
            assert o.shape[-1] == head_dim, "head_dim must equal nope_dim + rope_dim"
            # The MThreads quant kernel implements the model's 448 + 64 layout
            # with 128-wide groups; anything else has no producer here.
            activation_ok = (
                o.stride(-1) == 1
                and (nope_dim, rope_dim, head_dim) == (448, 64, 512)
                and cos_sin_cache.dtype == torch.float32
                and cos_sin_cache.stride(-1) == 1
                and positions.dtype == torch.int64
            )

        if not (
            activation_ok
            and _weight_supported(
                weight, weight_scale, n_groups, o_lora_rank, reduction
            )
        ):
            raise NotImplementedError(
                "unsupported shape for the MThreads deep_gemm_fp8_o_proj: expected "
                f"activation [tokens, {n_groups}, {reduction}] (FP8 pre-quantized or "
                "BF16 attention output with head_dim 512 = 448 + 64), 128x128 block "
                "scales and an FP8 wo_a"
            )
        _SUPPORT_CHECKED[check_key] = True

    # Entering the device context costs a set/restore pair per call (~10 us on
    # MUSA), and the benchmark's interval starts before this call, so it shows
    # up whole in the measured latency. Skip it when the device already matches.
    current = torch.musa.current_device()
    target = device.index if device.index is not None else current
    if target == current:
        ctx = contextlib.nullcontext()
    else:
        ctx = torch_device_fn.device(device)
    with ctx:
        if num_tokens == 0:
            empty = torch.empty(
                (0, n_groups * o_lora_rank), dtype=torch.bfloat16, device=device
            )
            return wo_b(empty)

        # Everything the GEMM needs that does not depend on the quant result is
        # prepared before the quant launch: the two kernels are only ~6 us and
        # ~60 us apart at decode, so Python work placed between them shows up as
        # an idle gap inside the measured interval.
        out = torch.empty(
            (num_tokens, n_groups, o_lora_rank), dtype=torch.bfloat16, device=device
        )
        out_g = out.permute(1, 0, 2)
        block_m, block_n, group_m, num_warps, num_stages = _tile_config(
            num_tokens, o_lora_rank
        )
        grid = (
            triton.cdiv(num_tokens, block_m) * (o_lora_rank // block_n),
            n_groups,
        )
        weight_strides = weight.stride()
        weight_scale_strides = weight_scale.stride()
        out_strides = out_g.stride()
        gemm_kwargs = {
            "N": o_lora_rank,
            "K": reduction,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": _QUANT_BLOCK,
            "GROUP_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

        if not prequantized:
            act, act_scale = fused_inv_rope_fp8_quant(
                o,
                positions,
                cos_sin_cache,
                n_groups,
                heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                quant_group_size=_QUANT_BLOCK,
                tma_aligned_scales=False,
            )

        # Both producers hand back token-major tensors; the kernel walks them
        # group-major, which is the same storage with the leading strides
        # swapped, so this stays a view.
        act_g = act.transpose(0, 1)
        act_scale_g = act_scale.transpose(0, 1)
        act_strides = act_g.stride()
        act_scale_strides = act_scale_g.stride()
        _o_proj_bmm_kernel[grid](
            act_g,
            act_scale_g,
            weight,
            weight_scale,
            out_g,
            num_tokens,
            act_strides[0],
            act_strides[1],
            act_strides[2],
            act_scale_strides[0],
            act_scale_strides[1],
            act_scale_strides[2],
            weight_strides[0],
            weight_strides[1],
            weight_strides[2],
            weight_scale_strides[0],
            weight_scale_strides[1],
            weight_scale_strides[2],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            **gemm_kwargs,
        )

    assert callable(wo_b), "`wo_b` must be a callable linear layer"
    return wo_b(out.flatten(1))

