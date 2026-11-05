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

Results are close to vLLM but not bit-identical: vLLM rounds activation scales up
to a power of two on Hopper and ``fused_inv_rope_fp8_quant`` does not.

Not covered (raise NotImplementedError, no torch fallback):
  * Backends other than NVIDIA Hopper.
  * vLLM's small-batch SM100/SM103 fused WO-A path (CuTe DSL, MXFP8 output).
  * Blackwell einsum recipes (1, 1, 32) / (1, 1, 128) with packed UE8M0 scales.
  * BF16 wo_a (vLLM's ``torch.bmm`` branch): FlagGems-vllm's
    ``fused_inv_rope_fp8_quant`` has no ``quantize=False`` mode yet.
  * A pre-quantized ``QuantizedActivation`` input (attention kernel already
    did inverse RoPE + FP8 cast).
"""

import logging
from typing import Callable, Optional

import torch

from flaggems_vllm.ops.fp8_einsum import fp8_einsum
from flaggems_vllm.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

logger = logging.getLogger(__name__)

# Hopper recipe (1, 128, 128): activation scales per 128 elements along K,
# weight scales per 128x128 block. This is what fp8_einsum supports today.
_QUANT_BLOCK = 128


def _as_grouped_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
    group_dim: int,
):
    """View wo_a weight/scale as [G, o_lora_rank, K] / [G, o_lora_rank/128, K/128].

    vLLM keeps wo_a as 2D [G * o_lora_rank, K] at load time; DeepGEMM's
    post-processing (``is_bmm=True``) reshapes it to 3D. Accept both, views only.
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
    wo_a_weight_scale: torch.Tensor,
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
        wo_a_weight: FP8 E4M3, [G * o_lora_rank, hpg * head_dim] or
            [G, o_lora_rank, hpg * head_dim].
        wo_a_weight_scale: FP32 128x128 block scales, 2D or 3D matching the weight:
            vLLM's loaded ``weight_scale_inv``, not the checkpoint's E8M0 ``scale``.
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
    if (
        device.type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(device)[0] != 9
    ):
        raise NotImplementedError("dsv41_o_proj requires an NVIDIA Hopper GPU")
    if any(t.requires_grad for t in (attn_out, wo_a_weight, wo_a_weight_scale)):
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

    if wo_a_weight.dtype != torch.float8_e4m3fn:
        # vLLM falls back to torch.bmm on a BF16 inverse-rotated input here.
        # That would be a torch compute fallback, which this repo forbids.
        raise NotImplementedError(
            "dsv41_o_proj: only FP8 (e4m3fn) wo_a is supported; BF16 wo_a needs a "
            "non-quantizing inverse-RoPE kernel"
        )
    if wo_a_weight_scale.dtype != torch.float32:
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

    # ---- step 1: inverse RoPE + per-128 FP8 quant ------------------------
    # Slicing off padded heads is a view; the kernel reads through strides.
    o = attn_out[:, :n_heads, :]
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
