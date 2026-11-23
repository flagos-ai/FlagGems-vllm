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
"""Ascend vector-core implementation of the standalone KDA gate."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime


@triton.jit
def _fused_kda_gate_ascend_softplus_kernel(
    G,
    A,
    BIAS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    STEPS: tl.constexpr,
):
    # A complete token row is contiguous. Keep only BT rows live in the 910B
    # vector buffer, then reuse their parameters across successive row tiles.
    program = tl.program_id(0)
    if T * H * D >= 2**31:
        program = program.to(tl.int64)
    first = program * (BT * STEPS)
    cols = tl.arange(0, H * D)
    if HAS_BIAS:
        bias = tl.load(BIAS + cols).to(tl.float32)
    scales = -tl.exp(tl.load(A + tl.arange(0, H)).to(tl.float32))
    scales = tl.reshape(tl.broadcast_to(scales[:, None], (H, D)), (H * D,))
    for step in range(STEPS):
        rows = first + step * BT + tl.arange(0, BT)
        offsets = rows[:, None] * (H * D) + cols[None, :]
        mask = rows[:, None] < T
        value = tl.load(G + offsets, mask, other=0).to(tl.float32)
        if HAS_BIAS:
            value += bias[None, :]
        scaled = value * BETA
        softplus = tl.where(
            scaled > THRESHOLD,
            value,
            (1.0 / BETA) * tl.log(1.0 + tl.exp(scaled)),
        )
        tl.store(Y + offsets, scales[None, :] * softplus, mask)


@triton.jit
def _fused_kda_gate_ascend_kernel(
    G,
    A,
    BIAS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if T * H * D >= 2**31:
        pid = pid.to(tl.int64)
    if BT > 0:
        head_groups: tl.constexpr = H // BH
        head = (pid % head_groups) * BH
        rows = (pid // head_groups) * BT + tl.arange(0, BT)
        channels = tl.arange(0, BH * BD)
        cols = head * D + channels
        offsets = rows[:, None] * (H * D) + cols[None, :]
        mask = rows[:, None] < T
        if BD != D:
            # Only single-head tiles need to mask padding within the row.
            mask = mask & (channels[None, :] < D)
        value = tl.load(G + offsets, mask, other=0).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(BIAS + cols, channels < BH * D, other=0).to(tl.float32)
            value += bias[None, :]
        if BH == 1:
            scale = tl.exp(tl.load(A + head).to(tl.float32))
        else:
            heads = head + tl.arange(0, BH)
            head_scale = tl.exp(tl.load(A + heads).to(tl.float32))
            scale = tl.reshape(tl.broadcast_to(head_scale[:, None], (BH, D)), (BH * D,))
            scale = scale[None, :]
    else:
        # Flat output blocks keep stores disjoint even for unaligned head widths.
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < T * H * D
        cols = offsets % (H * D)
        value = tl.load(G + offsets, mask, other=0).to(tl.float32)
        if HAS_BIAS:
            value += tl.load(BIAS + cols, mask, other=0).to(tl.float32)
        scale = tl.exp(tl.load(A + cols // D, mask, other=0).to(tl.float32))
    if SAFE_GATE:
        result = LOWER_BOUND / (1.0 + tl.exp(-(scale * value)))
    else:
        scaled = value * BETA
        softplus = tl.where(
            scaled > THRESHOLD, value, (1.0 / BETA) * tl.log(1.0 + tl.exp(scaled))
        )
        result = -scale * softplus
    tl.store(Y + offsets, result, mask)


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    safe_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """Return a fresh FP32 KDA gate of shape ``[..., H, head_k_dim]``.

    Ascend backend for the public fused_kda_gate interface. Preserve both
    thresholded softplus and bounded sigmoid semantics, optional bias, and
    the original contiguous-input validation. The GLM-5.3-Flash softplus
    path uses bounded vector tiles and reuses parameters across row tiles.
    Forward inference only; unsupported inputs raise explicitly.
    """
    if runtime.device.name == "cpu":
        raise NotImplementedError("fused_kda_gate does not support CPU")
    if g.ndim < 1 or head_k_dim < 1 or A.numel() < 1:
        raise ValueError("KDA gate requires a nonempty head/key dimension")
    heads = A.numel()
    hidden = heads * head_k_dim
    if g.shape[-1] != hidden:
        raise ValueError("g.shape[-1] must equal A.numel() * head_k_dim")
    if g_bias is not None and g_bias.numel() != hidden:
        raise ValueError("g_bias must contain H * head_k_dim elements")
    for tensor in (g, A, g_bias):
        if tensor is None:
            continue
        if tensor.device != g.device:
            raise ValueError("KDA gate inputs must be on the same device")
        if tensor.device.type != runtime.device.name:
            raise NotImplementedError("KDA gate requires the runtime accelerator")
        if tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise NotImplementedError("KDA gate requires floating point inputs")
        if tensor.dtype == torch.float64 and (not runtime.device.support_fp64):
            raise NotImplementedError("KDA gate FP64 is unsupported on this backend")
        if tensor.dtype == torch.bfloat16 and (not runtime.device.support_bf16):
            raise NotImplementedError("KDA gate BF16 is unsupported on this backend")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise NotImplementedError("KDA gate requires contiguous inputs")
    output = torch.empty(
        (*g.shape[:-1], heads, head_k_dim), device=g.device, dtype=torch.float32
    )
    tokens = g.numel() // hidden
    if tokens == 0:
        return output
    constants = (
        tokens,
        heads,
        head_k_dim,
        beta,
        threshold,
        safe_gate,
        -5.0 if lower_bound is None else lower_bound,
        g_bias is not None,
    )
    with runtime.torch_device_fn.device(g.device):
        if not safe_gate and heads == 16 and (head_k_dim == 128) and (tokens >= 128):
            steps = 2 if tokens < 512 else 8
            _fused_kda_gate_ascend_softplus_kernel[triton.cdiv(tokens, 4 * steps),](
                g,
                A,
                g_bias,
                output,
                tokens,
                heads,
                head_k_dim,
                beta,
                threshold,
                g_bias is not None,
                BT=4,
                STEPS=steps,
                multibuffer=False,
            )
            return output
        aligned_head = head_k_dim <= 256 and head_k_dim % 16 == 0
        bt = 0
        bh = 1
        bd = 0
        if aligned_head:
            bd = triton.next_power_of_2(head_k_dim)
            if safe_gate and heads == 16 and (head_k_dim == 128) and (tokens > 128):
                bh, bt = (8, 16)
            else:
                max_bt = (64 if tokens <= 512 else 128) if safe_gate else 32
                bt = min(triton.next_power_of_2(tokens), max_bt, 16384 // bd)
        block = 256
        grid = (
            triton.cdiv(tokens, bt) * (heads // bh)
            if aligned_head
            else triton.cdiv(g.numel(), block)
        )
        _fused_kda_gate_ascend_kernel[grid,](
            g, A, g_bias, output, *constants, BT=bt, BH=bh, BD=bd, BLOCK=block
        )
    return output


__all__ = ["fused_kda_gate"]
