# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure Torch SiLU/mul + group-128 E4M3FN reference for Ascend benchmarks."""

import torch


def _encode_e4m3fn(values):
    # values are FP32, clamped to [-448,448]. Produce E4M3FN bytes with RNE.
    isnan = torch.isnan(values)
    safe = torch.where(isnan, 0.0, values).contiguous()
    bits = safe.view(torch.int32)
    magnitude = bits & 0x7FFFFFFF
    shifted = magnitude - 0x3C000000
    rounded = shifted + 0x7FFFF + ((shifted >> 20) & 1)
    normal = rounded >> 20
    subnormal = (magnitude.view(torch.float32) * 512.0 + 8388608.0).view(
        torch.int32
    ) - 0x4B000000
    encoded = torch.where(magnitude >= 0x3C800000, normal, subnormal)
    sign = (bits >> 24) & 0x80
    encoded = (encoded | sign).to(torch.uint8)
    encoded = torch.where(isnan, (sign | 0x7F).to(torch.uint8), encoded)
    return encoded.view(torch.float8_e4m3fn)


def silu_mul_per_token_group_quant_fp8_colmajor(
    input, output=None, use_ue8m0=None, eps=1e-10, clamp_limit=None
):
    """Torch composition with software FP8 encoding for devices without FP8 casts.

    Keep vLLM/Triton's dtype rules: SiLU narrows to the input dtype, BF16
    multiplication promotes to FP32, and FP16 multiplication narrows to FP16.
    Scales are FP32 with column-major layout; FP8 conversion uses round-to-nearest-even.
    """
    x = input
    assert x.ndim == 2
    assert x.shape[0] % 128 == 0 and x.shape[1] % 256 == 0
    if use_ue8m0 is None:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        use_ue8m0 = is_deep_gemm_e8m0_used()
    if not x.numel():
        if output is None:
            output = torch.empty(
                (x.shape[0], x.shape[1] // 2),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            )
        scales = torch.empty_strided(
            (x.shape[0], x.shape[1] // 256),
            (1, x.shape[0]),
            dtype=torch.float32,
            device=x.device,
        )
        return output, scales
    gate, up = x.chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(-clamp_limit, clamp_limit)
    up = up.float().add(0.0).to(x.dtype)
    gate = gate.float()
    silu = (gate / (1.0 + torch.exp(-gate))).to(x.dtype)
    # Triton promotes BF16 arithmetic to FP32, unlike eager Torch BF16 mul.
    y = silu.float() * up.float() if x.dtype == torch.bfloat16 else (silu * up).float()
    grouped = y.reshape(x.shape[0], -1, 128)
    scales = grouped.abs().amax(dim=-1).clamp_min(eps) * (1.0 / 448.0)
    if use_ue8m0:
        b = scales.contiguous().view(torch.int32)
        exact = ((b + 0x7FFFFF) & 0x7F800000).view(torch.float32)
        scales = torch.where(scales < float("inf"), exact, scales)
    quant = (grouped / scales.unsqueeze(-1)).clamp(-448, 448)
    q = _encode_e4m3fn(quant.reshape_as(y))
    column_major = torch.empty_strided(
        scales.shape, (1, x.shape[0]), dtype=scales.dtype, device=scales.device
    )
    column_major.copy_(scales)
    scales = column_major
    if output is not None:
        output.view(torch.uint8).copy_(q.view(torch.uint8))
        q = output
    return q, scales
