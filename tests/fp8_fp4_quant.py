"""Unified FP8/FP4 quantization utilities with platform dispatch.

Provides per-token MXFP4 (E2M1) quantization/dequantization and per-dim
FP8 quantization with a consistent interface across vendors.
"""

import torch

import flaggems_vllm

_vendor = flaggems_vllm.vendor_name

MXFP4_BLOCK_SIZE = 32

# ── Platform-specific implementations ──────────────────────────────────────

if _vendor == "nvidia":
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp4 as _cast_back_from_fp4,
    )
    from vllm.third_party.deep_gemm.utils.math import (
        per_custom_dims_cast_to_fp8 as _per_custom_dims_cast_to_fp8,
    )
    from vllm.third_party.deep_gemm.utils.math import (
        per_token_cast_to_fp4 as _per_token_cast_to_fp4,
    )

    def per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=128, use_packed_ue8m0=False):
        return _per_token_cast_to_fp4(x, use_ue8m0, gran_k, use_packed_ue8m0)

    def cast_back_from_fp4(packed, sf, gran_k=128, use_packed_ue8m0=False):
        return _cast_back_from_fp4(packed, sf, gran_k, use_packed_ue8m0)

    def per_custom_dims_cast_to_fp8(x, dims, use_ue8m0=False):
        return _per_custom_dims_cast_to_fp8(x, dims, use_ue8m0)

elif _vendor == "mthreads":
    import math

    def _align(x, y):
        return math.ceil(x / y) * y

    def per_custom_dims_cast_to_fp8(x, dims, use_ue8m0=False):
        """Per-dim FP8 quantization."""
        excluded_dims = tuple(i for i in range(x.dim()) if i not in set(dims))
        x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
        sf = x_amax / 448.0
        if use_ue8m0:
            sf = torch.pow(2.0, torch.round(torch.log2(sf)).clamp(-126, 127))
        x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled, sf.squeeze(-1)

    def per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=128, use_packed_ue8m0=False):
        """MXFP4 quantization (E2M1)."""
        m, n = x.shape
        assert n % 2 == 0
        padded_n = _align(n, gran_k)
        x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
        x_padded[:, :n] = x
        x_view = x_padded.view(m, -1, gran_k)
        x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
        sf = x_amax / 6.0
        if use_ue8m0:
            sf = torch.pow(2.0, torch.ceil(torch.log2(sf.clamp(min=1e-12))))
        x_scaled = x_view * (1.0 / sf.unsqueeze(2))
        # Quantize to E2M1
        magnitudes = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device, dtype=torch.float32
        )
        x_abs = x_scaled.abs()
        codes = torch.zeros_like(x_scaled, dtype=torch.int8)
        sign = (x_scaled < 0).to(torch.int8)
        for i, mag in enumerate(magnitudes):
            mask = (x_abs - mag).abs() < 0.25
            codes[mask] = i | (sign[mask] << 3)
        codes = codes.view(m, padded_n)
        codes2 = codes.view(m, padded_n // 2, 2)
        packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
        return packed[:, : n // 2].contiguous().to(torch.int8), sf

    def cast_back_from_fp4(packed, sf, gran_k=128, use_packed_ue8m0=False):
        """MXFP4 dequantization (E2M1).

        E2M1 bit layout: bit3 = sign, bits[2:1] = 2-bit exponent (bias 1),
        bit0 = 1-bit mantissa.  Same as the Triton _e2m1_to_f32 helper.
        """
        m, n2 = packed.shape
        n = n2 * 2
        nibbles = torch.zeros((m, n), dtype=torch.int32, device=packed.device)
        nibbles[:, ::2] = packed.to(torch.int32) & 0x0F
        nibbles[:, 1::2] = (packed.to(torch.int32) >> 4) & 0x0F

        sign = 1.0 - 2.0 * ((nibbles >> 3) & 1).to(torch.float32)
        e = (nibbles >> 1) & 0x3
        man = (nibbles & 0x1).to(torch.float32)
        mag = torch.where(
            e == 0,
            0.5 * man,
            torch.where(
                e == 1, 1.0 + 0.5 * man, torch.where(e == 2, 2.0 + man, 4.0 + 2.0 * man)
            ),
        )
        x_dequant = sign * mag
        group_idx = torch.arange(n, device=packed.device) // gran_k
        x_restored = x_dequant * sf[:, group_idx]
        return x_restored

else:
    raise ImportError(f"Unsupported vendor for fp8_fp4_quant: {_vendor}")


# ── High-level MXFP4 quantize / dequantize ────────────────────────────────


def quantize_to_mxfp4(x):
    """Quantize a bf16/fp32 tensor to MXFP4 (E2M1) packed format.

    Returns:
        packed: int8 [..., D//2]
        scales: int32 [..., 1] (packed ue8m0 per-block scales)
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    flat = x.float().reshape(-1, D)

    packed_2d, sf_2d = per_token_cast_to_fp4(
        flat, use_ue8m0=False, gran_k=MXFP4_BLOCK_SIZE
    )

    n_blocks = D // MXFP4_BLOCK_SIZE
    ue8m0 = (
        sf_2d.abs().clamp(min=2**-126).log2().round().clamp(-127.0, 127.0) + 127.0
    ).to(torch.uint8)
    packed_scales = torch.zeros(flat.shape[0], dtype=torch.int32, device=flat.device)
    for i in range(n_blocks):
        packed_scales |= ue8m0[:, i].to(torch.int32) << (8 * i)
    packed_scales = packed_scales.reshape(*orig_shape[:-1], 1)

    packed = packed_2d.to(torch.int8).reshape(*orig_shape[:-1], D // 2)
    return packed, packed_scales


def dequantize_mxfp4(packed, scales, head_dim):
    """Dequantize MXFP4 packed E2M1 values to float32.

    Args:
        packed: int8 [..., D//2]
        scales: int32 [..., 1]
        head_dim: D
    Returns:
        x_f32: float32 [..., D]
    """
    orig_batch = packed.shape[:-1]
    D = head_dim
    n_blocks = D // MXFP4_BLOCK_SIZE

    scales_flat = scales.squeeze(-1).reshape(-1)
    block_id = torch.arange(n_blocks, device=packed.device)
    ue8m0_bytes = ((scales_flat[..., None] >> (8 * block_id[None, :])) & 0xFF).to(
        torch.float32
    )
    sf_2d = torch.exp2(ue8m0_bytes - 127.0)

    packed_2d = packed.reshape(-1, D // 2).to(torch.int8)
    x_2d = cast_back_from_fp4(packed_2d, sf_2d, gran_k=MXFP4_BLOCK_SIZE)
    return x_2d.reshape(*orig_batch, D)
