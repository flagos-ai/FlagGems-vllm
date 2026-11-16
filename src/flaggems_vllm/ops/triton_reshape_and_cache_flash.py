# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Native-compatible paged KV writes, including software E4M3 storage."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_ORDINARY = {"auto", "float16", "bfloat16", "float32", "half", "float"}
_FLOATS = (torch.float16, torch.bfloat16, torch.float32)
_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


@triton.jit
def _encode_e4m3(value, FNUZ: tl.constexpr):
    # Signed int32 bit operations also avoid an unsupported uint32->uint64
    # promotion in the Ascend vector compiler. Magnitudes fit signed int32.
    # IEEE float32 -> E4M3, RNE with finite saturation. FNUZ has bias 8,
    # unsigned zero and NaN 0x80; FN has bias 7, signed zero and NaN 0x7f.
    if FNUZ:
        maximum = 240.0
        bias = 8
        subnormal_scale = 1024.0
    else:
        maximum = 448.0
        bias = 7
        subnormal_scale = 512.0
    bits = value.to(tl.float32).to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 128
    magnitude = tl.minimum(tl.abs(value.to(tl.float32)), maximum)
    mag_bits = magnitude.to(tl.int32, bitcast=True)
    rounded = mag_bits + 0x7FFFF + ((mag_bits >> 20) & 1)
    normal = (rounded >> 20).to(tl.int32) - ((127 - bias) << 3)
    scaled = magnitude * subnormal_scale
    low = tl.floor(scaled)
    integer = low.to(tl.int32)
    fraction = scaled - low
    subnormal = integer + (
        (fraction > 0.5) | ((fraction == 0.5) & ((integer & 1) != 0))
    ).to(tl.int32)
    code = tl.where(mag_bits < ((127 + 1 - bias) << 23), subnormal, normal)
    code = code | sign.to(tl.int32)
    if FNUZ:
        code = tl.where((code & 127) == 0, 0, code)
        code = tl.where(value != value, 128, code)
    else:
        code = tl.where(value != value, 127, code)
    return code.to(tl.uint8)


@triton.jit
def _scale_fp8(value, scale, REFINE: tl.constexpr):
    quotient = tl.div_rn(value, scale)
    if REFINE:
        # MUSA 5.2 lowers precise_divf to a reciprocal approximation. Recover
        # its residual with one fused multiply-add before FP8 tie rounding.
        residual = tl.fma(-quotient, scale, value)
        corrected = quotient + tl.div_rn(residual, scale)
        # Only this interval can round to a finite, nonzero E4M3 value.
        # Bounded comparisons avoid MUSA's unlowered llvm.is.fpclass intrinsic
        # and preserve zero/NaN/Inf without feeding a NaN correction into them.
        refine = (tl.abs(quotient) > 1e-10) & (tl.abs(quotient) < 512.0)
        quotient = tl.where(refine, corrected, quotient)
    return quotient


@triton.jit
def _scale_fp8_musa(value, scale, REFINE: tl.constexpr):
    # MUSA flushes FP32 subnormal operands during division. Decode their
    # integer significands first, divide normal floats, then restore a bounded
    # power of two. Clipping only affects values already rounding to FP8 zero
    # or finite saturation. No floating arithmetic touches the original bits.
    xb = value.to(tl.int32, bitcast=True)
    yb = scale.to(tl.int32, bitcast=True)
    xe, ye = (xb >> 23) & 255, (yb >> 23) & 255
    xf, yf = xb & 0x7FFFFF, yb & 0x7FFFFF
    mx = (xf | tl.where(xe != 0, 0x800000, 0)).to(tl.float32)
    my = (yf | tl.where(ye != 0, 0x800000, 0)).to(tl.float32)
    quotient = tl.div_rn(mx, my)
    residual = tl.fma(-quotient, my, mx)
    corrected = quotient + tl.div_rn(residual, my)
    refine = (quotient > 1e-10) & (quotient < 33554432.0)
    quotient = tl.where(refine, corrected, quotient)
    exponent = tl.where(xe == 0, 1, xe) - tl.where(ye == 0, 1, ye)
    exponent = tl.minimum(tl.maximum(exponent, -100), 100)
    power = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    quotient = quotient * power
    infinity = tl.full((), 0x7F800000, tl.int32).to(tl.float32, bitcast=True)
    nan = tl.full((), 0x7FC00000, tl.int32).to(tl.float32, bitcast=True)
    quotient = tl.where(xe == 255, infinity, quotient)
    quotient = tl.where(ye == 255, 0.0, quotient)
    invalid = ((xe == 255) & (xf != 0)) | ((ye == 255) & (yf != 0))
    invalid = invalid | ((xe == 255) & (ye == 255))
    quotient = tl.where(invalid, nan, quotient)
    sign = (xb ^ yb) & -2147483648
    return (quotient.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)


# Keep the precise-division dependency unchanged on the other backends.
if runtime.device.vendor_name == "mthreads":
    _scale_fp8 = _scale_fp8_musa  # noqa: F811 - intentional backend dependency


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("triton_reshape_and_cache_flash"),
    key=["SPEC"],
    # SQL tuning storage accepts scalar keys; repr preserves every tuple field.
    strategy=repr,
    restore_value=["KC", "VC"],
)
@triton.jit
def _cache_flash(
    K,
    V,
    KC,
    VC,
    Slots,
    KScale,
    VScale,
    SPEC: tl.constexpr,
    TILE: tl.constexpr,
):
    # One immutable metadata tuple keeps all layout/dtype/stride paths distinct
    # without paying LibEntry's Python per-parameter specialization overhead.
    # SPEC[0] is the token-count tuning key; the launch grid already bounds it.
    H: tl.constexpr = SPEC[1]
    D: tl.constexpr = SPEC[2]
    BS: tl.constexpr = SPEC[3]
    B: tl.constexpr = SPEC[4]
    X: tl.constexpr = SPEC[5]
    HEAD_MAJOR: tl.constexpr = SPEC[6]
    FORMAT: tl.constexpr = SPEC[7]
    QUANTIZE: tl.constexpr = SPEC[8]
    K_BYTES: tl.constexpr = SPEC[9]
    V_BYTES: tl.constexpr = SPEC[10]
    KS: tl.constexpr = SPEC[11]
    VS: tl.constexpr = SPEC[12]
    CB: tl.constexpr = SPEC[13]
    CP: tl.constexpr = SPEC[14]
    CH: tl.constexpr = SPEC[15]
    CKD: tl.constexpr = SPEC[16]
    CVD: tl.constexpr = SPEC[17]
    REFINE_DIV: tl.constexpr = SPEC[18]
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slots + token).to(tl.int64)
    # Do not read source or scales for a negative slot.
    if (slot >= 0) & (slot < B * BS):
        feature = tl.program_id(1).to(tl.int64) * TILE + tl.arange(0, TILE)
        head, dim = feature // D, feature % D
        block, offset = slot // BS, slot % BS
        if HEAD_MAJOR:
            ki = block * CB + head * CH + (dim // X) * CKD + offset * X + dim % X
            vi = block * CB + head * CH + dim * CVD + offset
        else:
            ki = block * CB + offset * CP + head * CH + dim
            vi = ki
        mask = feature < H * D
        kval = tl.load(K + token * KS + feature, mask, other=0)
        vval = tl.load(V + token * VS + feature, mask, other=0)
        if FORMAT:
            if not K_BYTES:
                kval = kval.to(tl.float32)
                if QUANTIZE:
                    # Preserve FP8 halfway cases for non-power-of-two scales.
                    kval = _scale_fp8(kval, tl.load(KScale), REFINE_DIV)
                kval = _encode_e4m3(kval, FORMAT == 2)
            if not V_BYTES:
                vval = vval.to(tl.float32)
                if QUANTIZE:
                    vval = _scale_fp8(vval, tl.load(VScale), REFINE_DIV)
                vval = _encode_e4m3(vval, FORMAT == 2)
        tl.store(KC + ki, kval, mask)
        tl.store(VC + vi, vval, mask)


def triton_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    kv_cache_dtype: str,
    k_scale,
    v_scale,
) -> None:
    """Write first ``len(slot_mapping)`` tokens into ordinary/global-FP8 cache.

    Sources have dense [H,D] interiors and independent token strides. 4D K/V
    share all strides; head-major K/V share block/head strides. Nonnegative
    slots must be unique and below B*BS; duplicate writes have no ordering
    guarantee. No tensor data are inspected on the host.

    Like native vLLM, fp8* selectors mean the platform E4M3 format, including
    the legacy fp8_e5m2 selector: they do not select an E5M2 encoding. Typed
    caches must match that format. uint8 caches carry its raw bytes.
    """
    logger.debug("GEMS TRITON_RESHAPE_AND_CACHE_FLASH")
    tensors = (key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all tensor arguments, including scales, must be tensors")
    if key.device.type == "cpu" or any(t.device != key.device for t in tensors):
        raise ValueError("all tensors must be on the same accelerator device")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("key/value must have equal [T,H,D] shapes")
    tokens, heads, dim = key.shape
    if heads <= 0 or dim <= 0:
        raise ValueError("H and D must be positive")
    for source in (key, value):
        if (
            source.stride(2) != 1
            or source.stride(1) != dim
            or source.stride(0) < heads * dim
        ):
            raise ValueError("source H,D must be dense and token rows must not overlap")
    if (
        slot_mapping.ndim != 1
        or not slot_mapping.is_contiguous()
        or slot_mapping.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("slot_mapping must be contiguous int32/int64 [N]")
    if slot_mapping.numel() > tokens:
        raise ValueError("slot_mapping length cannot exceed source token count")
    if any(s.dtype != torch.float32 or s.numel() != 1 for s in (k_scale, v_scale)):
        raise ValueError("scales must be single-element float32 device tensors")
    if not isinstance(kv_cache_dtype, str) or not (
        kv_cache_dtype in _ORDINARY or kv_cache_dtype.startswith("fp8")
    ):
        raise ValueError("unsupported kv_cache_dtype selector")
    quantize = kv_cache_dtype.startswith("fp8")
    platform_fp8 = (
        torch.float8_e4m3fnuz
        if runtime.device.vendor_name in ("hygon", "amd")
        else torch.float8_e4m3fn
    )
    if key_cache.dtype != value_cache.dtype:
        raise ValueError("key/value caches must have equal dtypes")
    cache_dtype = key_cache.dtype
    if quantize:
        if cache_dtype not in (platform_fp8, torch.uint8):
            raise ValueError(
                "quantized cache must hold the platform FP8 dtype or uint8 bytes"
            )
        target = platform_fp8
    else:
        if cache_dtype not in _FLOATS + _FP8:
            raise ValueError("ordinary cache must have a supported floating dtype")
        target = cache_dtype
    if key.dtype not in _FLOATS + _FP8 or value.dtype not in _FLOATS + _FP8:
        raise ValueError("key/value must have supported floating dtypes")
    if any(t.dtype in _FP8 and t.dtype != target for t in (key, value)):
        raise NotImplementedError("cross-format FP8 source conversion is unsupported")
    head_major = key_cache.ndim == 5
    if head_major:
        blocks, cache_heads, chunks, block_size, x = key_cache.shape
        if (
            x <= 0
            or dim % x
            or (cache_heads, chunks * x) != (heads, dim)
            or value_cache.shape != (blocks, heads, dim, block_size)
        ):
            raise ValueError("invalid head-major K/V cache shapes")
        kb, kh, kd, ks, kx = key_cache.stride()
        vb, vh, vd, vs = value_cache.stride()
        if (
            (ks, kx, vs) != (x, 1, 1)
            or kb != vb
            or kh != vh
            or kd < block_size * x
            or vd < block_size
            or kh < max(chunks * kd, dim * vd)
            or kb < heads * kh
        ):
            raise ValueError(
                "head-major requires dense slot/x and shared nonoverlapping block/head strides"
            )
        page_stride, head_stride, dim_k, dim_v = 0, kh, kd, vd
    else:
        if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
            raise ValueError("Flash caches must have equal [B,BS,H,D] shapes")
        blocks, block_size, cache_heads, cache_dim = key_cache.shape
        if (cache_heads, cache_dim) != (heads, dim):
            raise ValueError("cache H,D must match sources")
        kb, page_stride, head_stride, last = key_cache.stride()
        if (
            key_cache.stride() != value_cache.stride()
            or last != 1
            or head_stride < dim
            or page_stride < heads * head_stride
            or kb < block_size * page_stride
        ):
            raise ValueError(
                "4D caches require shared nonoverlapping strides and contiguous D"
            )
        x, dim_k, dim_v = 1, 0, 0
    if blocks <= 0 or block_size <= 0:
        raise ValueError("cache block count and block size must be positive")
    if slot_mapping.numel() == 0:
        return None
    fmt = (
        2
        if target == torch.float8_e4m3fnuz
        else (1 if target == torch.float8_e4m3fn else 0)
    )
    k_bytes, v_bytes = key.dtype in _FP8, value.dtype in _FP8
    # CUDA-compatible index guards avoid repeated device-object normalization while
    # retaining correct launches when the caller's current device differs.
    guard = (
        torch_device_fn._DeviceGuard(key.device.index)
        if runtime.device.vendor_name in ("nvidia", "thead")
        else torch_device_fn.device(key.device)
    )
    with guard:
        _cache_flash[
            lambda meta: (slot_mapping.numel(), triton.cdiv(heads * dim, meta["TILE"]))
        ](
            key.view(torch.uint8) if k_bytes else key,
            value.view(torch.uint8) if v_bytes else value,
            key_cache.view(torch.uint8) if fmt else key_cache,
            value_cache.view(torch.uint8) if fmt else value_cache,
            slot_mapping,
            k_scale,
            v_scale,
            (
                slot_mapping.numel(),
                heads,
                dim,
                block_size,
                blocks,
                x,
                head_major,
                fmt,
                quantize,
                k_bytes,
                v_bytes,
                key.stride(0),
                value.stride(0),
                kb,
                page_stride,
                head_stride,
                dim_k,
                dim_v,
                runtime.device.vendor_name == "mthreads",
            ),
        )
    return None
