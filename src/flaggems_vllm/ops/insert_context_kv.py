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

import threading

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fp8_storage import encode_e4m3fn
from flaggems_vllm.utils import libentry


@triton.jit
def _encode_scaled_fp8(
    values, inverse, post_scale, ENCODE: tl.constexpr = encode_e4m3fn
):
    bits = values.to(tl.int32, bitcast=True)
    magnitude = (bits & 0x7FFFFFFF).to(tl.float32, bitcast=True)
    scaled = magnitude * inverse * post_scale
    scaled_bits = scaled.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    payload = ENCODE(scaled)
    signed = payload | ((bits >> 24) & 0x80).to(tl.uint8)
    # fmaxf(value, -448) maps NaN to -448. Preserve the source sign of
    # zero even when a backend uses a multiply that canonicalizes it.
    invalid = (scaled_bits > 0x7F800000) | (
        (inverse == 0.0) & ((bits & 0x7FFFFFFF) >= 0x7F800000)
    )
    return tl.where(invalid, 0xFE, signed).to(tl.uint8)


@triton.jit
def _fma_rn(a, b, c):
    return tl.fma(a, b, c)


@triton.jit
def _rotate_kv(kv, trig, offsets, FMA: tl.constexpr = _fma_rn):
    values = tl.load(kv + offsets).to(tl.float32)
    rope = offsets >= 448
    partner = tl.load(kv + (offsets ^ 1), mask=rope, other=0).to(tl.float32)
    pair = (offsets - 448) // 2
    cos = tl.load(trig + pair, mask=rope, other=1)
    sin = tl.load(trig + 32 + pair, mask=rope, other=0)
    even = (offsets & 1) == 0
    product = partner * sin
    # Flip the IEEE sign bit to preserve -0 for an exactly zero product.
    negative = (product.to(tl.int32, bitcast=True) ^ -2147483648).to(
        tl.float32, bitcast=True
    )
    rotated = FMA(
        tl.where(even, values, partner),
        tl.where(even, cos, sin),
        tl.where(even, negative, values * cos),
    )
    return tl.where(rope, rotated, values)


@triton.jit
def _correct_reciprocal(mantissa, estimate, half_ulp):
    residual = tl.fma(-mantissa, estimate, 1.0)
    midpoint = mantissa * half_ulp
    return (residual > midpoint).to(tl.int32) - (residual < -midpoint).to(tl.int32)


@triton.jit
def _encode_with_scale(values, scale, ENCODE: tl.constexpr = encode_e4m3fn):
    bits = scale.to(tl.int32, bitcast=True)
    # Positive scales <= 2**-128 have an infinite FP32 reciprocal. Substitute
    # 1 in the finite arithmetic, then select the saturated bytes below.
    overflow = bits <= 0x00200000
    bits = tl.where(overflow, 0x3F800000, bits)
    subnormal = bits < 0x00800000
    normalized = tl.where(
        subnormal, bits.to(tl.float32).to(tl.int32, bitcast=True), bits
    )
    exponent = (normalized >> 23) - tl.where(subnormal, 276, 127)
    mantissa = ((normalized & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    estimate = 1.0 / mantissa
    # Refine the platform division, then compare the exact FMA residual
    # against half an ULP in the normalized reciprocal's binade.
    estimate = tl.fma(estimate, tl.fma(-mantissa, estimate, 1.0), estimate)

    # Round a subnormal inverse on its own grid to avoid double rounding.
    # Keep it normalized during multiplication on FTZ devices.
    inverse_is_subnormal = exponent >= 126
    grid_inverse = tl.where(exponent == 126, 2.0**-23, 2.0**-22)
    units_per_one = tl.where(exponent == 126, 2.0**23, 2.0**22)
    units = (estimate * units_per_one).to(tl.int32)
    center = tl.where(
        inverse_is_subnormal, units.to(tl.float32) * grid_inverse, estimate
    )
    half_ulp = tl.where(inverse_is_subnormal, grid_inverse * 0.5, 2.0**-25)
    correction = _correct_reciprocal(mantissa, center, half_ulp)
    normal_bits = estimate.to(tl.int32, bitcast=True) + correction
    normal_bits -= tl.minimum(exponent, 125) << 23
    inverse = tl.where(
        inverse_is_subnormal,
        (units + correction).to(tl.float32) * 2.0**-23,
        normal_bits.to(tl.float32, bitcast=True),
    )
    post_scale = tl.where(inverse_is_subnormal, 2.0**-126, 1.0)
    encoded = _encode_scaled_fp8(values, inverse, post_scale, ENCODE)
    saturated = tl.where(values > 0, 0x7E, 0xFE)
    return tl.where(overflow, saturated, encoded).to(tl.uint8)


@triton.jit
def _quantize_groups(values, ENCODE: tl.constexpr = encode_e4m3fn):
    bits = values.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    magnitude = tl.where(bits > 0x7F800000, 0.0, tl.abs(values))
    amax = tl.maximum(tl.max(magnitude, axis=1), 1.0e-4)
    bits = amax.to(tl.int32, bitcast=True)
    # 448 = 1.75 * 2**8. Comparing the mantissa to 1.75 computes
    # ceil(log2(amax / 448)) plus the UE8M0 bias exactly.
    finite_code = (bits >> 23) - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    infinite = bits == 0x7F800000
    code = tl.where(infinite, 255, finite_code).to(tl.uint8)
    inverse = tl.where(
        infinite, 0.0, ((254 - finite_code) << 23).to(tl.float32, bitcast=True)
    )
    return _encode_scaled_fp8(values, inverse[:, None], 1.0, ENCODE), code


@triton.jit
def _store_v4_groups(
    kv,
    data,
    scales,
    START: tl.constexpr,
    COUNT: tl.constexpr,
    ENCODE: tl.constexpr = encode_e4m3fn,
):
    groups = START + tl.arange(0, COUNT)
    offsets = groups[:, None] * 64 + tl.arange(0, 64)[None, :]
    values = tl.load(kv + offsets).to(tl.float32)
    quantized, code = _quantize_groups(values, ENCODE)
    tl.store(data + offsets, quantized)
    tl.store(scales + groups, code)


@triton.jit
def _store_v4(kv, data, scales):
    _store_v4_groups(kv, data, scales, 0, 4)
    _store_v4_groups(kv, data, scales, 4, 2)
    _store_v4_groups(kv, data, scales, 6, 1)
    tl.store(scales + 7, 0)


@triton.jit
def _insert_context_kv_body(
    kv,
    cache,
    rope_cache,
    token,
    slot,
    positions,
    cos_sin_cache,
    fp8_scale,
    page_stride,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    FORMAT: tl.constexpr,
):
    position = tl.load(positions + token)
    page = cache + (slot // BLOCK_SIZE) * page_stride
    row = slot % BLOCK_SIZE
    trig = cos_sin_cache + position * 64

    source = kv + token * 512
    if FORMAT == "bf16" or FORMAT == "fp8":
        offsets = tl.arange(0, 512)
        values = _rotate_kv(source, trig, offsets)
        target = page + row * row_stride + offsets
        if FORMAT == "bf16":
            tl.store(target, values.to(tl.bfloat16))
        else:
            tl.store(target, _encode_with_scale(values, tl.load(fp8_scale)))
    elif FORMAT == "mxfp8":
        groups = tl.arange(0, 16)
        offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
        rotated = _rotate_kv(source, trig, offsets)
        rounded = rotated.to(tl.bfloat16).to(tl.float32)
        # The BF16 conversion can canonicalize zero on some backends.
        # Reattach the input sign bit without changing the rounded magnitude.
        bits = rounded.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        sign = rotated.to(tl.int32, bitcast=True) & -2147483648
        quantized, code = _quantize_groups((bits | sign).to(tl.float32, bitcast=True))
        data = page + row * 528
        tl.store(data + offsets, quantized)
        tl.store(data + 512 + groups, code)
    else:
        data = page + row * 576
        scales = page + BLOCK_SIZE * 576 + row * 8
        # Use disjoint stores for the seven FP8 groups and the BF16 tail.
        _store_v4(source, data, scales)
        tail = tl.arange(0, 64)
        rotated = _rotate_kv(source, trig, 448 + tail)
        rope = rope_cache
        rope += (slot // BLOCK_SIZE) * (page_stride // 2) + row * 288
        tl.store(rope + 224 + tail, rotated.to(tl.bfloat16))


@triton.jit(
    do_not_specialize=[
        "kv",
        "cache",
        "rope_cache",
        "slots",
        "positions",
        "cos_sin_cache",
        "fp8_scale",
        "page_stride",
        "row_stride",
    ]
)
def _insert_context_kv_kernel(
    kv,
    cache,
    rope_cache,
    slots,
    positions,
    cos_sin_cache,
    fp8_scale,
    page_stride,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    FORMAT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + token)
    if slot < 0:
        return
    _insert_context_kv_body(
        kv,
        cache,
        rope_cache,
        token,
        slot,
        positions,
        cos_sin_cache,
        fp8_scale,
        page_stride,
        row_stride,
        BLOCK_SIZE,
        FORMAT,
    )


_kernel_entry = libentry()(_insert_context_kv_kernel)
_compiled_kernels = {}
_compiled_kernel_lock = threading.Lock()
_triton_version = tuple(int(part) for part in triton.__version__.split(".")[:2])
_compiled_abi = (3, 3) <= _triton_version <= (3, 9)


def _launch_insert(grid, args, device_index):
    # Pointer dtypes follow from the validated format. Strides specialize only
    # on their signed integer ABI width, not on their values.
    key = (device_index, args[10], args[9], args[7] >= 2**31, args[8] >= 2**31)
    if _compiled_abi:
        with _compiled_kernel_lock:
            kernel = _compiled_kernels.get(key)
        if kernel is not None:
            # Standard CompiledKernel resolves this call's stream and hooks.
            kernel[(grid[0], 1, 1)](*args)
            return
    if not _compiled_abi:
        _insert_context_kv_kernel[grid](*args, enable_fp_fusion=False)
        return
    kernel, _ = _kernel_entry.run(
        *args, grid=grid, warmup=False, enable_fp_fusion=False
    )
    with _compiled_kernel_lock:
        if len(_compiled_kernels) >= 128:
            _compiled_kernels.pop(next(iter(_compiled_kernels)))
        _compiled_kernels[key] = kernel


def _make_insert_context_kv(launch_insert):
    def insert_context_kv(
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        cache_block_size,
        fp8_scale=None,
        kv_mxfp8=False,
    ):
        """Apply interleaved RoPE to normalized DSpark KV and update a paged cache.

        Negative slots are ignored. Active slots must be unique and in bounds,
        positions must index the trigonometric table, and inputs and cache may
        not overlap. Input values must be finite and the optional scale positive.
        """
        kv_shape = kv.shape
        kv_device = kv.device
        if len(kv_shape) != 2 or kv_shape[1] != 512 or kv.dtype != torch.bfloat16:
            raise NotImplementedError("kv must be bfloat16 [tokens, 512]")
        if not kv.is_contiguous():
            raise NotImplementedError("kv must be contiguous")
        for indices in (slot_mapping, position_ids):
            if (
                indices.ndim != 1
                or indices.dtype != torch.int64
                or not indices.is_contiguous()
            ):
                raise NotImplementedError(
                    "slots and positions must be contiguous int64 vectors"
                )
        num_slots = slot_mapping.numel()
        if position_ids.numel() != kv_shape[0] or num_slots > kv_shape[0]:
            raise NotImplementedError(
                "positions must match KV rows and slots may omit padding"
            )
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[1] != 64
            or cos_sin_cache.dtype != torch.float32
            or not cos_sin_cache.is_contiguous()
        ):
            raise NotImplementedError(
                "cos_sin_cache must be contiguous float32 [max_pos, 64]"
            )
        if (
            k_cache.device != kv_device
            or slot_mapping.device != kv_device
            or position_ids.device != kv_device
            or cos_sin_cache.device != kv_device
        ):
            raise NotImplementedError("all tensors must be on the KV device")
        cache_shape = k_cache.shape
        cache_strides = k_cache.stride()
        cache_dtype = k_cache.dtype
        if (
            not isinstance(cache_block_size, int)
            or cache_block_size <= 0
            or len(cache_shape) != 3
            or cache_shape[1] != cache_block_size
            or cache_strides[2] != 1
        ):
            raise NotImplementedError(
                "cache must be [pages, block_size, width] with unit last stride"
            )

        if cache_dtype == torch.uint8:
            cache_format, width = ("mxfp8", 528) if kv_mxfp8 else ("packed", 584)
            if cache_strides[1] != width or cache_strides[0] < cache_block_size * width:
                raise NotImplementedError(
                    "packed cache requires dense records within each page"
                )
            if cache_format == "packed" and (
                k_cache.data_ptr() % 2 or cache_strides[0] % 2
            ):
                raise NotImplementedError(
                    "BF16 fields of packed V4 pages must be aligned"
                )
        elif cache_dtype == torch.bfloat16:
            cache_format, width = "bf16", 512
        elif cache_dtype == getattr(torch, "float8_e4m3fn", None):
            cache_format, width = "fp8", 512
            if (
                fp8_scale is None
                or fp8_scale.numel() != 1
                or fp8_scale.dtype != torch.float32
                or fp8_scale.device != kv_device
            ):
                raise NotImplementedError(
                    "FP8 cache requires one float32 scale on the KV device"
                )
        else:
            raise NotImplementedError(
                "cache dtype must be uint8, bfloat16 or float8_e4m3fn"
            )
        if cache_shape[2] != width:
            raise NotImplementedError("cache width does not match the selected format")
        page_span = (cache_block_size - 1) * cache_strides[1] + width
        if cache_strides[1] < width or cache_strides[0] < page_span:
            raise NotImplementedError("cache rows and pages must not overlap")
        if num_slots == 0:
            return
        if kv_device.type == "cpu":
            raise NotImplementedError("insert_context_kv requires an accelerator")

        cache_storage = (
            triton.reinterpret(k_cache, tl.uint8) if cache_format == "fp8" else k_cache
        )
        # Supply the typed pointer at the host boundary: some compilers reject
        # cross-width pointer casts inside a kernel. Reinterpret moves no data.
        rope_storage = (
            triton.reinterpret(k_cache, tl.bfloat16)
            if cache_format == "packed"
            else None
        )
        # Fixed geometry for the in-place writer. Tune offline on private cache state.
        args = (
            kv,
            cache_storage,
            rope_storage,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale if cache_format == "fp8" else None,
            cache_strides[0],
            cache_strides[1],
            cache_block_size,
            cache_format,
        )
        # Avoid constructing a device context when it is already current. Never
        # cache the current device or stream: callers may switch either at any time.
        current_device = runtime.torch_device_fn.current_device()
        if current_device == kv_device.index:
            launch_insert((num_slots,), args, current_device)
        else:
            with runtime.torch_device_fn.device(kv_device):
                launch_insert((num_slots,), args, kv_device.index)

    return insert_context_kv


insert_context_kv = _make_insert_context_kv(_launch_insert)
