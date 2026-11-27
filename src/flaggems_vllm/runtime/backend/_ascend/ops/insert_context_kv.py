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

"""Ascend exact-FMA specialization for DSpark KV cache insertion.

The common host checks, rotation addressing and quantization are shared.
The local encoder uses signed32 operations for this compiler. This kernel
limits software FMA to the 64 RoPE values and uses the synchronization setting
validated by repeated V4.1 cache writes.
"""

import threading
from functools import lru_cache

import triton
import triton.language as tl

from flaggems_vllm.ops.insert_context_kv import (
    _encode_with_scale,
    _make_insert_context_kv,
    _quantize_groups,
    _rotate_kv,
    _store_v4_groups,
)
from flaggems_vllm.utils import libentry

_shared_rotate_kv = _rotate_kv


# Keep the signed32 operations used by this operator's validated Ascend path
# local. Other operators retain the common codec.
@triton.jit
def _encode_e4m3fn_signed32(x):
    """Saturating round-to-nearest-even conversion, preserving signed zero."""
    bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.where(is_nan, 0.0, tl.minimum(tl.abs(x), 448.0))
    magnitude_bits = magnitude.to(tl.int32, bitcast=True)
    # Discard 20 FP32 mantissa bits, round ties to an even E4M3 mantissa,
    # then change the exponent bias from 127 to 7.
    normal = ((magnitude_bits + 0x7FFFF + ((magnitude_bits >> 20) & 1)) >> 20) - 960
    # E4M3FN subnormals are integer multiples of 2**-9.
    scaled = magnitude * 512.0
    floor = tl.floor(scaled)
    remainder = scaled - floor
    mantissa = floor.to(tl.int32)
    subnormal = mantissa + (
        (remainder > 0.5) | ((remainder == 0.5) & ((mantissa & 1) != 0))
    ).to(tl.int32)
    payload = tl.where(magnitude < 0.015625, subnormal, normal)
    payload = tl.where(is_nan, 0x7F, payload)
    return (sign | payload).to(tl.uint8)


@triton.jit
def _ult(a, b):
    return (a ^ -2147483648) < (b ^ -2147483648)


@triton.jit
def _shr(x, n):
    # This compiler lowers uint32 right-shift as arithmetic shift. Mask the
    # signed32 shift explicitly; all counts are clamped to the 0..31 range.

    mask = tl.where(n > 0, 0x7FFFFFFF >> tl.maximum(n - 1, 0), -1)
    return (x >> n) & mask


@triton.jit
def _shift_pair(x, shift):
    x = x.to(tl.int32)
    left = tl.minimum(tl.maximum(shift, 0), 31)
    high = tl.where(
        shift >= 32,
        x << tl.minimum(tl.maximum(shift - 32, 0), 31),
        tl.where(shift > 0, _shr(x, tl.minimum(32 - left, 31)), 0),
    )
    low = tl.where(shift >= 32, 0, x << left)
    right = tl.minimum(tl.maximum(-shift, 0), 31)
    mask = (tl.full((), 1, tl.int32) << right) - 1
    shr = (_shr(x, right)) | ((x & mask) != 0).to(tl.int32)
    shr = tl.where(-shift >= 32, (x != 0).to(tl.int32), shr)
    return tl.where(shift < 0, 0, high), tl.where(shift < 0, shr, low)


@triton.jit
def fma_bf16(a, b, c):
    # a is a finite BF16 value widened to FP32; b is finite FP32. Their exact
    # product has at most 32 significand bits. Two signed32 words hold its
    # aligned sum with c. Right-jam shifts retain information below the final
    # rounding point, including cancellation and subnormal outputs.

    ai = a.to(tl.int32, bitcast=True)
    bi = b.to(tl.int32, bitcast=True)
    ci = c.to(tl.int32, bitcast=True)
    ae = (ai >> 23) & 255
    be = (bi >> 23) & 255
    ce = (ci >> 23) & 255
    am = ((ai & 0x7FFFFF) >> 16) | tl.where(ae > 0, 128, 0)
    bm = (bi & 0x7FFFFF) | tl.where(be > 0, 0x800000, 0)
    cm = (ci & 0x7FFFFF) | tl.where(ce > 0, 0x800000, 0)
    ep = tl.maximum(ae, 1) + tl.maximum(be, 1) - 254
    ec = tl.maximum(ce, 1) - 127
    pm = am.to(tl.int32) * bm.to(tl.int32)
    largest = tl.maximum(tl.where(pm != 0, ep + 1, -1024), tl.where(cm != 0, ec, -1024))
    ph, pl = _shift_pair(pm, 30 + ep - largest)
    qh, ql = _shift_pair(cm, 37 + ec - largest)
    psign = ((ai ^ bi) & -2147483648) != 0
    csign = (ci & -2147483648) != 0
    # High halves fit signed31, low halves use explicit unsigned comparisons.
    pbig = (ph > qh) | ((ph == qh) & ~_ult(pl, ql))
    big_h = tl.where(pbig, ph, qh)
    small_h = tl.where(pbig, qh, ph)
    big_l = tl.where(pbig, pl, ql)
    small_l = tl.where(pbig, ql, pl)
    diff_l = big_l - small_l
    diff_h = big_h - small_h - _ult(big_l, small_l).to(tl.int32)
    sum_l = pl + ql
    sum_h = ph + qh + _ult(sum_l, pl).to(tl.int32)
    same = psign == csign
    hi = tl.where(same, sum_h, diff_h)
    lo = tl.where(same, sum_l, diff_l)
    negative = tl.where(same, psign, tl.where(pbig, psign, csign))
    scan = tl.where(hi != 0, hi, lo)
    top = tl.where(hi != 0, 32, 0)
    for step in tl.static_range(4, -1, -1):
        count = 1 << step
        take = ~_ult(scan, tl.full((), 1 << count, tl.int32))
        top += tl.where(take, count, 0)
        scan = tl.where(take, _shr(scan, count), scan)
    exponent = largest - 60 + top
    sub = exponent < -126
    shift = tl.where(sub, -89 - largest, top - 23)
    right = tl.minimum(tl.maximum(shift, 0), 31)
    over32 = tl.minimum(tl.maximum(shift - 32, 0), 31)
    trunc = tl.where(
        shift >= 32,
        _shr(hi, over32),
        (_shr(lo, right)) | tl.where(shift > 0, hi << tl.minimum(32 - right, 31), 0),
    )
    trunc = tl.where(shift <= 0, lo << tl.minimum(tl.maximum(-shift, 0), 31), trunc)
    guard_low_shift = tl.minimum(tl.maximum(shift - 1, 0), 31)
    guard_high_shift = tl.minimum(tl.maximum(shift - 33, 0), 31)
    guard = tl.where(
        shift <= 32, (_shr(lo, guard_low_shift)) & 1, (_shr(hi, guard_high_shift)) & 1
    )
    mask_l = (tl.full((), 1, tl.int32) << guard_low_shift) - 1
    mask_h = (tl.full((), 1, tl.int32) << guard_high_shift) - 1
    sticky = tl.where(shift <= 32, (lo & mask_l) != 0, ((hi & mask_h) != 0) | (lo != 0))
    up = (shift > 0) & (shift < 64) & (guard != 0) & (sticky | ((trunc & 1) != 0))
    rounded = tl.where(shift >= 64, 0, trunc + up.to(tl.int32))
    carry = (rounded & 0x1000000) != 0
    normal = ((exponent + 127 + carry.to(tl.int32)) << 23).to(tl.int32) | (
        tl.where(carry, rounded >> 1, rounded) & 0x7FFFFF
    )
    payload = tl.where(sub, rounded, normal)
    payload = tl.where(exponent + carry.to(tl.int32) > 127, 0x7F800000, payload)
    iszero = (hi == 0) & (lo == 0)
    negative = tl.where(iszero, (pm == 0) & (cm == 0) & psign & csign, negative)
    payload = tl.where(iszero, 0, payload)
    answer = (payload | tl.where(negative, -2147483648, 0).to(tl.int32)).to(
        tl.float32, bitcast=True
    )
    return tl.where(ce == 255, c, answer)


@triton.jit
def _rotate_kv(kv, trig, offsets):
    return _shared_rotate_kv(kv, trig, offsets, fma_bf16)


@triton.jit
def _store_v41_prefix(kv, data, scales, start: tl.constexpr, GROUPS: tl.constexpr):
    groups = start + tl.arange(0, GROUPS)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    values = tl.load(kv + offsets).to(tl.float32)
    quantized, code = _quantize_groups(values, _encode_e4m3fn_signed32)
    tl.store(data + offsets, quantized)
    tl.store(scales + groups, code)


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

    position = tl.load(positions + token)
    page = cache + (slot // BLOCK_SIZE) * page_stride
    row = slot % BLOCK_SIZE
    trig = cos_sin_cache + position * 64

    source = kv + token * 512
    if FORMAT == "bf16" or FORMAT == "fp8":
        # Software FMA only operates on the 64 RoPE values. Prefix stores are
        # disjoint and do not create 448 unnecessary exact-FMA lanes.
        prefix = tl.arange(0, 512)
        prefix_mask = prefix < 448
        plain = tl.load(source + prefix, mask=prefix_mask, other=0).to(tl.float32)
        tail = tl.arange(0, 64)
        rotated = _shared_rotate_kv(source, trig, 448 + tail, fma_bf16)
        target = page + row * row_stride
        if FORMAT == "bf16":
            tl.store(target + prefix, plain.to(tl.bfloat16), mask=prefix_mask)
            tl.store(target + 448 + tail, rotated.to(tl.bfloat16))
        else:
            scale = tl.load(fp8_scale)
            tl.store(
                target + prefix,
                _encode_with_scale(plain, scale, _encode_e4m3fn_signed32),
                mask=prefix_mask,
            )
            tl.store(
                target + 448 + tail,
                _encode_with_scale(rotated, scale, _encode_e4m3fn_signed32),
            )
    elif FORMAT == "mxfp8":
        data = page + row * 528
        scales = data + 512
        _store_v41_prefix(source, data, scales, 0, 8)
        _store_v41_prefix(source, data, scales, 8, 4)
        _store_v41_prefix(source, data, scales, 12, 2)
        tail = tl.arange(0, 64)
        rotated = _shared_rotate_kv(source, trig, 448 + tail, fma_bf16)
        rounded = rotated.to(tl.bfloat16).to(tl.float32)
        bits = rounded.to(tl.int32, bitcast=True) & 0x7FFFFFFF
        sign = rotated.to(tl.int32, bitcast=True) & -2147483648
        groups = (bits | sign).to(tl.float32, bitcast=True).reshape((2, 32))
        quantized, code = _quantize_groups(groups, _encode_e4m3fn_signed32)
        tl.store(data + 448 + tail, quantized.reshape((64,)))
        tl.store(scales + 14 + tl.arange(0, 2), code)
    else:
        data = page + row * 576
        scales = page + BLOCK_SIZE * 576 + row * 8
        # Use disjoint stores for the seven FP8 groups and the BF16 tail.
        _store_v4_groups(source, data, scales, 0, 4, _encode_e4m3fn_signed32)
        _store_v4_groups(source, data, scales, 4, 2, _encode_e4m3fn_signed32)
        _store_v4_groups(source, data, scales, 6, 1, _encode_e4m3fn_signed32)
        tl.store(scales + 7, 0)
        tail = tl.arange(0, 64)
        rotated = _rotate_kv(source, trig, 448 + tail)
        rope = rope_cache
        rope += (slot // BLOCK_SIZE) * (page_stride // 2) + row * 288
        tl.store(rope + 224 + tail, rotated.to(tl.bfloat16))


@lru_cache(maxsize=128)
def _get_simd_compile_mode(device_index):
    # Use the compiler's own default mode rather than a build/version allowlist.
    # Mode names differ between compiler releases; probing arbitrary names can
    # silently succeed on compilers that retain unrecognized option strings.
    try:
        from triton.compiler.compiler import make_backend

        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "npu":
            raise ValueError("expected an NPU compiler backend")
        backend = make_backend(target)
        defaults = backend.parse_options({})
        compile_mode = getattr(defaults, "compile_mode", None)
        if (
            not isinstance(compile_mode, str)
            or not compile_mode
            or getattr(defaults, "parallel_mode", None) != "simd"
        ):
            raise ValueError("compiler default mode must use SIMD")
        for barrier_all in (True, False):
            options = backend.parse_options(
                {
                    "inject_barrier_all": barrier_all,
                    "enable_fp_fusion": False,
                    "compile_mode": compile_mode,
                }
            )
            if (
                getattr(options, "inject_barrier_all", None) is not barrier_all
                or getattr(options, "enable_fp_fusion", None) is not False
                or getattr(options, "compile_mode", None) != compile_mode
                or getattr(options, "parallel_mode", None) != "simd"
            ):
                raise ValueError("compiler does not retain SIMD/barrier/fusion options")
        return compile_mode
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise NotImplementedError(
            f"insert_context_kv requires an NPU SIMD compiler with "
            f"inject_barrier_all support: {error}"
        ) from error


_kernel_entry = libentry()(_insert_context_kv_kernel)
_compiled_kernels = {}
_compiled_kernel_lock = threading.Lock()
_triton_version = tuple(int(part) for part in triton.__version__.split(".")[:2])


def _launch_insert(grid, args, device_index):
    compile_mode = _get_simd_compile_mode(device_index)
    # Ascend910_9392 + CANN9.0.0 requires compiler-inserted full barriers here:
    # dense V4.1 records otherwise intermittently corrupt their last bytes.
    # Explicit tl.debug_barrier, disabling multibuffer or sub-block binding,
    # and padding the launch grid did not fix the 120-replay reproducer.

    # Match LibEntry's supported full-argument CompiledKernel ABI. Older
    # runtimes retain the ordinary JIT launch path.
    if not (3, 3) <= _triton_version <= (3, 9):
        _insert_context_kv_kernel[grid](
            *args,
            enable_fp_fusion=False,
            inject_barrier_all=True,
            compile_mode=compile_mode,
        )
        return
    # The JIT explicitly does not specialize pointers or stride values. Every
    # pointer dtype/None combination follows from the validated cache format;
    # strides only distinguish their signed integer ABI width. This key cannot
    # reuse an alignment-specialized kernel for an offset pointer.
    key = (
        device_index,
        args[10],
        args[9],
        args[7] >= 2**31,
        args[8] >= 2**31,
        True,  # inject_barrier_all is part of the compiled identity.
        compile_mode,
    )
    with _compiled_kernel_lock:
        kernel = _compiled_kernels.get(key)
    if kernel is None:
        kernel, _ = _kernel_entry.run(
            *args,
            grid=grid,
            warmup=False,
            enable_fp_fusion=False,
            inject_barrier_all=True,
            compile_mode=compile_mode,
        )
        with _compiled_kernel_lock:
            if len(_compiled_kernels) >= 128:
                _compiled_kernels.pop(next(iter(_compiled_kernels)))
            _compiled_kernels[key] = kernel
        return
    # CompiledKernel resolves the current stream on every call and receives all
    # current arguments; changed slots, positions or scales cannot be cached.
    kernel[(grid[0], 1, 1)](*args)


insert_context_kv = _make_insert_context_kv(_launch_insert)
insert_context_kv.__module__ = __name__
