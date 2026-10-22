# SPDX-License-Identifier: Apache-2.0
"""E4M3FN cache bytes without requiring native FP8 conversion instructions."""

import triton
import triton.language as tl


@triton.jit
def encode_e4m3fn(x):
    """Saturating round-to-nearest-even conversion, preserving signed zero."""
    bits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    magnitude = tl.where(is_nan, 0.0, tl.minimum(tl.abs(x), 448.0))
    magnitude_bits = magnitude.to(tl.uint32, bitcast=True)
    # Discard 20 FP32 mantissa bits, round ties to an even E4M3 mantissa,
    # then change the exponent bias from 127 to 7.
    normal = ((magnitude_bits + 0x7FFFF + ((magnitude_bits >> 20) & 1)) >> 20) - 960
    # E4M3FN subnormals are integer multiples of 2**-9.
    scaled = magnitude * 512.0
    floor = tl.floor(scaled)
    remainder = scaled - floor
    subnormal = floor.to(tl.uint32) + (
        (remainder > 0.5) | ((remainder == 0.5) & ((floor.to(tl.uint32) & 1) != 0))
    ).to(tl.uint32)
    payload = tl.where(magnitude < 0.015625, subnormal, normal)
    payload = tl.where(is_nan, 0x7F, payload)
    return (sign | payload).to(tl.uint8)


@triton.jit
def decode_e4m3fn(bits):
    """Decode all E4M3FN byte values to FP32, including subnormals and NaN."""
    bits = bits.to(tl.uint32)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 7
    normal_bits = ((exponent + 120) << 23) | (mantissa << 20)
    value = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * 0.001953125,
        normal_bits.to(tl.float32, bitcast=True),
    )
    value = tl.where((bits & 0x7F) == 0x7F, float("nan"), value)
    signed_bits = value.to(tl.uint32, bitcast=True) | ((bits & 0x80) << 24)
    return signed_bits.to(tl.float32, bitcast=True)
