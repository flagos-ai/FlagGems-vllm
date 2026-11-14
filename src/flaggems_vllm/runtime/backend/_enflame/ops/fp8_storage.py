# SPDX-License-Identifier: Apache-2.0
"""E4M3FN cache bytes on GCU: native fp8 conversion codecs.

The generic implementation simulates the conversion with integer bit math so
it runs on backends without fp8 conversion instructions. The GCU Triton
toolchain provides native ``float8e4nv`` conversions that are bit-exact for
every finite value, subnormal, signed zero, and rounding boundary, so this
backend-local copy replaces the bit simulation with the native instructions:

- ``encode_e4m3fn``: clamp to +-448 then ``.to(tl.float8e4nv)``. The native
  cast is non-saturating (an unclamped 500.0 encodes as NaN 0x7F), hence the
  clamp; ``tl.clamp`` resolves NaN inputs to a bound instead of propagating
  them, so NaN lanes are reconstructed from the input sign bit explicitly.
- ``decode_e4m3fn``: bitcast the byte to ``float8e4nv`` and convert to FP32.
  This reproduces the full 256-entry E4M3FN table including subnormals, both
  signed zeros, and both NaN encodings. The input must be a 1-byte integer
  type (uint8/int8).

Both directions are verified bit-exact against the CPU PyTorch cast by
``tests/test_fp8_storage.py`` (exhaustive rounding-boundary sweep for
encode, all 256 bytes for decode).
"""

import triton
import triton.language as tl


@triton.jit
def encode_e4m3fn(x):
    """Saturating round-to-nearest-even conversion, preserving signed zero."""
    v = x.to(tl.float32)
    encoded = tl.clamp(v, -448.0, 448.0).to(tl.float8e4nv)
    payload = encoded.to(tl.uint8, bitcast=True)
    bits = v.to(tl.uint32, bitcast=True)
    # NaN lanes: the top byte of an IEEE-754 NaN is 0x7F | sign already
    # (exponent bits 30-24 are all ones), so it *is* the canonical NaN
    # payload -- 0x7F for +NaN, 0xFF for -NaN. No OR with 0x7F is needed
    # (the OR is idempotent on {0x7F, 0xFF} and costs one ALU op per lane).
    nan_payload = bits >> 24
    return tl.where(v != v, nan_payload, payload)


@triton.jit
def decode_e4m3fn(bits):
    """Decode all E4M3FN byte values to FP32, including subnormals and NaN."""
    return bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)


_PUBLIC_CODECS = ("encode_e4m3fn", "decode_e4m3fn")


def _install():
    """Wire the GCU codecs into the direct-import entrypoints.

    Tests and consumers import ``from flaggems_vllm.ops.fp8_storage import
    encode_e4m3fn``, which resolves against the already-imported generic
    module, so the SpecOpRegistrar namespace swap cannot reach them. Replace
    the attributes on that module in place, mirroring the kpool override
    precedent: only a pristine generic function is swapped, so explicit user
    or test patches always win.
    """
    import sys

    mod = sys.modules.get("flaggems_vllm.ops.fp8_storage")
    if mod is None:
        return
    for name in _PUBLIC_CODECS:
        current = getattr(mod, name, None)
        if current is not None and getattr(current, "__module__", None) == (
            "flaggems_vllm.ops.fp8_storage"
        ):
            setattr(mod, name, globals()[name])


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
