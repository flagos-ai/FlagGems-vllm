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

import hashlib
import math
import os
import tempfile
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from triton.language import core as tl_core

from flaggems_vllm import runtime
from flaggems_vllm.utils import has_triton_tle_attrs, libentry, libtuner
from flaggems_vllm.utils.code_cache import code_cache_dir

if TYPE_CHECKING:
    from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
        FlashMLAFp8SplitKSchedMeta,
    )

HAS_TLE = has_triton_tle_attrs(
    ("gpu.wgmma", "gpu.warp_specialize", "gpu.copy"), 3, 6, 0
)
if HAS_TLE:
    import triton.experimental.tle.language as tle


@tl_core.builtin
def copy_with_arrival_count(
    src, dst, shape, offsets=None, barrier=None, _semantic=None
):
    from triton.experimental.tle.language.gpu.core import _barrier_slot

    slot = _barrier_slot(barrier, _semantic)
    assert slot.arrive_count == 3
    shape = tl_core._unwrap_if_constexpr(shape)
    if isinstance(shape, tl_core.tuple):
        shape = shape.values
    byte_count = dst.dtype.primitive_bitwidth // 8
    for dim in shape:
        byte_count *= int(tl_core._unwrap_if_constexpr(dim))
    # Three copies share one phase; each contributes its own byte budget and arrival.
    member = tle.gpu.barrier(
        slot.handle,
        slot.num_barriers,
        slot.arrive_count,
        slot.init,
        byte_count,
        slot.layout,
        _semantic,
        shape=slot.shape,
        static_index=slot.static_index,
        named_base_id=slot.named_base_id,
        allocation_key=slot.allocation_key,
    )
    tle.gpu.copy(src, dst, shape, offsets=offsets, barrier=member, _semantic=_semantic)


@tl_core.extern
def row_max8(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_max8", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def row_sum8(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_sum8", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def exp2_two(a, b, _semantic=None):
    """exp2 of two values in one instruction; the pair returns packed in a uint64."""
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b],
        {(tl.float32, tl.float32): ("mla_fp8_exp2_two", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.builtin
def inline_jit(fn, args, _semantic=None, _generator=None):
    """Use the same frontend inlining as WS for nested, role-local helpers.

    Normal nested JIT calls survive the stock inliner in isolated WS regions
    and lose the caller's explicit layout. This stays within the operator.
    """
    fn = tl_core._unwrap_if_constexpr(fn)
    args = tl_core._unwrap_if_constexpr(args)
    if isinstance(args, tl_core.tuple):
        args = tuple(args.values)
    return _generator.inline_JitFunction(fn, args, kwargs={})


@triton.jit
def exp2_pairs(x):
    """exp2 over the score tile, two elements per hardware instruction.

    Pairs must be thread-local. SQMMA puts the low three token bits across
    lanes, so neighbours along the last axis live in different threads; the
    same permute the row reduction uses brings the register-resident bits to
    the end, and the pairs are taken there.
    """
    x = tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)

    p04 = exp2_two(a0, a4)
    p26 = exp2_two(a2, a6)
    p15 = exp2_two(a1, a5)
    p37 = exp2_two(a3, a7)
    b0 = (p04 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b4 = (p04 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b2 = (p26 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b6 = (p26 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b1 = (p15 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b5 = (p15 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b3 = (p37 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b7 = (p37 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    y = tl.join(
        tl.join(tl.join(b0, b4), tl.join(b2, b6)),
        tl.join(tl.join(b1, b5), tl.join(b3, b7)),
    )
    return tl.reshape(tl.permute(y, (0, 2, 3, 4, 1)), (x.shape[0], 64))


@triton.jit
def row_partials(x, MAXIMUM: tl.constexpr):
    # SQMMA distributes low three K bits across lanes; high bits are registers.
    x = tl.permute(tl.reshape(x, (x.shape[0], 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)
    if MAXIMUM:
        partial = row_max8(a0, a1, a2, a3, a4, a5, a6, a7)
    else:
        partial = row_sum8(a0, a1, a2, a3, a4, a5, a6, a7)
    return partial


@triton.jit
def row_reduce(x, MAXIMUM: tl.constexpr):
    if MAXIMUM:
        partial = inline_jit(row_partials, (x, True))
        result = tl.max(partial, 1)
    else:
        partial = inline_jit(row_partials, (x, False))
        result = tl.sum(partial, 1)
    return result


@triton.jit
def sum_max_combine(sum0, max0, sum1, max1):
    return sum0 + sum1, tl.maximum(max0, max1)


@triton.jit
def row_sum_and_max(probabilities, weights):
    # Both reductions are independent after exponentiation; interleave shuffles.
    partial_sum = inline_jit(row_partials, (probabilities, False))
    partial_max = inline_jit(row_partials, (weights, True))
    return tl.reduce((partial_sum, partial_max), 1, sum_max_combine)


@tl_core.extern
def native_recip(value, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [value],
        {(tl.float32,): ("mla_fp8_recip", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def native_lma_wait(token, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [token],
        {(tl.int32,): ("mla_fp8_lma_complete", tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )


NATIVE_PRIMITIVES = r"""
target triple = "musa"

declare void @llvm.musa.lma.wait()
define i32 @mla_fp8_lma_complete(i32 %token) alwaysinline nounwind {
  call void @llvm.musa.lma.wait()
  ret i32 %token
}

declare <4 x float> @llvm.musa.max.f.bst4.src02(<4 x float>, <4 x float>)
declare <2 x float> @llvm.musa.max.f.bst2.src02(<2 x float>, <2 x float>)
declare <2 x float> @llvm.musa.add.f.bst2.src01(<2 x float>, <2 x float>)
declare float @llvm.maxnum.f32(float, float)
define float @mla_fp8_max8(float %a0, float %a1, float %a2, float %a3,
                          float %a4, float %a5, float %a6, float %a7)
    alwaysinline nounwind readnone {
  %l0 = insertelement <4 x float> poison, float %a0, i32 0
  %l1 = insertelement <4 x float> %l0, float %a1, i32 1
  %l2 = insertelement <4 x float> %l1, float %a2, i32 2
  %l3 = insertelement <4 x float> %l2, float %a3, i32 3
  %r0 = insertelement <4 x float> poison, float %a4, i32 0
  %r1 = insertelement <4 x float> %r0, float %a5, i32 1
  %r2 = insertelement <4 x float> %r1, float %a6, i32 2
  %r3 = insertelement <4 x float> %r2, float %a7, i32 3
  %v = call <4 x float> @llvm.musa.max.f.bst4.src02(<4 x float> %l3, <4 x float> %r3)
  %lo = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hi = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %p = call <2 x float> @llvm.musa.max.f.bst2.src02(<2 x float> %lo, <2 x float> %hi)
  %x = extractelement <2 x float> %p, i32 0
  %y = extractelement <2 x float> %p, i32 1
  %z = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %z
}
define float @mla_fp8_sum8(float %a0, float %a1, float %a2, float %a3,
                          float %a4, float %a5, float %a6, float %a7)
    alwaysinline nounwind readnone {
  %l0 = insertelement <4 x float> poison, float %a0, i32 0
  %l1 = insertelement <4 x float> %l0, float %a1, i32 1
  %l2 = insertelement <4 x float> %l1, float %a2, i32 2
  %l3 = insertelement <4 x float> %l2, float %a3, i32 3
  %r0 = insertelement <4 x float> poison, float %a4, i32 0
  %r1 = insertelement <4 x float> %r0, float %a5, i32 1
  %r2 = insertelement <4 x float> %r1, float %a6, i32 2
  %r3 = insertelement <4 x float> %r2, float %a7, i32 3
  %v = fadd <4 x float> %l3, %r3
  %lo = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %hi = shufflevector <4 x float> %v, <4 x float> poison, <2 x i32> <i32 2, i32 3>
  %p = call <2 x float> @llvm.musa.add.f.bst2.src01(<2 x float> %lo, <2 x float> %hi)
  %x = extractelement <2 x float> %p, i32 0
  %y = extractelement <2 x float> %p, i32 1
  %z = fadd float %x, %y
  ret float %z
}


declare <2 x float> @llvm.musa.exp2.f.bst2(<2 x float>)
define i64 @mla_fp8_exp2_two(float %a, float %b) alwaysinline nounwind readnone {
  %v0 = insertelement <2 x float> poison, float %a, i32 0
  %v1 = insertelement <2 x float> %v0, float %b, i32 1
  %r = call <2 x float> @llvm.musa.exp2.f.bst2(<2 x float> %v1)
  %i = bitcast <2 x float> %r to i64
  ret i64 %i
}

declare float @llvm.musa.rcp(float)
define float @mla_fp8_recip(float %x) alwaysinline nounwind readnone {
  %r = call float @llvm.musa.rcp(float %x)
  ret float %r
}


declare <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float>,<4 x float>)
define i64 @mla_fp8_mul4_lo(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 0,i32 1>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
define i64 @mla_fp8_mul4_hi(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.mul.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 2,i32 3>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
declare <4 x float> @llvm.musa.add.f.bst4.src01(<4 x float>,<4 x float>)
define i64 @mla_fp8_add4_lo(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.add.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 0,i32 1>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
define i64 @mla_fp8_add4_hi(float %a,float %b,float %c,float %d,
    float %e,float %f,float %g,float %h) alwaysinline nounwind readnone {
 %l0 = insertelement <4 x float> poison,float %a,i32 0
 %l1 = insertelement <4 x float> %l0,float %b,i32 1
 %l2 = insertelement <4 x float> %l1,float %c,i32 2
 %l3 = insertelement <4 x float> %l2,float %d,i32 3
 %r0 = insertelement <4 x float> poison,float %e,i32 0
 %r1 = insertelement <4 x float> %r0,float %f,i32 1
 %r2 = insertelement <4 x float> %r1,float %g,i32 2
 %r3 = insertelement <4 x float> %r2,float %h,i32 3
 %r = call <4 x float> @llvm.musa.add.f.bst4.src01(<4 x float> %l3,<4 x float> %r3)
 %p = shufflevector <4 x float> %r,<4 x float> poison,<2 x i32> <i32 2,i32 3>
 %bits = bitcast <2 x float> %p to i64
 ret i64 %bits
}
"""


@lru_cache(maxsize=1)
def primitives_lib():
    digest = hashlib.sha256(NATIVE_PRIMITIVES.encode()).hexdigest()[:16]
    path = code_cache_dir() / ("musa-mla-fp8-" + digest + ".ll")
    libraries = (("musa_mla_fp8", str(path)),)
    if path.exists():
        return libraries
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as source:
        source.write(NATIVE_PRIMITIVES)
    os.replace(source.name, path)
    return libraries


@lru_cache(maxsize=1)
def require_tle_pipeline():
    from triton._C.libtriton import mthreads

    try:
        supported = mthreads.supports_tle_sqmma_row_reduction_m_split()
    except AttributeError as error:
        raise NotImplementedError(
            "FP8 MLA requires FlagTree MUSA row-reduction SQMMA selection"
        ) from error
    if not supported:
        raise NotImplementedError(
            "FP8 MLA requires FlagTree MUSA row-reduction SQMMA selection"
        )


@tl_core.extern
def mul4_lo(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_mul4_lo", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def mul4_hi(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_mul4_hi", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def add4_lo(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_add4_lo", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl_core.extern
def add4_hi(a, b, c, d, e, f, g, h, _semantic=None):
    return tl_core.extern_elementwise(
        "",
        "",
        [a, b, c, d, e, f, g, h],
        {(tl.float32,) * 8: ("mla_fp8_add4_hi", tl.uint64)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def pointwise_vec4(x, y, MULTIPLY: tl.constexpr):
    y = tl.broadcast_to(y, x.shape)
    BH: tl.constexpr = x.shape[0]
    x = tl.permute(tl.reshape(x, (BH, 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    y = tl.permute(tl.reshape(y, (BH, 2, 2, 2, 8)), (0, 4, 1, 2, 3))
    x0, x1 = tl.split(x)
    x00, x01 = tl.split(x0)
    x10, x11 = tl.split(x1)
    a0, a4 = tl.split(x00)
    a2, a6 = tl.split(x01)
    a1, a5 = tl.split(x10)
    a3, a7 = tl.split(x11)
    y0, y1 = tl.split(y)
    y00, y01 = tl.split(y0)
    y10, y11 = tl.split(y1)
    c0, c4 = tl.split(y00)
    c2, c6 = tl.split(y01)
    c1, c5 = tl.split(y10)
    c3, c7 = tl.split(y11)
    if MULTIPLY:
        p04 = mul4_lo(a0, a4, a2, a6, c0, c4, c2, c6)
        p26 = mul4_hi(a0, a4, a2, a6, c0, c4, c2, c6)
        p15 = mul4_lo(a1, a5, a3, a7, c1, c5, c3, c7)
        p37 = mul4_hi(a1, a5, a3, a7, c1, c5, c3, c7)
    else:
        p04 = add4_lo(a0, a4, a2, a6, c0, c4, c2, c6)
        p26 = add4_hi(a0, a4, a2, a6, c0, c4, c2, c6)
        p15 = add4_lo(a1, a5, a3, a7, c1, c5, c3, c7)
        p37 = add4_hi(a1, a5, a3, a7, c1, c5, c3, c7)
    b0 = (p04 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b4 = (p04 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b2 = (p26 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b6 = (p26 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b1 = (p15 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b5 = (p15 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    b3 = (p37 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
    b7 = (p37 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    z = tl.join(
        tl.join(tl.join(b0, b4), tl.join(b2, b6)),
        tl.join(tl.join(b1, b5), tl.join(b3, b7)),
    )
    return tl.reshape(tl.permute(z, (0, 2, 3, 4, 1)), (BH, 64))


def prune_dense_tle_configs(configs, named_args, **kwargs):
    launch_args = {**named_args, **kwargs}
    head_rows = launch_args["H"] * launch_args["QLEN"]
    requested_splits = launch_args["REQUESTED_SPLITS"]
    return [
        config
        for config in configs
        if config.kwargs["BH"] <= head_rows
        and (requested_splits == 0 or config.kwargs["SPLITS"] == requested_splits)
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_dense_fp8_mla"),
    key=["BATCH", "HEADS", "QUERIES", "CAPACITY", "SPLITS"],
)
@triton.jit
def dense_fp8_mla_kernel(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    QUERIES: tl.constexpr,
    CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_tile * BH + tl.arange(0, BH)
    query = heads // HEADS
    head = heads % HEADS
    valid_head = heads < HEADS * QUERIES
    dims = tl.arange(0, 512)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, 64)
    q = tl.load(
        Q
        + batch * Q_STRIDES[0]
        + query[:, None] * Q_STRIDES[1]
        + head[:, None] * Q_STRIDES[2]
        + dims[None, :],
        valid_head[:, None],
        0.0,
    )
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + query[:, None] * QR_STRIDES[1]
        + head[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        valid_head[:, None],
        0.0,
    )
    qs = tl.load(
        QS + batch * QS_STRIDES[0] + query * QS_STRIDES[1] + head * QS_STRIDES[2],
        valid_head,
        0.0,
    )
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    pages = tl.cdiv(length, 64)
    first_page = pages * split // SPLITS
    end_page = pages * (split + 1) // SPLITS
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    accumulator = tl.zeros((BH, 512), tl.float32)
    for page_idx in range(first_page, end_page):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid_token = page_idx * 64 + tokens < length
        kv = tl.load(
            KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1] + dims[None, :],
            valid_token[:, None],
            0.0,
        )
        kr = tl.load(
            KR
            + page * KR_STRIDES[0]
            + tokens[:, None] * KR_STRIDES[1]
            + rope_dims[None, :],
            valid_token[:, None],
            0.0,
        )
        ks = tl.load(
            KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1], valid_token, 0.0
        )
        content_scores = tl.dot(q, tl.trans(kv), out_dtype=tl.float32).to(tl.float32)
        tl.debug_barrier()
        rope_scores = tl.dot(qr, tl.trans(kr), out_dtype=tl.float32)
        tl.debug_barrier()
        scores = (
            (content_scores + rope_scores)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        valid = valid_head[:, None] & valid_token[None, :]
        if CAUSAL:
            valid = valid & (
                page_idx * 64 + tokens[None, :] < length - QUERIES + query[:, None] + 1
            )
        scores = tl.where(valid, scores, -float("inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        probabilities = tl.exp2(scores - safe_maximum[:, None])
        rescale = tl.exp2(maximum - safe_maximum)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        scaled_probabilities = probabilities * ks[None, :]
        probability_scale = tl.maximum(tl.max(scaled_probabilities, 1) / 448.0, 1.0e-30)
        scaled = scaled_probabilities / probability_scale[:, None]
        quantized = scaled.to(tl.float8e4nv)
        contribution = tl.dot(quantized, kv, out_dtype=tl.float32).to(tl.float32)
        tl.debug_barrier()
        accumulator = (
            accumulator * rescale[:, None] + contribution * probability_scale[:, None]
        )
        maximum = new_maximum
    if SPLITS == 1:
        normalized = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
        tl.store(
            Output + (batch * HEADS * QUERIES + heads[:, None]) * 512 + dims[None, :],
            normalized,
            valid_head[:, None],
        )
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            -float("inf"),
        )
        tl.store(LSE + (batch * HEADS + head) * QUERIES + query, logsum, valid_head)
    else:
        tl.store(
            Partial
            + ((batch * SPLITS + split) * HEADS * QUERIES + heads[:, None]) * 512
            + dims[None, :],
            accumulator,
            valid_head[:, None],
        )
        tl.store(
            Stats + ((batch * SPLITS + split) * 2 * HEADS * QUERIES + heads),
            maximum,
            valid_head,
        )
        tl.store(
            Stats
            + (
                (batch * SPLITS + split) * 2 * HEADS * QUERIES + HEADS * QUERIES + heads
            ),
            denominator,
            valid_head,
        )


@libentry()
@triton.jit
def dense_fp8_mla_combine_kernel(
    Partial,
    Stats,
    Output,
    LSE,
    HEADS: tl.constexpr,
    QUERIES: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    splits = tl.arange(0, SPLITS)
    dims = block * 128 + tl.arange(0, 128)
    maximum = tl.load(Stats + (batch * SPLITS + splits) * 2 * HEADS * QUERIES + head)
    denominator = tl.load(
        Stats + (batch * SPLITS + splits) * 2 * HEADS * QUERIES + HEADS * QUERIES + head
    )
    merged_maximum = tl.max(maximum, 0)
    safe_maximum = tl.where(merged_maximum == -float("inf"), 0.0, merged_maximum)
    weights = tl.exp2(maximum - safe_maximum)
    total = tl.sum(denominator * weights, 0)
    partial = tl.load(
        Partial
        + ((batch * SPLITS + splits[:, None]) * HEADS * QUERIES + head) * 512
        + dims[None, :]
    )
    output = tl.sum(partial * weights[:, None], 0) / tl.where(total > 0, total, 1.0)
    tl.store(Output + (batch * HEADS * QUERIES + head) * 512 + dims, output)
    if block == 0:
        logsum = tl.where(
            total > 0, safe_maximum * 0.6931471805599453 + tl.log(total), -float("inf")
        )
        tl.store(LSE + (batch * HEADS + head % HEADS) * QUERIES + head // HEADS, logsum)


if HAS_TLE:

    @triton.jit
    def dense_fp8_mla_score_producer(
        Q,
        QR,
        QS,
        KS,
        Table,
        Lengths,
        LSE,
        Partial,
        Stats,
        KD,
        RD,
        SK0,
        SK1,
        SR,
        SP,
        SA,
        SD,
        SMax,
        SDenom,
        End,
        Full,
        Empty,
        Sync,
        KB0,
        QB: tl.constexpr,
        QQ: tl.constexpr,
        QH: tl.constexpr,
        QRB: tl.constexpr,
        QRQ: tl.constexpr,
        QRH: tl.constexpr,
        QSB: tl.constexpr,
        QSQ: tl.constexpr,
        QSH: tl.constexpr,
        KSB: tl.constexpr,
        KSTOKEN: tl.constexpr,
        TS: tl.constexpr,
        LS: tl.constexpr,
        H: tl.constexpr,
        QLEN: tl.constexpr,
        BH: tl.constexpr,
        SCALE: tl.constexpr,
        CAUSAL: tl.constexpr,
        PW: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        group, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        heads = group * BH + tl.arange(0, BH)
        valid_head = tl.full((BH,), True, tl.int1)
        queries = heads // H
        head = heads % H
        dims = tl.arange(0, 256)
        rope_dims = tl.arange(0, 64)
        tokens = tl.arange(0, 64)
        q = tl.load(
            Q + batch * QB + queries[:, None] * QQ + head[:, None] * QH + dims[None, :],
            valid_head[:, None],
            0.0,
        )
        q1 = tl.load(
            Q
            + batch * QB
            + queries[:, None] * QQ
            + head[:, None] * QH
            + dims[None, :]
            + 256,
            valid_head[:, None],
            0.0,
        )
        qr = tl.load(
            QR
            + batch * QRB
            + queries[:, None] * QRQ
            + head[:, None] * QRH
            + rope_dims[None, :],
            valid_head[:, None],
            0.0,
        )
        qs = tl.load(QS + batch * QSB + queries * QSQ + head * QSH, valid_head, 0.0)
        length = tl.load(Lengths + batch * LS)
        first_page = tl.cdiv(length, 64) * split // SPLITS
        end_page = tl.cdiv(length, 64) * (split + 1) // SPLITS
        maximum = tl.full((BH,), -float("inf"), tl.float32)
        denominator = tl.zeros((BH,), tl.float32)
        running_scale = tl.zeros((BH,), tl.float32)
        sq0 = tle.gpu.alloc(
            (BH, 256), tl.float8e4nv, init_value=q, nv_mma_shared_layout=True
        )
        sq1 = tle.gpu.alloc(
            (BH, 256), tl.float8e4nv, init_value=q1, nv_mma_shared_layout=True
        )
        sqr = tle.gpu.alloc(
            (BH, 64), tl.bfloat16, init_value=qr, nv_mma_shared_layout=True
        )
        native_lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(Sync, phaseIdx=0)
        tle.gpu.barrier_wait(Sync, phaseIdx=0)
        if first_page < end_page:
            page = tl.load(Table + batch * TS + first_page).to(tl.int64)
            slot = 0
            phase = 0
            tle.gpu.barrier_wait(Empty[slot], phaseIdx=phase)
            ready = KB0[slot]
            copy_with_arrival_count(
                KD,
                SK0.slot(slot),
                (64, 256),
                offsets=(page.to(tl.int32) * 64, 0),
                barrier=ready,
            )
            copy_with_arrival_count(
                KD,
                SK1.slot(slot),
                (64, 256),
                offsets=(page.to(tl.int32) * 64, 256),
                barrier=ready,
            )
            copy_with_arrival_count(
                RD,
                SR.slot(0),
                (64, 64),
                offsets=(page.to(tl.int32) * 64, 0),
                barrier=ready,
            )
        for page_idx in range(first_page, end_page):
            slot = (page_idx - first_page) & 1
            phase = ((page_idx - first_page) // 2) & 1
            page = tl.load(Table + batch * TS + page_idx).to(tl.int64)
            valid_token = page_idx * 64 + tokens < length
            ks = tl.load(KS + page * KSB + tokens * KSTOKEN, valid_token, 0.0)
            tle.gpu.barrier_wait(KB0[slot], phaseIdx=phase)
            content = tle.gpu.wgmma(sq0, SK0.slot(slot), trans_b=True)
            content = tle.gpu.wgmma(sq1, SK1.slot(slot), acc=content, trans_b=True)
            content = tle.gpu.wgmma(sqr, SR.slot(0), acc=content, trans_b=True)
            content = tle.gpu.wgmma_wait(0, content)
            if page_idx > first_page:
                # Finish the prior PV page before publishing or overwriting shared data.
                native_lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(End, phaseIdx=(page_idx - first_page - 1) & 1)
                tle.gpu.barrier_wait(End, phaseIdx=(page_idx - first_page - 1) & 1)
            next_slot = (page_idx - first_page + 1) & 1
            next_phase = ((page_idx - first_page + 1) // 2) & 1
            tle.gpu.barrier_wait(Empty[next_slot], phaseIdx=next_phase)
            if page_idx + 1 < end_page:
                if PW > 4:
                    # Both squads must finish reading the single RoPE buffer.
                    native_lma_wait(tl.full((), 0, tl.int32))
                    tle.gpu.barrier_arrive(
                        Sync, phaseIdx=(page_idx - first_page + 1) & 1
                    )
                    tle.gpu.barrier_wait(Sync, phaseIdx=(page_idx - first_page + 1) & 1)
                next_page = tl.load(Table + batch * TS + page_idx + 1).to(tl.int64)
                next_ready = KB0[next_slot]
                copy_with_arrival_count(
                    KD,
                    SK0.slot(next_slot),
                    (64, 256),
                    offsets=(next_page.to(tl.int32) * 64, 0),
                    barrier=next_ready,
                )
                copy_with_arrival_count(
                    KD,
                    SK1.slot(next_slot),
                    (64, 256),
                    offsets=(next_page.to(tl.int32) * 64, 256),
                    barrier=next_ready,
                )
                copy_with_arrival_count(
                    RD,
                    SR.slot(0),
                    (64, 64),
                    offsets=(next_page.to(tl.int32) * 64, 0),
                    barrier=next_ready,
                )
            scores = inline_jit(
                pointwise_vec4,
                (
                    inline_jit(pointwise_vec4, (content, qs[:, None], True)),
                    ks[None, :],
                    True,
                ),
            ) * (SCALE * 1.4426950408889634)
            if page_idx * 64 + 64 <= length - QLEN + 1:
                masked_scores = scores
            else:
                valid = valid_head[:, None] & valid_token[None, :]
                if CAUSAL:
                    valid = valid & (
                        page_idx * 64 + tokens[None, :]
                        < length - QLEN + queries[:, None] + 1
                    )
                masked_scores = tl.where(valid, scores, -float("inf"))
            scores = masked_scores
            new_maximum = tl.maximum(maximum, inline_jit(row_reduce, (scores, True)))
            safe_maximum = tl.where(new_maximum == -float("inf"), 0.0, new_maximum)
            probabilities = inline_jit(
                exp2_pairs,
                (inline_jit(pointwise_vec4, (scores, -safe_maximum[:, None], False)),),
            )
            alpha = tl.exp2(maximum - safe_maximum)
            weights = inline_jit(pointwise_vec4, (probabilities, ks[None, :], True))
            page_sum, page_max = inline_jit(row_sum_and_max, (probabilities, weights))
            denominator = denominator * alpha + page_sum
            scale = tl.maximum(page_max * (1.0 / 448.0), 1.0e-30)
            inverse_scale = native_recip(scale)
            scaled = inline_jit(pointwise_vec4, (weights, inverse_scale[:, None], True))
            high = scaled.to(tl.float8e4nv)
            tl.store(tle.gpu.local_ptr(SP.slot(0)), high)
            tl.store(
                tle.gpu.local_ptr(SA.slot(0)), running_scale * alpha * inverse_scale
            )
            tl.store(tle.gpu.local_ptr(SD.slot(0)), denominator * inverse_scale)
            # Publish before loop-exit layout conversion across the producer squads.
            if page_idx + 1 == end_page:
                tl.store(tle.gpu.local_ptr(SMax.slot(0)), new_maximum)
                tl.store(tle.gpu.local_ptr(SDenom.slot(0)), denominator)
            running_scale = scale
            # Complete shared writes before publishing P and its row scales.
            native_lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Full[slot], phaseIdx=phase)
            maximum = new_maximum
        if first_page < end_page:
            native_lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(End, phaseIdx=(end_page - first_page - 1) & 1)
            tle.gpu.barrier_wait(End, phaseIdx=(end_page - first_page - 1) & 1)
        final_phase = (end_page - first_page) & 1
        native_lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(End, phaseIdx=final_phase)
        tle.gpu.barrier_wait(End, phaseIdx=final_phase)
        native_lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(End, phaseIdx=final_phase ^ 1)
        tle.gpu.barrier_wait(End, phaseIdx=final_phase ^ 1)

    @triton.jit
    def dense_fp8_mla_value_consumer(
        Lengths,
        Output,
        Partial,
        LSE,
        Stats,
        SK0,
        SK1,
        VSync,
        SP,
        SA,
        SD,
        SMax,
        SDenom,
        End,
        Full,
        Empty,
        LS: tl.constexpr,
        H: tl.constexpr,
        QLEN: tl.constexpr,
        BH: tl.constexpr,
        HALF: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        group, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        heads = group * BH + tl.arange(0, BH)
        valid_head = tl.full((BH,), True, tl.int1)
        dims = HALF * 256 + tl.arange(0, 256)
        length = tl.load(Lengths + batch * LS)
        first_page = tl.cdiv(length, 64) * split // SPLITS
        end_page = tl.cdiv(length, 64) * (split + 1) // SPLITS
        accumulator = tl.zeros((BH, 256), tl.float32)
        denominator = tl.zeros((BH,), tl.float32)
        for page_idx in range(first_page, end_page):
            slot = (page_idx - first_page) & 1
            phase = ((page_idx - first_page) // 2) & 1
            tle.gpu.barrier_wait(Full[slot], phaseIdx=phase)
            if HALF == 0:
                values = SK0.slot(slot)
            else:
                values = SK1.slot(slot)
            if page_idx * 64 + 64 > length:
                # A narrow pointer tensor limits register use while clearing padding.
                columns = tl.arange(0, 256)
                for row in range(length - page_idx * 64, 64):
                    indices = tl.full((256,), row, tl.int32)
                    tl.store(tle.gpu.local_ptr(values, (indices, columns)), 0.0)
                native_lma_wait(tl.full((), 0, tl.int32))
                tle.gpu.barrier_arrive(VSync[HALF], phaseIdx=0)
                tle.gpu.barrier_wait(VSync[HALF], phaseIdx=0)
            rescale = tl.load(tle.gpu.local_ptr(SA.slot(0)))
            denominator = tl.load(tle.gpu.local_ptr(SD.slot(0)))
            accumulator = tle.gpu.wgmma(
                SP.slot(0), values, acc=accumulator * rescale[:, None]
            )
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
            # Finish local accesses before allowing the producer to reuse KV/P.
            native_lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(Empty[slot], phaseIdx=phase)
            native_lma_wait(tl.full((), 0, tl.int32))
            tle.gpu.barrier_arrive(End, phaseIdx=(page_idx - first_page) & 1)
            tle.gpu.barrier_wait(End, phaseIdx=(page_idx - first_page) & 1)
        inverse_denominator = native_recip(tl.where(denominator > 0, denominator, 1.0))
        output = accumulator * inverse_denominator[:, None]
        if SPLITS == 1:
            tl.store(
                Output + (batch * H * QLEN + heads[:, None]) * 512 + dims[None, :],
                output,
                valid_head[:, None],
            )
        else:
            tl.store(
                Partial
                + ((batch * SPLITS + split) * H * QLEN + heads[:, None]) * 512
                + dims[None, :],
                output,
                valid_head[:, None],
            )
        final_phase = (end_page - first_page) & 1
        native_lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(End, phaseIdx=final_phase)
        tle.gpu.barrier_wait(End, phaseIdx=final_phase)
        if HALF == 0:
            if first_page < end_page:
                row_maximum = tl.load(tle.gpu.local_ptr(SMax.slot(0)))
                row_denominator = tl.load(tle.gpu.local_ptr(SDenom.slot(0)))
            else:
                row_maximum = tl.full((BH,), -float("inf"), tl.float32)
                row_denominator = tl.zeros((BH,), tl.float32)
            logsum = tl.where(
                row_denominator > 0,
                row_maximum + tl.log2(row_denominator),
                -float("inf"),
            )
            if SPLITS == 1:
                tl.store(
                    LSE + (batch * H + heads % H) * QLEN + heads // H,
                    logsum * 0.6931471805599453,
                )
            else:
                tl.store(Stats + ((batch * SPLITS + split) * H * QLEN + heads), logsum)
        native_lma_wait(tl.full((), 0, tl.int32))
        tle.gpu.barrier_arrive(End, phaseIdx=final_phase ^ 1)
        tle.gpu.barrier_wait(End, phaseIdx=final_phase ^ 1)

    def bench_dense_tle(kernel_call, quantiles):
        from triton.testing import _summarize_statistics

        # Triton's graph timer currently hardcodes torch.cuda.
        device = runtime.torch_device_fn
        stream = device.Stream()
        stream.wait_stream(device.current_stream())
        with device.stream(stream):
            kernel_call()
            begin = device.Event(enable_timing=True)
            end = device.Event(enable_timing=True)
            begin.record()
            for _ in range(5):
                kernel_call()
            end.record()
            device.synchronize()
            estimate = begin.elapsed_time(end) / 5
            repeats = max(1, int(40 / estimate)) if estimate > 0 else 1000
            graph = torch.musa.MUSAGraph()
            with device.graph(graph):
                for _ in range(repeats):
                    kernel_call()
            device.synchronize()
            samples = []
            for _ in range(3):
                begin.record()
                graph.replay()
                end.record()
                device.synchronize()
                samples.append(begin.elapsed_time(end) / repeats)
        return _summarize_statistics(samples, quantiles, "median")

    def tune_dense_tle_combine(arguments: dict, exception: Optional[Exception]) -> None:
        if (
            exception is not None
            or arguments["REQUESTED_SPLITS"] != 0
            or arguments["SPLITS"] == 1
        ):
            return
        heads = arguments["H"] * arguments["QLEN"]
        dense_fp8_mla_tle_combine[(arguments["BATCH"], triton.cdiv(heads, 8))](
            arguments["Partial"],
            arguments["Stats"],
            arguments["Output"],
            arguments["LSE"],
            heads,
            arguments["H"],
            arguments["QLEN"],
            SPLITS=arguments["SPLITS"],
            num_warps=8,
        )

    @libtuner(
        configs=runtime.get_tuned_config("mthreads_dense_fp8_mla_tle"),
        key=["BATCH", "H", "QLEN", "CAPACITY", "REQUESTED_SPLITS"],
        prune_configs_by={"early_config_prune": prune_dense_tle_configs},
        do_bench=bench_dense_tle,
        post_hook=tune_dense_tle_combine,
    )
    @triton.jit
    def dense_fp8_mla_tle_kernel(
        Q,
        QR,
        QS,
        KS,
        Table,
        Lengths,
        KD,
        RD,
        Output,
        LSE,
        Partial,
        Stats,
        QST: tl.constexpr,
        QRST: tl.constexpr,
        QSST: tl.constexpr,
        KSST: tl.constexpr,
        TS: tl.constexpr,
        LS: tl.constexpr,
        H: tl.constexpr,
        QLEN: tl.constexpr,
        SCALE: tl.constexpr,
        BH: tl.constexpr,
        CW: tl.constexpr,
        PW: tl.constexpr,
        SPLITS: tl.constexpr,
        CAUSAL: tl.constexpr,
        BATCH: tl.constexpr,
        CAPACITY: tl.constexpr,
        REQUESTED_SPLITS: tl.constexpr,
    ):
        sk0 = tle.gpu.alloc((2, 64, 256), tl.float8e4nv, nv_mma_shared_layout=True)
        sk1 = tle.gpu.alloc((2, 64, 256), tl.float8e4nv, nv_mma_shared_layout=True)
        vsync = tle.gpu.alloc_barriers(2, arrive_count=CW)
        sr = tle.gpu.alloc((1, 64, 64), tl.bfloat16, nv_mma_shared_layout=True)
        sp = tle.gpu.alloc((1, BH, 64), tl.float8e4nv, nv_mma_shared_layout=True)
        sa = tle.gpu.alloc((1, BH), tl.float32, nv_mma_shared_layout=False)
        sd = tle.gpu.alloc((1, BH), tl.float32, nv_mma_shared_layout=False)
        smax = tle.gpu.alloc((1, BH), tl.float32, nv_mma_shared_layout=False)
        sdenom = tle.gpu.alloc((1, BH), tl.float32, nv_mma_shared_layout=False)
        full = tle.gpu.alloc_barriers(2, arrive_count=PW)
        empty = tle.gpu.alloc_barriers(2, arrive_count=2 * CW, init=tle.gpu.READY)
        sync = tle.gpu.alloc_barrier(arrive_count=PW)
        kb0 = tle.gpu.alloc_barriers(
            2, arrive_count=3, expect_bytes=64 * 512 + 64 * 64 * 2
        )
        end = tle.gpu.alloc_barrier(arrive_count=2 * CW + PW)
        tl.debug_barrier()
        tle.gpu.warp_specialize(
            [
                (
                    dense_fp8_mla_value_consumer,
                    (
                        Lengths,
                        Output,
                        Partial,
                        LSE,
                        Stats,
                        sk0,
                        sk1,
                        vsync,
                        sp,
                        sa,
                        sd,
                        smax,
                        sdenom,
                        end,
                        full,
                        empty,
                        LS,
                        H,
                        QLEN,
                        BH,
                        0,
                        SPLITS,
                    ),
                ),
                (
                    dense_fp8_mla_value_consumer,
                    (
                        Lengths,
                        Output,
                        Partial,
                        LSE,
                        Stats,
                        sk0,
                        sk1,
                        vsync,
                        sp,
                        sa,
                        sd,
                        smax,
                        sdenom,
                        end,
                        full,
                        empty,
                        LS,
                        H,
                        QLEN,
                        BH,
                        1,
                        SPLITS,
                    ),
                ),
                (
                    dense_fp8_mla_score_producer,
                    (
                        Q,
                        QR,
                        QS,
                        KS,
                        Table,
                        Lengths,
                        LSE,
                        Partial,
                        Stats,
                        KD,
                        RD,
                        sk0,
                        sk1,
                        sr,
                        sp,
                        sa,
                        sd,
                        smax,
                        sdenom,
                        end,
                        full,
                        empty,
                        sync,
                        kb0,
                        QST[0],
                        QST[1],
                        QST[2],
                        QRST[0],
                        QRST[1],
                        QRST[2],
                        QSST[0],
                        QSST[1],
                        QSST[2],
                        KSST[0],
                        KSST[1],
                        TS,
                        LS,
                        H,
                        QLEN,
                        BH,
                        SCALE,
                        CAUSAL,
                        PW,
                        SPLITS,
                    ),
                ),
            ],
            worker_num_warps=[CW, PW],
            worker_num_regs=[224, 128],
        )

    COMBINE_LAYOUT = tl.constexpr(
        tle.gpu.BlockEncoding([1, 4], [1, 32], [8, 1], [1, 0])
    )
    COMBINE_HEADS = tl.constexpr(tle.gpu.SlicedEncoding(1, COMBINE_LAYOUT.value))
    COMBINE_DIMS = tl.constexpr(tle.gpu.SlicedEncoding(0, COMBINE_LAYOUT.value))

    @triton.jit
    def dense_fp8_mla_tle_combine(
        partial,
        partial_lse,
        out,
        lse,
        HEADS: tl.constexpr,
        H: tl.constexpr,
        QLEN: tl.constexpr,
        SPLITS: tl.constexpr,
    ):
        request = tl.program_id(0)
        row = HEADS * 512
        splits = SPLITS
        first = request * splits
        end = first + splits
        if splits > 1:
            h = tl.program_id(1) * 8 + tle.gpu.set_layout(
                tl.arange(0, 8), COMBINE_HEADS
            )
            d = tle.gpu.set_layout(tl.arange(0, 512), COMBINE_DIMS)
            maximum = tl.full((8,), float("-inf"), tl.float32)
            denominator = tl.zeros((8,), tl.float32)
            result = tl.zeros((8, 512), tl.float32)
            for slot in range(first, end):
                value = tl.load(partial_lse + slot * HEADS + h)
                new_maximum = tl.maximum(maximum, value)
                safe_maximum = tl.where(new_maximum == float("-inf"), 0.0, new_maximum)
                alpha = tl.exp2(maximum - safe_maximum)
                weight = tl.exp2(value - safe_maximum)
                result = (
                    result * alpha[:, None]
                    + tl.load(partial + slot * row + h[:, None] * 512 + d[None, :])
                    * weight[:, None]
                )
                denominator = denominator * alpha + weight
                maximum = new_maximum
            valid = denominator > 0
            natural_lse = tl.where(
                valid,
                (tl.log2(denominator) + maximum) * 0.6931471805599453,
                float("-inf"),
            )
            factor = tl.where(valid, 1.0 / denominator, 0.0)
            tl.store(
                out + request * row + h[:, None] * 512 + d[None, :],
                result * factor[:, None],
            )
            tl.store(lse + (request * H + h % H) * QLEN + h // H, natural_lse)


def flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    tile_scheduler_metadata: "FlashMLAFp8SplitKSchedMeta | None" = None,
    num_splits: Optional[torch.Tensor | int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense FP8 MLA on MUSA; RoPE is divided by the content scale as in FP8 MLA.

    Inputs have 512 content dimensions, 64 RoPE dimensions and 64-token pages.
    Active scales must be finite and positive, page IDs and cache lengths valid.
    Lengths may change on the GPU without rebuilding the capacity-based schedule.
    Empty attention rows return zero output and negative-infinity LSE.
    """
    k_lora, k_rope = k_cache_lora, k_cache_rope
    if tile_scheduler_metadata is not None or isinstance(num_splits, torch.Tensor):
        raise NotImplementedError("MUSA FP8 MLA does not consume NVIDIA split metadata")
    if (
        q_nope.ndim != 4
        or q_nope.shape[1] not in (1, 2)
        or q_nope.shape[2] not in (16, 32, 64, 128)
        or q_nope.shape[3] != 512
    ):
        raise NotImplementedError("requires Q [batch, 1 or 2, 16/32/64/128 heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if (
        k_lora.ndim not in (3, 4)
        or k_lora.shape[1] != 64
        or k_lora.shape[-1] != 512
        or head_dim_v != 512
    ):
        raise NotImplementedError(
            "requires KV [pages, 64, (1,), 512] and head_dim_v=512"
        )
    if k_lora.ndim == 4 and k_lora.shape[2] != 1:
        raise NotImplementedError("MLA requires one KV head")
    expected_shapes = (
        (batch, queries, heads, 64),
        (*k_lora.shape[:-1], 64),
        (batch, queries, heads, 1),
        (*k_lora.shape[:-1], 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_rope, q_scale, k_scale, cache_seqlens), expected_shapes
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("block_table must have shape [batch, page capacity]")
    if q_nope.dtype != torch.float8_e4m3fn or k_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("Q and KV content must be FP8 E4M3")
    if (
        q_rope.dtype not in (torch.float16, torch.bfloat16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE tensors must have the same FP16 or BF16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("block_table and cache_seqlens must be INT32")
    tensors = (
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
    )
    if q_nope.device.type != "musa":
        raise ValueError("FP8 MLA requires MUSA tensors")
    for tensor in tensors:
        if tensor.device != q_nope.device or (
            tensor.numel() and tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all tensors must share a device and contiguous last dimension"
            )
    scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if num_splits is None:
        tiles = max(1, batch * triton.cdiv(queries * heads, 16))
        splits = min(32, triton.next_power_of_2(max(1, 128 // tiles)))
        splits = min(splits, triton.next_power_of_2(max(1, block_table.shape[1])))
    else:
        if num_splits not in (1, 2, 4, 8, 16, 32):
            raise ValueError("num_splits must be one of 1, 2, 4, 8, 16, 32")
        splits = num_splits
    if out is None:
        out = torch.empty(
            (batch, queries, heads, 512), dtype=q_rope.dtype, device=q_nope.device
        )
    if lse is None:
        lse = torch.empty(
            (batch, heads, queries), dtype=torch.float32, device=q_nope.device
        )
    for tensor, shape, dtype in (
        (out, (batch, queries, heads, 512), q_rope.dtype),
        (lse, (batch, heads, queries), torch.float32),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != q_nope.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"output must be contiguous with shape {shape}, dtype {dtype}, and input device"
            )
        if any(torch._C._overlaps(tensor, source) for source in tensors):
            raise ValueError("outputs must not alias inputs")
    if torch._C._overlaps(out, lse):
        raise ValueError("out and lse must not alias")
    if batch == 0:
        return out, lse
    if (
        HAS_TLE
        and splits == 1
        and q_rope.dtype == torch.bfloat16
        and k_lora.shape[0] > 0
        and k_lora.is_contiguous()
        and k_rope.is_contiguous()
    ):
        require_tle_pipeline()

        from triton.tools.tensor_descriptor import TensorDescriptor

        content_descriptor = TensorDescriptor.from_tensor(
            k_lora.view(-1, 512), block_shape=[64, 256]
        )
        rope_descriptor = TensorDescriptor.from_tensor(
            k_rope.view(-1, 64), block_shape=[64, 64]
        )
        requested_splits = 0 if num_splits is None else 1
        workspace_splits = 4 if requested_splits == 0 else 0
        tle_partial = torch.empty(
            (batch, workspace_splits, queries * heads, 512),
            dtype=torch.float32,
            device=q_nope.device,
        )
        tle_stats = torch.empty(
            (batch, workspace_splits, queries * heads),
            dtype=torch.float32,
            device=q_nope.device,
        )
        with runtime.torch_device_fn.device(q_nope.device):
            compiled_kernel = dense_fp8_mla_tle_kernel[
                lambda meta: (
                    triton.cdiv(heads * queries, meta["BH"]),
                    batch,
                    meta["SPLITS"],
                )
            ](
                q_nope,
                q_rope,
                q_scale,
                k_scale,
                block_table,
                cache_seqlens,
                content_descriptor,
                rope_descriptor,
                out,
                lse,
                tle_partial,
                tle_stats,
                q_nope.stride(),
                q_rope.stride(),
                q_scale.stride(),
                k_scale.stride(),
                block_table.stride(0),
                cache_seqlens.stride(0),
                heads,
                queries,
                scale,
                CAUSAL=causal,
                BATCH=batch,
                CAPACITY=block_table.shape[1],
                REQUESTED_SPLITS=requested_splits,
                extern_libs=primitives_lib(),
            )
            # Use the immutable compile constant of this invocation.
            split_parameter = (compiled_kernel.src.fn.arg_names.index("SPLITS"),)
            active_splits = compiled_kernel.src.constants[split_parameter]
            if active_splits > 1:
                dense_fp8_mla_tle_combine[(batch, triton.cdiv(queries * heads, 8))](
                    tle_partial,
                    tle_stats,
                    out,
                    lse,
                    queries * heads,
                    heads,
                    queries,
                    SPLITS=active_splits,
                    num_warps=8,
                )
        return out, lse
    workspace_splits = splits if splits > 1 else 0
    partial = torch.empty(
        (batch, workspace_splits, queries * heads, 512),
        dtype=torch.float32,
        device=q_nope.device,
    )
    stats = torch.empty(
        (batch, workspace_splits, 2, queries * heads),
        dtype=torch.float32,
        device=q_nope.device,
    )
    with runtime.torch_device_fn.device(q_nope.device):
        dense_fp8_mla_kernel[
            lambda meta: (triton.cdiv(heads * queries, meta["BH"]), batch, splits)
        ](
            *tensors,
            partial,
            stats,
            out,
            lse,
            q_nope.stride(),
            q_rope.stride(),
            k_lora.stride(),
            k_rope.stride(),
            q_scale.stride(),
            k_scale.stride(),
            block_table.stride(0),
            cache_seqlens.stride(0),
            batch,
            heads,
            queries,
            block_table.shape[1],
            splits,
            scale,
            causal,
        )
        if splits > 1:
            # Fixed reduction geometry: no matrix tiles or tunable pipeline stages.
            dense_fp8_mla_combine_kernel[(batch, heads * queries, 4)](
                partial,
                stats,
                out,
                lse,
                heads,
                queries,
                splits,
                num_warps=4,
            )
    return out, lse
