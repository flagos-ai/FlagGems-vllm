# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl
from triton.experimental.tle.language import gpu

from flaggems_vllm.runtime.backend._thead.ops.flash_attention import common


@triton.jit
def async_prefill_attention_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    CUQ,
    CUK,
    USEDK,
    TABLE,
    ALIBI,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    SOFTCAP: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr = False,
    SHORT_LIMIT: tl.constexpr = 0,
    ORDER: tl.constexpr = 0,
    WRITE_LSE: tl.constexpr = True,
    BATCHES: tl.constexpr = 1,
    FOLD: tl.constexpr = 4,
    KNORM=None,
    WORKLIST=None,
    CAPACITY: tl.constexpr = 0,
    FIXED_MODE: tl.constexpr = False,
):
    tl.static_assert(D == 256 and BM == 64 and (FOLD == 4) and PAGED)
    tl.static_assert(Q.dtype.element_ty == tl.bfloat16)
    tl.static_assert(KP == PAGE * KS and VP == PAGE * VS)
    index = tl.program_id(0)
    group = tl.program_id(2)
    if index >= tl.load(WORKLIST + 5 * CAPACITY):
        return
    qstart = tl.load(WORKLIST + index)
    qlen = tl.load(WORKLIST + CAPACITY + index)
    klen = tl.load(WORKLIST + 2 * CAPACITY + index)
    batch = tl.load(WORKLIST + 3 * CAPACITY + index)
    tile = tl.load(WORKLIST + 4 * CAPACITY + index)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    kh = group // (HQ // HK // 4)
    if tl.load(KNORM + batch * HK + kh) == 0:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, 16) - 1 - tile
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    p = lane // 4
    q = lane % 4
    qhead = group * 4 + warp
    word = tl.where(p % 4 == q, tl.where(lane >= 16, 1065353216, 16256), 0).to(
        tl.uint32
    )
    cm0 = tile * 16 + p + 0
    cn0 = q + 0
    bk0 = 2 * q + 0
    cm1 = tile * 16 + p + 0
    cn1 = q + 4
    bk1 = 2 * q + 1
    cm2 = tile * 16 + p + 0
    cn2 = q + 8
    bk2 = 2 * q + 8
    cm3 = tile * 16 + p + 0
    cn3 = q + 12
    bk3 = 2 * q + 9
    cm4 = tile * 16 + p + 8
    cn4 = q + 0
    bk4 = 2 * q + 0
    cm5 = tile * 16 + p + 8
    cn5 = q + 4
    bk5 = 2 * q + 1
    cm6 = tile * 16 + p + 8
    cn6 = q + 8
    bk6 = 2 * q + 8
    cm7 = tile * 16 + p + 8
    cn7 = q + 12
    bk7 = 2 * q + 9
    qa0_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa0_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa0_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa0_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa0_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa0_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa0_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa0_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 0 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa1_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa1_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa1_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa1_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa1_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa1_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa1_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa1_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 16 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa2_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa2_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa2_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa2_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa2_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa2_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa2_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa2_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 32 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa3_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa3_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa3_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa3_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa3_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa3_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa3_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa3_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 48 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa4_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa4_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa4_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa4_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa4_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa4_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa4_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa4_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 64 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa5_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa5_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa5_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa5_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa5_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa5_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa5_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa5_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 80 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa6_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa6_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa6_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa6_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa6_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa6_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa6_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa6_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 96 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa7_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa7_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa7_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa7_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa7_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa7_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa7_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa7_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 112 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa8_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa8_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa8_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa8_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa8_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa8_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa8_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa8_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 128 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa9_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa9_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa9_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa9_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa9_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa9_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa9_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa9_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 144 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa10_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa10_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa10_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa10_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa10_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa10_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa10_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa10_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 160 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa11_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa11_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa11_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa11_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa11_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa11_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa11_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa11_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 176 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa12_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa12_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa12_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa12_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa12_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa12_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa12_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa12_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 192 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa13_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa13_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa13_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa13_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa13_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa13_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa13_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa13_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 208 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa14_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa14_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa14_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa14_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa14_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa14_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa14_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa14_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 224 + bk7,
        cm7 < qlen,
        0.0,
    )
    qa15_0 = tl.load(
        Q + (qstart + cm0).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk0,
        cm0 < qlen,
        0.0,
    )
    qa15_1 = tl.load(
        Q + (qstart + cm1).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk1,
        cm1 < qlen,
        0.0,
    )
    qa15_2 = tl.load(
        Q + (qstart + cm2).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk2,
        cm2 < qlen,
        0.0,
    )
    qa15_3 = tl.load(
        Q + (qstart + cm3).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk3,
        cm3 < qlen,
        0.0,
    )
    qa15_4 = tl.load(
        Q + (qstart + cm4).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk4,
        cm4 < qlen,
        0.0,
    )
    qa15_5 = tl.load(
        Q + (qstart + cm5).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk5,
        cm5 < qlen,
        0.0,
    )
    qa15_6 = tl.load(
        Q + (qstart + cm6).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk6,
        cm6 < qlen,
        0.0,
    )
    qa15_7 = tl.load(
        Q + (qstart + cm7).to(tl.int64) * QS + qhead.to(tl.int64) * QH + 240 + bk7,
        cm7 < qlen,
        0.0,
    )
    qp0_0 = common.pack_bf(qa0_0, qa0_1)
    qp0_1 = common.pack_bf(qa0_2, qa0_3)
    qp0_2 = common.pack_bf(qa0_4, qa0_5)
    qp0_3 = common.pack_bf(qa0_6, qa0_7)
    qp1_0 = common.pack_bf(qa1_0, qa1_1)
    qp1_1 = common.pack_bf(qa1_2, qa1_3)
    qp1_2 = common.pack_bf(qa1_4, qa1_5)
    qp1_3 = common.pack_bf(qa1_6, qa1_7)
    qp2_0 = common.pack_bf(qa2_0, qa2_1)
    qp2_1 = common.pack_bf(qa2_2, qa2_3)
    qp2_2 = common.pack_bf(qa2_4, qa2_5)
    qp2_3 = common.pack_bf(qa2_6, qa2_7)
    qp3_0 = common.pack_bf(qa3_0, qa3_1)
    qp3_1 = common.pack_bf(qa3_2, qa3_3)
    qp3_2 = common.pack_bf(qa3_4, qa3_5)
    qp3_3 = common.pack_bf(qa3_6, qa3_7)
    qp4_0 = common.pack_bf(qa4_0, qa4_1)
    qp4_1 = common.pack_bf(qa4_2, qa4_3)
    qp4_2 = common.pack_bf(qa4_4, qa4_5)
    qp4_3 = common.pack_bf(qa4_6, qa4_7)
    qp5_0 = common.pack_bf(qa5_0, qa5_1)
    qp5_1 = common.pack_bf(qa5_2, qa5_3)
    qp5_2 = common.pack_bf(qa5_4, qa5_5)
    qp5_3 = common.pack_bf(qa5_6, qa5_7)
    qp6_0 = common.pack_bf(qa6_0, qa6_1)
    qp6_1 = common.pack_bf(qa6_2, qa6_3)
    qp6_2 = common.pack_bf(qa6_4, qa6_5)
    qp6_3 = common.pack_bf(qa6_6, qa6_7)
    qp7_0 = common.pack_bf(qa7_0, qa7_1)
    qp7_1 = common.pack_bf(qa7_2, qa7_3)
    qp7_2 = common.pack_bf(qa7_4, qa7_5)
    qp7_3 = common.pack_bf(qa7_6, qa7_7)
    qp8_0 = common.pack_bf(qa8_0, qa8_1)
    qp8_1 = common.pack_bf(qa8_2, qa8_3)
    qp8_2 = common.pack_bf(qa8_4, qa8_5)
    qp8_3 = common.pack_bf(qa8_6, qa8_7)
    qp9_0 = common.pack_bf(qa9_0, qa9_1)
    qp9_1 = common.pack_bf(qa9_2, qa9_3)
    qp9_2 = common.pack_bf(qa9_4, qa9_5)
    qp9_3 = common.pack_bf(qa9_6, qa9_7)
    qp10_0 = common.pack_bf(qa10_0, qa10_1)
    qp10_1 = common.pack_bf(qa10_2, qa10_3)
    qp10_2 = common.pack_bf(qa10_4, qa10_5)
    qp10_3 = common.pack_bf(qa10_6, qa10_7)
    qp11_0 = common.pack_bf(qa11_0, qa11_1)
    qp11_1 = common.pack_bf(qa11_2, qa11_3)
    qp11_2 = common.pack_bf(qa11_4, qa11_5)
    qp11_3 = common.pack_bf(qa11_6, qa11_7)
    qp12_0 = common.pack_bf(qa12_0, qa12_1)
    qp12_1 = common.pack_bf(qa12_2, qa12_3)
    qp12_2 = common.pack_bf(qa12_4, qa12_5)
    qp12_3 = common.pack_bf(qa12_6, qa12_7)
    qp13_0 = common.pack_bf(qa13_0, qa13_1)
    qp13_1 = common.pack_bf(qa13_2, qa13_3)
    qp13_2 = common.pack_bf(qa13_4, qa13_5)
    qp13_3 = common.pack_bf(qa13_6, qa13_7)
    qp14_0 = common.pack_bf(qa14_0, qa14_1)
    qp14_1 = common.pack_bf(qa14_2, qa14_3)
    qp14_2 = common.pack_bf(qa14_4, qa14_5)
    qp14_3 = common.pack_bf(qa14_6, qa14_7)
    qp15_0 = common.pack_bf(qa15_0, qa15_1)
    qp15_1 = common.pack_bf(qa15_2, qa15_3)
    qp15_2 = common.pack_bf(qa15_4, qa15_5)
    qp15_3 = common.pack_bf(qa15_6, qa15_7)
    o0_0 = tl.full((128,), 0.0, tl.float32)
    o0_1 = tl.full((128,), 0.0, tl.float32)
    o0_2 = tl.full((128,), 0.0, tl.float32)
    o0_3 = tl.full((128,), 0.0, tl.float32)
    o0_4 = tl.full((128,), 0.0, tl.float32)
    o0_5 = tl.full((128,), 0.0, tl.float32)
    o0_6 = tl.full((128,), 0.0, tl.float32)
    o0_7 = tl.full((128,), 0.0, tl.float32)
    o1_0 = tl.full((128,), 0.0, tl.float32)
    o1_1 = tl.full((128,), 0.0, tl.float32)
    o1_2 = tl.full((128,), 0.0, tl.float32)
    o1_3 = tl.full((128,), 0.0, tl.float32)
    o1_4 = tl.full((128,), 0.0, tl.float32)
    o1_5 = tl.full((128,), 0.0, tl.float32)
    o1_6 = tl.full((128,), 0.0, tl.float32)
    o1_7 = tl.full((128,), 0.0, tl.float32)
    o2_0 = tl.full((128,), 0.0, tl.float32)
    o2_1 = tl.full((128,), 0.0, tl.float32)
    o2_2 = tl.full((128,), 0.0, tl.float32)
    o2_3 = tl.full((128,), 0.0, tl.float32)
    o2_4 = tl.full((128,), 0.0, tl.float32)
    o2_5 = tl.full((128,), 0.0, tl.float32)
    o2_6 = tl.full((128,), 0.0, tl.float32)
    o2_7 = tl.full((128,), 0.0, tl.float32)
    o3_0 = tl.full((128,), 0.0, tl.float32)
    o3_1 = tl.full((128,), 0.0, tl.float32)
    o3_2 = tl.full((128,), 0.0, tl.float32)
    o3_3 = tl.full((128,), 0.0, tl.float32)
    o3_4 = tl.full((128,), 0.0, tl.float32)
    o3_5 = tl.full((128,), 0.0, tl.float32)
    o3_6 = tl.full((128,), 0.0, tl.float32)
    o3_7 = tl.full((128,), 0.0, tl.float32)
    o4_0 = tl.full((128,), 0.0, tl.float32)
    o4_1 = tl.full((128,), 0.0, tl.float32)
    o4_2 = tl.full((128,), 0.0, tl.float32)
    o4_3 = tl.full((128,), 0.0, tl.float32)
    o4_4 = tl.full((128,), 0.0, tl.float32)
    o4_5 = tl.full((128,), 0.0, tl.float32)
    o4_6 = tl.full((128,), 0.0, tl.float32)
    o4_7 = tl.full((128,), 0.0, tl.float32)
    o5_0 = tl.full((128,), 0.0, tl.float32)
    o5_1 = tl.full((128,), 0.0, tl.float32)
    o5_2 = tl.full((128,), 0.0, tl.float32)
    o5_3 = tl.full((128,), 0.0, tl.float32)
    o5_4 = tl.full((128,), 0.0, tl.float32)
    o5_5 = tl.full((128,), 0.0, tl.float32)
    o5_6 = tl.full((128,), 0.0, tl.float32)
    o5_7 = tl.full((128,), 0.0, tl.float32)
    o6_0 = tl.full((128,), 0.0, tl.float32)
    o6_1 = tl.full((128,), 0.0, tl.float32)
    o6_2 = tl.full((128,), 0.0, tl.float32)
    o6_3 = tl.full((128,), 0.0, tl.float32)
    o6_4 = tl.full((128,), 0.0, tl.float32)
    o6_5 = tl.full((128,), 0.0, tl.float32)
    o6_6 = tl.full((128,), 0.0, tl.float32)
    o6_7 = tl.full((128,), 0.0, tl.float32)
    o7_0 = tl.full((128,), 0.0, tl.float32)
    o7_1 = tl.full((128,), 0.0, tl.float32)
    o7_2 = tl.full((128,), 0.0, tl.float32)
    o7_3 = tl.full((128,), 0.0, tl.float32)
    o7_4 = tl.full((128,), 0.0, tl.float32)
    o7_5 = tl.full((128,), 0.0, tl.float32)
    o7_6 = tl.full((128,), 0.0, tl.float32)
    o7_7 = tl.full((128,), 0.0, tl.float32)
    o8_0 = tl.full((128,), 0.0, tl.float32)
    o8_1 = tl.full((128,), 0.0, tl.float32)
    o8_2 = tl.full((128,), 0.0, tl.float32)
    o8_3 = tl.full((128,), 0.0, tl.float32)
    o8_4 = tl.full((128,), 0.0, tl.float32)
    o8_5 = tl.full((128,), 0.0, tl.float32)
    o8_6 = tl.full((128,), 0.0, tl.float32)
    o8_7 = tl.full((128,), 0.0, tl.float32)
    o9_0 = tl.full((128,), 0.0, tl.float32)
    o9_1 = tl.full((128,), 0.0, tl.float32)
    o9_2 = tl.full((128,), 0.0, tl.float32)
    o9_3 = tl.full((128,), 0.0, tl.float32)
    o9_4 = tl.full((128,), 0.0, tl.float32)
    o9_5 = tl.full((128,), 0.0, tl.float32)
    o9_6 = tl.full((128,), 0.0, tl.float32)
    o9_7 = tl.full((128,), 0.0, tl.float32)
    o10_0 = tl.full((128,), 0.0, tl.float32)
    o10_1 = tl.full((128,), 0.0, tl.float32)
    o10_2 = tl.full((128,), 0.0, tl.float32)
    o10_3 = tl.full((128,), 0.0, tl.float32)
    o10_4 = tl.full((128,), 0.0, tl.float32)
    o10_5 = tl.full((128,), 0.0, tl.float32)
    o10_6 = tl.full((128,), 0.0, tl.float32)
    o10_7 = tl.full((128,), 0.0, tl.float32)
    o11_0 = tl.full((128,), 0.0, tl.float32)
    o11_1 = tl.full((128,), 0.0, tl.float32)
    o11_2 = tl.full((128,), 0.0, tl.float32)
    o11_3 = tl.full((128,), 0.0, tl.float32)
    o11_4 = tl.full((128,), 0.0, tl.float32)
    o11_5 = tl.full((128,), 0.0, tl.float32)
    o11_6 = tl.full((128,), 0.0, tl.float32)
    o11_7 = tl.full((128,), 0.0, tl.float32)
    o12_0 = tl.full((128,), 0.0, tl.float32)
    o12_1 = tl.full((128,), 0.0, tl.float32)
    o12_2 = tl.full((128,), 0.0, tl.float32)
    o12_3 = tl.full((128,), 0.0, tl.float32)
    o12_4 = tl.full((128,), 0.0, tl.float32)
    o12_5 = tl.full((128,), 0.0, tl.float32)
    o12_6 = tl.full((128,), 0.0, tl.float32)
    o12_7 = tl.full((128,), 0.0, tl.float32)
    o13_0 = tl.full((128,), 0.0, tl.float32)
    o13_1 = tl.full((128,), 0.0, tl.float32)
    o13_2 = tl.full((128,), 0.0, tl.float32)
    o13_3 = tl.full((128,), 0.0, tl.float32)
    o13_4 = tl.full((128,), 0.0, tl.float32)
    o13_5 = tl.full((128,), 0.0, tl.float32)
    o13_6 = tl.full((128,), 0.0, tl.float32)
    o13_7 = tl.full((128,), 0.0, tl.float32)
    o14_0 = tl.full((128,), 0.0, tl.float32)
    o14_1 = tl.full((128,), 0.0, tl.float32)
    o14_2 = tl.full((128,), 0.0, tl.float32)
    o14_3 = tl.full((128,), 0.0, tl.float32)
    o14_4 = tl.full((128,), 0.0, tl.float32)
    o14_5 = tl.full((128,), 0.0, tl.float32)
    o14_6 = tl.full((128,), 0.0, tl.float32)
    o14_7 = tl.full((128,), 0.0, tl.float32)
    o15_0 = tl.full((128,), 0.0, tl.float32)
    o15_1 = tl.full((128,), 0.0, tl.float32)
    o15_2 = tl.full((128,), 0.0, tl.float32)
    o15_3 = tl.full((128,), 0.0, tl.float32)
    o15_4 = tl.full((128,), 0.0, tl.float32)
    o15_5 = tl.full((128,), 0.0, tl.float32)
    o15_6 = tl.full((128,), 0.0, tl.float32)
    o15_7 = tl.full((128,), 0.0, tl.float32)
    sum0 = tl.full((128,), 0.0, tl.float32)
    sum1 = tl.full((128,), 0.0, tl.float32)
    hi = klen
    delta = klen - qlen
    if CAUSAL:
        hi = tl.minimum(klen, tl.maximum(0, (tile + 1) * 16 + delta))
    base = batch.to(tl.int64) * (TS * PAGE)
    kbuf = gpu.alloc(
        (256, 64),
        K.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    vbuf = gpu.alloc(
        (256, 64),
        V.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    if hi > 0:
        common.issue_copy(K + base * KS, kbuf, 0, klen, KS, 0, kh * KH)
        common.issue_copy(V + base * VS, vbuf, 0, klen, VS, 0, kh * VH)
        common.commit_copies()
    token = common.wait_copies() + tl.zeros((128,), tl.int32)
    for start in range(0, hi, 32):
        slot = start // 32 % 2
        if start + 32 < hi:
            common.issue_copy(
                K + base * KS, kbuf, start + 32, klen, KS, (1 - slot) * 32, kh * KH
            )
            common.issue_copy(
                V + base * VS, vbuf, start + 32, klen, VS, (1 - slot) * 32, kh * VH
            )
            common.commit_copies()
        s0_0 = tl.full((128,), 0.0, tl.float32)
        s0_1 = tl.full((128,), 0.0, tl.float32)
        s0_2 = tl.full((128,), 0.0, tl.float32)
        s0_3 = tl.full((128,), 0.0, tl.float32)
        s0_4 = tl.full((128,), 0.0, tl.float32)
        s0_5 = tl.full((128,), 0.0, tl.float32)
        s0_6 = tl.full((128,), 0.0, tl.float32)
        s0_7 = tl.full((128,), 0.0, tl.float32)
        s1_0 = tl.full((128,), 0.0, tl.float32)
        s1_1 = tl.full((128,), 0.0, tl.float32)
        s1_2 = tl.full((128,), 0.0, tl.float32)
        s1_3 = tl.full((128,), 0.0, tl.float32)
        s1_4 = tl.full((128,), 0.0, tl.float32)
        s1_5 = tl.full((128,), 0.0, tl.float32)
        s1_6 = tl.full((128,), 0.0, tl.float32)
        s1_7 = tl.full((128,), 0.0, tl.float32)
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 0, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp0_0,
            qp0_1,
            qp0_2,
            qp0_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 0, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp0_0,
            qp0_1,
            qp0_2,
            qp0_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 16, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp1_0,
            qp1_1,
            qp1_2,
            qp1_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 16, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp1_0,
            qp1_1,
            qp1_2,
            qp1_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 32, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp2_0,
            qp2_1,
            qp2_2,
            qp2_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 32, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp2_0,
            qp2_1,
            qp2_2,
            qp2_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 48, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp3_0,
            qp3_1,
            qp3_2,
            qp3_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 48, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp3_0,
            qp3_1,
            qp3_2,
            qp3_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 64, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp4_0,
            qp4_1,
            qp4_2,
            qp4_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 64, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp4_0,
            qp4_1,
            qp4_2,
            qp4_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 80, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp5_0,
            qp5_1,
            qp5_2,
            qp5_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 80, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp5_0,
            qp5_1,
            qp5_2,
            qp5_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 96, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp6_0,
            qp6_1,
            qp6_2,
            qp6_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 96, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp6_0,
            qp6_1,
            qp6_2,
            qp6_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 112, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp7_0,
            qp7_1,
            qp7_2,
            qp7_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 112, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp7_0,
            qp7_1,
            qp7_2,
            qp7_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 128, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp8_0,
            qp8_1,
            qp8_2,
            qp8_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 128, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp8_0,
            qp8_1,
            qp8_2,
            qp8_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 144, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp9_0,
            qp9_1,
            qp9_2,
            qp9_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 144, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp9_0,
            qp9_1,
            qp9_2,
            qp9_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 160, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp10_0,
            qp10_1,
            qp10_2,
            qp10_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 160, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp10_0,
            qp10_1,
            qp10_2,
            qp10_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 176, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp11_0,
            qp11_1,
            qp11_2,
            qp11_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 176, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp11_0,
            qp11_1,
            qp11_2,
            qp11_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 192, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp12_0,
            qp12_1,
            qp12_2,
            qp12_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 192, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp12_0,
            qp12_1,
            qp12_2,
            qp12_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 208, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp13_0,
            qp13_1,
            qp13_2,
            qp13_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 208, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp13_0,
            qp13_1,
            qp13_2,
            qp13_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 224, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp14_0,
            qp14_1,
            qp14_2,
            qp14_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 224, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp14_0,
            qp14_1,
            qp14_2,
            qp14_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 240, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp15_0,
            qp15_1,
            qp15_2,
            qp15_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 240, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp15_0,
            qp15_1,
            qp15_2,
            qp15_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        valid0 = (cm0 < qlen) & (start + 0 + cn0 < klen)
        if CAUSAL:
            valid0 = valid0 & (start + 0 + cn0 <= cm0 + delta)
        p0 = tl.exp2(
            tl.where(
                valid0,
                s0_0 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid1 = (cm1 < qlen) & (start + 0 + cn1 < klen)
        if CAUSAL:
            valid1 = valid1 & (start + 0 + cn1 <= cm1 + delta)
        p1 = tl.exp2(
            tl.where(
                valid1,
                s0_1 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid2 = (cm2 < qlen) & (start + 0 + cn2 < klen)
        if CAUSAL:
            valid2 = valid2 & (start + 0 + cn2 <= cm2 + delta)
        p2 = tl.exp2(
            tl.where(
                valid2,
                s0_2 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid3 = (cm3 < qlen) & (start + 0 + cn3 < klen)
        if CAUSAL:
            valid3 = valid3 & (start + 0 + cn3 <= cm3 + delta)
        p3 = tl.exp2(
            tl.where(
                valid3,
                s0_3 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid4 = (cm4 < qlen) & (start + 0 + cn4 < klen)
        if CAUSAL:
            valid4 = valid4 & (start + 0 + cn4 <= cm4 + delta)
        p4 = tl.exp2(
            tl.where(
                valid4,
                s0_4 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid5 = (cm5 < qlen) & (start + 0 + cn5 < klen)
        if CAUSAL:
            valid5 = valid5 & (start + 0 + cn5 <= cm5 + delta)
        p5 = tl.exp2(
            tl.where(
                valid5,
                s0_5 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid6 = (cm6 < qlen) & (start + 0 + cn6 < klen)
        if CAUSAL:
            valid6 = valid6 & (start + 0 + cn6 <= cm6 + delta)
        p6 = tl.exp2(
            tl.where(
                valid6,
                s0_6 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid7 = (cm7 < qlen) & (start + 0 + cn7 < klen)
        if CAUSAL:
            valid7 = valid7 & (start + 0 + cn7 <= cm7 + delta)
        p7 = tl.exp2(
            tl.where(
                valid7,
                s0_7 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        sum0 += p0 + p1 + p2 + p3
        sum1 += p4 + p5 + p6 + p7
        pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7 = common.mma_reorder(
            p0.to(tl.bfloat16),
            p1.to(tl.bfloat16),
            p2.to(tl.bfloat16),
            p3.to(tl.bfloat16),
            p4.to(tl.bfloat16),
            p5.to(tl.bfloat16),
            p6.to(tl.bfloat16),
            p7.to(tl.bfloat16),
            word,
        )
        pp0_0 = common.pack_bf(pa0, pa1)
        pp0_1 = common.pack_bf(pa2, pa3)
        pp0_2 = common.pack_bf(pa4, pa5)
        pp0_3 = common.pack_bf(pa6, pa7)
        valid0 = (cm0 < qlen) & (start + 16 + cn0 < klen)
        if CAUSAL:
            valid0 = valid0 & (start + 16 + cn0 <= cm0 + delta)
        p0 = tl.exp2(
            tl.where(
                valid0,
                s1_0 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid1 = (cm1 < qlen) & (start + 16 + cn1 < klen)
        if CAUSAL:
            valid1 = valid1 & (start + 16 + cn1 <= cm1 + delta)
        p1 = tl.exp2(
            tl.where(
                valid1,
                s1_1 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid2 = (cm2 < qlen) & (start + 16 + cn2 < klen)
        if CAUSAL:
            valid2 = valid2 & (start + 16 + cn2 <= cm2 + delta)
        p2 = tl.exp2(
            tl.where(
                valid2,
                s1_2 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid3 = (cm3 < qlen) & (start + 16 + cn3 < klen)
        if CAUSAL:
            valid3 = valid3 & (start + 16 + cn3 <= cm3 + delta)
        p3 = tl.exp2(
            tl.where(
                valid3,
                s1_3 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid4 = (cm4 < qlen) & (start + 16 + cn4 < klen)
        if CAUSAL:
            valid4 = valid4 & (start + 16 + cn4 <= cm4 + delta)
        p4 = tl.exp2(
            tl.where(
                valid4,
                s1_4 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid5 = (cm5 < qlen) & (start + 16 + cn5 < klen)
        if CAUSAL:
            valid5 = valid5 & (start + 16 + cn5 <= cm5 + delta)
        p5 = tl.exp2(
            tl.where(
                valid5,
                s1_5 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid6 = (cm6 < qlen) & (start + 16 + cn6 < klen)
        if CAUSAL:
            valid6 = valid6 & (start + 16 + cn6 <= cm6 + delta)
        p6 = tl.exp2(
            tl.where(
                valid6,
                s1_6 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid7 = (cm7 < qlen) & (start + 16 + cn7 < klen)
        if CAUSAL:
            valid7 = valid7 & (start + 16 + cn7 <= cm7 + delta)
        p7 = tl.exp2(
            tl.where(
                valid7,
                s1_7 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        sum0 += p0 + p1 + p2 + p3
        sum1 += p4 + p5 + p6 + p7
        pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7 = common.mma_reorder(
            p0.to(tl.bfloat16),
            p1.to(tl.bfloat16),
            p2.to(tl.bfloat16),
            p3.to(tl.bfloat16),
            p4.to(tl.bfloat16),
            p5.to(tl.bfloat16),
            p6.to(tl.bfloat16),
            p7.to(tl.bfloat16),
            word,
        )
        pp1_0 = common.pack_bf(pa0, pa1)
        pp1_1 = common.pack_bf(pa2, pa3)
        pp1_2 = common.pack_bf(pa4, pa5)
        pp1_3 = common.pack_bf(pa6, pa7)
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 0, token, True, ROW=slot * 32)
        o0_0, o0_1, o0_2, o0_3, o0_4, o0_5, o0_6, o0_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o0_0,
            o0_1,
            o0_2,
            o0_3,
            o0_4,
            o0_5,
            o0_6,
            o0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 0, token, True, ROW=slot * 32 + 16)
        o0_0, o0_1, o0_2, o0_3, o0_4, o0_5, o0_6, o0_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o0_0,
            o0_1,
            o0_2,
            o0_3,
            o0_4,
            o0_5,
            o0_6,
            o0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 16, token, True, ROW=slot * 32)
        o1_0, o1_1, o1_2, o1_3, o1_4, o1_5, o1_6, o1_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o1_0,
            o1_1,
            o1_2,
            o1_3,
            o1_4,
            o1_5,
            o1_6,
            o1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 16, token, True, ROW=slot * 32 + 16)
        o1_0, o1_1, o1_2, o1_3, o1_4, o1_5, o1_6, o1_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o1_0,
            o1_1,
            o1_2,
            o1_3,
            o1_4,
            o1_5,
            o1_6,
            o1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 32, token, True, ROW=slot * 32)
        o2_0, o2_1, o2_2, o2_3, o2_4, o2_5, o2_6, o2_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o2_0,
            o2_1,
            o2_2,
            o2_3,
            o2_4,
            o2_5,
            o2_6,
            o2_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 32, token, True, ROW=slot * 32 + 16)
        o2_0, o2_1, o2_2, o2_3, o2_4, o2_5, o2_6, o2_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o2_0,
            o2_1,
            o2_2,
            o2_3,
            o2_4,
            o2_5,
            o2_6,
            o2_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 48, token, True, ROW=slot * 32)
        o3_0, o3_1, o3_2, o3_3, o3_4, o3_5, o3_6, o3_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o3_0,
            o3_1,
            o3_2,
            o3_3,
            o3_4,
            o3_5,
            o3_6,
            o3_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 48, token, True, ROW=slot * 32 + 16)
        o3_0, o3_1, o3_2, o3_3, o3_4, o3_5, o3_6, o3_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o3_0,
            o3_1,
            o3_2,
            o3_3,
            o3_4,
            o3_5,
            o3_6,
            o3_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 64, token, True, ROW=slot * 32)
        o4_0, o4_1, o4_2, o4_3, o4_4, o4_5, o4_6, o4_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o4_0,
            o4_1,
            o4_2,
            o4_3,
            o4_4,
            o4_5,
            o4_6,
            o4_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 64, token, True, ROW=slot * 32 + 16)
        o4_0, o4_1, o4_2, o4_3, o4_4, o4_5, o4_6, o4_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o4_0,
            o4_1,
            o4_2,
            o4_3,
            o4_4,
            o4_5,
            o4_6,
            o4_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 80, token, True, ROW=slot * 32)
        o5_0, o5_1, o5_2, o5_3, o5_4, o5_5, o5_6, o5_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o5_0,
            o5_1,
            o5_2,
            o5_3,
            o5_4,
            o5_5,
            o5_6,
            o5_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 80, token, True, ROW=slot * 32 + 16)
        o5_0, o5_1, o5_2, o5_3, o5_4, o5_5, o5_6, o5_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o5_0,
            o5_1,
            o5_2,
            o5_3,
            o5_4,
            o5_5,
            o5_6,
            o5_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 96, token, True, ROW=slot * 32)
        o6_0, o6_1, o6_2, o6_3, o6_4, o6_5, o6_6, o6_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o6_0,
            o6_1,
            o6_2,
            o6_3,
            o6_4,
            o6_5,
            o6_6,
            o6_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 96, token, True, ROW=slot * 32 + 16)
        o6_0, o6_1, o6_2, o6_3, o6_4, o6_5, o6_6, o6_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o6_0,
            o6_1,
            o6_2,
            o6_3,
            o6_4,
            o6_5,
            o6_6,
            o6_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 112, token, True, ROW=slot * 32)
        o7_0, o7_1, o7_2, o7_3, o7_4, o7_5, o7_6, o7_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o7_0,
            o7_1,
            o7_2,
            o7_3,
            o7_4,
            o7_5,
            o7_6,
            o7_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 112, token, True, ROW=slot * 32 + 16)
        o7_0, o7_1, o7_2, o7_3, o7_4, o7_5, o7_6, o7_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o7_0,
            o7_1,
            o7_2,
            o7_3,
            o7_4,
            o7_5,
            o7_6,
            o7_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 128, token, True, ROW=slot * 32)
        o8_0, o8_1, o8_2, o8_3, o8_4, o8_5, o8_6, o8_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o8_0,
            o8_1,
            o8_2,
            o8_3,
            o8_4,
            o8_5,
            o8_6,
            o8_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 128, token, True, ROW=slot * 32 + 16)
        o8_0, o8_1, o8_2, o8_3, o8_4, o8_5, o8_6, o8_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o8_0,
            o8_1,
            o8_2,
            o8_3,
            o8_4,
            o8_5,
            o8_6,
            o8_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 144, token, True, ROW=slot * 32)
        o9_0, o9_1, o9_2, o9_3, o9_4, o9_5, o9_6, o9_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o9_0,
            o9_1,
            o9_2,
            o9_3,
            o9_4,
            o9_5,
            o9_6,
            o9_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 144, token, True, ROW=slot * 32 + 16)
        o9_0, o9_1, o9_2, o9_3, o9_4, o9_5, o9_6, o9_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o9_0,
            o9_1,
            o9_2,
            o9_3,
            o9_4,
            o9_5,
            o9_6,
            o9_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 160, token, True, ROW=slot * 32)
        o10_0, o10_1, o10_2, o10_3, o10_4, o10_5, o10_6, o10_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o10_0,
            o10_1,
            o10_2,
            o10_3,
            o10_4,
            o10_5,
            o10_6,
            o10_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 160, token, True, ROW=slot * 32 + 16)
        o10_0, o10_1, o10_2, o10_3, o10_4, o10_5, o10_6, o10_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o10_0,
            o10_1,
            o10_2,
            o10_3,
            o10_4,
            o10_5,
            o10_6,
            o10_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 176, token, True, ROW=slot * 32)
        o11_0, o11_1, o11_2, o11_3, o11_4, o11_5, o11_6, o11_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o11_0,
            o11_1,
            o11_2,
            o11_3,
            o11_4,
            o11_5,
            o11_6,
            o11_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 176, token, True, ROW=slot * 32 + 16)
        o11_0, o11_1, o11_2, o11_3, o11_4, o11_5, o11_6, o11_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o11_0,
            o11_1,
            o11_2,
            o11_3,
            o11_4,
            o11_5,
            o11_6,
            o11_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 192, token, True, ROW=slot * 32)
        o12_0, o12_1, o12_2, o12_3, o12_4, o12_5, o12_6, o12_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o12_0,
            o12_1,
            o12_2,
            o12_3,
            o12_4,
            o12_5,
            o12_6,
            o12_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 192, token, True, ROW=slot * 32 + 16)
        o12_0, o12_1, o12_2, o12_3, o12_4, o12_5, o12_6, o12_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o12_0,
            o12_1,
            o12_2,
            o12_3,
            o12_4,
            o12_5,
            o12_6,
            o12_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 208, token, True, ROW=slot * 32)
        o13_0, o13_1, o13_2, o13_3, o13_4, o13_5, o13_6, o13_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o13_0,
            o13_1,
            o13_2,
            o13_3,
            o13_4,
            o13_5,
            o13_6,
            o13_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 208, token, True, ROW=slot * 32 + 16)
        o13_0, o13_1, o13_2, o13_3, o13_4, o13_5, o13_6, o13_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o13_0,
            o13_1,
            o13_2,
            o13_3,
            o13_4,
            o13_5,
            o13_6,
            o13_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 224, token, True, ROW=slot * 32)
        o14_0, o14_1, o14_2, o14_3, o14_4, o14_5, o14_6, o14_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o14_0,
            o14_1,
            o14_2,
            o14_3,
            o14_4,
            o14_5,
            o14_6,
            o14_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 224, token, True, ROW=slot * 32 + 16)
        o14_0, o14_1, o14_2, o14_3, o14_4, o14_5, o14_6, o14_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o14_0,
            o14_1,
            o14_2,
            o14_3,
            o14_4,
            o14_5,
            o14_6,
            o14_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 240, token, True, ROW=slot * 32)
        o15_0, o15_1, o15_2, o15_3, o15_4, o15_5, o15_6, o15_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o15_0,
            o15_1,
            o15_2,
            o15_3,
            o15_4,
            o15_5,
            o15_6,
            o15_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 240, token, True, ROW=slot * 32 + 16)
        o15_0, o15_1, o15_2, o15_3, o15_4, o15_5, o15_6, o15_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o15_0,
            o15_1,
            o15_2,
            o15_3,
            o15_4,
            o15_5,
            o15_6,
            o15_7,
        )
        token = common.wait_copies() + tl.zeros((128,), tl.int32)
    den0 = tl.sum(sum0.reshape((4, 8, 4)), 2)
    den0 = tl.broadcast_to(den0[:, :, None], (4, 8, 4)).reshape((128,))
    inv0 = tl.div_rn(1.0, tl.where(den0 > 0.0, den0, 1.0))
    den1 = tl.sum(sum1.reshape((4, 8, 4)), 2)
    den1 = tl.broadcast_to(den1[:, :, None], (4, 8, 4)).reshape((128,))
    inv1 = tl.div_rn(1.0, tl.where(den1 > 0.0, den1, 1.0))
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn0,
        o0_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn1,
        o0_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn2,
        o0_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn3,
        o0_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn4,
        o0_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn5,
        o0_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn6,
        o0_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn7,
        o0_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn0,
        o1_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn1,
        o1_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn2,
        o1_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn3,
        o1_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn4,
        o1_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn5,
        o1_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn6,
        o1_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn7,
        o1_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn0,
        o2_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn1,
        o2_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn2,
        o2_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn3,
        o2_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn4,
        o2_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn5,
        o2_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn6,
        o2_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn7,
        o2_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn0,
        o3_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn1,
        o3_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn2,
        o3_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn3,
        o3_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn4,
        o3_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn5,
        o3_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn6,
        o3_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn7,
        o3_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn0,
        o4_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn1,
        o4_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn2,
        o4_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn3,
        o4_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn4,
        o4_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn5,
        o4_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn6,
        o4_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn7,
        o4_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn0,
        o5_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn1,
        o5_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn2,
        o5_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn3,
        o5_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn4,
        o5_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn5,
        o5_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn6,
        o5_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn7,
        o5_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn0,
        o6_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn1,
        o6_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn2,
        o6_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn3,
        o6_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn4,
        o6_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn5,
        o6_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn6,
        o6_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn7,
        o6_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn0,
        o7_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn1,
        o7_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn2,
        o7_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn3,
        o7_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn4,
        o7_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn5,
        o7_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn6,
        o7_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn7,
        o7_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn0,
        o8_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn1,
        o8_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn2,
        o8_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn3,
        o8_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn4,
        o8_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn5,
        o8_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn6,
        o8_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn7,
        o8_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn0,
        o9_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn1,
        o9_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn2,
        o9_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn3,
        o9_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn4,
        o9_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn5,
        o9_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn6,
        o9_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn7,
        o9_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn0,
        o10_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn1,
        o10_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn2,
        o10_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn3,
        o10_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn4,
        o10_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn5,
        o10_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn6,
        o10_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn7,
        o10_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn0,
        o11_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn1,
        o11_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn2,
        o11_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn3,
        o11_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn4,
        o11_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn5,
        o11_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn6,
        o11_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn7,
        o11_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn0,
        o12_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn1,
        o12_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn2,
        o12_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn3,
        o12_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn4,
        o12_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn5,
        o12_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn6,
        o12_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn7,
        o12_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn0,
        o13_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn1,
        o13_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn2,
        o13_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn3,
        o13_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn4,
        o13_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn5,
        o13_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn6,
        o13_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn7,
        o13_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn0,
        o14_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn1,
        o14_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn2,
        o14_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn3,
        o14_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn4,
        o14_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn5,
        o14_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn6,
        o14_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn7,
        o14_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn0,
        o15_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn1,
        o15_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn2,
        o15_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn3,
        o15_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn4,
        o15_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn5,
        o15_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn6,
        o15_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn7,
        o15_7 * inv1,
        cm7 < qlen,
    )
    if WRITE_LSE:
        lse0 = tl.where(
            den0 > 0.0, 32.0 + tl.log2(den0) * 0.6931471805599453, float("inf")
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + cm0).to(tl.int64),
            lse0,
            (cm0 < qlen) & (thread % 4 == 0),
        )
        lse1 = tl.where(
            den1 > 0.0, 32.0 + tl.log2(den1) * 0.6931471805599453, float("inf")
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + cm4).to(tl.int64),
            lse1,
            (cm4 < qlen) & (thread % 4 == 0),
        )


@triton.jit
def async_small_prefill_attention_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    CUQ,
    CUK,
    USEDK,
    TABLE,
    ALIBI,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KP: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    VP: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    OS: tl.constexpr,
    OH: tl.constexpr,
    TS: tl.constexpr,
    AS: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    SOFTCAP: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr = False,
    SHORT_LIMIT: tl.constexpr = 0,
    ORDER: tl.constexpr = 0,
    WRITE_LSE: tl.constexpr = True,
    BATCHES: tl.constexpr = 1,
    FOLD: tl.constexpr = 4,
    KNORM=None,
    WORKLIST=None,
    CAPACITY: tl.constexpr = 0,
    FIXED_MODE: tl.constexpr = False,
):
    tl.static_assert(D == 256 and BM == 64 and (FOLD == 4) and PAGED)
    tl.static_assert(Q.dtype.element_ty == tl.bfloat16)
    tl.static_assert(QH >= D and QS >= (HQ - 1) * QH + D)
    tl.static_assert(KP == PAGE * KS and VP == PAGE * VS)
    index = tl.program_id(0)
    group = tl.program_id(2)
    if index >= tl.load(WORKLIST + 5 * CAPACITY):
        return
    qstart = tl.load(WORKLIST + index)
    qlen = tl.load(WORKLIST + CAPACITY + index)
    klen = tl.load(WORKLIST + 2 * CAPACITY + index)
    batch = tl.load(WORKLIST + 3 * CAPACITY + index)
    tile = tl.load(WORKLIST + 4 * CAPACITY + index)
    if SHORT_LIMIT and qlen <= SHORT_LIMIT and (klen >= 8192):
        return
    kh = group // (HQ // HK // 4)
    if tl.load(KNORM + batch * HK + kh) == 0:
        return
    if REVERSE:
        tile = tl.cdiv(qlen, 16) - 1 - tile
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    p = lane // 4
    q = lane % 4
    qhead = group * 4 + warp
    word = tl.where(p % 4 == q, tl.where(lane >= 16, 1065353216, 16256), 0).to(
        tl.uint32
    )
    cm0 = tile * 16 + p + 0
    cn0 = q + 0
    cm1 = tile * 16 + p + 0
    cn1 = q + 4
    cm2 = tile * 16 + p + 0
    cn2 = q + 8
    cm3 = tile * 16 + p + 0
    cn3 = q + 12
    cm4 = tile * 16 + p + 8
    cn4 = q + 0
    cm5 = tile * 16 + p + 8
    cn5 = q + 4
    cm6 = tile * 16 + p + 8
    cn6 = q + 8
    cm7 = tile * 16 + p + 8
    cn7 = q + 12
    kbuf = gpu.alloc(
        (256, 64),
        K.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    vbuf = gpu.alloc(
        (256, 64),
        V.dtype.element_ty,
        layout=common.cube_layout(False),
        nv_mma_shared_layout=False,
    )
    (
        qp0_0,
        qp0_1,
        qp0_2,
        qp0_3,
        qp1_0,
        qp1_1,
        qp1_2,
        qp1_3,
        qp2_0,
        qp2_1,
        qp2_2,
        qp2_3,
        qp3_0,
        qp3_1,
        qp3_2,
        qp3_3,
        qp4_0,
        qp4_1,
        qp4_2,
        qp4_3,
        qp5_0,
        qp5_1,
        qp5_2,
        qp5_3,
        qp6_0,
        qp6_1,
        qp6_2,
        qp6_3,
        qp7_0,
        qp7_1,
        qp7_2,
        qp7_3,
        qp8_0,
        qp8_1,
        qp8_2,
        qp8_3,
        qp9_0,
        qp9_1,
        qp9_2,
        qp9_3,
        qp10_0,
        qp10_1,
        qp10_2,
        qp10_3,
        qp11_0,
        qp11_1,
        qp11_2,
        qp11_3,
        qp12_0,
        qp12_1,
        qp12_2,
        qp12_3,
        qp13_0,
        qp13_1,
        qp13_2,
        qp13_3,
        qp14_0,
        qp14_1,
        qp14_2,
        qp14_3,
        qp15_0,
        qp15_1,
        qp15_2,
        qp15_3,
    ) = common.load_q(Q, kbuf, qstart, qlen, tile, group, QS, QH)
    o0_0 = tl.full((128,), 0.0, tl.float32)
    o0_1 = tl.full((128,), 0.0, tl.float32)
    o0_2 = tl.full((128,), 0.0, tl.float32)
    o0_3 = tl.full((128,), 0.0, tl.float32)
    o0_4 = tl.full((128,), 0.0, tl.float32)
    o0_5 = tl.full((128,), 0.0, tl.float32)
    o0_6 = tl.full((128,), 0.0, tl.float32)
    o0_7 = tl.full((128,), 0.0, tl.float32)
    o1_0 = tl.full((128,), 0.0, tl.float32)
    o1_1 = tl.full((128,), 0.0, tl.float32)
    o1_2 = tl.full((128,), 0.0, tl.float32)
    o1_3 = tl.full((128,), 0.0, tl.float32)
    o1_4 = tl.full((128,), 0.0, tl.float32)
    o1_5 = tl.full((128,), 0.0, tl.float32)
    o1_6 = tl.full((128,), 0.0, tl.float32)
    o1_7 = tl.full((128,), 0.0, tl.float32)
    o2_0 = tl.full((128,), 0.0, tl.float32)
    o2_1 = tl.full((128,), 0.0, tl.float32)
    o2_2 = tl.full((128,), 0.0, tl.float32)
    o2_3 = tl.full((128,), 0.0, tl.float32)
    o2_4 = tl.full((128,), 0.0, tl.float32)
    o2_5 = tl.full((128,), 0.0, tl.float32)
    o2_6 = tl.full((128,), 0.0, tl.float32)
    o2_7 = tl.full((128,), 0.0, tl.float32)
    o3_0 = tl.full((128,), 0.0, tl.float32)
    o3_1 = tl.full((128,), 0.0, tl.float32)
    o3_2 = tl.full((128,), 0.0, tl.float32)
    o3_3 = tl.full((128,), 0.0, tl.float32)
    o3_4 = tl.full((128,), 0.0, tl.float32)
    o3_5 = tl.full((128,), 0.0, tl.float32)
    o3_6 = tl.full((128,), 0.0, tl.float32)
    o3_7 = tl.full((128,), 0.0, tl.float32)
    o4_0 = tl.full((128,), 0.0, tl.float32)
    o4_1 = tl.full((128,), 0.0, tl.float32)
    o4_2 = tl.full((128,), 0.0, tl.float32)
    o4_3 = tl.full((128,), 0.0, tl.float32)
    o4_4 = tl.full((128,), 0.0, tl.float32)
    o4_5 = tl.full((128,), 0.0, tl.float32)
    o4_6 = tl.full((128,), 0.0, tl.float32)
    o4_7 = tl.full((128,), 0.0, tl.float32)
    o5_0 = tl.full((128,), 0.0, tl.float32)
    o5_1 = tl.full((128,), 0.0, tl.float32)
    o5_2 = tl.full((128,), 0.0, tl.float32)
    o5_3 = tl.full((128,), 0.0, tl.float32)
    o5_4 = tl.full((128,), 0.0, tl.float32)
    o5_5 = tl.full((128,), 0.0, tl.float32)
    o5_6 = tl.full((128,), 0.0, tl.float32)
    o5_7 = tl.full((128,), 0.0, tl.float32)
    o6_0 = tl.full((128,), 0.0, tl.float32)
    o6_1 = tl.full((128,), 0.0, tl.float32)
    o6_2 = tl.full((128,), 0.0, tl.float32)
    o6_3 = tl.full((128,), 0.0, tl.float32)
    o6_4 = tl.full((128,), 0.0, tl.float32)
    o6_5 = tl.full((128,), 0.0, tl.float32)
    o6_6 = tl.full((128,), 0.0, tl.float32)
    o6_7 = tl.full((128,), 0.0, tl.float32)
    o7_0 = tl.full((128,), 0.0, tl.float32)
    o7_1 = tl.full((128,), 0.0, tl.float32)
    o7_2 = tl.full((128,), 0.0, tl.float32)
    o7_3 = tl.full((128,), 0.0, tl.float32)
    o7_4 = tl.full((128,), 0.0, tl.float32)
    o7_5 = tl.full((128,), 0.0, tl.float32)
    o7_6 = tl.full((128,), 0.0, tl.float32)
    o7_7 = tl.full((128,), 0.0, tl.float32)
    o8_0 = tl.full((128,), 0.0, tl.float32)
    o8_1 = tl.full((128,), 0.0, tl.float32)
    o8_2 = tl.full((128,), 0.0, tl.float32)
    o8_3 = tl.full((128,), 0.0, tl.float32)
    o8_4 = tl.full((128,), 0.0, tl.float32)
    o8_5 = tl.full((128,), 0.0, tl.float32)
    o8_6 = tl.full((128,), 0.0, tl.float32)
    o8_7 = tl.full((128,), 0.0, tl.float32)
    o9_0 = tl.full((128,), 0.0, tl.float32)
    o9_1 = tl.full((128,), 0.0, tl.float32)
    o9_2 = tl.full((128,), 0.0, tl.float32)
    o9_3 = tl.full((128,), 0.0, tl.float32)
    o9_4 = tl.full((128,), 0.0, tl.float32)
    o9_5 = tl.full((128,), 0.0, tl.float32)
    o9_6 = tl.full((128,), 0.0, tl.float32)
    o9_7 = tl.full((128,), 0.0, tl.float32)
    o10_0 = tl.full((128,), 0.0, tl.float32)
    o10_1 = tl.full((128,), 0.0, tl.float32)
    o10_2 = tl.full((128,), 0.0, tl.float32)
    o10_3 = tl.full((128,), 0.0, tl.float32)
    o10_4 = tl.full((128,), 0.0, tl.float32)
    o10_5 = tl.full((128,), 0.0, tl.float32)
    o10_6 = tl.full((128,), 0.0, tl.float32)
    o10_7 = tl.full((128,), 0.0, tl.float32)
    o11_0 = tl.full((128,), 0.0, tl.float32)
    o11_1 = tl.full((128,), 0.0, tl.float32)
    o11_2 = tl.full((128,), 0.0, tl.float32)
    o11_3 = tl.full((128,), 0.0, tl.float32)
    o11_4 = tl.full((128,), 0.0, tl.float32)
    o11_5 = tl.full((128,), 0.0, tl.float32)
    o11_6 = tl.full((128,), 0.0, tl.float32)
    o11_7 = tl.full((128,), 0.0, tl.float32)
    o12_0 = tl.full((128,), 0.0, tl.float32)
    o12_1 = tl.full((128,), 0.0, tl.float32)
    o12_2 = tl.full((128,), 0.0, tl.float32)
    o12_3 = tl.full((128,), 0.0, tl.float32)
    o12_4 = tl.full((128,), 0.0, tl.float32)
    o12_5 = tl.full((128,), 0.0, tl.float32)
    o12_6 = tl.full((128,), 0.0, tl.float32)
    o12_7 = tl.full((128,), 0.0, tl.float32)
    o13_0 = tl.full((128,), 0.0, tl.float32)
    o13_1 = tl.full((128,), 0.0, tl.float32)
    o13_2 = tl.full((128,), 0.0, tl.float32)
    o13_3 = tl.full((128,), 0.0, tl.float32)
    o13_4 = tl.full((128,), 0.0, tl.float32)
    o13_5 = tl.full((128,), 0.0, tl.float32)
    o13_6 = tl.full((128,), 0.0, tl.float32)
    o13_7 = tl.full((128,), 0.0, tl.float32)
    o14_0 = tl.full((128,), 0.0, tl.float32)
    o14_1 = tl.full((128,), 0.0, tl.float32)
    o14_2 = tl.full((128,), 0.0, tl.float32)
    o14_3 = tl.full((128,), 0.0, tl.float32)
    o14_4 = tl.full((128,), 0.0, tl.float32)
    o14_5 = tl.full((128,), 0.0, tl.float32)
    o14_6 = tl.full((128,), 0.0, tl.float32)
    o14_7 = tl.full((128,), 0.0, tl.float32)
    o15_0 = tl.full((128,), 0.0, tl.float32)
    o15_1 = tl.full((128,), 0.0, tl.float32)
    o15_2 = tl.full((128,), 0.0, tl.float32)
    o15_3 = tl.full((128,), 0.0, tl.float32)
    o15_4 = tl.full((128,), 0.0, tl.float32)
    o15_5 = tl.full((128,), 0.0, tl.float32)
    o15_6 = tl.full((128,), 0.0, tl.float32)
    o15_7 = tl.full((128,), 0.0, tl.float32)
    sum0 = tl.full((128,), 0.0, tl.float32)
    sum1 = tl.full((128,), 0.0, tl.float32)
    hi = klen
    delta = klen - qlen
    if CAUSAL:
        hi = tl.minimum(klen, tl.maximum(0, (tile + 1) * 16 + delta))
    base = batch.to(tl.int64) * (TS * PAGE)
    if hi > 0:
        common.issue_copy(K + base * KS, kbuf, 0, klen, KS, 0, kh * KH)
        common.issue_copy(V + base * VS, vbuf, 0, klen, VS, 0, kh * VH)
        common.commit_copies()
    token = common.wait_copies() + tl.zeros((128,), tl.int32)
    for start in range(0, hi, 32):
        slot = start // 32 % 2
        if start + 32 < hi:
            common.issue_copy(
                K + base * KS, kbuf, start + 32, klen, KS, (1 - slot) * 32, kh * KH
            )
            common.issue_copy(
                V + base * VS, vbuf, start + 32, klen, VS, (1 - slot) * 32, kh * VH
            )
            common.commit_copies()
        s0_0 = tl.full((128,), 0.0, tl.float32)
        s0_1 = tl.full((128,), 0.0, tl.float32)
        s0_2 = tl.full((128,), 0.0, tl.float32)
        s0_3 = tl.full((128,), 0.0, tl.float32)
        s0_4 = tl.full((128,), 0.0, tl.float32)
        s0_5 = tl.full((128,), 0.0, tl.float32)
        s0_6 = tl.full((128,), 0.0, tl.float32)
        s0_7 = tl.full((128,), 0.0, tl.float32)
        s1_0 = tl.full((128,), 0.0, tl.float32)
        s1_1 = tl.full((128,), 0.0, tl.float32)
        s1_2 = tl.full((128,), 0.0, tl.float32)
        s1_3 = tl.full((128,), 0.0, tl.float32)
        s1_4 = tl.full((128,), 0.0, tl.float32)
        s1_5 = tl.full((128,), 0.0, tl.float32)
        s1_6 = tl.full((128,), 0.0, tl.float32)
        s1_7 = tl.full((128,), 0.0, tl.float32)
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 0, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp0_0,
            qp0_1,
            qp0_2,
            qp0_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 0, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp0_0,
            qp0_1,
            qp0_2,
            qp0_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 16, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp1_0,
            qp1_1,
            qp1_2,
            qp1_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 16, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp1_0,
            qp1_1,
            qp1_2,
            qp1_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 32, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp2_0,
            qp2_1,
            qp2_2,
            qp2_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 32, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp2_0,
            qp2_1,
            qp2_2,
            qp2_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 48, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp3_0,
            qp3_1,
            qp3_2,
            qp3_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 48, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp3_0,
            qp3_1,
            qp3_2,
            qp3_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 64, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp4_0,
            qp4_1,
            qp4_2,
            qp4_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 64, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp4_0,
            qp4_1,
            qp4_2,
            qp4_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 80, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp5_0,
            qp5_1,
            qp5_2,
            qp5_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 80, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp5_0,
            qp5_1,
            qp5_2,
            qp5_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 96, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp6_0,
            qp6_1,
            qp6_2,
            qp6_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 96, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp6_0,
            qp6_1,
            qp6_2,
            qp6_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 112, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp7_0,
            qp7_1,
            qp7_2,
            qp7_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 112, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp7_0,
            qp7_1,
            qp7_2,
            qp7_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 128, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp8_0,
            qp8_1,
            qp8_2,
            qp8_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 128, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp8_0,
            qp8_1,
            qp8_2,
            qp8_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 144, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp9_0,
            qp9_1,
            qp9_2,
            qp9_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 144, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp9_0,
            qp9_1,
            qp9_2,
            qp9_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 160, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp10_0,
            qp10_1,
            qp10_2,
            qp10_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 160, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp10_0,
            qp10_1,
            qp10_2,
            qp10_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 176, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp11_0,
            qp11_1,
            qp11_2,
            qp11_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 176, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp11_0,
            qp11_1,
            qp11_2,
            qp11_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 192, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp12_0,
            qp12_1,
            qp12_2,
            qp12_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 192, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp12_0,
            qp12_1,
            qp12_2,
            qp12_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 208, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp13_0,
            qp13_1,
            qp13_2,
            qp13_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 208, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp13_0,
            qp13_1,
            qp13_2,
            qp13_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 224, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp14_0,
            qp14_1,
            qp14_2,
            qp14_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 224, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp14_0,
            qp14_1,
            qp14_2,
            qp14_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 240, token, False, ROW=slot * 32)
        s0_0, s0_1, s0_2, s0_3, s0_4, s0_5, s0_6, s0_7 = common.mma_packed(
            qp15_0,
            qp15_1,
            qp15_2,
            qp15_3,
            b0,
            b1,
            b2,
            b3,
            s0_0,
            s0_1,
            s0_2,
            s0_3,
            s0_4,
            s0_5,
            s0_6,
            s0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(kbuf, 240, token, False, ROW=slot * 32 + 16)
        s1_0, s1_1, s1_2, s1_3, s1_4, s1_5, s1_6, s1_7 = common.mma_packed(
            qp15_0,
            qp15_1,
            qp15_2,
            qp15_3,
            b0,
            b1,
            b2,
            b3,
            s1_0,
            s1_1,
            s1_2,
            s1_3,
            s1_4,
            s1_5,
            s1_6,
            s1_7,
        )
        valid0 = (cm0 < qlen) & (start + 0 + cn0 < klen)
        if CAUSAL:
            valid0 = valid0 & (start + 0 + cn0 <= cm0 + delta)
        p0 = tl.exp2(
            tl.where(
                valid0,
                s0_0 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid1 = (cm1 < qlen) & (start + 0 + cn1 < klen)
        if CAUSAL:
            valid1 = valid1 & (start + 0 + cn1 <= cm1 + delta)
        p1 = tl.exp2(
            tl.where(
                valid1,
                s0_1 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid2 = (cm2 < qlen) & (start + 0 + cn2 < klen)
        if CAUSAL:
            valid2 = valid2 & (start + 0 + cn2 <= cm2 + delta)
        p2 = tl.exp2(
            tl.where(
                valid2,
                s0_2 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid3 = (cm3 < qlen) & (start + 0 + cn3 < klen)
        if CAUSAL:
            valid3 = valid3 & (start + 0 + cn3 <= cm3 + delta)
        p3 = tl.exp2(
            tl.where(
                valid3,
                s0_3 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid4 = (cm4 < qlen) & (start + 0 + cn4 < klen)
        if CAUSAL:
            valid4 = valid4 & (start + 0 + cn4 <= cm4 + delta)
        p4 = tl.exp2(
            tl.where(
                valid4,
                s0_4 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid5 = (cm5 < qlen) & (start + 0 + cn5 < klen)
        if CAUSAL:
            valid5 = valid5 & (start + 0 + cn5 <= cm5 + delta)
        p5 = tl.exp2(
            tl.where(
                valid5,
                s0_5 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid6 = (cm6 < qlen) & (start + 0 + cn6 < klen)
        if CAUSAL:
            valid6 = valid6 & (start + 0 + cn6 <= cm6 + delta)
        p6 = tl.exp2(
            tl.where(
                valid6,
                s0_6 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid7 = (cm7 < qlen) & (start + 0 + cn7 < klen)
        if CAUSAL:
            valid7 = valid7 & (start + 0 + cn7 <= cm7 + delta)
        p7 = tl.exp2(
            tl.where(
                valid7,
                s0_7 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        sum0 += p0 + p1 + p2 + p3
        sum1 += p4 + p5 + p6 + p7
        pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7 = common.mma_reorder(
            p0.to(tl.bfloat16),
            p1.to(tl.bfloat16),
            p2.to(tl.bfloat16),
            p3.to(tl.bfloat16),
            p4.to(tl.bfloat16),
            p5.to(tl.bfloat16),
            p6.to(tl.bfloat16),
            p7.to(tl.bfloat16),
            word,
        )
        pp0_0 = common.pack_bf(pa0, pa1)
        pp0_1 = common.pack_bf(pa2, pa3)
        pp0_2 = common.pack_bf(pa4, pa5)
        pp0_3 = common.pack_bf(pa6, pa7)
        valid0 = (cm0 < qlen) & (start + 16 + cn0 < klen)
        if CAUSAL:
            valid0 = valid0 & (start + 16 + cn0 <= cm0 + delta)
        p0 = tl.exp2(
            tl.where(
                valid0,
                s1_0 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid1 = (cm1 < qlen) & (start + 16 + cn1 < klen)
        if CAUSAL:
            valid1 = valid1 & (start + 16 + cn1 <= cm1 + delta)
        p1 = tl.exp2(
            tl.where(
                valid1,
                s1_1 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid2 = (cm2 < qlen) & (start + 16 + cn2 < klen)
        if CAUSAL:
            valid2 = valid2 & (start + 16 + cn2 <= cm2 + delta)
        p2 = tl.exp2(
            tl.where(
                valid2,
                s1_2 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid3 = (cm3 < qlen) & (start + 16 + cn3 < klen)
        if CAUSAL:
            valid3 = valid3 & (start + 16 + cn3 <= cm3 + delta)
        p3 = tl.exp2(
            tl.where(
                valid3,
                s1_3 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid4 = (cm4 < qlen) & (start + 16 + cn4 < klen)
        if CAUSAL:
            valid4 = valid4 & (start + 16 + cn4 <= cm4 + delta)
        p4 = tl.exp2(
            tl.where(
                valid4,
                s1_4 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid5 = (cm5 < qlen) & (start + 16 + cn5 < klen)
        if CAUSAL:
            valid5 = valid5 & (start + 16 + cn5 <= cm5 + delta)
        p5 = tl.exp2(
            tl.where(
                valid5,
                s1_5 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid6 = (cm6 < qlen) & (start + 16 + cn6 < klen)
        if CAUSAL:
            valid6 = valid6 & (start + 16 + cn6 <= cm6 + delta)
        p6 = tl.exp2(
            tl.where(
                valid6,
                s1_6 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        valid7 = (cm7 < qlen) & (start + 16 + cn7 < klen)
        if CAUSAL:
            valid7 = valid7 & (start + 16 + cn7 <= cm7 + delta)
        p7 = tl.exp2(
            tl.where(
                valid7,
                s1_7 * (SCALE * 1.4426950408889634) - 32.0 * 1.4426950408889634,
                -float("inf"),
            )
        )
        sum0 += p0 + p1 + p2 + p3
        sum1 += p4 + p5 + p6 + p7
        pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7 = common.mma_reorder(
            p0.to(tl.bfloat16),
            p1.to(tl.bfloat16),
            p2.to(tl.bfloat16),
            p3.to(tl.bfloat16),
            p4.to(tl.bfloat16),
            p5.to(tl.bfloat16),
            p6.to(tl.bfloat16),
            p7.to(tl.bfloat16),
            word,
        )
        pp1_0 = common.pack_bf(pa0, pa1)
        pp1_1 = common.pack_bf(pa2, pa3)
        pp1_2 = common.pack_bf(pa4, pa5)
        pp1_3 = common.pack_bf(pa6, pa7)
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 0, token, True, ROW=slot * 32)
        o0_0, o0_1, o0_2, o0_3, o0_4, o0_5, o0_6, o0_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o0_0,
            o0_1,
            o0_2,
            o0_3,
            o0_4,
            o0_5,
            o0_6,
            o0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 0, token, True, ROW=slot * 32 + 16)
        o0_0, o0_1, o0_2, o0_3, o0_4, o0_5, o0_6, o0_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o0_0,
            o0_1,
            o0_2,
            o0_3,
            o0_4,
            o0_5,
            o0_6,
            o0_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 16, token, True, ROW=slot * 32)
        o1_0, o1_1, o1_2, o1_3, o1_4, o1_5, o1_6, o1_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o1_0,
            o1_1,
            o1_2,
            o1_3,
            o1_4,
            o1_5,
            o1_6,
            o1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 16, token, True, ROW=slot * 32 + 16)
        o1_0, o1_1, o1_2, o1_3, o1_4, o1_5, o1_6, o1_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o1_0,
            o1_1,
            o1_2,
            o1_3,
            o1_4,
            o1_5,
            o1_6,
            o1_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 32, token, True, ROW=slot * 32)
        o2_0, o2_1, o2_2, o2_3, o2_4, o2_5, o2_6, o2_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o2_0,
            o2_1,
            o2_2,
            o2_3,
            o2_4,
            o2_5,
            o2_6,
            o2_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 32, token, True, ROW=slot * 32 + 16)
        o2_0, o2_1, o2_2, o2_3, o2_4, o2_5, o2_6, o2_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o2_0,
            o2_1,
            o2_2,
            o2_3,
            o2_4,
            o2_5,
            o2_6,
            o2_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 48, token, True, ROW=slot * 32)
        o3_0, o3_1, o3_2, o3_3, o3_4, o3_5, o3_6, o3_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o3_0,
            o3_1,
            o3_2,
            o3_3,
            o3_4,
            o3_5,
            o3_6,
            o3_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 48, token, True, ROW=slot * 32 + 16)
        o3_0, o3_1, o3_2, o3_3, o3_4, o3_5, o3_6, o3_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o3_0,
            o3_1,
            o3_2,
            o3_3,
            o3_4,
            o3_5,
            o3_6,
            o3_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 64, token, True, ROW=slot * 32)
        o4_0, o4_1, o4_2, o4_3, o4_4, o4_5, o4_6, o4_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o4_0,
            o4_1,
            o4_2,
            o4_3,
            o4_4,
            o4_5,
            o4_6,
            o4_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 64, token, True, ROW=slot * 32 + 16)
        o4_0, o4_1, o4_2, o4_3, o4_4, o4_5, o4_6, o4_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o4_0,
            o4_1,
            o4_2,
            o4_3,
            o4_4,
            o4_5,
            o4_6,
            o4_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 80, token, True, ROW=slot * 32)
        o5_0, o5_1, o5_2, o5_3, o5_4, o5_5, o5_6, o5_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o5_0,
            o5_1,
            o5_2,
            o5_3,
            o5_4,
            o5_5,
            o5_6,
            o5_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 80, token, True, ROW=slot * 32 + 16)
        o5_0, o5_1, o5_2, o5_3, o5_4, o5_5, o5_6, o5_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o5_0,
            o5_1,
            o5_2,
            o5_3,
            o5_4,
            o5_5,
            o5_6,
            o5_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 96, token, True, ROW=slot * 32)
        o6_0, o6_1, o6_2, o6_3, o6_4, o6_5, o6_6, o6_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o6_0,
            o6_1,
            o6_2,
            o6_3,
            o6_4,
            o6_5,
            o6_6,
            o6_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 96, token, True, ROW=slot * 32 + 16)
        o6_0, o6_1, o6_2, o6_3, o6_4, o6_5, o6_6, o6_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o6_0,
            o6_1,
            o6_2,
            o6_3,
            o6_4,
            o6_5,
            o6_6,
            o6_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 112, token, True, ROW=slot * 32)
        o7_0, o7_1, o7_2, o7_3, o7_4, o7_5, o7_6, o7_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o7_0,
            o7_1,
            o7_2,
            o7_3,
            o7_4,
            o7_5,
            o7_6,
            o7_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 112, token, True, ROW=slot * 32 + 16)
        o7_0, o7_1, o7_2, o7_3, o7_4, o7_5, o7_6, o7_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o7_0,
            o7_1,
            o7_2,
            o7_3,
            o7_4,
            o7_5,
            o7_6,
            o7_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 128, token, True, ROW=slot * 32)
        o8_0, o8_1, o8_2, o8_3, o8_4, o8_5, o8_6, o8_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o8_0,
            o8_1,
            o8_2,
            o8_3,
            o8_4,
            o8_5,
            o8_6,
            o8_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 128, token, True, ROW=slot * 32 + 16)
        o8_0, o8_1, o8_2, o8_3, o8_4, o8_5, o8_6, o8_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o8_0,
            o8_1,
            o8_2,
            o8_3,
            o8_4,
            o8_5,
            o8_6,
            o8_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 144, token, True, ROW=slot * 32)
        o9_0, o9_1, o9_2, o9_3, o9_4, o9_5, o9_6, o9_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o9_0,
            o9_1,
            o9_2,
            o9_3,
            o9_4,
            o9_5,
            o9_6,
            o9_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 144, token, True, ROW=slot * 32 + 16)
        o9_0, o9_1, o9_2, o9_3, o9_4, o9_5, o9_6, o9_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o9_0,
            o9_1,
            o9_2,
            o9_3,
            o9_4,
            o9_5,
            o9_6,
            o9_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 160, token, True, ROW=slot * 32)
        o10_0, o10_1, o10_2, o10_3, o10_4, o10_5, o10_6, o10_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o10_0,
            o10_1,
            o10_2,
            o10_3,
            o10_4,
            o10_5,
            o10_6,
            o10_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 160, token, True, ROW=slot * 32 + 16)
        o10_0, o10_1, o10_2, o10_3, o10_4, o10_5, o10_6, o10_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o10_0,
            o10_1,
            o10_2,
            o10_3,
            o10_4,
            o10_5,
            o10_6,
            o10_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 176, token, True, ROW=slot * 32)
        o11_0, o11_1, o11_2, o11_3, o11_4, o11_5, o11_6, o11_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o11_0,
            o11_1,
            o11_2,
            o11_3,
            o11_4,
            o11_5,
            o11_6,
            o11_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 176, token, True, ROW=slot * 32 + 16)
        o11_0, o11_1, o11_2, o11_3, o11_4, o11_5, o11_6, o11_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o11_0,
            o11_1,
            o11_2,
            o11_3,
            o11_4,
            o11_5,
            o11_6,
            o11_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 192, token, True, ROW=slot * 32)
        o12_0, o12_1, o12_2, o12_3, o12_4, o12_5, o12_6, o12_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o12_0,
            o12_1,
            o12_2,
            o12_3,
            o12_4,
            o12_5,
            o12_6,
            o12_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 192, token, True, ROW=slot * 32 + 16)
        o12_0, o12_1, o12_2, o12_3, o12_4, o12_5, o12_6, o12_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o12_0,
            o12_1,
            o12_2,
            o12_3,
            o12_4,
            o12_5,
            o12_6,
            o12_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 208, token, True, ROW=slot * 32)
        o13_0, o13_1, o13_2, o13_3, o13_4, o13_5, o13_6, o13_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o13_0,
            o13_1,
            o13_2,
            o13_3,
            o13_4,
            o13_5,
            o13_6,
            o13_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 208, token, True, ROW=slot * 32 + 16)
        o13_0, o13_1, o13_2, o13_3, o13_4, o13_5, o13_6, o13_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o13_0,
            o13_1,
            o13_2,
            o13_3,
            o13_4,
            o13_5,
            o13_6,
            o13_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 224, token, True, ROW=slot * 32)
        o14_0, o14_1, o14_2, o14_3, o14_4, o14_5, o14_6, o14_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o14_0,
            o14_1,
            o14_2,
            o14_3,
            o14_4,
            o14_5,
            o14_6,
            o14_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 224, token, True, ROW=slot * 32 + 16)
        o14_0, o14_1, o14_2, o14_3, o14_4, o14_5, o14_6, o14_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o14_0,
            o14_1,
            o14_2,
            o14_3,
            o14_4,
            o14_5,
            o14_6,
            o14_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 240, token, True, ROW=slot * 32)
        o15_0, o15_1, o15_2, o15_3, o15_4, o15_5, o15_6, o15_7 = common.mma_packed(
            pp0_0,
            pp0_1,
            pp0_2,
            pp0_3,
            b0,
            b1,
            b2,
            b3,
            o15_0,
            o15_1,
            o15_2,
            o15_3,
            o15_4,
            o15_5,
            o15_6,
            o15_7,
        )
        b0, b1, b2, b3 = common.ld_fragment(vbuf, 240, token, True, ROW=slot * 32 + 16)
        o15_0, o15_1, o15_2, o15_3, o15_4, o15_5, o15_6, o15_7 = common.mma_packed(
            pp1_0,
            pp1_1,
            pp1_2,
            pp1_3,
            b0,
            b1,
            b2,
            b3,
            o15_0,
            o15_1,
            o15_2,
            o15_3,
            o15_4,
            o15_5,
            o15_6,
            o15_7,
        )
        token = common.wait_copies() + tl.zeros((128,), tl.int32)
    den0 = tl.sum(sum0.reshape((4, 8, 4)), 2)
    den0 = tl.broadcast_to(den0[:, :, None], (4, 8, 4)).reshape((128,))
    inv0 = tl.div_rn(1.0, tl.where(den0 > 0.0, den0, 1.0))
    den1 = tl.sum(sum1.reshape((4, 8, 4)), 2)
    den1 = tl.broadcast_to(den1[:, :, None], (4, 8, 4)).reshape((128,))
    inv1 = tl.div_rn(1.0, tl.where(den1 > 0.0, den1, 1.0))
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn0,
        o0_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn1,
        o0_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn2,
        o0_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn3,
        o0_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn4,
        o0_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn5,
        o0_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn6,
        o0_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 0 + cn7,
        o0_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn0,
        o1_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn1,
        o1_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn2,
        o1_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn3,
        o1_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn4,
        o1_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn5,
        o1_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn6,
        o1_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 16 + cn7,
        o1_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn0,
        o2_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn1,
        o2_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn2,
        o2_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn3,
        o2_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn4,
        o2_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn5,
        o2_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn6,
        o2_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 32 + cn7,
        o2_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn0,
        o3_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn1,
        o3_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn2,
        o3_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn3,
        o3_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn4,
        o3_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn5,
        o3_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn6,
        o3_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 48 + cn7,
        o3_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn0,
        o4_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn1,
        o4_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn2,
        o4_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn3,
        o4_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn4,
        o4_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn5,
        o4_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn6,
        o4_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 64 + cn7,
        o4_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn0,
        o5_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn1,
        o5_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn2,
        o5_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn3,
        o5_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn4,
        o5_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn5,
        o5_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn6,
        o5_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 80 + cn7,
        o5_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn0,
        o6_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn1,
        o6_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn2,
        o6_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn3,
        o6_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn4,
        o6_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn5,
        o6_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn6,
        o6_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 96 + cn7,
        o6_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn0,
        o7_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn1,
        o7_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn2,
        o7_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn3,
        o7_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn4,
        o7_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn5,
        o7_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn6,
        o7_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 112 + cn7,
        o7_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn0,
        o8_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn1,
        o8_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn2,
        o8_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn3,
        o8_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn4,
        o8_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn5,
        o8_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn6,
        o8_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 128 + cn7,
        o8_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn0,
        o9_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn1,
        o9_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn2,
        o9_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn3,
        o9_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn4,
        o9_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn5,
        o9_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn6,
        o9_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 144 + cn7,
        o9_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn0,
        o10_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn1,
        o10_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn2,
        o10_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn3,
        o10_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn4,
        o10_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn5,
        o10_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn6,
        o10_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 160 + cn7,
        o10_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn0,
        o11_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn1,
        o11_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn2,
        o11_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn3,
        o11_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn4,
        o11_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn5,
        o11_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn6,
        o11_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 176 + cn7,
        o11_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn0,
        o12_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn1,
        o12_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn2,
        o12_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn3,
        o12_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn4,
        o12_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn5,
        o12_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn6,
        o12_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 192 + cn7,
        o12_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn0,
        o13_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn1,
        o13_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn2,
        o13_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn3,
        o13_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn4,
        o13_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn5,
        o13_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn6,
        o13_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 208 + cn7,
        o13_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn0,
        o14_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn1,
        o14_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn2,
        o14_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn3,
        o14_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn4,
        o14_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn5,
        o14_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn6,
        o14_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 224 + cn7,
        o14_7 * inv1,
        cm7 < qlen,
    )
    tl.store(
        Out + (qstart + cm0).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn0,
        o15_0 * inv0,
        cm0 < qlen,
    )
    tl.store(
        Out + (qstart + cm1).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn1,
        o15_1 * inv0,
        cm1 < qlen,
    )
    tl.store(
        Out + (qstart + cm2).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn2,
        o15_2 * inv0,
        cm2 < qlen,
    )
    tl.store(
        Out + (qstart + cm3).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn3,
        o15_3 * inv0,
        cm3 < qlen,
    )
    tl.store(
        Out + (qstart + cm4).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn4,
        o15_4 * inv1,
        cm4 < qlen,
    )
    tl.store(
        Out + (qstart + cm5).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn5,
        o15_5 * inv1,
        cm5 < qlen,
    )
    tl.store(
        Out + (qstart + cm6).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn6,
        o15_6 * inv1,
        cm6 < qlen,
    )
    tl.store(
        Out + (qstart + cm7).to(tl.int64) * OS + qhead.to(tl.int64) * OH + 240 + cn7,
        o15_7 * inv1,
        cm7 < qlen,
    )
    if WRITE_LSE:
        lse0 = tl.where(
            den0 > 0.0, 32.0 + tl.log2(den0) * 0.6931471805599453, float("inf")
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + cm0).to(tl.int64),
            lse0,
            (cm0 < qlen) & (thread % 4 == 0),
        )
        lse1 = tl.where(
            den1 > 0.0, 32.0 + tl.log2(den1) * 0.6931471805599453, float("inf")
        )
        tl.store(
            LSE + qhead.to(tl.int64) * TQ + (qstart + cm4).to(tl.int64),
            lse1,
            (cm4 < qlen) & (thread % 4 == 0),
        )
