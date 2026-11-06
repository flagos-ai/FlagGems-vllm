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

import math
from typing import Optional

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import validate_dense_outputs
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.shape_utils import can_use_int32_index


@triton.jit
def mla_buffer_base(pointer, ALIGNMENT: tl.constexpr):
    # Keep bounded local offsets separate from the potentially large page base.
    address = tl.inline_asm_elementwise(
        "",
        constraints="=s,0",
        args=[pointer.to(tl.uint64)],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    )
    return tl.multiple_of(address.to(pointer.dtype), ALIGNMENT)


@triton.jit
def load_mla_page_pair(Pointer, ValidFirst, ValidSecond):
    # Constant memory is read-only during a dispatch; fallback keeps tails in bounds.
    address = Pointer.to(tl.uint64)
    read_pointer = address.to(tl.pointer_type(tl.int32, 4))
    if ValidFirst & ValidSecond & ((address & 7) == 0):
        word = tl.load(address.to(tl.pointer_type(tl.uint64, 4)))
    else:
        first = tl.load(read_pointer, ValidFirst, 0).to(tl.uint32).to(tl.uint64)
        second = tl.load(read_pointer + 1, ValidSecond, 0).to(tl.uint32).to(tl.uint64)
        word = first | (second << 32)
        word = tl.inline_asm_elementwise(
            "",
            constraints="=s,0",
            args=[word],
            dtype=tl.uint64,
            is_pure=True,
            pack=1,
        )
    return word


@triton.jit
def interleave_mla_pv_bytes(first, second, selector):
    return tl.inline_asm_elementwise(
        "v_perm_b32 $0, $2, $1, $3",
        constraints="=v,v,v,v",
        args=[first, second, selector],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def accumulate_mla_pv_limbs(High, Low, Base, State):
    return tl.inline_asm_elementwise(
        """s_waitcnt lgkmcnt(0)
v_xor_b32 v160, 64, $130
ds_read_m32x16_b16_alt v[0:3], $130 offset:0
ds_read_m32x16_b16_alt v[4:7], v160 offset:0
ds_read_m32x16_b16_alt v[8:11], $130 offset:128
ds_read_m32x16_b16_alt v[12:15], v160 offset:128
ds_read_m32x16_b16_alt v[16:19], $130 offset:256
ds_read_m32x16_b16_alt v[20:23], v160 offset:256
ds_read_m32x16_b16_alt v[24:27], $130 offset:384
ds_read_m32x16_b16_alt v[28:31], v160 offset:384
s_waitcnt lgkmcnt(7)
s_nop 0
v_mmac_i32_16x16x32_i8 v[32:35], $128, v[0:1], v[32:35]
v_mmac_i32_16x16x32_i8 v[64:67], $128, v[2:3], v[64:67]
v_mmac_i32_16x16x32_i8 v[96:99], $129, v[0:1], v[96:99]
v_mmac_i32_16x16x32_i8 v[128:131], $129, v[2:3], v[128:131]
s_waitcnt lgkmcnt(6)
s_nop 0
v_mmac_i32_16x16x32_i8 v[36:39], $128, v[4:5], v[36:39]
v_mmac_i32_16x16x32_i8 v[68:71], $128, v[6:7], v[68:71]
v_mmac_i32_16x16x32_i8 v[100:103], $129, v[4:5], v[100:103]
v_mmac_i32_16x16x32_i8 v[132:135], $129, v[6:7], v[132:135]
s_waitcnt lgkmcnt(5)
s_nop 0
v_mmac_i32_16x16x32_i8 v[40:43], $128, v[8:9], v[40:43]
v_mmac_i32_16x16x32_i8 v[72:75], $128, v[10:11], v[72:75]
v_mmac_i32_16x16x32_i8 v[104:107], $129, v[8:9], v[104:107]
v_mmac_i32_16x16x32_i8 v[136:139], $129, v[10:11], v[136:139]
s_waitcnt lgkmcnt(4)
s_nop 0
v_mmac_i32_16x16x32_i8 v[44:47], $128, v[12:13], v[44:47]
v_mmac_i32_16x16x32_i8 v[76:79], $128, v[14:15], v[76:79]
v_mmac_i32_16x16x32_i8 v[108:111], $129, v[12:13], v[108:111]
v_mmac_i32_16x16x32_i8 v[140:143], $129, v[14:15], v[140:143]
s_waitcnt lgkmcnt(3)
s_nop 0
v_mmac_i32_16x16x32_i8 v[48:51], $128, v[16:17], v[48:51]
v_mmac_i32_16x16x32_i8 v[80:83], $128, v[18:19], v[80:83]
v_mmac_i32_16x16x32_i8 v[112:115], $129, v[16:17], v[112:115]
v_mmac_i32_16x16x32_i8 v[144:147], $129, v[18:19], v[144:147]
s_waitcnt lgkmcnt(2)
s_nop 0
v_mmac_i32_16x16x32_i8 v[52:55], $128, v[20:21], v[52:55]
v_mmac_i32_16x16x32_i8 v[84:87], $128, v[22:23], v[84:87]
v_mmac_i32_16x16x32_i8 v[116:119], $129, v[20:21], v[116:119]
v_mmac_i32_16x16x32_i8 v[148:151], $129, v[22:23], v[148:151]
s_waitcnt lgkmcnt(1)
s_nop 0
v_mmac_i32_16x16x32_i8 v[56:59], $128, v[24:25], v[56:59]
v_mmac_i32_16x16x32_i8 v[88:91], $128, v[26:27], v[88:91]
v_mmac_i32_16x16x32_i8 v[120:123], $129, v[24:25], v[120:123]
v_mmac_i32_16x16x32_i8 v[152:155], $129, v[26:27], v[152:155]
s_waitcnt lgkmcnt(0)
s_nop 0
v_mmac_i32_16x16x32_i8 v[60:63], $128, v[28:29], v[60:63]
v_mmac_i32_16x16x32_i8 v[92:95], $128, v[30:31], v[92:95]
v_mmac_i32_16x16x32_i8 v[124:127], $129, v[28:29], v[124:127]
v_mmac_i32_16x16x32_i8 v[156:159], $129, v[30:31], v[156:159]
s_nop 0""",
        constraints=(
            "=&{v32},=&{v33},=&{v34},=&{v35},=&{v36},=&{v37},=&{v38},=&{v39},=&{v40},=&{v41},=&{v42},"
            "=&{v43},=&{v44},=&{v45},=&{v46},=&{v47},=&{v48},=&{v49},=&{v50},=&{v51},=&{v52},=&{v53},"
            "=&{v54},=&{v55},=&{v56},=&{v57},=&{v58},=&{v59},=&{v60},=&{v61},=&{v62},=&{v63},=&{v64},"
            "=&{v65},=&{v66},=&{v67},=&{v68},=&{v69},=&{v70},=&{v71},=&{v72},=&{v73},=&{v74},=&{v75},"
            "=&{v76},=&{v77},=&{v78},=&{v79},=&{v80},=&{v81},=&{v82},=&{v83},=&{v84},=&{v85},=&{v86},"
            "=&{v87},=&{v88},=&{v89},=&{v90},=&{v91},=&{v92},=&{v93},=&{v94},=&{v95},=&{v96},=&{v97},"
            "=&{v98},=&{v99},=&{v100},=&{v101},=&{v102},=&{v103},=&{v104},=&{v105},=&{v106},=&{v107},"
            "=&{v108},=&{v109},=&{v110},=&{v111},=&{v112},=&{v113},=&{v114},=&{v115},=&{v116},=&{v117"
            "},=&{v118},=&{v119},=&{v120},=&{v121},=&{v122},=&{v123},=&{v124},=&{v125},=&{v126},=&{v1"
            "27},=&{v128},=&{v129},=&{v130},=&{v131},=&{v132},=&{v133},=&{v134},=&{v135},=&{v136},=&{"
            "v137},=&{v138},=&{v139},=&{v140},=&{v141},=&{v142},=&{v143},=&{v144},=&{v145},=&{v146},="
            "&{v147},=&{v148},=&{v149},=&{v150},=&{v151},=&{v152},=&{v153},=&{v154},=&{v155},=&{v156}"
            ",=&{v157},=&{v158},=&{v159},v,v,v,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,2"
            "1,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50"
            ",51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,"
            "80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,"
            "107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,~{v0"
            "},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v1"
            "4},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~"
            "{v27},~{v28},~{v29},~{v30},~{v31},~{v160},~{memory}"
        ),
        args=(High, Low, Base) + State,
        dtype=(tl.int32,) * 128,
        is_pure=False,
        pack=1,
    )


@triton.jit
def quantize_mla_p16_word(word, scale, NORMALIZED: tl.constexpr = False):
    # The low limb is the signed seven-bit remainder of the biased FP32 bits.
    biased_words = ()
    if NORMALIZED:
        # Pair FP32 FMAs without changing the P14 rounding or probability precision.
        raw_words = ()
        for i in tl.static_range(4):
            raw = ((word >> (16 * i)) & 65535).to(tl.uint32).to(tl.float32)
            raw_words += (raw.to(tl.uint32, bitcast=True),)
        scale_bits = scale.to(tl.uint32, bitcast=True).to(tl.uint64)
        scale_pair = scale_bits | (scale_bits << 32)
        bias_pair = tl.full(word.shape, 0x4780000047800000, tl.uint64)
        for i in tl.static_range(2):
            raw_pair = raw_words[2 * i].to(tl.uint64) | (
                raw_words[2 * i + 1].to(tl.uint64) << 32
            )
            biased_pair = tl.inline_asm_elementwise(
                "v_pk_fma_f32 $0, $1, $2, $3",
                constraints="=v,v,v,v",
                args=[raw_pair, scale_pair, bias_pair],
                dtype=tl.uint64,
                is_pure=True,
                pack=1,
            )
            biased_words += (
                biased_pair.to(tl.uint32),
                (biased_pair >> 32).to(tl.uint32),
            )
    else:
        for i in tl.static_range(4):
            raw = ((word >> (16 * i)) & 65535).to(tl.uint32).to(tl.float32)
            normalized = tl.minimum(raw * scale, 127.0)
            biased = normalized + 65536.0
            biased_words += (biased.to(tl.uint32, bitcast=True),)
    pair_selector = tl.full(word.shape, 0x05040100, tl.uint32)
    merge_selector = tl.full(word.shape, 0x06040200, tl.uint32)
    sign_mask = tl.full(word.shape, 0x80808080, tl.uint32)
    rounding_pair = tl.full(word.shape, 0x00400040, tl.uint32)
    high, low = tl.inline_asm_elementwise(
        """v_perm_b32 v0, $3, $2, $6
v_perm_b32 v1, $5, $4, $6
v_perm_b32 $1, v1, v0, $7
v_lshlrev_b32 v2, 1, $1
v_bfi_b32 $1, $8, v2, $1
v_pk_add_u16 v0, v0, $9
v_pk_add_u16 v1, v1, $9
v_lshrrev_b32 v0, 7, v0
v_lshrrev_b32 v1, 7, v1
v_perm_b32 $0, v1, v0, $7""",
        constraints="=&v,=&v,v,v,v,v,v,v,v,v,~{v0},~{v1},~{v2}",
        args=[
            biased_words[0],
            biased_words[1],
            biased_words[2],
            biased_words[3],
            pair_selector,
            merge_selector,
            sign_mask,
            rounding_pair,
        ],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    if NORMALIZED:
        # Match minnum(NaN, 127) when normalization has nonfinite intermediates.
        is_nan = scale != scale
        return tl.where(is_nan, 0x7F7F7F7F, high), tl.where(is_nan, 0, low)
    return high, low


@triton.jit
def load_mla_buffer_u32x4(Base, ByteOffset, Mask):
    # Scalar page bases keep local offsets within the 2 GiB buffer range.
    address = Base.to(tl.uint64)
    offset = tl.where(Mask, ByteOffset.to(tl.uint32), 0x80000000)
    a, b, c, d = tl.inline_asm_elementwise(
        "buffer_load_dwordx4 v[0:3], $8, s[0:3], 0 offen\n" "s_waitcnt vmcnt(0)",
        constraints="=&{v0},=&{v1},=&{v2},=&{v3},{s0},{s1},{s2},{s3},v,~{memory}",
        args=[
            address.to(tl.uint32),
            (address >> 32).to(tl.uint32),
            tl.full((), 2147483646, tl.uint32),
            tl.full((), 0x27000, tl.uint32),
            offset,
        ],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return a, b, c, d


@triton.jit
def load_mla_probability_pair(Base, Offset, Mask):
    a, b, c, d = load_mla_buffer_u32x4(Base, Offset.to(tl.uint32) * 8, Mask)
    return a.to(tl.uint64) | (b.to(tl.uint64) << 32), c.to(tl.uint64) | (
        d.to(tl.uint64) << 32
    )


@triton.jit
def load_mla_kv_pair(Base, FirstOffset, SecondOffset, FirstMask, SecondMask):
    # Issue both token loads before waiting, preserving memory-level parallelism.
    address = Base.to(tl.uint64)
    first = tl.where(FirstMask, FirstOffset.to(tl.uint32) * 4, 0x80000000)
    second = tl.where(SecondMask, SecondOffset.to(tl.uint32) * 4, 0x80000000)
    a, b, c, d, e, f, g, h = tl.inline_asm_elementwise(
        "buffer_load_dwordx4 v[0:3], $12, s[0:3], 0 offen\n"
        "buffer_load_dwordx4 v[4:7], $13, s[0:3], 0 offen\n"
        "s_waitcnt vmcnt(0)",
        constraints=(
            "=&{v0},=&{v1},=&{v2},=&{v3},=&{v4},=&{v5},=&{v6},=&{v7},"
            "{s0},{s1},{s2},{s3},v,v,~{memory}"
        ),
        args=[
            address.to(tl.uint32),
            (address >> 32).to(tl.uint32),
            tl.full((), 2147483646, tl.uint32),
            tl.full((), 0x27000, tl.uint32),
            first,
            second,
        ],
        dtype=(tl.uint32,) * 8,
        is_pure=False,
        pack=1,
    )
    first_words = tl.join(tl.join(a, c), tl.join(b, d)).reshape((32, 64))
    second_words = tl.join(tl.join(e, g), tl.join(f, h)).reshape((32, 64))
    return first_words, second_words


@triton.jit
def native_int8_mla_pv_wide(
    KV,
    Table,
    Lengths,
    Packed,
    Weight,
    RowFactor,
    Output,
    KV_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    PARTS: tl.constexpr = 1,
    Order=None,
    NORMALIZED: tl.constexpr = False,
):
    tl.static_assert(PARTS == 1)
    PAGE_GROUP: tl.constexpr = 2
    SHARED_BYTES: tl.constexpr = 16384 * PAGE_GROUP
    C: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [16, 4], [8, 1], [0, 1])
    W: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [4, 16], [8, 1], [1, 0])
    dt, ht, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if Order is not None:
        batch = tl.load(Order + batch)
    mh: tl.constexpr = H * SQ
    local_heads = tle.gpu.set_layout(tl.arange(0, 128), tle.gpu.SlicedEncoding(1, C))
    heads = ht * 128 + local_heads
    groups = tle.gpu.set_layout(tl.arange(0, 4), tle.gpu.SlicedEncoding(0, C))
    ph = heads
    pg = groups
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    pages = tl.cdiv(length, 64)
    shared = tle.gpu.alloc(
        shape=(SHARED_BYTES,),
        dtype=tl.uint8,
        layout=tle.gpu.swizzled_shared_layout(16, 1, 1, [0], 1, [1], [1], [0]),
    )
    tail = tle.gpu.alloc(
        shape=(128, 4),
        dtype=tl.uint32,
        layout=tle.gpu.swizzled_shared_layout(
            1, 1, 1, [0, 1], 1, [1, 1], [1, 1], [0, 1]
        ),
        alias=shared,
        alias_offset_bytes=SHARED_BYTES - 2048,
    )
    tl.store(
        tle.gpu.local_ptr(tail), tle.gpu.set_layout(tl.full((128, 4), 0, tl.uint32), C)
    )
    tl.debug_barrier()
    zero = tl.inline_asm_elementwise(
        "v_and_b32 $0, $1, 0",
        constraints="=v,v,~{memory}",
        args=[tle.gpu.set_layout(tl.load(tle.gpu.local_ptr(tail)), C)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    # Each shared halfword holds one feature from two adjacent INT8 tokens.
    source_pair = (local_heads[:, None] & 15) // 4 + groups[None, :] * 4
    base_raw = (
        source_pair * 512
        + (((local_heads[:, None] & 3) * 8) ^ ((source_pair & 7) * 8)) * 2
        + zero
    )
    base = base_raw
    pairs = tle.gpu.set_layout(tl.arange(0, 32), tle.gpu.SlicedEncoding(1, W))
    features = dt * 16 + tle.gpu.set_layout(
        tl.arange(0, 16), tle.gpu.SlicedEncoding(0, W)
    )
    # Separate P14 limbs stay within INT32 at the 131072-token capacity limit.
    zero = tle.gpu.set_layout(tl.zeros((128, 4), tl.int32), C)
    state = (zero,) * 128
    for group in tl.range(0, tl.cdiv(pages, PAGE_GROUP), loop_unroll_factor=1):
        page_word = load_mla_page_pair(
            Table + batch * TABLE_STRIDE + group * 2,
            group * 2 < pages,
            group * 2 + 1 < pages,
        )
        tl.debug_barrier()
        for page_offset in tl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            page = (
                (page_word >> (32 * page_offset))
                .to(tl.uint32)
                .to(tl.int32, bitcast=True)
                .to(tl.int64)
            )
            first_token = 2 * pairs
            first, second = load_mla_kv_pair(
                KV + page * (KV_STRIDES[0] // 4),
                first_token[:, None] * (KV_STRIDES[1] // 4) + features[None, :] * 4,
                (first_token[:, None] + 1) * (KV_STRIDES[1] // 4)
                + features[None, :] * 4,
                page_idx * 64 + first_token[:, None] < length,
                page_idx * 64 + first_token[:, None] + 1 < length,
            )
            low_pair = interleave_mla_pv_bytes(first, second, 0x05010400)
            high_pair = interleave_mla_pv_bytes(first, second, 0x07030602)
            packed_pairs = tl.join(low_pair, high_pair).reshape((32, 128))
            tl.store(
                tle.gpu.set_layout(
                    tle.gpu.local_ptr(
                        tle.gpu.alloc(
                            shape=(32, 128),
                            dtype=tl.uint32,
                            layout=tle.gpu.swizzled_shared_layout(
                                4, 1, 8, [1, 0], 1, [1, 1], [1, 1], [0, 1]
                            ),
                            alias=shared,
                            alias_offset_bytes=(page_offset * 32) * 512,
                        )
                    ),
                    tle.gpu.BlockEncoding([1, 8], [4, 16], [8, 1], [1, 0]),
                ),
                tle.gpu.set_layout(
                    packed_pairs, tle.gpu.BlockEncoding([1, 8], [4, 16], [8, 1], [1, 0])
                ),
            )
        tl.debug_barrier()
        for page_offset in tl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            scale = tl.load(
                Weight + (batch * CAPACITY + page_idx) * mh + (ph),
                (ph < mh) & (page_idx < pages),
                0.0,
            )
            for half in tl.static_range(2):
                poffset = ph[:, None] * 16 + half * 8 + pg[None, :] * 2
                probability_base = Packed + (batch * CAPACITY + page_idx) * mh * 16
                p0, p1 = load_mla_probability_pair(
                    probability_base, poffset, (ph[:, None] < mh) & (page_idx < pages)
                )
                high0, low0 = quantize_mla_p16_word(p0, scale[:, None], NORMALIZED)
                high1, low1 = quantize_mla_p16_word(p1, scale[:, None], NORMALIZED)
                high = high0.to(tl.uint64) | (high1.to(tl.uint64) << 32)
                low = low0.to(tl.uint64) | (low1.to(tl.uint64) << 32)
                state = accumulate_mla_pv_limbs(
                    high, low, base + (2 * page_offset + half) * 8192, state
                )
    factor = tl.load(RowFactor + batch * mh + heads, heads < mh, 0.0)
    for register in tl.static_range(32):
        he, ho, le, lo = (
            state[register],
            state[32 + register],
            state[64 + register],
            state[96 + register],
        )
        columns = register * 4 + groups
        even = (he.to(tl.float32) * 128.0 + le.to(tl.float32)) * factor[:, None]
        odd = (ho.to(tl.float32) * 128.0 + lo.to(tl.float32)) * factor[:, None]
        word = pack_mla_output_pair(even, odd, Output.dtype.element_ty)
        offset = (batch * mh + heads[:, None]) * 256 + dt * 128 + columns[None, :]
        tl.store(
            Output.to(tl.pointer_type(tl.uint32)) + offset,
            word,
            heads[:, None] < mh,
        )


@triton.jit
def pack_mla_output_pair(even, odd, dtype: tl.constexpr):
    low = even.to(dtype).to(tl.uint16, bitcast=True).to(tl.uint32)
    high = odd.to(dtype).to(tl.uint16, bitcast=True).to(tl.uint32)
    return low | (high << 16)


@triton.jit
def store_mla_partial_limbs(
    Base,
    ByteOffsets,
    HighEven,
    LowEven,
    HighOdd,
    LowOdd,
    Valid,
    ColumnBytes: tl.constexpr,
    OddBytes: tl.constexpr,
):
    # Keep per-register column offsets inside asm to avoid extending their live ranges.
    address = Base.to(tl.uint64)
    offset = tl.where(Valid, ByteOffsets.to(tl.uint32), 0x80000000)
    tl.inline_asm_elementwise(
        "v_add_u32 v4, $10, $9\n"
        "buffer_store_dwordx2 v[0:1], v4, s[0:3], 0 offen\n"
        "v_add_u32 v5, $11, v4\n"
        "buffer_store_dwordx2 v[2:3], v5, s[0:3], 0 offen\n"
        "v_mov_b32 $0, 0",
        constraints=(
            "=v,{s0},{s1},{s2},{s3},{v0},{v1},{v2},{v3},v,n,n," "~{v4},~{v5},~{memory}"
        ),
        args=[
            address.to(tl.uint32),
            (address >> 32).to(tl.uint32),
            tl.full((), 2147483646, tl.uint32),
            tl.full((), 0x27000, tl.uint32),
            HighEven,
            LowEven,
            HighOdd,
            LowOdd,
            offset,
            tl.full((), ColumnBytes, tl.int32),
            tl.full((), OddBytes, tl.int32),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mla_probability_scale(max_scaled, QUANT_MAX: tl.constexpr):
    tl.static_assert(QUANT_MAX == 127 or QUANT_MAX == 65535)
    # Odd divisors and the scale floor make one residual correction correctly rounded.
    reciprocal: tl.constexpr = 1.0 / QUANT_MAX
    first = max_scaled * reciprocal
    residual = tl.fma(first, -QUANT_MAX * 1.0, max_scaled)
    quotient = tl.fma(residual, reciprocal, first)
    return tl.where(
        max_scaled == float("inf"), max_scaled, tl.maximum(quotient, 1.0e-30)
    )


@triton.jit
def accumulate_int8_pv(
    KVBase,
    Valid,
    PHigh,
    PLow,
    Rescale,
    PScale,
    Accumulator,
    OFFSET: tl.constexpr,
    BASE: tl.constexpr,
):
    # Two INT8 products preserve the softmax probabilities to about 14 bits.
    # Reload V in small tiles to bound live registers and shared-memory usage.
    dims = OFFSET + tl.arange(0, 128)
    value = tl.load(KVBase + dims[None, :], Valid[:, None], 0)
    high = tl.dot(tl.trans(value), tl.trans(PHigh), out_dtype=tl.int32)
    # At 64 tokens the combined INT32 magnitude is bounded by 133169152.
    combined = tl.dot(tl.trans(value), tl.trans(PLow), high * BASE, out_dtype=tl.int32)
    return Accumulator * Rescale[None, :] + combined.to(tl.float32) * (
        PScale[None, :] * (1.0 / BASE)
    )


def prune_mla_configs(configs, named_args, **kwargs):
    if named_args["H"] * named_args["SQ"] <= 32:
        return configs
    return [config for config in configs if config.kwargs["QK_BLOCK"] == 128]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dense_int8_mla_kernel"),
    key=["B", "H", "SQ", "SPLITS", "TABLE_STRIDE", "CAUSAL"],
    use_cuda_graph=True,
    prune_configs_by={"early_config_prune": prune_mla_configs},
)
@triton.jit
def dense_int8_mla_kernel(
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
    B: tl.constexpr,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    QK_BLOCK: tl.constexpr,
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Schedule one batch's splits together without changing their page order.
    if SPLITS > 1:
        linear = batch.to(tl.int64) + split.to(tl.int64) * B
        batch = (linear // SPLITS).to(tl.int32)
        split = (linear % SPLITS).to(tl.int32)
    heads = head_tile * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    merged_heads = SQ * H
    PRELOAD_Q: tl.constexpr = SQ * H <= 32
    PV_BASE: tl.constexpr = 127 if PRELOAD_Q else 128
    content_dims = tl.arange(0, QK_BLOCK)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, 64)
    q_base = (
        Q
        + batch * Q_STRIDES[0]
        + queries[:, None] * Q_STRIDES[1]
        + qheads[:, None] * Q_STRIDES[2]
    )
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + queries[:, None] * QR_STRIDES[1]
        + qheads[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        heads[:, None] < merged_heads,
        0,
    )
    qs = tl.load(
        QS + batch * QS_STRIDES[0] + queries * QS_STRIDES[1] + qheads * QS_STRIDES[2],
        heads < merged_heads,
        0,
    )
    query_tiles = ()
    if PRELOAD_Q:
        for chunk in tl.static_range(512 // QK_BLOCK):
            query_tile = tl.load(
                q_base + chunk * QK_BLOCK + content_dims[None, :],
                heads[:, None] < merged_heads,
                0,
            )
            query_tiles += (query_tile,)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    query_length = tl.full((BH,), length, tl.int32)
    if CAUSAL:
        query_length = tl.maximum(length - SQ + queries + 1, 0)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    acc0 = tl.zeros((128, BH), tl.float32)
    acc1 = tl.zeros((128, BH), tl.float32)
    acc2 = tl.zeros((128, BH), tl.float32)
    acc3 = tl.zeros((128, BH), tl.float32)
    valid_pages = tl.cdiv(length, 64)
    first_page = valid_pages * split // SPLITS
    end_page = valid_pages * (split + 1) // SPLITS
    for page_idx in range(first_page, end_page):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        kv_base = KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1]
        content_raw = tl.zeros((64, BH), tl.int32)
        if PRELOAD_Q:
            for chunk in tl.static_range(512 // QK_BLOCK):
                dims = chunk * QK_BLOCK + content_dims
                key = tl.load(kv_base + dims[None, :], valid[:, None], 0)
                content_raw = tl.dot(
                    key, tl.trans(query_tiles[chunk]), content_raw, out_dtype=tl.int32
                )
        else:
            # Keeping wide queries across PV spills the output accumulators.
            for chunk in range(4):
                dims = chunk * 128 + content_dims
                q = tl.load(q_base + dims[None, :], heads[:, None] < merged_heads, 0)
                key = tl.load(kv_base + dims[None, :], valid[:, None], 0)
                content_raw = tl.dot(key, tl.trans(q), content_raw, out_dtype=tl.int32)
        kr = tl.load(
            KR
            + page * KR_STRIDES[0]
            + tokens[:, None] * KR_STRIDES[1]
            + rope_dims[None, :],
            valid[:, None],
            0,
        )
        ks = tl.load(KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1], valid, 0)
        content_scores = tl.trans(content_raw).to(tl.float32)
        rope_scores = tl.trans(tl.dot(kr, tl.trans(qr), out_dtype=tl.float32))
        scores = (
            (content_scores + rope_scores)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        scores = tl.where(
            page_idx * 64 + tokens[None, :] < query_length[:, None],
            scores,
            -float("inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probabilities = tl.exp2(scores - safe_maximum[:, None])
        rescale = tl.exp2(maximum - safe_maximum)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        scaled = probabilities * ks[None, :]
        probability_scale = mla_probability_scale(tl.max(scaled, 1), 127)
        scaled /= probability_scale[:, None]
        if PRELOAD_Q:
            # The small-head query-preload path benefits from float residuals.
            high = (scaled + 0.5).to(tl.int8)
            low = tl.floor((scaled - high.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
        else:
            # scaled is in [0, 127]; the biased FP32 ULP is 1/128.
            quant = (scaled + 65536.0).to(tl.int32, bitcast=True) - 0x47800000
            high_raw = (quant + 64) >> 7
            high = high_raw.to(tl.int8)
            low = (quant - (high_raw << 7)).to(tl.int8)
        acc0 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc0, 0, PV_BASE
        )
        acc1 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc1, 128, PV_BASE
        )
        acc2 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc2, 256, PV_BASE
        )
        acc3 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc3, 384, PV_BASE
        )
        maximum = next_maximum
    out_dims = tl.arange(0, 128)
    if SPLITS == 1:
        inverse = 1.0 / tl.where(denominator > 0, denominator, 1.0)
        base = (
            Output + (batch * merged_heads + heads[None, :]) * 512 + out_dims[:, None]
        )
        tl.store(base, acc0 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 128, acc1 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 256, acc2 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 384, acc3 * inverse[None, :], heads[None, :] < merged_heads)
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(
            LSE + (batch * H + qheads) * SQ + queries, logsum, heads < merged_heads
        )
    else:
        base = (
            Partial
            + ((split * B + batch) * merged_heads + heads[None, :]) * 512
            + out_dims[:, None]
        )
        tl.store(base, acc0, heads[None, :] < merged_heads)
        tl.store(base + 128, acc1, heads[None, :] < merged_heads)
        tl.store(base + 256, acc2, heads[None, :] < merged_heads)
        tl.store(base + 384, acc3, heads[None, :] < merged_heads)
        tl.store(
            Stats + (split * 2 * B + batch) * merged_heads + heads,
            maximum,
            heads < merged_heads,
        )
        tl.store(
            Stats + ((split * 2 + 1) * B + batch) * merged_heads + heads,
            denominator,
            heads < merged_heads,
        )


@libentry()
@triton.jit
def dense_int8_mla_merge(
    Partial,
    Stats,
    Output,
    LSE,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    # Vectorize column loads and keep each split reduction within one thread.
    layout: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [1, 64], [1, 2], [1, 0])
    splits = tle.gpu.set_layout(tl.arange(0, SPLITS), tle.gpu.SlicedEncoding(1, layout))
    dims = tle.gpu.set_layout(tl.arange(0, 512), tle.gpu.SlicedEncoding(0, layout))
    merged_heads = H * SQ
    maxima = tl.load(Stats + (splits * 2 * B + batch) * merged_heads + head)
    maximum = tl.max(maxima, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - safe_maximum)
    sums = tl.load(Stats + ((splits * 2 + 1) * B + batch) * merged_heads + head)
    denominator = tl.sum(sums * weights, 0)
    values = tl.load(
        Partial
        + ((splits[:, None] * B + batch) * merged_heads + head) * 512
        + dims[None, :]
    )
    output = tl.sum(values * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(Output + (batch * merged_heads + head) * 512 + dims, output)
    logsum = tl.where(
        denominator > 0,
        maximum * 0.6931471805599453 + tl.log(denominator),
        float("inf"),
    )
    tl.store(LSE + (batch * H + head % H) * SQ + head // H, logsum)


@libentry()
@triton.jit
def _staged_int8_mla_qk(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Packed,
    Stats,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    PAGES: tl.constexpr,
    REBASE_KV: tl.constexpr = False,
    REBASE_PACKED: tl.constexpr = False,
    KV_ALIGNMENT: tl.constexpr = 1,
    PACKED_ALIGNMENT: tl.constexpr = 1,
):
    if PAGES > 1:
        Table = Table.to(tl.uint64).to(tl.pointer_type(tl.int32, 4))
    NOPE_TOKEN_FIRST: tl.constexpr = PAGES == 1 and BH == 32
    ht = 0 if H * SQ <= BH else tl.program_id(0)
    first_page, batch = tl.program_id(1), tl.program_id(2)
    first_page *= PAGES
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    if first_page * 64 >= length:
        return
    heads = ht * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    mh: tl.constexpr = H * SQ
    tokens = tl.arange(0, 64)
    dims = tl.arange(0, QK_BLOCK)
    rd = tl.arange(0, 64)
    qb = (
        Q
        + batch * Q_STRIDES[0]
        + queries[:, None] * Q_STRIDES[1]
        + qheads[:, None] * Q_STRIDES[2]
    )
    # Reuse Q across cache pages without retaining PV output accumulators.
    if PAGES > 1:
        query_tiles = ()
        for chunk in tl.static_range(512 // QK_BLOCK):
            query = tl.load(
                qb + chunk * QK_BLOCK + dims[None, :], heads[:, None] < mh, 0
            )
            query_tiles += (query,)
        qr = tl.load(
            QR
            + batch * QR_STRIDES[0]
            + queries[:, None] * QR_STRIDES[1]
            + qheads[:, None] * QR_STRIDES[2]
            + rd[None, :],
            heads[:, None] < mh,
            0,
        )
        qs = tl.load(
            QS
            + batch * QS_STRIDES[0]
            + queries * QS_STRIDES[1]
            + qheads * QS_STRIDES[2],
            heads < mh,
            0,
        )
    if PAGES == 1:
        pages_in_group = 1
    else:
        pages_in_group = tl.minimum(PAGES, tl.cdiv(length, 64) - first_page)
    for page_offset in range(pages_in_group):
        page_idx = first_page + page_offset
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        if REBASE_KV:
            page_base = mla_buffer_base(KV + page * KV_STRIDES[0], KV_ALIGNMENT)
            if 63 * KV_STRIDES[1] + 512 > 2**31 - 2:
                kv_tokens = tokens.to(tl.int64)
            else:
                kv_tokens = tokens
            kv_offsets = kv_tokens[:, None] * KV_STRIDES[1] + dims[None, :]
        else:
            kb = KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1]
        if NOPE_TOKEN_FIRST:
            content = tl.zeros((64, BH), tl.int32)
        else:
            content = tl.zeros((BH, 64), tl.int32)
        if PAGES > 1:
            for chunk in tl.static_range(512 // QK_BLOCK):
                if REBASE_KV:
                    key_pointer = page_base + (tl.trans(kv_offsets) + chunk * QK_BLOCK)
                else:
                    key_pointer = (
                        KV
                        + page * KV_STRIDES[0]
                        + tokens[None, :] * KV_STRIDES[1]
                        + chunk * QK_BLOCK
                        + dims[:, None]
                    )
                key_t = tl.load(key_pointer, valid[None, :], 0)
                content = tl.dot(query_tiles[chunk], key_t, content, out_dtype=tl.int32)
        else:
            if NOPE_TOKEN_FIRST:
                for chunk in tl.static_range(512 // QK_BLOCK):
                    qt = tl.load(
                        Q
                        + batch * Q_STRIDES[0]
                        + queries[None, :] * Q_STRIDES[1]
                        + qheads[None, :] * Q_STRIDES[2]
                        + chunk * QK_BLOCK
                        + dims[:, None],
                        heads[None, :] < mh,
                        0,
                    )
                    if REBASE_KV:
                        key_pointer = page_base + (kv_offsets + chunk * QK_BLOCK)
                    else:
                        key_pointer = kb + chunk * QK_BLOCK + dims[None, :]
                    k = tl.load(key_pointer, valid[:, None], 0)
                    content = tl.dot(k, qt, content, out_dtype=tl.int32)
            else:
                for chunk in range(512 // QK_BLOCK):
                    q = tl.load(
                        qb + chunk * QK_BLOCK + dims[None, :], heads[:, None] < mh, 0
                    )
                    if REBASE_KV:
                        key_pointer = page_base + (
                            tl.trans(kv_offsets) + chunk * QK_BLOCK
                        )
                    else:
                        key_pointer = (
                            KV
                            + page * KV_STRIDES[0]
                            + tokens[None, :] * KV_STRIDES[1]
                            + chunk * QK_BLOCK
                            + dims[:, None]
                        )
                    kt = tl.load(key_pointer, valid[None, :], 0)
                    content = tl.dot(q, kt, content, out_dtype=tl.int32)
        if NOPE_TOKEN_FIRST:
            content = tl.trans(content)
        if PAGES == 1:
            qr = tl.load(
                QR
                + batch * QR_STRIDES[0]
                + queries[:, None] * QR_STRIDES[1]
                + qheads[:, None] * QR_STRIDES[2]
                + rd[None, :],
                heads[:, None] < mh,
                0,
            )
        kr_t = tl.load(
            KR + page * KR_STRIDES[0] + rd[:, None] + tokens[None, :] * KR_STRIDES[1],
            valid[None, :],
            0,
        )
        if PAGES == 1:
            qs = tl.load(
                QS
                + batch * QS_STRIDES[0]
                + queries * QS_STRIDES[1]
                + qheads * QS_STRIDES[2],
                heads < mh,
                0,
            )
        ks = tl.load(KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1], valid, 0)
        rope_score = tl.dot(qr, kr_t, out_dtype=tl.float32)
        # Prevent dot-add fusion from rounding away content during RoPE cancellation.
        rope_score = tl.inline_asm_elementwise(
            "",
            constraints="=v,0",
            args=[rope_score],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        score = (
            (content.to(tl.float32) + rope_score)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        if CAUSAL:
            query_length = tl.maximum(length - SQ + queries + 1, 0)
        else:
            query_length = tl.full((BH,), length, tl.int32)
        score = tl.where(
            page_idx * 64 + tokens[None, :] < query_length[:, None],
            score,
            -float("inf"),
        )
        maximum = tl.max(score, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        prob = tl.exp2(score - safe[:, None])
        denom = tl.sum(prob, 1)
        scaled = prob * ks[None, :]
        ps = mla_probability_scale(tl.max(scaled, 1), 65535)
        inverse_ps = 1.0 / ps
        normalized = scaled * inverse_ps[:, None]
        code = (normalized + 0.5).to(tl.int32)
        if REBASE_PACKED:
            packed_base = mla_buffer_base(
                Packed
                + ((batch.to(tl.int64) * CAPACITY + page_idx) * mh + ht * BH) * 64,
                PACKED_ALIGNMENT,
            )
            local_heads = tl.arange(0, BH)
            offset = local_heads[:, None] * 64 + tokens[None, :]
            # Convert through INT32 before truncation: 65535 exceeds signed INT16.
            tl.store(packed_base + offset, code.to(tl.int16), heads[:, None] < mh)
        else:
            offset = (
                (batch * CAPACITY + page_idx) * mh + heads[:, None]
            ) * 64 + tokens[None, :]
            # Convert through INT32 before truncation: 65535 exceeds signed INT16.
            tl.store(Packed + offset, code.to(tl.int16), heads[:, None] < mh)
        stat_base = (batch * (3 * CAPACITY) + page_idx) * mh + heads
        tl.store(Stats + stat_base, maximum, heads < mh)
        tl.store(Stats + stat_base + CAPACITY * mh, denom, heads < mh)
        tl.store(Stats + stat_base + 2 * CAPACITY * mh, ps, heads < mh)


@libentry()
@triton.jit
def _staged_int8_mla_normalize(
    Stats,
    Weight,
    RowFactor,
    LSE,
    Lengths,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    NP: tl.constexpr,
    BH: tl.constexpr,
    B: tl.constexpr = 0,
    Order=None,
    ORDER_BLOCK: tl.constexpr = 1,
):
    # Adjacent heads make page-statistic reads and weight writes coalesced.
    ht, batch = tl.program_id(0), tl.program_id(1)
    if Order is not None:
        if (ht | batch) == 0:
            batches = tl.arange(0, ORDER_BLOCK)
            lengths = tl.maximum(
                tl.load(Lengths + batches * LENGTH_STRIDE, batches < B, 0), 0
            ).to(tl.int64)
            # Tie-breaking by original row preserves deterministic scheduling.
            ranked = tl.where(
                batches < B, lengths * ORDER_BLOCK + ORDER_BLOCK - 1 - batches, -1
            )
            ranked = tl.sort(ranked, descending=True)
            original = (ORDER_BLOCK - 1 - (ranked & (ORDER_BLOCK - 1))).to(tl.int32)
            tl.store(Order + batches, original, batches < B)
    mh: tl.constexpr = H * SQ
    heads = ht * BH + tl.arange(0, BH)
    idx = tl.arange(0, NP)
    pages = tl.cdiv(tl.load(Lengths + batch * LENGTH_STRIDE), 64)
    valid = (idx[None, :] < pages) & (heads[:, None] < mh)
    base = (batch * (3 * CAPACITY) + idx[None, :]) * mh + heads[:, None]
    m = tl.load(Stats + base, valid, -float("inf"))
    d = tl.load(Stats + base + CAPACITY * mh, valid, 0.0)
    ps = tl.load(Stats + base + 2 * CAPACITY * mh, valid, 0.0)
    maximum = tl.max(m, 1)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    beta = tl.exp2(m - safe[:, None])
    denom = tl.sum(d * beta, 1)
    inv = 1.0 / tl.where(denom > 0.0, denom, 1.0)
    global_scale = tl.maximum(tl.max(ps * beta, 1) * (65535.0 / 127.0), 1.0e-30)
    inverse_scale = 1.0 / global_scale
    factor = (beta * ps) * inverse_scale[:, None]
    tl.store(
        Weight + (batch * CAPACITY + idx[None, :]) * mh + heads[:, None], factor, valid
    )
    tl.store(
        RowFactor + batch * mh + heads, global_scale * ((1.0 / 128.0) * inv), heads < mh
    )
    lse = tl.where(
        denom > 0.0, maximum * 0.6931471805599453 + tl.log(denom), float("inf")
    )
    tl.store(LSE + (batch * H + heads % H) * SQ + heads // H, lse, heads < mh)


@libentry()
@triton.jit
def _staged_int8_mla_requant(
    Packed,
    Weight,
    Lengths,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    BH: tl.constexpr,
):
    ht, page, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    if page * 64 >= length:
        return
    mh: tl.constexpr = H * SQ
    heads = ht * BH + tl.arange(0, BH)
    tokens = tl.arange(0, 64)
    base = (batch * CAPACITY + page) * mh + heads
    offset = base[:, None] * 64 + tokens[None, :]
    raw = tl.load(Packed + offset, heads[:, None] < mh, 0).to(tl.int32) & 65535
    scale = tl.load(Weight + base, heads < mh, 0.0)
    normalized = tl.minimum(raw.to(tl.float32) * scale[:, None], 127.0)
    quant = (normalized + 65536.0).to(tl.int32, bitcast=True) - 0x47800000
    high = (quant + 64) >> 7
    low = quant - (high << 7)
    code = high | ((low & 255) << 8)
    tl.store(Packed + offset, code.to(tl.int16), heads[:, None] < mh)


@libentry()
@triton.jit
def _staged_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    RowFactor,
    Output,
    KV_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    dt, ht, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mh: tl.constexpr = H * SQ
    heads = ht * BH + tl.arange(0, BH)
    dims = dt * BD + tl.arange(0, BD)
    tokens = tl.arange(0, 64)
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    pages = tl.cdiv(length, 64)
    tl.static_assert(CAPACITY <= 2048)
    # At 131072 tokens, |high| <= 2130706432 and |low| <= 1073741824.
    # Keep the limbs separate until FP32 conversion to avoid INT32 overflow.
    hi = tl.zeros((BD, BH), tl.int32)
    lo = tl.zeros((BD, BH), tl.int32)
    for page_idx in range(pages):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        value = tl.load(
            KV + page * KV_STRIDES[0] + tokens[None, :] * KV_STRIDES[1] + dims[:, None],
            valid[None, :],
            0,
        )
        offset = ((batch * CAPACITY + page_idx) * mh + heads[None, :]) * 64 + tokens[
            :, None
        ]
        code = tl.load(Packed + offset, heads[None, :] < mh, 0)
        high = code.to(tl.int8)
        low = (code >> 8).to(tl.int8)
        hi = tl.dot(value, high, hi, out_dtype=tl.int32)
        lo = tl.dot(value, low, lo, out_dtype=tl.int32)
    total = hi.to(tl.float32) * 128.0 + lo.to(tl.float32)
    factor = tl.load(RowFactor + batch * mh + heads, heads < mh, 0.0)
    tl.store(
        Output + (batch * mh + heads[None, :]) * 512 + dims[:, None],
        total * factor[None, :],
        heads[None, :] < mh,
    )


@triton.jit
def native_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    Weight,
    RowFactor,
    Output,
    KV_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    PARTS: tl.constexpr = 1,
    Order=None,
):
    PAGE_GROUP: tl.constexpr = 1
    SHARED_BYTES: tl.constexpr = 16384 * PAGE_GROUP
    C: tl.constexpr = tle.gpu.BlockEncoding([1, 1], [16, 4], [4, 1], [0, 1])
    W: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [4, 16], [4, 1], [1, 0])
    dt, ht, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if Order is not None:
        batch = tl.load(Order + batch)
    if PARTS > 1:
        part = dt // 2
        dt = dt % 2
    mh: tl.constexpr = H * SQ
    local_heads = tle.gpu.set_layout(tl.arange(0, 64), tle.gpu.SlicedEncoding(1, C))
    heads = ht * 64 + local_heads
    groups = tle.gpu.set_layout(tl.arange(0, 4), tle.gpu.SlicedEncoding(0, C))
    ph = heads
    pg = groups
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    pages = tl.cdiv(length, 64)
    shared = tle.gpu.alloc(
        shape=(SHARED_BYTES,),
        dtype=tl.uint8,
        layout=tle.gpu.swizzled_shared_layout(16, 1, 1, [0], 1, [1], [1], [0]),
    )
    tail = tle.gpu.alloc(
        shape=(64, 4),
        dtype=tl.uint32,
        layout=tle.gpu.swizzled_shared_layout(
            1, 1, 1, [0, 1], 1, [1, 1], [1, 1], [0, 1]
        ),
        alias=shared,
        alias_offset_bytes=SHARED_BYTES - 1024,
    )
    tl.store(
        tle.gpu.local_ptr(tail), tle.gpu.set_layout(tl.full((64, 4), 0, tl.uint32), C)
    )
    tl.debug_barrier()
    zero = tl.inline_asm_elementwise(
        "v_and_b32 $0, $1, 0",
        constraints="=v,v,~{memory}",
        args=[tle.gpu.set_layout(tl.load(tle.gpu.local_ptr(tail)), C)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    # Each shared halfword holds one feature from two adjacent INT8 tokens.
    source_pair = (local_heads[:, None] & 15) // 4 + groups[None, :] * 4
    base_raw = (
        source_pair * 512
        + (((local_heads[:, None] & 3) * 8) ^ ((source_pair & 7) * 8)) * 2
        + zero
    )
    base = base_raw
    pairs = tle.gpu.set_layout(tl.arange(0, 32), tle.gpu.SlicedEncoding(1, W))
    features = dt * 64 + tle.gpu.set_layout(
        tl.arange(0, 64), tle.gpu.SlicedEncoding(0, W)
    )
    # Separate P14 limbs stay within INT32 at the 131072-token capacity limit.
    zero = tle.gpu.set_layout(tl.zeros((64, 4), tl.int32), C)
    state = (zero,) * 128
    if PARTS > 1:
        page_groups = tl.cdiv(pages, PAGE_GROUP)
        groups_per_part = tl.cdiv(page_groups, PARTS)
        first_group = part * groups_per_part
        last_group = tl.minimum(page_groups, first_group + groups_per_part)
    else:
        first_group = 0
        last_group = tl.cdiv(pages, PAGE_GROUP)
    for group in tl.range(first_group, last_group, loop_unroll_factor=1):
        if PARTS > 1 and PAGE_GROUP == 2:
            page_word = load_mla_page_pair(
                Table + batch * TABLE_STRIDE + group * 2,
                group * 2 < pages,
                group * 2 + 1 < pages,
            )
        tl.debug_barrier()
        for page_offset in tl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            if PARTS > 1 and PAGE_GROUP == 2:
                page = (
                    (page_word >> (32 * page_offset))
                    .to(tl.uint32)
                    .to(tl.int32, bitcast=True)
                    .to(tl.int64)
                )
            else:
                page = tl.load(
                    Table + batch * TABLE_STRIDE + page_idx, page_idx < pages, 0
                ).to(tl.int64)
            first_token = 2 * pairs
            first = tl.load(
                KV
                + page * (KV_STRIDES[0] // 4)
                + (first_token[:, None] * (KV_STRIDES[1] // 4) + features[None, :]),
                page_idx * 64 + first_token[:, None] < length,
                0,
            )
            second = tl.load(
                KV
                + page * (KV_STRIDES[0] // 4)
                + (
                    (first_token[:, None] + 1) * (KV_STRIDES[1] // 4)
                    + features[None, :]
                ),
                page_idx * 64 + first_token[:, None] + 1 < length,
                0,
            )
            low_pair = interleave_mla_pv_bytes(first, second, 0x05010400)
            high_pair = interleave_mla_pv_bytes(first, second, 0x07030602)
            packed_pairs = tl.join(low_pair, high_pair).reshape((32, 128))
            tl.store(
                tle.gpu.set_layout(
                    tle.gpu.local_ptr(
                        tle.gpu.alloc(
                            shape=(32, 128),
                            dtype=tl.uint32,
                            layout=tle.gpu.swizzled_shared_layout(
                                4, 1, 8, [1, 0], 1, [1, 1], [1, 1], [0, 1]
                            ),
                            alias=shared,
                            alias_offset_bytes=(page_offset * 32) * 512,
                        )
                    ),
                    tle.gpu.BlockEncoding([1, 8], [4, 16], [4, 1], [1, 0]),
                ),
                tle.gpu.set_layout(
                    packed_pairs, tle.gpu.BlockEncoding([1, 8], [4, 16], [4, 1], [1, 0])
                ),
            )
        tl.debug_barrier()
        for page_offset in tl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            scale = tl.load(
                Weight + (batch * CAPACITY + page_idx) * mh + (ph),
                (ph < mh) & (page_idx < pages),
                0.0,
            )
            for half in tl.static_range(2):
                poffset = ph[:, None] * 16 + half * 8 + pg[None, :] * 2
                probability_base = Packed + (batch * CAPACITY + page_idx) * mh * 16
                p0, p1 = load_mla_probability_pair(
                    probability_base, poffset, (ph[:, None] < mh) & (page_idx < pages)
                )
                high0, low0 = quantize_mla_p16_word(p0, scale[:, None])
                high1, low1 = quantize_mla_p16_word(p1, scale[:, None])
                high = high0.to(tl.uint64) | (high1.to(tl.uint64) << 32)
                low = low0.to(tl.uint64) | (low1.to(tl.uint64) << 32)
                state = accumulate_mla_pv_limbs(
                    high, low, base + (page_offset * 2 + half) * 8192, state
                )
    if PARTS == 1:
        factor = tl.load(RowFactor + batch * mh + heads, heads < mh, 0.0)
    else:
        HT: tl.constexpr = (H * SQ + 63) // 64
        tile = (batch * HT + ht) * 2 + dt
    for register in tl.static_range(32):
        he, ho, le, lo = (
            state[register],
            state[32 + register],
            state[64 + register],
            state[96 + register],
        )
        columns = register * 4 + groups
        if PARTS == 1:
            even = (he.to(tl.float32) * 128.0 + le.to(tl.float32)) * factor[:, None]
            odd = (ho.to(tl.float32) * 128.0 + lo.to(tl.float32)) * factor[:, None]
            word = pack_mla_output_pair(even, odd, Output.dtype.element_ty)
            offset = (batch * mh + heads[:, None]) * 256 + dt * 128 + columns[None, :]
            tl.store(
                Output.to(tl.pointer_type(tl.uint32)) + offset,
                word,
                heads[:, None] < mh,
            )
        else:
            partial_base = Output + (tile * PARTS + part) * 16384
            byte_offsets = (groups[None, :] * 64 + local_heads[:, None]) * 8
            store_mla_partial_limbs(
                partial_base,
                byte_offsets,
                he,
                le,
                ho,
                lo,
                heads[:, None] < mh,
                register * 2048,
                65536,
            )


@triton.jit
def merge_mla_pv_integer(
    Partial,
    RowFactor,
    Output,
    MH: tl.constexpr,
    PARTS: tl.constexpr,
    COLS: tl.constexpr,
):
    HEAD_TILE: tl.constexpr = 32 if MH == 32 else 64
    HT: tl.constexpr = (MH + HEAD_TILE - 1) // HEAD_TILE
    program, tile = tl.program_id(0), tl.program_id(1)
    if HEAD_TILE == 32:
        half, block = program // (64 // COLS), program % (64 // COLS)
    else:
        half, block = 0, program
    batch, ht, dt = tile // (HT * 2), (tile // 2) % HT, tile % 2
    C: tl.constexpr = tle.gpu.BlockEncoding(
        [1, 1], [16, 4], [2, 2] if MH == 32 else [4, 1], [0, 1]
    )
    local = tle.gpu.set_layout(tl.arange(0, HEAD_TILE), tle.gpu.SlicedEncoding(1, C))
    heads = ht * HEAD_TILE + local
    columns = block * COLS + tle.gpu.set_layout(
        tl.arange(0, COLS), tle.gpu.SlicedEncoding(0, C)
    )
    he = tle.gpu.set_layout(tl.zeros((HEAD_TILE, COLS), tl.int32), C)
    ho = tle.gpu.set_layout(tl.zeros((HEAD_TILE, COLS), tl.int32), C)
    le = tle.gpu.set_layout(tl.zeros((HEAD_TILE, COLS), tl.int32), C)
    lo = tle.gpu.set_layout(tl.zeros((HEAD_TILE, COLS), tl.int32), C)
    for part in tl.static_range(PARTS):
        if HEAD_TILE == 32:
            offset = (
                (((tile * PARTS + part) * 2 + half) * 2) * 64 + columns[None, :]
            ) * 32 + local[:, None]
            odd_offset = 64 * 32
        else:
            offset = (
                ((tile * PARTS + part) * 2) * 128 + columns[None, :]
            ) * 64 + local[:, None]
            odd_offset = 128 * 64
        even = tl.load(Partial + offset, heads[:, None] < MH, 0)
        odd = tl.load(
            Partial + offset + odd_offset,
            heads[:, None] < MH,
            0,
        )
        he += even.to(tl.uint32).to(tl.int32, bitcast=True)
        le += (even >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
        ho += odd.to(tl.uint32).to(tl.int32, bitcast=True)
        lo += (odd >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
    factor = tl.load(RowFactor + batch * MH + heads, heads < MH, 0.0)
    even = (he.to(tl.float32) * 128.0 + le.to(tl.float32)) * factor[:, None]
    odd = (ho.to(tl.float32) * 128.0 + lo.to(tl.float32)) * factor[:, None]
    word = pack_mla_output_pair(even, odd, Output.dtype.element_ty)
    offset = (
        (batch * MH + heads[:, None]) * 256 + dt * 128 + half * 64 + columns[None, :]
    )
    tl.store(Output.to(tl.pointer_type(tl.uint32)) + offset, word, heads[:, None] < MH)


@triton.jit
def accumulate_small_head_pv_limbs(High, Low, Base, State):
    return tl.inline_asm_elementwise(
        """s_waitcnt lgkmcnt(0)
s_setprio 1
v_xor_b32 v80, 64, $66
ds_read_m32x16_b16_alt v[0:3], $66 offset:0
ds_read_m32x16_b16_alt v[4:7], v80 offset:0
ds_read_m32x16_b16_alt v[8:11], $66 offset:128
ds_read_m32x16_b16_alt v[12:15], v80 offset:128
s_waitcnt lgkmcnt(3)
s_nop 0
v_mmac_i32_16x16x32_i8 v[16:19], $64, v[0:1], v[16:19]
v_mmac_i32_16x16x32_i8 v[32:35], $64, v[2:3], v[32:35]
v_mmac_i32_16x16x32_i8 v[48:51], $65, v[0:1], v[48:51]
v_mmac_i32_16x16x32_i8 v[64:67], $65, v[2:3], v[64:67]
s_waitcnt lgkmcnt(2)
s_nop 0
v_mmac_i32_16x16x32_i8 v[20:23], $64, v[4:5], v[20:23]
v_mmac_i32_16x16x32_i8 v[36:39], $64, v[6:7], v[36:39]
v_mmac_i32_16x16x32_i8 v[52:55], $65, v[4:5], v[52:55]
v_mmac_i32_16x16x32_i8 v[68:71], $65, v[6:7], v[68:71]
s_waitcnt lgkmcnt(1)
s_nop 0
v_mmac_i32_16x16x32_i8 v[24:27], $64, v[8:9], v[24:27]
v_mmac_i32_16x16x32_i8 v[40:43], $64, v[10:11], v[40:43]
v_mmac_i32_16x16x32_i8 v[56:59], $65, v[8:9], v[56:59]
v_mmac_i32_16x16x32_i8 v[72:75], $65, v[10:11], v[72:75]
s_waitcnt lgkmcnt(0)
s_nop 0
v_mmac_i32_16x16x32_i8 v[28:31], $64, v[12:13], v[28:31]
v_mmac_i32_16x16x32_i8 v[44:47], $64, v[14:15], v[44:47]
v_mmac_i32_16x16x32_i8 v[60:63], $65, v[12:13], v[60:63]
v_mmac_i32_16x16x32_i8 v[76:79], $65, v[14:15], v[76:79]
s_nop 0
s_setprio 0""",
        constraints=(
            "=&{v16},=&{v17},=&{v18},=&{v19},=&{v20},=&{v21},=&{v22},=&{v23},=&{v24},=&{v25},=&{v26},"
            "=&{v27},=&{v28},=&{v29},=&{v30},=&{v31},=&{v32},=&{v33},=&{v34},=&{v35},=&{v36},=&{v37},"
            "=&{v38},=&{v39},=&{v40},=&{v41},=&{v42},=&{v43},=&{v44},=&{v45},=&{v46},=&{v47},=&{v48},"
            "=&{v49},=&{v50},=&{v51},=&{v52},=&{v53},=&{v54},=&{v55},=&{v56},=&{v57},=&{v58},=&{v59},"
            "=&{v60},=&{v61},=&{v62},=&{v63},=&{v64},=&{v65},=&{v66},=&{v67},=&{v68},=&{v69},=&{v70},"
            "=&{v71},=&{v72},=&{v73},=&{v74},=&{v75},=&{v76},=&{v77},=&{v78},=&{v79},v,v,v,0,1,2,3,4,"
            "5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,"
            "36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,~{v0"
            "},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v1"
            "4},~{v15},~{v80},~{memory}"
        ),
        args=(High, Low, Base) + State,
        dtype=(tl.int32,) * 64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def small_head_int8_mla_pv(
    KV,
    Table,
    Lengths,
    Packed,
    Weight,
    RowFactor,
    Output,
    KV_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    PARTS: tl.constexpr = 1,
    Order=None,
):
    PAGE_GROUP: tl.constexpr = 1
    SHARED_BYTES: tl.constexpr = 16384 * PAGE_GROUP
    # Metadata remains read-only throughout this kernel dispatch.
    Table = Table.to(tl.uint64).to(tl.pointer_type(tl.int32, 4))
    C: tl.constexpr = tle.gpu.BlockEncoding([1, 1, 1], [16, 1, 4], [2, 2, 1], [0, 2, 1])
    W: tl.constexpr = tle.gpu.BlockEncoding([1, 4], [4, 16], [4, 1], [1, 0])
    dt, ht, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if Order is not None:
        batch = tl.load(Order + batch)
    part = dt // 2
    dt = dt % 2
    mh: tl.constexpr = H * SQ
    local = tle.gpu.set_layout(
        tl.arange(0, 32), tle.gpu.SlicedEncoding(1, tle.gpu.SlicedEncoding(2, C))
    )
    heads = ht * 32 + local
    groups = tle.gpu.set_layout(
        tl.arange(0, 4), tle.gpu.SlicedEncoding(0, tle.gpu.SlicedEncoding(1, C))
    )
    halves = tle.gpu.set_layout(
        tl.arange(0, 2), tle.gpu.SlicedEncoding(0, tle.gpu.SlicedEncoding(2, C))
    )
    ph = heads
    pg = groups
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    pages = tl.cdiv(length, 64)
    shared = tle.gpu.alloc(
        shape=(SHARED_BYTES,),
        dtype=tl.uint8,
        layout=tle.gpu.swizzled_shared_layout(16, 1, 1, [0], 1, [1], [1], [0]),
    )
    tail = tle.gpu.alloc(
        shape=(32, 2, 4),
        dtype=tl.uint32,
        layout=tle.gpu.swizzled_shared_layout(
            1, 1, 1, [0, 2, 1], 1, [1, 1, 1], [1, 1, 1], [0, 1, 2]
        ),
        alias=shared,
        alias_offset_bytes=SHARED_BYTES - 1024,
    )
    tl.store(
        tle.gpu.local_ptr(tail),
        tle.gpu.set_layout(tl.full((32, 2, 4), 0, tl.uint32), C),
    )
    tl.debug_barrier()
    zero = tl.inline_asm_elementwise(
        "v_and_b32 $0, $1, 0",
        constraints="=v,v,~{memory}",
        args=[tle.gpu.set_layout(tl.load(tle.gpu.local_ptr(tail)), C)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    source = (local[:, None, None] & 15) // 4 + groups[None, None, :] * 4
    raw = (
        source * 256
        + (((local[:, None, None] & 3) * 8) ^ ((source & 7) * 8)) * 2
        + halves[None, :, None] * 8192
        + zero
    )
    base = raw
    pairs = tle.gpu.set_layout(tl.arange(0, 32), tle.gpu.SlicedEncoding(1, W))
    features = dt * 64 + tle.gpu.set_layout(
        tl.arange(0, 64), tle.gpu.SlicedEncoding(0, W)
    )
    zero = tle.gpu.set_layout(tl.zeros((32, 2, 4), tl.int32), C)
    state = (zero,) * 64
    groups_total = tl.cdiv(pages, PAGE_GROUP)
    groups_per_part = tl.cdiv(groups_total, PARTS)
    first_group = part * groups_per_part
    last_group = tl.minimum(groups_total, first_group + groups_per_part)
    for group in tl.range(first_group, last_group, loop_unroll_factor=1):
        tl.debug_barrier()
        for page_offset in tl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            page = tl.load(
                Table + batch * TABLE_STRIDE + page_idx, page_idx < pages, 0
            ).to(tl.int64)
            token = 2 * pairs
            first = tl.load(
                KV
                + page * (KV_STRIDES[0] // 4)
                + (token[:, None] * (KV_STRIDES[1] // 4) + features[None, :]),
                page_idx * 64 + token[:, None] < length,
                0,
            )
            second = tl.load(
                KV
                + page * (KV_STRIDES[0] // 4)
                + ((token[:, None] + 1) * (KV_STRIDES[1] // 4) + features[None, :]),
                page_idx * 64 + token[:, None] + 1 < length,
                0,
            )
            low_pair = interleave_mla_pv_bytes(first, second, 0x05010400)
            high_pair = interleave_mla_pv_bytes(first, second, 0x07030602)
            packed = (
                tl.join(low_pair, high_pair).reshape((32, 2, 64)).permute((1, 0, 2))
            )
            tl.store(
                tle.gpu.set_layout(
                    tle.gpu.local_ptr(
                        tle.gpu.alloc(
                            shape=(2, 32, 64),
                            dtype=tl.uint32,
                            layout=tle.gpu.swizzled_shared_layout(
                                4, 1, 8, [2, 1, 0], 1, [1, 1, 1], [1, 1, 1], [0, 1, 2]
                            ),
                            alias=shared,
                            alias_offset_bytes=(page_offset * 2) * 8192,
                        )
                    ),
                    tle.gpu.BlockEncoding([1, 1, 8], [2, 4, 8], [1, 4, 1], [2, 0, 1]),
                ),
                tle.gpu.set_layout(
                    packed,
                    tle.gpu.BlockEncoding([1, 1, 8], [2, 4, 8], [1, 4, 1], [2, 0, 1]),
                ),
            )
        tl.debug_barrier()
        for page_offset in tl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            scale = tl.load(
                Weight + (batch * CAPACITY + page_idx) * mh + (ph),
                (ph < mh) & (page_idx < pages),
                0.0,
            )
            for half in tl.static_range(2):
                offset = (
                    ph[:, None, None] * 16
                    + half * 8
                    + pg[None, None, :] * 2
                    + tle.gpu.set_layout(tl.full((32, 2, 4), 0, tl.int32), C)
                )
                probability_base = Packed + (batch * CAPACITY + page_idx) * mh * 16
                p0, p1 = load_mla_probability_pair(
                    probability_base,
                    offset,
                    (ph[:, None, None] < mh) & (page_idx < pages),
                )
                h0, l0 = quantize_mla_p16_word(p0, scale[:, None, None])
                h1, l1 = quantize_mla_p16_word(p1, scale[:, None, None])
                high = h0.to(tl.uint64) | (h1.to(tl.uint64) << 32)
                low = l0.to(tl.uint64) | (l1.to(tl.uint64) << 32)
                state = accumulate_small_head_pv_limbs(
                    high, low, base + page_offset * 16384 + half * 4096, state
                )
    if PARTS == 1:
        factor = tl.load(RowFactor + batch * mh + heads, heads < mh, 0.0)
    else:
        HT: tl.constexpr = (H * SQ + 31) // 32
        tile = (batch * HT + ht) * 2 + dt
    for register in tl.static_range(16):
        he, ho, le, lo = (
            state[register],
            state[16 + register],
            state[32 + register],
            state[48 + register],
        )
        columns = register * 4 + groups
        if PARTS == 1:
            even = (he.to(tl.float32) * 128.0 + le.to(tl.float32)) * factor[
                :, None, None
            ]
            odd = (ho.to(tl.float32) * 128.0 + lo.to(tl.float32)) * factor[
                :, None, None
            ]
            word = pack_mla_output_pair(even, odd, Output.dtype.element_ty)
            offset = (
                (batch * mh + heads[:, None, None]) * 256
                + dt * 128
                + halves[None, :, None] * 64
                + columns[None, None, :]
            )
            tl.store(
                Output.to(tl.pointer_type(tl.uint32)) + offset,
                word,
                heads[:, None, None] < mh,
            )
        else:
            partial_base = Output + (tile * PARTS + part) * 8192
            byte_offsets = (
                halves[None, :, None] * 4096
                + groups[None, None, :] * 32
                + local[:, None, None]
            ) * 8
            store_mla_partial_limbs(
                partial_base,
                byte_offsets,
                he,
                le,
                ho,
                lo,
                heads[:, None, None] < mh,
                register * 1024,
                16384,
            )


class FlashMLAInt8PreparedHandle:
    """Reuse decode workspace; callers must serialize calls on one stream."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        splits: int,
        scale: float,
        causal: bool,
        automatic_splits: bool = False,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ):
        self.tensors = tensors
        query, rope = tensors[:2]
        self.batch, self.queries, self.heads, _ = query.shape
        self.scale, self.causal = scale, causal
        self.output = (
            torch.empty(
                (self.batch, self.queries, self.heads, 512),
                dtype=rope.dtype,
                device=query.device,
            )
            if out is None
            else out
        )
        self.lse = (
            torch.empty(
                (self.batch, self.heads, self.queries),
                dtype=torch.float32,
                device=query.device,
            )
            if lse is None
            else lse
        )
        self.capacity = tensors[6].shape[1]
        merged_heads = self.queries * self.heads
        # Bound the staged probability and page-statistics workspace to 4 GiB.
        staged_bytes = (
            self.batch * merged_heads * ((64 * 2 + 4 * 4) * self.capacity + 4)
        )
        self.staged = (
            (
                (splits == 1 and merged_heads >= 64)
                or (automatic_splits and merged_heads == 32 and self.batch >= 128)
            )
            and 64 <= self.capacity <= 2048
            and staged_bytes <= 4 * 1024**3
        )
        self.splits = 1 if self.staged else splits
        if self.staged:
            self.packed = torch.empty(
                (self.batch, self.capacity, merged_heads, 64),
                dtype=torch.int16,
                device=query.device,
            )
            self.staged_stats = torch.empty(
                (self.batch, self.capacity, merged_heads, 3),
                dtype=torch.float32,
                device=query.device,
            )
            self.weight = torch.empty(
                (self.batch, self.capacity, merged_heads),
                dtype=torch.float32,
                device=query.device,
            )
            self.rowfactor = torch.empty(
                (self.batch, merged_heads),
                dtype=torch.float32,
                device=query.device,
            )
            kv = tensors[2]
            has_aligned_kv_rows = (
                math.gcd(16, kv.data_ptr(), kv.stride(0), kv.stride(1)) == 16
            )
            self.qk_rebase_kv = (
                has_aligned_kv_rows and kv.untyped_storage().nbytes() >= 2**31
            )
            self.qk_rebase_packed = (
                has_aligned_kv_rows
                and self.packed.data_ptr() % 16 == 0
                and self.packed.untyped_storage().nbytes() >= 2**31
            )
        else:
            workspace_splits = splits if splits > 1 else 0
            self.partial = torch.empty(
                (workspace_splits, self.batch, self.queries * self.heads, 512),
                dtype=torch.float32,
                device=query.device,
            )
            self.stats = torch.empty(
                (workspace_splits, 2, self.batch, self.queries * self.heads),
                dtype=torch.float32,
                device=query.device,
            )

        self.native_pv_kernel = (
            small_head_int8_mla_pv if merged_heads == 32 else native_int8_mla_pv
        )
        self.native_pv_merge_kernel = merge_mla_pv_integer
        self.has_native_pv = False
        kv, table, lengths = tensors[2], tensors[6], tensors[7]
        if (
            self.staged
            and (merged_heads == 32 or merged_heads >= 64)
            and rope.dtype == torch.bfloat16
            and self.output.data_ptr() % 16 == 0
            and self.batch * triton.cdiv(merged_heads, 64) * 2
            >= _get_num_sms(query.device)
            and 63 * kv.stride(1) + 512 < 2**31
            and kv.stride(-1) == 1
            and kv.data_ptr() % 16 == 0
            and kv.stride(0) % 16 == 0
            and kv.stride(1) % 16 == 0
            and torch.cuda.get_device_properties(query.device).gcnArchName.startswith(
                "gfx936"
            )
        ):
            # Typed no-copy views avoid pointer bitcasts that block buffer lowering.
            self.native_kv = kv.view(kv.shape[0], 64, 512).view(torch.int32)
            self.native_packed = self.packed.view(torch.int64)
            with torch_device_fn.device(query.device):
                binary = self.native_pv_kernel.warmup(
                    self.native_kv,
                    table,
                    lengths,
                    self.native_packed,
                    self.weight,
                    self.rowfactor,
                    self.output,
                    kv.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    grid=(2, triton.cdiv(merged_heads, 64), self.batch),
                    num_warps=4,
                    num_stages=1,
                )
            # A full-span allocation of exactly 16 KiB fixes the native LDS base.
            self.has_native_pv = (
                binary.metadata.shared == 16384 and binary.metadata.warp_size == 64
            )

        # Private PV partitions preserve the public num_splits contract.
        self.pv_splits = 1
        partial_workspace_bytes = 0
        pv_heads = 32 if merged_heads == 32 else 64
        minimum_pages = 64 if merged_heads == 32 else 256
        if self.has_native_pv and self.capacity >= minimum_pages:
            candidate_splits = 3
            cta_count = self.batch * triton.cdiv(merged_heads, pv_heads) * 2
            # Require enough partitioning benefit to cover the merge overhead.
            group_size = (6 if merged_heads == 32 else 3) * _get_num_sms(query.device)
            groups = triton.cdiv(cta_count, group_size)
            split_groups = triton.cdiv(cta_count * candidate_splits, group_size)
            partial_bytes = (
                8
                * candidate_splits
                * self.batch
                * (pv_heads * triton.cdiv(merged_heads, pv_heads))
                * 512
            )
            if (
                groups * candidate_splits * 100 >= split_groups * 135
                and staged_bytes + partial_bytes <= 4 * 1024**3
            ):
                tiles = 2 * self.batch * triton.cdiv(merged_heads, pv_heads)
                partial_shape = (
                    (tiles, candidate_splits, 2, 2, 64, 32)
                    if merged_heads == 32
                    else (tiles, candidate_splits, 2, 128, 64)
                )
                self.pv_partials = torch.empty(
                    partial_shape,
                    dtype=torch.int64,
                    device=query.device,
                )
                partial_workspace_bytes = partial_bytes
                with torch_device_fn.device(query.device):
                    binary = self.native_pv_kernel.warmup(
                        self.native_kv,
                        table,
                        lengths,
                        self.native_packed,
                        self.weight,
                        self.rowfactor,
                        self.pv_partials,
                        kv.stride(),
                        table.stride(0),
                        self.capacity,
                        lengths.stride(0),
                        self.heads,
                        self.queries,
                        candidate_splits,
                        grid=(
                            2 * candidate_splits,
                            triton.cdiv(merged_heads, 64),
                            self.batch,
                        ),
                        num_warps=4,
                        num_stages=1,
                    )
                    binary._init_handles()
                if (
                    binary.metadata.shared == 16384
                    and binary.metadata.warp_size == 64
                    and binary.n_spills == 0
                ):
                    self.pv_splits = candidate_splits

        self.pv_order = None
        if (
            self.has_native_pv
            and 32 <= self.batch <= 1024
            and staged_bytes + partial_workspace_bytes + self.batch * 4 <= 4 * 1024**3
        ):
            self.pv_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=query.device
            )
            native_output = self.pv_partials if self.pv_splits > 1 else self.output
            with torch_device_fn.device(query.device):
                binary = self.native_pv_kernel.warmup(
                    self.native_kv,
                    table,
                    lengths,
                    self.native_packed,
                    self.weight,
                    self.rowfactor,
                    native_output,
                    kv.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    self.pv_splits,
                    Order=self.pv_order,
                    grid=(
                        2 * self.pv_splits,
                        triton.cdiv(merged_heads, pv_heads),
                        self.batch,
                    ),
                    num_warps=4,
                    num_stages=1,
                )
            if binary.metadata.shared != 16384 or binary.metadata.warp_size != 64:
                self.pv_order = None

        self.pv_head_tile = pv_heads
        self.pv_warps = 4
        if (
            self.has_native_pv
            and self.pv_splits == 1
            and merged_heads >= 128
            and merged_heads % 128 == 0
            and 2 * self.batch * triton.cdiv(merged_heads, 128)
            >= 2 * _get_num_sms(query.device)
        ):
            with torch_device_fn.device(query.device):
                binary = native_int8_mla_pv_wide.warmup(
                    self.native_kv,
                    table,
                    lengths,
                    self.native_packed,
                    self.weight,
                    self.rowfactor,
                    self.output,
                    kv.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    1,
                    Order=self.pv_order,
                    NORMALIZED=True,
                    grid=(2, triton.cdiv(merged_heads, 128), self.batch),
                    num_warps=8,
                    num_stages=1,
                )
                binary._init_handles()
            if (
                binary.metadata.shared == 32768
                and binary.metadata.warp_size == 64
                and binary.n_spills == 0
                and binary.n_regs <= 192
            ):
                self.native_pv_kernel = native_int8_mla_pv_wide
                self.pv_head_tile = 128
                self.pv_warps = 8

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        with torch_device_fn.device(query.device):
            if self.staged:
                merged_heads = self.queries * self.heads
                page_group = 4 if merged_heads > 128 else 1
                qk_heads = min(128, triton.next_power_of_2(merged_heads))
                qk_block = 256 if merged_heads in (32, 64) else 128
                pv_heads = self.pv_head_tile
                _staged_int8_mla_qk[
                    (
                        triton.cdiv(merged_heads, qk_heads),
                        triton.cdiv(self.capacity, page_group),
                        self.batch,
                    )
                ](
                    *self.tensors,
                    self.packed,
                    self.staged_stats,
                    query.stride(),
                    rope.stride(),
                    kv.stride(),
                    kv_rope.stride(),
                    qs.stride(),
                    ks.stride(),
                    table.stride(0),
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    self.scale,
                    self.causal,
                    qk_heads,
                    qk_block,
                    page_group,
                    REBASE_KV=self.qk_rebase_kv,
                    REBASE_PACKED=self.qk_rebase_packed,
                    KV_ALIGNMENT=16,
                    PACKED_ALIGNMENT=16,
                    num_warps=4,
                    num_stages=1,
                )
                normalize_pages = triton.next_power_of_2(self.capacity)
                if normalize_pages <= 1024:
                    # Coalesce head rows while bounding live page statistics.
                    normalize_heads = min(
                        64,
                        triton.next_power_of_2(merged_heads),
                        max(16, 8192 // normalize_pages),
                    )
                else:
                    normalize_heads = 1
                _staged_int8_mla_normalize[
                    (triton.cdiv(merged_heads, normalize_heads), self.batch)
                ](
                    self.staged_stats,
                    self.weight,
                    self.rowfactor,
                    self.lse,
                    lengths,
                    self.capacity,
                    lengths.stride(0),
                    self.heads,
                    self.queries,
                    normalize_pages,
                    BH=normalize_heads,
                    B=self.batch,
                    Order=self.pv_order,
                    ORDER_BLOCK=triton.next_power_of_2(self.batch),
                    num_warps=4,
                )
                if self.has_native_pv:
                    if self.pv_splits > 1:
                        self.native_pv_kernel[
                            (
                                2 * self.pv_splits,
                                triton.cdiv(merged_heads, 64),
                                self.batch,
                            )
                        ](
                            self.native_kv,
                            table,
                            lengths,
                            self.native_packed,
                            self.weight,
                            self.rowfactor,
                            self.pv_partials,
                            kv.stride(),
                            table.stride(0),
                            self.capacity,
                            lengths.stride(0),
                            self.heads,
                            self.queries,
                            self.pv_splits,
                            Order=self.pv_order,
                            num_warps=4,
                            num_stages=1,
                        )
                        # Both native layouts merge 4096 output elements per CTA.
                        merge_cols = 64 if pv_heads == 32 else 32
                        self.native_pv_merge_kernel[
                            (
                                128 // merge_cols,
                                2 * self.batch * triton.cdiv(merged_heads, 64),
                            )
                        ](
                            self.pv_partials,
                            self.rowfactor,
                            self.output,
                            merged_heads,
                            self.pv_splits,
                            merge_cols,
                            num_warps=4,
                        )
                    else:
                        self.native_pv_kernel[
                            (
                                2,
                                triton.cdiv(merged_heads, self.pv_head_tile),
                                self.batch,
                            )
                        ](
                            self.native_kv,
                            table,
                            lengths,
                            self.native_packed,
                            self.weight,
                            self.rowfactor,
                            self.output,
                            kv.stride(),
                            table.stride(0),
                            self.capacity,
                            lengths.stride(0),
                            self.heads,
                            self.queries,
                            self.pv_splits,
                            Order=self.pv_order,
                            num_warps=self.pv_warps,
                            num_stages=1,
                            **({"NORMALIZED": True} if self.pv_warps == 8 else {}),
                        )
                else:
                    _staged_int8_mla_requant[
                        (triton.cdiv(merged_heads, 32), self.capacity, self.batch)
                    ](
                        self.packed,
                        self.weight,
                        lengths,
                        self.capacity,
                        lengths.stride(0),
                        self.heads,
                        self.queries,
                        32,
                        num_warps=4,
                    )
                    _staged_int8_mla_pv[
                        (2, triton.cdiv(merged_heads, pv_heads), self.batch)
                    ](
                        kv,
                        table,
                        lengths,
                        self.packed,
                        self.rowfactor,
                        self.output,
                        kv.stride(),
                        table.stride(0),
                        self.capacity,
                        lengths.stride(0),
                        self.heads,
                        self.queries,
                        pv_heads,
                        256,
                        num_warps=4,
                        num_stages=1,
                    )
                return self.output, self.lse
            dense_int8_mla_kernel[
                lambda meta: (
                    triton.cdiv(self.queries * self.heads, meta["BH"]),
                    self.batch,
                    self.splits,
                )
            ](
                *self.tensors,
                self.partial,
                self.stats,
                self.output,
                self.lse,
                query.stride(),
                rope.stride(),
                kv.stride(),
                kv_rope.stride(),
                qs.stride(),
                ks.stride(),
                table.stride(0),
                lengths.stride(0),
                self.batch,
                self.heads,
                self.splits,
                self.scale,
                self.queries,
                self.causal,
            )
            if self.splits > 1:
                dense_int8_mla_merge[(self.batch, self.queries * self.heads)](
                    self.partial,
                    self.stats,
                    self.output,
                    self.lse,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.splits,
                    num_warps=2,
                )
        return self.output, self.lse


def prepare_flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[FlashMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Decode 64-token pages; RoPE is divided by the content scale as in PR851.

    Active page IDs and lengths must be valid, scales finite and positive.
    Causal queries attend the aligned suffix of the valid cache.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    if q_nope.device.type != "cuda":
        raise ValueError("requires Hygon device tensors")
    batch, queries, heads, _ = q_nope.shape
    if (
        k_lora.ndim not in (3, 4)
        or k_lora.shape[1] != 64
        or k_lora.shape[-1] != 512
        or (k_lora.ndim == 4 and k_lora.shape[2] != 1)
        or head_dim_v != 512
    ):
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
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
        if tensor.shape != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
    ):
        raise ValueError("block_table must have shape [batch, positive page capacity]")
    if q_nope.dtype != torch.int8 or k_lora.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
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
    for tensor in tensors:
        if tensor.device != q_nope.device or (
            tensor.numel() != 0 and tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all tensors must share a device and contiguous last dimension"
            )
    if not all(
        can_use_int32_index(tensor)
        for tensor in (q_nope, q_rope, q_scale, block_table, cache_seqlens)
    ):
        raise NotImplementedError("query and metadata offsets must fit signed INT32")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a Python number")
    scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if num_splits is None:
        splits = min(
            16,
            triton.next_power_of_2(
                max(
                    1,
                    triton.cdiv(
                        (8 if queries * heads == 16 else 4)
                        * _get_num_sms(q_nope.device),
                        max(1, batch * triton.cdiv(queries * heads, 16)),
                    ),
                )
            ),
        )
        if queries * heads == 16:
            # Limit page work per partition for long requests with small heads.
            length_splits = min(
                16, triton.next_power_of_2(triton.cdiv(block_table.shape[1], 16))
            )
            # Preserve the supported range of the original workspace check.
            if length_splits * batch * queries * heads * 512 * 4 < 2**31:
                splits = max(splits, length_splits)
        splits = min(splits, max(1, triton.next_power_of_2(block_table.shape[1])))
    else:
        splits = num_splits
    if (
        isinstance(splits, bool)
        or not isinstance(splits, int)
        or splits not in (1, 2, 4, 8, 16)
    ):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    validate_dense_outputs(out, lse, q_nope, q_rope.dtype, head_dim_v)
    output_bytes = batch * queries * heads * 512 * q_rope.element_size()
    partial_bytes = (splits if splits > 1 else 0) * batch * queries * heads * 512 * 4
    if max(output_bytes, partial_bytes) > 2**31 - 1:
        raise NotImplementedError(
            "output and partial workspaces must fit signed INT32 indexing"
        )
    handle = FlashMLAInt8PreparedHandle(
        tensors,
        splits,
        scale,
        causal,
        automatic_splits=num_splits is None,
        out=out,
        lse=lse,
    )
    return handle, handle()


def flash_mla_with_kvcache_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: object | None = None,
    num_splits: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tile_scheduler_metadata is not None:
        raise NotImplementedError("HCU does not accept external scheduler metadata")
    _, outputs = prepare_flash_mla_with_kvcache_w8a8(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        softmax_scale=softmax_scale,
        causal=causal,
        num_splits=num_splits,
        out=out,
        lse=lse,
    )
    return outputs
