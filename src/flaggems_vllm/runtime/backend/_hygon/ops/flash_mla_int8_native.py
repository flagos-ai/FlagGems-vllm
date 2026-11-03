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


import triton
import triton.experimental.tle.language as tle
import triton.language as tl


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
    BATCH: tl.constexpr = 0,
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
