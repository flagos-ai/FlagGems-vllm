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


"""AIU loads and MMA kernels for packed expert tiles on T-Head."""

import triton
import triton.language as tl

from flaggems_vllm.runtime.backend._thead.fused.marlin_fragments import (
    _TL_QUANT_TYPE_FP4_E2M1,
    _TL_QUANT_TYPE_UINT4B8,
    _TL_QUANT_TYPE_UINT8B128,
    decode_fp8_pair,
    decode_i4_register,
    decode_int8_fragments,
    decode_mx_register,
    e8m0_to_bf16_bits,
    fold_fp8_scale_fragment,
    tle_async,
)
from flaggems_vllm.utils import libentry

W8_TILE_BYTES = tl.constexpr(128 * 128)
W8_HALF_TILE_BYTES = tl.constexpr(64 * 128)
EXPERT_ID_MASK = tl.constexpr(0x3FFFFFFF)
SHORT_TILE_FLAG = tl.constexpr(0x40000000)


@triton.jit
def load_w8_packed_scales(ptr, stride_n, FOLD: tl.constexpr):
    words = ptr.to(tl.pointer_type(tl.uint32))
    first = tl.load(words)
    second = tl.load(words + 4 * stride_n)
    if FOLD:
        first = fold_fp8_scale_fragment(first)
        second = fold_fp8_scale_fragment(second)
    return first, second


@triton.jit
def w8_scale_fragment(scale, k_tile, PACKED: tl.constexpr, FOLD: tl.constexpr):
    if PACKED:
        bits = (scale >> (16 * k_tile)) & 0xFFFF
    else:
        bits = scale.to(tl.uint16, bitcast=True).to(tl.uint32)
    fragment = bits | (bits << 16)
    if FOLD and not PACKED:
        fragment = fold_fp8_scale_fragment(fragment)
    return fragment


@triton.jit
def mma_bf16_m16n8(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    return tl.inline_asm_elementwise(
        (
            "ppu.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
        ),
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        [a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def copy_w8_tile_async(
    dst, src, num_rows: tl.constexpr, KS, KT: tl.constexpr, start, copy_token
):
    # AIU copies are collective within the first warp.
    return tl.inline_asm_elementwise(
        (
            "{ .reg .u32 tid; .reg .pred p; ppu.mov.u32 tid,%tid.x; ppu.setp.lt.u32 p,tid,32;"
            " @p ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill.2d.b8 "
            "[$1],[$2],{1,$3,$4,0,0,$5},{1,$3,$6,1}; ppu.mov.b32 $0,$7; }"
        ),
        "=r,r,l,r,r,r,r,r,~{memory}",
        [dst, src, num_rows, KS, start, KT, copy_token],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def prefetch_activations(
    activation_smem,
    activations,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    ROWS: tl.constexpr,
    PAIRED_ROWS: tl.constexpr = False,
):
    activation_base = (
        activations.to(tl.pointer_type(tl.uint8))
        + row_tile.to(tl.int64) * ROWS * reduction_size * 2
    )
    if PAIRED_ROWS:
        # Two consecutive K16 fragments of eight routes form one 16-row AIU tile.
        tl.static_assert(ROWS == 8)
        copy_token = copy_w8_tile_async(
            activation_smem,
            activation_base,
            16,
            reduction_size,
            128,
            k_tile * 128,
            copy_token,
        )
    else:
        for half in tl.static_range(2):
            copy_token = copy_w8_tile_async(
                activation_smem + half * ROWS * 128,
                activation_base,
                ROWS,
                reduction_size * 2,
                128,
                k_tile * 256 + half * 128,
                copy_token,
            )
    return copy_token


@triton.jit
def prefetch_w8_tile(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    ROWS: tl.constexpr,
    IS_SPLIT_Q: tl.constexpr = False,
    PAIRED_ROWS: tl.constexpr = False,
):
    copy_token = prefetch_activations(
        activation_smem,
        activations,
        reduction_size,
        row_tile,
        k_tile,
        copy_token,
        ROWS,
        PAIRED_ROWS,
    )
    if IS_SPLIT_Q:
        # Split the two warp groups into independent 8 KiB swizzled blocks.
        copy_token = copy_w8_tile_async(
            weight_smem, weight_bytes, 64, QN, 128, k_tile * 128, copy_token
        )
        copy_token = copy_w8_tile_async(
            weight_smem + W8_HALF_TILE_BYTES,
            weight_bytes + 64 * QN,
            64,
            QN,
            128,
            k_tile * 128,
            copy_token,
        )
    else:
        copy_token = copy_w8_tile_async(
            weight_smem, weight_bytes, 128, QN, 128, k_tile * 128, copy_token
        )
    return issue_copy_group(copy_token, "ppu.cp.async.commit_group;")


@triton.jit
def issue_copy_group(copy_token, instruction: tl.constexpr):
    return tl.inline_asm_elementwise(
        instruction + " ppu.mov.b32 $0,$1;",
        "=r,r,~{memory}",
        [copy_token],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_w8_tile(copy_token, KEEP: tl.constexpr):
    instruction: tl.constexpr = (
        "ppu.cp.async.wait_group 1;" if KEEP else "ppu.cp.async.wait_group 0;"
    )
    token = issue_copy_group(copy_token, instruction)
    tl.debug_barrier()
    return token


@triton.jit
def mma_bf16_square(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7):
    return tl.inline_asm_elementwise(
        (
            "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
            "{$0,$1,$2,$3,$4,$5,$6,$7},{$8,$9,$10,$11},{$12,$13,$14,$15},{$16,$17,$18,$19,$20"
            ",$21,$22,$23};"
        ),
        "=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f",
        [a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7],
        dtype=(tl.float32,) * 8,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_matrix_x4(shared, row, CW: tl.constexpr, channel, copy_token):
    # PPU x4 has row0-low, row0-high, row8-low, row8-high register order.
    return tl.inline_asm_elementwise(
        (
            "ppu.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 "
            "{$0,$1,$2,$3},[$4],{$5,$6,$7,$8,$9,$10};"
        ),
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,~{memory}",
        [shared, 0, row, 1, CW, 1, channel, copy_token],
        dtype=(tl.uint32,) * 4,
        is_pure=False,
        pack=1,
    )


@triton.jit
def store_k32(k):
    # FP4 has eight K values per loaded word; its permutation is self-inverse.
    return (k & ~31) | (k & 1) | ((k & 6) << 2) | ((k & 24) >> 2)


@triton.jit
def mx_scale_source(ptr, PACKED_SCALE: tl.constexpr):
    if PACKED_SCALE:
        return tl.load(ptr.to(tl.pointer_type(tl.uint32)))
    else:
        return ptr


@triton.jit
def mx_scale_fragment(
    source, stride_group, group: tl.constexpr, PACKED_SCALE: tl.constexpr
):
    if PACKED_SCALE:
        byte = (source >> (group * 8)) & 255
    else:
        byte = tl.load(source + group * stride_group).to(tl.uint32)
    bits = e8m0_to_bf16_bits(byte)
    return bits | (bits << 16)


@triton.jit
def decode_w4_fragment_pair(q0, q8, s0, s1, half: tl.constexpr, MX: tl.constexpr):
    lo, hi = q0 >> (half * 16), q8 >> (half * 16)
    if MX:
        a0, a1 = decode_mx_register(lo, s0, s0)
        a2, a3 = decode_mx_register(hi, s1, s1)
    else:
        codes = tl.join(lo, hi)
        scales = tl.join(s0, s1)
        low, high = decode_i4_register(codes, scales, scales)
        a0, a2 = tl.split(low)
        a1, a3 = tl.split(high)
    return a0, a1, a2, a3


@triton.jit
def prefetch_w4_square(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    ROWS: tl.constexpr,
):
    copy_token = copy_w8_tile_async(
        weight_smem, weight_bytes, 64, QN, 64, k_tile * 64, copy_token
    )
    copy_token = prefetch_activations(
        activation_smem,
        activations,
        reduction_size,
        row_tile,
        k_tile,
        copy_token,
        ROWS,
    )
    return issue_copy_group(copy_token, "ppu.cp.async.commit_group;")


@triton.jit
def store_k16(k):
    # The permutation stays inside a scale group and avoids weight repacking.
    return (k & ~15) | (k & 1) | ((k & 2) << 2) | ((k & 12) >> 1)


@triton.jit
def load_activation_fragment_x2(shared, channel: tl.constexpr, lane):
    # The AIU swizzle rotates four-row groups and swaps the two 16-byte halves.
    sector = channel // 32
    skew = ((sector & 1) << 1) | (sector >> 1)
    row = lane // 4
    rotated_row = (row & ~3) | ((row + skew) & 3)
    word = (lane % 4) ^ ((row // 4) * 4)
    offset = sector * 64 + rotated_row * 8 + word
    ptr = shared.to(tl.uint64).to(tl.pointer_type(tl.uint32, 3))
    return tl.load(ptr + offset), tl.load(ptr + (offset ^ 4))


@triton.jit
def w8_slot_address(buffer, stage):
    return tle_async.gpu.local_ptr(buffer, (stage, 0)).to(tl.int64).to(tl.int32)


@triton.jit
def square_accumulate(
    qmem,
    amem,
    weight_bytes,
    activations,
    scales,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    expert,
    n0,
    warp,
    SE,
    SG,
    SN,
    FP8: tl.constexpr,
    FAST: tl.constexpr,
    initial_dep=None,
    SCALE_PAIR: tl.constexpr = False,
):
    if initial_dep is None:
        copy_token = tl.full((256,), 0, tl.int32)
        copy_token = prefetch_w8_tile(
            w8_slot_address(qmem, 0),
            w8_slot_address(amem, 0),
            weight_bytes,
            activations,
            QN,
            reduction_size,
            row_tile,
            0,
            copy_token,
            16,
            IS_SPLIT_Q=True,
        )
    else:
        copy_token = initial_dep
    c0 = tl.full((256,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    c4 = c0
    c5 = c0
    c6 = c0
    c7 = c0
    # One packed BF16 load covers two successive K128 groups.
    tiles_per_scale_load: tl.constexpr = 2 if SCALE_PAIR else 1
    unroll: tl.constexpr = (
        2 if FP8 and reduction_size % 256 == 0 and not SCALE_PAIR else 1
    )
    for chunk in tl.range(
        reduction_size // (tiles_per_scale_load * 128), loop_unroll_factor=unroll
    ):
        if SCALE_PAIR:
            scale0, scale1 = load_w8_packed_scales(
                scales + expert * SE + n0 * SN + chunk * 2 * SG, SN, FAST
            )
        for inner_k in tl.static_range(tiles_per_scale_load):
            k_tile = chunk * tiles_per_scale_load + inner_k
            if not SCALE_PAIR:
                scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
                scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
            if k_tile + 1 < reduction_size // 128:
                copy_token = prefetch_w8_tile(
                    w8_slot_address(qmem, (k_tile + 1) % 2),
                    w8_slot_address(amem, (k_tile + 1) % 2),
                    weight_bytes,
                    activations,
                    QN,
                    reduction_size,
                    row_tile,
                    k_tile + 1,
                    copy_token,
                    16,
                    IS_SPLIT_Q=True,
                )
                copy_token = wait_w8_tile(copy_token, True)
            else:
                copy_token = wait_w8_tile(copy_token, False)
            s0 = w8_scale_fragment(scale0, inner_k, SCALE_PAIR, FP8 and FAST)
            s1 = w8_scale_fragment(scale1, inner_k, SCALE_PAIR, FP8 and FAST)
            qhere = w8_slot_address(qmem, k_tile % 2)
            ahere = w8_slot_address(amem, k_tile % 2)
            for group in tl.static_range(4):
                (r0, r1, r2, r3) = load_matrix_x4(
                    qhere + (warp // 4) * W8_HALF_TILE_BYTES,
                    (warp % 4) * 16,
                    64,
                    group * 32,
                    copy_token,
                )
                for part in tl.static_range(2):
                    q0 = r0 if part == 0 else r1
                    q8 = r2 if part == 0 else r3
                    sub = group * 2 + part
                    if FP8:
                        a0, a1, a2, a3 = decode_fp8_pair(q0, q8, s0, s1, FAST)
                        (b0, b1, b2, b3) = load_matrix_x4(
                            ahere + sub // 4 * 2048, 0, 16, sub % 4 * 32, copy_token
                        )
                    else:
                        (b0, b1, b2, b3) = load_matrix_x4(
                            ahere + sub // 4 * 2048, 0, 16, sub % 4 * 32, copy_token
                        )
                        a0, a1, a2, a3 = decode_int8_fragments(q0, q8, s0, s1)
                    (c0, c1, c2, c3, c4, c5, c6, c7) = mma_bf16_square(
                        a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7
                    )
            tl.debug_barrier()
    return (c0, c1, c2, c3, c4, c5, c6, c7)


@triton.jit
def marlin_moe_gemm_grouped_square_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    scale_flags,
    FP8: tl.constexpr,
    USE32: tl.constexpr = False,
    PAIRED_DOWN: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
    PREFETCH_ROUTES: tl.constexpr = False,
    PACK_WIDTH: tl.constexpr = False,
):
    tl.static_assert(num_columns % 128 == 0 and reduction_size % 128 == 0)
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 128)
    col_tile = pid % (num_columns // 128)
    if row_tile * 16 >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert_word = tl.load(block_experts + row_tile)
    if PACK_WIDTH:
        expert = (expert_word & EXPERT_ID_MASK).to(offset_type)
        short_tile = (expert_word & SHORT_TILE_FLAG) != 0
    else:
        expert = expert_word.to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    lane_group = lane // 4
    lane_in_group = lane % 4
    n0 = col_tile * 128 + warp * 16 + lane_group
    qmem = tle_async.gpu.alloc(
        [2, W8_TILE_BYTES],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    amem = tle_async.gpu.alloc(
        [2, 16 * 128 * 2],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    weight_smem = w8_slot_address(qmem, 0)
    activation_smem = w8_slot_address(amem, 0)
    weight_base = weight_bytes + expert * QE + col_tile * 128 * QN
    inner_n = warp * 16 + lane_group
    if not PACK_WIDTH:
        short_tile = tl.load(route_ids + row_tile * 16 + 8) >= VALID
    if short_tile:
        if PAIRED_DOWN:
            copy_token = tl.full((256,), 0, tl.int32)
            copy_token = prefetch_w8_tile(
                weight_smem,
                activation_smem,
                weight_base,
                activations,
                QN,
                reduction_size,
                row_tile * 2,
                0,
                copy_token,
                8,
                True,
                PAIRED_ROWS=True,
            )
        else:
            copy_token = None
        if PREFETCH_ROUTES:
            early0 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group)
            early1 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group + 1)
            if MUL:
                w0 = tl.load(
                    router_weights + early0.to(offset_type), early0 < VALID, other=0
                ).to(tl.float32)
                w1 = tl.load(
                    router_weights + early1.to(offset_type), early1 < VALID, other=0
                ).to(tl.float32)
        tile_inputs = (
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            scales,
            QN,
        )
        tile_coordinates = (row_tile * 2, expert, n0, inner_n, SE, SG, SN)
        if FP8:
            can_fold = (
                tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
            )
            if can_fold:
                (c0, c1, c2, c3) = w8_narrow_accumulate(
                    *tile_inputs,
                    reduction_size,
                    *tile_coordinates,
                    True,
                    True,
                    IS_SPLIT_Q=True,
                    initial_dep=copy_token,
                    PAIRED_ROWS=PAIRED_DOWN,
                    SCALE_PAIR=SCALE_PAIR,
                )
            else:
                (c0, c1, c2, c3) = w8_narrow_accumulate(
                    *tile_inputs,
                    reduction_size,
                    *tile_coordinates,
                    True,
                    False,
                    IS_SPLIT_Q=True,
                    initial_dep=copy_token,
                    PAIRED_ROWS=PAIRED_DOWN,
                    SCALE_PAIR=SCALE_PAIR,
                )
        else:
            (c0, c1, c2, c3) = w8_narrow_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                False,
                False,
                IS_SPLIT_Q=True,
                initial_dep=copy_token,
                PAIRED_ROWS=PAIRED_DOWN,
            )
        if PREFETCH_ROUTES:
            token0 = early0.to(offset_type)
            token1 = early1.to(offset_type)
        else:
            token0 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group).to(
                offset_type
            )
            token1 = tl.load(route_ids + row_tile * 16 + 2 * lane_in_group + 1).to(
                offset_type
            )
        if MUL:
            if not PREFETCH_ROUTES:
                w0 = tl.load(router_weights + token0, token0 < VALID, other=0).to(
                    tl.float32
                )
                w1 = tl.load(router_weights + token1, token1 < VALID, other=0).to(
                    tl.float32
                )
        else:
            w0 = 1.0
            w1 = 1.0
        c0 *= w0
        c2 *= w0
        c1 *= w1
        c3 *= w1
        tl.store(
            output_tile + token0 * CM + n0 * CN,
            c0.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + n0 * CN,
            c1.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
        tl.store(
            output_tile + token0 * CM + (n0 + 8) * CN,
            c2.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + (n0 + 8) * CN,
            c3.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
        return
    if PAIRED_DOWN:
        copy_token = tl.full((256,), 0, tl.int32)
        copy_token = prefetch_w8_tile(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            QN,
            reduction_size,
            row_tile,
            0,
            copy_token,
            16,
            IS_SPLIT_Q=True,
        )
    else:
        copy_token = None
    if PREFETCH_ROUTES:
        tokens = ()
        weights = ()
        for j in tl.static_range(4):
            token = tl.load(route_ids + row_tile * 16 + lane_in_group + 4 * j)
            tokens += (token,)
            if MUL:
                weight = tl.load(
                    router_weights + token.to(offset_type), token < VALID, other=0
                ).to(tl.float32)
                weights += (weight,)
    tile_inputs = (qmem, amem, weight_base, activations, scales, QN)
    tile_coordinates = (row_tile, expert, n0, warp, SE, SG, SN)
    if FP8:
        can_fold = (
            tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
        )
        if can_fold:
            c = square_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                True,
                initial_dep=copy_token,
                SCALE_PAIR=SCALE_PAIR,
            )
        else:
            c = square_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                False,
                initial_dep=copy_token,
                SCALE_PAIR=SCALE_PAIR,
            )
    else:
        c = square_accumulate(
            *tile_inputs,
            reduction_size,
            *tile_coordinates,
            False,
            False,
            initial_dep=copy_token,
        )
    for j in tl.static_range(4):
        if PREFETCH_ROUTES:
            token = tokens[j].to(offset_type)
        else:
            token = tl.load(route_ids + row_tile * 16 + lane_in_group + 4 * j).to(
                offset_type
            )
        v0 = c[j]
        v1 = c[j + 4]
        if MUL:
            if PREFETCH_ROUTES:
                w = weights[j]
            else:
                w = tl.load(router_weights + token, token < VALID, other=0).to(
                    tl.float32
                )
            v0 *= w
            v1 *= w
        tl.store(
            output_tile + token * CM + n0 * CN,
            v0.to(output_tile.dtype.element_ty),
            token < VALID,
        )
        tl.store(
            output_tile + token * CM + (n0 + 8) * CN,
            v1.to(output_tile.dtype.element_ty),
            token < VALID,
        )


@libentry()
@triton.jit
def raw_w4_square_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    MX: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    ROWS: tl.constexpr = 16,
    USE32: tl.constexpr = False,
):
    tl.static_assert(num_columns % 64 == 0 and reduction_size % 128 == 0)
    tl.static_assert(ROWS == 8 or ROWS == 16)
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 64)
    col_tile = pid % (num_columns // 64)
    if row_tile * ROWS >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(block_experts + row_tile).to(offset_type)
    thread = tl.arange(0, 128)
    lane = thread % 32
    warp = thread // 32
    lane_group = lane // 4
    lane_in_group = lane % 4
    n0 = col_tile * 64 + warp * 16 + lane_group
    qm = tle_async.gpu.alloc(
        [2 * 64 * 64],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    am = tle_async.gpu.alloc(
        [ROWS * 512],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    weight_smem = tle_async.gpu.local_ptr(qm, (0,)).to(tl.int64).to(tl.int32)
    activation_smem = tle_async.gpu.local_ptr(am, (0,)).to(tl.int64).to(tl.int32)
    weight_base = weight_bytes + expert * QE + col_tile * 64 * QN
    copy_token = tl.full((128,), 0, tl.int32)
    copy_token = prefetch_w4_square(
        weight_smem,
        activation_smem,
        weight_base,
        activations,
        QN,
        reduction_size,
        row_tile,
        0,
        copy_token,
        ROWS,
    )
    c0 = tl.full((128,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    c4 = c0
    c5 = c0
    c6 = c0
    c7 = c0
    for k_tile in range(reduction_size // 128):
        if MX:
            scale_ptr = scales + expert * SE + (k_tile * 4) * SG + n0 * SN
            src0 = mx_scale_source(scale_ptr, PACKED_SCALE)
            src1 = mx_scale_source(scale_ptr + 8 * SN, PACKED_SCALE)
        if not MX and ROWS == 16:
            scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
            scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
        if k_tile + 1 < reduction_size // 128:
            copy_token = prefetch_w4_square(
                weight_smem + ((k_tile + 1) % 2) * 4096,
                activation_smem + ((k_tile + 1) % 2) * (ROWS * 256),
                weight_base,
                activations,
                QN,
                reduction_size,
                row_tile,
                k_tile + 1,
                copy_token,
                ROWS,
            )
            copy_token = wait_w8_tile(copy_token, True)
        else:
            copy_token = wait_w8_tile(copy_token, False)
        qhere = weight_smem + (k_tile % 2) * 4096
        ahere = activation_smem + (k_tile % 2) * (ROWS * 256)
        if not MX:
            if ROWS == 8:
                scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
                scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
            s0 = scale0.to(tl.uint16, bitcast=True).to(tl.uint32)
            s1 = scale1.to(tl.uint16, bitcast=True).to(tl.uint32)
            s0 = s0 | (s0 << 16)
            s1 = s1 | (s1 << 16)
        for group in tl.static_range(2):
            r0, r1, r2, r3 = load_matrix_x4(
                qhere, warp * 16, 64, group * 32, copy_token
            )
            for chunk in tl.static_range(2):
                if MX:
                    s0 = mx_scale_fragment(src0, SG, group * 2 + chunk, PACKED_SCALE)
                    s1 = mx_scale_fragment(src1, SG, group * 2 + chunk, PACKED_SCALE)
                q0 = r0 if chunk == 0 else r1
                q8 = r2 if chunk == 0 else r3
                for part in tl.static_range(2):
                    a0, a1, a2, a3 = decode_w4_fragment_pair(q0, q8, s0, s1, part, MX)
                    sub = group * 4 + chunk * 2 + part
                    if ROWS == 8:
                        b0, b1 = load_activation_fragment_x2(
                            ahere + (sub // 4) * 1024, (sub % 4) * 32, lane
                        )
                        c0, c1, c2, c3 = mma_bf16_m16n8(
                            a0, a2, a1, a3, b0, b1, c0, c1, c2, c3
                        )
                    else:
                        b0, b1, b2, b3 = load_matrix_x4(
                            ahere + (sub // 4) * 2048, 0, 16, (sub % 4) * 32, copy_token
                        )
                        c0, c1, c2, c3, c4, c5, c6, c7 = mma_bf16_square(
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            b2,
                            b3,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                        )
        tl.debug_barrier()
    if ROWS == 8:
        token0 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group).to(offset_type)
        token1 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group + 1).to(
            offset_type
        )
        if MUL:
            w0 = tl.load(router_weights + token0, token0 < VALID, other=0).to(
                tl.float32
            )
            w1 = tl.load(router_weights + token1, token1 < VALID, other=0).to(
                tl.float32
            )
            c0 *= w0
            c2 *= w0
            c1 *= w1
            c3 *= w1
        tl.store(
            output_tile + token0 * CM + n0 * CN,
            c0.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + n0 * CN,
            c1.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
        tl.store(
            output_tile + token0 * CM + (n0 + 8) * CN,
            c2.to(output_tile.dtype.element_ty),
            token0 < VALID,
        )
        tl.store(
            output_tile + token1 * CM + (n0 + 8) * CN,
            c3.to(output_tile.dtype.element_ty),
            token1 < VALID,
        )
    else:
        c = (c0, c1, c2, c3, c4, c5, c6, c7)
        for j in tl.static_range(4):
            token = tl.load(route_ids + row_tile * 16 + lane_in_group + 4 * j).to(
                offset_type
            )
            v0 = c[j]
            v1 = c[j + 4]
            if MUL:
                w = tl.load(router_weights + token, token < VALID, other=0).to(
                    tl.float32
                )
                v0 *= w
                v1 *= w
            tl.store(
                output_tile + token * CM + n0 * CN,
                v0.to(output_tile.dtype.element_ty),
                token < VALID,
            )
            tl.store(
                output_tile + token * CM + (n0 + 8) * CN,
                v1.to(output_tile.dtype.element_ty),
                token < VALID,
            )


@triton.jit
def w8_narrow_accumulate(
    shared,
    ashared,
    qbase,
    activations,
    scales,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    expert,
    n0,
    inner_n,
    SE,
    SG,
    SN,
    FP8: tl.constexpr,
    FAST: tl.constexpr,
    IS_SPLIT_Q: tl.constexpr = False,
    initial_dep=None,
    PAIRED_ROWS: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
):
    # Two K=128 buffers overlap AIU transfer and eight 16-K MMAs.
    lane = tl.arange(0, 256) % 32
    if initial_dep is None:
        copy_token = tl.full((256,), 0, tl.int32)
        copy_token = prefetch_w8_tile(
            shared,
            ashared,
            qbase,
            activations,
            QN,
            reduction_size,
            row_tile,
            0,
            copy_token,
            8,
            IS_SPLIT_Q,
            PAIRED_ROWS=PAIRED_ROWS,
        )
    else:
        copy_token = initial_dep
    c0 = tl.full((256,), 0.0, tl.float32)
    c1 = c0
    c2 = c0
    c3 = c0
    # One packed BF16 load covers two successive K128 groups.
    tiles_per_scale_load: tl.constexpr = 2 if SCALE_PAIR else 1
    unroll: tl.constexpr = (
        2 if FP8 and reduction_size % 256 == 0 and not SCALE_PAIR else 1
    )
    for chunk in tl.range(
        reduction_size // (tiles_per_scale_load * 128), loop_unroll_factor=unroll
    ):
        if SCALE_PAIR:
            scale0, scale1 = load_w8_packed_scales(
                scales + expert * SE + n0 * SN + chunk * 2 * SG, SN, FAST
            )
        for inner_k in tl.static_range(tiles_per_scale_load):
            k_tile = chunk * tiles_per_scale_load + inner_k
            if not SCALE_PAIR:
                scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
                scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
            if k_tile + 1 < reduction_size // 128:
                copy_token = prefetch_w8_tile(
                    shared + (k_tile + 1) % 2 * W8_TILE_BYTES,
                    ashared + (k_tile + 1) % 2 * 2048,
                    qbase,
                    activations,
                    QN,
                    reduction_size,
                    row_tile,
                    k_tile + 1,
                    copy_token,
                    8,
                    IS_SPLIT_Q,
                    PAIRED_ROWS=PAIRED_ROWS,
                )
                copy_token = wait_w8_tile(copy_token, True)
            else:
                copy_token = wait_w8_tile(copy_token, False)
            s0 = w8_scale_fragment(scale0, inner_k, SCALE_PAIR, FP8 and FAST)
            s1 = w8_scale_fragment(scale1, inner_k, SCALE_PAIR, FP8 and FAST)
            here = shared + k_tile % 2 * W8_TILE_BYTES
            ahere = ashared + k_tile % 2 * 2048
            warp = inner_n // 16
            for group in tl.static_range(4):
                if IS_SPLIT_Q:
                    (r0, r1, r2, r3) = load_matrix_x4(
                        here + (warp // 4) * W8_HALF_TILE_BYTES,
                        (warp % 4) * 16,
                        64,
                        group * 32,
                        copy_token,
                    )
                else:
                    (r0, r1, r2, r3) = load_matrix_x4(
                        here, warp * 16, 128, group * 32, copy_token
                    )
                if PAIRED_ROWS:
                    first0, first1, second0, second1 = load_matrix_x4(
                        ahere, 0, 16, group * 32, copy_token
                    )
                for part in tl.static_range(2):
                    q0 = r0 if part == 0 else r1
                    q8 = r2 if part == 0 else r3
                    sub = group * 2 + part
                    if FP8:
                        if FAST:
                            (a0, a2, a1, a3) = decode_fp8_pair(q0, q8, s0, s1, True)
                        else:
                            (a0, a2, a1, a3) = decode_fp8_pair(q0, q8, s0, s1, False)
                    else:
                        a0, a2, a1, a3 = decode_int8_fragments(q0, q8, s0, s1)
                    if PAIRED_ROWS:
                        b0 = first0 if part == 0 else second0
                        b1 = first1 if part == 0 else second1
                    else:
                        (b0, b1) = load_activation_fragment_x2(
                            ahere + sub // 4 * 1024, sub % 4 * 32, lane
                        )
                    (c0, c1, c2, c3) = mma_bf16_m16n8(
                        a0, a1, a2, a3, b0, b1, c0, c1, c2, c3
                    )
            tl.debug_barrier()
    return (c0, c1, c2, c3)


@triton.jit
def marlin_moe_gemm_grouped_narrow_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    scale_flags,
    FP8: tl.constexpr,
    USE32: tl.constexpr = False,
    PAIRED_DOWN: tl.constexpr = False,
    SCALE_PAIR: tl.constexpr = False,
):
    # Each warp computes 16 output channels for eight routed rows.
    tl.static_assert(num_columns % 128 == 0 and reduction_size % 128 == 0)
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 128)
    col_tile = pid % (num_columns // 128)
    if row_tile * 8 >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(block_experts + row_tile).to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    lane_group = lane // 4
    lane_in_group = lane % 4
    inner_n = warp * 16 + lane_group
    n0 = col_tile * 128 + inner_n
    smem = tle_async.gpu.alloc(
        [32768],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    act = tle_async.gpu.alloc(
        [4096],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared = tle_async.gpu.local_ptr(smem, (0,)).to(tl.int64).to(tl.int32)
    ashared = tle_async.gpu.local_ptr(act, (0,)).to(tl.int64).to(tl.int32)
    qbase = weight_bytes + expert * QE + col_tile * 128 * QN
    tile_inputs = (shared, ashared, qbase, activations, scales, QN)
    tile_coordinates = (row_tile, expert, n0, inner_n, SE, SG, SN)
    if FP8:
        if PAIRED_DOWN:
            copy_token = tl.full((256,), 0, tl.int32)
            copy_token = prefetch_w8_tile(
                shared,
                ashared,
                qbase,
                activations,
                QN,
                reduction_size,
                row_tile,
                0,
                copy_token,
                8,
                PAIRED_ROWS=True,
            )
        else:
            copy_token = None
        can_fold = (
            tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
        )
        if can_fold:
            (c0, c1, c2, c3) = w8_narrow_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                True,
                initial_dep=copy_token,
                PAIRED_ROWS=PAIRED_DOWN,
                SCALE_PAIR=SCALE_PAIR,
            )
        else:
            (c0, c1, c2, c3) = w8_narrow_accumulate(
                *tile_inputs,
                reduction_size,
                *tile_coordinates,
                True,
                False,
                initial_dep=copy_token,
                PAIRED_ROWS=PAIRED_DOWN,
                SCALE_PAIR=SCALE_PAIR,
            )
    else:
        c0, c1, c2, c3 = w8_narrow_accumulate(
            *tile_inputs,
            reduction_size,
            *tile_coordinates,
            False,
            False,
            PAIRED_ROWS=PAIRED_DOWN,
        )
    token0 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group).to(offset_type)
    token1 = tl.load(route_ids + row_tile * 8 + 2 * lane_in_group + 1).to(offset_type)
    if MUL:
        w0 = tl.load(router_weights + token0, token0 < VALID, other=0).to(tl.float32)
        w1 = tl.load(router_weights + token1, token1 < VALID, other=0).to(tl.float32)
        c0 *= w0
        c2 *= w0
        c1 *= w1
        c3 *= w1
    tl.store(
        output_tile + token0 * CM + n0 * CN,
        c0.to(output_tile.dtype.element_ty),
        token0 < VALID,
    )
    tl.store(
        output_tile + token1 * CM + n0 * CN,
        c1.to(output_tile.dtype.element_ty),
        token1 < VALID,
    )
    tl.store(
        output_tile + token0 * CM + (n0 + 8) * CN,
        c2.to(output_tile.dtype.element_ty),
        token0 < VALID,
    )
    tl.store(
        output_tile + token1 * CM + (n0 + 8) * CN,
        c3.to(output_tile.dtype.element_ty),
        token1 < VALID,
    )


@triton.jit
def prefetch_quantized_wide_tile(
    weight_smem,
    activation_smem,
    weight_bytes,
    activations,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    k_tile,
    copy_token,
    WEIGHT_K_BYTES: tl.constexpr,
):
    copy_token = prefetch_activations(
        activation_smem, activations, reduction_size, row_tile, k_tile, copy_token, 64
    )
    copy_token = copy_w8_tile_async(
        weight_smem,
        weight_bytes,
        128,
        QN,
        WEIGHT_K_BYTES,
        k_tile * WEIGHT_K_BYTES,
        copy_token,
    )
    return issue_copy_group(copy_token, "ppu.cp.async.commit_group;")


@triton.jit
def accumulate_quantized_wide_tile(
    weight_smem,
    activation_smem,
    weight_base,
    activations,
    scales,
    QN,
    reduction_size: tl.constexpr,
    row_tile,
    expert,
    n0,
    warp,
    SE,
    SG,
    SN,
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    FAST: tl.constexpr,
    LIVE: tl.constexpr,
):
    MX: tl.constexpr = QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    W4: tl.constexpr = MX or QTYPE == _TL_QUANT_TYPE_UINT4B8
    WEIGHT_K_BYTES: tl.constexpr = 64 if W4 else 128
    copy_token = tl.full((256,), 0, tl.int32)
    copy_token = prefetch_quantized_wide_tile(
        weight_smem,
        activation_smem,
        weight_base,
        activations,
        QN,
        reduction_size,
        row_tile,
        0,
        copy_token,
        WEIGHT_K_BYTES,
    )
    zero = tl.full((256,), 0.0, tl.float32)
    acc = ()
    for j in tl.static_range(LIVE // 2):
        acc += (zero,)
    for k_tile in range(reduction_size // 128):
        if not MX:
            scale0 = tl.load(scales + expert * SE + k_tile * SG + n0 * SN)
            scale1 = tl.load(scales + expert * SE + k_tile * SG + (n0 + 8) * SN)
        else:
            scale_ptr = scales + expert * SE + (k_tile * 4) * SG + n0 * SN
            src0 = mx_scale_source(scale_ptr, PACKED_SCALE)
            src1 = mx_scale_source(scale_ptr + 8 * SN, PACKED_SCALE)
        if k_tile + 1 < reduction_size // 128:
            next_offset = (k_tile + 1) % 2 * W8_TILE_BYTES
            copy_token = prefetch_quantized_wide_tile(
                weight_smem + (k_tile + 1) % 2 * 128 * WEIGHT_K_BYTES,
                activation_smem + next_offset,
                weight_base,
                activations,
                QN,
                reduction_size,
                row_tile,
                k_tile + 1,
                copy_token,
                WEIGHT_K_BYTES,
            )
            copy_token = wait_w8_tile(copy_token, True)
        else:
            copy_token = wait_w8_tile(copy_token, False)
        if not MX:
            s0 = w8_scale_fragment(scale0, k_tile, False, FAST)
            s1 = w8_scale_fragment(scale1, k_tile, False, FAST)
        current_offset = k_tile % 2 * W8_TILE_BYTES
        qhere = weight_smem + k_tile % 2 * 128 * WEIGHT_K_BYTES
        ahere = activation_smem + current_offset
        for group in tl.static_range(WEIGHT_K_BYTES // 32):
            r0, r1, r2, r3 = load_matrix_x4(
                qhere, warp * 16, 128, group * 32, copy_token
            )
            for part in tl.static_range(2):
                q0 = r0 if part == 0 else r1
                q8 = r2 if part == 0 else r3
                if MX:
                    s0 = mx_scale_fragment(src0, SG, group * 2 + part, PACKED_SCALE)
                    s1 = mx_scale_fragment(src1, SG, group * 2 + part, PACKED_SCALE)
                for half in tl.static_range(2 if W4 else 1):
                    if W4:
                        sub = group * 4 + part * 2 + half
                        lo, hi = q0 >> (half * 16), q8 >> (half * 16)
                        if MX:
                            a0, a1 = decode_mx_register(lo, s0, s0, WIDE=True)
                            a2, a3 = decode_mx_register(hi, s1, s1, WIDE=True)
                        else:
                            a2, a3 = decode_i4_register(hi, s1, s1)
                            a0, a1 = decode_i4_register(lo, s0, s0)
                    else:
                        sub = group * 2 + part
                        if QTYPE == _TL_QUANT_TYPE_UINT8B128:
                            a0, a1, a2, a3 = decode_int8_fragments(q0, q8, s0, s1)
                        else:
                            a0, a1, a2, a3 = decode_fp8_pair(q0, q8, s0, s1, FAST)
                    # Reuse the decoded fragment for each live group of 16 routes.
                    next_acc = ()
                    for chunk in tl.static_range(LIVE // 16):
                        b0, b1, b2, b3 = load_matrix_x4(
                            ahere + sub // 4 * W8_HALF_TILE_BYTES,
                            chunk * 16,
                            64,
                            sub % 4 * 32,
                            copy_token,
                        )
                        next_acc += mma_bf16_square(
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            b2,
                            b3,
                            acc[chunk * 8],
                            acc[chunk * 8 + 1],
                            acc[chunk * 8 + 2],
                            acc[chunk * 8 + 3],
                            acc[chunk * 8 + 4],
                            acc[chunk * 8 + 5],
                            acc[chunk * 8 + 6],
                            acc[chunk * 8 + 7],
                        )
                    acc = next_acc
        tl.debug_barrier()
    return acc


@triton.jit
def project_quantized_wide_tile(
    weight_smem,
    activation_smem,
    weight_base,
    activations,
    output_tile,
    scales,
    router_weights,
    route_ids,
    scale_flags,
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    reduction_size: tl.constexpr,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    VALID,
    MUL: tl.constexpr,
    LIVE: tl.constexpr,
    offset_type: tl.constexpr,
    expert,
    n0,
    warp,
    row_tile,
    lane_in_group,
    amem,
    col_tile,
):
    can_fold = tl.load(scale_flags + expert) != 0 if scale_flags is not None else False
    tile_inputs = (weight_smem, activation_smem, weight_base, activations, scales, QN)
    tile_coordinates = (row_tile, expert, n0, warp, SE, SG, SN)
    if can_fold:
        acc = accumulate_quantized_wide_tile(
            *tile_inputs,
            reduction_size,
            *tile_coordinates,
            QTYPE,
            PACKED_SCALE,
            True,
            LIVE,
        )
    else:
        acc = accumulate_quantized_wide_tile(
            *tile_inputs,
            reduction_size,
            *tile_coordinates,
            QTYPE,
            PACKED_SCALE,
            False,
            LIVE,
        )
    if reduction_size == 256:
        for chunk in tl.static_range(LIVE // 16):
            for j in tl.static_range(4):
                row = chunk * 16 + lane_in_group + 4 * j
                col = n0 % 128
                # Separate lanes that write the same column into distinct banks.
                bank = (row % 4) * 8
                tl.store(
                    tle_async.gpu.local_ptr(amem, (row * 128 + (col ^ bank),)),
                    acc[chunk * 8 + j],
                )
                tl.store(
                    tle_async.gpu.local_ptr(amem, (row * 128 + ((col + 8) ^ bank),)),
                    acc[chunk * 8 + j + 4],
                )
        tl.debug_barrier()
        rows = tl.arange(0, 64)
        cols = tl.arange(0, 128)
        bank = (rows[:, None] % 4) * 8
        indices = rows[:, None] * 128 + (cols[None, :] ^ bank)
        acc_output = tl.load(
            tle_async.gpu.local_ptr(amem, (indices,)), rows[:, None] < LIVE, other=0
        )
        tokens = tl.load(route_ids + row_tile * 64 + rows).to(offset_type)
        valid = tokens < VALID
        if MUL:
            weights = tl.load(router_weights + tokens, valid, other=0).to(tl.float32)
            acc_output *= weights[:, None]
        tl.store(
            output_tile + tokens[:, None] * CM + (col_tile * 128 + cols[None, :]) * CN,
            acc_output.to(output_tile.dtype.element_ty),
            valid[:, None],
        )
    else:
        for chunk in tl.static_range(LIVE // 16):
            for j in tl.static_range(4):
                token = tl.load(
                    route_ids + row_tile * 64 + chunk * 16 + lane_in_group + 4 * j
                ).to(offset_type)
                v0 = acc[chunk * 8 + j]
                v1 = acc[chunk * 8 + j + 4]
                if MUL:
                    w = tl.load(router_weights + token, token < VALID, other=0).to(
                        tl.float32
                    )
                    v0 *= w
                    v1 *= w
                tl.store(
                    output_tile + token * CM + n0 * CN,
                    v0.to(output_tile.dtype.element_ty),
                    token < VALID,
                )
                tl.store(
                    output_tile + token * CM + (n0 + 8) * CN,
                    v1.to(output_tile.dtype.element_ty),
                    token < VALID,
                )


@libentry()
@triton.jit
def quantized_wide_grouped_kernel(
    activations,
    weight_bytes,
    output_tile,
    scales,
    router_weights,
    route_ids,
    block_experts,
    padded_count,
    num_columns: tl.constexpr,
    reduction_size: tl.constexpr,
    VALID,
    QE,
    QN,
    SE,
    SG,
    SN,
    CM,
    CN,
    MUL: tl.constexpr,
    scale_flags,
    QTYPE: tl.constexpr,
    PACKED_SCALE: tl.constexpr,
    USE32: tl.constexpr,
):
    pid = tl.program_id(0)
    row_tile = pid // (num_columns // 128)
    col_tile = pid % (num_columns // 128)
    if row_tile * 64 >= tl.load(padded_count):
        return
    offset_type: tl.constexpr = tl.int32 if USE32 else tl.int64
    expert = tl.load(block_experts + row_tile).to(offset_type)
    thread = tl.arange(0, 256)
    lane = thread % 32
    warp = thread // 32
    n0 = col_tile * 128 + warp * 16 + lane // 4
    lane_in_group = lane % 4
    W4: tl.constexpr = (
        QTYPE == _TL_QUANT_TYPE_UINT4B8 or QTYPE == _TL_QUANT_TYPE_FP4_E2M1
    )
    qmem = tle_async.gpu.alloc(
        [W8_TILE_BYTES if W4 else 2 * W8_TILE_BYTES],
        tl.uint8,
        layout=None,
        scope=tle_async.gpu.smem,
        nv_mma_shared_layout=False,
    )
    if reduction_size == 256:
        # Reuse the activation buffer to assemble contiguous FP32 output rows.
        amem = tle_async.gpu.alloc(
            [64 * 128],
            tl.float32,
            layout=None,
            scope=tle_async.gpu.smem,
            nv_mma_shared_layout=False,
        )
    else:
        amem = tle_async.gpu.alloc(
            [32768],
            tl.uint8,
            layout=None,
            scope=tle_async.gpu.smem,
            nv_mma_shared_layout=False,
        )
    weight_smem = tle_async.gpu.local_ptr(qmem, (0,)).to(tl.int64).to(tl.int32)
    activation_smem = tle_async.gpu.local_ptr(amem, (0,)).to(tl.int64).to(tl.int32)
    weight_base = weight_bytes + expert * QE + col_tile * 128 * QN
    # Padding is contiguous at the end of each expert tile.
    if tl.load(route_ids + row_tile * 64 + 48) >= VALID:
        project_quantized_wide_tile(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            output_tile,
            scales,
            router_weights,
            route_ids,
            scale_flags,
            QTYPE,
            PACKED_SCALE,
            reduction_size,
            QN,
            SE,
            SG,
            SN,
            CM,
            CN,
            VALID,
            MUL,
            48,
            offset_type,
            expert,
            n0,
            warp,
            row_tile,
            lane_in_group,
            amem,
            col_tile,
        )
    else:
        project_quantized_wide_tile(
            weight_smem,
            activation_smem,
            weight_base,
            activations,
            output_tile,
            scales,
            router_weights,
            route_ids,
            scale_flags,
            QTYPE,
            PACKED_SCALE,
            reduction_size,
            QN,
            SE,
            SG,
            SN,
            CM,
            CN,
            VALID,
            MUL,
            64,
            offset_type,
            expert,
            n0,
            warp,
            row_tile,
            lane_in_group,
            amem,
            col_tile,
        )
