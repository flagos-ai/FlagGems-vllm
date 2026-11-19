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

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import has_triton_tle_attrs, libentry

if has_triton_tle_attrs(("load",), 3, 6, 0):
    try:
        import triton.experimental.tle.language as tle
        from triton._C.libtriton import ppu

        HAS_AIU_K = ppu.passes.ttppugpuir.add_tle_promote_async_load_to_aiu is not None
    except (ImportError, AttributeError):
        HAS_AIU_K = False
        tle = None
else:
    HAS_AIU_K = False
    tle = None


HAS_PAGED_PIPE = False
if HAS_AIU_K and has_triton_tle_attrs(
    ("pipe", "gpu.alloc", "gpu.copy", "gpu.local_ptr", "gpu.swizzled_shared_layout"),
    3,
    6,
    0,
):
    try:
        HAS_PAGED_PIPE = ppu.passes.ttppugpuir.add_tle_lower_pipe is not None
    except AttributeError:
        HAS_PAGED_PIPE = False


LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)
# Signed INT8 stores 0..255 probability levels with a -128 zero point.
PROB_QUANT_LEVELS = tl.constexpr(255)
# Two signed bytes represent hi * 256 + lo without overflowing the high byte.
PRECISE_PROB_LEVELS = tl.constexpr(127 * 256)
# A power-of-two scale preserves P's significand and keeps its maximum in FP16 range.
PV_SCALE = tl.constexpr(2.0**15)
PV_UNSCALE = tl.constexpr(2.0**-15)
# Budget for the absolute output contribution from probability rounding only.
PV_ROUNDING_BUDGET = tl.constexpr(2.0**-8)
# FP16 round-to-nearest bounds; the PPU dot preserves subnormal operands.
PV_RELATIVE_ROUNDING = tl.constexpr(2.0**-11)
PV_ABSOLUTE_ROUNDING = tl.constexpr(2.0**-25)
# Keep scaling when accumulated subnormal error would tighten the fast-path limit.
PV_UNSCALED_MAX_LENGTH = (
    int(PV_RELATIVE_ROUNDING.value / PV_ABSOLUTE_ROUNDING.value) + 1
)
PV_SCALED_UNDERFLOW_BOUND = tl.constexpr(2.0**-29)
# Descales are indexed by logical sequence position in blocks of DESCALE_BLOCK.
DESCALE_BLOCK = tl.constexpr(128)
PACK_TILE = tl.constexpr(32)
BOUNDARY_KV_TILE = tl.constexpr(16)
SHORT_QUERY_LIMIT = tl.constexpr(16)
WORKLIST_TILE = tl.constexpr(128)
WORKLIST_INTERLEAVE_MAX_BATCH = 64
TRANSPOSE_V_MIN_QUERY_LENGTH = 1024
DECODE_KV_SPLITS = tl.constexpr(2)
SPLIT_KV_MIN_LENGTH = tl.constexpr(512)
SPLIT_KV_MAX_PARALLEL_HEADS = tl.constexpr(64)
KV_PACK_MAX_BYTES = 512 * 1024 * 1024
SPLIT_KV_MAX_BYTES = 256 * 1024 * 1024
SPLIT_KV_TARGET_BLOCKS = 128
MAX_KV_SPLITS = 32
SHORT_SPLIT_KV_MIN_LENGTH = 8192
D256_FOLDED_QUERY_LIMIT = 256


@triton.jit
def _flash_int8_prepare_worklist(
    CUQ,
    WORK,
    BATCH: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE_PITCH: tl.constexpr,
    ROUND_ROBIN: tl.constexpr,
    SHORT_STATS: tl.constexpr = False,
):
    batch = tl.program_id(0)
    batches = tl.arange(0, triton.next_power_of_2(BATCH))
    begins = tl.load(CUQ + batches, batches < BATCH, 0)
    ends = tl.load(CUQ + batches + 1, batches < BATCH, 0)
    lengths = ends - begins
    counts = tl.where(
        (batches < BATCH) & (lengths > SHORT_QUERY_LIMIT),
        tl.cdiv(lengths, QUERY_TILE),
        0,
    )
    offset = tl.sum(tl.where(batches < batch, counts, 0))
    count = tl.sum(tl.where(batches == batch, counts, 0))
    if batch == 0:
        tl.store(WORK + CAPACITY, tl.sum(counts))
        if SHORT_STATS:
            tl.store(WORK + CAPACITY + 1, 0)
            tl.store(WORK + CAPACITY + 2, SHORT_QUERY_LIMIT)
    for start in range(tl.cdiv(count, WORKLIST_TILE)):
        tile = start * WORKLIST_TILE + tl.arange(0, WORKLIST_TILE)
        if ROUND_ROBIN:
            # Rank (tile, request) without atomics, preserving ragged request tails.
            previous_tiles = tl.sum(tl.minimum(counts[None, :], tile[:, None]), 1)
            previous_requests = tl.sum(
                ((batches[None, :] < batch) & (counts[None, :] > tile[:, None])).to(
                    tl.int32
                ),
                1,
            )
            position = previous_tiles + previous_requests
        else:
            position = offset + tile
        tl.store(WORK + position, batch * TILE_PITCH + tile, tile < count)


@triton.jit
def _flash_int8_pack_kv(
    K,
    V,
    KP,
    VP,
    TABLE,
    USED,
    CUQ,
    N: tl.constexpr,
    D: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    TS: tl.constexpr,
    PK: tl.constexpr,
    SK: tl.constexpr,
    HK: tl.constexpr,
    PV: tl.constexpr,
    SV: tl.constexpr,
    HV: tl.constexpr,
    LOGICAL: tl.constexpr,
    SKIP_SHORT: tl.constexpr,
    TRANSPOSE_V: tl.constexpr = False,
):
    n = tl.program_id(0) * PACK_TILE + tl.arange(0, PACK_TILE)
    d = tl.arange(0, D)
    if LOGICAL:
        batch, h = tl.program_id(1), tl.program_id(2)
    else:
        batch, h = 0, tl.program_id(1)
    should_pack = tl.full((), True, tl.int1)
    if SKIP_SHORT:
        nq = tl.load(CUQ + batch + 1) - tl.load(CUQ + batch)
        should_pack = nq > SHORT_QUERY_LIMIT
    if should_pack:
        if LOGICAL:
            nk = tl.load(USED + batch)
            page = tl.load(TABLE + batch * TS + n // PAGE, n < nk, 0)
        else:
            nk = N
            page = n // PAGE
        # Shared with MetaX: page IDs are int32, but cache offsets may not fit.
        page = page.to(tl.int64)
        k = tl.load(
            K + page[:, None] * PK + (n[:, None] % PAGE) * SK + h * HK + d[None, :],
            n[:, None] < nk,
            0,
        )
        v = tl.load(
            V + page[:, None] * PV + (n[:, None] % PAGE) * SV + h * HV + d[None, :],
            n[:, None] < nk,
            0,
        )
        off = ((batch * HEADS + h) * N + n[:, None]) * D + d[None, :]
        tl.store(KP + off, k, n[:, None] < N)
        v_off = (
            ((batch * HEADS + h) * D + d[None, :]) * N + n[:, None]
            if TRANSPOSE_V
            else off
        )
        tl.store(VP + v_off, v.to(tl.float16), n[:, None] < N)


@triton.jit
def _flash_int8_stage_paged(
    writer,
    tile_index,
    first,
    K,
    V,
    TABLE,
    batch,
    head,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    PK: tl.constexpr,
    SK: tl.constexpr,
    HK: tl.constexpr,
    PV: tl.constexpr,
    SV: tl.constexpr,
    HV: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    n = (first + tile_index) * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    page = tl.load(TABLE + batch * TS + n // PAGE).to(tl.int64)
    slot = writer.acquire(tile_index)
    tle.gpu.copy(
        K + page[:, None] * PK + (n[:, None] % PAGE) * SK + head * HK + d[None, :],
        slot.k,
        [BN, D],
    )
    tle.gpu.copy(
        V + page[:, None] * PV + (n[:, None] % PAGE) * SV + head * HV + d[None, :],
        slot.v,
        [BN, D],
    )
    writer.commit(tile_index)


@libentry()
@triton.jit
def _flash_int8_fwd(
    Q,
    K,
    V,
    O,
    LSE,
    CUQ,
    CUK,
    USED,
    TABLE,
    WORK,
    QS,
    KS,
    VS,
    ALIBI,
    sq: tl.constexpr,
    hq: tl.constexpr,
    sk: tl.constexpr,
    hk: tl.constexpr,
    pk: tl.constexpr,
    sv: tl.constexpr,
    hv: tl.constexpr,
    dv: tl.constexpr,
    pv: tl.constexpr,
    so: tl.constexpr,
    ho: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    table_stride: tl.constexpr,
    alibi_stride: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    PAGED: tl.constexpr,
    CAUSAL: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    CAP: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_ALIBI: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HALF_PV: tl.constexpr,
    COMPENSATE_PV: tl.constexpr,
    SCALED_PV: tl.constexpr,
    PRECISE_PV: tl.constexpr,
    ASYNC_K: tl.constexpr,
    PREFETCH_V: tl.constexpr,
    PIPE_DEPTH: tl.constexpr,
    REORDER_CAUSAL: tl.constexpr,
    TAIL_N: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    OUT_SPLIT_STRIDE: tl.constexpr,
    LSE_SPLIT_STRIDE: tl.constexpr,
    SEPARATE_MASK: tl.constexpr,
    LOGICAL_KV: tl.constexpr,
    BATCH: tl.constexpr,
    COMPACT: tl.constexpr,
    USE_WORKLIST: tl.constexpr,
    WORK_CAPACITY: tl.constexpr,
    TILE_PITCH: tl.constexpr,
    SHORT_ONLY: tl.constexpr,
    SHORT_OUTPUT: tl.constexpr,
    FOLD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    program_tile = tl.program_id(2) if REORDER_CAUSAL else tl.program_id(0)
    tile, batch, head = (
        program_tile // KV_SPLITS,
        tl.program_id(1),
        (tl.program_id(0) if REORDER_CAUSAL else tl.program_id(2)),
    )
    split_id = program_tile % KV_SPLITS
    kv_head = head if FOLD else head // GROUP
    query_tile: tl.constexpr = BM // GROUP if FOLD else BM
    active = tl.full((), True, tl.int1)
    if USE_WORKLIST:
        count = tl.load(WORK + WORK_CAPACITY)
        active = tile < count
        entry = tl.load(WORK + tile, active, 0)
        batch = entry // TILE_PITCH
        tile = entry % TILE_PITCH
    elif COMPACT:
        batches = tl.arange(0, triton.next_power_of_2(BATCH))
        begins = tl.load(CUQ + batches, batches < BATCH, 0)
        ends = tl.load(CUQ + batches + 1, batches < BATCH, 0)
        counts = tl.cdiv(ends - begins, query_tile)
        cumulative = tl.cumsum(counts)
        # The host grid bounds sum(ceil(q_len / query_tile)); excess tiles
        # map past the final request and are skipped by the query-length guard.
        batch = tl.minimum(
            tl.sum(((tile >= cumulative) & (batches < BATCH)).to(tl.int32)), BATCH - 1
        )
        tile -= tl.sum(tl.where(batches < batch, counts, 0))
    q_start = tl.load(CUQ + batch)
    nq = tl.load(CUQ + batch + 1) - q_start
    output_q_start = batch * SHORT_QUERY_LIMIT if SHORT_OUTPUT else q_start
    if REORDER_CAUSAL:
        # Later causal query tiles have more KV work. Schedule them first.
        count_q_tiles = tl.cdiv(nq, query_tile)
        active &= tile < count_q_tiles
        tile = count_q_tiles - 1 - tile
    if SHORT_ONLY:
        active &= nq <= SHORT_QUERY_LIMIT
    if PAGED:
        k_start = 0
        nk = tl.load(USED + batch)
    else:
        k_start = tl.load(CUK + batch)
        nk = tl.load(CUK + batch + 1) - k_start
    # FP16 PV already keeps its head scale live; reuse uniform scales across tiles.
    if ks2 == 0 and HALF_PV:
        head_k_scale = tl.load(KS + batch * ks0 + kv_head * ks1, nk > 0, 0)
    if vs2 == 0 and HALF_PV:
        head_v_scale = tl.load(VS + batch * vs0 + kv_head * vs1, nk > 0, 0)
    if HALF_PV and vs2 == 0:
        v_scale = head_v_scale
        # A maximum probability is exactly 1, so its FP16 cast has no error.
        # For D = sum(P) >= 1, the remaining error divided by D is bounded by
        # max(u, (nk - 1) * epsilon), since u * (D - 1) + (nk - 1) * epsilon
        # bounds the unnormalized error. |V_int8| <= 128 then bounds each output.
        if SCALED_PV:
            probability_error = PV_RELATIVE_ROUNDING + nk * PV_SCALED_UNDERFLOW_BOUND
        else:
            probability_error = tl.maximum(
                PV_RELATIVE_ROUNDING, tl.maximum(nk - 1, 0) * PV_ABSOLUTE_ROUNDING
            )
        rounding_bound = 128.0 * tl.abs(v_scale) * probability_error
        needs_compensation = rounding_bound > PV_ROUNDING_BUDGET
        if COMPENSATE_PV:
            active &= needs_compensation
        else:
            active &= ~needs_compensation
    # A single valid key has unit probability. Keep this in the existing
    # kernel so mixed/padded batches use their actual on-device lengths.
    # Large matrix tiles use the ordinary path to bound scalar LSE registers.
    if D == 256 and BM <= 16:
        single_key = active & (nq == 1) & (nk == 1) & (tile == 0)
        if single_key:
            one_rows = tl.arange(0, BM)
            one_heads = (
                kv_head * GROUP + one_rows if FOLD else tl.full((BM,), head, tl.int32)
            )
            one_mask = one_rows < (GROUP if FOLD else 1)
            one_d = tl.arange(0, D)
            if LOGICAL_KV:
                one_k_row, one_v_row = batch * pk, batch * pv
            elif PAGED:
                one_page = tl.load(TABLE + batch * table_stride).to(tl.int64)
                one_k_row, one_v_row = one_page * pk, one_page * pv
            else:
                one_k_row, one_v_row = k_start * sk, k_start * sv
            one_v = tl.load(V + one_v_row + kv_head * hv + one_d * dv)
            one_vs = tl.load(VS + batch * vs0 + kv_head * vs1)
            one_result = tl.where(split_id == 0, one_v.to(tl.float32) * one_vs, 0.0)
            tl.store(
                O
                + split_id * OUT_SPLIT_STRIDE
                + output_q_start * so
                + one_heads[:, None] * ho
                + one_d[None, :],
                tl.broadcast_to(one_result[None, :], (BM, D)),
                one_mask[:, None],
            )
            if WRITE_LSE:
                one_q = tl.load(
                    Q + q_start * sq + one_heads[:, None] * hq + one_d[None, :],
                    one_mask[:, None],
                    0,
                )
                one_k = tl.load(K + one_k_row + kv_head * hk + one_d)
                one_qs = tl.load(QS + batch * qs0 + one_heads * qs1, one_mask, 0)
                one_ks = tl.load(KS + batch * ks0 + kv_head * ks1)
                one_score = tl.sum(one_q.to(tl.int32) * one_k[None, :].to(tl.int32), 1)
                one_score = one_score.to(tl.float32) * (one_qs * one_ks * SCALE)
                if CAP > 0:
                    one_score = CAP * (2 / (1 + tl.exp(-2 * one_score / CAP)) - 1)
                one_address = (
                    LSE
                    + split_id * LSE_SPLIT_STRIDE
                    + one_heads * TOTAL_Q
                    + output_q_start
                )
                if KV_SPLITS > 1:
                    tl.store(
                        one_address,
                        tl.where(split_id == 0, one_score * LOG2E, float("-inf")),
                        one_mask,
                    )
                    tl.store(
                        one_address + LSE_SPLIT_STRIDE // 2,
                        tl.where(split_id == 0, 1.0, 0.0),
                        one_mask,
                    )
                else:
                    tl.store(one_address, one_score, one_mask)
        active &= ~single_key
    if SHORT_ONLY:
        query_iterations = tl.where(active, tl.cdiv(nq, query_tile), 0)
    else:
        query_iterations = 1
    for query_index in range(query_iterations):
        current_tile = query_index if SHORT_ONLY else tile
        rows = current_tile * BM + tl.arange(0, BM)
        if ASYNC_K and FOLD:
            # Keep one head's query rows together in the matrix-warp layout.
            local_rows = tl.arange(0, BM)
            m = current_tile * query_tile + local_rows % query_tile
            h = kv_head * GROUP + local_rows // query_tile
        else:
            m = rows // GROUP if FOLD else rows
            h = (
                kv_head * GROUP + rows % GROUP
                if FOLD
                else tl.full((BM,), head, tl.int32)
            )
        d = tl.arange(0, D)
        if active & (current_tile * query_tile < nq):
            q = tl.load(
                Q + (q_start + m[:, None]) * sq + h[:, None] * hq + d[None, :],
                m[:, None] < nq,
                0,
            )
            q_scale = tl.load(
                QS + batch * qs0 + h * qs1 + (m // DESCALE_BLOCK) * qs2, m < nq, 0
            )
            if HALF_PV and not COMPACT and CAP <= 0 and not HAS_ALIBI:
                q_scale = q_scale * (SCALE * LOG2E)
            maximum = tl.full((BM,), float("-inf"), tl.float32)
            denom = tl.full((BM,), 0, tl.float32)
            acc = tl.full((BM, D), 0, tl.float32)
            if HAS_ALIBI:
                slope = tl.load(ALIBI + batch * alibi_stride + h)
            # Skip KV blocks that are masked for every query in this query tile.
            first = 0
            end = nk
            if LEFT >= 0:
                first = tl.maximum(0, current_tile * query_tile + nk - nq - LEFT) // BN
            if CAUSAL:
                end = tl.minimum(end, (current_tile + 1) * query_tile + nk - nq)
            if RIGHT >= 0:
                end = tl.minimum(end, (current_tile + 1) * query_tile + nk - nq + RIGHT)
            # Only the boundary interval needs per-row causal and length masks.
            use_dense: tl.constexpr = SEPARATE_MASK and LEFT < 0 and RIGHT < 0
            end_block = tl.cdiv(tl.maximum(end, 0), BN)
            if use_dense:
                full_keys = nk
                if CAUSAL:
                    full_keys = tl.minimum(nk, current_tile * query_tile + nk - nq + 1)
                full_hi = tl.maximum(full_keys, 0) // BN
            # Avoid computing fully masked columns without narrowing dense KV tiles.
            small_boundary: tl.constexpr = (
                use_dense and HALF_PV and LOGICAL_KV and query_tile < BN and TAIL_N < BN
            )
            phase_blocks: tl.constexpr = (BN, TAIL_N if small_boundary else BN)
            for mask_phase in tl.static_range(2 if use_dense else 1):
                if use_dense:
                    begin_block = (
                        first
                        if mask_phase == 0
                        else (full_hi * BN // TAIL_N if small_boundary else full_hi)
                    )
                    stop_block = (
                        full_hi
                        if mask_phase == 0
                        else (
                            tl.cdiv(tl.maximum(end, 0), TAIL_N)
                            if small_boundary
                            else end_block
                        )
                    )
                else:
                    begin_block, stop_block = first, end_block
                if KV_SPLITS > 1:
                    blocks_per_split = tl.cdiv(tl.cdiv(nk, BN), KV_SPLITS)
                    split_tile_scale = BN // phase_blocks[mask_phase]
                    begin_block = tl.maximum(
                        begin_block, split_id * blocks_per_split * split_tile_scale
                    )
                    stop_block = tl.minimum(
                        stop_block, (split_id + 1) * blocks_per_split * split_tile_scale
                    )
                if PIPE_DEPTH > 0 and use_dense and mask_phase == 0:
                    k_layout: tl.constexpr = tle.gpu.swizzled_shared_layout(
                        16, 1, 8, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0]
                    )
                    k_ring = tle.gpu.alloc(
                        [PIPE_DEPTH, BN, D], dtype=tl.int8, layout=k_layout
                    )
                    v_ring = tle.gpu.alloc(
                        [PIPE_DEPTH, BN, D], dtype=tl.int8, layout=k_layout
                    )
                    pipe = tle.pipe(capacity=PIPE_DEPTH, k=k_ring, v=v_ring)
                    writer = pipe.writer()
                    reader = pipe.reader()
                    tile_count = tl.maximum(stop_block - begin_block, 0)
                    for initial in range(tl.minimum(tile_count, PIPE_DEPTH)):
                        _flash_int8_stage_paged(
                            writer,
                            initial,
                            begin_block,
                            K,
                            V,
                            TABLE,
                            batch,
                            kv_head,
                            table_stride,
                            PAGE,
                            pk,
                            sk,
                            hk,
                            pv,
                            sv,
                            hv,
                            BN,
                            D,
                        )
                for start in range(begin_block, stop_block):
                    n = start * phase_blocks[mask_phase] + tl.arange(
                        0, phase_blocks[mask_phase]
                    )
                    if PIPE_DEPTH > 0 and use_dense and mask_phase == 0:
                        pipe_index = start - begin_block
                        slot = reader.wait(pipe_index).slot
                        k = tl.trans(tl.load(tle.gpu.local_ptr(slot.k)))
                    elif LOGICAL_KV:
                        k_row = batch * pk + n * sk
                        v_row = batch * pv + n * sv
                    elif PAGED:
                        if not use_dense or mask_phase == 1:
                            page = tl.load(
                                TABLE + batch * table_stride + n // PAGE, n < nk, 0
                            )
                        else:
                            page = tl.load(TABLE + batch * table_stride + n // PAGE)
                        # A large physical cache can exceed int32 element offsets.
                        page = page.to(tl.int64)
                        k_row = page * pk + (n % PAGE) * sk
                        v_row = page * pv + (n % PAGE) * sv
                    else:
                        k_row = (k_start + n) * sk
                        v_row = (k_start + n) * sv
                    if PIPE_DEPTH > 0 and use_dense and mask_phase == 0:
                        pass
                    elif ASYNC_K and use_dense and mask_phase == 0:
                        # AIU copies contiguous packed K without materializing a
                        # per-element global-load address in every thread.
                        k_ptr = tl.make_block_ptr(
                            K + batch * pk + kv_head * hk,
                            shape=(D, nk),
                            strides=(1, sk),
                            offsets=(0, start * phase_blocks[mask_phase]),
                            block_shape=(D, phase_blocks[mask_phase]),
                            order=(0, 1),
                        )
                        k = tle.load(k_ptr, is_async=True)
                    elif not use_dense or mask_phase == 1:
                        k = tl.load(
                            K + k_row[None, :] + kv_head * hk + d[:, None],
                            n[None, :] < nk,
                            0,
                        )
                    else:
                        k = tl.load(K + k_row[None, :] + kv_head * hk + d[:, None])
                    if PREFETCH_V and not (
                        PIPE_DEPTH > 0 and use_dense and mask_phase == 0
                    ):
                        # Loading raw V before score updates reduces short-query spills.
                        v = tl.load(
                            V + v_row[:, None] + kv_head * hv + d[None, :] * dv,
                            (
                                n[:, None] < nk
                                if not use_dense or mask_phase == 1
                                else True
                            ),
                            0,
                        )
                    descale_block = start * phase_blocks[mask_phase] // DESCALE_BLOCK
                    if ks2 == 0 and HALF_PV:
                        ks = head_k_scale
                    else:
                        ks = tl.load(
                            KS + batch * ks0 + kv_head * ks1 + descale_block * ks2
                        )
                    if vs2 == 0 and HALF_PV:
                        vs = head_v_scale
                    else:
                        vs = tl.load(
                            VS + batch * vs0 + kv_head * vs1 + descale_block * vs2
                        )
                    scores = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
                    if HALF_PV and not COMPACT and CAP <= 0 and not HAS_ALIBI:
                        scores = scores * (q_scale * ks)[:, None]
                    else:
                        scores = scores * (q_scale * ks * SCALE)[:, None]
                    if CAP > 0:
                        scores = CAP * (2 / (1 + tl.exp(2 * (-scores / CAP))) - 1)
                    position = m + nk - nq
                    if HAS_ALIBI:
                        scores -= slope[:, None] * tl.abs(
                            position[:, None] - n[None, :]
                        )
                    if not use_dense or mask_phase == 1:
                        if SEPARATE_MASK:
                            # Invalid query rows are excluded by Q loads and output stores.
                            valid = n[None, :] < nk
                        else:
                            valid = (m[:, None] < nq) & (n[None, :] < nk)
                        if CAUSAL:
                            valid &= n[None, :] <= position[:, None]
                        if LEFT >= 0:
                            valid &= n[None, :] >= position[:, None] - LEFT
                        if RIGHT >= 0:
                            valid &= n[None, :] <= position[:, None] + RIGHT
                    if not (HALF_PV and not COMPACT and CAP <= 0 and not HAS_ALIBI):
                        scores = scores * LOG2E
                    if not use_dense or mask_phase == 1:
                        scores = tl.where(valid, scores, float("-inf"))
                    tile_max = tl.max(scores, 1)
                    new_max = tl.maximum(maximum, tile_max)
                    safe_max = tl.where(new_max == float("-inf"), 0, new_max)
                    alpha = tl.exp2(maximum - safe_max)
                    if HALF_PV:
                        p = tl.exp2(scores - safe_max[:, None])
                        denom = denom * alpha + tl.sum(p, 1)
                    else:
                        # Normalize within this KV tile so probability quantization needs
                        # only a constant multiply. beta brings its sum/PV into the running scale.
                        safe_tile = tl.where(tile_max == float("-inf"), 0, tile_max)
                        p = tl.exp2(scores - safe_tile[:, None])
                        beta = tl.exp2(tile_max - safe_max)
                        denom = denom * alpha + tl.sum(p, 1) * beta
                        if PRECISE_PV:
                            tl.static_assert(phase_blocks[mask_phase] <= 128)
                            p_scale = beta * (1.0 / PRECISE_PROB_LEVELS)
                            fixed = (p * PRECISE_PROB_LEVELS + 0.5).to(tl.int32)
                            p_hi = ((fixed + 128) >> 8).to(tl.int8)
                            p_lo = fixed.to(tl.int8)
                        else:
                            p_scale = beta * (1.0 / PROB_QUANT_LEVELS)
                            # Nonnegative inputs make truncation round-half-up.
                            p_int8 = (
                                (p * PROB_QUANT_LEVELS + 0.5).to(tl.int32) - 128
                            ).to(tl.int8)
                    if PIPE_DEPTH > 0 and use_dense and mask_phase == 0:
                        v = tl.load(tle.gpu.local_ptr(slot.v))
                        reader.release(pipe_index)
                        if pipe_index + PIPE_DEPTH < tile_count:
                            _flash_int8_stage_paged(
                                writer,
                                pipe_index + PIPE_DEPTH,
                                begin_block,
                                K,
                                V,
                                TABLE,
                                batch,
                                kv_head,
                                table_stride,
                                PAGE,
                                pk,
                                sk,
                                hk,
                                pv,
                                sv,
                                hv,
                                BN,
                                D,
                            )
                    elif (
                        not PREFETCH_V
                        and (COMPENSATE_PV or dv != 1)
                        and ASYNC_K
                        and use_dense
                        and mask_phase == 0
                    ):
                        # Transposed V keeps the reduction dimension contiguous
                        # for AIU copies; compensation also benefits on the old layout.
                        # AIU derives pitch from shape, so describe the packed
                        # allocation, including padding for shorter requests.
                        v_ptr = tl.make_block_ptr(
                            V + batch * pv + kv_head * hv,
                            shape=(dv if dv != 1 else nk, D),
                            strides=(sv, dv),
                            offsets=(start * phase_blocks[mask_phase], 0),
                            block_shape=(phase_blocks[mask_phase], D),
                            order=(0, 1) if dv != 1 else (1, 0),
                        )
                        v = tle.load(v_ptr, is_async=True)
                    elif not PREFETCH_V and (not use_dense or mask_phase == 1):
                        v = tl.load(
                            V + v_row[:, None] + kv_head * hv + d[None, :] * dv,
                            n[:, None] < nk,
                            0,
                        )
                    elif not PREFETCH_V:
                        v = tl.load(V + v_row[:, None] + kv_head * hv + d[None, :] * dv)
                    if HALF_PV:
                        p_scaled = p * PV_SCALE if COMPENSATE_PV or SCALED_PV else p
                        p_hi = p_scaled.to(tl.float16)
                        if COMPENSATE_PV:
                            p_lo = (p_scaled - p_hi.to(tl.float32)).to(tl.float16)
                        if vs2 == 0:
                            if BM >= 128:
                                # Rescale the large-tile accumulator before the dot.
                                acc = acc * alpha[:, None]
                                acc = tl.dot(p_hi, v.to(tl.float16), acc)
                            else:
                                acc = tl.dot(
                                    p_hi,
                                    v.to(tl.float16),
                                    acc * alpha[:, None],
                                )
                            if COMPENSATE_PV:
                                acc = tl.dot(p_lo, v.to(tl.float16), acc)
                        else:
                            partial = tl.dot(
                                p_hi, v.to(tl.float16), out_dtype=tl.float32
                            )
                            if COMPENSATE_PV:
                                partial = tl.dot(p_lo, v.to(tl.float16), partial)
                            acc = acc * alpha[:, None] + partial * vs
                    else:
                        if PRECISE_PV:
                            # phase_blocks[mask_phase] <= 128 keeps the combined INT32 result in range.
                            high = tl.dot(p_hi, v, out_dtype=tl.int32)
                            partial = tl.dot(p_lo, v, high << 8, out_dtype=tl.int32).to(
                                tl.float32
                            )
                        else:
                            correction = tl.dot(
                                tl.full((BM, phase_blocks[mask_phase]), -128, tl.int8),
                                v,
                                out_dtype=tl.int32,
                            )
                            partial = tl.dot(
                                p_int8, v, -correction, out_dtype=tl.int32
                            ).to(tl.float32)
                        acc = acc * alpha[:, None] + partial * (p_scale * vs)[:, None]
                    maximum = new_max
            if HALF_PV and vs2 == 0 and KV_SPLITS == 1:
                # Fold the head scale into the row factor before scaling the tile.
                output_scale = v_scale * (
                    PV_UNSCALE if COMPENSATE_PV or SCALED_PV else 1.0
                )
                output_scale = output_scale / tl.where(denom > 0, denom, 1)
                result = acc * output_scale[:, None]
            else:
                if KV_SPLITS > 1:
                    # Defer normalization to the merge instead of rounding twice.
                    result = acc
                else:
                    result = acc / tl.where(denom > 0, denom, 1)[:, None]
                if HALF_PV:
                    if vs2 == 0:
                        result *= v_scale * (
                            PV_UNSCALE if COMPENSATE_PV or SCALED_PV else 1.0
                        )
                    elif COMPENSATE_PV or SCALED_PV:
                        result *= PV_UNSCALE
            tl.store(
                O
                + split_id * OUT_SPLIT_STRIDE
                + (output_q_start + m[:, None]) * so
                + h[:, None] * ho
                + d[None, :],
                result,
                m[:, None] < nq,
            )
            if WRITE_LSE:
                if KV_SPLITS > 1:
                    lse_address = (
                        LSE
                        + split_id * LSE_SPLIT_STRIDE
                        + h * TOTAL_Q
                        + output_q_start
                        + m
                    )
                    tl.store(lse_address, maximum, m < nq)
                    tl.store(lse_address + LSE_SPLIT_STRIDE // 2, denom, m < nq)
                else:
                    lse = tl.where(
                        denom > 0, maximum * LN2 + tl.log(denom), float("inf")
                    )
                    tl.store(LSE + h * TOTAL_Q + output_q_start + m, lse, m < nq)


def flash_attn_varlen_func_w8a8_int8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """PPU INT8 variable-length attention, inference forward only.

    Q is [total_q, heads, D]; K/V are [total_k, kv_heads, D] or paged
    [pages, page_size, kv_heads, D]. D is 64, 128 or 256, heads must be a multiple
    of kv_heads, and the last dimension must be contiguous. Q/K/V are INT8.
    Required FP32 descales have shape [batch, heads, ceil(max_length / 128)]
    (kv_heads for K/V), indexed by logical sequence position even for paged KV.
    Each real value is the INT8 value multiplied by its descale.

    QK uses INT8 dot with INT32 accumulation. Paged long-query paths pack
    K/V by KV head and convert V to FP16. The D=128, GQA=4 fast path gathers
    logical KV positions while staying within the physical-cache workspace
    budget; D=256 uses logical packing within a 512 MiB scratch budget.
    Other long-query paths preserve physical page indices. When the
    compiler provides PPU AIU loading, dense tiles use asynchronous INT8 K
    copies. Long logical-query paths pack V with a contiguous reduction dimension
    for asynchronous FP16 V copies. Smaller queries keep the original V layout;
    compensated PV uses asynchronous V copies on either layout. Masked boundary
    tiles retain regular loads. Folded AIU tiles group query rows by head.
    Dense causal prefill visits query tiles in descending order. Small mixed
    batches interleave the long-request worklist by query tile. Packed FP16-PV
    boundary blocks use 16 KV positions when the query tile is narrower than
    the dense KV block. Dense blocks keep their original width. Selected
    short-query paths also use FP16 PV, converting V inside the kernel.
    Four-token paged GQA may stage raw K/V in a shared buffer when the
    compiler supports TLE pipes; partial KV tiles retain masked loads.
    Remaining paths quantize probabilities to 256 levels and use INT8 PV
    with zero-point correction. D=128, GQA=4 queries of up to four tokens with
    at most 256 KV positions use two signed INT8 probability components to reduce
    quantization error. Small-batch single-token decode splits KV into two
    independent intervals when there are at most 64 parallel KV heads and
    at least 512 maximum KV positions. Partial FP32 accumulators and softmax
    statistics are merged by a Triton kernel. Softmax and online accumulation
    use FP32.
    D=256 accepts a split-count hint capped by available parallelism and a
    256 MiB workspace budget. Its scheduler metadata is validated but the
    kernel builds its own schedule. Single-query, single-key requests directly
    dequantize V and calculate QK only when LSE is requested.
    Mixed batches reuse the long-request worklist. Short requests with long KV
    split their reduction into compact per-request workspaces, avoiding partial
    buffers for the long-prefill rows. The existing merge computation maps these
    compact rows back to their original output positions.
    Packing requires at most three bytes per element of the physical K cache.
    Output is BF16 by default, or uses the supplied FP16/BF16 out buffer.
    LSE is [heads, total_q], FP32. Fully masked rows return zero and LSE +inf,
    following FlashAttention's convention. Inputs and out must not overlap.
    """
    if dropout_p != 0 or return_attn_probs or q_v is not None:
        raise NotImplementedError(
            "Only attention inference without dropout is supported"
        )
    if q.shape[-1] == 256:
        if isinstance(num_splits, bool) or not isinstance(num_splits, int):
            raise TypeError(f"num_splits must be an integer in [0, {MAX_KV_SPLITS}]")
        if not 0 <= num_splits <= MAX_KV_SPLITS:
            raise ValueError(f"num_splits must be in [0, {MAX_KV_SPLITS}]")
        if scheduler_metadata is not None and (
            scheduler_metadata.device != q.device
            or scheduler_metadata.dtype != torch.int32
            or scheduler_metadata.ndim != 1
            or not scheduler_metadata.is_contiguous()
        ):
            raise ValueError(
                "scheduler_metadata must be a contiguous int32 vector on q.device"
            )
    elif scheduler_metadata is not None or num_splits != 0:
        raise NotImplementedError(
            "Scheduler and explicit split-KV hints require head dimension 256"
        )
    if s_aux is not None:
        raise NotImplementedError("Auxiliary attention is unsupported")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("Context parallel attention is unsupported")
    if fa_version != 2:
        raise NotImplementedError("Only FA2 is implemented")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise NotImplementedError("Q/K/V must use INT8")
    if q.shape[-1] not in (64, 128, 256):
        raise NotImplementedError("Only head dimensions 64, 128 and 256 are supported")
    if seqused_k is not None and block_table is None:
        raise NotImplementedError("seqused_k requires paged KV")

    batch = cu_seqlens_q.numel() - 1
    total, heads, dim = q.shape
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = (
        torch.empty((heads, total), dtype=torch.float32, device=q.device)
        if return_softmax_lse
        else None
    )
    if total == 0:
        return (out, lse) if return_softmax_lse else out
    left, right = (-1, -1) if window_size is None else window_size
    paged = block_table is not None
    group = heads // k.shape[-2]
    kv_heads = k.shape[-2]
    long_query = paged and max_seqlen_q >= 128
    pack_kv = long_query and k.shape[0] > 0
    common_gqa = paged and dim == 128 and group == 4
    # Do not exceed the physical-cache workspace budget for shared KV pages.
    logical_kv = (
        pack_kv
        and (common_gqa or dim == 256)
        and max_seqlen_k > 0
        and batch * max_seqlen_k <= k.shape[0] * k.shape[1]
    )
    if dim == 256:
        # Do not scan or duplicate a model's entire physical cache allocation.
        logical_kv = (
            logical_kv
            and max_seqlen_q >= 1024
            and batch * max_seqlen_k * kv_heads * dim * 3 <= KV_PACK_MAX_BYTES
        )
        pack_kv = logical_kv
    elif pack_kv and not logical_kv and k.numel() >= 2**31:
        # Physical packing uses int32 destinations. Read the original cache
        # directly when its extent exceeds that representation.
        pack_kv = False
    use_aiu_k = logical_kv and HAS_AIU_K and dim != 256
    # Transposed V amortizes its pipeline setup only beyond small prefill tiles.
    transpose_v = use_aiu_k and max_seqlen_q >= TRANSPOSE_V_MIN_QUERY_LENGTH
    reorder_causal = use_aiu_k and causal and left < 0 and right < 0
    block_m = 64 if not paged and not causal and max_seqlen_q >= 512 else 16
    fold = paged and group > 1 and group <= 16 and group & (group - 1) == 0
    if fold:
        block_m = 32 if max_seqlen_q > 16 else 16
    block_n = 128 if long_query else 64
    num_warps = 8 if block_m == 64 and dim == 128 else 4
    num_stages = 3 if long_query else 1
    half_pv = long_query
    if logical_kv:
        # A 64-column tile reduces pressure on the long-query accumulator.
        block_m, block_n, num_warps = 64, 64, 4
        fold = reorder_causal or max_seqlen_q > 512
        if not use_aiu_k and max_seqlen_q >= 4096 and left < 0 and right < 0:
            block_m = 128
            num_stages = 2 if max_seqlen_q < 8192 else 3
    elif common_gqa and max_seqlen_q <= 16:
        parallel_heads = batch * kv_heads
        if max_seqlen_q > 4:
            # Reuse each KV tile across more short-query rows.
            block_m, block_n, num_stages, half_pv = 32, 128, 3, True
            num_stages = 2 if left < 0 and right < 0 else 3
        elif max_seqlen_q == 1 and parallel_heads <= 64:
            block_n, num_warps, half_pv = 128, 8, True
        elif left < 0 and right < 0:
            # One/two-token GQA uses at most eight rows; avoid padded work.
            block_m = 8 if max_seqlen_q <= 2 else 16
            block_n, num_warps, num_stages = 128, 2, 2
        elif parallel_heads > 128:
            block_n = 128
    if dim == 256:
        fold = (
            (max_seqlen_q <= D256_FOLDED_QUERY_LIMIT or logical_kv)
            and group > 1
            and group <= 16
            and group & (group - 1) == 0
        )
        block_m = 64 if logical_kv else 32 if max_seqlen_q > 16 else 16
        block_n, num_warps, half_pv = 64, 4, True
        num_stages = 2 if logical_kv or max_seqlen_k >= 512 else 1
    split_kv = (
        common_gqa
        and max_seqlen_q == 1
        and batch * kv_heads <= SPLIT_KV_MAX_PARALLEL_HEADS.value
        and max_seqlen_k >= SPLIT_KV_MIN_LENGTH.value
        and left < 0
        and right < 0
    )
    if split_kv:
        # More CTAs improve occupancy when one-token decode has few KV heads.
        block_m, block_n, num_warps, num_stages, half_pv = 8, 128, 2, 2, False
    splits = DECODE_KV_SPLITS.value if split_kv else 1
    if dim == 256 and num_splits > 1:
        # The hint is an upper bound: already-parallel batches need fewer splits.
        parallel_heads = batch * (kv_heads if fold else heads)
        workspace_cap = max(1, SPLIT_KV_MAX_BYTES // (total * heads * (dim + 2) * 4))
        splits = min(
            num_splits,
            max(1, triton.cdiv(SPLIT_KV_TARGET_BLOCKS, parallel_heads)),
            max(1, triton.cdiv(max_seqlen_k, block_n)),
            workspace_cap,
        )
    split_kv = splits > 1
    query_tile = block_m // group if fold else block_m
    grid_heads = kv_heads if fold else heads
    compact = paged and max_seqlen_q > 16 and total * 2 < batch * max_seqlen_q
    split_queries = (
        compact
        and group <= 16
        and group & (group - 1) == 0
        and ((logical_kv and not split_kv) or (dim == 256 and split_kv))
    )
    tile_pitch = triton.next_power_of_2(triton.cdiv(max_seqlen_q, query_tile))
    split_queries = split_queries and batch * tile_pitch < 2**31
    if split_queries:
        compact = False
    elif logical_kv and compact:
        num_stages = 2
    short_splits = 1
    if (
        dim == 256
        and split_queries
        and not split_kv
        and max_seqlen_k >= SHORT_SPLIT_KV_MIN_LENGTH
    ):
        short_splits = min(
            MAX_KV_SPLITS,
            max(1, triton.cdiv(SPLIT_KV_TARGET_BLOCKS, batch * kv_heads)),
            triton.cdiv(max_seqlen_k, 64),
        )
    work_capacity = triton.cdiv(total, query_tile) + batch - 1
    work = None
    raw_k, raw_v = k, v
    grid = (
        (triton.cdiv(total, query_tile) + batch - 1, 1, grid_heads)
        if compact
        else (triton.cdiv(max_seqlen_q, query_tile), batch, grid_heads)
    )
    with torch_device_fn.device(q.device):
        kernel_out, kernel_stats = out, lse
        if split_kv:
            kernel_out = torch.empty(
                (splits, total, heads, dim),
                dtype=torch.float32,
                device=q.device,
            )
            kernel_stats = torch.empty(
                (splits, 2, heads, total),
                dtype=torch.float32,
                device=q.device,
            )
            grid = (grid[0] * splits, grid[1], grid[2])
        if split_queries:
            work = torch.empty(
                (work_capacity + (3 if short_splits > 1 else 1),),
                dtype=torch.int32,
                device=q.device,
            )
            _flash_int8_prepare_worklist[(batch,)](
                cu_seqlens_q,
                work,
                batch,
                query_tile,
                work_capacity,
                tile_pitch,
                ROUND_ROBIN=reorder_causal and batch <= WORKLIST_INTERLEAVE_MAX_BATCH,
                SHORT_STATS=short_splits > 1,
            )
            grid = (work_capacity * splits, 1, grid_heads)
        if pack_kv:
            if logical_kv:
                n = max_seqlen_k
                packed_shape = (batch, n, kv_heads, dim)
                packed_stride = (kv_heads * n * dim, dim, n * dim, 1)
                pack_grid = (triton.cdiv(n, PACK_TILE.value), batch, kv_heads)
            else:
                n = k.shape[0] * k.shape[1]
                packed_shape = k.shape
                packed_stride = (k.shape[1] * dim, dim, n * dim, 1)
                pack_grid = (triton.cdiv(n, PACK_TILE.value), kv_heads)
            kp = torch.empty_strided(
                packed_shape, packed_stride, dtype=torch.int8, device=k.device
            )
            vp = torch.empty_strided(
                packed_shape,
                (kv_heads * n * dim, 1, n * dim, n) if transpose_v else packed_stride,
                dtype=torch.float16,
                device=v.device,
            )
            _flash_int8_pack_kv[pack_grid](
                k,
                v,
                kp,
                vp,
                block_table,
                seqused_k,
                cu_seqlens_q,
                n,
                dim,
                kv_heads,
                k.shape[1],
                block_table.stride(0),
                *k.stride()[:3],
                *v.stride()[:3],
                LOGICAL=logical_kv,
                TRANSPOSE_V=transpose_v,
                SKIP_SHORT=split_queries,
            )
            k, v = kp, vp
        for phase in range(2 if split_queries else 1):
            short_phase = phase == 1
            phase_splits, kernel_total = splits, total
            if short_phase:
                k, v = raw_k, raw_v
                fold, logical_kv, compact = True, False, False
                block_m, num_warps, num_stages, half_pv = 16, 4, 1, False
                # An occupancy hint only; GPU CUQ values still decide which
                # requests execute, including empty and padded requests.
                min_long = triton.cdiv(
                    max(total - batch * SHORT_QUERY_LIMIT.value, 0),
                    max_seqlen_q - SHORT_QUERY_LIMIT.value,
                )
                short_head_bound = max(batch - min_long, 0) * kv_heads
                block_n = 128 if short_head_bound > 128 else 64
                if short_head_bound <= 64:
                    block_n, num_warps, half_pv = 128, 8, True
                elif left < 0 and right < 0:
                    block_n, num_warps, num_stages = 128, 2, 2
                if dim == 256:
                    block_m, block_n, num_warps, num_stages, half_pv = (
                        8 if split_kv and group <= 8 else 16,
                        64,
                        2 if split_kv and group <= 8 else 4,
                        2 if split_kv or short_splits > 1 else 1,
                        True,
                    )
                if short_splits > 1:
                    phase_splits = short_splits
                    kernel_total = batch * SHORT_QUERY_LIMIT.value
                    kernel_out = torch.empty(
                        (phase_splits, kernel_total, heads, dim),
                        dtype=torch.float32,
                        device=q.device,
                    )
                    kernel_stats = torch.empty(
                        (phase_splits, 2, heads, kernel_total),
                        dtype=torch.float32,
                        device=q.device,
                    )
                grid = (phase_splits, batch, kv_heads)
            reorder_phase = reorder_causal and not short_phase
            launch_grid = (grid[2], grid[1], grid[0]) if reorder_phase else grid
            # Uniform head scales admit a cheap error bound before either loop.
            # The launches have complementary guards, so each CTA writes only once.
            pv_passes = (
                (False, True) if half_pv and v_descale.stride(2) == 0 else (True,)
            )
            for compensate_pv in pv_passes:
                _flash_int8_fwd[launch_grid](
                    q,
                    k,
                    v,
                    kernel_out,
                    kernel_stats,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqused_k,
                    block_table,
                    work,
                    q_descale,
                    k_descale,
                    v_descale,
                    alibi_slopes,
                    q.stride(0),
                    q.stride(1),
                    k.stride(-3),
                    k.stride(-2),
                    k.stride(0) if paged else 0,
                    v.stride(-3),
                    v.stride(-2),
                    v.stride(-1),
                    v.stride(0) if paged else 0,
                    kernel_out.stride(-3),
                    kernel_out.stride(-2),
                    *q_descale.stride(),
                    *k_descale.stride(),
                    *v_descale.stride(),
                    block_table.stride(0) if paged else 0,
                    (
                        alibi_slopes.stride(0)
                        if alibi_slopes is not None and alibi_slopes.ndim == 2
                        else 0
                    ),
                    kernel_total,
                    heads // k.shape[-2],
                    dim,
                    k.shape[1] if paged else 0,
                    paged,
                    causal,
                    left,
                    right,
                    softcap,
                    dim**-0.5 if softmax_scale is None else softmax_scale,
                    alibi_slopes is not None,
                    return_softmax_lse or phase_splits > 1,
                    HALF_PV=half_pv,
                    COMPENSATE_PV=compensate_pv,
                    SCALED_PV=max_seqlen_k > PV_UNSCALED_MAX_LENGTH,
                    ASYNC_K=use_aiu_k and not short_phase,
                    PIPE_DEPTH=(
                        1
                        if HAS_PAGED_PIPE
                        and common_gqa
                        and max_seqlen_q == 4
                        and left < 0
                        and right < 0
                        else 0
                    ),
                    PREFETCH_V=(
                        common_gqa
                        and 4 < max_seqlen_q <= SHORT_QUERY_LIMIT.value
                        and left < 0
                        and right < 0
                    ),
                    REORDER_CAUSAL=reorder_phase,
                    TAIL_N=BOUNDARY_KV_TILE.value,
                    KV_SPLITS=phase_splits,
                    OUT_SPLIT_STRIDE=kernel_out.stride(0) if phase_splits > 1 else 0,
                    LSE_SPLIT_STRIDE=kernel_stats.stride(0) if phase_splits > 1 else 0,
                    PRECISE_PV=(
                        common_gqa
                        and max_seqlen_q <= 4
                        and max_seqlen_k <= 2 * DESCALE_BLOCK.value
                    ),
                    SEPARATE_MASK=common_gqa or dim == 256,
                    LOGICAL_KV=logical_kv,
                    BATCH=batch,
                    COMPACT=compact,
                    USE_WORKLIST=split_queries and not short_phase,
                    WORK_CAPACITY=(
                        work_capacity if split_queries and not short_phase else 0
                    ),
                    TILE_PITCH=tile_pitch if split_queries and not short_phase else 1,
                    SHORT_ONLY=short_phase,
                    SHORT_OUTPUT=short_phase and short_splits > 1,
                    FOLD=fold,
                    BM=block_m,
                    BN=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            if short_phase and short_splits > 1:
                flash_int8_merge_short_splits[(SHORT_QUERY_LIMIT.value, heads, batch)](
                    kernel_out,
                    kernel_stats,
                    out,
                    lse,
                    cu_seqlens_q,
                    work,
                    work_capacity,
                    total,
                    kernel_total,
                    heads,
                    dim,
                    phase_splits,
                    out.stride(0),
                    out.stride(1),
                    return_softmax_lse,
                    num_warps=1,
                )
        if split_kv:
            _flash_int8_merge_splits[(total, heads)](
                kernel_out,
                kernel_stats,
                out,
                lse,
                cu_seqlens_q,
                total,
                heads,
                dim,
                batch,
                splits,
                out.stride(0),
                out.stride(1),
                return_softmax_lse,
                num_warps=1,
            )
    return (out, lse) if return_softmax_lse else out


@libentry()
@triton.jit
def _flash_int8_merge_splits(
    PARTIAL,
    PARTIAL_STATS,
    OUT,
    LSE,
    CUQ,
    TOTAL_Q: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BATCH: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    first_q = tl.load(CUQ)
    end_q = tl.load(CUQ + BATCH)
    valid_q = (row >= first_q) & (row < end_q)
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    d = tl.arange(0, D)
    maxima = tl.load(
        PARTIAL_STATS + (splits * 2 * HEADS + head) * TOTAL_Q + row,
        (splits < SPLITS) & valid_q,
        float("-inf"),
    )
    maximum = tl.max(maxima, 0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = tl.exp2(maxima - safe_max)
    denominators = tl.load(
        PARTIAL_STATS + ((splits * 2 + 1) * HEADS + head) * TOTAL_Q + row,
        (splits < SPLITS) & valid_q,
        0.0,
    )
    denominator = tl.sum(weight * denominators, 0)
    values = tl.load(
        PARTIAL + ((splits[:, None] * TOTAL_Q + row) * HEADS + head) * D + d[None, :],
        (splits[:, None] < SPLITS) & valid_q,
        0.0,
    )
    result = tl.sum(values * weight[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(OUT + row * OUT_ROW + head * OUT_HEAD + d, result, valid_q)
    if WRITE_LSE:
        result_lse = tl.where(
            denominator > 0, maximum * LN2 + tl.log(denominator), float("inf")
        )
        tl.store(LSE + head * TOTAL_Q + row, result_lse, valid_q)


merge_splits_jit = _flash_int8_merge_splits.fn


@libentry()
@triton.jit
def flash_int8_merge_short_splits(
    PARTIAL,
    STATS,
    OUT,
    LSE,
    CUQ,
    WORK,
    CAPACITY: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SHORT_TOTAL: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    row, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin = tl.load(CUQ + batch)
    length = tl.load(CUQ + batch + 1) - begin
    if (length <= SHORT_QUERY_LIMIT) & (row < length):
        if WRITE_LSE:
            LSE = LSE + begin + head * (TOTAL_Q - SHORT_TOTAL)
        # Reuse normalization, remapping only the compact short-request rows.
        merge_splits_jit(
            PARTIAL + batch * SHORT_QUERY_LIMIT * HEADS * D,
            STATS + batch * SHORT_QUERY_LIMIT,
            OUT + begin * OUT_ROW,
            LSE,
            WORK + CAPACITY + 1,
            SHORT_TOTAL,
            HEADS,
            D,
            1,
            SPLITS,
            OUT_ROW,
            OUT_HEAD,
            WRITE_LSE,
        )
