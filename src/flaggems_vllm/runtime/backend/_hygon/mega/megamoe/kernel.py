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

"""Hygon prequantized single-launch W8A8 MegaMoE kernel.

This file is the HCU rewrite of NVIDIA v234's monolithic ``ws_megakernel``.
Every rank launches exactly one persistent grid. That grid publishes an
already-quantized qA1 input, gathers each peer token exactly once, constructs
the dynamic route queue, executes two W8A8 expert projections, returns the
partials, and performs the ordered top-k combine before it exits.

Unsupported Hopper pipeline, warp-specialization, and WGMMA APIs are replaced
by an explicit wave64-friendly phase schedule and ordinary ``tl.dot``. Each
source qA1 row is copied from its DUSHMEM peer mapping into an all-gathered
local token cache once, then reused for every local expert route and every L1
N tile. DUSHMEM allocation and peer mapping are caller responsibilities; the
kernel accesses the prepared mappings with ordinary loads/stores and
system-scope release/acquire atomics.
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _peer_i8(table, pe):
    return tl.load(table + pe).to(tl.pointer_type(tl.int8))


@triton.jit
def _peer_i32(table, pe):
    return tl.load(table + pe).to(tl.pointer_type(tl.int32))


@triton.jit
def _peer_i64(table, pe):
    return tl.load(table + pe).to(tl.pointer_type(tl.int64))


@triton.jit
def _peer_f32(table, pe):
    return tl.load(table + pe).to(tl.pointer_type(tl.float32))


@triton.jit
def _peer_bf16(table, pe):
    return tl.load(table + pe).to(tl.pointer_type(tl.bfloat16))


@triton.jit
def _quantize_symmetric_int8(x, scale):
    # Match vLLM-HCU's dynamic INT8 qA2 exactly.  In particular, nearbyint
    # uses round-to-nearest-even; the old half-away-from-zero expression
    # accumulated a measurable relative-L2 delta across the eight routes.
    rounded = libdevice.nearbyint(x / scale)
    return tl.maximum(-127.0, tl.minimum(127.0, rounded)).to(tl.int8)


@triton.jit
def _local_epoch_barrier(
    done,
    released,
    sync_status,
    epoch,
    PHASE: tl.constexpr,
    NCTA: tl.constexpr,
    NCTA_POW2: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    """Grid barrier for a deliberately fully-resident persistent grid."""

    pid = tl.program_id(0)
    tl.debug_barrier()
    tl.atomic_xchg(
        done + PHASE * NCTA + pid,
        epoch,
        sem="release",
        scope="sys",
    )
    if pid == 0:
        ctas = tl.arange(0, NCTA_POW2)
        live = ctas < NCTA
        observed = tl.atomic_add(
            done + PHASE * NCTA + ctas,
            0,
            mask=live,
            sem="acquire",
            scope="sys",
        )
        minimum = tl.min(tl.where(live, observed, epoch), axis=0)
        spins = 0
        while (minimum < epoch) & (spins < MAX_SPINS):
            observed = tl.atomic_add(
                done + PHASE * NCTA + ctas,
                0,
                mask=live,
                sem="acquire",
                scope="sys",
            )
            minimum = tl.min(tl.where(live, observed, epoch), axis=0)
            spins += 1
        if minimum < epoch:
            tl.atomic_xchg(sync_status, 1, sem="release", scope="sys")
        tl.atomic_xchg(
            released + PHASE,
            epoch,
            sem="release",
            scope="sys",
        )
    else:
        ready = tl.atomic_add(
            released + PHASE,
            0,
            sem="acquire",
            scope="sys",
        )
        spins = 0
        while (ready < epoch) & (spins < MAX_SPINS):
            ready = tl.atomic_add(
                released + PHASE,
                0,
                sem="acquire",
                scope="sys",
            )
            spins += 1
        if ready < epoch:
            tl.atomic_xchg(sync_status, 1, sem="release", scope="sys")
    tl.debug_barrier()


@triton.jit
def _rank_epoch_barrier(
    flag_table,
    done,
    released,
    rank_released,
    sync_status,
    epoch,
    LOCAL_PHASE: tl.constexpr,
    RANK_PHASE: tl.constexpr,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NPES_POW2: tl.constexpr,
    NCTA: tl.constexpr,
    NCTA_POW2: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    """Publish all preceding peer traffic and wait for every rank."""

    # With one CTA there is no grid rendezvous to perform.  Keeping the
    # atomic/spin barrier in front of the peer barrier triggers a gfx936 HCU
    # lowering/runtime deadlock even though each barrier passes in isolation.
    # A workgroup barrier is the exact required operation for the single-CTA
    # bring-up specialization.  Multi-CTA specializations retain the epoch
    # rendezvous below.
    if NCTA == 1:
        tl.debug_barrier()
    else:
        _local_epoch_barrier(
            done,
            released,
            sync_status,
            epoch,
            LOCAL_PHASE,
            NCTA,
            NCTA_POW2,
            MAX_SPINS,
        )
    pid = tl.program_id(0)
    if pid == 0:
        for peer in tl.static_range(0, NPES):
            # The HCU/DUSHMEM probe for this environment established the
            # release/acquire protocol with 32-bit system-scope atomics.  Keep
            # the production barrier on that proven representation instead of
            # relying on unverified gfx936 i64 atomic lowering.
            remote_flags = _peer_i32(flag_table, peer)
            tl.atomic_xchg(
                remote_flags + RANK_PHASE * NPES + MY_PE,
                epoch,
                sem="release",
                scope="sys",
            )
        local_flags = _peer_i32(flag_table, MY_PE)
        peers = tl.arange(0, NPES_POW2)
        live = peers < NPES
        observed = tl.atomic_add(
            local_flags + RANK_PHASE * NPES + peers,
            0,
            mask=live,
            sem="acquire",
            scope="sys",
        )
        minimum = tl.min(tl.where(live, observed, epoch), axis=0)
        spins = 0
        while (minimum < epoch) & (spins < MAX_SPINS):
            observed = tl.atomic_add(
                local_flags + RANK_PHASE * NPES + peers,
                0,
                mask=live,
                sem="acquire",
                scope="sys",
            )
            minimum = tl.min(tl.where(live, observed, epoch), axis=0)
            spins += 1
        if minimum < epoch:
            tl.atomic_xchg(sync_status, 1, sem="release", scope="sys")
        tl.atomic_xchg(
            rank_released + RANK_PHASE,
            epoch,
            sem="release",
            scope="sys",
        )
    else:
        ready = tl.atomic_add(
            rank_released + RANK_PHASE,
            0,
            sem="acquire",
            scope="sys",
        )
        spins = 0
        while (ready < epoch) & (spins < MAX_SPINS):
            ready = tl.atomic_add(
                rank_released + RANK_PHASE,
                0,
                sem="acquire",
                scope="sys",
            )
            spins += 1
        if ready < epoch:
            tl.atomic_xchg(sync_status, 1, sem="release", scope="sys")
    tl.debug_barrier()


@triton.jit(do_not_specialize=["epoch"])
def hcu_megamoe(
    hidden_q,
    hidden_scale,
    topk_weights,
    topk_ids,
    w1_gate_t,
    w1_up_t,
    w1_gate_scale,
    w1_up_scale,
    w2_t,
    w2_scale,
    final_output,
    q_local,
    qscale_local,
    ids_local,
    route_weights_local,
    partials_local,
    q_table,
    qscale_table,
    ids_table,
    route_weights_table,
    partials_table,
    flag_table,
    expert_counts,
    expert_offsets,
    expert_write,
    expert_block_offsets,
    route_queue,
    gathered_q,
    block_to_expert,
    gathered_qscale,
    queue_route_weight,
    activation_tmp,
    activation_amax,
    q2,
    q2_scale,
    owner_accum,
    phase_done,
    phase_released,
    rank_released,
    sync_status,
    epoch,
    MY_PE: tl.constexpr,
    NPES: tl.constexpr,
    NPES_POW2: tl.constexpr,
    NCTA: tl.constexpr,
    NCTA_POW2: tl.constexpr,
    NTOK: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    NEXP: tl.constexpr,
    EPR: tl.constexpr,
    HIDDEN: tl.constexpr,
    INTER: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NI: tl.constexpr,
    NH: tl.constexpr,
    NI_POW2: tl.constexpr,
    AGGREGATE_RETURN: tl.constexpr,
    SYNC_MAX_SPINS: tl.constexpr,
):
    """Complete dynamic W8A8 MegaMoE forward in one per-rank launch."""

    # FlagTree PR1345's Triton AST does not lower Python ``in`` inside a
    # static assertion.  Spell the finite set out so the check remains a
    # compile-time condition on every specialization.
    tl.static_assert(
        (NPES == 2) or (NPES == 4) or (NPES == 8),
        "MegaMoE supports EP2/EP4/EP8",
    )
    tl.static_assert(NEXP % NPES == 0, "experts must shard evenly")
    tl.static_assert(EPR * NPES == NEXP, "EPR must match the expert shard")
    tl.static_assert(HIDDEN % BLOCK_Q == 0, "qA1 tile must divide H")
    tl.static_assert(HIDDEN % BLOCK_K == 0, "GEMM K tile must divide H")
    tl.static_assert(INTER % BLOCK_K == 0, "GEMM K tile must divide I")
    tl.static_assert(INTER % BLOCK_N == 0, "L1 N tile must divide I")
    tl.static_assert(HIDDEN % BLOCK_N == 0, "L2 N tile must divide H")
    tl.static_assert(NI * BLOCK_N == INTER, "NI must cover I exactly")
    tl.static_assert(NH * BLOCK_N == HIDDEN, "NH must cover H exactly")
    tl.static_assert(NPES_POW2 >= NPES, "peer reduction extent is too small")
    tl.static_assert(TOPK_POW2 >= TOPK, "top-k vector extent is too small")
    tl.static_assert(NCTA_POW2 >= NCTA, "CTA reduction extent is too small")
    tl.static_assert(NI_POW2 >= NI, "activation reduction extent is too small")
    tl.static_assert(
        (BLOCK_M == 16) or (BLOCK_M == 32) or (BLOCK_M == 64) or (BLOCK_M == 128),
        "gfx936 M tile must be >=16 and a multiple of 16",
    )
    tl.static_assert(
        (BLOCK_N == 32) or (BLOCK_N == 64) or (BLOCK_N == 128),
        "gfx936 N tile must be one of 32/64/128",
    )
    tl.static_assert(
        (BLOCK_K == 32) or (BLOCK_K == 64) or (BLOCK_K == 128),
        "gfx936 INT8 MMAC K tile must be >=32 and a multiple of 32",
    )
    tl.static_assert(
        (BLOCK_Q == 16)
        or (BLOCK_Q == 32)
        or (BLOCK_Q == 64)
        or (BLOCK_Q == 128)
        or (BLOCK_Q == 256),
        "qA1 tile must be a supported power of two",
    )
    tl.static_assert(
        (BLOCK_R == 4)
        or (BLOCK_R == 8)
        or (BLOCK_R == 16)
        or (BLOCK_R == 32)
        or (BLOCK_R == 64)
        or (BLOCK_R == 128),
        "route tile must be a supported power of two",
    )

    pid = tl.program_id(0)
    routes_per_rank = NTOK * TOPK
    total_routes = NPES * routes_per_rank

    mi = tl.arange(0, BLOCK_M)
    ni = tl.arange(0, BLOCK_N)
    ki = tl.arange(0, BLOCK_K)

    # Large-token NP2/NP4 specializations aggregate all routes owned by this
    # rank before returning one contribution per source token.  Clear the
    # FP32 owner accumulator while the same persistent grid publishes qA1;
    # the phase-0 rank barrier below also fences these local stores before L2.
    if AGGREGATE_RETURN:
        zero_task = pid
        zero_tasks = NPES * NTOK * NH
        while zero_task < zero_tasks:
            global_token = zero_task // NH
            nblock = zero_task - global_token * NH
            ncols = nblock * BLOCK_N + ni
            tl.store(
                owner_accum + global_token * HIDDEN + ncols,
                tl.zeros((BLOCK_N,), dtype=tl.float32),
            )
            zero_task += NCTA

    # Phase 0: publish the prequantized qA1 input and live routes into the
    # symmetric slab.  qA1 is deliberately outside this version's timing
    # boundary; the unavoidable symmetric staging copy remains inside it.
    rlane = tl.arange(0, BLOCK_R)
    token0 = pid * BLOCK_R
    token_stride = NCTA * BLOCK_R
    qcols = tl.arange(0, BLOCK_Q)
    topk_col = tl.arange(0, TOPK_POW2)
    while token0 < NTOK:
        tokens = token0 + rlane
        token_live = tokens < NTOK
        scale = tl.load(
            hidden_scale + tokens,
            mask=token_live,
            other=0.0,
        ).to(tl.float32)
        tl.store(qscale_local + tokens, scale, mask=token_live)
        for h0 in tl.range(0, HIDDEN, BLOCK_Q):
            hcols = h0 + qcols
            q = tl.load(
                hidden_q + tokens[:, None] * HIDDEN + hcols[None, :],
                mask=token_live[:, None],
                other=0,
            )
            tl.store(
                q_local + tokens[:, None] * HIDDEN + hcols[None, :],
                q,
                mask=token_live[:, None],
            )
        tk_mask = token_live[:, None] & (topk_col[None, :] < TOPK)
        ids = tl.load(
            topk_ids + tokens[:, None] * TOPK + topk_col[None, :],
            mask=tk_mask,
            other=-1,
        ).to(tl.int32)
        weights = tl.load(
            topk_weights + tokens[:, None] * TOPK + topk_col[None, :],
            mask=tk_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            ids_local + tokens[:, None] * TOPK + topk_col[None, :],
            ids,
            mask=tk_mask,
        )
        tl.store(
            route_weights_local + tokens[:, None] * TOPK + topk_col[None, :],
            weights,
            mask=tk_mask,
        )
        token0 += token_stride

    _rank_epoch_barrier(
        flag_table,
        phase_done,
        phase_released,
        rank_released,
        sync_status,
        epoch,
        0,
        0,
        MY_PE,
        NPES,
        NPES_POW2,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )

    # Phase 1: all-gather qA1 once per source token and build an expert-major
    # compact route queue.  The token cache makes inter-rank qA1 traffic
    # independent of the number of local expert hits.
    global_token0 = pid * BLOCK_R
    global_tokens = NPES * NTOK
    global_token_stride = NCTA * BLOCK_R
    while global_token0 < global_tokens:
        global_token = global_token0 + rlane
        token_live = global_token < global_tokens
        source_rank = global_token // NTOK
        source_token = global_token - source_rank * NTOK
        safe_source_rank = tl.where(token_live, source_rank, 0)
        source_q = _peer_i8(q_table, safe_source_rank)
        source_scale = _peer_f32(qscale_table, safe_source_rank)
        tl.store(
            gathered_qscale + global_token,
            tl.load(source_scale + source_token, mask=token_live, other=0.0),
            mask=token_live,
        )
        for h0 in tl.range(0, HIDDEN, BLOCK_Q):
            hcols = h0 + qcols
            cached_q = tl.load(
                source_q[:, None] + source_token[:, None] * HIDDEN + hcols[None, :],
                mask=token_live[:, None],
                other=0,
            )
            tl.store(
                gathered_q + global_token[:, None] * HIDDEN + hcols[None, :],
                cached_q,
                mask=token_live[:, None],
            )
        global_token0 += global_token_stride

    eidx = pid
    while eidx < EPR:
        tl.store(expert_counts + eidx, 0)
        tl.store(expert_write + eidx, 0)
        eidx += NCTA
    if pid == 0:
        tl.store(route_queue, 0)
    _local_epoch_barrier(
        phase_done,
        phase_released,
        sync_status,
        epoch,
        1,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )

    route0 = pid * BLOCK_R
    route_stride = NCTA * BLOCK_R
    while route0 < total_routes:
        gid = route0 + rlane
        live = gid < total_routes
        src = gid // routes_per_rank
        local_route = gid - src * routes_per_rank
        safe_src = tl.where(live, src, 0)
        expert = tl.load(
            _peer_i32(ids_table, safe_src) + local_route,
            mask=live,
            other=-1,
        )
        owner = expert // EPR
        local_expert = expert - owner * EPR
        owned = live & (expert >= 0) & (expert < NEXP) & (owner == MY_PE)
        tl.atomic_add(
            expert_counts + tl.where(owned, local_expert, 0),
            1,
            mask=owned,
            sem="relaxed",
            scope="gpu",
        )
        route0 += route_stride

    _local_epoch_barrier(
        phase_done,
        phase_released,
        sync_status,
        epoch,
        2,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )
    if pid == 0:
        route_prefix = tl.full((), 0, tl.int32)
        block_prefix = tl.full((), 0, tl.int32)
        for expert in tl.range(0, EPR):
            count = tl.load(expert_counts + expert)
            tl.store(expert_offsets + expert, route_prefix)
            tl.store(expert_block_offsets + expert, block_prefix)
            route_prefix += count
            block_prefix += (count + BLOCK_M - 1) // BLOCK_M
        tl.store(expert_offsets + EPR, route_prefix)
        tl.store(expert_block_offsets + EPR, block_prefix)
    _local_epoch_barrier(
        phase_done,
        phase_released,
        sync_status,
        epoch,
        3,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )

    route0 = pid * BLOCK_R
    while route0 < total_routes:
        gid = route0 + rlane
        live = gid < total_routes
        src = gid // routes_per_rank
        local_route = gid - src * routes_per_rank
        safe_src = tl.where(live, src, 0)
        expert = tl.load(
            _peer_i32(ids_table, safe_src) + local_route,
            mask=live,
            other=-1,
        )
        owner = expert // EPR
        local_expert = expert - owner * EPR
        owned = live & (expert >= 0) & (expert < NEXP) & (owner == MY_PE)
        safe_expert = tl.where(owned, local_expert, 0)
        slot = tl.atomic_add(
            expert_write + safe_expert,
            1,
            mask=owned,
            sem="relaxed",
            scope="gpu",
        )
        safe_slot = tl.where(owned, slot, 0)
        destination = tl.load(expert_offsets + safe_expert) + safe_slot
        tl.store(route_queue + destination, gid, mask=owned)
        cached_route_weight = tl.load(
            _peer_f32(route_weights_table, safe_src) + local_route,
            mask=owned,
            other=0.0,
        )
        tl.store(
            queue_route_weight + destination,
            cached_route_weight,
            mask=owned,
        )
        block_start = owned & ((safe_slot % BLOCK_M) == 0)
        block_id = tl.load(expert_block_offsets + safe_expert) + safe_slot // BLOCK_M
        tl.store(
            block_to_expert + block_id,
            safe_expert,
            mask=block_start,
        )
        route0 += route_stride

    _local_epoch_barrier(
        phase_done,
        phase_released,
        sync_status,
        epoch,
        4,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )

    # Phase 2: L1 (expert M block, I tile), with per-tile activation amax.
    total_blocks = tl.load(expert_block_offsets + EPR)
    l1_task = pid
    l1_tasks = total_blocks * NI
    while l1_task < l1_tasks:
        block_id = l1_task // NI
        nblock = l1_task - block_id * NI
        expert = tl.load(block_to_expert + block_id)
        expert_block = block_id - tl.load(expert_block_offsets + expert)
        first_row = tl.load(expert_offsets + expert) + expert_block * BLOCK_M
        expert_end = tl.load(expert_offsets + expert + 1)
        rows = first_row + mi
        row_live = rows < expert_end
        safe_rows = tl.where(row_live, rows, 0)
        gids = tl.load(route_queue + safe_rows, mask=row_live, other=0)
        src = gids // routes_per_rank
        local_route = gids - src * routes_per_rank
        source_token = local_route // TOPK
        gathered_token = src * NTOK + source_token
        input_scale = tl.load(
            gathered_qscale + gathered_token,
            mask=row_live,
            other=0.0,
        )
        ncols = nblock * BLOCK_N + ni
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in tl.range(0, HIDDEN, BLOCK_K):
            aptrs = gathered_q + gathered_token[:, None] * HIDDEN + k0 + ki[None, :]
            atile = tl.load(aptrs, mask=row_live[:, None], other=0)
            gate_ptrs = (
                w1_gate_t
                + (expert * HIDDEN + k0 + ki[:, None]) * INTER
                + ncols[None, :]
            )
            up_ptrs = (
                w1_up_t + (expert * HIDDEN + k0 + ki[:, None]) * INTER + ncols[None, :]
            )
            gate_acc = tl.dot(
                atile, tl.load(gate_ptrs), acc=gate_acc, out_dtype=tl.int32
            )
            up_acc = tl.dot(atile, tl.load(up_ptrs), acc=up_acc, out_dtype=tl.int32)
        gate_scale = tl.load(w1_gate_scale + expert * INTER + ncols)
        up_scale = tl.load(w1_up_scale + expert * INTER + ncols)
        gate_bf16 = (
            gate_acc.to(tl.float32) * input_scale[:, None] * gate_scale[None, :]
        ).to(tl.bfloat16)
        up_bf16 = (up_acc.to(tl.float32) * input_scale[:, None] * up_scale[None, :]).to(
            tl.bfloat16
        )
        gate = gate_bf16.to(tl.float32)
        silu_bf16 = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16)
        activated = (silu_bf16.to(tl.float32) * up_bf16.to(tl.float32)).to(tl.bfloat16)
        tl.store(
            activation_tmp + rows[:, None] * INTER + ncols[None, :],
            activated,
            mask=row_live[:, None],
        )
        tile_amax = tl.max(tl.abs(activated.to(tl.float32)), axis=1)
        tl.store(
            activation_amax + rows * NI + nblock,
            tile_amax,
            mask=row_live,
        )
        l1_task += NCTA

    _local_epoch_barrier(
        phase_done,
        phase_released,
        sync_status,
        epoch,
        5,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )

    # Phase 3: qA2 scale reduction and quantization.
    q2_task = pid
    amax_cols = tl.arange(0, NI_POW2)
    while q2_task < total_blocks:
        expert = tl.load(block_to_expert + q2_task)
        expert_block = q2_task - tl.load(expert_block_offsets + expert)
        first_row = tl.load(expert_offsets + expert) + expert_block * BLOCK_M
        expert_end = tl.load(expert_offsets + expert + 1)
        rows = first_row + mi
        row_live = rows < expert_end
        partial_max = tl.load(
            activation_amax + rows[:, None] * NI + amax_cols[None, :],
            mask=row_live[:, None] & (amax_cols[None, :] < NI),
            other=0.0,
        )
        full_amax = tl.max(partial_max, axis=1)
        scale = tl.where(full_amax > 0.0, full_amax / 127.0, 1.0)
        tl.store(q2_scale + rows, scale, mask=row_live)
        for nblock in tl.range(0, NI):
            ncols = nblock * BLOCK_N + ni
            activated = tl.load(
                activation_tmp + rows[:, None] * INTER + ncols[None, :],
                mask=row_live[:, None],
                other=0.0,
            ).to(tl.float32)
            quantized = _quantize_symmetric_int8(activated, scale[:, None])
            tl.store(
                q2 + rows[:, None] * INTER + ncols[None, :],
                quantized,
                mask=row_live[:, None],
            )
        q2_task += NCTA

    _local_epoch_barrier(
        phase_done,
        phase_released,
        sync_status,
        epoch,
        6,
        NCTA,
        NCTA_POW2,
        SYNC_MAX_SPINS,
    )

    # Phase 4: L2 and direct weighted peer scatter.
    l2_task = pid
    l2_tasks = total_blocks * NH
    while l2_task < l2_tasks:
        block_id = l2_task // NH
        nblock = l2_task - block_id * NH
        expert = tl.load(block_to_expert + block_id)
        expert_block = block_id - tl.load(expert_block_offsets + expert)
        first_row = tl.load(expert_offsets + expert) + expert_block * BLOCK_M
        expert_end = tl.load(expert_offsets + expert + 1)
        rows = first_row + mi
        row_live = rows < expert_end
        safe_rows = tl.where(row_live, rows, 0)
        gids = tl.load(route_queue + safe_rows)
        src = gids // routes_per_rank
        local_route = gids - src * routes_per_rank
        ncols = nblock * BLOCK_N + ni
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in tl.range(0, INTER, BLOCK_K):
            aptrs = q2 + rows[:, None] * INTER + k0 + ki[None, :]
            atile = tl.load(aptrs, mask=row_live[:, None], other=0)
            bptrs = w2_t + (expert * INTER + k0 + ki[:, None]) * HIDDEN + ncols[None, :]
            acc = tl.dot(
                atile,
                tl.load(bptrs),
                acc=acc,
                out_dtype=tl.int32,
            )
        a_scale = tl.load(q2_scale + safe_rows)
        weight_scale = tl.load(w2_scale + expert * HIDDEN + ncols)
        route_weight = tl.load(
            queue_route_weight + safe_rows,
            mask=row_live,
            other=0.0,
        )
        partial = (
            acc.to(tl.float32)
            * a_scale[:, None]
            * weight_scale[None, :]
            * route_weight[:, None]
        ).to(tl.bfloat16)
        if AGGREGATE_RETURN:
            source_token = local_route // TOPK
            global_token = src * NTOK + source_token
            tl.atomic_add(
                owner_accum + global_token[:, None] * HIDDEN + ncols[None, :],
                partial.to(tl.float32),
                mask=row_live[:, None],
                sem="relaxed",
                scope="gpu",
            )
        else:
            destination = _peer_bf16(partials_table, src)
            tl.store(
                destination[:, None] + local_route[:, None] * HIDDEN + ncols[None, :],
                partial,
                mask=row_live[:, None],
            )
        l2_task += NCTA

    if AGGREGATE_RETURN:
        _local_epoch_barrier(
            phase_done,
            phase_released,
            sync_status,
            epoch,
            7,
            NCTA,
            NCTA_POW2,
            SYNC_MAX_SPINS,
        )
        return_task = pid
        return_tasks = NPES * NTOK * NH
        while return_task < return_tasks:
            global_token = return_task // NH
            nblock = return_task - global_token * NH
            src = global_token // NTOK
            source_token = global_token - src * NTOK
            ncols = nblock * BLOCK_N + ni
            contribution = tl.load(owner_accum + global_token * HIDDEN + ncols).to(
                tl.bfloat16
            )
            destination = _peer_bf16(partials_table, src)
            tl.store(
                destination + (source_token * TOPK + MY_PE) * HIDDEN + ncols,
                contribution,
            )
            return_task += NCTA
        _rank_epoch_barrier(
            flag_table,
            phase_done,
            phase_released,
            rank_released,
            sync_status,
            epoch,
            8,
            1,
            MY_PE,
            NPES,
            NPES_POW2,
            NCTA,
            NCTA_POW2,
            SYNC_MAX_SPINS,
        )
    else:
        _rank_epoch_barrier(
            flag_table,
            phase_done,
            phase_released,
            rank_released,
            sync_status,
            epoch,
            7,
            1,
            MY_PE,
            NPES,
            NPES_POW2,
            NCTA,
            NCTA_POW2,
            SYNC_MAX_SPINS,
        )

    # Phase 5: source-local fixed-slot FP32 combine and BF16 output.
    out_lane = tl.arange(0, 4)
    out_block0 = pid * 4
    output_blocks = NTOK * NH
    while out_block0 < output_blocks:
        out_blocks = out_block0 + out_lane
        out_live = out_blocks < output_blocks
        tokens = out_blocks // NH
        nblocks = out_blocks - tokens * NH
        ncols = nblocks[:, None] * BLOCK_N + ni[None, :]
        combined = tl.zeros((4, BLOCK_N), dtype=tl.float32)
        if AGGREGATE_RETURN:
            for owner in tl.static_range(0, NPES):
                partial = tl.load(
                    partials_local + (tokens[:, None] * TOPK + owner) * HIDDEN + ncols,
                    mask=out_live[:, None],
                    other=0.0,
                )
                combined += partial.to(tl.float32)
        else:
            for slot in tl.static_range(0, TOPK):
                expert = tl.load(
                    ids_local + tokens * TOPK + slot,
                    mask=out_live,
                    other=-1,
                )
                partial = tl.load(
                    partials_local + (tokens[:, None] * TOPK + slot) * HIDDEN + ncols,
                    mask=(
                        out_live[:, None]
                        & (expert[:, None] >= 0)
                        & (expert[:, None] < NEXP)
                    ),
                    other=0.0,
                )
                combined += partial.to(tl.float32)
        tl.store(
            final_output + tokens[:, None] * HIDDEN + ncols,
            combined.to(tl.bfloat16),
            mask=out_live[:, None],
        )
        out_block0 += NCTA * 4


KERNEL_LAUNCHES_PER_FORWARD = 1

__all__ = ["KERNEL_LAUNCHES_PER_FORWARD", "hcu_megamoe"]
