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

"""Ascend-specialized top_k_per_row_prefill (DeepSeek V4 sparse attention).

Routes (per-row, resolved inside the threshold kernel from row_len; the
host launches one pipeline that serves all of them):

  trivial     row_len <= TOPK.  Every element wins; the row is copied out
              directly, padded with -1.  No sampling, no packing.
  select_all  row_len <= CAP.   All elements fit the candidate buffer, so
              the threshold degenerates to -inf and K2-K5 run on an
              all-ones mask.  No sampling.
  sampled     row_len > CAP.    K1 estimates a threshold from a sorted
              sample (plus a +-6-sigma bracket pair for fixup), K2-K5
              count/compact/sort the survivors.
  rowsort     launch-level route for narrow rows (k == 512,
              vocab <= 6144, any batch size): one exact-width hardware
              sort per row (a subview of the pow2 load buffer keeps
              ragged tails out of the sort), unpacked in-kernel — no
              proposal round-trip, no merge.  Beats the sampled pipeline
              2-7x on narrow rows (measured); the last two rows read from
              a -inf-padded scratch (masked lanes still issue addresses).
  segsort     launch-level route for medium rows (k == 512,
              6144 < vocab <= 32768) in small batches (rows <= 64):
              fixed-width (2048/4096) segment sorts + a two-level 4-way
              exhaustion merge.  2 launches beat the 7-launch sampled
              pipeline while the batch is latency-bound (probe).
  fixup       sampled rows whose survivor count at the threshold lands
              outside [TOPK, CAP] (degenerate distributions).  Exact
              tie-fill / bit-descent reselection in one kernel; rare.
  chunked     launch-level route for top_k > CHUNK_K: ceil(top_k/CHUNK_K)
              consecutive rounds of the threshold pipeline, each emitting
              up to CHUNK_K indices at a column offset.  A lexicographic
              (value, position) bound recorded by a boundary kernel
              excludes everything earlier rounds emitted, so ties can
              neither duplicate nor lose elements across rounds.

Design notes (all numbers measured on 910B-4, CANN 9.0, triton 3.5.1):

- Global-memory histogram atomics and shared-memory scratch (generic impl)
  are pathologically slow (~110ms/row).
- Backend cost model measured with micro-benchmarks and msprof
  PipeUtilization: this backend issues per-lane address arithmetic and
  per-lane load/store on the SCALAR pipe, and lowers tl.sum/tl.cumsum to
  scalar add chains. Dense affine load/store are full-width vector ops
  (memory-bound, nearly free); fp32 compares are free. Therefore:
  * all hot-path reductions use split+add trees (reshape (G,W)->(G,W/2,2)
    -> tl.split -> add, log2(W) levels) which run on the vector pipe —
    measured 184us -> 7us scalar on the pack counting workload;
  * trees only lower correctly from dense (affine) loads and within a
    single control-flow scope — the partial tail tile lives in a separate
    kernel (_ascend_topk_pack_tail_kernel);
  * no tl.cumsum in hot passes, no scatter stores in hot passes (they
    scalarize per lane).
- fp32->fp16 conversion lowers to slow scalar code; unsigned shifts must
  be emulated with mask + arithmetic shift (FlagTree issue #1121).
- The backend compiler loses store->load dependencies inside a mega-kernel
  (program 0 at grid>=64 read back its own stores), so intermediate
  results cross kernel boundaries; counts and exclusive bases live in
  separate buffers and the segment base table is derived from registers.
- One dense pass over 32 rows x 129280 fp32 takes ~55us (300GB/s
  aggregate), vs ~265us for torch.topk. The pipeline touches the full row
  once (pack); extraction works on the packed masks only:

  K1 threshold  Per row, DMA-copy 4 evenly spaced 1024-tiles into UB and
               hardware-sort them (sort_1d_pack raw op); the exact sample
               quantile at rank ~2x TOPK becomes the mask threshold, with
               a +-6-sigma bracket pair stored alongside for the fixup
               path's reselection.  Short rows (row_len <= TOPK) are
               copied out directly, padded with -1.
  K2 pack      One full-row pass (rows x C blocks): order-preserving
               compares against the threshold (exactly representable
               floats), bit-pack one mask per 64-element subtile and
               store per-subtile counts (all via split+add trees).
               The partial tail tile is packed by separate tiny kernels
               (K2t dense window + K2s partial subtile).
  K3 reduce    Per-row sum -> L, the route decision (0: L in
               [TOPK, CAP], 2: fixup) and the exclusive 4096-segment
               bases of the mask.
  K4 compact   grid (rows x NSEG): per 4096-segment, two vreducev2 calls
               (gather_mask_custom_pattern custom op) compact candidate
               values and positions against the padded mask words (no
               scalar GM access), emitting a compact (value, index)
               candidate list.  The mask words are stored in the padded
               vreducev2 layout: the 2 words of each 64-element subtile
               head a 32-byte block (8 words); the hardware reads only
               those 64 bits per repeat (probe-verified with poisoned
               pads).
  K5 sort      grid (rows): one sort_1d_pack over the <=CAP candidates
               (hardware vbitsort/vmrgsort4 via the TLE custom ops) +
               unpack + tl.gather (hivm.vgather) index remap writes
               exactly TOPK indices.  Skips fixup rows.
  K6 fixup     Route-2 rows only (launched when the device-side flag
               fires; the reduce kernel ORs per-row flags via
               atomic_max).  The bracket threshold is picked by
               direction: L < TOPK -> thr_lo, L > CAP -> thr_hi.
               Typical case (massive tie at one value): strict/tie fill,
               3 read passes.  Otherwise a 32-pass descent over the
               unsigned rank of the order-preserving int32 fp32-bit
               mapping finds the exact top-k boundary value.

All custom ops (sort_1d_pack / unpack_sort / gather_mask_custom_pattern)
are registered by FlagTree's triton.experimental.tle.language.dsa.ascend.
custom_ops package and called via tle.dsa.ascend.raw; TLE is mandatory
(the host entry raises NotImplementedError without it).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables (measured on 910B-4, CANN 9.0). Each constant names one hardware
# or compiler constraint; the module docstring lists the pipeline stages
# that consume them.
# ---------------------------------------------------------------------------
CAP = 4096  # candidate capacity; sort_1d_pack UB budget ~20B/elem -> 80KB/192KB
SCAN_BLOCK = 1024  # dense load/store tile width for the counting passes
PACK_BLOCK = 2048  # wide tile for the pack kernel (fewer vector instructions)
SUB = 64  # bit-pack subtile: one mask word pair per SUB elements
MASK_BLK = tl.constexpr(8)  # padded mask layout for vreducev2
# (gather_mask_custom_pattern): the subtile's word pair heads a 32B block
# (8 int32 words); the hardware reads only the first 64 bits of each block
# (probe: poisoned pads never affect the result), so the pad words stay
# uninitialized.
SUBTILES_PER_BLOCK = SCAN_BLOCK // SUB  # mask word pairs per scan block
COMPACT_SEG = 4096  # mask segment width consumed by one compact program
DENSIFY_BLOCK = 1024  # copy tile for the stride1 != 1 densify pre-route
SAMPLE_TILES = 4  # sampler: DMA tiles of SCAN_BLOCK elements per row
SAMPLE_SORT_OUT = 512  # sampler sort output length: >= the +6-sigma
# bracket rank used by the fixup path (thr_lo's clamp ceiling)
SORT_TMP_MUL = 4  # sort_1d_pack workspace: 4 floats per sorted element
TB_MIN = 16  # SB-geometry floor: keeps the mask/count geometry in the
# range the pipeline is validated against
NEG_INF_BITS = tl.constexpr(-8388608)  # 0xFF800000: float("-inf") as int32

# Row-sort route (narrow rows): one exact-width hardware sort per row with
# in-kernel unpack — no proposal round-trip, no merge kernel.
ROWSORT_MAX_VOCAB = 6144  # UB budget: pow2 load buf + 4x sort workspace
# (measured: 6144 compiles at ~172KB UB; 7000/8192 overflow the 192KB UB)
ROWSORT_K = 512  # DeepSeek V4 sparse-attention k this route serves

# Seg-sort route (medium rows, small batches): fixed-width segment sorts
# (2048-wide up to 8 segments; 4096-wide beyond that) + a two-level 4-way
# exhaustion merge.  Latency-bound batches beat the 7-launch threshold
# pipeline here (2 launches); large batches stay on the threshold
# pipeline, which has better per-row throughput at scale.
SEGSORT_SEG = 2048  # segment sort width (sort_1d_pack sweet spot)
SEGSORT_MAX_VOCAB = 32768  # 8 segments of 4096; merge caps at 2 levels
SEGSORT_MAX_ROWS = 64  # latency-bound batches only; throughput crossover
# vs the threshold pipeline sits around 32-64 rows (probe)

# Chunked route (top_k > CHUNK_K): per-round extraction cap.  One round
# reliably emits at most CHUNK_K indices: the sampler targets 2x TOPK
# capped at 3/4 CAP, and the [TOPK, CAP] route window needs ~4 sigma of
# headroom on both sides.
CHUNK_K = 2048


# ---------------------------------------------------------------------------
# Raw custom-op path (CANN >= 9.1, TLE required).
# ---------------------------------------------------------------------------
try:
    import triton.experimental.tle as tle

    # Registers sort_1d_pack / unpack_sort / gather_mask_custom_pattern
    # into tle.dsa.ascend.raw.
    import triton.experimental.tle.language.dsa.ascend.custom_ops  # noqa: F401

    _TLE_IMPORT_OK = True
except Exception:  # noqa: BLE001
    tle = None
    _TLE_IMPORT_OK = False

_SORT_IMPL_BASE = tl.constexpr(0)  # vbitsort + vmrgsort4 path, any N/K

_neginf_tile = None


def _get_neginf_tile(device):
    # Constant -inf tile used to pad the sampler buffer beyond n_samp tiles.
    # One-time torch.full is a host-side constant fill, not a compute op in
    # the op's data path.
    global _neginf_tile
    if _neginf_tile is None:
        _neginf_tile = torch.full(
            (SCAN_BLOCK,), float("-inf"), device=device, dtype=torch.float32
        )
    return _neginf_tile


@triton.jit
def _nan_as_inf(x):
    # NaN canonicalization at numeric load points.  torch.topk ranks NaN
    # above every finite value and +inf, but compares like `x >= thr`
    # silently drop NaN (probe: 10 NaNs in a row, 0 selected).  The
    # pipeline emits INDICES — values are read back from the original
    # logits — so mapping NaN -> +inf preserves the selected value
    # multiset.
    return tl.where(x != x, float("inf"), x)


@triton.jit
def _pad_mask_blk(w2):
    """(G, 2) mask words -> (G, 8) padded vreducev2 layout (2 words + 6 zeros).

    tl.join would INTERLEAVE the pair with the zeros (minor-dim stacking);
    split + broadcast-select keeps [w0, w1, 0,0, 0,0,0,0] per row.  Writing
    the pads as explicit zeros keeps the GM store DENSE (a strided
    2-of-8-words store costs 3x the pack kernel — msprof 100us -> 286us).
    """
    w0, w1 = tl.split(w2)  # (G,) each
    j = tl.arange(0, 8)[None, :]
    return w0[:, None] * (j == 0).to(tl.int32) + w1[:, None] * (j == 1).to(tl.int32)


@triton.jit
def _tree_sum_lastdim(v, W: tl.constexpr, LOGW: tl.constexpr):
    """(G, W) int32 -> (G,) row sums via LOGW levels of reshape+split+add.

    tl.sum lowers to a scalar add chain on this backend (measured on the
    pack count workload: 184us on the scalar pipe vs 7us for this tree);
    split+add runs as full-width vector adds.
    """
    for i in tl.static_range(0, LOGW):
        v = tl.reshape(v, (v.shape[0], W >> (i + 1), 2))
        a, b = tl.split(v)
        v = a + b
    return tl.reshape(v, (v.shape[0],))


@triton.jit
def _tree_sum_all(v, W: tl.constexpr, LOGW: tl.constexpr):
    """(W,) int32 -> scalar: full-width tree reduction (see _tree_sum_lastdim)."""
    for i in tl.static_range(0, LOGW):
        v = tl.reshape(v, (W >> (i + 1), 2))
        a, b = tl.split(v)
        v = a + b
    return tl.sum(v, axis=0)  # (1,) -> scalar, single lane


@triton.jit
def _tree_pack32(v):
    """(G, 32) int32 0/1 -> (G,) bit-packed word (bit j = element j).

    Pairwise combine with a shift that doubles per level: a + (b << w).
    """
    v = tl.reshape(v, (v.shape[0], 16, 2))
    a, b = tl.split(v)
    v = a + (b << 1)
    v = tl.reshape(v, (v.shape[0], 8, 2))
    a, b = tl.split(v)
    v = a + (b << 2)
    v = tl.reshape(v, (v.shape[0], 4, 2))
    a, b = tl.split(v)
    v = a + (b << 4)
    v = tl.reshape(v, (v.shape[0], 2, 2))
    a, b = tl.split(v)
    v = a + (b << 8)
    v = tl.reshape(v, (v.shape[0], 1, 2))
    a, b = tl.split(v)
    v = a + (b << 16)
    return tl.reshape(v, (v.shape[0],))


@triton.jit
def _ascend_topk_pack_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    stride0,
    mask_ptr,  # [num_rows, SB, MASK_BLK] int32 (padded vreducev2 layout)
    cnt_ptr,  # [num_rows, SB] int32
    totals_ptr,  # [num_rows, 8]
    TOPK: tl.constexpr,
    PB: tl.constexpr,  # wide tile size (2048): fewer vector instructions
    SB: tl.constexpr,
    SUB: tl.constexpr,
    C: tl.constexpr,
    hi_ptr,  # [num_rows, 4] int32 chunk state (dummy when HI=0)
    REPACK: tl.constexpr = 0,  # 1: second pass, re-routed (route==3) rows only
    HI: tl.constexpr = 0,  # 1: chunked round, apply the lexicographic bound
):
    pid = tl.program_id(0)
    row_id = pid // C
    cid = pid % C
    if REPACK:
        skip = tl.load(totals_ptr + row_id * 8 + 3) != 3
    else:
        skip = tl.load(totals_ptr + row_id * 8) < 0  # short row
    if skip:
        return
    # Threshold as an exact float: selection is one fp32 compare.
    thr = tl.load(totals_ptr + row_id * 8 + 4).to(tl.float32, bitcast=True)
    if HI:
        # Chunked round: exclude everything earlier rounds emitted.  The
        # bound is a total order ((value desc, position asc); positions are
        # unique), so no element can be re-emitted or lost across rounds.
        hi_v = tl.load(hi_ptr + row_id * 4 + 0).to(tl.float32, bitcast=True)
        hi_i = tl.load(hi_ptr + row_id * 4 + 1)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    base_ptr = logits_ptr + row_id * stride0 + row_start
    mask_ptr += row_id * SB * MASK_BLK
    cnt_ptr += row_id * SB

    lane = tl.arange(0, PB)
    NSUB: tl.constexpr = PB // SUB
    t32 = tl.arange(0, NSUB)
    n_pt = tl.cdiv(row_len, PB)
    n_full = row_len // PB
    tpc = tl.cdiv(n_pt, C)
    t0 = cid * tpc
    t1 = tl.minimum(t0 + tpc, n_pt)
    # Full tiles: unmasked dense loads (clamping defeats vectorization).
    # The clamped tail stays outside the loop: two load branches in the
    # loop body double the software-pipeline buffers and overflow UB.
    # Row totals are derived from the counts by the reduce kernel (a full
    # 2048-lane reduction per tile here would cost ~30% of the kernel).
    # Bit-packing and per-subtile counts use split+add trees: tl.sum lowers
    # to a scalar add chain here, the tree runs on the vector pipe (~4.6x
    # faster on this workload). The partial tail tile is handled by a
    # separate kernel: this backend's storage-align pass cannot propagate
    # the tree's reshape/split chains through two control-flow scopes.
    t1_full = tl.minimum(t1, n_full)
    for t in tl.range(t0, t1_full):
        offs = t * PB + lane
        x = tl.load(base_ptr + offs)
        sub_id = t * NSUB + t32
        if HI:
            # Scalar pack form (the proven tail-kernel pattern): the
            # lexicographic filter's extra op chain between the dense
            # load and the split+add tree breaks the tree's
            # storage-align lowering (cannot align 1 axis).  HI is the
            # cold chunked path, so scalar packing is acceptable.
            # NaN is canonicalized to +inf so the lexicographic bound
            # stays a total order (raw NaN fails both sides of the
            # filter and would leak out of the round bookkeeping).
            xc = _nan_as_inf(x)
            sel_f = (xc >= thr).to(tl.int32) * (
                (xc < hi_v) | ((xc == hi_v) & (offs > hi_i))
            ).to(tl.int32)
            POW2 = (1 << tl.arange(0, 32)).to(tl.int32)
            pack = tl.sum(
                tl.reshape(sel_f, (NSUB, 2, 32)) * POW2[None, None, :], axis=2
            )
            tl.store(
                mask_ptr
                + (sub_id * MASK_BLK)[:, None]
                + tl.arange(0, MASK_BLK)[None, :],
                _pad_mask_blk(pack),
            )
            cnt = tl.sum(tl.reshape(sel_f, (NSUB, 64)), axis=1)
            tl.store(cnt_ptr + sub_id, cnt)
        else:
            # NaN-as-largest with zero extra ops on the load->tree chain:
            # pack the COMPLEMENTED compare (NaN fails x < thr, so it is
            # selected) and invert the packed words / counts instead.
            # Full tiles only, so every packed bit is a real lane and the
            # inversion is exact.  A where(x!=x) select on the load would
            # break the tree lowering here (same failure as the HI form).
            lt_i32 = (x < thr).to(tl.int32)
            pack = ~_tree_pack32(tl.reshape(lt_i32, (NSUB * 2, 32)))
            tl.store(
                mask_ptr
                + (sub_id * MASK_BLK)[:, None]
                + tl.arange(0, MASK_BLK)[None, :],
                _pad_mask_blk(tl.reshape(pack, (NSUB, 2))),
            )
            cnt = 64 - _tree_sum_lastdim(tl.reshape(lt_i32, (NSUB, 64)), 64, 6)
            tl.store(cnt_ptr + sub_id, cnt)


@triton.jit
def _ascend_topk_pack_tail_dense_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    stride0,
    mask_ptr,  # [num_rows, SB, MASK_BLK] int32 (padded vreducev2 layout)
    cnt_ptr,  # [num_rows, SB] int32
    totals_ptr,  # [num_rows, 8]
    PB: tl.constexpr,
    SB: tl.constexpr,
    SUB: tl.constexpr,
    hi_ptr,  # [num_rows, 4] int32 chunk state (dummy when HI=0)
    REPACK: tl.constexpr = 0,
    HI: tl.constexpr = 0,
):
    # Tail subtiles [n_full*PB, A) via a SHIFTED DENSE WINDOW [A-PB, A),
    # where A = row_len rounded down to SUB: every lane is a real element,
    # so the split+add tree lowering works (masked/clamped tails defeat it).
    # The kernel is BRANCHLESS on
    # purpose: any extra control-flow scope in the same kernel poisons the
    # tree's storage-align lowering (measured 226us/call when the clamp and
    # the scalar tail shared the kernel).  Requires stride0 >= PB (host
    # guarantees it): then the window always stays inside the row's stride,
    # even for the last row.  Overlap with the last full tile rewrites
    # identical values (idempotent); store masks restrict writes to the
    # subtiles the pack kernel did not cover.  The final partial subtile
    # [A, row_len) is handled by _ascend_topk_pack_tail_sub_kernel.
    row_id = tl.program_id(0)
    if REPACK:
        skip = tl.load(totals_ptr + row_id * 8 + 3) != 3
    else:
        skip = tl.load(totals_ptr + row_id * 8) < 0  # short row
    if skip:
        return
    thr = tl.load(totals_ptr + row_id * 8 + 4).to(tl.float32, bitcast=True)
    if HI:
        hi_v = tl.load(hi_ptr + row_id * 4 + 0).to(tl.float32, bitcast=True)
        hi_i = tl.load(hi_ptr + row_id * 4 + 1)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    n_full = row_len // PB
    if n_full * PB >= row_len:
        return  # no tail
    base_ptr = logits_ptr + row_id * stride0 + row_start
    mask_ptr += row_id * SB * MASK_BLK
    cnt_ptr += row_id * SB
    NSUB: tl.constexpr = PB // SUB
    A = row_len & ~(SUB - 1)
    first_sub = n_full * NSUB  # first subtile not covered by the pack kernel
    last_dense = A // SUB
    wsub = tl.maximum(A - PB, 0) // SUB  # window start, in subtiles
    lane = tl.arange(0, PB)
    x = tl.load(base_ptr + wsub * SUB + lane)
    offs = wsub * SUB + lane
    t32 = tl.arange(0, NSUB)
    keep = (t32 >= first_sub - wsub) & (t32 < last_dense - wsub)
    if HI:
        # Scalar pack form (see the pack kernel): the lexicographic
        # filter's op chain breaks the tree's storage-align lowering.
        # NaN is canonicalized to +inf (total-order bound, see there).
        xc = _nan_as_inf(x)
        sel_f = (xc >= thr).to(tl.int32) * (
            (xc < hi_v) | ((xc == hi_v) & (offs > hi_i))
        ).to(tl.int32)
        POW2 = (1 << tl.arange(0, 32)).to(tl.int32)
        pack = tl.sum(tl.reshape(sel_f, (NSUB, 2, 32)) * POW2[None, None, :], axis=2)
        tl.store(
            mask_ptr
            + ((wsub + t32) * MASK_BLK)[:, None]
            + tl.arange(0, MASK_BLK)[None, :],
            _pad_mask_blk(pack),
            mask=keep[:, None],
        )
        cnt = tl.sum(tl.reshape(sel_f, (NSUB, 64)), axis=1)
        tl.store(cnt_ptr + wsub + t32, cnt, mask=keep)
    else:
        # Complemented-compare form (see the pack kernel): NaN-as-largest,
        # zero extra ops on the load->tree chain, exact since every lane
        # of the dense window is a real element.
        lt_i32 = (x < thr).to(tl.int32)
        pack = ~_tree_pack32(tl.reshape(lt_i32, (NSUB * 2, 32)))
        tl.store(
            mask_ptr
            + ((wsub + t32) * MASK_BLK)[:, None]
            + tl.arange(0, MASK_BLK)[None, :],
            _pad_mask_blk(tl.reshape(pack, (NSUB, 2))),
            mask=keep[:, None],
        )
        cnt = 64 - _tree_sum_lastdim(tl.reshape(lt_i32, (NSUB, 64)), 64, 6)
        tl.store(cnt_ptr + wsub + t32, cnt, mask=keep)


@triton.jit
def _ascend_topk_pack_tail_sub_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    stride0,
    mask_ptr,  # [num_rows, SB, MASK_BLK] int32 (padded vreducev2 layout)
    cnt_ptr,  # [num_rows, SB] int32
    totals_ptr,  # [num_rows, 8]
    PB: tl.constexpr,
    SB: tl.constexpr,
    SUB: tl.constexpr,
    hi_ptr,  # [num_rows, 4] int32 chunk state (dummy when HI=0)
    REPACK: tl.constexpr = 0,
    HI: tl.constexpr = 0,
):
    # Final partial subtile [A, row_len), A = row_len & ~(SUB-1): at most
    # SUB-1 elements, scalar form.  Kept in its own kernel: sharing one
    # with the tree-based dense kernel breaks the tree lowering.
    row_id = tl.program_id(0)
    if REPACK:
        skip = tl.load(totals_ptr + row_id * 8 + 3) != 3
    else:
        skip = tl.load(totals_ptr + row_id * 8) < 0  # short row
    if skip:
        return
    thr = tl.load(totals_ptr + row_id * 8 + 4).to(tl.float32, bitcast=True)
    if HI:
        hi_v = tl.load(hi_ptr + row_id * 4 + 0).to(tl.float32, bitcast=True)
        hi_i = tl.load(hi_ptr + row_id * 4 + 1)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    A = row_len & ~(SUB - 1)
    if A >= row_len:
        return  # no partial subtile
    base_ptr = logits_ptr + row_id * stride0 + row_start
    # Compute on 128 explicit lanes even though the subtile is SUB=64:
    # reductions (the pack/cnt tl.sums) on a 64-wide value read the
    # hardware vector padding beyond lane 63, which holds leftover x /
    # compare bits from earlier in the kernel (probe: lanes 56..63 of w1
    # set when thr <= 0 and hi_v >= 0 -> 8 OOB candidates per corrupted
    # row).  Widening the compute domain to the 128-lane vector width
    # turns that padding into explicit in_range zeros.  A direct 64-lane
    # store of sel_i32 is unaffected, which is why the bug only shows in
    # the packed mask words and counts.
    lane = tl.arange(0, 128)
    offs = A + lane
    in_range = (offs < row_len) & (lane < SUB)
    x = tl.load(base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0)
    # Complemented compare gives NaN-as-largest without a select on the
    # load: a where(x!=x) canonicalization here miscompiles this kernel's
    # scalar pack (probe: corrupted mask word -> OOB indices downstream).
    sel_i32 = (~(x < thr)).to(tl.int32) * in_range.to(tl.int32)
    if HI:
        # The lexicographic bound needs the canonical value: raw NaN
        # fails both sides of the filter and leaks out of bookkeeping.
        xc = _nan_as_inf(x)
        sel_i32 = sel_i32 * ((xc < hi_v) | ((xc == hi_v) & (offs > hi_i))).to(tl.int32)
    j32 = tl.arange(0, 32)
    POW2 = (1 << j32).to(tl.int32)
    pack = tl.sum(tl.reshape(sel_i32, (128 // 32, 32)) * POW2[None, :], axis=1)
    sub_id = A // SUB
    # 2-word strided store, pads left uninitialized: routing the scalar pack
    # through the (G,8) pad chain in this kernel's control flow miscompiles
    # (w1 high byte corrupted; probe: constexpr VALID + pad -> 0x80000000).
    # vreducev2 reads only the head 2 words of each 8-word block (verified by
    # pad-poisoning), and this is one store per row so store density is moot.
    # Words 2..3 of the 128-lane pack are zero by construction (lanes >= SUB
    # are out of range); only words 0..1 are stored.
    tl.store(
        mask_ptr + row_id * SB * MASK_BLK + sub_id * MASK_BLK + tl.arange(0, 4),
        pack,
        mask=tl.arange(0, 4) < 2,
    )
    cnt = tl.sum(sel_i32, axis=0)
    tl.store(cnt_ptr + row_id * SB + sub_id, cnt)


@triton.jit
def _ascend_topk_pack_tail_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    stride0,
    mask_ptr,  # [num_rows, SB, MASK_BLK] int32 (padded vreducev2 layout)
    cnt_ptr,  # [num_rows, SB] int32
    totals_ptr,  # [num_rows, 8]
    PB: tl.constexpr,
    SB: tl.constexpr,
    SUB: tl.constexpr,
    hi_ptr,  # [num_rows, 4] int32 chunk state (dummy when HI=0)
    REPACK: tl.constexpr = 0,
    HI: tl.constexpr = 0,
):
    # Whole-tail scalar form, used only when stride0 < PB (tiny vocabs),
    # where the dense window cannot be guaranteed in-bounds.  One partial
    # tile per row (elements [n_full*PB, row_len)), at most PB-1 elements.
    # Kept scalar: masked/clamped tail addressing defeats the tree's
    # storage-align lowering (one tile per row).
    row_id = tl.program_id(0)
    if REPACK:
        skip = tl.load(totals_ptr + row_id * 8 + 3) != 3
    else:
        skip = tl.load(totals_ptr + row_id * 8) < 0  # short row
    if skip:
        return
    thr = tl.load(totals_ptr + row_id * 8 + 4).to(tl.float32, bitcast=True)
    if HI:
        hi_v = tl.load(hi_ptr + row_id * 4 + 0).to(tl.float32, bitcast=True)
        hi_i = tl.load(hi_ptr + row_id * 4 + 1)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    n_full = row_len // PB
    if n_full * PB >= row_len:
        return  # no tail
    base_ptr = logits_ptr + row_id * stride0 + row_start
    mask_ptr += row_id * SB * MASK_BLK
    cnt_ptr += row_id * SB
    NSUB: tl.constexpr = PB // SUB
    lane = tl.arange(0, PB)
    t32 = tl.arange(0, NSUB)
    offs = n_full * PB + lane
    in_range = offs < row_len
    x = tl.load(base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0)
    # Complemented compare for NaN-as-largest (see _ascend_topk_pack_tail_sub_kernel).
    # Arithmetic combination with in_range (same miscompile as tail_sub).
    sel_i32 = (~(x < thr)).to(tl.int32) * in_range.to(tl.int32)
    if HI:
        xc = _nan_as_inf(x)
        sel_i32 = sel_i32 * ((xc < hi_v) | ((xc == hi_v) & (offs > hi_i))).to(tl.int32)
    sub_id = n_full * NSUB + t32
    j32 = tl.arange(0, 32)
    POW2 = (1 << j32).to(tl.int32)
    pack = tl.sum(tl.reshape(sel_i32, (NSUB, 2, 32)) * POW2[None, None, :], axis=2)
    tl.store(
        mask_ptr + (sub_id * MASK_BLK)[:, None] + tl.arange(0, MASK_BLK)[None, :],
        _pad_mask_blk(pack),
    )
    cnt = tl.sum(tl.reshape(sel_i32, (NSUB, 64)), axis=1)
    tl.store(cnt_ptr + sub_id, cnt)


@triton.jit
def _ascend_topk_reduce_seg_kernel(
    row_starts,
    row_ends,
    totals_ptr,  # [num_rows, 8]
    fixup_flag_ptr,  # [1] int32: OR of all per-row fixup flags (atomic_max)
    cnt_ptr,  # [num_rows, SB] per-subtile counts (input)
    seg_base_ptr,  # [num_rows, SB // (COMPACT_SEG // SUB)] exclusive segment bases
    TOPK: tl.constexpr,
    SB: tl.constexpr,
    SUB: tl.constexpr,
    CSEG_SUB: tl.constexpr,  # subtiles per COMPACT_SEG segment
    CAP: tl.constexpr,
    LOGW_SB: tl.constexpr,
    REPACK: tl.constexpr = 0,
):
    # Per-row sums over the subtile counts -> L, the route decision (0:
    # survivor count inside [TOPK, CAP], 2: fixup) and the exclusive
    # COMPACT_SEG-segment bases.  Sums go through a split+add tree (vector
    # pipe); the exclusive scan over the NSEG segment sums is a short
    # tl.cumsum.  REPACK=1 (second pass): only re-routed (route==3) rows,
    # which KEEP route 3 when the new threshold lands in the window so the
    # compact/sort REPACK pass can still tell them apart from pass-1 rows.
    row_id = tl.program_id(0)
    totals_ptr += row_id * 8
    if REPACK:
        skip = tl.load(totals_ptr + 3) != 3
    else:
        skip = tl.load(totals_ptr + 0) < 0  # short row
    if skip:
        return
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    n_sub = tl.cdiv(row_end - row_start, SUB)
    cnt_ptr += row_id * SB
    sb_lane = tl.arange(0, SB)
    in_sb = sb_lane < n_sub
    cnts = tl.where(in_sb, tl.load(cnt_ptr + sb_lane), 0)
    L = _tree_sum_all(cnts, SB, LOGW_SB)
    if REPACK:
        route = tl.where((L >= TOPK) & (L <= CAP), 3, 2)
    else:
        route = tl.where((L >= TOPK) & (L <= CAP), 0, 2)
    tl.store(totals_ptr + 0, L)
    tl.store(totals_ptr + 3, route)
    tl.atomic_max(fixup_flag_ptr, (route == 2).to(tl.int32))
    if route != 2:
        # SB >= CSEG_SUB guaranteed whenever NSEG > 1 (vocab >
        # COMPACT_SEG); single-segment rows never read seg_base (compact
        # takes base=0, c=L).
        if SB >= CSEG_SUB:
            NSEG_SB: tl.constexpr = SB // CSEG_SUB
            seg_sums = _tree_sum_lastdim(
                tl.reshape(cnts, (NSEG_SB, CSEG_SUB)),
                CSEG_SUB,
                (CSEG_SUB - 1).bit_length(),
            )
            excl = tl.cumsum(seg_sums, axis=0) - seg_sums
            tl.store(seg_base_ptr + row_id * NSEG_SB + tl.arange(0, NSEG_SB), excl)


@triton.jit
def _ascend_topk_threshold_kernel(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    totals_ptr,  # [num_rows, 8]
    fixup_flag_ptr,  # [1] int32
    neginf_ptr,  # [SCAN_BLOCK] f32 constant tile of -inf (sampler fill)
    hi_ptr,  # [num_rows, 4] int32 chunk state (dummy when HI=0)
    out_stride,  # out_indices row stride (== TOPK unless chunked)
    out_off,  # this round's column offset into out_indices
    emit_off,  # elements earlier chunk rounds already emitted
    TOPK: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    SAMP: tl.constexpr,  # max sample tiles (4 x SCAN_BLOCK elements)
    KSAMP: tl.constexpr,  # sort output size (>= sample target count)
    TMP: tl.constexpr,  # sort workspace: SAMP * SCAN_BLOCK * 4 floats
    HI: tl.constexpr = 0,  # 1: chunked round, apply the lexicographic bound
):
    # Sort-based threshold: one hardware sort over the sample replaces the
    # 16-pass bit descent (76us -> ~10us at rows=1, and the sort yields
    # the exact sample quantile: no bucket quantization at all). The
    # sample tile count adapts to row_len so the target rank always fits
    # the sort output; rows barely above TOPK get the select-all
    # threshold (-inf), which lands L' = row_len inside [TOPK, CAP].
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(fixup_flag_ptr, 0)  # reduce kernel ORs into this
    row_id = pid
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    base_ptr = logits_ptr + row_id * stride0 + row_start
    totals_row = totals_ptr + row_id * 8
    rem = row_len - emit_off  # remaining pool size (== row_len unchunked)
    if HI:
        # Chunked round: rows the short kernel finished (row_len <= full
        # top_k) carry the done flag; mark them short so every downstream
        # kernel takes its usual short-row skip this round.
        if tl.load(hi_ptr + row_id * 4 + 2) != 0:
            tl.store(totals_row + 0, -1)
            tl.store(totals_row + 3, 0)
            return
    if row_len <= TOPK:
        # Short row: copy everything, pad with -1; mark done via L=-1.
        # (Dead in chunked mode: those rows carry the done flag above.)
        lane = tl.arange(0, SCAN_BLOCK)
        out_ptr = out_indices_ptr + row_id * out_stride + out_off
        for c in tl.range(0, tl.cdiv(TOPK, SCAN_BLOCK)):
            pos = c * SCAN_BLOCK + lane
            tl.store(out_ptr + pos, pos.to(tl.int32), mask=pos < row_len)
            tl.store(out_ptr + pos, -1, mask=(pos >= row_len) & (pos < TOPK))
        tl.store(totals_row + 0, -1)
        tl.store(totals_row + 3, 0)
    else:
        if rem <= CAP:
            # Degenerate narrow pool: every remaining element is a
            # candidate, so L' = rem always lands in (TOPK, CAP] — fixup
            # can never fire and the sample/sort below is skipped.
            # Without this the sampler starves on narrow rows (n_cap
            # collapses the sample to a single leading tile) and every
            # row falls into fixup.
            tl.store(
                totals_row + 4 + tl.arange(0, 3),
                tl.full((3,), NEG_INF_BITS, tl.int32),
            )
        else:
            # Target 2x TOPK (capped at 3/4 CAP): keeps L' comfortably
            # inside the [TOPK, CAP] route window given the sampler's
            # spread.
            target_l = tl.minimum(TOPK * 2, CAP * 3 // 4)
            n_tiles = tl.cdiv(row_len, SCAN_BLOCK)
            # Sample tile count: largest pow2 n with the target rank fitting
            # the sort output (target_l * n*SCAN_BLOCK/row_len <= KSAMP).
            n_cap = row_len * KSAMP // (target_l * SCAN_BLOCK)
            n_raw = tl.minimum(tl.minimum(SAMP, n_tiles), tl.maximum(1, n_cap))
            n_samp = tl.where(
                n_raw >= 8,
                SAMP,
                tl.where(
                    n_raw >= 6,
                    6,
                    tl.where(n_raw >= 4, 4, tl.where(n_raw >= 2, 2, 1)),
                ),
            )
            tgt = target_l * (n_samp * SCAN_BLOCK) // row_len
            valid = tl.minimum(n_samp * SCAN_BLOCK, row_len)
            tmp = tl.zeros([TMP], dtype=tl.float32)
            props = tl.zeros([KSAMP * 2], dtype=tl.float32)
            if HI:
                # Chunked round: elements earlier rounds emitted are sunk
                # to -inf, so the sorted sample's head is a uniform
                # thinning of the REMAINING pool and the tgt/n_cap math
                # above (pool rank -> sample rank) carries over unchanged.
                # Register-tensor staging costs ~90us of scalar
                # materialization (the reason the common path DMA-copies
                # instead); acceptable on this cold path.
                hi_v = tl.load(hi_ptr + row_id * 4 + 0).to(tl.float32, bitcast=True)
                hi_i = tl.load(hi_ptr + row_id * 4 + 1)
                ti = tl.arange(0, SAMP)[:, None]
                lane2 = tl.arange(0, SCAN_BLOCK)[None, :]
                toff = ((ti * n_tiles) // n_samp) * SCAN_BLOCK
                toff = tl.maximum(0, tl.minimum(toff, row_len - SCAN_BLOCK))
                pos2 = toff + lane2
                x2 = _nan_as_inf(tl.load(base_ptr + pos2))
                ok2 = (ti < n_samp) & ((x2 < hi_v) | ((x2 == hi_v) & (pos2 > hi_i)))
                x2 = tl.where(ok2, x2, float("-inf"))
                samp_t = tle.dsa.to_buffer(
                    tl.reshape(x2, [SAMP * SCAN_BLOCK]),
                    space=tle.dsa.ascend.UB,
                )
                props = tle.dsa.ascend.raw(
                    "sort_1d_pack",
                    tle.dsa.to_tensor(samp_t),
                    tmp,
                    True,
                    KSAMP,
                    0,
                    _SORT_IMPL_BASE,
                    out=props,
                )
            else:
                # Sample tiles as scalar-offset DMA copies straight into UB
                # (loading via a computed tl tensor and handing it to the
                # raw op costs ~90us of scalar per-lane input
                # materialization). Tile source windows are clamped to stay
                # in-bounds. Tiles beyond n_samp are filled with -inf:
                # duplicating real tiles would repeat them in the sort rank
                # space and push the threshold up by ~+1 sigma, collapsing
                # L' to ~TOPK and forcing mass fixup.
                samp_ub = tle.dsa.alloc(
                    [SAMP * SCAN_BLOCK],
                    dtype=tl.float32,
                    mem_addr_space=tle.dsa.ascend.UB,
                )
                for i in tl.static_range(0, SAMP):
                    sub = tle.dsa.subview(samp_ub, [i * SCAN_BLOCK], [SCAN_BLOCK], [1])
                    if i < n_samp:
                        # Evenly spaced tiles: exact i*n_tiles//n_samp (a
                        # shift by log2(n_samp) overflows past n_tiles for
                        # i > n_samp/2 when n_samp is not 4, silently
                        # duplicating last tiles).
                        tile_off = ((i * n_tiles) // n_samp) * SCAN_BLOCK
                        tile_off = tl.maximum(
                            0, tl.minimum(tile_off, row_len - SCAN_BLOCK)
                        )
                        tle.dsa.copy(
                            base_ptr + tile_off + tl.arange(0, SCAN_BLOCK),
                            sub,
                            [SCAN_BLOCK],
                        )
                    else:
                        tle.dsa.copy(
                            neginf_ptr + tl.arange(0, SCAN_BLOCK), sub, [SCAN_BLOCK]
                        )
                props = tle.dsa.ascend.raw(
                    "sort_1d_pack",
                    tle.dsa.to_tensor(samp_ub),
                    tmp,
                    True,
                    KSAMP,
                    0,
                    _SORT_IMPL_BASE,
                    out=props,
                )
            # Clamp the rank to the sort output (rows barely above CAP can
            # compute tgt > KSAMP when n_samp collapses to 1 tile).  At the
            # clamp thr is the smallest sampled value and the expected
            # survivor count row_len * KSAMP / valid still lands inside
            # [TOPK, CAP] — far cheaper than routing the row to fixup.
            tgt = tl.maximum(1, tl.minimum(tgt, KSAMP))
            pl = tl.arange(0, KSAMP * 2)
            pick = tl.where(pl == 2 * (tgt - 1), props, 0.0)
            thr = _tree_sum_all(pick, KSAMP * 2, (KSAMP * 2 - 1).bit_length())
            thr_bits = thr.to(tl.int32, bitcast=True)
            # Bracket thresholds at +/-6 sampling sigmas (sample rank
            # space: sigma = sqrt(tgt * (1 - tgt/valid))), used ONLY by
            # the fixup kernel when the survivor count misses the
            # [TOPK, CAP] window: thr_lo serves the L < TOPK direction
            # (threshold overshot), thr_hi the L > CAP direction; the
            # common path never touches them.
            tgt_f = tgt.to(tl.float32)
            sig = tl.sqrt(tgt_f * (1.0 - tgt_f / valid))
            tgt_lo = tl.minimum(tgt + (6.0 * sig).to(tl.int32) + 1, KSAMP)
            pick_lo = tl.where(pl == 2 * (tgt_lo - 1), props, 0.0)
            thr_lo = _tree_sum_all(pick_lo, KSAMP * 2, (KSAMP * 2 - 1).bit_length())
            thr_lo_bits = thr_lo.to(tl.int32, bitcast=True)
            tgt_hi = tl.maximum(tgt - (6.0 * sig).to(tl.int32) - 1, 1)
            pick_hi = tl.where(pl == 2 * (tgt_hi - 1), props, 0.0)
            thr_hi = _tree_sum_all(pick_hi, KSAMP * 2, (KSAMP * 2 - 1).bit_length())
            thr_hi_bits = thr_hi.to(tl.int32, bitcast=True)
            tl.store(
                totals_row + 4 + tl.arange(0, 4),
                tl.where(
                    tl.arange(0, 4) == 1,
                    thr_lo_bits,
                    tl.where(tl.arange(0, 4) == 2, thr_hi_bits, thr_bits),
                ),
                mask=tl.arange(0, 4) < 3,
            )
        tl.store(totals_row + 0, 0)
        tl.store(totals_row + 3, 0)


@triton.jit
def _ascend_topk_compact_seg_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    stride0,
    mask_ptr,  # [num_rows, SB, MASK_BLK] int32 (padded vreducev2 layout) (K2 output)
    seg_base_ptr,  # [num_rows, SB // (COMPACT_SEG // SUB)] exclusive segment bases
    cand_vals_ptr,  # [num_rows, CAP] fp32
    cand_idx_ptr,  # [num_rows, CAP] int32
    totals_ptr,  # [num_rows, 8]
    NSEG: tl.constexpr,  # COMPACT_SEG-element segments per vocab row
    SB: tl.constexpr,
    CSEG: tl.constexpr,  # == COMPACT_SEG
    CSEG_SUB: tl.constexpr,  # subtiles per CSEG segment
    SUB: tl.constexpr,  # elements per mask subtile (== one vreducev2 repeat)
    TOPK: tl.constexpr,
    CAP_SEG: tl.constexpr,  # == CAP; per-segment count <= L' <= CAP
    REPACK: tl.constexpr = 0,
):
    # Per-COMPACT_SEG-segment candidate compaction. Two vreducev2 calls
    # (gather_mask_custom_pattern) compact the candidate values and their
    # segment-local positions against the padded K2 mask words; no scalar
    # GM access and no ctz/vgather custom op.
    pid = tl.program_id(0)
    row_id = pid // NSEG
    seg = pid % NSEG
    totals_ptr += row_id * 8
    L = tl.load(totals_ptr + 0)
    if L < 0:
        return  # short row
    route = tl.load(totals_ptr + 3)
    if REPACK:
        skip = route != 3  # second pass: re-routed rows only
    else:
        skip = route == 2  # fixup row
    if skip:
        return
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    nseg_row = tl.cdiv(row_len, CSEG)
    if seg >= nseg_row:
        return
    seg0 = seg * CSEG
    valid = tl.minimum(row_len - seg0, CSEG)
    x_ptr = logits_ptr + row_id * stride0 + row_start
    lanes = tl.arange(0, CSEG)
    if valid == CSEG:
        seg_t = tl.load(x_ptr + seg0 + lanes)
    else:
        # Tail segment: masked load keeps the read in-bounds (the valid
        # window can end before the segment does, esp. the last row).
        # Values stay RAW here (NaN included): canonicalization to +inf
        # happens on the candidate loads in sort_final, which touch only
        # CAP elements per row instead of this full-row scan.
        seg_t = tl.load(x_ptr + seg0 + lanes, mask=lanes < valid, other=0.0)
    seg_buf = tle.dsa.to_buffer(seg_t, space=tle.dsa.ascend.UB)
    # Padded mask layout: MASK_BLK words per 64-element subtile, of which
    # vreducev2 reads only the heading 64 bits (src1RepeatStride is in
    # 32-byte blocks).  Subtiles past the valid window are never read:
    # repeat_times covers exactly the valid subtiles.
    PATW: tl.constexpr = CSEG_SUB * MASK_BLK
    ml = tl.arange(0, PATW)
    nw = tl.cdiv(valid, SUB) * MASK_BLK
    msk_t = tl.load(
        mask_ptr + row_id * SB * MASK_BLK + seg * PATW + ml,
        mask=ml < nw,
        other=0,
    )
    msk_buf = tle.dsa.to_buffer(
        msk_t.to(tl.uint32, bitcast=True), space=tle.dsa.ascend.UB
    )
    # Segment-local candidate positions come from compacting an arange with
    # the same mask (second vreducev2 call) instead of ctz extraction.
    ar_buf = tle.dsa.to_buffer(lanes.to(tl.int32), space=tle.dsa.ascend.UB)
    # Segment base/count from the reduce kernel's exclusive segment bases.
    if NSEG == 1:
        base = tl.zeros((), dtype=tl.int32)
        c = L
    else:
        SEG_STRIDE: tl.constexpr = SB // CSEG_SUB
        base = tl.load(seg_base_ptr + row_id * SEG_STRIDE + seg)
        nxt = tl.load(
            seg_base_ptr
            + row_id * SEG_STRIDE
            + seg
            + tl.where(seg + 1 < nseg_row, 1, 0)
        )
        c = tl.where(seg + 1 < nseg_row, nxt, L) - base
    vals = tl.zeros([CAP_SEG], dtype=tl.float32)
    idxs = tl.zeros([CAP_SEG], dtype=tl.int32)
    rc1 = tl.zeros([1], dtype=tl.int64)
    rc2 = tl.zeros([1], dtype=tl.int64)
    reps = tl.cdiv(valid, SUB)
    vals, rc1 = tle.dsa.ascend.raw(
        "gather_mask_custom_pattern",
        tle.dsa.to_tensor(seg_buf),
        tle.dsa.to_tensor(msk_buf),
        True,
        SUB,
        1,
        reps,
        8,
        1,
        out=[vals, rc1],
    )
    idxs, rc2 = tle.dsa.ascend.raw(
        "gather_mask_custom_pattern",
        tle.dsa.to_tensor(ar_buf),
        tle.dsa.to_tensor(msk_buf),
        True,
        SUB,
        1,
        reps,
        8,
        1,
        out=[idxs, rc2],
    )
    pl = tl.arange(0, CAP_SEG)
    m = pl < c
    # Canonicalize NaN -> +inf on the GATHERED candidates (CAP registers,
    # no extra load): cheaper than canonicalizing the full-row load, and
    # keeps sort_final's cand load directly feedable to the sort raw op
    # (a where between that load and to_buffer would force scalar
    # per-lane materialization of the sort input).
    tl.store(cand_vals_ptr + row_id * CAP_SEG + base + pl, _nan_as_inf(vals), mask=m)
    tl.store(cand_idx_ptr + row_id * CAP_SEG + base + pl, idxs + seg0, mask=m)


@triton.jit
def _ascend_topk_sort_final_kernel(
    out_indices_ptr,
    cand_vals_ptr,  # [num_rows, CAP] fp32
    cand_idx_ptr,  # [num_rows, CAP] int32
    totals_ptr,  # [num_rows, 8]
    out_stride,  # out_indices row stride (== TOPK unless chunked)
    out_off,  # this round's column offset into out_indices
    TOPK: tl.constexpr,
    CAP: tl.constexpr,
    TMP: tl.constexpr,  # sort_1d_pack BASE workspace: CAP * 4 floats
    REPACK: tl.constexpr = 0,
):
    # Exact selection over the L' <= CAP candidates: one hardware sort
    # (vbitsort + vmrgsort4 via the TLE custom ops).
    # Rows with L' <= CAP//2 take a half-width sort (~2x cheaper); the
    # sampler targets L' ~ 2x TOPK, so k = CAP//4 rows (e.g. k=1024)
    # land there most of the time.
    row_id = tl.program_id(0)
    totals_ptr += row_id * 8
    L = tl.load(totals_ptr + 0)
    if L < 0:
        return  # short row
    if REPACK:
        skip = tl.load(totals_ptr + 3) != 3  # second pass: re-routed only
    else:
        skip = tl.load(totals_ptr + 3) == 2  # fixup row
    if skip:
        return  # the fixup kernel writes a fixup row's output directly
    out_ptr = out_indices_ptr + row_id * out_stride + out_off
    CAP4: tl.constexpr = CAP // 4
    if TOPK <= CAP4:
        if L <= CAP4:
            pl = tl.arange(0, CAP4)
            pm = pl < L
            vals_t = tl.load(
                cand_vals_ptr + row_id * CAP + pl, mask=pm, other=float("-inf")
            )
            idx_t = tl.load(cand_idx_ptr + row_id * CAP + pl, mask=pm, other=0)
            vals_buf = tle.dsa.to_buffer(vals_t, space=tle.dsa.ascend.UB)
            tmp = tl.zeros([CAP4 * 4], dtype=tl.float32)
            props = tl.zeros([TOPK * 2], dtype=tl.float32)
            props = tle.dsa.ascend.raw(
                "sort_1d_pack",
                tle.dsa.to_tensor(vals_buf),
                tmp,
                True,
                TOPK,
                0,
                _SORT_IMPL_BASE,
                out=props,
            )
            dval = tl.zeros([TOPK], dtype=tl.float32)
            didx = tl.zeros([TOPK], dtype=tl.int32)
            dval, didx = tle.dsa.ascend.raw(
                "unpack_sort", props, TOPK, out=[dval, didx]
            )
            final = tl.gather(idx_t.to(tl.float32, bitcast=True), didx, axis=0)
            tl.store(
                out_ptr + tl.arange(0, TOPK),
                final.to(tl.int32, bitcast=True),
            )
            return
    CAP2: tl.constexpr = CAP // 2
    if L <= CAP2:
        pl = tl.arange(0, CAP2)
        pm = pl < L
        vals_t = tl.load(
            cand_vals_ptr + row_id * CAP + pl, mask=pm, other=float("-inf")
        )
        idx_t = tl.load(cand_idx_ptr + row_id * CAP + pl, mask=pm, other=0)
        vals_buf = tle.dsa.to_buffer(vals_t, space=tle.dsa.ascend.UB)
        tmp = tl.zeros([CAP2 * 4], dtype=tl.float32)
        props = tl.zeros([TOPK * 2], dtype=tl.float32)
        props = tle.dsa.ascend.raw(
            "sort_1d_pack",
            tle.dsa.to_tensor(vals_buf),
            tmp,
            True,
            TOPK,
            0,
            _SORT_IMPL_BASE,
            out=props,
        )
        dval = tl.zeros([TOPK], dtype=tl.float32)
        didx = tl.zeros([TOPK], dtype=tl.int32)
        dval, didx = tle.dsa.ascend.raw("unpack_sort", props, TOPK, out=[dval, didx])
        final = tl.gather(idx_t.to(tl.float32, bitcast=True), didx, axis=0)
        tl.store(
            out_ptr + tl.arange(0, TOPK),
            final.to(tl.int32, bitcast=True),
        )
        return
    pl = tl.arange(0, CAP)
    pm = pl < L
    vals_t = tl.load(cand_vals_ptr + row_id * CAP + pl, mask=pm, other=float("-inf"))
    vals_buf = tle.dsa.to_buffer(vals_t, space=tle.dsa.ascend.UB)
    tmp = tl.zeros([TMP], dtype=tl.float32)
    props = tl.zeros([TOPK * 2], dtype=tl.float32)
    props = tle.dsa.ascend.raw(
        "sort_1d_pack",
        tle.dsa.to_tensor(vals_buf),
        tmp,
        True,
        TOPK,
        0,
        _SORT_IMPL_BASE,
        out=props,
    )
    dval = tl.zeros([TOPK], dtype=tl.float32)
    didx = tl.zeros([TOPK], dtype=tl.int32)
    dval, didx = tle.dsa.ascend.raw("unpack_sort", props, TOPK, out=[dval, didx])
    # Proposals carry candidate-local positions; remap to the original
    # (row_start-relative) indices. tl.gather lowers to hivm.vgather;
    # the bitcast works around the frontend's float-only source list.
    idx_t = tl.load(cand_idx_ptr + row_id * CAP + pl, mask=pm, other=0)
    final = tl.gather(idx_t.to(tl.float32, bitcast=True), didx, axis=0)
    tl.store(
        out_ptr + tl.arange(0, TOPK),
        final.to(tl.int32, bitcast=True),
    )


@triton.jit
def _ascend_topk_fixup_kernel(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    totals_ptr,  # [num_rows, 8]
    fixup_flag_ptr,  # [1] int32
    hi_ptr,  # [num_rows, 4] int32 chunk state (dummy when HI=0)
    out_stride,  # out_indices row stride (== TOPK unless chunked)
    out_off,  # this round's column offset into out_indices
    TOPK: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    FINAL: tl.constexpr = 0,
    HI: tl.constexpr = 0,  # 1: chunked round, apply the lexicographic bound
):
    # Exact fallback for route-2 rows: the survivor count at thr landed
    # outside [TOPK, CAP], which means the distribution is degenerate in a
    # way the sample cannot represent.  The bracket threshold is picked by
    # direction: L < TOPK (overshot) -> thr_lo, L > CAP (undershot) ->
    # thr_hi.  Three cases:
    #   * massive tie at one value (typical, e.g. quantized scores):
    #     count(x > t_sel) <= TOPK <= count(x >= t_sel) — emit the
    #     strict winners, fill the rest with ties (any subset: identical
    #     values are interchangeable in the top-k set).  3 read passes.
    #   * bracket threshold strictly past thr (the realistic overshoot on
    #     continuous data: the +-6-sigma bracket lands ~2x TOPK): RE-ROUTE
    #     the row (route=3) with t_sel as the new threshold — no full-row
    #     pass at all; the host re-runs the vectorized pack/reduce/
    #     compact/sort pass and the reduce kernel re-checks the window.
    #     Skipped when FINAL=1 (a re-routed row that missed the window
    #     again falls through to the descent below).
    #   * otherwise (~never): 32-pass binary descent over the
    #     order-preserving descending int32 key of the fp32 bit pattern
    #     (smaller key = larger float) pinpoints the exact top-k boundary
    #     value, then strict/tie fill.  ~34 read passes.
    # Scalar tl.sum reductions throughout: the nested loops defeat the
    # split+add tree lowering, and this path is rare by construction.
    row_id = tl.program_id(0)
    totals_ptr += row_id * 8
    if tl.load(totals_ptr + 3) != 2:
        return
    L = tl.load(totals_ptr + 0)
    t_bits = tl.where(
        L < TOPK,
        tl.load(totals_ptr + 5),  # thr_lo: threshold overshot
        tl.load(totals_ptr + 6),  # thr_hi: threshold undershot
    )
    if FINAL == 0:
        if t_bits != tl.load(totals_ptr + 4):
            # Bracket threshold is strictly past thr (continuous data: the
            # +-6-sigma bracket lands ~2x TOPK, inside [TOPK, CAP]):
            # RE-ROUTE without any full-row pass — the reduce kernel
            # re-checks the window after the re-pack, and a second miss
            # falls to the FINAL=1 descent below.  A COLLAPSED bracket
            # (t_sel == thr) means a massive tie sits on the boundary;
            # that case needs the exact tie-fill below.
            tl.store(totals_ptr + 4, t_bits)
            tl.store(totals_ptr + 3, 3)
            tl.atomic_max(fixup_flag_ptr, 2)
            return
    t_sel = t_bits.to(tl.float32, bitcast=True)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    row_len = row_end - row_start
    base_ptr = logits_ptr + row_id * stride0 + row_start
    out_ptr = out_indices_ptr + row_id * out_stride + out_off
    if HI:
        # Chunked round: every counting/emission pass below excludes what
        # earlier rounds emitted (lexicographic bound, total order).
        hi_v = tl.load(hi_ptr + row_id * 4 + 0).to(tl.float32, bitcast=True)
        hi_i = tl.load(hi_ptr + row_id * 4 + 1)
    lane = tl.arange(0, SCAN_BLOCK)
    n_tiles = tl.cdiv(row_len, SCAN_BLOCK)

    L_s = tl.zeros((), dtype=tl.int32)
    E = tl.zeros((), dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        offs = t * SCAN_BLOCK + lane
        in_range = offs < row_len
        # Clamp masked-lane addresses: masked lanes still issue them.
        x = _nan_as_inf(
            tl.load(base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0)
        )
        pool = in_range
        if HI:
            pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
        L_s += tl.sum((pool & (x > t_sel)).to(tl.int32), axis=0)
        E += tl.sum((pool & (x == t_sel)).to(tl.int32), axis=0)

    if (L_s <= TOPK) & (L_s + E >= TOPK):
        base_out = tl.zeros((), dtype=tl.int32)
        for t in tl.range(0, n_tiles):
            offs = t * SCAN_BLOCK + lane
            in_range = offs < row_len
            x = _nan_as_inf(
                tl.load(
                    base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0
                )
            )
            pool = in_range
            if HI:
                pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
            take = pool & (x > t_sel)
            take_i32 = take.to(tl.int32)
            pos = base_out + tl.cumsum(take_i32, axis=0) - take_i32
            ok = take & (pos < TOPK)
            tl.store(out_ptr + tl.where(ok, pos, 0), offs.to(tl.int32), mask=ok)
            base_out += tl.sum(take_i32, axis=0)
        for t in tl.range(0, n_tiles):
            if base_out < TOPK:
                offs = t * SCAN_BLOCK + lane
                in_range = offs < row_len
                x = _nan_as_inf(
                    tl.load(
                        base_ptr + tl.where(in_range, offs, 0),
                        mask=in_range,
                        other=0.0,
                    )
                )
                pool = in_range
                if HI:
                    pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
                take = pool & (x == t_sel)
                take_i32 = take.to(tl.int32)
                pos = base_out + tl.cumsum(take_i32, axis=0) - take_i32
                ok = take & (pos < TOPK)
                tl.store(out_ptr + tl.where(ok, pos, 0), offs.to(tl.int32), mask=ok)
                base_out += tl.sum(take_i32, axis=0)
        tl.store(totals_ptr + 3, 4)  # done
    else:
        # 32-pass descent over the descending-order int32 key
        #   skey = bits < 0 ? (bits & 0x7FFFFFFF) : ~bits
        # (smaller skey = larger float; signed int32 order on skey matches
        # the float order for any mixture of signs).  The sign bit is
        # decided by a separate counting pass: the descent test
        # "count(skey < T|bit) < TOPK" derives prefix ranges that only
        # partition the key space within one sign half — at the bit-31
        # step the trial (INT_MIN) is below every key, so the sign bit
        # was always kept and any row whose top-k boundary value is
        # NEGATIVE (skey >= 0) converged to a nonexistent key and emitted
        # garbage (probe: 2029/2048 valid).  (An unsigned ukey domain
        # fixes this too, but this backend cannot codegen uint32
        # compares: hivm.hir.vcast uint32->uint64 unsupported.)  T ends
        # as the exact TOPK-th smallest skey; winners fill strictly below
        # T, ties at T fill the rest.
        S: tl.constexpr = -2147483647 - 1  # 0x80000000 bit pattern
        cnt0 = tl.zeros((), dtype=tl.int32)
        for t in tl.range(0, n_tiles):
            offs = t * SCAN_BLOCK + lane
            in_range = offs < row_len
            x = _nan_as_inf(
                tl.load(
                    base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0
                )
            )
            pool = in_range
            if HI:
                pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
            bits = x.to(tl.int32, bitcast=True)
            skey = tl.where(bits < 0, bits & 0x7FFFFFFF, ~bits)
            cnt0 += tl.sum((pool & (skey < 0)).to(tl.int32), axis=0)
        T = tl.where(cnt0 >= TOPK, S, 0)
        for bp_i in tl.range(1, 32):
            bit = 1 << (31 - bp_i)
            trial = T | bit
            cnt = tl.zeros((), dtype=tl.int32)
            for t in tl.range(0, n_tiles):
                offs = t * SCAN_BLOCK + lane
                in_range = offs < row_len
                x = _nan_as_inf(
                    tl.load(
                        base_ptr + tl.where(in_range, offs, 0),
                        mask=in_range,
                        other=0.0,
                    )
                )
                pool = in_range
                if HI:
                    pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
                bits = x.to(tl.int32, bitcast=True)
                skey = tl.where(bits < 0, bits & 0x7FFFFFFF, ~bits)
                cnt += tl.sum((pool & (skey < trial)).to(tl.int32), axis=0)
            # Branchless update: a runtime `if` around the loop-carried T
            # risks a scalar-phi miscompile on this backend.
            T = tl.where(cnt < TOPK, trial, T)
        base_out = tl.zeros((), dtype=tl.int32)
        for t in tl.range(0, n_tiles):
            offs = t * SCAN_BLOCK + lane
            in_range = offs < row_len
            x = _nan_as_inf(
                tl.load(
                    base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0
                )
            )
            pool = in_range
            if HI:
                pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
            bits = x.to(tl.int32, bitcast=True)
            skey = tl.where(bits < 0, bits & 0x7FFFFFFF, ~bits)
            take = pool & (skey < T)
            take_i32 = take.to(tl.int32)
            pos = base_out + tl.cumsum(take_i32, axis=0) - take_i32
            ok = take & (pos < TOPK)
            tl.store(out_ptr + tl.where(ok, pos, 0), offs.to(tl.int32), mask=ok)
            base_out += tl.sum(take_i32, axis=0)
        for t in tl.range(0, n_tiles):
            if base_out < TOPK:
                offs = t * SCAN_BLOCK + lane
                in_range = offs < row_len
                x = _nan_as_inf(
                    tl.load(
                        base_ptr + tl.where(in_range, offs, 0),
                        mask=in_range,
                        other=0.0,
                    )
                )
                pool = in_range
                if HI:
                    pool = pool & ((x < hi_v) | ((x == hi_v) & (offs > hi_i)))
                bits = x.to(tl.int32, bitcast=True)
                skey = tl.where(bits < 0, bits & 0x7FFFFFFF, ~bits)
                take = pool & (skey == T)
                take_i32 = take.to(tl.int32)
                pos = base_out + tl.cumsum(take_i32, axis=0) - take_i32
                ok = take & (pos < TOPK)
                tl.store(out_ptr + tl.where(ok, pos, 0), offs.to(tl.int32), mask=ok)
                base_out += tl.sum(take_i32, axis=0)
        tl.store(totals_ptr + 3, 4)  # done


@triton.jit
def _ascend_topk_chunk_short_kernel(
    out_indices_ptr,
    row_starts,
    row_ends,
    state_ptr,  # [num_rows, 4] int32 chunk state
    out_stride,  # out_indices row stride (== full top_k)
    top_k,  # full top_k (runtime: chunked mode is top_k > CHUNK_K)
    BLOCK: tl.constexpr,
):
    # Chunked route (top_k > CHUNK_K) pre-pass.  Short rows
    # (row_len <= top_k) are fully emitted here — positions then -1 pad —
    # and marked done so every round of the threshold pipeline skips them.
    # Other rows get the round-0 lexicographic bound (+inf, -1), which
    # lets every element through the HI filter.
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_len = tl.load(row_ends + row_id) - row_start
    state_ptr += row_id * 4
    if row_len <= top_k:
        lane = tl.arange(0, BLOCK)
        out_ptr = out_indices_ptr + row_id * out_stride
        for c0 in tl.range(0, tl.cdiv(top_k, BLOCK)):
            pos = c0 * BLOCK + lane
            tl.store(out_ptr + pos, pos.to(tl.int32), mask=pos < row_len)
            tl.store(out_ptr + pos, -1, mask=(pos >= row_len) & (pos < top_k))
        tl.store(state_ptr + 2, 1)  # done
    else:
        tl.store(state_ptr + 0, 2139095040)  # 0x7F800000: float("+inf") bits
        tl.store(state_ptr + 1, -1)
        tl.store(state_ptr + 2, 0)


@triton.jit
def _ascend_topk_chunk_boundary_kernel(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    stride0,
    state_ptr,  # [num_rows, 4] int32 chunk state
    out_stride,  # out_indices row stride (== full top_k)
    out_off,  # this round's column offset into out_indices
    SCAN_BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,  # == CHUNK_K: indices emitted this round
):
    # Lexicographic round boundary for the chunked route.  v is the
    # round's smallest selected value; c of this round's entries tie at
    # v.  v MUST come from a full-segment min: sort_final emits
    # value-sorted output (last slot = min), but the fixup paths emit
    # winners position-ordered, so the last slot can be any winner
    # (probe: seg[-1]=-0.320 while vmin=-3.169 → pass 1 underfilled,
    # stale slots duplicated).  The round segment is REBUILT from
    # scratch in position order — a tie at v may sit in an early winner
    # slot and a suffix-only rewrite would duplicate it.  Two passes
    # over the PRE-round pool
    # ((x < hi_v) | ((x == hi_v) & (pos > cursor_prev))) rewrite the
    # segment to: every pool element with x > v (exactly CHUNK - c of
    # them) in slots [0, CHUNK-c), then the first c pool elements with
    # x == v in slots [CHUNK-c, CHUNK).  The value multiset is unchanged
    # (any tie subset is a valid top-k), and the emitted set then
    # matches "every value > v, plus value == v with
    # cursor_prev < pos <= cursor" EXACTLY, so the next round's filter
    # can neither re-emit nor lose an element however degenerate the tie
    # distribution is.
    row_id = tl.program_id(0)
    state_ptr += row_id * 4
    if tl.load(state_ptr + 2) != 0:
        return  # short row, finished by the short kernel
    row_start = tl.load(row_starts + row_id)
    row_len = tl.load(row_ends + row_id) - row_start
    base_ptr = logits_ptr + row_id * stride0 + row_start
    seg_ptr = out_indices_ptr + row_id * out_stride + out_off
    pl = tl.arange(0, CHUNK)
    idx_seg = tl.load(seg_ptr + pl)
    vals = _nan_as_inf(tl.load(base_ptr + idx_seg))
    v = tl.min(vals, axis=0)
    c = tl.sum((vals == v).to(tl.int32), axis=0)
    hi_v = tl.load(state_ptr + 0).to(tl.float32, bitcast=True)
    cursor_prev = tl.load(state_ptr + 1)
    lane = tl.arange(0, SCAN_BLOCK)
    n_tiles = tl.cdiv(row_len, SCAN_BLOCK)
    w = CHUNK - c  # winner count == first tie slot
    # Pass 1: re-emit winners (x > v) position-ordered into [0, w).
    base = tl.zeros((), dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        if base < w:
            offs = t * SCAN_BLOCK + lane
            in_range = offs < row_len
            x = _nan_as_inf(
                tl.load(
                    base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0
                )
            )
            pool = in_range & ((x < hi_v) | ((x == hi_v) & (offs > cursor_prev)))
            m = pool & (x > v)
            m_i32 = m.to(tl.int32)
            rank = base + tl.cumsum(m_i32, axis=0)
            ok = m & (rank <= w)
            tl.store(seg_ptr - 1 + rank, offs.to(tl.int32), mask=ok)
            base += tl.sum(m_i32, axis=0)
    # Pass 2: re-emit the first c ties (x == v) into [w, CHUNK) and
    # record the new cursor (position of the last emitted tie).
    base = tl.zeros((), dtype=tl.int32)
    cursor = tl.zeros((), dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        if base < c:
            offs = t * SCAN_BLOCK + lane
            in_range = offs < row_len
            x = _nan_as_inf(
                tl.load(
                    base_ptr + tl.where(in_range, offs, 0), mask=in_range, other=0.0
                )
            )
            pool = in_range & ((x < hi_v) | ((x == hi_v) & (offs > cursor_prev)))
            m = pool & (x == v)
            m_i32 = m.to(tl.int32)
            rank = base + tl.cumsum(m_i32, axis=0)
            ok = m & (rank <= c)
            tl.store(seg_ptr + w - 1 + rank, offs.to(tl.int32), mask=ok)
            hit = m & (rank == c)
            cursor += tl.sum(tl.where(hit, offs, 0), axis=0)
            base += tl.sum(m_i32, axis=0)
    tl.store(state_ptr + 0, v.to(tl.int32, bitcast=True))
    tl.store(state_ptr + 1, cursor)


@triton.jit
def _ascend_topk_row_sort_pad_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    pad_ptr,  # [pad_rows, BUFW] fp32 scratch, -inf-padded row copies
    stride0,
    num_rows,
    pad_rows,  # min(2, num_rows): grid size and live scratch slots
    TOPK: tl.constexpr,
    BUFW: tl.constexpr,
):
    # Copy each of the last pad_rows rows into its scratch slot, -inf-padded
    # to the pow2 load width.  The clamp is safe here: this kernel has no
    # sort workspace, so the BUFW-wide clamp vector fits UB trivially.  Rows
    # are copied unconditionally (trivial rows included); only rows with
    # row_len > TOPK are ever read back through the scratch.
    row_id = num_rows - pad_rows + tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_len = tl.load(row_ends + row_id) - row_start
    lane = tl.arange(0, BUFW)
    m = lane < row_len
    src_ptr = logits_ptr + row_id * stride0 + row_start
    x = _nan_as_inf(
        tl.load(src_ptr + tl.where(m, lane, 0), mask=m, other=float("-inf"))
    )
    tl.store(pad_ptr + tl.program_id(0) * BUFW + lane, x)


@triton.jit
def _ascend_topk_row_sort_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    pad_ptr,  # [pad_rows, BUFW] fp32 scratch: -inf-padded tail-row copies
    out_indices_ptr,
    stride0,
    num_rows,
    pad_rows,  # min(2, num_rows)
    TOPK: tl.constexpr,
    VOCAB: tl.constexpr,  # exact sort width (== the batch's vocab_size)
    BUFW: tl.constexpr,  # pow2 load width covering VOCAB
    TMP_SZ: tl.constexpr,
    SORT_IMPL: tl.constexpr,
):
    # One CTA per row: a single exact-width hardware sort over the whole
    # row, unpack in-kernel, emit TOPK indices.  Sorting the exact VOCAB
    # (a subview of the pow2 load buffer, so the -inf tail of ragged rows
    # never widens the sort) costs ~7.5us/CTA at 4-5K wide vs ~2.9us at
    # 2048 wide, but deletes the segment-merge kernel and the proposal
    # round-trip — a net win at every benchmarked narrow shape.
    # Masked lanes still issue their addresses on this backend, so a row
    # whose pow2 load window reaches past the logits storage would fault.
    # Only the last two rows can over-run (BUFW < 2*VOCAB <= 2*stride0);
    # a tiny pad kernel copies them into a -inf-padded scratch first (a
    # per-lane address clamp would materialize a BUFW-wide vector and
    # overflow UB at BUFW=8192, measured).
    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_len = tl.load(row_ends + row_id) - row_start
    if row_len <= TOPK:
        # Trivial row: every element wins, emit 0..row_len-1
        # (row_start-relative), padded with -1.
        lanes = tl.arange(0, TOPK)
        tl.store(
            out_indices_ptr + row_id * TOPK + lanes,
            tl.where(lanes < row_len, lanes, -1),
        )
        return
    lane = tl.arange(0, BUFW)
    m = lane < row_len
    if row_id >= num_rows - pad_rows:
        # Pointer phis across runtime branches miscompile on this backend
        # (measured: every row read the scratch slots); the load itself
        # stays inside the branch so the phi is on the loaded tensor.
        x = tl.load(
            pad_ptr + (row_id - (num_rows - pad_rows)) * BUFW + lane,
            mask=m,
            other=float("-inf"),
        )
    else:
        x = _nan_as_inf(
            tl.load(
                logits_ptr + row_id * stride0 + row_start + lane,
                mask=m,
                other=float("-inf"),
            )
        )
    src_ub = tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB)
    tmp = tl.zeros([TMP_SZ], dtype=tl.float32)
    props = tl.zeros([TOPK * 2], dtype=tl.float32)
    if VOCAB < BUFW:
        sub = tle.dsa.subview(src_ub, [0], [VOCAB], [1])
        props = tle.dsa.ascend.raw(
            "sort_1d_pack",
            tle.dsa.to_tensor(sub),
            tmp,
            True,
            TOPK,
            0,
            SORT_IMPL,
            out=props,
        )
    else:
        props = tle.dsa.ascend.raw(
            "sort_1d_pack",
            tle.dsa.to_tensor(src_ub),
            tmp,
            True,
            TOPK,
            0,
            SORT_IMPL,
            out=props,
        )
    p_ub = tle.dsa.to_buffer(props, space=tle.dsa.ascend.UB)
    v1 = tl.zeros([TOPK], dtype=tl.float32)
    i1 = tl.zeros([TOPK], dtype=tl.int32)
    v1, i1 = tle.dsa.ascend.raw(
        "unpack_sort", tle.dsa.to_tensor(p_ub), TOPK, out=[v1, i1]
    )
    tl.store(out_indices_ptr + row_id * TOPK + tl.arange(0, TOPK), i1)


@triton.jit
def _ascend_topk_seg_pad_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    pad_ptr,  # [pad_rows, PADW] fp32 scratch, tail-row copies
    stride0,
    num_rows,
    pad_rows,  # min(2, num_rows)
    PADW: tl.constexpr,  # == SEGS * SEG_LEN
    BLOCK: tl.constexpr,
):
    # Copy the valid window [row_start, row_start + copy_w) of each of the
    # last pad_rows rows into its scratch slot, where copy_w = min(PADW,
    # stride0 - row_start) never crosses the logits storage.  Scratch slots
    # beyond copy_w stay garbage — the seg-sort kernel's masked load never
    # picks those lanes up (mask < row_len <= copy_w, other=-inf).
    # All loads are unmasked affine: the tail CTA shifts its window left
    # (base = copy_w - BLOCK, legal because stride0 > BLOCK keeps the
    # global address non-negative) instead of clamping per lane — a
    # tl.where address clamp scalarizes the load and slows the CTA ~40x
    # (msprof: 233us vs 6us).  For copy_w < BLOCK the base goes negative;
    # the store masks those lanes off (masked stores with out-of-range
    # addresses are safe on this backend, probed), preserving the
    # scratch[j] == logits[row_start + j] mapping.
    slot = tl.program_id(0)
    cid = tl.program_id(1)
    row_id = num_rows - pad_rows + slot
    row_start = tl.load(row_starts + row_id)
    copy_w = tl.minimum(PADW, stride0 - row_start)
    if cid * BLOCK >= copy_w:
        return
    base = tl.minimum(cid * BLOCK, copy_w - BLOCK)
    lane = tl.arange(0, BLOCK)
    x = _nan_as_inf(tl.load(logits_ptr + row_id * stride0 + row_start + base + lane))
    tl.store(pad_ptr + slot * PADW + base + lane, x, mask=base + lane >= 0)


@triton.jit
def _ascend_topk_seg_sort_kernel(
    logits_ptr,
    row_starts,
    row_ends,
    pad_ptr,  # [pad_rows, PADW] fp32 scratch: -inf-padded tail-row copies
    props_ptr,  # [num_rows, SLOTS, 2*TOPK] fp32 proposal streams
    stride0,
    num_rows,
    pad_rows,  # min(2, num_rows)
    TOPK: tl.constexpr,
    SEGS: tl.constexpr,
    SLOTS: tl.constexpr,  # SEGS + merge scratch slots (SA)
    SEG_LEN: tl.constexpr,
    PADW: tl.constexpr,  # == SEGS * SEG_LEN
    TMP_SZ: tl.constexpr,
    SORT_IMPL: tl.constexpr,
):
    # One CTA per (row, segment): sort the SEG_LEN-wide segment, emit TOPK
    # proposals.  seg_off as index_offset makes proposals carry
    # row_start-relative positions directly.  Masked lanes still issue
    # their addresses on this backend, and a segment window can reach past
    # the logits storage for the last TWO rows (SEG_LEN <= stride0, so
    # row n-3 and earlier stay in bounds); those rows read from the
    # -inf-padded scratch written by the pad kernel instead (a tl.where
    # address clamp here would scalarize the load and slow the whole
    # kernel ~40x, msprof).
    pid = tl.program_id(0)
    seg = pid % SEGS
    row_id = pid // SEGS
    row_start = tl.load(row_starts + row_id)
    row_len = tl.load(row_ends + row_id) - row_start
    if row_len <= TOPK:
        return  # trivial row, written by the merge kernel
    seg_off = seg * SEG_LEN
    if seg_off >= row_len:
        # Empty segment (ragged batch): skip the sort entirely; the merge
        # kernel clamps this way to length 0 and never reads the stream.
        return
    lane = tl.arange(0, SEG_LEN)
    offs = seg_off + lane
    m = offs < row_len
    if row_id >= num_rows - pad_rows:
        # Pointer phis across runtime branches miscompile on this backend;
        # the load stays inside the branch so the phi is on the tensor.
        x = tl.load(
            pad_ptr + (row_id - (num_rows - pad_rows)) * PADW + offs,
            mask=m,
            other=float("-inf"),
        )
    else:
        x = _nan_as_inf(
            tl.load(
                logits_ptr + row_id * stride0 + row_start + offs,
                mask=m,
                other=float("-inf"),
            )
        )
    src_ub = tle.dsa.to_buffer(x, space=tle.dsa.ascend.UB)
    tmp = tl.zeros([TMP_SZ], dtype=tl.float32)
    props = tl.zeros([TOPK * 2], dtype=tl.float32)
    props = tle.dsa.ascend.raw(
        "sort_1d_pack",
        tle.dsa.to_tensor(src_ub),
        tmp,
        True,
        TOPK,
        seg_off,
        SORT_IMPL,
        out=props,
    )
    tl.store(
        props_ptr + (row_id * SLOTS + seg) * 2 * TOPK + tl.arange(0, 2 * TOPK), props
    )


@triton.jit
def _merge4_exhaust(in_ub_t, dst_ptr, n_ways, l0, l1, l2, l3, TOPK: tl.constexpr):
    # Merge up to 4 sorted packed proposal streams held in in_ub (way w at
    # [w*2*TOPK, (w+1)*2*TOPK), length l_w) and append up to TOPK merged
    # proposals at dst_ptr; returns the number of proposals written.
    # merge_exhaust_sort4 stops as soon as one way runs out and reports the
    # per-way consumed counts; repeated rounds append consecutive sorted
    # prefixes until TOPK proposals are collected (4 rounds suffice: each
    # round exhausts at least one way, so round r finishes with at most
    # 3-r live ways).
    c0 = tl.zeros((), dtype=tl.int32)
    c1 = tl.zeros((), dtype=tl.int32)
    c2 = tl.zeros((), dtype=tl.int32)
    c3 = tl.zeros((), dtype=tl.int32)
    got = tl.zeros((), dtype=tl.int32)
    cons = tl.zeros([4], dtype=tl.int32)
    idx4 = tl.arange(0, 4)
    oo = tl.arange(0, 2 * TOPK)
    for _ in tl.static_range(0, 4):
        if got < TOPK:
            out_t = tl.zeros([2 * TOPK], dtype=tl.float32)
            out_t, cons = tle.dsa.ascend.raw(
                "merge_exhaust_sort4",
                in_ub_t,
                n_ways,
                c0,
                TOPK + c1,
                2 * TOPK + c2,
                3 * TOPK + c3,
                l0,
                l1,
                l2,
                l3,
                out=[out_t, cons],
            )
            cons0 = tl.sum(tl.where(idx4 == 0, cons, 0))
            cons1 = tl.sum(tl.where(idx4 == 1, cons, 0))
            cons2 = tl.sum(tl.where(idx4 == 2, cons, 0))
            cons3 = tl.sum(tl.where(idx4 == 3, cons, 0))
            take = tl.minimum(cons0 + cons1 + cons2 + cons3, TOPK - got)
            c0 += cons0
            c1 += cons1
            c2 += cons2
            c3 += cons3
            l0 -= cons0
            l1 -= cons1
            l2 -= cons2
            l3 -= cons3
            tl.store(dst_ptr + got * 2 + oo, out_t, mask=oo < take * 2)
            got += take
    return got


@triton.jit
def _ascend_topk_seg_merge_kernel(
    row_starts,
    row_ends,
    props_ptr,  # [num_rows, SLOTS, 2*TOPK] fp32 proposal streams
    out_indices_ptr,
    TOPK: tl.constexpr,
    SEGS: tl.constexpr,
    SLOTS: tl.constexpr,
    SEG_LEN: tl.constexpr,
):
    # One CTA per row: merge the SEGS proposal streams with a 4-way
    # exhaustion merge, emit TOPK indices.  SEGS <= 4 merges directly;
    # 5..8 first merge ways 0-3 into scratch slot SA (props slot SEGS),
    # then SA + ways 4..SEGS-1 into the front of slot 0, which is
    # unpacked from there.  Per-way lengths are clamped to the row's real
    # extent: partial segments contribute only their real proposals (the
    # sort pads with -inf) and empty segments (seg_sort early-exits
    # without writing) contribute nothing — their stream memory is never
    # consumed.  SEGS is capped at 8 (the host picks SEG_LEN accordingly):
    # a three-level variant faulted (MTE invalid GM) for SEGS == 9 at
    # rows >= 41 (measured) — a backend issue we route around, not an
    # addressing bug (SEGS 8/10 were clean at every probed batch size).
    row_id = tl.program_id(0)
    row_len = tl.load(row_ends + row_id) - tl.load(row_starts + row_id)
    if row_len <= TOPK:
        # Trivial row: every element wins, emit 0..row_len-1
        # (row_start-relative), padded with -1.
        lanes = tl.arange(0, TOPK)
        tl.store(
            out_indices_ptr + row_id * TOPK + lanes,
            tl.where(lanes < row_len, lanes, -1),
        )
        return
    base = props_ptr + row_id * SLOTS * 2 * TOPK
    in_ub = tle.dsa.alloc(
        [4 * 2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    in_t = tle.dsa.to_tensor(in_ub)
    if SEGS <= 4:
        for w in tl.static_range(0, 4):
            if w < SEGS:
                view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
                tle.dsa.copy(
                    base + w * 2 * TOPK + tl.arange(0, 2 * TOPK), view, [2 * TOPK]
                )
        l0 = tl.minimum(TOPK, tl.maximum(row_len, 0))
        l1 = tl.minimum(TOPK, tl.maximum(row_len - SEG_LEN, 0)) if SEGS > 1 else 0
        l2 = tl.minimum(TOPK, tl.maximum(row_len - 2 * SEG_LEN, 0)) if SEGS > 2 else 0
        l3 = tl.minimum(TOPK, tl.maximum(row_len - 3 * SEG_LEN, 0)) if SEGS > 3 else 0
        _merge4_exhaust(in_t, base, SEGS, l0, l1, l2, l3, TOPK)
    if SEGS > 4:
        # Stage A: ways 0..3 -> SA (props slot SEGS).
        for w in tl.static_range(0, 4):
            view = tle.dsa.subview(in_ub, [w * 2 * TOPK], [2 * TOPK], [1])
            tle.dsa.copy(base + w * 2 * TOPK + tl.arange(0, 2 * TOPK), view, [2 * TOPK])
        l0 = tl.minimum(TOPK, tl.maximum(row_len, 0))
        l1 = tl.minimum(TOPK, tl.maximum(row_len - SEG_LEN, 0))
        l2 = tl.minimum(TOPK, tl.maximum(row_len - 2 * SEG_LEN, 0))
        l3 = tl.minimum(TOPK, tl.maximum(row_len - 3 * SEG_LEN, 0))
        got_a = _merge4_exhaust(in_t, base + SEGS * 2 * TOPK, 4, l0, l1, l2, l3, TOPK)
        # Final: SA + ways 4..SEGS-1.
        view = tle.dsa.subview(in_ub, [0], [2 * TOPK], [1])
        tle.dsa.copy(base + SEGS * 2 * TOPK + tl.arange(0, 2 * TOPK), view, [2 * TOPK])
        for w in tl.static_range(4, 8):
            if w < SEGS:
                view = tle.dsa.subview(in_ub, [(w - 3) * 2 * TOPK], [2 * TOPK], [1])
                tle.dsa.copy(
                    base + w * 2 * TOPK + tl.arange(0, 2 * TOPK), view, [2 * TOPK]
                )
        l1 = tl.minimum(TOPK, tl.maximum(row_len - 4 * SEG_LEN, 0)) if SEGS > 4 else 0
        l2 = tl.minimum(TOPK, tl.maximum(row_len - 5 * SEG_LEN, 0)) if SEGS > 5 else 0
        l3 = tl.minimum(TOPK, tl.maximum(row_len - 6 * SEG_LEN, 0)) if SEGS > 6 else 0
        _merge4_exhaust(in_t, base, 1 + (SEGS - 4), got_a, l1, l2, l3, TOPK)
    s_ub = tle.dsa.alloc([2 * TOPK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(base + tl.arange(0, 2 * TOPK), s_ub, [2 * TOPK])
    dval = tl.zeros([TOPK], dtype=tl.float32)
    didx = tl.zeros([TOPK], dtype=tl.int32)
    dval, didx = tle.dsa.ascend.raw(
        "unpack_sort", tle.dsa.to_tensor(s_ub), TOPK, out=[dval, didx]
    )
    tl.store(out_indices_ptr + row_id * TOPK + tl.arange(0, TOPK), didx)


@triton.jit
def _ascend_topk_densify_kernel(
    src_ptr,
    dst_ptr,
    stride0,
    stride1,
    vocab_size,
    BLOCK: tl.constexpr,
):
    # Fallback for stride1 != 1 (parity with the generic implementation's
    # general strided path): gather the strided rows into a contiguous
    # buffer, then the main path runs with stride0=vocab_size, stride1=1.
    row_id = tl.program_id(0)
    cid = tl.program_id(1)
    offs = cid * BLOCK + tl.arange(0, BLOCK)
    m = offs < vocab_size
    x = tl.load(src_ptr + row_id * stride0 + offs * stride1, mask=m)
    tl.store(dst_ptr + row_id * vocab_size + offs, x, mask=m)


def _mask_geometry(vocab_size):
    """SB subtile-count geometry for the mask/count buffers.

    SB must exactly tile the 1024-wide scan blocks and stay a power of two
    for the reduce kernel's split+add trees: SB = TB * SUBTILES_PER_BLOCK
    with TB = max(next_pow2(cdiv(vocab_size, SCAN_BLOCK)), TB_MIN), which
    is always >= next_pow2(cdiv(vocab_size, SUB)).  Subtiles beyond the
    row length are tolerated everywhere: the reduce kernel masks their
    garbage counts out of the tree sums.  The TB floor keeps SB in the
    geometry range the pipeline is validated against.
    """
    tb = triton.next_power_of_2(triton.cdiv(vocab_size, SCAN_BLOCK))
    tb = max(tb, TB_MIN)
    return tb * SUBTILES_PER_BLOCK


def _pack_blocks_per_row(num_rows):
    # Blocks per row in the pack kernel: keep rows * C near the 40 vector
    # cores without over-splitting when rows already fill the device.
    if num_rows >= 128:
        return 1
    if num_rows >= 64:
        return 4
    if num_rows >= 16:
        return 8
    if num_rows == 1:
        return 32  # fill all 40 vector cores with a single row's chunks
    return 16


def top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for prefill phase of DeepSeek V4 sparse attention.

    Ascend-specialized implementation (sampled threshold + bit-packed gather
    extraction + register-resident exact candidate refinement). Selects top_k
    indices from each row of logits within [row_start, row_end); output
    indices are 0-based relative to row_starts[i], padded with -1.
    """
    logger.debug("GEMS_ASCEND TOP_K_PER_ROW_PREFILL")
    _top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
    )


def _top_k_per_row_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    # Log-free body: decode forwards here after emitting its own log line.
    vocab_size = logits.shape[1]
    assert num_rows == logits.shape[0]
    if not _TLE_IMPORT_OK:
        raise NotImplementedError(
            "ascend top_k_per_row_prefill requires TLE custom ops "
            "(triton.experimental.tle)"
        )
    if stride1 != 1:
        # The pack/sample kernels use unmasked dense row loads (address
        # clamping defeats load vectorization on this backend), so the hot
        # path requires contiguous rows; densify strided inputs instead.
        dense = torch.empty(
            (num_rows, vocab_size), device=logits.device, dtype=logits.dtype
        )
        _ascend_topk_densify_kernel[(num_rows, triton.cdiv(vocab_size, DENSIFY_BLOCK))](
            logits, dense, stride0, stride1, vocab_size, BLOCK=DENSIFY_BLOCK
        )
        logits = dense
        stride0 = vocab_size
        stride1 = 1
    # Per-row routes (trivial / select_all / sampled) are resolved inside
    # the threshold kernel from row_len; one pipeline launch serves the
    # whole batch.  Fixup is a device-side sub-route of sampled.
    # Narrow rows with k == ROWSORT_K take the row-sort route instead: one
    # exact-width hardware sort per row beats the sampled pipeline by 2-7x
    # there (measured), at any batch size.  Medium rows (up to 8 fixed-width
    # segments) in small, latency-bound batches take the seg-sort route:
    # 2 launches beat the 7-launch threshold pipeline until the throughput
    # crossover (~64 rows, probe).  Only the BASE sort impl is used: the
    # 4096-wide K-pruned impl measures ~100x slower (msprof), and vocabs
    # beyond the UB/slot budgets take the threshold pipeline instead.
    if top_k == ROWSORT_K:
        if vocab_size <= ROWSORT_MAX_VOCAB:
            _run_rowsort_route(
                logits,
                row_starts,
                row_ends,
                indices,
                num_rows,
                stride0,
                top_k,
                vocab_size,
            )
            return
        if vocab_size <= SEGSORT_MAX_VOCAB and num_rows <= SEGSORT_MAX_ROWS:
            _run_segsort_route(
                logits,
                row_starts,
                row_ends,
                indices,
                num_rows,
                stride0,
                top_k,
                vocab_size,
            )
            return
    if top_k > CHUNK_K:
        _run_chunked_pipeline(
            logits,
            row_starts,
            row_ends,
            indices,
            num_rows,
            stride0,
            top_k,
            vocab_size,
        )
        return
    _run_threshold_pipeline(
        logits, row_starts, row_ends, indices, num_rows, stride0, top_k, vocab_size
    )


def _run_rowsort_route(
    logits, row_starts, row_ends, indices, num_rows, stride0, top_k, vocab_size
):
    bufw = max(2048, triton.next_power_of_2(vocab_size))
    pad_rows = min(2, num_rows)
    pad = torch.empty((pad_rows, bufw), device=logits.device, dtype=torch.float32)
    _ascend_topk_row_sort_pad_kernel[(pad_rows,)](
        logits,
        row_starts,
        row_ends,
        pad,
        stride0,
        num_rows,
        pad_rows,
        TOPK=top_k,
        BUFW=bufw,
    )
    _ascend_topk_row_sort_kernel[(num_rows,)](
        logits,
        row_starts,
        row_ends,
        pad,
        indices,
        stride0,
        num_rows,
        pad_rows,
        TOPK=top_k,
        VOCAB=vocab_size,
        BUFW=bufw,
        TMP_SZ=triton.cdiv(vocab_size, 32) * 32 * SORT_TMP_MUL,
        SORT_IMPL=_SORT_IMPL_BASE,
    )


def _run_segsort_route(
    logits, row_starts, row_ends, indices, num_rows, stride0, top_k, vocab_size
):
    # SEG_LEN keeps the segment count at <= 8 (two merge levels max) and
    # pow2 (tl.arange): 2048-wide sorts are the sort_1d_pack sweet spot;
    # wider vocabularies switch to 4096-wide segments.
    seg_len = SEGSORT_SEG if vocab_size <= 8 * SEGSORT_SEG else 2 * SEGSORT_SEG
    segs = triton.cdiv(vocab_size, seg_len)
    # One scratch slot (SA) appended after the raw ways for the second
    # merge level when SEGS > 4.
    slots = segs + (1 if segs > 4 else 0)
    padw = segs * seg_len
    pad_rows = min(2, num_rows)
    pad = torch.empty((pad_rows, padw), device=logits.device, dtype=torch.float32)
    props = torch.empty(
        (num_rows, slots, 2 * top_k), device=logits.device, dtype=torch.float32
    )
    _ascend_topk_seg_pad_kernel[(pad_rows, triton.cdiv(padw, 4096))](
        logits,
        row_starts,
        row_ends,
        pad,
        stride0,
        num_rows,
        pad_rows,
        PADW=padw,
        BLOCK=4096,
    )
    _ascend_topk_seg_sort_kernel[(num_rows * segs,)](
        logits,
        row_starts,
        row_ends,
        pad,
        props,
        stride0,
        num_rows,
        pad_rows,
        TOPK=top_k,
        SEGS=segs,
        SLOTS=slots,
        SEG_LEN=seg_len,
        PADW=padw,
        TMP_SZ=seg_len * SORT_TMP_MUL,
        SORT_IMPL=_SORT_IMPL_BASE,
    )
    _ascend_topk_seg_merge_kernel[(num_rows,)](
        row_starts,
        row_ends,
        props,
        indices,
        TOPK=top_k,
        SEGS=segs,
        SLOTS=slots,
        SEG_LEN=seg_len,
    )


def _run_threshold_pipeline(
    logits,
    row_starts,
    row_ends,
    indices,
    num_rows,
    stride0,
    top_k,
    vocab_size,
    emit_off=0,  # elements earlier chunk rounds emitted (0 unless chunked)
    out_stride=None,  # out_indices row stride (defaults to top_k)
    out_off=0,  # this round's column offset into out_indices
    hi_state=None,  # [num_rows, 4] int32 chunk state (None unless chunked)
):
    device = logits.device
    if out_stride is None:
        out_stride = top_k
    hi = 1 if hi_state is not None else 0
    sb = _mask_geometry(vocab_size)
    pack_c = _pack_blocks_per_row(num_rows)
    # Full CAP: the wide [TOPK, CAP] route window absorbs the sampler's
    # spread, so the fixup path stays off on random data (sort_final on
    # 4096 costs ~3us more per row, a fixup row costs ~3 read passes).
    cap = CAP
    # +1 PATW of slack: the compact kernel's masked pattern load computes
    # addresses up to one padded segment past the row end (masked-off
    # lanes still emit addresses on this backend).
    masks = torch.empty(
        (num_rows * sb * MASK_BLK.value + COMPACT_SEG // SUB * MASK_BLK.value,),
        device=device,
        dtype=torch.int32,
    )
    cnts = torch.empty((num_rows, sb), device=device, dtype=torch.int32)
    cand_logits = torch.empty((num_rows, cap), device=device, dtype=torch.float32)
    cand_idx = torch.empty((num_rows, cap), device=device, dtype=torch.int32)
    totals = torch.empty((num_rows, 8), device=device, dtype=torch.int32)
    fixup_flag = torch.empty((1,), device=device, dtype=torch.int32)
    hi_arg = hi_state if hi_state is not None else totals  # dummy when HI=0
    grid_rows = (num_rows,)
    grid_rc = (num_rows * pack_c,)
    # Sampler: SAMP x SCAN_BLOCK elements into a SAMPLE_SORT_OUT-wide
    # hardware sort.  Small batches use 2 tiles: the wider sample's
    # quality margin only matters at scale (it keeps the per-row fixup
    # rate ~1e-5; at 2 tiles it is ~1e-3 but a fixup row costs little
    # when few rows are in flight), and the halved DMA + sort input is
    # worth ~4us/row at rows=1.  Tiles beyond n_samp are filled with
    # -inf (see kernel); rows within CAP never reach the sampler at all
    # (degenerate select-all threshold).
    samp = 2 if num_rows <= 8 else SAMPLE_TILES
    _ascend_topk_threshold_kernel[grid_rows](
        logits,
        indices,
        row_starts,
        row_ends,
        stride0,
        totals,
        fixup_flag,
        _get_neginf_tile(device),
        hi_arg,
        out_stride,
        out_off,
        emit_off,
        TOPK=top_k,
        SCAN_BLOCK=SCAN_BLOCK,
        CAP=cap,
        SAMP=samp,
        KSAMP=SAMPLE_SORT_OUT,
        TMP=samp * SCAN_BLOCK * SORT_TMP_MUL,
        HI=hi,
    )
    _ascend_topk_pack_kernel[grid_rc](
        logits,
        row_starts,
        row_ends,
        stride0,
        masks,
        cnts,
        totals,
        TOPK=top_k,
        PB=PACK_BLOCK,
        SB=sb,
        SUB=SUB,
        C=pack_c,
        hi_ptr=hi_arg,
        REPACK=0,
        HI=hi,
    )
    if stride0 >= PACK_BLOCK:
        # Dense-window tail (vectorized) + the final partial subtile; the
        # window stays inside the row's stride for every row, last included.
        _ascend_topk_pack_tail_dense_kernel[grid_rows](
            logits,
            row_starts,
            row_ends,
            stride0,
            masks,
            cnts,
            totals,
            PB=PACK_BLOCK,
            SB=sb,
            SUB=SUB,
            hi_ptr=hi_arg,
            REPACK=0,
            HI=hi,
        )
        _ascend_topk_pack_tail_sub_kernel[grid_rows](
            logits,
            row_starts,
            row_ends,
            stride0,
            masks,
            cnts,
            totals,
            PB=PACK_BLOCK,
            SB=sb,
            SUB=SUB,
            hi_ptr=hi_arg,
            REPACK=0,
            HI=hi,
        )
    else:
        # Tiny vocabs: the dense window could leave the row's stride; use
        # the whole-tail scalar form instead.
        _ascend_topk_pack_tail_kernel[grid_rows](
            logits,
            row_starts,
            row_ends,
            stride0,
            masks,
            cnts,
            totals,
            PB=PACK_BLOCK,
            SB=sb,
            SUB=SUB,
            hi_ptr=hi_arg,
            REPACK=0,
            HI=hi,
        )
    seg_base = torch.empty(
        (num_rows, sb // (COMPACT_SEG // SUB)), device=device, dtype=torch.int32
    )
    _ascend_topk_reduce_seg_kernel[grid_rows](
        row_starts,
        row_ends,
        totals,
        fixup_flag,
        cnts,
        seg_base,
        TOPK=top_k,
        SB=sb,
        SUB=SUB,
        CSEG_SUB=COMPACT_SEG // SUB,
        CAP=cap,
        LOGW_SB=sb.bit_length() - 1,
        REPACK=0,
    )
    # The fixup kernel only runs for rows where BOTH threshold survivor
    # counts missed the [TOPK, CAP] window (degenerate distributions).
    # Checking on the host saves a kernel launch (~80us on this stack) in
    # the common case.  The flag is pre-reduced on device (atomic_max in
    # the reduce kernel), so this is one 4B D2H read with no extra reduce
    # kernel launch.
    need_fixup = bool(fixup_flag.item())
    # Raw custom-op path: vgather-based compaction + hardware sort.
    nseg = triton.cdiv(vocab_size, COMPACT_SEG)
    _ascend_topk_compact_seg_kernel[(num_rows * nseg,)](
        logits,
        row_starts,
        row_ends,
        stride0,
        masks,
        seg_base,
        cand_logits,
        cand_idx,
        totals,
        NSEG=nseg,
        SB=sb,
        CSEG=COMPACT_SEG,
        CSEG_SUB=COMPACT_SEG // SUB,
        SUB=SUB,
        TOPK=top_k,
        CAP_SEG=cap,
        REPACK=0,
    )
    _ascend_topk_sort_final_kernel[(num_rows,)](
        indices,
        cand_logits,
        cand_idx,
        totals,
        out_stride,
        out_off,
        TOPK=top_k,
        CAP=cap,
        TMP=cap * SORT_TMP_MUL,
        REPACK=0,
    )
    if need_fixup:
        # Route-2 rows only (both thresholds missed the window): massive
        # ties get exact tie-fill; the realistic overshoot case on
        # continuous data is RE-ROUTED (route=3): the bracket threshold
        # lands inside [TOPK, CAP], so a second pack/reduce/compact/sort
        # pass over just those rows replaces the ~34-pass scalar descent.
        _ascend_topk_fixup_kernel[grid_rows](
            logits,
            indices,
            row_starts,
            row_ends,
            stride0,
            totals,
            fixup_flag,
            hi_arg,
            out_stride,
            out_off,
            TOPK=top_k,
            SCAN_BLOCK=SCAN_BLOCK,
            CAP=cap,
            FINAL=0,
            HI=hi,
        )
        if int(fixup_flag.item()) == 2:
            # Re-routed rows: re-run the vectorized pipeline gated on
            # route==3, then a final fixup pass (FINAL=1) descends any
            # row that missed the window again (~never).
            _ascend_topk_pack_kernel[grid_rc](
                logits,
                row_starts,
                row_ends,
                stride0,
                masks,
                cnts,
                totals,
                TOPK=top_k,
                PB=PACK_BLOCK,
                SB=sb,
                SUB=SUB,
                C=pack_c,
                hi_ptr=hi_arg,
                REPACK=1,
                HI=hi,
            )
            if stride0 >= PACK_BLOCK:
                _ascend_topk_pack_tail_dense_kernel[grid_rows](
                    logits,
                    row_starts,
                    row_ends,
                    stride0,
                    masks,
                    cnts,
                    totals,
                    PB=PACK_BLOCK,
                    SB=sb,
                    SUB=SUB,
                    hi_ptr=hi_arg,
                    REPACK=1,
                    HI=hi,
                )
                _ascend_topk_pack_tail_sub_kernel[grid_rows](
                    logits,
                    row_starts,
                    row_ends,
                    stride0,
                    masks,
                    cnts,
                    totals,
                    PB=PACK_BLOCK,
                    SB=sb,
                    SUB=SUB,
                    hi_ptr=hi_arg,
                    REPACK=1,
                    HI=hi,
                )
            else:
                _ascend_topk_pack_tail_kernel[grid_rows](
                    logits,
                    row_starts,
                    row_ends,
                    stride0,
                    masks,
                    cnts,
                    totals,
                    PB=PACK_BLOCK,
                    SB=sb,
                    SUB=SUB,
                    hi_ptr=hi_arg,
                    REPACK=1,
                    HI=hi,
                )
            _ascend_topk_reduce_seg_kernel[grid_rows](
                row_starts,
                row_ends,
                totals,
                fixup_flag,
                cnts,
                seg_base,
                TOPK=top_k,
                SB=sb,
                SUB=SUB,
                CSEG_SUB=COMPACT_SEG // SUB,
                CAP=cap,
                LOGW_SB=sb.bit_length() - 1,
                REPACK=1,
            )
            _ascend_topk_compact_seg_kernel[(num_rows * nseg,)](
                logits,
                row_starts,
                row_ends,
                stride0,
                masks,
                seg_base,
                cand_logits,
                cand_idx,
                totals,
                NSEG=nseg,
                SB=sb,
                CSEG=COMPACT_SEG,
                CSEG_SUB=COMPACT_SEG // SUB,
                SUB=SUB,
                TOPK=top_k,
                CAP_SEG=cap,
                REPACK=1,
            )
            _ascend_topk_sort_final_kernel[(num_rows,)](
                indices,
                cand_logits,
                cand_idx,
                totals,
                out_stride,
                out_off,
                TOPK=top_k,
                CAP=cap,
                TMP=cap * SORT_TMP_MUL,
                REPACK=1,
            )
            _ascend_topk_fixup_kernel[grid_rows](
                logits,
                indices,
                row_starts,
                row_ends,
                stride0,
                totals,
                fixup_flag,
                hi_arg,
                out_stride,
                out_off,
                TOPK=top_k,
                SCAN_BLOCK=SCAN_BLOCK,
                CAP=cap,
                FINAL=1,
                HI=hi,
            )


def _run_chunked_pipeline(
    logits, row_starts, row_ends, indices, num_rows, stride0, top_k, vocab_size
):
    # top_k > CHUNK_K: ceil(top_k / CHUNK_K) consecutive rounds of the
    # threshold pipeline, each emitting up to CHUNK_K indices at its
    # column offset.  Between rounds the boundary kernel rewrites the
    # round's tie suffix into ascending-position order and records the
    # lexicographic (value, position) bound the next round's HI filter
    # excludes by — a total order, so ties can neither duplicate nor
    # lose elements across rounds however degenerate the data is.
    # Non-short rows always have row_len - r*CHUNK_K > k_r pool elements
    # left in round r (row_len > top_k), so every round emits exactly
    # k_r real indices and the boundary value is always well-defined.
    device = logits.device
    state = torch.empty((num_rows, 4), device=device, dtype=torch.int32)
    _ascend_topk_chunk_short_kernel[(num_rows,)](
        indices, row_starts, row_ends, state, top_k, top_k, BLOCK=CHUNK_K
    )
    n_rounds = triton.cdiv(top_k, CHUNK_K)
    for r in range(n_rounds):
        k_r = min(CHUNK_K, top_k - r * CHUNK_K)
        out_off = r * CHUNK_K
        _run_threshold_pipeline(
            logits,
            row_starts,
            row_ends,
            indices,
            num_rows,
            stride0,
            k_r,
            vocab_size,
            emit_off=out_off,
            out_stride=top_k,
            out_off=out_off,
            hi_state=state,
        )
        if r + 1 < n_rounds:
            _ascend_topk_chunk_boundary_kernel[(num_rows,)](
                logits,
                indices,
                row_starts,
                row_ends,
                stride0,
                state,
                top_k,
                out_off,
                SCAN_BLOCK=SCAN_BLOCK,
                CHUNK=CHUNK_K,
            )
