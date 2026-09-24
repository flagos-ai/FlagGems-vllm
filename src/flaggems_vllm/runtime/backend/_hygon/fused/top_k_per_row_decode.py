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

"""top_k_per_row_decode on Hygon BW1000: one pass over the logits, not two.

The merge only needs a SUPERSET of each row's top-k, so the threshold can come
from a sample of the row instead of a full histogram pass. Three kernels then
do in ~1.02 passes what the generic radix algorithm does in two.

Dispatch, buffer limits and the environment switch are documented on
`top_k_per_row_decode` at the bottom of this file.
"""

import functools
import os
import threading

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.top_k_per_row_decode import NUM_BINS
from flaggems_vllm.ops.top_k_per_row_decode import _convert_to_trt_uint16_hi11 as _key
from flaggems_vllm.ops.top_k_per_row_decode import _convert_to_uint32 as _key32
from flaggems_vllm.ops.top_k_per_row_decode import (
    top_k_per_row_decode as _generic_decode,
)

# The key is the operator's own STEP-0 key, imported rather than copied so the
# two cannot drift apart: fp16 bits mapped so ascending uint16 means descending
# float, then the top 11 bits. NUM_BINS is 2048 == 1 << 11.

BLOCK = 512
WARPS = 8
SAMPLE_TILES = 8  # tiles the sample reads, whatever the row length
SAFETY = 4  # admit about SAFETY * top_k elements
CAP_FACTOR = 16  # candidate buffer, as a multiple of top_k
RADIX = 256  # bins per round of the tail's exact radix
MAX_CAND = 1 << 24  # refuse shapes whose buffers would be absurd
MIN_VOCAB = 2048
MAX_TOP_K = 2048

# Programs per row for the select pass: a flat 32, since tables keyed on the
# row count over 8, 16 and 32 did no better. At or beyond one row per SM the
# rows alone fill the card, and _split_factor returns 1.
_SPLIT = 32
MIN_CHUNK = 8192  # smallest chunk worth its own program


@triton.jit
def _scan_threshold(base, target, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Lowest bin whose prefix count reaches `target`. Bin 0 holds the largest
    values, so 'at or above the threshold' means bin <= thr."""
    lane = tl.arange(0, BLOCK)
    carry = tl.zeros([], tl.int32)
    thr = tl.full([], NB - 1, tl.int32)
    found = tl.full([], False, tl.int1)
    for t in tl.static_range(NB // BLOCK):
        bins = t * BLOCK + lane
        c = tl.load(base + bins)
        pre = carry + tl.cumsum(c, axis=0)
        hit = (pre >= target) & (not found)
        cand = tl.min(tl.where(hit, bins, NB - 1), axis=0)
        if (not found) & (tl.max(hit.to(tl.int32), axis=0) > 0):
            thr = cand
            found = tl.full([], True, tl.int1)
        carry += tl.sum(c, axis=0)
    return thr


@triton.jit
def _hist_total(base, NB: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    total = tl.zeros([], tl.int32)
    for t in tl.static_range(NB // BLOCK):
        total += tl.sum(tl.load(base + t * BLOCK + lane), axis=0)
    return total


@triton.jit
def _hist_pass(
    logits_ptr, base, row, stride0, n, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    """Histogram every STRIDE-th tile of the row into `base` (STRIDE 1: all of
    it). Whole tiles, not strided elements: a strided element sample touches
    one cache line per value and so costs a full pass for a fraction of the
    data."""
    lane = tl.arange(0, BLOCK)
    for t in tl.range(0, tl.cdiv(n, BLOCK * STRIDE)):
        i = t * BLOCK * STRIDE + lane
        m = i < n
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        tl.atomic_add(
            base + _key(x),
            tl.full([BLOCK], 1, tl.int32),
            mask=m,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _prepare(
    logits_ptr,
    seq_lens_ptr,
    hist_ptr,
    thr_ptr,
    cnt_ptr,
    stride0,
    TOPK: tl.constexpr,
    SAFETY: tl.constexpr,
    NB: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero, sample and threshold, in one program per row.

    One program owns the row's histogram, so tl.debug_barrier() is all the
    ordering the three phases need and the atomics stay inside one CTA. The
    admit rank comes from the sample actually taken, not from STRIDE, so a row
    far shorter than the vocabulary still gets a usable estimate.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for t in tl.static_range(NB // BLOCK):
        tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
    tl.store(cnt_ptr + row, 0)
    tl.debug_barrier()
    n = tl.load(seq_lens_ptr + row)
    _hist_pass(logits_ptr, base, row, stride0, n, STRIDE, BLOCK)
    tl.debug_barrier()
    total = _hist_total(base, NB, BLOCK)
    target = tl.maximum(tl.cdiv(TOPK * total, tl.maximum(n, 1)), 1) * SAFETY
    tl.store(thr_ptr + row, _scan_threshold(base, target, NB, BLOCK))


@triton.jit
def _select(
    logits_ptr,
    seq_lens_ptr,
    thr_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    stride0,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The one pass: append every element at or above the row's threshold bin.

    SPLIT programs share a row, so the counter they append through is written
    from several CTAs and the atomic has to be scoped to the whole device.
    """
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    thr = tl.load(thr_ptr + row)
    n = tl.load(seq_lens_ptr + row)
    start = chunk * CHUNK
    end = tl.minimum(start + CHUNK, n)
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK)):
        i = start + t * BLOCK + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        take = m & (_key(x) <= thr)
        pos = tl.atomic_add(
            cnt_ptrs,
            tl.full([BLOCK], 1, tl.int32),
            mask=take,
            sem="relaxed",
            scope="gpu",
        )
        keep = take & (pos < CAP)
        tl.store(cand_idx_ptr + row * CAP + pos, i.to(tl.int32), mask=keep)
        tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)


@triton.jit
def _tail(
    logits_ptr,
    seq_lens_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    out_ptr,
    counts_ptr,
    slot_ptr,
    stride0,
    TOPK: tl.constexpr,
    CAP: tl.constexpr,
    RADIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The fallback decision, then the exact top-k of the candidates.

    A row that admitted between TOPK and CAP candidates already holds a
    superset of its top-k. One outside that range is redone here over the
    whole row on the full 32-bit ordered key, so only exact ties can be
    dropped. Either way the answer comes from four 8-bit radix rounds over
    that key and is written straight to the output.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    bins = tl.arange(0, RADIX)
    ones = tl.full([BLOCK], 1, tl.int32)
    n = tl.load(seq_lens_ptr + row)
    c = tl.load(cnt_ptr + row)
    obase = out_ptr + row * TOPK
    cbase = counts_ptr + row * RADIX
    if (c < tl.minimum(TOPK, n)) | (c > CAP):
        # Not the sample's 11-bit key: it resolves magnitude/32, so a row in a
        # narrow band away from zero collapses into one or two bins. This path
        # has no STEP 1-3 to refine through; the 32-bit ordered key is
        # injective on distinct floats.
        rdesired = tl.zeros((), dtype=tl.uint32)
        rmask = tl.zeros((), dtype=tl.uint32)
        r_to_find = TOPK + 1
        row_tiles = tl.cdiv(n, BLOCK)
        for rdpos in tl.static_range(24, -1, -8):
            if r_to_find > 1:
                tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
                tl.debug_barrier()
                for rt in tl.range(0, row_tiles):
                    ri = rt * BLOCK + lane
                    rvalid = ri < n
                    rkey = _key32(
                        tl.load(logits_ptr + row * stride0 + ri, mask=rvalid, other=0.0)
                    )
                    rdigit = ((rkey >> rdpos) & (RADIX - 1)).to(tl.int32)
                    tl.atomic_add(
                        cbase + rdigit,
                        ones,
                        mask=rvalid & ((rkey & rmask) == rdesired),
                        sem="relaxed",
                        scope="cta",
                    )
                tl.debug_barrier()
                rcounts = tl.load(cbase + bins)
                rprefix = tl.cumsum(rcounts, axis=0) - rcounts
                rhit = (rprefix < r_to_find) & (rprefix + rcounts >= r_to_find)
                rb = tl.min(tl.where(rhit, bins, RADIX), axis=0).to(tl.int32)
                rb = tl.where(rb == RADIX, RADIX - 1, rb)
                rlt = tl.max(tl.where(bins == rb, rprefix, 0), axis=0).to(tl.int32)
                rdesired = rdesired | (rb.to(tl.uint32) << rdpos)
                rmask = rmask | (tl.full((), RADIX - 1, tl.uint32) << rdpos)
                r_to_find = r_to_find - rlt
        rthr = rdesired
        tl.store(slot_ptr + row, 0)
        tl.debug_barrier()
        rslots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
        # strictly better than the k-th, then its exact ties
        for req in tl.static_range(2):
            for rt2 in tl.range(0, row_tiles):
                ri2 = rt2 * BLOCK + lane
                rvalid2 = ri2 < n
                rkey2 = _key32(
                    tl.load(logits_ptr + row * stride0 + ri2, mask=rvalid2, other=0.0)
                )
                if req == 0:
                    rtake = rvalid2 & (rkey2 < rthr)
                else:
                    rtake = rvalid2 & (rkey2 == rthr)
                rq = tl.atomic_add(rslots, ones, mask=rtake, sem="relaxed", scope="cta")
                tl.store(obase + rq, ri2.to(tl.int32), mask=rtake & (rq < TOPK))
            tl.debug_barrier()
        # a row shorter than TOPK leaves the rest of the output padded
        rfilled = tl.load(slot_ptr + row)
        for rp in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            rj = rp * BLOCK + lane
            tl.store(obase + rj, -1, mask=(rj >= rfilled) & (rj < TOPK))
        return

    m = tl.minimum(tl.load(cnt_ptr + row), CAP)
    vbase = cand_val_ptr + row * CAP
    ibase = cand_idx_ptr + row * CAP
    tiles = tl.cdiv(m, BLOCK)

    if m <= TOPK:
        # fewer candidates than asked for: all of them go out, -1 pads
        for t in tl.static_range((TOPK + BLOCK - 1) // BLOCK):
            j = t * BLOCK + lane
            idx = tl.load(ibase + j, mask=j < m, other=-1)
            tl.store(obase + j, tl.where(j < m, idx, -1), mask=j < TOPK)
        return

    desired = tl.zeros((), tl.uint32)
    desired_mask = tl.zeros((), tl.uint32)
    k_to_find = TOPK + 1
    for digit_pos in tl.static_range(24, -1, -8):
        if k_to_find > 1:
            tl.store(cbase + bins, tl.zeros([RADIX], tl.int32))
            tl.debug_barrier()
            for t in tl.range(0, tiles):
                pos = t * BLOCK + lane
                valid = pos < m
                key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
                digit = ((key >> digit_pos) & (RADIX - 1)).to(tl.int32)
                tl.atomic_add(
                    cbase + digit,
                    ones,
                    mask=valid & ((key & desired_mask) == desired),
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()
            counts = tl.load(cbase + bins)
            prefix = tl.cumsum(counts, axis=0) - counts
            hit = (prefix < k_to_find) & (prefix + counts >= k_to_find)
            rb = tl.min(tl.where(hit, bins, RADIX), axis=0).to(tl.int32)
            rb = tl.where(rb == RADIX, RADIX - 1, rb)
            counts_lt = tl.max(tl.where(bins == rb, prefix, 0), axis=0).to(tl.int32)
            desired = desired | (rb.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX - 1, tl.uint32) << digit_pos
            )
            k_to_find = k_to_find - counts_lt

    thr_key = desired
    tl.store(slot_ptr + row, 0)
    tl.debug_barrier()
    slots = slot_ptr + row + tl.zeros([BLOCK], tl.int32)
    # everything strictly better than the k-th, then its equals; the rounds
    # above leave the first group short of TOPK and the two together at least
    # TOPK, so this lands exactly on TOPK
    for equal in tl.static_range(2):
        for t in tl.range(0, tiles):
            pos = t * BLOCK + lane
            valid = pos < m
            key = _key32(tl.load(vbase + pos, mask=valid, other=0.0))
            if equal == 0:
                take = valid & (key < thr_key)
            else:
                take = valid & (key == thr_key)
            q = tl.atomic_add(slots, ones, mask=take, sem="relaxed", scope="cta")
            idx = tl.load(ibase + pos, mask=take, other=-1)
            tl.store(obase + q, idx, mask=take & (q < TOPK))
        tl.debug_barrier()


@functools.lru_cache(maxsize=1)
def _sm_count():
    try:
        props = torch.cuda.get_device_properties(0)
        return int(getattr(props, "multi_processor_count", 0)) or 80
    except Exception:  # noqa: BLE001 - detection must never break dispatch
        return 80


def _enabled():
    return os.environ.get("FLAGGEMS_HYGON_TOPK_DECODE", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _split_factor(num_rows, vocab_size, top_k):
    """Programs per row for the select pass."""
    if num_rows >= _sm_count():
        return 1
    split = _SPLIT
    while split > 1 and (
        vocab_size % split or vocab_size // split < max(MIN_CHUNK, top_k)
    ):
        split //= 2
    return split


def _sample_stride(vocab_size):
    return max(1, vocab_size // (BLOCK * SAMPLE_TILES))


def _cap(top_k):
    return max(BLOCK, triton.next_power_of_2(top_k * CAP_FACTOR))


class _Launch:
    """One kernel of the pipeline: JIT on first use, direct afterwards."""

    __slots__ = ("jit", "grid", "grid3", "constexprs", "num_warps", "runner")

    def __init__(self, jit, grid, constexprs, num_warps):
        self.jit = jit
        self.grid = grid
        self.grid3 = tuple(grid) + (1,) * (3 - len(grid))
        self.constexprs = constexprs
        self.num_warps = num_warps
        self.runner = None

    def __call__(self, *args):
        if self.runner is not None:
            self.runner(*args, *self.constexprs.values())
            return
        ck = self.jit.run(
            *args,
            **self.constexprs,
            num_warps=self.num_warps,
            grid=self.grid,
            warmup=False,
        )
        if ck is not None:
            self.runner = ck[self.grid3]


class _Plan:
    """Buffers and launchers for one (shape, specialization). Three launches:
    sample the row for a threshold, select against it in one pass, then the
    fallback decision and the exact answer together."""

    def __init__(self, dev, dtype, num_rows, vocab, top_k, split):
        cap = _cap(top_k)
        chunk = vocab // split
        self.hist = torch.empty((num_rows, NUM_BINS), dtype=torch.int32, device=dev)
        self.thr = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cnt = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.cand_idx = torch.empty((num_rows, cap), dtype=torch.int32, device=dev)
        self.cand_val = torch.empty((num_rows, cap), dtype=dtype, device=dev)
        self.counts = torch.empty((num_rows, RADIX), dtype=torch.int32, device=dev)
        self.slot = torch.empty((num_rows,), dtype=torch.int32, device=dev)
        self.prepare = _Launch(
            _prepare,
            (num_rows,),
            {
                "TOPK": top_k,
                "SAFETY": SAFETY,
                "NB": NUM_BINS,
                "STRIDE": _sample_stride(vocab),
                "BLOCK": BLOCK,
            },
            WARPS,
        )
        self.select = _Launch(
            _select,
            (num_rows * split,),
            {"CHUNK": chunk, "SPLIT": split, "CAP": cap, "BLOCK": BLOCK},
            WARPS,
        )
        self.tail = _Launch(
            _tail,
            (num_rows,),
            {
                "TOPK": top_k,
                "CAP": cap,
                "RADIX": RADIX,
                "BLOCK": BLOCK,
            },
            WARPS,
        )

    def run(self, logits, seq_lens, indices, stride0):
        self.prepare(logits, seq_lens, self.hist, self.thr, self.cnt, stride0)
        self.select(
            logits,
            seq_lens,
            self.thr,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            stride0,
        )
        self.tail(
            logits,
            seq_lens,
            self.cnt,
            self.cand_idx,
            self.cand_val,
            indices,
            self.counts,
            self.slot,
            stride0,
        )


_PLANS = {}
_PLANS_MAX = 32
_LOCK = threading.Lock()


def _aligned(t):
    return t.data_ptr() % 16 == 0


def top_k_per_row_decode(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Top-K per row for DeepSeek V4 decode, threshold picked from a sample.

    Same contract as the generic operator. Three launches per call, all
    through the cached plan below:

      prepare  one program per row. Histograms SAMPLE_TILES tiles of the row
               and scans them for the bin holding rank top_k, scaled to the
               sample actually taken and loosened by SAFETY.
      select   _split_factor() programs per row. One pass over the row,
               appending every element at or above that bin into a per-row
               buffer of _cap(top_k) entries.
      tail     one program per row. The fallback decision, then the exact
               top-k of the candidates.

    Falls back to the generic operator -- always correct, never faster -- when
    any of these does not hold: next_n == 1, stride1 == 1, stride0 ==
    vocab_size, logits float32, seq_lens int32, vocab_size >= MIN_VOCAB, top_k
    <= min(MAX_TOP_K, vocab_size), and num_rows * _cap(top_k) <= MAX_CAND.

    Plans are cached on (device, shape, split, caller pointer alignment),
    because Triton specializes a compiled kernel on integer argument values and
    on data_ptr % 16; at most _PLANS_MAX are kept. A Triton version whose `run`
    returns no CompiledKernel falls back to ordinary JIT launches.

    FLAGGEMS_HYGON_TOPK_DECODE=0 disables the override; every call then goes
    to the generic operator.
    """
    vocab_size = logits.shape[1]
    if (
        not _enabled()
        or next_n != 1
        or stride1 != 1
        or stride0 != vocab_size
        or logits.dtype != torch.float32
        or seq_lens.dtype != torch.int32
        or vocab_size < MIN_VOCAB
        or top_k > MAX_TOP_K
        or top_k > vocab_size
        or num_rows * _cap(top_k) > MAX_CAND
    ):
        return _generic_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    split = _split_factor(num_rows, vocab_size, top_k)
    key = (
        logits.device,
        num_rows,
        vocab_size,
        top_k,
        split,
        _aligned(logits),
        _aligned(seq_lens),
        _aligned(indices),
    )
    with _LOCK:
        plan = _PLANS.get(key)
        if plan is None:
            if len(_PLANS) >= _PLANS_MAX:
                _PLANS.pop(next(iter(_PLANS)))
            plan = _PLANS[key] = _Plan(
                logits.device, logits.dtype, num_rows, vocab_size, top_k, split
            )
        plan.run(logits, seq_lens, indices, stride0)
    return indices
