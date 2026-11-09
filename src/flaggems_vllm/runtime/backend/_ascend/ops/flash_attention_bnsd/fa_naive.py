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

"""FlashAttention forward with no hand-written CV pipeline: plain `tl.dot` + `tl.max/exp/sum`.

Started as a control experiment against fa_950.py's hand-written 4-stage
Cube/Vector pipeline. It is now the faster kernel of the two on all 120 shapes
in bench_fa.py (mean ratio 1.73x; 0.483 vs 0.854 ms at B=1 Hq=32 S=4096 causal)
at cos=1.000000 throughout, and reaches 0.716x of the vendor's
aclnnFlashAttentionScoreV4 on average. So it is the real implementation, not a
probe.

Four things make it fast, and only the first is about CV scheduling:

1. **Letting the compiler schedule CV.** No `cann.scope`, no
   `sync_block_set/wait`. The compiler splits cube and vector work itself
   (`--set-cv-pipeline-mode`), which is the route all of triton-ascend's own
   attention examples take, and it beats the hand-written schedule.

2. **Host-side causal load balancing** -- worth 2.1x, the single biggest factor.
   With a `range(pid, ntiles, NUM_CORES)` grid stride and
   S/BLOCK_M == NUM_CORES == 32, every core gets a constant `mi == pid`, so
   under causal masking core 0 walks 1 KV block per tile while core 31 walks 32.
   That 32x imbalance makes the causal kernel as slow as dense and is easy to
   mistake for the kernel being slow. `build_tile_order` sorts tiles by
   descending KV length and deals them round robin
   (longest-processing-time-first).

3. **`propagate_nan=ALL` on the row max** -- worth 1.28x, see `_max_nan`. The
   default `tl.max` lowering spends 12 vector instructions on NaN handling that
   attention scores can never need. This cost is invisible in both the Triton
   source and the linalg IR; it only shows up in the emitted vector functions.

4. **`enable_dynamic_cv_pipeline=False`** -- worth 6%, see `_COMPILE_OPTS`.

Structure otherwise follows triton-ascend's `tutorials/06-fused-attention.py`:
a persistent grid over NUM_CORES, with the diagonal block peeled out so only it
pays for the causal mask. Specialised to our case: fp16, D == 64, BNSD, GQA
through `Hq % Hkv == 0`.

BLOCK_N is 128, not the 256 fa_950 prefers: the peeled diagonal block is a
second copy of the block body and N=256 then overflows UB
(`requires 2375680 bits while 2031616 bits available`).

`tl.exp` here always runs on the fp32 dot accumulator, so the fp16
`exp(a - b)` miscompile that fa_950 works around does not apply.

Measurements behind the structural choices: OPTIMIZATION_NOTES.md section 1,
reproducible via debug/dbg_naive_peel.py and debug/dbg_naive_cv.py.
"""

import torch
import triton
import triton.language as tl

DIM = 64


@triton.jit
def _max_nan(a, b):
    """`tl.maximum` with NaN propagation, which is the *cheap* lowering here.

    `tl.max`/`tl.maximum` default to `PropagateNan.NONE` -> `arith.maxnumf`,
    and the backend expands each operand into an explicit NaN flush before the
    compare: `vand 0x7fffffff; vadds -2139095040; vsmins 1; vsmaxs 0; vcmp NE;
    vsel -inf` -- 12 vector ops guarding a single `vmax`. `PropagateNan.ALL`
    lowers to `arith.maximumf`, which needs no flush.

    Worth 1.28x (0.661 -> 0.515 ms). Scores out of `tl.dot` on finite fp16
    inputs are never NaN, so the two agree bit for bit here; cos=1.000000 and
    all bench_fa.py shapes pass. See debug/dbg_maxnum.py.
    """
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _attn_fwd(Q, K, V, Out, Tiles, ntiles, sm_scale,
              sQb, sQh, sQs, sKb, sKh, sKs, sOb, sOh, sOs,
              B, Hq, S,
              GQA: tl.constexpr, NUM_CORES: tl.constexpr,
              D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              PRESCALED: tl.constexpr, CAUSAL: tl.constexpr,
              PEEL: tl.constexpr):
    nblk_m = S // BLOCK_M
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_m0 = tl.arange(0, BLOCK_M)

    # `Tiles` carries packed (hz * nblk_m + mi) in an order the host chose so
    # that a grid-strided walk gives every core a similar total KV length.
    for t in range(pid, ntiles, NUM_CORES):
        blk = tl.load(Tiles + t)
        hz = blk // nblk_m
        mi = blk % nblk_m
        b = hz // Hq
        h = hz % Hq
        hkv = h // GQA

        q_base = Q + b * sQb + h * sQh
        k_base = K + b * sKb + hkv * sKh
        v_base = V + b * sKb + hkv * sKh
        o_base = Out + b * sOb + h * sOh

        offs_m = mi * BLOCK_M + offs_m0
        q = tl.load(q_base + offs_m[:, None] * sQs + offs_d[None, :])

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # The two block bodies below are written out rather than shared through
        # a `@triton.jit` helper. The helper version is 7% slower (0.658 vs
        # 0.613 ms) even though it is supposed to inline -- passing `acc`/`l_i`/
        # `m_i` through a call boundary apparently blocks some scheduling the
        # compiler does on the straight-line form. Measured, not assumed; keep
        # them inline and keep them in sync.

        # Causal normally stops one block short and the diagonal block is
        # peeled out below (PEEL=True). With PEEL=False the diagonal block
        # stays in this loop and takes the mask under a runtime branch, the
        # way vendor does it -- see the PEEL note in flash_attention_fwd.
        diag = mi * BLOCK_N
        if CAUSAL:
            hi = diag if PEEL else diag + BLOCK_N
        else:
            hi = S
        for start_n in range(0, hi, BLOCK_N):
            kn = start_n + offs_n
            kb = tl.load(k_base + kn[:, None] * sKs + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kb))
            if not PRESCALED:
                qk = qk * sm_scale
            if CAUSAL and not PEEL:
                # Only the final block straddles the diagonal, so interior
                # blocks pay just the compare. Same shape as vendor's runtime
                # `if (needAtten)`.
                if start_n == diag:
                    # True -inf, not -1e6: see the note in fa_causal.py. This
                    # module's `flash_attention_fwd` is no longer reachable
                    # from the shipping router (it is a copy of the pre-split
                    # kernel), but the same mask bug lived here.
                    qk = tl.where(offs_m0[:, None] >= offs_n[None, :],
                                  qk, float("-inf"))
            m_ij = _max_nan(m_i, tl.reduce(qk, 1, _max_nan))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            # V is loaded after the Q@K^T dot, not alongside K: its MTE2 then
            # overlaps the cube work instead of queueing behind K's load.
            vb = tl.load(v_base + kn[:, None] * sKs + offs_d[None, :])
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(vb.dtype), vb, acc)
            m_i = m_ij

        if CAUSAL and PEEL:
            # The diagonal block as straight-line code rather than a second
            # loop. BLOCK_M == BLOCK_N makes it exactly one block carrying the
            # fixed lower-triangular pattern, so it needs no loop and no bias
            # plane table: 0.613 ms against 0.649 for a second loop.
            kn = diag + offs_n
            kb = tl.load(k_base + kn[:, None] * sKs + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kb))
            if not PRESCALED:
                qk = qk * sm_scale
            qk = tl.where(offs_m0[:, None] >= offs_n[None, :],
                          qk, float("-inf"))
            m_ij = _max_nan(m_i, tl.reduce(qk, 1, _max_nan))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            vb = tl.load(v_base + kn[:, None] * sKs + offs_d[None, :])
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(vb.dtype), vb, acc)

        acc = acc / l_i[:, None]
        tl.store(o_base + offs_m[:, None] * sOs + offs_d[None, :],
                 acc.to(Out.dtype.element_ty))


_tile_cache = {}


def check_bnsd(q, k, v):
    """Tensor-level checks that do not need S, callable *before* unpacking.

    Both entries start with `B, Hq, S, D = q.shape`, so a 3-D input would hit
    the tuple unpack first and raise `not enough values to unpack (expected 4,
    got 3)` -- true, but it names neither the argument nor the expected layout.
    """
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t)}")
        if t.ndim != 4:
            raise ValueError(
                f"{name} must be 4-D BNSD [B, H, S, D], got {t.ndim}-D shape "
                f"{tuple(t.shape)}")
        if t.dtype != torch.float16:
            raise ValueError(
                f"{name} must be fp16 (the kernels are fp16-only), got {t.dtype}")
        if t.device != q.device:
            raise ValueError(f"{name} is on {t.device} but q is on {q.device}")
        # One `sKs` is passed for both K and V, so both must be dense-contiguous.
        if not t.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous: the kernel indexes with the "
                f"caller's strides, and a non-contiguous {name} is read at "
                f"wrong offsets (silently wrong, not an error). Call "
                f".contiguous() first. shape={tuple(t.shape)} "
                f"strides={tuple(t.stride())}")
    return True


def validate_shape(q, k, v, block_m, block_n, causal=False):
    """Shape/divisibility checks. Needs `block_m`/`block_n`, so it runs after
    the tile config is resolved. Tensor-level checks live in `check_bnsd`,
    which the entries call before unpacking `q.shape`.
    """
    B, Hq, S, D = q.shape
    ks, vs = k.shape, v.shape          # .shape is a dispatch; index it once
    Hk, Hv = ks[1], vs[1]
    if D != DIM:
        raise ValueError(f"D must be {DIM}, got {D}")
    if ks[0] != B or vs[0] != B:
        raise ValueError(
            f"batch mismatch: q B={B}, k B={ks[0]}, v B={vs[0]}")
    if ks[2] != S or vs[2] != S:
        raise ValueError(
            f"asymmetric Sq/Skv is not supported: Sq={S}, Skv={ks[2]}/{vs[2]}")
    if ks[3] != D or vs[3] != D:
        raise ValueError(
            f"K/V head dim must be {D}, got {ks[3]}/{vs[3]}")
    if Hk != Hv:
        raise ValueError(f"K and V must have the same head count: {Hk} vs {Hv}")
    if Hk <= 0 or Hq % Hk:
        raise ValueError(f"Hq={Hq} must be a positive multiple of Hkv={Hk}")
    if S <= 0 or B <= 0:
        raise ValueError(f"empty input: B={B} S={S}")
    if S % block_m or S % block_n:
        raise ValueError(
            f"S={S} must be divisible by block_m={block_m} and "
            f"block_n={block_n}")
    if causal and block_m != block_n:
        raise ValueError(
            f"causal requires block_m == block_n (the diagonal block is one "
            f"square KV block), got {block_m} vs {block_n}")
    return True


def validate_inputs(q, k, v, block_m, block_n, causal=False):
    """Both halves, for callers that have not already run `check_bnsd`.

    The kernels do not use this form: they call `check_bnsd` at the top (before
    `q.shape` is unpacked) and `validate_shape` after the tile config is known,
    so no check runs twice.

    Two real bugs motivated all of this, both of which used to be silent:

    * **V stride.** Both kernels take one `sKs` and use it for K *and* V, so a
      non-contiguous V is read at K's addresses. With
      `v = randn(B, H, D, S).transpose(-1, -2)` that produced
      `max|ours - vendor| = 5.998` (contiguous V: 4.9e-04).
    * **Prescaled Q stride.** `qs = q * sm_scale` at `S >= 4096`, and the launch
      used to pass `q.stride()`. An elementwise op can return a tensor whose
      strides differ from `q`'s, in which case the kernel indexes `qs` wrongly.

    Raising rather than silently calling `.contiguous()` is deliberate: a hidden
    copy would change both the memory traffic and the measured time, making a
    perf number that no longer describes the input the caller passed.

    Cost: ~2.4 us of host work per call, of which `check_bnsd` is ~1.5 us (three
    `is_contiguous()` dispatches dominate). At S=1024 the *wall* time is
    host-bound, so that reads as a few percent there -- but the device kernel is
    untouched: profiled duration is identical with and without it
    (38.7/38.7 us at S=1024, 1360.6/1360.6 at S=8192), because host work
    overlaps the async device queue. If a latency-bound decode path ever shows
    it, gate the call site rather than weaken the checks.
    """
    check_bnsd(q, k, v)
    return validate_shape(q, k, v, block_m, block_n, causal)


def build_tile_order(B, Hq, S, block_m, is_causal, device):
    """Packed tile ids ordered so a grid-strided walk balances the cores.

    Under causal masking tile `mi` costs `mi + 1` KV blocks, so the natural
    order gives core `pid` only the tiles with `mi == pid` (mod NUM_CORES) and
    the spread is 32x. Sorting by descending cost and dealing round robin is
    longest-processing-time-first, which bounds the imbalance to one tile.

    Dense tiles all cost the same, so the order is irrelevant there and the
    natural one is used.
    """
    key = (B, Hq, S, block_m, is_causal, str(device))
    if key not in _tile_cache:
        nblk_m = S // block_m
        tiles = [(hz, mi) for hz in range(B * Hq) for mi in range(nblk_m)]
        if is_causal:
            tiles.sort(key=lambda t: -(t[1] + 1))
        packed = [hz * nblk_m + mi for hz, mi in tiles]
        _tile_cache[key] = torch.tensor(packed, dtype=torch.int32, device=device)
    return _tile_cache[key]


# On a 950 target `enable_dynamic_cv_pipeline` defaults to True. Measured 6%
# faster with it off (0.630 vs 0.669 ms at B=1 Hq=32 S=4096 causal), which is
# the opposite of what the name suggests for a kernel that wants the compiler to
# schedule CV: turning off DynamicCV does not turn off CV splitting, it only
# drops the dynamic (runtime-decided) variant.
_COMPILE_OPTS = dict(enable_dynamic_cv_pipeline=False)


# Tile configs worth trying, as (block_m, block_n). `num_cores` is deliberately
# not tuned: swept 16/32/48/64 at seven shapes and 32 or 64 always tie to within
# 0.6% while 48 is 20% worse (it does not divide the tile count evenly) and 16
# starves the device. Adding it would triple tuning cost for noise.
#
# Only two shapes of tile survive a sweep (debug/dbg_small_s_sweep.py, 4x4x4
# over M/N/cores at seven shapes):
#   128/128  the default, best or tied nearly everywhere
#   64/256   same 16384 elements but a 4x wider row, so the row max and row sum
#            reduce over 256 contiguous elements with half as many rows. Wins
#            on *large dense* only, and loses clearly at small S.
# Causal is excluded: the peeled diagonal block requires block_m == block_n, and
# the doubly-inlined loop body makes N=256 overflow UB there.
_AUTOTUNE_CONFIGS = ((128, 128), (64, 256))

# shape key -> (block_m, block_n), filled by the autotuner at run time.
_autotune_cache: dict = {}


def _autotune_dense(q, k, v, S, launch, warmup=3, rep=7):
    """Pick (block_m, block_n) for one dense shape by measuring both.

    Host-side rather than `@triton.autotune` because `Tiles` is built on the
    host and keyed to `block_m` (see `build_tile_order`): a decorator sweeping
    BLOCK_M as a `tl.constexpr` would hand the kernel a tile list built for a
    *different* block size, which is silently wrong rather than slow. Each
    config here gets its own correctly-sized tile list.

    This applies to triton-ascend's own autotuner too
    (`triton.backends.ascend.runtime.autotuner.autotune`) -- it takes the same
    `configs`/`key` decorator form over constexpr kwargs. Its `pre_hook` cannot
    rescue the case either: `ntiles` *changes with* `block_m`, so the fix needs
    a differently-sized positional tensor, not an in-place mutation of the
    existing one.

    Deliberately cheap. Measured gains are 1.03-1.07x, so a tuner that costs
    more than a few launches per shape can easily exceed what it wins; the
    result is cached per shape key, and `rep=7` with a median is enough to
    separate 1.03x from a <1% spread on an idle device.
    """
    import time as _time
    best, best_ms = None, float("inf")
    for bm, bn in _AUTOTUNE_CONFIGS:
        if S % bm or S % bn:
            continue
        try:
            for _ in range(warmup):
                launch(bm, bn)
            torch.npu.synchronize()
            lat = []
            for _ in range(rep):
                torch.npu.synchronize()
                t0 = _time.perf_counter()
                launch(bm, bn)
                torch.npu.synchronize()
                lat.append(_time.perf_counter() - t0)
            lat.sort()
            ms = lat[len(lat) // 2]
        except Exception:
            continue          # a config that will not compile is not a candidate
        if ms < best_ms:
            best, best_ms = (bm, bn), ms
    return best or (128, 128)


def flash_attention_fwd(q, k, v, is_causal=False, block_m=None, block_n=None,
                        num_cores=32, autotune=True, **opts):
    """q[B, Hq, S, D], k/v[B, Hkv, S, D], fp16, D == 64, Hq % Hkv == 0.

    `block_m`/`block_n` default to None, meaning "choose". Pass them explicitly
    to pin a config (which is what the sweep probes do, and what `autotune=False`
    is for).
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert D == DIM and Hq % Hkv == 0

    explicit = block_m is not None or block_n is not None
    if block_m is None:
        block_m = 128
    if block_n is None:
        block_n = 128
    # Autotune only where the sweep found a real choice: dense, S >= 4096.
    # Causal is pinned by the peeled-diagonal constraint (block_m == block_n)
    # plus UB overflow at N=256, so there is nothing to choose; and at small S
    # every alternative measured worse, so tuning would only cost launches.
    if (autotune and not explicit and not is_causal and S >= 4096
            and len(_AUTOTUNE_CONFIGS) > 1):
        key = (B, Hq, Hkv, S, D, str(q.device))
        if key not in _autotune_cache:
            def _probe(bm, bn):
                return flash_attention_fwd(q, k, v, is_causal=False,
                                           block_m=bm, block_n=bn,
                                           num_cores=num_cores, **opts)
            _autotune_cache[key] = _autotune_dense(q, k, v, S, _probe)
        block_m, block_n = _autotune_cache[key]

    assert S % block_m == 0 and S % block_n == 0
    # The peeled diagonal block assumes it is exactly one KV block.
    assert (not is_causal) or block_m == block_n, "causal requires block_m == block_n"
    out = torch.empty_like(q)
    tiles = build_tile_order(B, Hq, S, block_m, is_causal, q.device)
    # Fold the softmax scale into Q instead of scaling each score tile in the
    # kernel. Vector cycles over the BLOCK_M x BLOCK_N score tile are what bound
    # this kernel and one full-tile pass costs ~0.05 ms at S=4096 (calibrated in
    # debug/dbg_vec_passes.py); scaling Q touches 1/BLOCK_N as many elements.
    #
    # Do it **in fp16**. The obvious `(q.float() * sm).to(q.dtype)` makes three
    # passes over Q (widen, multiply, narrow) and at S=4096 that launch is
    # ~55 us, not the ~9 us an earlier note here claimed -- 13% of the kernel.
    # `q * sm_scale` is one pass and is **bit-identical**, not merely close:
    #   D=64  -> sm_scale == 0.125 == 2^-3 exactly, so this is an exponent
    #            shift and every finite input is exact (subnormals included,
    #            since scaling down by 2^-3 cannot round in fp16's gradual
    #            underflow range)
    #   D=128 -> 2^-3.5 is not a power of two, but double rounding through
    #            fp32 is still innocuous because 24 >= 2*11 + 2
    # Verified bit-identical by comparing int16 views, including a
    # subnormal-range input set (debug/dbg_prescale_cost.py).
    #
    # Measured interleaved, 6 blocks x median-of-15 (a single median is not
    # enough at this effect size -- that is how `enable_preload` faked +2.4%):
    #   S=4096 causal 0.476 -> 0.443  1.073x   S=4096 dense 0.734 -> 0.703
    #   S=8192 causal 1.464 -> 1.417  1.034x   S=2048       0.270 -> 0.239
    #   S=1024 (B8)   0.245 -> 0.217  1.130x   S=16384      within spread
    #
    # The `S >= 4096` gate stays. Making the prescale 3x cheaper does not move
    # the break-even, because below it the competitor is not the expensive
    # prescale -- it is **no prescale at all** (`PRESCALED=False`, scaling the
    # score tile in-kernel), and that has no launch to pay for. Three-way, from
    # debug/dbg_prescale_cost.py (f32pre / f16pre / in-kernel, ms):
    #   S=8192 causal  1.473 / 1.428 / 1.584   -> prescale
    #   S=4096 causal  0.470 / 0.441 / 0.466   -> prescale
    #   S=4096 dense   0.728 / 0.700 / 0.784   -> prescale
    #   S=2048 (B2)    0.262 / 0.236 / 0.233   -> in-kernel, by ~1%
    #   S=1024 (B8)    0.238 / 0.212 / 0.204   -> in-kernel
    #   S=1024 (B1)    0.132 / 0.103 / 0.087   -> in-kernel, clearly
    # I briefly removed the gate on the strength of the fp16-vs-fp32 numbers
    # alone and it cost 0.218 -> 0.231 at S=1024 and 0.248 -> 0.254 at S=2048:
    # that comparison held PRESCALED=True on both sides, so it never priced the
    # path small-S actually takes.
    #
    # Scaling the Q tile *inside* the kernel is a third option and is worse than
    # both (0.476 at S=4096 causal, 0.776 dense): it re-reads and re-scales Q
    # once per tile instead of once per launch. In fp32 it breaks tl.dot's
    # same-dtype rule.
    sm_scale = (1.0 / D) ** 0.5
    prescale = S >= 4096
    qs = q * sm_scale if prescale else q

    # Peel the causal diagonal block out of the KV loop only at large S.
    #
    # The peeled block is straight-line code sitting at the *per-tile* level of
    # the loop nest (IR ind=6, next to the Q load), so it is charged once per
    # tile and gets no overlap with the pipelined loop body. That is a good
    # trade when a tile walks many KV blocks and a bad one when it walks few.
    #
    # Round 6 measured the gap as per-tile 3.69x against per-KV-block 1.12x, so
    # this is exactly the kind of cost that dominates at small S. Vendor does
    # not peel at all: every block including the diagonal stays in the one
    # pipelined loop with a runtime mask branch.
    #
    # OPTIMIZATION_NOTES §1 recorded peeling as faster (0.613 vs 0.649 ms) --
    # true, but measured at S=4096 where per-tile cost is only ~6% of runtime.
    # Re-measured across S (debug/dbg_depeel.py, interleaved 6 blocks x
    # median-of-15, de-peeled / peeled):
    #   S=512  B16Hq16  1.118x      S=1536 B8Hq16  1.004x
    #   S=512  B8Hq16   1.081x      S=2048 B4Hq16  0.968x
    #   S=1024 B16Hq16  1.062x      S=4096 B1Hq32  0.819x
    #   S=1024 B8Hq16   1.055x      S=8192 B1Hq16  0.765x
    # Monotone crossover at S~1536, so gate at S <= 1024. cos=1.000000 on both
    # arms at every shape.
    peel = not is_causal or S > 1024
    # Overridable so A/B probes can force either arm (debug/dbg_depeel.py).
    peel = bool(opts.pop("PEEL", peel))

    launch_opts = dict(_COMPILE_OPTS)
    launch_opts.update(opts)
    _attn_fwd[(num_cores,)](
        qs, k, v, out, tiles, tiles.numel(), sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, Hq, S,
        GQA=Hq // Hkv, NUM_CORES=num_cores, D=D,
        BLOCK_M=block_m, BLOCK_N=block_n,
        PRESCALED=prescale, CAUSAL=is_causal, PEEL=peel,
        **launch_opts,
    )
    return out
