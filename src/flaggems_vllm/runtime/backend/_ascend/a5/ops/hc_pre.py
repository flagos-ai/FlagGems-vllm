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

"""HcPre (the mHC pre block) for Ascend 950DT (A5).

Replaces the vllm-ascend vendor custom op ``torch.ops._C_ascend.npu_hc_pre_v2``
(``csrc/moe/hc_pre``).  The profiler kernel name is plain ``HcPre`` with no
``aclnn`` prefix because it is not a CANN built-in: ``libopapi.so`` ships
``aclnnMhcPre`` / ``aclnnMhcPreV2`` / ``aclnnMhcPreSinkhorn`` but no
``aclnnHcPre``.  It is a vendor op that vllm-ascend compiles itself
(``OP_ADD(HcPre)``), fusing mhc_pre and sinkhorn into one kernel.  Calling the
baseline therefore needs
``vllm_ascend.utils.bootstrap_custom_op_env(include_vendor_lib=True)`` *before*
``import torch_npu``, or it raises ``aclnnHcPre or
aclnnHcPreGetWorkspaceSize not in libopapi.so``; see ``tests/test_hc_pre.py``.

Shape in the DSV4 profile (``op_list_dsv4_0731_meta.csv``, the only HcPre
variant there)::

    x        [1, 4, 4096]   bf16    # [T, n, D]; T=1 is one decode token
    hc_fn    [24, 16384]    fp32    # [n*n+2n, n*D]
    hc_scale [3]            fp32
    hc_base  [24]           fp32
    y        [1, 4096]      bf16
    post     [1, 4]         fp32
    comb     [1, 4, 4]      fp32
    avg 11.062us x 89440 calls = 989ms = 4.48% of the trace

The authoritative math is vllm's own reference,
``vllm/model_executor/kernels/mhc/torch.py::mhc_pre_torch`` -- *not* the CANN
``aclnnMhcPre`` document, which describes a different op with sinkhorn split
out::

    xf     = x.reshape(T, n*D).float()
    mixes  = (xf @ fn.T) * rsqrt(mean(xf^2, -1) + norm_eps)       # [T, 24]
    pre    = sigmoid(mixes[:, 0:n]  * s0 + base[0:n])  + hc_eps   # [T, n]
    post   = sigmoid(mixes[:, n:2n] * s1 + base[n:2n]) * 2.0      # [T, n]
    comb   = softmax(mixes[:, 2n:].view(T,n,n) * s2
                     + base[2n:].view(n,n), -1) + hc_eps
    comb   = comb / (comb.sum(-2, keepdim) + hc_eps)   # columns only, step 1
    repeat (iters-1) times:  comb /= comb.sum(-1, keepdim) + hc_eps
                             comb /= comb.sum(-2, keepdim) + hc_eps
    y      = sum_i pre[:, i] * x[:, i, :]                         # [T, D] bf16

``post_mult=2.0`` is hard-coded in the vendor op (``Muls(mix, mix, 2.0)`` at
``hc_pre_base_arch35.h:763``), not a parameter, so it is hard-coded here too.
``n`` is capped at 4, the same limit as the baseline (``HC_PRE_HC_LIMIT`` at
``torch_binding.cpp:1540``); other values raise ``ValueError``.

Dispatch on T: two independent thresholds
-----------------------------------------

The best implementation forks on T, and the fork has two *independent* axes,
each with its own crossover::

    how mix is computed    T < 16     vector       grid=(T/BT, SK)
                           T >= 16    cube + split-K, grid=(ceil(T/BT), SK)

    how the epilogue splits
                           T < 256    fused        grid=(T, SPLIT)
                           T >= 256   comb + y     grid=(ceil(T/BT),), (T,)

The axes are independent because both mix kernels write the *same*
``[T*SK, BLOCK_J]`` partial-sum layout.  Only one combination (cube mix feeding
comb+y) needs an extra ``_hc_reduce_sk_kernel`` pass in between, because comb
wants token-innermost.

Mix crossover, measured kernel-only with each side at its own best config (us):

===== ========== ============== ================
T     vector     cube+split-K   winner
===== ========== ============== ================
1     **2.63**   5.73           vector 2.18x
2     **3.79**   5.64           vector 1.49x
8     **5.69**   5.70           tie
16    8.89       **5.83**       cube 1.53x
64    17.44      **8.26**       cube 2.11x
256   62.08      **16.99**      cube 3.65x
512   117.01     **28.17**      cube 4.15x
===== ========== ============== ================

One conclusion must be read with its qualifier attached:

    "This op should not touch the cube" holds **only for T < 16**.  A kernel
    with ``tl.dot`` pays 1.3-1.5us more fixed cost at every grid size, and the
    whole op at T=1 is 7us.  From T>=16 the fixed cost amortizes and
    cube+split-K is 4.15x faster at T=512.

That table also replaces an earlier crossover of T=256, which was measured with
SK=1 only (the whole k-loop inside one program).  SK=1 cube mix is ~90us across
T=64..2048 -- 32x the work in the same time.  Those 90us are not throughput,
they are the serial chain of 64 k-loop steps inside one program.  Splitting K
moved the crossover all the way left to 16.

Small-T path (T=1 is the production case)
-----------------------------------------

At T=1 this op is dominated by fixed cost, neither bandwidth nor compute:
shrinking D from 4096 to 64 (fn 64x smaller) only takes CANN from 8.9us to
8.2us, and takes our mix kernel down to exactly the empty-kernel floor at the
same grid.  Real memory traffic accounts for 0.38us (1.03 TB/s).

So the thing being optimized is the *number of kernels and programs*, not the
access pattern.  Four decisions follow:

1. **Two kernels is the floor.**  A single kernel would need a cross-program
   reduction and Triton has no barrier; grid=1 streaming 1.57MB through one
   core takes 33-37us (single-core DMA 42-47 GB/s).
2. **SK=32** (the split-K count).  The cost model
   ``T(P) ~= floor(P) + 1.6MB/(P * 43GB/s)`` is minimal at P=32 and predicts
   2.28us; measured 2.307us.
3. **fn's 24 rows are read as two unmasked segments, 16+8.**  Padding to 32
   rows with a static mask is shorter code, but the 8 masked rows are still
   DMA'd into UB -- a static mask saves instructions, not bytes.
4. **The epilogue splits by task, SPLIT=2**: program 0 does comb+sinkhorn,
   program 1 does pre+y.  The two have no data dependency, so y's 0.93us hides
   entirely inside sinkhorn's 2.78us.

Mid/large-T path
----------------

1. **Cube mix with split-K**, ``mixes[T,J] = x[T,K] @ fn^T[K,J]``,
   grid=(ceil(T/BT), SK).  ``fn^T`` is loaded through transposed indices rather
   than pre-transposed (pre-transposing measured exactly zero benefit).
   Splitting K is the single largest win on this path (T=512: 90.26 -> 28.17us)
   and it *improves* accuracy (maxdiff 4.7e-5 -> 1.5e-5: shorter accumulation
   chains per partial).
2. **Fill BT first (cap 256), then use SK to reach 32 programs.**  The order
   matters: at the same 32 programs, T=1024 with BT=64/SK=2 takes 49.05us while
   BT=32/SK=1 takes 69.71us; the difference is how many times fn is re-read
   (16 vs 32 passes).  Both criteria were later found verbatim in the baseline
   tiling (``hc_pre_tiling_arch35.h:732``
   ``mDimNum = min(aicCoreNum, ceil(bs/M_L1_MAX_SIZE))`` and ``:734``
   ``kDimNum = aicCoreNum / mDimNum``): M first, K to fill, product equal to
   the AIC core count -- and the 32 arrived at by measurement *is* that count.
3. **The mix GEMM runs in HF32** (``input_precision="hf32"``).  The baseline
   does the same on the cube (``hc_pre_cube_compute_arch35.h:88``) while
   triton-ascend defaults to ``"ieee"``.  Sweeping parameters can never find
   this one; only reading the vendor source can.  mix 1.24-1.91x.
4. **Split-K has two boundaries.**  Beyond 32 programs it gets worse (more fn
   re-reads buy no parallelism), and from T>=1024 the limit is cube throughput
   rather than chain length.
5. **comb's sinkhorn unrolls both N axes into registers and never calls
   ``tl.sum``.**  The old ``[N,N,BT]`` tile with ``tl.sum(axis=0/1)`` cost
   402ns/step; unrolled it is 16ns/step, 25x faster.  The baseline uses exactly
   this layout (``VFProcessCombFragPremixElementMajor`` at
   ``hc_pre_base_arch35.h:1167``: 16 registers ``m00..m33``, tokens occupy
   lanes, zero cross-lane reduction).  The criterion is "reduction on a 3-D
   tile plus broadcast along the third axis is expensive", *not* "``tl.sum`` is
   expensive" -- the 2-D ``tl.sum`` in the y kernel is already the best
   lowering and unrolling it is slower.
6. **comb's BT criterion differs from mix's: fill 64 cores, not 32 programs.**
   comb reads each token's data once, so extra programs cost no extra bytes and
   it can use every core; ``BT = max(8, next_pow2(T/64))``.
7. **y gets its own kernel**, grid=(T,), ``BLOCK_D`` up to 4096.  It wants long
   contiguous runs while comb wants a large BT, and those two compete for the
   same UB budget, hence the split.
8. **y's ``sq``/``mpre`` prelude loads directly instead of reading full width
   and selecting** (T=2048: 41.45 -> 32.39us, with the numerics bit-identical).
   The criterion is "does this kernel need the full width", not "select vs
   direct load" -- y needs only 2 compile-time-known lanes, so a ``[BLOCK_J]``
   read wastes 32 lanes, while the four sections of ``_hc_epilogue_kernel``
   together do need the full width and the same select there measures even,
   with narrow direct reads 2-4x slower.

Known gaps
----------

* The y kernel's prelude still has ~11.17us unexplained at T=2048.  A bare
  accumulate at the same traffic reaches 4.51 TB/s while y reaches 2.2 TB/s, so
  half the time is compute; differencing four shapes shows the 33.5M fp32
  multiplies cost 0.30us and the rest is the per-token scalar prelude
  (5.5ns/token for one scalar load, one ``[N]`` load, ``rsqrt`` and one
  ``sigmoid`` over ``[N]``).
* ``reduce_sk`` (5.69us at T=2048) moves only 1.05MB and is almost entirely
  launch floor.  It is needed only when SK != 1 and would disappear if mix
  wrote both layouts directly.
* T=2048 is back to being mix-dominated (40.70us); T=1024 and T=2048 both work
  out to ~11 T MAC/s, so that is cube throughput, not chain length.

Accuracy
--------

Split by path: the vector path is more accurate than the baseline, the cube
path is on par with it.  Against the fp32 torch reference (``hc_pre_ref`` in
the test file), max absolute error:

================= ======== ======== ========
                  y        post     comb
================= ======== ======== ========
CANN              3.1e-2   1.4e-4   1.2e-4
ours, vector path 2.0e-3   1.5e-6   1.1e-6
ours, cube path   3.1e-2   7.0e-6   1.1e-5
================= ======== ======== ========

y on the cube path is 3.1e-2 because we enable the same HF32 switch the
baseline does (``SetHF32Mode(1)`` + ``SetHF32TransMode(1)``,
``hc_pre_cube_compute_arch35.h:88``); y is a bf16 output and at T=2048 the
magnitude of |y| is such that 3.1e-2 is one bf16 ulp.  Tests therefore use a
4e-2 tolerance, the baseline's own error level; demanding better than the op
being replaced would not be meaningful.

Measured device kernel time vs the vendor op (profiler ``task_time.csv``,
alternating in one process, 950DT, CANN 9.2.1):

======== ======== ======== ========
case     CANN     ours     ratio
======== ======== ======== ========
T=1      9.721    7.255    1.340x
T=8      10.723   10.881   0.985x
T=16     11.001   12.881   0.854x
T=64     12.717   16.458   0.773x
T=128    14.533   22.918   0.634x
T=256    20.113   28.408   0.708x
T=1000   46.276   56.288   0.822x
T=2048   81.227   89.915   0.903x
======== ======== ======== ========

Only T=1 appears in the DSV4 profile.  The prefill shapes (T>=16) are all below
the baseline; the remaining difference there is launch count, not bytes moved:
the baseline packs both stages into one launch (``hc_pre.cpp:51-63``, Part1 ->
``pipe.Destroy()`` -> new TPipe -> Part2, intermediates through ``userWs``,
aligned with ``CrossCoreSetFlag``/``CrossCoreWaitFlag``/``SyncAll<false>()``).
Both of the baseline's parts take ``xGm`` (``hc_pre.cpp:55`` and ``:61``), so it
reads x twice as well.  Triton has no cross-core barrier, so we must use
separate launches.
"""

import torch
import triton
import triton.language as tl

__all__ = ["hc_pre"]

HC_POST_MULT = 2.0  # hard-coded in the vendor op, hc_pre_base_arch35.h:763
HC_MULT_LIMIT = 4  # baseline limit too, torch_binding.cpp:1540 HC_PRE_HC_LIMIT

_BK_MAX = 1024  # fn tile is [32, BK] fp32; BK=1024 is 128KB
_BK_MIN = 256  # below this only the floor is left, not worth more programs
# Crossover of the launch floor and single-core bandwidth.  This was 16 when
# measured on a version that stored psq with its own 4-byte scatter store;
# that store alone cost 1.4us and raised the floor, pushing the optimum toward
# fewer programs.  After folding it into lane 24: SK=32 gives 2.307us against
# 3.212 (SK=16) and 3.040 (SK=64).
_SK_PREFER = 32

# ---- large-T cube path (see _hc_mix_dot_kernel / _hc_comb_kernel) ----
_BT_VEC_MAX = 8  # vector mix token tile cap: saturates past 8 (vector MAC bound)
_BK_CUBE = 256  # cube k step for BT<=64; large BT needs 128, see _pick_bk_cube
# Cube-path mix token tile cap, taken from the baseline's ``M_L1_MAX_SIZE =
# 256`` (``hc_pre_tiling_arch35.h:55``).  It used to be 64, which was an
# artifact of never sweeping higher rather than a measured limit.  Total fn
# traffic is ``ceil(T/BT) * 1.57MB``, independent of SK, so BT is the only
# knob and it should be pushed up.  Measured alternating in one process:
#
#   T     before BT/BK/SK     candidate         speedup
#   512   64/256/4   15.50    128/128/8  13.76  1.13x
#   1024  64/256/2   27.77    256/128/8  22.35  1.24x
#   2048  64/256/1   48.95    256/128/8  42.67  1.15x
#
# Much less than the 55-75us predicted from fn traffic alone (50MB -> 12.6MB),
# but consistent across three shapes and far above this machine's 27% jitter.
_BT_CUBE_MAX = 256
_BT_COMB_MIN = 8  # comb token tile floor; per-wave cost rises again at BT=4
_BT_REDUCE_MAX = 32  # token tile cap for the reduce pass
_CORES = 64  # programs that run in one wave; comb's BT is derived from it
_T_CUBE_MIN = 16  # mix vector -> cube crossover, measured
# Epilogue fused -> comb+y crossover.  It was 512 when measured against the old
# comb (``[N,N,BT]`` + ``tl.sum``): at T=256 fused was 23.75 vs 26.30, fused
# winning by 2.55us.  Unrolling comb saved ~13us and flipped the comparison.
#
# Deciding this threshold went wrong twice, both times on methodology.  The
# final protocol: both paths use the real (BT, SK) from ``_pick_bt_sk_cube``,
# share the mix kernel, alternate in one process over three rounds taking the
# minimum, and the comb+y side *includes* ``reduce_sk`` (the layout bridge that
# SK != 1 requires):
#
#   T    BT/SK     fused  reduce   comb      y   sum    winner
#   64   64/32      9.22    2.91   6.77   3.73  13.42   fused
#   128  128/32    14.43    2.95   7.21   4.75  14.91   fused   (by 0.48us)
#   256  256/32    24.71    3.28   8.20   6.96  18.44   comb+y
#   512  256/16    44.96    4.32   9.40  10.69  24.41   comb+y
#
# 256 is a measured point, not an extrapolation.  At T=128 the two paths differ
# by 0.48us, within jitter, but neither choice loses, so the measured winner
# stands.
#
# First error: the probe fed a synthetic ``pmT`` to both paths and forgot
# ``reduce_sk`` (3.0-4.3us), which put the crossover at 128.  Second error:
# suspecting the fused path's cost grows with SK (it reads ``[BLOCK_SK,
# BLOCK_J]`` and reduces axis 0, so 32x the bytes at SK=32) and that feeding it
# SK=1 flattered it.  That one was unfounded: fused measures 14.43 at SK=32 and
# 14.07 at SK=1, a 0.36us difference, because partial-sum traffic is a small
# part of a fused epilogue dominated by sinkhorn and y.
#
# Raising ``_BT_CUBE_MAX`` pushed SK from 1-4 up to 4-32, which moved *both*
# paths at once (adding the reduce bridge to comb+y, adding partial-sum reads to
# fused), and the first two attempts each accounted for only one side.  After
# changing one constant, every threshold that was set using numbers derived
# from it has to be re-measured end to end.
_T_EPI_SPLIT = 256
_PROG_TARGET = 32  # shared BT criterion for both paths: ~32 programs total

# Cube precision mode for the mix GEMM.  The baseline has always run this way:
# ``SetHF32Mode(1)`` + ``SetHF32TransMode(1)`` at
# ``hc_pre_cube_compute_arch35.h:88``, with L0 A/B/C still
# ``LocalTensor<float>`` (fp32 layout, HF32 multipliers).  We passed no
# ``input_precision`` at all and so took the backend default ``"ieee"``
# (``triton/backends/ascend/compiler.py``, allowed values ``("ieee", "hf32")``)
# -- giving away accuracy for free, when the tolerance was already set at
# 4e-2, the baseline's own error level.
#
# Measured (npubin hash e7600df755e1 -> 32eb22f43e2a, so not a no-op knob):
#
#   T      ieee    hf32   speedup   ieee err   hf32 err
#   64     8.96    7.22   1.24x     4.5e-07    2.2e-04
#   256   16.36   10.73   1.52x     9.2e-07    2.5e-04
#   512   28.17   15.46   1.82x     1.5e-06    2.1e-04
#   1024  49.74   27.69   1.80x     3.1e-06    2.3e-04
#   2048  93.36   48.80   1.91x     4.2e-06    2.2e-04
#
# The speedup grows with T because HF32 cannot buy back the floor, only the
# cube-throughput-bound large shapes.  2.2e-04 is still two orders of magnitude
# better than the baseline's 3.1e-02.  This only touches the cube path
# (T >= 16); T=1 goes through ``tl.sum`` and is unchanged.
#
# Backend limitation: HF32 applies to fp32 x fp32 only
# (``backends/ascend/__init__.py``).  x is cast to fp32 and fn is already fp32,
# so it applies; feeding bf16 x directly would *silently* fall back to ieee
# without an error, and this table would need re-measuring.
#
# It has to be the ``tl.constexpr("hf32")`` instance -- a bare string or the
# annotated form ``_DOT_PREC: tl.constexpr = "hf32"`` makes the JIT raise
# ``NameError: Cannot access global variable ... from within @jit'ed function``.
_DOT_PREC = tl.constexpr("hf32")


def _pick_bt(T: int) -> int:
    """Vector-path mix token tile: the largest divisor of T under the cap.

    Two caps:

    1. ``_BT_VEC_MAX = 8`` -- past that, fn traffic keeps dropping but time
       gets worse; the limit is the vector MAC rate (see ``_hc_mix_kernel``).
    2. Total programs ``(T/BT) * SK`` must fill the cores.  SK is already
       16-32, so ``T/BT >= 2`` suffices.  At T=8, BT=4 (2x16=32 programs)
       measures 5.79us against BT=8 (1x16=16 programs) at 8.73us -- halving
       the program count costs more than doubling fn reuse buys.

    The divisibility requirement keeps a runtime ``t < T`` branch out of the
    loop.  T=1 is this op's production case (DSV4 decode, 4.48% of the trace)
    and must not carry a branch for the sake of large T.  The cost is that a
    prime T falls back to BT=1 with no reuse; real shapes are almost always
    even.  The cube path uses a mask instead, since BT=64 rarely divides T.
    """
    cap = min(_BT_VEC_MAX, max(1, T // 2))
    for bt in range(cap, 0, -1):
        if T % bt == 0:
            return bt
    return 1


def _pick_bt_comb(T: int) -> int:
    """comb kernel token tile.  Criterion: fill 64 cores, not 32 programs.

    Measured (us, minimum of two rounds per cell):

    ==== ====== ====== ====== ====== ===== =====
    T    BT=4   8      16     32     64    128
    ==== ====== ====== ====== ====== ===== =====
    256  24.22  19.10  19.62  21.28  25.41 53.80
    512  45.31  20.50  20.12  21.50  25.52 53.84
    1000 87.56  37.73  21.38  22.08  25.76 54.09
    2048 171.90 72.52  39.68  23.22  26.28 54.15
    ==== ====== ====== ====== ====== ===== =====

    Every row's optimum sits one step left of the largest BT with
    ``ceil(T/BT) <= 64``.  T=512/BT=8 is 64 programs (exactly one wave) at
    20.50us, while T=1000/BT=8 is 125 programs (two waves) at 37.73us, nearly
    double.  So BT is the smallest value that still fits in one wave::

        BT = max(8, next_pow2(T / 64))

    Three of four rows land on the optimum; at T=512 the formula gives BT=8
    (20.50) against the best BT=16 (20.12), a 0.38us gap inside cross-process
    jitter, not worth a special case.
    Why 64 cores here but only 32 for mix: every extra mix program re-reads the
    whole of fn, so mix has a real knee at 32; comb reads each token's data
    once, so extra programs cost no extra bytes and it can use every core.

    This table also refuted the "width cost" model behind the probe (BT 64->8
    should cut per-step width 8x and the time to ~3us).  The measured gain is
    1.33x and BT=4 rises again.  The fitted shape is ``~18us per wave +
    0.1us/token``: the 18us constant is independent of BT, which only affects
    that small linear term.  So lowering BT pays off purely by filling cores
    with more programs, not by narrowing each step.

    Those 18us are 39 sinkhorn steps, ~460ns/step, while the fused epilogue
    runs the same 39 steps on a 2-D ``[N,N]`` tile in 2.78us (71ns/step): the
    same amount of data for one vector instruction, 6.5x more expensive on a
    3-D tile.  Root cause undetermined (it needs an IR dump of how the axis
    0/1 reductions and broadcasts lower), so this is an open item rather than
    a settled conclusion.  The jump to 54us at BT=128 is also unexplained, but
    it is outside the usable range.
    """
    want = triton.next_power_of_2(max(1, triton.cdiv(T, _CORES)))
    return max(_BT_COMB_MIN, min(want, 64))


def _pick_bt_sk_cube(T: int, K: int) -> tuple[int, int]:
    """Pick (BT, SK) for cube mix (``_hc_mix_dot_sk_kernel``).

    BT always takes its cap and SK alone tunes the program count.  This is the
    opposite of the earlier ``_pick_bt_cube`` (deleted), where SK was fixed at
    1 so BT was the only knob for the program count and therefore had to track
    T (BT=16 at T=256, BT=64 at T=2048) -- and a small BT makes fn be re-read
    T/BT times.  Once SK is tunable the dilemma disappears: BT takes the cap
    for fn reuse, SK handles the program count.

    The criterion is still "reach ``_PROG_TARGET`` programs", and two probes
    agree on six of six shapes:

    ==== ============== ======== ===== =====================
    T    best config    programs us    what the formula says
    ==== ============== ======== ===== =====================
    16   BT=16/SK=16    16       5.83  BT=16/SK=32 (7.24)
    32   BT=16/SK=16    32       7.13  BT=32/SK=32 (7.20)
    64   BT=64/SK=32    32       8.26  same
    128  BT=64/SK=16    32       11.02 same
    256  BT=64/SK=8     32       16.99 same
    512  BT=64/SK=4     32       28.17 same
    ==== ============== ======== ===== =====================
    T=16 is the one exception, and it is a known trade-off rather than an
    oversight.  That shape has a single token tile, so SK equals the program
    count directly, and total fn traffic is ``programs * 1.57MB``.  T=16
    already sits on the floor (the k-chain is down to one step), so splitting K
    further only doubles fn traffic from 25MB to 50MB for nothing: SK=32
    measures 7.24us against 5.83us for SK=16.

    The cost is explicit: the formula spends 1.33us extra at T=16, which is why
    that shape benchmarks at 0.837x instead of ~0.93x.  Constraining SK the
    other way (keep the chain at 4+ steps) loses 0.07us at T=32 and 1.4us at
    T=64, where SK=32 really is best.  Either rule pays somewhere; no single
    rule wins both.

    What is actually missing is a cost model.  T=16 prefers low fn traffic and
    T=64 prefers a short chain, and the two differ by 4x in total MAC -- i.e.
    fn traffic dominates at small T and parallelism dominates at large T, and a
    model of the form ``T(SK) ~= floor(P) + max(MAC/P, fn_bytes/bandwidth)``
    should cover both.  But neither the effective fn bandwidth (1.57MB probably
    lands in L2) nor the cube MAC rate on small tiles has been measured, and
    setting a piecewise criterion from unmeasured numbers is the mistake this
    op has already made twice.  Open item: measure those two before touching
    T=16.

    SK constraint: ``K/SK`` must be a whole multiple of ``BLOCK_K``, otherwise
    the k-loop ends in a runtime half step, and a masked tail makes the
    compiler lose the regular access pattern.
    """
    bt = min(_BT_CUBE_MAX, triton.next_power_of_2(T))
    bk = _pick_bk_cube(bt)
    want = max(1, _PROG_TARGET // triton.cdiv(T, bt))
    cands = [s for s in (1, 2, 4, 8, 16, 32, 64) if K % (s * bk) == 0]
    if not cands:
        cands = [1]
    return bt, min(cands, key=lambda s: (abs(s - want), s))


def _pick_bk_cube(bt: int) -> int:
    """Cube mix k step, derived from BT: a large tile needs a small BK to fit UB.

    ``xt`` cast to fp32 occupies ``BT * BK * 4`` bytes, and ``tl.dot`` operands
    can only come through UB (the baseline has no such dilemma -- it stages x in
    L1, ``A_L1_SIZE = 128*256``)::

        BT=64,  BK=256 ->  64KB    ok
        BT=128, BK=256 -> 128KB    ok, but measures worse than BK=128
        BT=256, BK=256 -> 256KB    MLIRCompilationError, UB overflow

    All three BT=256/BK=256 configs failed to compile, matching this budget.
    BT=128 with BK=256 compiles but is slower (T=512/SK=8: 14.37 vs 13.67), so
    the threshold is 128 rather than 256.
    """
    return 128 if bt >= 128 else _BK_CUBE


@triton.jit
def _hc_mix_kernel(
    x_ptr,
    fn_ptr,
    pmix_ptr,
    K: tl.constexpr,  # n*D
    J: tl.constexpr,  # n*n+2n = 24
    SK: tl.constexpr,
    BLOCK_J: tl.constexpr,  # J padded to a power of 2 (32); lane J holds sq
    BLOCK_K: tl.constexpr,
    J0: tl.constexpr,  # fn rows read as two segments, [0,J0) and [J0,J)
    BT: tl.constexpr,  # BT tokens per program, sharing one fn tile
):
    """grid=(ceil(T/BT), SK).  Partial sums of mixes plus of x's squares.

    BT reuses one fn tile in UB across BT tokens.  At T=1, BT=1 and this
    degenerates to the original form.  Without reuse fn is re-read T times --
    3.2GB at T=2048, where mix measured 1148us and the whole op 0.061x:

    ===== ======= ========= ====================
    BT    T=64    T=2048    fn traffic at T=2048
    ===== ======= ========= ====================
    1     36.56   1148.46   3221 MB
    4     19.80   497.87    805 MB
    8     17.54   465.42    403 MB
    16    19.83   511.87    201 MB
    32    40.68   607.21    101 MB
    ===== ======= ========= ====================

    It saturates past BT=8: another 4x drop in fn traffic makes the time worse,
    so the limit is no longer bandwidth but the vector MAC rate (805M MAC /
    465us = 1.73 T MAC/s).  Large T therefore has to move to the cube (see
    ``_hc_mix_dot_kernel``); this kernel only covers small T.

    BT must divide T (``_pick_bt`` guarantees it) so the loop needs no runtime
    ``t < T`` branch.  The 1.34x at T=1 is the production case and must not
    carry a branch for large T's benefit; the cube path uses a mask because
    there BT rarely divides T.
    Each token issues exactly three stores, all unmasked and >= 32B: lanes 0-23
    are the mixes partials, lane 24 is the partial sum of squares, lanes 25-31
    are left dirty.  The sum of squares does not get its own ``psq`` array
    because a 4-byte scatter store is the single most expensive thing on this
    backend: at grid=16, "every program writes 4B" costs 2.354us while "only p0
    writes" costs 0.996us and "every program writes 128B" costs 1.128us --
    writing 4 bytes costs twice as much as writing 128, because 4B is below an
    alignment granule and turns into read-modify-write.  The old line
    ``tl.store(psq_ptr + t*SK + s, ...)`` was worth 1.4us on its own, 29% of
    the mix kernel.

    The bonus: when the epilogue reduces the whole ``[SK, 32]`` block along
    axis 0, lane 24 of the result *is* the total sum of squares, with no second
    load and no second reduction.

    fn's rows are read as two segments, J0 and (J-J0), neither masked.  It used
    to be one ``[BLOCK_J=32, BK]`` load with a static mask zeroing rows 24-31.
    That is shorter, but the masked rows really are DMA'd: at the same config,
    "read 32 rows (8 masked)" costs 3.656us and "read 16 rows" costs 2.323us,
    a 1.333us gap that matches the transfer time of the extra 16 rows.  Two
    segments took it from 3.656 to 3.212 (SK=16) and 2.532 to 2.307 (SK=32).

    The lesson: a static mask saves instructions, not bytes.  An earlier note on
    another op recorded "static masks are free on both loads and stores", which
    was about *instruction* cost and does not extrapolate to "masked data is
    not fetched".  The same holds in the space dimension: a masked ``[32,
    2048]`` tile overflows UB while ``[16,2048]`` + ``[8,2048]`` fits.

    Partials are stored untransposed.  Storing them transposed would make the
    epilogue's reduction axis contiguous, but that saves only 0.08us there
    while costing 2.8us here.
    """
    tb = tl.program_id(0)
    s = tl.program_id(1)
    offs_k = s * BLOCK_K + tl.arange(0, BLOCK_K)

    # fn's two segments are read once and shared by BT tokens
    j0 = tl.arange(0, J0)
    f0 = tl.load(fn_ptr + j0[:, None] * K + offs_k[None, :])
    J1: tl.constexpr = J - J0
    j1 = tl.arange(0, J1)
    f1 = tl.load(fn_ptr + (J0 + j1)[:, None] * K + offs_k[None, :])
    for i in tl.static_range(BT):
        t = tb * BT + i
        xv = tl.load(x_ptr + t * K + offs_k).to(tl.float32)  # [BK]
        o = pmix_ptr + (t * SK + s) * BLOCK_J
        tl.store(o + j0, tl.sum(f0 * xv[None, :], axis=1))
        tl.store(o + J0 + j1, tl.sum(f1 * xv[None, :], axis=1))
        # Sum of squares goes in lane J.  Three stores rather than one 128B
        # store: the two segments (16, 8 rows) plus this lane add up to 25,
        # which cannot be one unmasked write.  The measured 2.307us already
        # matches the cost model's lower bound (1.112 floor + 1.16 transfer),
        # so the two extra stores have no visible cost -- 64B and 32B are both
        # at or above the alignment granule; only 4B scatters are expensive.
        tl.store(o + J, tl.sum(xv * xv, axis=0))
        # Lanes J+1 .. BLOCK_J-1 are never written; the epilogue must read
        # only the first J+1 lanes.


@triton.jit
def _hc_epilogue_kernel(
    x_ptr,
    pmix_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    comb_ptr,
    inv_hk,  # 1/(n*D)
    rms_eps,
    hc_eps,
    POST_MULT: tl.constexpr,
    N: tl.constexpr,  # n (=4)
    NN: tl.constexpr,  # n*n (=16)
    J: tl.constexpr,  # n*n+2n (=24); lane 24 holds the sum of squares
    D: tl.constexpr,
    SK: tl.constexpr,
    ITERS: tl.constexpr,  # sinkhorn iterations
    BLOCK_J: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT: tl.constexpr,  # second grid axis, split by *task*; see below
):
    """grid=(T, SPLIT).  Reduce partials -> pre/post/comb(+sinkhorn) -> y.

    Split by task, not by data.  At SPLIT=2, program 0 does post + comb +
    sinkhorn and program 1 does pre + y.  Neither depends on the other (y needs
    pre, not comb), so the kernel time goes from "floor + sinkhorn + y" to
    "floor + max(sinkhorn, y)": y's 0.93us hides entirely inside sinkhorn's
    2.78us.  Measured 5.287 -> 4.645, and the 0.642 saved is exactly 0.93 of
    transfer minus the 0.322 floor difference between grid 1 and 2.  The cost
    is computing the partial reduction and rstd twice, 0.08us, in parallel and
    so free.  SPLIT=3/5 (halving or quartering y as well) measure 4.620/4.712,
    already noise: y is fully covered at SPLIT=2 and there is nothing left.
    What was rejected earlier is spreading y across 16 programs *by data*: that
    pays 16 program floors (~0.1us each) and the transfer saved cannot repay
    it.  Splitting by task needs only 2.

    Partials are read as one whole ``[BLOCK_SK, BLOCK_J]`` block, reduced along
    axis 0 into ``[BLOCK_J]``, and the three sections are then selected out of
    that in-UB vector with masks.  Loading the sections separately costs 7.25us
    against 4.84us for the whole block.  The IR shows why: a slice like
    ``pmix[:, a:b]`` has memref type ``strided<[32,1]>``, for which the
    compiler emits ``memref.alloc`` + ``memref.copy`` to materialize a
    contiguous temporary -- three times for three sections.  A whole-block read
    has no slice and no copy.

    The sum of squares lives in lane J (=24), so that same axis-0 reduction
    produces it for free, with no separate ``psq`` read (the 4B scatter store
    that used to feed it was worth 1.4us; see ``_hc_mix_kernel``).
    """
    t = tl.program_id(0)
    s = tl.program_id(1)
    ks = tl.arange(0, BLOCK_SK)
    kmask = ks < SK
    js = tl.arange(0, BLOCK_J)
    r = tl.arange(0, N)
    cj = tl.arange(0, NN)

    # ---- partials: one whole-block read + one axis-0 reduction (both
    # programs need it) ----
    # Only the first J+1 lanes are valid (0-23 mixes, 24 the sum of squares);
    # lanes 25-31 are uninitialized ``torch.empty`` bytes the mix kernel never
    # wrote.  This is a *static* mask (J is constexpr), not a slice, so it
    # triggers no memref.copy; those bytes have been DMA'd in regardless (see
    # ``_hc_mix_kernel``) and masking them only keeps dirty values, possibly
    # NaN bit patterns, out of the reduction.
    full = tl.sum(
        tl.load(
            pmix_ptr + t * SK * BLOCK_J + ks[:, None] * BLOCK_J + js[None, :],
            mask=kmask[:, None] & (js < J + 1)[None, :],
            other=0.0,
        ),
        axis=0,
    )  # [BLOCK_J]
    sqsum = tl.sum(tl.where(js == J, full, 0.0), axis=0)
    rstd = tl.rsqrt(sqsum * inv_hk + rms_eps)
    if s == 0:
        # ---- program 0: post + comb + sinkhorn ----
        mix_post = tl.sum(
            tl.where(js[None, :] == (N + r)[:, None], full[None, :], 0.0), axis=1
        )
        post = (
            tl.sigmoid(
                mix_post * rstd * tl.load(scale_ptr + 1) + tl.load(base_ptr + N + r)
            )
            * POST_MULT
        )
        tl.store(post_ptr + t * N + r, post)

        mix_c = tl.reshape(
            tl.sum(
                tl.where(js[None, :] == (2 * N + cj)[:, None], full[None, :], 0.0),
                axis=1,
            ),
            (N, N),
        )
        # sinkhorn is 2*ITERS-1 *serial* 4x4 reductions, measured 2.78us
        # (70ns/step), the largest single item in this epilogue.  Four ways out
        # were tried and none worked:
        #   - flatten to 16 scalars (copying the baseline's element-major
        #     ``VFElementMajorSoftmax``): backwards, 29.8us vs 6.94us --
        #     scalars are billed as vector instructions on this backend too;
        #   - ``tl.trans`` to turn the column normalization into a row one:
        #     166ns/step, 2.4x worse;
        #   - hoping axis 0 (cross-lane) is dearer than axis 1 (contiguous) so
        #     the asymmetry could be exploited: 62.8 vs 76.6 ns/step, i.e.
        #     cross-lane is *cheaper*, no asymmetry to use;
        #   - replacing 16 divisions with 4 reciprocals + 16 multiplies:
        #     69.8 -> 73.1 ns/step, slightly slower.  Division is not the cost;
        #     the reduce-and-broadcast is.
        # So 2.78us is accepted, and hiding y in its shadow is the only use
        # made of it.
        b_c = tl.reshape(tl.load(base_ptr + 2 * N + cj), (N, N))
        logits = mix_c * rstd * tl.load(scale_ptr + 2) + b_c
        e = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        c = e / tl.sum(e, axis=1)[:, None] + hc_eps
        # Step 1 normalizes columns only (dim=-2), matching the reference
        c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
        for _ in tl.static_range(ITERS - 1):
            c = c / (tl.sum(c, axis=1)[:, None] + hc_eps)
            c = c / (tl.sum(c, axis=0)[None, :] + hc_eps)
        tl.store(comb_ptr + t * NN + cj, tl.reshape(c, (NN,)))
    if (SPLIT == 1) or (s > 0):
        # ---- programs 1..: pre + y = sum_i pre_i * x[t, i, :] ----
        mix_pre = tl.sum(
            tl.where(js[None, :] == r[:, None], full[None, :], 0.0), axis=1
        )
        pre = (
            tl.sigmoid(mix_pre * rstd * tl.load(scale_ptr + 0) + tl.load(base_ptr + r))
            + hc_eps
        )
        NY: tl.constexpr = SPLIT - 1 if SPLIT > 1 else 1
        iy = s - 1 if SPLIT > 1 else 0
        CH: tl.constexpr = (D + NY - 1) // NY
        for d0 in tl.range(0, CH, BLOCK_D):
            offs_d = iy * CH + d0 + tl.arange(0, BLOCK_D)
            dmask = offs_d < D
            xx = tl.load(
                x_ptr + t * N * D + r[:, None] * D + offs_d[None, :],
                mask=dmask[None, :],
                other=0.0,
            ).to(tl.float32)
            yv = tl.sum(pre[:, None] * xx, axis=0)
            tl.store(
                y_ptr + t * D + offs_d,
                yv.to(y_ptr.dtype.element_ty),
                mask=dmask,
            )


@triton.jit
def _hc_mix_dot_kernel(
    x_ptr,
    fn_ptr,
    pmix_ptr,
    pmixT_ptr,
    T,
    K: tl.constexpr,
    J: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BT: tl.constexpr,
):
    """Large-T mix, cube path.  grid=(ceil(T/BT),), SK=1 (k-loop in-program).

    ``mixes[T, J] = x[T, K] @ fn^T[K, J]``, J=24 padded to 32, K=16384.
    Why the cube here when the T=1 path explicitly avoids it: at T=1 a kernel
    with ``tl.dot`` pays 1.3-1.5us extra fixed cost at every grid size and the
    whole op is 7us, which is where "this op cannot use the cube" came from.
    That conclusion holds only where fixed cost dominates.  At large T the
    fixed cost is negligible and throughput is everything: at T=2048 the
    vector path tops out at 465us (already at the vector MAC ceiling, see
    ``_hc_mix_kernel``) while the cube path takes 93.3us, 5.07x.  At T=64 it
    reverses: cube 65.6us loses to vector 17.5us.  Hence the dispatch on T.

    ``fn^T`` is loaded through transposed indices (``fn_ptr + offs_k[:,None] +
    js[None,:]*K``) rather than pre-transposed.  A probe variant fed a
    torch-pre-transposed ``fn^T`` prepared outside the timed region and
    measured identically (133.32 vs 133.67), so transposed-index reads of fn
    are free while pre-transposing costs an extra 1.57MB copy.  (That variant's
    numerics were wrong -- its ``fn^T`` row stride was BLOCK_J instead of J --
    but the timing comparison still holds, since both read the same bytes.)

    Computing ``tl.dot(fn[J,.], x^T)`` to get mixes^T was also tried: 110us at
    T=2048, 17us worse.  But its output is exactly the token-innermost layout
    the comb kernel wants, so this kernel instead keeps the faster algorithm
    and writes *both* layouts: an extra T*32*4B (256KB at T=2048), cheaper than
    those 17us.

    Accuracy: fn is fp32, x is bf16 cast to fp32.  The cube path's maxdiff is
    4.6e-5 (4.0e-6 relative), far better than the baseline's 3.1e-2, because
    the baseline enables HF32 on the cube (``SetHF32Mode(1)`` at
    ``hc_pre_cube_compute_arch35.h:88``) -- which we now do as well, see
    ``_DOT_PREC``.
    """
    tb = tl.program_id(0)
    rows = tb * BT + tl.arange(0, BT)
    rmask = rows < T  # cube BT rarely divides T, so this one needs a mask
    js = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BT, BLOCK_J), dtype=tl.float32)
    sq = tl.zeros((BT,), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xt = tl.load(
            x_ptr + rows[:, None] * K + offs_k[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        fb = tl.load(
            fn_ptr + offs_k[:, None] + js[None, :] * K,
            mask=(js < J)[None, :],
            other=0.0,
        )
        acc += tl.dot(xt, fb, input_precision=_DOT_PREC)
        sq += tl.sum(xt * xt, axis=1)
    jm = (js < J)[None, :]
    tl.store(
        pmix_ptr + rows[:, None] * BLOCK_J + js[None, :],
        acc,
        mask=rmask[:, None] & jm,
    )
    tl.store(pmix_ptr + rows * BLOCK_J + J, sq, mask=rmask)
    # the token-innermost copy, for the comb kernel
    tl.store(pmixT_ptr + js[None, :] * T + rows[:, None], acc, mask=rmask[:, None] & jm)
    tl.store(pmixT_ptr + J * T + rows, sq, mask=rmask)


@triton.jit
def _hc_mix_dot_sk_kernel(
    x_ptr,
    fn_ptr,
    pmix_ptr,
    T,
    K: tl.constexpr,
    J: tl.constexpr,
    SK: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BT: tl.constexpr,
):
    """Mid-T mix: cube plus split-K.  grid=(ceil(T/BT), SK).

    The only difference from ``_hc_mix_dot_kernel`` is that the k-loop covers
    K/SK and writes partial sums.  That one difference turns the whole mid-T
    range around.

    The problem was never throughput, it was the serial k-chain inside one
    program.  At one BT=64/SK=1 config, T=512 takes 90.26us (8 programs, 64
    k-loop steps) and T=2048 takes 93.50us (32 programs, the same 64 steps):
    4x the work in the same time.  The time simply equals the chain length.
    Cube BT cannot shrink further (a narrow M leaves the tile underfilled), but
    K can be split.

    Measured (us; the control column is each shape's best vector BT):

    ==== ========= ===== ===== ===== ===== ===== ==================
    T    vector    SK=1  2     4     8     16    best (BT=64)
    ==== ========= ===== ===== ===== ===== ===== ==================
    64   17.57     89.37 46.23 24.61 14.15 9.49  SK=16 -> 1.85x
    128  33.06     89.67 46.39 25.09 14.99 11.02 SK=16 -> 3.00x
    256  62.08     89.89 46.95 26.09 16.99 17.40 SK=8  -> 3.65x
    512  117.01    90.26 48.04 28.17 28.82 30.27 SK=4  -> 4.15x
    ==== ========= ===== ===== ===== ===== ===== ==================
    The four optima are at 16/32/32/32 programs, the same criterion as the
    other three kernels (``_PROG_TARGET``).  Note the SK=1 column is ~90us for
    all four shapes, direct evidence of "time equals chain length", and each
    row's optimum appears where the chain is cut to roughly 4 steps.

    Two side effects:

    1. Accuracy improves with SK (T=256: 4.1e-5 at SK=1 -> 5.5e-6 at SK=16),
       because each partial accumulates over a shorter chain.
    2. The partial layout ``[T*SK, BLOCK_J]`` is identical to the vector path's,
       so the existing fused epilogue consumes it unchanged.  This path is
       only a different mix kernel.
    """
    tb = tl.program_id(0)
    s = tl.program_id(1)
    rows = tb * BT + tl.arange(0, BT)
    rmask = rows < T
    js = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BT, BLOCK_J), dtype=tl.float32)
    sq = tl.zeros((BT,), dtype=tl.float32)

    KS: tl.constexpr = K // SK
    for k0 in tl.range(s * KS, s * KS + KS, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xt = tl.load(
            x_ptr + rows[:, None] * K + offs_k[None, :],
            mask=rmask[:, None],
            other=0.0,
        ).to(tl.float32)
        fb = tl.load(
            fn_ptr + offs_k[:, None] + js[None, :] * K,
            mask=(js < J)[None, :],
            other=0.0,
        )
        acc += tl.dot(xt, fb, input_precision=_DOT_PREC)
        sq += tl.sum(xt * xt, axis=1)

    o = (rows * SK + s)[:, None] * BLOCK_J
    tl.store(pmix_ptr + o + js[None, :], acc, mask=rmask[:, None] & (js < J)[None, :])
    tl.store(pmix_ptr + (rows * SK + s) * BLOCK_J + J, sq, mask=rmask)


@triton.jit
def _hc_reduce_sk_kernel(
    pmix_ptr,
    out_ptr,
    outT_ptr,
    T,
    SK: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BT: tl.constexpr,
):
    """Reduce ``[T*SK, BLOCK_J]`` partials into both layouts.
    grid=(ceil(T/BT),).

    Needed only where split-K cube mix feeds the comb+y epilogue: mix produces
    token-outermost partials, while comb needs token-innermost (see
    ``_hc_comb_kernel`` for where that layout comes from -- the reductions fall
    on the outer axes, so each step is a full-width vector op).  Neither side
    will yield, so a pass goes in between.

    It is cheap: at T=512/SK=4 it reads 256KB and writes 2x64KB, about one
    kernel floor.  What it buys is a comb+y epilogue that is more than twice as
    fast as the fused one at large T.

    BT is clamped to [8, 32].  Measured (us, program count in parentheses):

    ============ ========= ========= ========= ======== ======== ========= =========
    BT           4         8         16        32       64       128       384
    ============ ========= ========= ========= ======== ======== ========= =========
    T=512 SK=4   3.01(128) 3.32(64)  2.98(32)  3.89(16) 6.51(8)  12.12(4)  34.02(2)
    T=1000 SK=2  3.85(250) 2.54(125) 2.95(63)  2.40(32) 2.82(16) 4.47(8)   11.32(3)
    ============ ========= ========= ========= ======== ======== ========= =========

    This curve has a different shape and a different mechanism from comb's.
    comb drops into a second wave at 128-250 programs and doubles; this pass
    still costs 3.85us at 250 programs, because it has no serial chain and
    extra programs cost it nothing.  What hurts it is a *large tile*: BT=384
    takes 11.32us (12 GB/s), while BT=4's 2KB tile is too fragmented.  The
    valley is flat (BT=8-32 all land in 2.4-3.3us), so borrowing comb's
    starting point and clamping at 32 is enough.  Why large tiles are expensive
    was not root-caused (the transposed ``outT_ptr + js*T + rows`` strided
    store is the suspect), but inside the [8,32] window it does not bind.

    The first version derived this BT straight from the UB budget
    (``98304 // (SK*BLOCK_J*4)`` = 384), which gave T=1000 only 3 programs and
    cost 11.62us in the end-to-end benchmark -- matching the 11.32us the probe
    measured at BT=384.  A UB budget is an upper bound, not a value to use.
    Why the reduction is not folded into comb itself: comb reads ``[J, BT]``,
    and taking that from token-outermost partials means transposing in
    registers first, where ``tl.trans`` measures 166ns/step against 70ns
    (2.4x).  One transpose is cheap; one inside a 39-step chain is not.
    """
    tb = tl.program_id(0)
    rows = tb * BT + tl.arange(0, BT)
    rmask = rows < T
    js = tl.arange(0, BLOCK_J)
    ks = tl.arange(0, BLOCK_SK)
    p = tl.load(
        pmix_ptr
        + (rows[:, None, None] * SK + ks[None, :, None]) * BLOCK_J
        + js[None, None, :],
        mask=rmask[:, None, None] & (ks < SK)[None, :, None],
        other=0.0,
    )
    v = tl.sum(p, axis=1)
    tl.store(out_ptr + rows[:, None] * BLOCK_J + js[None, :], v, mask=rmask[:, None])
    tl.store(outT_ptr + js[None, :] * T + rows[:, None], v, mask=rmask[:, None])


@triton.jit
def _hc_comb_kernel(
    pmixT_ptr,
    scale_ptr,
    base_ptr,
    post_ptr,
    comb_ptr,
    T,
    inv_hk,
    rms_eps,
    hc_eps,
    POST_MULT: tl.constexpr,
    N: tl.constexpr,
    NN: tl.constexpr,
    J: tl.constexpr,
    ITERS: tl.constexpr,
    BT: tl.constexpr,
):
    """Large-T post + comb + sinkhorn.  grid=(ceil(T/BT),), one token per lane.

    sinkhorn is 2*ITERS-1 serial 4x4 reductions and the step count is
    independent of the token count, so handling BT tokens in one program
    amortizes those 39 steps.
    Why not ``tl.sum``
    ~~~~~~~~~~~~~~~~~~

    Neither direction of this 4x4 should use ``tl.sum``.  N=4 is a compile-time
    constant (this op only supports n=4; ``hc_mult != 4`` raises), so with both
    N axes expanded into separate ``[BT]`` tiles::

        row sum (over c):  a[r][0]+a[r][1]+a[r][2]+a[r][3]   3 register adds
        col sum (over r):  a[0][j]+a[1][j]+a[2][j]+a[3][j]   3, same
        broadcast:         none -- the divisor *is* that [BT] register

    Measured at ITERS=20 (39 steps):

    ==== ==== ===================== ======================= ======
    T    BT   ``[N,N,BT]`` + tl.sum 16 x ``[BT]`` unrolled
    ==== ==== ===================== ======================= ======
    1000 16   21.60                 9.88                    2.19x
    2048 32   23.26                 9.88                    2.35x
    ==== ==== ===================== ======================= ======

    Per step that is 402ns -> 16ns, 25x.  This is the root cause of the earlier
    open item "reduction on a 3-D tile costs 6.5x more than on a 2-D one": not
    the data volume (both sides do exactly the same number of multiplies and
    divides) but the lowering of ``tl.sum`` reducing along an outer axis and
    broadcasting back along the third.

    The lead came from the baseline's ``VFProcessCombFragPremixElementMajor``
    (``hc_pre_base_arch35.h:1167``): 16 registers ``m00..m33``, both axes
    expanded, one token per lane, zero cross-lane reduction.  The baseline's
    non-premix path uses something else (``VFProcessCombFragPacked`` at
    ``:1012``: expand r, keep c in lanes, so the row direction pays 6 butterfly
    shuffles).  Expanding both axes saves one more layer than the baseline.

    BT criterion unchanged, but the old reasoning was wrong
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    At 16ns/step, sinkhorn is only 0.6us (39 x 16ns) and comb becomes dominated
    by fixed cost (~8.8us of load/store/post/softmax).  So BT still follows the
    wave model in ``_pick_bt_comb`` and no longer has anything to do with
    sinkhorn.  BT=32 (9.39) and BT=64 (9.43) measure even -- the prediction
    that full width needs BT>=64 was falsified.

    Both layouts turn at BT>=128 (per step 424 -> 932 -> 1514ns, linear in BT
    after that) and are flat at BT<=64.  The knee lands exactly on the
    baseline's ``VL_FP32 = 64``.
    The old layouts, for the record
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ========================== ====== ====== ====== ====== ====== ======
    layout                     BT=1   4      16     64     128    256
    ========================== ====== ====== ====== ====== ====== ======
    ``c[BT,N,N]`` token-outer  115.41 111.48 102.66 197.32 391.34 780.22
    ``c[N,N,BT]`` token-inner  109.59 171.86 39.78  26.29  54.20  91.92
    ========================== ====== ====== ====== ====== ====== ======

    (T=2048, us, both with ``tl.sum``.)  Token-outermost puts the reduction on
    an inner axis of length 4, so a 128-lane vector instruction uses 3% of its
    width and larger BT is slower.

    Two mistakes happened here.  The first probe tried only token-outermost,
    saw BT=2 run 2.6x slower, and rejected the whole idea of tiling at large T.
    The second added token-innermost but, to fit the UB budget, *also* cut
    BLOCK_D from 4096 to 384, which shortened y's contiguous runs 10x, so both
    layouts "lost".  An experiment that moves two variables is not an
    experiment.
    """
    tb = tl.program_id(0)
    cols = tb * BT + tl.arange(0, BT)
    cm = cols < T
    r = tl.arange(0, N)

    sq = tl.load(pmixT_ptr + J * T + cols, mask=cm, other=1.0)
    rstd = tl.rsqrt(sq * inv_hk + rms_eps)  # [BT]

    mp = tl.load(
        pmixT_ptr + (N + r)[:, None] * T + cols[None, :], mask=cm[None, :], other=0.0
    )  # [N, BT]
    post = (
        tl.sigmoid(
            mp * rstd[None, :] * tl.load(scale_ptr + 1)
            + tl.load(base_ptr + N + r)[:, None]
        )
        * POST_MULT
    )
    tl.store(post_ptr + cols[None, :] * N + r[:, None], post, mask=cm[None, :])

    # ---- sinkhorn: both N axes in registers, not one tl.sum call ----
    # Rationale in the "Why not tl.sum" section above.
    w = rstd * tl.load(scale_ptr + 2)
    p = pmixT_ptr + 2 * N * T
    bp = base_ptr + 2 * N
    a0 = tl.load(p + 0 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 0)
    a1 = tl.load(p + 1 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 1)
    a2 = tl.load(p + 2 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 2)
    a3 = tl.load(p + 3 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 3)
    b0 = tl.load(p + 4 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 4)
    b1 = tl.load(p + 5 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 5)
    b2 = tl.load(p + 6 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 6)
    b3 = tl.load(p + 7 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 7)
    c0 = tl.load(p + 8 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 8)
    c1 = tl.load(p + 9 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 9)
    c2 = tl.load(p + 10 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 10)
    c3 = tl.load(p + 11 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 11)
    d0 = tl.load(p + 12 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 12)
    d1 = tl.load(p + 13 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 13)
    d2 = tl.load(p + 14 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 14)
    d3 = tl.load(p + 15 * T + cols, mask=cm, other=0.0) * w + tl.load(bp + 15)

    # softmax over c (row direction): max and sum become register ops
    ma = tl.maximum(tl.maximum(a0, a1), tl.maximum(a2, a3))
    mb = tl.maximum(tl.maximum(b0, b1), tl.maximum(b2, b3))
    mc = tl.maximum(tl.maximum(c0, c1), tl.maximum(c2, c3))
    md = tl.maximum(tl.maximum(d0, d1), tl.maximum(d2, d3))
    a0 = tl.exp(a0 - ma)
    a1 = tl.exp(a1 - ma)
    a2 = tl.exp(a2 - ma)
    a3 = tl.exp(a3 - ma)
    b0 = tl.exp(b0 - mb)
    b1 = tl.exp(b1 - mb)
    b2 = tl.exp(b2 - mb)
    b3 = tl.exp(b3 - mb)
    c0 = tl.exp(c0 - mc)
    c1 = tl.exp(c1 - mc)
    c2 = tl.exp(c2 - mc)
    c3 = tl.exp(c3 - mc)
    d0 = tl.exp(d0 - md)
    d1 = tl.exp(d1 - md)
    d2 = tl.exp(d2 - md)
    d3 = tl.exp(d3 - md)

    sa = a0 + a1 + a2 + a3
    sb = b0 + b1 + b2 + b3
    sc = c0 + c1 + c2 + c3
    sd = d0 + d1 + d2 + d3
    a0 = a0 / sa + hc_eps
    a1 = a1 / sa + hc_eps
    a2 = a2 / sa + hc_eps
    a3 = a3 / sa + hc_eps
    b0 = b0 / sb + hc_eps
    b1 = b1 / sb + hc_eps
    b2 = b2 / sb + hc_eps
    b3 = b3 / sb + hc_eps
    c0 = c0 / sc + hc_eps
    c1 = c1 / sc + hc_eps
    c2 = c2 / sc + hc_eps
    c3 = c3 / sc + hc_eps
    d0 = d0 / sd + hc_eps
    d1 = d1 / sd + hc_eps
    d2 = d2 / sd + hc_eps
    d3 = d3 / sd + hc_eps

    # Step 1 normalizes columns only (over r), matching the fused epilogue
    t0 = a0 + b0 + c0 + d0 + hc_eps
    t1 = a1 + b1 + c1 + d1 + hc_eps
    t2 = a2 + b2 + c2 + d2 + hc_eps
    t3 = a3 + b3 + c3 + d3 + hc_eps
    a0 = a0 / t0
    a1 = a1 / t1
    a2 = a2 / t2
    a3 = a3 / t3
    b0 = b0 / t0
    b1 = b1 / t1
    b2 = b2 / t2
    b3 = b3 / t3
    c0 = c0 / t0
    c1 = c1 / t1
    c2 = c2 / t2
    c3 = c3 / t3
    d0 = d0 / t0
    d1 = d1 / t1
    d2 = d2 / t2
    d3 = d3 / t3
    for _ in tl.static_range(ITERS - 1):
        sa = a0 + a1 + a2 + a3 + hc_eps
        sb = b0 + b1 + b2 + b3 + hc_eps
        sc = c0 + c1 + c2 + c3 + hc_eps
        sd = d0 + d1 + d2 + d3 + hc_eps
        a0 = a0 / sa
        a1 = a1 / sa
        a2 = a2 / sa
        a3 = a3 / sa
        b0 = b0 / sb
        b1 = b1 / sb
        b2 = b2 / sb
        b3 = b3 / sb
        c0 = c0 / sc
        c1 = c1 / sc
        c2 = c2 / sc
        c3 = c3 / sc
        d0 = d0 / sd
        d1 = d1 / sd
        d2 = d2 / sd
        d3 = d3 / sd

        t0 = a0 + b0 + c0 + d0 + hc_eps
        t1 = a1 + b1 + c1 + d1 + hc_eps
        t2 = a2 + b2 + c2 + d2 + hc_eps
        t3 = a3 + b3 + c3 + d3 + hc_eps
        a0 = a0 / t0
        a1 = a1 / t1
        a2 = a2 / t2
        a3 = a3 / t3
        b0 = b0 / t0
        b1 = b1 / t1
        b2 = b2 / t2
        b3 = b3 / t3
        c0 = c0 / t0
        c1 = c1 / t1
        c2 = c2 / t2
        c3 = c3 / t3
        d0 = d0 / t0
        d1 = d1 / t1
        d2 = d2 / t2
        d3 = d3 / t3
    o = comb_ptr + cols * NN
    tl.store(o + 0, a0, mask=cm)
    tl.store(o + 1, a1, mask=cm)
    tl.store(o + 2, a2, mask=cm)
    tl.store(o + 3, a3, mask=cm)
    tl.store(o + 4, b0, mask=cm)
    tl.store(o + 5, b1, mask=cm)
    tl.store(o + 6, b2, mask=cm)
    tl.store(o + 7, b3, mask=cm)
    tl.store(o + 8, c0, mask=cm)
    tl.store(o + 9, c1, mask=cm)
    tl.store(o + 10, c2, mask=cm)
    tl.store(o + 11, c3, mask=cm)
    tl.store(o + 12, d0, mask=cm)
    tl.store(o + 13, d1, mask=cm)
    tl.store(o + 14, d2, mask=cm)
    tl.store(o + 15, d3, mask=cm)


@triton.jit
def _hc_y_kernel(
    x_ptr,
    pmix_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    inv_hk,
    rms_eps,
    hc_eps,
    N: tl.constexpr,
    J: tl.constexpr,
    D: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Large-T pre + y.  grid=(T,), one token per program, BLOCK_D maximal.

    This section reads T*n*D*2B of x and writes T*D*2B of y, so it wants long
    contiguous runs, which conflicts with comb's need for a large BT -- hence
    separate kernels.  It uses the token-outermost partials; the
    token-innermost copy costs 62.67us, 20us more for the strided reads.

    It is not pure data movement; about half the time is compute
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    An earlier note here read "T=2048 measures 42.81us (~2 TB/s, close to
    peak)".  That "close to peak" was an estimate and it was wrong.  A bare
    accumulate at the same traffic measures 4.51 TB/s while y reaches only
    2.2 TB/s, so half the time is not spent moving data.  The estimate came
    from extrapolating "single-core DMA 43-47 GB/s x 64 cores = 2.9 TB/s", and
    single-core bandwidth does not multiply like that -- that number was
    measured with one core owning the bus.
    Three rounds of differencing at T=2048:

    ==================== =====================================================
    probe                finding
    ==================== =====================================================
    y vs floor           y 37.96 vs 18.60 at equal traffic -> 19.4us is compute
    unrolling ``tl.sum`` expanding into 4 adds is *slower*, 0.93x -> not the
                         reduction
    decomposition        4 ``Mul`` passes 0.30us, prelude 12.24us -> all in the
                         prelude
    ==================== =====================================================

    33.5M fp32 multiplies cost 0.30us while the per-token scalar prelude costs
    12.24us (6ns/token).  A factor of 40.

    ``tl.sum`` is the best lowering here; do not carry comb's conclusion over.
    comb gains 25x from unrolling ``tl.sum`` because its tile is 3-D
    ``[N,N,BT]``, both reduction axes are outer, and it broadcasts back along
    the third.  Here the tile is 2-D ``[N,BLOCK_D]``, the reduction is axis 0,
    and the inner axis D is contiguous and full width, so ``tl.sum`` already
    lowers to just 3 full-width adds and hand-unrolling saves nothing (and is
    0.93x, the extra scalar loads not paying off).  The criterion is "reduction
    on a 3-D tile plus a broadcast along the third axis is expensive", not
    "``tl.sum`` is expensive".

    Prelude sq/mpre: load directly, do not read full width and select
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ``sq`` is at index ``J`` and ``mpre`` at ``0..N-1``, both compile-time known
    and contiguous, and this kernel needs only those lanes -- reading
    ``[BLOCK_J]`` and picking with two ``tl.where`` calls wastes 32 lanes and a
    whole reduction.  Probe (numerics identical to the bit) and the end-to-end
    result:

    ==== ============= ============= =================
    T    probe select  probe direct  y end-to-end
    ==== ============= ============= =================
    256  7.04          6.23          7.05 -> 6.44
    1000 18.52         16.02         19.43 -> 16.34
    2048 34.20         29.41         41.45 -> 32.39
    ==== ============= ============= =================

    The criterion is "does it need the full width", not "select vs direct".
    ``_hc_epilogue_kernel`` uses the same select pattern and should *not* be
    changed: its four sections (sq/pre/post/comb) together need all ``J+1``
    lanes.  Measured there, reading ``sq`` as its own column is even within
    1%, and converting all four sections to narrow direct reads is 2-4x
    slower.  "select is expensive" is a bad criterion; do not apply it to other
    kernels.
    Two things that are not understood, and should not be treated as if they
    were:

    1. Differencing does not add up here.  Pricing the prelude on its own saves
       1.57us (12.73 -> 11.17) while total time drops 4.79us, with the main
       loop byte-for-byte unchanged.  Explaining that needs an IR dump and a
       register-allocation comparison.  The end-to-end gain was larger still
       (9.06us), so a probe's absolute numbers are not a budget for the real
       thing.
    2. The remaining 11.17us of prelude is still unexplained: 5.5ns/token for
       one scalar load, one ``[N]`` load, an ``rsqrt`` and one ``sigmoid`` over
       ``[N]`` is still too expensive.  Next step is to price the scalar
       transcendentals separately.
    """
    t = tl.program_id(0)
    r = tl.arange(0, N)
    p = pmix_ptr + t * BLOCK_J
    sq = tl.load(p + J)  # scalar, index known at compile time
    mpre = tl.load(p + r)  # [N], contiguous
    rstd = tl.rsqrt(sq * inv_hk + rms_eps)
    pre = (
        tl.sigmoid(mpre * rstd * tl.load(scale_ptr + 0) + tl.load(base_ptr + r))
        + hc_eps
    )
    for d0 in tl.range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        dm = offs_d < D
        xx = tl.load(
            x_ptr + t * N * D + r[:, None] * D + offs_d[None, :],
            mask=dm[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            y_ptr + t * D + offs_d,
            tl.sum(pre[:, None] * xx, axis=0).to(y_ptr.dtype.element_ty),
            mask=dm,
        )


def _pick_j0(J: int) -> int:
    """Split fn's J rows into ``[0,J0)`` and ``[J0,J)``, both powers of two.

    ``tl.arange`` requires a power-of-two length, so ``tl.arange(0, 24)`` is out,
    and padding to 32 with a static mask still DMAs those 8 rows (a static mask
    saves instructions, not bytes; see ``_hc_mix_kernel``).  J=24 -> 16+8.
    """
    j0 = 1 << (J.bit_length() - 1)  # largest power of two <= J
    if j0 == J:  # J is already a power of two, no split needed
        return J
    j1 = J - j0
    if j1 & (j1 - 1):
        raise ValueError(
            f"J={J} does not split into two powers of two ({j0}+{j1}); "
            "this kernel supports exactly two segments"
        )
    return j0


def _pick_sk(K: int, T: int, split_k=None):
    """Pick SK, the split-K count, for the vector mix path.

    Three constraints:

    1. SK must divide K.  A masked runtime tail makes the compiler lose the
       regular access pattern (measured 13x on a masked scale load in another
       op on this backend).
    2. ``BK = K/SK`` must be <= 1024: the fn tile is ``[32, BK]`` fp32, so
       BK=1024 is already 128KB and more overflows UB (256KB x 80% = 204.8KB).
    3. BK should not go below 256; below that only the floor is left.

    Among the divisors satisfying 1-3, take the one closest to 32 (the
    crossover of the launch floor and single-core bandwidth, see
    ``_SK_PREFER``).  The total program count is T*SK, so SK is scaled down as
    T grows.
    """
    if split_k is not None:
        if K % split_k:
            raise ValueError(f"split_k={split_k} must divide K={K}")
        return split_k
    prefer = max(1, _SK_PREFER // max(1, T))
    cands = [
        s
        for s in range(1, 257)
        if K % s == 0 and K // s <= _BK_MAX and K // s >= _BK_MIN
    ]
    if not cands:
        # K is too small (e.g. D=16 -> K=64) for the BK floor to leave any
        # candidate; fall back to the whole block in one program.
        cands = [s for s in range(1, 257) if K % s == 0 and K // s <= _BK_MAX]
    return min(cands, key=lambda s: (abs(s - prefer), s))


def hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    hc_sinkhorn_iters: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    *,
    split_k: int | None = None,
    block_d: int | None = None,
    **backend_opts,
):
    """Same signature and returns as ``torch.ops._C_ascend.npu_hc_pre_v2``.

    Args:
        x: ``[*outer, n, D]`` bf16 residual stack, n = ``hc_mult``.
        hc_fn: ``[n*n+2n, n*D]`` fp32 projection.
        hc_scale: ``[3]`` fp32, the pre / post / comb logit scales.
        hc_base: ``[n*n+2n]`` fp32 biases, laid out pre | post | comb.
        hc_mult: must equal ``x.shape[-2]`` and must be 4.
        hc_sinkhorn_iters: >= 1; the first iteration normalizes columns only.
        norm_eps: epsilon inside the rsqrt of the RMS normalization.
        hc_eps: epsilon added to pre and inside every sinkhorn division.
        split_k: override the split-K count; for benchmarking only.
        block_d: override ``BLOCK_D`` in the y loop; for benchmarking only.
        **backend_opts: forwarded to every kernel launch (e.g. ``debug``).

    Returns:
        ``(y, post, comb)`` with ``y [*outer, D]`` bf16,
        ``post [*outer, n]`` fp32 and ``comb [*outer, n, n]`` fp32, where
        ``outer`` is ``x`` without its last two dims
        (``[B,S,n,D] -> [B,S]``, ``[T,n,D] -> [T]``).

    Raises:
        ValueError: on any unsupported rank, dtype, shape or ``hc_mult != 4``.
    """
    if x.dim() not in (3, 4):
        raise ValueError(f"x must be rank 3 or 4, got {x.dim()}")
    n = x.shape[-2]
    D = x.shape[-1]
    outer = x.shape[:-2]
    T = 1
    for s in outer:
        T *= s
    if hc_mult != n:
        raise ValueError(f"hc_mult ({hc_mult}) must equal x.shape[-2] ({n})")
    if n != HC_MULT_LIMIT:
        raise ValueError(
            f"only n={HC_MULT_LIMIT} is supported (the baseline's limit too), "
            f"got {n}"
        )
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16, got {x.dtype}")
    J = n * n + 2 * n
    K = n * D
    if tuple(hc_fn.shape) != (J, K):
        raise ValueError(f"hc_fn must be [{J}, {K}], got {tuple(hc_fn.shape)}")
    if hc_base.numel() != J:
        raise ValueError(f"hc_base must hold {J} values, got {hc_base.numel()}")
    if hc_scale.numel() != 3:
        raise ValueError(f"hc_scale must hold 3 values, got {hc_scale.numel()}")
    if hc_sinkhorn_iters < 1:
        raise ValueError(f"hc_sinkhorn_iters must be >= 1, got {hc_sinkhorn_iters}")
    for t_, nm in ((hc_fn, "hc_fn"), (hc_scale, "hc_scale"), (hc_base, "hc_base")):
        if t_.dtype != torch.float32:
            raise ValueError(f"{nm} must be fp32, got {t_.dtype}")

    xf = x.contiguous()
    fn = hc_fn.contiguous()

    BLOCK_J = triton.next_power_of_2(J)
    if BLOCK_J <= J:
        raise ValueError(
            f"BLOCK_J ({BLOCK_J}) must exceed J ({J}): lane J holds the sum of "
            "squares of x"
        )

    y = torch.empty((T, D), device=x.device, dtype=x.dtype)
    post = torch.empty((T, n), device=x.device, dtype=torch.float32)
    comb = torch.empty((T, n, n), device=x.device, dtype=torch.float32)
    bd = block_d if block_d is not None else min(triton.next_power_of_2(D), 4096)
    if T >= _T_EPI_SPLIT:
        # ---- large T: cube mix + comb + y, three kernels ----
        # The three stages want conflicting tilings, so each gets a kernel:
        #   mix  wants a large BT (the cube's M) and enough programs -> BT=64
        #   comb wants a large BT (to amortize 39 serial steps) and tokens
        #        innermost
        #   y    wants a large BLOCK_D (long contiguous runs), BT irrelevant
        #        -> one token per program
        # Fusing them would force a shared BT and BLOCK_D, and one stage would
        # always lose.
        pmix = torch.empty((T, BLOCK_J), device=x.device, dtype=torch.float32)
        pmixT = torch.empty((BLOCK_J, T), device=x.device, dtype=torch.float32)

        btm, skm = _pick_bt_sk_cube(T, K)
        if split_k is not None:
            skm = split_k
        if skm == 1:
            # At SK=1 the mix kernel writes both layouts itself, so the
            # reduce pass is not needed.
            _hc_mix_dot_kernel[(triton.cdiv(T, btm),)](
                xf,
                fn,
                pmix,
                pmixT,
                T,
                K=K,
                J=J,
                BLOCK_J=BLOCK_J,
                BLOCK_K=_pick_bk_cube(btm),
                BT=btm,
                **backend_opts,
            )
        else:
            part = torch.empty((T * skm, BLOCK_J), device=x.device, dtype=torch.float32)
            _hc_mix_dot_sk_kernel[(triton.cdiv(T, btm), skm)](
                xf,
                fn,
                part,
                T,
                K=K,
                J=J,
                SK=skm,
                BLOCK_J=BLOCK_J,
                BLOCK_K=_pick_bk_cube(btm),
                BT=btm,
                **backend_opts,
            )
            # BT: see ``_hc_reduce_sk_kernel``.  Clamped to [8, 32], starting
            # from comb's criterion.  The UB budget is an upper bound, not a
            # value: deriving BT from it directly (98304 // (SK*BLOCK_J*4))
            # gave T=1000/SK=2 a BT of 384 and only 3 programs, costing
            # 11.62us where the probe had measured 11.32us at BT=384.
            btr = max(
                1,
                min(
                    _pick_bt_comb(T),
                    _BT_REDUCE_MAX,
                    triton.next_power_of_2(T),
                    98304 // (skm * BLOCK_J * 4),
                ),
            )
            _hc_reduce_sk_kernel[(triton.cdiv(T, btr),)](
                part,
                pmix,
                pmixT,
                T,
                SK=skm,
                BLOCK_J=BLOCK_J,
                BLOCK_SK=triton.next_power_of_2(skm),
                BT=btr,
                **backend_opts,
            )
        btc = min(_pick_bt_comb(T), triton.next_power_of_2(T))
        _hc_comb_kernel[(triton.cdiv(T, btc),)](
            pmixT,
            hc_scale,
            hc_base,
            post,
            comb,
            T,
            1.0 / K,
            norm_eps,
            hc_eps,
            POST_MULT=HC_POST_MULT,
            N=n,
            NN=n * n,
            J=J,
            ITERS=hc_sinkhorn_iters,
            BT=btc,
            **backend_opts,
        )
        _hc_y_kernel[(T,)](
            xf,
            pmix,
            hc_scale,
            hc_base,
            y,
            1.0 / K,
            norm_eps,
            hc_eps,
            N=n,
            J=J,
            D=D,
            BLOCK_J=BLOCK_J,
            BLOCK_D=bd,
            **backend_opts,
        )
        return (
            y.view(*outer, D),
            post.view(*outer, n),
            comb.view(*outer, n, n),
        )

    # ---- small/mid T: mix (vector or cube+split-K) + fused epilogue ----
    # Both mix kernels write the *same* [T*SK, BLOCK_J] partial layout, so the
    # epilogue is shared.
    if T >= _T_CUBE_MIN:
        # Mid T: cube with split-K.  See ``_hc_mix_dot_sk_kernel`` (up to
        # 4.15x over the vector path).
        bt, sk = (
            _pick_bt_sk_cube(T, K)
            if split_k is None
            else (min(_BT_CUBE_MAX, triton.next_power_of_2(T)), split_k)
        )
        pmix = torch.empty((T * sk, BLOCK_J), device=x.device, dtype=torch.float32)
        _hc_mix_dot_sk_kernel[(triton.cdiv(T, bt), sk)](
            xf,
            fn,
            pmix,
            T,
            K=K,
            J=J,
            SK=sk,
            BLOCK_J=BLOCK_J,
            BLOCK_K=_pick_bk_cube(bt),
            BT=bt,
            **backend_opts,
        )
    else:
        sk = _pick_sk(K, T, split_k)
        bt = _pick_bt(T)
        pmix = torch.empty((T * sk, BLOCK_J), device=x.device, dtype=torch.float32)
        _hc_mix_kernel[(T // bt, sk)](
            xf,
            fn,
            pmix,
            K=K,
            J=J,
            SK=sk,
            BLOCK_J=BLOCK_J,
            BLOCK_K=K // sk,
            J0=_pick_j0(J),
            BT=bt,
            **backend_opts,
        )

    # y / post / comb and bd were allocated before the dispatch; both paths
    # share them.
    # SPLIT=2: program 0 does comb+sinkhorn, program 1 does pre+y, hiding y in
    # sinkhorn's shadow (measured -0.642us; see ``_hc_epilogue_kernel``).  With
    # a single sinkhorn iteration there is only a softmax left and no shadow to
    # hide in, so fall back to SPLIT=1 and save the second program's floor.
    split = 2 if hc_sinkhorn_iters >= 4 else 1
    if split > 1:
        bd = min(bd, triton.next_power_of_2(triton.cdiv(D, split - 1)))
    _hc_epilogue_kernel[(T, split)](
        xf,
        pmix,
        hc_scale,
        hc_base,
        y,
        post,
        comb,
        1.0 / K,
        norm_eps,
        hc_eps,
        POST_MULT=HC_POST_MULT,
        N=n,
        NN=n * n,
        J=J,
        D=D,
        SK=sk,
        ITERS=hc_sinkhorn_iters,
        BLOCK_J=BLOCK_J,
        BLOCK_SK=triton.next_power_of_2(sk),
        BLOCK_D=bd,
        SPLIT=split,
        **backend_opts,
    )

    return (y.view(*outer, D), post.view(*outer, n), comb.view(*outer, n, n))
