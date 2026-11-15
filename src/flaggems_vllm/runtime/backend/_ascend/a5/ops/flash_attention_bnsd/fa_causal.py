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

"""Causal FlashAttention forward, split out from the dense kernel.

Why a separate kernel: once the diagonal block stopped being peeled out of the
KV loop (round 6), the mask became a plain row/column compare applied to the
one block that straddles the diagonal. That compare does not care whether the
block is square, so the `block_m == block_n` assertion the peeled version
needed is gone -- and with it the reason causal could not use a taller Q tile.

That matters because round 6 identified the gap as per-tile fixed cost (3.69x
vs vendor) against a near-parity loop body (1.12x). Halving the tile count by
doubling BLOCK_M attacks exactly that term, and it is the only way to do so
that Triton gives us: the per-tile pipeline prime/drain is compiler-generated
and the eight compile options that name it are no-ops (catalog 坑 8).

Shares `build_tile_order`, `_max_nan` and `_COMPILE_OPTS` with fa_naive; only
the kernel body and the tile-config choice are separate.
"""

import torch
import triton
import triton.language as tl

from .fa_naive import (_COMPILE_OPTS, _max_nan, build_tile_order,
                      check_bnsd, validate_shape)


@triton.jit
def _attn_fwd_causal(Q, K, V, Out, Tiles, ntiles, sm_scale,
                     sQb, sQh, sQs, sKb, sKh, sKs, sOb, sOh, sOs,
                     B, Hq, S,
                     GQA: tl.constexpr, NUM_CORES: tl.constexpr,
                     D: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, PRESCALED: tl.constexpr,
                     PEEL: tl.constexpr):
    """Causal only. One loop over KV blocks, no peeled diagonal.

    Every block including the diagonal one stays inside the pipelined loop;
    the blocks that straddle the diagonal take the mask under a runtime
    branch, which is the shape vendor uses (`if (needAtten)`).

    With BLOCK_M != BLOCK_N a Q tile spans several diagonal-crossing KV
    blocks, so the mask condition is the general one: absolute query row index
    against absolute key column index.
    """
    nblk_m = S // BLOCK_M
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_m0 = tl.arange(0, BLOCK_M)

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

        # The tile's last query row is its causal horizon: keys beyond it are
        # masked for every row, so the loop can stop there.
        hi = mi * BLOCK_M if PEEL else (mi + 1) * BLOCK_M
        # Keys strictly below this are visible to all BLOCK_M rows and need no
        # mask at all.
        full = mi * BLOCK_M
        for start_n in range(0, hi, BLOCK_N):
            kn = start_n + offs_n
            kb = tl.load(k_base + kn[:, None] * sKs + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kb))
            if not PRESCALED:
                qk = qk * sm_scale
            # Under PEEL the loop stops *before* the diagonal, so every block
            # here is fully visible and needs no mask at all. Without the
            # `not PEEL` guard the final iteration still satisfies
            # `start_n + BLOCK_N > full` (since hi == full) and pays for a
            # mask it does not need.
            if not PEEL and start_n + BLOCK_N > full:
                # The mask must be built from indices whose *pattern* does not
                # depend on `mi`. Using absolute `offs_m[:, None] >= kn` makes
                # the whole BLOCK_M x BLOCK_N predicate a function of `mi`, and
                # it is then rebuilt per block: measured 1.48-1.71x slower at
                # identical work (S=1024/2048/4096). Subtracting `full` keeps
                # the row index tile-relative; only the scalar column shift
                # `rel` varies, and when BLOCK_M == BLOCK_N it is always 0.
                # The mask is a true -inf substitution, not `qk + (-1e6)`. A
                # finite bias can be penetrated: with q·k·scale near +1e6 the
                # masked entry survives the subtraction and the row softmaxes
                # over a key it must not see. Reproduced with q=k=1000 (and
                # k[:, :, 0, :] = -1000, v[:, :, 0, :] = 0): the -1e6 form
                # returned 1.0 where vendor returns 0.0. `tl.where(cond, qk,
                # -inf)` is a pure substitution, so it is also correct if qk is
                # itself +inf (which `qk - inf` would turn into nan).
                #
                # Every row keeps at least its diagonal entry, so `m_ij` is
                # never -inf and `exp(-inf - finite) == 0`. Cost, measured
                # interleaved over 12 shapes: 1.0026x net, and *faster* at
                # S<=1024 because the select replaces the add.
                rel = start_n - full
                qk = tl.where(offs_m0[:, None] >= rel + offs_n[None, :],
                              qk, float("-inf"))
            m_ij = _max_nan(m_i, tl.reduce(qk, 1, _max_nan))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            # V after the Q@K^T dot, not alongside K: its MTE2 then overlaps
            # the cube work instead of queueing behind K's load.
            vb = tl.load(v_base + kn[:, None] * sKs + offs_d[None, :])
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(vb.dtype), vb, acc)
            m_i = m_ij

        if PEEL:
            # Straight-line diagonal block, requires BLOCK_M == BLOCK_N so the
            # pattern is the fixed lower triangle. Costs a per-tile body that
            # gets no pipeline overlap, which is why it only pays off at large
            # S -- see `_pick`.
            kn = mi * BLOCK_N + offs_n
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


# (block_m, block_n) candidates for causal, filled in by measurement below.
_CONFIGS = ((128, 128),)
_cache: dict = {}


def flash_attention_fwd_causal(q, k, v, block_m=None, block_n=None,
                               num_cores=32, **opts):
    check_bnsd(q, k, v)
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]

    if block_m is None or block_n is None:
        bm, bn = _pick(S)
        block_m = block_m or bm
        block_n = block_n or bn

    # Peel the diagonal block only where it pays (large S). See `_pick`.
    peel = bool(opts.pop("PEEL", S > 1024))
    if peel and block_m != block_n:
        # The peeled diagonal block is exactly one square KV block.
        peel = False

    validate_shape(q, k, v, block_m, block_n, causal=True)

    out = torch.empty_like(q)
    tiles = build_tile_order(B, Hq, S, block_m, True, q.device)

    sm_scale = (1.0 / D) ** 0.5
    prescale = S >= 4096
    qs = q * sm_scale if prescale else q
    if not qs.is_contiguous():
        # A fresh contiguous copy of q has the same layout, so this is only
        # reachable if the elementwise op ever returns something exotic.
        qs = qs.contiguous()

    launch_opts = dict(_COMPILE_OPTS)
    launch_opts.update(opts)
    # `qs`, not `q`: qs is what the kernel reads, and an elementwise launch may
    # hand back different strides (see validate_shape).
    _attn_fwd_causal[(num_cores,)](
        qs, k, v, out, tiles, tiles.numel(), sm_scale,
        qs.stride(0), qs.stride(1), qs.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, Hq, S,
        GQA=Hq // Hkv, NUM_CORES=num_cores, D=D,
        BLOCK_M=block_m, BLOCK_N=block_n, PRESCALED=prescale, PEEL=peel,
        **launch_opts,
    )
    return out


def _pick(S):
    """Tile config by S.

    Swept M/N over {64,128,256} x {64,128,256} at nine shapes
    (debug/dbg_causal_tiles.py): 128/128 wins or ties everywhere. Taller Q
    tiles would attack the per-tile cost directly, but BLOCK_M=256 overflows
    UB on this kernel (397568 B against 253952) -- the general row/column
    mask needs more live UB than the peeled version's fixed pattern, so
    de-peeling does not buy back enough to afford a taller tile. Wider N
    loses for the same reason round 4 found: 128/256 overflows too.

    So the only real knob here is PEEL, handled in flash_attention_fwd_causal.
    """
    return _cache.get(S, (128, 128))
