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

"""Dense (non-causal) FlashAttention forward, split out from fa_naive.

Separated from causal so the two can be tuned independently: the dense kernel
has no mask, no diagonal, and every Q tile walks the full S/BLOCK_N KV blocks,
so its tile-shape tradeoff is different. Round 4's sweep found dense alone
benefits from a wide-N tile (64/256) at large S, which causal cannot use.

Shares `build_tile_order`, `_max_nan` and `_COMPILE_OPTS` with fa_naive.
"""

import torch
import triton
import triton.language as tl

from .fa_naive import (_COMPILE_OPTS, _max_nan, build_tile_order,
                      check_bnsd, validate_shape)


@triton.jit
def _attn_fwd_dense(Q, K, V, Out, Tiles, ntiles, sm_scale,
                    sQb, sQh, sQs, sKb, sKh, sKs, sOb, sOh, sOs,
                    B, Hq, S,
                    GQA: tl.constexpr, NUM_CORES: tl.constexpr,
                    D: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr, PRESCALED: tl.constexpr):
    """No mask, no peel: every tile walks all S/BLOCK_N KV blocks."""
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

        for start_n in range(0, S, BLOCK_N):
            kn = start_n + offs_n
            kb = tl.load(k_base + kn[:, None] * sKs + offs_d[None, :])
            qk = tl.dot(q, tl.trans(kb))
            if not PRESCALED:
                qk = qk * sm_scale
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

        acc = acc / l_i[:, None]
        tl.store(o_base + offs_m[:, None] * sOs + offs_d[None, :],
                 acc.to(Out.dtype.element_ty))


# (block_m, block_n). 64/256 has the same 16384 elements as 128/128 but a 4x
# wider row, so the row max and row sum reduce over 256 contiguous elements
# with half as many rows. Round 4 measured it ahead only on large dense.
_CONFIGS = ((128, 128), (64, 256))
_cache: dict = {}


def flash_attention_fwd_dense(q, k, v, block_m=None, block_n=None,
                              num_cores=32, autotune=True, **opts):
    check_bnsd(q, k, v)
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]

    explicit = block_m is not None or block_n is not None
    if block_m is None:
        block_m = 128
    if block_n is None:
        block_n = 128

    validate_shape(q, k, v, block_m, block_n, causal=False)

    if autotune and not explicit and S >= 4096 and len(_CONFIGS) > 1:
        key = (B, Hq, Hkv, S, D, str(q.device))
        if key not in _cache:
            def _probe(bm, bn):
                return flash_attention_fwd_dense(
                    q, k, v, block_m=bm, block_n=bn, num_cores=num_cores,
                    autotune=False, **opts)
            _cache[key] = _tune(q, k, v, S, _probe)
        block_m, block_n = _cache[key]

    out = torch.empty_like(q)
    tiles = build_tile_order(B, Hq, S, block_m, False, q.device)

    sm_scale = (1.0 / D) ** 0.5
    prescale = S >= 4096
    qs = q * sm_scale if prescale else q
    if not qs.is_contiguous():
        qs = qs.contiguous()

    launch_opts = dict(_COMPILE_OPTS)
    launch_opts.update(opts)
    # `qs`, not `q`: see validate_shape.
    _attn_fwd_dense[(num_cores,)](
        qs, k, v, out, tiles, tiles.numel(), sm_scale,
        qs.stride(0), qs.stride(1), qs.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, Hq, S,
        GQA=Hq // Hkv, NUM_CORES=num_cores, D=D,
        BLOCK_M=block_m, BLOCK_N=block_n, PRESCALED=prescale,
        **launch_opts,
    )
    return out


def _tune(q, k, v, S, launch, warmup=3, rep=7):
    """Pick (block_m, block_n) for one shape by measuring both.

    Host-side rather than `@triton.autotune` because `Tiles` is built on the
    host and keyed to `block_m`: a decorator sweeping BLOCK_M as a
    `tl.constexpr` would hand the kernel a tile list built for a *different*
    block size, which is silently wrong rather than slow. `pre_hook` cannot
    rescue it either -- `ntiles` changes with `block_m`, so the fix needs a
    differently-sized positional tensor, not an in-place mutation.

    Gains are 1.03-1.07x, so the tuner must stay cheaper than what it wins:
    result cached per shape key, median of 7.
    """
    import time as _time
    best, best_ms = None, float("inf")
    for bm, bn in _CONFIGS:
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
            continue
        if ms < best_ms:
            best, best_ms = (bm, bn), ms
    return best or (128, 128)
