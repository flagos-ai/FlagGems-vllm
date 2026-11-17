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

"""DeepSeek-V4.1 attention output projection (``dsv41_o_proj``) on Hygon DCU.

As in vLLM, the precision follows ``wo_a``:

* FP8 E4M3 ``wo_a`` with FP32 128x128 scales: vLLM quantizes to FP8. Hygon BW cards
  have no FP8 tensor cores, so this version uses INT8 instead, as this repo's Hygon
  ``fp8_einsum`` does::

    attn_out[:, :n_heads]                       [T, H_pad, 512] bf16
      -> inverse GPT-J RoPE on the last 64 dims of every head
      -> per-128 INT8 activation quant with FP32 scales
      -> grouped INT8 GEMM with wo_a requantized to INT8 per 128x128 block
      -> z  [T, G * o_lora_rank] bf16, then wo_b_proj(z) if a callback is given

  The INT8 copy of ``wo_a`` is made by a Triton kernel on the first call and cached
  on the weight tensor that was passed in.

* BF16 ``wo_a`` without scales: nothing is quantized, as in vLLM's non-quantizing
  branch (and vllm_hcu's own path). The BF16 GEMM reads x straight from attn_out;
  only the 64 rotated dims of each head differ, and they are either rotated inside
  the GEMM (small batches, one launch) or written first to a small [T, H, 64] buffer
  (larger batches). Small batches also split the K loop over heads, with the same
  deterministic last-arriver sum as the INT8 fused path. If attn_out's heads are not
  512 apart, a full BF16 copy plus the general module's grouped GEMM is used instead.

The INT8 path has two variants, chosen by shape:
  * Fused: one kernel. Each output tile quantizes its activations on the fly; work is
    split over heads and the last program of a tile to finish sums the parts in a
    fixed order, so results are deterministic. Used for small batches.
  * Two kernels: quantize once, then the INT8 GEMM. Used for larger batches, where
    quantizing inside every output tile costs more than a second launch.

Both quantize identically. The INT8 result differs from exact activations and weights
by about 1.3% per (token, group) row on random data.

Autotune exemption: tile configurations are fixed from sweeps on BW1000, and compiled
kernels are launched directly. ``@libentry`` and ``@libtuner`` would add back the
CPU time per call that small batches cannot afford.
"""

import logging
from functools import lru_cache
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.dsv41_o_proj import (
    _bf16_grouped_gemm_kernel,
    _inv_rope_bf16_kernel,
)

logger = logging.getLogger(__name__)

_QUANT_BLOCK = 128
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_EPS = 1e-10
_MAX_TILES = 1 << 16
_COMPILED = {}


@triton.jit
def _quant_rows_int8(x, eps):
    # One FP32 scale per row; round half away from zero.
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
    q = x * (127.0 / amax)[:, None]
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    return q, amax * (1.0 / 127.0)


@triton.jit
def _load_chunk(
    head, c: tl.constexpr, cols, m_mask, rope_mask, is_rope, even, cos, sin
):
    # One 128-wide chunk of a head; the last chunk holds the 64 rotated dims.
    x = tl.load(head + c * 128 + cols[None, :], mask=m_mask[:, None], other=0.0).to(
        tl.float32
    )
    if c == 3:
        partner = tl.load(
            head + c * 128 + (cols ^ 1)[None, :], mask=rope_mask, other=0.0
        ).to(tl.float32)
        rotated = tl.where(
            even[None, :], x * cos + partner * sin, x * cos - partner * sin
        )
        x = tl.where(is_rope[None, :], rotated, x)
    return x


@triton.jit
def _weight_int8_kernel(
    w_ptr,
    s_ptr,
    q_ptr,
    qs_ptr,
    K: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    EPS: tl.constexpr,
):
    # One 128x128 block of the [G * N, K] weight per program.
    rb = tl.program_id(0).to(tl.int64)
    kb = tl.program_id(1)
    NKB: tl.constexpr = K // 128
    rows = rb * 128 + tl.arange(0, 128)
    cols = kb * 128 + tl.arange(0, 128)
    w = tl.load(w_ptr + rows[:, None] * K + cols[None, :]).to(tl.float32)
    if HAS_SCALE:
        w = w * tl.load(s_ptr + rb * NKB + kb)
    amax = tl.maximum(tl.max(tl.max(tl.abs(w), axis=1), axis=0), EPS)
    q = w * (127.0 / amax)
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + rows[:, None] * K + cols[None, :], q)
    tl.store(qs_ptr + rb * NKB + kb, amax * (1.0 / 127.0))


@triton.jit(do_not_specialize=["T", "T_pad"])
def _fused_kernel(
    o_ptr,
    pos_ptr,
    cache_ptr,
    w_ptr,
    ws_ptr,
    z_ptr,
    part_ptr,
    cnt_ptr,
    T,
    T_pad,
    o_sT: tl.constexpr,
    o_sH: tl.constexpr,
    cache_sP: tl.constexpr,
    z_sB: tl.constexpr,
    z_sM: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    HPG: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_gs = tl.program_id(2)
    g = (pid_gs // SPLIT).to(tl.int64)
    s = pid_gs % SPLIT
    K: tl.constexpr = HPG * 512
    NKB: tl.constexpr = K // 128
    HEADS: tl.constexpr = HPG // SPLIT
    rm = pid_m * BM + tl.arange(0, BM)
    m_mask = rm < T
    rn = pid_n * BN + tl.arange(0, BN)
    cols = tl.arange(0, 128)
    is_rope = cols >= ROPE_START
    rope_col = cols - ROPE_START
    cs = tl.maximum(rope_col >> 1, 0)
    even = (rope_col & 1) == 0
    rope_mask = m_mask[:, None] & is_rope[None, :]
    pos = tl.load(pos_ptr + rm, mask=m_mask, other=0)
    cache_row = cache_ptr + pos[:, None] * cache_sP
    cos = tl.load(cache_row + cs[None, :], mask=rope_mask, other=1.0)
    sin = tl.load(cache_row + HALF_ROPE + cs[None, :], mask=rope_mask, other=0.0)
    row_base = o_ptr + rm.to(tl.int64)[:, None] * o_sT + (g * HPG) * o_sH
    w_base = w_ptr + g * N * K + rn[None, :] * K + cols[:, None]
    ws_base = ws_ptr + (g * (N // 128) + (pid_n * BN) // 128) * NKB
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for hh in range(HEADS):
        h = s * HEADS + hh
        head = row_base + h * o_sH
        for c in tl.static_range(4):
            x = _load_chunk(head, c, cols, m_mask, rope_mask, is_rope, even, cos, sin)
            q, scale = _quant_rows_int8(x, EPS)
            kb = h * 4 + c
            w = tl.load(w_base + kb * 128)
            part = tl.dot(q, w, out_dtype=tl.int32).to(tl.float32)
            acc += part * (scale[:, None] * tl.load(ws_base + kb))
    z_ptrs = z_ptr + g * z_sB + rm[:, None] * z_sM + rn[None, :]
    if SPLIT == 1:
        tl.store(z_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])
    else:
        # Park this split's tile; the last of the SPLIT programs to arrive sums all of
        # them in split order and resets the counter for the next launch.
        rows = (s * G + g) * T_pad + rm
        tl.store(part_ptr + rows.to(tl.int64)[:, None] * N + rn[None, :], acc)
        tl.debug_barrier()
        tile = (g * (T_pad // BM) + pid_m) * (N // BN) + pid_n
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if ticket == SPLIT - 1:
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s2 in tl.static_range(SPLIT):
                rows2 = (s2 * G + g) * T_pad + rm
                total += tl.load(
                    part_ptr + rows2.to(tl.int64)[:, None] * N + rn[None, :]
                )
            tl.store(z_ptrs, total.to(tl.bfloat16), mask=m_mask[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0)


@triton.jit(do_not_specialize=["T"])
def _quant_kernel(
    o_ptr,
    pos_ptr,
    cache_ptr,
    x_ptr,
    xs_ptr,
    T,
    o_sT: tl.constexpr,
    o_sH: tl.constexpr,
    cache_sP: tl.constexpr,
    HPG: tl.constexpr,
    BT: tl.constexpr,
    HB: tl.constexpr,
    NOPE: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    EPS: tl.constexpr,
):
    # BT tokens x HB heads per program, a whole 512-wide head per load.
    tb = tl.program_id(0)
    hb = tl.program_id(1).to(tl.int64)
    K: tl.constexpr = HPG * 512
    NKB: tl.constexpr = K // 128
    rm = tb * BT + tl.arange(0, BT)
    m_mask = rm < T
    cols = tl.arange(0, 512)
    is_rope = cols >= NOPE
    rope_col = cols - NOPE
    cs = tl.maximum(rope_col >> 1, 0)
    even = (rope_col & 1) == 0
    rope_mask = m_mask[:, None] & is_rope[None, :]
    pos = tl.load(pos_ptr + rm, mask=m_mask, other=0)
    cache_row = cache_ptr + pos[:, None] * cache_sP
    cos = tl.load(cache_row + cs[None, :], mask=rope_mask, other=1.0)
    sin = tl.load(cache_row + HALF_ROPE + cs[None, :], mask=rope_mask, other=0.0)
    row_o = o_ptr + rm.to(tl.int64)[:, None] * o_sT
    blocks = tl.arange(0, 4)
    for i in tl.static_range(HB):
        head_id = hb * HB + i
        g = head_id // HPG
        h = head_id % HPG
        base = row_o + head_id * o_sH
        x = tl.load(base + cols[None, :], mask=m_mask[:, None], other=0.0).to(
            tl.float32
        )
        partner = tl.load(base + (cols ^ 1)[None, :], mask=rope_mask, other=0.0).to(
            tl.float32
        )
        rotated = tl.where(
            even[None, :], x * cos + partner * sin, x * cos - partner * sin
        )
        x = tl.where(is_rope[None, :], rotated, x)
        # Same quantization as _quant_rows_int8, on 4 chunks of 128 at once.
        amax = tl.maximum(tl.max(tl.reshape(tl.abs(x), (BT * 4, 128)), axis=1), EPS)
        inv = tl.reshape(
            tl.broadcast_to(tl.reshape(127.0 / amax, (BT * 4, 1)), (BT * 4, 128)),
            (BT, 512),
        )
        q = x * inv
        q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int32)
        q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
        rows = g * T + rm.to(tl.int64)
        tl.store(
            x_ptr + rows[:, None] * K + h * 512 + cols[None, :], q, mask=m_mask[:, None]
        )
        scale = tl.reshape(amax * (1.0 / 127.0), (BT, 4))
        tl.store(
            xs_ptr + rows[:, None] * NKB + h * 4 + blocks[None, :],
            scale,
            mask=m_mask[:, None],
        )


@triton.jit(do_not_specialize=["T"])
def _gemm_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    z_ptr,
    T,
    z_sB: tl.constexpr,
    z_sM: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.program_id(1).to(tl.int64)
    NKB: tl.constexpr = K // 128
    num_m = tl.cdiv(T, BM)
    num_n: tl.constexpr = N // BN
    first_m = pid // (GROUP_M * num_n) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    local = pid % (GROUP_M * num_n)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BM + tl.arange(0, BM)
    m_mask = rm < T
    # Each 128-column half of the tile has one weight scale per K block, so a
    # 256-wide tile runs as two halves sharing the activation tile.
    HALF: tl.constexpr = BN if BN <= 128 else 128
    rn0 = pn * BN + tl.arange(0, HALF)
    cols = tl.arange(0, 128)
    rows = g * T + rm.to(tl.int64)
    a_ptrs = x_ptr + rows[:, None] * K + cols[None, :]
    b0_ptrs = w_ptr + g * N * K + rn0[None, :] * K + cols[:, None]
    b1_ptrs = b0_ptrs + 128 * K
    xs_ptrs = xs_ptr + rows * NKB
    ws0_ptr = ws_ptr + (g * (N // 128) + (pn * BN) // 128) * NKB
    ws1_ptr = ws0_ptr + NKB
    acc0 = tl.zeros((BM, HALF), dtype=tl.float32)
    acc1 = tl.zeros((BM, HALF), dtype=tl.float32)
    for kk in range(NKB // UNROLL):
        for uu in tl.static_range(UNROLL):
            kb = kk * UNROLL + uu
            if EVEN_M:
                a = tl.load(a_ptrs + kb * 128)
                xsv = tl.load(xs_ptrs + kb)
            else:
                a = tl.load(a_ptrs + kb * 128, mask=m_mask[:, None], other=0)
                xsv = tl.load(xs_ptrs + kb, mask=m_mask, other=0.0)
            part0 = tl.dot(a, tl.load(b0_ptrs + kb * 128), out_dtype=tl.int32)
            acc0 = tl.fma(
                part0.to(tl.float32), xsv[:, None] * tl.load(ws0_ptr + kb), acc0
            )
            if BN > 128:
                part1 = tl.dot(a, tl.load(b1_ptrs + kb * 128), out_dtype=tl.int32)
                acc1 = tl.fma(
                    part1.to(tl.float32), xsv[:, None] * tl.load(ws1_ptr + kb), acc1
                )
    z_ptrs = z_ptr + g * z_sB + rm[:, None] * z_sM + rn0[None, :]
    if EVEN_M:
        tl.store(z_ptrs, acc0.to(tl.bfloat16))
        if BN > 128:
            tl.store(z_ptrs + 128, acc1.to(tl.bfloat16))
    else:
        tl.store(z_ptrs, acc0.to(tl.bfloat16), mask=m_mask[:, None])
        if BN > 128:
            tl.store(z_ptrs + 128, acc1.to(tl.bfloat16), mask=m_mask[:, None])


_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_raw_stream(index):
    # The raw handle costs ~0.3 us; torch.cuda.current_stream() costs ~5.7 us.
    if _RAW_STREAM is not None:
        return _RAW_STREAM(index)
    return torch.cuda.current_stream(index).cuda_stream


class _Kernel:
    """One kernel launch with a fixed grid, compiled through Triton once.

    The first call goes through Triton, which compiles. Later calls, and other grids of
    the same compiled kernel, call it directly and skip Triton's per-call argument
    handling. ``key`` must hold everything that selects the compiled kernel;
    ``constexprs`` must follow the kernel's parameter order. ``opts`` are fixed compile
    options (one set per ``name``) such as ``kpack``.
    """

    __slots__ = ("kernel", "grid", "full_key", "constexprs", "cvals", "opts", "runner")

    def __init__(
        self, name, kernel, grid, key, constexprs, num_warps, num_stages, **opts
    ):
        self.kernel = kernel
        self.grid = grid
        self.full_key = (name,) + key
        self.constexprs = constexprs
        self.cvals = tuple(constexprs.values())
        self.opts = dict(opts, num_warps=num_warps, num_stages=num_stages)
        compiled = _COMPILED.get(self.full_key)
        self.runner = None if compiled is None else compiled[grid]

    def __call__(self, *args):
        runner = self.runner
        if runner is not None:
            runner(*args, *self.cvals)
            return
        compiled = _COMPILED.get(self.full_key)
        if compiled is None:
            compiled = self.kernel[self.grid](*args, **self.constexprs, **self.opts)
            _COMPILED[self.full_key] = compiled
            self.runner = compiled[self.grid]
            return
        self.runner = compiled[self.grid]
        self.runner(*args, *self.cvals)


_COUNTERS = {}


def _counters_for(device, raw_stream):
    # One set per stream: launches on a stream never overlap. Counters return to zero.
    key = (device, raw_stream)
    counters = _COUNTERS.get(key)
    if counters is None:
        counters = _COUNTERS[key] = torch.zeros(
            _MAX_TILES, dtype=torch.int32, device=device
        )
    return counters


def _tile_counters(stream, device):
    return _counters_for(device, stream.cuda_stream)


def _largest_divisor(n, choices):
    return next(c for c in choices if n % c == 0)


@lru_cache(maxsize=4096)
def _plan(T, G, HPG, N):
    """Path and tile configuration for one shape, from sweeps on BW1000."""
    if T <= 16 or (G == 1 and T <= 256):
        BM = 16
        BN = 64 if (G == 1 and T <= 16) else 128
        base = G * triton.cdiv(T, BM) * (N // BN)
        target = 256 if G == 1 else 128
        split = 1
        while split < HPG and base * split < target and HPG % (split * 2) == 0:
            split *= 2
        return ("fused", BM, BN, split, 4)
    if T <= 1024:
        quant = (1, _largest_divisor(G * HPG, (4 if G == 1 else 8, 4, 2, 1)), 1)
    else:
        quant = (2 if G <= 8 else 8, _largest_divisor(G * HPG, (8, 4, 2, 1)))
        quant += (2 if G <= 8 else 8,)
    if G == 1 and T > 256:
        if T >= 2048:
            gemm = (128, 128, 4)
        elif T >= 1024:
            gemm = (64, 128, 4)
        elif T >= 512:
            gemm = (64, 64, 8)
        else:
            gemm = (32, 128, 4)
    elif T > 256:
        # 128-row tiles only when they fill up; T=257 would waste most of one.
        gemm = (128, 256, 8) if T % 128 == 0 else (64, 256, 4)
    elif T > 128:
        gemm = (64, 256, 4)
    elif T > 64:
        gemm = (64, 128, 4)
    elif T > 32:
        gemm = (64, 64, 8)
    else:
        gemm = (32, 128, 4)
    if N % gemm[1]:
        gemm = (gemm[0], 128, gemm[2])
    return ("two", quant, gemm)


def _is_contiguous(shape, stride):
    if 0 in shape:
        return True
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size != 1 and step != expected:
            return False
        expected *= size
    return True


def _int8_weight(weight, weight_scale, n_groups, n, k):
    """INT8 copy of wo_a with FP32 128x128 scales, cached on the caller's tensor.

    The cache follows the tensors' version counters, so an in-place update of the
    weight or its scale makes a fresh copy.
    """
    key = (
        weight.data_ptr(),
        weight._version,
        weight_scale.data_ptr(),
        weight_scale._version,
    )
    cached = getattr(weight, "_flaggems_dsv41_o_proj_int8", None)
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    q = torch.empty((n_groups, n, k), dtype=torch.int8, device=weight.device)
    qs = torch.empty(
        (n_groups, n // _QUANT_BLOCK, k // _QUANT_BLOCK),
        dtype=torch.float32,
        device=weight.device,
    )
    _weight_int8_kernel[(n_groups * n // _QUANT_BLOCK, k // _QUANT_BLOCK)](
        weight,
        weight_scale,
        q,
        qs,
        K=k,
        HAS_SCALE=True,
        EPS=_EPS,
        num_warps=4,
        num_stages=1,
    )
    weight._flaggems_dsv41_o_proj_int8 = (key, q, qs)
    return q, qs


def _build_fused(plan, device, T, G, N, HPG, o_sT, o_sH, cache_sP, pos_dtype, aligns):
    """Launch recipe of the fused INT8 kernel for one input signature."""
    _, BM, BN, split, num_warps = plan
    T_pad = triton.cdiv(T, BM) * BM
    grid = (N // BN, T_pad // BM, G * split)
    if grid[0] * grid[1] * G > _MAX_TILES:
        raise NotImplementedError("dsv41_o_proj: too many output tiles for split-K")
    constexprs = dict(
        o_sT=o_sT,
        o_sH=o_sH,
        cache_sP=cache_sP,
        z_sB=N,
        z_sM=G * N,
        G=G,
        N=N,
        HPG=HPG,
        SPLIT=split,
        BM=BM,
        BN=BN,
        ROPE_START=_NOPE_DIM % _QUANT_BLOCK,
        HALF_ROPE=_ROPE_DIM // 2,
        EPS=_EPS,
    )
    key = (device, pos_dtype, *aligns, num_warps, *constexprs.values())
    launch = _Kernel("fused", _fused_kernel, grid, key, constexprs, num_warps, 1)
    part_shape = (split * G * T_pad, N) if split > 1 else None
    index = device.index

    def run(o, positions, cos_sin_cache, wq, wqs, out):
        if part_shape is None:  # part and counters are unused without a split
            launch(o, positions, cos_sin_cache, wq, wqs, out, out, out, T, T_pad)
            return
        part = torch.empty(part_shape, dtype=torch.float32, device=device)
        counters = _counters_for(device, _current_raw_stream(index))
        launch(o, positions, cos_sin_cache, wq, wqs, out, part, counters, T, T_pad)

    return run


def _build_two(plan, device, T, G, N, HPG, o_sT, o_sH, cache_sP, pos_dtype, aligns):
    """Launch recipe of the quant kernel + INT8 GEMM for one input signature."""
    _, (BT, HB, q_warps), (BM, BN, g_warps) = plan
    o_al, p_al, c_al, z_al = aligns
    K = HPG * _HEAD_DIM
    q_constexprs = dict(
        o_sT=o_sT,
        o_sH=o_sH,
        cache_sP=cache_sP,
        HPG=HPG,
        BT=BT,
        HB=HB,
        NOPE=_NOPE_DIM,
        HALF_ROPE=_ROPE_DIM // 2,
        EPS=_EPS,
    )
    q_key = (device, pos_dtype, o_al, p_al, c_al, q_warps, *q_constexprs.values())
    quant = _Kernel(
        "quant",
        _quant_kernel,
        (triton.cdiv(T, BT), G * HPG // HB, 1),
        q_key,
        q_constexprs,
        q_warps,
        1,
    )
    g_constexprs = dict(
        z_sB=N,
        z_sM=G * N,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        GROUP_M=4,
        EVEN_M=T % BM == 0,
        # K blocks per loop iteration: unrolling helps the small tiles of small
        # batches (-5% to -14%) and slows the 128x256 tiles of large ones (+6-9%).
        UNROLL=4 if T <= 128 else 1,
    )
    g_key = (device, z_al, g_warps, *g_constexprs.values())
    gemm = _Kernel(
        "gemm",
        _gemm_kernel,
        (triton.cdiv(T, BM) * (N // BN), G, 1),
        g_key,
        g_constexprs,
        g_warps,
        1,
        kpack=2,  # 1-4% faster on every shape tried; output is bit-identical
    )
    x_shape = (G * T, K)
    xs_shape = (G * T, K // _QUANT_BLOCK)

    def run(o, positions, cos_sin_cache, wq, wqs, out):
        x = torch.empty(x_shape, dtype=torch.int8, device=device)
        xs = torch.empty(xs_shape, dtype=torch.float32, device=device)
        quant(o, positions, cos_sin_cache, x, xs, T)
        gemm(x, xs, wq, wqs, out, T)

    return run


@triton.jit
def _rotate_rope_bf16(x, cos, sin, ROWS: tl.constexpr, ROPE: tl.constexpr):
    # Inverse GPT-J RoPE of fp32 [ROWS, ROPE] with cos/sin [ROWS, ROPE // 2], to bf16.
    xe, xo = tl.split(tl.reshape(x, (ROWS, ROPE // 2, 2)))
    y = tl.join(xe * cos + xo * sin, xo * cos - xe * sin)
    return tl.reshape(y, (ROWS, ROPE)).to(tl.bfloat16)


@triton.jit(do_not_specialize=["T"])
def _bf16_rope_tail_kernel(
    o_ptr,
    pos_ptr,
    cache_ptr,
    r_ptr,
    T,
    o_sT,
    o_sH,
    cache_sP,
    NH: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    BT: tl.constexpr,
    HB: tl.constexpr,
):
    # r[t, h, :] = inverse RoPE of attn_out[t, h, NOPE:]; only these 64 dims change.
    rm = tl.program_id(0) * BT + tl.arange(0, BT)
    m_mask = rm < T
    rr = tl.arange(0, ROPE)
    rh = tl.arange(0, ROPE // 2)
    pos = tl.load(pos_ptr + rm, mask=m_mask, other=0)
    cache_row = cache_ptr + pos[:, None] * cache_sP
    cos = tl.load(cache_row + rh[None, :], mask=m_mask[:, None], other=1.0)
    sin = tl.load(cache_row + ROPE // 2 + rh[None, :], mask=m_mask[:, None], other=0.0)
    rows = rm.to(tl.int64)[:, None]
    for i in tl.static_range(HB):
        h = tl.program_id(1).to(tl.int64) * HB + i
        x = tl.load(
            o_ptr + rows * o_sT + h * o_sH + NOPE + rr[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        y = _rotate_rope_bf16(x, cos, sin, BT, ROPE)
        tl.store(r_ptr + (rows * NH + h) * ROPE + rr[None, :], y, mask=m_mask[:, None])


@triton.jit(do_not_specialize=["T", "T_pad"])
def _bf16_gemm_kernel(
    o_ptr,
    r_ptr,
    pos_ptr,
    cache_ptr,
    w_ptr,
    z_ptr,
    part_ptr,
    cnt_ptr,
    T,
    T_pad,
    o_sT,
    cache_sP,
    z_sB,
    z_sM,
    G: tl.constexpr,
    N: tl.constexpr,
    HPG: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    FUSE_ROPE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    # z[g] = x[:, g] @ wo_a[g]^T, reading x straight from attn_out: a group's heads are
    # contiguous (head stride 512), so x[t, g, :] is one row of K = HPG * 512 values.
    # Per head, the unrotated columns come from attn_out and the last 64 from the
    # rotated tail: computed here (FUSE_ROPE, small batches) or read from r.
    pid = tl.program_id(0)
    g = (tl.program_id(1) // SPLIT).to(tl.int64)
    s = tl.program_id(1) % SPLIT
    HEAD: tl.constexpr = NOPE + ROPE
    K: tl.constexpr = HPG * HEAD
    SUBS: tl.constexpr = HEAD // BK
    TAIL: tl.constexpr = NOPE - (SUBS - 1) * BK  # unrotated columns of the last block
    HEADS_PER: tl.constexpr = HPG // SPLIT
    num_m = T_pad // BM
    num_n: tl.constexpr = N // BN
    first_m = pid // (GROUP_M * num_n) * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    local = pid % (GROUP_M * num_n)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BM + tl.arange(0, BM)
    m_mask = rm < T
    if EVEN_M:
        m_mask = tl.full((BM,), 1, tl.int1)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rr = tl.arange(0, ROPE)
    o_rows = o_ptr + rm.to(tl.int64)[:, None] * o_sT + g * K
    w_rows = w_ptr + g * N * K + rn[None, :] * K
    if FUSE_ROPE:
        pos = tl.load(pos_ptr + rm, mask=m_mask, other=0)
        cache_row = cache_ptr + pos[:, None] * cache_sP
        rh = tl.arange(0, ROPE // 2)
    else:
        r_rows = r_ptr + (rm.to(tl.int64)[:, None] * (G * HPG) + g * HPG) * ROPE
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # UNROLL heads per iteration let the compiler overlap one head's loads with the
    # previous head's dots; this kernel is latency-bound per workgroup.
    for hh in range(HEADS_PER // UNROLL):
        for uu in tl.static_range(UNROLL):
            head = s * HEADS_PER + hh * UNROLL + uu
            k0 = head * HEAD
            for j in tl.static_range(SUBS - 1):
                a = tl.load(
                    o_rows + k0 + j * BK + rk[None, :], mask=m_mask[:, None], other=0.0
                )
                acc = tl.dot(a, tl.load(w_rows + (k0 + j * BK + rk)[:, None]), acc)
            kt = k0 + (SUBS - 1) * BK
            if TAIL > 0:
                rt = tl.arange(0, TAIL)
                a = tl.load(o_rows + kt + rt[None, :], mask=m_mask[:, None], other=0.0)
                acc = tl.dot(a, tl.load(w_rows + (kt + rt)[:, None]), acc)
            if FUSE_ROPE:
                x = tl.load(
                    o_rows + kt + TAIL + rr[None, :], mask=m_mask[:, None], other=0.0
                ).to(tl.float32)
                cos = tl.load(cache_row + rh[None, :], mask=m_mask[:, None], other=1.0)
                sin = tl.load(
                    cache_row + ROPE // 2 + rh[None, :], mask=m_mask[:, None], other=0.0
                )
                a = _rotate_rope_bf16(x, cos, sin, BM, ROPE)
            else:
                a = tl.load(
                    r_rows + head * ROPE + rr[None, :], mask=m_mask[:, None], other=0.0
                )
            acc = tl.dot(a, tl.load(w_rows + (kt + TAIL + rr)[:, None]), acc)
    z_ptrs = z_ptr + g * z_sB + rm[:, None] * z_sM + rn[None, :]
    if SPLIT == 1:
        tl.store(z_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])
    else:
        # Split over heads: park the FP32 part; the last program of the tile sums the
        # parts in split order (deterministic) and resets the counter.
        rows = (s * G + g) * T_pad + rm
        tl.store(part_ptr + rows.to(tl.int64)[:, None] * N + rn[None, :], acc)
        tl.debug_barrier()
        tile = (g * num_m + pm) * num_n + pn
        ticket = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel")
        if ticket == SPLIT - 1:
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s2 in tl.static_range(SPLIT):
                r2 = (s2 * G + g) * T_pad + rm
                total += tl.load(part_ptr + r2.to(tl.int64)[:, None] * N + rn[None, :])
            tl.store(z_ptrs, total.to(tl.bfloat16), mask=m_mask[:, None])
            tl.atomic_xchg(cnt_ptr + tile, 0)


def _bf16_gemm_config(T, G):
    """(BM, BN, BK, num_warps, num_stages) for the copy-path BF16 GEMM.

    Used only when attn_out's heads are not contiguous. Every choice keeps
    (BM * BK + BK * BN) * 2 bytes * stages within the 64 KB LDS.
    """
    if G == 1:
        return (32, 64, 128, 4, 2) if T <= 128 else (128, 128, 64, 8, 2)
    if T <= 32:
        return 32, 64, 128, 4, 2
    if T <= 64:
        return 64, 64, 128, 4, 2
    if T <= 128:
        return 64, 128, 128, 4, 1
    if T < 512:
        return 64, 256, 64, 8, 1
    return 128, 256, 64, 8, 1


@lru_cache(maxsize=4096)
def _bf16_plan(T, G, HPG, N):
    """(fuse_rope, rope-tail config, GEMM tile) for the BF16 path, from BW1000 sweeps.

    rope-tail config: (tokens, heads per program, warps). GEMM tile: (BM, BN, BK,
    warps, stages, split over heads, kpack). A fused rope saves a launch but slows
    big tiles, so only small batches use it.
    """
    fuse = T <= 64 or (T <= 128 and G <= 8)
    if G == 1:
        if T <= 16:
            tile = (16, 64, 64, 4, 1, 8, 1)
        elif T <= 32:
            tile = (32, 64, 128, 4, 1, 8, 1)
        elif T <= 128:
            tile = (64, 128, 64, 4, 2, 8, 1)
        else:
            tile = (64, 128, 128, 4, 1, 1, 2)
    elif T <= 16:
        tile = (16, 128, 128, 4, 1, 1, 1)
    elif T <= 32:
        tile = (32, 128, 128, 4, 1, 2, 1) if G <= 8 else (32, 128, 128, 8, 1, 1, 1)
    elif T <= 64:
        tile = (64, 128, 128, 4, 1, 2, 1)
    elif T <= 128:
        tile = (128, 128, 128, 8, 1, 1, 1)
    elif T < 512 or T % 128:
        # 128-row tiles only when they fill up; T=257 would waste most of one.
        tile = (64, 256, 64, 4, 1, 1, 2)
    else:
        tile = (128, 256, 64, 8, 1, 1, 2)
    BM, BN, BK, warps, stages, split, kpack = tile
    while HPG % split:
        split //= 2
    if N % BN:
        BN = 128
    heads = 1 if fuse else (4 if T <= 128 else 8)
    while (G * HPG) % heads:
        heads //= 2
    rope = (4, heads, 2)
    return fuse, rope, (BM, BN, BK, warps, stages, split, kpack)


def _build_bf16(device, T, G, N, HPG, o_sT, o_sH, cache_sP, pos_dtype, aligns):
    """Launch recipe of the non-quantizing path; heads must be contiguous (stride 512)."""
    fuse, (BT, HB, rope_warps), tile = _bf16_plan(T, G, HPG, N)
    BM, BN, BK, num_warps, num_stages, split, kpack = tile
    o_al, p_al, c_al, z_al = aligns
    T_pad = triton.cdiv(T, BM) * BM
    grid = (T_pad // BM * (N // BN), G * split, 1)
    if split > 1 and grid[0] * G > _MAX_TILES:
        raise NotImplementedError("dsv41_o_proj: too many output tiles for split-K")
    n_heads = G * HPG
    rope = None
    if not fuse:
        r_constexprs = dict(
            o_sT=o_sT,
            o_sH=o_sH,
            cache_sP=cache_sP,
            NH=n_heads,
            NOPE=_NOPE_DIM,
            ROPE=_ROPE_DIM,
            BT=BT,
            HB=HB,
        )
        r_key = (
            device,
            pos_dtype,
            o_al,
            p_al,
            c_al,
            rope_warps,
            *r_constexprs.values(),
        )
        rope = _Kernel(
            "bf16_rope",
            _bf16_rope_tail_kernel,
            (triton.cdiv(T, BT), n_heads // HB, 1),
            r_key,
            r_constexprs,
            rope_warps,
            1,
        )
    # Unrolling heads pays off for the plain GEMM; with the rope fused in it lost a little.
    unroll = 1 if fuse else next(u for u in (4, 2, 1) if (HPG // split) % u == 0)
    constexprs = dict(
        o_sT=o_sT,
        cache_sP=cache_sP,
        z_sB=N,
        z_sM=G * N,
        G=G,
        N=N,
        HPG=HPG,
        NOPE=_NOPE_DIM,
        ROPE=_ROPE_DIM,
        BM=BM,
        BN=BN,
        BK=BK,
        SPLIT=split,
        GROUP_M=8,
        EVEN_M=T % BM == 0,
        FUSE_ROPE=fuse,
        UNROLL=unroll,
    )
    key = (
        device,
        pos_dtype,
        o_al,
        p_al,
        c_al,
        z_al,
        num_warps,
        num_stages,
        kpack,
        *constexprs.values(),
    )
    gemm = _Kernel(
        "bf16_gemm",
        _bf16_gemm_kernel,
        grid,
        key,
        constexprs,
        num_warps,
        num_stages,
        kpack=kpack,
    )
    r_shape = (T, n_heads, _ROPE_DIM)
    part_shape = (split * G * T_pad, N) if split > 1 else None
    index = device.index

    def run(o, positions, cos_sin_cache, w, _unused, out):
        r = o  # unused when the rope is fused into the GEMM
        if rope is not None:
            r = torch.empty(r_shape, dtype=torch.bfloat16, device=device)
            rope(o, positions, cos_sin_cache, r, T)
        if part_shape is None:  # part and counters are unused without a split
            gemm(o, r, positions, cos_sin_cache, w, out, out, out, T, T_pad)
            return
        part = torch.empty(part_shape, dtype=torch.float32, device=device)
        counters = _counters_for(device, _current_raw_stream(index))
        gemm(o, r, positions, cos_sin_cache, w, out, part, counters, T, T_pad)

    return run


def _run_bf16_copy(o, positions, cos_sin_cache, w3, z):
    """Fallback when attn_out's heads are not contiguous: BF16 inverse-RoPE copy,
    then the general module's grouped BF16 GEMM."""
    G, T, N = z.shape
    n_heads = o.shape[1]
    K = w3.shape[2]
    x = torch.empty((T, n_heads, _HEAD_DIM), dtype=torch.bfloat16, device=o.device)
    BT, rope_warps = _BF16_ROPE_CONFIG
    _inv_rope_bf16_kernel[(triton.cdiv(T, BT), n_heads)](
        o,
        positions,
        cos_sin_cache,
        x,
        T,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        NH=n_heads,
        HEAD_DIM=_HEAD_DIM,
        NOPE=_NOPE_DIM,
        HALF_ROPE=_ROPE_DIM // 2,
        BT=BT,
        num_warps=rope_warps,
    )
    BM, BN, BK, num_warps, num_stages = _bf16_gemm_config(T, G)
    if N % BN:
        BN = 64
    _bf16_grouped_gemm_kernel[(triton.cdiv(T, BM) * (N // BN), G)](
        x,
        w3,
        z,
        T,
        z.stride(0),
        z.stride(1),
        G=G,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        BK=BK,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )


_BF16_ROPE_CONFIG = (2, 4)  # tokens per program, warps


_EMPTY, _INT8, _BF16, _BF16_COPY = range(4)


@lru_cache(maxsize=4096)
def _call_plan(
    a_shape,
    a_stride,
    a_dev,
    p_shape,
    p_dtype,
    p_dev,
    c_shape,
    c_stride,
    c_dtype,
    c_dev,
    w_shape,
    w_stride,
    w_dtype,
    w_dev,
    s_sig,
    out_sig,
    n_groups,
    heads_per_group,
    o_lora_rank,
    aligns,
):
    """Validate one input signature and build its launch recipe.

    Everything checked or derived here depends only on shapes, strides, dtypes,
    devices, scalar arguments and pointer alignment, so a call with a signature seen
    before skips all of it. Invalid inputs raise and are never cached. Returns
    ``(kind, run, out_shape)``.
    """
    num_tokens, num_heads_padded, head_dim = a_shape
    n_heads = n_groups * heads_per_group
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim must be {_HEAD_DIM}, got {head_dim}")
    if num_heads_padded < n_heads:
        raise ValueError(
            f"attn_out has {num_heads_padded} heads, need n_groups*heads_per_group={n_heads}"
        )
    if a_stride[-1] != 1:
        raise ValueError("attn_out must be contiguous in head_dim")
    if o_lora_rank % _QUANT_BLOCK:
        raise NotImplementedError("o_lora_rank must be a multiple of 128")
    if tuple(p_shape) != (num_tokens,) or p_dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be [num_tokens] int32 or int64")
    if (
        len(c_shape) != 2
        or c_shape[-1] != _ROPE_DIM
        or c_dtype != torch.float32
        or c_stride[-1] != 1
    ):
        raise ValueError(f"cos_sin_cache must be [max_position, {_ROPE_DIM}] float32")

    if w_dtype == torch.float8_e4m3fn:
        if s_sig is None or s_sig[2] != torch.float32:
            raise NotImplementedError(
                "dsv41_o_proj: FP8 wo_a needs FP32 128x128 block scales"
            )
    elif w_dtype == torch.bfloat16:
        if s_sig is not None:
            raise ValueError("dsv41_o_proj: BF16 wo_a takes no weight_scale")
    else:
        raise NotImplementedError("dsv41_o_proj: wo_a must be FP8 E4M3 or BF16")
    devices = [p_dev, c_dev, w_dev] + ([] if s_sig is None else [s_sig[3]])
    if any(d != a_dev for d in devices):
        raise ValueError("all inputs must be on the same device")

    group_dim = heads_per_group * head_dim
    w_shape3 = (n_groups, o_lora_rank, group_dim)
    w_dims = tuple(w_shape)
    if w_dims != (n_groups * o_lora_rank, group_dim) and w_dims != w_shape3:
        raise ValueError(f"wo_a weight must be {w_shape3} or its 2D form, got {w_dims}")
    if not _is_contiguous(w_shape, w_stride):
        raise ValueError("wo_a weight must be contiguous")
    if s_sig is not None:
        s_dims = tuple(s_sig[0])
        s_shape3 = (n_groups, o_lora_rank // _QUANT_BLOCK, group_dim // _QUANT_BLOCK)
        if s_dims != (s_shape3[0] * s_shape3[1], s_shape3[2]) and s_dims != s_shape3:
            raise ValueError(
                f"wo_a weight_scale must be {s_shape3} or its 2D form, got {s_dims}"
            )
        if not _is_contiguous(s_sig[0], s_sig[1]):
            raise ValueError("wo_a weight_scale must be contiguous")

    out_shape = (num_tokens, n_groups * o_lora_rank)
    if out_sig is not None and (
        tuple(out_sig[0]) != out_shape
        or out_sig[2] != torch.bfloat16
        or out_sig[3] != a_dev
        or not _is_contiguous(out_sig[0], out_sig[1])
    ):
        raise ValueError(f"out must be a contiguous {out_shape} bfloat16 on {a_dev}")
    if num_tokens == 0:
        return _EMPTY, None, out_shape

    common = (
        a_dev,
        num_tokens,
        n_groups,
        o_lora_rank,
        heads_per_group,
        a_stride[0],
        a_stride[1],
        c_stride[0],
        p_dtype,
        aligns,
    )
    if w_dtype == torch.bfloat16:
        if a_stride[1] == _HEAD_DIM:
            return _BF16, _build_bf16(*common), out_shape
        return _BF16_COPY, None, out_shape
    plan = _plan(num_tokens, n_groups, heads_per_group, o_lora_rank)
    build = _build_fused if plan[0] == "fused" else _build_two
    return _INT8, build(plan, *common), out_shape


def dsv41_o_proj(
    attn_out: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_a_weight_scale: Optional[torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    o_lora_rank: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    out: Optional[torch.Tensor] = None,
    wo_b_proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Inverse RoPE + grouped wo_a + optional wo_b on Hygon DCU.

    Same contract as the general ``dsv41_o_proj``: an FP8 wo_a runs in INT8 here, a
    BF16 wo_a (``wo_a_weight_scale=None``) is not quantized.

    Args:
        attn_out: [num_tokens, num_heads_padded, 512], bf16. Only the first
            ``n_groups * heads_per_group`` heads are used.
        positions: [num_tokens] integer positions.
        cos_sin_cache: [max_position, 64] float32, laid out cos || sin.
        wo_a_weight: [G * o_lora_rank, hpg * 512] or [G, o_lora_rank, hpg * 512],
            FP8 E4M3 or BF16.
        wo_a_weight_scale: FP32 128x128 block scales for FP8 (vLLM's
            ``weight_scale_inv``, 2D or 3D), or None for BF16.
        out: optional [num_tokens, G * o_lora_rank] bf16 buffer for ``z``.
        wo_b_proj: optional callback applied to ``z``.

    Returns:
        ``wo_b_proj(z)`` if a callback is given, otherwise z: [num_tokens,
        G * o_lora_rank] bf16.
    """
    logger.debug("GEMS_HYGON DSV41 O PROJ")

    if attn_out.ndim != 3:
        raise ValueError("attn_out must be [num_tokens, num_heads, head_dim]")
    if attn_out.dtype != torch.bfloat16:
        raise NotImplementedError("dsv41_o_proj: attn_out must be bfloat16")
    if (nope_dim, rope_dim) != (_NOPE_DIM, _ROPE_DIM):
        raise NotImplementedError("dsv41_o_proj on Hygon supports nope 448 + rope 64")
    if attn_out.requires_grad or wo_a_weight.requires_grad:
        raise NotImplementedError("dsv41_o_proj does not support autograd")

    # Checks and launch parameters depend only on this signature, so they are done once.
    scale = wo_a_weight_scale
    kind, run, out_shape = _call_plan(
        attn_out.shape,
        attn_out.stride(),
        attn_out.device,
        positions.shape,
        positions.dtype,
        positions.device,
        cos_sin_cache.shape,
        cos_sin_cache.stride(),
        cos_sin_cache.dtype,
        cos_sin_cache.device,
        wo_a_weight.shape,
        wo_a_weight.stride(),
        wo_a_weight.dtype,
        wo_a_weight.device,
        (
            None
            if scale is None
            else (scale.shape, scale.stride(), scale.dtype, scale.device)
        ),
        None if out is None else (out.shape, out.stride(), out.dtype, out.device),
        n_groups,
        heads_per_group,
        o_lora_rank,
        (
            attn_out.data_ptr() % 16 == 0,
            positions.data_ptr() % 16 == 0,
            cos_sin_cache.data_ptr() % 16 == 0,
            out is None or out.data_ptr() % 16 == 0,
        ),
    )
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=attn_out.device)
    group_dim = heads_per_group * _HEAD_DIM
    if kind == _INT8:
        wq, wqs = _int8_weight(wo_a_weight, scale, n_groups, o_lora_rank, group_dim)
        run(attn_out, positions, cos_sin_cache, wq, wqs, out)
    elif kind == _BF16:
        run(attn_out, positions, cos_sin_cache, wo_a_weight, None, out)
    elif kind == _BF16_COPY:
        _run_bf16_copy(
            attn_out[:, : n_groups * heads_per_group, :],
            positions,
            cos_sin_cache,
            wo_a_weight.view(n_groups, o_lora_rank, group_dim),
            out.view(out_shape[0], n_groups, o_lora_rank).permute(1, 0, 2),
        )
    return out if wo_b_proj is None else wo_b_proj(out)
