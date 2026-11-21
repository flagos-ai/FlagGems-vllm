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

"""Ascend ``deep_gemm_fp8_o_proj``: inverse RoPE + FP8 block quant + einsum.

The generic implementation composes ``fused_inv_rope_fp8_quant`` with
``fp8_einsum``. On Ascend both steps are unavailable: ``fp8_einsum`` rejects
every non-NVIDIA device and BiShengHIR cannot lower a ``float8_e4m3fn`` cast
(the FP8 byte grid has to be built with integer ops).

This backend therefore transports the values as INT8 fixed point:

1. ``_inv_rope_quant_kernel`` applies the inverse RoPE, rounds every
   128-element block onto the E4M3 grid (integer RNE on the FP32 bits, i.e. the
   exact FP8 value the reference keeps) and renormalizes the row onto a single
   ``row_amax / 127`` scale. Rows of one head are contiguous, so the kernel is a
   streaming pass; the 128-wide block that holds the rope columns is quantized
   from its even/odd halves.
2. ``wo_a`` is a model parameter: it is decoded once into an INT8 K-major copy
   (``[G, K, N]``, so the GEMM reads contiguous tiles) with one scale per 128
   output columns, cached per module or tensor, keyed by ``data_ptr + _version``.
3. ``_bmm_kernel`` runs the grouped INT8 tensor-core product as a single INT32
   accumulator over the whole K loop and applies the rank-1
   ``act_scale[row] * weight_scale[col block]`` factor in the epilogue. Both
   scales are necessarily coarse: a per-128-K-block scale has to be applied
   inside the K loop, which pushes the accumulator out of the cube every block
   and costs ~15x on this backend (12.7 TFLOPS against 178 TFLOPS measured here,
   234 TOPS for the bare INT8 dot).

vLLM may pass a ``QuantizedActivation`` (the attention kernel already applied
the inverse RoPE and the FP8 cast); its payload is decoded and renormalized in
``_prequant_to_int8``, skipping the RoPE.

Renormalizing the exact FP8 values onto the row scale adds at most half an INT8
step of the row maximum (<= 0.4% of the row peak), next to the ~3% the E4M3 grid
already carries per element.
"""

import logging
import os
import weakref
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.deep_gemm_fp8_o_proj import (
    _SUPPORTED_RECIPE,
    _is_quantized_activation,
    _weight_and_scale,
)
from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_QUANT_BLOCK = 128
_FP8_MAX = 448.0
_ROPE_DIM = 64
# ``nope_dim % 128``: the rope columns sit at the tail of the last 128 block.
_ROPE_START = 64

# The Ascend Triton dispatcher costs ~55 us per launch (the empty-kernel floor on
# this FlagTree build), which dominates decode. Caching the bound
# ``CompiledKernel`` runner drops that to ~33 us.
_LEAN_LAUNCH = os.environ.get("FLAGGEMS_O_PROJ_LEAN_LAUNCH", "1") != "0"
_RUNNERS = {}
_RUNNER_LIMIT = 256


def _disable_all_blocks_parallel():
    """Turn off Ascend's ``TRITON_ALL_BLOCKS_PARALLEL`` compile mode.

    ``flag_gems`` exports ``TRITON_ALL_BLOCKS_PARALLEL=1`` on import, and with
    that knob this FlagTree build miscompiles every kernel below (measured:
    NaN activations from the weight decode and a GEMM that drops part of the K
    loop). The knob is read when a kernel is compiled, so clearing it before
    each first launch is enough; correctness comes first here.
    """
    os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)


_disable_all_blocks_parallel()


def _launch(tag, jit_fn, grid, args, key_extra, launch_kwargs=None):
    """Launch a Triton kernel, reusing its bound CompiledKernel when possible."""
    launch_kwargs = launch_kwargs or {}
    if not _LEAN_LAUNCH:
        jit_fn[grid](*args, **launch_kwargs)
        return
    key = (tag, tuple(grid), key_extra, tuple(sorted(launch_kwargs.items())))
    runner = _RUNNERS.get(key)
    if runner is not None:
        runner(*args)
        return
    _disable_all_blocks_parallel()
    compiled = jit_fn[grid](*args, **launch_kwargs)
    try:
        if len(_RUNNERS) >= _RUNNER_LIMIT:
            _RUNNERS.clear()
        _RUNNERS[key] = compiled[
            (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
        ]
    except Exception:  # pragma: no cover - defensive
        pass


# ---------------------------------------------------------------------------
# Activation: inverse RoPE + FP8 block quantization -> dequantized FP16
# ---------------------------------------------------------------------------


@triton.jit
def _round_e4m3(v):
    """Round FP32 magnitudes onto the E4M3 grid (RNE, three mantissa bits).

    Adding the RNE bias to the FP32 bit pattern and masking the low 20 mantissa
    bits matches a hardware FP8 cast for normal values; Ascend cannot lower the
    cast itself, and this needs four integer ops per element.
    """
    bits = v.to(tl.int32, bitcast=True)
    bits = bits + 0x0007FFFF + ((bits >> 20) & 1)
    bits = bits & -1048576
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _round_to_int8(v):
    """Round a value already scaled onto the INT8 grid, half away from zero.

    The cast goes straight to INT8: an INT32 intermediate doubles the live
    bytes of the tile and overflows the 192 KB UB on this backend.
    """
    return tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int8)


@triton.jit
def _fp8_grid_to_int8(v, blk_scale, row_scale):
    """``v`` -> the exact FP8 value, normalized by the per-row INT8 scale.

    ``blk_scale`` is the reference 128-element block scale ``block_amax / 448``
    (so ``_round_e4m3(v * (1 / blk_scale))`` reproduces the FP8 value the
    reference would keep) and ``row_scale`` is ``row_amax / 127``, the single
    scale of the whole ``(token, group)`` row. Folding one scale per row instead
    of one per 128-wide K block is what lets the GEMM keep the INT8 dot inside
    the cube for the whole K loop (see ``_bmm_kernel``).

    The reciprocal is taken once per row: an FP32 divide per element costs ~2.5x
    the whole pass on the Ascend vector unit (and this backend lowers it to a
    reciprocal multiply anyway). Either way a small share of the values sit on
    an E4M3 midpoint and round to the neighbouring grid point, which moves a
    single element by up to one E4M3 ulp; ``wo_b`` dilutes that to well under
    the accuracy budget.
    """
    inv = 1.0 / blk_scale
    grid = _round_e4m3(v * inv[:, None]) * (blk_scale / row_scale)[:, None]
    return _round_to_int8(tl.minimum(tl.maximum(grid, -127.0), 127.0))


@triton.jit
def _block_scale(amax):
    """Reference FP8 block scale, guarded against an all-zero block."""
    return tl.where(amax == 0, 1.0, amax * (1.0 / 448.0))


@triton.jit
def _block_amax(src):
    """Per-row absmax of a [ROWS, 128] tile (one 128-wide block per row)."""
    return tl.max(tl.abs(src), axis=1)


@triton.jit
def _quant_row_store(
    src, amax, out_ptr, off, row_scale, valid, ROWS: tl.constexpr
):
    """Quantize one 128-wide block of a row and store it as INT8.

    ``amax`` is the block absmax the caller already reduced for the row maximum:
    a 128-wide reduce is the most expensive op of this pass on Ascend, so it is
    taken once and reused.
    """
    _store_payload(
        _fp8_grid_to_int8(src, _block_scale(amax), row_scale),
        out_ptr,
        off,
        valid,
        ROWS,
    )


@triton.jit
def _store_payload(payload, out_ptr, off, valid, ROWS: tl.constexpr):
    tl.store(
        out_ptr + off[:, None] + tl.arange(0, 128)[None, :],
        payload,
        mask=valid[:, None],
    )


@triton.jit
def _quant_pair_store(even, odd, amax, out_ptr, off, row_scale, valid, ROWS: tl.constexpr):
    """Quantize a rotated rope block from its even/odd halves and store it.

    The 128-wide block is reduced as two [ROWS, 64] halves instead of the
    joined tile: reducing a ``tl.join`` result is correct but compiles to a
    cross-lane storm on FlagTree 0.7.0 (CANN 9.0.0), where it costs ~45x the
    rest of the pass. Both halves share the block scale, so the result is the
    same tile.
    """
    blk_scale = _block_scale(amax)
    ev = _fp8_grid_to_int8(even, blk_scale, row_scale)
    od = _fp8_grid_to_int8(odd, blk_scale, row_scale)
    tl.store(
        out_ptr + off[:, None] + tl.arange(0, 128)[None, :],
        tl.reshape(tl.join(ev, od), (ROWS, 128)),
        mask=valid[:, None],
    )


@triton.jit
def _inv_rope_quant_kernel(
    o_ptr,
    pos_ptr,
    tab_ptr,
    out_ptr,
    s_ptr,
    num_tokens,
    tab_stride,
    rows_per_token: tl.constexpr,
    heads_per_group: tl.constexpr,
    TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_WIDTH: tl.constexpr,
    GROUP_DIM: tl.constexpr,
):
    """One program covers ``TOKENS`` tokens of one group, head-tile by head-tile.

    A ``(token, group)`` pair owns ``heads_per_group * HEAD_DIM`` contiguous
    elements on both sides, so every tile below is a plain 2D block whose rows
    are one head apart. The output is group-major (``[G, T, K]``) so the GEMM
    reads contiguous tiles; the inner 128 block that holds the rope columns is
    rotated with a pair split/join.

    The row carries a single INT8 scale, so the row maximum has to be known
    before the first block is written: the kernel loads every block of the row
    once, keeps the tiles in registers across the two passes and quantizes from
    those registers, so the activation is read (and the rope rotated) once.

    Loads are deliberately unmasked: the token index is clamped to the last
    valid row, so every load stays in bounds and only the store is predicated.
    A masked 128-wide load out of this tile shape is miscompiled by FlagTree
    0.7.0 (CANN 9.0.0): it returns garbage, which turns the rope block into
    inf/NaN even though the same expression is exact without the mask.
    """
    pid_t = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    heads = tl.arange(0, heads_per_group)
    nope_cols: tl.constexpr = NOPE_DIM - (128 - ROPE_WIDTH)
    nope_blocks: tl.constexpr = nope_cols // 128
    pairs: tl.constexpr = (128 - ROPE_WIDTH) // 2 + ROPE_WIDTH // 2
    pair_cols = tl.arange(0, pairs)
    last_cols = tl.arange(0, 128)
    block_cols = tl.arange(0, 128)
    head_valid = heads < heads_per_group
    for j in tl.static_range(TOKENS):
        token = pid_t * TOKENS + j
        in_range = token < num_tokens
        valid = head_valid & in_range
        token_ld = tl.minimum(token, num_tokens - 1).to(tl.int64)
        in_base = (token_ld * rows_per_token + group * heads_per_group) * HEAD_DIM
        out_base = (group * num_tokens + token) * GROUP_DIM
        row_off = heads * HEAD_DIM
        # --- pass 1: rotate the rope block once, then the row absmax ----------
        # A wider int8 payload tile overflows the 192 KB UB on FlagTree 0.7.0
        # (the cast chain needs ~144 bytes/element), so blocks stay 128 wide.
        # The blocks are held in registers across both passes (hence the fixed
        # unroll: the frontend has no list/tuple), so the row is read once.
        base = o_ptr + in_base + row_off[:, None]
        t0 = tl.load(base + block_cols[None, :]).to(tl.float32)
        t1 = (
            tl.load(base + 128 + block_cols[None, :]).to(tl.float32)
            if nope_blocks > 1
            else t0
        )
        t2 = (
            tl.load(base + 256 + block_cols[None, :]).to(tl.float32)
            if nope_blocks > 2
            else t0
        )
        t3 = (
            tl.load(base + 384 + block_cols[None, :]).to(tl.float32)
            if nope_blocks > 3
            else t0
        )
        pair = tl.reshape(
            tl.load(base + nope_cols + last_cols[None, :]).to(tl.float32),
            (heads_per_group, pairs, 2),
        )
        even, odd = tl.split(pair)
        pos = tl.load(pos_ptr + token_ld).to(tl.int64)
        cos_v = tl.load(tab_ptr + pos * tab_stride + pair_cols)
        sin_v = tl.load(tab_ptr + pairs + pos * tab_stride + pair_cols)
        rope_even = even * cos_v[None, :] + odd * sin_v[None, :]
        rope_odd = odd * cos_v[None, :] - even * sin_v[None, :]
        rope_amax = tl.maximum(
            tl.max(tl.abs(rope_even), axis=1), tl.max(tl.abs(rope_odd), axis=1)
        )
        a0 = _block_amax(t0)
        a1 = _block_amax(t1) if nope_blocks > 1 else a0
        a2 = _block_amax(t2) if nope_blocks > 2 else a0
        a3 = _block_amax(t3) if nope_blocks > 3 else a0
        amax = rope_amax
        amax = tl.maximum(amax, a0)
        if nope_blocks > 1:
            amax = tl.maximum(amax, a1)
        if nope_blocks > 2:
            amax = tl.maximum(amax, a2)
        if nope_blocks > 3:
            amax = tl.maximum(amax, a3)
        row_scale = tl.max(amax) * (1.0 / 127.0)
        tl.store(s_ptr + group * num_tokens + token, row_scale, mask=in_range)
        # --- pass 2: quantize onto the FP8 grid and the row scale --------------
        _quant_row_store(
            t0, a0, out_ptr, out_base + row_off, row_scale, valid, heads_per_group
        )
        if nope_blocks > 1:
            _quant_row_store(
                t1, a1, out_ptr, out_base + row_off + 128, row_scale, valid,
                heads_per_group,
            )
        if nope_blocks > 2:
            _quant_row_store(
                t2, a2, out_ptr, out_base + row_off + 256, row_scale, valid,
                heads_per_group,
            )
        if nope_blocks > 3:
            _quant_row_store(
                t3, a3, out_ptr, out_base + row_off + 384, row_scale, valid,
                heads_per_group,
            )
        _quant_pair_store(
            rope_even,
            rope_odd,
            rope_amax,
            out_ptr,
            out_base + row_off + nope_cols,
            row_scale,
            valid,
            heads_per_group,
        )


_TABLE_CACHE = OrderedDict()


def _rope_table(cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """Expand the ``cos || sin`` cache for pair-wise rotation of a 128 block.

    Entry ``p`` of a pair holds ``(cos, sin) = (1, 0)`` for the pass-through
    columns and ``(cos_p, sin_p)`` for the rope columns, so the kernel can treat
    the whole trailing 128 block uniformly.
    """
    key = (
        cos_sin_cache.data_ptr(),
        cos_sin_cache._version,
        tuple(cos_sin_cache.shape),
        cos_sin_cache.device,
    )
    entry = _TABLE_CACHE.get(key)
    if entry is not None:
        _TABLE_CACHE.move_to_end(key)
        return entry
    half = _ROPE_DIM // 2
    table = torch.empty(
        (cos_sin_cache.shape[0], 4 * half),
        dtype=cos_sin_cache.dtype,
        device=cos_sin_cache.device,
    )
    table[:, :half] = 1.0
    table[:, half : 2 * half] = cos_sin_cache[:, :half]
    table[:, 2 * half : 3 * half] = 0.0
    table[:, 3 * half :] = cos_sin_cache[:, half:]
    while len(_TABLE_CACHE) >= 4:
        _TABLE_CACHE.popitem(last=False)
    _TABLE_CACHE[key] = table
    return table


def _quant_tokens(num_tokens: int, n_groups: int) -> int:
    """Tokens per CTA, capped so the launch stays inside the 65535 block limit.

    Four tokens keep one program at ~16 KB of tile traffic; Ascend rejects a
    launch whose grid product exceeds 65535, so large token counts fall back to
    fewer tokens per program. The pass is vector-throughput bound, so larger
    tiles and software pipelining were measured and do not help (see A9).
    """
    limit = 65500 // max(n_groups, 1)
    for candidate in (4, 2, 1):
        if num_tokens % candidate == 0 and triton.cdiv(num_tokens, candidate) <= limit:
            return candidate
    return 1


def _quant_activation(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
):
    """Return group-major ``[groups, tokens, group_dim]`` INT8 + row scales."""
    num_tokens, _, head_dim = o.shape
    group_dim = heads_per_group * head_dim
    out = torch.empty(
        (n_groups, num_tokens, group_dim), dtype=torch.int8, device=o.device
    )
    scales = torch.empty(
        (n_groups, num_tokens), dtype=torch.float32, device=o.device
    )
    if num_tokens == 0:
        return out, scales
    table = _rope_table(cos_sin_cache)
    tokens = _quant_tokens(num_tokens, n_groups)
    args = (
        o,
        positions,
        table,
        out,
        scales,
        num_tokens,
        table.stride(0),
        n_groups * heads_per_group,
        heads_per_group,
        tokens,
        head_dim,
        nope_dim,
        rope_dim,
        group_dim,
    )
    _launch(
        "inv_rope_quant",
        _inv_rope_quant_kernel,
        (triton.cdiv(num_tokens, tokens), n_groups),
        args,
        (tuple(t.stride() for t in (o, out, scales)), num_tokens == 1),
        {"num_warps": 8, "num_stages": 1},
    )
    return out, scales


# ---------------------------------------------------------------------------
# FP8 payload -> INT8 transport (pre-quantized activation and wo_a weights)
# ---------------------------------------------------------------------------


@triton.jit
def _e4m3_byte_to_f32(bits):
    """Decode E4M3 bytes into the exact FP32 value without a hardware cast."""
    b = bits.to(tl.int32) & 0xFF
    sign = (b & 0x80) << 8
    exp = (b >> 3) & 0xF
    man = b & 0x7
    packed = sign | ((exp + 120) << 7) | (man << 4)
    normal = (packed << 16).to(tl.float32, bitcast=True)
    sub = man.to(tl.float32) * 0.001953125  # mantissa * 2^-9
    sub = tl.where(sign != 0, -sub, sub)
    return tl.where(exp == 0, sub, normal)


def _decode_fp8(weight: torch.Tensor) -> torch.Tensor:
    """Decode an E4M3 payload to FP32 values (Ascend has no FP8 cast)."""
    values = torch.empty(weight.shape, dtype=torch.float32, device=weight.device)
    numel = values.numel()
    if numel == 0:
        return values
    block = 4096
    _launch(
        "fp8_decode",
        _fp8_to_f32_kernel,
        (triton.cdiv(numel, block),),
        (weight.view(torch.uint8), values, numel, block),
        (numel,),
        {"num_warps": 4, "num_stages": 1},
    )
    return values


def _prequant_to_int8(x: torch.Tensor, x_scale: torch.Tensor):
    """``[tokens, groups, k]`` FP8 + block scales -> group-major INT8 + row scales.

    The payload is already on the FP8 grid, so it only has to be renormalized
    onto one INT8 scale per ``(group, token)`` row; the block scales ride into
    the FP32 values first. This path is off the attention hot path, so the
    tensor ops below (one extra FP32 copy of the activation) are acceptable.
    """
    num_tokens, n_groups, group_dim = x.shape
    values = _decode_fp8(x)
    if num_tokens == 0:
        return (
            torch.empty(
                (n_groups, num_tokens, group_dim), dtype=torch.int8, device=x.device
            ),
            torch.empty((n_groups, num_tokens), dtype=torch.float32, device=x.device),
        )
    values = values * x_scale.repeat_interleave(_QUANT_BLOCK, dim=-1)
    amax = values.abs().amax(dim=-1)  # [tokens, groups]
    scale = (amax / 127.0).clamp_min(1e-30)
    payload = (
        torch.round(values / scale[..., None])
        .clamp_(-127.0, 127.0)
        .to(torch.int8)
        .permute(1, 0, 2)
        .contiguous()
    )
    return payload, scale.permute(1, 0).contiguous()


@triton.jit
def _fp8_to_f32_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    """Flat E4M3 payload -> FP32 decode (no scales)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    value = _e4m3_byte_to_f32(tl.load(src_ptr + offs, mask=mask, other=0))
    tl.store(dst_ptr + offs, value, mask=mask)


def _quantize_weight_int8(weight: torch.Tensor, scale: torch.Tensor):
    """Decode ``wo_a`` (FP8 + 128x128 scales) into a K-major INT8 copy.

    The 128x128 FP8 block scales are folded into the FP32 values, then the
    whole ``n`` block is renormalized onto one INT8 scale per 128 output
    columns. That single scale per output block (instead of one per
    ``(n block, k block)``) is what keeps the GEMM's epilogue rank-1, so the
    INT8 dot stays in the cube for the whole K loop.

    Only the decode is a Triton kernel (Ascend cannot cast FP8 in hardware);
    the requantization runs once per weight tensor.
    """
    n_groups, n, k = weight.shape
    values = _decode_fp8(weight)
    view = values.reshape(n_groups, n // 128, 128, k // 128, 128)
    block_scale = scale.contiguous().reshape(n_groups, n // 128, k // 128)
    real = view * block_scale[:, :, None, :, None]
    amax = real.abs().amax(dim=(2, 3, 4))  # [groups, n blocks]
    inv = 127.0 / amax.clamp_min(1e-30)
    payload = (
        torch.round(real * inv[:, :, None, None, None])
        .clamp_(-127.0, 127.0)
        .to(torch.int8)
        .reshape(n_groups, n, k)
        .permute(0, 2, 1)
        .contiguous()
    )
    return payload, (amax / 127.0).contiguous()


_MODULE_CACHE = weakref.WeakKeyDictionary()
_TENSOR_CACHE = OrderedDict()
_TENSOR_CACHE_BYTES = 0
_TENSOR_CACHE_LIMIT = 1 << 30


def _cache_key(weight: torch.Tensor, scale: torch.Tensor):
    return (
        weight.data_ptr(),
        weight._version,
        tuple(weight.shape),
        scale.data_ptr(),
        scale._version,
        tuple(scale.shape),
    )


def _cached_weight(owner, weight: torch.Tensor, scale: torch.Tensor):
    """Return the cached K-major INT8 ``wo_a`` and its block scales."""
    global _TENSOR_CACHE_BYTES
    key = _cache_key(weight, scale)
    if isinstance(owner, torch.nn.Module):
        entry = _MODULE_CACHE.get(owner)
        if entry is not None and entry[0] == key:
            return entry[1]
        quantized = _quantize_weight_int8(weight, scale)
        try:
            _MODULE_CACHE[owner] = (key, quantized)
        except TypeError:  # pragma: no cover - module without weakref support
            pass
        return quantized

    entry = _TENSOR_CACHE.get(key)
    if entry is not None:
        _TENSOR_CACHE.move_to_end(key)
        return entry
    quantized = _quantize_weight_int8(weight, scale)
    nbytes = sum(t.numel() * t.element_size() for t in quantized)
    while _TENSOR_CACHE and _TENSOR_CACHE_BYTES + nbytes > _TENSOR_CACHE_LIMIT:
        _, evicted = _TENSOR_CACHE.popitem(last=False)
        _TENSOR_CACHE_BYTES -= sum(t.numel() * t.element_size() for t in evicted)
    _TENSOR_CACHE[key] = quantized
    _TENSOR_CACHE_BYTES += nbytes
    return quantized


# ---------------------------------------------------------------------------
# Batched INT8 GEMM with one activation scale per row and one weight scale per
# 128 output columns
# ---------------------------------------------------------------------------


@triton.jit
def _bmm_kernel(
    A,
    W,
    AS,
    WS,
    O,
    M,
    SA0,
    SA1,
    SW0,
    SW1,
    SAS0,
    SWS0,
    SO0,
    SO1,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped INT8 GEMM with a rank-1 scale applied after the K loop.

    Both operands carry a single scale per row (activation) and per 128 output
    columns (weight), so the whole K reduction stays inside the cube as one
    INT32 accumulator; the ``act_scale * weight_scale`` factor is one FMA per
    output element. Applying a scale per 128-wide K block instead costs ~15x on
    this backend (the accumulator has to leave the cube every block).
    """
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    cols_per_row = N // BLOCK_N
    row_block = pid // cols_per_row
    col_block = pid % cols_per_row
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    row_valid = rows < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + kk
        a = tl.load(
            A + group * SA0 + rows[:, None] * SA1 + k_off[None, :],
            mask=row_valid[:, None],
            other=0,
        )
        w = tl.load(W + group * SW0 + k_off[:, None] * SW1 + cols[None, :])
        acc += tl.dot(a, w, out_dtype=tl.int32)
    a_s = tl.load(AS + group * SAS0 + rows, mask=row_valid, other=0.0)
    # One weight scale per 128 output columns; ``BLOCK_N`` may cover several.
    w_s = tl.load(WS + group * SWS0 + cols // 128)
    tl.store(
        O + group * SO0 + rows[:, None] * SO1 + cols[None, :],
        (acc.to(tl.float32) * (a_s[:, None] * w_s[None, :])).to(tl.bfloat16),
        mask=row_valid[:, None],
    )


def _multi_stage_allowed() -> bool:
    """Ascend's ``TRITON_ALL_BLOCKS_PARALLEL`` knob breaks multi-stage pipelining.

    ``flag_gems`` exports ``TRITON_ALL_BLOCKS_PARALLEL=1`` on import, and with
    that knob a ``num_stages >= 2`` GEMM silently drops part of the K loop
    (measured: 0.94 relative error, while ``num_stages=1`` is exact). Read the
    environment on every call because the flag can be set after this module is
    imported.
    """
    return os.environ.get("TRITON_ALL_BLOCKS_PARALLEL", "0").lower() not in (
        "1",
        "true",
        "yes",
    )


def _bmm_tile(m: int):
    """Per-M tiles measured on Ascend910B4 for the rank-1 INT8 GEMM.

    Decode sizes are bound by streaming ``wo_a`` (16 MB of INT8 per 8 groups),
    so they want a narrow M tile; from ``m == 128`` on the projection is
    compute-bound and a 128-row / 256-column tile reaches 178 TFLOPS on the
    bare INT8 dot. Two-stage pipelining is dropped when the backend's
    all-blocks-parallel knob is active.
    """
    stages = 2 if _multi_stage_allowed() else 1
    if m <= 32:
        return (16, 256, 256, 8, stages)
    if m <= 64:
        return (32, 256, 256, 8, stages)
    if m <= 256:
        return (64, 256, 256, 8, stages)
    return (128, 256, 256, 8, stages)


def _bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out[t, g, :] = act[g, t, :] @ weight[g]`` with rank-1 scales."""
    n_groups, m, k = act.shape
    n = weight.shape[2]
    block_m, block_n, block_k, warps, stages = _bmm_tile(m)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), n_groups)
    args = (
        act,
        weight,
        act_scale,
        weight_scale,
        out,
        m,
        act.stride(0),
        act.stride(1),
        weight.stride(0),
        weight.stride(1),
        act_scale.stride(0),
        weight_scale.stride(0),
        out.stride(1),
        out.stride(0),
        n,
        k,
        block_m,
        block_n,
        block_k,
    )
    _launch(
        "bmm",
        _bmm_kernel,
        grid,
        args,
        (
            tuple(t.stride() for t in (act, weight, act_scale, weight_scale, out)),
            m == 1,
            m % block_m == 0,
        ),
        {"num_warps": warps, "num_stages": stages},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _weight_supported(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
    reduction: int,
) -> bool:
    if weight.shape != (n_groups, o_lora_rank, reduction):
        return False
    if reduction % _QUANT_BLOCK or o_lora_rank % _QUANT_BLOCK:
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.stride(-1) != 1:
        return False
    if weight_scale.dtype != torch.float32:
        return False
    return weight_scale.shape == (
        n_groups,
        o_lora_rank // _QUANT_BLOCK,
        reduction // _QUANT_BLOCK,
    )


def deep_gemm_fp8_o_proj(
    o,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    wo_b: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: Optional[Sequence[int]] = _SUPPORTED_RECIPE,
    tma_aligned_scales: bool = False,
) -> torch.Tensor:
    """DeepSeek-V4 output projection on Ascend (INT8 tensor-core transport)."""
    logger.debug("GEMS DEEP GEMM FP8 O PROJ (ASCEND)")

    if tuple(einsum_recipe) != _SUPPORTED_RECIPE or tma_aligned_scales:
        raise NotImplementedError(
            "the Ascend implementation only supports the Hopper FP32 block-scale "
            f"recipe (1, 128, 128) with 128-wide K blocks, got "
            f"einsum_recipe={tuple(einsum_recipe)}, "
            f"tma_aligned_scales={tma_aligned_scales}"
        )
    if rope_dim != _ROPE_DIM or nope_dim % _QUANT_BLOCK != _ROPE_START:
        raise NotImplementedError(
            "the Ascend implementation requires the model layout "
            f"(nope_dim % 128 == 64, rope_dim == 64), got nope_dim={nope_dim}, "
            f"rope_dim={rope_dim}"
        )
    if nope_dim > 4 * _QUANT_BLOCK + _ROPE_START:
        raise NotImplementedError(
            "the Ascend quantization kernel unrolls at most four 128-wide nope "
            f"blocks per head, got nope_dim={nope_dim}"
        )
    assert o_lora_rank % _QUANT_BLOCK == 0, "o_lora_rank must be a multiple of 128"

    head_dim = nope_dim + rope_dim
    reduction = heads_per_group * head_dim
    weight, weight_scale = _weight_and_scale(wo_a)
    prequantized = _is_quantized_activation(o)

    if prequantized:
        act_fp8, act_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
        assert act_fp8.ndim == 3, "`o.data` must be [num_tokens, n_groups, k]"
        assert act_fp8.dtype == torch.float8_e4m3fn, "`o.data` must be float8_e4m3fn"
        assert act_scale.dtype == torch.float32, "`o.scale` must be float32"
        num_tokens = act_fp8.shape[0]
        device = act_fp8.device
        activation_ok = act_fp8.shape[1:] == (n_groups, reduction)
        activation_ok = activation_ok and act_scale.shape == (
            num_tokens,
            n_groups,
            reduction // _QUANT_BLOCK,
        )
    else:
        assert o.ndim == 3, "`o` must be [num_tokens, num_heads, head_dim]"
        assert positions.ndim == 1 and positions.shape[0] == o.shape[0]
        assert cos_sin_cache.ndim == 2, "`cos_sin_cache` must be 2D"
        assert o.dtype == torch.bfloat16, "`o` must be bfloat16"
        assert (
            o.shape[1] == n_groups * heads_per_group
        ), "num_heads must equal n_groups * heads_per_group"
        assert o.shape[-1] == head_dim, "head_dim must equal nope_dim + rope_dim"
        num_tokens = o.shape[0]
        device = o.device
        activation_ok = o.stride(-1) == 1 and o.stride(1) == head_dim

    if not (
        activation_ok
        and _weight_supported(weight, weight_scale, n_groups, o_lora_rank, reduction)
    ):
        raise NotImplementedError(
            "unsupported shape for the Ascend deep_gemm_fp8_o_proj: expected an "
            f"activation [tokens, {n_groups}, {reduction}] (FP8 pre-quantized or "
            "BF16 attention output), 128x128 block scales and an FP8 wo_a"
        )

    owner = wo_a if isinstance(wo_a, torch.nn.Module) else None
    with torch_device_fn.device(device):
        if num_tokens == 0:
            empty = torch.empty(
                (0, n_groups * o_lora_rank), dtype=torch.bfloat16, device=device
            )
            return wo_b(empty)
        if prequantized:
            act, act_scales = _prequant_to_int8(act_fp8, act_scale)
        else:
            act, act_scales = _quant_activation(
                o,
                positions,
                cos_sin_cache,
                n_groups,
                heads_per_group,
                nope_dim,
                rope_dim,
            )
        weight_i8, weight_scales = _cached_weight(owner, weight, weight_scale)
        out = torch.empty(
            (num_tokens, n_groups, o_lora_rank), dtype=torch.bfloat16, device=device
        )
        _bmm(act, act_scales, weight_i8, weight_scales, out)

    assert callable(wo_b), "`wo_b` must be a callable linear layer"
    return wo_b(out.reshape(num_tokens, n_groups * o_lora_rank))
