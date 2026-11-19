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
every non-NVIDIA device, BiShengHIR cannot lower a ``float8_e4m3fn`` cast (the
FP8 byte grid has to be built with integer ops), and a per-128-block
dequantization inside a ``tl.dot`` K loop costs several times the dot itself
(measured 15 TOPS against 139 TOPS for the bare INT8 dot).

This backend therefore keeps the operator's FP8 block-wise semantics but
transports the values as FP16:

1. ``_inv_rope_quant_kernel`` applies the inverse RoPE and rounds every
   128-element block onto the E4M3 grid (integer RNE on the FP32 bits), then
   writes the dequantized FP16 value. Rows of one head are contiguous, so the
   kernel is a streaming pass; the 128-wide block that holds the rope columns
   is handled with a pair split/join instead of a strided gather.
2. ``wo_a`` is a model parameter: it is decoded once into an FP16 K-major copy
   (``[G, K, N]``, so the GEMM reads contiguous tiles) and cached per module or
   tensor, keyed by ``data_ptr + _version``.
3. ``_bmm_kernel`` is a plain FP16 tensor-core batched GEMM with FP32
   accumulation; the block scales are already folded into the operands, so it
   runs at the shape's peak instead of paying a per-block promotion.

vLLM may pass a ``QuantizedActivation`` (the attention kernel already applied
the inverse RoPE and the FP8 cast); its payload is decoded and rescaled in
``_prequant_to_f16_kernel``, skipping the RoPE.

FP16 (11 mantissa bits) adds ~0.05% on top of the E4M3 grid, far below the
grid's own ~3% quantization error.
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
def _quant_blocks(src, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    """Per-128 block absmax quantize a [ROWS, WIDTH] tile, FP32 out."""
    q = tl.reshape(src, (ROWS * (WIDTH // 128), 128))
    amax = tl.max(tl.abs(q), axis=1)
    scale = amax * (1.0 / 448.0)
    inv = 1.0 / tl.where(scale == 0, 1.0, scale)
    return _round_e4m3(q * inv[:, None]) * scale[:, None]


@triton.jit
def _quant_store(src, out_ptr, off, valid, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    """Per-128 block absmax quantize ``src`` [ROWS, WIDTH] and store FP16.

    The reciprocal is taken once per row: an FP32 division per element costs
    ~2.5x the whole pass on the Ascend vector unit.
    """
    cols = tl.arange(0, WIDTH)
    tl.store(
        out_ptr + off[:, None] + cols[None, :],
        tl.reshape(_quant_blocks(src, ROWS, WIDTH), (ROWS, WIDTH)).to(tl.float16),
        mask=valid[:, None],
    )


@triton.jit
def _quant_pair_store(even, odd, out_ptr, off, valid, ROWS: tl.constexpr):
    """Quantize a rotated rope block from its even/odd halves and store it.

    The 128-wide block is reduced as two [ROWS, 64] halves instead of the
    joined tile: reducing a ``tl.join`` result is correct but compiles to a
    cross-lane storm on FlagTree 0.7.0 (CANN 9.0.0), where it costs ~45x the
    rest of the pass. Both halves share the block scale, so the result is the
    same tile.
    """
    amax = tl.maximum(tl.max(tl.abs(even), axis=1), tl.max(tl.abs(odd), axis=1))
    scale = amax * (1.0 / 448.0)
    inv = 1.0 / tl.where(scale == 0, 1.0, scale)
    dq = tl.join(
        _round_e4m3(even * inv[:, None]) * scale[:, None],
        _round_e4m3(odd * inv[:, None]) * scale[:, None],
    )
    tl.store(
        out_ptr + off[:, None] + tl.arange(0, 128)[None, :],
        tl.reshape(dq, (ROWS, 128)).to(tl.float16),
        mask=valid[:, None],
    )


@triton.jit
def _inv_rope_quant_kernel(
    o_ptr,
    pos_ptr,
    tab_ptr,
    out_ptr,
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
    pairs: tl.constexpr = (128 - ROPE_WIDTH) // 2 + ROPE_WIDTH // 2
    pair_cols = tl.arange(0, pairs)
    last_cols = tl.arange(0, 128)
    head_valid = heads < heads_per_group
    for j in tl.static_range(TOKENS):
        token = pid_t * TOKENS + j
        in_range = token < num_tokens
        valid = head_valid & in_range
        token_ld = tl.minimum(token, num_tokens - 1).to(tl.int64)
        in_base = (token_ld * rows_per_token + group * heads_per_group) * HEAD_DIM
        out_base = (group * num_tokens + token) * GROUP_DIM
        row_off = heads * HEAD_DIM
        _quant_store(
            tl.load(
                o_ptr
                + in_base
                + row_off[:, None]
                + tl.arange(0, nope_cols)[None, :]
            ).to(tl.float32),
            out_ptr,
            out_base + row_off,
            valid,
            heads_per_group,
            nope_cols,
        )
        last = tl.load(
            o_ptr + in_base + row_off[:, None] + nope_cols + last_cols[None, :]
        ).to(tl.float32)
        pair = tl.reshape(last, (heads_per_group, pairs, 2))
        even, odd = tl.split(pair)
        pos = tl.load(pos_ptr + token_ld).to(tl.int64)
        cos_v = tl.load(tab_ptr + pos * tab_stride + pair_cols)
        sin_v = tl.load(tab_ptr + pairs + pos * tab_stride + pair_cols)
        _quant_pair_store(
            even * cos_v[None, :] + odd * sin_v[None, :],
            odd * cos_v[None, :] - even * sin_v[None, :],
            out_ptr,
            out_base + row_off + nope_cols,
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
) -> torch.Tensor:
    """Return a group-major ``[groups, tokens, group_dim]`` FP16 activation."""
    num_tokens, _, head_dim = o.shape
    group_dim = heads_per_group * head_dim
    out = torch.empty(
        (n_groups, num_tokens, group_dim), dtype=torch.float16, device=o.device
    )
    if num_tokens == 0:
        return out
    table = _rope_table(cos_sin_cache)
    tokens = _quant_tokens(num_tokens, n_groups)
    args = (
        o,
        positions,
        table,
        out,
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
        (tuple(t.stride() for t in (o, out)), num_tokens == 1),
        {"num_warps": 8, "num_stages": 1},
    )
    return out


# ---------------------------------------------------------------------------
# FP8 payload -> FP16 (pre-quantized activation and wo_a weights)
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


@triton.jit
def _prequant_to_f16_kernel(
    x_ptr,
    s_ptr,
    out_ptr,
    num_rows,
    num_tokens,
    num_groups: tl.constexpr,
    blocks_per_row: tl.constexpr,
    ROWS: tl.constexpr,
    GRID_X: tl.constexpr,
):
    """``[tokens, groups, k]`` E4M3 payload -> group-major FP16 with block scales."""
    pid = (tl.program_id(1) * GRID_X + tl.program_id(0)).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS)
    valid = rows < num_rows
    cols = tl.arange(0, blocks_per_row * 128)
    x = tl.load(
        x_ptr + rows[:, None] * (blocks_per_row * 128) + cols[None, :],
        mask=valid[:, None],
        other=0,
    )
    value = _e4m3_byte_to_f32(x)
    scale = tl.load(
        s_ptr + rows[:, None] * blocks_per_row + tl.arange(0, blocks_per_row)[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    out = (
        tl.reshape(value, (ROWS * blocks_per_row, 128))
        * tl.reshape(
            tl.broadcast_to(scale[:, :, None], (ROWS, blocks_per_row, 128)),
            (ROWS * blocks_per_row, 128),
        )
    ).to(tl.float16)
    token = rows // num_groups
    group = rows % num_groups
    off = (group * num_tokens + token) * (blocks_per_row * 128)
    tl.store(
        out_ptr + off[:, None] + cols[None, :],
        tl.reshape(out, (ROWS, blocks_per_row * 128)),
        mask=valid[:, None],
    )


def _prequant_to_f16(x: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    num_tokens, n_groups, group_dim = x.shape
    out = torch.empty(
        (n_groups, num_tokens, group_dim), dtype=torch.float16, device=x.device
    )
    rows = num_tokens * n_groups
    if rows == 0:
        return out
    blocks_per_row = group_dim // _QUANT_BLOCK
    row_tile = 1
    while triton.cdiv(rows, row_tile) > 32768:
        row_tile *= 2
    grid_x = triton.cdiv(rows, row_tile)
    _launch(
        "prequant_f16",
        _prequant_to_f16_kernel,
        (grid_x,),
        (
            x.view(torch.uint8),
            x_scale,
            out,
            rows,
            num_tokens,
            n_groups,
            blocks_per_row,
            row_tile,
            grid_x,
        ),
        (tuple(t.stride() for t in (x, x_scale, out)),),
        {"num_warps": 4, "num_stages": 1},
    )
    return out


@triton.jit
def _fp8_to_f16_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    """Flat E4M3 payload -> FP16 decode (no scales)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    value = _e4m3_byte_to_f32(tl.load(src_ptr + offs, mask=mask, other=0))
    tl.store(dst_ptr + offs, value.to(tl.float16), mask=mask)


@triton.jit
def _transpose_scale_kernel(
    src_ptr,
    s_ptr,
    dst_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    n_blocks: tl.constexpr,
    k_blocks: tl.constexpr,
    TILE: tl.constexpr,
):
    """``[G, N, K]`` FP16 -> ``[G, K, N]`` FP16, applying the 128x128 block scale.

    Producing the K-major copy makes the GEMM read contiguous tiles; the
    transpose happens once per weight tensor.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    group = tl.program_id(2).to(tl.int64)
    rn = pid_n * TILE + tl.arange(0, TILE)
    rk = pid_k * TILE + tl.arange(0, TILE)
    # Load already transposed ([k, n]) so no register-level ``tl.trans`` is
    # needed: the layout conversion misbehaves under Ascend's
    # TRITON_ALL_BLOCKS_PARALLEL kernel mode (NaN output).
    x = tl.load(src_ptr + group * (N * K) + rn[None, :] * K + rk[:, None])
    scale = tl.load(
        s_ptr
        + group * (n_blocks * k_blocks)
        + (pid_n * TILE // 128) * k_blocks
        + pid_k * TILE // 128
    )
    tl.store(
        dst_ptr + group * (K * N) + rk[:, None] * N + rn[None, :],
        (x * scale).to(tl.float16),
    )


def _quantize_weight_f16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode ``wo_a`` (FP8 + 128x128 scales) into a K-major FP16 copy."""
    n_groups, n, k = weight.shape
    flat = torch.empty((n_groups, n, k), dtype=torch.float16, device=weight.device)
    numel = flat.numel()
    block = 4096
    _launch(
        "weight_decode",
        _fp8_to_f16_kernel,
        (triton.cdiv(numel, block),),
        (weight.view(torch.uint8), flat, numel, block),
        (numel,),
        {"num_warps": 4, "num_stages": 1},
    )
    out = torch.empty((n_groups, k, n), dtype=torch.float16, device=weight.device)
    tile = 64
    _launch(
        "weight_transpose",
        _transpose_scale_kernel,
        (n // tile, k // tile, n_groups),
        (flat, scale.contiguous(), out, n, k, n // 128, k // 128, tile),
        (weight.data_ptr(), weight._version),
        {"num_warps": 4, "num_stages": 1},
    )
    return out


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
    """Return the cached K-major FP16 ``wo_a``; weights are model parameters."""
    global _TENSOR_CACHE_BYTES
    key = _cache_key(weight, scale)
    if isinstance(owner, torch.nn.Module):
        entry = _MODULE_CACHE.get(owner)
        if entry is not None and entry[0] == key:
            return entry[1]
        quantized = _quantize_weight_f16(weight, scale)
        try:
            _MODULE_CACHE[owner] = (key, quantized)
        except TypeError:  # pragma: no cover - module without weakref support
            pass
        return quantized

    entry = _TENSOR_CACHE.get(key)
    if entry is not None:
        _TENSOR_CACHE.move_to_end(key)
        return entry
    quantized = _quantize_weight_f16(weight, scale)
    nbytes = quantized.numel() * quantized.element_size()
    while _TENSOR_CACHE and _TENSOR_CACHE_BYTES + nbytes > _TENSOR_CACHE_LIMIT:
        _, evicted = _TENSOR_CACHE.popitem(last=False)
        _TENSOR_CACHE_BYTES -= evicted.numel() * evicted.element_size()
    _TENSOR_CACHE[key] = quantized
    _TENSOR_CACHE_BYTES += nbytes
    return quantized


# ---------------------------------------------------------------------------
# Batched FP16 GEMM with FP32 accumulation
# ---------------------------------------------------------------------------


@triton.jit
def _bmm_kernel(
    A,
    W,
    O,
    M,
    SA0,
    SA1,
    SW0,
    SW1,
    SO0,
    SO1,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    cols_per_row = N // BLOCK_N
    row_block = pid // cols_per_row
    col_block = pid % cols_per_row
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K // BLOCK_K):
        k_off = kb * BLOCK_K + kk
        a = tl.load(
            A + group * SA0 + rows[:, None] * SA1 + k_off[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        )
        w = tl.load(W + group * SW0 + k_off[:, None] * SW1 + cols[None, :])
        acc = tl.dot(a, w, acc, out_dtype=tl.float32)
    tl.store(
        O + group * SO0 + rows[:, None] * SO1 + cols[None, :],
        acc.to(tl.bfloat16),
        mask=rows[:, None] < M,
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
    """Per-M tiles measured on Ascend910B4.

    Decode sizes are bound by streaming ``wo_a`` (64 MB of FP16 per group
    count), so they want a narrow M tile and a wide N tile to keep the weight
    reads large and contiguous; from ``m == 128`` on the projection is
    compute-bound and a 256-wide N tile with two-stage pipelining wins (223
    TFLOPS at 8192 tokens). Pipelining is dropped when the backend's
    all-blocks-parallel knob is active.
    """
    stages = 2 if _multi_stage_allowed() else 1
    if m <= 32:
        return (16, 512, 128, 8, stages)
    if m <= 64:
        return (32, 256, 128, 8, stages)
    block_k = 256 if m >= 128 else 128
    return (128, 256, block_k, 8, stages)


def _bmm(act: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """``out[t, g, :] = act[g, t, :] @ weight[g]`` for every group."""
    n_groups, m, k = act.shape
    n = weight.shape[2]
    block_m, block_n, block_k, warps, stages = _bmm_tile(m)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), n_groups)
    args = (
        act,
        weight,
        out,
        m,
        act.stride(0),
        act.stride(1),
        weight.stride(0),
        weight.stride(1),
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
            tuple(t.stride() for t in (act, weight, out)),
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
    """DeepSeek-V4 output projection on Ascend (FP16 tensor-core transport)."""
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
            act = _prequant_to_f16(act_fp8, act_scale)
        else:
            act = _quant_activation(
                o,
                positions,
                cos_sin_cache,
                n_groups,
                heads_per_group,
                nope_dim,
                rope_dim,
            )
        weight_f16 = _cached_weight(owner, weight, weight_scale)
        out = torch.empty(
            (num_tokens, n_groups, o_lora_rank), dtype=torch.bfloat16, device=device
        )
        _bmm(act, weight_f16, out)

    assert callable(wo_b), "`wo_b` must be a callable linear layer"
    return wo_b(out.reshape(num_tokens, n_groups * o_lora_rank))
