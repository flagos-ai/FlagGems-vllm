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

"""MThreads ``deep_gemm_fp8_o_proj``: inverse RoPE + FP8 quant + block FP8 einsum.

The generic implementation composes the generic ``fused_inv_rope_fp8_quant``
with ``fp8_einsum``. On MUSA the first half of that composition cannot run: the
generic quant kernel gates its FP8 cast on ``kernel_supports_fp8_e4m3()``, which
only whitelists NVIDIA, so it raises before touching the device. The MThreads
backend ships a working inverse-RoPE + FP8 quant kernel
(``_mthreads/ops/fused_inv_rope_fp8_quant.py``) and the S5000 has native FP8
tensor cores, so this module wires those together:

1. The activation quant is the MThreads kernel, owned by this module so the
   tile width follows the S5000 sweeps: one head per CTA for decode, four-head
   tiles for prefill. It produces the same block-wise FP8 payload and scales
   the shared ``fp8_einsum`` would consume.
2. The GEMM is either a dedicated block-scaled batched kernel or, from 512
   tokens up, the TLE warp-specialized one. The shared TLE
   ``w8a8_block_fp8_bmm`` cannot back this operator on its own: it searches
   tiles for the generic bmm shapes, and with this operator's ``N = 1024``,
   ``B = n_groups`` and short ``K`` it resolves to one CTA per weight tile at
   ``M <= 128`` (0.29 ms for 34 MB of weights at ``M = 16``) and faults with
   ``misaligned address`` once ``M >= 512``. The dedicated kernel below walks K
   in 128-wide blocks, applies both block scales after each ``tl.dot`` and
   keeps a fixed tile per shape, so no autotune is paid at benchmark time.
3. Both kernels are relaunched through their compiled-kernel handles: FlagTree's
   dispatch costs ~115-220 us per launch, and up to 256 tokens this operator is
   entirely host bound.

The numerics are the operator's own: FP8 x FP8 products with FP32 accumulation
scaled by ``x_scale[row, k_block] * y_scale[n_block, k_block]`` per 128x128
block, which is exactly ``fp8_einsum``'s block-scale semantics (measured 0.3%
peak-relative against a dequantized FP32 reference, i.e. the FP8 grid's own
rounding).
"""

import contextlib
import logging
import os
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import triton
import triton.language as tl
from triton import knobs

from flaggems_vllm.ops.deep_gemm_fp8_o_proj import (
    _SUPPORTED_RECIPE,
    _is_quantized_activation,
    _weight_and_scale,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._mthreads.ops.per_token_group_quant_fp8 import (
    _quant_groups,
)
from flaggems_vllm.runtime.backend._mthreads.ops.w8a8_block_fp8_bmm import (
    LLC_OPTIONS,
    _derived_values,
    extern_libs_thread_index,
    ws_gemm_fp8,
)

try:  # TMA descriptors are only needed by the TLE path.
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:  # pragma: no cover - older FlagTree builds
    TensorDescriptor = None

logger = logging.getLogger(__name__)

_QUANT_BLOCK = 128
# Tensor-metadata signatures whose support verdict has already been computed;
# the value holds that signature's cached kernel launchers.
_SUPPORT_CHECKED = {}
# TLE path: the weight TMA descriptor and the extern-libs handle are cached
# (rebuilding the latter costs ~19 us per call).
_TLE_WEIGHT_DESCS = {}
_EXTERN_LIBS = None
# Set when a direct launch fails, so every later call takes the regular path.
_LAUNCH_DISABLED = False


def _current_stream():
    """Current MUSA stream handle, as the Triton launcher wants it."""
    stream = torch.musa.current_stream()
    return getattr(stream, "musa_stream", getattr(stream, "cuda_stream", stream))


class _CachedLaunch:
    """A compiled Triton kernel relaunched through its C launcher.

    FlagTree's ``JITFunction.run`` re-binds the arguments, re-derives the
    specialization of every scalar and pointer, rebuilds the cache key and only
    then expands the descriptor arguments. On the S5000 that costs ~115 us for
    a plain kernel and ~220 us for the 41-parameter TLE one, against ~20 us
    (~75 us with three TMA descriptors) for the launcher itself, and at 256
    tokens and below this operator is entirely host bound. This class keeps the
    compiled kernel and the constant argument tail so later calls go straight
    to the launcher. It is only built for signatures whose arguments the caller
    already validated, and every failure drops back to the regular dispatch.
    """

    __slots__ = ("kernel", "grid", "tail")

    def __init__(self, kernel, arg_names, grid, num_args, kwargs):
        self.kernel = kernel
        size = len(grid)
        self.grid = (
            grid[0],
            grid[1] if size > 1 else 1,
            grid[2] if size > 2 else 1,
        )
        self.tail = [kwargs[name] for name in arg_names[num_args:]]

    def __call__(self, args, stream):
        kernel = self.kernel
        kernel.run(
            self.grid[0],
            self.grid[1],
            self.grid[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            None,
            None,
            None,
            *args,
            *self.tail,
        )


def _launch_hooks_absent() -> bool:
    """Whether Triton has no launch hooks installed (profiler/debugger attach)."""
    for name in ("launch_enter_hook", "launch_exit_hook"):
        hook = getattr(knobs.runtime, name, None)
        if hook is None:
            continue
        calls = getattr(hook, "calls", None)
        if calls is None or calls:
            return False
    return True


def _dispatch(jit_fn, grid, args, kwargs, record, slot, stream):
    """Launch ``jit_fn``, reusing its compiled kernel after the first call."""
    global _LAUNCH_DISABLED
    launcher = record.get(slot)
    if launcher is not None:
        try:
            launcher(args, stream)
            return
        except Exception:  # pragma: no cover - defensive, layout changes
            logger.warning("direct launch failed; using the regular dispatch")
            record.pop(slot, None)
            _LAUNCH_DISABLED = True
    kernel = jit_fn.run(*args, grid=grid, warmup=False, **kwargs)
    # The launcher bypasses Triton's launch hooks, so only use it when none is
    # installed. Building it can only fail on a signature we cannot mirror (a
    # parameter the caller passes positionally and as a keyword), which just
    # means this shape keeps the regular dispatch.
    if not _LAUNCH_DISABLED and _launch_hooks_absent():
        try:
            record[slot] = _CachedLaunch(
                kernel, jit_fn.arg_names, grid, len(args), kwargs
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("cannot cache the launch of %s", jit_fn.arg_names[0])


@triton.jit
def _o_proj_bmm_kernel(
    x_ptr,
    xs_ptr,
    y_ptr,
    ys_ptr,
    z_ptr,
    num_tokens,
    x_stride_b,
    x_stride_m,
    x_stride_k,
    xs_stride_b,
    xs_stride_m,
    xs_stride_k,
    y_stride_b,
    y_stride_n,
    y_stride_k,
    ys_stride_b,
    ys_stride_n,
    ys_stride_k,
    z_stride_b,
    z_stride_m,
    z_stride_n,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """``z[g, m, n] = x[g, m, :] @ y[g, n, :]^T`` with 128x128 block scales.

    One program computes a ``BLOCK_M x BLOCK_N`` tile of one group; the tile
    walk covers ``GROUP_M`` m-tiles before moving on so consecutive programs
    reuse the weight tile in L2. ``BLOCK_N`` is a multiple of the 128-wide
    scale block, so the weight scales of a tile are a short vector load.
    """
    pid = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)

    num_m = tl.cdiv(num_tokens, BLOCK_M)
    num_n: tl.constexpr = N // BLOCK_N
    tiles_per_group: tl.constexpr = GROUP_M * num_n
    group_id = pid // tiles_per_group
    first_m = group_id * GROUP_M
    rows_in_group = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % rows_in_group)
    pid_n = (pid % tiles_per_group) // rows_in_group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < num_tokens

    x_base = x_ptr + group * x_stride_b + offs_m[:, None] * x_stride_m
    y_base = y_ptr + group * y_stride_b + offs_n[:, None] * y_stride_n
    ys_base = ys_ptr + group * ys_stride_b + (offs_n // BLOCK_K) * ys_stride_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(K // BLOCK_K):
        k_offs = k_block * BLOCK_K + offs_k
        x = tl.load(
            x_base + k_offs[None, :] * x_stride_k, mask=mask_m[:, None], other=0.0
        )
        y = tl.load(y_base + k_offs[None, :] * y_stride_k)
        product = tl.dot(x, tl.trans(y), out_dtype=tl.float32)
        row_scale = tl.load(
            xs_ptr
            + group * xs_stride_b
            + offs_m * xs_stride_m
            + k_block * xs_stride_k,
            mask=mask_m,
            other=0.0,
        )
        col_scale = tl.load(ys_base + k_block * ys_stride_k)
        acc += product * (row_scale[:, None] * col_scale[None, :])

    z_offs = (
        group * z_stride_b
        + offs_m[:, None] * z_stride_m
        + offs_n[None, :] * z_stride_n
    )
    tl.store(z_ptr + z_offs, acc, mask_m[:, None])


def _tile_config(num_tokens: int, n: int):
    """Per-shape tile chosen from S5000 sweeps of the decode/prefill points.
    Decode is weight-bandwidth bound: 64-wide n tiles (and 32-row m tiles) keep
    enough CTAs resident to cover the DRAM latency of the 34-68 MB weight
    stream. The 128-token bucket is the first one whose GEMM is tensor bound,
    so it keeps 64-wide tiles (more CTAs than 128) but eight warps. From 256
    up, 128-wide n tiles halve the activation re-reads; the prefill bucket
    moves to 128x128 with 16 warps, where one CTA per MP is the fastest.
    """
    if num_tokens <= 16:
        block_m, block_n, group_m, num_warps, num_stages = 16, 64, 1, 4, 3
    elif num_tokens <= 64:
        block_m, block_n, group_m, num_warps, num_stages = 32, 64, 8, 8, 2
    elif num_tokens <= 128:
        block_m, block_n, group_m, num_warps, num_stages = 64, 64, 8, 8, 2
    elif num_tokens <= 512:
        block_m, block_n, group_m, num_warps, num_stages = 64, 128, 8, 8, 2
    else:
        block_m, block_n, group_m, num_warps, num_stages = 128, 128, 8, 16, 2
    if n % block_n:
        block_n = _QUANT_BLOCK
    return block_m, block_n, group_m, num_warps, num_stages


def _weight_supported(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_groups: int,
    o_lora_rank: int,
    reduction: int,
) -> bool:
    if weight.shape != (n_groups, o_lora_rank, reduction):
        return False
    if weight.dtype != torch.float8_e4m3fn or weight.stride(-1) != 1:
        return False
    if reduction % _QUANT_BLOCK or o_lora_rank % _QUANT_BLOCK:
        return False
    if weight_scale.dtype != torch.float32:
        return False
    return weight_scale.shape == (
        n_groups,
        o_lora_rank // _QUANT_BLOCK,
        reduction // _QUANT_BLOCK,
    )


@triton.jit
def _o_proj_inv_rope_quant(
    O,
    Positions,
    Cache,
    Q,
    S,
    tokens,
    heads,
    heads_per_group: tl.constexpr,
    o_token_stride,
    o_head_stride,
    cache_stride,
    aligned_tokens,
    eps,
    HEADS: tl.constexpr,
    SCALE_DENSE: tl.constexpr,
):
    """Inverse interleaved RoPE + 128-group FP8 quantization for prefill.

    One CTA quantizes ``HEADS`` heads of one token: rows are (head, 128-block)
    pairs so each row is exactly one quantization group. Same math as the
    shared kernel, with the RoPE partner taken from the loaded tile (the pair
    swap over the last axis) instead of a second masked global load.

    ``SCALE_DENSE`` picks the scale layout: dense ``[group, token, column]``
    writes this CTA's ``HEADS * 4`` scales as one contiguous run (the TLE GEMM
    wants exactly that tile), while the token-major layout the plain GEMM's row
    load wants scatters them one per 64 KB-strided address. Measured at 4096
    tokens: 368 vs 310 us for the same kernel without the scale store.
    """
    token = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, HEADS * 4)
    head = tl.program_id(1) * HEADS + rows // 4
    chunk = rows % 4
    group = head // heads_per_group
    local_head = head % heads_per_group
    valid_head = head < heads
    scale_cols: tl.constexpr = heads_per_group * 4
    if SCALE_DENSE:
        scale_offsets = (
            group.to(tl.int64) * scale_cols * tokens
            + token * scale_cols
            + local_head * 4
            + chunk
        )
    else:
        scale_offsets = (
            group.to(tl.int64) * scale_cols * aligned_tokens
            + (local_head * 4 + chunk) * aligned_tokens
            + token
        )
    if token >= tokens:
        tl.store(S + scale_offsets, 0.0, valid_head)
        return

    cols = tl.arange(0, 128)
    source = (
        token * o_token_stride
        + head.to(tl.int64)[:, None] * o_head_stride
        + chunk[:, None] * 128
        + cols[None, :]
    )
    x = tl.load(O + source, valid_head[:, None], other=0.0).to(tl.float32)
    # partner[r, c] = x[r, c ^ 1]: swap the adjacent lanes in registers.
    pairs = tl.reshape(x, (HEADS * 4, 64, 2))
    even, odd = tl.split(pairs)
    partner = tl.reshape(tl.join(odd, even), (HEADS * 4, 128))

    # Every head of this token shares the same 32 cosine/sine entries.
    position = tl.load(Positions + token)
    cache_idx = tl.maximum((cols - 64) >> 1, 0)
    cosine = tl.load(Cache + position * cache_stride + cache_idx, cols >= 64, other=1.0)
    sine = tl.load(
        Cache + position * cache_stride + 32 + cache_idx, cols >= 64, other=0.0
    )
    x_add = x * cosine[None, :] + partner * sine[None, :]
    x_sub = x * cosine[None, :] - partner * sine[None, :]
    rotated = tl.where((cols[None, :] & 1) == 0, x_add, x_sub)
    is_rope = (chunk[:, None] == 3) & (cols[None, :] >= 64)
    x = tl.where(is_rope, rotated, x)
    quantized, scales = _quant_groups(x, eps, -448.0, 448.0, 1.0 / 448.0, False, False)
    target = (
        group.to(tl.int64)[:, None] * tokens * heads_per_group * 512
        + token * heads_per_group * 512
        + local_head[:, None] * 512
        + chunk[:, None] * 128
        + cols[None, :]
    )
    tl.store(Q + target, quantized.to(Q.dtype.element_ty), valid_head[:, None])
    tl.store(S + scale_offsets, scales, valid_head)


def _quant_heads(
    o,
    positions,
    cos_sin_cache,
    n_groups,
    heads_per_group,
    nope_dim,
    rope_dim,
    record,
    stream,
    dense_scales,
):
    """Inverse RoPE + group FP8 quant for every batch size.

    Same kernel and math as ``fused_inv_rope_fp8_quant``; only the tile width
    and warp count differ. That wrapper tiles eight heads per CTA from 128
    tokens up, which measures 11-13% slower on the S5000 than four heads at
    every prefill point (4096/8: 490 -> 430 us, 4096/16: 922 -> 806 us,
    8192/8: 929 -> 830 us, 2048/16: 486 -> 434 us), because the eight-head tile
    halves the CTA count while each CTA is already only a few KB of traffic.
    Decode keeps the wrapper's one-head tile, which it picked for the same
    reason in reverse (from 64 tokens down the batch has more CTAs than the
    chip has lanes, so a narrow tile is what keeps them short).

    ``dense_scales`` emits the scale block the TLE GEMM consumes directly
    (``[groups, tokens, columns]``); the token-major layout the plain GEMM's
    row load wants is then only built for the shapes that use it.

    The output and scale buffers never escape this operator, so they are kept
    in the signature's slot and reused. They are keyed by stream as well, so a
    second stream cannot overtake a buffer that the first one still reads.
    """
    heads = o.shape[1]
    tokens = o.shape[0]
    heads_per_program = 1 if tokens < 128 else 4
    aligned_tokens = triton.cdiv(tokens, 4) * 4
    scale_cols = heads_per_group * 4
    key = (stream, tokens, heads, n_groups, heads_per_group, dense_scales)
    workspace = record.get("ws")
    if workspace is None or workspace[0] != key:
        output = torch.empty(
            (n_groups, tokens, heads_per_group * (nope_dim + rope_dim)),
            device=o.device,
            dtype=torch.float8_e4m3fn,
        )
        if dense_scales:
            scales = torch.empty(
                (n_groups, tokens, scale_cols), device=o.device, dtype=torch.float32
            )
            tokens_axis = tokens
        else:
            scales = torch.empty(
                n_groups * scale_cols * aligned_tokens,
                device=o.device,
                dtype=torch.float32,
            ).as_strided(
                (n_groups, tokens, scale_cols),
                (scale_cols * aligned_tokens, 1, aligned_tokens),
            )
            tokens_axis = aligned_tokens
        workspace = (
            key,
            output,
            scales,
            output.transpose(0, 1),
            scales.transpose(0, 1),
            tokens_axis,
        )
        record["ws"] = workspace
    _, output, scales, output_t, scales_t, tokens_axis = workspace
    _dispatch(
        _o_proj_inv_rope_quant,
        (tokens_axis, triton.cdiv(heads, heads_per_program)),
        (
            o,
            positions,
            cos_sin_cache,
            output,
            scales,
            tokens,
            heads,
            heads_per_group,
            o.stride(0),
            o.stride(1),
            cos_sin_cache.stride(0),
            aligned_tokens,
            1e-10,
        ),
        {
            "HEADS": heads_per_program,
            "SCALE_DENSE": dense_scales,
            "num_warps": 1 if heads_per_program == 1 else 4,
            "num_stages": 1,
        },
        record,
        "quant",
        stream,
    )
    return output_t, scales_t


def _use_tle(num_tokens: int, n_groups: int) -> bool:
    """Whether the TLE (SQMMA warp-specialized) GEMM beats the plain one.

    S5000 sweep of the o_proj shapes: the TLE kernel runs at 280-313 TFLOPS
    against the plain kernel's 165, but it needs enough work to fill its
    persistent 60-CTA grid and the JIT dispatch over its 41 parameters costs
    ~160 us per call (with the descriptor/extern-libs caches below; it was
    ~280 us before them). Operator-level A/B including the quant: 512/8 269 vs
    326 us, 512/16 410 vs 568 us, 1024/8 440 vs 573 us, 4096/16 3031 vs 3696 us,
    while 128/8 (281 vs 119 us) and 256/8 (276 vs 199 us) still favour the
    plain kernel.
    """
    override = os.environ.get("FLAGGEMS_O_PROJ_TLE")
    if override is not None:
        return override != "0" and TensorDescriptor is not None
    if TensorDescriptor is None:
        return False
    return num_tokens >= 512 or (num_tokens >= 256 and n_groups > 8)


# (BLOCK_M, BLOCK_N, CONSUMER_WARPS, SPLIT_K, PROMOTE) for the TLE path; the
# 128x128/16-warp tile is the fastest at every token count above the crossover.
_TLE_TILE = (128, 128, 16, 1, 0)


def _tle_bmm(
    act, act_scale, weight, weight_scale, out, num_tokens, n_groups, record, stream
):
    """Block-scaled FP8 batched GEMM through the TLE warp-specialized kernel.

    ``act``/``act_scale`` are token-major ``[tokens, groups, k]``; the kernel
    walks the activation and its scales as ``[groups, tokens, *]`` (the
    ``X_BATCH_MAJOR=1`` layout its TMA descriptor wants) and the weight as
    ``[groups * n, k]``. A copy only happens for pre-quantized activations,
    whose token-major slice is not group-major contiguous.
    """
    block_m, block_n, warps, split_k, promote = _TLE_TILE
    k = act.shape[2]
    n = weight.shape[1]
    k_blocks = k // _QUANT_BLOCK
    act_g = act.transpose(0, 1)
    if not act_g.is_contiguous():
        act_g = act_g.contiguous()
    # The quant buffer's scales are padded on the token axis. Re-laying them as
    # a dense [groups, tokens, k_blocks] block is not just descriptor hygiene:
    # the kernel's per-block row-scale load then walks 128 consecutive floats,
    # and measured 14% faster at 4096 tokens than indexing the padded buffer.
    scale_g = act_scale.transpose(0, 1)
    if not scale_g.is_contiguous():
        scale_g = scale_g.contiguous()
    row_scale_pad = max(4, triton.next_power_of_2(k_blocks // split_k))
    out_g = out.permute(1, 0, 2)
    derived = _derived_values(
        block_m,
        block_n,
        warps,
        split_k,
        num_tokens,
        n,
        k_blocks,
        n_groups,
        out_g.device.index,
        out_g.stride(),
    )
    scale_desc = TensorDescriptor.from_tensor(
        scale_g.view(n_groups * num_tokens, k_blocks),
        block_shape=[block_m, row_scale_pad],
    )

    x_desc = TensorDescriptor.from_tensor(
        act_g.view(n_groups * num_tokens, k), block_shape=[block_m, _QUANT_BLOCK]
    )
    y_desc = _weight_desc(
        weight, n_groups * n, k, block_n // (2 if block_m == 256 else 1)
    )
    jit_fn = getattr(ws_gemm_fp8.fn, "fn", ws_gemm_fp8.fn)
    _dispatch(
        jit_fn,
        (derived["NUM_CTAS"],),
        (
            x_desc,
            y_desc,
            scale_desc,
            scale_g,
            weight_scale,
            out_g,
            out_g,
            num_tokens,
            n,
            k_blocks,
            scale_g.stride(1),
            scale_g.stride(0),
            scale_g.stride(2),
            weight_scale.stride(0),
            weight_scale.stride(1),
            weight_scale.stride(2),
            out_g.stride(1),
            out_g.stride(0),
            out_g.stride(2),
            derived["num_units"],
            derived["num_m_tiles"],
            derived["partial_plane"],
        ),
        {
            "BATCH": n_groups,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "CONSUMER_WARPS": warps,
            "SPLIT_K": split_k,
            "PROMOTE": promote,
            "NUM_N_TILES": derived["NUM_N_TILES"],
            "UNIT_K_BLOCKS": derived["UNIT_K_BLOCKS"],
            "GROUP_M": derived["GROUP_M"],
            "NUM_CTAS": derived["NUM_CTAS"],
            "STAGES": derived["STAGES"],
            "MMA_LAYOUT": derived["MMA_LAYOUT"],
            "ROW_LAYOUT": derived["ROW_LAYOUT"],
            "N_SLICES": derived["N_SLICES"],
            "STAGE_ROW_SCALES": derived["STAGE_ROW_SCALES"],
            "ROW_SCALE_PAD": derived["ROW_SCALE_PAD"],
            "Z_STRIDE_M": derived["Z_STRIDE_M"],
            "Z_STRIDE_B": derived["Z_STRIDE_B"],
            "X_BATCH_MAJOR": 1,
            "llc_options": LLC_OPTIONS,
            "extern_libs": _extern_libs(),
            "num_warps": warps,
        },
        record,
        "tle",
        stream,
    )


def _weight_desc(weight, rows, k, block_n):
    """Weight TMA descriptor; the weight tensor is a model parameter."""
    key = (weight.data_ptr(), rows, k, block_n)
    desc = _TLE_WEIGHT_DESCS.get(key)
    if desc is None:
        desc = TensorDescriptor.from_tensor(
            weight.view(rows, k), block_shape=[block_n, _QUANT_BLOCK]
        )
        if len(_TLE_WEIGHT_DESCS) > 32:
            _TLE_WEIGHT_DESCS.clear()
        _TLE_WEIGHT_DESCS[key] = desc
    return desc


def _extern_libs():
    global _EXTERN_LIBS
    if _EXTERN_LIBS is None:
        _EXTERN_LIBS = extern_libs_thread_index()
    return _EXTERN_LIBS


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
    """DeepSeek-V4 output projection on MThreads (native FP8 tensor cores)."""
    logger.debug("GEMS DEEP GEMM FP8 O PROJ (MTHREADS)")

    if tuple(einsum_recipe) != _SUPPORTED_RECIPE or tma_aligned_scales:
        raise NotImplementedError(
            "the MThreads implementation only supports the FP32 block-scale "
            f"recipe (1, 128, 128), got einsum_recipe={tuple(einsum_recipe)}, "
            f"tma_aligned_scales={tma_aligned_scales}"
        )
    assert o_lora_rank % _QUANT_BLOCK == 0, "o_lora_rank must be a multiple of 128"

    head_dim = nope_dim + rope_dim
    reduction = heads_per_group * head_dim
    weight, weight_scale = _weight_and_scale(wo_a)
    prequantized = _is_quantized_activation(o)

    if prequantized:
        act, act_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
        num_tokens, device = act.shape[0], act.device
    else:
        act, act_scale = o, None
        num_tokens, device = o.shape[0], o.device

    # Entering the device context costs a set/restore pair per call (~10 us on
    # MUSA), and the benchmark's interval starts before this call, so it shows
    # up whole in the measured latency. Skip it when the device already matches.
    current = torch.musa.current_device()
    target = device.index if device.index is not None else current

    # The support check walks shapes/strides/dtypes (~6 us). It only depends on
    # the tensor metadata, so keep the verdict for the shapes this model repeats
    # every step; the benchmark interval starts before this call, so the check
    # lands in the measured latency. The verdict's slot also holds this
    # signature's cached kernel launchers, which is why the device is part of
    # the key: a compiled kernel belongs to one device.
    check_key = (
        prequantized,
        target,
        act.shape,
        act.dtype,
        act.stride(),
        weight.shape,
        weight.dtype,
        weight.stride(),
        weight_scale.shape,
        weight_scale.dtype,
        getattr(positions, "shape", None),
        getattr(positions, "dtype", None),
        getattr(positions, "stride", lambda: None)(),
        getattr(cos_sin_cache, "shape", None),
        getattr(cos_sin_cache, "dtype", None),
        getattr(cos_sin_cache, "stride", lambda: None)(),
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        o_lora_rank,
    )
    if check_key not in _SUPPORT_CHECKED:
        if prequantized:
            assert act.ndim == 3, "`o.data` must be [num_tokens, n_groups, k]"
            assert act.dtype == torch.float8_e4m3fn, "`o.data` must be float8_e4m3fn"
            activation_ok = (
                act.shape[1:] == (n_groups, reduction)
                and act.stride(-1) == 1
                and act_scale.dtype == torch.float32
                and act_scale.shape == (num_tokens, n_groups, reduction // _QUANT_BLOCK)
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
            # The MThreads quant kernel implements the model's 448 + 64 layout
            # with 128-wide groups; anything else has no producer here.
            activation_ok = (
                o.stride(-1) == 1
                and (nope_dim, rope_dim, head_dim) == (448, 64, 512)
                and cos_sin_cache.dtype == torch.float32
                and cos_sin_cache.stride(-1) == 1
                and positions.dtype == torch.int64
                and positions.stride(0) == 1
            )

        if not (
            activation_ok
            and _weight_supported(
                weight, weight_scale, n_groups, o_lora_rank, reduction
            )
        ):
            raise NotImplementedError(
                "unsupported shape for the MThreads deep_gemm_fp8_o_proj: expected "
                f"activation [tokens, {n_groups}, {reduction}] (FP8 pre-quantized or "
                "BF16 attention output with head_dim 512 = 448 + 64), 128x128 block "
                "scales and an FP8 wo_a"
            )
        _SUPPORT_CHECKED[check_key] = record = {}
    else:
        record = _SUPPORT_CHECKED[check_key]

    if target == current:
        ctx = contextlib.nullcontext()
    else:
        ctx = torch_device_fn.device(device)
    with ctx:
        if num_tokens == 0:
            empty = torch.empty(
                (0, n_groups * o_lora_rank), dtype=torch.bfloat16, device=device
            )
            return wo_b(empty)
        stream = _current_stream()

        # Everything the GEMM needs that does not depend on the quant result is
        # prepared before the quant launch: the two kernels are only ~6 us and
        # ~60 us apart at decode, so Python work placed between them shows up as
        # an idle gap inside the measured interval.
        out = torch.empty(
            (num_tokens, n_groups, o_lora_rank), dtype=torch.bfloat16, device=device
        )
        use_tle = TensorDescriptor is not None and _use_tle(num_tokens, n_groups)
        out_g = out.permute(1, 0, 2)
        block_m, block_n, group_m, num_warps, num_stages = _tile_config(
            num_tokens, o_lora_rank
        )
        grid = (
            triton.cdiv(num_tokens, block_m) * (o_lora_rank // block_n),
            n_groups,
        )
        weight_strides = weight.stride()
        weight_scale_strides = weight_scale.stride()
        out_strides = out_g.stride()
        gemm_kwargs = {
            "N": o_lora_rank,
            "K": reduction,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": _QUANT_BLOCK,
            "GROUP_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

        if not prequantized:
            act, act_scale = _quant_heads(
                o,
                positions,
                cos_sin_cache,
                n_groups,
                heads_per_group,
                nope_dim,
                rope_dim,
                record,
                stream,
                use_tle,
            )

        if use_tle:
            _tle_bmm(
                act,
                act_scale,
                weight,
                weight_scale,
                out,
                num_tokens,
                n_groups,
                record,
                stream,
            )
        else:
            # Both producers hand back token-major tensors; the kernel walks them
            # group-major, which is the same storage with the leading strides
            # swapped, so this stays a view.
            act_g = act.transpose(0, 1)
            act_scale_g = act_scale.transpose(0, 1)
            act_strides = act_g.stride()
            act_scale_strides = act_scale_g.stride()
            _dispatch(
                _o_proj_bmm_kernel,
                grid,
                (
                    act_g,
                    act_scale_g,
                    weight,
                    weight_scale,
                    out_g,
                    num_tokens,
                    act_strides[0],
                    act_strides[1],
                    act_strides[2],
                    act_scale_strides[0],
                    act_scale_strides[1],
                    act_scale_strides[2],
                    weight_strides[0],
                    weight_strides[1],
                    weight_strides[2],
                    weight_scale_strides[0],
                    weight_scale_strides[1],
                    weight_scale_strides[2],
                    out_strides[0],
                    out_strides[1],
                    out_strides[2],
                ),
                gemm_kwargs,
                record,
                "gemm",
                stream,
            )

    assert callable(wo_b), "`wo_b` must be a callable linear layer"
    return wo_b(out.flatten(1))


