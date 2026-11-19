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

"""Hopper (H100) ``deep_gemm_fp8_o_proj`` decode fast path.

Decode steps (``num_tokens <= 128``) spend most of their wall time outside the
GPU. Measured on H100 for ``(T=16, g=8)`` in the repo benchmark: 84 us total,
while the two kernels only take ~30 us. The rest is FlagGems' per-call Python:
``fp8_einsum`` re-validates every tensor before forwarding into
``w8a8_block_fp8_bmm``, which validates again, and both launches go through the
argument re-binding in ``LibEntry.run``.

This backend keeps the same quantization kernel but calls both kernels one level
lower: the activation is produced by ``fused_inv_rope_fp8_quant`` and the GEMM
goes straight to a grouped FP8 BMM with a preallocated output, so the per-call
Python of ``fp8_einsum`` and ``LibEntry.run`` is skipped. The GEMM runs on the
TMA + warp-specialized kernel further down (one producer warp issuing TMA
copies, wgmma consumers), which measured faster than the masked-load kernel at
every core shape; ``FLAGGEMS_O_PROJ_WS_BMM=0`` selects the older kernel. The
scales, layouts and math are identical to the generic composition.

``FLAGGEMS_O_PROJ_FUSED_DECODE=1`` additionally enables the experimental
single-kernel path (inverse RoPE + quant fused into the GEMM K loop) for
``T <= 16``. It is correct but measured slower than the composition, so it
stays opt-in.
"""

import os
from typing import Callable, Optional, Sequence, Union

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.deep_gemm_fp8_o_proj import (
    _SUPPORTED_RECIPE,
    _is_quantized_activation,
    _weight_and_scale,
)
from flaggems_vllm.ops.deep_gemm_fp8_o_proj import (
    deep_gemm_fp8_o_proj as _generic_deep_gemm_fp8_o_proj,
)
from flaggems_vllm.ops.fused_inv_rope_fp8_quant import (
    _fused_inv_rope_fp8_quant_per_head,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
    w8a8_block_fp8_bmm as _bmm_mod,
)
from flaggems_vllm.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_bmm import (
    w8a8_block_fp8_bmm as _block_fp8_bmm,
)

# Decode regime covered by the fused kernel, and the fixed tiles it uses.
# T=32 is excluded: a 32-row tile measured 4x slower than T<=16 on H100.
_FUSED_MAX_TOKENS = 16
# The direct launch is also faster at prefill: 128x128 tiles measured 1.04-1.09x
# over the general kernel's autotuned choice at M=4096, so the lean path covers
# every token count.
_FAST_MAX_TOKENS = 1 << 30
_SCALE_BLOCK_N = 128
_BLOCK_N = 128
_GROUP_M = 1
_BLOCK_M = 16
_NUM_WARPS = 8
_NUM_STAGES = 3

# Opt-in until the schedule beats the generic composition. Measured on H100
# (80 GB HBM3, T=1..16, g=8, 128x128 scales): generic 86-95 us vs fused
# 112-123 us across every tile tried (BN in 64/128/256, warps 4/8,
# stages 2..5). The fused kernel pays for the per-lane RoPE gathers and loses
# the tuned BMM's pipelining, which outweighs skipping the FP8 activation
# round trip at this size.
_ENABLE_FUSED_DECODE = os.environ.get("FLAGGEMS_O_PROJ_FUSED_DECODE", "0") == "1"

# Decode GEMM tile used by the direct launch below. It trades a slightly larger
# tile (128 rows) for skipping `LibEntry.run`'s per-call argument rebinding:
# measured on H100 (T<=128, g=8/16) the direct launch costs ~19.5 us of host
# time against ~60 us through `w8a8_block_fp8_bmm`, which dominates the tile
# difference.
_DIRECT_BLOCK_M = 128
_DIRECT_BLOCK_N = 128
_DIRECT_GROUP_M = 1
_DIRECT_WARPS = 8
_DIRECT_STAGES = 4


def _bmm_jit_function():
    fn = _bmm_mod.w8a8_block_fp8_bmm_kernel_general
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = getattr(fn, "fn", None)
        if fn is None:
            raise RuntimeError("cannot unwrap w8a8_block_fp8_bmm kernel")
    return fn


try:
    _DIRECT_BMM_KERNEL = _bmm_jit_function()
except RuntimeError:  # pragma: no cover - defensive for other repo revisions
    _DIRECT_BMM_KERNEL = None


_COMPILED_LAUNCHERS = {}

# Activation scratch is consumed inside the same call (quant -> GEMM), so it can
# be reused per shape. Only decode-sized buffers are kept: they are a few MiB,
# while a prefill buffer would pin hundreds of MiB per shape.
_QUANT_SCRATCH = {}
_QUANT_SCRATCH_LIMIT = 32 << 20  # bytes


def _launch_compiled(name, jit_fn, grid, positional, launch_options, key=None):
    """Launch a Triton kernel through its bound CompiledKernel when possible.

    ``jit_fn[grid](...)`` re-binds arguments and rebuilds the specialization key
    on every call (~11 us on this FlagTree build). Once the kernel exists, the
    bound runner only pays the driver launch (~5.5 us). The cache key includes
    the grid and the pointer alignment Triton specializes on, and any capture
    failure silently falls back to the normal dispatch path.

    ``key`` overrides the derived key for launches whose arguments (TMA
    descriptors) are not tensors and must not be hashed by identity.
    """
    if key is None:
        key_parts = []
        for arg in positional:
            if isinstance(arg, torch.Tensor):
                # Triton specializes on 16-byte pointer alignment.
                key_parts.append(arg.data_ptr() % 16)
            else:
                key_parts.append(arg)
        key = (name, grid, tuple(key_parts))
    runner = _COMPILED_LAUNCHERS.get(key)
    if runner is not None:
        runner(*positional)
        return

    device = torch.cuda.current_device()
    kernel_cache = jit_fn.device_caches[device][0]
    before = set(kernel_cache)
    jit_fn[grid](*positional, **launch_options)
    new_keys = [k for k in kernel_cache if k not in before]
    if len(new_keys) == 1:
        compiled = kernel_cache[new_keys[0]]
        try:
            _COMPILED_LAUNCHERS[key] = compiled[(grid[0], grid[1], grid[2])]
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass


def _direct_block_fp8_bmm(x, xs, weight, weight_scale, out):
    """Launch the tuned grouped FP8 BMM without FlagGems' host wrappers.

    ``x``/``xs`` must already be in the kernel's ``[n_groups, tokens, k]``
    layout, which ``_quant_lean`` produces directly.
    """
    z = out.permute(1, 0, 2)
    b, m, k = x.shape
    n = weight.shape[1]
    grid = (
        b * triton.cdiv(m, _DIRECT_BLOCK_M) * triton.cdiv(n, _DIRECT_BLOCK_N),
        1,
        1,
    )
    positional = (
        x,
        weight,
        xs,
        z,
        weight_scale,
        b,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        xs.stride(0),
        xs.stride(1),
        xs.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        weight_scale.stride(0),
        weight_scale.stride(1),
        weight_scale.stride(2),
        _SCALE_BLOCK_N,
        _DIRECT_BLOCK_M,
        _DIRECT_BLOCK_N,
        _SCALE_BLOCK_N,
        _DIRECT_GROUP_M,
    )
    _launch_compiled(
        "block_fp8_bmm",
        _DIRECT_BMM_KERNEL,
        grid,
        positional,
        {"num_warps": _DIRECT_WARPS, "num_stages": _DIRECT_STAGES},
    )


# --------------------------------------------------------------------------
# TMA + warp-specialized grouped FP8 GEMM.
#
# The composition above moves the weight once per call, so the GEMM is a pure
# streaming problem: 32 MiB of fp8 weights for g=8 at decode, 1-2 GiB of tile
# traffic at prefill. A producer warp issues TMA loads into a staged shared
# buffer and the consumer warps run wgmma out of it, which keeps the copies off
# the math warps' critical path. Measured on H100 against the masked-load
# kernel above (kernel mode, cold L2): T=16 23.3 us vs 30.4, T=128 28.4 vs
# 33.1, T=4096 302 vs 356.
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _HAS_WS_BMM = hasattr(tle.gpu, "warp_specialize") and hasattr(tle.gpu, "wgmma")
except ImportError:  # pragma: no cover - FlagTree builds without TLE
    tle = None
    TensorDescriptor = None
    _HAS_WS_BMM = False

# Decode keeps 16-row tiles: 8 groups x 16 N tiles = 128 CTAs on 132 SMs, the
# most parallelism available without splitting K. The mid bucket uses a 128-row
# tile with 64 N columns when there is only one group set, so the weight is read
# once while the grid stays at 128 CTAs; with 16 groups the extra M parallelism
# is already there and 128x128 minimises traffic. Prefill keeps one CTA per tile.
_ENABLE_WS_BMM = os.environ.get("FLAGGEMS_O_PROJ_WS_BMM", "1") == "1"

_WS_DESCRIPTORS = {}
_WS_DESCRIPTOR_LIMIT = 32
_ALLOCATOR_READY = False

# The fast quant kernel drops the second (permuted) load and the per-element
# divide. That pays off once the activation is large enough to be throughput
# bound (T=4096: 171 -> 162 us); at decode the kernel is at the small-kernel
# floor and the two variants measure the same, so the tuned one stays default
# there. ``FLAGGEMS_O_PROJ_FAST_QUANT=0`` disables it everywhere.
_FAST_QUANT = os.environ.get("FLAGGEMS_O_PROJ_FAST_QUANT", "1") == "1"
_FAST_QUANT_MIN_TOKENS = 256


def _ws_tile(m, b):
    """Return ``(BLOCK_M, BLOCK_N, stages, warps, swap_ab)`` for this shape.

    The tile is picked so the tile grid stays within one wave (``B * M/BLOCK_M *
    N/BLOCK_N <= 132``): every tile re-reads its ``BLOCK_N x K`` weight slice, so
    a BLOCK_M smaller than the token count multiplies the weight traffic. That
    made ``T=64`` read the weights 4-8x and cost 4x the T=128 time until the
    16-row tile was confined to ``T <= 16``.
    """
    if m <= 16:
        return 16, 64, 8, 4, True
    if m <= 64:
        return (64, 64, 4, 4, True) if b <= 8 else (128, 128, 4, 8, False)
    if m <= 128 and b <= 8:
        return 128, 64, 4, 4, False
    return 128, 128, 4, 8, False


def _ensure_tma_allocator(device):
    """Install Triton's scratch allocator once; TMA lowering may request it."""
    global _ALLOCATOR_READY
    if _ALLOCATOR_READY:
        return

    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)
    _ALLOCATOR_READY = True


@triton.jit
def _ws_bmm_tile(
    tile_id,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    tiles_per_batch: tl.constexpr,
):
    batch_id = tile_id // tiles_per_batch
    local_id = tile_id % tiles_per_batch
    m_tile_id = local_id // num_n_tiles
    n_tile_id = local_id % num_n_tiles
    return batch_id, m_tile_id, n_tile_id


@triton.jit
def _ws_bmm_compute(
    x_smem,
    y_smem,
    empty_x,
    empty_y,
    full_x,
    full_y,
    xs_ptr,
    z_ptr,
    ys_ptr,
    xs_sB: tl.constexpr,
    xs_sM: tl.constexpr,
    xs_sKb: tl.constexpr,
    z_sB: tl.constexpr,
    z_sM: tl.constexpr,
    z_sN: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
    M_aligned: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    num_sms: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_m_tiles: tl.constexpr = M_aligned // BLOCK_M
    num_n_tiles: tl.constexpr = N // BLOCK_N
    num_k_blocks: tl.constexpr = K // BLOCK_K
    tiles_per_batch: tl.constexpr = num_m_tiles * num_n_tiles
    num_tiles: tl.constexpr = B * tiles_per_batch
    lane = tl.arange(0, BLOCK_M)

    for tile_id in range(start_pid, num_tiles, num_sms):
        batch_id, m_tile_id, n_tile_id = _ws_bmm_tile(
            tile_id, num_m_tiles, num_n_tiles, tiles_per_batch
        )
        m_start = m_tile_id * BLOCK_M
        n_start = n_tile_id * BLOCK_N
        # The weight scale grid is (N/128, K/128): one scalar per k block.
        ys_base = (batch_id * (N // 128) + n_start // 128) * num_k_blocks
        offs_m = m_start + lane
        xs_row = batch_id * xs_sB + offs_m * xs_sM
        xs_mask = offs_m < M

        if SWAP_AB:
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, num_k_blocks):
            index = k_block % NUM_STAGES
            phase = k_block // NUM_STAGES
            x_slot = x_smem.slot(index).slot(0)
            y_slot = y_smem.slot(index)
            tle.gpu.barrier_wait(full_x[index], phaseIdx=phase)
            tle.gpu.barrier_wait(full_y[index], phaseIdx=phase)

            x_s = tl.load(xs_ptr + xs_row + k_block * xs_sKb, mask=xs_mask, other=0.0)
            y_s = tl.load(ys_ptr + ys_base + k_block)
            xy_s = x_s * y_s

            if SWAP_AB:
                partial = tle.gpu.wgmma(
                    y_slot, x_slot, out_dtype=tl.float32, trans_b=True
                )
                partial = tle.gpu.wgmma_wait(0, partial)
                acc = acc + partial * xy_s[None, :]
            else:
                partial = tle.gpu.wgmma(
                    x_slot, y_slot, out_dtype=tl.float32, trans_b=True
                )
                partial = tle.gpu.wgmma_wait(0, partial)
                acc = acc + partial * xy_s[:, None]

            tle.gpu.barrier_arrive(empty_x[index], phaseIdx=phase)
            tle.gpu.barrier_arrive(empty_y[index], phaseIdx=phase)

        offs_n = n_start + tl.arange(0, BLOCK_N)
        acc_out = tl.trans(acc) if SWAP_AB else acc
        z_ptrs = (
            z_ptr + batch_id * z_sB + offs_m[:, None] * z_sM + offs_n[None, :] * z_sN
        )
        z_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(z_ptrs, acc_out.to(z_ptr.dtype.element_ty), mask=z_mask)


@triton.jit
def _ws_bmm_load(
    x_desc,
    y_desc,
    x_smem,
    y_smem,
    empty_x,
    empty_y,
    full_x,
    full_y,
    B: tl.constexpr,
    M: tl.constexpr,
    M_aligned: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    num_sms: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_m_tiles: tl.constexpr = M_aligned // BLOCK_M
    num_n_tiles: tl.constexpr = N // BLOCK_N
    tiles_per_batch: tl.constexpr = num_m_tiles * num_n_tiles
    num_tiles: tl.constexpr = B * tiles_per_batch

    for tile_id in range(start_pid, num_tiles, num_sms):
        batch_id, m_tile_id, n_tile_id = _ws_bmm_tile(
            tile_id, num_m_tiles, num_n_tiles, tiles_per_batch
        )
        m_start = m_tile_id * BLOCK_M
        n_start = n_tile_id * BLOCK_N
        # ``y`` is viewed as (B*N, K); rows outside [batch*N, batch*N+N) are OOB.
        y_row = batch_id * N + n_start

        for k_block in range(0, K // BLOCK_K):
            index = k_block % NUM_STAGES
            phase = k_block // NUM_STAGES
            k_start = k_block * BLOCK_K
            tle.gpu.barrier_wait(empty_x[index], phaseIdx=phase)
            tle.gpu.copy(
                x_desc,
                x_smem.slot(index),
                [1, BLOCK_M, BLOCK_K],
                [batch_id, m_start, k_start],
                barrier=full_x[index],
            )
            tle.gpu.barrier_wait(empty_y[index], phaseIdx=phase)
            tle.gpu.copy(
                y_desc,
                y_smem.slot(index),
                [BLOCK_N, BLOCK_K],
                [y_row, k_start],
                barrier=full_y[index],
            )


@triton.jit
def _ws_bmm_kernel(
    x_desc,
    y_desc,
    xs_ptr,
    z_ptr,
    ys_ptr,
    xs_sB: tl.constexpr,
    xs_sM: tl.constexpr,
    xs_sKb: tl.constexpr,
    z_sB: tl.constexpr,
    z_sM: tl.constexpr,
    z_sN: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
    M_aligned: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWAP_AB: tl.constexpr,
    X_ELEM_BYTES: tl.constexpr,
    Y_ELEM_BYTES: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    LOAD_WARPS: tl.constexpr,
    LOAD_REGS: tl.constexpr,
    num_sms: tl.constexpr,
):
    x_smem = tle.gpu.alloc(
        [NUM_STAGES, 1, BLOCK_M, BLOCK_K],
        dtype=x_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
    )
    y_smem = tle.gpu.alloc(
        [NUM_STAGES, BLOCK_N, BLOCK_K],
        dtype=y_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
    )
    empty_x = tle.gpu.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=1, init=tle.gpu.READY
    )
    empty_y = tle.gpu.alloc_barriers(
        num_barriers=NUM_STAGES, arrive_count=1, init=tle.gpu.READY
    )
    full_x = tle.gpu.alloc_barriers(
        num_barriers=NUM_STAGES,
        arrive_count=1,
        expect_bytes=BLOCK_M * BLOCK_K * X_ELEM_BYTES,
    )
    full_y = tle.gpu.alloc_barriers(
        num_barriers=NUM_STAGES,
        arrive_count=1,
        expect_bytes=BLOCK_N * BLOCK_K * Y_ELEM_BYTES,
    )

    tle.gpu.warp_specialize(
        [
            (
                _ws_bmm_compute,
                (
                    x_smem,
                    y_smem,
                    empty_x,
                    empty_y,
                    full_x,
                    full_y,
                    xs_ptr,
                    z_ptr,
                    ys_ptr,
                    xs_sB,
                    xs_sM,
                    xs_sKb,
                    z_sB,
                    z_sM,
                    z_sN,
                    B,
                    M,
                    M_aligned,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    SWAP_AB,
                    NUM_STAGES,
                    num_sms,
                ),
            ),
            (
                _ws_bmm_load,
                (
                    x_desc,
                    y_desc,
                    x_smem,
                    y_smem,
                    empty_x,
                    empty_y,
                    full_x,
                    full_y,
                    B,
                    M,
                    M_aligned,
                    N,
                    K,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    NUM_STAGES,
                    num_sms,
                ),
            ),
        ],
        [1],
        [24],
    )


def _ws_bmm_supported(x, xs, weight, weight_scale) -> bool:
    """The TMA path needs contiguous fp8 operands and a 128x128 scale grid."""
    if not (_ENABLE_WS_BMM and _HAS_WS_BMM):
        return False
    if not (
        x.is_contiguous() and weight.is_contiguous() and weight_scale.is_contiguous()
    ):
        return False
    if weight.dtype != torch.float8_e4m3fn or x.dtype != torch.float8_e4m3fn:
        return False
    if weight_scale.dtype != torch.float32 or xs.dtype != torch.float32:
        return False
    b, m, k = x.shape
    n = weight.shape[1]
    if k % _SCALE_BLOCK_N or n % _SCALE_BLOCK_N:
        return False
    if weight_scale.shape != (b, n // _SCALE_BLOCK_N, k // _SCALE_BLOCK_N):
        return False
    return True


def _ws_block_fp8_bmm(x, xs, weight, weight_scale, out):
    """Run the grouped FP8 BMM on the TMA + warp-specialized kernel."""
    _ensure_tma_allocator(x.device)
    z = out.permute(1, 0, 2)
    b, m, k = x.shape
    n = weight.shape[1]
    block_m, block_n, stages, warps, swap_ab = _ws_tile(m, b)
    m_aligned = triton.cdiv(m, block_m) * block_m

    # TMA descriptors embed the global shape, and the activation buffer is
    # reallocated per call once it is too large to cache, so pointer reuse must
    # not be mistaken for the same descriptor: a stale, smaller M would make TMA
    # clamp whole rows to zero.
    desc_key = (x.data_ptr(), weight.data_ptr(), b, m, n, k, block_m, block_n)
    descs = _WS_DESCRIPTORS.get(desc_key)
    if descs is None:
        if len(_WS_DESCRIPTORS) >= _WS_DESCRIPTOR_LIMIT:
            _WS_DESCRIPTORS.clear()
        descs = (
            TensorDescriptor.from_tensor(x, [1, block_m, _SCALE_BLOCK_N]),
            TensorDescriptor.from_tensor(
                weight.view(b * n, k), [block_n, _SCALE_BLOCK_N]
            ),
        )
        _WS_DESCRIPTORS[desc_key] = descs
    x_desc, y_desc = descs

    num_tiles = b * (m_aligned // block_m) * (n // block_n)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(num_sms, num_tiles), 1, 1)
    positional = (
        x_desc,
        y_desc,
        xs,
        z,
        weight_scale,
        xs.stride(0),
        xs.stride(1),
        xs.stride(2),
        z.stride(0),
        z.stride(1),
        z.stride(2),
        b,
        m,
        m_aligned,
        n,
        k,
        block_m,
        block_n,
        _SCALE_BLOCK_N,
        swap_ab,
        x.element_size(),
        weight.element_size(),
        stages,
        1,
        24,
        grid[0],
    )
    # Descriptors are not hashable and their pointers change with the scratch
    # buffers, so key the runner on the launch shape plus pointer alignment.
    key = (
        "ws_block_fp8_bmm",
        grid,
        block_m,
        block_n,
        stages,
        swap_ab,
        b,
        m,
        n,
        k,
        x.data_ptr() % 16,
        xs.data_ptr() % 16,
        z.data_ptr() % 16,
        weight_scale.data_ptr() % 16,
    )
    _launch_compiled(
        "ws_block_fp8_bmm",
        _ws_bmm_kernel,
        grid,
        positional,
        {"num_warps": warps},
        key=key,
    )


@triton.jit
def _fused_inv_rope_quant_gemm_kernel(
    O,
    POS,
    CACHE,
    W,
    WS,
    Z,
    T: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    heads_per_group: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    o_stride_t,
    o_stride_h,
    cache_stride_pos,
    w_stride_g,
    w_stride_r,
    w_stride_d,
    ws_stride_g,
    ws_stride_r,
    ws_stride_d,
    z_stride_t,
    z_stride_g,
    z_stride_r,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # The activation quantization grid is 128 elements wide; the K tile must
    # line up with it so `block_amax` matches the operator's scales.
    tl.static_assert(BLOCK_K == 128)
    pid = tl.program_id(0).to(tl.int64)
    num_pid_m = tl.cdiv(T, BLOCK_M)
    num_pid_n = tl.cdiv(R, BLOCK_N)
    tiles_per_group = num_pid_m * num_pid_n
    group_id = pid // tiles_per_group
    local_pid = pid % tiles_per_group

    num_pid_in_group = GROUP_M * num_pid_n
    pid_group = local_pid // num_pid_in_group
    first_pid_m = pid_group * GROUP_M
    pid_in_group = local_pid % num_pid_in_group
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

    m_mask = offs_m < T
    n_mask = offs_n < R
    head_base = group_id * heads_per_group

    positions = tl.load(POS + offs_m, mask=m_mask, other=0).to(tl.int64)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, D // BLOCK_K):
        k = k_block * BLOCK_K + offs_k
        head_in_group = k // head_dim
        dim = k % head_dim
        head = head_base + head_in_group
        is_rope = dim >= nope_dim

        o_offsets = (
            offs_m[:, None] * o_stride_t + head[None, :] * o_stride_h + dim[None, :]
        )
        x = tl.load(O + o_offsets, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # Interleaved inverse RoPE: lane ``d`` pairs with lane ``d ^ 1``.
        partner = k ^ 1
        partner_head = head_base + partner // head_dim
        partner_dim = partner % head_dim
        x_partner = tl.load(
            O
            + offs_m[:, None] * o_stride_t
            + partner_head[None, :] * o_stride_h
            + partner_dim[None, :],
            mask=m_mask[:, None] & is_rope[None, :],
            other=0.0,
        ).to(tl.float32)

        rope_mask = is_rope[None, :] & m_mask[:, None]
        cache_offsets = (
            positions[:, None] * cache_stride_pos + ((dim - nope_dim) >> 1)[None, :]
        )
        cos_v = tl.load(CACHE + cache_offsets, mask=rope_mask, other=1.0)
        sin_v = tl.load(
            CACHE + cache_offsets + rope_dim // 2, mask=rope_mask, other=0.0
        )
        is_even = (dim & 1) == 0
        rotated = tl.where(
            is_even[None, :],
            x * cos_v + x_partner * sin_v,
            x * cos_v - x_partner * sin_v,
        )
        x = tl.where(is_rope[None, :], rotated, x)

        # Per-token FP8 quant over this 128-element block, matching
        # ``fused_inv_rope_fp8_quant`` (amax / fp8_max, no UE8M0 rounding).
        block_amax = tl.max(tl.abs(x), axis=1)
        scale = tl.maximum(block_amax, eps) * (1.0 / fp8_max)
        x_fp8 = tl.clamp(x / scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)

        y = tl.load(
            W
            + group_id * w_stride_g
            + k[:, None] * w_stride_d
            + offs_n[None, :] * w_stride_r,
            mask=n_mask[None, :],
            other=0.0,
        )
        y_scale = tl.load(
            WS
            + group_id * ws_stride_g
            + (offs_n // 128) * ws_stride_r
            + k_block * ws_stride_d,
            mask=n_mask,
            other=0.0,
        )
        acc += tl.dot(x_fp8, y, out_dtype=tl.float32) * scale[:, None] * y_scale

    z_ptrs = (
        Z
        + offs_m[:, None] * z_stride_t
        + group_id * z_stride_g
        + offs_n[None, :] * z_stride_r
    )
    mask = m_mask[:, None] & n_mask[None, :]
    if Z.dtype.element_ty == tl.bfloat16:
        tl.store(z_ptrs, acc.to(tl.bfloat16), mask=mask)
    elif Z.dtype.element_ty == tl.float16:
        tl.store(z_ptrs, acc.to(tl.float16), mask=mask)
    else:
        tl.store(z_ptrs, acc.to(tl.float32), mask=mask)


def _fast_weights(
    wo_a,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
):
    """Return ``(weight, weight_scale)`` when the FP8 block layout is supported."""
    head_dim = nope_dim + rope_dim
    rope_offset = nope_dim % _SCALE_BLOCK_N
    if head_dim % _SCALE_BLOCK_N or rope_offset % 2:
        return None
    if rope_offset + rope_dim > _SCALE_BLOCK_N:
        return None

    try:
        weight, weight_scale = _weight_and_scale(wo_a)
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(weight, torch.Tensor) or weight.ndim != 3:
        return None
    if weight.dtype != torch.float8_e4m3fn or weight.stride(-1) != 1:
        return None
    if not isinstance(weight_scale, torch.Tensor):
        return None
    if weight_scale.ndim != 3 or weight_scale.dtype != torch.float32:
        return None

    group_dim = heads_per_group * head_dim
    if weight.shape != (n_groups, o_lora_rank, group_dim):
        return None
    if group_dim % _SCALE_BLOCK_N or o_lora_rank % _SCALE_BLOCK_N:
        return None
    if weight_scale.shape != (
        n_groups,
        o_lora_rank // _SCALE_BLOCK_N,
        group_dim // _SCALE_BLOCK_N,
    ):
        return None
    return weight, weight_scale


def _fast_shape(
    o,
    wo_a,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    max_tokens: int,
    einsum_recipe: Sequence[int],
    tma_aligned_scales: bool,
):
    """Return ``(weight, weight_scale)`` when a decode fast path applies."""
    if tma_aligned_scales or tuple(einsum_recipe) != tuple(_SUPPORTED_RECIPE):
        return None
    if not isinstance(o, torch.Tensor) or o.ndim != 3:
        return None
    if o.dtype != torch.bfloat16 or o.stride(-1) != 1:
        return None
    if not 0 < o.shape[0] <= max_tokens:
        return None
    if o.shape[1] != n_groups * heads_per_group:
        return None
    if o.shape[2] != nope_dim + rope_dim:
        return None
    return _fast_weights(
        wo_a, n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank
    )


@triton.jit
def _inv_rope_quant_fast_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
):
    """Inverse RoPE + FP8 quant, one (token, head) per CTA, cheaper math.

    Same grid and layout as ``_fused_inv_rope_fp8_quant_per_head`` but
    - the rope partner comes from a pair split of the already loaded row instead
      of a second permuted load of the same 512 elements,
    - the per-element divide becomes one reciprocal per 128-wide block.
    Both measured as the dominant cost of the quantized activation at decode.
    """
    pid_token = tl.program_id(0)
    pid_gh = tl.program_id(1)
    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    qb_start = head_in_group * CHUNKS_PER_HEAD

    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            tl.store(
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k,
                tl.zeros((), dtype=tl.int32),
            )
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            tl.store(
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + (qb_start + block_offsets) * scale_stride_k,
                tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32),
            )
        return

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    PAIRS: tl.constexpr = HEAD_DIM // 2
    input_base = o_ptr + pid_token * o_stride_token + pid_gh * o_stride_head
    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offs).to(tl.float32)

    # Interleaved inverse RoPE on the (even, odd) lane pairs of one row.
    rope_pair: tl.constexpr = (
        (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    ) // 2
    pair = tl.arange(0, PAIRS)
    in_rope = pair >= rope_pair
    even, odd = tl.split(tl.reshape(x, (PAIRS, 2)))
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    cs_idx = tl.where(in_rope, pair - rope_pair, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=in_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=in_rope, other=0.0)
    rot_even = tl.where(in_rope, even * cos_v + odd * sin_v, even)
    rot_odd = tl.where(in_rope, odd * cos_v - even * sin_v, odd)
    x = tl.reshape(tl.join(rot_even, rot_odd), (HEAD_DIM,))

    blocks = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    absmax = tl.maximum(tl.max(blocks, axis=1), eps)
    scales = absmax * (1.0 / fp8_max)
    if TMA_ALIGNED_SCALES:
        scales = tl.math.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scales), 1e-10))))
    inv = 1.0 / scales
    inv_exp = tl.reshape(
        tl.broadcast_to(inv[:, None], (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE)),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x * inv_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offs, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        packed = tl.sum(((scale_bits >> 23) & 0xFF) << (block_offsets * 8))
        tl.store(
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k,
            packed,
        )
    else:
        tl.store(
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + (qb_start + block_offsets) * scale_stride_k,
            scales,
        )


def _quant_lean(
    o,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    eps: float = 1e-10,
) -> tuple:
    """Inverse RoPE + FP8 quant, emitted in the BMM's ``[groups, tokens, k]``."""
    num_tokens, _, head_dim = o.shape
    chunks_per_head = head_dim // _SCALE_BLOCK_N
    group_dim = heads_per_group * head_dim
    scale_inner = group_dim // _SCALE_BLOCK_N
    aligned_t = (num_tokens + 3) // 4 * 4

    fp8_bytes = n_groups * num_tokens * group_dim
    scratch_key = (n_groups, num_tokens, group_dim, o.device.index)
    if fp8_bytes <= _QUANT_SCRATCH_LIMIT:
        buffers = _QUANT_SCRATCH.get(scratch_key)
    else:
        buffers = None
    if buffers is None:
        fp8_q = torch.empty(
            (n_groups, num_tokens, group_dim),
            dtype=torch.float8_e4m3fn,
            device=o.device,
        )
        scale = torch.empty(
            n_groups * scale_inner * aligned_t, dtype=torch.float32, device=o.device
        ).as_strided(
            (n_groups, num_tokens, scale_inner),
            (scale_inner * aligned_t, 1, aligned_t),
        )
        if fp8_bytes <= _QUANT_SCRATCH_LIMIT:
            _QUANT_SCRATCH[scratch_key] = (fp8_q, scale)
    else:
        fp8_q, scale = buffers
    grid = (aligned_t, n_groups * heads_per_group, 1)
    positional = (
        o,
        positions,
        cos_sin_cache,
        fp8_q,
        scale,
        num_tokens,
        heads_per_group,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        fp8_q.stride(0),
        fp8_q.stride(1),
        scale.stride(0),
        scale.stride(2),
        torch.finfo(torch.float8_e4m3fn).max,
        eps,
        _SCALE_BLOCK_N,
        chunks_per_head,
        nope_dim % _SCALE_BLOCK_N,
        rope_dim // 2,
        False,
    )
    use_fast = _FAST_QUANT and num_tokens >= _FAST_QUANT_MIN_TOKENS
    _launch_compiled(
        "inv_rope_fp8_quant_fast" if use_fast else "inv_rope_fp8_quant",
        _inv_rope_quant_fast_kernel if use_fast else _fused_inv_rope_fp8_quant_per_head,
        grid,
        positional,
        {"num_warps": 1, "num_stages": 1},
    )
    return fp8_q, scale


def _lean_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    wo_b,
    n_groups: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Grouped FP8 GEMM into a preallocated output, without the host wrappers.

    ``x`` / ``x_scale`` are the group-major ``[n_groups, tokens, k]`` activation.
    """
    out = torch.empty(
        (x.shape[1], n_groups, o_lora_rank),
        device=x.device,
        dtype=torch.bfloat16,
    )
    if _ws_bmm_supported(x, x_scale, weight, weight_scale):
        _ws_block_fp8_bmm(x, x_scale, weight, weight_scale, out)
    elif _DIRECT_BMM_KERNEL is not None:
        _direct_block_fp8_bmm(x, x_scale, weight, weight_scale, out)
    else:
        _block_fp8_bmm(
            x,
            weight,
            x_scale,
            weight_scale,
            block_size=(_SCALE_BLOCK_N, _SCALE_BLOCK_N),
            z=out.permute(1, 0, 2),
            output_dtype=torch.bfloat16,
        )
    return wo_b(out.flatten(1))


def _lean_decode(
    o,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    wo_b,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """Same kernels as the generic path, without the two host wrappers."""
    o_fp8, o_scale = _quant_lean(
        o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim, rope_dim
    )
    return _lean_gemm(o_fp8, o_scale, weight, weight_scale, wo_b, n_groups, o_lora_rank)


def _fused_decode(
    o,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    wo_b,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    num_tokens = o.shape[0]
    out = torch.empty(
        (num_tokens, n_groups, o_lora_rank), device=o.device, dtype=torch.bfloat16
    )
    grid = (
        triton.cdiv(num_tokens, _BLOCK_M)
        * triton.cdiv(o_lora_rank, _BLOCK_N)
        * n_groups,
    )
    _fused_inv_rope_quant_gemm_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        weight,
        weight_scale,
        out,
        T=num_tokens,
        R=o_lora_rank,
        D=weight.shape[2],
        heads_per_group=heads_per_group,
        head_dim=nope_dim + rope_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_stride_t=o.stride(0),
        o_stride_h=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        w_stride_g=weight.stride(0),
        w_stride_r=weight.stride(1),
        w_stride_d=weight.stride(2),
        ws_stride_g=weight_scale.stride(0),
        ws_stride_r=weight_scale.stride(1),
        ws_stride_d=weight_scale.stride(2),
        z_stride_t=out.stride(0),
        z_stride_g=out.stride(1),
        z_stride_r=out.stride(2),
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        eps=1e-10,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_SCALE_BLOCK_N,
        GROUP_M=_GROUP_M,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return wo_b(out.flatten(1))


def _fast_prequantized(
    o,
    wo_a,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: Sequence[int],
    tma_aligned_scales: bool,
):
    """Return ``(x, x_scale, weight, weight_scale)`` for a quantized activation.

    ``x`` / ``x_scale`` are converted to the group-major ``[groups, tokens, k]``
    layout the GEMM consumes; no copy is needed for the group-major buffers the
    quantization kernel writes.
    """
    if tma_aligned_scales or tuple(einsum_recipe) != tuple(_SUPPORTED_RECIPE):
        return None
    weights = _fast_weights(
        wo_a, n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank
    )
    if weights is None:
        return None
    x, x_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
    if x.ndim != 3 or x.shape[0] == 0:
        return None
    if x.shape[1:] != (n_groups, weights[0].shape[2]):
        return None
    if x.dtype != torch.float8_e4m3fn or x_scale.dtype != torch.float32:
        return None
    if x.stride(-1) != 1:
        return None
    if x_scale.shape != (x.shape[0], n_groups, weights[0].shape[2] // _SCALE_BLOCK_N):
        return None
    return x.permute(1, 0, 2), x_scale.permute(1, 0, 2), *weights


def deep_gemm_fp8_o_proj(
    o: Union[torch.Tensor, "QuantizedActivation"],
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a,
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
    """Hopper entry point: lean launches for supported shapes, generic otherwise.

    A pre-quantized ``QuantizedActivation`` (vLLM >= 0.32 already rotated and
    quantized it inside the attention kernel) skips the quantization step and
    goes straight to the grouped GEMM.
    """
    if _is_quantized_activation(o):
        prequantized = _fast_prequantized(
            o,
            wo_a,
            n_groups,
            heads_per_group,
            nope_dim,
            rope_dim,
            o_lora_rank,
            einsum_recipe,
            tma_aligned_scales,
        )
        if prequantized is not None:
            return _lean_gemm(*prequantized, wo_b, n_groups, o_lora_rank)
    shape_args = (
        o,
        wo_a,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        o_lora_rank,
    )
    recipe_args = (einsum_recipe, tma_aligned_scales)
    call_args = (
        o,
        positions,
        cos_sin_cache,
        wo_b,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
        o_lora_rank,
    )

    if _ENABLE_FUSED_DECODE:
        weights = _fast_shape(*shape_args, _FUSED_MAX_TOKENS, *recipe_args)
        if weights is not None:
            return _fused_decode(*call_args[:3], *weights, *call_args[3:])

    weights = _fast_shape(*shape_args, _FAST_MAX_TOKENS, *recipe_args)
    if weights is not None:
        return _lean_decode(*call_args[:3], *weights, *call_args[3:])

    return _generic_deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        wo_b,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        o_lora_rank=o_lora_rank,
        einsum_recipe=einsum_recipe,
        tma_aligned_scales=tma_aligned_scales,
    )
