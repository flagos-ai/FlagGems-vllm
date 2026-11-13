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

"""Inverse RoPE + FP8 quant + grouped WGMMA in one kernel launch, for small batches.

Two separate launches leave the GPU idle while Python starts the second one, which
dominates small batches. Here one launch has two phases:

1. Every CTA takes (token, group) items in turn and writes their quantized
   activations and scales to temporary buffers. Each item is computed once.
2. CTAs meet at a ticket barrier, then run the tile loop of
   ``w8a8_block_fp8_bmm_kernel`` on those buffers.

The barrier counter only grows and needs no reset, so a replayed launch still
synchronizes. The launch is cooperative, which keeps all CTAs resident while they
wait. Phase 1 runs on 4 warps per CTA, slower than a quant kernel that fills the SM,
so large batches should take the two-kernel path instead.

Autotune exemption: the configuration is fixed to the best measured one of
``w8a8_block_fp8_bmm`` on H20 (tile order 0, 4 warps, 4 stages). Going through
``@libentry`` and ``@libtuner`` would add back launch time, which is what this
kernel removes.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_bmm import (
    HAS_TLE_W8A8_BLOCK_FP8_BMM,
    tle,
)

_NUM_WARPS = 4
_NUM_STAGES = 4
_FP8_MAX = 448.0
_EPS = 1e-10
_COMPILED = {}


@triton.jit
def _fence_proxy_async_global(dummy):
    # Phase 1 writes x with ordinary stores and phase 2 reads it with TMA, which runs in
    # the async proxy. PTX orders accesses across proxies only through a proxy fence.
    tl.inline_asm_elementwise(
        "fence.proxy.async.global;\n\tmov.u32 $0, $1;",
        "=r,r",
        [dummy],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _get_tile(tile_id, num_m_tiles, num_n_tiles: tl.constexpr, num_tiles_per_batch):
    # Tile order 0: N is fastest within a batch, which reuses x across the N sweep.
    batch_id = tile_id // num_tiles_per_batch
    local_id = tile_id % num_tiles_per_batch
    return batch_id, local_id // num_n_tiles, local_id % num_n_tiles


@triton.jit
def _single_launch_compute_partition(
    x_smem,
    y_smem,
    empty_x,
    empty_y,
    full_x,
    full_y,
    quant_done,
    o_ptr,
    positions_ptr,
    cache_ptr,
    x_ptr,
    xs_ptr,
    counter_ptr,
    z_ptr,
    ys_ptr,
    T,
    T_aligned,
    x_sB,
    xs_sB,
    o_sT: tl.constexpr,
    o_sH: tl.constexpr,
    cache_sP: tl.constexpr,
    x_sM: tl.constexpr,
    xs_sM: tl.constexpr,
    z_sB: tl.constexpr,
    z_sM: tl.constexpr,
    z_sN: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    SWAP_AB: tl.constexpr,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    num_stages: tl.constexpr,
    num_sms: tl.constexpr,
):
    start_pid = tl.program_id(0)
    # These strides scale with the token count: keep the address math in int64.
    x_sB = x_sB.to(tl.int64)
    xs_sB = xs_sB.to(tl.int64)

    # ---- phase 1: each (token, group) once, all of its heads together --------
    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * BLOCK_K
    ROPE_ABS: tl.constexpr = (CHUNKS_PER_HEAD - 1) * BLOCK_K + ROPE_START
    NUM_BLOCKS: tl.constexpr = HEADS_PER_GROUP * CHUNKS_PER_HEAD
    heads = tl.arange(0, HEADS_PER_GROUP)
    offsets = tl.arange(0, HEAD_DIM)
    is_rope = offsets >= ROPE_ABS
    rope_local = offsets - ROPE_ABS
    cs_idx = tl.maximum(rope_local >> 1, 0)
    is_even = (rope_local & 1) == 0
    is_rope_2d = tl.broadcast_to(is_rope[None, :], (HEADS_PER_GROUP, HEAD_DIM))
    is_even_2d = tl.broadcast_to(is_even[None, :], (HEADS_PER_GROUP, HEAD_DIM))
    head_offs = heads[:, None] * o_sH
    x_offs = heads[:, None] * HEAD_DIM + offsets[None, :]
    block_offs = tl.arange(0, NUM_BLOCKS)
    for item in range(start_pid, T * B, num_sms):
        token = (item // B).to(tl.int64)
        group = item % B
        o_base = o_ptr + token * o_sT + group * HEADS_PER_GROUP * o_sH
        x = tl.load(o_base + head_offs + offsets[None, :]).to(tl.float32)
        partner = tl.load(
            o_base + head_offs + (offsets ^ 1)[None, :], mask=is_rope_2d, other=0.0
        ).to(tl.float32)
        cache_base = cache_ptr + tl.load(positions_ptr + token) * cache_sP
        cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
        sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
        x_add = x * cos_v[None, :] + partner * sin_v[None, :]
        x_sub = x * cos_v[None, :] - partner * sin_v[None, :]
        x = tl.where(is_rope_2d, tl.where(is_even_2d, x_add, x_sub), x)
        abs_blocks = tl.reshape(tl.abs(x), (NUM_BLOCKS, BLOCK_K))
        scales = tl.maximum(tl.max(abs_blocks, axis=1), eps) * (1.0 / fp8_max)
        scales_exp = tl.reshape(
            tl.broadcast_to(tl.reshape(scales, (NUM_BLOCKS, 1)), (NUM_BLOCKS, BLOCK_K)),
            (HEADS_PER_GROUP, HEAD_DIM),
        )
        x_fp8 = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(x_ptr + group * x_sB + token * x_sM + x_offs, x_fp8)
        tl.store(xs_ptr + group * xs_sB + token * xs_sM + block_offs, scales)

    # ---- ticket barrier: every CTA of this launch has finished phase 1 --------
    # Each launch adds exactly num_sms, so the next multiple closes this launch. The
    # launch is cooperative, so all CTAs are resident and none waits forever.
    _fence_proxy_async_global(start_pid)
    ticket = tl.atomic_add(counter_ptr, 1, sem="acq_rel")
    target = (ticket // num_sms + 1) * num_sms
    seen = ticket + 1
    while seen < target:
        seen = tl.atomic_add(counter_ptr, 0, sem="acquire")
    tle.gpu.barrier_arrive(quant_done, phaseIdx=0)

    # ---- phase 2: grouped WGMMA over the temporary buffers --------------------
    num_m_tiles = T_aligned // BLOCK_M
    num_n_tiles: tl.constexpr = N // BLOCK_N
    num_k_blocks: tl.constexpr = K // BLOCK_K
    num_tiles_per_batch = num_m_tiles * num_n_tiles
    num_tiles = B * num_tiles_per_batch
    lanes = tl.arange(0, BLOCK_M)
    for tile_id in range(start_pid, num_tiles, num_sms):
        batch_id, m_tile_id, n_tile_id = _get_tile(
            tile_id, num_m_tiles, num_n_tiles, num_tiles_per_batch
        )
        n_start = n_tile_id * BLOCK_N
        ys_base = (batch_id * num_n_tiles + n_tile_id) * num_k_blocks
        offs_m = m_tile_id * BLOCK_M + lanes
        m_mask = offs_m < T
        xs_row = batch_id * xs_sB + offs_m * xs_sM

        if SWAP_AB:
            acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block_idx in range(0, num_k_blocks):
            index = k_block_idx % num_stages
            phase = k_block_idx // num_stages
            x_slot = x_smem.slot(index).slot(0)
            y_slot = y_smem.slot(index)
            tle.gpu.barrier_wait(full_x[index], phaseIdx=phase)
            tle.gpu.barrier_wait(full_y[index], phaseIdx=phase)
            x_s = tl.load(xs_ptr + xs_row + k_block_idx, mask=m_mask, other=0.0)
            y_s = tl.load(ys_ptr + ys_base + k_block_idx)
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

        if SWAP_AB:
            acc_out = tl.trans(acc)
        else:
            acc_out = acc
        offs_n = n_start + tl.arange(0, BLOCK_N)
        z_ptrs = (
            z_ptr + batch_id * z_sB + offs_m[:, None] * z_sM + offs_n[None, :] * z_sN
        )
        tl.store(z_ptrs, acc_out, mask=m_mask[:, None] & (offs_n[None, :] < N))


@triton.jit
def _single_launch_load_partition(
    x_desc,
    y_desc,
    x_smem,
    y_smem,
    empty_x,
    empty_y,
    full_x,
    full_y,
    quant_done,
    T_aligned,
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_sms: tl.constexpr,
):
    # The activations do not exist until every CTA has finished phase 1.
    tle.gpu.barrier_wait(quant_done, phaseIdx=0)

    start_pid = tl.program_id(0)
    _fence_proxy_async_global(start_pid)
    num_m_tiles = T_aligned // BLOCK_M
    num_n_tiles: tl.constexpr = N // BLOCK_N
    num_tiles_per_batch = num_m_tiles * num_n_tiles
    num_tiles = B * num_tiles_per_batch
    for tile_id in range(start_pid, num_tiles, num_sms):
        batch_id, m_tile_id, n_tile_id = _get_tile(
            tile_id, num_m_tiles, num_n_tiles, num_tiles_per_batch
        )
        m_start = m_tile_id * BLOCK_M
        y_row = batch_id * N + n_tile_id * BLOCK_N
        for k_block_idx in range(0, K // BLOCK_K):
            index = k_block_idx % num_stages
            phase = k_block_idx // num_stages
            k_start = k_block_idx * BLOCK_K
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


if HAS_TLE_W8A8_BLOCK_FP8_BMM:

    @triton.jit(do_not_specialize=["T", "T_aligned", "x_sB", "xs_sB"])
    def _dsv41_o_proj_single_launch_kernel(
        x_desc,
        y_desc,
        o_ptr,
        positions_ptr,
        cache_ptr,
        x_ptr,
        xs_ptr,
        counter_ptr,
        z_ptr,
        ys_ptr,
        T,
        T_aligned,
        x_sB,
        xs_sB,
        o_sT: tl.constexpr,
        o_sH: tl.constexpr,
        cache_sP: tl.constexpr,
        x_sM: tl.constexpr,
        xs_sM: tl.constexpr,
        z_sB: tl.constexpr,
        z_sM: tl.constexpr,
        z_sN: tl.constexpr,
        B: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HEADS_PER_GROUP: tl.constexpr,
        CHUNKS_PER_HEAD: tl.constexpr,
        ROPE_START: tl.constexpr,
        HALF_ROPE: tl.constexpr,
        SWAP_AB: tl.constexpr,
        fp8_max: tl.constexpr,
        eps: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
        num_sms: tl.constexpr,
    ):
        _ = num_warps
        x_smem = tle.gpu.alloc(
            [num_stages, 1, BLOCK_M, BLOCK_K],
            dtype=x_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        y_smem = tle.gpu.alloc(
            [num_stages, BLOCK_N, BLOCK_K],
            dtype=y_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        empty_x = tle.gpu.alloc_barriers(
            num_barriers=num_stages, arrive_count=1, init=tle.gpu.READY
        )
        empty_y = tle.gpu.alloc_barriers(
            num_barriers=num_stages, arrive_count=1, init=tle.gpu.READY
        )
        full_x = tle.gpu.alloc_barriers(
            num_barriers=num_stages, arrive_count=1, expect_bytes=BLOCK_M * BLOCK_K
        )
        full_y = tle.gpu.alloc_barriers(
            num_barriers=num_stages, arrive_count=1, expect_bytes=BLOCK_N * BLOCK_K
        )
        quant_done = tle.gpu.alloc_barrier(arrive_count=1)

        tle.gpu.warp_specialize(
            [
                (
                    _single_launch_compute_partition,
                    (
                        x_smem,
                        y_smem,
                        empty_x,
                        empty_y,
                        full_x,
                        full_y,
                        quant_done,
                        o_ptr,
                        positions_ptr,
                        cache_ptr,
                        x_ptr,
                        xs_ptr,
                        counter_ptr,
                        z_ptr,
                        ys_ptr,
                        T,
                        T_aligned,
                        x_sB,
                        xs_sB,
                        o_sT,
                        o_sH,
                        cache_sP,
                        x_sM,
                        xs_sM,
                        z_sB,
                        z_sM,
                        z_sN,
                        B,
                        N,
                        K,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        HEADS_PER_GROUP,
                        CHUNKS_PER_HEAD,
                        ROPE_START,
                        HALF_ROPE,
                        SWAP_AB,
                        fp8_max,
                        eps,
                        num_stages,
                        num_sms,
                    ),
                ),
                (
                    _single_launch_load_partition,
                    (
                        x_desc,
                        y_desc,
                        x_smem,
                        y_smem,
                        empty_x,
                        empty_y,
                        full_x,
                        full_y,
                        quant_done,
                        T_aligned,
                        B,
                        N,
                        K,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        num_stages,
                        num_sms,
                    ),
                ),
            ],
            [1],
            [24],
        )


@lru_cache(maxsize=None)
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@lru_cache(maxsize=None)
def _descriptor_allocator(device: torch.device):
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    return alloc_fn


@lru_cache(maxsize=None)
def _ticket_counter(stream: torch.cuda.Stream, device: torch.device) -> torch.Tensor:
    # One per stream: launches on a stream never overlap, so tickets cannot mix.
    return torch.zeros(1, dtype=torch.int64, device=device)


@lru_cache(maxsize=None)
def _is_sm90(device: torch.device) -> bool:
    # The kernel uses TMA and WGMMA, so it is Hopper-only; other GPUs take other paths.
    return (
        device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device)[0] == 9
    )


def single_launch_supported(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    y: torch.Tensor,
    ys: torch.Tensor,
    heads_per_group: int,
    rope_dim: int,
    quant_block: int,
) -> bool:
    """The two-kernel path handles, or rejects, everything this returns False for."""
    head_dim = o.shape[2]
    device = o.device
    return (
        HAS_TLE_W8A8_BLOCK_FP8_BMM
        and _is_sm90(device)
        and y.device == device
        and ys.device == device
        and positions.device == device
        and cos_sin_cache.device == device
        # Tile shapes must be powers of two.
        and heads_per_group & (heads_per_group - 1) == 0
        and head_dim % quant_block == 0
        and head_dim & (head_dim - 1) == 0
        and rope_dim % 2 == 0
        and 0 < rope_dim < quant_block
        and o.stride(-1) == 1
        and positions.shape == (o.shape[0],)
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[-1] == rope_dim
        and cos_sin_cache.dtype == torch.float32
        and y.is_contiguous()
        and ys.is_contiguous()
        and y.data_ptr() % 16 == 0
    )


def dsv41_o_proj_single_launch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    y: torch.Tensor,
    ys: torch.Tensor,
    z: torch.Tensor,
    heads_per_group: int,
    rope_dim: int,
    quant_block: int,
) -> None:
    """Write ``z[G, T, N]`` from ``o[T, H, D]``, ``y[G, N, K]`` and ``ys``.

    Arguments must be validated and pass ``single_launch_supported``.
    """
    device = o.device
    B, T, N = z.shape
    K = y.shape[2]
    head_dim = o.shape[2]
    num_kb = K // quant_block
    # Same tile geometry as w8a8_block_fp8_bmm, so results stay identical to it.
    BLOCK_M = max(8, min(64, 1 << ((T - 1).bit_length())))
    SWAP_AB = 1 if BLOCK_M < 64 else 0
    T_aligned = triton.cdiv(T, BLOCK_M) * BLOCK_M

    x = torch.empty((B, T, K), dtype=torch.float8_e4m3fn, device=device)
    xs = torch.empty((B, T, num_kb), dtype=torch.float32, device=device)
    triton.set_allocator(_descriptor_allocator(device))
    x_desc = TensorDescriptor.from_tensor(x, block_shape=[1, BLOCK_M, quant_block])
    y_desc = TensorDescriptor.from_tensor(
        y.view(B * N, K), block_shape=[quant_block, quant_block]
    )
    o_sT, o_sH, _ = o.stride()
    cache_sP = cos_sin_cache.stride(0)
    z_sB, z_sM, z_sN = z.stride()
    num_sms = _num_sms(device)
    counter = _ticket_counter(torch.cuda.current_stream(device), device)

    # Triton would rebuild its key from every argument on each launch. This key holds
    # everything that selects the compiled kernel, so repeat launches go straight to it.
    # The other dtypes are fixed by the validation this function requires.
    key = (
        device,
        positions.dtype,
        z.dtype,
        o.data_ptr() % 16 == 0,
        positions.data_ptr() % 16 == 0,
        cos_sin_cache.data_ptr() % 16 == 0,
        z.data_ptr() % 16 == 0,
        ys.data_ptr() % 16 == 0,
        T * K < 2**31,  # runtime ints compile as i32 or i64
        o_sT,
        o_sH,
        cache_sP,
        z_sB,
        z_sM,
        z_sN,
        B,
        N,
        K,
        BLOCK_M,
        heads_per_group,
        head_dim,
        rope_dim,
        quant_block,
    )
    kernel = _COMPILED.get(key)
    if kernel is not None:
        kernel[(num_sms, 1, 1)](
            x_desc,
            y_desc,
            o,
            positions,
            cos_sin_cache,
            x,
            xs,
            counter,
            z,
            ys,
            T,
            T_aligned,
            T * K,
            T * num_kb,
            o_sT,
            o_sH,
            cache_sP,
            K,
            num_kb,
            z_sB,
            z_sM,
            z_sN,
            B,
            N,
            K,
            BLOCK_M,
            quant_block,
            quant_block,
            heads_per_group,
            head_dim // quant_block,
            (head_dim - rope_dim) % quant_block,
            rope_dim // 2,
            SWAP_AB,
            _FP8_MAX,
            _EPS,
            _NUM_WARPS,
            _NUM_STAGES,
            num_sms,
        )
        return

    _COMPILED[key] = _dsv41_o_proj_single_launch_kernel[(num_sms,)](
        x_desc,
        y_desc,
        o,
        positions,
        cos_sin_cache,
        x,
        xs,
        counter,
        z,
        ys,
        T,
        T_aligned,
        T * K,
        T * num_kb,
        o_sT=o_sT,
        o_sH=o_sH,
        cache_sP=cache_sP,
        x_sM=K,
        xs_sM=num_kb,
        z_sB=z_sB,
        z_sM=z_sM,
        z_sN=z_sN,
        B=B,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=quant_block,
        BLOCK_K=quant_block,
        HEADS_PER_GROUP=heads_per_group,
        CHUNKS_PER_HEAD=head_dim // quant_block,
        ROPE_START=(head_dim - rope_dim) % quant_block,
        HALF_ROPE=rope_dim // 2,
        SWAP_AB=SWAP_AB,
        fp8_max=_FP8_MAX,
        eps=_EPS,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        num_sms=num_sms,
        # The ticket barrier needs every CTA resident at once. One CTA fills an SM's
        # registers, so a concurrent launch could otherwise hold the SMs it waits for.
        launch_cooperative_grid=True,
    )
