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

# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::cp_gather_indexer_k_quant_cache_kernel
#
# GCU (gcu400) adaptation notes:
# - The generic two-axis (token x quant-block) grid re-resolves the batch and
#   block table per program and per quant block; for batch > 16 that binary
#   search issues per-lane gather loads which serialize (~51us/program on
#   gcu400, 8.7ms for the 32-batch core shape).
# - gcu400 memory-path rules measured for this op (see knowledge base):
#   plain `base + tl.arange` flat ops are the only fast shape; per-lane
#   computed / where-clamped offsets, 2D row tiles and i64 vector math all
#   degrade to gather-speed. Within one block the cache row is
#   [block values][block scales], so each block contributes one contiguous
#   value segment and one contiguous scale segment.
# - This kernel therefore walks token tiles with a scalar batch scan (one
#   vector load of cu_seqlen + reductions, no per-lane gathers), extends runs
#   across physically-consecutive blocks (probing block_table with unrolled
#   scalar loads), and copies per block with unmasked full-block fast paths.
# - int64 appears only as a scalar base fold (block base, dst base when the
#   output exceeds 2**31 bytes); all lane arithmetic stays int32.

import sys

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.cp_gather_indexer_k_quant_cache import (
    cp_gather_indexer_k_quant_cache as _general_cp_gather_indexer_k_quant_cache,
)


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    cache_ptr,
    k_out_ptr,
    scale_out_ptr,
    block_table_ptr,
    cu_seqlen_ptr,
    num_tokens,
    cache_stride,
    bt_stride,
    batch_size,
    BS: tl.constexpr,
    HD: tl.constexpr,
    SB: tl.constexpr,
    VP2: tl.constexpr,
    VEXACT: tl.constexpr,
    SP2: tl.constexpr,
    SEXACT: tl.constexpr,
    TB: tl.constexpr,
    MAX_EXT: tl.constexpr,
    SCAN_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    GRID_X: tl.constexpr,
    DST_I64: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X
    t_begin = pid * TB
    t_end = tl.minimum(t_begin + TB, num_tokens)
    t = t_begin
    while t < t_end:
        cnt = tl.zeros((), dtype=tl.int32)
        bstart = tl.full((), -1, dtype=tl.int32)
        bend = tl.full((), 2147483647, dtype=tl.int32)
        for c in tl.static_range(0, NUM_CHUNKS):
            soffs = c * SCAN_SIZE + tl.arange(0, SCAN_SIZE)
            starts = tl.load(
                cu_seqlen_ptr + soffs, mask=soffs <= batch_size, other=2147483647
            )
            le = starts <= t
            cnt += tl.sum(le.to(tl.int32))
            bstart = tl.maximum(bstart, tl.max(tl.where(le, starts, -1)))
            bend = tl.minimum(bend, tl.min(tl.where(le, 2147483647, starts)))
        b = cnt - 1
        if (b >= 0) & (b < batch_size):
            b_end_t = tl.minimum(bend, t_end)
            while t < b_end_t:
                boff = t - bstart
                c0 = boff // BS
                inb0 = boff - c0 * BS
                p0 = tl.load(block_table_ptr + b * bt_stride + c0)
                run_cap = b_end_t - t
                nblk_cap = tl.minimum(tl.cdiv(inb0 + run_cap, BS), MAX_EXT)
                ext = tl.zeros((), dtype=tl.int32) + 1
                for k in tl.static_range(1, MAX_EXT):
                    pk = tl.load(
                        block_table_ptr + b * bt_stride + c0 + k,
                        mask=k < nblk_cap,
                        other=-1,
                    )
                    ext = tl.where((k == ext) & (pk == p0 + k), ext + 1, ext)
                run = tl.minimum(ext * BS - inb0, run_cap)

                tok = tl.zeros((), dtype=tl.int32)
                for k in range(0, ext):
                    inb = inb0 if k == 0 else 0
                    seg = tl.minimum(BS - inb, run - tok)
                    if seg > 0:
                        pb = p0 + k
                        voffs = tl.arange(0, VP2)
                        soffs2 = tl.arange(0, SP2)
                        if (seg == BS) & VEXACT:
                            v = tl.load(
                                cache_ptr
                                + pb.to(tl.int64) * (BS * cache_stride)
                                + inb * HD
                                + voffs
                            )
                            if DST_I64:
                                vd = (t + tok).to(tl.int64) * HD
                            else:
                                vd = (t + tok) * HD
                            tl.store(
                                k_out_ptr.to(tl.pointer_type(tl.uint8)) + vd + voffs,
                                v,
                            )
                            if SEXACT:
                                s = tl.load(
                                    cache_ptr
                                    + pb.to(tl.int64) * (BS * cache_stride)
                                    + BS * HD
                                    + inb * SB
                                    + soffs2
                                )
                                if DST_I64:
                                    sd = (t + tok).to(tl.int64) * SB
                                else:
                                    sd = (t + tok) * SB
                                tl.store(scale_out_ptr + sd + soffs2, s)
                            else:
                                sm = soffs2 < BS * SB
                                s = tl.load(
                                    cache_ptr
                                    + pb.to(tl.int64) * (BS * cache_stride)
                                    + BS * HD
                                    + inb * SB
                                    + soffs2,
                                    mask=sm,
                                )
                                if DST_I64:
                                    sd = (t + tok).to(tl.int64) * SB
                                else:
                                    sd = (t + tok) * SB
                                tl.store(scale_out_ptr + sd + soffs2, s, mask=sm)
                        else:
                            vm = voffs < seg * HD
                            v = tl.load(
                                cache_ptr
                                + pb.to(tl.int64) * (BS * cache_stride)
                                + inb * HD
                                + voffs,
                                mask=vm,
                            )
                            if DST_I64:
                                vd = (t + tok).to(tl.int64) * HD
                            else:
                                vd = (t + tok) * HD
                            tl.store(
                                k_out_ptr.to(tl.pointer_type(tl.uint8)) + vd + voffs,
                                v,
                                mask=vm,
                            )
                            sm = soffs2 < seg * SB
                            s = tl.load(
                                cache_ptr
                                + pb.to(tl.int64) * (BS * cache_stride)
                                + BS * HD
                                + inb * SB
                                + soffs2,
                                mask=sm,
                            )
                            if DST_I64:
                                sd = (t + tok).to(tl.int64) * SB
                            else:
                                sd = (t + tok) * SB
                            tl.store(scale_out_ptr + sd + soffs2, s, mask=sm)
                    tok = tok + seg
                t = t + run
        else:
            t = t_end


_GRID_X_CAP = 65535


def cp_gather_indexer_k_quant_cache(
    k_cache: torch.Tensor,
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
):
    num_tokens = k_fp8.size(0)
    if num_tokens == 0:
        return
    block_size = k_cache.size(1)
    head_dim = k_fp8.shape[-1]
    num_scale_bytes = k_fp8_scale.size(1)
    quant_block_size = head_dim * 4 // num_scale_bytes
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size
    scale_bytes = num_quant_blocks * 4

    if (
        k_cache.stride(2) != 1
        or k_cache.stride(1) != k_cache.size(2)
        or k_cache.stride(0) != block_size * k_cache.size(2)
        or k_cache.size(2) < head_dim + scale_bytes
    ):
        raise ValueError(
            "k_cache must be contiguous with per-token row "
            "[head_dim value bytes][scale bytes]"
        )
    if k_fp8.stride(0) != head_dim or k_fp8.stride(1) != 1:
        raise ValueError("k_fp8 must be row-contiguous")
    if k_fp8_scale.stride(0) != scale_bytes or k_fp8_scale.stride(1) != 1:
        raise ValueError("k_fp8_scale must be row-contiguous")
    if cu_seqlen.dtype != torch.int32:
        cu_seqlen = cu_seqlen.to(torch.int32)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    cache_stride = k_cache.size(2)
    bt_stride = block_table.stride(0)
    batch_size = block_table.shape[0]

    vp2 = triton.next_power_of_2(block_size * head_dim)
    vexact = vp2 == block_size * head_dim
    sp2 = triton.next_power_of_2(block_size * scale_bytes)
    sexact = sp2 == block_size * scale_bytes

    if num_tokens <= 2048:
        token_block = 64
    elif head_dim >= 256:
        token_block = 128
    else:
        token_block = 96
    max_ext = min(triton.cdiv(token_block, block_size) + 1, 18)

    scan_size = min(triton.next_power_of_2(batch_size + 1), 256)
    num_chunks = triton.cdiv(batch_size + 1, scan_size)

    num_programs = triton.cdiv(num_tokens, token_block)
    grid_x = min(num_programs, _GRID_X_CAP)
    grid_y = triton.cdiv(num_programs, grid_x)
    dst_i64 = max(num_tokens * head_dim, num_tokens * scale_bytes) >= 2**31

    _cp_gather_indexer_quant_cache_kernel[(grid_x, grid_y)](
        k_cache,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        num_tokens,
        cache_stride,
        bt_stride,
        batch_size,
        block_size,
        head_dim,
        scale_bytes,
        vp2,
        vexact,
        sp2,
        sexact,
        token_block,
        max_ext,
        scan_size,
        num_chunks,
        grid_x,
        dst_i64,
        num_warps=1,
    )


def _install():
    """Wire the GCU implementation into the direct-import entrypoints.

    Tests and benchmarks import `from flaggems_vllm.ops import
    cp_gather_indexer_k_quant_cache`, which resolves against the
    already-imported generic modules, so the SpecOpRegistrar namespace swap
    alone can not reach them. Replace the attributes on those modules in
    place, mirroring the bf16_paged_mqa_logits backend override.
    """
    for module_name in (
        "flaggems_vllm.ops.cp_gather_indexer_k_quant_cache",
        "flaggems_vllm.ops",
    ):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        cur = getattr(mod, "cp_gather_indexer_k_quant_cache", None)
        if cur is _general_cp_gather_indexer_k_quant_cache:
            mod.cp_gather_indexer_k_quant_cache = cp_gather_indexer_k_quant_cache


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
