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

"""BF16 Paged MQA Logits Triton Kernel.

Ported from FlagGems `src/flag_gems/fused/bf16_paged_mqa_logits.py`.

Computes multi-head weighted ReLU attention logits on paged KV cache:
  logits[b*next_n+n, pos] = sum_h( relu(q[b,n,h,:] @ K[pos,:]) * w[b*next_n+n, h] )

Uses shape-specialized kernels (_kernel_H32_D128, _kernel_H64_D128) with
hardcoded shape constants as source-level literals for zero-constexpr dispatch.

GCU (enflame) tuning:
- GBLOCK dispatch: one program processes GB consecutive KV blocks of one row,
  hoisting the per-row prologue (Q transpose, weights, context length) and
  rotating a one-block-deep K prefetch so the next block's DMA overlaps the
  current dot. GB is picked host-side from shape metadata as the largest
  value in {1,2,4,8} that still yields >= 32 programs, capped at 8.
- Block-table hoisting (2026-11-12 round 2): all GB block-table entries are
  loaded as unmasked scalar loads issued back-to-back at program start
  (index clamped to the table row width, page id clamped to [0, num_phys-1]
  for ragged-tail safety). This removes the per-iteration bt->K DMA
  serialization that dominated the per-block serial chain (-14% alone).
- num_warps=2 on the GBLOCK kernels: doubles the per-program K-tile DMA
  issue rate (the post-hoist wall is per-program DMA rate, not latency);
  combined with hoisting this is -25~-34% vs the round-1 kernel. The
  single-block (gb=1) kernels keep num_warps=1 (single dot, no loop).
"""

import logging
import sys

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.bf16_paged_mqa_logits import (
    bf16_paged_mqa_logits as _general_bf16_paged_mqa_logits,
)
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


# H=32, D=128, block_size=64 — shape constants as source-level literals
# for zero-constexpr function params and trivial dispatch cache
@triton.jit
def _kernel_H32_D128(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_ctx,
    stride_bt,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    # block_size=64 hardcoded for H=32 specialization
    kv_pos = pid_blk * 64
    if kv_pos >= ctx_len:
        return

    # H=32, D=128 hardcoded for this specialization
    h_range = tl.arange(0, 32)
    d_range = tl.arange(0, 128)
    # block_size=64 hardcoded
    pos_range = tl.arange(0, 64)

    # Load Q [32, 128] bf16 → transpose to [128, 32]
    q_base = pid_row * (32 * 128)
    q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
    q_block = tl.load(q_ptr + q_offs, eviction_policy="evict_last")
    q_t = tl.trans(q_block)

    # Load weights [32] fp32
    w = tl.load(weights_ptr + pid_row * 32 + h_range, eviction_policy="evict_last")

    # Load K [64, 128] bf16 from paged cache
    b_idx = pid_row // next_n
    phys_blk = tl.load(block_table_ptr + b_idx * stride_bt + pid_blk)
    # block_size=64, D=128 hardcoded
    k_offs = phys_blk * (64 * 128) + pos_range[:, None] * 128 + d_range[None, :]
    k_block = tl.load(kv_cache_ptr + k_offs, eviction_policy="evict_first")

    # GEMM: K[64,128] @ Q^T[128,32] → scores[64,32] fp32
    scores = tl.dot(k_block, q_t)

    # ReLU + weighted sum → logits[64] fp32
    # Workaround: synthesized zero tensor avoids a rank-0 constant src1
    scores = tl.maximum(scores, scores * 0.0)
    logits_val = tl.sum(scores * w[None, :], axis=1)

    # Store (mask only the last partial block)
    out_base = pid_row * max_ctx + kv_pos
    # block_size=64 hardcoded
    if kv_pos + 64 <= ctx_len:
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            eviction_policy="evict_first",
        )
    else:
        mask = pos_range < (ctx_len - kv_pos)
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            mask=mask,
            eviction_policy="evict_first",
        )


# H=64, D=128, block_size=64 — shape constants as source-level literals
@triton.jit
def _kernel_H64_D128(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_ctx,
    stride_bt,
):
    pid_row = tl.program_id(0)
    pid_blk = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    # block_size=64 hardcoded for H=64 specialization
    kv_pos = pid_blk * 64
    if kv_pos >= ctx_len:
        return

    # H=64, D=128 hardcoded for this specialization
    h_range = tl.arange(0, 64)
    d_range = tl.arange(0, 128)
    # block_size=64 hardcoded
    pos_range = tl.arange(0, 64)

    q_base = pid_row * (64 * 128)
    q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
    q_block = tl.load(q_ptr + q_offs, eviction_policy="evict_last")
    q_t = tl.trans(q_block)

    w = tl.load(weights_ptr + pid_row * 64 + h_range, eviction_policy="evict_last")

    b_idx = pid_row // next_n
    phys_blk = tl.load(block_table_ptr + b_idx * stride_bt + pid_blk)
    # block_size=64, D=128 hardcoded
    k_offs = phys_blk * (64 * 128) + pos_range[:, None] * 128 + d_range[None, :]
    k_block = tl.load(kv_cache_ptr + k_offs, eviction_policy="evict_first")

    scores = tl.dot(k_block, q_t)
    # Workaround: synthesized zero tensor avoids a rank-0 constant src1
    scores = tl.maximum(scores, scores * 0.0)
    logits_val = tl.sum(scores * w[None, :], axis=1)

    out_base = pid_row * max_ctx + kv_pos
    # block_size=64 hardcoded
    if kv_pos + 64 <= ctx_len:
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            eviction_policy="evict_first",
        )
    else:
        mask = pos_range < (ctx_len - kv_pos)
        tl.store(
            logits_ptr + out_base + pos_range,
            logits_val,
            mask=mask,
            eviction_policy="evict_first",
        )


# GBLOCK variant of _kernel_H32_D128: one program handles GB consecutive KV
# blocks of one row. GCU tuning (2026-11-12 round 2, RSI attribution):
# - The kernel is DMA-latency bound: per-block serial chain was
#   block-table scalar load -> K tile DMA (~2.7us) -> dot. Hoisting ALL GB
#   block-table entries as unmasked scalar loads issued back-to-back removes
#   the per-iteration bt->K serialization (-14% on gb>=2 shapes).
# - Address safety for ragged rows: bt entry index clamped to the table row
#   width (stride_bt-1) and the page id value clamped to [0, num_phys-1], so
#   every K load stays in bounds even when the group tail reaches beyond the
#   row's valid blocks (dead blocks: loaded but dot/store guarded by ctx).
# - Deeper K prefetch windows (PF 2-4) measured neutral-to-worse than the
#   1-deep rotation here (per-program DMA rate is the wall, not latency).
@triton.jit
def _kernel_gb_H32_D128(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_ctx,
    stride_bt,
    num_phys,
    GB: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_grp = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    kv_pos0 = pid_grp * GB * 64
    if kv_pos0 < ctx_len:
        # H=32, D=128, block_size=64 hardcoded for this specialization
        h_range = tl.arange(0, 32)
        d_range = tl.arange(0, 128)
        pos_range = tl.arange(0, 64)

        # Load Q [32, 128] bf16 -> transpose to [128, 32] once per program
        q_base = pid_row * (32 * 128)
        q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
        q_block = tl.load(q_ptr + q_offs, eviction_policy="evict_last")
        q_t = tl.trans(q_block)

        # Load weights [32] fp32 once per program
        w = tl.load(weights_ptr + pid_row * 32 + h_range, eviction_policy="evict_last")

        b_idx = pid_row // next_n
        out_base = pid_row * max_ctx

        # ---- hoisted unmasked scalar block-table loads (index+value clamped) ----
        bt_base = block_table_ptr + b_idx * stride_bt + pid_grp * GB
        mb1 = stride_bt - 1
        pb_v0 = tl.maximum(tl.minimum(tl.load(bt_base), num_phys - 1), 0)
        if GB >= 2:
            pb_v1 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(1, mb1)), num_phys - 1), 0
            )
        if GB >= 4:
            pb_v2 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(2, mb1)), num_phys - 1), 0
            )
            pb_v3 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(3, mb1)), num_phys - 1), 0
            )
        if GB >= 8:
            pb_v4 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(4, mb1)), num_phys - 1), 0
            )
            pb_v5 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(5, mb1)), num_phys - 1), 0
            )
            pb_v6 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(6, mb1)), num_phys - 1), 0
            )
            pb_v7 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(7, mb1)), num_phys - 1), 0
            )

        # one-block-deep K prefetch (addresses known upfront -> DMA issues
        # immediately, overlapping the current dot)
        k_offs = pos_range[:, None] * 128 + d_range[None, :]
        k_next = tl.load(
            kv_cache_ptr + pb_v0 * (64 * 128) + k_offs,
            eviction_policy="evict_first",
        )

        for i in tl.static_range(GB):
            pid_blk = pid_grp * GB + i
            kv_pos = pid_blk * 64
            k_cur = k_next
            if i + 1 < GB:
                if i + 1 == 1:
                    pb_n = pb_v1
                elif i + 1 == 2:
                    pb_n = pb_v2
                elif i + 1 == 3:
                    pb_n = pb_v3
                elif i + 1 == 4:
                    pb_n = pb_v4
                elif i + 1 == 5:
                    pb_n = pb_v5
                elif i + 1 == 6:
                    pb_n = pb_v6
                else:
                    pb_n = pb_v7
                k_next = tl.load(
                    kv_cache_ptr + pb_n * (64 * 128) + k_offs,
                    eviction_policy="evict_first",
                )
            if kv_pos < ctx_len:
                # GEMM: K[64,128] @ Q^T[128,32] -> scores[64,32] fp32
                scores = tl.dot(k_cur, q_t)

                # ReLU + weighted sum -> logits[64] fp32
                # Workaround: synthesized zero tensor avoids a rank-0 constant src1
                scores = tl.maximum(scores, scores * 0.0)
                logits_val = tl.sum(scores * w[None, :], axis=1)

                # Store (mask only the last partial block)
                if kv_pos + 64 <= ctx_len:
                    tl.store(
                        logits_ptr + out_base + kv_pos + pos_range,
                        logits_val,
                        eviction_policy="evict_first",
                    )
                else:
                    mask = pos_range < (ctx_len - kv_pos)
                    tl.store(
                        logits_ptr + out_base + kv_pos + pos_range,
                        logits_val,
                        mask=mask,
                        eviction_policy="evict_first",
                    )


# GBLOCK variant of _kernel_H64_D128 (same structure as the H32 version):
# hoisted clamped scalar block-table loads + one-block-deep K prefetch,
# num_warps=2 at launch (doubles the per-program DMA issue rate; measured
# fastest for both specializations after the bt-hoisting restructure).
@triton.jit
def _kernel_gb_H64_D128(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    next_n,
    max_ctx,
    stride_bt,
    num_phys,
    GB: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_grp = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + pid_row)
    kv_pos0 = pid_grp * GB * 64
    if kv_pos0 < ctx_len:
        # H=64, D=128, block_size=64 hardcoded for this specialization
        h_range = tl.arange(0, 64)
        d_range = tl.arange(0, 128)
        pos_range = tl.arange(0, 64)

        # Load Q [64, 128] bf16 -> transpose to [128, 64] once per program
        q_base = pid_row * (64 * 128)
        q_offs = q_base + h_range[:, None] * 128 + d_range[None, :]
        q_block = tl.load(q_ptr + q_offs, eviction_policy="evict_last")
        q_t = tl.trans(q_block)

        # Load weights [64] fp32 once per program
        w = tl.load(weights_ptr + pid_row * 64 + h_range, eviction_policy="evict_last")

        b_idx = pid_row // next_n
        out_base = pid_row * max_ctx

        # ---- hoisted unmasked scalar block-table loads (index+value clamped) ----
        bt_base = block_table_ptr + b_idx * stride_bt + pid_grp * GB
        mb1 = stride_bt - 1
        pb_v0 = tl.maximum(tl.minimum(tl.load(bt_base), num_phys - 1), 0)
        if GB >= 2:
            pb_v1 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(1, mb1)), num_phys - 1), 0
            )
        if GB >= 4:
            pb_v2 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(2, mb1)), num_phys - 1), 0
            )
            pb_v3 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(3, mb1)), num_phys - 1), 0
            )
        if GB >= 8:
            pb_v4 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(4, mb1)), num_phys - 1), 0
            )
            pb_v5 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(5, mb1)), num_phys - 1), 0
            )
            pb_v6 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(6, mb1)), num_phys - 1), 0
            )
            pb_v7 = tl.maximum(
                tl.minimum(tl.load(bt_base + tl.minimum(7, mb1)), num_phys - 1), 0
            )

        # one-block-deep K prefetch (addresses known upfront -> DMA issues
        # immediately, overlapping the current dot)
        k_offs = pos_range[:, None] * 128 + d_range[None, :]
        k_next = tl.load(
            kv_cache_ptr + pb_v0 * (64 * 128) + k_offs,
            eviction_policy="evict_first",
        )

        for i in tl.static_range(GB):
            pid_blk = pid_grp * GB + i
            kv_pos = pid_blk * 64
            k_cur = k_next
            if i + 1 < GB:
                if i + 1 == 1:
                    pb_n = pb_v1
                elif i + 1 == 2:
                    pb_n = pb_v2
                elif i + 1 == 3:
                    pb_n = pb_v3
                elif i + 1 == 4:
                    pb_n = pb_v4
                elif i + 1 == 5:
                    pb_n = pb_v5
                elif i + 1 == 6:
                    pb_n = pb_v6
                else:
                    pb_n = pb_v7
                k_next = tl.load(
                    kv_cache_ptr + pb_n * (64 * 128) + k_offs,
                    eviction_policy="evict_first",
                )
            if kv_pos < ctx_len:
                # GEMM: K[64,128] @ Q^T[128,64] -> scores[64,64] fp32
                scores = tl.dot(k_cur, q_t)

                # ReLU + weighted sum -> logits[64] fp32
                scores = tl.maximum(scores, scores * 0.0)
                logits_val = tl.sum(scores * w[None, :], axis=1)

                if kv_pos + 64 <= ctx_len:
                    tl.store(
                        logits_ptr + out_base + kv_pos + pos_range,
                        logits_val,
                        eviction_policy="evict_first",
                    )
                else:
                    mask = pos_range < (ctx_len - kv_pos)
                    tl.store(
                        logits_ptr + out_base + kv_pos + pos_range,
                        logits_val,
                        mask=mask,
                        eviction_policy="evict_first",
                    )


@libentry()
@triton.jit
def _clean_logits_kernel(
    logits_ptr,
    context_lens_ptr,
    max_ctx,
    BLOCK: tl.constexpr,
):
    """Set logits beyond each row's context length to -inf."""
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    col = pid_col * BLOCK + tl.arange(0, BLOCK)
    ctx_len = tl.load(context_lens_ptr + pid_row)
    mask = (col >= ctx_len) & (col < max_ctx)

    tl.store(
        logits_ptr + pid_row * max_ctx + col,
        float("-inf") + tl.zeros([BLOCK], dtype=tl.float32),
        mask=mask,
    )


def bf16_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_metadata,
    max_context_len,
    clean_logits=False,
    logits_dtype=torch.float32,
):
    """BF16 Paged MQA Logits — Triton implementation.

    Computes weighted ReLU attention logits on paged KV cache:
      logits[row, pos] = sum_h( relu(q[b,n,h,:] · K[pos,:]) * w[row, h] )

    Args:
        q: [B, next_n, H, D] bfloat16 query tensor.
        kv_cache: [num_blocks, block_size, 1, D] bfloat16 paged KV cache.
        weights: [B*next_n, H] float32 per-head weights.
        context_lens: [B, next_n] int32 context lengths.
        block_table: [B, max_blocks] int32 block table mapping.
        schedule_metadata: [num_sms+1, 2] int32 (unused, kept for API
            compatibility with vLLM).
        max_context_len: Maximum context length.
        clean_logits: If True, set positions beyond context length to -inf.
        logits_dtype: Output dtype (default float32).

    Returns:
        Logits tensor [total_tokens, max_context_len] in logits_dtype.
    """
    logger.debug("GEMS BF16_PAGED_MQA_LOGITS")

    B, next_n, H, D = q.shape
    total_tokens = B * next_n

    logits = torch.empty(
        total_tokens,
        max_context_len,
        dtype=logits_dtype,
        device=q.device,
    )

    if total_tokens == 0 or max_context_len == 0:
        return logits

    # block_size=64 hardcoded for both specializations
    num_kv_blocks = (max_context_len + 63) >> 6
    stride_bt = block_table.stride(0)

    # GBLOCK dispatch (shape metadata only): amortize the per-program prologue
    # (Q transpose, weights, context scalar chain) over GB consecutive KV
    # blocks while keeping the grid wide enough to fill the device. Empirical
    # line on GCU400: >= 32 programs, GB capped at 8 (16 already underfills).
    gb = 1
    for candidate in (2, 4, 8):
        groups = (num_kv_blocks + candidate - 1) // candidate
        if total_tokens * groups >= 32:
            gb = candidate

    # num_warps: single-block (gb=1) kernels keep nw=1 (single dot, no loop,
    # measured fastest); the GBLOCK kernels launch with num_warps=2 — after
    # block-table hoisting the wall is the per-program K-tile DMA issue rate,
    # which a second warp doubles (2-4 warps were neutral before hoisting
    # because the per-iteration bt->K serialization shadowed it).
    if gb == 1:
        grid = (total_tokens, num_kv_blocks)
        if H == 32:
            _kernel_H32_D128[grid](
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                logits,
                next_n,
                max_context_len,
                stride_bt,
                num_warps=1,
                num_stages=1,
            )
        else:
            _kernel_H64_D128[grid](
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                logits,
                next_n,
                max_context_len,
                stride_bt,
                num_warps=1,
                num_stages=1,
            )
    else:
        grid = (total_tokens, (num_kv_blocks + gb - 1) // gb)
        num_phys = kv_cache.shape[0]
        if H == 32:
            _kernel_gb_H32_D128[grid](
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                logits,
                next_n,
                max_context_len,
                stride_bt,
                num_phys,
                GB=gb,
                num_warps=2,
                num_stages=1,
            )
        else:
            _kernel_gb_H64_D128[grid](
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                logits,
                next_n,
                max_context_len,
                stride_bt,
                num_phys,
                GB=gb,
                num_warps=2,
                num_stages=1,
            )

    if clean_logits:
        CLEAN_BLOCK = 256
        clean_grid = (total_tokens, triton.cdiv(max_context_len, CLEAN_BLOCK))
        _clean_logits_kernel[clean_grid](
            logits,
            context_lens,
            max_context_len,
            BLOCK=CLEAN_BLOCK,
            num_warps=4,
        )

    return logits


def _install():
    """Wire the GCU implementation into the direct-import entrypoints.

    Tests and benchmarks import `from flaggems_vllm.ops import
    bf16_paged_mqa_logits`, which resolves against the already-imported generic
    modules, so the SpecOpRegistrar namespace swap alone can not reach them.
    Replace the attributes on those modules in place, mirroring the _kunlunxin
    bf16_paged_mqa_logits / mhc_pre backend overrides.
    """
    for module_name in ("flaggems_vllm.ops.bf16_paged_mqa_logits", "flaggems_vllm.ops"):
        mod = sys.modules.get(module_name)
        if mod is None:
            continue
        cur = getattr(mod, "bf16_paged_mqa_logits", None)
        if cur is _general_bf16_paged_mqa_logits:
            mod.bf16_paged_mqa_logits = bf16_paged_mqa_logits


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
