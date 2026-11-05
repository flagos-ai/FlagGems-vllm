# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Length-aware, sorted indexer TopK with the serving invalid-index epilogue.

Algorithm provenance: FlagGems TopK2's stable fp32 radix selection, ordered
compaction and packed-key sort. Device prefix bounds select between a complete
short-prefix sort, prefix-bounded radix, and the original fixed-width radix
scanner for long rows inside its descriptor domain. Long full-width scans use
the producer strict -inf tail contract. Runtime lengths never reach
Python. The caller owns the guarantee that scores outside each prefix are
-inf, as provided by flag_gems.fused.int8_sparse_indexer._paged_logits.

The public result matches topk(...).indices followed by isfinite(values), -1
and int32 conversion. It does not promise the raw indices of nonfinite
positions. The scanner preserves the selected stable-key and prefix-bound semantics.
"""

import logging

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)


if HAS_TLE:
    import triton.experimental.tle.language as tle_gpu

    # Keep the actual TopK2 key exactly, including signed-zero normalization
    # and its bit-key behavior for NaNs (which were outside v2's qualified
    # domain). Do not silently replace it with a new NaN ordering policy.
    from flag_gems.fused.indexer_topk_key import topk_stable_fp32_key

    @libentry()
    @triton.jit
    def topk_kernel_radix_stable_k2048_tle_prefix(
        X,
        Lengths,
        Out,
        XM: tl.constexpr,
        XN: tl.constexpr,
        LS: tl.constexpr,
        OM: tl.constexpr,
        ON: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
        V2_FULL_WIDTH: tl.constexpr,
        LONG_PREFIX_CUTOVER: tl.constexpr,
    ):
        tl.static_assert(K > 0)
        tl.static_assert(K <= N)
        tl.static_assert(N <= 204800)
        INDEX_BITS: tl.constexpr = 16 if N <= 65535 else 32
        INDEX_MASK: tl.constexpr = 65535 if N <= 65535 else 4294967295
        tl.static_assert(K <= K_PAD)
        tl.static_assert(K_PAD <= 2048)
        tl.static_assert(RADIX_BITS == 4 or RADIX_BITS == 8)
        row = tl.program_id(0)
        length = tl.load(Lengths + row.to(tl.int64) * LS).to(tl.int64)
        end = tl.minimum(tl.maximum(length, 0), N).to(tl.int32)
        out_cols = tl.arange(0, K_PAD)
        selected_buf = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=tl.uint64,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        if end <= K_PAD:
            short_valid = out_cols < end
            short_value = tl.load(
                X + row.to(tl.int64) * XM + out_cols.to(tl.int64) * XN,
                short_valid,
                other=float("-inf"),
            )
            short_key = topk_stable_fp32_key(short_value)
            short_packed = short_key.to(tl.uint64) << INDEX_BITS | (N - out_cols).to(
                tl.uint64
            )
            short_packed = tl.where(
                short_valid, short_packed, tl.full([K_PAD], 0, tl.uint64)
            )
            packed_sorted = tl.sort(short_packed, dim=0, descending=True)
        elif V2_FULL_WIDTH & (end >= LONG_PREFIX_CUTOVER):
            v2_RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
            v2_RADIX_MASK: tl.constexpr = v2_RADIX_SIZE - 1
            v2_cols = tl.arange(0, BLOCK_N)
            v2_bins = tl.arange(0, v2_RADIX_SIZE)
            v2_desired = tl.full((), 0, tl.uint32)
            v2_desired_mask = tl.full((), 0, tl.uint32)
            v2_k_to_find = tl.full((), K, tl.int32)
            for v2_digit_pos in tl.static_range(32 - RADIX_BITS, -1, -RADIX_BITS):
                v2_counts = tl.zeros([v2_RADIX_SIZE], tl.int32)
                for v2_start in tl.range(0, N, BLOCK_N):
                    v2_offsets = v2_start + v2_cols
                    v2_valid = v2_offsets < N
                    v2_x = tl.load(
                        X + row * XM + v2_offsets * XN,
                        mask=v2_valid,
                        other=float("-inf"),
                    )
                    v2_key = topk_stable_fp32_key(v2_x)
                    v2_matches = v2_key & v2_desired_mask == v2_desired
                    v2_digit = (v2_key >> v2_digit_pos & v2_RADIX_MASK).to(tl.int32)
                    v2_counts += tl.histogram(
                        v2_digit, v2_RADIX_SIZE, v2_valid & v2_matches
                    )
                v2_cumulative = tl.cumsum(v2_counts, axis=0, reverse=True)
                v2_greater = v2_cumulative - v2_counts
                v2_selected = (v2_cumulative >= v2_k_to_find) & (
                    v2_greater < v2_k_to_find
                )
                v2_selected_digit = tl.sum(tl.where(v2_selected, v2_bins, 0), axis=0)
                v2_count_gt = tl.sum(tl.where(v2_selected, v2_greater, 0), axis=0)
                v2_desired |= v2_selected_digit.to(tl.uint32) << v2_digit_pos
                v2_desired_mask |= tl.full((), v2_RADIX_MASK, tl.uint32) << v2_digit_pos
                v2_k_to_find -= v2_count_gt
            v2_equal_seen = tl.full((), 0, tl.int32)
            v2_output_seen = tl.full((), 0, tl.int32)
            for v2_start in tl.range(0, N, BLOCK_N):
                v2_offsets = v2_start + v2_cols
                v2_valid = v2_offsets < N
                v2_x = tl.load(
                    X + row * XM + v2_offsets * XN, mask=v2_valid, other=float("-inf")
                )
                v2_key = topk_stable_fp32_key(v2_x)
                v2_equal = v2_valid & (v2_key == v2_desired)
                v2_equal_rank = v2_equal_seen + tl.cumsum(v2_equal.to(tl.int32), axis=0)
                v2_take = v2_valid & (
                    (v2_key > v2_desired) | v2_equal & (v2_equal_rank <= v2_k_to_find)
                )
                v2_positions = (
                    v2_output_seen + tl.cumsum(v2_take.to(tl.int32), axis=0) - 1
                )
                v2_safe_positions = tl.where(v2_take, v2_positions, 0)
                v2_pointers = tle_gpu.gpu.local_ptr(selected_buf, (v2_safe_positions,))
                v2_packed = v2_key.to(tl.uint64) << INDEX_BITS | (N - v2_offsets).to(
                    tl.uint64
                )
                tl.store(v2_pointers, v2_packed, mask=v2_take)
                v2_equal_seen += tl.sum(v2_equal.to(tl.int32), axis=0)
                v2_output_seen += tl.sum(v2_take.to(tl.int32), axis=0)
            tl.debug_barrier()
            v2_selected_ptrs = tle_gpu.gpu.local_ptr(selected_buf, (out_cols,))
            packed_sorted = tl.sort(tl.load(v2_selected_ptrs), dim=0, descending=True)
        else:
            RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
            RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
            cols = tl.arange(0, BLOCK_N)
            bins = tl.arange(0, RADIX_SIZE)
            desired = tl.full((), 0, tl.uint32)
            desired_mask = tl.full((), 0, tl.uint32)
            k_to_find = tl.full((), K, tl.int32)
            for digit_pos in tl.static_range(32 - RADIX_BITS, -1, -RADIX_BITS):
                counts = tl.zeros([RADIX_SIZE], tl.int32)
                for start in tl.range(0, end, BLOCK_N):
                    offsets = start + cols
                    valid = offsets < end
                    value = tl.load(
                        X + row.to(tl.int64) * XM + offsets.to(tl.int64) * XN,
                        valid,
                        other=float("-inf"),
                    )
                    key = topk_stable_fp32_key(value)
                    matches = key & desired_mask == desired
                    digit = (key >> digit_pos & RADIX_MASK).to(tl.int32)
                    counts += tl.histogram(digit, RADIX_SIZE, valid & matches)
                cumulative = tl.cumsum(counts, axis=0, reverse=True)
                greater = cumulative - counts
                selected = (cumulative >= k_to_find) & (greater < k_to_find)
                selected_digit = tl.sum(tl.where(selected, bins, 0), axis=0)
                count_gt = tl.sum(tl.where(selected, greater, 0), axis=0)
                desired |= selected_digit.to(tl.uint32) << digit_pos
                desired_mask |= tl.full((), RADIX_MASK, tl.uint32) << digit_pos
                k_to_find -= count_gt
            if K != K_PAD:
                selected_ptrs = tle_gpu.gpu.local_ptr(selected_buf, (out_cols,))
                tl.store(selected_ptrs, tl.full([K_PAD], 0, tl.uint64))
                tl.debug_barrier()
            equal_seen = tl.full((), 0, tl.int32)
            output_seen = tl.full((), 0, tl.int32)
            for start in tl.range(0, end, BLOCK_N):
                offsets = start + cols
                valid = offsets < end
                value = tl.load(
                    X + row.to(tl.int64) * XM + offsets.to(tl.int64) * XN,
                    valid,
                    other=float("-inf"),
                )
                key = topk_stable_fp32_key(value)
                equal = valid & (key == desired)
                equal_rank = equal_seen + tl.cumsum(equal.to(tl.int32), axis=0)
                take = valid & ((key > desired) | equal & (equal_rank <= k_to_find))
                positions = output_seen + tl.cumsum(take.to(tl.int32), axis=0) - 1
                safe_positions = tl.where(take, positions, 0)
                pointers = tle_gpu.gpu.local_ptr(selected_buf, (safe_positions,))
                packed = key.to(tl.uint64) << INDEX_BITS | (N - offsets).to(tl.uint64)
                tl.store(pointers, packed, take)
                equal_seen += tl.sum(equal.to(tl.int32), axis=0)
                output_seen += tl.sum(take.to(tl.int32), axis=0)
            tl.debug_barrier()
            selected_ptrs = tle_gpu.gpu.local_ptr(selected_buf, (out_cols,))
            packed_sorted = tl.sort(tl.load(selected_ptrs), dim=0, descending=True)
        real = packed_sorted != 0
        index = N - (packed_sorted & INDEX_MASK).to(tl.int32)
        safe_index = tl.where(real, index, 0)
        value = tl.load(
            X + row.to(tl.int64) * XM + safe_index.to(tl.int64) * XN,
            real,
            other=float("-inf"),
        )
        finite = (value > float("-inf")) & (value < float("inf"))
        result = tl.where(real & finite, index, -1).to(tl.int32)
        tl.store(
            Out + row.to(tl.int64) * OM + out_cols.to(tl.int64) * ON,
            result,
            out_cols < K,
        )


def _validate(logits, lengths, k, out):
    if logits.ndim != 2:
        raise ValueError("indexer logits must be 2D")
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise NotImplementedError("indexer prefix TopK supports CUDA float32 logits")
    if type(k) is not int or not 0 <= k <= logits.shape[1]:
        raise ValueError("k must be an integer in [0, logits width]")
    if (
        lengths.ndim != 1
        or lengths.shape[0] != logits.shape[0]
        or lengths.device != logits.device
    ):
        raise ValueError("lengths must be one same-device value per row")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError("indexer prefix TopK supports int32/int64 lengths")
    if out is None:
        out = torch.empty((logits.shape[0], k), device=logits.device, dtype=torch.int32)
    if (
        out.shape != (logits.shape[0], k)
        or out.device != logits.device
        or out.dtype != torch.int32
    ):
        raise ValueError("output must be same-device int32 with shape [rows, k]")
    if any(torch._C._overlaps(out, value) for value in (logits, lengths)):
        raise ValueError("TopK output may not overlap scores or lengths")
    return out


def _supported(logits, lengths, k, out):
    # Host descriptors only. No data-dependent launch choice or host length read.
    return (
        HAS_TLE
        and 0 < k <= 2048
        and 0 < logits.shape[1] <= 204800
        and logits.stride(0) >= 0
        and logits.stride(1) > 0
        and lengths.stride(0) >= 0
        and out.stride(0) >= 0
        and out.stride(1) > 0
        and (logits.shape[0] <= 1 or out.stride(0) >= (k - 1) * out.stride(1) + 1)
    )


def supports_indexer_prefix_topk(logits, lengths, k, out):
    """Complete host-descriptor/TLE predicate for a caller's pre-launch route.

    No tensor contents, length values, NaN scan, allocation, launch or recovery.
    The caller guarantees strict -inf tails and disjoint output storage.
    """
    logger.debug("GEMS_NVIDIA SUPPORTS_INDEXER_PREFIX_TOPK")
    return (
        logits.ndim == 2
        and logits.is_cuda
        and logits.dtype == torch.float32
        and type(k) is int
        and 0 <= k <= logits.shape[1]
        and lengths.ndim == 1
        and lengths.shape[0] == logits.shape[0]
        and lengths.device == logits.device
        and lengths.dtype in (torch.int32, torch.int64)
        and out.shape == (logits.shape[0], k)
        and out.device == logits.device
        and out.dtype == torch.int32
        and (logits.shape[0] == 0 or k == 0 or _supported(logits, lengths, k, out))
    )


def _v2_full_width_supported(logits, k):
    """Original v2 address domain, using host descriptors only.

    This selects compiler capability. Each GPU row still selects the long
    scanner from its own device length. General r4 support is unchanged.
    """
    m, n = logits.shape
    return (
        k == 2048
        and 1 <= m <= 1024
        and 4096 <= n <= 34880
        and logits.stride(0) >= 0
        and logits.stride(1) > 0
        and (
            (m - 1) * logits.stride(0)
            + (triton.cdiv(n, 1024) * 1024 - 1) * logits.stride(1)
            <= 0x7FFFFFFF
        )
    )


def _launch(logits, lengths, k, out):
    with torch_device_fn.device(logits.device):
        topk_kernel_radix_stable_k2048_tle_prefix[logits.shape[0],](
            logits,
            lengths,
            out,
            logits.stride(0),
            logits.stride(1),
            lengths.stride(0),
            out.stride(0),
            out.stride(1),
            N=logits.shape[1],
            K=k,
            K_PAD=triton.next_power_of_2(k),
            BLOCK_N=1024,
            RADIX_BITS=4,
            V2_FULL_WIDTH=_v2_full_width_supported(logits, k),
            LONG_PREFIX_CUTOVER=24576,
            num_warps=8,
            num_stages=1,
        )
    return out


def indexer_prefix_topk(logits, lengths, k, out=None):
    """Sorted TopK for all rows, with GPU prefix lengths and int32 -1 output.

    Non-NaN values sort descending with original-column ascending ties. Signed
    zeros compare equal. The current TopK2 bit key is reused for NaN behavior;
    NaNs and both infinities then become -1 under the existing serving contract.
    Lengths clamp to [0, width]. Inactive/padded rows must carry length zero.
    Input and output must not overlap; output rows must not alias one another.
    """
    logger.debug("GEMS_NVIDIA INDEXER_PREFIX_TOPK")
    out = _validate(logits, lengths, k, out)
    if logits.shape[0] == 0 or k == 0:
        return out
    if not _supported(logits, lengths, k, out):
        raise NotImplementedError(
            "unsupported indexer prefix TopK descriptor or TLE capability"
        )
    return _launch(logits, lengths, k, out)


__all__ = ["indexer_prefix_topk", "supports_indexer_prefix_topk"]
