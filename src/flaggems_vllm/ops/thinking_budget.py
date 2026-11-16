# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Thinking-budget filtering aligned with vLLM Model Runner V2.

Source: vllm/v1/worker/gpu/sample/thinking_budget.py
Commit: 947dd62c0700851a79b0a8f32cb823e25a003b31
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

tle_api = None
tle = None
if runtime.device.vendor_name == "nvidia":
    try:
        import triton.experimental.tle as tle_api
        import triton.experimental.tle.language as tle
    except ModuleNotFoundError as exc:
        if exc.name not in (
            "triton.experimental",
            "triton.experimental.tle",
            "triton.experimental.tle.language",
        ):
            raise
        tle_api = None
        tle = None


@lru_cache(maxsize=None)
def _use_tle(device):
    if runtime.device.vendor_name != "nvidia" or tle_api is None:
        return False
    if runtime.torch_device_fn.get_device_capability(device) != (9, 0):
        return False
    return tle_api.is_primitive_supported("nvidia", "gpu.set_layout")


@triton.jit
def _load_effective_token(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    pos,
):
    if pos < total_len:
        return tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos)
    # In decode/spec-decode, input_ids at local position 0 is the already
    # committed last sampled token. Effective draft-prefix positions start at
    # local position 1.
    input_pos = cur_req_first_pos + pos - total_len + 1
    return tl.load(input_ids_ptr + input_pos)


@triton.jit
def _update_committed_marker_cache(
    req_state_idx,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_TLE: tl.constexpr,
    WARPS: tl.constexpr,
    ELEMENTS: tl.constexpr,
):
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if scan_pos == 0 and last_start < 0 and last_end < 0:
        # Cold scan: walk backward in vectorized blocks, stopping at the first
        # block with a marker. Preserve both marker positions from that block.
        offsets = tl.arange(0, BLOCK)
        if USE_TLE:
            # TLE reduction hints require a parent encoding of rank >= 2.
            offsets = tle.gpu.set_layout(
                offsets[None, :],
                tle.gpu.BlockEncoding([1, ELEMENTS], [1, 32], [1, WARPS], [1, 0]),
            )
        # Zero means no match; bit 0 distinguishes end from start. Encode
        # tile-local offsets so keys do not grow with the history length.
        marker_keys = (offsets + 1) * 2
        block_hi = total_len
        while block_hi > 0 and last_start < 0 and last_end < 0:
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + offsets

            start_match = (offs < block_hi) & (offs + START_LEN <= total_len)
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)

            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)

            # Empty blocks only need one reduction. On a hit, the key already
            # gives one marker's last position; reduce only the other kind.
            last_key = tl.max(
                tl.where(
                    end_match,
                    marker_keys + 1,
                    tl.where(start_match, marker_keys, 0),
                ),
                axis=1 if USE_TLE else 0,
            )
            if USE_TLE:
                last_key = tl.reshape(last_key, ())
            if last_key != 0:
                is_end = (last_key & 1) != 0
                last_pos = block_lo + (last_key >> 1) - 1
                other_match = tl.where(is_end, start_match, end_match)
                other_pos = tl.max(
                    tl.where(other_match, offs, -1), axis=1 if USE_TLE else 0
                )
                if USE_TLE:
                    other_pos = tl.reshape(other_pos, ())
                last_start = tl.where(is_end, other_pos, last_pos)
                last_end = tl.where(is_end, last_pos, other_pos)
            else:
                last_start = -1
                last_end = -1
            block_hi = block_lo
    else:
        for i in tl.range(scan_pos, total_len):
            if i + START_LEN <= total_len:
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

            if i + NATURAL_END_LEN <= total_len:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = tl.load(
                        all_token_ids_ptr + req_state_idx * all_token_ids_stride + i + j
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    new_scan_pos = total_len - (MAX_LEN - 1)
    if new_scan_pos < 0:
        new_scan_pos = 0
    tl.store(cached_scan_pos_ptr + req_state_idx, new_scan_pos)
    return last_start, last_end


@triton.jit
def _update_committed_marker_cache_kernel(
    req_ids_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_TLE: tl.constexpr,
    WARPS: tl.constexpr,
    ELEMENTS: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    _update_committed_marker_cache(
        req_state_idx,
        all_token_ids_ptr,
        all_token_ids_stride,
        total_len,
        cached_last_start_ptr,
        cached_last_end_ptr,
        cached_scan_pos_ptr,
        reasoning_start_token_ids_ptr,
        natural_reasoning_end_token_ids_ptr,
        START_LEN,
        NATURAL_END_LEN,
        MAX_LEN,
        BLOCK,
        USE_TLE,
        WARPS,
        ELEMENTS,
    )


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    cached_scan_pos_ptr=None,
    BLOCK: tl.constexpr = 1024,
    USE_TLE: tl.constexpr = False,
    WARPS: tl.constexpr = 4,
    ELEMENTS: tl.constexpr = 1,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    if cached_scan_pos_ptr is not None:
        # The host only fuses when each request owns exactly one logits row.
        # Reuse the updated positions directly, without a second launch/load.
        last_start, last_end = _update_committed_marker_cache(
            req_state_idx,
            all_token_ids_ptr,
            all_token_ids_stride,
            total_len,
            cached_last_start_ptr,
            cached_last_end_ptr,
            cached_scan_pos_ptr,
            reasoning_start_token_ids_ptr,
            natural_reasoning_end_token_ids_ptr,
            START_LEN,
            NATURAL_END_LEN,
            START_LEN if START_LEN >= NATURAL_END_LEN else NATURAL_END_LEN,
            BLOCK,
            USE_TLE,
            WARPS,
            ELEMENTS,
        )
    else:
        last_start = tl.load(cached_last_start_ptr + req_state_idx)
        last_end = tl.load(cached_last_end_ptr + req_state_idx)

    start_lo = total_len - START_LEN + 1
    if start_lo < 0:
        start_lo = 0
    for i in tl.range(start_lo, effective_len - START_LEN + 1):
        start_match = True
        for j in tl.static_range(0, START_LEN):
            expected = tl.load(reasoning_start_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            start_match = start_match & (actual == expected)
        if start_match:
            last_start = i

    end_lo = total_len - NATURAL_END_LEN + 1
    if end_lo < 0:
        end_lo = 0
    for i in tl.range(end_lo, effective_len - NATURAL_END_LEN + 1):
        end_match = True
        for j in tl.static_range(0, NATURAL_END_LEN):
            expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
            actual = _load_effective_token(
                all_token_ids_ptr,
                all_token_ids_stride,
                input_ids_ptr,
                cur_req_first_pos,
                req_state_idx,
                total_len,
                i + j,
            )
            end_match = end_match & (actual == expected)
        if end_match:
            last_end = i

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    # If the tail already ends with a prefix of the forced end sequence
    # (even from a resumed prompt), continue from the next marker token.
    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len < max_prefix_len:
        max_prefix_len = effective_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start + j,
                    )
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def _prune_cache_configs(configs, named_args, **kwargs):
    if kwargs["USE_TLE"]:
        return configs
    return [config for config in configs if config.kwargs["ELEMENTS"] == 1]


_cache_entry = _update_committed_marker_cache_kernel
_thinking_entry = _thinking_budget_kernel
_fused_entry = _thinking_budget_kernel
if runtime.device.vendor_name == "nvidia":
    _cache_entry = libtuner(
        configs=runtime.get_tuned_config("thinking_budget_cache"),
        key=["START_LEN", "NATURAL_END_LEN", "USE_TLE"],
        prune_configs_by={"early_config_prune": _prune_cache_configs},
        restore_value=[
            "cached_last_start_ptr",
            "cached_last_end_ptr",
            "cached_scan_pos_ptr",
        ],
        warmup=5,
        rep=30,
        use_cuda_graph=False,
    )(_cache_entry)
    _thinking_entry = libtuner(
        configs=runtime.get_tuned_config("thinking_budget"),
        key=["START_LEN", "NATURAL_END_LEN", "END_LEN"],
        restore_value=["logits_ptr"],
        warmup=5,
        rep=30,
        use_cuda_graph=False,
    )(_thinking_entry)
    _fused_entry = libtuner(
        configs=runtime.get_tuned_config("thinking_budget_fused"),
        key=["START_LEN", "NATURAL_END_LEN", "END_LEN", "USE_TLE"],
        prune_configs_by={"early_config_prune": _prune_cache_configs},
        restore_value=[
            "logits_ptr",
            "cached_last_start_ptr",
            "cached_last_end_ptr",
            "cached_scan_pos_ptr",
        ],
        warmup=5,
        rep=30,
        use_cuda_graph=False,
    )(_fused_entry)
_cache_entry = libentry()(_cache_entry)
_thinking_entry = libentry()(_thinking_entry)
_fused_entry = libentry()(_fused_entry)


def apply_thinking_budget(
    logits: torch.Tensor,
    req_ids: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    thinking_token_budget: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cached_last_start: torch.Tensor,
    cached_last_end: torch.Tensor,
    cached_scan_pos: torch.Tensor,
    reasoning_start_token_ids: torch.Tensor,
    natural_reasoning_end_token_ids: torch.Tensor,
    reasoning_end_token_ids: torch.Tensor,
) -> None:
    """Update committed marker caches and force the next reasoning-end token."""
    if logits.device.type == "cpu":
        raise NotImplementedError("apply_thinking_budget requires an accelerator")
    if logits.stride(1) != 1 or all_token_ids.stride(1) != 1:
        raise ValueError("logits and all_token_ids require unit column stride")
    num_tokens = logits.shape[0]
    start_len = reasoning_start_token_ids.shape[0]
    natural_end_len = natural_reasoning_end_token_ids.shape[0]
    end_len = reasoning_end_token_ids.shape[0]

    cache_options = {}
    thinking_options = {}
    if runtime.device.vendor_name != "nvidia":
        cache_options.update(WARPS=4, ELEMENTS=1, num_warps=4, num_stages=1)
        thinking_options.update(num_warps=4, num_stages=1)
    with runtime.torch_device_fn.device(logits.device):
        # vLLM's common no-draft path reuses idx_mapping directly. Identity
        # plus matching row counts preserves the one-CTA-per-request contract.
        if (
            runtime.device.vendor_name == "nvidia"
            and req_ids is expanded_idx_mapping
            and req_ids.shape[0] == num_tokens
        ):
            if num_tokens:
                _fused_entry[(num_tokens,)](
                    logits,
                    logits.stride(0),
                    expanded_idx_mapping,
                    thinking_token_budget,
                    all_token_ids,
                    all_token_ids.stride(0),
                    total_len,
                    input_ids,
                    expanded_local_pos,
                    cached_last_start,
                    cached_last_end,
                    reasoning_start_token_ids,
                    natural_reasoning_end_token_ids,
                    reasoning_end_token_ids,
                    START_LEN=start_len,
                    NATURAL_END_LEN=natural_end_len,
                    END_LEN=end_len,
                    cached_scan_pos_ptr=cached_scan_pos,
                    BLOCK=1024,
                    USE_TLE=_use_tle(logits.device),
                )
            return
        if req_ids.shape[0]:
            _cache_entry[(req_ids.shape[0],)](
                req_ids,
                thinking_token_budget,
                all_token_ids,
                all_token_ids.stride(0),
                total_len,
                cached_last_start,
                cached_last_end,
                cached_scan_pos,
                reasoning_start_token_ids,
                natural_reasoning_end_token_ids,
                START_LEN=start_len,
                NATURAL_END_LEN=natural_end_len,
                MAX_LEN=max(start_len, natural_end_len),
                BLOCK=1024,
                USE_TLE=_use_tle(logits.device),
                **cache_options,
            )
        if num_tokens:
            _thinking_entry[(num_tokens,)](
                logits,
                logits.stride(0),
                expanded_idx_mapping,
                thinking_token_budget,
                all_token_ids,
                all_token_ids.stride(0),
                total_len,
                input_ids,
                expanded_local_pos,
                cached_last_start,
                cached_last_end,
                reasoning_start_token_ids,
                natural_reasoning_end_token_ids,
                reasoning_end_token_ids,
                START_LEN=start_len,
                NATURAL_END_LEN=natural_end_len,
                END_LEN=end_len,
                **thinking_options,
            )
