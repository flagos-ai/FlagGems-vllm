# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""In-place min-p filtering aligned with vLLM's GPU sampler."""

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
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    USE_TLE: tl.constexpr,
    WARPS: tl.constexpr,
    ELEMENTS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    if USE_TLE:
        # Keep a row dimension: TLE's reduction layout propagation requires
        # a parent encoding of rank >= 2 when constructing a slice encoding.
        offsets = tle.gpu.set_layout(
            offsets[None, :],
            tle.gpu.BlockEncoding([1, ELEMENTS], [1, 32], [1, WARPS], [1, 0]),
        )
    row_ptr = logits_ptr + token_idx * logits_stride
    if SINGLE_TILE:
        # Keep the loaded row live for filtering below.
        logits = tl.load(row_ptr + offsets, offsets < vocab_size, other=-float("inf"))
        max_val = tl.max(
            logits.to(tl.float32), 1 if USE_TLE else 0, keep_dims=USE_TLE
        )
    else:
        # Accumulate per-lane maxima and perform one cross-lane reduction.
        maxima = tl.full(offsets.shape, -float("inf"), tl.float32)
        for i in range(0, vocab_size, BLOCK_SIZE):
            block = i + offsets
            logits = tl.load(row_ptr + block, block < vocab_size, other=-float("inf"))
            maxima = tl.maximum(maxima, logits.to(tl.float32))
        max_val = tl.max(maxima, 1 if USE_TLE else 0, keep_dims=USE_TLE)

    threshold = max_val + tl.log(min_p)
    if SINGLE_TILE:
        filtered = tl.where(logits < threshold, -float("inf"), logits)
        tl.store(row_ptr + offsets, filtered, offsets < vocab_size)
    else:
        for i in range(0, vocab_size, BLOCK_SIZE):
            block = i + offsets
            logits = tl.load(row_ptr + block, block < vocab_size, other=-float("inf"))
            filtered = tl.where(logits < threshold, -float("inf"), logits)
            tl.store(row_ptr + block, filtered, block < vocab_size)


def _prune_min_p_configs(configs, named_args, **kwargs):
    # ELEMENTS only affects TLE layouts; avoid equivalent plain-Triton trials.
    if kwargs["USE_TLE"]:
        return configs
    return [config for config in configs if config.kwargs["ELEMENTS"] == 1]


def _make_entry(config_name):
    entry = _min_p_kernel
    if runtime.device.vendor_name == "nvidia":
        entry = libtuner(
            configs=runtime.get_tuned_config(config_name),
            key=["vocab_size", "USE_TLE"],
            prune_configs_by={"early_config_prune": _prune_min_p_configs},
            restore_value=["logits_ptr"],
            warmup=5,
            rep=30,
        )(entry)
    return libentry()(entry)


_min_p_single_tile_entry = _make_entry("min_p_single_tile")
_min_p_streaming_entry = _make_entry("min_p_streaming")


def apply_min_p(
    logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor
) -> None:
    """Apply vLLM min-p filtering in place; logits columns must be contiguous."""
    if logits.device.type == "cpu":
        raise NotImplementedError("apply_min_p requires the active accelerator")
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    if logits.stride(1) != 1:
        raise ValueError("apply_min_p requires unit column stride for logits")

    single_tile = vocab_size <= 8192
    entry = _min_p_single_tile_entry if single_tile else _min_p_streaming_entry
    launch_args = {}
    if single_tile:
        launch_args["BLOCK_SIZE"] = triton.next_power_of_2(vocab_size)
    if runtime.device.vendor_name != "nvidia":
        launch_args.setdefault("BLOCK_SIZE", 1024)
        launch_args.update(WARPS=4, ELEMENTS=1, num_warps=4, num_stages=1)
    with runtime.torch_device_fn.device(logits.device):
        entry[(num_tokens,)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            min_p,
            vocab_size,
            SINGLE_TILE=single_tile,
            USE_TLE=_use_tle(logits.device),
            **launch_args,
        )
