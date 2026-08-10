# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import contextlib
import threading
from typing import Any

import flaggems_vllm.ops.fused_moe as generic_fused_moe

from .moe_sum import moe_sum as metax_moe_sum

_PATCH_LOCK = threading.RLock()
_GENERIC_GET_DEFAULT_CONFIG = generic_fused_moe.get_default_config
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")
_DIRECT_SUM_DISABLED_MIN_TOKENS = 1 << 60
_ROUTE_WEIGHT_STATE = threading.local()


def _is_qwen_moe_shape(
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    gemm_stage: str,
) -> bool:
    if dtype not in _PLAIN_HALF_CONFIG_DTYPES:
        return False

    if E == 512 and topk == 10:
        if gemm_stage == "gemm1":
            return (N, K) == (2048, 4096)
        if gemm_stage == "gemm2":
            return (N, K) == (4096, 1024)

    if E == 256 and topk == 8:
        if gemm_stage == "gemm1":
            return (N, K) in ((1024, 2048), (256, 2048))
        if gemm_stage == "gemm2":
            return (N, K) in ((2048, 512), (2048, 128))

    return False


_METAX_MAX_SHARED_MEMORY = 65536  # 64KB hardware limit for MetaX C550


def _adjust_config_for_shared_memory(config: dict[str, Any]) -> dict[str, Any]:
    """Adjust kernel config to fit within MetaX 64KB shared memory limit.

    Strategy (preserving performance as much as possible):
    1. First reduce num_stages to 2
    2. Then reduce BLOCK_SIZE_K (less impact on parallelism)
    3. Then reduce BLOCK_SIZE_N (affects output tile)
    4. Finally reduce BLOCK_SIZE_M if still needed
    """
    block_m = config.get("BLOCK_SIZE_M", 128)
    block_n = config.get("BLOCK_SIZE_N", 128)
    block_k = config.get("BLOCK_SIZE_K", 64)
    num_stages = config.get("num_stages", 3)

    def calc_smem():
        return (block_m * block_k + block_k * block_n) * 2 * num_stages

    # Step 1: Reduce num_stages to 2
    if calc_smem() > _METAX_MAX_SHARED_MEMORY:
        num_stages = 2

    # Step 2: Reduce BLOCK_SIZE_K (32 is minimum for good vectorization)
    if calc_smem() > _METAX_MAX_SHARED_MEMORY and block_k > 32:
        block_k = 32

    # Step 3: Reduce BLOCK_SIZE_N
    if calc_smem() > _METAX_MAX_SHARED_MEMORY and block_n > 128:
        block_n = 128

    # Step 4: Reduce BLOCK_SIZE_M if still over
    if calc_smem() > _METAX_MAX_SHARED_MEMORY and block_m > 64:
        block_m = 64

    # Step 5: Further reduce BLOCK_SIZE_N if needed
    if calc_smem() > _METAX_MAX_SHARED_MEMORY and block_n > 64:
        block_n = 64

    config["BLOCK_SIZE_M"] = block_m
    config["BLOCK_SIZE_N"] = block_n
    config["BLOCK_SIZE_K"] = block_k
    config["num_stages"] = num_stages
    return config


def _metax_get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    if not _is_qwen_moe_shape(E, N, K, topk, dtype, gemm_stage):
        config = _GENERIC_GET_DEFAULT_CONFIG(
            M,
            E,
            N,
            K,
            topk,
            dtype,
            block_shape,
            gemm_stage,
            enable_gemm_fast_path,
        )
        return _adjust_config_for_shared_memory(config)

    if M <= 1024:
        block_m, block_n, block_k = 16, 128, 64
        group_m, num_warps, num_stages = 1, 4, 2
    elif M <= 2048:
        block_m, block_n, block_k = 64, 128, 64
        group_m, num_warps, num_stages = 1, 8, 2
    elif M <= 4096:
        block_m, block_n, block_k = 64, 256, 32
        group_m, num_warps, num_stages = 1, 8, 2
    elif M <= 8192:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = 1, 8
        num_stages = 2 if gemm_stage == "gemm2" else 3
    elif M <= 16384:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = 1, 8
        num_stages = 2 if gemm_stage == "gemm2" else 3
    else:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = 8, 8
        num_stages = 2 if gemm_stage == "gemm2" else 3

    if gemm_stage == "gemm1" and (E, N, K, topk) == (256, 256, 2048, 8):
        block_n = min(block_n, 128)

    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    use_pair_gate_up_dot = gemm_stage == "gemm1" and (
        M >= 8192 or (M >= 4096 and (E, N, K, topk) == (256, 256, 2048, 8))
    )
    if use_pair_gate_up_dot:
        config["PAIR_GATE_UP_DOT"] = True
    swap_ab_shape = (E, N, K, topk)
    if gemm_stage == "gemm1" and M <= 8 and swap_ab_shape == (512, 2048, 4096, 10):
        config["SWAP_AB"] = True
    elif gemm_stage == "gemm2" and M <= 1024:
        config["SWAP_AB"] = True

    if E == 512:
        if M in (8192, 16384):
            config["BLOCK_SIZE_K"] = 16
        elif M == 32768 and gemm_stage == "gemm1":
            config["num_stages"] = 2
    elif M == 4096:
        config["num_warps"] = 4
    elif M in (8192, 16384):
        config["num_warps"] = 4
        config["num_stages"] = 2
    elif M == 32768:
        qwen3_6_i128 = (N, K) in ((256, 2048), (2048, 128))
        if qwen3_6_i128:
            config["BLOCK_SIZE_K"] = 16
        else:
            config["num_warps"] = 4
            config["num_stages"] = 2

    if 4096 <= M < 8192:
        if gemm_stage == "gemm1" and (E, N, K, topk) == (256, 1024, 2048, 8):
            config.update(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "num_stages": 2,
                }
            )
        elif gemm_stage == "gemm2" and (E, N, K, topk) == (256, 2048, 512, 8):
            config.update(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "num_stages": 2,
                }
            )
    return config


def _is_qwen_plain_half_call(args, kwargs) -> bool:
    try:
        hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
        w1 = args[1] if len(args) > 1 else kwargs["w1"]
        w2 = args[2] if len(args) > 2 else kwargs["w2"]
        topk_ids = args[4] if len(args) > 4 else kwargs["topk_ids"]
    except (KeyError, IndexError):
        return False

    if str(hidden_states.dtype) not in ("torch.float16", "torch.bfloat16"):
        return False
    if topk_ids.ndim != 2:
        return False

    shapes = (tuple(w1.shape), tuple(w2.shape), topk_ids.size(1))
    return shapes in (
        ((512, 2048, 4096), (512, 4096, 1024), 10),
        ((256, 1024, 2048), (256, 2048, 512), 8),
        ((256, 256, 2048), (256, 2048, 128), 8),
    )


def _use_metax_moe_sum(args, kwargs) -> bool:
    try:
        hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
        w1 = args[1] if len(args) > 1 else kwargs["w1"]
        w2 = args[2] if len(args) > 2 else kwargs["w2"]
        topk_ids = args[4] if len(args) > 4 else kwargs["topk_ids"]
    except (KeyError, IndexError):
        return False

    if str(hidden_states.dtype) not in ("torch.float16", "torch.bfloat16"):
        return False
    if topk_ids.ndim != 2:
        return False

    shapes = (tuple(w1.shape), tuple(w2.shape), topk_ids.size(1))
    return shapes in (
        ((512, 2048, 4096), (512, 4096, 1024), 10),
        ((256, 1024, 2048), (256, 2048, 512), 8),
        ((256, 256, 2048), (256, 2048, 128), 8),
    )


def _router_weight_is_already_applied(args, kwargs, positional_index: int) -> bool:
    if "apply_router_weight_on_input" in kwargs:
        return bool(kwargs["apply_router_weight_on_input"])
    if len(args) > positional_index:
        return bool(args[positional_index])
    return False


def _should_defer_router_weight_to_sum(args, kwargs, positional_index: int) -> bool:
    """Select the backend-only fused GEMM2+weighted-reduction path."""
    if not _is_qwen_plain_half_call(args, kwargs):
        return False
    if _router_weight_is_already_applied(args, kwargs, positional_index):
        return False

    hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
    w1 = args[1] if len(args) > 1 else kwargs["w1"]
    num_tokens = hidden_states.size(0)
    w1_shape = tuple(w1.shape)
    if w1_shape == (512, 2048, 4096):
        return 16 <= num_tokens <= 4096
    if w1_shape == (256, 1024, 2048):
        return 8 <= num_tokens <= 8192
    return num_tokens >= 8


def _is_qwen_gemm2_dispatch(args, kwargs) -> bool:
    """Identify GEMM2 without relying on a change to generic fused_moe.py."""
    weights = args[1] if len(args) > 1 else kwargs.get("B")
    if weights is None:
        return False
    return tuple(weights.shape) in (
        (512, 4096, 1024),
        (256, 2048, 512),
        (256, 2048, 128),
    )


@contextlib.contextmanager
def _metax_moe_config_patch(
    disable_direct_sum: bool,
    use_metax_moe_sum: bool,
    defer_router_weight_to_sum: bool,
):
    with _PATCH_LOCK:
        original_get_default_config = generic_fused_moe.get_default_config
        original_direct_sum_min_tokens = generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS
        original_moe_sum = generic_fused_moe.moe_sum
        original_dispatch = generic_fused_moe.dispatch_fused_moe_kernel
        generic_fused_moe.get_default_config = _metax_get_default_config
        if disable_direct_sum:
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = (
                _DIRECT_SUM_DISABLED_MIN_TOKENS
            )
        if use_metax_moe_sum:
            generic_fused_moe.moe_sum = metax_moe_sum
        if defer_router_weight_to_sum:
            _ROUTE_WEIGHT_STATE.router_weights = None

            def dispatch_with_deferred_weight(*dispatch_args, **dispatch_kwargs):
                if not _is_qwen_gemm2_dispatch(dispatch_args, dispatch_kwargs):
                    return original_dispatch(*dispatch_args, **dispatch_kwargs)
                router_weights = (
                    dispatch_args[6]
                    if len(dispatch_args) > 6
                    else dispatch_kwargs.get("topk_weights")
                )
                _ROUTE_WEIGHT_STATE.router_weights = router_weights
                if dispatch_args:
                    mutable_args = list(dispatch_args)
                    # dispatch_fused_moe_kernel's mul_routed_weight position.
                    mutable_args[10] = False
                    return original_dispatch(*mutable_args, **dispatch_kwargs)
                dispatch_kwargs["mul_routed_weight"] = False
                return original_dispatch(**dispatch_kwargs)

            def weighted_metax_moe_sum(input_tensor, output_tensor):
                router_weights = getattr(_ROUTE_WEIGHT_STATE, "router_weights", None)
                if router_weights is None:
                    return original_moe_sum(input_tensor, output_tensor)
                return metax_moe_sum(input_tensor, output_tensor, router_weights)

            generic_fused_moe.dispatch_fused_moe_kernel = dispatch_with_deferred_weight
            generic_fused_moe.moe_sum = weighted_metax_moe_sum
        try:
            yield
        finally:
            generic_fused_moe.get_default_config = original_get_default_config
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = original_direct_sum_min_tokens
            generic_fused_moe.moe_sum = original_moe_sum
            generic_fused_moe.dispatch_fused_moe_kernel = original_dispatch
            if defer_router_weight_to_sum:
                _ROUTE_WEIGHT_STATE.router_weights = None


def fused_experts_impl(*args, **kwargs):
    is_qwen_half = _is_qwen_plain_half_call(args, kwargs)
    with _metax_moe_config_patch(
        is_qwen_half,
        _use_metax_moe_sum(args, kwargs),
        _should_defer_router_weight_to_sum(args, kwargs, 7),
    ):
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs):
    is_qwen_half = _is_qwen_plain_half_call(args, kwargs)
    with _metax_moe_config_patch(
        is_qwen_half,
        _use_metax_moe_sum(args, kwargs),
        _should_defer_router_weight_to_sum(args, kwargs, 6),
    ):
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    is_qwen_half = _is_qwen_plain_half_call(args, kwargs)
    with _metax_moe_config_patch(
        is_qwen_half,
        _use_metax_moe_sum(args, kwargs),
        _should_defer_router_weight_to_sum(args, kwargs, 6),
    ):
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)
