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


def _is_qwen_moe_shape(E, N, K, topk, dtype, gemm_stage):
    if dtype not in _PLAIN_HALF_CONFIG_DTYPES:
        return False
    if E == 512 and topk == 10:
        return (N, K) == ((2048, 4096) if gemm_stage == "gemm1" else (4096, 1024))
    if E == 256 and topk == 8:
        if gemm_stage == "gemm1":
            return (N, K) in ((1024, 2048), (256, 2048))
        return (N, K) in ((2048, 512), (2048, 128))
    return False


def _metax_get_default_config(
    M,
    E,
    N,
    K,
    topk,
    dtype,
    block_shape=None,
    gemm_stage="gemm1",
    enable_gemm_fast_path=False,
):
    if not _is_qwen_moe_shape(E, N, K, topk, dtype, gemm_stage):
        return _GENERIC_GET_DEFAULT_CONFIG(
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

    if M <= 1024:
        block_m, block_n, block_k = 16, 128, 64
        group_m, num_warps, num_stages = 1, 4, 2
    elif M <= 2048:
        block_m, block_n, block_k = 64, 128, 64
        group_m, num_warps, num_stages = 1, 8, 2
    elif M <= 4096:
        block_m, block_n, block_k = 64, 256, 32
        group_m, num_warps, num_stages = 1, 8, 2
    else:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = (1, 8) if M <= 16384 else (8, 8)
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
    if gemm_stage == "gemm1" and (
        M >= 8192
        or (M >= 4096 and (E, N, K, topk) == (256, 256, 2048, 8))
    ):
        config["PAIR_GATE_UP_DOT"] = True

    shape = (E, N, K, topk)
    if gemm_stage == "gemm1" and M <= 8 and shape == (512, 2048, 4096, 10):
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
        if (N, K) in ((256, 2048), (2048, 128)):
            config["BLOCK_SIZE_K"] = 16
        else:
            config["num_warps"] = 4
            config["num_stages"] = 2

    if 4096 <= M < 8192:
        if shape == (256, 1024, 2048, 8) and gemm_stage == "gemm1":
            config.update(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, num_stages=2
            )
        elif shape == (256, 2048, 512, 8) and gemm_stage == "gemm2":
            config.update(
                BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, num_stages=2
            )
    return config


def _is_qwen_plain_half_call(args, kwargs):
    try:
        hidden_states = args[0] if args else kwargs["hidden_states"]
        w1 = args[1] if len(args) > 1 else kwargs["w1"]
        w2 = args[2] if len(args) > 2 else kwargs["w2"]
        topk_ids = args[4] if len(args) > 4 else kwargs["topk_ids"]
    except (KeyError, IndexError):
        return False
    if str(hidden_states.dtype) not in ("torch.float16", "torch.bfloat16"):
        return False
    if topk_ids.ndim != 2:
        return False
    return (
        (tuple(w1.shape), tuple(w2.shape), topk_ids.size(1))
        in (
            ((512, 2048, 4096), (512, 4096, 1024), 10),
            ((256, 1024, 2048), (256, 2048, 512), 8),
            ((256, 256, 2048), (256, 2048, 128), 8),
        )
    )


def _router_weight_is_already_applied(args, kwargs, positional_index):
    if "apply_router_weight_on_input" in kwargs:
        return bool(kwargs["apply_router_weight_on_input"])
    if len(args) > positional_index:
        return bool(args[positional_index])
    return False


def _should_defer_router_weight_to_sum(args, kwargs, positional_index):
    if not _is_qwen_plain_half_call(args, kwargs):
        return False
    if _router_weight_is_already_applied(args, kwargs, positional_index):
        return False

    hidden_states = args[0] if args else kwargs["hidden_states"]
    w1 = args[1] if len(args) > 1 else kwargs["w1"]
    num_tokens = hidden_states.size(0)
    w1_shape = tuple(w1.shape)
    if w1_shape == (512, 2048, 4096):
        return 16 <= num_tokens <= 4096
    if w1_shape == (256, 1024, 2048):
        return 8 <= num_tokens <= 8192
    return num_tokens >= 8


@contextlib.contextmanager
def _metax_moe_config_patch(use_metax, defer_router_weight):
    if not use_metax:
        yield
        return
    with _PATCH_LOCK:
        original_get_default_config = generic_fused_moe.get_default_config
        original_direct_sum_min_tokens = generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS
        original_moe_sum = generic_fused_moe.moe_sum
        original_dispatch = generic_fused_moe.dispatch_fused_moe_kernel
        router_weights_for_sum = None

        def metax_dispatch(*dispatch_args, **dispatch_kwargs):
            nonlocal router_weights_for_sum
            # Generic GEMM2 uses top_k=1 and out_top_k=the model top-k.  For
            # the selected MC550 ranges, leave the router weights for moe_sum.
            positional = list(dispatch_args)
            top_k = positional[11] if len(positional) > 11 else dispatch_kwargs.get("top_k")
            topk_weights = (
                positional[6]
                if len(positional) > 6
                else dispatch_kwargs.get("topk_weights")
            )
            out_top_k = dispatch_kwargs.get("out_top_k", 1)
            if defer_router_weight and top_k == 1 and out_top_k in (8, 10):
                router_weights_for_sum = topk_weights
                if len(positional) > 10:
                    positional[10] = False
                else:
                    dispatch_kwargs["mul_routed_weight"] = False
                dispatch_args = tuple(positional)
            return original_dispatch(*dispatch_args, **dispatch_kwargs)

        def metax_sum(input, output):
            return metax_moe_sum(input, output, router_weights_for_sum)

        generic_fused_moe.get_default_config = _metax_get_default_config
        generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = _DIRECT_SUM_DISABLED_MIN_TOKENS
        generic_fused_moe.dispatch_fused_moe_kernel = metax_dispatch
        generic_fused_moe.moe_sum = metax_sum
        try:
            yield
        finally:
            generic_fused_moe.get_default_config = original_get_default_config
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = original_direct_sum_min_tokens
            generic_fused_moe.moe_sum = original_moe_sum
            generic_fused_moe.dispatch_fused_moe_kernel = original_dispatch


def fused_experts_impl(*args, **kwargs):
    is_qwen = _is_qwen_plain_half_call(args, kwargs)
    with _metax_moe_config_patch(
        is_qwen,
        _should_defer_router_weight_to_sum(args, kwargs, 7),
    ):
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs):
    is_qwen = _is_qwen_plain_half_call(args, kwargs)
    with _metax_moe_config_patch(
        is_qwen,
        _should_defer_router_weight_to_sum(args, kwargs, 6),
    ):
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    is_qwen = _is_qwen_plain_half_call(args, kwargs)
    with _metax_moe_config_patch(
        is_qwen,
        _should_defer_router_weight_to_sum(args, kwargs, 6),
    ):
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)
