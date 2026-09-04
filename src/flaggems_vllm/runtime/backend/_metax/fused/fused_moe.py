# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import contextlib
import threading
from typing import Any

import triton
import triton.language as tl

import flaggems_vllm.ops.fused_moe as generic_fused_moe
from flaggems_vllm.runtime.backend._metax.fused.moe_sum import moe_sum as metax_moe_sum

_PATCH_LOCK = threading.RLock()
_GENERIC_GET_DEFAULT_CONFIG = generic_fused_moe.get_default_config
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")
_DIRECT_SUM_DISABLED_MIN_TOKENS = 1 << 60
_write_zeros_to_output = generic_fused_moe.write_zeros_to_output


@triton.jit
def _metax_pair_fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,
    stride_bbn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_K: tl.constexpr,
    N_DIVISIBLE_BY_BLOCK_N: tl.constexpr,
    PAIR_GATE_UP_DOT: tl.constexpr,
    DIRECT_SUM: tl.constexpr,
    OUT_TOP_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
):
    """MC550 plain-half GEMM1 kernel for paired gate/up projection."""
    pid = tl.program_id(axis=0)
    N_out = N // 2
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    if not naive_block_assignment:
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N_out,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_pair = tl.arange(0, BLOCK_SIZE_N * 2).to(tl.int64)
    offs_pair_bn = tl.where(
        offs_pair < BLOCK_SIZE_N,
        pid_n * BLOCK_SIZE_N + offs_pair,
        N_out + pid_n * BLOCK_SIZE_N + offs_pair - BLOCK_SIZE_N,
    )
    a_ptrs = (
        a_ptr + (offs_token[:, None] // top_k * stride_am) + offs_k[None, :] * stride_ak
    )
    b_pair_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_pair_bn[None, :] * stride_bn
    )
    pair_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N * 2), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if K_DIVISIBLE_BY_BLOCK_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            if N_DIVISIBLE_BY_BLOCK_N:
                b_pair = tl.load(b_pair_ptrs)
            else:
                b_pair = tl.load(
                    b_pair_ptrs,
                    mask=offs_pair_bn[None, :] < N,
                    other=0.0,
                )
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b_pair = tl.load(
                b_pair_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_pair_bn[None, :] < N),
                other=0.0,
            )
        pair_acc += tl.dot(a, b_pair)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_pair_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        pair_bias_ptrs = (
            b_bias_ptr + off_experts * stride_bbe + offs_pair_bn * stride_bbn
        )
        pair_bias = tl.load(pair_bias_ptrs, mask=offs_pair_bn < N, other=0.0)
        pair_acc += pair_bias[None, :]

    # FlagTree Triton requires permutation dimensions as positional arguments.
    gate_up = tl.trans(
        tl.reshape(pair_acc, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N)),
        0,
        2,
        1,
    )
    gate_acc, up_acc = tl.split(gate_up)
    gate_sig = tl.sigmoid(gate_acc)
    accumulator = (
        gate_acc.to(compute_type) * gate_sig.to(compute_type) * up_acc.to(compute_type)
    )

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if DIRECT_SUM:
        offs_c = offs_token // OUT_TOP_K
    else:
        offs_c = offs_token
    c_ptrs = c_ptr + stride_cm * offs_c[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None]
    if not N_DIVISIBLE_BY_BLOCK_N:
        c_mask = c_mask & (offs_cn[None, :] < N_out)
    if DIRECT_SUM:
        tl.atomic_add(c_ptrs, accumulator, sem="relaxed", mask=c_mask)
    else:
        tl.store(c_ptrs, accumulator, mask=c_mask)


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
        return (N, K) == ((2048, 4096) if gemm_stage == "gemm1" else (4096, 1024))
    if E == 256 and topk == 8:
        if gemm_stage == "gemm1":
            return (N, K) in ((1024, 2048), (256, 2048))
        return (N, K) in ((2048, 512), (2048, 128))
    return False


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
    # PAIR ranges retained from the measured MC550 configuration snapshot.
    if gemm_stage == "gemm1" and (
        (E == 512 and topk == 10 and M >= 8192)
        or (E, N, K, topk, M) == (256, 256, 2048, 8, 4096)
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
    if M >= 8192 and E == 256 and topk == 8:
        if (gemm_stage, N, K) in (
            ("gemm1", 1024, 2048),
            ("gemm2", 2048, 512),
            ("gemm1", 256, 2048),
            ("gemm2", 2048, 128),
        ):
            config.update(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                    "SPLIT_K": 1,
                    "num_warps": 8,
                    "num_stages": 2,
                    "PAIR_GATE_UP_DOT": False,
                }
            )

    # MC550 FlagTree pipeline selections retained from real-shape tuning.
    if (
        M >= 8192
        and E == 256
        and topk == 8
        and gemm_stage == "gemm2"
        and (N, K) == (2048, 512)
    ):
        config.update(
            {
                "pipeline": "basic",
                "num_stages": 1,
                "pipeline_load_num": -1,
            }
        )
    if (
        M >= 8192
        and E == 256
        and topk == 8
        and gemm_stage == "gemm1"
        and (N, K) == (256, 2048)
    ):
        config.update(
            {
                "pipeline": "basic",
                "num_stages": 3,
                "pipeline_load_num": -1,
            }
        )
    if (
        M >= 8192
        and E == 256
        and topk == 8
        and gemm_stage == "gemm2"
        and (N, K) == (2048, 128)
    ):
        config.update(
            {
                "pipeline": "basic",
                "num_stages": 1,
                "pipeline_load_num": 1,
            }
        )

    # MC550 autotuned range: four_k_i128/pair_bm64_bn64.
    if 4097 <= M <= 8191 and E == 256 and topk == 8:
        if gemm_stage == "gemm1" and (N, K) == (256, 2048):
            config["BLOCK_SIZE_M"] = 64
            config["BLOCK_SIZE_N"] = 64
            config["BLOCK_SIZE_K"] = 32
            config["GROUP_SIZE_M"] = 1
            config["num_warps"] = 8
            config["num_stages"] = 2
            config["PAIR_GATE_UP_DOT"] = True
        elif gemm_stage == "gemm2" and (N, K) == (2048, 128):
            config["BLOCK_SIZE_M"] = 64
            config["BLOCK_SIZE_N"] = 128
            config["BLOCK_SIZE_K"] = 32
            config["GROUP_SIZE_M"] = 1
            config["num_warps"] = 8
            config["num_stages"] = 2
            config.pop("SWAP_AB", None)

    # MC550 autotuned range: four_k_i512/no_pair_bm64_bk64.
    if 4097 <= M <= 8191 and E == 256 and topk == 8:
        if gemm_stage == "gemm1" and (N, K) == (1024, 2048):
            config["BLOCK_SIZE_M"] = 64
            config["BLOCK_SIZE_N"] = 128
            config["BLOCK_SIZE_K"] = 64
            config["GROUP_SIZE_M"] = 1
            config["num_warps"] = 8
            config["num_stages"] = 2
            config.pop("PAIR_GATE_UP_DOT", None)
        elif gemm_stage == "gemm2" and (N, K) == (2048, 512):
            config["BLOCK_SIZE_M"] = 64
            config["BLOCK_SIZE_N"] = 256
            config["BLOCK_SIZE_K"] = 32
            config["GROUP_SIZE_M"] = 1
            config["num_warps"] = 8
            config["num_stages"] = 2
            config.pop("SWAP_AB", None)

    # MC550 autotuned range: small_i512/bm32_bn128_bk64.
    if 448 <= M <= 1024 and E == 256 and topk == 8:
        if gemm_stage == "gemm1" and (N, K) == (1024, 2048):
            config["BLOCK_SIZE_M"] = 32
            config["BLOCK_SIZE_N"] = 128
            config["BLOCK_SIZE_K"] = 64
            config["GROUP_SIZE_M"] = 1
            config["num_warps"] = 4
            config["num_stages"] = 2
            config.pop("PAIR_GATE_UP_DOT", None)
        elif gemm_stage == "gemm2" and (N, K) == (2048, 512):
            config["BLOCK_SIZE_M"] = 32
            config["BLOCK_SIZE_N"] = 128
            config["BLOCK_SIZE_K"] = 64
            config["GROUP_SIZE_M"] = 1
            config["num_warps"] = 4
            config["num_stages"] = 2
            config["SWAP_AB"] = True

    return config


def _is_qwen_plain_half_call(args, kwargs) -> bool:
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
    return (tuple(w1.shape), tuple(w2.shape), topk_ids.size(1)) in (
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
        return num_tokens >= 8
    return num_tokens >= 8


@contextlib.contextmanager
def _metax_moe_config_patch(use_metax: bool, defer_router_weight: bool):
    if not use_metax:
        yield
        return
    with _PATCH_LOCK:
        original_get_default_config = generic_fused_moe.get_default_config
        original_direct_sum_min_tokens = generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS
        original_moe_sum = generic_fused_moe.moe_sum
        original_dispatch = generic_fused_moe.dispatch_fused_moe_kernel
        original_kernel = generic_fused_moe.fused_moe_kernel
        router_weights_for_sum = None

        def metax_dispatch(*dispatch_args, **dispatch_kwargs):
            nonlocal router_weights_for_sum
            # Generic GEMM2 uses top_k=1 and out_top_k=the model top-k.  For
            # the selected MC550 ranges, leave the router weights for moe_sum.
            positional = list(dispatch_args)
            top_k = (
                positional[11] if len(positional) > 11 else dispatch_kwargs.get("top_k")
            )
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
            config = (
                positional[12]
                if len(positional) > 12
                else dispatch_kwargs.get("config", {})
            )
            use_pair_kernel = bool(config.get("PAIR_GATE_UP_DOT", False))
            if use_pair_kernel:
                generic_fused_moe.fused_moe_kernel = _metax_pair_fused_moe_kernel
            try:
                return original_dispatch(*dispatch_args, **dispatch_kwargs)
            finally:
                if use_pair_kernel:
                    generic_fused_moe.fused_moe_kernel = original_kernel

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
            generic_fused_moe.fused_moe_kernel = original_kernel


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
