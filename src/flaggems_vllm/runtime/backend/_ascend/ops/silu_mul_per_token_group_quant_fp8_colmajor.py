# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend fused SiLU/mul and group-128 E4M3FN quantization."""

import math

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.silu_mul_per_token_group_quant_fp8_colmajor import (
    _validate_and_allocate,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.ops.per_token_group_quant_fp8 import (
    _f32_to_fp8_e4m3fn,
)
from flaggems_vllm.utils import libentry, libtuner


def _prune_configs(configs, named_args, **kwargs):
    m, n = named_args["M"], named_args["N"]
    return [
        c
        for c in configs
        if m % c.kwargs["BR"] == 0
        and (n // 256) % c.kwargs["BG"] == 0
        and (m // c.kwargs["BR"]) * (n // (256 * c.kwargs["BG"])) <= 65535
        and (named_args["DTYPE"] != "torch.float32" or c.kwargs["BR"] <= 16)
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("silu_mul_quant_fp8_ascend"),
    key=["M", "N", "DTYPE", "UE", "HAS_CLAMP", "INDEX64"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit(do_not_specialize=["CLAMP"])
def _silu_mul_quant_kernel(
    X,
    Q,
    S,
    M,
    N: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP,
    UE: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    DTYPE: tl.constexpr,
    INDEX64: tl.constexpr,
    BR: tl.constexpr,
    BG: tl.constexpr,
):
    group_tiles = N // (256 * BG)
    task = tl.program_id(0)
    ids = tl.arange(0, BR * BG)
    rows = (task // group_tiles) * BR + ids // BG
    groups = (task % group_tiles) * BG + ids % BG
    if INDEX64:
        rows = rows.to(tl.int64)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    offsets = rows[:, None] * N + cols
    gate = tl.load(X + offsets)
    up = tl.load(X + offsets + N // 2)
    if HAS_CLAMP:
        gate = tl.minimum(gate.to(tl.float32), CLAMP).to(X.dtype.element_ty)
        up = tl.clamp(up.to(tl.float32), -CLAMP, CLAMP).to(X.dtype.element_ty)
    up = tl.where(up == 0.0, 0.0, up).to(X.dtype.element_ty)
    gate = gate.to(tl.float32)
    glu = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty)
    values = (glu * up).to(X.dtype.element_ty).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), EPS)
    scale = amax * (1.0 / 448.0)
    if UE:
        # Exact power-of-two rounding avoids Ascend log2/exp2 approximation.
        bits = (scale.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000
        scale = tl.where(scale < float("inf"), bits.to(tl.float32, bitcast=True), scale)
    normalized = values / scale[:, None]
    quant = tl.clamp(normalized, -448.0, 448.0)
    # Flatten the conversion to avoid inflated 2D cast buffers on Ascend.
    flat = tl.reshape(quant, (BR * BG * 128,))
    isnan = flat != flat
    encoded = _f32_to_fp8_e4m3fn(tl.where(isnan, 0.0, flat))
    sign = ((flat.to(tl.int32, bitcast=True) >> 24) & 0x80).to(tl.uint8)
    encoded = tl.where(isnan, sign | 0x7F, encoded).to(tl.uint8)
    tl.store(Q + rows[:, None] * (N // 2) + cols, tl.reshape(encoded, (BR * BG, 128)))
    tl.store(S + groups * M + rows, scale)


def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
    clamp_limit: float | None = None,
):
    # The integer UE shortcut requires a finite positive normal scale.
    if not math.isfinite(eps) or not 448.0 * 2.0**-126 <= eps <= 1.0:
        raise NotImplementedError("eps must be finite and in [448 * 2**-126, 1]")
    if clamp_limit is not None and (not math.isfinite(clamp_limit) or clamp_limit < 0):
        raise NotImplementedError("clamp_limit must be finite and nonnegative")
    output, scales, ue = _validate_and_allocate(
        input, output, use_ue8m0, eps, clamp_limit
    )
    if output.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Ascend output must be float8_e4m3fn")
    if not input.numel():
        return output, scales
    m, n = input.shape

    def grid(meta):
        tasks = (m // meta["BR"]) * (n // (256 * meta["BG"]))
        return (tasks,)

    # Tiny workloads need enough programs; avoid noisy event-based tuning.
    small = (
        m * (n // 256) <= 256 and input.dtype != torch.float32 and clamp_limit is None
    )
    kernel = _silu_mul_quant_kernel.jit_function if small else _silu_mul_quant_kernel
    config = {"BR": 8, "BG": 1, "num_warps": 4, "num_stages": 1} if small else {}
    with torch_device_fn.device(input.device):
        kernel[grid](
            input,
            output.view(torch.uint8),
            scales,
            m,
            n,
            eps,
            clamp_limit if clamp_limit is not None else 0.0,
            ue,
            clamp_limit is not None,
            str(input.dtype),
            input.numel() >= 2**31,
            # Larger tiles need the UB space otherwise used for multi-buffering.
            multibuffer=small,
            enable_fp_fusion=False,
            **config,
        )
    return output, scales
