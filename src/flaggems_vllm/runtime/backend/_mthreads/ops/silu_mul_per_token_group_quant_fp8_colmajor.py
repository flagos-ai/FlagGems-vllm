# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""mthreads SiLU/mul FP8 quantization with default/expand CUDA Graph tuning.

Default schedules are searched for each input key; explicit FlagTune enables
the larger expand space. The arithmetic keeps vLLM 0.23's input-precision activation/multiply boundaries.
"""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.silu_mul_per_token_group_quant_fp8_colmajor import (
    _launch_platform_fp8,
    _needs_general_quantization,
    _prune_silu_tiles,
    _silu_mul_quant_tile,
    _validate_and_allocate,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config(
        "silu_mul_per_token_group_quant_fp8_tile_mthreads_default"
    ),
    flagtune_op_name="silu_mul_per_token_group_quant_fp8_colmajor",
    flagtune_expand_op_name="silu_mul_per_token_group_quant_fp8_tile_mthreads",
    key=["M", "N", "DTYPE", "UE", "HAS_CLAMP", "DEVICE_NAME", "INDEX64"],
    prune_configs_by={"early_config_prune": _prune_silu_tiles},
    use_cuda_graph=True,
    rep=30,
)
@triton.jit(do_not_specialize=["CLAMP"])
def _silu_mul_quant_autotuned(
    X,
    Q,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP,
    HAS_CLAMP: tl.constexpr,
    UE: tl.constexpr,
    DTYPE: tl.constexpr,
    DEVICE_NAME: tl.constexpr,
    INDEX64: tl.constexpr,
    BR: tl.constexpr,
    BG: tl.constexpr,
):
    _silu_mul_quant_tile(
        X, Q, S, M, N, EPS, CLAMP, HAS_CLAMP, UE, BR, BG, "mthreads", INDEX64
    )


def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
    clamp_limit: float | None = None,
):
    """Return quantized activations and column-major FP32 group-128 scales."""
    if runtime.device.vendor_name != "mthreads" or input.device.type != "musa":
        raise NotImplementedError("this implementation requires a MUSA device")
    if torch_device_fn.get_device_capability(input.device) < (3, 1):
        raise NotImplementedError("native E4M3FN conversion requires MP 3.1 or newer")
    output, scales, use_ue8m0 = _validate_and_allocate(
        input, output, use_ue8m0, eps, clamp_limit
    )
    m, n = input.shape
    if m == 0 or n == 0:
        return output, scales

    guard = (
        nullcontext()
        if torch_device_fn.current_device() == input.device.index
        else torch_device_fn.device(input.device)
    )
    with guard:
        if _needs_general_quantization(output, use_ue8m0, eps, clamp_limit):
            _launch_platform_fp8(input, output, scales, use_ue8m0, eps, clamp_limit)
            return output, scales
        _silu_mul_quant_autotuned[
            lambda meta: (m // meta["BR"], n // (256 * meta["BG"]))
        ](
            input,
            output,
            scales,
            m,
            n,
            eps,
            clamp_limit if clamp_limit is not None else 0.0,
            clamp_limit is not None,
            use_ue8m0,
            str(input.dtype),
            torch_device_fn.get_device_name(input.device),
            input.numel() >= 2**31,
            enable_fp_fusion=False,
        )
    return output, scales
