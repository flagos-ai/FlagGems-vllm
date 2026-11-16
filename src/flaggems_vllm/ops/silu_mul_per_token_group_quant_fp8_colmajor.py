# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused vLLM SiLU/multiply and per-token group-128 FP8 quantization."""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.device_info import kernel_supports_fp8_e4m3


def _platform_fp8_dtype():
    """Use the same platform FP8 dtype as the vLLM baseline."""
    from vllm.platforms import current_platform

    return current_platform.fp8_dtype()


def _launch_platform_fp8(input, output, scales, ue, eps, clamp_limit):
    from vllm.platforms import current_platform

    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = 224.0 if current_platform.is_fp8_fnuz() else torch.finfo(fp8_dtype).max
    m, n = input.shape
    _silu_mul_quant_kernel[lambda meta: (m // meta["BLOCK_M"], n // 256)](
        input,
        output,
        scales,
        m,
        n // 2,
        eps,
        clamp_limit if clamp_limit is not None else 0.0,
        str(input.dtype),
        ue,
        clamp_limit is not None,
        FP8_MAX=fp8_max,
        NORMALIZE_UP_ZERO=runtime.device.vendor_name
        in ("metax", "thead", "hygon", "mthreads"),
    )


def _validate_and_allocate(input, output, use_ue8m0, eps, clamp_limit):
    """Validate metadata only; this helper never computes or copies tensor data."""
    # Keep vLLM's rank/divisibility assertions and their exception types.
    assert input.ndim == 2
    if output is not None:
        assert output.ndim == 2
    m, n = input.shape
    assert m % 128 == 0
    assert n % 256 == 0
    if not input.is_contiguous():
        raise NotImplementedError("input must be contiguous")
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("input must have float16, bfloat16, or float32 dtype")
    if input.device.type == "cpu":
        raise NotImplementedError("FP8 quantization requires a supported accelerator")
    if use_ue8m0 is None:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        use_ue8m0 = is_deep_gemm_e8m0_used()
    fp8_dtype = _platform_fp8_dtype()
    if output is None:
        output = torch.empty((m, n // 2), dtype=fp8_dtype, device=input.device)
    else:
        if output.shape != (m, n // 2):
            raise ValueError("output shape must be [M, N // 2]")
        if output.device != input.device:
            raise ValueError("output must be on the input device")
        if not output.is_contiguous():
            raise NotImplementedError("output must be contiguous")
        if input.numel():
            input_start = input.data_ptr()
            output_start = output.data_ptr()
            input_end = input_start + input.numel() * input.element_size()
            output_end = output_start + output.numel() * output.element_size()
            if input_start < output_end and output_start < input_end:
                raise ValueError("input and output must not overlap")
    scales = torch.empty(
        (n // 256, m), dtype=torch.float32, device=input.device
    ).transpose(0, 1)
    return output, scales, use_ue8m0


def _needs_general_quantization(output, use_ue8m0, eps, clamp_limit):
    """Select native output conversion; vendor FP8 paths keep software encoding."""
    if output.dtype != torch.float8_e4m3fn:
        return True
    return runtime.device.vendor_name == "nvidia" and (
        not isinstance(use_ue8m0, bool)
        or not math.isfinite(eps)
        or eps <= 0
        or (
            clamp_limit is not None
            and (not math.isfinite(clamp_limit) or clamp_limit < 0)
        )
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("silu_mul_per_token_group_quant_fp8_colmajor"),
    key=["M", "K", "DTYPE", "USE_UE8M0", "HAS_CLAMP", "FP8_MAX"],
    use_cuda_graph=True,
)
@triton.jit(do_not_specialize=["eps", "clamp_limit"])
def _silu_mul_quant_kernel(
    X,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    eps,
    clamp_limit,
    DTYPE: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    HAS_CLAMP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FP8_MAX: tl.constexpr = 448.0,
    NORMALIZE_UP_ZERO: tl.constexpr = False,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1).to(tl.int64)
    cols = group * 128 + tl.arange(0, 128)
    offsets = rows[:, None] * (2 * K) + cols[None, :]
    gate = tl.load(X + offsets)
    up = tl.load(X + offsets + K)
    if HAS_CLAMP:
        gate = tl.minimum(gate.to(tl.float32), clamp_limit).to(X.dtype.element_ty)
        up = tl.clamp(up.to(tl.float32), -clamp_limit, clamp_limit).to(
            X.dtype.element_ty
        )
    if NORMALIZE_UP_ZERO:
        # vLLM's default up + beta (beta=+0) canonicalizes input signed zero.
        up = tl.where(up == 0.0, 0.0, up).to(X.dtype.element_ty)
    gate_f32 = gate.to(tl.float32)
    silu = (gate_f32 / (1.0 + tl.exp(-gate_f32))).to(X.dtype.element_ty)
    values = (silu * up).to(X.dtype.element_ty).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values), axis=1), eps)
    scales = absmax * (1.0 / FP8_MAX)
    if USE_UE8M0:
        scales = tl.exp2(tl.ceil(tl.log2(scales)))
    quantized = tl.clamp(values / scales[:, None], -FP8_MAX, FP8_MAX)
    tl.store(Q + rows[:, None] * K + cols[None, :], quantized.to(Q.dtype.element_ty))
    tl.store(S + group * M + rows, scales)


@triton.jit(do_not_specialize=["CLAMP"])
def _silu_mul_quant_tile(
    X,
    Q,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP,
    HAS_CLAMP: tl.constexpr,
    UE: tl.constexpr,
    BR: tl.constexpr,
    BG: tl.constexpr,
    VENDOR: tl.constexpr,
    INDEX64: tl.constexpr = False,
):
    # The host proves full tiles and selects the required offset width.
    # Axes are independent rows, adjacent groups, and each 128-value reduction.
    row_pid = tl.program_id(0)
    group_pid = tl.program_id(1)
    if INDEX64:
        row_pid = row_pid.to(tl.int64)
        group_pid = group_pid.to(tl.int64)
    rows = row_pid * BR + tl.arange(0, BR)
    groups = group_pid * BG + tl.arange(0, BG)
    cols = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    offsets = rows[:, None, None] * N + cols[None, :, :]
    gate = tl.load(X + offsets)
    up = tl.load(X + offsets + N // 2)
    if HAS_CLAMP:
        gate = tl.minimum(gate.to(tl.float32), CLAMP).to(X.dtype.element_ty)
        up = tl.clamp(up.to(tl.float32), -CLAMP, CLAMP).to(X.dtype.element_ty)
    if VENDOR != "nvidia":
        up = tl.where(up == 0.0, 0.0, up).to(X.dtype.element_ty)
    gate = gate.to(tl.float32)
    # Preserve both input-precision rounding boundaries from vLLM 0.23.
    silu = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty)
    y = (silu * up).to(X.dtype.element_ty).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), axis=2), EPS)
    scale = amax * (1.0 / 448.0)
    if UE:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    # On the small NVIDIA tile path, a normal power-of-two scale has an exact
    # FP32 reciprocal; this avoids per-value division without changing rounding.
    if (
        (
            VENDOR == "metax"
            or VENDOR == "thead"
            or VENDOR == "mthreads"
            or (VENDOR == "nvidia" and M <= 2048 and N >= 2048)
        )
        and UE
        and (VENDOR == "thead" or BR * BG > 1)
        and EPS >= 7.888609052210118e-31
        and EPS <= 3.4028234663852886e38
    ):
        normalized = y * (1.0 / scale)[:, :, None]
    else:
        normalized = y / scale[:, :, None]
    quant = tl.clamp(normalized, -448.0, 448.0)
    if VENDOR == "nvidia" or VENDOR == "mthreads":
        q = quant.to(Q.dtype.element_ty)
    else:
        # Native E4M3FN conversion is unavailable on these vendor backends.
        # Encode finite clamped values, then restore the NaN mask and zero sign.
        isnan = quant != quant
        safe = tl.where(isnan, 0.0, quant)
        b = safe.to(tl.uint32, bitcast=True)
        a = b & 0x7FFFFFFF
        t = a - 0x3C000000
        t += 0x0007FFFF + ((t >> 20) & 1)
        normal = t >> 20
        sub = (a.to(tl.float32, bitcast=True) * 512.0 + 8388608.0).to(
            tl.uint32, bitcast=True
        ) - 0x4B000000
        encoded = (tl.where(a >= 0x3C800000, normal, sub) | ((b >> 24) & 0x80)).to(
            tl.uint8
        )
        if (
            VENDOR == "metax"
            and UE
            and BR * BG > 1
            and EPS >= 7.888609052210118e-31
            and EPS <= 3.4028234663852886e38
        ):
            sign = ((y.to(tl.uint32, bitcast=True) >> 24) & 0x80).to(tl.uint8)
            encoded = tl.where(quant == 0.0, encoded | sign, encoded)
        sign = ((quant.to(tl.uint32, bitcast=True) >> 24) & 0x80).to(tl.uint8)
        encoded = tl.where(isnan, sign | 0x7F, encoded).to(tl.uint8)
        q = encoded.to(Q.dtype.element_ty, bitcast=True)
    tl.store(Q + rows[:, None, None] * (N // 2) + cols[None, :, :], q)
    tl.store(S + groups[None, :] * M + rows[:, None], scale)


def _prune_silu_tiles(configs, named_args, **kwargs):
    """Only launch full tiles; do not encode benchmark shapes in dispatch."""
    m, n = named_args["M"], named_args["N"]
    return [
        config
        for config in configs
        if m % config.kwargs["BR"] == 0
        and (n // 256) % config.kwargs["BG"] == 0
        and config.kwargs["BR"] * config.kwargs["BG"] <= 64
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("silu_mul_per_token_group_quant_fp8_tile_default"),
    flagtune_op_name="silu_mul_per_token_group_quant_fp8_colmajor",
    flagtune_expand_op_name="silu_mul_per_token_group_quant_fp8_tile",
    key=["M", "N", "DTYPE", "UE", "HAS_CLAMP", "DEVICE_NAME"],
    prune_configs_by={"early_config_prune": _prune_silu_tiles},
    use_cuda_graph=True,
    rep=30,
)
@triton.jit(do_not_specialize=["CLAMP"])
def _silu_mul_quant_tile_nvidia(
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
    BR: tl.constexpr,
    BG: tl.constexpr,
):
    _silu_mul_quant_tile(X, Q, S, M, N, EPS, CLAMP, HAS_CLAMP, UE, BR, BG, "nvidia")


def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    use_ue8m0: Optional[bool] = None,
    eps: float = 1e-10,
    clamp_limit: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``SiLU(input[:, :K]) * input[:, K:]`` with group size 128.

    Matches vLLM 0.23's input-precision activation/multiply rounding. Output
    defaults to the vLLM platform FP8 dtype; a supplied output keeps its dtype.
    Scales are FP32 with
    vLLM's transposed allocation layout. UE8M0 mode
    rounds scales up to powers of two but does not pack them into bytes.
    ``use_ue8m0=None`` follows vLLM's ``is_deep_gemm_e8m0_used()``.
    Forward inference only; noncontiguous input/output is unsupported. This
    generic launch supports NVIDIA SM90+; other vendors need their override.
    """
    if runtime.device.vendor_name != "nvidia" or not kernel_supports_fp8_e4m3():
        raise NotImplementedError(
            "this kernel requires NVIDIA SM90+ or a vendor override"
        )
    output, scales, use_ue8m0 = _validate_and_allocate(
        input, output, use_ue8m0, eps, clamp_limit
    )
    m, n = input.shape
    if not input.numel():
        return output, scales
    if _needs_general_quantization(output, use_ue8m0, eps, clamp_limit):
        with torch_device_fn.device(input.device):
            _launch_platform_fp8(input, output, scales, use_ue8m0, eps, clamp_limit)
        return output, scales
    # Full tiles with int32 addressing share one tuning space. Each new
    # shape/dtype/scale mode and GPU model is measured independently.
    if input.dtype in (torch.float16, torch.bfloat16) and input.numel() < 2**31:
        with torch_device_fn.device(input.device):
            _silu_mul_quant_tile_nvidia[
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
                torch.cuda.get_device_name(input.device),
            )
        return output, scales
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), n // 256)
    with torch_device_fn.device(input.device):
        _silu_mul_quant_kernel[grid](
            input,
            output,
            scales,
            m,
            n // 2,
            eps,
            clamp_limit if clamp_limit is not None else 0.0,
            str(input.dtype),
            use_ue8m0,
            clamp_limit is not None,
            FP8_MAX=448.0,
        )
    return output, scales
