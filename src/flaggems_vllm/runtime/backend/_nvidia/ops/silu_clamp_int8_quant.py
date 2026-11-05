# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Selected clamped SiLU/INT8 arithmetic, isolated from generic MoE quantizers."""

import logging
import sys

import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)
_INT8_QUANT_REQUIRED_FG_KEYS = frozenset(
    ("abs", "amax", "clamp", "div.Scalar", "div.Tensor", "round")
)
_INT8_QUANT_ROUTING_FG_KEYS = _INT8_QUANT_REQUIRED_FG_KEYS | {"_to_copy"}


def _int8_quant_dispatch_spec() -> tuple[bool, bool] | None:
    """Read the live seven-key signature once; never cache across calls.

    The pinned registrar exposes mutable lists without a public generation.
    One key pass and one config pass detect even same-length in-place edits.
    Legacy full/plain behavior and the narrow mixed producer contract remain.
    """
    package = sys.modules.get("flag_gems")
    registrar = getattr(package, "current_work_registrar", None)
    if registrar is None:
        return False, False
    if not hasattr(registrar, "get_all_keys"):
        return None
    keys = registrar.get_all_keys()
    positions = {}
    for index, key in enumerate(keys):
        if key in _INT8_QUANT_ROUTING_FG_KEYS:
            positions[key] = -1 if key in positions else index
    if _INT8_QUANT_REQUIRED_FG_KEYS.issubset(positions):
        return True, False
    if (type(registrar).__module__, type(registrar).__name__) != (
        "flag_gems.runtime.op_registrar",
        "GeneralOpRegistrar",
    ) or getattr(registrar, "reg_key", None) != "CUDA":
        return None
    if not hasattr(registrar, "get_all_ops") or not hasattr(registrar, "config"):
        return None
    names = registrar.get_all_ops()
    if len(keys) != len(names) or {"div.Scalar", "div.Tensor"} & positions.keys():
        return None
    configs = {}
    for entry in registrar.config:
        key = entry[0]
        if key in _INT8_QUANT_ROUTING_FG_KEYS:
            if key in {"div.Scalar", "div.Tensor"} or key in configs:
                return None
            configs[key] = entry
    expected = (
        ("abs", "flag_gems.ops.abs", "abs"),
        ("amax", "flag_gems.ops.amax", "amax"),
        ("clamp", "flag_gems.ops.clamp", "clamp"),
        ("round", "flag_gems.ops.round", "round"),
        ("_to_copy", "flag_gems.ops.to", "to_copy"),
    )
    for key, module_name, function_name in expected:
        index = positions.get(key, -1)
        canonical = getattr(sys.modules.get(module_name), function_name, None)
        config = configs.get(key)
        if (
            index < 0
            or names[index] != function_name
            or canonical is None
            or getattr(package, function_name, None) is not canonical
            or getattr(canonical, "__module__", None) != module_name
            or getattr(canonical, "__name__", None) != function_name
            or config is None
            or config[1] is not canonical
            or (len(config) > 2 and config[2])
        ):
            return None
    return True, True


@triton.jit
def _round_to_int8_even(x, FG_DISPATCH: tl.constexpr):
    """Portable round-to-nearest-even for the bounded INT8 quotient."""
    # PyTorch's round() precedes clamp() and casts NaN to zero on the target
    # integer path.  Clamp first is equivalent for finite inputs, including
    # half-way cases at the endpoints, and makes Inf handling explicit.
    if FG_DISPATCH:
        # FlagGems round/clamp/cast turns NaN quotients into -128 on the
        # pinned live backend; plain Torch turns them into zero.
        x = tl.where(x == x, x, -128.0)
    else:
        x = tl.where(x == x, x, 0.0)
    x = tl.minimum(tl.maximum(x, -128.0), 127.0)
    floor_x = tl.floor(x)
    fraction = x - floor_x
    odd = (floor_x.to(tl.int32) & 1) != 0
    rounded = floor_x + ((fraction > 0.5) | ((fraction == 0.5) & odd)).to(tl.float32)
    return rounded.to(tl.int8)


@triton.jit
def _per_token_int8_values(
    x,
    valid,
    storage_dtype: tl.constexpr,
    FG_DISPATCH: tl.constexpr,
    NATIVE_DIV: tl.constexpr = False,
):
    """Match _int8_quantize's source-dtype amax/clamp and FP32 scale/divide."""
    x = x.to(tl.float32)
    amax = tl.max(tl.where(valid, tl.abs(x), 0.0), axis=0)
    # tl.max ignores NaN while torch.amax propagates it.  The second reduction
    # is zero for finite input and NaN if any valid lane contains NaN.
    if not FG_DISPATCH:
        nan_term = tl.sum(tl.where(valid & (x != x), x, 0.0), axis=0)
    # Form the literal in FP32 first. Triton's low-precision tl.full can
    # underflow this BF16 clamp constant on the target compiler.
    eps = tl.full((), 1e-10, tl.float32).to(storage_dtype).to(tl.float32)
    amax = tl.maximum(amax, eps)
    if not FG_DISPATCH:
        amax += nan_term
    # PyTorch's Tensor / Python-int scalar path lowers to FP32 reciprocal
    # multiplication on this runtime; rounded division differs by one ULP
    # for values such as amax=52 and can flip INT8 half-way decisions.
    scale = amax * tl.full((), 1.0 / 127.0, tl.float32)
    # FlagGems true_divide uses Triton's default approximate f32 division;
    # its half-way decisions can differ from PyTorch's rounded division.
    if FG_DISPATCH and not NATIVE_DIV:
        quotient = x / scale
    else:
        quotient = tl.div_rn(x, scale)
    return _round_to_int8_even(quotient, FG_DISPATCH), scale


@libentry()
@triton.jit
def _swigluoai_quantize_per_token_kernel(
    input_ptr,
    activation_ptr,
    quant_ptr,
    scale_ptr,
    K: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_om: tl.constexpr,
    stride_ok: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    clamp_limit: tl.constexpr,
    FG_DISPATCH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NATIVE_DIV: tl.constexpr = False,
    SILU: tl.constexpr = False,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    valid = cols < K
    gate = tl.load(input_ptr + row * stride_im + cols * stride_ik, valid, other=0).to(
        tl.float32
    )
    up = tl.load(
        input_ptr + row * stride_im + (K + cols) * stride_ik, valid, other=0
    ).to(tl.float32)
    gate = tl.minimum(gate, clamp_limit)
    up = tl.minimum(tl.maximum(up, -clamp_limit), clamp_limit)
    if SILU:
        activated = gate / (1.0 + tl.exp(-gate)) * up
    else:
        activated = gate * tl.sigmoid(alpha * gate) * (up + beta)
    # Retain the observable BF16/FP16 workspace rounding before quant2.
    activated = activated.to(activation_ptr.dtype.element_ty)
    tl.store(activation_ptr + row * stride_om + cols * stride_ok, activated, valid)
    q, scale = _per_token_int8_values(
        activated, valid, activation_ptr.dtype.element_ty, FG_DISPATCH, NATIVE_DIV
    )
    tl.store(quant_ptr + row * K + cols, q, valid)
    tl.store(scale_ptr + row, scale)


def _swigluoai_quantize_per_token_triton(
    input: torch.Tensor,
    activation_output: torch.Tensor,
    alpha: float,
    beta: float,
    clamp_limit: float,
    *,
    _silu: bool = False,
    _validated_spec: tuple[bool, bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OAI + quant2, retaining the rounded activation workspace for inspection."""
    assert input.ndim == activation_output.ndim == 2
    assert input.shape == (activation_output.size(0), activation_output.size(1) * 2)
    assert input.dtype == activation_output.dtype
    assert input.dtype in (torch.bfloat16, torch.float16)
    M, K = activation_output.shape
    assert 0 < K <= 8192
    dispatch_spec = (
        _int8_quant_dispatch_spec() if _validated_spec is None else _validated_spec
    )
    if dispatch_spec is None:
        raise RuntimeError("Unknown FlagGems quant registration requires original path")
    fg_dispatch, native_div = dispatch_spec
    if _silu:
        alpha, beta = 1.0, 0.0
    q = torch.empty((M, K), device=input.device, dtype=torch.int8)
    scale = torch.empty((M, 1), device=input.device, dtype=torch.float32)
    if M:
        _swigluoai_quantize_per_token_kernel[(M,)](
            input,
            activation_output,
            q,
            scale,
            K,
            input.stride(0),
            input.stride(1),
            activation_output.stride(0),
            activation_output.stride(1),
            alpha,
            beta,
            clamp_limit,
            FG_DISPATCH=fg_dispatch,
            NATIVE_DIV=native_div,
            SILU=_silu,
            BLOCK_K=triton.next_power_of_2(K),
            num_warps=4 if K <= 2048 else 8,
        )
    return q, scale


@libentry()
@triton.jit
def _swigluoai_uninterleave_kernel(
    input_ptr,
    output_ptr,
    n_inter,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    alpha,
    beta,
    clamp_limit,
    BLOCK_I: tl.constexpr,
    SILU: tl.constexpr = False,
):
    """Split gate/up layout; compute in FP32 and round on output store."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter
    gate = tl.load(
        input_ptr + row * stride_im + cols * stride_in, mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(
        input_ptr + row * stride_im + (n_inter + cols) * stride_in,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.minimum(gate, clamp_limit)
    up = tl.minimum(tl.maximum(up, -clamp_limit), clamp_limit)
    if SILU:
        # Match the pad-aware SiLU reference without changing the OAI path.
        activated = gate / (1.0 + tl.exp(-gate)) * up
    else:
        activated = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        output_ptr + row * stride_om + cols * stride_on,
        activated.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@libentry()
@triton.jit
def _int8_quantize_per_token_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    K: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    FG_DISPATCH: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NATIVE_DIV: tl.constexpr = False,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    valid = cols < K
    x = tl.load(input_ptr + row * stride_im + cols * stride_ik, valid, other=0)
    q, scale = _per_token_int8_values(
        x, valid, input_ptr.dtype.element_ty, FG_DISPATCH, NATIVE_DIV
    )
    tl.store(output_ptr + row * K + cols, q, valid)
    tl.store(scale_ptr + row, scale)


def _int8_quantize_per_token_triton(
    A: torch.Tensor,
    *,
    _validated_spec: tuple[bool, bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stride-aware 2D dynamic INT8 quantizer; one program per input row."""
    assert A.ndim == 2 and A.dtype in (torch.bfloat16, torch.float16)
    M, K = A.shape
    assert 0 < K <= 8192
    dispatch_spec = (
        _int8_quant_dispatch_spec() if _validated_spec is None else _validated_spec
    )
    if dispatch_spec is None:
        raise RuntimeError("Unknown FlagGems quant registration requires original path")
    fg_dispatch, native_div = dispatch_spec
    A_q = torch.empty((M, K), device=A.device, dtype=torch.int8)
    scale = torch.empty((M, 1), device=A.device, dtype=torch.float32)
    if M:
        _int8_quantize_per_token_kernel[(M,)](
            A,
            A_q,
            scale,
            K,
            A.stride(0),
            A.stride(1),
            FG_DISPATCH=fg_dispatch,
            NATIVE_DIV=native_div,
            BLOCK_K=triton.next_power_of_2(K),
            num_warps=4 if K <= 2048 else 8,
        )
    return A_q, scale


def supports_silu_clamp_int8_quant(input, activation_output, clamp_limit):
    logger.debug("GEMS_NVIDIA SUPPORTS_SILU_CLAMP_INT8_QUANT")
    if not isinstance(input, torch.Tensor) or not isinstance(
        activation_output, torch.Tensor
    ):
        return False
    return (
        input.ndim == activation_output.ndim == 2
        and input.shape == (activation_output.shape[0], activation_output.shape[1] * 2)
        and input.dtype == activation_output.dtype
        and input.dtype in (torch.bfloat16, torch.float16)
        and input.device == activation_output.device
        and input.is_cuda
        and torch.version.hip is None
        and 0 < activation_output.shape[1] <= 8192
        and input.stride(1) > 0
        and activation_output.stride(1) > 0
        and input.stride(0) >= (input.shape[1] - 1) * input.stride(1) + 1
        and activation_output.stride(0)
        >= (activation_output.shape[1] - 1) * activation_output.stride(1) + 1
        and not torch._C._overlaps(input, activation_output)
        and not torch.is_grad_enabled()
        and not input.requires_grad
        and not activation_output.requires_grad
        and not input.is_conj()
        and not input.is_neg()
        and not activation_output.is_conj()
        and not activation_output.is_neg()
        and type(clamp_limit) in (int, float)
        and clamp_limit >= 0
        and clamp_limit < float("inf")
        and _int8_quant_dispatch_spec() is not None
    )


def silu_clamp_int8_quant(input, activation_output, clamp_limit):
    logger.debug("GEMS_NVIDIA SILU_CLAMP_INT8_QUANT")
    if not supports_silu_clamp_int8_quant(input, activation_output, clamp_limit):
        raise NotImplementedError(
            "unsupported SiLU/INT8 metadata or quantization policy"
        )
    with torch.cuda.device(input.device):
        return _swigluoai_quantize_per_token_triton(
            input,
            activation_output,
            1.0,
            0.0,
            clamp_limit,
            _silu=True,
            _validated_spec=_int8_quant_dispatch_spec(),
        )


def silu_clamp(input, activation_output, clamp_limit):
    logger.debug("GEMS_NVIDIA SILU_CLAMP")
    if not supports_silu_clamp_int8_quant(input, activation_output, clamp_limit):
        raise NotImplementedError("unsupported clamped SiLU metadata")
    if activation_output.shape[0]:
        with torch.cuda.device(input.device):
            _swigluoai_uninterleave_kernel[
                (
                    activation_output.shape[0],
                    triton.cdiv(activation_output.shape[1], 256),
                )
            ](
                input,
                activation_output,
                activation_output.shape[1],
                input.stride(0),
                input.stride(1),
                activation_output.stride(0),
                activation_output.stride(1),
                1.0,
                0.0,
                clamp_limit,
                BLOCK_I=256,
                SILU=True,
                num_warps=4,
            )
    return activation_output


__all__ = ["silu_clamp", "silu_clamp_int8_quant", "supports_silu_clamp_int8_quant"]


def supports_int8_quantize_per_token(x):
    logger.debug("GEMS_NVIDIA SUPPORTS_INT8_QUANTIZE_PER_TOKEN")
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_cuda
        and torch.version.hip is None
        and x.layout == torch.strided
        and 0 < x.shape[1] <= 8192
        and x.stride(1) > 0
        and x.stride(0) >= (x.shape[1] - 1) * x.stride(1) + 1
        and not x.is_conj()
        and not x.is_neg()
        and not x.requires_grad
        and not torch.is_grad_enabled()
        and _int8_quant_dispatch_spec() is not None
    )


def int8_quantize_per_token(x):
    logger.debug("GEMS_NVIDIA INT8_QUANTIZE_PER_TOKEN")
    if not supports_int8_quantize_per_token(x):
        raise NotImplementedError(
            "unsupported per-token INT8 metadata or quantization policy"
        )
    with torch.cuda.device(x.device):
        return _int8_quantize_per_token_triton(
            x, _validated_spec=_int8_quant_dispatch_spec()
        )


__all__ += ["int8_quantize_per_token", "supports_int8_quantize_per_token"]
