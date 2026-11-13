# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Kimi-K3 full-key copy, with no cache, RoPE, quantization, or query work."""

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def _prune_copy_configs(configs, named_args, **kwargs):
    if runtime.device.vendor_name == "hygon":
        elements = named_args["OUTPUT_ELEMENTS"]
        heads = named_args["H"]
        # Keep the formerly AABS-generated tiny tiles explicit. More programs
        # reduce launch-limited latency for a few narrow rows on DCU.
        tiny_block = 1 if elements == 192 else 8
        if elements <= 8192 and heads <= 3:
            return [
                config
                for config in configs
                if config.kwargs["BLOCK"] == tiny_block
                and not config.kwargs["ROW_LAYOUT"]
            ]
        return [config for config in configs if config.kwargs["BLOCK"] >= 256]
    # MUSA benefits from more programs for tiny copies. Avoid benchmarking
    # that tiny tile over large projections, where it creates millions of CTAs.
    if (
        runtime.device.vendor_name == "mthreads"
        and named_args["OUTPUT_ELEMENTS"] > 8192
    ):
        return [config for config in configs if config.kwargs["BLOCK"] >= 256]
    return configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_kimi_k3_mla_kv_concat"),
    key=["OUTPUT_ELEMENTS", "H", "NS0", "NS1", "PS0"],
    restore_value=["OUT"],
    prune_configs_by={"early_config_prune": _prune_copy_configs},
)
@triton.jit
def _kv_concat_kernel(
    N,
    P,
    OUT,
    OUTPUT_ELEMENTS: tl.constexpr,
    H: tl.constexpr,
    NS0,
    NS1,
    PS0,
    BLOCK: tl.constexpr,
    ROW_LAYOUT: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    program = tl.program_id(0)
    if WIDE_INDEX:
        program = program.to(tl.int64)
    index = program * BLOCK + tl.arange(0, BLOCK)
    if ROW_LAYOUT:
        # A power-of-two logical row exposes contiguous source vectors. The
        # final 64 lanes are masked; output rows remain densely packed at 192.
        dim = index % 256
        row = index // 256
        head = row % H
        token = row // H
        out_index = row * 192 + dim
    else:
        dim = index % 192
        head = index // 192 % H
        token = index // (192 * H)
        out_index = index
    # BLOCK tiles elements. Derive the token limit from that element count:
    # a bare token-count argument makes AABS pair BLOCK with the wrong unit.
    # Keep a token-uniform mask so complete rows retain vectorized accesses.
    valid = token < OUTPUT_ELEMENTS // (H * 192)
    # Arguments are metadata-only int16 views; no floating-point conversion.
    n_offset = token.to(tl.int64) * NS0 + head.to(tl.int64) * NS1 + dim
    nv = tl.load(N + n_offset, valid & (dim < 128), other=0)
    pv = tl.load(
        P + token.to(tl.int64) * PS0 + dim - 128,
        valid & (dim >= 128) & (dim < 192),
        other=0,
    )
    tl.store(OUT + out_index, tl.where(dim < 128, nv, pv), valid & (dim < 192))


def fused_kimi_k3_mla_kv_concat(
    k_nope: torch.Tensor, k_pe: torch.Tensor, k_out: torch.Tensor
) -> None:
    """Write [T,H,128] + shared [T,64] into caller-owned [T,H,192].

    Only the source last strides must be one. The output must be contiguous.
    Output/input byte-span overlap is unsupported and rejected conservatively
    (including gaps in strided inputs). The native CUDA pointers are restrict
    qualified and provide no overlap safety guarantee. Empty outputs launch
    nothing. All copies preserve the 16-bit storage representation.
    """
    logger.debug("GEMS FUSED_KIMI_K3_MLA_KV_CONCAT")
    if any(not isinstance(x, torch.Tensor) for x in (k_nope, k_pe, k_out)):
        raise TypeError("k_nope, k_pe, and k_out must be tensors")
    if k_nope.ndim != 3 or k_pe.ndim != 2 or k_out.ndim != 3:
        raise ValueError("expected k_nope [T,H,128], k_pe [T,64], k_out [T,H,192]")
    tokens, heads, width = k_nope.shape
    if (
        width != 128
        or k_pe.shape != (tokens, 64)
        or k_out.shape != (tokens, heads, 192)
    ):
        raise ValueError("expected matching T/H and fixed widths 128/64/192")
    if k_nope.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("only FP16 and BF16 are supported")
    if not (k_nope.dtype == k_pe.dtype == k_out.dtype):
        raise TypeError("all tensor dtypes must match")
    if not (k_nope.device == k_pe.device == k_out.device):
        raise ValueError("all tensors must be on the same device")
    if any(x.layout != torch.strided for x in (k_nope, k_pe, k_out)):
        raise ValueError("only strided tensors are supported")
    if k_nope.stride(-1) != 1 or k_pe.stride(-1) != 1:
        raise ValueError("source last-dimension strides must be one")
    if not k_out.is_contiguous():
        raise ValueError("k_out must be contiguous")
    if k_nope.device.type != runtime.device.name or k_nope.device.type == "cpu":
        raise NotImplementedError("an active accelerator device is required")
    if not k_out.numel():
        return None
    out_start = k_out.data_ptr()
    out_end = out_start + k_out.numel() * k_out.element_size()
    for source in (k_nope, k_pe):
        start = source.data_ptr()
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(source.shape, source.stride())
        )
        end = start + span * source.element_size()
        if start < out_end and out_start < end:
            raise ValueError("k_out must not overlap either input byte span")
    with torch_device_fn.device(k_nope.device):
        if runtime.device.vendor_name == "ascend":
            from flaggems_vllm.runtime.backend._ascend.ops.fused_kimi_k3_mla_kv_concat import (
                _launch_ascend,
            )

            _launch_ascend(
                k_nope.view(torch.int16),
                k_pe.view(torch.int16),
                k_out.view(torch.int16),
            )
            return None
        _kv_concat_kernel[
            lambda meta: (
                triton.cdiv(
                    tokens * heads * (256 if meta["ROW_LAYOUT"] else 192), meta["BLOCK"]
                ),
            )
        ](
            k_nope.view(torch.int16),
            k_pe.view(torch.int16),
            k_out.view(torch.int16),
            tokens * heads * 192,
            heads,
            k_nope.stride(0),
            k_nope.stride(1),
            k_pe.stride(0),
            # Leave a factor-of-two margin for the last masked block. Source
            # pointer arithmetic is always int64, including sparse views.
            WIDE_INDEX=tokens * heads * 256 >= 2**30,
        )
    return None


__all__ = ["fused_kimi_k3_mla_kv_concat"]
