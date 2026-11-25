# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash Normal BF16 MegaMHC on the Hygon backend."""

from typing import NamedTuple, Optional

import torch
import triton

from .kernels import (
    _pack_mhc_weights,
    mega_mhc_batch_kernel,
    mega_mhc_decode_kernel,
    mega_mhc_packed_kernel,
    mega_mhc_row_kernel,
    mega_mhc_tile_kernel,
)

HIDDEN_SIZE = 4096


_MIN_PACKED_TOKENS = 416
_MAX_PACKED_TOKENS = 4096
_MAX_DECODE_TOKENS = 128
_DECODE_SPLITS = 16


class _PreparedMegaMHC(NamedTuple):
    fn: torch.Tensor
    packed: torch.Tensor
    partials: Optional[torch.Tensor]
    arrivals: Optional[torch.Tensor]


def mega_mhc_prepare(fn, *, max_decode_tokens=None, out=None):
    """Prepare projection weights and optional decode workspace before capture.

    Pass max_decode_tokens to reserve Split-K workspace for up to 128 tokens. Calls and
    graph replays sharing a decode workspace must be ordered. Give each
    concurrently executing stream or graph a separate prepared object.

    Keep the returned object alive through graph replay. Refresh it in place
    with ``out=prepared`` after updating fn and before any subsequent call or
    replay. Order the refresh after all calls that are using the old weights.
    """
    k = 4 * HIDDEN_SIZE
    if (
        not isinstance(fn, torch.Tensor)
        or fn.shape != (24, k)
        or fn.device.type != "cuda"
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
        or fn.requires_grad
    ):
        raise NotImplementedError("MegaMHC requires contiguous FP32 HCU fn [24, 16384]")
    if out is not None and (not isinstance(out, _PreparedMegaMHC) or out.fn is not fn):
        raise ValueError("Refresh requires the same MegaMHC weights")

    if max_decode_tokens is not None and (
        type(max_decode_tokens) is not int
        or not 0 <= max_decode_tokens <= _MAX_DECODE_TOKENS
    ):
        raise ValueError(
            f"MegaMHC decode capacity must be an integer from 0 to {_MAX_DECODE_TOKENS}"
        )

    with torch.cuda.device(fn.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare MegaMHC outside CUDA Graph capture")
        if out is None:
            packed = torch.empty((32, 2 * k), device=fn.device, dtype=torch.bfloat16)
            partials = arrivals = None
            if max_decode_tokens:
                partials = torch.empty(
                    (max_decode_tokens, 2, _DECODE_SPLITS, 32),
                    device=fn.device,
                    dtype=torch.float32,
                )
                arrivals = torch.empty(
                    (max_decode_tokens, 2), device=fn.device, dtype=torch.int32
                )
            out = _PreparedMegaMHC(fn, packed, partials, arrivals)
        elif max_decode_tokens is not None:
            capacity = 0 if out.arrivals is None else out.arrivals.shape[0]
            if max_decode_tokens != capacity:
                raise ValueError("Refresh requires the same MegaMHC decode capacity")
        _pack_mhc_weights[(triton.cdiv(32 * k, 256),)](
            fn,
            out.packed,
            out.arrivals,
            k,
            out.arrivals.numel() if out.arrivals is not None else 0,
            256,
            num_warps=4,
        )
    return out


def mega_mhc(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
    *,
    prepared=None,
):
    """Apply Normal BF16 mHC post, projection, and normalized pre in one GPU launch.

    Returns updated residual, post mix, combination mix, and BF16 layer input.
    See ``mega_mhc_prepare`` for prepacked weights and the optional decode workspace.
    Warm up before graph capture to compile the Triton specializations.
    """
    if x.ndim != 2 or x.shape[1] != HIDDEN_SIZE or x.device.type != "cuda":
        raise NotImplementedError("MegaMHC requires HCU x with shape [T, 4096]")
    if (
        rms_eps != 1e-6
        or hc_pre_eps != 1e-6
        or hc_sinkhorn_eps != 1e-6
        or norm_eps != 1e-6
        or hc_post_mult_value != 2.0
        or type(sinkhorn_repeat) is not int
        or sinkhorn_repeat != 20
        or n_splits != 1
        or tile_n != 1
    ):
        raise NotImplementedError(
            "MegaMHC requires the Normal BF16 V4-Flash parameters"
        )

    tokens, hidden = x.shape
    if tokens * 4 * hidden > 2**31:
        raise NotImplementedError("MegaMHC residual offsets exceed int32 addressing")
    bf16, fp32 = torch.bfloat16, torch.float32
    specs = (
        ("x", x, (tokens, hidden), bf16),
        ("residual", residual, (tokens, 4, hidden), bf16),
        ("post_layer_mix", post_layer_mix, (tokens, 4, 1), fp32),
        ("comb_res_mix", comb_res_mix, (tokens, 4, 4), fp32),
        ("fn", fn, (24, 4 * hidden), fp32),
        ("hc_scale", hc_scale, (3,), fp32),
        ("hc_base", hc_base, (24,), fp32),
        ("norm_weight", norm_weight, (hidden,), bf16),
    )
    for name, tensor, shape, dtype in specs:
        if (
            tensor is None
            or tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise NotImplementedError(
                f"MegaMHC requires contiguous {name} with {shape}, {dtype}, "
                f"on {x.device} and no gradients"
            )

    if prepared is not None and (
        not isinstance(prepared, _PreparedMegaMHC) or prepared.fn is not fn
    ):
        raise ValueError("MegaMHC prepared weights must match fn")

    def empty(shape, dtype):
        return torch.empty(shape, device=x.device, dtype=dtype)

    updated = empty(residual.shape, bf16)
    post = empty(post_layer_mix.shape, fp32)
    comb = empty(comb_res_mix.shape, fp32)
    y = empty(x.shape, bf16)
    if tokens == 0:
        return updated, post, comb, y

    use_decode = (
        prepared is not None
        and prepared.arrivals is not None
        and tokens <= prepared.arrivals.shape[0]
    )
    use_packed = (
        prepared is not None and _MIN_PACKED_TOKENS <= tokens <= _MAX_PACKED_TOKENS
    )
    buffers = (
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        prepared.packed if use_packed else fn,
        hc_scale,
        hc_base,
        norm_weight,
        updated,
        post,
        comb,
        y,
    )
    constants = (
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    )
    # A wider K split helps around 64 complete 16-token tiles.
    use_eight_partitions = 1024 <= tokens < 1040
    with torch.cuda.device(x.device):
        if use_decode:
            interleave_groups = 16 <= tokens <= 32
            splits = 8 if interleave_groups else _DECODE_SPLITS
            grid = (tokens, 2, splits) if interleave_groups else (tokens, splits, 2)
            mega_mhc_decode_kernel[grid](
                *buffers,
                prepared.partials,
                prepared.arrivals,
                hidden,
                splits,
                tokens == 1,
                interleave_groups,
                num_warps=4 if tokens <= 32 else 2,
            )
        elif use_packed:
            split_heads = tokens <= 1280
            partitions = 8 if split_heads else 4
            if tokens <= 640:
                block_m = 8
            else:
                block_m = 16 if tokens <= 2560 else 32
            mega_mhc_packed_kernel[
                (triton.cdiv(tokens, block_m), 2 if split_heads else 1)
            ](
                *buffers,
                tokens,
                hidden,
                block_m,
                64,
                partitions,
                split_heads,
                *constants,
                num_warps=4,
                num_stages=1,
                matrix_instr_nonkdim=16,
                kpack=2,
            )
        elif tokens <= 384:
            unroll = 2 if tokens <= 32 else 4
            mega_mhc_row_kernel[(tokens, 2)](
                *buffers,
                hidden,
                512,
                unroll,
                *constants,
                num_warps=4,
            )
        elif tokens <= 1536 and not use_eight_partitions:
            block_m = 8
            block_k, warps = (256, 8) if tokens <= 640 else (128, 4)
            mega_mhc_batch_kernel[(triton.cdiv(tokens, block_m), 2)](
                *buffers,
                tokens,
                hidden,
                block_m,
                block_k,
                1 if tokens <= 640 else 2,
                *constants,
                num_warps=warps,
            )
        else:
            if use_eight_partitions:
                block_m, block_k, panel, partitions, warps, stages = (
                    16,
                    128,
                    256,
                    8,
                    8,
                    1,
                )
            elif tokens <= 2560:
                block_m, block_k, panel, partitions, warps, stages = (
                    16,
                    64,
                    256,
                    4,
                    4,
                    2,
                )
            else:
                block_m, block_k, partitions, warps, stages = 32, 128, 1, 4, 2
                panel = 256 if tokens <= 4096 else 128
            mega_mhc_tile_kernel[(triton.cdiv(tokens, block_m),)](
                *buffers,
                tokens,
                hidden,
                block_m,
                block_k,
                128,
                panel,
                partitions,
                2,
                *constants,
                num_warps=warps,
                num_stages=stages,
                matrix_instr_nonkdim=16,
                kpack=2,
            )
    return updated, post, comb, y
