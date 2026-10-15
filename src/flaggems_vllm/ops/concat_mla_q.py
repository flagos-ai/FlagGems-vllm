# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""In-place MLA query concatenation without a Torch compute fallback."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("concat_mla_q"),
    key=["SIZE", "WIDTH"],
    strategy=["log", "log"],
)
@triton.jit
def _concat_mla_q_kernel(
    nope,
    rope,
    out,
    SIZE,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    WIDTH: tl.constexpr,
    ns0,
    ns1,
    ns2,
    rs0,
    rs1,
    rs2,
    os0,
    os1,
    os2,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    col = index % WIDTH
    head = (index // WIDTH) % HEADS
    token = index // (WIDTH * HEADS)
    nval = tl.load(
        nope + token * ns0 + head * ns1 + col * ns2,
        mask=(index < SIZE) & (col < NOPE),
        other=0,
    )
    rval = tl.load(
        rope + token * rs0 + head * rs1 + (col - NOPE) * rs2,
        mask=(index < SIZE) & (col >= NOPE),
        other=0,
    )
    tl.store(
        out + token * os0 + head * os1 + col * os2,
        tl.where(col < NOPE, nval, rval),
        mask=index < SIZE,
    )


def _validate_inputs(ql_nope, q_pe, q_out):
    if any(t.ndim != 3 for t in (ql_nope, q_pe, q_out)):
        raise ValueError("concat_mla_q requires rank-3 tensors")
    tokens, heads, nope = ql_nope.shape
    rope = q_pe.shape[-1]
    if heads <= 0 or nope <= 0 or rope <= 0:
        raise ValueError("concat_mla_q requires positive head count and widths")
    if q_pe.shape[:2] not in ((tokens, heads), (tokens, 1)):
        raise ValueError("concat_mla_q rope token/head dimensions do not match")
    if q_out.shape != (tokens, heads, nope + rope):
        raise ValueError("concat_mla_q output shape does not match")
    if not (ql_nope.dtype == q_pe.dtype == q_out.dtype):
        raise ValueError("concat_mla_q tensors must have the same dtype")
    if not (ql_nope.device == q_pe.device == q_out.device):
        raise ValueError("concat_mla_q tensors must be on the same device")
    if q_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("concat_mla_q supports FP16/BF16/FP32")
    if any(stride < 0 for t in (ql_nope, q_pe, q_out) for stride in t.stride()):
        raise NotImplementedError("concat_mla_q does not support negative strides")
    # Conservative metadata-only proof of disjoint output elements. Gaps in
    # storage are allowed, including padded token/head/feature strides.
    span = 1
    for stride, size in sorted(zip(q_out.stride(), q_out.shape)):
        if size <= 1:
            continue
        if stride < span:
            raise ValueError("concat_mla_q output elements overlap")
        span += (size - 1) * stride
    if q_out.numel():
        storage = q_out.untyped_storage().data_ptr()
        if any(storage == t.untyped_storage().data_ptr() for t in (ql_nope, q_pe)):
            raise ValueError("concat_mla_q output must not alias its inputs")
    if q_out.device.type != "cuda":
        raise NotImplementedError("concat_mla_q requires CUDA tensors")


def concat_mla_q(
    ql_nope: torch.Tensor, q_pe: torch.Tensor, q_out: torch.Tensor
) -> None:
    """Write ``[ql_nope, q_pe]`` into q_out, broadcasting one rope head.

    Inputs and output may be strided. Output aliasing is rejected before any
    launch. This is an inference-only data copy, with no dtype conversion.
    """
    _validate_inputs(ql_nope, q_pe, q_out)
    if not q_out.numel():
        return
    tokens, heads, nope = ql_nope.shape
    width = q_out.shape[-1]
    with torch.cuda.device(ql_nope.device):
        _concat_mla_q_kernel[lambda meta: (triton.cdiv(q_out.numel(), meta["BLOCK"]),)](
            ql_nope,
            q_pe,
            q_out,
            tokens * heads * width,
            heads,
            nope,
            width,
            *ql_nope.stride(),
            q_pe.stride(0),
            0 if q_pe.shape[1] == 1 else q_pe.stride(1),
            q_pe.stride(2),
            *q_out.stride(),
        )


__all__ = ["concat_mla_q"]
