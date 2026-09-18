# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence

import torch


def fp8_einsum(
    equation: str,
    x: torch.Tensor,
    xs: torch.Tensor,
    y: torch.Tensor,
    ys: torch.Tensor,
    block_size: Sequence[int] = (128, 128),
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute block-scaled FP8 ``bhr,hdr->bhd`` on NVIDIA Hopper.

    ``x[b,h,r]`` and ``y[h,d,r]`` contain E4M3 data. FP32 scales have shapes
    ``xs[b,h,r/128]`` and ``ys[h,d/128,r/128]``. The reduction and output
    feature dimensions must be multiples of 128. Inputs may be strided,
    provided the data tensors have a contiguous last dimension.

    Return a new tensor, or write into and return ``out``. Output supports
    BF16 (default), FP16 and FP32. This is an inference-only operator.
    """
    if equation != "bhr,hdr->bhd":
        raise NotImplementedError("fp8_einsum only supports 'bhr,hdr->bhd'")
    if tuple(block_size) != (128, 128):
        raise NotImplementedError("fp8_einsum requires 128x128 scale blocks")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError("output_dtype must be bfloat16, float16 or float32")
    inputs = (x, xs, y, ys)
    if any(t.layout != torch.strided or t.ndim != 3 for t in inputs):
        raise ValueError("inputs must be strided rank-3 tensors")
    if any(t.device != x.device for t in inputs):
        raise ValueError("all inputs must be on the same device")
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise NotImplementedError("fp8_einsum requires an NVIDIA Hopper GPU")
    if torch.cuda.get_device_capability(x.device)[0] != 9:
        raise NotImplementedError("fp8_einsum requires an NVIDIA Hopper GPU")
    if x.dtype != torch.float8_e4m3fn or y.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("x and y must have dtype float8_e4m3fn")
    if xs.dtype != torch.float32 or ys.dtype != torch.float32:
        raise NotImplementedError("xs and ys must have dtype float32")
    if any(t.requires_grad for t in inputs):
        raise NotImplementedError("fp8_einsum does not support autograd")

    b, h, r = x.shape
    h2, d, r2 = y.shape
    if h2 != h or r2 != r:
        raise ValueError("x and y head/reduction dimensions must match")
    if r % 128 or d % 128:
        raise NotImplementedError("r and d must be multiples of 128")
    if xs.shape != (b, h, r // 128) or ys.shape != (h, d // 128, r // 128):
        raise ValueError("scale shapes do not match the 128x128 block grid")
    if x.stride(-1) != 1 or y.stride(-1) != 1:
        raise NotImplementedError("x and y must have a contiguous last dimension")

    if out is None:
        out = torch.empty((b, h, d), device=x.device, dtype=output_dtype)
    else:
        if (
            out.shape != (b, h, d)
            or out.device != x.device
            or out.dtype != output_dtype
        ):
            raise ValueError("out must match the output shape, device and dtype")
        if out.layout != torch.strided or out.requires_grad:
            raise ValueError("out must be a strided tensor without autograd")
        # Dense last dimensions and monotonically separated rows exclude
        # overlapping output views while allowing padded output buffers.
        if (
            out.stride(-1) != 1
            or out.stride(1) < d
            or out.stride(0) < h * out.stride(1)
        ):
            raise ValueError("out must have non-overlapping rows in [b,h,d] order")
        if out.numel() and any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in inputs
        ):
            raise ValueError("out must not share storage with an input")

    if out.numel() == 0:
        return out

    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_bmm import (
        w8a8_block_fp8_bmm,
    )

    w8a8_block_fp8_bmm(
        x.permute(1, 0, 2),
        y,
        xs.permute(1, 0, 2),
        ys,
        block_size=block_size,
        z=out.permute(1, 0, 2),
        output_dtype=output_dtype,
    )
    return out
