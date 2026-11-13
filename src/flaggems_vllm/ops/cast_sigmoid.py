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

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_RANK = 8


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cast_sigmoid"),
    key=["N"],
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _cast_sigmoid_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # exp2 only accepts FP32/FP64 on the supported backends, so the sigmoid is
    # always evaluated in FP32 and cast once on the way out.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # 1 / (1 + exp(-x)) == 1 / (1 + exp2(-x * log2(e))); exp2 maps directly to
    # the SFU instruction and the multiply folds into the argument.
    y = 1.0 / (1.0 + tl.math.exp2(-x * 1.4426950408889634))
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def cast_sigmoid(
    x: torch.Tensor,
    out_dtype: torch.dtype = None,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply sigmoid element-wise with explicit output and compute dtypes.

    Used by vLLM's grouped-topk gating scores: the sigmoid is evaluated in
    ``compute_dtype`` (FP32 by default, for numerical stability) and the result
    is stored as ``out_dtype`` (the input dtype when ``None``).

    Args:
        x: strided tensor of rank at most eight, FP16/BF16/FP32.
        out_dtype: optional output dtype, one of FP16/BF16/FP32.
        compute_dtype: internal computation dtype; only FP32 is supported today
            because the backend's exp2 accepts FP32/FP64 only.

    Returns:
        A new tensor with the shape of ``x`` and dtype ``out_dtype``.
    """
    logger.debug("GEMS CAST_SIGMOID")

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.device.type in ("cpu", "meta"):
        raise NotImplementedError("cast_sigmoid requires an accelerator device")
    if x.layout != torch.strided:
        raise NotImplementedError("cast_sigmoid requires strided tensors")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError("cast_sigmoid supports FP16, BF16 and FP32")
    if x.ndim > _MAX_RANK:
        raise NotImplementedError("cast_sigmoid supports rank at most eight")

    if out_dtype is None:
        y_dtype = x.dtype
    else:
        if out_dtype not in _SUPPORTED_DTYPES:
            raise NotImplementedError("cast_sigmoid output must be FP16, BF16 or FP32")
        y_dtype = out_dtype

    if compute_dtype is None:
        compute_dtype = torch.float32
    if compute_dtype != torch.float32:
        raise NotImplementedError(
            "cast_sigmoid currently computes in FP32 only; pass "
            "compute_dtype=torch.float32"
        )

    output = torch.empty(x.shape, dtype=y_dtype, device=x.device)
    n = x.numel()
    if n == 0:
        return output

    # A non-contiguous input comes from an upstream view/transpose; one copy is
    # cheaper than a general stride-decode in this bandwidth-bound kernel.
    inp = x if x.is_contiguous() else x.contiguous()

    with runtime.torch_device_fn.device(x.device):
        _cast_sigmoid_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)](
            inp,
            output,
            n,
        )
    return output
