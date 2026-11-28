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
import triton.language.extra.libdevice as libdevice

from flaggems_vllm import runtime
from flaggems_vllm.ops.per_token_quant_int8 import (
    _per_token_quant_int8_kernel as _common_kernel,
)
from flaggems_vllm.ops.per_token_quant_int8 import (
    _prune_quant_configs as _prune_common_configs,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)
_quantize_rows = _common_kernel.jit_function


def _prune_quant_configs(configs, named_args, **kwargs):
    configs = _prune_common_configs(configs, named_args, **kwargs)
    M = named_args["M"]
    block_n = named_args["BLOCK_N"]
    cta_budget = triton.next_power_of_2(get_sm_count())
    pruned = []
    for config in configs:
        parts = config.kwargs["QUANT_PARTS"]
        # Local BF16 reductions need aligned rows; large grids favor the
        # ordinary reduction without the additional layout conversion.
        if config.kwargs["REDUCE_GROUP"] and (
            M > cta_budget
            or block_n > 8192
            or named_args["N"] % 8
            or named_args["stride_xm"] % 8
        ):
            continue
        if block_n > 8192 and config.kwargs["VEC_SIZE"] != 8:
            continue
        # Replicated maxima stop helping after enough CTAs fill the device.
        if parts > 1 and M * parts > cta_budget:
            continue
        if block_n <= 1024:
            if M >= 1024:
                lower, max_warps = 1, 2
            else:
                lower, max_warps = min(4, max(1, block_n // 128)), 4
        elif parts > 1:
            # Limit the dependent quantization work to eight values per lane.
            lower = min(8, max(1, triton.cdiv(block_n // parts, 64 * 8)))
            max_warps = 8
        elif (
            M <= cta_budget
            or named_args["x_ptr"].dtype == torch.float32
            or block_n > 8192
        ):
            # Wide single-wave rows spill; FP32 and small grids need more waves.
            lower = max_warps = 8
        elif block_n <= 4096 or M >= 1024:
            # Four waves balance packed stores and occupancy for 16-bit prefill.
            lower = max_warps = 4
        else:
            lower = max_warps = 8
        if lower <= config.num_warps <= max_warps:
            pruned.append(config)
    return pruned


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_quant_int8"),
    key=["M", "N", "stride_xm", "stride_xn"],
    prune_configs_by={"early_config_prune": _prune_quant_configs},
    rep=20,
)
@triton.jit
def _per_token_quant_int8_metax_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    ROW_TILES: tl.constexpr,
    CONTIGUOUS_ROWS: tl.constexpr,
    REDUCE_GROUP: tl.constexpr,
    QUANT_PARTS: tl.constexpr = 1,
):
    # Inline the shared arithmetic under a separate tuning/cache identity.
    _quantize_rows(
        x_ptr,
        xq_ptr,
        scale_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        BLOCK_N,
        BLOCK_M,
        VEC_SIZE,
        ROW_TILES,
        CONTIGUOUS_ROWS,
        REDUCE_GROUP,
        QUANT_PARTS,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_quant_int8_flat"),
    key=["M", "N", "stride_xm", "stride_xn", "ALIGNED"],
    rep=20,
)
@triton.jit
def _per_token_quant_int8_metax_flat_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    BLOCK_N: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    STREAM_STORE: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    # One CTA owns one wide row. Keeping the row one-dimensional avoids the
    # extra layout and register pressure of the general row-tiled kernel.
    row = tl.program_id(0)
    cols = tl.max_contiguous(tl.arange(0, BLOCK_N), VEC_SIZE)
    mask = (row < M) & (cols < N)
    in_base = row * stride_xm
    out_base = row * N
    if ALIGNED:
        in_base = tl.multiple_of(in_base, 16)
        out_base = tl.multiple_of(out_base, 16)
    x = tl.load(x_ptr + in_base + cols * stride_xn, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(xf), axis=0).to(tl.float32), 1e-10)
    scale = absmax / 127.0
    xq = libdevice.round(xf * (127.0 / absmax)).to(tl.int8)
    if STREAM_STORE:
        tl.store(
            xq_ptr + out_base + cols,
            xq,
            mask=mask,
            cache_modifier=".cs",
        )
    else:
        tl.store(xq_ptr + out_base + cols, xq, mask=mask)
    tl.store(scale_ptr + row, scale, mask=row < M)


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logger.debug("GEMS PER TOKEN QUANT INT8 METAX")
    shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, shape[-1])
    N = x.shape[-1]
    M = x.numel() // N
    xq = torch.empty(shape, device=x.device, dtype=torch.int8)
    scales = torch.empty(shape[:-1] + (1,), device=x.device, dtype=torch.float32)

    with torch_device_fn.device(x.device):
        block_n = triton.next_power_of_2(N)
        aligned = (
            N % 16 == 0
            and x.stride(-2) % 16 == 0
            and x.stride(-1) == 1
            and x.data_ptr() % 16 == 0
        )
        if M >= 64 and 5120 <= N <= 16384:
            _per_token_quant_int8_metax_flat_kernel[(M,)](
                x,
                xq,
                scales,
                M,
                N,
                x.stride(-2),
                x.stride(-1),
                block_n,
                ALIGNED=aligned,
            )
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"] * meta["ROW_TILES"]),
                meta["QUANT_PARTS"],
            )
            _per_token_quant_int8_metax_kernel[grid](
                x,
                xq,
                scales,
                M,
                N,
                x.stride(-2),
                x.stride(-1),
                block_n,
            )
    return xq, scales
