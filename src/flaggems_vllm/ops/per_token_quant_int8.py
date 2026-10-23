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
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def _prune_quant_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    block_n = named_args["BLOCK_N"]
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        # A single warp already covers short rows without extra vector variants.
        if block_n <= 128 and config.kwargs["VEC_SIZE"] != 16:
            continue
        # Packing wide rows raises register pressure and reduces parallelism.
        if block_m > M or (block_m > 1 and block_n > 2048):
            continue
        # Short rows fit within a warp and do not benefit from CTA-wide reduction.
        if block_n <= 128 and config.num_warps > block_m:
            continue
        tile_size = block_m * block_n
        if config.num_warps * 32 > max(tile_size, 32):
            continue
        # Avoid more than 256 values per thread, keeping the widest candidate.
        if tile_size > 8192 * config.num_warps and config.num_warps < 16:
            continue
        pruned.append(config)
    return pruned


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("per_token_quant_int8"),
    key=["M", "N", "stride_xm", "stride_xn"],
    prune_configs_by={"early_config_prune": _prune_quant_configs},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit
def _per_token_quant_int8_kernel(
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
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # Balance input load coalescing against packed INT8 stores.
    cols = tl.max_contiguous(tl.arange(0, BLOCK_N), VEC_SIZE)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    )
    # Short rows favor FP32 arithmetic; wide rows need fewer live registers.
    if BLOCK_N <= 128:
        x = x.to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1).to(tl.float32), 1e-10)
    scale = absmax / 127.0
    # vLLM rounds halfway cases away from zero, unlike nearbyint / torch.round.
    xq = libdevice.round(x.to(tl.float32) * (127.0 / absmax[:, None])).to(tl.int8)
    # Streaming output stores reduce eviction of input data near L2 capacity.
    tl.store(
        xq_ptr + rows[:, None] * N + cols[None, :],
        xq,
        mask=mask,
        cache_modifier=".cs",
    )
    tl.store(scale_ptr + rows, scale, mask=rows < M)


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each last-dimension row with vLLM's symmetric INT8 semantics."""
    logger.debug("GEMS PER TOKEN QUANT INT8")
    shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, shape[-1])
    N = x.shape[-1]
    M = x.numel() // N
    xq = torch.empty(shape, device=x.device, dtype=torch.int8)
    scales = torch.empty(shape[:-1] + (1,), device=x.device, dtype=torch.float32)
    if M == 0:
        return xq, scales

    with torch_device_fn.device(x.device):
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _per_token_quant_int8_kernel[grid](
            x,
            xq,
            scales,
            M,
            N,
            x.stride(-2),
            x.stride(-1),
            triton.next_power_of_2(N),
        )
    return xq, scales
