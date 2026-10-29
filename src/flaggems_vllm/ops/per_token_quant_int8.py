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


def _get_quant_configs():
    configs = runtime.get_tuned_config("per_token_quant_int8")
    if configs:
        return configs
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "VEC_SIZE": vec_size,
                "ROW_TILES": row_tiles,
                "CONTIGUOUS_ROWS": contiguous_rows,
                "REDUCE_GROUP": reduce_group,
                "QUANT_PARTS": 1,
            },
            num_warps=warps,
            num_stages=1,
        )
        for block_m in (1, 2)
        for row_tiles in (1, 2)
        for contiguous_rows in (False, True)
        for reduce_group in (0, 4, 8, 16)
        for vec_size in (4, 8, 16)
        for warps in (1, 2, 4, 8, 16)
    ] + [
        triton.Config(
            {
                "BLOCK_M": 1,
                "VEC_SIZE": 8,
                "ROW_TILES": 1,
                "CONTIGUOUS_ROWS": False,
                "REDUCE_GROUP": 8,
                "QUANT_PARTS": parts,
            },
            num_warps=warps,
            num_stages=1,
        )
        for parts in (2, 4, 8, 16)
        for warps in (4, 8, 16)
    ]


def _prune_quant_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    block_n = named_args["BLOCK_N"]
    warp_size = triton.runtime.driver.active.get_current_target().warp_size
    # Bound CTA size on both 32-lane and 64-lane targets.
    max_warps = min(max(config.num_warps for config in configs), 512 // warp_size)
    pruned = []
    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        reduce_group = config.kwargs["REDUCE_GROUP"]
        quant_parts = config.kwargs["QUANT_PARTS"]
        if config.num_warps > max_warps:
            continue
        # Duplicate the reduction only while extra CTAs can hide row latency.
        if quant_parts > 1 and (
            M > 64
            or M * quant_parts > 256
            or block_n < 2048
            or block_n > 8192
            or named_args["x_ptr"].dtype != torch.bfloat16
            or named_args["stride_xn"] != 1
        ):
            continue
        if reduce_group and (
            block_n <= 128
            or named_args["x_ptr"].dtype != torch.bfloat16
            or named_args["stride_xn"] != 1
            or reduce_group > config.kwargs["VEC_SIZE"]
            or config.kwargs["ROW_TILES"] > 1
        ):
            continue
        if config.kwargs["CONTIGUOUS_ROWS"] and config.kwargs["ROW_TILES"] == 1:
            continue
        # Reusing a CTA only helps once enough wide rows fill the GPU.
        if config.kwargs["ROW_TILES"] > 1 and (
            M < 1024 or block_n < 2048 or block_m > 1
        ):
            continue
        # A single warp already covers short rows without extra vector variants.
        if block_n <= 128 and config.kwargs["VEC_SIZE"] != 16:
            continue
        # Packing wide rows raises register pressure and reduces parallelism.
        if block_m > 1 and (block_m > M or block_n > 2048):
            continue
        # Short rows fit within a warp and do not benefit from CTA-wide reduction.
        if block_n <= 128 and config.num_warps > block_m:
            continue
        tile_size = block_m * block_n
        if config.num_warps * warp_size > max(tile_size, warp_size):
            continue
        # Avoid more than 256 values per thread, keeping the widest candidate.
        if (
            tile_size > 256 * warp_size * config.num_warps
            and config.num_warps < max_warps
        ):
            continue
        pruned.append(config)
    return pruned


@libentry()
@libtuner(
    configs=_get_quant_configs(),
    key=["M", "N", "stride_xm", "stride_xn"],
    prune_configs_by={"early_config_prune": _prune_quant_configs},
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
    ROW_TILES: tl.constexpr,
    CONTIGUOUS_ROWS: tl.constexpr,
    REDUCE_GROUP: tl.constexpr,
    QUANT_PARTS: tl.constexpr = 1,
):
    # Row tiles reduce CTA scheduling and improve locality near L2 capacity.
    for row_tile in range(ROW_TILES):
        if CONTIGUOUS_ROWS:
            row_block = tl.program_id(0) * ROW_TILES + row_tile
        else:
            row_block = tl.program_id(0) + row_tile * tl.num_programs(0)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
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
        if REDUCE_GROUP:
            # Reduce locally before promoting to FP32, then compare positive bits.
            local_max = tl.max(
                tl.reshape(tl.abs(x), (BLOCK_M, BLOCK_N // REDUCE_GROUP, REDUCE_GROUP)),
                axis=2,
            ).to(tl.float32)
            bits = local_max.to(tl.uint32, bitcast=True)
            # Float max ignores NaNs; zero is neutral before the scale clamp.
            bits = tl.where(bits > 0x7F800000, 0, bits)
            absmax = tl.max(bits, axis=1).to(tl.float32, bitcast=True)
        else:
            absmax = tl.max(tl.abs(x), axis=1).to(tl.float32)
        absmax = tl.maximum(absmax, 1e-10)
        # Match the reference scalar division on each row. Tensor/scalar
        # division can become an approximate reciprocal on some backends.
        if BLOCK_M == 1:
            scale = (tl.sum(absmax, axis=0) / 127.0)[None]
        else:
            absmax0, absmax1 = tl.split(absmax)
            scale = tl.join(absmax0 / 127.0, absmax1 / 127.0)
        if QUANT_PARTS > 1:
            # Each CTA computes the full maximum, then quantizes its own columns.
            # Reloading a smaller tile shortens the dependent round/store path.
            cols = tl.program_id(1) * (BLOCK_N // QUANT_PARTS) + tl.max_contiguous(
                tl.arange(0, BLOCK_N // QUANT_PARTS), VEC_SIZE
            )
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            x = tl.load(
                x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn,
                mask=mask,
                other=0.0,
            )
        # vLLM rounds halfway cases away from zero, unlike nearbyint / torch.round.
        xq = libdevice.round(x.to(tl.float32) * (127.0 / absmax[:, None])).to(tl.int8)
        # Streaming output stores reduce eviction of input data near L2 capacity.
        tl.store(
            xq_ptr + rows[:, None] * N + cols[None, :],
            xq,
            mask=mask,
            cache_modifier=".cs",
        )
        if QUANT_PARTS == 1:
            tl.store(scale_ptr + rows, scale, mask=rows < M)
        elif tl.program_id(1) == 0:
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

    with torch_device_fn.device(x.device):
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"] * meta["ROW_TILES"]),
            meta["QUANT_PARTS"],
        )
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
