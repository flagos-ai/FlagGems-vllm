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

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import fill_tensor
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.shape_utils import can_use_int32_index

MEDIUM_BATCH_MIN_ROWS = 96
MEDIUM_BATCH_MAX_ROWS = 384
LARGE_BATCH_SCALE_ROWS = tl.constexpr(8192)
EARLY_DOT_MIN_ELEMENTS = tl.constexpr(4096)
SPLIT_REDUCTION_BLOCK = 1024
NON_POWER_SPLIT_ROW_RANGES = ((192, 256), (384, 512))
NON_POWER_SPLIT_TILE = 128
NON_POWER_SPLIT_CTA_BUDGET = 96


def bmm_tuning_key(value):
    # The persistent SQL cache supports scalar keys, not stride tuples.
    return str(value) if isinstance(value, tuple) else value


def prune_wide_scale_configs(configs, named_args, **kwargs):
    if all(can_use_int32_index(named_args[name]) for name in ("AS", "WS")):
        return configs
    # Wide scale arithmetic can make large tiles exceed C550's default
    # private-memory limit. Keep the existing conservative configuration.
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_M"] == 16
        and config.kwargs["BLOCK_N"] == 64
        and config.num_warps == 4
        and config.num_stages == 1
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_bmm"),
    key=[
        "M",
        "N",
        "K",
        "BATCH",
        "SPLIT_K",
        "SCALE_N",
        "stride_am",
        "stride_ak",
        "stride_wn",
        "stride_wk",
        "USE_INT64",
        "SAS",
        "SWS",
    ],
    strategy=bmm_tuning_key,
    prune_configs_by={"early_config_prune": prune_wide_scale_configs},
)
@triton.jit
def block_int8_bmm_kernel(
    A,
    W,
    AS,
    WS,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    SCALE_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BATCH: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    USE_INT64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if BATCH == 1:
        # Eliminate dynamic head offsets without narrowing partial-buffer indices.
        batch = tl.full((), 0, tl.int64)
    else:
        batch = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2)
    row_tile = tl.program_id(0) // tl.cdiv(N, BLOCK_N)
    column_tile = tl.program_id(0) % tl.cdiv(N, BLOCK_N)
    if USE_INT64:
        row_tile = row_tile.to(tl.int64)
        column_tile = column_tile.to(tl.int64)
    row = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    col = column_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    red = tl.arange(0, 128)
    if USE_INT64:
        red = red.to(tl.int64)
    scale_column = column_tile * BLOCK_N // SCALE_N
    is_early_dot: tl.constexpr = (
        M < LARGE_BATCH_SCALE_ROWS and BLOCK_M * BLOCK_N >= EARLY_DOT_MIN_ELEMENTS
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for group in range(split, tl.cdiv(K, 128), SPLIT_K):
        # Widen before multiplication: scale strides can exceed the i32 span
        # even when M, N and K themselves are small.
        scale_group = group.to(tl.int64) if USE_INT64 else group
        offsets_k = scale_group * 128 + red
        activation = tl.load(
            A
            + batch * SA[0]
            + row[:, None] * stride_am
            + offsets_k[None, :] * stride_ak,
            (row[:, None] < M) & (offsets_k[None, :] < K),
            other=0,
        )
        weight = tl.load(
            W
            + batch * SW[0]
            + col[None, :] * stride_wn
            + offsets_k[:, None] * stride_wk,
            (col[None, :] < N) & (offsets_k[:, None] < K),
            other=0,
        )
        # Scalar-scale tiles benefit from converting the dot result first.
        if is_early_dot:
            partial = tl.dot(activation, weight, out_dtype=tl.int32).to(tl.float32)
        else:
            pass
        if M >= LARGE_BATCH_SCALE_ROWS:
            # Rank-two scale loads improve MetaX lowering for large row counts.
            # Only lane zero is loaded; the masked lane is reduced away.
            scale_lane = tl.arange(0, 2)
            activation_scale = tl.sum(
                tl.load(
                    AS
                    + batch * SAS[0]
                    + row[:, None] * SAS[1]
                    + (scale_group + scale_lane[None, :]) * SAS[2],
                    (row[:, None] < M)
                    & (scale_lane[None, :] == 0)
                    & (group < tl.cdiv(K, 128)),
                    other=0,
                ).to(tl.float32),
                1,
            )
        else:
            activation_scale = tl.load(
                AS + batch * SAS[0] + row * SAS[1] + scale_group * SAS[2],
                row < M,
                other=0,
            ).to(tl.float32)
        if SCALE_N % BLOCK_N == 0:
            if M >= LARGE_BATCH_SCALE_ROWS:
                weight_scale = tl.sum(
                    tl.sum(
                        tl.load(
                            WS
                            + batch * SWS[0]
                            + scale_column * SWS[1]
                            + (scale_group + scale_lane[None, :]) * SWS[2]
                            + tl.arange(0, 1)[:, None],
                            (scale_lane[None, :] == 0) & (group < tl.cdiv(K, 128)),
                            other=0,
                        ).to(tl.float32),
                        1,
                    ),
                    0,
                )
            else:
                weight_scale = tl.load(
                    WS + batch * SWS[0] + scale_column * SWS[1] + scale_group * SWS[2]
                ).to(tl.float32)
            scale = (activation_scale * weight_scale)[:, None]
        else:
            weight_scale = tl.load(
                WS + batch * SWS[0] + (col // SCALE_N) * SWS[1] + scale_group * SWS[2],
                col < N,
                other=0,
            ).to(tl.float32)
            scale = activation_scale[:, None] * weight_scale[None, :]
        if is_early_dot:
            pass
        else:
            partial = tl.dot(activation, weight, out_dtype=tl.int32).to(tl.float32)
        accumulator = tl.fma(partial, scale, accumulator)
    if SPLIT_K == 1:
        pointers = Output + batch * SO[0] + row[:, None] * SO[1] + col[None, :] * SO[2]
    else:
        pointers = (
            Output + ((split * BATCH + batch) * M + row[:, None]) * N + col[None, :]
        )
    tl.store(pointers, accumulator, (row[:, None] < M) & (col[None, :] < N))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("w8a8_block_int8_gemv"),
    key=["N", "K", "BATCH", "SCALE_N"],
)
@triton.jit
def block_int8_gemv_kernel(
    A,
    W,
    AS,
    WS,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    SCALE_N: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    head = tl.program_id(2).to(tl.int64)
    row = tl.program_id(1).to(tl.int64)
    col = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    groups = tl.arange(0, BLOCK_K // 128)
    red = groups[:, None] * 128 + tl.arange(0, 128)[None, :]
    if USE_INT64:
        col = col.to(tl.int64)
        groups = groups.to(tl.int64)
        red = red.to(tl.int64)
    else:
        col = col.to(tl.int32)
    a = tl.load(A + head * SA[0] + row * SA[1] + red * SA[2], red < K, other=0).to(
        tl.int32
    )
    w = tl.load(
        W + head * SW[0] + col[:, None, None] * SW[1] + red[None, :, :] * SW[2],
        (col[:, None, None] < N) & (red[None, :, :] < K),
        other=0,
    ).to(tl.int32)
    asc = tl.load(
        AS + head * SAS[0] + row * SAS[1] + groups * SAS[2],
        groups < tl.cdiv(K, 128),
        other=0,
    ).to(tl.float32)
    wsc = tl.load(
        WS
        + head * SWS[0]
        + (col[:, None] // SCALE_N) * SWS[1]
        + groups[None, :] * SWS[2],
        (col[:, None] < N) & (groups[None, :] < tl.cdiv(K, 128)),
        other=0,
    ).to(tl.float32)
    partial = tl.sum(a[None, :, :] * w, 2).to(tl.float32)
    output = tl.sum(partial * (asc[None, :] * wsc), 1)
    tl.store(Output + head * SO[0] + row * SO[1] + col * SO[2], output, col < N)


@libentry()
@triton.jit
def reduce_split_kernel(
    Partial,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH: tl.constexpr,
    SO: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, triton.next_power_of_2(SPLIT_K)).to(tl.int64)
    elements: tl.constexpr = BATCH * M * N
    partials = tl.load(
        Partial + splits[:, None] * elements + offsets[None, :],
        (splits[:, None] < SPLIT_K) & (offsets[None, :] < elements),
        other=0,
    )
    if (
        SO[2] == 1
        and SO[1] == N
        and (BATCH == 1 or SO[0] == M * N)
        and (elements & (elements - 1)) != 0
    ):
        # Avoid non-power-of-two address division for contiguous outputs.
        pointers = Output + offsets
    else:
        batch = offsets // (M * N)
        row = offsets // N % M
        col = offsets % N
        pointers = Output + batch * SO[0] + row * SO[1] + col * SO[2]
    tl.store(
        pointers,
        tl.sum(partials, 0),
        offsets < elements,
    )


def w8a8_block_int8_bmm(
    x: torch.Tensor,
    y: torch.Tensor,
    xs: torch.Tensor | None,
    ys: torch.Tensor | None,
    block_size: tuple[int, int] = (128, 128),
    z: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-scaled INT8 BMM with FP32 accumulation on MetaX."""
    if x.requires_grad or y.requires_grad or (z is not None and z.requires_grad):
        raise NotImplementedError("W8A8 BMM is forward-only")
    if x.dtype != y.dtype:
        raise TypeError("W8A8 BMM inputs must have matching dtypes")
    if x.dtype != torch.int8:
        raise TypeError("MetaX W8A8 BMM requires INT8 inputs")
    if xs is None or ys is None:
        raise ValueError("INT8 W8A8 BMM requires both scale tensors")
    tensors = (x, y, xs, ys)
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("inputs and scales must have three dimensions")
    if x.device.type != "cuda" or any(t.device != x.device for t in tensors):
        raise ValueError("all inputs and scales must be on the same MetaX GPU")
    if any(t.requires_grad for t in tensors):
        raise NotImplementedError("INT8 BMM is forward-only")
    if xs.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ) or ys.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("scales must be float32, bfloat16 or float16")
    if len(block_size) != 2 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain two positive integers")
    block_n, block_k = block_size
    if block_k != 128 or block_n < 16 or block_n & (block_n - 1):
        raise NotImplementedError("requires power-of-two block_n >= 16 and block_k=128")
    batch, rows, reduction = x.shape
    columns = y.shape[1]
    if y.shape != (batch, columns, reduction):
        raise ValueError("W8A8 BMM input shape mismatch")
    if xs.shape != (batch, rows, triton.cdiv(reduction, block_k)) or ys.shape != (
        batch,
        triton.cdiv(columns, block_n),
        triton.cdiv(reduction, block_k),
    ):
        raise ValueError("incorrect W8A8 BMM scale shape")
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("unsupported output dtype")
    if z is None:
        z = torch.empty((batch, rows, columns), device=x.device, dtype=output_dtype)
    elif (
        z.shape != (batch, rows, columns)
        or z.device != x.device
        or z.dtype != output_dtype
    ):
        raise ValueError("z must have matching shape, device and dtype")
    else:
        # Both ordinary BMM and the einsum's interleaved head view are supported.
        axes = sorted(zip(z.stride(), z.shape))
        span = 1
        for stride, size in axes:
            if size > 1 and stride < span:
                raise ValueError("z must have non-overlapping strides")
            span += (size - 1) * stride
        if z.numel() and any(
            z.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in tensors
        ):
            raise ValueError("z must not share storage with an input")
    if z.numel() == 0:
        return z
    with torch_device_fn.device(x.device):
        if reduction == 0:
            return fill_tensor(z, 0.0)
        use_int64 = not all(can_use_int32_index(t) for t in (*tensors, z))
        if rows == 1 and 2048 <= reduction <= 8192:
            grid = lambda meta: (triton.cdiv(columns, meta["BLOCK_N"]), rows, batch)
            block_int8_gemv_kernel[grid](
                x,
                y,
                xs,
                ys,
                z,
                rows,
                columns,
                reduction,
                x.stride(),
                y.stride(),
                xs.stride(),
                ys.stride(),
                z.stride(),
                block_n,
                batch,
                BLOCK_K=triton.next_power_of_2(reduction),
                USE_INT64=use_int64,
            )
            return z
        # Small output grids need additional independent CTAs to cover the C550.
        # The larger reduction block permits more split-K for medium batches.
        planning_rows = (
            128 if MEDIUM_BATCH_MIN_ROWS < rows <= MEDIUM_BATCH_MAX_ROWS else 32
        )
        tiles = batch * triton.cdiv(rows, planning_rows) * triton.cdiv(columns, 64)
        if reduction >= 512 and any(
            lower < rows <= upper for lower, upper in NON_POWER_SPLIT_ROW_RANGES
        ):
            # These row ranges benefit from splits between powers of two.
            split_tiles = (
                batch
                * triton.cdiv(rows, NON_POWER_SPLIT_TILE)
                * triton.cdiv(columns, NON_POWER_SPLIT_TILE)
            )
            split_k = max(1, min(8, NON_POWER_SPLIT_CTA_BUDGET // split_tiles))
        else:
            split_k = (
                min(8, triton.next_power_of_2(triton.cdiv(128, tiles)))
                if reduction >= 512
                else 1
            )
        partials = (
            torch.empty(
                (split_k, batch, rows, columns), device=x.device, dtype=torch.float32
            )
            if split_k > 1
            else z
        )
        grid = lambda meta: (
            triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),
            batch,
            split_k,
        )
        block_int8_bmm_kernel[grid](
            x,
            y,
            xs,
            ys,
            partials,
            rows,
            columns,
            reduction,
            x.stride(),
            y.stride(),
            xs.stride(),
            ys.stride(),
            z.stride(),
            block_n,
            split_k,
            batch,
            x.stride(1),
            x.stride(2),
            y.stride(1),
            y.stride(2),
            USE_INT64=use_int64,
        )
        if split_k == 1:
            return z
        reduce_split_kernel[(triton.cdiv(z.numel(), SPLIT_REDUCTION_BLOCK),)](
            partials,
            z,
            rows,
            columns,
            batch,
            z.stride(),
            split_k,
            BLOCK=SPLIT_REDUCTION_BLOCK,
        )
    return z
