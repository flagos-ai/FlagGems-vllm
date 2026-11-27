# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

_INDEX_DTYPES = (torch.int32, torch.int64)
_WEIGHT_DTYPES = (torch.float16, torch.bfloat16)
_FLOAT_DTYPES = (*_WEIGHT_DTYPES, torch.float32)
_GROUPED_DOT_VENDORS = {"nvidia", "metax", "hygon"}


@triton.jit
def _expand_kernel(
    X,
    B0,
    B1,
    B2,
    OUT,
    SORTED,
    COUNTS,
    STARTS,
    IDS,
    offset,
    tile_start,
    T: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    ACTIVE: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    NSLICES: tl.constexpr,
    XS: tl.constexpr,
    XT: tl.constexpr,
    XR: tl.constexpr,
    YT: tl.constexpr,
    ADD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tile_start + tl.program_id(0)
    slice_id = tl.program_id(1)
    group = 0 if ACTIVE == 1 else tl.program_id(2)
    tiles_m = tl.cdiv(T, BLOCK_M)
    first = (tile % tiles_m) * BLOCK_M
    first_col = (tile // tiles_m) * BLOCK_N
    adapter = tl.load(IDS + group)
    count = tl.load(COUNTS + group)
    if (adapter < 0) | (adapter >= L) | (first >= count):
        return

    # Direct weight arguments keep the current pointers and offset in the launch.
    # Common QKV/merged-column projections fit in a single launch.
    weight = B0
    width = W0
    if NSLICES >= 2:
        offset = offset + tl.where(slice_id >= 1, W0, 0)
        if slice_id == 1:
            weight = B1
            width = W1
    if NSLICES == 3:
        offset = offset + tl.where(slice_id == 2, W1, 0)
        if slice_id == 2:
            weight = B2
            width = W2
    if first_col >= width:
        return

    start = tl.load(STARTS + group).to(tl.int64)
    cols = first_col + tl.arange(0, BLOCK_N)
    weight += (adapter.to(tl.int64) * width + cols[:, None]) * R
    X += slice_id.to(tl.int64) * XS
    if BLOCK_M == 1:
        if (start < 0) | (start > T - 1 - first):
            return
        row = tl.load(SORTED + start + first).to(tl.int64)
        if (row < 0) | (row >= T):
            return
        x = X + row * XT
        acc = tl.full((BLOCK_N,), 0, tl.float32)
        for k in range(tl.cdiv(R, BLOCK_K)):
            ranks = k * BLOCK_K + tl.arange(0, BLOCK_K)
            vec = tl.load(x + ranks * XR, ranks < R, other=0).to(B0.dtype.element_ty)
            w = tl.load(
                weight + ranks[None, :],
                (cols[:, None] < width) & (ranks[None, :] < R),
                other=0,
            )
            acc += tl.sum(w.to(tl.float32) * vec[None, :].to(tl.float32), axis=1)
        dst = OUT + row * YT + offset + cols
        mask = cols < width
    else:
        positions = first + tl.arange(0, BLOCK_M)
        live = (positions < count) & (start >= 0) & (start <= T - 1 - positions)
        rows = tl.load(SORTED + start + positions, live, other=-1).to(tl.int64)
        live = live & (rows >= 0) & (rows < T)
        x = X + rows[:, None] * XT
        acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
        for k in range(tl.cdiv(R, BLOCK_K)):
            ranks = k * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(
                x + ranks[None, :] * XR,
                live[:, None] & (ranks[None, :] < R),
                other=0,
            ).to(B0.dtype.element_ty)
            w = tl.load(
                weight + ranks[None, :],
                (cols[:, None] < width) & (ranks[None, :] < R),
                other=0,
            )
            acc += tl.dot(a, tl.trans(w))
        dst = OUT + rows[:, None] * YT + offset + cols[None, :]
        mask = live[:, None] & (cols[None, :] < width)

    # vLLM rounds the projection to weight precision before accumulating into OUT.
    value = acc.to(B0.dtype.element_ty)
    if ADD:
        old = tl.load(dst, mask, other=0)
        value = value + old
    tl.store(dst, value, mask)


class _WarmLaunch:
    """Reuse the compiled entry while passing the caller's current tensors and stream."""

    def __init__(self, entry):
        self.entry = entry
        self.cache = {}

    def __getitem__(self, grid):
        def run(*args):
            # LibEntry's specialization includes pointer alignment and backend
            # attributes. No tensor, storage address or metadata value is retained.
            key = (args[0].device, self.entry.key(args[:11], (), []), args[11:])
            cached = self.cache.get(key)
            if cached is None:
                compiled, meta = self.entry[grid](*args)
                if len(self.cache) >= 256:
                    self.cache.clear()
                self.cache[key] = compiled, meta
            else:
                compiled, meta = cached
                compiled[grid(meta)](*args, meta["BLOCK_M"], meta["BLOCK_N"])

        return run


def _tuned_kernel(name):
    configs = runtime.get_tuned_config(name)
    entry = libentry()(
        libtuner(
            configs=configs,
            key=[
                "T",
                "R",
                "L",
                "ACTIVE",
                "W0",
                "W1",
                "W2",
                "NSLICES",
                "offset",
                "XS",
                "XT",
                "XR",
                "YT",
                "ADD",
            ],
            restore_value=["OUT"],
            use_cuda_graph=False,
        )(_expand_kernel)
    )
    # The 3.3–3.9 launch ABI accepts constexprs in declaration order.
    # Multiple configurations and launch hooks require the full dispatcher.
    version = tuple(int(part) for part in triton.__version__.split(".")[:2])
    if (
        runtime.device.vendor_name in {"nvidia", "thead", "metax", "hygon"}
        and (3, 3) <= version <= (3, 9)
        and len(configs) == 1
        and configs[0].pre_hook is None
        and not getattr(entry.fn, "shared_config_pre_hook", None)
        and not entry._has_flagtune_tuner
    ):
        return _WarmLaunch(entry)
    return entry


_gemv = _tuned_kernel("lora_expand_gemv")
_grouped = _tuned_kernel("lora_expand_grouped")
if runtime.device.vendor_name in _GROUPED_DOT_VENDORS | {"ascend"}:
    _wide_gemv = _tuned_kernel("lora_expand_wide_gemv")


def _ascend_launches(tiles, active):
    # CANN bounds the product of the grid dimensions, including empty groups.
    # Ascend dispatches one slice at a time and each program reduces the full rank.
    limit = 65535
    for group in range(0, active, limit):
        groups = min(limit, active - group)
        tile_step = limit // groups
        for tile in range(0, tiles, tile_step):
            yield (min(tile_step, tiles - tile), 1, groups), tile, group


def _cpu_control(tensor, name, dtypes):
    if tensor.device.type != "cpu" or tensor.numel() != 1:
        raise ValueError(f"{name} must contain one CPU value")
    if tensor.dtype not in dtypes:
        raise TypeError(f"invalid dtype for {name}")
    return tensor.item()


def _check_alias(output, sources):
    # Only shared storage needs a byte-span check. Disjoint slices can coexist.
    for source in sources:
        if torch._C._overlaps(output, source) and output.numel() and source.numel():
            begin = source.data_ptr()
            span = 1 + sum((n - 1) * s for n, s in zip(source.shape, source.stride()))
            out = output.data_ptr()
            if (
                begin < out + output.nbytes
                and out < begin + span * source.element_size()
            ):
                raise ValueError("output_tensor overlaps an input")


def _validate(inputs, weights, output, metadata, active, offset):
    if inputs.ndim != 3 or not weights or len(weights) != inputs.size(0):
        raise ValueError("expected inputs [S,T,R] and one nonempty list of S weights")
    _, tokens, rank = inputs.shape
    if output.ndim != 2 or output.size(0) != tokens or not output.is_contiguous():
        raise ValueError("output_tensor must be contiguous with shape [T,Hout]")
    if inputs.layout != torch.strided:
        raise ValueError("inputs must be strided")
    device = inputs.device
    if device.type != runtime.device.name or device.type == "cpu":
        raise NotImplementedError("lora_expand requires an accelerator")
    if any(t.device != device for t in (output, *weights, *metadata)):
        raise ValueError(
            "inputs, weights, output and grouped metadata must share a device"
        )
    dtype = weights[0].dtype
    if (
        dtype not in _WEIGHT_DTYPES
        or inputs.dtype not in (dtype, torch.float32)
        or output.dtype not in _FLOAT_DTYPES
    ):
        raise TypeError(
            "expected FP16/BF16 weights, matching or FP32 inputs, and floating output"
        )
    adapters = weights[0].size(0)
    for w in weights:
        if w.ndim not in (3, 4) or (w.ndim == 4 and w.size(1) != 1):
            raise ValueError("expected weights [L,H,R] or [L,1,H,R]")
        if w.size(0) != adapters or w.size(-1) != rank or not w.is_contiguous():
            raise ValueError(
                "weights must be contiguous with matching adapter and rank dimensions"
            )
        if w.dtype != dtype:
            raise TypeError("weight slices must have the same dtype")
    if (
        not isinstance(offset, int)
        or offset < 0
        or offset + sum(w.size(-2) for w in weights) > output.size(1)
    ):
        raise ValueError(
            "the expanded slices must fit in output_tensor at offset_start"
        )

    groups = metadata[-1].numel()
    if not 0 <= active <= groups:
        raise ValueError("num_active_loras exceeds grouped metadata capacity")
    for tensor, length in zip(metadata, (tokens, tokens, groups, groups + 1, groups)):
        if tensor.shape != (length,) or not tensor.is_contiguous():
            raise ValueError("invalid grouped metadata shape or layout")
        if tensor.dtype not in _INDEX_DTYPES:
            raise TypeError("grouped metadata must use int32 or int64")
    _check_alias(output, (inputs, *weights, *metadata))


def lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    num_active_loras: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """Project grouped token rows through LoRA B into consecutive output slices.

    Each active metadata group supplies distinct token rows and one adapter.
    Both overwrite and add preserve unassigned rows and columns outside the
    slices. A true no-LoRA flag leaves the entire output untouched.
    """
    logger.debug("GEMS LORA_EXPAND")
    if _cpu_control(no_lora_flag_cpu, "no_lora_flag_cpu", (torch.bool,)):
        return
    active = _cpu_control(num_active_loras, "num_active_loras", _INDEX_DTYPES)
    grouped = (
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
    )
    _validate(
        inputs,
        lora_b_weights,
        output_tensor,
        (token_lora_mapping, *grouped),
        active,
        offset_start,
    )
    slices, tokens, rank = inputs.shape
    adapters = lora_b_weights[0].size(0)
    if not (tokens and active and adapters and output_tensor.numel()):
        return

    vendor = runtime.device.vendor_name
    matrix = (
        (vendor in _GROUPED_DOT_VENDORS and tokens > 8)
        or (vendor == "thead" and tokens > 8 and rank >= 16)
        or (vendor == "mthreads" and tokens >= 32)
    )
    rank_tile = max(
        16 if matrix else 1,
        min(32 if matrix else 64, triton.next_power_of_2(max(rank, 1))),
    )
    # Group up to three direct weights when the backend supports pointer selection.
    batch = 3 if vendor in {"nvidia", "metax", "hygon", "thead", "mthreads"} else 1
    with runtime.torch_device_fn.device(inputs.device):
        for first in range(0, slices, batch):
            weights = lora_b_weights[first : first + batch]
            n = len(weights)
            widths = [w.size(-2) for w in weights]
            width = max(widths)
            if width:
                kernel = _grouped if matrix else _gemv
                if (
                    vendor in _GROUPED_DOT_VENDORS
                    and 1 < tokens <= 8
                    and rank >= 16
                    and width >= 1024
                ) or (
                    vendor == "ascend" and tokens >= 128 and rank <= 16 and width >= 512
                ):
                    kernel = _wide_gemv

                def grid(meta):
                    return (
                        triton.cdiv(tokens, meta["BLOCK_M"])
                        * triton.cdiv(width, meta["BLOCK_N"]),
                        n,
                        active,
                    )

                launches = ((grid, 0, 0),)
                if vendor == "ascend":
                    tiles = grid(kernel.fn.configs[0].kwargs)[0]
                    launches = _ascend_launches(tiles, active)
                for launch, tile, group in launches:
                    sorted_rows, counts, starts, ids = grouped
                    if group:
                        counts, starts, ids = (
                            counts[group:],
                            starts[group:],
                            ids[group:],
                        )
                    kernel[launch](
                        inputs if first == 0 else inputs[first:],
                        weights[0],
                        weights[min(1, n - 1)],
                        weights[-1],
                        output_tensor,
                        sorted_rows,
                        counts,
                        starts,
                        ids,
                        offset_start,
                        tile,
                        tokens,
                        rank,
                        adapters,
                        active,
                        *(widths + [0] * (3 - n)),
                        n,
                        *inputs.stride(),
                        output_tensor.stride(0),
                        add_inputs,
                        rank_tile,
                    )
            offset_start += sum(widths)


__all__ = ["lora_expand"]
