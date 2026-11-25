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

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import triton

from flaggems_vllm.ops.flash_mla import _get_num_sms as get_num_sms
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE
from flaggems_vllm.utils.libentry import LibEntry

if HAS_TLE:
    from flaggems_vllm.runtime.backend._nvidia.sparse_mla_compact import (
        sparse_fp8_compact,
        sparse_fp8_tile,
    )
    from flaggems_vllm.runtime.backend._nvidia.sparse_mla_model1 import prepare_model1
    from flaggems_vllm.runtime.backend._nvidia.sparse_mla_reduce import (
        sparse_fp8_empty,
        sparse_fp8_fast_merge,
        sparse_fp8_repair,
        sparse_fp8_short,
    )
    from flaggems_vllm.runtime.backend._nvidia.sparse_mla_warp_specialized import (
        sparse_fp8_warp_specialized,
    )


SPARSE_LAUNCH_CACHE_LIMIT = 128


SPARSE_LAUNCH_CACHE: dict[tuple, tuple[Callable[..., None], tuple[int | bool, ...]]] = (
    {}
)


def launch_sparse_entry(
    entry: LibEntry,
    grid: tuple[int, ...] | Callable[[dict[str, object]], tuple[int, ...]],
    *arguments: torch.Tensor | int | float | bool | None,
    **launch_kwargs: int | bool,
) -> None:
    # Retain dynamic FlagTune hooks; ordinary entries tune once per exact ABI.
    if entry._has_flagtune_tuner:
        entry[grid](*arguments, **launch_kwargs)
        return
    signature = tuple(
        (
            (argument.dtype, argument.data_ptr() % entry.divisibility == 0)
            if isinstance(argument, torch.Tensor)
            else (type(argument), argument)
        )
        for argument in arguments
    )
    static_grid = None if callable(grid) else grid
    key = (
        entry,
        torch.cuda.current_device(),
        static_grid,
        signature,
        tuple(launch_kwargs.items()),
    )
    prepared = SPARSE_LAUNCH_CACHE.get(key)
    if prepared is None:
        kernel, constants = entry[grid](*arguments, **launch_kwargs)
        trailing = tuple(
            (
                launch_kwargs[param.name]
                if param.name in launch_kwargs
                else constants[param.name]
            )
            for param in entry.jit_function.params[len(arguments) :]
        )
        if callable(grid):
            resolved_grid = grid(
                {**dict(zip(entry.arg_names, arguments)), **launch_kwargs, **constants}
            )
        else:
            resolved_grid = grid
        while len(SPARSE_LAUNCH_CACHE) >= SPARSE_LAUNCH_CACHE_LIMIT:
            SPARSE_LAUNCH_CACHE.clear()
        # Retain launch metadata without holding input or output Tensor storage.
        SPARSE_LAUNCH_CACHE[key] = (kernel[resolved_grid], trailing)
        return
    runner, trailing = prepared
    runner(*arguments, *trailing)


def flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: Optional[torch.Tensor],
    q_scale: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    cache_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode using the separate per-token cache format of dense MLA.

    q_nope [B, 1, H, 512] and k_cache_lora [P, 64, 512] are FP8 e4m3fn;
    q_rope [B, 1, H, R] and k_cache_rope [P, 64, R] are BF16.
    R is 0 for inputs without RoPE, or 64 for the RoPE component.
    Both NoPE and RoPE store values divided by the corresponding FP32 scale:
    q_scale [B, 1, H, 1] and k_scale [P, 64, 1]. This directly accepts
    quantize_q_ckv_per_token / quantize_k_ckv_per_token outputs from dense MLA.
    V is the dequantized 512-dimensional NoPE cache. H must be 64 or 128.

    indices [B, 1, topk] contains int32 physical token IDs (page * 64 + slot).
    Negative and out-of-range IDs are ignored. topk_length [B] optionally
    limits the number of entries read per request. attn_sink [H] affects
    output only. Inputs must be finite, with positive finite scales.

    Native V4 takes 448 contiguous FP8 query channels, seven FP32 scales and
    64 unscaled BF16 RoPE channels. Pass the uint8 [P,64,1,584] page view
    as k_cache_lora with k_cache_rope and k_scale set to None. cache_tokens
    optionally excludes padding in the last page.

    Returns BF16 output [B, 1, H, 512] and natural-log FP32 LSE [B, H, 1].
    Empty attention produces zero output and +inf LSE. Forward only.
    Sensitive rows use FP32 recomputation inline, during merge, or in the
    existing separate repair kernel, depending on the selected path.
    Requires FlagTree cross-dtype WGMMA support (PR #1001).
    """
    if not HAS_TLE:
        raise NotImplementedError(
            "FP8 sparse MLA requires Hopper and FlagTree GPU extensions"
        )
    if (
        q_nope.device.type != "cuda"
        or torch.cuda.get_device_capability(q_nope.device)[0] != 9
    ):
        raise NotImplementedError("FP8 sparse MLA requires NVIDIA Hopper CUDA")
    if k_cache_lora.dtype == torch.uint8:
        if k_cache_rope is not None or k_scale is not None:
            raise ValueError("Native V4 cache pages already contain RoPE and scales")
        if attn_sink is not None or topk_length is not None:
            raise NotImplementedError("Native V4 uses the full supplied index row")
        if (
            q_nope.ndim != 4
            or q_nope.shape[1] != 1
            or q_rope.ndim != 4
            or q_scale.ndim != 4
        ):
            raise ValueError("Expected query components with singleton query length")
        if q_rope.shape[1] != 1 or q_scale.shape[1] != 1:
            raise ValueError("Expected singleton RoPE and scale query length")
        output, lse = prepare_model1(
            q_nope[:, 0],
            q_rope[:, 0],
            q_scale[:, 0],
            k_cache_lora,
            indices,
            tokens=k_cache_lora.shape[0] * 64 if cache_tokens is None else cache_tokens,
            scale=512**-0.5 if softmax_scale is None else float(softmax_scale),
        )()
        return output[:, None], lse[:, :, None]
    if cache_tokens is not None or k_cache_rope is None or k_scale is None:
        raise ValueError("Separate V32 pages require explicit RoPE and scales")
    if q_nope.dtype != torch.float8_e4m3fn or k_cache_lora.dtype != torch.float8_e4m3fn:
        raise TypeError("NoPE tensors must have dtype float8_e4m3fn")
    if q_rope.dtype != torch.bfloat16 or k_cache_rope.dtype != torch.bfloat16:
        raise TypeError("RoPE tensors must have dtype bfloat16")
    if q_nope.ndim != 4 or k_cache_lora.ndim != 3 or indices.ndim != 3:
        raise ValueError("q_nope, cache and indices must have ranks 4, 3 and 3")
    batch, query_length, heads, dim = q_nope.shape
    pages = k_cache_lora.shape[0]
    topk = indices.shape[-1]
    if query_length != 1 or heads not in (64, 128) or dim != 512:
        raise NotImplementedError(
            "Requires one query, 64/128 heads and 512 NoPE dimensions"
        )
    if (
        q_rope.ndim != 4
        or q_rope.shape[:3] != (batch, 1, heads)
        or q_rope.shape[-1] not in (0, 64)
    ):
        raise ValueError("q_rope must have shape [batch, 1, heads, 0 or 64]")
    rope_dim = q_rope.shape[-1]
    has_rope = rope_dim != 0
    if k_cache_lora.shape != (pages, 64, 512) or k_cache_rope.shape != (
        pages,
        64,
        rope_dim,
    ):
        raise ValueError(
            "Caches require page size 64 and the matching query RoPE dimension"
        )
    if indices.shape != (batch, 1, topk) or indices.dtype != torch.int32:
        raise ValueError("indices must be int32 [batch, 1, topk]")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("q_scale and k_scale must have dtype float32")
    if q_scale.shape != (batch, 1, heads, 1) or k_scale.shape != (pages, 64, 1):
        raise ValueError("Scale shapes must be [batch, 1, heads, 1] and [pages, 64, 1]")
    for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope):
        if tensor.stride(-1) != 1:
            raise ValueError("NoPE and RoPE must be contiguous in the last dimension")
    for tensor in (
        q_rope,
        k_cache_lora,
        k_cache_rope,
        indices,
        q_scale,
        k_scale,
        attn_sink,
        topk_length,
    ):
        if tensor is not None and tensor.device != q_nope.device:
            raise ValueError("All tensors must be on the same CUDA device")
    if attn_sink is not None and (
        attn_sink.shape != (heads,)
        or attn_sink.dtype != torch.float32
        or not attn_sink.is_contiguous()
    ):
        raise ValueError("attn_sink must be contiguous float32 [heads]")
    if topk_length is not None and (
        topk_length.shape != (batch,)
        or topk_length.dtype != torch.int32
        or not topk_length.is_contiguous()
    ):
        raise ValueError("topk_length must be contiguous int32 [batch]")
    output = torch.empty(
        (batch, 1, heads, 512), device=q_nope.device, dtype=torch.bfloat16
    )
    lse = torch.empty((batch, heads, 1), device=q_nope.device, dtype=torch.float32)
    if batch == 0:
        return output, lse
    if pages == 0 or topk == 0:
        rows = batch * heads
        sparse_fp8_empty[(triton.cdiv(rows * 512, 1024),)](
            output, lse, rows, num_warps=4
        )
        return output, lse
    softmax_scale = (
        (512 + rope_dim) ** -0.5 if softmax_scale is None else float(softmax_scale)
    )
    if batch <= 16 and not has_rope and topk <= 128:
        launch_sparse_entry(
            sparse_fp8_short,
            lambda meta: (
                batch,
                triton.cdiv(heads, meta["BLOCK_H"]),
                512 // meta["VALUE_DIM"],
            ),
            q_nope,
            k_cache_lora,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            output,
            lse,
            q_nope.stride(0),
            q_nope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            topk_length is not None,
            attn_sink is not None,
        )
        return output, lse
    # TLE does not yet lower the FP8-only compact shared-memory layout.
    use_partitioned = batch <= 16 and has_rope
    use_tile = False
    num_sms = get_num_sms(q_nope.device)
    head_groups = batch * (heads // 64)
    desired_splits = max(1, num_sms // head_groups)
    keys_per_split = 64 if batch <= 16 else 256
    max_splits = min(desired_splits, 32, max(1, topk // keys_per_split))
    splits = 1 << (max_splits.bit_length() - 1)
    if use_partitioned:
        can_async_copy = all(
            tensor.data_ptr() % 16 == 0
            and tensor.stride(0) * tensor.element_size() % 16 == 0
            and tensor.stride(-2) * tensor.element_size() % 16 == 0
            for tensor in (q_nope, q_rope, k_cache_lora, k_cache_rope)
        )
        tile_splits = triton.next_power_of_2(max(1, triton.cdiv(topk, 64)))
        # Keep value tiling in its validated workset; short rows use the compact CTA.
        use_tile = topk >= 512 and head_groups <= 8 and tile_splits <= 32
        if use_tile:
            splits = tile_splits
        else:
            # Checked merge consumes partial statistics, including for empty/short rows.
            splits = max(splits, 2)
    # Bound FP8 tensor-core accumulation; split outputs are merged in FP32.
    max_accumulation_keys = 4096
    splits = max(
        splits, triton.next_power_of_2(triton.cdiv(topk, max_accumulation_keys))
    )
    repair_splits = splits
    flag_heads = heads // 64
    repair_flags = (
        None
        if use_tile
        else torch.empty(
            (batch, flag_heads, splits), device=q_nope.device, dtype=torch.int32
        )
    )
    if splits > 1:
        partial = torch.empty(
            (batch, splits, heads, 512),
            device=q_nope.device,
            dtype=torch.bfloat16,
        )
        stats = torch.empty(
            (batch, splits, heads, 2), device=q_nope.device, dtype=torch.float32
        )
    else:
        partial, stats = output, lse
    if use_tile:
        launch_sparse_entry(
            sparse_fp8_tile,
            lambda meta: (batch, heads // 64, splits * (512 // meta["BLOCK_D"])),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CAN_ASYNC=can_async_copy,
            HAS_ROPE=has_rope,
        )
    elif use_partitioned:
        launch_sparse_entry(
            sparse_fp8_compact,
            (batch, heads // 64, splits),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            repair_flags,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CAN_ASYNC=can_async_copy,
            HAS_ROPE=has_rope,
        )
    else:
        b_bucket = 0 if head_groups * splits <= 128 else 1
        launch_sparse_entry(
            sparse_fp8_warp_specialized,
            (batch * (heads // 64), 1, splits),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            b_bucket,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            topk_length is not None,
            splits,
            attn_sink is not None,
            repair_flags,
            HAS_ROPE=has_rope,
            CAN_ASYNC_ROPE=(
                has_rope
                and k_cache_rope.data_ptr() % 16 == 0
                and (k_cache_rope.stride(0) * 2) % 16 == 0
                and (k_cache_rope.stride(1) * 2) % 16 == 0
            ),
            CAN_ASYNC_NOPE=(
                k_cache_lora.data_ptr() % 16 == 0
                and k_cache_lora.stride(0) % 16 == 0
                and k_cache_lora.stride(1) % 16 == 0
            ),
        )
    if not use_partitioned and splits == 1:
        launch_sparse_entry(
            sparse_fp8_repair,
            lambda meta: (
                triton.cdiv(batch, meta["REPAIR_GROUP"]),
                triton.cdiv(heads, meta["BLOCK_H"]),
                splits,
            ),
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            indices,
            q_scale,
            k_scale,
            attn_sink,
            topk_length,
            output,
            lse,
            partial,
            stats,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            indices.stride(0),
            indices.stride(2),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            batch,
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            attn_sink is not None,
            topk_length is not None,
            repair_flags,
            REPAIR_SPLITS=repair_splits,
            HAS_ROPE=has_rope,
        )
    if splits > 1:
        head_tile = 2 if splits == 2 else 4
        sparse_fp8_fast_merge[(batch, heads // head_tile)](
            q_nope,
            q_rope,
            k_cache_lora,
            k_cache_rope,
            q_scale,
            k_scale,
            indices,
            indices if topk_length is None else topk_length,
            q_scale if attn_sink is None else attn_sink,
            stats if repair_flags is None else repair_flags,
            partial,
            stats,
            output,
            lse,
            q_nope.stride(0),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(2),
            k_cache_lora.stride(0),
            k_cache_lora.stride(1),
            k_cache_rope.stride(0),
            k_cache_rope.stride(1),
            q_scale.stride(0),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            indices.stride(0),
            indices.stride(2),
            heads,
            pages * 64,
            topk,
            softmax_scale,
            splits,
            topk_length is not None,
            attn_sink is not None,
            CHECK_REPAIR=not use_tile,
            HEAD_TILE=head_tile,
            HAS_ROPE=has_rope,
            num_warps=2,
        )
    return output, lse
