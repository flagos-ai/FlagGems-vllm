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

"""Projection launches and direct/grouped execution."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    _PPU_DENSE_MIN_TOKENS,
    tle_async,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.gemm import (
    marlin_moe_gemm_direct_kernel,
    marlin_moe_gemm_grouped_kernel,
    marlin_moe_gemm_reduce_direct_kernel,
    marlin_moe_gemm_silu_direct_kernel,
    marlin_moe_matvec_direct_kernel,
    reduce_split_silu_kernel,
    silu_and_stage_routed_kernel,
    stage_bf16_activation_pairs_kernel,
    stage_routed_activations_kernel,
    tiny_fp8_down_reduce,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.policy import (
    MarlinCall,
    QuantizedStage,
    SparsePlan,
    direct_split_count,
    select_direct_block_n,
    select_grouped_block_m,
    select_grouped_config,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.routing import (
    align_grouped_tokens,
    w8_small_routes_kernel,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.sparse_tiles import (
    marlin_moe_gemm_grouped_narrow_kernel,
    marlin_moe_gemm_grouped_square_kernel,
    raw_w4_square_kernel,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.wide_tiles import (
    quantized_wide_grouped_kernel,
)

_PPU_W8_PAIR_ALIGN_MAX_ROUTES = 32
_PPU_SMALL_ALIGN_MAX_EXPERTS = 256
_PPU_SMALL_ALIGN_MAX_TOPK = 6


def invoke_marlin_moe_gemm_direct(
    activations: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    a_route_divisor: int,
    group_size: int,
    compute_type: tl.dtype,
    quant_type_id: int,
    use_matvec: bool = False,
    fuse_silu: bool = False,
    split_k: int = 1,
    pipeline_stages: Optional[int] = None,
    use_wide_gate: bool = False,
) -> None:
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = B_scale.stride()
    reduction_size = activations.size(1)
    num_columns = output_tile.size(-1) if fuse_silu else B.size(2)
    routes = topk_ids.numel()
    if fuse_silu:
        block_n = 32 if routes >= 12 else select_direct_block_n(num_columns)
        # Fill more of the 64 PPU compute units for sparse decode batches.
        if routes * triton.cdiv(num_columns, block_n) < 64:
            block_n = 32
    else:
        block_n = 64 if use_matvec else select_direct_block_n(num_columns)
        if (
            use_matvec
            and min(reduction_size, num_columns) <= 512
            and (
                quant_type_id == QUANT_TYPE_UINT8B128
                or (
                    quant_type_id == QUANT_TYPE_FP8_E4M3
                    and compute_type == tl.bfloat16
                    and B_scale.dtype == torch.bfloat16
                )
            )
        ):
            block_n = 32 if reduction_size > num_columns else 128
            four_token_gate = (
                activations.size(0) == 4
                and reduction_size == 4096
                and num_columns == 512
                and group_size == 128
                and 4 <= a_route_divisor <= 8
            )
            if (
                quant_type_id == QUANT_TYPE_FP8_E4M3
                and (routes <= 16 or four_token_gate or use_wide_gate)
                and reduction_size >= 1024
                and num_columns % 128 == 0
            ):
                # Use fewer CTAs for the long-K projection of very few routes.
                block_n = 128
    # Packed FP8 decode favors more long-K tiles and wider short-K tiles.
    # The PPU pipeline pass allocates num_stages - 1 loop buffers.  Three
    # scheduling stages therefore provide the two buffers required to overlap
    # the next AIU copy with the current tile's unpack/dequantize/dot work.
    pipeline_stages = pipeline_stages or (3 if reduction_size > 128 else 1)
    grid = (routes * triton.cdiv(num_columns, block_n),)

    kernel = (
        marlin_moe_gemm_silu_direct_kernel
        if fuse_silu
        else (
            marlin_moe_matvec_direct_kernel
            if use_matvec
            else marlin_moe_gemm_direct_kernel
        )
    )

    target = output_tile
    if split_k > 1:
        assert use_matvec and not fuse_silu
        target = torch.empty(
            (split_k, routes, num_columns),
            device=output_tile.device,
            dtype=torch.float32,
        )
        grid = (*grid, split_k)
        pipeline_stages = min(
            pipeline_stages, triton.cdiv(triton.cdiv(reduction_size, 128), split_k)
        )
    extra = (
        dict(SPLIT_K=split_k, SPLIT_STRIDE=routes * num_columns) if use_matvec else {}
    )
    kernel[grid](
        activations,
        B,
        target,
        B_scale,
        topk_weights,
        topk_ids,
        num_columns,
        reduction_size,
        activations.stride(0),
        activations.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        target.stride(-2),
        target.stride(-1),
        stride_se,
        stride_sg,
        stride_sn,
        A_ROUTE_DIVISOR=a_route_divisor,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
        **extra,
    )

    if split_k > 1:
        reduce_split_silu_kernel[(routes, triton.cdiv(num_columns // 2, 128))](
            target,
            output_tile,
            topk_weights,
            num_columns // 2,
            routes,
            split_k,
            mul_routed_weight,
            128,
            num_warps=4,
        )


def invoke_marlin_moe_gemm_reduce_direct(
    activations: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    mul_routed_weight: bool,
    group_size: int,
    compute_type: tl.dtype,
    quant_type_id: int,
) -> None:
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")

    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = B_scale.stride()
    reduction_size = activations.size(1)
    num_columns = B.size(2)
    top_k = topk_ids.size(1)
    tokens = activations.size(0) // top_k
    block_n = (64 if num_columns >= 1024 else 128) if tokens == 1 else 32
    pipeline_stages = 3 if reduction_size > 128 else 1
    grid = (tokens * triton.cdiv(num_columns, block_n),)

    marlin_moe_gemm_reduce_direct_kernel[grid](
        activations,
        B,
        output_tile,
        B_scale,
        topk_weights,
        topk_ids,
        num_columns,
        reduction_size,
        activations.stride(0),
        activations.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        output_tile.stride(0),
        output_tile.stride(1),
        stride_se,
        stride_sg,
        stride_sn,
        TOP_K=top_k,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        BLOCK_SIZE_N=block_n,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
    )


def stage_grouped_activations(
    activations: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    em: int,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    deinterleave_k: bool = False,
    permute_k16: bool = False,
    permute_k32: bool = False,
    should_skip_inactive: bool = False,
    use_packed_gather: bool = False,
) -> torch.Tensor:
    routed_a = torch.empty(
        (em, activations.size(1)), dtype=activations.dtype, device=activations.device
    )
    if (
        use_packed_gather
        and (permute_k16 or permute_k32)
        and activations.dtype == torch.bfloat16
        and activations.size(1) >= 1024
        and activations.size(1) % (32 if permute_k32 else 16) == 0
        and activations.stride(1) == 1
        and activations.stride(0) % 2 == 0
        and activations.data_ptr() % 4 == 0
    ):
        k_pairs = activations.size(1) // 2
        stage_bf16_activation_pairs_kernel[
            (triton.cdiv(em, 16), triton.cdiv(k_pairs, 256))
        ](
            activations,
            routed_a,
            sorted_token_ids,
            num_tokens_post_padded,
            em,
            k_pairs,
            num_valid_tokens,
            activations.stride(0) // 2,
            top_k,
            16,
            256,
            SKIP_UNUSED_UPPER_ROWS=block_m == 16 and not permute_k32,
            PERMUTE_K32=permute_k32,
            num_warps=8,
            num_stages=1,
        )
        return routed_a
    grid = (triton.cdiv(em, block_m), triton.cdiv(activations.size(1), 128))
    stage_routed_activations_kernel[grid](
        activations,
        routed_a,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        reduction_size=activations.size(1),
        num_valid_tokens=num_valid_tokens,
        stride_am=activations.stride(0),
        stride_ak=activations.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=128,
        DEINTERLEAVE_K=deinterleave_k,
        PERMUTE_K16=permute_k16,
        PERMUTE_K32=permute_k32,
        SKIP_INACTIVE=should_skip_inactive,
        num_warps=4,
        num_stages=1,
    )
    return routed_a


def silu_and_stage_grouped(
    intermediate1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    em: int,
    num_valid_tokens: int,
    block_m: int,
    deinterleave_k: bool = False,
    permute_k16: bool = False,
    permute_k32: bool = False,
    should_skip_inactive: bool = False,
    skip_unused_upper_rows: bool = False,
) -> torch.Tensor:
    n = intermediate1.size(1) // 2
    routed = torch.empty(
        (em, n), dtype=intermediate1.dtype, device=intermediate1.device
    )
    stage_rows = 8 if skip_unused_upper_rows else block_m
    grid = (triton.cdiv(em, stage_rows), triton.cdiv(n, 128))
    silu_and_stage_routed_kernel[grid](
        intermediate1,
        routed,
        sorted_token_ids,
        num_tokens_post_padded,
        EM=em,
        num_columns=n,
        num_valid_tokens=num_valid_tokens,
        BLOCK_SIZE_M=stage_rows,
        BLOCK_SIZE_N=128,
        DEINTERLEAVE_K=deinterleave_k,
        PERMUTE_K16=permute_k16,
        PERMUTE_K32=permute_k32,
        SKIP_INACTIVE=should_skip_inactive,
        SKIP_UNUSED_UPPER_ROWS=skip_unused_upper_rows,
        num_warps=4,
        num_stages=1,
    )
    return routed


def launch_raw_projection(
    routed_a: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    block_m: int,
    quant_type_id: int,
    is_down_projection: bool = False,
    raw_w4: bool = False,
    packed_width: bool = False,
) -> None:
    B, B_scale = stage.weight, stage.scale
    em = sorted_token_ids.size(0)
    num_valid_tokens = output_tile.numel() // output_tile.size(-1)
    n = B.size(2)
    stride_se, stride_sg, stride_sn = B_scale.stride()
    use32 = False
    if raw_w4 or quant_type_id == QUANT_TYPE_FP8_E4M3:
        largest_offset = max(
            routed_a.nbytes - 1,
            sum((length - 1) * stride for length, stride in zip(B.shape, B.stride())),
            sum(
                (length - 1) * stride
                for length, stride in zip(B_scale.shape, B_scale.stride())
            ),
            num_valid_tokens * output_tile.stride(-2) + n * output_tile.stride(-1),
        )
        use32 = largest_offset < (1 << 31)
    packed_mx_scales = (
        raw_w4
        and quant_type_id == QUANT_TYPE_FP4_E2M1
        and stride_sg == 1
        and stride_se % 4 == 0
        and stride_sn % 4 == 0
        and B_scale.data_ptr() % 4 == 0
    )
    tile_args = (
        routed_a,
        B,
        output_tile,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        routed_a.size(1),
        num_valid_tokens,
        B.stride(0),
        B.stride(2),
        stride_se,
        stride_sg,
        stride_sn,
        output_tile.stride(-2),
        output_tile.stride(-1),
        mul_routed_weight,
    )
    if block_m == 64:
        quantized_wide_grouped_kernel[(triton.cdiv(em, 64) * triton.cdiv(n, 128),)](
            *tile_args,
            stage.scale_safe,
            quant_type_id,
            packed_mx_scales,
            use32,
            num_warps=8,
            num_stages=1,
        )
        return
    if raw_w4:
        raw_w4_square_kernel[(triton.cdiv(em, block_m) * triton.cdiv(n, 64),)](
            *tile_args,
            quant_type_id == QUANT_TYPE_FP4_E2M1,
            packed_mx_scales,
            ROWS=block_m,
            USE32=use32,
            num_warps=4,
            num_stages=1,
        )
        return
    # The first down-projection tile is shared by both FP8 scale paths.
    prefetch_before_scale = (
        quant_type_id == QUANT_TYPE_FP8_E4M3
        and is_down_projection
        and routed_a.size(1) == 256
        and n == 4096
    )
    # The two adjacent BF16 scale groups can be loaded and folded together.
    scale_pair = (
        prefetch_before_scale
        and B_scale.dtype == torch.bfloat16
        and stride_sg == 1
        and stride_se % 2 == 0
        and stride_sn % 2 == 0
        and B_scale.data_ptr() % 4 == 0
    )
    if block_m == 16:
        marlin_moe_gemm_grouped_square_kernel[
            (triton.cdiv(em, 16) * triton.cdiv(n, 128),)
        ](
            *tile_args,
            stage.scale_safe,
            quant_type_id == QUANT_TYPE_FP8_E4M3,
            PREFETCH_ROUTES=(
                is_down_projection
                and quant_type_id == QUANT_TYPE_FP8_E4M3
                and routed_a.size(1) == 256
                and num_valid_tokens >= (3 if prefetch_before_scale else 9) * B.size(0)
            ),
            USE32=use32,
            PREFETCH_BEFORE_SCALE=prefetch_before_scale,
            PACK_WIDTH=packed_width,
            SCALE_PAIR=scale_pair,
            num_warps=8,
            num_stages=1,
        )
        return
    assert block_m == 8
    marlin_moe_gemm_grouped_narrow_kernel[(triton.cdiv(em, 8) * triton.cdiv(n, 128),)](
        *tile_args,
        stage.scale_safe,
        quant_type_id == QUANT_TYPE_FP8_E4M3,
        USE32=use32,
        PREFETCH_BEFORE_SCALE=prefetch_before_scale,
        SCALE_PAIR=scale_pair,
        num_warps=8,
        num_stages=1,
    )


def invoke_marlin_moe_gemm_grouped(
    routed_a: torch.Tensor,
    output_tile: torch.Tensor,
    stage: QuantizedStage,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int,
    group_size: int,
    compute_type: tl.dtype,
    quant_type_id: int,
    is_down_projection: bool = False,
    raw_w8: bool = False,
    raw_w4: bool = False,
    packed_width: bool = False,
) -> None:
    if tle_async is None:
        raise RuntimeError("PPU fused Marlin MoE requires Triton TLE")
    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = B_scale.stride()
    em = sorted_token_ids.size(0)
    num_valid_tokens = output_tile.numel() // output_tile.size(-1)
    n = B.size(2)
    if raw_w4 or raw_w8:
        launch_raw_projection(
            routed_a,
            output_tile,
            stage,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight=mul_routed_weight,
            block_m=block_m,
            quant_type_id=quant_type_id,
            is_down_projection=is_down_projection,
            raw_w4=raw_w4,
            packed_width=packed_width,
        )
        return
    batch_m = (
        output_tile.size(0) if is_down_projection else output_tile.size(0) // top_k
    )
    block_n, group_m, num_warps, pipeline_stages = select_grouped_config(
        batch_m, routed_a.size(1), n, block_m, quant_type_id
    )
    # A 64-K tile bounds decoded-weight state; FP8 can then use a wider down tile.
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and group_size == 128 and block_m <= 32:
        block_n = 128
    if (
        quant_type_id == QUANT_TYPE_FP8_E4M3
        and compute_type == tl.bfloat16
        and 1024 <= batch_m < _PPU_DENSE_MIN_TOKENS
        and group_size == 128
        and block_m == 64
        and is_down_projection
    ):
        block_n, num_warps = 128, 4
    stride_cm, stride_cn = output_tile.stride(-2), output_tile.stride(-1)
    grid = (triton.cdiv(em, block_m) * triton.cdiv(n, block_n),)
    marlin_moe_gemm_grouped_kernel[grid](
        routed_a,
        B,
        output_tile,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_columns=n,
        reduction_size=routed_a.size(1),
        EM=em,
        num_valid_tokens=num_valid_tokens,
        stride_be=B.stride(0),
        stride_bk=B.stride(1),
        stride_bn=B.stride(2),
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        stride_bse=stride_se,
        stride_bsg=stride_sg,
        stride_bsn=stride_sn,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=(
            64
            if (
                quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
                and group_size == 128
            )
            or (
                quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
                and block_m >= 64
                and batch_m >= 4096
            )
            else 128
        ),
        GROUP_SIZE_M=group_m,
        GROUP_SIZE_K=group_size,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        PIPELINE_STAGES=pipeline_stages,
        compute_type=compute_type,
        num_warps=num_warps,
        num_stages=pipeline_stages,
        QUANT_TYPE=quant_type_id,
    )


def project_direct_gate(
    call: MarlinCall,
    plan: SparsePlan,
    c1: Optional[torch.Tensor],
    c2: torch.Tensor,
    stage1: QuantizedStage,
    split_silu: bool,
    use_matvec: bool,
    gemv_split_k: int,
) -> None:
    stage1_topk_weights = (
        call.topk_weights if call.apply_router_weight_on_input else None
    )
    if split_silu:
        invoke_marlin_moe_gemm_direct(
            activations=call.hidden_states,
            output_tile=c1 if gemv_split_k == 1 else c2,
            stage=stage1,
            topk_weights=stage1_topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=call.apply_router_weight_on_input,
            a_route_divisor=call.shape.top_k,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
            use_matvec=use_matvec,
            split_k=gemv_split_k,
            pipeline_stages=(
                2
                if plan.small_fp8_gate
                or (
                    plan.decode_fp8_gate
                    and (
                        gemv_split_k == 2 or call.shape.tokens * call.shape.top_k >= 128
                    )
                )
                else None
            ),
            use_wide_gate=plan.decode_fp8_gate,
        )
        if gemv_split_k == 1:
            silu_and_mul_out(
                c1[:, : call.shape.intermediate], c1[:, call.shape.intermediate :], c2
            )
    else:
        invoke_marlin_moe_gemm_direct(
            activations=call.hidden_states,
            output_tile=c2,
            stage=stage1,
            topk_weights=stage1_topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=call.apply_router_weight_on_input,
            a_route_divisor=call.shape.top_k,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
            fuse_silu=True,
        )


def project_direct_down(
    call: MarlinCall,
    plan: SparsePlan,
    c2: torch.Tensor,
    c3: Optional[torch.Tensor],
    stage2: QuantizedStage,
    tiny_fp8_down: bool,
    reduced: bool,
    use_matvec: bool,
) -> None:
    stage2_topk_weights = (
        None if call.apply_router_weight_on_input else call.topk_weights
    )
    if tiny_fp8_down:
        block_n = 16
        tiny_fp8_down_reduce[(call.shape.tokens, call.shape.hidden // block_n)](
            c2,
            call.w2.view(torch.uint8),
            call.w2_scale,
            call.topk_ids,
            call.topk_weights,
            call.out,
            call.shape.hidden,
            call.shape.intermediate,
            call.shape.top_k,
            *call.w2_scale.stride(),
            block_n,
            not call.apply_router_weight_on_input,
            num_warps=4,
        )
    elif reduced:
        invoke_marlin_moe_gemm_reduce_direct(
            activations=c2,
            output_tile=call.out,
            stage=stage2,
            topk_weights=call.topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=not call.apply_router_weight_on_input,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
        )
    else:
        invoke_marlin_moe_gemm_direct(
            activations=c2,
            output_tile=c3,
            stage=stage2,
            topk_weights=stage2_topk_weights,
            topk_ids=call.topk_ids,
            mul_routed_weight=not call.apply_router_weight_on_input,
            a_route_divisor=1,
            group_size=call.group_size,
            compute_type=call.compute_type,
            quant_type_id=call.quant_type_id,
            use_matvec=use_matvec,
            pipeline_stages=(
                1 if plan.small_fp8_gate else 2 if plan.decode_fp8_gate else None
            ),
        )


def run_direct(
    call: MarlinCall, plan: SparsePlan, stage1: QuantizedStage, stage2: QuantizedStage
) -> None:
    split_silu = call.shape.intermediate >= 1024 or call.quant_type_id in (
        QUANT_TYPE_UINT8B128,
        QUANT_TYPE_FP8_E4M3,
    )
    use_matvec = plan.packed_fp8 or (
        call.shape.tokens == 1
        and call.quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
    )
    gemv_split_k = direct_split_count(call, plan)
    tiny_fp8_down = (
        plan.small_fp8_gate
        and call.shape.tokens == 2
        and call.w2.is_contiguous()
        and call.w2_scale.is_contiguous()
        and call.w2.numel() < (1 << 31)
    )
    reduced = tiny_fp8_down or (call.shape.tokens <= 2 and not split_silu)
    c2 = torch.empty(
        (call.shape.tokens * call.shape.top_k, call.shape.intermediate),
        device=call.out.device,
        dtype=call.out.dtype,
    )
    c1 = (
        torch.empty(
            (call.shape.tokens * call.shape.top_k, 2 * call.shape.intermediate),
            device=call.out.device,
            dtype=call.out.dtype,
        )
        if split_silu and gemv_split_k == 1
        else None
    )
    c3 = (
        None
        if reduced
        else torch.empty(
            (call.shape.tokens, call.shape.top_k, call.shape.hidden),
            device=call.out.device,
            dtype=call.out.dtype,
        )
    )
    project_direct_gate(
        call, plan, c1, c2, stage1, split_silu, use_matvec, gemv_split_k
    )
    project_direct_down(call, plan, c2, c3, stage2, tiny_fp8_down, reduced, use_matvec)
    if not reduced:
        moe_sum(c3, call.out)


def group_routes(
    call: MarlinCall, plan: SparsePlan, stage1: QuantizedStage, stage2: QuantizedStage
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
    packed_width = False
    bm = select_grouped_block_m(call.shape.tokens, call.shape.experts, call.shape.top_k)
    if (plan.raw_w8 or plan.raw_w4) and plan.use_wide_tile:
        bm = 64
    elif plan.raw_w8:
        narrow_density = 5 if call.quant_type_id == QUANT_TYPE_FP8_E4M3 else 6
        if plan.wide_fp8_gate_shape:
            narrow_density = 3
        bm = (
            8
            if call.shape.tokens * call.shape.top_k
            <= narrow_density * call.shape.experts
            else 16
        )
    elif plan.raw_w4:
        bm = 8 if call.shape.tokens * call.shape.top_k <= 5 * call.shape.experts else 16
    if (
        (plan.raw_w4 or (plan.raw_w8 and call.quant_type_id == QUANT_TYPE_UINT8B128))
        and call.shape.tokens <= 2
        and call.shape.top_k <= _PPU_SMALL_ALIGN_MAX_TOPK
        and call.shape.tokens * call.shape.top_k
        <= call.shape.experts
        <= _PPU_SMALL_ALIGN_MAX_EXPERTS
    ):
        if call.shape.tokens == 1:
            # One token typically routes to distinct experts.
            sorted_ids, experts, padded = moe_align_block_size_singleton(
                call.topk_ids, bm
            )
        else:
            # Group routes so both tokens can share an expert tile.
            sorted_ids, experts, padded = moe_align_block_size_small_grouped(
                call.topk_ids, call.shape.experts, bm
            )
    elif (
        plan.raw_w8
        and call.quant_type_id == QUANT_TYPE_UINT8B128
        and call.shape.tokens * call.shape.top_k <= _PPU_W8_PAIR_ALIGN_MAX_ROUTES
    ):
        routes = call.shape.tokens * call.shape.top_k
        max_padded_routes = min(routes * bm, routes + call.shape.experts * (bm - 1))
        sorted_ids = torch.empty(
            (max_padded_routes,), device=call.topk_ids.device, dtype=torch.int32
        )
        experts = torch.empty((routes,), device=call.topk_ids.device, dtype=torch.int32)
        padded = torch.empty((1,), device=call.topk_ids.device, dtype=torch.int32)
        w8_small_routes_kernel[(1,)](
            call.topk_ids,
            sorted_ids,
            experts,
            padded,
            routes,
            bm,
            triton.next_power_of_2(max(routes, bm)),
            num_warps=4,
        )
    else:
        is_one_cta = (
            (plan.raw_w8 or plan.raw_w4)
            and call.shape.tokens * call.shape.top_k <= 4096
            and call.shape.experts <= 1024
        )
        if plan.is_joint_safety:
            packed_width = is_one_cta and bm == 16
            sorted_ids, experts, padded = align_grouped_tokens(
                call.topk_ids,
                bm,
                call.shape.experts,
                fp8_scales=(call.w1_scale, call.w2_scale),
                fp8_safety=(stage1.scale_safe, stage2.scale_safe),
                packed_width=packed_width,
                is_one_cta=is_one_cta,
            )
        else:
            sorted_ids, experts, padded = align_grouped_tokens(
                call.topk_ids,
                bm,
                call.shape.experts,
                is_one_cta=is_one_cta,
            )
    return sorted_ids, experts, padded, bm, packed_width


def run_grouped(
    call: MarlinCall, plan: SparsePlan, stage1: QuantizedStage, stage2: QuantizedStage
) -> None:
    stage1_topk_weights = (
        call.topk_weights if call.apply_router_weight_on_input else None
    )
    stage2_topk_weights = (
        None if call.apply_router_weight_on_input else call.topk_weights
    )
    sorted_ids, experts, padded, bm, packed_width = group_routes(
        call, plan, stage1, stage2
    )
    c3 = torch.empty(
        (call.shape.tokens, call.shape.top_k, call.shape.hidden),
        device=call.out.device,
        dtype=call.out.dtype,
    )
    c1 = torch.empty(
        (call.shape.tokens * call.shape.top_k, 2 * call.shape.intermediate),
        device=call.out.device,
        dtype=call.out.dtype,
    )
    staging = dict(
        em=sorted_ids.numel(),
        num_valid_tokens=call.shape.tokens * call.shape.top_k,
        block_m=bm,
        deinterleave_k=call.quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1)
        and call.compute_type == tl.bfloat16
        and bm <= 32
        and not plan.raw_w4,
        permute_k16=plan.raw_w8,
        permute_k32=plan.raw_w4,
        should_skip_inactive=plan.raw_w8
        or plan.raw_w4
        or call.quant_type_id == QUANT_TYPE_FP8_E4M3,
    )
    invoke_marlin_moe_gemm_grouped(
        routed_a=stage_grouped_activations(
            call.hidden_states,
            sorted_ids,
            padded,
            top_k=call.shape.top_k,
            use_packed_gather=plan.raw_w4
            or (plan.raw_w8 and (plan.packed_fp8 or plan.use_wide_tile)),
            **staging,
        ),
        output_tile=c1,
        stage=stage1,
        topk_weights=stage1_topk_weights,
        sorted_token_ids=sorted_ids,
        expert_ids=experts,
        num_tokens_post_padded=padded,
        mul_routed_weight=call.apply_router_weight_on_input,
        top_k=call.shape.top_k,
        block_m=bm,
        group_size=call.group_size,
        compute_type=call.compute_type,
        quant_type_id=call.quant_type_id,
        raw_w8=plan.raw_w8,
        raw_w4=plan.raw_w4,
        packed_width=packed_width,
    )
    routed_c2 = silu_and_stage_grouped(
        c1,
        sorted_ids,
        padded,
        **staging,
        skip_unused_upper_rows=(
            plan.raw_w8 and call.quant_type_id == QUANT_TYPE_FP8_E4M3 and bm == 16
        ),
    )
    invoke_marlin_moe_gemm_grouped(
        routed_a=routed_c2,
        output_tile=c3,
        stage=stage2,
        topk_weights=stage2_topk_weights,
        sorted_token_ids=sorted_ids,
        expert_ids=experts,
        num_tokens_post_padded=padded,
        mul_routed_weight=not call.apply_router_weight_on_input,
        top_k=1,
        block_m=bm,
        group_size=call.group_size,
        compute_type=call.compute_type,
        quant_type_id=call.quant_type_id,
        is_down_projection=True,
        raw_w8=plan.raw_w8,
        raw_w4=plan.raw_w4,
        packed_width=packed_width,
    )
    moe_sum(c3, call.out)
