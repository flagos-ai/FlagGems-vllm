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

"""T-Head public entry point for quantized fused Marlin MoE."""

from enum import Enum
from typing import Callable, Optional, Union

import torch
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as generic_fused_marlin_moe,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.dense import run_dense_batches
from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import tle_async
from flaggems_vllm.runtime.backend._thead.fused.marlin.launch import (
    run_direct,
    run_grouped,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.policy import (
    MarlinCall,
    eligible_for_dense,
    prepare_quantized_weights,
    select_sparse_plan,
    validate_inputs,
)

_PPU_DENSE_GATE_TILE = 256
_PPU_DENSE_DOWN_TILE = 1024
_PPU_DENSE_MAX_WORKSPACE_BYTES = 2 << 30
_PPU_QUANT_TYPES = (
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
)
_VLLM_QUANT_TYPE_FP8_E4M3 = 2814749767172868


def dispatch_marlin_call(call: MarlinCall) -> None:
    with torch.cuda.device(call.hidden_states.device):
        if eligible_for_dense(call):
            run_dense_batches(
                call,
                _PPU_DENSE_GATE_TILE,
                _PPU_DENSE_DOWN_TILE,
                _PPU_DENSE_MAX_WORKSPACE_BYTES,
            )
        else:
            plan = select_sparse_plan(call, _PPU_DENSE_MAX_WORKSPACE_BYTES)
            stage1, stage2 = prepare_quantized_weights(
                call.w1,
                call.w2,
                call.w1_scale,
                call.w2_scale,
                needs_expert_flags=plan.is_joint_safety,
                is_mxfp4=call.quant_type_id == QUANT_TYPE_FP4_E2M1,
            )
            if plan.direct:
                run_direct(call, plan, stage1, stage2)
            else:
                run_grouped(call, plan, stage1, stage2)


def run_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    apply_router_weight_on_input: bool = False,
    inplace: bool = False,
    output: Optional[torch.Tensor] = None,
    group_size: int = 128,
    quant_type_id: int = QUANT_TYPE_UINT4B8,
) -> torch.Tensor:
    """Forward SwiGLU MoE with output-major quantized weights.

    All weights and scales are read directly on every invocation. MXFP4
    E8M0 scales are decoded on every invocation without a persistent copy.
    Expert IDs must be in [0, E); expert parallel maps are not supported.
    """
    shape = validate_inputs(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        inplace,
        group_size,
        quant_type_id,
    )
    if tle_async is None:
        raise NotImplementedError(
            "PPU fused Marlin MoE requires a TLE-enabled FlagTree build"
        )
    out = hidden_states if inplace else output
    if out is None:
        out = torch.empty_like(hidden_states)
    if shape.tokens == 0:
        return out
    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16
    call = MarlinCall(
        shape,
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        out,
        group_size,
        quant_type_id,
        apply_router_weight_on_input,
        compute_type,
    )
    dispatch_marlin_call(call)
    return out


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[Union[str, Enum]] = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """PPU override for INT4, MXFP4, INT8 and FP8 fused Marlin MoE."""
    if quant_type_id == _VLLM_QUANT_TYPE_FP8_E4M3:
        quant_type_id = QUANT_TYPE_FP8_E4M3
    if quant_type_id not in _PPU_QUANT_TYPES:
        return generic_fused_marlin_moe(**locals())
    activation_str = getattr(
        activation, "value", getattr(activation, "name", activation)
    )
    if activation_str is not None and str(activation_str).lower() != "silu":
        raise NotImplementedError("PPU fused Marlin MoE supports only SiLU")
    if any(
        value is not None for value in (g_idx1, g_idx2, sort_indices1, sort_indices2)
    ):
        raise NotImplementedError("PPU fused Marlin MoE does not support act_order")
    if input_dtype is not None:
        raise NotImplementedError("PPU fused Marlin MoE does not support FP8 input")
    unsupported = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        w1_zeros,
        w2_zeros,
        workspace,
        intermediate_cache13,
        intermediate_cache2,
        clamp_limit,
    )
    if any(value is not None for value in unsupported) or not is_k_full:
        raise NotImplementedError(
            "Unsupported PPU fused Marlin MoE option "
            "(bias/map/scaling/workspace/activation)"
        )
    if w1.ndim != 3 or global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("PPU fused Marlin MoE requires local weights")
    return run_marlin_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
        inplace=inplace,
        output=output,
        group_size=group_size,
        quant_type_id=quant_type_id,
    )


__all__ = ["fused_marlin_moe"]
