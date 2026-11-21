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

"""Input validation, tensor layouts and sparse scheduling."""

from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin.formats import (
    _PPU_DENSE_MIN_TOKENS,
)

_PPU_DIRECT_ROUTE_LIMIT = 32
MAX_COMPACT_SCALE_ELEMENTS = 16 * 1024
_PPU_WIDE_SHARED_BYTES = 64 << 10
_PPU_DENSE_MAX_DECODE_BYTES = 1 << 30


class QuantizedStage(NamedTuple):
    weight: torch.Tensor
    scale: torch.Tensor
    scale_safe: Optional[torch.Tensor]


def select_direct_block_n(n: int) -> int:
    if n <= 32:
        return 32
    if n < 512:
        return 64
    return 128


def select_grouped_config(
    num_rows: int,
    reduction_size: int,
    num_columns: int,
    block_m: int,
    quant_type_id: int,
) -> Tuple[int, int, int, int]:
    if quant_type_id == QUANT_TYPE_UINT8B128 and block_m <= 32:
        # Sparse INT8 expert batches spill with 256-wide decoded tiles. A
        # 128-wide tile keeps both GEMMs resident across the upstream shapes.
        return 128, 1, 4, 3
    if quant_type_id == QUANT_TYPE_FP8_E4M3 and block_m <= 32:
        # Sparse FP8 expert batches need smaller decoded tiles to avoid spills.
        return (
            128 if reduction_size > num_columns else 64,
            1 if num_rows < 512 else 8,
            4,
            3,
        )
    if 256 <= num_rows <= 512 and reduction_size > num_columns:
        return 128, 1, 4, 3
    if num_rows < 512:
        return 256, 1, 8, 3
    if num_rows >= 4096:
        return 128, 8, 4, 3
    if num_rows == 2048 and reduction_size > num_columns:
        return 256, 8, 8, 2
    return 256, 8, 8, 3


def select_grouped_block_m(num_rows: int, num_experts: int, top_k: int) -> int:
    routes = num_rows * top_k
    if routes <= 16 * num_experts:
        return 16
    if routes <= 32 * num_experts:
        return 32
    return 64


def use_direct_route(
    num_rows: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    routes = num_rows * top_k
    if num_experts <= 8 and routes >= num_experts:
        return False
    max_output_n = max(hidden_size, 2 * intermediate_size)
    block_n = select_direct_block_n(max_output_n)
    n_tiles = triton.cdiv(max_output_n, block_n)
    max_routes_by_grid = 65535 // n_tiles
    return routes <= min(_PPU_DIRECT_ROUTE_LIMIT, max_routes_by_grid)


class MoEShape(NamedTuple):
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    top_k: int


def validate_w8(
    shape: MoEShape,
    activation_dtype: torch.dtype,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    quant_type_id: int,
) -> None:
    _, k, n, e, _ = shape
    if w1.shape != (e, 2 * n, k) or w2.shape != (e, k, n):
        raise ValueError("8-bit weight shapes do not match activations")
    allowed_group_sizes = (
        (-1, 32, 64, 128) if quant_type_id == QUANT_TYPE_FP8_E4M3 else (128,)
    )
    if group_size not in allowed_group_sizes:
        raise NotImplementedError(
            f"quant_type_id={quant_type_id} does not support group_size="
            f"{group_size}"
        )
    if group_size != -1 and (k % group_size or n % group_size):
        raise ValueError("Input dimensions must be divisible by group_size")
    g1 = 1 if group_size == -1 else k // group_size
    g2 = 1 if group_size == -1 else n // group_size
    if w1_scale.shape != (e, 2 * n, g1) or w2_scale.shape != (e, k, g2):
        raise ValueError("8-bit scale shapes must match the selected group size")
    if quant_type_id == QUANT_TYPE_FP8_E4M3:
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
        if w1.dtype not in (torch.uint8, fp8_dtype) or w2.dtype not in (
            torch.uint8,
            fp8_dtype,
        ):
            raise NotImplementedError("Expected output-major E4M3FN weights")
        if w1_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32) or (
            w2_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise NotImplementedError("FP8 scales must be FP16, BF16 or FP32")
    elif w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("INT8 weights must use output-major uint8")
    elif w1_scale.dtype != activation_dtype or w2_scale.dtype != activation_dtype:
        raise NotImplementedError("INT8 scales must match activation dtype")


def validate_w4(
    shape: MoEShape,
    activation_dtype: torch.dtype,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    group_size: int,
    quant_type_id: int,
) -> None:
    _, k, n, e, _ = shape
    if w1.shape != (e, 2 * n, k // 2) or w2.shape != (e, k, n // 2):
        raise ValueError("4-bit weight shapes do not match activations")
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("Expected output-major packed uint8 weights")
    expected_group_size = 128 if quant_type_id == QUANT_TYPE_UINT4B8 else 32
    if group_size != expected_group_size:
        raise NotImplementedError(
            f"quant_type_id={quant_type_id} requires group_size="
            f"{expected_group_size}"
        )
    if k % 128 or n % 128:
        raise ValueError(
            "4-bit hidden_size and intermediate_size must be multiples of 128"
        )
    if w1_scale.shape != (e, 2 * n, k // group_size) or w2_scale.shape != (
        e,
        k,
        n // group_size,
    ):
        raise ValueError("4-bit scale shapes do not match the selected group size")
    if quant_type_id == QUANT_TYPE_UINT4B8:
        if w1_scale.dtype != activation_dtype or w2_scale.dtype != activation_dtype:
            raise NotImplementedError("INT4 scales must match activation dtype")
    else:
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
        if w1_scale.dtype not in (torch.uint8, e8m0_dtype) or w2_scale.dtype not in (
            torch.uint8,
            e8m0_dtype,
        ):
            raise NotImplementedError("MXFP4 scales must use E8M0 bytes")


def validate_output(
    hidden_states: torch.Tensor,
    output: Optional[torch.Tensor],
    inplace: bool,
    live_inputs: Tuple[torch.Tensor, ...],
) -> None:
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if output is not None:
        if (
            output.shape != hidden_states.shape
            or output.dtype != hidden_states.dtype
            or output.device != hidden_states.device
        ):
            raise ValueError("Output shape/dtype/device must match hidden_states")
        if not output.is_contiguous() or output.requires_grad:
            raise ValueError("Output must be contiguous and inference-only")
    # Weights, scales and routing remain live during GEMM2.
    target = hidden_states if inplace else output
    if target is not None and target.numel():
        for tensor in live_inputs:
            if (
                tensor.numel()
                and target.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
            ):
                raise ValueError("Output must not alias weights, scales or routing")


def validate_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: Optional[torch.Tensor],
    inplace: bool,
    group_size: int,
    quant_type_id: int,
) -> MoEShape:
    if any(tensor.layout != torch.strided for tensor in (w1, w2, w1_scale, w2_scale)):
        raise NotImplementedError(
            "PPU Marlin weights and scales require strided tensors"
        )
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("Expected activations/routing rank 2 and weights rank 3")
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise NotImplementedError(
            "PPU Marlin requires contiguous output-major weights with unit K stride"
        )
    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n = n2 // 2
    if e <= 0 or n2 % 2 or min(k, n) <= 0 or k % 32 or n % 32:
        raise ValueError(
            "Positive K/N multiples of 32 and paired gate/up weights required"
        )
    if (
        topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != m
        or not 1 <= topk_ids.shape[1] <= e
    ):
        raise ValueError("Routing must have shape [M, topk], 1 <= topk <= E")
    shape = MoEShape(m, k, n, e, topk_ids.shape[1])
    validate_format = (
        validate_w8
        if quant_type_id in (QUANT_TYPE_UINT8B128, QUANT_TYPE_FP8_E4M3)
        else validate_w4
    )
    validate_format(
        shape,
        hidden_states.dtype,
        w1,
        w2,
        w1_scale,
        w2_scale,
        group_size,
        quant_type_id,
    )
    if any(
        scale.size(-1) > 1 and scale.stride(-1) != 1 for scale in (w1_scale, w2_scale)
    ):
        raise NotImplementedError(
            "PPU Marlin scales require unit stride along the group dimension"
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("PPU fused Marlin MoE requires FP16 or BF16")
    if topk_ids.dtype not in (torch.int32, torch.int64) or topk_weights.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Unsupported routing dtype")
    tensors = (hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
    if hidden_states.device.type != "cuda" or any(
        tensor.device != hidden_states.device for tensor in tensors
    ):
        raise ValueError("All tensors must reside on the same PPU device")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("PPU fused Marlin MoE is inference-only")
    if any(
        not tensor.is_contiguous() for tensor in (hidden_states, topk_weights, topk_ids)
    ):
        raise NotImplementedError("Activations and routing must be contiguous")
    validate_output(
        hidden_states,
        output,
        inplace,
        (w1, w2, w1_scale, w2_scale, topk_weights, topk_ids),
    )
    return shape


def prepare_quantized_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    needs_expert_flags: bool,
    is_mxfp4: bool,
) -> Tuple[QuantizedStage, QuantizedStage]:
    b1 = w1.view(torch.uint8).transpose(1, 2)
    b2 = w2.view(torch.uint8).transpose(1, 2)
    s1 = w1_scale.view(torch.uint8) if is_mxfp4 else w1_scale
    s2 = w2_scale.view(torch.uint8) if is_mxfp4 else w2_scale
    if needs_expert_flags:
        safe1 = torch.empty((w1.size(0),), device=w1.device, dtype=torch.int32)
        safe2 = torch.empty_like(safe1)
    else:
        safe1, safe2 = None, None
    return (
        QuantizedStage(b1, s1.transpose(1, 2), safe1),
        QuantizedStage(b2, s2.transpose(1, 2), safe2),
    )


class MarlinCall(NamedTuple):
    shape: MoEShape
    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    out: torch.Tensor
    group_size: int
    quant_type_id: int
    apply_router_weight_on_input: bool
    compute_type: tl.dtype


class SparsePlan(NamedTuple):
    packed_fp8: bool
    wide_fp8_gate_shape: bool
    use_wide_tile: bool
    raw_w8: bool
    raw_w4: bool
    direct: bool
    is_tiny_int8_matvec: bool
    small_fp8_gate: bool
    decode_fp8_gate: bool
    is_joint_safety: bool


def raw_tile_modes(call: MarlinCall, use_wide_tile: bool) -> Tuple[bool, bool]:
    tokens, hidden, intermediate, experts, top_k = call.shape
    if (
        call.compute_type != tl.bfloat16
        or hidden % 128
        or intermediate % 128
        or not (tokens * top_k <= 16 * experts or use_wide_tile)
    ):
        return False, False
    bf16_group = (
        call.group_size == 128
        and call.w1_scale.dtype == call.w2_scale.dtype == torch.bfloat16
    )
    raw_w8 = bf16_group and call.quant_type_id in (
        QUANT_TYPE_UINT8B128,
        QUANT_TYPE_FP8_E4M3,
    )
    raw_w4 = (bf16_group and call.quant_type_id == QUANT_TYPE_UINT4B8) or (
        call.group_size == 32 and call.quant_type_id == QUANT_TYPE_FP4_E2M1
    )
    return raw_w8, raw_w4


def direct_schedule(
    call: MarlinCall,
    workspace_limit: int,
    packed_fp8: bool,
    wide_fp8_gate_shape: bool,
    raw_w8: bool,
    raw_w4: bool,
) -> Tuple[bool, bool, bool, bool]:
    tokens, hidden, intermediate, experts, top_k = call.shape
    routes = tokens * top_k
    # Sparse raw tiles pad to eight rows; account for both staged activations.
    padded_routes = routes + 7 * experts
    sparse_bytes = (
        4 * (experts + 1) ** 2
        + 16 * padded_routes
        + 2 * padded_routes * (hidden + intermediate)
        + 2 * tokens * top_k * (hidden + 2 * intermediate)
        + 8
    )
    sparse_raw = raw_w4 or (raw_w8 and call.quant_type_id == QUANT_TYPE_UINT8B128)
    direct = use_direct_route(
        tokens,
        experts,
        top_k,
        hidden,
        intermediate,
    ) and not (sparse_raw and sparse_bytes <= workspace_limit)
    # A one-token gate/up grid may not fill the PPU without split K.
    is_tiny_int8_matvec = (
        raw_w8
        and call.quant_type_id == QUANT_TYPE_UINT8B128
        and tokens == 1
        and hidden >= 1024
        and hidden % 1024 == 0
        and 2 * hidden < (1 << 17)
        and 2 * intermediate <= 1024
        and top_k * triton.cdiv(2 * intermediate, 128) < 32
    )
    direct = direct or is_tiny_int8_matvec
    if (
        packed_fp8
        and experts > 8
        and routes <= 160
        and max(hidden, 2 * intermediate) <= 4096
        and not raw_w8
    ):
        direct = True
    # Reuse routed weights once the grouped projection can fill the PPU.
    is_raw_fp8_sparse = (
        raw_w8
        and packed_fp8
        and routes <= min(32, experts // 2)
        and tokens * top_k * triton.cdiv(2 * intermediate, 128) >= 64
        and sparse_bytes <= workspace_limit
    )
    small_fp8_gate = wide_fp8_gate_shape and tokens in (1, 2, 4)
    decode_fp8_gate = (
        wide_fp8_gate_shape
        and raw_w8
        and 4 < tokens <= 32
        and 32 <= routes <= 192
        # Denser routes reuse expert weights in the grouped projection.
        and 4 * routes <= 3 * experts
    )
    direct = direct and (not is_raw_fp8_sparse or (small_fp8_gate and tokens == 4))
    direct = direct or decode_fp8_gate
    return direct, is_tiny_int8_matvec, small_fp8_gate, decode_fp8_gate


def select_sparse_plan(call: MarlinCall, workspace_limit: int) -> SparsePlan:
    tokens, hidden, intermediate, experts, top_k = call.shape
    routes = tokens * top_k
    packed_fp8 = (
        call.quant_type_id == QUANT_TYPE_FP8_E4M3
        and call.hidden_states.dtype
        == call.w1_scale.dtype
        == call.w2_scale.dtype
        == torch.bfloat16
    )
    wide_gate_shape = hidden == 4096 and intermediate == 256 and 4 <= top_k <= 8
    wide_fp8_gate_shape = packed_fp8 and call.group_size == 128 and wide_gate_shape
    # A 64-row tile reuses each decoded weight fragment across three or four MMAs.
    use_wide_tile = (
        wide_gate_shape
        and call.hidden_states.dtype == torch.bfloat16
        and 1024 <= tokens < _PPU_DENSE_MIN_TOKENS
        and 32 * experts < routes <= 64 * experts
        and torch.cuda.get_device_properties(
            call.hidden_states.device
        ).shared_memory_per_block_optin
        >= _PPU_WIDE_SHARED_BYTES
    )
    raw_w8, raw_w4 = raw_tile_modes(call, use_wide_tile)
    direct, is_tiny_int8_matvec, small_fp8_gate, decode_fp8_gate = direct_schedule(
        call, workspace_limit, packed_fp8, wide_fp8_gate_shape, raw_w8, raw_w4
    )
    is_joint_safety = (
        raw_w8
        and packed_fp8
        and not direct
        and tokens < _PPU_DENSE_MIN_TOKENS
        and max(
            call.w1_scale.size(1) * call.w1_scale.size(2),
            call.w2_scale.size(1) * call.w2_scale.size(2),
        )
        <= MAX_COMPACT_SCALE_ELEMENTS
    )
    return SparsePlan(
        packed_fp8,
        wide_fp8_gate_shape,
        use_wide_tile,
        raw_w8,
        raw_w4,
        direct,
        is_tiny_int8_matvec,
        small_fp8_gate,
        decode_fp8_gate,
        is_joint_safety,
    )


def eligible_for_dense(call: MarlinCall) -> bool:
    return (
        call.shape.tokens >= _PPU_DENSE_MIN_TOKENS
        and call.shape.tokens * call.shape.top_k > 32 * call.shape.experts
        and call.hidden_states.dtype == torch.bfloat16
        and (
            (
                call.group_size == 128
                and call.w1_scale.dtype == call.w2_scale.dtype == torch.bfloat16
            )
            or call.quant_type_id == QUANT_TYPE_FP4_E2M1
        )
        and max(call.w1.numel(), call.w2.numel())
        * (4 if call.quant_type_id in (QUANT_TYPE_UINT4B8, QUANT_TYPE_FP4_E2M1) else 2)
        <= _PPU_DENSE_MAX_DECODE_BYTES
    )


def direct_split_count(call: MarlinCall, plan: SparsePlan) -> int:
    gemv_split_k = 1
    if (
        (plan.packed_fp8 or plan.is_tiny_int8_matvec)
        and call.group_size == 128
        and call.shape.hidden >= 1024
        and call.shape.hidden % 1024 == 0
        and 2 * call.shape.intermediate <= 1024
    ):
        routes = call.shape.tokens * call.shape.top_k
        # Fill the compute units for very sparse routes; the 33--64 route
        # band already has enough column tiles without another reduction.
        if plan.small_fp8_gate:
            # The wide gate tiles need more CTAs at low token counts.
            gemv_split_k = 16 // call.shape.tokens
        elif plan.decode_fp8_gate:
            gemv_split_k = 2 if routes <= 64 else 8
        else:
            gemv_split_k = (
                min(8, 64 // triton.next_power_of_2(routes)) if routes <= 64 else 4
            )
    return gemv_split_k
