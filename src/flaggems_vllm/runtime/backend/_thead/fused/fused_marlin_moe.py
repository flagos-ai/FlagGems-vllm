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

"""T-Head PPU specialization for quantized fused Marlin MoE."""

from enum import Enum
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakTensorKeyDictionary

from flaggems_vllm.ops.fused_marlin_moe import (
    QUANT_TYPE_FP4_E2M1,
    QUANT_TYPE_FP8_E4M3,
    QUANT_TYPE_UINT4B8,
    QUANT_TYPE_UINT8B128,
)
from flaggems_vllm.ops.fused_marlin_moe import (
    fused_marlin_moe as generic_fused_marlin_moe,
)
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)
from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._thead.fused.marlin_dense import (
    MIN_OVERLAP_TOKENS,
    OVERLAP_BLOCK_ROWS,
    dense_routing_sizes,
    eager_decode_stream,
    run_dense_wna16,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin_fragments import tle_async
from flaggems_vllm.runtime.backend._thead.fused.marlin_quantized import (
    align_grouped_tokens,
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
    w8_small_routes_kernel,
)
from flaggems_vllm.runtime.backend._thead.fused.marlin_tiles import (
    marlin_moe_gemm_grouped_narrow_kernel,
    marlin_moe_gemm_grouped_square_kernel,
    quantized_wide_grouped_kernel,
    raw_w4_square_kernel,
)

try:
    import triton.experimental.tle as tle_api
except ImportError:
    tle_api = None

_SCALE_CACHE = WeakTensorKeyDictionary()


DENSE_WORKSPACE_HEADROOM_BYTES = 16 << 20
_PPU_DENSE_MIN_TOKENS = 4096


_PPU_DIRECT_ROUTE_LIMIT = 32
MAX_COMPACT_SCALE_ELEMENTS = 16 * 1024
_PPU_DENSE_MAX_DECODE_BYTES = 1 << 30


class ScaleLayout(NamedTuple):
    # Kernel argument order: expert, quantization group, output channel.
    strides: Tuple[int, int, int]
    max_offset: int


class QuantizedStage(NamedTuple):
    weight: torch.Tensor
    scale: torch.Tensor
    scale_safe: Optional[torch.Tensor]
    layout: ScaleLayout


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


def overlaps_live_input(start: int, end: int, tensor: torch.Tensor) -> bool:
    first = tensor.data_ptr()
    if end <= first or not tensor.numel():
        return False
    if tensor.is_contiguous():
        return start < first + tensor.nbytes

    # Only scales may be strided. Each [expert, column] group row is contiguous.
    experts, columns, groups = tensor.shape
    item = tensor.element_size()
    expert_step, column_step, _ = (stride * item for stride in tensor.stride())
    width = groups * item
    if (
        start
        >= first + (experts - 1) * expert_step + (columns - 1) * column_step + width
    ):
        return False
    for expert in range(experts):
        row = first + expert * expert_step
        # First column whose last byte can reach the output, including padded rows.
        column = max(0, (start - row - width) // column_step + 1) if column_step else 0
        row += column * column_step
        if column < columns and row < end and start < row + width:
            return True
    return False


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
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    device = hidden_states.device
    live_inputs = (w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
    if device.type != "cuda" or any(t.device != device for t in live_inputs):
        raise ValueError("All tensors must reside on the same PPU device")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("PPU fused Marlin MoE requires FP16 or BF16")
    if hidden_states.ndim != 2 or topk_ids.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("Expected activations/routing rank 2 and weights rank 3")
    if any(t.requires_grad for t in (hidden_states, *live_inputs)):
        raise NotImplementedError("PPU fused Marlin MoE is inference-only")
    m, k = hidden_states.shape
    e, n2, packed_k = w1.shape
    n = n2 // 2
    if min(k, n, e) <= 0 or n2 % 2:
        raise ValueError("Positive K/N/E and paired gate/up weights required")
    rows, top_k = topk_ids.shape
    if rows != m or not 1 <= top_k <= e or topk_weights.shape != topk_ids.shape:
        raise ValueError("Routing must have shape [M, topk], 1 <= topk <= E")
    float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (
        topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype not in float_dtypes
    ):
        raise NotImplementedError("Unsupported routing dtype")
    if not all(t.is_contiguous() for t in (hidden_states, topk_weights, topk_ids)):
        raise NotImplementedError("Activations and routing must be contiguous")
    if output is not None and (
        output.shape != hidden_states.shape
        or output.dtype != hidden_states.dtype
        or output.device != device
        or not output.is_contiguous()
        or output.requires_grad
    ):
        raise ValueError(
            "Output must match hidden_states shape/dtype/device and be contiguous and inference-only"
        )

    is_fp8 = quant_type_id == QUANT_TYPE_FP8_E4M3
    is_mxfp4 = quant_type_id == QUANT_TYPE_FP4_E2M1
    is_w4 = is_mxfp4 or quant_type_id == QUANT_TYPE_UINT4B8
    if is_fp8:
        allowed_groups = (-1, 32, 64, 128)
    else:
        allowed_groups = (32,) if is_mxfp4 else (128,)
    if group_size not in allowed_groups:
        raise NotImplementedError(
            f"quant_type_id={quant_type_id} does not support group_size={group_size}"
        )
    alignment = 128 if is_w4 else (32 if group_size == -1 else group_size)
    if k % alignment or n % alignment:
        raise ValueError(
            f"hidden_size and intermediate_size must be multiples of {alignment}"
        )

    packing = 2 if is_w4 else 1
    if packed_k != k // packing or w2.shape != (e, k, n // packing):
        raise ValueError("Quantized weight shapes do not match activations")
    weight_dtypes = (torch.uint8, torch.float8_e4m3fn) if is_fp8 else (torch.uint8,)
    for weight in (w1, w2):
        if weight.dtype not in weight_dtypes or not weight.is_contiguous():
            raise NotImplementedError(
                "Expected contiguous output-major weights of the selected quantized dtype"
            )

    if is_fp8:
        scale_dtypes = float_dtypes
    elif is_mxfp4:
        scale_dtypes = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
    else:
        scale_dtypes = (hidden_states.dtype,)
    for scale, columns, reduction in ((w1_scale, n2, k), (w2_scale, k, n)):
        groups = 1 if group_size == -1 else reduction // group_size
        if scale.shape != (e, columns, groups):
            raise ValueError("Scale shape does not match the selected group size")
        if (
            scale.dtype not in scale_dtypes
            or scale.layout != torch.strided
            or (groups > 1 and scale.stride(-1) != 1)
        ):
            raise NotImplementedError(
                "Expected strided scales of the selected dtype with unit group stride"
            )
    target = hidden_states if inplace else output
    if target is not None and target.numel():
        start = target.data_ptr()
        end = start + target.nbytes
        if any(overlaps_live_input(start, end, tensor) for tensor in live_inputs):
            raise ValueError("Output must not overlap weights, scales or routing")
    return MoEShape(m, k, n, e, top_k)


def scale_layout(scale: torch.Tensor, is_mxfp4: bool) -> ScaleLayout:
    shape, strides = scale.shape, scale.stride()
    version = scale._version if is_mxfp4 and not scale.is_inference() else None
    if version is not None:
        cached = _SCALE_CACHE.get(scale)
        # Rebinding .data can change shape or strides without bumping the version.
        if cached is not None and cached[:3] == (version, shape, strides):
            return cached[3]

    se, sn, sg = strides
    layout = ScaleLayout(
        (se, sg, sn),
        sum((size - 1) * stride for size, stride in zip(shape, strides)),
    )
    if version is not None:
        _SCALE_CACHE[scale] = (version, shape, strides, layout)
    return layout


def prepare_quantized_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    needs_expert_flags: bool,
    is_mxfp4: bool,
) -> Tuple[QuantizedStage, QuantizedStage]:
    b1 = (w1 if w1.dtype == torch.uint8 else w1.view(torch.uint8)).transpose(1, 2)
    b2 = (w2 if w2.dtype == torch.uint8 else w2.view(torch.uint8)).transpose(1, 2)
    s1 = w1_scale.view(torch.uint8) if is_mxfp4 else w1_scale
    s2 = w2_scale.view(torch.uint8) if is_mxfp4 else w2_scale
    if needs_expert_flags:
        safe1 = torch.empty((w1.size(0),), device=w1.device, dtype=torch.int32)
        safe2 = torch.empty_like(safe1)
    else:
        safe1, safe2 = None, None
    return (
        QuantizedStage(b1, s1, safe1, scale_layout(w1_scale, is_mxfp4)),
        QuantizedStage(b2, s2, safe2, scale_layout(w2_scale, is_mxfp4)),
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
        and 4 < tokens
        and 32 <= routes <= 256
        # Up to one route per expert on average favors direct split-K.
        and routes <= experts
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


def run_dense_batches(
    call: MarlinCall, gate_limit: int, down_limit: int, workspace_limit: int
) -> None:
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import get_default_config

    gate_tile = min(2 * call.shape.intermediate, gate_limit)
    down_tile = min(call.shape.hidden, down_limit)
    decode_elements = call.shape.experts * max(
        gate_tile * call.shape.hidden, down_tile * call.shape.intermediate
    )
    projection_width = max(2 * call.shape.intermediate, down_tile)
    tokens, k, n, experts, top_k = call.shape
    config_tokens = max(tokens, _PPU_DENSE_MIN_TOKENS)
    gate_config = get_default_config(
        config_tokens, experts, 2 * n, k, top_k, "bf16", gemm_stage="gemm1"
    )
    down_config = get_default_config(
        config_tokens, experts, k, n, top_k, "bf16", gemm_stage="gemm2"
    )
    block_m = gate_config["BLOCK_SIZE_M"]
    # Both projections consume routes grouped by the gate's row block.
    configs = (gate_config, dict(down_config, BLOCK_SIZE_M=block_m))
    can_overlap = (
        call.quant_type_id == QUANT_TYPE_FP8_E4M3
        and experts in (128, 256)
        and k == 4096
        and n == 256
        and 4 <= top_k <= 8
        and gate_tile == n
        and down_tile <= k // 2
        and k % down_tile == 0
        and down_tile % 256 == 0
    )

    def workspace_words(batch: int) -> Tuple[int, int, int, int]:
        routes = batch * top_k
        routing = sum(dense_routing_sizes(routes, experts, block_m)[1])
        if can_overlap and batch >= MIN_OVERLAP_TOKENS:
            # The tail may use ordinary GEMMs with a different expert-block count.
            routing = max(
                routing,
                sum(dense_routing_sizes(routes, experts, OVERLAP_BLOCK_ROWS)[1]),
            )
        # Each int32 word holds two BF16 elements; both projection widths are even.
        return (
            decode_elements // 2,
            routes * projection_width // 2,
            routes * n // 2,
            routing,
        )

    # Leave room for allocation and capture overhead even when a chunk fills the budget.
    headroom = min(DENSE_WORKSPACE_HEADROOM_BYTES, workspace_limit // 4)
    usable_words = (workspace_limit - headroom) // 4
    batch_tokens = tokens
    if sum(workspace_words(tokens)) > usable_words:
        low, high = 0, tokens - 1
        while low < high:
            mid = (low + high + 1) // 2
            if sum(workspace_words(mid)) <= usable_words:
                low = mid
            else:
                high = mid - 1
        batch_tokens = low
    if batch_tokens < 1:
        raise NotImplementedError("PPU dense MoE workspace exceeds its budget")
    if batch_tokens < call.shape.tokens and call.out is not call.hidden_states:
        # Both views are contiguous with equal byte length after input validation.
        distance = abs(call.out.data_ptr() - call.hidden_states.data_ptr())
        if 0 < distance < call.hidden_states.nbytes:
            raise ValueError(
                "output must not partially overlap hidden_states in chunked dense MoE"
            )
    sizes = workspace_words(batch_tokens)
    # One main-stream allocation survives every chunk. The side stream waits on
    # main before reusing it, and main joins all side-stream work before return.
    storage = call.hidden_states.new_empty(sum(sizes), dtype=torch.int32)
    banks, projections, middle, route_storage = storage.split(sizes)
    scratch = banks.view(torch.bfloat16)
    cache = projections.view(torch.bfloat16)
    activations = middle.view(torch.bfloat16)
    side = None
    if can_overlap and batch_tokens >= MIN_OVERLAP_TOKENS:
        device = call.hidden_states.device
        side = (
            torch.cuda.Stream(device=device, priority=-1)
            if torch.cuda.is_current_stream_capturing()
            else eager_decode_stream(device)
        )
    for start in range(0, call.shape.tokens, batch_tokens):
        end = min(start + batch_tokens, call.shape.tokens)
        run_dense_wna16(
            call.hidden_states[start:end],
            call.w1,
            call.w2,
            call.w1_scale,
            call.w2_scale,
            call.topk_weights[start:end],
            call.out[start:end],
            call.topk_ids[start:end],
            call.quant_type_id,
            call.apply_router_weight_on_input,
            gate_tile=gate_tile,
            down_tile=down_tile,
            scratch=scratch,
            cache=cache,
            activations=activations,
            route_storage=route_storage,
            side=side if end - start >= MIN_OVERLAP_TOKENS else None,
            configs=configs,
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
    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = stage.layout.strides
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
    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = stage.layout.strides
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
    paired_down: bool = False,
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
        PAIRED_DOWN=paired_down,
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
    paired_down: bool = False,
) -> None:
    B, B_scale = stage.weight, stage.scale
    em = sorted_token_ids.size(0)
    num_valid_tokens = output_tile.numel() // output_tile.size(-1)
    n = B.size(2)
    stride_se, stride_sg, stride_sn = stage.layout.strides
    use32 = False
    if raw_w4 or quant_type_id == QUANT_TYPE_FP8_E4M3:
        largest_offset = max(
            routed_a.nbytes - 1,
            sum((length - 1) * stride for length, stride in zip(B.shape, B.stride())),
            stage.layout.max_offset,
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
    # Adjacent K groups share an aligned 32-bit scale load.
    scale_pair = (
        quant_type_id == QUANT_TYPE_FP8_E4M3
        and routed_a.size(1) % 256 == 0
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
                and num_valid_tokens >= (3 if paired_down else 9) * B.size(0)
            ),
            USE32=use32,
            PAIRED_DOWN=paired_down,
            PACK_WIDTH=packed_width,
            SCALE_PAIR=scale_pair,
            num_warps=8,
            num_stages=1,
        )
    else:
        marlin_moe_gemm_grouped_narrow_kernel[
            (triton.cdiv(em, 8) * triton.cdiv(n, 128),)
        ](
            *tile_args,
            stage.scale_safe,
            quant_type_id == QUANT_TYPE_FP8_E4M3,
            USE32=use32,
            PAIRED_DOWN=paired_down,
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
    paired_down: bool = False,
) -> None:
    B, B_scale = stage.weight, stage.scale
    stride_se, stride_sg, stride_sn = stage.layout.strides
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
            paired_down=paired_down,
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
                2 if plan.small_fp8_gate or plan.decode_fp8_gate else None
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
            use_packed_gather=plan.raw_w4 or plan.raw_w8,
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
    paired_down = (
        plan.raw_w8
        and bm in (8, 16)
        and call.shape.intermediate == 256
        and call.shape.hidden == 4096
    )
    routed_c2 = silu_and_stage_grouped(
        c1,
        sorted_ids,
        padded,
        **staging,
        paired_down=paired_down,
        skip_unused_upper_rows=(
            plan.raw_w8 and bm == 16 and (plan.packed_fp8 or paired_down)
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
        paired_down=paired_down,
        raw_w8=plan.raw_w8,
        raw_w4=plan.raw_w4,
        packed_width=packed_width,
    )
    moe_sum(c3, call.out)


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


@lru_cache(maxsize=16)
def supports_marlin_tle(device: torch.device) -> bool:
    if tle_api is None or tle_async is None:
        return False
    features = ("load", "gpu.alloc", "gpu.local_ptr")
    with torch.cuda.device(device):
        try:
            if hasattr(tle_api, "is_tle_supported"):
                return tle_api.is_tle_supported(features)
            if hasattr(tle_api, "get_supported_primitives"):
                from triton.compiler.compiler import make_backend

                target = triton.runtime.driver.active.get_current_target()
                backend = type(make_backend(target)).__module__.split(".")[-2]
                supported = tle_api.get_supported_primitives(backend)
                return all(feature in supported for feature in features)
        except ValueError:
            # Older FlagTree versions can reject an unrecognized primitive name.
            return False
    return False


def dispatch_marlin_call(call: MarlinCall) -> None:
    with torch.cuda.device(call.hidden_states.device):
        if eligible_for_dense(call):
            # A full down tile avoids repeated strided reductions for this INT8 layout.
            down_tile = (
                call.shape.hidden
                if call.quant_type_id == QUANT_TYPE_UINT8B128
                and call.shape.hidden == 4096
                and call.shape.intermediate == 256
                else _PPU_DENSE_DOWN_TILE
            )
            run_dense_batches(
                call,
                _PPU_DENSE_GATE_TILE,
                down_tile,
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
    if not supports_marlin_tle(hidden_states.device):
        raise NotImplementedError(
            "PPU fused Marlin MoE requires TLE load, gpu.alloc and gpu.local_ptr"
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
    if isinstance(activation, Enum):
        if (
            activation.name.lower() != "silu"
            and str(activation.value).lower() != "silu"
        ):
            raise NotImplementedError("PPU fused Marlin MoE supports only SiLU")
    elif activation is not None and str(activation).lower() != "silu":
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
    if global_num_experts != -1 and (w1.ndim != 3 or global_num_experts != w1.shape[0]):
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
