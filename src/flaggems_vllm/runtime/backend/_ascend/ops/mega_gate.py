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

"""Fused router projection and expert selection on Ascend."""

import math

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _promote_activation_kernel(src, dst, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(src + offsets, offsets < count, other=0)
    tl.store(dst + offsets, values.to(tl.float32), offsets < count)


def _promote_activation(x, out):
    count = x.numel()
    _promote_activation_kernel[(triton.cdiv(count, 2048),)](
        x,
        out,
        count,
        BLOCK=2048,
        use_bytecode=False,
    )
    return out


@triton.jit
def _project_router_tile(
    x_ptr,
    weight_ptr,
    rows,
    columns,
    tokens,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
    for chunk in range(HIDDEN // BLOCK_K):
        reduction = chunk * BLOCK_K + k
        a = tl.load(
            x_ptr + rows[:, None] * HIDDEN + reduction[None, :],
            rows[:, None] < tokens,
            other=0,
        )
        b = tl.load(weight_ptr + columns[None, :] * HIDDEN + reduction[:, None])
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")
    return acc


@triton.jit
def _mega_gate_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    input_ids_ptr,
    table_ptr,
    vision_bias_ptr,
    workspace_ptr,
    indices_ptr,
    weights_ptr,
    tokens,
    scale,
    IMAGE_LO: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAD_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_ROWS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    HARDWARE_TOPK: tl.constexpr,
    USE_BOTH_SUBCORES: tl.constexpr,
    PROJECTION_SPLITS: tl.constexpr,
    TAIL_WINDOW: tl.constexpr,
    ROUND_SPLIT_LOGITS: tl.constexpr,
):
    block = tl.program_id(0)
    workers = tl.num_programs(0)
    expert_tiles: tl.constexpr = EXPERTS // BLOCK_N
    row_tiles = tl.cdiv(tokens, BLOCK_M)
    # Keeping the reduction inline preserves the compiled projection used by
    # the short and mid-length 910B routes.
    k = tl.arange(0, BLOCK_K)
    plane_stride = row_tiles * BLOCK_M * PAD_EXPERTS
    for job in range(block, row_tiles * expert_tiles * PROJECTION_SPLITS, workers):
        task = job // PROJECTION_SPLITS
        plane = job % PROJECTION_SPLITS
        rows = task // expert_tiles * BLOCK_M + tl.arange(0, BLOCK_M)
        if TAIL_WINDOW:
            start = tl.minimum(task // expert_tiles * BLOCK_M, tokens - BLOCK_M)
            load_rows = start + tl.arange(0, BLOCK_M)
        columns = task % expert_tiles * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
        for chunk in range(HIDDEN // (PROJECTION_SPLITS * BLOCK_K)):
            reduction = plane * (HIDDEN // PROJECTION_SPLITS) + chunk * BLOCK_K + k
            if TAIL_WINDOW:
                a = tl.load(x_ptr + load_rows[:, None] * HIDDEN + reduction[None, :])
            else:
                a = tl.load(
                    x_ptr + rows[:, None] * HIDDEN + reduction[None, :],
                    rows[:, None] < tokens,
                    other=0,
                )
            b = tl.load(weight_ptr + columns[None, :] * HIDDEN + reduction[:, None])
            acc = tl.dot(a, b, acc=acc, input_precision="ieee")
        plane_ptr = workspace_ptr + plane * plane_stride
        if TAIL_WINDOW and PROJECTION_SPLITS > 1:
            # Keep the partial planes row-major. The prefix of the shifted
            # tile was already computed by the preceding projection tile.
            tl.store(
                plane_ptr + load_rows[:, None] * PAD_EXPERTS + columns[None, :],
                acc,
                load_rows[:, None] >= task // expert_tiles * BLOCK_M,
            )
        else:
            tl.store(plane_ptr + rows[:, None] * PAD_EXPERTS + columns[None, :], acc)

    # All projection tiles must be visible before a vector worker reads a row.
    tle.dsa.ascend.sync_block_all("all", 0)

    if USE_BOTH_SUBCORES:
        route_worker = 2 * block + tle.dsa.ascend.sub_vec_id()
        route_workers = 2 * workers
    else:
        route_worker = block
        route_workers = workers
    for task in range(route_worker, tl.cdiv(tokens, ROUTE_ROWS), route_workers):
        rows = task * ROUTE_ROWS + tl.arange(0, ROUTE_ROWS)
        _route_rows(
            workspace_ptr,
            bias_ptr,
            input_ids_ptr,
            table_ptr,
            vision_bias_ptr,
            IMAGE_LO,
            indices_ptr,
            weights_ptr,
            rows,
            tokens,
            scale,
            HAS_BIAS,
            EXPERTS,
            PAD_EXPERTS,
            PAD_EXPERTS,
            TOPK,
            SCORING,
            GROUPS,
            TOP_GROUPS,
            ROUTE_ROWS,
            BLOCK_TOPK,
            HARDWARE_TOPK,
            USE_BOTH_SUBCORES,
            PROJECTION_SPLITS,
            plane_stride,
            TAIL_WINDOW if PROJECTION_SPLITS == 1 else 0,
            ROUND_SPLIT_LOGITS,
        )


@triton.jit
def _mega_gate_pipelined_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    input_ids_ptr,
    table_ptr,
    vision_bias_ptr,
    workspace_ptr,
    indices_ptr,
    weights_ptr,
    tokens,
    scale,
    IMAGE_LO: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAD_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_ROWS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    HARDWARE_TOPK: tl.constexpr,
    USE_BOTH_SUBCORES: tl.constexpr,
):
    block = tl.program_id(0)
    workers = tl.num_programs(0)
    row_tiles = tl.cdiv(tokens, BLOCK_M)
    sub = tle.dsa.ascend.sub_vec_id()
    for tile in range(block, row_tiles, workers):
        odd = (tile // workers) % 2 != 0
        # Each pair owns complete rows. Acknowledge slot reuse before the Cube
        # can issue a third ready signal; the final two tiles need no ack.
        if tile >= 2 * workers:
            if odd:
                tle.dsa.ascend.sync_block_wait("vector", "cube", 5)
            else:
                tle.dsa.ascend.sync_block_wait("vector", "cube", 4)
        project_rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        for n_tile in range(EXPERTS // BLOCK_N):
            columns = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = _project_router_tile(
                x_ptr,
                weight_ptr,
                project_rows,
                columns,
                tokens,
                HIDDEN,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )
            tl.store(
                workspace_ptr + project_rows[:, None] * EXPERTS + columns[None, :], acc
            )
        if odd:
            tle.dsa.ascend.sync_block_set("cube", "vector", 3)
            tle.dsa.ascend.sync_block_wait("cube", "vector", 3)
        else:
            tle.dsa.ascend.sync_block_set("cube", "vector", 2)
            tle.dsa.ascend.sync_block_wait("cube", "vector", 2)
        for stripe in range(sub, BLOCK_M // ROUTE_ROWS, 2):
            rows = tile * BLOCK_M + stripe * ROUTE_ROWS + tl.arange(0, ROUTE_ROWS)
            _route_rows(
                workspace_ptr,
                bias_ptr,
                input_ids_ptr,
                table_ptr,
                vision_bias_ptr,
                IMAGE_LO,
                indices_ptr,
                weights_ptr,
                rows,
                tokens,
                scale,
                HAS_BIAS,
                EXPERTS,
                PAD_EXPERTS,
                EXPERTS,
                TOPK,
                SCORING,
                GROUPS,
                TOP_GROUPS,
                ROUTE_ROWS,
                BLOCK_TOPK,
                HARDWARE_TOPK,
                USE_BOTH_SUBCORES,
            )
        if tile + 2 * workers < row_tiles:
            if odd:
                tle.dsa.ascend.sync_block_set("vector", "cube", 5)
            else:
                tle.dsa.ascend.sync_block_set("vector", "cube", 4)


@triton.jit
def _mask_unselected_groups(
    ranking,
    ROWS: tl.constexpr,
    EXPERTS: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
):
    group_width: tl.constexpr = EXPERTS // GROUPS
    grouped = tl.reshape(ranking, (ROWS, GROUPS, group_width))
    first_value, first = tl.max(grouped, axis=2, return_indices=True)
    lanes = tl.arange(0, group_width)
    remaining = tl.where(
        lanes[None, None, :] == first[:, :, None], -float("inf"), grouped
    )
    scores = first_value + tl.max(remaining, axis=2)
    groups = tl.arange(0, GROUPS)
    selected = tl.full((ROWS, GROUPS), False, tl.int1)
    for _ in tl.static_range(TOP_GROUPS):
        winner = tl.argmax(scores, axis=1, tie_break_left=True)
        picked = groups[None, :] == winner[:, None]
        selected |= picked
        scores = tl.where(picked, -float("inf"), scores)
    return tl.reshape(
        tl.where(selected[:, :, None], grouped, -float("inf")), (ROWS, EXPERTS)
    )


@triton.jit
def _mask_sorted_groups(
    ranking, EXPERTS: tl.constexpr, GROUPS: tl.constexpr, TOP_GROUPS: tl.constexpr
):
    # Each 32-expert group maps to one hardware sort. Only its first two
    # scores are needed, so the interleaved index payload is unused here.
    scores = tl.reshape(ranking, (EXPERTS,))
    buffer = tle.dsa.alloc(
        [2 * EXPERTS], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
    )
    packed = tle.dsa.ascend.raw(
        "sort32",
        scores,
        scores.to(tl.uint32, bitcast=True),
        GROUPS,
        out=tle.dsa.to_tensor(buffer, writable=True),
    )
    groups = tl.arange(0, GROUPS)
    group_scores = tl.gather(packed, groups * 64, axis=0) + tl.gather(
        packed, groups * 64 + 2, axis=0
    )
    ahead = (group_scores[:, None] > group_scores[None, :]) | (
        (group_scores[:, None] == group_scores[None, :])
        & (groups[:, None] < groups[None, :])
    )
    selected = tl.reshape(tl.sum(ahead.to(tl.int32), axis=0) < TOP_GROUPS, (1, GROUPS))
    grouped = tl.reshape(ranking, (1, GROUPS, 32))
    return tl.reshape(
        tl.where(selected[:, :, None], grouped, -float("inf")), (1, EXPERTS)
    )


@triton.jit
def _route_rows(
    workspace_ptr,
    bias_ptr,
    input_ids_ptr,
    table_ptr,
    vision_bias_ptr,
    IMAGE_LO: tl.constexpr,
    indices_ptr,
    weights_ptr,
    rows,
    tokens,
    scale,
    HAS_BIAS: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAD_EXPERTS: tl.constexpr,
    LOGIT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    HARDWARE_TOPK: tl.constexpr,
    USE_BOTH_SUBCORES: tl.constexpr,
    PROJECTION_SPLITS: tl.constexpr = 1,
    plane_stride=0,
    TAIL_WINDOW: tl.constexpr = 0,
    ROUND_SPLIT_LOGITS: tl.constexpr = False,
):
    columns = tl.arange(0, PAD_EXPERTS)
    logit_rows = rows
    if TAIL_WINDOW:
        # The last Cube tile reads a full contiguous window ending at T.
        # Only its suffix belongs to this tile's output rows.
        slack = tl.cdiv(tokens, TAIL_WINDOW) * TAIL_WINDOW - tokens
        logit_rows = tl.where(
            rows >= tokens // TAIL_WINDOW * TAIL_WINDOW,
            tl.minimum(rows, tokens - 1) + slack,
            rows,
        )
    logits_ptr = workspace_ptr + logit_rows[:, None] * LOGIT_STRIDE + columns[None, :]
    if PAD_EXPERTS == EXPERTS:
        logits = tl.load(logits_ptr)
    else:
        logits = tl.load(logits_ptr, columns[None, :] < EXPERTS, other=-float("inf"))

    # Split projections store FP32 partials. Sum them before scoring or bias.
    for plane in tl.static_range(1, PROJECTION_SPLITS):
        ptr = logits_ptr + plane * plane_stride
        if PAD_EXPERTS == EXPERTS:
            logits += tl.load(ptr)
        else:
            logits += tl.load(ptr, columns[None, :] < EXPERTS, other=0)
    if ROUND_SPLIT_LOGITS:
        # Match the model's BF16 linear result after summing both FP32 planes.
        logits = logits.to(tl.bfloat16)
    logits = logits.to(tl.float32)
    RANK_LOGITS: tl.constexpr = (
        SCORING == "softmax" and not HAS_BIAS and TOP_GROUPS == GROUPS
    )
    KEEP_SELECTED_LOGITS: tl.constexpr = (
        USE_BOTH_SUBCORES
        and RANK_LOGITS
        and PAD_EXPERTS >= 512
        and ROWS * PAD_EXPERTS >= 8192
    )
    if SCORING == "sqrtsoftplus":
        e = tl.exp(-tl.abs(logits))
        # The small-e branch avoids cancellation in log(1 + e).
        log1p = tl.where(e < 0.001, e * (1.0 - 0.5 * e), tl.log(1.0 + e))
        raw_scores = tl.sqrt(tl.maximum(logits, 0.0) + log1p)
    elif SCORING == "sigmoid":
        raw_scores = 1.0 / (1.0 + tl.exp(-logits))
    elif RANK_LOGITS:
        # Softmax preserves order and its common denominator cancels on renorm.
        raw_scores = logits
    else:
        exp_logits = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        raw_scores = exp_logits / tl.sum(exp_logits, axis=1)[:, None]
    slots = tl.arange(0, BLOCK_TOPK)
    indices = tl.full((ROWS, BLOCK_TOPK), 0, tl.int32)
    if input_ids_ptr is not None:
        token_ids = tl.load(input_ids_ptr + rows, rows < tokens, other=0)
        # The V4 router maps sequence-parallel padding (-1) to token zero.
        token_ids = tl.where(token_ids == -1, 0, token_ids)
    if vision_bias_ptr is not None:
        image = (token_ids >= IMAGE_LO) & (token_ids < IMAGE_LO + 5)
    if table_ptr is None or vision_bias_ptr is not None:
        ranking = raw_scores
        if HAS_BIAS:
            if PAD_EXPERTS == EXPERTS:
                bias = tl.load(bias_ptr + columns)
            else:
                bias = tl.load(bias_ptr + columns, columns < EXPERTS, other=0)
            ranking = ranking + bias[None, :]
        if vision_bias_ptr is not None:
            vision_bias = tl.load(vision_bias_ptr + columns, columns < EXPERTS, other=0)
            ranking = tl.where(
                image[:, None], raw_scores + vision_bias[None, :], ranking
            )
        if PAD_EXPERTS != EXPERTS:
            ranking = tl.where(columns[None, :] < EXPERTS, ranking, -float("inf"))
        if TOP_GROUPS < GROUPS and not HARDWARE_TOPK:
            ranking = _mask_unselected_groups(
                ranking, ROWS, EXPERTS, GROUPS, TOP_GROUPS
            )

        if HARDWARE_TOPK:
            # Each sort returns interleaved scores and expert IDs. A batched
            # Vector tile shares the bias load and writes consecutive outputs.
            scratch = tl.zeros((4 * PAD_EXPERTS,), tl.float32)
            proposals = tle.dsa.alloc(
                [32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB
            )
            for row in tl.static_range(ROWS):
                if ROWS == 1:
                    row_ranks = tl.reshape(ranking, (PAD_EXPERTS,))
                else:
                    row_ranks = tl.gather(
                        tl.reshape(ranking, (ROWS * PAD_EXPERTS,)),
                        row * PAD_EXPERTS + columns,
                        axis=0,
                    )
                if TOP_GROUPS < GROUPS:
                    tl.static_assert(EXPERTS == GROUPS * 32)
                    row_ranks = tl.reshape(
                        _mask_sorted_groups(row_ranks, EXPERTS, GROUPS, TOP_GROUPS),
                        (EXPERTS,),
                    )
                packed = tle.dsa.ascend.raw(
                    "sort_1d_pack",
                    row_ranks,
                    scratch,
                    True,
                    16,
                    0,
                    0,
                    out=tle.dsa.to_tensor(proposals, writable=True),
                )
                ids = tl.gather(packed, 2 * slots + 1, axis=0).to(
                    tl.int32, bitcast=True
                )
                if ROWS == 1:
                    indices = tl.reshape(ids, (1, BLOCK_TOPK))
                else:
                    indices = tl.where(
                        tl.arange(0, ROWS)[:, None] == row, ids[None, :], indices
                    )
        else:
            if KEEP_SELECTED_LOGITS:
                selected_logits = tl.full((ROWS, BLOCK_TOPK), 0.0, tl.float32)
            for slot in tl.static_range(TOPK):
                if KEEP_SELECTED_LOGITS:
                    value, winner = tl.max(ranking, axis=1, return_indices=True)
                    selected_logits = tl.where(
                        slots[None, :] == slot, value[:, None], selected_logits
                    )
                else:
                    winner = tl.argmax(ranking, axis=1, tie_break_left=True)
                ranking = tl.where(
                    columns[None, :] == winner[:, None], -float("inf"), ranking
                )
                indices = tl.where(slots[None, :] == slot, winner[:, None], indices)

    if table_ptr is not None:
        lookup_row = rows < tokens
        if vision_bias_ptr is not None:
            lookup_row &= ~image
        lookup_ids = tl.load(
            table_ptr + token_ids[:, None].to(tl.int64) * TOPK + slots[None, :],
            lookup_row[:, None] & (slots[None, :] < TOPK),
            other=0,
        ).to(tl.int32)
        indices = tl.where(lookup_row[:, None], lookup_ids, indices)

    if KEEP_SELECTED_LOGITS and not HARDWARE_TOPK:
        weights = selected_logits
    else:
        weights = tl.gather(raw_scores, indices, axis=1)
    if RANK_LOGITS:
        picked_logits = tl.where(slots[None, :] < TOPK, weights, -float("inf"))
        weights = tl.exp(picked_logits - tl.max(picked_logits, axis=1)[:, None])
    weights = tl.where(slots[None, :] < TOPK, weights, 0.0)
    total = tl.sum(weights, axis=1)

    output = rows[:, None] * TOPK + slots[None, :]
    mask = (rows[:, None] < tokens) & (slots[None, :] < TOPK)
    tl.store(indices_ptr + output, indices, mask)
    tl.store(weights_ptr + output, weights * (scale / (total[:, None] + 1e-20)), mask)


def _validate_routing(topk, groups, top_groups, logits_dtype):
    assert logits_dtype in (
        torch.float32,
        torch.bfloat16,
    ), "logits_dtype must be FP32 or BF16"
    assert (
        type(topk) is type(groups) is type(top_groups) is int
    ), "topk, groups and top_groups must be integers"


def _validate_gate_tensors(x, weight, bias):
    device = x.device
    assert (
        weight.device == device
        and x.dtype in (torch.bfloat16, torch.float32)
        and (weight.dtype == torch.float32 or weight.dtype == x.dtype)
        and x.is_contiguous()
        and weight.is_contiguous()
    ), "x and weight must be contiguous BF16 or FP32 on the same NPU, with FP32 weights for FP32 x"
    assert bias is None or (
        bias.shape == (weight.shape[0],)
        and bias.dtype == torch.float32
        and bias.device == device
        and bias.is_contiguous()
    ), "bias must be contiguous FP32 [E] on the same NPU"


def _validate_v4_inputs(x, weight, input_ids, tid2eid, bias_vl, config):
    topk, scoring, groups, top_groups, _, image_lo, _, _ = config
    if tid2eid is None and bias_vl is None:
        assert input_ids is None, "input_ids require hash or vision routing"
        return
    assert (
        scoring == "sqrtsoftplus" and groups == top_groups == 1
    ), "V4 routing requires flat sqrtsoftplus"
    assert input_ids is not None and (
        input_ids.shape == (x.shape[0],)
        and input_ids.dtype in (torch.int32, torch.int64)
        and input_ids.device == x.device
        and input_ids.is_contiguous()
    ), "input_ids must be contiguous integer [T] on the same NPU"
    if tid2eid is not None:
        assert (
            tid2eid.ndim == 2
            and tid2eid.shape[0] > 0
            and tid2eid.shape[1] == topk
            and tid2eid.dtype in (torch.int32, torch.int64)
            and tid2eid.device == x.device
            and tid2eid.is_contiguous()
        ), "tid2eid must be contiguous integer [vocab, K] on the same NPU"
    if bias_vl is not None:
        assert type(image_lo) is int and image_lo >= 0, "invalid image_sentinel_lo"
        assert (
            bias_vl.shape == (weight.shape[0],)
            and bias_vl.dtype == torch.float32
            and bias_vl.device == x.device
            and bias_vl.is_contiguous()
        ), "bias_vl must be contiguous FP32 [E] on the same NPU"


def _validate_gate(x, weight, bias, topk, scoring, groups, top_groups, scale):
    assert (
        x.ndim == weight.ndim == 2 and x.shape[1] == weight.shape[1]
    ), "x and weight must be [T, H] and [E, H]"
    experts, hidden = weight.shape
    assert (
        hidden > 0 and hidden % 128 == 0 and 16 <= experts <= 512 and experts % 16 == 0
    ), "mega_gate requires H aligned to 128 and E aligned to 16 in [16, 512]"
    assert scoring in (
        "sqrtsoftplus",
        "sigmoid",
        "softmax",
    ), f"Unsupported MegaGate scoring: {scoring}"
    assert 1 <= topk <= 16, "topk must be between 1 and 16"
    assert (
        groups > 0 and experts % groups == 0 and 1 <= top_groups <= groups
    ), "groups must divide experts and contain top_groups"
    if top_groups < groups:
        group_width = experts // groups
        assert (
            topk <= top_groups * group_width
        ), "selected groups must contain at least topk experts"
        assert (
            experts & (experts - 1) == 0 and groups <= 16 and group_width >= 2
        ), "grouped selection requires power-of-two E, at most 16 groups and group width >= 2"
    assert isinstance(scale, (float, int)) and math.isfinite(
        scale
    ), "scale must be finite"
    assert x.device.type == "npu", "MegaGate requires NPU inputs"
    _validate_gate_tensors(x, weight, bias)


def _gate_launch_args(
    x, weight, bias, topk, scoring, groups, top_groups, routing_inputs, logits_dtype
):
    tokens, hidden = x.shape
    experts = weight.shape[0]
    has_bias = bias is not None
    grouped_route = top_groups < groups
    rank_logits = scoring == "softmax" and not has_bias and not grouped_route
    cube_cores = driver.active.utils.get_aicore_num()
    expert_alignment = experts & -experts
    # Wider projections amortize weight reads over 128 rows once there are
    # enough rows to distribute at least four tiles.
    wide_projection = experts >= 512 and tokens >= 4 * 128
    wide_row_tiles = triton.cdiv(tokens, 128)
    weight_bytes = hidden * experts * weight.element_size()
    # The small weight footprint can be reused across a wider row tile.
    long_narrow_projection = (
        rank_logits
        and experts <= 256
        and weight_bytes <= 1024 * 1024
        and tokens >= 2048
        and (wide_row_tiles <= cube_cores or wide_row_tiles > 3 * cube_cores)
    )
    max_block_m = 128 if wide_projection or long_narrow_projection else 64
    block_m = min(max_block_m, max(16, triton.next_power_of_2(triton.cdiv(tokens, 2))))
    if tokens == 1:
        block_m = 1
    row_tiles = triton.cdiv(tokens, block_m)
    # Keep every program resident across the projection/routing barrier.
    target_splits = max(1, cube_cores // row_tiles)
    padded_experts = triton.next_power_of_2(experts)
    splits = min(padded_experts // 16, 1 << (target_splits.bit_length() - 1))
    # Use a power-of-two divisor so every weight tile has all its columns.
    block_n = min(256, padded_experts // splits, expert_alignment)
    if long_narrow_projection and row_tiles > cube_cores:
        block_n = min(block_n, 128)
    short_projection = (
        tokens <= 16
        and hidden <= 4096
        and (rank_logits or (tokens > 1 and scoring == "sqrtsoftplus"))
    )
    if short_projection:
        # Fewer, wider weight tiles and a single buffer reduce the setup
        # cost when the projection has at most one short row tile.
        short_block_n = 64 if rank_logits else 32
        block_n = max(block_n, min(short_block_n, expert_alignment))
    workers = min(cube_cores, row_tiles * (experts // block_n))
    # Preload the 64 KiB weight tiles when the short projection feeds
    # flat sqrtsoftplus routing. The same tiles regressed other route paths.
    short_wave_preload = (
        scoring == "sqrtsoftplus"
        and not grouped_route
        and block_m == 16
        and row_tiles == 2
        and block_n == 32
        and hidden % 1024 == 0
    )
    if short_projection or short_wave_preload:
        block_k = 1024
    elif block_m == 1:
        block_k = 2048
    elif block_m >= 64:
        # Bound each wide tile's weight operand to 128 KiB.
        block_k = min(1024, 65536 // block_n)
        if block_m + block_n >= 128 or hidden <= 2048:
            block_k = min(512, block_k)
    else:
        block_k = 512
    if weight.dtype == torch.float32:
        block_k = min(block_k, 32768 // block_n, 256)
    block_k = min(block_k, hidden & -hidden)
    if row_tiles == 1:
        route_rows = min(4, triton.next_power_of_2(tokens))
    else:
        rows_per_worker = triton.next_power_of_2(triton.cdiv(tokens, workers))
        route_elements = 4096 if row_tiles > cube_cores else 2048
        max_route_rows = 16
        if rank_logits:
            # Ranking logits directly leaves room for a larger Vector tile.
            route_elements = 8192
            max_route_rows = 32
        elif grouped_route:
            route_elements = 4096
        route_rows = min(
            route_elements // padded_experts, max_route_rows, rows_per_worker, block_m
        )
    route_waves = triton.cdiv(triton.cdiv(tokens, route_rows), workers)
    parallel_flat_route = (
        scoring in ("sigmoid", "sqrtsoftplus") and not grouped_route and route_waves > 1
    )
    widen_flat_route = parallel_flat_route and (scoring == "sigmoid" or route_waves > 2)
    use_both_subcores = (
        (rank_logits and row_tiles >= 3)
        or (grouped_route and (route_waves > 1 or route_rows >= 16))
        or parallel_flat_route
    )
    if widen_flat_route:
        # Reuse the bias and scoring setup across a full Vector tile.
        route_rows = min(4096 // padded_experts, 16, block_m)
    elif use_both_subcores:
        # Each AIV handles its own rows. Halve the tile when that removes
        # padded rows on the busiest AIV, and keep the wider tile on a tie.
        rows_per_vector = triton.cdiv(tokens, 2 * workers)
        half_rows = route_rows // 2
        if (
            half_rows
            and triton.cdiv(rows_per_vector, half_rows) * half_rows
            < triton.cdiv(rows_per_vector, route_rows) * route_rows
        ):
            route_rows = half_rows
    _, table, vision_bias = routing_inputs
    if table is not None or vision_bias is not None:
        # Additional per-row indirect loads and vision rankings share the AIV
        # local buffer. The combined route needs room for both live sets.
        max_rows = 2 if table is not None and vision_bias is not None else 4
        route_rows = min(route_rows, max_rows, max(1, 1024 // padded_experts))
        use_both_subcores = triton.cdiv(tokens, route_rows) > workers
    wide_sigmoid = (
        weight.dtype == torch.bfloat16
        and logits_dtype == torch.float32
        and hidden == 7168
        and scoring == "sigmoid"
        and has_bias
        and topk == 8
        and (
            (experts == 256 and groups == 8 and top_groups == 4)
            or (experts == 384 and groups == top_groups == 1)
        )
    )
    custom_ops = getattr(tle.dsa.ascend, "custom_ops", None)
    split_group = (
        wide_sigmoid
        and grouped_route
        and 16 < tokens < 2 * cube_cores * 64
        and hasattr(custom_ops, "sort32")
        and hasattr(custom_ops, "sort_1d_pack")
    )
    split_flat = (
        wide_sigmoid
        and not grouped_route
        and (64 < tokens <= 128 or 512 < tokens < 2 * cube_cores * 64)
        and hasattr(custom_ops, "sort_1d_pack")
    )
    split_sigmoid = split_group or split_flat
    short_pipeline = (
        wide_sigmoid
        and not split_sigmoid
        and cube_cores * 64 <= tokens < 2 * cube_cores * 64
    )
    local_pipeline = (
        weight.dtype == torch.bfloat16
        and experts >= 256
        and expert_alignment >= 128
        and topk <= 10
        and (tokens >= 2 * cube_cores * 64 or short_pipeline)
        and table is None
        and vision_bias is None
        and (
            rank_logits
            or (scoring == "sigmoid" and (not grouped_route or experts == 256))
        )
    )
    fp32_router = (
        weight.dtype == torch.float32
        and logits_dtype == torch.float32
        and hidden == 4096
        and experts == 256
        and topk == 6
        and scoring == "sqrtsoftplus"
        and groups == top_groups == 1
    )
    fp32_score = fp32_router and table is None and vision_bias is None
    fp32_pipeline = fp32_router and tokens >= cube_cores * 64
    if fp32_pipeline:
        # Each Cube owns a full expert row. Its Vector subcores rank rows
        # while the Cube projects the next tile.
        block_m, block_n, block_k = 64, experts, 128
        route_rows = 8 if vision_bias is None else (2 if table is not None else 4)
        if fp32_score and 96 * triton.cdiv(tokens, 96 * cube_cores) < 64 * triton.cdiv(
            tokens, 64 * cube_cores
        ):
            block_m, route_rows = 96, 16
    elif local_pipeline:
        block_m = (
            128 if tokens >= 8 * cube_cores * 64 and experts != padded_experts else 64
        )
        block_n = min(expert_alignment, 128)
        block_k = min(512, hidden & -hidden)
        route_rows = 16 if rank_logits or padded_experts <= 256 else 8
        if short_pipeline:
            block_m = 48 if grouped_route and tokens % 64 else 64
            route_rows = 8
        if wide_sigmoid and experts == 384 and tokens < 8 * cube_cores * 64:
            padded_experts = experts
    local_pipeline = local_pipeline or fp32_pipeline
    if local_pipeline:
        row_tiles = triton.cdiv(tokens, block_m)
        workers = min(cube_cores, row_tiles)
        use_both_subcores = True
    # A row sort is cheap for wide expert sets while each AIV owns a
    # small batch. The repeated reductions amortize better on longer runs.
    sort_min = 64 if experts == 256 else 24
    sort_max = 256 if experts == 256 else 2 * cube_cores * 32
    flat_sort = (
        weight.dtype == torch.bfloat16
        and not grouped_route
        and not local_pipeline
        and (rank_logits or (scoring == "sigmoid" and experts == 384))
        and experts in (256, 384, 512)
        and sort_min <= tokens <= sort_max
        and table is None
        and vision_bias is None
    )
    group_sort = (
        wide_sigmoid
        and grouped_route
        and tokens <= 320
        and hasattr(custom_ops, "sort32")
    )
    hardware_topk = (flat_sort or group_sort) and hasattr(custom_ops, "sort_1d_pack")
    if hardware_topk:
        route_rows, use_both_subcores = 1, True
        if group_sort and 4 < tokens <= 16:
            block_m, block_n, block_k = 16, 32, 1024
    tail_window = 0
    if (
        wide_sigmoid
        and grouped_route
        and not split_group
        and 320 < tokens <= 512
        and tokens % 64
        and tokens % 16 == 0
    ):
        block_m, block_n, block_k = 64, 128, 512
        route_rows, use_both_subcores = 16, True
        tail_window = block_m
    projection_splits = 1
    if wide_sigmoid and not grouped_route and hardware_topk:
        if tokens <= 32:
            block_m, block_n, block_k = 32, 64, 1024
        elif 128 < tokens < cube_cores * 64 and (tokens % 128 or tokens > 384):
            # Splitting K spreads narrow expert projections over more Cubes.
            # Window the last activation tile so both halves can use dense loads.
            block_m = 128 if tokens % 128 else 64
            block_n, block_k, projection_splits = 128, 512, 2
            tail_window = block_m if tokens % block_m else 0
    if wide_sigmoid and not split_group and 512 < tokens < cube_cores * 64:
        if grouped_route:
            block_m, block_n, block_k = 256, 64, 256
            hardware_topk = hasattr(custom_ops, "sort32") and hasattr(
                custom_ops, "sort_1d_pack"
            )
            route_rows = 2 if hardware_topk else 16
        elif hardware_topk:
            route_rows = 4
        use_both_subcores = True
    logit_stride = experts if local_pipeline else padded_experts
    compile_options = {}
    if short_projection:
        compile_options["multibuffer"] = False
    elif short_wave_preload:
        compile_options["enable_preload"] = True
    # A 32-expert weight tile fits a long K block with one buffer for decode.
    if (
        tokens <= 4
        and (
            (
                weight.dtype == torch.float32
                and scoring == "sqrtsoftplus"
                and not grouped_route
            )
            or (weight.dtype == torch.bfloat16 and grouped_route)
        )
        and table is None
        and vision_bias is None
        and hidden % 1024 == 0
        and experts % 32 == 0
    ):
        block_n, route_rows = 32, 1
        block_k = 512 if grouped_route and tokens == 1 else 1024
        use_both_subcores = hardware_topk
        compile_options["multibuffer"] = False
    if (
        (fp32_score or split_sigmoid)
        and not local_pipeline
        and hasattr(custom_ops, "sort_1d_pack")
    ):
        hardware_topk = True
        use_both_subcores = True
        route_rows = 1 if tokens == 1 or 16 < tokens <= 64 else 2
        if tokens > 16:
            # Additional K partitions occupy otherwise idle Cubes on short
            # projections. Larger row tiles already expose enough parallel work.
            block_m, block_n = (32, 64) if tokens <= 64 else (64, 128)
            block_k = 256
            projection_splits = 2 if split_sigmoid or tokens <= 256 else 1
            if (split_group and tokens > 256) or (split_flat and tokens > 512):
                block_m, block_n, block_k = 128, 128, 512
            tail_window = block_m if tokens >= block_m and tokens % block_m else 0
    qwen_router = (
        rank_logits
        and weight.dtype == torch.bfloat16
        and logits_dtype == torch.bfloat16
        and (hidden, experts, topk) in ((2048, 256, 8), (4096, 512, 10))
    )
    if (
        qwen_router
        and not local_pipeline
        and tokens <= 2048
        and hasattr(custom_ops, "sort_1d_pack")
    ):
        hardware_topk = use_both_subcores = True
        route_rows = min(2, block_m)
        if tokens > 16:
            wide_tile = experts == 512 and tokens > 256
            block_m, block_n, block_k = (128, 128, 512) if wide_tile else (64, 128, 256)
            short_limit = 128 if experts == 256 else 64
            projection_splits = (
                2 if tokens <= short_limit or wide_tile or tokens > 512 else 1
            )
            tail_window = block_m if tokens >= block_m and tokens % block_m else 0
    if not local_pipeline:
        row_tiles = triton.cdiv(tokens, block_m)
        workers = min(cube_cores, row_tiles * (experts // block_n) * projection_splits)
        compile_options["ROUND_SPLIT_LOGITS"] = (
            projection_splits > 1 and logits_dtype == torch.bfloat16
        )
        compile_options["PROJECTION_SPLITS"] = projection_splits
        compile_options["TAIL_WINDOW"] = tail_window
    # BF16 logits round at this store. Split projections use FP32 planes,
    # which the Vector pass combines before scoring.
    workspace = torch.empty(
        (projection_splits * row_tiles * block_m, logit_stride),
        dtype=torch.float32 if projection_splits > 1 else logits_dtype,
        device=x.device,
    )
    return (
        _mega_gate_pipelined_kernel if local_pipeline else _mega_gate_kernel,
        (workers,),
        workspace,
        dict(
            HAS_BIAS=has_bias,
            HIDDEN=hidden,
            EXPERTS=experts,
            PAD_EXPERTS=padded_experts,
            TOPK=topk,
            SCORING=scoring,
            GROUPS=groups,
            TOP_GROUPS=top_groups,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            ROUTE_ROWS=route_rows,
            BLOCK_TOPK=triton.next_power_of_2(topk),
            HARDWARE_TOPK=hardware_topk,
            USE_BOTH_SUBCORES=use_both_subcores,
            enable_auto_bind_sub_block=False if use_both_subcores else None,
            enable_sync_block_lock=not hardware_topk,
            disable_auto_inject_block_sync=True,
            use_bytecode=False,
            **compile_options,
        ),
    )


class _PreparedMegaGate:
    def __init__(self, x, weight, bias, routing_inputs, config):
        import torch_npu

        (
            topk,
            scoring,
            groups,
            top_groups,
            scale,
            image_lo,
            logits_dtype,
            output_dtypes,
        ) = config
        tokens = x.shape[0]
        self.output_dtypes = _resolve_output_dtypes(weight, output_dtypes)
        # Expanded scalar templates describe the two fresh, contiguous outputs
        # without retaining full-sized output buffers in a prepared handle.
        self.output_templates = tuple(
            x.new_empty((1,), dtype=dtype).expand(tokens, topk)
            for dtype in self.output_dtypes
        )
        self.config = config
        self.x_shape = x.shape
        self.weight_shape = weight.shape
        self.bias_shape = None if bias is None else bias.shape
        tensors = (x, weight, bias, *routing_inputs)
        self.dtypes = tuple(t.dtype if t is not None else None for t in tensors)
        self.input_dtypes = self.dtypes[:3]
        self.routing_dtypes = self.dtypes[3:]
        self.device = x.device
        # Resolve the same raw-stream accessor as the Ascend Triton driver once.
        self.get_stream = getattr(
            torch_npu._C,
            "_npu_getCurrentRawStreamNoWait",
            driver.active.get_current_stream,
        )
        self.stream = self.get_stream(x.device.index)
        self.aligned_inputs = tuple(
            tensor is not None and tensor.data_ptr() % 16 == 0 for tensor in tensors
        )
        self.fp32_input = None
        self.launcher = None
        if tokens == 0:
            return
        if x.dtype != weight.dtype:
            self.fp32_input = torch.empty_like(x, dtype=torch.float32)
            x = _promote_activation(x, self.fp32_input)
        kernel, grid, self.workspace, options = _gate_launch_args(
            x,
            weight,
            bias,
            topk,
            scoring,
            groups,
            top_groups,
            routing_inputs,
            logits_dtype,
        )
        options["IMAGE_LO"] = image_lo
        # FlagTree warmup accepts output dtypes as aligned placeholder tensors.
        warmup_args = (
            x,
            weight,
            bias,
            *routing_inputs,
            self.workspace,
            *self.output_dtypes,
            tokens,
            scale,
        )
        compiled = kernel.warmup(*warmup_args, grid=grid, **options)
        if hasattr(compiled, "result"):
            compiled = compiled.result()
        assert compiled is not None and tuple(compiled.src.signature) == tuple(
            kernel.arg_names
        ), "Unsupported compiled MegaGate launcher signature"
        self.launch_scalars = (tokens, scale) + tuple(
            options[name] for name in kernel.arg_names[len(warmup_args) :]
        )
        self.launcher = compiled[grid + (1, 1)]

    def run(self, x, weight, bias, routing_inputs, config):
        assert config == self.config, "MegaGate routing parameters differ from prepare"
        assert (
            x.shape == self.x_shape
            and weight.shape == self.weight_shape
            and x.device == self.device
        ), "MegaGate inputs differ from the prepared shape or device"
        assert (
            x.dtype,
            weight.dtype,
            None if bias is None else bias.dtype,
        ) == self.input_dtypes, (
            "MegaGate input dtypes or routing mode differ from prepare"
        )
        assert (
            weight.device == self.device
            and x.is_contiguous()
            and weight.is_contiguous()
        ), "x and weight must be contiguous on the prepared NPU"
        if bias is not None:
            assert (
                bias.shape == self.bias_shape
                and bias.device == self.device
                and bias.is_contiguous()
            ), "bias must be contiguous FP32 [E] on the same NPU"
        input_ids, table, vision_bias = routing_inputs
        if (
            self.routing_dtypes[0] is not None
            or input_ids is not None
            or table is not None
            or vision_bias is not None
        ):
            assert (
                tuple(t.dtype if t is not None else None for t in routing_inputs)
                == self.routing_dtypes
            ), "MegaGate input dtypes or routing mode differ from prepare"
            _validate_v4_inputs(x, weight, *routing_inputs, config)
        tensors = (x, weight, bias, *routing_inputs)
        device = torch.npu.current_device()
        assert (
            device == self.device.index
        ), "MegaGate current device differs from prepare"
        assert (
            self.get_stream(device) == self.stream
        ), "Prepare MegaGate on the current stream before calling"
        # The compiled kernel may specialize on 16-byte pointer alignment.
        assert all(
            not aligned or tensor.data_ptr() % 16 == 0
            for aligned, tensor in zip(self.aligned_inputs, tensors)
        ), "MegaGate input alignment differs from prepare"
        ids = torch.empty_like(
            self.output_templates[0], memory_format=torch.contiguous_format
        )
        weights = torch.empty_like(
            self.output_templates[1], memory_format=torch.contiguous_format
        )
        if self.launcher is not None:
            if self.fp32_input is not None:
                x = _promote_activation(x, self.fp32_input)
            self.launcher(
                x,
                weight,
                bias,
                *routing_inputs,
                self.workspace,
                ids,
                weights,
                *self.launch_scalars,
                stream=self.stream,
            )
        return ids, weights


def _resolve_output_dtypes(weight, output_dtypes):
    if output_dtypes is None:
        return (
            torch.int32 if weight.dtype == torch.float32 else torch.int64
        ), torch.float32
    assert (
        isinstance(output_dtypes, tuple)
        and len(output_dtypes) == 2
        and output_dtypes[0] in (torch.int32, torch.int64)
        and output_dtypes[1] in (torch.float32, torch.bfloat16)
    ), "output_dtypes must be (int32 or int64, FP32 or BF16)"
    return output_dtypes


def _gate_outputs(x, topk, dtypes):
    shape = (x.shape[0], topk)
    return (
        x.new_empty(shape, dtype=dtypes[0]),
        x.new_empty(shape, dtype=dtypes[1]),
    )


def prepare_mega_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    topk: int = 6,
    scoring: str = "sqrtsoftplus",
    groups: int = 1,
    top_groups: int = 1,
    scale: float = 1.5,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 129257,
    logits_dtype: torch.dtype = torch.float32,
    output_dtypes: tuple[torch.dtype, torch.dtype] | None = None,
) -> _PreparedMegaGate:
    """Prepare one shape and routing configuration on the current NPU stream.

    logits_dtype and output_dtypes are bound to the handle alongside routing.
    Input values and storage may change between calls with compatible alignment.
    Prepare a separate handle for each execution stream. Keep the handle alive
    until its captured graphs are reset, and replay them on the prepared stream
    or serialize their workspace use with stream dependencies.
    """
    _validate_routing(topk, groups, top_groups, logits_dtype)
    _validate_gate(x, weight, bias, topk, scoring, groups, top_groups, scale)
    assert (
        driver.active.get_current_device() == x.device.index
    ), "Prepare MegaGate on the input device"
    assert (
        not torch.npu.is_current_stream_capturing()
    ), "Prepare MegaGate before capturing a graph"
    config = (
        topk,
        scoring,
        groups,
        top_groups,
        scale,
        image_sentinel_lo,
        logits_dtype,
        output_dtypes,
    )
    routing_inputs = (input_ids, tid2eid, bias_vl)
    _validate_v4_inputs(x, weight, *routing_inputs, config)
    return _PreparedMegaGate(x, weight, bias, routing_inputs, config)


def mega_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    topk: int = 6,
    scoring: str = "sqrtsoftplus",
    groups: int = 1,
    top_groups: int = 1,
    scale: float = 1.5,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
    bias_vl: torch.Tensor | None = None,
    image_sentinel_lo: int = 129257,
    logits_dtype: torch.dtype = torch.float32,
    output_dtypes: tuple[torch.dtype, torch.dtype] | None = None,
    prepared: _PreparedMegaGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return expert IDs and normalized routing weights.

    V4 projects FP32 activations (or promotes a BF16 activation) with FP32
    router weights. Pass row-aligned input_ids and a [vocab, topk] tid2eid
    table for hash layers. Text IDs index valid table rows containing expert
    IDs in [0, E). With bias_vl, the five image sentinels use dynamic TopK.
    Input tensors use contiguous ND storage.

    logits_dtype selects FP32 (default) or BF16 projection logits before
    scoring. output_dtypes optionally selects the (ID, weight) storage types
    for the final stores. The default returns FP32 weights and int32 IDs
    with FP32 projection weights, or int64 IDs with BF16 projection weights.
    Supports sqrtsoftplus, sigmoid and softmax with optional grouped selection.
    Bias changes expert selection while the weights use the unbiased scores.
    Pass a handle from prepare_mega_gate to reuse its workspace and launcher.
    """
    assert prepared is None or isinstance(
        prepared, _PreparedMegaGate
    ), "Invalid prepared MegaGate handle"
    _validate_routing(topk, groups, top_groups, logits_dtype)
    config = (
        topk,
        scoring,
        groups,
        top_groups,
        scale,
        image_sentinel_lo,
        logits_dtype,
        output_dtypes,
    )
    routing_inputs = (input_ids, tid2eid, bias_vl)
    if prepared is not None:
        return prepared.run(x, weight, bias, routing_inputs, config)
    _validate_v4_inputs(x, weight, *routing_inputs, config)
    _validate_gate(x, weight, bias, topk, scoring, groups, top_groups, scale)
    tokens = x.shape[0]
    assert (
        not tokens or driver.active.get_current_device() == x.device.index
    ), "Run MegaGate on the input device"
    indices, weights = _gate_outputs(
        x, topk, _resolve_output_dtypes(weight, output_dtypes)
    )
    if tokens:
        if x.dtype != weight.dtype:
            x = _promote_activation(x, torch.empty_like(x, dtype=torch.float32))
        kernel, grid, workspace, options = _gate_launch_args(
            x,
            weight,
            bias,
            topk,
            scoring,
            groups,
            top_groups,
            routing_inputs,
            logits_dtype,
        )
        kernel[grid](
            x,
            weight,
            bias,
            *routing_inputs,
            workspace,
            indices,
            weights,
            tokens,
            scale,
            IMAGE_LO=image_sentinel_lo,
            **options,
        )
    return indices, weights
