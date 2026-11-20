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

"""Fused BF16 router projection and expert selection on Ascend."""

import math

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _mega_gate_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    workspace_ptr,
    indices_ptr,
    weights_ptr,
    tokens,
    scale,
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
    USE_BOTH_SUBCORES: tl.constexpr,
):
    block = tl.program_id(0)
    workers = tl.num_programs(0)
    expert_tiles: tl.constexpr = EXPERTS // BLOCK_N
    row_tiles = tl.cdiv(tokens, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    for task in range(block, row_tiles * expert_tiles, workers):
        rows = task // expert_tiles * BLOCK_M + tl.arange(0, BLOCK_M)
        columns = task % expert_tiles * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.full((BLOCK_M, BLOCK_N), 0.0, tl.float32)
        for chunk in range(HIDDEN // BLOCK_K):
            reduction = chunk * BLOCK_K + k
            a = tl.load(
                x_ptr + rows[:, None] * HIDDEN + reduction[None, :],
                rows[:, None] < tokens,
                other=0,
            )
            b = tl.load(weight_ptr + columns[None, :] * HIDDEN + reduction[:, None])
            acc = tl.dot(a, b, acc=acc)
        tl.store(workspace_ptr + rows[:, None] * PAD_EXPERTS + columns[None, :], acc)

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
            indices_ptr,
            weights_ptr,
            rows,
            tokens,
            scale,
            HAS_BIAS,
            EXPERTS,
            PAD_EXPERTS,
            TOPK,
            SCORING,
            GROUPS,
            TOP_GROUPS,
            ROUTE_ROWS,
            BLOCK_TOPK,
            USE_BOTH_SUBCORES,
        )


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
def _route_rows(
    workspace_ptr,
    bias_ptr,
    indices_ptr,
    weights_ptr,
    rows,
    tokens,
    scale,
    HAS_BIAS: tl.constexpr,
    EXPERTS: tl.constexpr,
    PAD_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    SCORING: tl.constexpr,
    GROUPS: tl.constexpr,
    TOP_GROUPS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    USE_BOTH_SUBCORES: tl.constexpr,
):
    columns = tl.arange(0, PAD_EXPERTS)
    logits_ptr = workspace_ptr + rows[:, None] * PAD_EXPERTS + columns[None, :]
    if PAD_EXPERTS == EXPERTS:
        logits = tl.load(logits_ptr)
    else:
        logits = tl.load(logits_ptr, columns[None, :] < EXPERTS, other=-float("inf"))

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
    ranking = raw_scores
    if HAS_BIAS:
        if PAD_EXPERTS == EXPERTS:
            bias = tl.load(bias_ptr + columns)
        else:
            bias = tl.load(bias_ptr + columns, columns < EXPERTS, other=0)
        ranking = ranking + bias[None, :]
    if PAD_EXPERTS != EXPERTS:
        ranking = tl.where(columns[None, :] < EXPERTS, ranking, -float("inf"))
    if TOP_GROUPS < GROUPS:
        ranking = _mask_unselected_groups(ranking, ROWS, EXPERTS, GROUPS, TOP_GROUPS)

    slots = tl.arange(0, BLOCK_TOPK)
    indices = tl.full((ROWS, BLOCK_TOPK), 0, tl.int32)
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
        ranking = tl.where(columns[None, :] == winner[:, None], -float("inf"), ranking)
        indices = tl.where(slots[None, :] == slot, winner[:, None], indices)

    if KEEP_SELECTED_LOGITS:
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
    tl.store(indices_ptr + output, indices.to(tl.int64), mask)
    tl.store(weights_ptr + output, weights * (scale / (total[:, None] + 1e-20)), mask)


def _validate_routing_sizes(topk, groups, top_groups):
    assert (
        type(topk) is type(groups) is type(top_groups) is int
    ), "topk, groups and top_groups must be integers"


def _validate_gate_tensors(x, weight, bias):
    device = x.device
    assert (
        weight.device == device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
    ), "x and weight must be contiguous BF16 on the same NPU"
    assert bias is None or (
        bias.shape == (weight.shape[0],)
        and bias.dtype == torch.float32
        and bias.device == device
        and bias.is_contiguous()
    ), "bias must be contiguous FP32 [E] on the same NPU"


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


def _gate_launch_args(x, weight, bias, topk, scoring, groups, top_groups):
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
    weight_bytes = hidden * experts * 2  # BF16
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
    compile_options = {}
    if short_projection:
        compile_options["multibuffer"] = False
    elif short_wave_preload:
        compile_options["enable_preload"] = True
    workspace = torch.empty(
        (row_tiles * block_m, padded_experts), dtype=torch.float32, device=x.device
    )
    return (
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
            USE_BOTH_SUBCORES=use_both_subcores,
            enable_auto_bind_sub_block=False if use_both_subcores else None,
            enable_sync_block_lock=True,
            disable_auto_inject_block_sync=True,
            use_bytecode=False,
            **compile_options,
        ),
    )


class _PreparedMegaGate:
    def __init__(self, x, weight, bias, config):
        topk, scoring, groups, top_groups, scale = config
        tokens = x.shape[0]
        self.config = config
        self.x_shape = x.shape
        self.weight_shape = weight.shape
        self.has_bias = bias is not None
        self.device = x.device
        self.stream = driver.active.get_current_stream(x.device.index)
        self.aligned_inputs = tuple(
            tensor is not None and tensor.data_ptr() % 16 == 0
            for tensor in (x, weight, bias)
        )
        self.launcher = None
        if tokens == 0:
            return
        grid, self.workspace, options = _gate_launch_args(
            x, weight, bias, topk, scoring, groups, top_groups
        )
        # FlagTree warmup accepts output dtypes as aligned placeholder tensors.
        warmup_args = (
            x,
            weight,
            bias,
            self.workspace,
            torch.int64,
            torch.float32,
            tokens,
            scale,
        )
        compiled = _mega_gate_kernel.warmup(*warmup_args, grid=grid, **options)
        if hasattr(compiled, "result"):
            compiled = compiled.result()
        assert compiled is not None and tuple(compiled.src.signature) == tuple(
            _mega_gate_kernel.arg_names
        ), "Unsupported compiled MegaGate launcher signature"
        self.launch_scalars = (tokens, scale) + tuple(
            options[name] for name in _mega_gate_kernel.arg_names[len(warmup_args) :]
        )
        self.launcher = compiled[grid + (1, 1)]

    def run(self, x, weight, bias, config):
        assert config == self.config, "MegaGate routing parameters differ from prepare"
        assert (
            x.shape == self.x_shape
            and weight.shape == self.weight_shape
            and x.device == self.device
        ), "MegaGate inputs differ from the prepared shape or device"
        assert (
            bias is not None
        ) == self.has_bias, "MegaGate bias presence differs from prepare"
        _validate_gate_tensors(x, weight, bias)
        active_driver = driver.active
        device = active_driver.get_current_device()
        assert (
            device == self.device.index
        ), "MegaGate current device differs from prepare"
        assert (
            active_driver.get_current_stream(device) == self.stream
        ), "Prepare MegaGate on the current stream before calling"
        # The compiled kernel may specialize on 16-byte pointer alignment.
        assert all(
            not aligned or tensor.data_ptr() % 16 == 0
            for aligned, tensor in zip(self.aligned_inputs, (x, weight, bias))
        ), "MegaGate input alignment differs from prepare"
        ids, weights = _gate_outputs(x, config[0])
        if self.launcher is not None:
            self.launcher(
                x,
                weight,
                bias,
                self.workspace,
                ids,
                weights,
                *self.launch_scalars,
                stream=self.stream,
            )
        return ids, weights


def _gate_outputs(x, topk):
    shape = (x.shape[0], topk)
    return (
        torch.empty(shape, dtype=torch.int64, device=x.device),
        torch.empty(shape, dtype=torch.float32, device=x.device),
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
) -> _PreparedMegaGate:
    """Prepare one shape and routing configuration on the current NPU stream.

    Input values and storage may change between calls with compatible alignment.
    Prepare a separate handle for each execution stream. Keep the handle alive
    until its captured graphs are reset, and replay them on the prepared stream
    or serialize their workspace use with stream dependencies.
    """
    _validate_routing_sizes(topk, groups, top_groups)
    _validate_gate(x, weight, bias, topk, scoring, groups, top_groups, scale)
    assert (
        driver.active.get_current_device() == x.device.index
    ), "Prepare MegaGate on the input device"
    assert (
        not torch.npu.is_current_stream_capturing()
    ), "Prepare MegaGate before capturing a graph"
    return _PreparedMegaGate(
        x, weight, bias, (topk, scoring, groups, top_groups, scale)
    )


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
    prepared: _PreparedMegaGate | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int64 expert IDs and FP32 weights from contiguous BF16 router inputs.

    Supports sqrtsoftplus, sigmoid and softmax with optional grouped selection.
    Bias changes expert selection while the weights use the unbiased scores.
    Pass a handle from prepare_mega_gate to reuse its workspace and launcher.
    """
    assert prepared is None or isinstance(
        prepared, _PreparedMegaGate
    ), "Invalid prepared MegaGate handle"
    _validate_routing_sizes(topk, groups, top_groups)
    if prepared is not None:
        return prepared.run(x, weight, bias, (topk, scoring, groups, top_groups, scale))
    _validate_gate(x, weight, bias, topk, scoring, groups, top_groups, scale)
    tokens = x.shape[0]
    assert (
        not tokens or driver.active.get_current_device() == x.device.index
    ), "Run MegaGate on the input device"
    indices, weights = _gate_outputs(x, topk)
    if tokens:
        grid, workspace, options = _gate_launch_args(
            x, weight, bias, topk, scoring, groups, top_groups
        )
        _mega_gate_kernel[grid](
            x, weight, bias, workspace, indices, weights, tokens, scale, **options
        )
    return indices, weights
