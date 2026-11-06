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

"""Optimized Triton-Ascend implementation of the Compressor operator.

The public interface matches the vLLM Ascend custom operator:

    torch.ops._C_ascend.compressor(
        x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
        state_block_table, cu_seqlens, seqused, start_pos,
        rope_head_dim, cmp_ratio, coff, norm_eps, rotary_mode, cache_mode,
    )

The implementation contains dedicated prefill and decode paths:
1. project x with wkv and wgate;
2. write the projected kv/score states into state_cache;
3. read complete compression groups from state_cache, apply APE, softmax and
   weighted reduction;
4. apply RMSNorm and RoPE.

The state cache is updated in place. For latency-sensitive execution, callers
can prepare host metadata and workspace once, then execute the bound launch
plan directly.

The public API is kept unchanged from the source implementation.
"""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Ascend NPU hardware query helpers
# ---------------------------------------------------------------------------
_npu_core_counts = None


def _get_npu_core_counts():
    """Return (vectorcore_num, aicore_num) for current device, cached."""
    global _npu_core_counts
    if _npu_core_counts is not None:
        return _npu_core_counts
    import torch_npu
    import triton.runtime.driver as driver

    props = driver.active.utils.get_device_properties(torch_npu.npu.current_device())
    _npu_core_counts = (props["num_vectorcore"], props["num_aicore"])
    return _npu_core_counts


PROJECT_BLOCK_K_MIN = 16
PROJECT_BLOCK_K_ALIGNMENT = 16


@dataclass(frozen=True)
class OnChipMemoryCapacity:
    soc_version: str
    l0_a_bytes: int
    l0_b_bytes: int
    l0_c_bytes: int
    l1_bytes: int
    ub_bytes: int
    config_path: str


@dataclass(frozen=True)
class ProjectionResourceUsage:
    l0_a_bytes: int | None
    l0_b_bytes: int | None
    l0_c_bytes: int | None
    l1_bytes: int | None
    ub_bytes: int | None = None


@dataclass(frozen=True)
class ProjectionResourceModel:
    fixed_usage: ProjectionResourceUsage
    per_block_k_usage: ProjectionResourceUsage

    def usage_for(self, block_k):
        def evaluate(field):
            fixed = getattr(self.fixed_usage, field)
            per_block_k = getattr(self.per_block_k_usage, field)
            if fixed is None or per_block_k is None:
                return None
            return fixed + per_block_k * block_k

        return ProjectionResourceUsage(
            l0_a_bytes=evaluate("l0_a_bytes"),
            l0_b_bytes=evaluate("l0_b_bytes"),
            l0_c_bytes=evaluate("l0_c_bytes"),
            l1_bytes=evaluate("l1_bytes"),
            ub_bytes=evaluate("ub_bytes"),
        )


@dataclass(frozen=True)
class ProjectionResourcePlan:
    kernel: str
    block_k: int
    logical_tile_usage: ProjectionResourceUsage
    capacity: OnChipMemoryCapacity


_on_chip_capacity = None
_projection_resource_plans = {}


def _normalize_soc_name(value):
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized.removeprefix("ascend")


def _runtime_soc_names():
    names = []
    override = os.getenv("COMPRESSOR_SOC_VERSION")
    if override:
        names.append(override)
    try:
        import torch_npu

        names.append(torch_npu.npu.get_device_name(torch_npu.npu.current_device()))
    except Exception:
        pass
    try:
        from triton.runtime import driver

        names.append(driver.active.get_current_target().arch)
    except Exception:
        pass
    return tuple(dict.fromkeys(str(name) for name in names if name))


def _platform_config_paths():
    roots = []
    for variable in (
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_TOOLKIT_LATEST_HOME",
    ):
        value = os.getenv(variable)
        if value:
            roots.append(Path(value))
    roots.extend((Path("/usr/local/Ascend/ascend-toolkit/latest"),))

    paths = []
    for root in dict.fromkeys(roots):
        for relative in (
            "aarch64-linux/data/platform_config",
            "x86_64-linux/data/platform_config",
            "data/platform_config",
        ):
            directory = root / relative
            if directory.is_dir():
                paths.extend(sorted(directory.glob("*.ini")))
    return tuple(dict.fromkeys(paths))


def _read_on_chip_capacity(path):
    parser = configparser.ConfigParser()
    parser.read(path)
    version = parser["version"]
    spec = parser["AICoreSpec"]
    return OnChipMemoryCapacity(
        soc_version=version["soc_version"],
        l0_a_bytes=spec.getint("l0_a_size"),
        l0_b_bytes=spec.getint("l0_b_size"),
        l0_c_bytes=spec.getint("l0_c_size"),
        l1_bytes=spec.getint("l1_size"),
        ub_bytes=spec.getint("ub_size"),
        config_path=str(path),
    ), version.get("short_soc_version", "")


def _get_on_chip_capacity():
    global _on_chip_capacity
    if _on_chip_capacity is not None:
        return _on_chip_capacity

    runtime_names = _runtime_soc_names()
    normalized_names = {_normalize_soc_name(name) for name in runtime_names}
    matches = []
    for path in _platform_config_paths():
        try:
            capacity, short_name = _read_on_chip_capacity(path)
        except (KeyError, ValueError, configparser.Error):
            continue
        config_names = {
            _normalize_soc_name(capacity.soc_version),
            _normalize_soc_name(short_name),
            _normalize_soc_name(path.stem),
        }
        if normalized_names & config_names:
            matches.append(capacity)

    exact_matches = [
        capacity
        for capacity in matches
        if _normalize_soc_name(capacity.soc_version) in normalized_names
    ]
    selected = exact_matches or matches
    if not selected:
        names = ", ".join(runtime_names) or "unknown"
        raise RuntimeError(
            "Cannot find the CANN platform_config for the active NPU "
            f"({names}). Source the matching CANN setenv.bash or set "
            "COMPRESSOR_SOC_VERSION to the exact SoC version."
        )

    _on_chip_capacity = selected[0]
    return _on_chip_capacity


def _pair_projection_resource_model(block_m, block_d, dtype_bytes):
    # The compiler report exposes the full CBUF/CC layout. CA/CB tiling is
    # internal to hivm.hir.mmadL1 and must not be represented by a guessed
    # 16x16 fragment size.
    return ProjectionResourceModel(
        fixed_usage=ProjectionResourceUsage(
            l0_a_bytes=None,
            l0_b_bytes=None,
            l0_c_bytes=2 * block_m * block_d * 4,
            l1_bytes=0,
        ),
        per_block_k_usage=ProjectionResourceUsage(
            l0_a_bytes=None,
            l0_b_bytes=None,
            l0_c_bytes=0,
            l1_bytes=2 * dtype_bytes * (block_m + 2 * block_d),
        ),
    )


def _concat_projection_resource_model(block_m, block_n, dtype_bytes):
    return ProjectionResourceModel(
        fixed_usage=ProjectionResourceUsage(
            l0_a_bytes=None,
            l0_b_bytes=None,
            l0_c_bytes=block_m * block_n * 4,
            l1_bytes=0,
        ),
        per_block_k_usage=ProjectionResourceUsage(
            l0_a_bytes=None,
            l0_b_bytes=None,
            l0_c_bytes=0,
            l1_bytes=2 * dtype_bytes * (block_m + block_n),
        ),
    )


def _fits_resource_capacity(usage, capacity):
    def within(actual, limit):
        return actual is None or actual <= limit

    return (
        within(usage.l0_a_bytes, capacity.l0_a_bytes)
        and within(usage.l0_b_bytes, capacity.l0_b_bytes)
        and within(usage.l0_c_bytes, capacity.l0_c_bytes)
        and within(usage.l1_bytes, capacity.l1_bytes)
        and within(usage.ub_bytes, capacity.ub_bytes)
    )


def _floor_power_of_two(value):
    value = int(value)
    if value < 1:
        return 0
    return 1 << (value.bit_length() - 1)


def _ceil_power_of_two(value):
    value = int(value)
    if value < 1:
        return 0
    return 1 << ((value - 1).bit_length())


def _modeled_resource_fields(resource_model):
    fields = (
        "l0_a_bytes",
        "l0_b_bytes",
        "l0_c_bytes",
        "l1_bytes",
        "ub_bytes",
    )
    modeled = []
    for field in fields:
        intercept = getattr(resource_model.fixed_usage, field)
        slope = getattr(resource_model.per_block_k_usage, field)
        if intercept is None and slope is None:
            continue
        if intercept is None or slope is None:
            raise RuntimeError(f"Incomplete projection resource model for {field}")
        modeled.append((field, intercept, slope))
    return tuple(modeled)


def _calculate_projection_block_k(resource_model, capacity, h):
    """Select a measured-efficient BLOCK_K within exact resource limits."""
    if h <= 0:
        raise ValueError(f"Projection reduction dimension must be positive, got {h}")

    # A power-of-two tile may pad past H because all projection loads are
    # masked. Capping at ceil_pow2(H) avoids an unnecessary extra K loop for
    # non-power-of-two hidden sizes without allocating a larger useless tile.
    upper_bound = max(_ceil_power_of_two(h), PROJECT_BLOCK_K_MIN)

    for field, intercept, slope in _modeled_resource_fields(resource_model):

        limit = getattr(capacity, field)
        if slope < 0 or intercept < 0:
            raise RuntimeError(f"Non-monotonic projection resource model for {field}")
        if intercept > limit:
            raise RuntimeError(
                f"Projection fixed {field} usage {intercept} exceeds capacity {limit}"
            )
        if slope > 0:
            upper_bound = min(upper_bound, (limit - intercept) // slope)

    # SIMD tl.arange accepts non-power-of-two extents, but the Cube/CBUF
    # lowering rounds K to 16 elements. Current-shape sweeps also show that
    # power-of-two K tiles substantially outperform larger irregular tiles.
    aligned_upper_bound = (
        upper_bound // PROJECT_BLOCK_K_ALIGNMENT * PROJECT_BLOCK_K_ALIGNMENT
    )
    block_k = _floor_power_of_two(aligned_upper_bound)
    if block_k < PROJECT_BLOCK_K_MIN:
        raise RuntimeError(
            "No legal BLOCK_K fits the modeled on-chip resource capacities"
        )

    logical_usage = resource_model.usage_for(block_k)
    if not _fits_resource_capacity(logical_usage, capacity):
        raise RuntimeError(
            f"Calculated BLOCK_K={block_k} does not fit the logical resource model"
        )
    return block_k


def _select_projection_block_k(
    *,
    cache_key,
    kernel_name,
    resource_model,
    h,
):
    """Select and cache a resource-safe BLOCK_K without process-global side effects."""
    cached = _projection_resource_plans.get(cache_key)
    if cached is not None:
        return cached.block_k

    capacity = _get_on_chip_capacity()
    block_k = _calculate_projection_block_k(resource_model, capacity, h)
    plan = ProjectionResourcePlan(
        kernel=kernel_name,
        block_k=block_k,
        logical_tile_usage=resource_model.usage_for(block_k),
        capacity=capacity,
    )
    _projection_resource_plans[cache_key] = plan
    if os.getenv("COMPRESSOR_PRINT_RESOURCE_PLAN", "0") == "1":
        print(f"[compressor] projection resource plan: {plan}")
    return block_k


@triton.jit
def _zero_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)


@triton.jit
def _project_pair_kernel(
    x_ptr,
    wkv_ptr,
    wgate_ptr,
    state_proj_ptr,
    T: tl.constexpr,
    STORE_T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    HEAD_D: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_w_o: tl.constexpr,
    stride_w_h: tl.constexpr,
    BLOCKS_D: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    NUM_TILE_ROUNDS: tl.constexpr,
    COFF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    for tile_round in range(0, NUM_TILE_ROUNDS):
        tile_id = pid + tile_round * NUM_PROGRAMS
        pid_m = tile_id // BLOCKS_D
        pid_d = tile_id - pid_m * BLOCKS_D

        m_offset = pid_m * BLOCK_M
        d_base_offset = pid_d * BLOCK_D
        offs_m = m_offset + tl.arange(0, BLOCK_M)
        offs_d = d_base_offset + tl.arange(0, BLOCK_D)

        for coff_id in range(0, COFF):
            acc_kv = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            acc_score = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            for k0 in range(0, tl.cdiv(H, BLOCK_K)):
                h_offset = k0 * BLOCK_K
                offs_h = h_offset + tl.arange(0, BLOCK_K)
                weight_d = coff_id * HEAD_D + offs_d
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_x_t + offs_h[None, :] * stride_x_h,
                    mask=(offs_m[:, None] < T) & (offs_h[None, :] < H),
                    other=0.0,
                )
                wkv = tl.load(
                    wkv_ptr
                    + weight_d[:, None] * stride_w_o
                    + offs_h[None, :] * stride_w_h,
                    mask=(offs_d[:, None] < HEAD_D) & (offs_h[None, :] < H),
                    other=0.0,
                )
                wgate = tl.load(
                    wgate_ptr
                    + weight_d[:, None] * stride_w_o
                    + offs_h[None, :] * stride_w_h,
                    mask=(offs_d[:, None] < HEAD_D) & (offs_h[None, :] < H),
                    other=0.0,
                )
                acc_kv += tl.dot(x, tl.trans(wkv))
                acc_score += tl.dot(x, tl.trans(wgate))

            out_d = coff_id * HEAD_D + offs_d
            out_mask = (offs_m[:, None] < STORE_T) & (offs_d[None, :] < HEAD_D)
            tl.store(
                state_proj_ptr + offs_m[:, None] * (2 * K) + out_d[None, :],
                acc_kv,
                mask=out_mask,
            )
            tl.store(
                state_proj_ptr + offs_m[:, None] * (2 * K) + K + out_d[None, :],
                acc_score,
                mask=out_mask,
            )


@triton.jit
def _project_concat_kernel(
    x_ptr,
    weight_ptr,
    state_proj_ptr,
    T: tl.constexpr,
    STORE_T: tl.constexpr,
    H: tl.constexpr,
    OUT_DIM: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_w_o: tl.constexpr,
    stride_w_h: tl.constexpr,
    BLOCKS_N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    NUM_TILE_ROUNDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Project with a persistent [wkv; wgate] weight matrix."""
    pid = tl.program_id(0)
    for tile_round in range(0, NUM_TILE_ROUNDS):
        tile_id = pid + tile_round * NUM_PROGRAMS
        if tile_id < NUM_TILES:
            pid_m = tile_id // BLOCKS_N
            pid_n = tile_id - pid_m * BLOCKS_N
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, tl.cdiv(H, BLOCK_K)):
                offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
                x = tl.load(
                    x_ptr + offs_m[:, None] * stride_x_t + offs_k[None, :] * stride_x_h,
                    mask=(offs_m[:, None] < T) & (offs_k[None, :] < H),
                    other=0.0,
                )
                weight = tl.load(
                    weight_ptr
                    + offs_n[:, None] * stride_w_o
                    + offs_k[None, :] * stride_w_h,
                    mask=(offs_n[:, None] < OUT_DIM) & (offs_k[None, :] < H),
                    other=0.0,
                )
                acc += tl.dot(x, tl.trans(weight))
            tl.store(
                state_proj_ptr + offs_m[:, None] * OUT_DIM + offs_n[None, :],
                acc,
                mask=(offs_m[:, None] < STORE_T) & (offs_n[None, :] < OUT_DIM),
            )


@triton.jit
def _update_state_batched_kernel(
    state_proj_ptr,
    state_cache_ptr,
    state_block_table_ptr,
    start_pos_ptr,
    ape_ptr,
    token_bases_ptr,
    seq_used_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    NUM_CORES: tl.constexpr,
    MAX_S: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    table_stride_0: tl.constexpr,
    table_stride_1: tl.constexpr,
    ape_stride_0: tl.constexpr,
    ape_stride_1: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HAS_START_POS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CACHE_PROGRAM_APE: tl.constexpr,
    WRITEBACK_APE: tl.constexpr,
):
    """Prefill update: batch outer loop, grid=(NUM_CORES,).

    One token per iteration: cache_block and block_off are scalars, so each
    write is a fully contiguous 1D DMA over the K channels (no scatter).
    The K channels are tiled by BLOCK_K to fit UB.
    """
    pid = tl.program_id(0)
    K_BLOCKS: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K

    for b_idx in range(B):
        token_base = tl.load(token_bases_ptr + b_idx).to(tl.int64)
        seq_len = tl.load(seq_used_ptr + b_idx).to(tl.int64)
        has_work = seq_len > 0
        if has_work:
            start = tl.full((), 0, dtype=tl.int64)
            if HAS_START_POS:
                start = tl.load(start_pos_ptr + b_idx).to(tl.int64)

            if CACHE_PROGRAM_APE:
                program_ape_row = (start + pid) % CMP_RATIO
                program_ape_offsets = tl.arange(0, BLOCK_K)
                program_ape_mask = program_ape_offsets < K
                cached_program_ape = tl.load(
                    ape_ptr
                    + program_ape_row * ape_stride_0
                    + program_ape_offsets * ape_stride_1,
                    mask=program_ape_mask,
                    other=0.0,
                ).to(tl.float32)

            for local_t in range(pid, MAX_S, NUM_CORES):
                if local_t < seq_len:
                    pos = start + local_t
                    blk = pos // BLOCK_SIZE_CACHE
                    block_off = pos - blk * BLOCK_SIZE_CACHE

                    cache_block = blk
                    blk_ok = (blk >= 0) & (blk < STATE_BLOCKS)
                    if HAS_BLOCK_TABLE:
                        blk_in_tbl = (blk >= 0) & (blk < TABLE_WIDTH)
                        cache_block = tl.load(
                            state_block_table_ptr
                            + b_idx * table_stride_0
                            + blk * table_stride_1,
                            mask=blk_in_tbl,
                            other=0,
                        ).to(tl.int64)
                        blk_ok = (
                            blk_in_tbl
                            & (cache_block != 0)
                            & (cache_block < STATE_BLOCKS)
                        )

                    if blk_ok:
                        token_idx = token_base + local_t
                        dst_base = (
                            cache_block * state_stride_0 + block_off * state_stride_1
                        )
                        src_base = token_idx * (2 * K)

                        for k_block in tl.static_range(K_BLOCKS):
                            offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
                            k_mask = offs_k < K
                            kv = tl.load(
                                state_proj_ptr + src_base + offs_k,
                                mask=k_mask,
                                other=0.0,
                            )
                            score = tl.load(
                                state_proj_ptr + src_base + K + offs_k,
                                mask=k_mask,
                                other=0.0,
                            )
                            if CACHE_PROGRAM_APE:
                                ape = cached_program_ape
                            else:
                                ape_row = tl.minimum(
                                    tl.maximum(pos % CMP_RATIO, 0), CMP_RATIO - 1
                                )
                                ape = tl.load(
                                    ape_ptr
                                    + ape_row * ape_stride_0
                                    + offs_k * ape_stride_1,
                                    mask=k_mask,
                                    other=0.0,
                                ).to(tl.float32)
                            score_with_ape = score + ape
                            tl.store(
                                state_cache_ptr + dst_base + offs_k * state_stride_2,
                                kv,
                                mask=k_mask,
                            )
                            tl.store(
                                state_cache_ptr
                                + dst_base
                                + (K + offs_k) * state_stride_2,
                                score_with_ape,
                                mask=k_mask,
                            )
                            if WRITEBACK_APE:
                                tl.store(
                                    state_proj_ptr + src_base + K + offs_k,
                                    score_with_ape,
                                    mask=k_mask,
                                )


@triton.jit
def _register_tail(value, VALUE_SIZE: tl.constexpr, TAIL_SIZE: tl.constexpr):
    tail_offsets = VALUE_SIZE - TAIL_SIZE + tl.arange(0, TAIL_SIZE)
    return tl.gather(value, tail_offsets, axis=0)


@triton.jit
def _store_normed_with_rope_mode2(
    normed,
    rope_sin_ptr,
    rope_cos_ptr,
    output_ptr,
    row,
    value_valid,
    row_in_bounds,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
):
    tl.static_assert(BLOCK_D == D, "register RoPE tail requires BLOCK_D == D")
    d = tl.arange(0, BLOCK_D)
    rope_start: tl.constexpr = D - ROPE_HEAD_DIM
    rope_offs = tl.arange(0, ROPE_HEAD_DIM)

    tl.store(
        output_ptr + row * D + d,
        normed,
        mask=row_in_bounds & (d < rope_start),
    )

    rope_normed = _register_tail(normed, BLOCK_D, ROPE_HEAD_DIM)
    rope_sin = tl.load(
        rope_sin_ptr + row * ROPE_HEAD_DIM + rope_offs,
        mask=value_valid,
        other=0.0,
    ).to(tl.float32)
    rope_cos = tl.load(
        rope_cos_ptr + row * ROPE_HEAD_DIM + rope_offs,
        mask=value_valid,
        other=1.0,
    ).to(tl.float32)

    value_2d = tl.reshape(rope_normed, (ROPE_HEAD_DIM // 2, 2))
    value_even, value_odd = tl.split(value_2d)
    sin_2d = tl.reshape(rope_sin, (ROPE_HEAD_DIM // 2, 2))
    sin_even, sin_odd = tl.split(sin_2d)
    cos_2d = tl.reshape(rope_cos, (ROPE_HEAD_DIM // 2, 2))
    cos_even, cos_odd = tl.split(cos_2d)
    rotated_even = value_even * cos_even - value_odd * sin_even
    rotated_odd = value_odd * cos_odd + value_even * sin_odd
    tl.store(
        output_ptr + row * D + rope_start + rope_offs,
        tl.interleave(rotated_even, rotated_odd),
        mask=row_in_bounds,
    )


@triton.jit
def _compress_groups_batched_kernel(
    state_cache_ptr,
    state_block_table_ptr,
    start_pos_ptr,
    comp_ptr,
    norm_weight_ptr,
    rope_sin_ptr,
    rope_cos_ptr,
    output_ptr,
    output_bases_ptr,
    n_groups_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    COFF: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    table_stride_0: tl.constexpr,
    table_stride_1: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HAS_START_POS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    INNER_G: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROTARY_MODE: tl.constexpr,
    FUSE_POST: tl.constexpr,
):
    """Prefill compress: batch outer loop, grid=(NUM_CORES,)."""
    pid = tl.program_id(0)

    d = tl.arange(0, BLOCK_D)
    NEG_INF: tl.constexpr = -3.4028234663852886e38
    TOTAL_G: tl.constexpr = CMP_RATIO * COFF

    for b_idx in range(B):
        n_groups = tl.load(n_groups_ptr + b_idx).to(tl.int64)
        output_base = tl.load(output_bases_ptr + b_idx).to(tl.int64)
        has_work = n_groups > 0
        if has_work:
            start = tl.full((), 0, dtype=tl.int64)
            if HAS_START_POS:
                start = tl.load(start_pos_ptr + b_idx).to(tl.int64)

            for group_idx in range(pid, MAX_GROUPS, NUM_CORES):
                valid_row = group_idx < n_groups
                if valid_row:
                    group_anchor = start + group_idx * CMP_RATIO
                    group_start = (group_anchor // CMP_RATIO) * CMP_RATIO
                    valid_d = d < D

                    if COFF == 2:
                        # c4: split into left (pos=group_start-CMP_RATIO+rng, state_d=d)
                        # and right (pos=group_start+rng, state_d=D+d) contiguous loads.
                        # Avoids tl.where in address computation (which scalarizes loads).
                        rng = tl.arange(0, CMP_RATIO)
                        dd = d[None, :]

                        left_pos = group_start - CMP_RATIO + rng
                        left_blk = left_pos // BLOCK_SIZE_CACHE
                        left_off = left_pos - left_blk * BLOCK_SIZE_CACHE
                        left_cache = left_blk
                        left_vb = (
                            (left_pos >= 0)
                            & (left_blk >= 0)
                            & (left_blk < STATE_BLOCKS)
                        )
                        if HAS_BLOCK_TABLE:
                            lt_mask = (left_blk >= 0) & (left_blk < TABLE_WIDTH)
                            left_cache = tl.load(
                                state_block_table_ptr
                                + b_idx * table_stride_0
                                + left_blk * table_stride_1,
                                mask=lt_mask,
                                other=0,
                            ).to(tl.int64)
                            left_vb = (
                                lt_mask
                                & (left_cache != 0)
                                & (left_cache < STATE_BLOCKS)
                                & (left_pos >= 0)
                            )
                        left_base = (
                            left_cache[:, None] * state_stride_0
                            + left_off[:, None] * state_stride_1
                        )
                        left_valid = valid_d[None, :] & left_vb[:, None]
                        kv_left = tl.load(
                            state_cache_ptr + left_base + dd * state_stride_2,
                            mask=left_valid,
                            other=0.0,
                        ).to(tl.float32)
                        score_left = tl.load(
                            state_cache_ptr + left_base + (K + dd) * state_stride_2,
                            mask=left_valid,
                            other=NEG_INF,
                        ).to(tl.float32)

                        right_pos = group_start + rng
                        right_blk = right_pos // BLOCK_SIZE_CACHE
                        right_off = right_pos - right_blk * BLOCK_SIZE_CACHE
                        right_cache = right_blk
                        right_vb = (
                            (right_pos >= 0)
                            & (right_blk >= 0)
                            & (right_blk < STATE_BLOCKS)
                        )
                        if HAS_BLOCK_TABLE:
                            rt_mask = (right_blk >= 0) & (right_blk < TABLE_WIDTH)
                            right_cache = tl.load(
                                state_block_table_ptr
                                + b_idx * table_stride_0
                                + right_blk * table_stride_1,
                                mask=rt_mask,
                                other=0,
                            ).to(tl.int64)
                            right_vb = (
                                rt_mask
                                & (right_cache != 0)
                                & (right_cache < STATE_BLOCKS)
                                & (right_pos >= 0)
                            )
                        right_base = (
                            right_cache[:, None] * state_stride_0
                            + right_off[:, None] * state_stride_1
                        )
                        rcd = (D + d)[None, :]
                        right_valid = valid_d[None, :] & right_vb[:, None]
                        kv_right = tl.load(
                            state_cache_ptr + right_base + rcd * state_stride_2,
                            mask=right_valid,
                            other=0.0,
                        ).to(tl.float32)
                        score_right = tl.load(
                            state_cache_ptr + right_base + (K + rcd) * state_stride_2,
                            mask=right_valid,
                            other=NEG_INF,
                        ).to(tl.float32)

                        smax = tl.maximum(
                            tl.max(score_left, axis=0), tl.max(score_right, axis=0)
                        )
                        w_left = tl.exp(score_left - smax[None, :])
                        w_right = tl.exp(score_right - smax[None, :])
                        denom = tl.sum(w_left, axis=0) + tl.sum(w_right, axis=0)
                        comp_final = (
                            tl.sum(w_left * kv_left, axis=0)
                            + tl.sum(w_right * kv_right, axis=0)
                        ) / denom
                    else:
                        score_max = tl.full((BLOCK_D,), NEG_INF, dtype=tl.float32)
                        comp = tl.zeros((BLOCK_D,), dtype=tl.float32)
                        denom = tl.zeros((BLOCK_D,), dtype=tl.float32)

                        for g_base in range(0, BLOCK_G, INNER_G):
                            bg = g_base + tl.arange(0, INNER_G)
                            valid_g_dim = bg < TOTAL_G
                            pos = group_start + bg
                            state_d = d[None, :]

                            block_idx = pos // BLOCK_SIZE_CACHE
                            block_off = pos - block_idx * BLOCK_SIZE_CACHE
                            cache_block = block_idx
                            valid_block = (
                                (pos >= 0)
                                & (block_idx >= 0)
                                & (block_idx < STATE_BLOCKS)
                            )
                            if HAS_BLOCK_TABLE:
                                tbl_mask = (block_idx >= 0) & (block_idx < TABLE_WIDTH)
                                cache_block = tl.load(
                                    state_block_table_ptr
                                    + b_idx * table_stride_0
                                    + block_idx * table_stride_1,
                                    mask=tbl_mask,
                                    other=0,
                                ).to(tl.int64)
                                valid_block = (
                                    tbl_mask
                                    & (cache_block != 0)
                                    & (cache_block < STATE_BLOCKS)
                                    & (pos >= 0)
                                )

                            full_valid = (
                                valid_g_dim[:, None]
                                & valid_d[None, :]
                                & valid_block[:, None]
                            )
                            cache_base = (
                                cache_block[:, None] * state_stride_0
                                + block_off[:, None] * state_stride_1
                            )

                            kv_g = tl.load(
                                state_cache_ptr + cache_base + state_d * state_stride_2,
                                mask=full_valid,
                                other=0.0,
                            ).to(tl.float32)
                            score_g = tl.load(
                                state_cache_ptr
                                + cache_base
                                + (K + state_d) * state_stride_2,
                                mask=full_valid,
                                other=NEG_INF,
                            ).to(tl.float32)

                            chunk_max = tl.max(score_g, axis=0)
                            new_max = tl.maximum(score_max, chunk_max)
                            if g_base > 0:
                                scale = tl.exp(score_max - new_max)
                                comp = comp * scale
                                denom = denom * scale
                            score_max = new_max
                            w = tl.exp(score_g - score_max[None, :])
                            comp += tl.sum(w * kv_g, axis=0)
                            denom += tl.sum(w, axis=0)

                        comp_final = tl.where(denom > 0, comp / denom, 0.0)

                    out_row = output_base + group_idx
                    if FUSE_POST:
                        square_sum = tl.sum(comp_final * comp_final, axis=0)
                        rstd = 1.0 / tl.sqrt(square_sum / D + NORM_EPS)
                        weight = tl.load(
                            norm_weight_ptr + d, mask=valid_d, other=1.0
                        ).to(tl.float32)
                        normed = comp_final * rstd * weight
                        if ROTARY_MODE == 2 and ROPE_HEAD_DIM > 0:
                            _store_normed_with_rope_mode2(
                                normed,
                                rope_sin_ptr,
                                rope_cos_ptr,
                                output_ptr,
                                out_row,
                                valid_row,
                                valid_row,
                                D,
                                BLOCK_D,
                                ROPE_HEAD_DIM,
                            )
                        else:
                            tl.store(
                                output_ptr + out_row * D + d,
                                normed,
                                mask=valid_d,
                            )
                    else:
                        tl.store(comp_ptr + out_row * D + d, comp_final, mask=valid_d)


@triton.jit
def _compress_projection_b1_kernel(
    state_proj_ptr,
    ape_ptr,
    norm_weight_ptr,
    rope_sin_ptr,
    rope_cos_ptr,
    output_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    COFF: tl.constexpr,
    NUM_CORES: tl.constexpr,
    ape_stride_0: tl.constexpr,
    ape_stride_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    VALID_GROUPS: tl.constexpr,
    INNER_G: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    ROTARY_MODE: tl.constexpr,
    SCORE_HAS_APE: tl.constexpr,
):
    """B=1,start=0 prefill compression directly from contiguous projection."""
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    valid_d = d < D
    negative_inf: tl.constexpr = -3.4028234663852886e38

    if COFF == 2:
        cached_norm_weight = tl.load(norm_weight_ptr + d, mask=valid_d, other=1.0).to(
            tl.float32
        )
        if not SCORE_HAS_APE:
            ape_group = tl.arange(0, CMP_RATIO)
            ape_mask = valid_d[None, :]
            cached_ape_left = tl.load(
                ape_ptr + ape_group[:, None] * ape_stride_0 + d[None, :] * ape_stride_1,
                mask=ape_mask,
                other=0.0,
            ).to(tl.float32)
            cached_ape_right = tl.load(
                ape_ptr
                + ape_group[:, None] * ape_stride_0
                + (D + d[None, :]) * ape_stride_1,
                mask=ape_mask,
                other=0.0,
            ).to(tl.float32)

    for group_idx in range(pid, VALID_GROUPS, NUM_CORES):
        group_start = group_idx * CMP_RATIO
        if COFF == 2:
            rng = tl.arange(0, CMP_RATIO)
            left_token = group_start - CMP_RATIO + rng
            right_token = group_start + rng
            left_valid = (left_token >= 0) & (left_token < T)
            right_valid = right_token < T
            left_base = left_token[:, None] * (2 * K)
            right_base = right_token[:, None] * (2 * K)
            left_channel = d[None, :]
            right_channel = (D + d)[None, :]
            left_mask = left_valid[:, None] & valid_d[None, :]
            right_mask = right_valid[:, None] & valid_d[None, :]

            kv_left = tl.load(
                state_proj_ptr + left_base + left_channel,
                mask=left_mask,
                other=0.0,
            ).to(tl.float32)
            score_left = tl.load(
                state_proj_ptr + left_base + K + left_channel,
                mask=left_mask,
                other=negative_inf,
            ).to(tl.float32)
            kv_right = tl.load(
                state_proj_ptr + right_base + right_channel,
                mask=right_mask,
                other=0.0,
            ).to(tl.float32)
            score_right = tl.load(
                state_proj_ptr + right_base + K + right_channel,
                mask=right_mask,
                other=negative_inf,
            ).to(tl.float32)
            if SCORE_HAS_APE:
                score_left = tl.where(left_mask, score_left, negative_inf)
                score_right = tl.where(right_mask, score_right, negative_inf)
            else:
                score_left = tl.where(
                    left_mask, score_left + cached_ape_left, negative_inf
                )
                score_right = tl.where(
                    right_mask, score_right + cached_ape_right, negative_inf
                )
            score_max = tl.maximum(
                tl.max(score_left, axis=0), tl.max(score_right, axis=0)
            )
            weight_left = tl.exp(score_left - score_max[None, :])
            weight_right = tl.exp(score_right - score_max[None, :])
            denom = tl.sum(weight_left, axis=0) + tl.sum(weight_right, axis=0)
            comp_final = (
                tl.sum(weight_left * kv_left, axis=0)
                + tl.sum(weight_right * kv_right, axis=0)
            ) / denom
        else:
            score_max = tl.full((BLOCK_D,), negative_inf, dtype=tl.float32)
            comp = tl.zeros((BLOCK_D,), dtype=tl.float32)
            denom = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for g_base in range(0, CMP_RATIO, INNER_G):
                bg = g_base + tl.arange(0, INNER_G)
                token = group_start + bg
                token_valid = (bg < CMP_RATIO) & (token < T)
                base = token[:, None] * (2 * K)
                channel = d[None, :]
                mask = token_valid[:, None] & valid_d[None, :]
                kv = tl.load(
                    state_proj_ptr + base + channel,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.load(
                    state_proj_ptr + base + K + channel,
                    mask=mask,
                    other=negative_inf,
                ).to(tl.float32)
                if SCORE_HAS_APE:
                    score = tl.where(mask, score, negative_inf)
                else:
                    ape_value = tl.load(
                        ape_ptr + bg[:, None] * ape_stride_0 + channel * ape_stride_1,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    score = tl.where(mask, score + ape_value, negative_inf)
                chunk_max = tl.max(score, axis=0)
                new_max = tl.maximum(score_max, chunk_max)
                if g_base > 0:
                    scale = tl.exp(score_max - new_max)
                    comp = comp * scale
                    denom = denom * scale
                score_max = new_max
                weight = tl.exp(score - score_max[None, :])
                comp += tl.sum(weight * kv, axis=0)
                denom += tl.sum(weight, axis=0)
            comp_final = tl.where(denom > 0, comp / denom, 0.0)

        square_sum = tl.sum(comp_final * comp_final, axis=0)
        rstd = 1.0 / tl.sqrt(square_sum / D + NORM_EPS)
        if COFF == 2:
            norm_weight = cached_norm_weight
        else:
            norm_weight = tl.load(norm_weight_ptr + d, mask=valid_d, other=1.0).to(
                tl.float32
            )
        normed = comp_final * rstd * norm_weight
        if ROTARY_MODE == 2 and ROPE_HEAD_DIM > 0:
            _store_normed_with_rope_mode2(
                normed,
                rope_sin_ptr,
                rope_cos_ptr,
                output_ptr,
                group_idx,
                True,
                True,
                D,
                BLOCK_D,
                ROPE_HEAD_DIM,
            )
        else:
            tl.store(
                output_ptr + group_idx * D + d,
                normed,
                mask=valid_d,
            )


@triton.jit
def _compress_projection_batched_kernel(
    state_proj_ptr,
    start_pos_ptr,
    ape_ptr,
    norm_weight_ptr,
    rope_sin_ptr,
    rope_cos_ptr,
    output_ptr,
    token_bases_ptr,
    seq_used_ptr,
    output_bases_ptr,
    n_groups_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    COFF: tl.constexpr,
    NUM_CORES: tl.constexpr,
    HAS_START_POS: tl.constexpr,
    OFFSET_GROUPS: tl.constexpr,
    ape_stride_0: tl.constexpr,
    ape_stride_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TASK_GROUPS: tl.constexpr,
    INNER_G: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    ROTARY_MODE: tl.constexpr,
    ZERO_INVALID: tl.constexpr,
):
    """Compress current-chunk groups directly from contiguous projection."""
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    valid_d = d < D
    negative_inf: tl.constexpr = -3.4028234663852886e38
    total_tasks: tl.constexpr = B * TASK_GROUPS

    if COFF == 2:
        cached_norm_weight = tl.load(norm_weight_ptr + d, mask=valid_d, other=1.0).to(
            tl.float32
        )
        ape_group = tl.arange(0, CMP_RATIO)
        ape_mask = valid_d[None, :]
        cached_ape_left = tl.load(
            ape_ptr + ape_group[:, None] * ape_stride_0 + d[None, :] * ape_stride_1,
            mask=ape_mask,
            other=0.0,
        ).to(tl.float32)
        cached_ape_right = tl.load(
            ape_ptr
            + ape_group[:, None] * ape_stride_0
            + (D + d[None, :]) * ape_stride_1,
            mask=ape_mask,
            other=0.0,
        ).to(tl.float32)

    for task_idx in range(pid, total_tasks, NUM_CORES):
        b_idx = task_idx // TASK_GROUPS
        local_group_idx = task_idx - b_idx * TASK_GROUPS
        token_base = tl.load(token_bases_ptr + b_idx).to(tl.int64)
        seq_len = tl.load(seq_used_ptr + b_idx).to(tl.int64)
        output_base = tl.load(output_bases_ptr + b_idx).to(tl.int64)
        n_groups = tl.load(n_groups_ptr + b_idx).to(tl.int64)
        valid_group = local_group_idx < n_groups
        out_row = output_base + local_group_idx

        if valid_group:
            if OFFSET_GROUPS:
                start = tl.full((), 0, dtype=tl.int64)
                if HAS_START_POS:
                    start = tl.load(start_pos_ptr + b_idx).to(tl.int64)
                group_idx = local_group_idx
                group_idx += (1 if COFF == 2 else 0) + (start % CMP_RATIO != 0)
                group_anchor = start + group_idx * CMP_RATIO
                group_start = (group_anchor // CMP_RATIO) * CMP_RATIO
            else:
                group_start = local_group_idx * CMP_RATIO
            if COFF == 2:
                rng = tl.arange(0, CMP_RATIO)
                if OFFSET_GROUPS:
                    left_pos = group_start - CMP_RATIO + rng
                    right_pos = group_start + rng
                    left_token = left_pos - start
                    right_token = right_pos - start
                    left_valid = (left_token >= 0) & (left_token < seq_len)
                    right_valid = (right_token >= 0) & (right_token < seq_len)
                else:
                    left_token = group_start - CMP_RATIO + rng
                    right_token = group_start + rng
                    left_valid = (left_token >= 0) & (left_token < seq_len)
                    right_valid = right_token < seq_len
                left_base = (token_base + left_token)[:, None] * (2 * K)
                right_base = (token_base + right_token)[:, None] * (2 * K)
                left_channel = d[None, :]
                right_channel = (D + d)[None, :]
                left_mask = left_valid[:, None] & valid_d[None, :]
                right_mask = right_valid[:, None] & valid_d[None, :]

                kv_left = tl.load(
                    state_proj_ptr + left_base + left_channel,
                    mask=left_mask,
                    other=0.0,
                ).to(tl.float32)
                score_left = tl.load(
                    state_proj_ptr + left_base + K + left_channel,
                    mask=left_mask,
                    other=negative_inf,
                ).to(tl.float32)
                kv_right = tl.load(
                    state_proj_ptr + right_base + right_channel,
                    mask=right_mask,
                    other=0.0,
                ).to(tl.float32)
                score_right = tl.load(
                    state_proj_ptr + right_base + K + right_channel,
                    mask=right_mask,
                    other=negative_inf,
                ).to(tl.float32)
                score_left = tl.where(
                    left_mask, score_left + cached_ape_left, negative_inf
                )
                score_right = tl.where(
                    right_mask, score_right + cached_ape_right, negative_inf
                )

                score_max = tl.maximum(
                    tl.max(score_left, axis=0), tl.max(score_right, axis=0)
                )
                weight_left = tl.exp(score_left - score_max[None, :])
                weight_right = tl.exp(score_right - score_max[None, :])
                denom = tl.sum(weight_left, axis=0) + tl.sum(weight_right, axis=0)
                comp_final = (
                    tl.sum(weight_left * kv_left, axis=0)
                    + tl.sum(weight_right * kv_right, axis=0)
                ) / denom
            else:
                score_max = tl.full((BLOCK_D,), negative_inf, dtype=tl.float32)
                comp = tl.zeros((BLOCK_D,), dtype=tl.float32)
                denom = tl.zeros((BLOCK_D,), dtype=tl.float32)
                for g_base in range(0, CMP_RATIO, INNER_G):
                    bg = g_base + tl.arange(0, INNER_G)
                    if OFFSET_GROUPS:
                        pos = group_start + bg
                        local_token = pos - start
                        token_valid = (
                            (bg < CMP_RATIO)
                            & (local_token >= 0)
                            & (local_token < seq_len)
                        )
                    else:
                        local_token = group_start + bg
                        token_valid = (bg < CMP_RATIO) & (local_token < seq_len)
                    base = (token_base + local_token)[:, None] * (2 * K)
                    channel = d[None, :]
                    mask = token_valid[:, None] & valid_d[None, :]
                    kv = tl.load(
                        state_proj_ptr + base + channel,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    score = tl.load(
                        state_proj_ptr + base + K + channel,
                        mask=mask,
                        other=negative_inf,
                    ).to(tl.float32)
                    ape_value = tl.load(
                        ape_ptr + bg[:, None] * ape_stride_0 + channel * ape_stride_1,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                    score = tl.where(mask, score + ape_value, negative_inf)
                    chunk_max = tl.max(score, axis=0)
                    new_max = tl.maximum(score_max, chunk_max)
                    if g_base > 0:
                        scale = tl.exp(score_max - new_max)
                        comp = comp * scale
                        denom = denom * scale
                    score_max = new_max
                    weight = tl.exp(score - score_max[None, :])
                    comp += tl.sum(weight * kv, axis=0)
                    denom += tl.sum(weight, axis=0)
                comp_final = tl.where(denom > 0, comp / denom, 0.0)

            square_sum = tl.sum(comp_final * comp_final, axis=0)
            rstd = 1.0 / tl.sqrt(square_sum / D + NORM_EPS)
            if COFF == 2:
                norm_weight = cached_norm_weight
            else:
                norm_weight = tl.load(norm_weight_ptr + d, mask=valid_d, other=1.0).to(
                    tl.float32
                )
            normed = comp_final * rstd * norm_weight
            if ROTARY_MODE == 2 and ROPE_HEAD_DIM > 0:
                _store_normed_with_rope_mode2(
                    normed,
                    rope_sin_ptr,
                    rope_cos_ptr,
                    output_ptr,
                    out_row,
                    True,
                    True,
                    D,
                    BLOCK_D,
                    ROPE_HEAD_DIM,
                )
            else:
                tl.store(
                    output_ptr + out_row * D + d,
                    normed,
                    mask=valid_d,
                )
        elif ZERO_INVALID:
            tl.store(
                output_ptr + out_row * D + d,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=valid_d,
            )


@triton.jit
def _update_compress_projection_fused_kernel(
    state_proj_ptr,
    state_cache_ptr,
    state_block_table_ptr,
    start_pos_ptr,
    ape_ptr,
    norm_weight_ptr,
    rope_sin_ptr,
    rope_cos_ptr,
    output_ptr,
    token_bases_ptr,
    seq_used_ptr,
    output_bases_ptr,
    n_groups_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    COFF: tl.constexpr,
    UPDATE_PROGRAMS: tl.constexpr,
    FUSED_PROGRAMS: tl.constexpr,
    MAX_S: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    table_stride_0: tl.constexpr,
    table_stride_1: tl.constexpr,
    ape_stride_0: tl.constexpr,
    ape_stride_1: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HAS_START_POS: tl.constexpr,
    CACHE_BLOCK_K: tl.constexpr,
    CACHE_PROGRAM_APE: tl.constexpr,
    TASK_GROUPS: tl.constexpr,
    INNER_G: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    ROTARY_MODE: tl.constexpr,
    ZERO_INVALID: tl.constexpr,
    B1_FAST_PATH: tl.constexpr,
    OFFSET_GROUPS: tl.constexpr,
    VALID_GROUPS: tl.constexpr,
    VALID_ROWS: tl.constexpr,
    FLAT_ROWS: tl.constexpr,
):
    """Fuse Prefill cache update, compression, normalization and RoPE."""
    pid = tl.program_id(0)

    # Compression reads the immutable projection workspace, so it does not
    # depend on cache update completion. Avoid APE writeback to keep the two
    # phases independent and eliminate the need for a global AIV barrier.
    if pid < UPDATE_PROGRAMS:
        _update_state_batched_kernel(
            state_proj_ptr,
            state_cache_ptr,
            state_block_table_ptr,
            start_pos_ptr,
            ape_ptr,
            token_bases_ptr,
            seq_used_ptr,
            B=B,
            K=K,
            CMP_RATIO=CMP_RATIO,
            NUM_CORES=UPDATE_PROGRAMS,
            MAX_S=MAX_S,
            BLOCK_SIZE_CACHE=BLOCK_SIZE_CACHE,
            STATE_BLOCKS=STATE_BLOCKS,
            TABLE_WIDTH=TABLE_WIDTH,
            state_stride_0=state_stride_0,
            state_stride_1=state_stride_1,
            state_stride_2=state_stride_2,
            table_stride_0=table_stride_0,
            table_stride_1=table_stride_1,
            ape_stride_0=ape_stride_0,
            ape_stride_1=ape_stride_1,
            HAS_BLOCK_TABLE=HAS_BLOCK_TABLE,
            HAS_START_POS=HAS_START_POS,
            BLOCK_K=CACHE_BLOCK_K,
            CACHE_PROGRAM_APE=CACHE_PROGRAM_APE,
            WRITEBACK_APE=False,
        )

    if B1_FAST_PATH:
        _compress_projection_b1_kernel(
            state_proj_ptr,
            ape_ptr,
            norm_weight_ptr,
            rope_sin_ptr,
            rope_cos_ptr,
            output_ptr,
            T=T,
            D=D,
            K=K,
            CMP_RATIO=CMP_RATIO,
            COFF=COFF,
            NUM_CORES=FUSED_PROGRAMS,
            ape_stride_0=ape_stride_0,
            ape_stride_1=ape_stride_1,
            BLOCK_D=D,
            VALID_GROUPS=VALID_GROUPS,
            INNER_G=INNER_G,
            NORM_EPS=NORM_EPS,
            ROPE_HEAD_DIM=ROPE_HEAD_DIM,
            ROTARY_MODE=ROTARY_MODE,
            SCORE_HAS_APE=False,
        )
    else:
        _compress_projection_batched_kernel(
            state_proj_ptr,
            start_pos_ptr,
            ape_ptr,
            norm_weight_ptr,
            rope_sin_ptr,
            rope_cos_ptr,
            output_ptr,
            token_bases_ptr,
            seq_used_ptr,
            output_bases_ptr,
            n_groups_ptr,
            B=B,
            D=D,
            K=K,
            CMP_RATIO=CMP_RATIO,
            COFF=COFF,
            NUM_CORES=FUSED_PROGRAMS,
            HAS_START_POS=HAS_START_POS,
            OFFSET_GROUPS=OFFSET_GROUPS,
            ape_stride_0=ape_stride_0,
            ape_stride_1=ape_stride_1,
            BLOCK_D=D,
            TASK_GROUPS=TASK_GROUPS,
            INNER_G=INNER_G,
            NORM_EPS=NORM_EPS,
            ROPE_HEAD_DIM=ROPE_HEAD_DIM,
            ROTARY_MODE=ROTARY_MODE,
            ZERO_INVALID=ZERO_INVALID,
        )

    padding_d = tl.arange(0, D)
    for padding_offset in range(pid, FLAT_ROWS - VALID_ROWS, FUSED_PROGRAMS):
        padding_row = VALID_ROWS + padding_offset
        tl.store(
            output_ptr + padding_row * D + padding_d,
            tl.zeros((D,), dtype=tl.float32),
        )


@triton.jit
def _decode_update_only_kernel(
    state_proj_ptr,
    state_cache_ptr,
    state_block_table_ptr,
    start_pos_ptr,
    ape_ptr,
    token_bases_ptr,
    seq_used_ptr,
    B: tl.constexpr,
    S_MAX: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    COFF: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    table_stride_0: tl.constexpr,
    table_stride_1: tl.constexpr,
    ape_stride_0: tl.constexpr,
    ape_stride_1: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HAS_START_POS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Decode update-only path: assign one program to each batch."""
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)

    for b_idx in range(pid, B, NUM_CORES):
        token_base = tl.load(token_bases_ptr + b_idx).to(tl.int64)
        seq_len = tl.load(seq_used_ptr + b_idx)
        start = tl.full((), 0, dtype=tl.int64)
        if HAS_START_POS:
            start = tl.load(start_pos_ptr + b_idx).to(tl.int64)

        valid_batch = b_idx < B
        valid_d = d < D
        for t_off in tl.static_range(0, S_MAX):
            token_valid = (t_off < seq_len) & valid_batch
            pos = start + t_off
            block_idx_val = pos // BLOCK_SIZE_CACHE
            block_off_val = pos - block_idx_val * BLOCK_SIZE_CACHE

            cache_block = block_idx_val
            valid_block = (
                (pos >= 0) & (block_idx_val >= 0) & (block_idx_val < STATE_BLOCKS)
            )
            if HAS_BLOCK_TABLE:
                table_mask = (block_idx_val >= 0) & (block_idx_val < TABLE_WIDTH)
                cache_block = tl.load(
                    state_block_table_ptr
                    + b_idx * table_stride_0
                    + block_idx_val * table_stride_1,
                    mask=table_mask,
                    other=0,
                ).to(tl.int64)
                valid_block = (
                    table_mask
                    & (cache_block != 0)
                    & (cache_block < STATE_BLOCKS)
                    & (pos >= 0)
                )

            ape_row = tl.minimum(tl.maximum(pos % CMP_RATIO, 0), CMP_RATIO - 1)
            dst_base = cache_block * state_stride_0 + block_off_val * state_stride_1
            for coff_id in tl.static_range(0, COFF):
                coff_d = coff_id * D + d
                src_kv_offsets = (token_base + t_off) * (2 * K) + coff_d
                src_score_offsets = (token_base + t_off) * (2 * K) + K + coff_d
                ape_offsets = ape_row * ape_stride_0 + coff_d * ape_stride_1
                dst_kv = dst_base + coff_d * state_stride_2
                dst_score = dst_base + (K + coff_d) * state_stride_2

                mask = token_valid & valid_block & valid_d
                kv = tl.load(state_proj_ptr + src_kv_offsets, mask=mask, other=0.0)
                score = tl.load(
                    state_proj_ptr + src_score_offsets, mask=mask, other=0.0
                )
                ape = tl.load(ape_ptr + ape_offsets, mask=mask, other=0.0).to(
                    tl.float32
                )
                tl.store(state_cache_ptr + dst_kv, kv, mask=mask)
                tl.store(state_cache_ptr + dst_score, score + ape, mask=mask)


@triton.jit
def _decode_fused_kernel(
    state_proj_ptr,
    state_cache_ptr,
    state_block_table_ptr,
    start_pos_ptr,
    ape_ptr,
    comp_ptr,
    token_bases_ptr,
    seq_used_ptr,
    output_bases_ptr,
    B: tl.constexpr,
    S_MAX: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    COFF: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_SIZE_CACHE: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    state_stride_0: tl.constexpr,
    state_stride_1: tl.constexpr,
    state_stride_2: tl.constexpr,
    table_stride_0: tl.constexpr,
    table_stride_1: tl.constexpr,
    ape_stride_0: tl.constexpr,
    ape_stride_1: tl.constexpr,
    HAS_BLOCK_TABLE: tl.constexpr,
    HAS_START_POS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused update + compress for decode (S <= S_MAX).

    1D grid (vectorcore_num,): each AI Core handles batches via strided assignment.
    Compress: 8 tokens per iteration with online softmax (single-pass, stable).
    BLOCK_D = D; head_dim is validated to be a power of two.
    """
    pid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    NEG_INF: tl.constexpr = -3.4028234663852886e38

    for b_idx in range(pid, B, NUM_CORES):
        # --- Load per-batch params ---
        token_base = tl.load(token_bases_ptr + b_idx).to(tl.int64)
        seq_len = tl.load(seq_used_ptr + b_idx)
        output_base = tl.load(output_bases_ptr + b_idx).to(tl.int64)
        start = tl.full((), 0, dtype=tl.int64)
        if HAS_START_POS:
            start = tl.load(start_pos_ptr + b_idx).to(tl.int64)

        valid_batch = b_idx < B
        valid_group = valid_batch & (
            ((start + seq_len) // CMP_RATIO - start // CMP_RATIO) > 0
        )
        valid_d = d < D

        # ============================================================
        # Phase 1: Update state_cache
        # ============================================================
        for t_off in tl.static_range(0, S_MAX):
            token_valid = (t_off < seq_len) & valid_batch
            pos = start + t_off
            block_idx_val = pos // BLOCK_SIZE_CACHE
            block_off_val = pos - block_idx_val * BLOCK_SIZE_CACHE

            cache_block = block_idx_val
            valid_block = (
                (pos >= 0) & (block_idx_val >= 0) & (block_idx_val < STATE_BLOCKS)
            )
            if HAS_BLOCK_TABLE:
                table_mask = (block_idx_val >= 0) & (block_idx_val < TABLE_WIDTH)
                cache_block = tl.load(
                    state_block_table_ptr
                    + b_idx * table_stride_0
                    + block_idx_val * table_stride_1,
                    mask=table_mask,
                    other=0,
                ).to(tl.int64)
                valid_block = (
                    table_mask
                    & (cache_block != 0)
                    & (cache_block < STATE_BLOCKS)
                    & (pos >= 0)
                )

            safe_ape_row = tl.minimum(tl.maximum(pos % CMP_RATIO, 0), CMP_RATIO - 1)
            dst_base = cache_block * state_stride_0 + block_off_val * state_stride_1

            for coff_id in tl.static_range(0, COFF):
                coff_d = coff_id * D + d
                src_kv_offsets = (token_base + t_off) * (2 * K) + coff_d
                src_score_offsets = (token_base + t_off) * (2 * K) + K + coff_d
                ape_offsets = safe_ape_row * ape_stride_0 + coff_d * ape_stride_1
                dst_kv = dst_base + coff_d * state_stride_2
                dst_score = dst_base + (K + coff_d) * state_stride_2

                wmask = token_valid & valid_block & valid_d
                kv = tl.load(state_proj_ptr + src_kv_offsets, mask=wmask, other=0.0)
                score = tl.load(
                    state_proj_ptr + src_score_offsets, mask=wmask, other=0.0
                )
                ape_val = tl.load(ape_ptr + ape_offsets, mask=wmask, other=0.0).to(
                    tl.float32
                )
                tl.store(state_cache_ptr + dst_kv, kv, mask=wmask)
                tl.store(state_cache_ptr + dst_score, score + ape_val, mask=wmask)

        # ============================================================
        # Phase 2+3: Compress
        # c4 (BLOCK_G ≤ 8): vectorized softmax, one-shot load
        # c128: 8-token batched online softmax
        # ============================================================
        group_idx = 0
        group_anchor = start + group_idx * CMP_RATIO
        group_start = (group_anchor // CMP_RATIO) * CMP_RATIO
        total_g = CMP_RATIO * COFF
        INNER_G: tl.constexpr = 8

        if BLOCK_G <= 8:
            # --- c4: split left/right loads to avoid tl.where in addresses ---
            if COFF == 1:
                g = tl.arange(0, BLOCK_G)
                pos_g = group_start + g
                state_d = d[None, :]
                valid_g = g < total_g

                block_idx_g = pos_g // BLOCK_SIZE_CACHE
                block_off_g = pos_g - block_idx_g * BLOCK_SIZE_CACHE
                cache_block_g = block_idx_g
                valid_block_g = (
                    (pos_g >= 0) & (block_idx_g >= 0) & (block_idx_g < STATE_BLOCKS)
                )
                if HAS_BLOCK_TABLE:
                    tbl_mask = (block_idx_g >= 0) & (block_idx_g < TABLE_WIDTH)
                    cache_block_g = tl.load(
                        state_block_table_ptr
                        + b_idx * table_stride_0
                        + block_idx_g * table_stride_1,
                        mask=tbl_mask,
                        other=0,
                    ).to(tl.int64)
                    valid_block_g = (
                        tbl_mask & (cache_block_g != 0) & (cache_block_g < STATE_BLOCKS)
                    )
                cache_base_g = (
                    cache_block_g[:, None] * state_stride_0
                    + block_off_g[:, None] * state_stride_1
                )
                valid = (
                    valid_g[:, None]
                    & valid_d[None, :]
                    & valid_block_g[:, None]
                    & valid_group
                )
                kv = tl.load(
                    state_cache_ptr + cache_base_g + state_d * state_stride_2,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                score = tl.load(
                    state_cache_ptr + cache_base_g + (K + state_d) * state_stride_2,
                    mask=valid,
                    other=NEG_INF,
                ).to(tl.float32)

                score_max = tl.max(score, axis=0)
                numerator = tl.exp(score - score_max[None, :])
                denominator = tl.sum(numerator, axis=0)
                comp = tl.sum(numerator / denominator[None, :] * kv, axis=0)

            else:
                # COFF == 2: split into left (g=0..3) and right (g=4..7)
                rng = tl.arange(0, CMP_RATIO)  # 0..3
                dd = d[None, :]  # (1, BLOCK_D), kv half of the projection row

                # --- Left tokens: pos = group_start - CMP_RATIO + rng, state_d = d ---
                left_pos = group_start - CMP_RATIO + rng
                left_block_idx = left_pos // BLOCK_SIZE_CACHE
                left_block_off = left_pos - left_block_idx * BLOCK_SIZE_CACHE
                left_cache = left_block_idx
                left_valid_block = (
                    (left_pos >= 0)
                    & (left_block_idx >= 0)
                    & (left_block_idx < STATE_BLOCKS)
                )
                if HAS_BLOCK_TABLE:
                    lt_mask = (left_block_idx >= 0) & (left_block_idx < TABLE_WIDTH)
                    left_cache = tl.load(
                        state_block_table_ptr
                        + b_idx * table_stride_0
                        + left_block_idx * table_stride_1,
                        mask=lt_mask,
                        other=0,
                    ).to(tl.int64)
                    left_valid_block = (
                        lt_mask
                        & (left_cache != 0)
                        & (left_cache < STATE_BLOCKS)
                        & (left_pos >= 0)
                    )
                left_base = (
                    left_cache[:, None] * state_stride_0
                    + left_block_off[:, None] * state_stride_1
                )
                left_valid = (
                    (rng < CMP_RATIO)[:, None]
                    & valid_d
                    & left_valid_block[:, None]
                    & valid_group
                )
                kv_left = tl.load(
                    state_cache_ptr + left_base + dd * state_stride_2,
                    mask=left_valid,
                    other=0.0,
                ).to(tl.float32)
                score_left = tl.load(
                    state_cache_ptr + left_base + (K + dd) * state_stride_2,
                    mask=left_valid,
                    other=NEG_INF,
                ).to(tl.float32)

                # --- Right tokens: pos = group_start + rng, state_d = D + d ---
                right_pos = group_start + rng
                right_block_idx = right_pos // BLOCK_SIZE_CACHE
                right_block_off = right_pos - right_block_idx * BLOCK_SIZE_CACHE
                right_cache = right_block_idx
                right_valid_block = (
                    (right_pos >= 0)
                    & (right_block_idx >= 0)
                    & (right_block_idx < STATE_BLOCKS)
                )
                if HAS_BLOCK_TABLE:
                    rt_mask = (right_block_idx >= 0) & (right_block_idx < TABLE_WIDTH)
                    right_cache = tl.load(
                        state_block_table_ptr
                        + b_idx * table_stride_0
                        + right_block_idx * table_stride_1,
                        mask=rt_mask,
                        other=0,
                    ).to(tl.int64)
                    right_valid_block = (
                        rt_mask
                        & (right_cache != 0)
                        & (right_cache < STATE_BLOCKS)
                        & (right_pos >= 0)
                    )
                right_base = (
                    right_cache[:, None] * state_stride_0
                    + right_block_off[:, None] * state_stride_1
                )
                rcd = (D + d)[None, :]  # (1, BLOCK_D), gate half of the projection row
                right_valid = (
                    (rng < CMP_RATIO)[:, None]
                    & valid_d
                    & right_valid_block[:, None]
                    & valid_group
                )
                kv_right = tl.load(
                    state_cache_ptr + right_base + rcd * state_stride_2,
                    mask=right_valid,
                    other=0.0,
                ).to(tl.float32)
                score_right = tl.load(
                    state_cache_ptr + right_base + (K + rcd) * state_stride_2,
                    mask=right_valid,
                    other=NEG_INF,
                ).to(tl.float32)

                # Two-half softmax: left max + right max, then combined weighted sum
                score_left_max = tl.max(score_left, axis=0)
                score_right_max = tl.max(score_right, axis=0)
                score_max_comb = tl.maximum(score_left_max, score_right_max)

                w_left = tl.exp(score_left - score_max_comb[None, :])
                w_right = tl.exp(score_right - score_max_comb[None, :])
                denom = tl.sum(w_left, axis=0) + tl.sum(w_right, axis=0)
                comp = (
                    tl.sum(w_left * kv_left, axis=0)
                    + tl.sum(w_right * kv_right, axis=0)
                ) / denom
        else:
            # --- c128: batched online softmax ---
            score_max = tl.full((BLOCK_D,), NEG_INF, dtype=tl.float32)
            comp = tl.zeros((BLOCK_D,), dtype=tl.float32)
            denom = tl.zeros((BLOCK_D,), dtype=tl.float32)

            for g_base in range(0, BLOCK_G, INNER_G):
                g = g_base + tl.arange(0, INNER_G)
                valid_g_dim = g < total_g

                if COFF == 1:
                    pos_g = group_start + g
                    state_d = d[None, :]
                else:
                    is_left = g < CMP_RATIO
                    rel = tl.where(is_left, g, g - CMP_RATIO)
                    pos_g = tl.where(
                        is_left, group_start - CMP_RATIO + rel, group_start + rel
                    )
                    state_d = tl.where(is_left[:, None], d[None, :], D + d[None, :])

                block_idx_g = pos_g // BLOCK_SIZE_CACHE
                block_off_g = pos_g - block_idx_g * BLOCK_SIZE_CACHE
                cache_block_g = block_idx_g
                valid_block_g = (
                    (pos_g >= 0) & (block_idx_g >= 0) & (block_idx_g < STATE_BLOCKS)
                )
                if HAS_BLOCK_TABLE:
                    tbl_mask = (block_idx_g >= 0) & (block_idx_g < TABLE_WIDTH)
                    cache_block_g = tl.load(
                        state_block_table_ptr
                        + b_idx * table_stride_0
                        + block_idx_g * table_stride_1,
                        mask=tbl_mask,
                        other=0,
                    ).to(tl.int64)
                    valid_block_g = (
                        tbl_mask & (cache_block_g != 0) & (cache_block_g < STATE_BLOCKS)
                    )

                cache_base_g = (
                    cache_block_g[:, None] * state_stride_0
                    + block_off_g[:, None] * state_stride_1
                )
                valid = (
                    valid_g_dim[:, None]
                    & valid_d[None, :]
                    & valid_block_g[:, None]
                    & valid_group
                )

                kv_g = tl.load(
                    state_cache_ptr + cache_base_g + state_d * state_stride_2,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                score_g = tl.load(
                    state_cache_ptr + cache_base_g + (K + state_d) * state_stride_2,
                    mask=valid,
                    other=NEG_INF,
                ).to(tl.float32)

                chunk_max = tl.max(score_g, axis=0)
                new_max = tl.maximum(score_max, chunk_max)
                scale = tl.exp(score_max - new_max)
                comp = comp * scale
                denom = denom * scale
                score_max = new_max
                w = tl.exp(score_g - score_max[None, :])
                comp += tl.sum(w * kv_g, axis=0)
                denom += tl.sum(w, axis=0)

            comp = tl.where(denom > 0, comp / denom, 0.0)

        out_row = output_base + group_idx
        tl.store(
            comp_ptr + out_row * D + d,
            comp,
            mask=valid_group & valid_d & (out_row >= 0),
        )


@triton.jit
def _postprocess_kernel(
    comp_ptr,
    norm_weight_ptr,
    rope_sin_ptr,
    rope_cos_ptr,
    output_ptr,
    total_rows: tl.constexpr,
    valid_rows,
    D: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROTARY_MODE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)

    for row in range(pid, total_rows, NUM_CORES):
        row_in_bounds = row < total_rows
        valid_row = row < valid_rows
        load_mask = valid_row & (offs < D)
        store_mask = row_in_bounds & (offs < D)

        comp = tl.load(comp_ptr + row * D + offs, mask=load_mask, other=0.0).to(
            tl.float32
        )
        square_sum = tl.sum(comp * comp, axis=0)
        rstd = 1.0 / tl.sqrt(square_sum / D + NORM_EPS)
        weight = tl.load(norm_weight_ptr + offs, mask=offs < D, other=1.0).to(
            tl.float32
        )
        normed = comp * rstd * weight

        if (ROPE_HEAD_DIM > 0 and ROTARY_MODE == 2) and BLOCK_D == D:
            _store_normed_with_rope_mode2(
                normed,
                rope_sin_ptr,
                rope_cos_ptr,
                output_ptr,
                row,
                valid_row,
                row_in_bounds,
                D,
                BLOCK_D,
                ROPE_HEAD_DIM,
            )
        elif ROPE_HEAD_DIM > 0:
            rope_start = D - ROPE_HEAD_DIM
            rope_offs = tl.arange(0, ROPE_HEAD_DIM)

            # Reload the contiguous RoPE tail to avoid a register gather.
            rope_kv = tl.load(
                comp_ptr + row * D + rope_start + rope_offs, mask=valid_row, other=0.0
            ).to(tl.float32)
            rope_nw = tl.load(
                norm_weight_ptr + rope_start + rope_offs, mask=valid_row, other=1.0
            ).to(tl.float32)
            rope_normed = rope_kv * rstd * rope_nw
            rope_sin = tl.load(
                rope_sin_ptr + row * ROPE_HEAD_DIM + rope_offs,
                mask=valid_row,
                other=0.0,
            ).to(tl.float32)
            rope_cos = tl.load(
                rope_cos_ptr + row * ROPE_HEAD_DIM + rope_offs,
                mask=valid_row,
                other=1.0,
            ).to(tl.float32)

            if ROTARY_MODE == 1:
                # mode=1: swap halves, sign=-1 for first half
                half = ROPE_HEAD_DIM // 2
                kv_2d = tl.reshape(rope_normed, (2, half))
                first, second = tl.split(kv_2d)
                sin_2d = tl.reshape(rope_sin, (2, half))
                s1, s2 = tl.split(sin_2d)
                cos_2d = tl.reshape(rope_cos, (2, half))
                c1, c2 = tl.split(cos_2d)
                r1 = first * c1 - second * s1
                r2 = second * c2 + first * s2
                rot_rope = tl.interleave(r1, r2)

            else:
                # mode=2: pairwise even/odd swap
                kv_2d = tl.reshape(rope_normed, (ROPE_HEAD_DIM // 2, 2))
                kv_even, kv_odd = tl.split(kv_2d)
                sin_2d = tl.reshape(rope_sin, (ROPE_HEAD_DIM // 2, 2))
                sin_even, sin_odd = tl.split(sin_2d)
                cos_2d = tl.reshape(rope_cos, (ROPE_HEAD_DIM // 2, 2))
                cos_even, cos_odd = tl.split(cos_2d)
                rot_even = kv_even * cos_even - kv_odd * sin_even
                rot_odd = kv_odd * cos_odd + kv_even * sin_odd
                rot_rope = tl.interleave(rot_even, rot_odd)

            # Write: non-RoPE (0..rope_start-1) and RoPE (rope_start..D-1) in two stores
            tl.store(
                output_ptr + row * D + offs,
                normed,
                mask=store_mask & (offs < rope_start),
            )
            tl.store(
                output_ptr + row * D + rope_start + rope_offs,
                rot_rope,
                mask=row_in_bounds & (rope_offs < ROPE_HEAD_DIM),
            )

        else:
            tl.store(output_ptr + row * D + offs, normed, mask=store_mask)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b if a > 0 else 0


def _compressed_group_count(start: int, used: int, cmp_ratio: int) -> int:
    used = max(int(used), 0)
    start = int(start)
    return max((start + used) // cmp_ratio - start // cmp_ratio, 0)


def _projection_boundary_group_count(
    start: int, groups: int, cmp_ratio: int, coff: int
) -> int:
    """Groups whose compression window reaches before the current chunk."""
    if start <= 0 or groups <= 0:
        return 0
    boundary = (1 if coff == 2 else 0) + (1 if start % cmp_ratio else 0)
    return min(boundary, groups)


def _as_int_list(value, length: int | None = None, default: int = 0) -> list[int]:
    if value is None:
        if length is None:
            return []
        return [default for _ in range(length)]
    if torch.is_tensor(value):
        vals = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        vals = list(value)
    else:
        if length is None:
            return [int(value)]
        return [int(value) for _ in range(length)]
    vals = [int(v) for v in vals]
    if length is not None:
        if len(vals) < length:
            vals.extend([default] * (length - len(vals)))
        vals = vals[:length]
    return vals


def _build_batch_meta(
    x: torch.Tensor,
    rope_sin: torch.Tensor,
    state_block_table: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    seqused: torch.Tensor | None,
    start_pos: torch.Tensor | None,
    cmp_ratio: int,
) -> tuple[list[int], list[int], list[int], list[int], tuple[int, ...], int, int]:
    if x.dim() == 3:
        batch, seq_len, _ = x.shape
        token_bases = [b * seq_len for b in range(batch)]
        seq_lens = [seq_len for _ in range(batch)]
        seq_used = (
            _as_int_list(seqused, batch, seq_len)
            if seqused is not None
            else seq_lens[:]
        )
        start_vals = _as_int_list(start_pos, batch, 0)
        out_per_batch = _ceil_div(seq_len, cmp_ratio)
        output_shape = (batch, out_per_batch, 0)
        flat_rows = batch * out_per_batch
        return (
            token_bases,
            seq_lens,
            seq_used,
            start_vals,
            output_shape,
            flat_rows,
            out_per_batch,
        )

    total_tokens = x.shape[0]
    if cu_seqlens is not None:
        cu_vals = _as_int_list(cu_seqlens)
        batch = max(len(cu_vals) - 1, 0)
        token_bases = cu_vals[:-1]
        seq_lens = [max(cu_vals[i + 1] - cu_vals[i], 0) for i in range(batch)]
    elif state_block_table is not None:
        batch = int(state_block_table.shape[0])
        if batch != 1:
            raise ValueError(
                "cu_seqlens is required for 2D x when batch size is greater than 1"
            )
        token_bases = [0]
        seq_lens = [total_tokens]
    else:
        batch = 1
        token_bases = [0]
        seq_lens = [total_tokens]

    seq_used = _as_int_list(seqused, batch, 0) if seqused is not None else seq_lens[:]
    start_vals = _as_int_list(start_pos, batch, 0)
    flat_rows = int(rope_sin.shape[0])
    output_shape = (flat_rows, 0)
    return token_bases, seq_lens, seq_used, start_vals, output_shape, flat_rows, 0


@dataclass(frozen=True)
class CompressorBatchMetadata:
    """Host-derived metadata that never requires an NPU-to-CPU copy."""

    token_bases: tuple[int, ...]
    seq_used: tuple[int, ...]
    start_vals: tuple[int, ...]
    output_bases: tuple[int, ...]
    max_groups: tuple[int, ...]
    flat_rows: int
    out_per_batch: int
    cursor: int


def _host_int_tuple(
    value: Sequence[int] | torch.Tensor | int | None,
    length: int | None = None,
    default: int = 0,
) -> tuple[int, ...]:
    if value is None:
        return () if length is None else (default,) * length
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError(
                "CPU metadata must be supplied from host memory; implicit NPU-to-CPU copies are disabled"
            )
        values = value.reshape(-1).tolist()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [int(value)] if length is None else [int(value)] * length
    values = [int(item) for item in values]
    if length is not None:
        values = (values + [default] * max(length - len(values), 0))[:length]
    return tuple(values)


def build_compressor_metadata(
    *,
    x_shape: Sequence[int],
    rope_rows: int,
    state_block_table_batch: int | None,
    cu_seqlens: Sequence[int] | torch.Tensor | None,
    seqused: Sequence[int] | torch.Tensor | None,
    start_pos: Sequence[int] | torch.Tensor | int | None,
    cmp_ratio: int,
) -> CompressorBatchMetadata:
    """Build reusable compressor metadata entirely from CPU values."""
    x_shape = tuple(int(dim) for dim in x_shape)
    if len(x_shape) not in (2, 3):
        raise ValueError(f"x_shape must have rank 2 or 3, got {x_shape}")
    if cmp_ratio <= 0:
        raise ValueError("cmp_ratio should be greater than 0")

    if len(x_shape) == 3:
        batch, seq_len, _ = x_shape
        token_bases = tuple(batch_idx * seq_len for batch_idx in range(batch))
        seq_used = (
            _host_int_tuple(seqused, batch, seq_len)
            if seqused is not None
            else (seq_len,) * batch
        )
        start_vals = _host_int_tuple(start_pos, batch, 0)
        out_per_batch = _ceil_div(seq_len, cmp_ratio)
        flat_rows = batch * out_per_batch
        output_bases = tuple(batch_idx * out_per_batch for batch_idx in range(batch))
        max_groups = tuple(
            min(_compressed_group_count(start, used, cmp_ratio), out_per_batch)
            for start, used in zip(start_vals, seq_used)
        )
    else:
        total_tokens = x_shape[0]
        if cu_seqlens is not None:
            cu_vals = _host_int_tuple(cu_seqlens)
            batch = max(len(cu_vals) - 1, 0)
            token_bases = cu_vals[:-1]
            seq_lens = tuple(
                max(cu_vals[idx + 1] - cu_vals[idx], 0) for idx in range(batch)
            )
        elif state_block_table_batch is not None:
            batch = int(state_block_table_batch)
            if batch != 1:
                raise ValueError(
                    "cu_seqlens is required for 2D x when batch size is greater than 1"
                )
            token_bases = (0,)
            seq_lens = (total_tokens,)
        else:
            batch = 1
            token_bases = (0,)
            seq_lens = (total_tokens,)

        seq_used = (
            _host_int_tuple(seqused, batch, 0) if seqused is not None else seq_lens
        )
        start_vals = _host_int_tuple(start_pos, batch, 0)
        flat_rows = int(rope_rows)
        out_per_batch = 0
        output_bases_list = []
        max_groups_list = []
        cursor = 0
        for start, used in zip(start_vals, seq_used):
            output_bases_list.append(cursor)
            groups = _compressed_group_count(start, used, cmp_ratio)
            groups = max(min(groups, flat_rows - cursor), 0)
            max_groups_list.append(groups)
            cursor += groups
        output_bases = tuple(output_bases_list)
        max_groups = tuple(max_groups_list)

    return CompressorBatchMetadata(
        token_bases=tuple(token_bases),
        seq_used=tuple(seq_used),
        start_vals=tuple(start_vals),
        output_bases=tuple(output_bases),
        max_groups=tuple(max_groups),
        flat_rows=flat_rows,
        out_per_batch=out_per_batch,
        cursor=sum(max_groups),
    )


@dataclass
class CompressorWorkspace:
    token_bases: tuple[int, ...]
    seq_used: tuple[int, ...]
    start_vals: tuple[int, ...]
    output_bases: tuple[int, ...]
    max_groups: tuple[int, ...]
    projection_boundary_groups: tuple[int, ...]
    projection_output_bases: tuple[int, ...]
    projection_groups: tuple[int, ...]
    flat_rows: int
    out_per_batch: int
    cursor: int
    project_rows: int
    # Single source of truth for the projection BLOCK_M: state_proj is allocated
    # for project_rows in prepare, so the kernels must consume the same value here.
    block_m: int
    output: torch.Tensor
    state_proj: torch.Tensor
    comp: torch.Tensor | None
    token_bases_t: torch.Tensor
    seq_used_t: torch.Tensor
    output_bases_t: torch.Tensor
    n_groups_t: torch.Tensor
    projection_boundary_groups_t: torch.Tensor
    projection_output_bases_t: torch.Tensor
    projection_groups_t: torch.Tensor
    launches: tuple[Callable[[], None], ...] = ()


def _validate_compressor_inputs(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    rotary_mode: int,
    cache_mode: int,
) -> None:
    if x.dim() not in (2, 3):
        raise ValueError(f"x dim num[{x.dim()}] should be 2 or 3")
    if rope_sin.dim() != x.dim() or rope_cos.dim() != x.dim():
        raise ValueError("rope_sin and rope_cos dim must match x dim")
    if cmp_ratio <= 0:
        raise ValueError("cmp_ratio should be greater than 0")
    if coff not in (1, 2):
        raise ValueError("coff must be 1 or 2")
    if rotary_mode not in (1, 2):
        raise ValueError("rotary_mode must be 1 or 2")
    if cache_mode != 1:
        raise ValueError("only cache_mode=1 is supported")

    hidden_size = x.shape[-1]
    head_dim = int(norm_weight.shape[0])
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(
            f"head_dim[{head_dim}] must be a positive power of two: compress "
            "kernels use BLOCK_D == head_dim"
        )
    projection_dim = coff * head_dim
    expected_weight_shape = (projection_dim, hidden_size)
    if wkv.shape != expected_weight_shape or wgate.shape != expected_weight_shape:
        raise ValueError(
            f"wkv/wgate must be {expected_weight_shape}, got {tuple(wkv.shape)} and {tuple(wgate.shape)}"
        )
    if state_cache.dim() != 3 or state_cache.shape[2] < 2 * projection_dim:
        raise ValueError(
            "state_cache must have shape [blocks, block_size, at least 2 * coff * head_dim]"
        )
    if ape.dim() != 2 or ape.shape[0] < cmp_ratio or ape.shape[1] < projection_dim:
        raise ValueError(
            f"ape must have at least shape ({cmp_ratio}, {projection_dim})"
        )
    if rope_head_dim < 0 or rope_head_dim > head_dim:
        raise ValueError("rope_head_dim must be between 0 and head_dim")
    if rope_sin.shape[-1] != rope_head_dim or rope_cos.shape[-1] != rope_head_dim:
        raise ValueError(
            f"rope_sin and rope_cos last dimension must equal rope_head_dim={rope_head_dim}"
        )


def prepare_compressor_workspace(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_block_table: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    start_pos: torch.Tensor | None = None,
    rope_head_dim: int = 64,
    cmp_ratio: int = 128,
    coff: int = 1,
    norm_eps: float = 1e-6,
    rotary_mode: int = 2,
    cache_mode: int = 1,
    metadata: CompressorBatchMetadata | None = None,
    out: torch.Tensor | None = None,
) -> CompressorWorkspace:
    """Prepare metadata, output, temporary storage, and bound eager launches."""
    _validate_compressor_inputs(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight,
        rope_sin,
        rope_cos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        coff=coff,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )

    if metadata is None:
        token_bases, _, seq_used, start_vals, _, flat_rows, out_per_batch = (
            _build_batch_meta(
                x,
                rope_sin,
                state_block_table,
                cu_seqlens,
                seqused,
                start_pos,
                cmp_ratio,
            )
        )
        metadata = build_compressor_metadata(
            x_shape=tuple(x.shape),
            rope_rows=flat_rows,
            state_block_table_batch=(
                int(state_block_table.shape[0])
                if state_block_table is not None
                else None
            ),
            cu_seqlens=(
                tuple(token_bases) + (x.shape[0],)
                if x.dim() == 2 and cu_seqlens is not None
                else None
            ),
            seqused=seq_used,
            start_pos=start_vals,
            cmp_ratio=cmp_ratio,
        )

    token_bases = metadata.token_bases
    seq_used = metadata.seq_used
    start_vals = metadata.start_vals
    output_bases = metadata.output_bases
    max_groups = metadata.max_groups
    flat_rows = metadata.flat_rows
    out_per_batch = metadata.out_per_batch
    hidden_size = x.shape[-1]
    head_dim = int(norm_weight.shape[0])
    proj_dim = coff * head_dim
    total_tokens = x.numel() // hidden_size

    if x.dim() == 3:
        final_shape = (len(token_bases), out_per_batch, head_dim)
    else:
        final_shape = (flat_rows, head_dim)

    if out is None:
        output = torch.empty(final_shape, device=x.device, dtype=x.dtype)
    else:
        if (
            tuple(out.shape) != final_shape
            or out.device != x.device
            or out.dtype != x.dtype
        ):
            raise ValueError(
                f"out must have shape {final_shape}, device {x.device}, and dtype {x.dtype}"
            )
        output = out

    cursor = metadata.cursor
    projection_boundary_groups = tuple(
        _projection_boundary_group_count(start, groups, cmp_ratio, coff)
        for start, groups in zip(start_vals, max_groups)
    )
    projection_output_bases = tuple(
        output_base + boundary
        for output_base, boundary in zip(output_bases, projection_boundary_groups)
    )
    projection_groups = tuple(
        groups - boundary
        for groups, boundary in zip(max_groups, projection_boundary_groups)
    )
    wide_projection = total_tokens >= 128 and coff == 1 and head_dim >= 512
    if wide_projection:
        block_m = 128
    else:
        block_m = (128 if coff == 2 else 256) if total_tokens >= 128 else 32
    project_rows = _ceil_div(total_tokens, block_m) * block_m
    state_proj = torch.empty(
        (project_rows, 2 * proj_dim), device=x.device, dtype=torch.float32
    )

    max_s = max(seq_used) if seq_used else 0
    needs_comp = total_tokens > 0 and flat_rows > 0 and (max_s > 4 or cursor > 0)
    comp = (
        torch.empty((flat_rows, head_dim), device=x.device, dtype=torch.float32)
        if needs_comp
        else None
    )

    def int32_tensor(values):
        return torch.tensor(values, device=x.device, dtype=torch.int32)

    workspace = CompressorWorkspace(
        token_bases=tuple(token_bases),
        seq_used=tuple(seq_used),
        start_vals=tuple(start_vals),
        output_bases=tuple(output_bases),
        max_groups=tuple(max_groups),
        projection_boundary_groups=projection_boundary_groups,
        projection_output_bases=projection_output_bases,
        projection_groups=projection_groups,
        flat_rows=flat_rows,
        out_per_batch=out_per_batch,
        cursor=cursor,
        project_rows=project_rows,
        block_m=block_m,
        output=output,
        state_proj=state_proj,
        comp=comp,
        token_bases_t=int32_tensor(token_bases),
        seq_used_t=int32_tensor(seq_used),
        output_bases_t=int32_tensor(output_bases),
        n_groups_t=int32_tensor(max_groups),
        projection_boundary_groups_t=int32_tensor(projection_boundary_groups),
        projection_output_bases_t=int32_tensor(projection_output_bases),
        projection_groups_t=int32_tensor(projection_groups),
    )
    workspace.launches = _bind_compressor_launches(
        workspace,
        x=x,
        wkv=wkv,
        wgate=wgate,
        state_cache=state_cache,
        ape=ape,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        state_block_table=state_block_table,
        start_pos=start_pos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=norm_eps,
        rotary_mode=rotary_mode,
    )
    return workspace


def _bound_zero_launch(tensor: torch.Tensor) -> Callable[[], None] | None:
    if tensor.numel() == 0:
        return None
    block = 1024
    grid = (triton.cdiv(tensor.numel(), block),)
    return partial(_zero_kernel[grid], tensor, tensor.numel(), BLOCK_SIZE=block)


def _bind_compressor_launches(
    workspace: CompressorWorkspace,
    *,
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_block_table: torch.Tensor | None,
    start_pos: torch.Tensor | None,
    rope_head_dim: int,
    cmp_ratio: int,
    coff: int,
    norm_eps: float,
    rotary_mode: int,
) -> tuple[Callable[[], None], ...]:
    """Bind all static arguments once so eager calls only enqueue kernels."""
    hidden_size = x.shape[-1]
    head_dim = int(norm_weight.shape[0])
    proj_dim = coff * head_dim
    total_tokens = x.numel() // hidden_size
    launches: list[Callable[[], None]] = []

    if total_tokens == 0 or workspace.flat_rows == 0:
        zero_launch = _bound_zero_launch(workspace.output)
        return () if zero_launch is None else (zero_launch,)

    x_2d = x.contiguous().reshape(total_tokens, hidden_size)
    wkv_t = wkv.contiguous()
    wgate_t = wgate.contiguous()
    ape_t = ape.contiguous()
    norm_weight_t = norm_weight.contiguous()
    rope_sin_t = rope_sin.contiguous().reshape(-1, rope_head_dim)
    rope_cos_t = rope_cos.contiguous().reshape(-1, rope_head_dim)

    has_block_table = state_block_table is not None
    has_start_pos = start_pos is not None
    table_width = (
        int(state_block_table.shape[1])
        if has_block_table
        else int(state_cache.shape[0])
    )
    table_stride_0 = int(state_block_table.stride(0)) if has_block_table else 0
    table_stride_1 = int(state_block_table.stride(1)) if has_block_table else 0
    block_table_arg = state_block_table if has_block_table else None
    start_pos_arg = start_pos if has_start_pos else None

    wide_projection = total_tokens >= 128 and coff == 1 and head_dim >= 512
    block_m = workspace.block_m
    block_d = (
        64
        if total_tokens >= 128 and head_dim >= 64
        else (32 if head_dim >= 32 else triton.next_power_of_2(head_dim))
    )
    project_programs = 24
    project_blocks_m = _ceil_div(total_tokens, block_m)
    if total_tokens >= 128:
        block_n = 256 if wide_projection else 128
        project_blocks_n = _ceil_div(2 * proj_dim, block_n)
        project_tiles = project_blocks_m * project_blocks_n
        project_tile_rounds = _ceil_div(project_tiles, project_programs)
        guarded_project_tiles = (
            project_tiles
            if (coff == 2 and head_dim == 128) or coff == 1
            else project_tile_rounds * project_programs
        )
        # Only the concat path needs the fused (2 * proj_dim, hidden) weight;
        # the pair path below reads wkv_t / wgate_t directly.
        projection_weight_t = torch.cat((wkv_t, wgate_t), dim=0)
        project_args = (
            x_2d,
            projection_weight_t,
            workspace.state_proj,
            total_tokens,
            workspace.project_rows,
            hidden_size,
            2 * proj_dim,
            x_2d.stride(0),
            x_2d.stride(1),
            projection_weight_t.stride(0),
            projection_weight_t.stride(1),
        )
        project_constexprs = {
            "BLOCKS_N": project_blocks_n,
            "NUM_TILES": guarded_project_tiles,
            "NUM_PROGRAMS": project_programs,
            "NUM_TILE_ROUNDS": project_tile_rounds,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
        }
        block_k = _select_projection_block_k(
            cache_key=(
                "concat",
                str(x_2d.dtype),
                total_tokens,
                workspace.project_rows,
                hidden_size,
                2 * proj_dim,
                x_2d.stride(0),
                x_2d.stride(1),
                projection_weight_t.stride(0),
                projection_weight_t.stride(1),
                *project_constexprs.values(),
            ),
            kernel_name="project_concat",
            resource_model=_concat_projection_resource_model(
                block_m, block_n, x_2d.element_size()
            ),
            h=hidden_size,
        )
        launches.append(
            partial(
                _project_concat_kernel[(project_programs,)],
                *project_args,
                **project_constexprs,
                BLOCK_K=block_k,
            )
        )
    else:
        project_blocks_d = _ceil_div(head_dim, block_d)
        project_tiles = project_blocks_m * project_blocks_d
        if head_dim <= 128:
            project_programs = 8
        project_tile_rounds = _ceil_div(project_tiles, project_programs)
        project_args = (
            x_2d,
            wkv_t,
            wgate_t,
            workspace.state_proj,
            total_tokens,
            workspace.project_rows,
            hidden_size,
            proj_dim,
            head_dim,
            x_2d.stride(0),
            x_2d.stride(1),
            wkv_t.stride(0),
            wkv_t.stride(1),
        )
        project_constexprs = {
            "BLOCKS_D": project_blocks_d,
            "NUM_TILES": project_tiles,
            "NUM_PROGRAMS": project_programs,
            "NUM_TILE_ROUNDS": project_tile_rounds,
            "COFF": coff,
            "BLOCK_M": block_m,
            "BLOCK_D": block_d,
        }
        block_k = _select_projection_block_k(
            cache_key=(
                "pair",
                str(x_2d.dtype),
                total_tokens,
                workspace.project_rows,
                hidden_size,
                proj_dim,
                head_dim,
                x_2d.stride(0),
                x_2d.stride(1),
                wkv_t.stride(0),
                wkv_t.stride(1),
                *project_constexprs.values(),
            ),
            kernel_name="project_pair",
            resource_model=_pair_projection_resource_model(
                block_m, block_d, x_2d.element_size()
            ),
            h=hidden_size,
        )
        launches.append(
            partial(
                _project_pair_kernel[(project_programs,)],
                *project_args,
                **project_constexprs,
                BLOCK_K=block_k,
            )
        )

    max_s = max(workspace.seq_used) if workspace.seq_used else 0
    batch_count = len(workspace.token_bases)
    vc_num, _ = _get_npu_core_counts()
    cache_args = {
        "BLOCK_SIZE_CACHE": int(state_cache.shape[1]),
        "STATE_BLOCKS": int(state_cache.shape[0]),
        "TABLE_WIDTH": table_width,
        "state_stride_0": int(state_cache.stride(0)),
        "state_stride_1": int(state_cache.stride(1)),
        "state_stride_2": int(state_cache.stride(2)),
        "table_stride_0": table_stride_0,
        "table_stride_1": table_stride_1,
        "HAS_BLOCK_TABLE": has_block_table,
        "HAS_START_POS": has_start_pos,
    }

    if max_s <= 4:
        fused_block_g = triton.next_power_of_2(cmp_ratio * coff)
        fused_block_d = head_dim
        if workspace.cursor > 0:
            assert workspace.comp is not None
            if x.dim() == 3:
                zero_launch = _bound_zero_launch(workspace.comp)
                if zero_launch is not None:
                    launches.append(zero_launch)
            launches.append(
                partial(
                    _decode_fused_kernel[(min(batch_count, vc_num),)],
                    workspace.state_proj,
                    state_cache,
                    block_table_arg,
                    start_pos_arg,
                    ape_t,
                    workspace.comp,
                    workspace.token_bases_t,
                    workspace.seq_used_t,
                    workspace.output_bases_t,
                    B=batch_count,
                    S_MAX=max_s,
                    D=head_dim,
                    K=proj_dim,
                    CMP_RATIO=cmp_ratio,
                    COFF=coff,
                    NUM_CORES=vc_num,
                    ape_stride_0=int(ape_t.stride(0)),
                    ape_stride_1=int(ape_t.stride(1)),
                    BLOCK_G=fused_block_g,
                    BLOCK_D=fused_block_d,
                    **cache_args,
                )
            )
            post_valid_rows = workspace.cursor if x.dim() == 2 else workspace.flat_rows
            launches.append(
                partial(
                    _postprocess_kernel[(min(workspace.flat_rows, vc_num),)],
                    workspace.comp,
                    norm_weight_t,
                    rope_sin_t,
                    rope_cos_t,
                    workspace.output,
                    workspace.flat_rows,
                    post_valid_rows,
                    head_dim,
                    rope_head_dim,
                    float(norm_eps),
                    rotary_mode,
                    NUM_CORES=vc_num,
                    BLOCK_D=triton.next_power_of_2(head_dim),
                )
            )
        else:
            launches.append(
                partial(
                    _decode_update_only_kernel[(min(batch_count, vc_num),)],
                    workspace.state_proj,
                    state_cache,
                    block_table_arg,
                    start_pos_arg,
                    ape_t,
                    workspace.token_bases_t,
                    workspace.seq_used_t,
                    B=batch_count,
                    S_MAX=max_s,
                    D=head_dim,
                    K=proj_dim,
                    CMP_RATIO=cmp_ratio,
                    COFF=coff,
                    NUM_CORES=vc_num,
                    ape_stride_0=int(ape_t.stride(0)),
                    ape_stride_1=int(ape_t.stride(1)),
                    BLOCK_D=fused_block_d,
                    **cache_args,
                )
            )
            zero_launch = _bound_zero_launch(workspace.output)
            if zero_launch is not None:
                launches.append(zero_launch)
        return tuple(launches)

    assert workspace.comp is not None
    fuse_prefill_post = rotary_mode == 2
    compress_from_projection_b1 = (
        fuse_prefill_post
        and x.dim() == 2
        and batch_count == 1
        and workspace.token_bases == (0,)
        and workspace.seq_used == (total_tokens,)
        and workspace.start_vals == (0,)
    )
    projection_batches_in_bounds = all(
        token_base >= 0 and seq_used >= 0 and token_base + seq_used <= total_tokens
        for token_base, seq_used in zip(workspace.token_bases, workspace.seq_used)
    )
    compress_from_projection_batched = (
        fuse_prefill_post
        and not compress_from_projection_b1
        and projection_batches_in_bounds
    )
    hybrid_projection_cache = compress_from_projection_batched and any(
        start != 0 for start in workspace.start_vals
    )
    if hybrid_projection_cache:
        projection_output_bases_t = workspace.projection_output_bases_t
        projection_n_groups_t = workspace.projection_groups_t
        max_projection_groups = (
            max(workspace.projection_groups) if workspace.projection_groups else 0
        )
    else:
        projection_output_bases_t = workspace.output_bases_t
        projection_n_groups_t = workspace.n_groups_t
        max_projection_groups = max(workspace.max_groups) if workspace.max_groups else 0
    projection_task_groups = (
        workspace.out_per_batch
        if x.dim() == 3 and not hybrid_projection_cache
        else max_projection_groups
    )
    max_boundary_groups = (
        max(workspace.projection_boundary_groups)
        if hybrid_projection_cache and workspace.projection_boundary_groups
        else 0
    )
    writeback_ape_scores = compress_from_projection_b1 and coff == 2 and head_dim == 128
    max_n_groups = max(workspace.max_groups) if workspace.max_groups else 0
    block_g = triton.next_power_of_2(cmp_ratio * coff)
    compress_programs = vc_num
    cache_block_k = min(512, triton.next_power_of_2(proj_dim))
    cache_program_ape = (
        cmp_ratio == 4 and proj_dim <= cache_block_k and vc_num % cmp_ratio == 0
    )
    two_kernel_prefill = (
        os.getenv("COMPRESSOR_PREFILL_TWO_KERNEL_FUSION", "1") == "1"
        and max_s > 0
        and max_n_groups > 0
        and (compress_from_projection_b1 or compress_from_projection_batched)
    )
    if two_kernel_prefill:
        launches.append(
            partial(
                _update_compress_projection_fused_kernel[(compress_programs,)],
                workspace.state_proj,
                state_cache,
                block_table_arg,
                start_pos_arg,
                ape_t,
                norm_weight_t,
                rope_sin_t,
                rope_cos_t,
                workspace.output,
                workspace.token_bases_t,
                workspace.seq_used_t,
                projection_output_bases_t,
                projection_n_groups_t,
                B=batch_count,
                T=total_tokens,
                D=head_dim,
                K=proj_dim,
                CMP_RATIO=cmp_ratio,
                COFF=coff,
                UPDATE_PROGRAMS=vc_num,
                FUSED_PROGRAMS=compress_programs,
                MAX_S=max_s,
                ape_stride_0=int(ape_t.stride(0)),
                ape_stride_1=int(ape_t.stride(1)),
                CACHE_BLOCK_K=cache_block_k,
                CACHE_PROGRAM_APE=cache_program_ape,
                TASK_GROUPS=projection_task_groups,
                INNER_G=8,
                NORM_EPS=float(norm_eps),
                ROPE_HEAD_DIM=rope_head_dim,
                ROTARY_MODE=rotary_mode,
                ZERO_INVALID=x.dim() == 3 and not hybrid_projection_cache,
                B1_FAST_PATH=compress_from_projection_b1,
                OFFSET_GROUPS=hybrid_projection_cache,
                VALID_GROUPS=workspace.cursor,
                VALID_ROWS=workspace.cursor if x.dim() == 2 else workspace.flat_rows,
                FLAT_ROWS=workspace.flat_rows,
                **cache_args,
            )
        )
        if max_boundary_groups > 0:
            boundary_programs = min(max_boundary_groups, vc_num)
            launches.append(
                partial(
                    _compress_groups_batched_kernel[(boundary_programs,)],
                    state_cache,
                    block_table_arg,
                    start_pos_arg,
                    workspace.comp,
                    norm_weight_t,
                    rope_sin_t,
                    rope_cos_t,
                    workspace.output,
                    workspace.output_bases_t,
                    workspace.projection_boundary_groups_t,
                    B=batch_count,
                    D=head_dim,
                    K=proj_dim,
                    CMP_RATIO=cmp_ratio,
                    COFF=coff,
                    NUM_CORES=boundary_programs,
                    BLOCK_G=block_g,
                    BLOCK_D=head_dim,
                    MAX_GROUPS=max_boundary_groups,
                    INNER_G=8,
                    ROPE_HEAD_DIM=rope_head_dim,
                    NORM_EPS=float(norm_eps),
                    ROTARY_MODE=rotary_mode,
                    FUSE_POST=True,
                    **cache_args,
                )
            )
        return tuple(launches)

    if fuse_prefill_post:
        if x.dim() == 3:
            zero_launch = (
                None
                if (
                    compress_from_projection_batched
                    and not hybrid_projection_cache
                    and workspace.cursor > 0
                )
                else _bound_zero_launch(workspace.output)
            )
        else:
            padding = workspace.output[workspace.cursor :]
            zero_launch = _bound_zero_launch(padding)
        if zero_launch is not None:
            launches.append(zero_launch)
    elif x.dim() == 3:
        zero_launch = _bound_zero_launch(workspace.comp)
        if zero_launch is not None:
            launches.append(zero_launch)

    if max_s > 0:
        launches.append(
            partial(
                _update_state_batched_kernel[(vc_num,)],
                workspace.state_proj,
                state_cache,
                block_table_arg,
                start_pos_arg,
                ape_t,
                workspace.token_bases_t,
                workspace.seq_used_t,
                B=batch_count,
                K=proj_dim,
                CMP_RATIO=cmp_ratio,
                NUM_CORES=vc_num,
                MAX_S=max_s,
                ape_stride_0=int(ape_t.stride(0)),
                ape_stride_1=int(ape_t.stride(1)),
                BLOCK_K=cache_block_k,
                CACHE_PROGRAM_APE=cache_program_ape,
                WRITEBACK_APE=writeback_ape_scores,
                **cache_args,
            )
        )

    if max_n_groups > 0:
        if compress_from_projection_b1:
            launches.append(
                partial(
                    _compress_projection_b1_kernel[(compress_programs,)],
                    workspace.state_proj,
                    ape_t,
                    norm_weight_t,
                    rope_sin_t,
                    rope_cos_t,
                    workspace.output,
                    T=total_tokens,
                    D=head_dim,
                    K=proj_dim,
                    CMP_RATIO=cmp_ratio,
                    COFF=coff,
                    NUM_CORES=compress_programs,
                    ape_stride_0=int(ape_t.stride(0)),
                    ape_stride_1=int(ape_t.stride(1)),
                    BLOCK_D=head_dim,
                    VALID_GROUPS=workspace.cursor,
                    INNER_G=8,
                    NORM_EPS=float(norm_eps),
                    ROPE_HEAD_DIM=rope_head_dim,
                    ROTARY_MODE=rotary_mode,
                    SCORE_HAS_APE=writeback_ape_scores,
                )
            )
        elif compress_from_projection_batched:
            launches.append(
                partial(
                    _compress_projection_batched_kernel[(compress_programs,)],
                    workspace.state_proj,
                    start_pos_arg,
                    ape_t,
                    norm_weight_t,
                    rope_sin_t,
                    rope_cos_t,
                    workspace.output,
                    workspace.token_bases_t,
                    workspace.seq_used_t,
                    projection_output_bases_t,
                    projection_n_groups_t,
                    B=batch_count,
                    D=head_dim,
                    K=proj_dim,
                    CMP_RATIO=cmp_ratio,
                    COFF=coff,
                    NUM_CORES=compress_programs,
                    HAS_START_POS=has_start_pos,
                    OFFSET_GROUPS=hybrid_projection_cache,
                    ape_stride_0=int(ape_t.stride(0)),
                    ape_stride_1=int(ape_t.stride(1)),
                    BLOCK_D=head_dim,
                    TASK_GROUPS=projection_task_groups,
                    INNER_G=8,
                    NORM_EPS=float(norm_eps),
                    ROPE_HEAD_DIM=rope_head_dim,
                    ROTARY_MODE=rotary_mode,
                    ZERO_INVALID=x.dim() == 3 and not hybrid_projection_cache,
                )
            )
        else:
            launches.append(
                partial(
                    _compress_groups_batched_kernel[(vc_num,)],
                    state_cache,
                    block_table_arg,
                    start_pos_arg,
                    workspace.comp,
                    norm_weight_t,
                    rope_sin_t,
                    rope_cos_t,
                    workspace.output,
                    workspace.output_bases_t,
                    workspace.n_groups_t,
                    B=batch_count,
                    D=head_dim,
                    K=proj_dim,
                    CMP_RATIO=cmp_ratio,
                    COFF=coff,
                    NUM_CORES=vc_num,
                    BLOCK_G=block_g,
                    BLOCK_D=head_dim,
                    MAX_GROUPS=max_n_groups,
                    INNER_G=8,
                    ROPE_HEAD_DIM=rope_head_dim,
                    NORM_EPS=float(norm_eps),
                    ROTARY_MODE=rotary_mode,
                    FUSE_POST=fuse_prefill_post,
                    **cache_args,
                )
            )
    if max_boundary_groups > 0:
        boundary_programs = min(max_boundary_groups, vc_num)
        launches.append(
            partial(
                _compress_groups_batched_kernel[(boundary_programs,)],
                state_cache,
                block_table_arg,
                start_pos_arg,
                workspace.comp,
                norm_weight_t,
                rope_sin_t,
                rope_cos_t,
                workspace.output,
                workspace.output_bases_t,
                workspace.projection_boundary_groups_t,
                B=batch_count,
                D=head_dim,
                K=proj_dim,
                CMP_RATIO=cmp_ratio,
                COFF=coff,
                NUM_CORES=boundary_programs,
                BLOCK_G=block_g,
                BLOCK_D=head_dim,
                MAX_GROUPS=max_boundary_groups,
                INNER_G=8,
                ROPE_HEAD_DIM=rope_head_dim,
                NORM_EPS=float(norm_eps),
                ROTARY_MODE=rotary_mode,
                FUSE_POST=fuse_prefill_post,
                **cache_args,
            )
        )
    if not fuse_prefill_post:
        post_valid_rows = workspace.cursor if x.dim() == 2 else workspace.flat_rows
        launches.append(
            partial(
                _postprocess_kernel[(min(workspace.flat_rows, vc_num),)],
                workspace.comp,
                norm_weight_t,
                rope_sin_t,
                rope_cos_t,
                workspace.output,
                workspace.flat_rows,
                post_valid_rows,
                head_dim,
                rope_head_dim,
                float(norm_eps),
                rotary_mode,
                NUM_CORES=vc_num,
                BLOCK_D=triton.next_power_of_2(head_dim),
            )
        )
    return tuple(launches)


def _run_compressor_launches(workspace: CompressorWorkspace) -> None:
    for launch in workspace.launches:
        launch()


def compressor_prepared(
    workspace: CompressorWorkspace,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute a prepared compressor plan without graph capture or host rebuilding."""
    if out is not None and out is not workspace.output:
        raise ValueError("out must be the tensor bound by prepare_compressor_workspace")
    _run_compressor_launches(workspace)
    return workspace.output


def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_block_table: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    start_pos: torch.Tensor | None = None,
    rope_head_dim: int = 64,
    cmp_ratio: int = 128,
    coff: int = 1,
    norm_eps: float = 1e-6,
    rotary_mode: int = 2,
    cache_mode: int = 1,
    workspace: CompressorWorkspace | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Compressor directly or execute an already prepared launch plan."""
    if workspace is None:
        workspace = prepare_compressor_workspace(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            rope_sin,
            rope_cos,
            state_block_table,
            cu_seqlens,
            seqused,
            start_pos,
            rope_head_dim,
            cmp_ratio,
            coff,
            norm_eps,
            rotary_mode,
            cache_mode,
            out=out,
        )
    return compressor_prepared(workspace, out=out)


__all__ = [
    "CompressorBatchMetadata",
    "CompressorWorkspace",
    "build_compressor_metadata",
    "compressor",
    "compressor_prepared",
    "prepare_compressor_workspace",
]
