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

# enflame-local grouped_topk.
#
# The generic implementation packs (fp32 value, index) pairs into uint64 keys
# and runs warp-level bitonic sorts on 2-D score tiles
# (`triton_grouped_topk_fused_small_expert_count_kernel`). On the GCU backend
# that shape cannot be used: fresh compiles hit the `memref_ext.slice_start`
# segment-size regression introduced by the 2026-11-07 backend reinstall
# (2-D broadcast tiles; same cluster as compiler_issues/vllm_outer.md).
# (The baseline FAIL 0/21 was additionally caused by the *test reference*:
# its int64 packed keys are silently narrowed by GCU torch.topk — see
# compiler_issues/vllm_grouped_topk.md.)
# This rewrite is a pure 1-D row-streaming kernel: one program per token, all
# operands 1-D, no uint64 arithmetic, no cumsum/gather/cat, selection done
# with plain tl.max/tl.min reductions with leftmost tie-breaking, which
# matches the packed-key ordering (ties -> lowest index) of the reference.
#
# Perf note (2026-11-12): the one-token-per-program form is bound by the
# serial reduction chain of a single token (DS config: 56 dependent
# 256-lane trees ~= 4.5us per program at saturation) and programs do not
# overlap on one SP, so latency scales ~linearly with tokens / 48 SPs.
# Batching ROWS_PER_PROG tokens per program (all loads staged first, then
# the independent per-token chains interleaved phase-split: all m1 -> all
# i1 -> all m2, ...) keeps the per-token arithmetic order - and therefore
# the results - bitwise identical to the single-row kernel. On GCU400 the
# benefit is wave quantization, not ILP: at saturation per-program time is
# linear in U (no cross-chain overlap win), but near/below the 48-SP wave
# boundary collapsing ceil(tokens/U) programs into one wave wins up to
# 1.6x (64-512 tokens); U>8 spills registers and regresses everywhere.
# Thresholds in _DISPATCH_TABLE are probe+framework calibrated. Triton's
# frontend cannot hold lists of tensors, so the unrolled kernel body is
# source-generated (same code_cache infra as pointwise_dynamic) and cached
# under FLAGGEMS_CACHE_DIR/code_cache.

import hashlib
import importlib.util
import logging

import torch
import triton

from flaggems_vllm.utils.code_cache import code_cache_dir
from flaggems_vllm.utils.code_utils import write_atomic

logger = logging.getLogger(__name__)

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255

# Single 1-D kernel keeps one expert lane per element on chip; 1024 experts
# (BLOCK_EXPERT <= 1024) is a few tens of KB of on-chip buffers.
_MAX_EXPERTS = 1024

# Rows (tokens) per program for large token counts. Values are baked into
# generated kernels; thresholds come from the U-sweep probe on GCU400
# (see _pick_rows_per_prog).


def _row_grid(m):
    """Spread ``m`` logical tokens over a 3-D grid within GCU launch limits.

    The kernel recombines (pid_x, pid_y, pid_z) into a flat token id via the
    ``GRID_X``/``GRID_XY`` runtime arguments, so any decomposition with
    ``grid_x * grid_y * grid_z >= m`` is correct; ids beyond ``m`` are
    masked by the in-kernel ``pid < num_tokens`` guard.
    """
    if m <= _GRID_X_MAX:
        return (m, 1, 1)

    grid_y = triton.cdiv(m, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(m, grid_y), grid_y, 1)

    # m > 65535 * 255: spill into grid.z as well (supports up to
    # 65535 * 255 * 255 ~= 4.26e9 tokens).
    grid_z = triton.cdiv(m, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(m, _GRID_X_MAX * grid_z)
    return (triton.cdiv(m, grid_y * grid_z), grid_y, grid_z)


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    logger.debug("GEMS GROUPED TOPK")
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D Tensor")
    num_tokens, num_experts = scores.shape
    if num_experts % n_group != 0:
        raise ValueError("num_experts must be divisible by n_group")
    if n_group > 32:
        raise ValueError("n_group should be smaller than or equal to 32")
    if topk > 32:
        raise ValueError("topk should be smaller than or equal to 32 for now")
    if scoring_func not in (0, 1):
        raise ValueError("scoring_func must be 0 (none) or 1 (sigmoid)")
    if scores.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype: {scores.dtype}")
    if num_experts > _MAX_EXPERTS:
        raise NotImplementedError(
            f"grouped_topk on the enflame backend supports at most "
            f"{_MAX_EXPERTS} experts, got {num_experts}"
        )

    if bias.ndim != 1:
        bias = bias.flatten()
    if len(bias) != num_experts:
        raise ValueError(
            f"bias length ({len(bias)}) must match num_experts ({num_experts})"
        )

    num_experts_per_group = num_experts // n_group

    if not scores.is_contiguous():
        scores = scores.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    topk_values = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.float32,
    )
    topk_indices = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.int32,
    )
    if num_tokens == 0:
        return topk_values, topk_indices

    # Keep grid.x within the GCU 65535 limit (grid.y/z within 255) for
    # large token counts; the kernel recombines the axes into a flat id.
    rows_per_prog = _pick_rows_per_prog(num_tokens)
    grid = _row_grid(triton.cdiv(num_tokens, rows_per_prog))
    kernel = _get_rows_kernel(rows_per_prog)
    kernel[grid](
        scores,
        bias,
        topk_values,
        topk_indices,
        num_tokens,
        num_experts,
        num_experts_per_group,
        routed_scaling_factor,
        scores.stride(0),
        topk_values.stride(0),
        grid[0],
        grid[0] * grid[1],
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        BLOCK_GROUP=triton.next_power_of_2(n_group),
        BLOCK_EXPERT=triton.next_power_of_2(num_experts),
        TOPK_PAD=triton.next_power_of_2(topk),
        RENORMALIZE=int(renormalize),
        SCORING_FUNC=scoring_func,
        num_warps=1,
    )

    return topk_values, topk_indices


def _emit_rows_kernel_source(rows_per_prog):
    """Emit the source of the U-token-per-program grouped-topk kernel.

    ``U = rows_per_prog`` tokens are processed by one program with their
    (independent) reduction chains interleaved: inside every
    ``tl.static_range`` step the U copies are emitted phase-split (all
    first-tree ops, then all second-tree ops, ...), so a dependent link is
    separated from its use by ~(U-1) independent trees and the tree latency
    is hidden. Per-token op order matches the single-row kernel exactly.
    """
    u_count = rows_per_prog
    tokens = range(u_count)
    lines = []
    w = lines.append

    w("# Auto-generated by flaggems_vllm grouped_topk.py - do not edit.")
    w("import triton")
    w("import triton.language as tl")
    w("")
    w("from flaggems_vllm.utils import tl_extra_shim")
    w("from flaggems_vllm.utils import triton_lang_extension as ext")
    w("")
    w("")
    w("# Sentinel for the leftmost-argmax reduction (int32 lanes only).")
    w("_ABSENT = tl.constexpr(2147483647)")
    w("")
    w("")
    w("@triton.jit")
    w("def _sigmoid(x):")
    w("    log2e: tl.constexpr = 1.4426950408889634")
    w("    return 1 / (1 + tl_extra_shim.exp2(-x * log2e))")
    w("")
    w("")
    w("@triton.jit")
    w(f"def gcu_grouped_topk_kernel_rows{u_count}(")
    w("    scores_ptr,")
    w("    bias_ptr,")
    w("    topk_values_ptr,")
    w("    topk_indices_ptr,")
    w("    num_tokens,")
    w("    num_experts,")
    w("    num_experts_per_group,")
    w("    routed_scaling_factor,")
    w("    stride_scores_token,")
    w("    stride_out_token,")
    w("    GRID_X,")
    w("    GRID_XY,")
    w("    N_GROUP: tl.constexpr,")
    w("    TOPK_GROUP: tl.constexpr,")
    w("    TOPK: tl.constexpr,")
    w("    BLOCK_GROUP: tl.constexpr,")
    w("    BLOCK_EXPERT: tl.constexpr,")
    w("    TOPK_PAD: tl.constexpr,")
    w("    RENORMALIZE: tl.constexpr,")
    w("    SCORING_FUNC: tl.constexpr,")
    w("):")
    w("    # Recombine the 3-D launch grid into a flat program id (grid.x is")
    w("    # capped at 65535 on the GCU, grid.y/z at 255).")
    w("    pid = (")
    w("        ext.program_id(0)")
    w("        + ext.program_id(1) * GRID_X")
    w("        + ext.program_id(2) * GRID_XY")
    w("    )")
    w(f"    base_row = pid * {u_count}")
    w("    if base_row >= num_tokens:")
    w("        return")
    w("")
    w('    neg_inf = -float("inf")')
    w("")
    w("    expert_lane = tl.arange(0, BLOCK_EXPERT)")
    w("    valid_expert = expert_lane < num_experts")
    w("    bias_val = tl.load(")
    w("        bias_ptr + expert_lane, mask=valid_expert, other=0.0")
    w("    ).to(tl.float32)")
    w("")
    w("    # -- staged loads: bias once, then all U score rows ---------------")
    for u in tokens:
        w(f"    row{u} = base_row + {u}")
        w(f"    tok_ok{u} = row{u} < num_tokens")
        w(f"    lane_mask{u} = valid_expert & tok_ok{u}")
        w(
            f"    score{u} = tl.load("
            f"scores_ptr + row{u} * stride_scores_token + expert_lane, "
            f"mask=lane_mask{u}, other=neg_inf"
            f").to(tl.float32)"
        )
    w("")
    w("    # -- elementwise prep (invalid lanes stay at -inf) ----------------")
    w("    if SCORING_FUNC == 1:")
    for u in tokens:
        w(f"        base_score{u} = _sigmoid(score{u})")
    w("    else:")
    for u in tokens:
        w(f"        base_score{u} = score{u}")
    for u in tokens:
        w(f"    base_score{u} = tl.where(valid_expert, base_score{u}, neg_inf)")
        w(f"    biased{u} = tl.where(valid_expert, base_score{u} + bias_val, neg_inf)")
    _emit_step1(w, tokens)
    _emit_step2(w, tokens)
    _emit_step3(w, tokens)
    _emit_epilogue(w, tokens)
    w("")
    return "\n".join(lines)


def _emit_step1(w, tokens):
    """Per-group top-2 (biased) sum, phase-split across the U tokens."""
    w("")
    w("    # -- step 1: per-group top-2 (biased) sum ------------------------")
    w("    group_lane = tl.arange(0, BLOCK_GROUP)")
    for u in tokens:
        w(f"    group_scores{u} = tl.full(" f"(BLOCK_GROUP,), neg_inf, tl.float32)")
    w("    for g in tl.static_range(N_GROUP):")
    w("        g_lo = g * num_experts_per_group")
    w("        g_hi = g_lo + num_experts_per_group")
    w("        in_group = (expert_lane >= g_lo) & (expert_lane < g_hi)")
    for u in tokens:
        w(f"        gsel{u} = tl.where(in_group, biased{u}, neg_inf)")
    for u in tokens:
        w(f"        m1_{u} = tl.max(gsel{u}, axis=0)")
    for u in tokens:
        w(
            f"        i1_{u} = tl.min("
            f"tl.where(gsel{u} == m1_{u}, expert_lane, _ABSENT), axis=0)"
        )
    for u in tokens:
        w(f"        gsel{u} = tl.where(expert_lane == i1_{u}, neg_inf, gsel{u})")
    for u in tokens:
        w(f"        m2_{u} = tl.max(gsel{u}, axis=0)")
    for u in tokens:
        w(
            f"        group_scores{u} = tl.where("
            f"group_lane == g, m1_{u} + m2_{u}, group_scores{u})"
        )


def _emit_step2(w, tokens):
    """Pick the top ``TOPK_GROUP`` groups (ties -> lowest group id)."""
    w("")
    w("    # -- step 2: pick top TOPK_GROUP groups --------------------------")
    for u in tokens:
        w(f"    expert_selected{u} = expert_lane < 0")
        w(f"    group_cand{u} = group_scores{u}")
    w("    for _ in tl.static_range(TOPK_GROUP):")
    for u in tokens:
        w(f"        gmax{u} = tl.max(group_cand{u}, axis=0)")
    for u in tokens:
        w(
            f"        gi{u} = tl.min("
            f"tl.where(group_cand{u} == gmax{u}, group_lane, _ABSENT), axis=0)"
        )
    for u in tokens:
        w(f"        g_lo{u} = gi{u} * num_experts_per_group")
        w(f"        g_hi{u} = g_lo{u} + num_experts_per_group")
        w(f"        sel_group{u} = (expert_lane >= g_lo{u}) & (expert_lane < g_hi{u})")
        w(
            f"        expert_selected{u} = expert_selected{u} "
            f"| (sel_group{u} & (gmax{u} > neg_inf))"
        )
        w(
            f"        group_cand{u} = tl.where("
            f"group_lane == gi{u}, neg_inf, group_cand{u})"
        )


def _emit_step3(w, tokens):
    """Top ``TOPK`` experts among selected groups (ties -> lowest id)."""
    w("")
    w("    # -- step 3: top TOPK experts among selected groups --------------")
    for u in tokens:
        w(f"    cand{u} = tl.where(expert_selected{u}, biased{u}, neg_inf)")
        w(f"    proceed{u} = tl.max(group_scores{u}, axis=0) > neg_inf")
    w("    pos = tl.arange(0, TOPK_PAD)")
    for u in tokens:
        w(f"    out_idx{u} = tl.full((TOPK_PAD,), 0, tl.int32)")
        w(f"    out_w{u} = tl.full((TOPK_PAD,), 0.0, tl.float32)")
        w(f"    weight_sum{u} = 0.0")
    w("    default_idx = pos.to(tl.int32)")
    w("    default_w = tl.full((TOPK_PAD,), 1.0 / TOPK, tl.float32)")
    w("    for i in tl.static_range(TOPK):")
    for u in tokens:
        w(f"        mval{u} = tl.max(cand{u}, axis=0)")
    for u in tokens:
        w(
            f"        midx{u} = tl.min("
            f"tl.where(cand{u} == mval{u}, expert_lane, _ABSENT), axis=0)"
        )
    for u in tokens:
        w(
            f"        w{u} = tl.sum("
            f"tl.where(expert_lane == midx{u}, base_score{u}, 0.0), axis=0)"
        )
    for u in tokens:
        w(f"        out_idx{u} = tl.where(pos == i, midx{u}.to(tl.int32), out_idx{u})")
        w(f"        out_w{u} = tl.where(pos == i, w{u}, out_w{u})")
        w(f"        weight_sum{u} += w{u}")
        w(f"        cand{u} = tl.where(expert_lane == midx{u}, neg_inf, cand{u})")


def _emit_epilogue(w, tokens):
    """Renormalize over the unbiased topk weights and store."""
    w("")
    w("    # -- step 4: renormalize + identity fallback + stores ------------")
    w("    if RENORMALIZE:")
    for u in tokens:
        w(
            f"        scale{u} = routed_scaling_factor.to(tl.float32) "
            f"/ (weight_sum{u} + 1e-20)"
        )
    w("    else:")
    for u in tokens:
        w(f"        scale{u} = routed_scaling_factor.to(tl.float32)")
    for u in tokens:
        w(f"    out_w{u} = out_w{u} * scale{u}")
    w("    # vLLM semantics: with no finite group score, fall back to")
    w("    # identity routing (slot i -> expert i, uniform 1/topk weights).")
    for u in tokens:
        w(f"    final_w{u} = tl.where(proceed{u}, out_w{u}, default_w)")
        w(f"    final_idx{u} = tl.where(proceed{u}, out_idx{u}, default_idx)")
    for u in tokens:
        w(
            f"    tl.store("
            f"topk_values_ptr + row{u} * stride_out_token + pos, "
            f"final_w{u}, mask=(pos < TOPK) & tok_ok{u})"
        )
        w(
            f"    tl.store("
            f"topk_indices_ptr + row{u} * stride_out_token + pos, "
            f"final_idx{u}, mask=(pos < TOPK) & tok_ok{u})"
        )


_ROWS_KERNEL_CACHE = {}


def _get_rows_kernel(rows_per_prog):
    """Build (or fetch) the generated kernel for ``rows_per_prog`` tokens."""
    kernel = _ROWS_KERNEL_CACHE.get(rows_per_prog)
    if kernel is not None:
        return kernel

    source = _emit_rows_kernel_source(rows_per_prog)
    tag = hashlib.sha256(source.encode()).hexdigest()[:12]
    file_path = code_cache_dir() / f"grouped_topk_rows{rows_per_prog}_{tag}.py"
    write_atomic(str(file_path), source, make_dirs=True)
    module_name = f"_gen_grouped_topk_rows{rows_per_prog}_{tag}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, f"gcu_grouped_topk_kernel_rows{rows_per_prog}")
    _ROWS_KERNEL_CACHE[rows_per_prog] = kernel
    return kernel


# Rows (tokens) per program, calibrated on GCU400 (E=256, DS ng=8/tkg=4/k=8
# and GLM ng=1/tkg=1/k=8) by the 2026-11-12 U-sweep probe (steady cells,
# do_bench median) and confirmed/corrected per shape by the framework bench
# (base1 vs new1/new2 fresh runs). At scale the kernel is throughput-bound
# (per-program time is linear in U: T_prog(U) ~= U * 4.5us + launch floor),
# so batching only pays by collapsing program waves near/below the 48-SP
# wave boundary - and the biggest shapes only gain ~1.5% (U8 vs U1).
# Framework-verified cells: 64->U2 1.45-1.59x, 496/512->U4 1.18-1.24x,
# 193->U2 ~1.0x, 16384->U8 1.5%; probe-steady cells: 128->U4 19.4us
# (U2 regresses to 22.9us in the framework bench), 2048->U4.
_DISPATCH_TABLE = (
    (48, 1),
    (96, 2),
    (192, 4),
    (256, 2),
    (2048, 4),
)


def _pick_rows_per_prog(num_tokens):
    """Rows per program by token count (probe-calibrated on GCU400).

    Below one full wave (48 SPs) single-row programs already run in parallel
    and only pay the launch floor; above it, batching U rows per program
    removes whole serialization waves of the per-token reduction chain.
    """
    for threshold, rows in _DISPATCH_TABLE:
        if num_tokens <= threshold:
            return rows
    return 8


# ---------------------------------------------------------------------------
# vLLM registers `grouped_topk` in the `_moe_C` CUDA extension namespace
# (vllm._custom_ops.grouped_topk -> torch.ops._moe_C.grouped_topk). The GCU
# container has no libcuda, so the extension never loads and the qualified op
# is missing: vLLM's grouped-topk router fails to resolve it, and the
# accuracy tests fall back to a pure-torch packing reference that corrupts
# its int64 packed keys on the GCU (silent Long->Int narrowing in torch.topk).
# Register this backend's Triton implementation under the same qualified
# name so both the production routing path and the test reference resolve
# on GCU. Hosts where the native CUDA op is already registered (real CUDA
# boxes) skip the override.
# ---------------------------------------------------------------------------
if not hasattr(torch.ops._moe_C, "grouped_topk"):
    _moe_c_lib = torch.library.Library("_moe_C", "FRAGMENT")
    _moe_c_lib.define(
        "grouped_topk(Tensor scores, int num_expert_group, int topk_group, "
        "int topk, bool renormalize, float routed_scaling_factor, Tensor bias, "
        "int scoring_func=0) -> (Tensor topk_weights, Tensor topk_ids)"
    )
    _moe_c_lib.impl("grouped_topk", grouped_topk)
