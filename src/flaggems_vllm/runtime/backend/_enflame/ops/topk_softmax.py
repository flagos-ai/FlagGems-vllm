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

"""TopK-gating softmax for MoE routers, GCU (enflame) adaptation.

Semantics aligned with vLLM 0.20.0 ``csrc/moe/topk_softmax_kernels.cu``:
softmax over each gating row (fp32 compute), iterative top-k selection over
``softmax_prob (+ optional e_score_correction_bias)`` with ties resolving to
the lowest expert index, output weights are the *unbiased* softmax
probabilities, optional renormalization divides the k weights by their sum
(``> 0 ? sum : 1``), and ``token_expert_indices[r, k] = k * num_rows + r``
as int32.

GCU-specific notes:

- torch_gcu int64 layout is env-dependent (compiler issue
  ``triton_i64_narrow_load``): without ``TORCH_GCU_ENABLE_INT64_AND_UINT64``
  int64 tensors are compact int32 words and a Triton store through the
  int64-typed tensor writes 8-byte elements and corrupts the buffer, so int64
  ``topk_indices`` are written through a zero-copy int32 word view (expert
  ids are < 2^31, i32 safe domain); with the env set (service default since
  2026-11-11) int64 is a true 8-byte layout and the kernel stores i64
  directly. Dispatch mirrors ``topk_sigmoid``.
- **2026-11-12 perf rewrite (round 2)**: the 2026-11-07 fix used one program
  per token row, so every row paid ``K`` dependent ``tl.max`` reductions
  (~1 us each; measured 2026-11-12 probe: 2048x64x8 = 943 us) and the grid
  paid ~50 ns/CTA dispatch. The toolchain ``memref_ext.slice_*`` regression
  that blocked 2D tiles on 2026-11-07 has since been rolled back (backend
  .so mtime 2026-11-06), so the main path is now a **2D row-batched
  persistent kernel**: ``BLOCK_R`` token rows per tile amortize each argmax
  reduction, a fixed 48-program grid-stride loop removes the dispatch tax,
  and ``BLOCK_E`` is padded to >= 128 because width-64 reductions are
  pathologically slow on this backend (2048x64x8: 943 us @ BE=64 vs 380 us
  @ BE=128, 1D nw=1). Probe (2048-row, do_bench w150/r100, direction only):
  E64 943->163, E128 384->163, E256-K8 389->165, E256-K32 1266->662 us.
  Rows with ``num_experts > 512`` (no router in practice) keep the original
  1D row-streaming kernel as fallback: the (BLOCK_R, 1024+) fp32 tile would
  risk the per-program DSM budget.
"""

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _gcu_int64_enabled():
    """True when torch_gcu uses the true 8-byte int64 physical layout.

    Without ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` torch_gcu stores int64
    tensors as compact int32 words (compiler issue triton_i64_narrow_load);
    with it (service default since 2026-11-11) int64 is a true 8-byte layout
    and direct i64 stores address correctly. Mirrors ``topk_sigmoid``.
    """
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "").lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255

# One persistent program per SP (48 on gcu410): grid-stride over row tiles.
# More programs just queue in waves (measured -3~-5% vs exact-tile grids in
# the 2026-11-12 probe; mega_moe P_cap=48 precedent).
_NUM_PROGS = 48

# Width-64 reductions are ~2.5x slower than width-128 on this backend, so
# small expert counts are padded up (probe: 1D E64 nw4 943 us vs 1D E64
# padded to BE=128 nw1 380 us).
_MIN_BLOCK_E = 128

# Largest expert axis served by the 2D tile; above this the (BLOCK_R, BLOCK_E)
# fp32 tile risks the per-program DSM budget and the 1D kernel takes over.
_MAX_2D_BLOCK_E = 512


def _row_grid(num_rows):
    """Spread ``num_rows`` logical rows over a 3-D grid within GCU limits."""
    if num_rows <= _GRID_X_MAX:
        return (num_rows, 1, 1)

    grid_y = triton.cdiv(num_rows, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(num_rows, grid_y), grid_y, 1)

    grid_z = triton.cdiv(num_rows, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(num_rows, _GRID_X_MAX * grid_z)
    return (triton.cdiv(num_rows, grid_y * grid_z), grid_y, grid_z)


def _int32_slot_view(tensor):
    """Zero-copy int32 view of a GCU int64 tensor's 4-byte physical slots."""
    words = torch.empty(0, dtype=torch.int32, device=tensor.device)
    return words.set_(
        tensor.untyped_storage(),
        int(tensor.storage_offset()),
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
    )


@triton.jit
def topk_gating_softmax_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    indices_ptr,
    source_rows_ptr,
    num_rows,
    num_experts,
    K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    INDEX_TY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_E: tl.constexpr,
    MASK_COLS: tl.constexpr,
):
    """2D row-batched persistent top-k softmax.

    Each program grid-strides over row tiles of ``BLOCK_R`` token rows; all
    per-row work (softmax + iterative argmax selection) is expressed on a
    ``(BLOCK_R, BLOCK_E)`` tile with axis-1 reductions, so one reduction
    serves ``BLOCK_R`` rows instead of one. Stores stay per-slot ``[BLOCK_R]``
    vectors (``[BLOCK_R, K]`` accumulators measured slower + DSM-risky).
    """
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(num_rows, BLOCK_R)
    # NOTE: tl.range(..., num_stages=2) to pipeline the tile loop is rejected
    # by the enflame frontend (CompilationError at the range expr, 2026-11-12);
    # the loop runs unpipelined.
    for tile in tl.range(pid, n_tiles, tl.num_programs(0)):
        rows = tile * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < num_rows
        cols = tl.arange(0, BLOCK_E)
        if MASK_COLS:
            cmask = cols < num_experts
            lmask = rmask[:, None] & cmask[None, :]
            logits = tl.load(
                input_ptr + rows[:, None] * num_experts + cols[None, :],
                mask=lmask,
                other=-float("inf"),
            ).to(tl.float32)
        else:
            lmask = rmask[:, None]
            logits = tl.load(
                input_ptr + rows[:, None] * num_experts + cols[None, :],
                mask=lmask,
            ).to(tl.float32)

        row_max = tl.max(logits, axis=1)
        exp_vals = tl.exp(logits - row_max[:, None])
        probs = exp_vals / tl.sum(exp_vals, axis=1)[:, None]

        # Selection scores: softmax probs, optionally biased. Weights stored
        # below are always the *unbiased* probs (vLLM CUDA semantics); the
        # bias only steers which experts are selected. Padding columns stay
        # at -inf so they can never win, even with a very negative bias on
        # real columns.
        if HAS_BIAS:
            if MASK_COLS:
                bias = tl.load(bias_ptr + cols, mask=cols < num_experts, other=0.0).to(
                    tl.float32
                )
            else:
                bias = tl.load(bias_ptr + cols).to(tl.float32)
            choice = probs + bias[None, :]
        else:
            choice = probs
        choice = tl.where(lmask, choice, -float("inf"))

        selected_sum = tl.zeros((BLOCK_R,), dtype=tl.float32)
        for ki in tl.static_range(K):
            # tl.max ties resolve to the lowest column, matching the CUDA
            # gatherTopK / cub::ArgMax fill rule.
            curr_max, curr_arg = tl.max(choice, axis=1, return_indices=True)
            if HAS_BIAS:
                weight = tl.sum(
                    tl.where(cols[None, :] == curr_arg[:, None], probs, 0.0),
                    axis=1,
                )
            else:
                weight = curr_max
            tl.store(output_ptr + rows * K + ki, weight, mask=rmask)
            tl.store(indices_ptr + rows * K + ki, curr_arg.to(INDEX_TY), mask=rmask)
            tl.store(
                source_rows_ptr + rows * K + ki,
                (ki * num_rows + rows).to(tl.int32),
                mask=rmask,
            )
            if RENORMALIZE:
                selected_sum += weight
            choice = tl.where(cols[None, :] == curr_arg[:, None], -float("inf"), choice)

        if RENORMALIZE:
            denom = tl.where(selected_sum > 0.0, selected_sum, 1.0)
            for ki in tl.static_range(K):
                idx = rows * K + ki
                val = tl.load(output_ptr + idx, mask=rmask)
                tl.store(output_ptr + idx, val / denom, mask=rmask)


@triton.jit
def _topk_gating_softmax_row_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    indices_ptr,
    source_rows_ptr,
    num_rows,
    num_experts,
    K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    INDEX_TY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """1D row-streaming fallback (num_experts > 512): one program per row."""
    grid_x = tl.num_programs(0)
    row = (
        tl.program_id(0)
        + tl.program_id(1) * grid_x
        + tl.program_id(2) * (grid_x * tl.num_programs(1))
    )
    if row >= num_rows:
        return

    cols = tl.arange(0, BLOCK_E)
    valid_cols = cols < num_experts
    logits = tl.load(
        input_ptr + row * num_experts + cols, mask=valid_cols, other=-float("inf")
    ).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    exp_vals = tl.exp(logits - row_max)
    probs = exp_vals / tl.sum(exp_vals, axis=0)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + cols, mask=valid_cols, other=0.0).to(tl.float32)
        choice = probs + bias
    else:
        choice = probs
    choice = tl.where(valid_cols, choice, -float("inf"))

    selected_sum = 0.0
    for ki in tl.static_range(K):
        curr_max, curr_arg = tl.max(choice, axis=0, return_indices=True)
        if HAS_BIAS:
            weight = tl.sum(tl.where(cols == curr_arg, probs, 0.0), axis=0)
        else:
            weight = curr_max
        tl.store(output_ptr + row * K + ki, weight)
        tl.store(indices_ptr + row * K + ki, curr_arg.to(INDEX_TY))
        tl.store(source_rows_ptr + row * K + ki, (ki * num_rows + row).to(tl.int32))
        if RENORMALIZE:
            selected_sum += weight
        choice = tl.where(cols == curr_arg, -float("inf"), choice)

    if RENORMALIZE:
        denom = tl.where(selected_sum > 0.0, selected_sum, 1.0)
        for ki in tl.static_range(K):
            idx = row * K + ki
            val = tl.load(output_ptr + idx)
            tl.store(output_ptr + idx, val / denom)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    bias: torch.Tensor | None = None,
) -> None:
    """Fused softmax + top-k gating kernel for MoE routers.

    Interface aligned with the vLLM CUDA operator ``_moe_C::topk_softmax``
    (vllm 0.20.0)::

        topk_softmax(Tensor! topk_weights, Tensor! topk_indices,
                     Tensor! token_expert_indices, Tensor gating_output,
                     bool renormalize, Tensor? bias) -> ()

    Args:
        topk_weights: Output [num_tokens, topk], float32.
        topk_indices: Output [num_tokens, topk], int32/uint32/int64.
        token_expert_indices: Output [num_tokens, topk], int32.
        gating_output: Logits [num_tokens, num_experts], fp32/fp16/bf16.
        renormalize: Divide the k selected weights by their sum.
        bias: Optional float32 [num_experts] e_score_correction_bias; only
            affects expert selection, never the stored weights.
    """
    logger.debug("GEMS TOPK SOFTMAX")
    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.size(-1)
    assert 0 < topk <= min(32, num_experts)

    assert gating_output.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert topk_weights.dtype == torch.float32
    assert token_expert_indices.dtype == torch.int32
    assert gating_output.is_contiguous()
    assert topk_weights.is_contiguous()
    assert token_expert_indices.is_contiguous()
    assert topk_indices.is_contiguous()

    index_dtype = topk_indices.dtype
    if index_dtype == torch.int64:
        if _gcu_int64_enabled():
            # True 8-byte int64 layout: the kernel's i64 stores (INDEX_TY =
            # tl.int64) address elements correctly.
            kernel_indices = topk_indices
            index_ty = tl.int64
        else:
            # Compact int32-word physical layout: a Triton store through the
            # int64-typed tensor writes 8-byte elements and corrupts the
            # buffer (compiler_issues/triton_i64_narrow_load.md). Write
            # through a zero-copy int32 word view instead (expert ids are
            # < 2^31, i32 safe domain).
            kernel_indices = _int32_slot_view(topk_indices)
            index_ty = tl.int32
    elif index_dtype == torch.int32:
        kernel_indices = topk_indices
        index_ty = tl.int32
    elif index_dtype == torch.uint32:
        kernel_indices = topk_indices
        index_ty = tl.uint32
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    if bias is not None:
        assert bias.dtype == torch.float32
        assert bias.dim() == 1
        assert bias.size(0) == num_experts
        assert bias.is_contiguous()
        assert bias.device == gating_output.device

    if num_tokens == 0:
        return

    block_e = ((triton.next_power_of_2(num_experts) + 31) // 32) * 32
    if block_e <= _MAX_2D_BLOCK_E:
        # 2D row-batched persistent path (probe 2026-11-12: 1.9-5.8x over the
        # 1D row kernel on the bench shape family; correctness bit-stable on
        # indices/source_rows, weights within fp LSB of the 1D kernel).
        block_e = max(block_e, _MIN_BLOCK_E)
        block_r = 8 if topk <= 16 else 4
        n_tiles = triton.cdiv(num_tokens, block_r)
        grid = (min(_NUM_PROGS, n_tiles),)
        # The bias path (axis-1 reductions -> [R,1] broadcast divide -> 1D
        # [BLOCK_E] broadcast add) miscompiles silently at num_warps >= 2
        # with BLOCK_E >= 256 on this backend (maxdiff ~2.2 on ~39% of
        # elements; verified matrix 2026-11-12: nw=1 or BLOCK_E=128 are
        # LSB-correct). Bench shapes carry no bias, so pin the bias arm to
        # nw=1 for correctness; see compiler_issues/ for the filed bug.
        # nw=8 for BLOCK_E >= 256 measured WITHIN-NOISE vs nw=4 (framework
        # bench 2026-11-12, total +0.0%); keep nw=4.
        launch_warps = 1 if bias is not None else 4
        topk_gating_softmax_kernel[grid](
            gating_output,
            bias if bias is not None else gating_output,
            topk_weights,
            kernel_indices,
            token_expert_indices,
            num_tokens,
            num_experts,
            K=topk,
            RENORMALIZE=renormalize,
            INDEX_TY=index_ty,
            HAS_BIAS=bias is not None,
            BLOCK_R=block_r,
            BLOCK_E=block_e,
            MASK_COLS=(block_e != num_experts),
            num_warps=launch_warps,
            num_stages=1,
        )
    else:
        # Wide-expert fallback: 1D row streaming within GCU grid limits.
        num_warps = 4 if block_e <= 64 else 1
        grid = _row_grid(num_tokens)
        _topk_gating_softmax_row_kernel[grid](
            gating_output,
            bias if bias is not None else gating_output,
            topk_weights,
            kernel_indices,
            token_expert_indices,
            num_tokens,
            num_experts,
            K=topk,
            RENORMALIZE=renormalize,
            INDEX_TY=index_ty,
            HAS_BIAS=bias is not None,
            BLOCK_E=block_e,
            num_warps=num_warps,
            num_stages=1,
        )


# ---------------------------------------------------------------------------
# vLLM torch.ops registration.
#
# On the GCU there is no CUDA driver, so vLLM's `_moe_C` extension never loads
# and `torch.ops._moe_C.topk_softmax` stays unregistered: vLLM's
# `topk_softmax` wrapper (used by every fused-topk router path and by this
# repo's vs-vLLM tests/benchmarks) then fails with `AttributeError:
# '_OpNamespace' '_moe_C' object has no attribute 'topk_softmax'`. Register
# this backend's Triton implementation under the same qualified name so the
# wrapper resolves on GCU. Hosts where the native CUDA op is already
# registered (real CUDA boxes) skip the override.
# ---------------------------------------------------------------------------
if not hasattr(torch.ops._moe_C, "topk_softmax"):
    _moe_c_lib = torch.library.Library("_moe_C", "FRAGMENT")
    _moe_c_lib.define(
        "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
        "token_expert_indices, Tensor gating_output, bool renormalize, Tensor? "
        "bias) -> ()"
    )
    _moe_c_lib.impl("topk_softmax", topk_softmax)


def _install():
    """Wire the GCU implementation into the ``flaggems_vllm.ops`` entrypoints.

    ``benchmark/test_topk_softmax.py`` (and any direct importer) resolves
    ``flaggems_vllm.ops.topk_softmax``, which the SpecOpRegistrar top-level
    namespace swap alone cannot reach. Replace the attribute on the already
    imported modules in place, mirroring the kpool / kunlunxin mhc_pre
    precedents: only a pristine generic function is swapped, so explicit user
    or test patches always win.
    """
    import sys

    impl = topk_softmax
    pkg = sys.modules.get("flaggems_vllm.ops")
    if pkg is not None:
        current = getattr(pkg, "topk_softmax", None)
        if current is not None and getattr(current, "__module__", None) == (
            "flaggems_vllm.ops.topk_softmax"
        ):
            pkg.topk_softmax = impl
    sub = sys.modules.get("flaggems_vllm.ops.topk_softmax")
    if sub is not None:
        current = getattr(sub, "topk_softmax", None)
        if current is not None and getattr(current, "__module__", None) == (
            "flaggems_vllm.ops.topk_softmax"
        ):
            sub.topk_softmax = impl


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
