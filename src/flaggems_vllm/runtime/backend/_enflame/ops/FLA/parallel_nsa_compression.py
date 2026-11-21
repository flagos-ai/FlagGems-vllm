# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import re

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_token_indices,
)
from flaggems_vllm.ops.FLA.triton_ops_helper import autotune_cache_kwargs, exp, log
from flaggems_vllm.ops.FLA.utils import check_shared_mem, input_guard

# Enflame GCU hardware grid limits: grid.x <= 65535 (see
# compiler_problems/P19_grid_hard_limits). The T dimension of the
# forward/dq grids is launched in <=65535-token chunks via a T_OFF offset,
# mirroring the adaptation in parallel_nsa.py.
_GCU_GRID_X_MAX = 65535


# Enflame GCU: torch_gcu narrows int64 (Long) storage to 32-bit while the
# tensor still reports torch.int64, so Triton kernels would read the buffer
# with 8-byte strides over 4-byte elements (garbage indices). Normalize every
# integer index tensor to int32 before it reaches a kernel launch (same
# adaptation as parallel_nsa.py).
def _ensure_i32(t):
    if torch.is_tensor(t) and t.dtype == torch.int64:
        return t.to(torch.int32)
    return t


# Backward preprocessing (local copy): the generic
# parallel_attn_bwd_kernel_preprocess launches one CTA per flattened row
# (grid=(numel,)), which overflows the GCU grid.x limit of 65535 for
# B*T*HQ > 65535 (e.g. 1*65536*256). This variant decomposes the flat row id
# over a capped 3D grid with an in-kernel bound guard; the math is identical
# to the generic kernel (same as _nsa_attn_bwd_preprocess_kernel).
@triton.jit(do_not_specialize=["N", "GRID_X", "GRID_XY"])
def _nsa_cmp_attn_bwd_preprocess_kernel(
    o,
    do,
    delta,
    N,
    GRID_X,
    GRID_XY,
    B: tl.constexpr,
    V: tl.constexpr,
):
    i_n = tl.program_id(0) + tl.program_id(1) * GRID_X + tl.program_id(2) * GRID_XY
    if i_n < N:
        o_d = tl.arange(0, B)
        m_d = o_d < V
        b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
        b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
        b_delta = tl.sum(b_o * b_do)
        tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def _nsa_cmp_parallel_attn_bwd_preprocess(o, do):
    """Compute delta = sum(o * do, -1) with a GCU grid.x-safe launch."""
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    N = delta.numel()
    grid_x = min(N, _GCU_GRID_X_MAX)
    remainder = triton.cdiv(N, grid_x)
    grid_y = min(remainder, 255)
    grid_z = triton.cdiv(remainder, grid_y)
    _nsa_cmp_attn_bwd_preprocess_kernel[(grid_x, grid_y, grid_z)](
        o=o,
        do=do,
        delta=delta,
        N=N,
        GRID_X=grid_x,
        GRID_XY=grid_x * grid_y,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


# ===========================================================================
# GCU compiler-toolchain dialect compatibility shim
# ===========================================================================
# The installed FlagTree wheel emits MLIR where several `memref_ext.*` /
# `gcu.matmul` ops carry a trailing zero operand segment, while the external
# `gcu-compiler-opt` tool shipped in /opt/triton_gcu (triton-gcu 1.0.20261020)
# still registers the older operand layouts. The mismatch makes every kernel
# with 2-D tensor accesses (block pointers, masked 2-D loads, tl.dot tiles)
# fail to compile with "operandSegmentSizes ... size mismatch". The shim below
# rewrites the generated MLIR text to the layout the external tool accepts:
#   * `memref_ext.memset_start` must carry no properties at all;
#   * other `memref_ext.*` / `gcu.matmul` ops drop their trailing zero
#     segments (pure metadata, no operand rebinding);
#   * `memref.reinterpret_cast` / `memref.subview` need explicit segment
#     sizes derived from their INT64_MIN sentinel counts when missing.
# It is idempotent and leaves already-compatible IR untouched.

_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if op_name in ("memref.reinterpret_cast", "memref.subview") and not seg_match:
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = len([x for x in opers.group(1).split(",") if x.strip()]) if opers else 1
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(
                _fix_gcu_dialect_line(line) for line in content.split("\n")
            )
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


# ===========================================================================
# Forward kernel
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BC": bc}, num_warps=num_warps)
        for bc in [64, 32, 16]
        for num_warps in [1, 2, 4]
    ],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T_OFF"])
def parallel_nsa_compression_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    T_OFF,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t = i_t + T_OFF
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)
    # max scores for the current block
    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    # lse = log(acc) + m
    b_acc = tl.zeros([G], dtype=tl.float32)

    for i_c in range(0, NC, BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (boc * H + i_h) * V,
            (TC, V),
            (H * V, 1),
            (i_c, i_v * BV),
            (BC, BV),
            (1, 0),
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

        # [G]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp(b_mp - b_m)
        # [G, BC]
        b_p = exp(b_s - b_m[:, None])
        # [G]
        b_acc = tl.math.fma(b_acc, b_r, tl.sum(b_p, 1))

        # [G, BV]
        b_o = tl.math.fma(b_o, b_r[:, None], tl.dot(b_p.to(b_q.dtype), b_v))

        b_mp = b_m
    if NC == 0:
        b_lse = tl.zeros([G], dtype=tl.float32)
    else:
        b_o = b_o / b_acc[:, None]
        b_lse = b_m + log(b_acc)

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(
            lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G),
            b_lse.to(lse.dtype.element_ty),
        )


# ===========================================================================
# Forward kernel (token-tiled, dense path)
# ===========================================================================
# The per-token kernel above launches one program per token; every program
# re-streams the whole compressed k/v prefix, so k/v traffic is O(T^2/BS).
# This dense-path variant processes BT consecutive tokens per program:
# all BT*G query rows share one k/v tile per iteration (loaded once), cutting
# k/v traffic by ~BT. The math (online softmax, masked (t+1)//BS blocks)
# is identical to the per-token kernel; rows whose token has no complete
# block (NC == 0) are zeroed through the same NC == 0 epilogue branch.
# The k/v loop is kept mask-only (no scf.if around loop-carried tiles) and
# always runs >= 1 iteration (max(NC_MAX, 1)) to avoid the GCU zero-trip
# dynamic-loop accumulator clobber (see the bwd dq guard below).


def _prune_tiled_fwd_configs(configs, nargs, **kwargs):
    merged = {**nargs, **kwargs}
    G = merged.get("G", 1)
    BK = merged.get("BK", 64)
    BV = merged.get("BV", 64)
    # Conservative 2D DSM peak per program: q[R,BK] + k[BK,BC] + v[BC,BV]
    # + s[R,BC]fp32 + p[R,BC]bf16 + dot_pv[R,BV]fp32 + o[R,BV]fp32 (worst-case
    # lifecycle overlap sum). > 448KB locks 1 program/SP on Libra-class parts,
    # so cap the pool at the 2-program budget and keep the smallest fallback.
    budget = 448 * 1024

    def dsm(cfg):
        bt = cfg.kwargs.get("BT", 8)
        bc = cfg.kwargs.get("BC", 64)
        R = bt * G
        return (
            R * BK * 2
            + BK * bc * 2
            + bc * BV * 2
            + R * bc * 4
            + R * bc * 2
            + R * BV * 4
            + R * BV * 4
        )

    pruned = [cfg for cfg in configs if dsm(cfg) <= budget]
    return pruned if pruned else [configs[0]]


@triton.autotune(
    configs=[
        triton.Config({"BT": bt, "BC": bc}, num_warps=num_warps)
        for bt in [8, 16, 32]
        for bc in [64, 128]
        for num_warps in [2, 4]
    ],
    prune_configs_by={"early_config_prune": _prune_tiled_fwd_configs},
    key=["G", "T", "BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_compression_fwd_tiled_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_tb, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T

    R: tl.constexpr = BT * G
    r = tl.arange(0, R)
    t = i_tb * BT + r // G
    g = r % G
    row_valid = t < T
    # number of complete compression blocks visible to each row's token
    NC_row = (t + 1) // BS
    # loop bound: the largest NC among valid tokens of this tile
    t_hi = tl.minimum(i_tb * BT + BT, T)
    NC_max = t_hi // BS

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    # [R, BK] — each token's G query rows are contiguous in memory
    q_ptrs = (
        q + (bos + t)[:, None] * (HQ * K) + (i_h * G + g)[:, None] * K + o_k[None, :]
    )
    b_q = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0)
    b_q = (b_q * scale).to(b_q.dtype)

    b_o = tl.zeros([R, BV], dtype=tl.float32)
    b_m = tl.full([R], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([R], dtype=tl.float32)

    TC = tl.cdiv(T, BS)
    k_base = k + (i_b * TC * H + i_h) * K
    v_base = v + (i_b * TC * H + i_h) * V
    # always at least one iteration: avoids the zero-trip clobber while
    # keeping the loop mask-only; empty tiles fully mask their rows.
    for i_c in range(0, tl.maximum(NC_max, 1), BC):
        o_c = i_c + tl.arange(0, BC)
        p_k = tl.make_block_ptr(k_base, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1))
        p_v = tl.make_block_ptr(
            v_base, (TC, V), (H * V, 1), (i_c, i_v * BV), (BC, BV), (1, 0)
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [R, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_c[None, :] < NC_row[:, None], b_s, float("-inf"))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = exp(b_m - b_m_new)
        b_p = exp(b_s - b_m_new[:, None])
        b_acc = tl.math.fma(b_acc, b_r, tl.sum(b_p, 1))
        b_o = tl.math.fma(b_o, b_r[:, None], tl.dot(b_p.to(b_q.dtype), b_v))
        b_m = b_m_new

    b_acc = tl.where(NC_row == 0, 1.0, b_acc)
    # rows whose token has no complete block (NC == 0) accumulated NaNs from
    # the all-masked softmax path; zero them exactly like the per-token kernel
    b_o = tl.where(NC_row[:, None] == 0, 0.0, b_o / b_acc[:, None])
    b_lse = tl.where(NC_row == 0, 0.0, b_m + log(b_acc))

    o_ptrs = (
        o + (bos + t)[:, None] * (HQ * V) + (i_h * G + g)[:, None] * V + o_v[None, :]
    )
    tl.store(o_ptrs, b_o.to(o.dtype.element_ty), mask=row_valid[:, None])
    if i_v == 0:
        lse_ptrs = lse + (bos + t) * HQ + i_h * G + g
        tl.store(lse_ptrs, b_lse.to(lse.dtype.element_ty), mask=row_valid)


# ===========================================================================
# Backward kernel: dQ
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "T_OFF"])
def parallel_nsa_compression_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    T_OFF,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t = i_t + T_OFF
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    # GCU: a zero-trip dynamic loop may leave the pre-loop accumulator registers
    # clobbered (non-finite dq for tokens with NC == 0), so guard the loop with a
    # positive-trip condition and store the zeros-initialized accumulator directly.
    if NC > 0:
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)
            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
            )
            p_v = tl.make_block_ptr(
                v + (boc * H + i_h) * V,
                (V, TC),
                (1, H * V),
                (i_v * BV, i_c),
                (BV, BC),
                (0, 1),
            )
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BC]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((o_c < NC)[None, :], b_p, 0)

            # [G, BV] @ [BV, BC] -> [G, BC]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BC] @ [BC, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Backward kernel: dK / dV
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "TC"])
def parallel_nsa_compression_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    scale,
    T,
    TC,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * TC

    if IS_VARLEN:
        i_n, i_c = tl.load(chunk_indices + i_c * 2).to(tl.int32), tl.load(
            chunk_indices + i_c * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        # the number of compression representations in total
        TC = tl.cdiv(T, BS)
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_k = tl.make_block_ptr(
        k + (boc * H + i_h) * K, (TC, K), (H * K, 1), (i_c * BC, 0), (BC, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (boc * H + i_h) * V,
        (TC, V),
        (H * V, 1),
        (i_c * BC, i_v * BV),
        (BC, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * all * H + boc * H + i_h) * K,
        (TC, K),
        (H * K, 1),
        (i_c * BC, 0),
        (BC, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (boc * H + i_h) * V,
        (TC, V),
        (H * V, 1),
        (i_c * BC, i_v * BV),
        (BC, BV),
        (1, 0),
    )

    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    # [BC, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)

    for i in range(i_c * BC * BS, T):
        o_c = i_c * BC + tl.arange(0, BC)

        p_q = tl.make_block_ptr(
            q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_do = tl.make_block_ptr(
            do + (bos + i) * HQ * V,
            (HQ, V),
            (V, 1),
            (i_h * G, i_v * BV),
            (G, BV),
            (1, 0),
        )
        p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [G]
        b_lse = tl.load(p_lse)
        b_delta = tl.load(p_delta)
        # [BC, G]
        b_s = tl.dot(b_k, tl.trans(b_q))
        b_p = exp(b_s - b_lse[None, :])
        b_p = tl.where((i >= max(0, (o_c + 1) * BS - 1))[:, None], b_p, 0)
        # [BC, G] @ [G, BV] -> [BC, BV]
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        # [BC, BV] @ [BV, G] -> [BC, G]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BC, G]
        b_ds = b_p * (b_dp - b_delta[None, :])
        # [BC, G] @ [G, BK] -> [BC, BK]
        b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Python-level wrapper functions (forward / backward)
# ===========================================================================


def parallel_nsa_compression_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    H = k.shape[2]
    G = HQ // H
    BS = block_size
    if check_shared_mem("hopper", q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    cu_seqlens = _ensure_i32(cu_seqlens)
    token_indices = _ensure_i32(token_indices)
    chunk_offsets = (
        _ensure_i32(prepare_chunk_offsets(cu_seqlens, BS))
        if cu_seqlens is not None
        else None
    )

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    if cu_seqlens is None and triton.cdiv(T, 8) <= _GCU_GRID_X_MAX:
        # dense path: token-tiled kernel — BT tokens share the k/v prefix
        # stream, cutting k/v traffic by ~BT vs one program per token.
        grid = lambda META: (triton.cdiv(T, META["BT"]), NV, B * H)
        parallel_nsa_compression_fwd_tiled_kernel[grid](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            scale=scale,
            T=T,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BS=BS,
            BK=BK,
            BV=BV,
        )
        return o, lse

    # varlen / extreme-T path: per-token kernel, T launched in <=65535 chunks.
    for t0 in range(0, T, _GCU_GRID_X_MAX):
        grid = (min(_GCU_GRID_X_MAX, T - t0), NV, B * H)
        parallel_nsa_compression_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            T=T,
            T_OFF=t0,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BS=BS,
            BK=BK,
            BV=BV,
        )
    return o, lse


def parallel_nsa_compression_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, HQ, K, V = *q.shape, v.shape[-1]
    TC = k.shape[1]
    H = k.shape[2]
    G = HQ // H
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)
    cu_seqlens = _ensure_i32(cu_seqlens)
    token_indices = _ensure_i32(token_indices)
    if cu_seqlens is not None:
        chunk_offsets = _ensure_i32(prepare_chunk_offsets(cu_seqlens, BS))
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(chunk_offsets, BC)
        chunk_indices = _ensure_i32(chunk_indices)
        NC = len(chunk_indices)
    else:
        chunk_indices, chunk_offsets = None, None
        NC = triton.cdiv(triton.cdiv(T, BS), BC)

    delta = _nsa_cmp_parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(
        NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device
    )
    # GCU grid.x hardware limit: launch the T dimension in <=65535 chunks.
    for t0 in range(0, T, _GCU_GRID_X_MAX):
        grid = (min(_GCU_GRID_X_MAX, T - t0), NV, B * H)
        parallel_nsa_compression_bwd_kernel_dq[grid](
            q=q,
            k=k,
            v=v,
            lse=lse,
            delta=delta,
            do=do,
            dq=dq,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            T=T,
            T_OFF=t0,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BC=BC,
            BS=BS,
            BK=BK,
            BV=BV,
        )
    dq = dq.sum(0)

    dk = torch.empty(
        NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device
    )
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NC, B * H)
    parallel_nsa_compression_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        TC=TC,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BC=BC,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


# ===========================================================================
# Autograd Function
# ===========================================================================


class ParallelNSACompressionFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q,
        k,
        v,
        block_size,
        scale,
        cu_seqlens,
    ):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the cu_seqlens of tokens in each sequence
        # for example, if the passed `cu_seqlens` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = (
            prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
        )

        o, lse = parallel_nsa_compression_fwd(
            q=q,
            k=k,
            v=v,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.cu_seqlens = cu_seqlens
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype), lse

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do, *args):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_compression_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_size=ctx.block_size,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None


# ===========================================================================
# Public API
# ===========================================================================


def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    """NSA-style parallel compression forward with backward support.

    Args:
        q: Query tensor of shape ``[B, T, HQ, K]``.
        k: Key tensor of shape ``[B, T, H, K]`` (compressed representation).
        v: Value tensor of shape ``[B, T, H, V]`` (compressed representation).
        block_size: Compression block size (default 64).
        scale: Attention scale. If ``None``, defaults to ``K ** -0.5``.
        cu_seqlens: Optional CUDA-style cumulative sequence lengths for
            variable-length sequences.

    Returns:
        Output tensor ``o`` of shape ``[B, T, HQ, V]`` and log-sum-exp
        tensor ``lse`` of shape ``[B, T, HQ]``.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return ParallelNSACompressionFunction.apply(
        q,
        k,
        v,
        block_size,
        scale,
        cu_seqlens,
    )
