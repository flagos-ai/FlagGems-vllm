# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Enflame GCU adaptations (local backend copy, do not sync back to generic):
# 1. int64 index tensors (block_indices / block_counts / cu_seqlens /
#    token_indices / chunk_indices / chunk_offsets) are normalized to int32 on
#    the host before every kernel launch: torch_gcu silently narrows Long
#    storage to 32-bit ("GCU not support Long use Int replace"), while the
#    Triton launcher still reads the buffer at 8-byte stride -> garbage
#    indices -> wrong results / device abort (varlen baseline crash).
# 2. The TLE async-load forward variant is gated OFF by default
#    (FLA_NSA_TLE=1 to opt in): tle.load(is_async=...) is unvalidated on GCU
#    and the plain kernel is bit-identical in math.
# 3. mean_pooling of k/v is only computed when g_cmp is provided (the pooled
#    tensors are unused otherwise); saves two kernel launches per call.
# 4. GCU grid.x is hardware-limited to 65535, so the T dimension of the
#    forward/dq grids is launched in <=65535-token chunks via a T_OFF offset,
#    and the backward delta preprocessing uses a local grid.x-capped variant
#    (the generic one-row-per-CTA launch overflows for B*T*HQ > 65535).
# 5. Toolchain skew shim (2026-11-07 20:22): the in-process GCU pass library
#    emits GCUIR whose memref_ext.slice_*/gcu.matmul ops carry one extra empty
#    variadic segment in operandSegmentSizes (plus an inputPrecision property
#    on gcu.matmul and a properties dict on memref_ext.memset_start) that the
#    older external gcu-compiler-opt rejects at parse time ("size mismatch in
#    attribute conversion"). The shim downgrades those encodings to the old
#    format on a retry path only, scoped to the whitelisted kernels, so kernels
#    that already compile are untouched. See
#    compiler_issues/vllm_parallel_nsa_gcu_matmul.md; remove once the toolchain
#    is re-synced.
# 6. Forward kernel S-loop unrolled when block counts are fixed (int): the
#    dynamic scf.for + scf.if iteration machinery costs ~1us per skipped
#    block on GCU (measured: all-skip workload spends ~15.5us/program in the
#    16-iteration loop body doing nothing but load-index + branch).
#    tl.static_range tracing-time unrolling removes the loop machinery
#    (-6.2% end-to-end on the core NSA shapes, -18.7% on the skip-heavy
#    regime, bit-identical output). The value tile load is also issued right
#    after the key tile load (its latency hides behind the score dot +
#    softmax chain) and the output rescale is folded into tl.dot(acc=).
#    Variable block counts (tensor) keep the original dynamic loop.

import os
import warnings

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_lens,
    prepare_token_indices,
)
from flaggems_vllm.ops.FLA.mean_pooling import mean_pooling
from flaggems_vllm.ops.FLA.parallel_nsa_compression import parallel_nsa_compression
from flaggems_vllm.ops.FLA.triton_ops_helper import autotune_cache_kwargs, exp, log
from flaggems_vllm.ops.FLA.utils import _bitonic_merge, check_shared_mem, input_guard

try:
    HAS_TLE = False
    import triton.experimental.tle.language as tle  # noqa: F401

    HAS_TLE = True
except ImportError:
    tle = None
    HAS_TLE = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None


# Enflame GCU: torch_gcu narrows int64 (Long) storage to 32-bit while the
# tensor still reports torch.int64, so Triton kernels would read the buffer
# with 8-byte strides over 4-byte elements (garbage indices). Normalize every
# integer tensor to int32 before it reaches a kernel launch.
def _ensure_i32(t):
    if torch.is_tensor(t) and t.dtype == torch.int64:
        return t.to(torch.int32)
    return t


# Enflame GCU hardware grid limits (see compiler_issues/grid_hard_limits.md).
_GCU_GRID_X_MAX = 65535


# Kernels whose GCUIR may be downgraded by the toolchain-compat shim below.
# fp8_fp4_mqa_logits kernels (2D-tile tl.dot) hit the same 20:22 slice_pad_start
# skew on first compile; they are whitelisted here so the retry-path downgrade
# covers them too (job j20261107224029_8c49d854).
_COMPAT_SCOPED_KERNELS = (
    "parallel_nsa_fwd_kernel",
    "parallel_nsa_fwd_kernel_tle",
    "parallel_nsa_bwd_kernel_dq",
    "parallel_nsa_bwd_kernel_dkv",
    "parallel_nsa_kernel_topk",
    "parallel_nsa_kernel_mask",
    "_fp8_fp4_mqa_logits_kernel",
    "_fp8_fp4_mqa_logits_mxfp4_kernel",
    "_fp8_fp4_mqa_logits_kernel_tle",
    "_clean_logits_kernel",
)


def _downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form.

    The 2026-11-07 20:22 in-process pass library emits:
    - memref_ext.slice_* / gcu.matmul with one extra trailing empty variadic
      segment in operandSegmentSizes;
    - gcu.matmul with an inputPrecision property;
    - memref_ext.memset_start with a properties dict.
    The older external gcu-compiler-opt rejects all three. The rewrite only
    fires on the retry path of the shim, so valid old-format IR is untouched.
    """
    import re

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _install_gcuir_text_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_parallel_nsa_compat", False):
        return
    original = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return original(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(name in text for name in _COMPAT_SCOPED_KERNELS):
                raise first_error
            try:
                return original(_downgrade_gcuir_for_external_tool(text), *args)
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._parallel_nsa_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


try:
    _install_gcuir_text_compat()
except Exception:
    # pytest assertion-rewrite 上下文中 triton.jit 源定位可能失败(collect 期
    # 动态触碰模块链触发);shim 失败不影响 op 路由,由 conftest 的共享
    # gcuir_compat(_gcuir_compat.py)在测试进程统一安装兜底。
    pass


# ===========================================================================
# Top-K selection kernel
# ===========================================================================


@triton.jit
def _nsa_fwd_block(
    b_q,
    k,
    v,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    i_v,
    i_t,
    i_s,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    b_m,
    b_acc,
    b_o,
):
    # Key and value tiles are issued back-to-back so the value load latency
    # hides behind the score dot + softmax chain (GCU: the original
    # load-after-score ordering leaves the value DMA latency exposed).
    p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
    # [BK, BS]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [G, BS] — compute QK^T scores
    b_s = tl.dot(b_q, b_k)
    b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf"))

    # [G]
    b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
    b_r = exp(b_mp - b_m)
    # [G, BS]
    b_p = exp(b_s - b_m[:, None])
    # [G]
    b_acc = b_acc * b_r + tl.sum(b_p, 1)
    # [G, BV] — rescale folded into the MMA accumulator chain
    b_o = tl.dot(b_p.to(b_q.dtype), b_v, acc=b_o * b_r[:, None])
    return b_m, b_acc, b_o


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_kernel_topk(
    q,
    k,
    lse,
    scale,
    block_indices,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    ################################
    # 1. lse computation
    ################################
    if lse is not None:
        b_lse = tl.load(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G))
    else:
        # max scores for the current block
        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        # lse = log(acc) + m
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)

            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
            )
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BC]
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([G], dtype=tl.float32)
        else:
            b_lse = b_m + log(b_acc)

    ################################
    # 2. topk selection
    ################################
    # [BC]
    b_i = tl.full([BC], -1, dtype=tl.float32)
    o_i = tl.zeros([BC], dtype=tl.int32)
    m_i = tl.arange(0, BC) < BC // 2

    IC = i_t // BS
    for i_c in range(0, tl.cdiv(i_t + 1, BS), BC):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
        )
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [G, BC]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(o_c < IC, b_s, float("-inf"))
        # [G, BC]
        # the 1st and the last 2 blocks are always selected
        b_p = tl.where(
            (o_c == 0) | ((o_c == IC - 1) | (o_c == IC)), 1.0, exp(b_s - b_lse[:, None])
        )
        # the importance scores of the current block
        # [BC]
        b_i, b_ip = tl.sum(b_p, 0), b_i
        # blocks with index < 0 will be skipped
        o_i, o_ip = tl.where(o_c <= IC, o_c, -1), o_i

        n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(
                b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)

    m_top = tl.arange(0, BC // S) == 0
    b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC // S, S]), 0)

    p_b = tl.make_block_ptr(
        block_indices + (bos + i_t) * H * S, (H * S,), (1,), (i_h * S,), (S,), (0,)
    )
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


# ===========================================================================
# Forward kernel
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T_OFF"])
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    T_OFF,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
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
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    if USE_BLOCK_COUNTS:
        # per-query block counts are runtime data: keep the dynamic loop
        for i in range(NS):
            i_s = tl.load(block_indices + i).to(tl.int32) * BS
            if i_s <= i_t and i_s >= 0:
                b_m, b_acc, b_o = _nsa_fwd_block(
                    b_q, k, v, T, H, K, V, i_v, i_t, i_s, BS, BK, BV, b_m, b_acc, b_o
                )
    else:
        # NS == S (constexpr): trace-time unrolled. The dynamic scf.for +
        # scf.if machinery costs ~1us per block on GCU even when the causal
        # guard skips all work; unrolling removes it. Bit-identical math.
        for i in tl.static_range(0, S):
            i_s = tl.load(block_indices + i).to(tl.int32) * BS
            if i_s <= i_t and i_s >= 0:
                b_m, b_acc, b_o = _nsa_fwd_block(
                    b_q, k, v, T, H, K, V, i_v, i_t, i_s, BS, BK, BV, b_m, b_acc, b_o
                )
    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


# ===========================================================================
# TLE-optimized forward kernel (Triton Language Extensions)
# ===========================================================================

if HAS_TLE:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
            "USE_BLOCK_COUNTS": lambda args: isinstance(
                args["block_counts"], torch.Tensor
            ),
        }
    )
    @triton.autotune(
        configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
        key=["BS", "BK", "BV"],
        **autotune_cache_kwargs,
    )
    @triton.jit(do_not_specialize=["T_OFF"])
    def parallel_nsa_fwd_kernel_tle(
        q,
        k,
        v,
        o,
        lse,
        scale,
        block_indices,
        block_counts,
        cu_seqlens,
        token_indices,
        T,
        T_OFF,
        H: tl.constexpr,
        HQ: tl.constexpr,
        G: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        S: tl.constexpr,
        BS: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        USE_BLOCK_COUNTS: tl.constexpr,
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
        else:
            bos, eos = i_b * T, i_b * T + T

        k += (bos * H + i_h) * K
        v += (bos * H + i_h) * V
        block_indices += (bos + i_t) * H * S + i_h * S

        if USE_BLOCK_COUNTS:
            NS = tl.load(block_counts + (bos + i_t) * H + i_h)
        else:
            NS = S

        p_q = tl.make_block_ptr(
            q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_o = tl.make_block_ptr(
            o + (bos + i_t) * HQ * V,
            (HQ, V),
            (V, 1),
            (i_h * G, i_v * BV),
            (G, BV),
            (1, 0),
        )
        p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_o = tl.zeros([G, BV], dtype=tl.float32)

        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        b_acc = tl.zeros([G], dtype=tl.float32)

        # Precompute indices for async K/V loads via tle.load(is_async=True).
        offs_k = tl.arange(0, BK)
        offs_s = tl.arange(0, BS)
        offs_v = tl.arange(0, BV)
        k_base = k  # already offset to (bos*H + i_h)*K
        v_base = v  # already offset to (bos*H + i_h)*V
        v_start = i_v * BV

        for i in range(NS):
            i_s = tl.load(block_indices + i).to(tl.int32) * BS
            if i_s <= i_t and i_s >= 0:
                # Async K load via regular pointer arithmetic
                k_ptrs = k_base + offs_k[:, None] + (i_s + offs_s[None, :]) * H * K
                k_mask = (offs_k[:, None] < K) & ((i_s + offs_s[None, :]) < T)
                b_k = tle.load(k_ptrs, mask=k_mask, other=0.0, is_async=True)

                # Compute QK^T scores
                b_s = tl.dot(b_q, b_k.to(b_q.dtype))
                b_s = tl.where(
                    (i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf")
                )

                # [G]
                b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                b_r = exp(b_mp - b_m)
                # [G, BS] — b_s registers reused for softmax output b_p
                b_p = exp(b_s - b_m[:, None])
                # [G]
                b_acc = b_acc * b_r + tl.sum(b_p, 1)

                # Async V load AFTER score computation
                v_ptrs = (
                    v_base
                    + (i_s + offs_s[:, None]) * H * V
                    + (v_start + offs_v[None, :])
                )
                v_mask = ((i_s + offs_s[:, None]) < T) & (
                    (v_start + offs_v[None, :]) < V
                )
                b_v = tle.load(v_ptrs, mask=v_mask, other=0.0, is_async=True)

                # [G, BV]
                b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

                b_mp = b_m
        b_o = b_o / b_acc[:, None]
        b_m += log(b_acc)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


# ===========================================================================
# Block mask kernel
# ===========================================================================


@triton.heuristics(
    {
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(
            block_counts + i_b * T * H + i_t * H + i_h
        )
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(
            block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i,
            b_m.to(block_mask.dtype.element_ty),
        )


# ===========================================================================
# Backward kernel: dQ
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "T_OFF"])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    T_OFF,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
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
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
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
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(
            chunk_indices + i_s * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * all * H + bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_s * BS, 0),
        (BS, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m:
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
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Python-level wrapper functions
# ===========================================================================


def parallel_nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.LongTensor:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    # the number of selected blocks for each token
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S)
    # here we set BC = BS, but beware that they can be chosen separately if required
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    assert BC >= 2 * S, f"BC ({BC}) must be greater than or equal to 2 * S ({S})"

    cu_seqlens = _ensure_i32(cu_seqlens)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    token_indices = (
        _ensure_i32(prepare_token_indices(cu_seqlens))
        if cu_seqlens is not None
        else None
    )
    chunk_offsets = (
        _ensure_i32(prepare_chunk_offsets(cu_seqlens, BS))
        if cu_seqlens is not None
        else None
    )
    grid = (T, B * H)
    # the 1st and the last 2 blocks are always selected
    parallel_nsa_kernel_topk[grid](
        q=q,
        k=k,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        S=S,
        BC=BC,
        BS=BS,
        BK=BK,
    )
    return block_indices


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    block_indices = _ensure_i32(block_indices)
    block_counts = _ensure_i32(block_counts)
    cu_seqlens = _ensure_i32(cu_seqlens)
    token_indices = _ensure_i32(token_indices)
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
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

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    # Dispatch to TLE-optimized kernel only when explicitly requested:
    # tle.load(is_async=True) is unvalidated on GCU, the plain kernel is
    # mathematically identical (FLA_NSA_TLE=1 restores the generic default).
    _use_tle = HAS_TLE and os.environ.get("FLA_NSA_TLE", "0") == "1"
    if _use_tle:
        kernel = parallel_nsa_fwd_kernel_tle
    else:
        kernel = parallel_nsa_fwd_kernel

    # GCU grid.x hardware limit: launch the T dimension in <=65535 chunks.
    for t0 in range(0, T, _GCU_GRID_X_MAX):
        grid = (min(_GCU_GRID_X_MAX, T - t0), NV, B * H)
        kernel[grid](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            scale=scale,
            block_indices=block_indices,
            block_counts=block_counts,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            T=T,
            T_OFF=t0,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            S=S,
            BS=BS,
            BK=BK,
            BV=BV,
        )
    return o, lse


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(prepare_lens(cu_seqlens).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H * S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS,
    )
    return block_mask


# Backward preprocessing (local copy): the generic
# parallel_attn_bwd_kernel_preprocess launches one CTA per flattened row
# (grid=(numel,)), which overflows the GCU grid.x limit of 65535 for
# B*T*HQ > 65535 (e.g. 3*1024*32 = 98304). This variant decomposes the flat
# row id over a capped 3D grid with an in-kernel bound guard; the math is
# identical to the generic kernel.
@triton.jit(do_not_specialize=["N", "GRID_X", "GRID_XY"])
def _nsa_attn_bwd_preprocess_kernel(
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


def _nsa_parallel_attn_bwd_preprocess(o, do):
    """Compute delta = sum(o * do, -1) with a GCU grid.x-safe launch."""
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    N = delta.numel()
    grid_x = min(N, _GCU_GRID_X_MAX)
    remainder = triton.cdiv(N, grid_x)
    grid_y = min(remainder, 255)
    grid_z = triton.cdiv(remainder, grid_y)
    _nsa_attn_bwd_preprocess_kernel[(grid_x, grid_y, grid_z)](
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


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    block_indices = _ensure_i32(block_indices)
    block_counts = _ensure_i32(block_counts)
    cu_seqlens = _ensure_i32(cu_seqlens)
    token_indices = _ensure_i32(token_indices)
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)

    delta = _nsa_parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(
        NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device
    )
    # GCU grid.x hardware limit: launch the T dimension in <=65535 chunks.
    for t0 in range(0, T, _GCU_GRID_X_MAX):
        grid = (min(_GCU_GRID_X_MAX, T - t0), NV, B * H)
        parallel_nsa_bwd_kernel_dq[grid](
            q=q,
            k=k,
            v=v,
            lse=lse,
            delta=delta,
            do=do,
            dq=dq,
            block_indices=block_indices,
            block_counts=block_counts,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            scale=scale,
            T=T,
            T_OFF=t0,
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            S=S,
            BS=BS,
            BK=BK,
            BV=BV,
        )
    dq = dq.sum(0)

    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        chunk_indices = _ensure_i32(chunk_indices)
        NS = len(chunk_indices)
    else:
        NS = triton.cdiv(T, BS)

    block_mask = parallel_nsa_block_mask(
        block_indices, block_counts, cu_seqlens, block_size
    )
    dk = torch.empty(
        NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device
    )
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


# ===========================================================================
# Autograd Function
# ===========================================================================


class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens
    ):
        ctx.dtype = q.dtype
        block_indices = _ensure_i32(block_indices)
        block_counts = _ensure_i32(block_counts)
        cu_seqlens = _ensure_i32(cu_seqlens)

        token_indices = (
            _ensure_i32(prepare_token_indices(cu_seqlens))
            if cu_seqlens is not None
            else None
        )

        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.cu_seqlens = cu_seqlens
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Native Sparse Attention (NSA) — parallel implementation.

    Args:
        q (torch.Tensor):
            queries of shape ``[B, T, HQ, K]``.
        k (torch.Tensor):
            keys of shape ``[B, T, H, K]``.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H)
            must be a power of 2 and >= 16.
        v (torch.Tensor):
            values of shape ``[B, T, H, V]``.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape ``[B, T, HQ]``. If provided,
            ``block_indices`` will be computed automatically.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape ``[B, T, HQ]``.
        g_swa (torch.Tensor):
            Gate score for sliding window attention of shape ``[B, T, HQ]``.
        block_indices (torch.LongTensor):
            Pre-computed block indices of shape ``[B, T, H, S]``.
            If ``g_cmp`` is provided, this is computed automatically.
        block_counts (int or torch.LongTensor):
            Number of selected blocks for each query. If a tensor, shape ``[B, T, H]``.
            Default: 16.
        block_size (int):
            Size of each selected block. Default: 64.
        window_size (int):
            Sliding window size. 0 means no sliding attention. Default: 0.
        scale (float):
            Scale factor. If ``None``, defaults to ``K ** -0.5``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths for variable-length sequences.

    Returns:
        torch.Tensor of shape ``[B, T, HQ, V]``.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    assert (
        q.shape[2] % (k.shape[2] * 16) == 0
    ), "Group size must be a multiple of 16 in NSA"

    # GCU: int64 index tensors are narrowed to int32 storage by torch_gcu;
    # normalize before any kernel sees them.
    block_indices = _ensure_i32(block_indices)
    block_counts = _ensure_i32(block_counts)
    cu_seqlens = _ensure_i32(cu_seqlens)

    k_cmp = v_cmp = None
    o_cmp, lse_cmp = None, None
    if g_cmp is not None:
        # The pooled/compressed tensors are only consumed on the g_cmp path,
        # so skip the two mean_pooling launches entirely otherwise.
        k_cmp = mean_pooling(k, block_size, cu_seqlens)
        v_cmp = mean_pooling(v, block_size, cu_seqlens)
        o_cmp, lse_cmp = parallel_nsa_compression(
            q=q,
            k=k_cmp,
            v=v_cmp,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        if block_indices is not None:
            warnings.warn("`block_indices` will be ignored when `g_cmp` is provided")
        block_indices = parallel_nsa_topk(
            q=q,
            k=k_cmp,
            lse=lse_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
    o = o_slc = ParallelNSAFunction.apply(
        q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens
    )
    if g_slc is not None:
        o = o_slc * g_slc.unsqueeze(-1)
    if o_cmp is not None:
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
    if window_size > 0:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError:
            raise ImportError(
                "Flash Attention is required for sliding window attention. "
                "Install it via `pip install flash-attn --no-build-isolation`"
            )
        if cu_seqlens is not None:
            max_seqlen = q.shape[1]
            o_swa = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(window_size - 1, 0),
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(window_size - 1, 0),
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
    return o
