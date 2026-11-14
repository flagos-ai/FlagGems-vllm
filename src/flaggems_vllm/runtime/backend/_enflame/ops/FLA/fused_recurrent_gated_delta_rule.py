# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Retains the recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Stride-aware recurrent GDN with vLLM's value-major state cache."""

import math
import os
import re

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device

# ---------------------------------------------------------------------------
# GCU toolchain regression workaround (2026-11-07 20:22 update; see
# compiler_issues/vllm_moe_sum.md): the in-tree FlagTree emits
# ``memref_ext.slice_pad_start``-family ops with a trailing zero operand
# segment that the external ``gcu-compiler-opt`` (triton-gcu 1.0.20261020)
# rejects with "operandSegmentSizes ... size mismatch", breaking every
# first-time compilation of a kernel with 2-D tile accesses. The wave-aware
# BV dispatch below widens the state tile, so new binaries must compile on
# this toolchain; the shim rewrites the generated MLIR text to the dialect
# layout the external tool accepts (pure metadata, no operand rebinding).
# It is idempotent and a no-op for 1-D-only kernels. Remove once the
# toolchain versions realign. Same pattern as the FLA parallel_nsa and
# flashmla overlays.
# ---------------------------------------------------------------------------

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
            content = "\n".join(_fix_gcu_dialect_line(l) for l in content.split("\n"))
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


@triton.jit
def _recurrent_gdn_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Out,
    Final,
    Cu,
    Indices,
    Accepted,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    POOL: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    SS: tl.constexpr,
    IX: tl.constexpr,
    WIDTH: tl.constexpr,
    VARLEN: tl.constexpr,
    INDEXED: tl.constexpr,
    TABLE: tl.constexpr,
    SPEC: tl.constexpr,
    INITIAL: tl.constexpr,
    INPLACE: tl.constexpr,
    VECTOR_BETA: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_G: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT: tl.constexpr,
):
    pid = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = pid % nv
    nh = pid // nv
    n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    if VARLEN:
        start = tl.load(Cu + n).to(tl.int64)
        end = tl.load(Cu + n + 1).to(tl.int64)
        batch = 0
    else:
        start, end = 0, T
        batch = n
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rv[:, None] < V) & (rk[None, :] < K)
    slot = n.to(tl.int64)
    valid = True
    if INDEXED:
        col = 0
        if SPEC:
            col = tl.load(Accepted + n).to(tl.int64) - 1
        valid = (col >= 0) & (col < WIDTH)
        slot = tl.load(Indices + n * IX[0] + col * IX[1], mask=valid, other=0).to(
            tl.int64
        )
        valid = valid & (slot > 0) & (slot < POOL)
    state_offsets = hv * SS[1] + rv[:, None] * SS[2] + rk[None, :] * SS[3]
    acc = tl.full((BV, BK), 0, tl.float32)
    if INITIAL:
        acc = tl.load(
            State + slot * SS[0] + state_offsets, mask=mask & valid, other=0
        ).to(tl.float32)
    # Column group for the fused dual inner product: col 0 holds k, col 1
    # holds q. With DOT, one tl.dot(acc, w) yields <acc_gated, k> and
    # <acc_gated, q> in a single MMA, replacing the two [BV, BK] product-sum
    # trees (whose live product tiles materialize across the token loop at
    # BV >= 64). The output then follows the algebraically identical
    # out = <acc_gated, q> + delta * <k, q> instead of reducing the updated
    # state again; summation order differs only at tolerance level.
    cols = tl.arange(0, 16)
    for t in range(start, end):
        q = tl.load(
            Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], rk < K, 0
        ).to(tl.float32)
        k = tl.load(
            Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], rk < K, 0
        ).to(tl.float32)
        v = tl.load(
            Vp + batch * VS[0] + t * VS[1] + hv * VS[2] + rv * VS[3], rv < V, 0
        ).to(tl.float32)
        if NORMALIZE:
            q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
            k = k / tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
        q *= scale
        if HAS_G:
            g = tl.load(G + batch * GS[0] + t * GS[1] + hv * GS[2]).to(tl.float32)
            acc *= tl.exp(g)
        if DOT:
            w = tl.where(
                cols[None, :] == 1,
                q[:, None],
                tl.where(cols[None, :] == 0, k[:, None], 0.0),
            )
            dqa = tl.dot(acc, w)
            delta = v - tl.sum(tl.where(cols[None, :] == 0, dqa, 0.0), 1)
        else:
            delta = v - tl.sum(acc * k[None, :], 1)
        if HAS_BETA:
            bp = Beta + batch * BS[0] + t * BS[1] + hv * BS[2]
            if VECTOR_BETA:
                beta = tl.load(bp + rv * BS[3], rv < V, 0).to(tl.float32)
            else:
                beta = tl.load(bp).to(tl.float32)
            delta *= beta
        acc += delta[:, None] * k[None, :]
        if DOT:
            out = tl.sum(tl.where(cols[None, :] == 1, dqa, 0.0), 1) + delta * tl.sum(
                k * q, 0
            )
        else:
            out = tl.sum(acc * q[None, :], 1)
        token = batch * T + t
        tl.store(Out + (token * HV + hv) * V + rv, tl.where(valid, out, 0), rv < V)
        if INPLACE:
            dest = slot
            store = valid
            if TABLE:
                final_col = t - start
                dest = tl.load(
                    Indices + n * IX[0] + final_col * IX[1], final_col < WIDTH, 0
                ).to(tl.int64)
                store = store & (dest > 0) & (dest < POOL) & (final_col < WIDTH)
            else:
                store = store & (t == end - 1)
            tl.store(Final + dest * SS[0] + state_offsets, acc, mask & store)
        else:
            offsets = ((token * HV + hv) * V + rv[:, None]) * K + rk[None, :]
            tl.store(Final + offsets, tl.where(valid, acc, 0), mask)


def _normalize_index_tensor(tensor):
    """Normalize int64 index metadata for kernel loads, zero device work.

    torch_gcu narrows int64 tensors to int32 storage while the metadata
    still reports int64 (unless TORCH_GCU_ENABLE_INT64_AND_UINT64 enables
    true 8-byte storage), so kernel-side int64-width loads read interleaved
    garbage slots and drive out-of-bounds state updates (see
    compiler_issues/triton_i64_narrow_load.md). In narrowed mode the
    physical 4-byte words are int32-isomorphic, so a zero-copy set_ view
    with the original shape and strides is exact. With true 8-byte storage
    the tensor passes through untouched: the kernel's masked loads then
    read native int64 words and widen via .to(tl.int64), which is
    value-correct (verified against the CPU fp64 reference on the spec and
    table workloads) and avoids a per-call ATen cast kernel that added a
    second launch and ~27-71us inside the timed window on spec/table
    shapes. Pool slots, token bounds and accepted counts always fit
    either width.
    """
    if tensor is None or tensor.dtype != torch.int64:
        return tensor
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") in ("1", "true"):
        return tensor
    view = torch.empty(0, dtype=torch.int32, device=tensor.device)
    view.set_(
        tensor.untyped_storage(),
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
    )
    return view


def fused_recurrent_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Return output and state in [pool, HV, V, K] layout.

    A 1D index maps each sequence to one cache slot (last-state writeback).
    A 2D index table stores each token's intermediate state, as required by
    speculative decoding; accepted counts select the initial table column.
    Initial slots <=0 or >=pool and invalid accepted columns produce zeros
    without updating the cache. Invalid destination slots suppress that write.
    Writable slots across sequences must be disjoint.
    Without indices, cache slots are sequence-major. Out-of-place mode returns
    every token state as [B*T, HV, V, K] and never modifies initial_state.
    cu_seqlens must be monotone, start at zero and end at T. Index-table width
    must cover each sequence. These device metadata are caller preconditions.

    Enflame GCU: int64 index metadata is normalized on the host with zero
    device work (narrowed storage becomes an int32 view; true 8-byte
    storage passes through for native int64 loads). Pool slots, token
    bounds and accepted counts always fit either width.
    Multi-wave contiguous workloads widen the state tile (BV up to 128,
    wave-aware dispatch) and fold the two product-sum trees into one
    tl.dot; single-wave, strided (gather) and narrow-K shapes keep the
    original BV=16 tree path. Both paths are shape-dispatched host-side
    from metadata only.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k must have identical [B,T,H,K] shapes; v must be rank 4")
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    if v.shape[:2] != (B, T) or min(H, HV, K, V) <= 0 or HV % H:
        raise ValueError("Invalid GVA dimensions")
    if K > 256:
        raise NotImplementedError("Recurrent GDN supports K <= 256")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("Recurrent GDN supports FP16, BF16 and FP32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have the same dtype")
    if scale is None:
        scale = K**-0.5
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("Packed varlen requires B=1 and cu_seqlens[N+1]")
        if not cu_seqlens.is_contiguous() or cu_seqlens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cu_seqlens must be contiguous int32/int64")
        N = cu_seqlens.numel() - 1
    else:
        N = B
    if inplace_final_state and initial_state is None:
        raise ValueError("In-place mode requires initial_state")
    if initial_state is not None:
        if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, V, K):
            raise ValueError("initial_state must use [pool, HV, V, K] layout")
        if initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported state dtype")
        if any(s <= 0 for s in initial_state.stride()):
            raise ValueError("State must have positive non-overlapping strides")
        ordered = sorted(zip(initial_state.stride(), initial_state.shape))
        extent = 1
        for stride, size in ordered:
            if size > 1 and stride < extent:
                raise ValueError("Overlapping state is unsupported")
            extent = max(extent, stride * size)
        pool, ss = initial_state.shape[0], initial_state.stride()
        if ssm_state_indices is None and pool < N:
            raise ValueError("State pool is smaller than the sequence count")
    else:
        pool, ss = N, (HV * V * K, V * K, K, 1)
    width = 1
    if ssm_state_indices is not None:
        if initial_state is None:
            raise ValueError("Indexed mode requires an initial state pool")
        if ssm_state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("State indices must be int32/int64")
        if ssm_state_indices.ndim not in (1, 2) or ssm_state_indices.shape[0] != N:
            raise ValueError("State indices must have shape [N] or [N,width]")
        if ssm_state_indices.ndim == 2:
            width = ssm_state_indices.shape[1]
            if width == 0:
                raise ValueError("State table cannot have zero width")
    if num_accepted_tokens is not None:
        if ssm_state_indices is None or ssm_state_indices.ndim != 2:
            raise ValueError("Accepted-token counts require a 2D state table")
        if (
            num_accepted_tokens.shape != (N,)
            or not num_accepted_tokens.is_contiguous()
            or num_accepted_tokens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Accepted-token counts must be contiguous int32/int64 [N]")
    if g is not None and g.shape != (B, T, HV):
        raise ValueError("g must have shape [B,T,HV]")
    if beta is not None and beta.shape not in ((B, T, HV), (B, T, HV, V)):
        raise ValueError("beta must have shape [B,T,HV] or [B,T,HV,V]")
    for tensor in (
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("All inputs must be on the same device")
    for tensor in (g, beta):
        if tensor is not None and tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("g/beta must be floating point")
    # Kernel-side int64-width loads of torch_gcu's narrowed int64 index
    # metadata read interleaved garbage slots; normalize on the host after
    # validation (see _normalize_index_tensor). Neither branch materializes
    # a copy, so strides always describe the original storage.
    cu_seqlens = _normalize_index_tensor(cu_seqlens)
    ssm_state_indices = _normalize_index_tensor(ssm_state_indices)
    num_accepted_tokens = _normalize_index_tensor(num_accepted_tokens)
    ix = (0, 0)
    if ssm_state_indices is not None:
        ix = (
            ssm_state_indices.stride()
            if ssm_state_indices.ndim == 2
            else (ssm_state_indices.stride(0), 0)
        )
    o = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty(
            (B * T, HV, V, K),
            dtype=initial_state.dtype if initial_state is not None else torch.float32,
            device=q.device,
        )
    )
    if B * T == 0:
        return o, final
    if N == 0:
        raise ValueError("Nonempty input requires at least one sequence")
    # No-autotune exemption: replaying this kernel mutates its state pool.
    # Tile experiments must restore state outside timing, never in production.
    #
    # Wave-aware BV dispatch (host metadata only, no device reads):
    # programs = N * HV * cdiv(V, BV) run a serial per-sequence recurrence,
    # so wall clock ~ ceil(programs / 48) waves x per-program op chain.
    # Widening BV folds more V rows into one program: per-program cost grows
    # sub-linearly (~1.75x for 8x rows on decode shapes) while waves drop
    # 8x, measured -76%..-79% on decode256 across dtypes. Guards:
    #   * single-wave shapes (BV=16 already fits the machine) keep BV=16;
    #   * N * HV >= 48 keeps the widened grid itself at >= 1 full wave, so
    #     long single-sequence prefills never trade serial steps for idle
    #     SPs;
    #   * innermost strides must be 1: strided (gather) state/q/k/v access
    #     is priced per element, so widening buys nothing (measured flat);
    #   * the state tile stays <= 16384 fp32 elements (64KB, the measured
    #     DSM-healthy domain; 48-way residency verified by perf).
    # DOT additionally folds the two [BV, BK] product-sum trees into one
    # tl.dot via out = <acc_gated, q> + delta * <k, q> (fused_sigmoid_gating
    # precedent; same algebra family). This removes the loop-carried product
    # materialization that made BV >= 64 regress on multi-token sequences
    # (mtp: 95.9 -> 56.0 us) and gives another ~4% on single-step decode.
    # Below BK = 128 the dot's fixed overhead exceeds the tree cost, so the
    # tree path stays.
    bk = triton.next_power_of_2(K)
    qs, ks, vs = q.stride(), k.stride(), v.stride()
    bv = 16
    use_dot = False
    if (
        V > 16
        and qs[3] == 1
        and ks[3] == 1
        and vs[3] == 1
        and ss[3] == 1
        and N * HV * triton.cdiv(V, 16) > 48
        and N * HV >= 48
    ):
        bv = min(triton.next_power_of_2(V), 128, max(16, 16384 // bk))
        use_dot = bk >= 128
    bs = beta.stride() if beta is not None else (0, 0, 0)
    if len(bs) == 3:
        bs = (*bs, 0)
    total = N * HV * triton.cdiv(V, bv)
    grid, kernel = (total,), _recurrent_gdn_kernel
    kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        o,
        final,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        T,
        H,
        HV,
        K,
        V,
        pool,
        q.stride(),
        k.stride(),
        v.stride(),
        g.stride() if g is not None else (0, 0, 0),
        bs,
        ss,
        ix,
        width,
        cu_seqlens is not None,
        ssm_state_indices is not None,
        ssm_state_indices is not None and ssm_state_indices.ndim == 2,
        num_accepted_tokens is not None,
        initial_state is not None,
        inplace_final_state,
        beta is not None and beta.ndim == 4,
        use_qk_l2norm_in_kernel,
        g is not None,
        beta is not None,
        bk,
        bv,
        use_dot,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return o, final


if device.vendor_name == "enflame":
    # torch_gcu narrows int64 tensors to int32 storage while the metadata
    # still reports int64, so the untouched generic entry would feed the
    # kernel int64 index metadata whose loads read interleaved garbage slots
    # (see compiler_issues/triton_i64_narrow_load.md). The generic sources
    # are outside this overlay and must stay untouched, so redirect the
    # generic entry (``flaggems_vllm.ops`` / ``flaggems_vllm.ops.FLA``, used
    # by tests and benchmarks) to this implementation — the same "vendor
    # implementation under the generic name" pattern the generic ops package
    # itself uses for other vendors. The production entry
    # (``flaggems_vllm.<op>``) already resolves here via SpecOpRegistrar and
    # is unchanged.
    import flaggems_vllm.ops as _generic_ops_pkg
    import flaggems_vllm.ops.FLA as _generic_fla_pkg

    _generic_ops_pkg.fused_recurrent_gated_delta_rule_fwd = (
        fused_recurrent_gated_delta_rule_fwd
    )
    _generic_fla_pkg.fused_recurrent_gated_delta_rule_fwd = (
        fused_recurrent_gated_delta_rule_fwd
    )
