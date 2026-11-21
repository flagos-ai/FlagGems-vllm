# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
# Adapted from vLLM 0.24.0 layers/fla/ops/fused_recurrent.py.
# The upstream file includes MIT-licensed flash-linear-attention code:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.


import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import device

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255

# Wave-aware BV dispatch constants (Libra/GCU400: 48 stream processors).
_GCU_NUM_SP = 48

# DSM-healthy state tile bound (fp32 elements) for the widened arm.
_MAX_STATE_TILE_ELEMS = 16384

# Beta rounding is only preserved on the Ascend lowering; hoisted so the
# launch path avoids a per-call vendor-name comparison.
_IS_ASCEND = device.vendor_name == "ascend"


def _next_pow2(n):
    # Bit-trick next power of two >= n (n >= 1); avoids the triton python
    # wrapper (unwrap/specialize machinery) on the per-call launch path.
    return 1 << max(0, (n - 1).bit_length())


def _cdiv(a, b):
    return -(-a // b)


@triton.jit
def _round_beta_ascend(x, DTYPE: tl.constexpr):
    # Ascend scalar lowering can eliminate a narrowing/widening cast pair.
    # Preserve vLLM 0.24 beta rounding explicitly; sigmoid is in [0, 1].
    bits = x.to(tl.uint32, bitcast=True)
    if DTYPE == tl.bfloat16:
        rounded = (bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000
        result = rounded.to(tl.float32, bitcast=True)
    elif DTYPE == tl.float16:
        rounded = (bits + 4095 + ((bits >> 13) & 1)) & 0xFFFFE000
        normal = rounded.to(tl.float32, bitcast=True)
        scaled = x * 16777216.0
        lower = tl.floor(scaled)
        fraction = scaled - lower
        up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
        subnormal = (lower + up.to(tl.float32)) * 5.960464477539063e-8
        result = tl.where(x < 6.103515625e-5, subnormal, normal)
    else:
        result = x
    return tl.where(x != x, x, result)


@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok,
    stride_a_tok,
    stride_b_tok,
    stride_init_state_token,
    stride_final_state_token,
    stride_indices_seq,
    grid_x,
    grid_xy,
    num_programs,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    EVEN_V: tl.constexpr,
    EVEN_K: tl.constexpr,
    PRESERVE_BETA_ROUNDING: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    # Enflame GCU adaptation notes:
    # 1. The flat program id is recombined from a host-side grid folded within
    #    the GCU launch limits (grid.x <= 65535, grid.y <= 255, grid.z <= 255).
    # 2. ssm_state_indices is normalized to int32 on the host: torch_gcu
    #    narrows int64 tensors to int32 storage while metadata still reports
    #    int64, so kernel-side int64-width loads would read garbage. The slot
    #    base offset still uses int64 arithmetic for large state pools.
    # 3. The [BV, BK] state tile is loaded before the small qkv/scalar loads
    #    ("big DMA first"): the largest DMA is emitted while the gate chain's
    #    serial dependencies resolve. For this kernel family the early state
    #    load combined with num_warps=1 is also the load-order combination
    #    validated for correctness (a late state load silently miscomputed
    #    multi-V-block shapes under the 2026-11-07 toolchain).
    # 4. The delta-rule single step is fully vectorized over the 2-D tile:
    #        h2[v, k] = decay * h[v, k] + vc[v] * k[k]
    #        o[v]     = decay * (h @ q)[v] + vc[v] * (k @ q)
    #    with vc = (v - decay * (h @ k)) * beta, so no serial per-row loop is
    #    needed (pure affine 2-D tiles compile on the current toolchain).
    pid = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if pid >= num_programs:
        return
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # Skip if state index is invalid (NULL_BLOCK_ID=0)
    if state_idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token + i_hv * V * K
    offs = o_v[:, None] * K + o_k[None, :]
    if EVEN_V and EVEN_K:
        b_h = tl.load(p_h0 + offs).to(tl.float32)
    elif EVEN_K:
        b_h = tl.load(p_h0 + offs, mask=mask_v[:, None], other=0).to(tl.float32)
    elif EVEN_V:
        b_h = tl.load(p_h0 + offs, mask=mask_k[None, :], other=0).to(tl.float32)
    else:
        b_h = tl.load(p_h0 + offs, mask=mask_v[:, None] & mask_k[None, :], other=0).to(
            tl.float32
        )

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + (H * K) + i_h * K + o_k, mask=mask_k, other=0).to(
        tl.float32
    )
    b_v = tl.load(p_mixed + (2 * H * K) + i_hv * V + o_v, mask=mask_v, other=0).to(
        tl.float32
    )

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    if PRESERVE_BETA_ROUNDING:
        beta_val = _round_beta_ascend(tl.sigmoid(b_val), b.dtype.element_ty)
    else:
        beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    decay = tl.exp(g_val)
    b_hk = tl.sum(b_h * b_k[None, :], 1)
    b_hq = tl.sum(b_h * b_q[None, :], 1)
    b_kq = tl.sum(b_k * b_q, 0)
    b_vc = (b_v - decay * b_hk) * beta_val
    b_o = decay * b_hq + b_vc * b_kq
    b_h2 = decay * b_h + b_vc[:, None] * b_k[None, :]

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token + i_hv * V * K
    if EVEN_V and EVEN_K:
        tl.store(p_ht + offs, b_h2.to(ht.dtype.element_ty))
    elif EVEN_K:
        tl.store(p_ht + offs, b_h2.to(ht.dtype.element_ty), mask=mask_v[:, None])
    elif EVEN_V:
        tl.store(p_ht + offs, b_h2.to(ht.dtype.element_ty), mask=mask_k[None, :])
    else:
        tl.store(
            p_ht + offs,
            b_h2.to(ht.dtype.element_ty),
            mask=mask_v[:, None] & mask_k[None, :],
        )


def _decode_grid(total):
    """Fold ``total`` logical programs into a GCU-safe 3-D grid.

    Any decomposition with ``grid_x * grid_y * grid_z >= total`` is correct;
    the kernel recombines the flat program id via ``grid_x``/``grid_xy``
    runtime arguments and guards ``pid < num_programs``. Covers up to
    65535 * 255 * 255 ~= 4.26e9 programs.
    """
    if total <= _GRID_X_MAX:
        return (total, 1, 1)
    grid_y = _cdiv(total, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (_cdiv(total, grid_y), grid_y, 1)
    grid_z = _cdiv(total, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = _cdiv(total, _GRID_X_MAX * grid_z)
    return (_cdiv(total, grid_y * grid_z), grid_y, grid_z)


def _pick_bv(B, HV, V, BK):
    """Wave-aware BV widening (host metadata only, no device reads).

    Programs = B * HV * ceil(V / BV) each stream one [BV, BK] state tile
    (load -> reductions -> rank-1 update -> store), so wall clock is roughly
    ceil(programs / 48) waves. Widening BV folds more V rows into one
    program: the per-program cost grows sub-linearly while the wave count
    drops. The widened arm is taken only when it strictly reduces the wave
    count and the widened tile stays within the DSM-healthy bound; the host
    validation above already guarantees dense state HV/V/K rows.
    """
    BV = min(64, _next_pow2(V))
    wide_bv = min(_next_pow2(V), 128, max(1, _MAX_STATE_TILE_ELEMS // BK))
    if wide_bv > BV:
        programs_base = B * HV * _cdiv(V, BV)
        programs_wide = B * HV * _cdiv(V, wide_bv)
        if _cdiv(programs_base, _GCU_NUM_SP) > _cdiv(programs_wide, _GCU_NUM_SP):
            BV = wide_bv
    return BV


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM 0.24.0 single-token packed GDN decode, updating state in place.

    State is [pool, HV, V, K]. Positive indices must be in bounds and unique;
    nonpositive indices produce zero output without updating the state pool.
    The state head/V/K dimensions must be dense; pool rows may be padded.
    Output and state must not overlap each other or read-only inputs.
    This is inference-only. Beta rounds through b.dtype, as in vLLM 0.24.0.

    Fixed launch configuration is intentional: runtime autotuning would replay
    a stateful operation. Configuration experiments use externally restored
    state and no-Graph timing (see the accompanying benchmark).

    Enflame GCU: state indices are normalized to int32 on the host (see the
    kernel notes), the launch grid is folded within GCU limits, and the BV
    block size is picked wave-aware (see ``_pick_bv``).
    """
    if device.vendor_name in ("nvidia", "hygon", "thead"):
        return _packed_decode_gpu(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            initial_state,
            out,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )
    # Validation reads each shape/stride tuple once and compares tuples: the
    # single-token decode is launch-latency sensitive on small batches, and
    # per-attribute Python hops are measurable inside the timed window.
    ms, ash, bsh = mixed_qkv.shape, a.shape, b.shape
    als, dts = A_log.shape, dt_bias.shape
    hs, outs, ids = initial_state.shape, out.shape, ssm_state_indices.shape
    float_types = (torch.float16, torch.bfloat16, torch.float32)
    if any(
        t.dtype not in float_types
        for t in (mixed_qkv, a, b, A_log, dt_bias, initial_state, out)
    ):
        raise NotImplementedError("Packed decode supports FP16, BF16 and FP32.")
    if ssm_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("State indices must be int32 or int64.")
    if len(ms) != 2 or len(ash) != 2 or len(bsh) != 2:
        raise ValueError("mixed_qkv, a and b must be 2D tensors.")
    if len(als) != 1 or len(dts) != 1 or len(ids) != 1:
        raise ValueError("A_log, dt_bias and state indices must be 1D tensors.")
    if len(hs) != 4:
        raise ValueError("initial_state must be a 4D tensor.")
    B = ms[0]
    HV, V, K = hs[1:]
    if min(HV, V, K) <= 0:
        raise ValueError("State head and feature dimensions must be positive.")
    if ash != (B, HV) or bsh != (B, HV) or ids != (B,):
        raise ValueError("Mismatched batch or head dimensions.")
    if als != (HV,) or dts != (HV,):
        raise ValueError("A_log and dt_bias must have HV elements.")
    if outs != (B, 1, HV, V):
        raise ValueError("out must have shape (B, 1, HV, V).")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous.")
    sm, sa, sb, sh = mixed_qkv.stride(), a.stride(), b.stride(), initial_state.stride()
    if B and ms[1] and sm[1] != 1:
        raise ValueError("mixed_qkv must be contiguous in the last dim.")
    if B and (sa[1] != 1 or sb[1] != 1):
        raise ValueError("a/b must be contiguous in the last dim.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("A_log/dt_bias must be contiguous.")
    if sh[3] != 1 or sh[2] != K or sh[1] != V * K or sh[0] < HV * V * K:
        raise ValueError("State must have dense HV/V/K dimensions and disjoint rows.")
    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")
    qk_dim = ms[1] - HV * V
    if qk_dim <= 0 or qk_dim % (2 * K):
        raise ValueError("Packed Q/K width must be a positive multiple of 2*K.")
    H = qk_dim // (2 * K)
    if HV % H:
        raise ValueError("HV must be divisible by H.")

    if B == 0:
        return out, initial_state

    if ssm_state_indices.dtype == torch.int64:
        # torch_gcu silently narrows int64 tensors to int32 storage while the
        # metadata still reports int64 (unless TORCH_GCU_ENABLE_INT64_AND_UINT64
        # is set), so kernel-side int64-width loads read garbage slots and
        # drive out-of-bounds state updates. Pool slots always fit int32.
        ssm_state_indices = ssm_state_indices.to(torch.int32)

    BK = _next_pow2(K)
    if BK < K:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = _pick_bv(B, HV, V, BK)

    NV = _cdiv(V, BV)
    total = NV * B * HV
    grid = _decode_grid(total)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        initial_state,
        initial_state,
        ssm_state_indices,
        scale,
        sm[0],
        sa[0],
        sb[0],
        sh[0],
        sh[0],
        ssm_state_indices.stride(0),
        grid[0],
        grid[0] * grid[1],
        total,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        NV,
        (V % BV) == 0,
        BK == K,
        _IS_ASCEND,
        20.0,
        use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=1,
    )
    return out, initial_state


def _packed_decode_gpu(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Read each shape/stride tuple once. These checks also run on warm calls;
    # metadata is not cached by tensor identity and may change between calls.
    ms, ash, bsh = mixed_qkv.shape, a.shape, b.shape
    als, dts = A_log.shape, dt_bias.shape
    hs, outs, ids = initial_state.shape, out.shape, ssm_state_indices.shape
    float_types = (torch.float16, torch.bfloat16, torch.float32)
    if any(
        t.dtype not in float_types
        for t in (mixed_qkv, a, b, A_log, dt_bias, initial_state, out)
    ):
        raise NotImplementedError("Packed decode supports FP16, BF16 and FP32.")
    if ssm_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("State indices must be int32 or int64.")
    if len(ms) != 2 or len(ash) != 2 or len(bsh) != 2:
        raise ValueError("mixed_qkv, a and b must be 2D tensors.")
    if len(als) != 1 or len(dts) != 1 or len(ids) != 1:
        raise ValueError("A_log, dt_bias and state indices must be 1D tensors.")
    if len(hs) != 4:
        raise ValueError("initial_state must be a 4D tensor.")
    B = ms[0]
    HV, V, K = hs[1:]
    if min(HV, V, K) <= 0:
        raise ValueError("State head and feature dimensions must be positive.")
    if ash != (B, HV) or bsh != (B, HV) or ids != (B,):
        raise ValueError("Mismatched batch or head dimensions.")
    if als != (HV,) or dts != (HV,):
        raise ValueError("A_log and dt_bias must have HV elements.")
    if outs != (B, 1, HV, V):
        raise ValueError("out must have shape (B, 1, HV, V).")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous.")
    sm, sa, sb, sh = mixed_qkv.stride(), a.stride(), b.stride(), initial_state.stride()
    if B and ms[1] and sm[1] != 1:
        raise ValueError("mixed_qkv must be contiguous in the last dim.")
    if B and (sa[1] != 1 or sb[1] != 1):
        raise ValueError("a/b must be contiguous in the last dim.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("A_log/dt_bias must be contiguous.")
    if sh[3] != 1 or sh[2] != K or sh[1] != V * K or sh[0] < HV * V * K:
        raise ValueError("State must have dense HV/V/K dimensions and disjoint rows.")
    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")
    qk_dim = ms[1] - HV * V
    if qk_dim <= 0 or qk_dim % (2 * K):
        raise ValueError("Packed Q/K width must be a positive multiple of 2*K.")
    H = qk_dim // (2 * K)
    if HV % H:
        raise ValueError("HV must be divisible by H.")
    if B == 0:
        return out, initial_state
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    total = NV * B * HV
    grid = _decode_grid(total)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        initial_state,
        initial_state,
        ssm_state_indices,
        scale,
        sm[0],
        sa[0],
        sb[0],
        sh[0],
        sh[0],
        ssm_state_indices.stride(0),
        grid[0],
        grid[0] * grid[1],
        total,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        NV,
        (V % BV) == 0,
        BK == K,
        False,
        20.0,
        use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=1,
    )
    return out, initial_state


if device.vendor_name == "enflame":
    # The generic implementation's 2D state tile could not be compiled by the
    # 2026-11-07 enflame toolchain (memref_ext.slice_* operandSegmentSizes
    # mismatch, see compiler_issues/vllm_moe_sum.md), and the generic sources
    # are outside this overlay and must stay untouched. Until the toolchain is
    # verified fixed for every shape arm, redirect the generic entry
    # (``flaggems_vllm.ops`` / ``flaggems_vllm.ops.FLA``, used by tests and
    # benchmarks) to this implementation — the same "vendor implementation
    # under the generic name" pattern the generic ops package itself uses for
    # other vendors. The production entry (``flaggems_vllm.<op>``) already
    # resolves here via SpecOpRegistrar and is unchanged.
    import flaggems_vllm.ops as _generic_ops_pkg
    import flaggems_vllm.ops.FLA as _generic_fla_pkg

    _generic_ops_pkg.fused_recurrent_gated_delta_rule_packed_decode = (
        fused_recurrent_gated_delta_rule_packed_decode
    )
    _generic_fla_pkg.fused_recurrent_gated_delta_rule_packed_decode = (
        fused_recurrent_gated_delta_rule_packed_decode
    )
