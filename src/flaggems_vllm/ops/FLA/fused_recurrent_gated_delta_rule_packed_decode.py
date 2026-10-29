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
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PRESERVE_BETA_ROUNDING: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1) + tl.program_id(2) * HV
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    # Skip if state index is invalid (NULL_BLOCK_ID=0)
    if state_idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

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

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


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
    float_types = (torch.float16, torch.bfloat16, torch.float32)
    tensors = (mixed_qkv, a, b, A_log, dt_bias, initial_state, out)
    if any(t.dtype not in float_types for t in tensors):
        raise NotImplementedError("Packed decode supports FP16, BF16 and FP32.")
    if ssm_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("State indices must be int32 or int64.")
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.numel() and mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.numel() and (a.stride(-1) != 1 or b.stride(-1) != 1):
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

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

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if min(HV, V, K) <= 0:
        raise ValueError("State head and feature dimensions must be positive.")
    if (
        initial_state.stride(1) != V * K
        or initial_state.stride(2) != K
        or initial_state.stride(0) < HV * V * K
    ):
        raise ValueError("State must have dense HV/V/K dimensions and disjoint rows.")
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements "
            f"(got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )

    if B == 0:
        return out, initial_state

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 1
    num_warps = 1

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    grid = (NV, HV, B) if B * HV > 65535 else (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        PRESERVE_BETA_ROUNDING=device.vendor_name == "ascend",
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
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
    grid = (NV, HV, B) if B * HV > 65535 else (NV, B * HV)
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
        H,
        HV,
        K,
        V,
        BK,
        BV,
        False,
        20.0,
        use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=1,
    )
    return out, initial_state
