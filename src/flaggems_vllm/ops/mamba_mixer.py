# SPDX-License-Identifier: Apache-2.0
"""Correctness-first Triton implementation of the vLLM Mamba-1 mixer.

Scope of this first implementation:
  * Mamba-1 only (not Mamba-2).
  * TP=1, no LoRA, no Mamba cache-all/APC, no chunked prefill.
  * Packed variable-length prefill via query_start_loc, or decode with one
    token per sequence (is_decode=True). Mixed prefill/decode batches are not
    supported by this first version.
  * FP16/BF16/FP32 activations and weights; FP32 SSM math/state accumulation.
  * Strided vLLM Conv/SSM cache views are supported; cache slot indices must
    be passed explicitly when a cache is supplied and NULL_BLOCK_ID is 0.

The host function is a FlagGems-vLLM operator API, not a drop-in replacement
for torch.ops.vllm.mamba_mixer. The vLLM plugin must pass the layer weights,
cache tensors, and metadata explicitly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


_LINEAR_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
]
# Stateful kernels below deliberately use fixed BLOCK_I values instead of
# runtime autotuning. Autotuning replays a kernel on its arguments and would
# mutate the caller's Conv/SSM cache multiple times.
_CONV_BLOCK_I = 16
_SCAN_BLOCK_I = 8


@libentry()
@libtuner(configs=_LINEAR_CONFIGS, key=["M", "N", "K"], strategy=["log", "log", "log"])
@triton.jit
def _mamba_linear_kernel(
    X,
    W,
    BIAS,
    Y,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Y = X @ W.T + bias, with W stored as [N, K]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        x = tl.load(
            X + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            W + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        if X.dtype.element_ty == tl.float32:
            # H20 (SM90): tf32x3 preserves FP32 GEMM accuracy much better
            # than the default tf32 mode. Do not round activations to BF16.
            acc += tl.dot(x, w, out_dtype=tl.float32,
                          input_precision="tf32x3")
        else:
            acc += tl.dot(x, w, out_dtype=tl.float32)

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    tl.store(
        Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(Y.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@libentry()
@triton.jit
def _mamba_conv1d_varlen_kernel(
    PROJ,
    WEIGHT,
    BIAS,
    CONV_STATE,
    QUERY_START_LOC,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    CONV_OUT,
    total_tokens,
    intermediate_size,
    num_sequences,
    stride_w_i,
    stride_w_k,
    stride_state_slot,
    stride_state_i,
    stride_state_k,
    HAS_BIAS: tl.constexpr,
    HAS_CONV_STATE: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    APPLY_SILU: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Depthwise causal Conv1D over packed sequences; updates conv cache."""
    seq_id = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    channel_mask = channel < intermediate_size

    if IS_DECODE:
        seq_start = seq_id
        seq_end = seq_id + 1
    elif HAS_QUERY_START_LOC:
        seq_start = tl.load(QUERY_START_LOC + seq_id).to(tl.int32)
        seq_end = tl.load(QUERY_START_LOC + seq_id + 1).to(tl.int32)
    else:
        seq_start = 0
        seq_end = total_tokens

    slot = seq_id
    if HAS_CACHE_INDICES:
        slot = tl.load(CACHE_INDICES + seq_id).to(tl.int32)
    # vLLM uses slot 0 as NULL_BLOCK_ID; negative padded indices are
    # invalid as well and must never be dereferenced.
    slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)

    init_state = INIT_ALL_TRUE
    if HAS_INITIAL_STATE_PTR:
        init_state = tl.load(HAS_INITIAL_STATE + seq_id).to(tl.int1)

    for token in range(seq_start, seq_end):
        acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)

        for tap in tl.static_range(KERNEL_SIZE):
            source_token = token - (KERNEL_SIZE - 1 - tap)
            if source_token >= seq_start:
                value = tl.load(
                    PROJ + source_token * (2 * intermediate_size) + channel,
                    mask=channel_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                value = tl.full((BLOCK_I,), 0.0, dtype=tl.float32)
                if HAS_CONV_STATE:
                    if slot_valid and init_state:
                        history_idx = source_token - seq_start + KERNEL_SIZE - 1
                        value = tl.load(
                            CONV_STATE
                            + slot * stride_state_slot
                            + channel * stride_state_i
                            + history_idx * stride_state_k,
                            mask=channel_mask,
                            other=0.0,
                        ).to(tl.float32)

            weight = tl.load(
                WEIGHT + channel * stride_w_i + tap * stride_w_k,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            acc += value * weight

        if APPLY_SILU:
            acc = acc / (1.0 + tl.exp(-acc))

        tl.store(
            CONV_OUT + token * intermediate_size + channel,
            acc.to(CONV_OUT.dtype.element_ty),
            mask=channel_mask,
        )

    # Keep the last K-1 *input* samples, not the convolved outputs.
    if HAS_CONV_STATE:
        if slot_valid and (seq_end > seq_start):
            for j in tl.static_range(KERNEL_SIZE - 1):
                source_token = seq_end - (KERNEL_SIZE - 1) + j
                if source_token >= seq_start:
                    # Keep branch values in FP32 so Triton can merge the
                    # source-token and history-state paths for FP16/BF16.
                    value = tl.load(
                        PROJ + source_token * (2 * intermediate_size) + channel,
                        mask=channel_mask,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    value = tl.full((BLOCK_I,), 0.0, dtype=tl.float32)
                    if init_state:
                        history_idx = source_token - seq_start + KERNEL_SIZE - 1
                        value = tl.load(
                            CONV_STATE
                            + slot * stride_state_slot
                            + channel * stride_state_i
                            + history_idx * stride_state_k,
                            mask=channel_mask,
                            other=0.0,
                        ).to(tl.float32)
                tl.store(
                    CONV_STATE
                    + slot * stride_state_slot
                    + channel * stride_state_i
                    + j * stride_state_k,
                    value.to(CONV_STATE.dtype.element_ty),
                    mask=channel_mask,
                )


@libentry()
@triton.jit
def _mamba_rms_norm_kernel(
    X,
    WEIGHT,
    rows,
    width,
    offset,
    stride_row,
    stride_col,
    eps,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In-place RMSNorm on one contiguous segment of each row."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < width
    ptr = X + row * stride_row + (offset + col) * stride_col
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
    if HAS_WEIGHT:
        gamma = tl.load(WEIGHT + col, mask=mask, other=1.0).to(tl.float32)
    else:
        gamma = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    tl.store(ptr, (x * inv_rms * gamma).to(X.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _mamba_selective_scan_varlen_kernel(
    PROJ,
    CONV_OUT,
    SSM_PARAMS,
    DT,
    A,
    D,
    DT_BIAS,
    SSM_STATE,
    QUERY_START_LOC,
    CACHE_INDICES,
    HAS_INITIAL_STATE,
    SCAN_OUT,
    total_tokens,
    intermediate_size,
    num_sequences,
    time_step_rank,
    params_width,
    stride_a_i,
    stride_a_s,
    stride_state_slot,
    stride_state_i,
    stride_state_s,
    HAS_DT_BIAS: tl.constexpr,
    HAS_SSM_STATE: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    HAS_INITIAL_STATE_PTR: tl.constexpr,
    INIT_ALL_TRUE: tl.constexpr,
    IS_DECODE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Sequential recurrence per (sequence, channel tile), correctness path."""
    seq_id = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
    state_idx = tl.arange(0, BLOCK_S)
    channel_mask = channel < intermediate_size
    state_mask = state_idx < STATE_SIZE

    if IS_DECODE:
        seq_start = seq_id
        seq_end = seq_id + 1
    elif HAS_QUERY_START_LOC:
        seq_start = tl.load(QUERY_START_LOC + seq_id).to(tl.int32)
        seq_end = tl.load(QUERY_START_LOC + seq_id + 1).to(tl.int32)
    else:
        seq_start = 0
        seq_end = total_tokens

    slot = seq_id
    if HAS_CACHE_INDICES:
        slot = tl.load(CACHE_INDICES + seq_id).to(tl.int32)
    # vLLM uses slot 0 as NULL_BLOCK_ID; negative padded indices are
    # invalid as well and must never be dereferenced.
    slot_valid = (slot != NULL_BLOCK_ID) & (slot >= 0)

    init_state = INIT_ALL_TRUE
    if HAS_INITIAL_STATE_PTR:
        init_state = tl.load(HAS_INITIAL_STATE + seq_id).to(tl.int1)

    state = tl.zeros((BLOCK_I, BLOCK_S), dtype=tl.float32)
    if HAS_SSM_STATE:
        if slot_valid and init_state:
            state = tl.load(
                SSM_STATE
                + slot * stride_state_slot
                + channel[:, None] * stride_state_i
                + state_idx[None, :] * stride_state_s,
                mask=channel_mask[:, None] & state_mask[None, :],
                other=0.0,
            ).to(tl.float32)

    a = tl.load(
        A + channel[:, None] * stride_a_i + state_idx[None, :] * stride_a_s,
        mask=channel_mask[:, None] & state_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    d = tl.load(D + channel, mask=channel_mask, other=0.0).to(tl.float32)

    for token in range(seq_start, seq_end):
        u = tl.load(CONV_OUT + token * intermediate_size + channel, mask=channel_mask, other=0.0).to(tl.float32)
        dt = tl.load(DT + token * intermediate_size + channel, mask=channel_mask, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(DT_BIAS + channel, mask=channel_mask, other=0.0).to(tl.float32)
        # Stable softplus used by Mamba's delta discretization.
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)

        b = tl.load(
            SSM_PARAMS + token * params_width + time_step_rank + state_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
        c = tl.load(
            SSM_PARAMS + token * params_width + time_step_rank + STATE_SIZE + state_idx,
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)

        d_a = tl.exp(dt[:, None] * a)
        d_b = dt[:, None] * b[None, :]
        state = state * d_a + d_b * u[:, None]
        y = tl.sum(state * c[None, :], axis=1) + d * u

        gate = tl.load(
            PROJ + token * (2 * intermediate_size) + intermediate_size + channel,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        y = y * (gate / (1.0 + tl.exp(-gate)))
        tl.store(
            SCAN_OUT + token * intermediate_size + channel,
            y.to(SCAN_OUT.dtype.element_ty),
            mask=channel_mask,
        )

    if HAS_SSM_STATE:
        if slot_valid and (seq_end > seq_start):
            tl.store(
                SSM_STATE
                + slot * stride_state_slot
                + channel[:, None] * stride_state_i
                + state_idx[None, :] * stride_state_s,
                state.to(SSM_STATE.dtype.element_ty),
                mask=channel_mask[:, None] & state_mask[None, :],
            )


def _check_contiguous(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous in this first implementation")


def _linear(x, weight, bias, out):
    m, k = x.shape
    n = weight.shape[0]
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _mamba_linear_kernel[grid](
        x,
        weight,
        bias,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
    )


def mamba_mixer(
    hidden_states: torch.Tensor,
    in_proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    out_proj_weight: torch.Tensor,
    *,
    in_proj_bias: torch.Tensor | None = None,
    conv1d_bias: torch.Tensor | None = None,
    dt_proj_bias: torch.Tensor | None = None,
    out_proj_bias: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    ssm_state: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    dt_norm_weight: torch.Tensor | None = None,
    b_norm_weight: torch.Tensor | None = None,
    c_norm_weight: torch.Tensor | None = None,
    use_rms_norm: bool = False,
    rms_norm_eps: float = 1e-5,
    activation: str | None = "silu",
    is_decode: bool = False,
    output: torch.Tensor | None = None,
    num_actual_tokens: int | None = None,
    null_block_id: int = 0,
) -> torch.Tensor:
    """Run the Mamba-1 mixer with Triton kernels.

    Args:
        hidden_states: [total_tokens, hidden_size].
        in_proj_weight: [2 * intermediate_size, hidden_size].
        conv1d_weight: [intermediate_size, kernel_size] or
            [intermediate_size, 1, kernel_size].
        x_proj_weight: [time_step_rank + 2 * state_size, intermediate_size].
        dt_proj_weight: [intermediate_size, time_step_rank]. Bias is applied
            inside selective scan, matching vLLM's skip_bias_add path.
        A: already transformed as -exp(A_log), shape [intermediate_size, state_size].
        D: [intermediate_size].
        out_proj_weight: [hidden_size, intermediate_size].
        conv_state: optional [num_cache_slots, intermediate_size, kernel_size - 1].
            Arbitrary strides are supported (including vLLM's transposed view).
        ssm_state: optional [num_cache_slots, intermediate_size, state_size].
            Arbitrary strides are supported.
        query_start_loc: [num_sequences + 1] for packed variable-length prefill.
            If omitted for prefill, all actual tokens are treated as one sequence.
        cache_indices: [num_sequences], maps each sequence to a cache slot.
            Required when a cache is supplied with vLLM's NULL_BLOCK_ID=0.
            Negative indices and NULL_BLOCK_ID are skipped.
        has_initial_state: optional bool [num_sequences]. For prefill, omitted
            means no prior state (zero initial state). For decode, omitted means
            the referenced cache slots already contain valid state. Pass the
            vLLM metadata flag explicitly for chunked/continued prefill.
        is_decode: when True, each actual row of hidden_states is one independent
            single-token sequence; query_start_loc must be omitted.
        output: optional output buffer with shape [>=num_actual_tokens, H].
            Only the first num_actual_tokens rows are written.
        num_actual_tokens: optional valid-token count. If omitted, all rows in
            hidden_states are treated as actual tokens. When supplied, packed
            query_start_loc must end at this count.
        null_block_id: cache slot reserved as Null Block; vLLM uses 0.

    Returns:
        Output tensor. Cache tensors are updated in-place when supplied.

    Notes:
        This is a forward-only inference operator. It deliberately raises on
        unsupported layouts/dtypes rather than using a PyTorch compute fallback.
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [total_tokens, hidden_size]")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "mamba_mixer supports FP16, BF16 and FP32 activations only"
        )
    if activation not in (None, "silu", "swish", True):
        raise NotImplementedError(f"unsupported activation: {activation}")
    if is_decode and query_start_loc is not None:
        raise ValueError("query_start_loc must be omitted when is_decode=True")

    padded_tokens, hidden_size = hidden_states.shape
    if num_actual_tokens is None:
        total_tokens = padded_tokens
    else:
        if isinstance(num_actual_tokens, bool) or not isinstance(num_actual_tokens, int):
            raise TypeError("num_actual_tokens must be an int or None")
        if not 0 <= num_actual_tokens <= padded_tokens:
            raise ValueError(
                f"num_actual_tokens must be in [0, {padded_tokens}], "
                f"got {num_actual_tokens}"
            )
        total_tokens = num_actual_tokens

    if output is not None:
        if output.ndim != 2 or output.shape[1] != hidden_size:
            raise ValueError("output must have shape [>=num_actual_tokens, hidden_size]")
        if output.shape[0] < total_tokens:
            raise ValueError("output has fewer rows than num_actual_tokens")
        if output.dtype != hidden_states.dtype:
            raise ValueError("output dtype must match hidden_states dtype")
        if output.device != hidden_states.device:
            raise ValueError("output and hidden_states must be on the same device")

    if total_tokens == 0:
        if output is not None:
            return output
        return torch.empty((0, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)

    # Do not let padded rows enter any projection or final output GEMM.
    hidden_states = hidden_states[:total_tokens]

    if in_proj_weight.ndim != 2 or in_proj_weight.shape[1] != hidden_size or in_proj_weight.shape[0] % 2:
        raise ValueError("in_proj_weight must have shape [2 * intermediate_size, hidden_size]")
    intermediate_size = in_proj_weight.shape[0] // 2

    if conv1d_weight.ndim == 3:
        if conv1d_weight.shape[1] != 1:
            raise ValueError("3D conv1d_weight must have shape [I, 1, K]")
        kernel_size = conv1d_weight.shape[2]
        conv_w_stride_i = conv1d_weight.stride(0)
        conv_w_stride_k = conv1d_weight.stride(2)
    elif conv1d_weight.ndim == 2:
        kernel_size = conv1d_weight.shape[1]
        conv_w_stride_i = conv1d_weight.stride(0)
        conv_w_stride_k = conv1d_weight.stride(1)
    else:
        raise ValueError("conv1d_weight must be [I, K] or [I, 1, K]")
    if conv1d_weight.shape[0] != intermediate_size:
        raise ValueError("conv1d_weight dimensions do not match intermediate_size")
    if not 2 <= kernel_size <= 7:
        raise NotImplementedError(
            "vLLM-compatible causal Conv1D currently requires kernel_size in [2, 7]"
        )

    if x_proj_weight.ndim != 2 or x_proj_weight.shape[1] != intermediate_size:
        raise ValueError("x_proj_weight must have shape [R + 2*S, intermediate_size]")
    if dt_proj_weight.ndim != 2 or dt_proj_weight.shape[0] != intermediate_size:
        raise ValueError("dt_proj_weight must have shape [intermediate_size, R]")
    time_step_rank = dt_proj_weight.shape[1]
    if (x_proj_weight.shape[0] - time_step_rank) % 2:
        raise ValueError("x_proj_weight output dimension must equal R + 2*S")
    state_size = (x_proj_weight.shape[0] - time_step_rank) // 2
    if state_size < 1 or state_size > 64:
        raise NotImplementedError("state_size must be in [1, 64] for this first Triton path")
    if A.shape != (intermediate_size, state_size) or D.shape != (intermediate_size,):
        raise ValueError("A or D shape does not match intermediate_size/state_size")
    if out_proj_weight.shape != (hidden_size, intermediate_size):
        raise ValueError("out_proj_weight must have shape [hidden_size, intermediate_size]")
    optional_shapes = (
        ("in_proj_bias", in_proj_bias, (2 * intermediate_size,)),
        ("conv1d_bias", conv1d_bias, (intermediate_size,)),
        ("dt_proj_bias", dt_proj_bias, (intermediate_size,)),
        ("out_proj_bias", out_proj_bias, (hidden_size,)),
    )
    for name, tensor, expected_shape in optional_shapes:
        if tensor is not None and tensor.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
    if A.dtype != torch.float32 or D.dtype != torch.float32:
        raise ValueError("A and D must be FP32")
    if dt_proj_bias is not None and dt_proj_bias.dtype != torch.float32:
        raise ValueError("dt_proj_bias must be FP32, matching vLLM's scan bias path")

    if is_decode:
        num_sequences = total_tokens
    elif query_start_loc is None:
        num_sequences = 1
    else:
        if query_start_loc.ndim != 1 or query_start_loc.dtype != torch.int32:
            raise ValueError("query_start_loc must be a 1D int32 tensor")
        num_sequences = query_start_loc.shape[0] - 1
        if num_sequences < 1:
            raise ValueError("query_start_loc must contain at least two entries")
        # The caller must guarantee query_start_loc[0] == 0,
        # query_start_loc[-1] == total_tokens, and nondecreasing offsets.

    has_cache = conv_state is not None or ssm_state is not None
    if (conv_state is None) != (ssm_state is None):
        raise ValueError("conv_state and ssm_state must either both be provided or both be omitted")
    if has_cache:
        if conv_state.ndim != 3 or conv_state.shape[1:] != (intermediate_size, kernel_size - 1):
            raise ValueError("conv_state must have shape [cache_slots, I, K-1]")
        if ssm_state.ndim != 3 or ssm_state.shape[1:] != (intermediate_size, state_size):
            raise ValueError("ssm_state must have shape [cache_slots, I, S]")
        if conv_state.shape[0] != ssm_state.shape[0]:
            raise ValueError("conv_state and ssm_state must have the same number of cache slots")
        allowed_state_dtypes = (torch.float16, torch.bfloat16, torch.float32)
        if conv_state.dtype not in allowed_state_dtypes or ssm_state.dtype not in allowed_state_dtypes:
            raise NotImplementedError("conv_state and ssm_state must be FP16, BF16, or FP32")
        # vLLM's causal-conv path casts its input to the Conv cache dtype.
        # This implementation supports same-dtype caches and FP32 caches;
        # cross-casting FP16 <-> BF16 is deliberately rejected.
        if conv_state.dtype not in (hidden_states.dtype, torch.float32):
            raise NotImplementedError(
                "conv_state dtype must match hidden_states or be FP32"
            )
        if cache_indices is None and null_block_id == 0:
            raise ValueError(
                "cache_indices must be provided when using vLLM's NULL_BLOCK_ID=0"
            )
    elif cache_indices is not None or has_initial_state is not None:
        raise ValueError("cache_indices/has_initial_state require conv_state and ssm_state")

    if cache_indices is not None and cache_indices.shape != (num_sequences,):
        raise ValueError("cache_indices must have shape [num_sequences]")
    if cache_indices is not None and cache_indices.dtype != torch.int32:
        raise ValueError("cache_indices must be int32")
    if has_initial_state is not None and has_initial_state.shape != (num_sequences,):
        raise ValueError("has_initial_state must have shape [num_sequences]")
    if has_initial_state is not None and has_initial_state.dtype != torch.bool:
        raise ValueError("has_initial_state must be bool")

    # Model tensors and metadata are contiguous in this first version.
    # Cache tensors are intentionally excluded: their actual strides are passed
    # to the Triton kernels so vLLM's transposed Conv State view remains writable.
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("in_proj_weight", in_proj_weight),
        ("conv1d_weight", conv1d_weight),
        ("x_proj_weight", x_proj_weight),
        ("dt_proj_weight", dt_proj_weight),
        ("A", A),
        ("D", D),
        ("out_proj_weight", out_proj_weight),
        ("in_proj_bias", in_proj_bias),
        ("conv1d_bias", conv1d_bias),
        ("dt_proj_bias", dt_proj_bias),
        ("out_proj_bias", out_proj_bias),
        ("query_start_loc", query_start_loc),
        ("cache_indices", cache_indices),
        ("has_initial_state", has_initial_state),
        ("dt_norm_weight", dt_norm_weight),
        ("b_norm_weight", b_norm_weight),
        ("c_norm_weight", c_norm_weight),
    ):
        _check_contiguous(name, tensor)

    for name, tensor in (
        ("in_proj_weight", in_proj_weight),
        ("conv1d_weight", conv1d_weight),
        ("x_proj_weight", x_proj_weight),
        ("dt_proj_weight", dt_proj_weight),
        ("A", A),
        ("D", D),
        ("out_proj_weight", out_proj_weight),
        ("in_proj_bias", in_proj_bias),
        ("conv1d_bias", conv1d_bias),
        ("dt_proj_bias", dt_proj_bias),
        ("out_proj_bias", out_proj_bias),
        ("conv_state", conv_state),
        ("ssm_state", ssm_state),
        ("query_start_loc", query_start_loc),
        ("cache_indices", cache_indices),
        ("has_initial_state", has_initial_state),
        ("dt_norm_weight", dt_norm_weight),
        ("b_norm_weight", b_norm_weight),
        ("c_norm_weight", c_norm_weight),
    ):
        if tensor is not None and tensor.device != hidden_states.device:
            raise ValueError(f"{name} and hidden_states must be on the same device")

    if in_proj_weight.dtype != hidden_states.dtype or conv1d_weight.dtype != hidden_states.dtype:
        raise ValueError("in_proj_weight and conv1d_weight must match hidden_states dtype")
    if x_proj_weight.dtype != hidden_states.dtype or dt_proj_weight.dtype != hidden_states.dtype:
        raise ValueError("x_proj_weight and dt_proj_weight must match hidden_states dtype")
    if out_proj_weight.dtype != hidden_states.dtype:
        raise ValueError("out_proj_weight must match hidden_states dtype")
    if use_rms_norm and any(w is not None and w.numel() != width for w, width in (
        (dt_norm_weight, time_step_rank), (b_norm_weight, state_size), (c_norm_weight, state_size)
    )):
        raise ValueError("RMSNorm weight has an incorrect width")
    if not use_rms_norm and any(w is not None for w in (dt_norm_weight, b_norm_weight, c_norm_weight)):
        raise ValueError("norm weights were supplied but use_rms_norm=False")

    projected = torch.empty((total_tokens, 2 * intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
    conv_out = torch.empty((total_tokens, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
    params_width = time_step_rank + 2 * state_size
    ssm_params = torch.empty((total_tokens, params_width), device=hidden_states.device, dtype=hidden_states.dtype)
    dt = torch.empty((total_tokens, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
    scan_out = torch.empty((total_tokens, intermediate_size), device=hidden_states.device, dtype=hidden_states.dtype)
    if output is None:
        output = torch.empty(
            (total_tokens, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    has_qsl = query_start_loc is not None
    has_cache_indices = cache_indices is not None
    has_init_ptr = has_initial_state is not None
    # A new prefill starts from zero state unless metadata explicitly says
    # that a chunk continues an existing sequence. Decode normally consumes
    # the state produced by a previous prefill/decode step.
    init_all_true = has_cache and is_decode and has_initial_state is None
    apply_silu = activation in ("silu", "swish", True)
    block_s = triton.next_power_of_2(state_size)

    with torch_device_fn.device(hidden_states.device):
        _linear(hidden_states, in_proj_weight, in_proj_bias, projected)

        conv_grid = (
            num_sequences,
            triton.cdiv(intermediate_size, _CONV_BLOCK_I),
        )
        _mamba_conv1d_varlen_kernel[conv_grid](
            projected,
            conv1d_weight,
            conv1d_bias,
            conv_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            conv_out,
            total_tokens,
            intermediate_size,
            num_sequences,
            conv_w_stride_i,
            conv_w_stride_k,
            conv_state.stride(0) if has_cache else 0,
            conv_state.stride(1) if has_cache else 0,
            conv_state.stride(2) if has_cache else 0,
            HAS_BIAS=conv1d_bias is not None,
            HAS_CONV_STATE=has_cache,
            HAS_QUERY_START_LOC=has_qsl,
            HAS_CACHE_INDICES=has_cache_indices,
            HAS_INITIAL_STATE_PTR=has_init_ptr,
            INIT_ALL_TRUE=init_all_true,
            IS_DECODE=is_decode,
            APPLY_SILU=apply_silu,
            NULL_BLOCK_ID=null_block_id,
            KERNEL_SIZE=kernel_size,
            BLOCK_I=_CONV_BLOCK_I,
        )

        _linear(conv_out, x_proj_weight, None, ssm_params)

        if use_rms_norm:
            _mamba_rms_norm_kernel[(total_tokens,)](
                ssm_params, dt_norm_weight, total_tokens, time_step_rank, 0,
                ssm_params.stride(0), ssm_params.stride(1), rms_norm_eps,
                HAS_WEIGHT=dt_norm_weight is not None,
                BLOCK=triton.next_power_of_2(time_step_rank),
            )
            _mamba_rms_norm_kernel[(total_tokens,)](
                ssm_params, b_norm_weight, total_tokens, state_size, time_step_rank,
                ssm_params.stride(0), ssm_params.stride(1), rms_norm_eps,
                HAS_WEIGHT=b_norm_weight is not None,
                BLOCK=block_s,
            )
            _mamba_rms_norm_kernel[(total_tokens,)](
                ssm_params, c_norm_weight, total_tokens, state_size, time_step_rank + state_size,
                ssm_params.stride(0), ssm_params.stride(1), rms_norm_eps,
                HAS_WEIGHT=c_norm_weight is not None,
                BLOCK=block_s,
            )

        dt_lowrank = ssm_params[:, :time_step_rank]
        _linear(dt_lowrank, dt_proj_weight, None, dt)

        scan_grid = (
            num_sequences,
            triton.cdiv(intermediate_size, _SCAN_BLOCK_I),
        )
        _mamba_selective_scan_varlen_kernel[scan_grid](
            projected,
            conv_out,
            ssm_params,
            dt,
            A,
            D,
            dt_proj_bias,
            ssm_state,
            query_start_loc,
            cache_indices,
            has_initial_state,
            scan_out,
            total_tokens,
            intermediate_size,
            num_sequences,
            time_step_rank,
            params_width,
            A.stride(0),
            A.stride(1),
            ssm_state.stride(0) if has_cache else 0,
            ssm_state.stride(1) if has_cache else 0,
            ssm_state.stride(2) if has_cache else 0,
            HAS_DT_BIAS=dt_proj_bias is not None,
            HAS_SSM_STATE=has_cache,
            HAS_QUERY_START_LOC=has_qsl,
            HAS_CACHE_INDICES=has_cache_indices,
            HAS_INITIAL_STATE_PTR=has_init_ptr,
            INIT_ALL_TRUE=init_all_true,
            IS_DECODE=is_decode,
            NULL_BLOCK_ID=null_block_id,
            STATE_SIZE=state_size,
            BLOCK_S=block_s,
            BLOCK_I=_SCAN_BLOCK_I,
        )

        _linear(scan_out, out_proj_weight, out_proj_bias, output)

    return output


__all__ = ["mamba_mixer"]

