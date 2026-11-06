# SPDX-License-Identifier: Apache-2.0
"""Stateful depthwise causal convolution for regular continuous batching."""
import torch
import triton
import triton.language as tl

from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def _conv_kernel(
    X,
    W,
    BIAS,
    STATE,
    CU,
    SLOTS,
    INITIAL,
    OUT,
    D: tl.constexpr,
    T: tl.constexpr,
    WIDTH: tl.constexpr,
    HISTORY: tl.constexpr,
    SX_REQ: tl.constexpr,
    SX_D: tl.constexpr,
    SX_T: tl.constexpr,
    SO_REQ: tl.constexpr,
    SO_D: tl.constexpr,
    SO_T: tl.constexpr,
    SS_REQ: tl.constexpr,
    SS_D: tl.constexpr,
    SS_T: tl.constexpr,
    SW_D: tl.constexpr,
    SW_T: tl.constexpr,
    VARLEN: tl.constexpr,
    HAS_SLOTS: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    PAD: tl.constexpr,
    NULL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BW: tl.constexpr,
):
    req = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    j = tl.arange(0, BW)
    start = tl.load(CU + req) if VARLEN else 0
    end = tl.load(CU + req + 1) if VARLEN else T
    if start == end:
        return
    slot = tl.load(SLOTS + req) if HAS_SLOTS else req
    valid = (slot >= 0) & (slot != PAD) & (slot != NULL)
    use_state = tl.load(INITIAL + req) if HAS_INITIAL else True
    st_base = STATE + slot * SS_REQ + channel[:, None] * SS_D
    history = tl.load(
        st_base + j[None, :] * SS_T,
        (channel[:, None] < D) & (j[None, :] < WIDTH - 1) & valid & use_state,
        other=0,
    ).to(tl.float32)
    weight = tl.load(
        W + channel[:, None] * SW_D + j[None, :] * SW_T,
        (channel[:, None] < D) & (j[None, :] < WIDTH),
        other=0,
    ).to(tl.float32)
    for token in range(start, end):
        value = tl.load(
            X + req * SX_REQ + channel * SX_D + token * SX_T, channel < D, other=0
        )
        value = value.to(STATE.dtype.element_ty).to(tl.float32)
        window = tl.where(j[None, :] == WIDTH - 1, value[:, None], history)
        result = tl.sum(window * weight, 1)
        if HAS_BIAS:
            result += tl.load(BIAS + channel, channel < D, other=0).to(tl.float32)
        if SILU:
            result = result / (1.0 + tl.exp(-result))
        result = result.to(STATE.dtype.element_ty)
        tl.store(
            OUT + req * SO_REQ + channel * SO_D + token * SO_T,
            tl.where(valid, result, 0),
            channel < D,
        )
        nxt = tl.broadcast_to(((j + 1) % BW)[None, :], (BLOCK_D, BW))
        history = tl.gather(window, nxt, 1)
    tl.store(
        st_base + j[None, :] * SS_T,
        history,
        (channel[:, None] < D) & (j[None, :] < WIDTH - 1) & valid,
    )


def _launch(
    x, weight, bias, state, starts, slots, initial, activation, pad, null, mode
):
    if activation not in (True, False, None, "silu", "swish"):
        raise ValueError(f"Unsupported causal-conv activation: {activation}")
    if weight.ndim != 2 or state.ndim != 3 or weight.shape[0] != state.shape[1]:
        raise ValueError(
            "Causal convolution requires [D,width] weights and [slots,D,history] state"
        )
    dim, width = weight.shape
    if width < 2 or width > 8 or state.shape[-1] < width - 1:
        raise NotImplementedError(
            "Causal convolution supports widths 2..8 and sufficient history"
        )
    if x.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ) or state.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise NotImplementedError("Causal convolution supports fp32/fp16/bf16")
    if bias is not None and bias.shape != (dim,):
        raise ValueError("Causal convolution bias shape mismatch")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16) or (
        bias is not None
        and bias.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        raise NotImplementedError(
            "Causal convolution weights/bias support fp32/fp16/bf16"
        )
    if bias is not None and not bias.is_contiguous():
        raise ValueError("Causal convolution bias must be contiguous")
    for name, metadata in (("query starts", starts), ("state slots", slots)):
        if metadata is not None and (
            metadata.ndim != 1
            or metadata.dtype not in (torch.int32, torch.int64)
            or not metadata.is_contiguous()
        ):
            raise ValueError(
                f"Causal convolution {name} must be a contiguous integer vector"
            )
    if initial is not None and (
        initial.dtype != torch.bool or not initial.is_contiguous()
    ):
        raise ValueError(
            "Causal convolution initial-state mask must be contiguous bool"
        )
    if any(
        t is not None and t.device != x.device
        for t in (weight, bias, state, starts, slots, initial)
    ):
        raise ValueError("Causal convolution tensors must share input device")
    varlen = starts is not None
    requests = starts.numel() - 1 if varlen else x.shape[0]
    if slots is not None and (slots.ndim != 1 or slots.numel() != requests):
        raise ValueError("Causal convolution needs one state slot per request")
    if initial is not None and initial.shape != (requests,):
        raise ValueError("Causal convolution initial-state mask shape mismatch")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if mode == "prefill":
        if x.ndim != 2 or x.shape[0] != dim or not varlen:
            raise ValueError("Prefill expects [D,total_tokens] and query_start_loc")
        sx = (0, x.stride(0), x.stride(1))
        so = (0, out.stride(0), out.stride(1))
        tokens = x.shape[1]
    elif varlen:
        if x.ndim != 2 or x.shape[1] != dim:
            raise ValueError("Varlen decode expects [total_tokens,D]")
        sx = (0, x.stride(1), x.stride(0))
        so = (0, out.stride(1), out.stride(0))
        tokens = x.shape[0]
    else:
        if x.ndim != 3 or x.shape[1] != dim:
            raise ValueError("Decode expects [B,D,tokens]")
        sx = x.stride()
        so = out.stride()
        tokens = x.shape[2]
    if requests and dim:
        _conv_kernel[(requests, triton.cdiv(dim, 32))](
            x,
            weight,
            bias,
            state,
            starts,
            slots,
            initial,
            out,
            dim,
            tokens,
            width,
            state.shape[-1],
            *sx,
            *so,
            *state.stride(),
            *weight.stride(),
            varlen,
            slots is not None,
            initial is not None,
            bias is not None,
            activation is True or activation in ("silu", "swish"),
            pad,
            null,
            BLOCK_D=32,
            BW=triton.next_power_of_2(width),
            num_warps=4,
        )
    return out


def causal_conv1d_fn(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
    null_block_id=-1,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    num_computed_tokens=None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    if any(
        v is not None
        for v in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
        )
    ):
        raise NotImplementedError("Causal convolution APC metadata is unsupported")
    return _launch(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        "prefill",
    )


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    null_block_id=-1,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
):
    if any(
        v is not None
        for v in (
            num_accepted_tokens,
            block_idx_last_scheduled_token,
            initial_state_idx,
        )
    ):
        raise NotImplementedError(
            "Causal convolution speculative/APC rotation is unsupported"
        )
    squeeze = x.ndim == 2 and query_start_loc is None
    inp = x.unsqueeze(-1) if squeeze else x
    out = _launch(
        inp,
        weight,
        bias,
        conv_state,
        query_start_loc,
        conv_state_indices,
        None,
        activation,
        -1,
        null_block_id,
        "decode",
    )
    return out.squeeze(-1) if squeeze else out
