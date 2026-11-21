# SPDX-License-Identifier: Apache-2.0
"""Stateful depthwise causal convolution for regular continuous batching.

GCU (enflame) adaptation notes
------------------------------
The generic seed kernel keeps a rolling ``(BLOCK_D, BW)`` window tile in
registers and rotates it every token with ``tl.gather``.  Neither pattern
survives the enflame Triton backend:

* ``tl.gather`` is not implemented (``tt.gather`` fails to legalize and the
  compiler pass manager aborts — see compiler_issues/tl_gather_unimplemented.md).
* Any hand-written 2-D tile load/store currently fails first-time compilation
  with a ``memref_ext.slice_*`` operand-segment mismatch (toolchain regression
  documented in compiler_issues/vllm_moe_sum.md).
* Scalar ``int64`` global loads fault the device nondeterministically (Sip
  error surfacing as SIGABRT at the next sync), pointer bitcasts are
  silently ignored by the backend, and — worse — torch_gcu stores int64
  tensors with an int32-compact layout behind 8-byte metadata ("Long use
  Int replace"), so a raw read of int64 elements picks up uninitialized
  neighbour bytes.  int64 metadata (``query_start_loc`` /
  ``cache_indices``) is therefore materialized into fresh int32 buffers
  host-side with the allowed ``torch.empty`` + ``copy_`` pair, and the
  kernel only ever issues int32 loads.

This rewrite therefore keeps the exact host-side dispatch, validation and
launch contract of the seed, but the kernel body uses only 1-D affine
accesses (``scalar base + arange(BLOCK_D)``) and rotates the window through
statically unrolled named registers ``h0..h6`` (``WIDTH`` is a constexpr in
[2, 8], so every guard below is pruned at compile time).  Per token the
kernel performs exactly one input load and one output store, mirroring the
seed's data flow; weights and the initial history are loaded once per
program, and the final history is written back with ``WIDTH - 1`` masked
1-D stores.

Numerics match the seed bit-for-bit: the current token value is rounded
through the state dtype before the fp32 dot product, the activated result
is rounded to the state dtype before the (implicitly cast) output store,
and the rolling history stays in fp32 (every history element is either an
exact state-dtype roundtrip or a state-dtype-rounded input, so the final
masked state store is bit-identical to the reference).

Grid: ``(requests, cdiv(dim, BLOCK_D))`` with ``BLOCK_D`` growing in powers
of two so the channel-block count respects the 255 limit on ``grid.y``;
``requests`` stays on ``grid.x`` (continuous-batching batch sizes are far
below the 65535 cap).
"""

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
):
    req = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    inb = channel < D
    # CU/SLOTS always arrive as int32 tensors (int64 metadata is materialized
    # host-side with an allowed empty+copy_).  The device must never see an
    # int64 load here: torch_gcu stores int64 tensors with an int32-compact
    # layout behind 8-byte metadata, and scalar int64 loads fault the device
    # nondeterministically.
    start = tl.load(CU + req) if VARLEN else 0
    end = tl.load(CU + req + 1) if VARLEN else T
    if start == end:
        return
    slot = tl.load(SLOTS + req) if HAS_SLOTS else req
    valid = (slot >= 0) & (slot != PAD) & (slot != NULL)
    use_state = tl.load(INITIAL + req) if HAS_INITIAL else True
    st_base = STATE + slot * SS_REQ + channel * SS_D
    x_base = X + req * SX_REQ + channel * SX_D
    o_base = OUT + req * SO_REQ + channel * SO_D
    w_base = W + channel * SW_D
    w_last = tl.load(w_base + (WIDTH - 1) * SW_T, mask=inb, other=0).to(tl.float32)
    h0 = tl.load(st_base + 0 * SS_T, mask=inb & valid & use_state, other=0).to(
        tl.float32
    )
    if WIDTH > 2:
        h1 = tl.load(st_base + 1 * SS_T, mask=inb & valid & use_state, other=0).to(
            tl.float32
        )
    if WIDTH > 3:
        h2 = tl.load(st_base + 2 * SS_T, mask=inb & valid & use_state, other=0).to(
            tl.float32
        )
    if WIDTH > 4:
        h3 = tl.load(st_base + 3 * SS_T, mask=inb & valid & use_state, other=0).to(
            tl.float32
        )
    if WIDTH > 5:
        h4 = tl.load(st_base + 4 * SS_T, mask=inb & valid & use_state, other=0).to(
            tl.float32
        )
    if WIDTH > 6:
        h5 = tl.load(st_base + 5 * SS_T, mask=inb & valid & use_state, other=0).to(
            tl.float32
        )
    if WIDTH > 7:
        h6 = tl.load(st_base + 6 * SS_T, mask=inb & valid & use_state, other=0).to(
            tl.float32
        )
    w0 = tl.load(w_base + 0 * SW_T, mask=inb, other=0).to(tl.float32)
    if WIDTH > 2:
        w1 = tl.load(w_base + 1 * SW_T, mask=inb, other=0).to(tl.float32)
    if WIDTH > 3:
        w2 = tl.load(w_base + 2 * SW_T, mask=inb, other=0).to(tl.float32)
    if WIDTH > 4:
        w3 = tl.load(w_base + 3 * SW_T, mask=inb, other=0).to(tl.float32)
    if WIDTH > 5:
        w4 = tl.load(w_base + 4 * SW_T, mask=inb, other=0).to(tl.float32)
    if WIDTH > 6:
        w5 = tl.load(w_base + 5 * SW_T, mask=inb, other=0).to(tl.float32)
    if WIDTH > 7:
        w6 = tl.load(w_base + 6 * SW_T, mask=inb, other=0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + channel, mask=inb, other=0).to(tl.float32)
    for token in range(start, end):
        value = tl.load(x_base + token * SX_T, mask=inb, other=0)
        value = value.to(STATE.dtype.element_ty).to(tl.float32)
        result = h0 * w0 + value * w_last
        if WIDTH > 2:
            result += h1 * w1
        if WIDTH > 3:
            result += h2 * w2
        if WIDTH > 4:
            result += h3 * w3
        if WIDTH > 5:
            result += h4 * w4
        if WIDTH > 6:
            result += h5 * w5
        if WIDTH > 7:
            result += h6 * w6
        if HAS_BIAS:
            result += bias
        if SILU:
            result = result * tl.sigmoid(result)
        result = result.to(STATE.dtype.element_ty)
        tl.store(o_base + token * SO_T, tl.where(valid, result, 0), mask=inb)
        if WIDTH > 2:
            h0 = h1
        if WIDTH > 3:
            h1 = h2
        if WIDTH > 4:
            h2 = h3
        if WIDTH > 5:
            h3 = h4
        if WIDTH > 6:
            h4 = h5
        if WIDTH > 7:
            h5 = h6
        if WIDTH == 2:
            h0 = value
        if WIDTH == 3:
            h1 = value
        if WIDTH == 4:
            h2 = value
        if WIDTH == 5:
            h3 = value
        if WIDTH == 6:
            h4 = value
        if WIDTH == 7:
            h5 = value
        if WIDTH == 8:
            h6 = value
    tl.store(st_base + 0 * SS_T, h0, mask=inb & valid)
    if WIDTH > 2:
        tl.store(st_base + 1 * SS_T, h1, mask=inb & valid)
    if WIDTH > 3:
        tl.store(st_base + 2 * SS_T, h2, mask=inb & valid)
    if WIDTH > 4:
        tl.store(st_base + 3 * SS_T, h3, mask=inb & valid)
    if WIDTH > 5:
        tl.store(st_base + 4 * SS_T, h4, mask=inb & valid)
    if WIDTH > 6:
        tl.store(st_base + 5 * SS_T, h5, mask=inb & valid)
    if WIDTH > 7:
        tl.store(st_base + 6 * SS_T, h6, mask=inb & valid)


def _md_i32(metadata):
    """Return an int32 view of request metadata for the kernel.

    torch_gcu implements int64 tensors with an int32-compact storage behind
    8-byte metadata ("GCU not support Long use Int replace"): a raw device
    read of an int64 element picks up uninitialized neighbour bytes, which
    nondeterministically corrupts slot indices or faults the device.  The
    kernel therefore only ever consumes int32 tensors; int64 metadata is
    materialized into a fresh int32 buffer with the allowed
    ``torch.empty`` + ``copy_`` pair, which reads the source through the
    same conversion path torch itself uses and stays correct if int64
    storage semantics are ever fixed upstream.  Cache slots and
    packed-token offsets are far below 2**31, so narrowing is lossless.
    """
    if metadata is None or metadata.dtype != torch.int64:
        return metadata
    narrowed = torch.empty(metadata.numel(), dtype=torch.int32, device=metadata.device)
    narrowed.copy_(metadata)
    return narrowed


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
        starts_arg = _md_i32(starts)
        slots_arg = _md_i32(slots)
        block_d = 32
        while triton.cdiv(dim, block_d) > 255 and block_d < 1024:
            block_d *= 2
        num_warps = 1 if block_d <= 64 else (2 if block_d <= 512 else 4)
        _conv_kernel[(requests, triton.cdiv(dim, block_d))](
            x,
            weight,
            bias if bias is not None else x,
            state,
            starts_arg if starts is not None else x,
            slots_arg if slots is not None else x,
            initial if initial is not None else x,
            out,
            dim,
            tokens,
            width,
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
            BLOCK_D=block_d,
            num_warps=num_warps,
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
