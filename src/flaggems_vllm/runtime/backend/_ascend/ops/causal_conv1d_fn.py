# Copyright 2026- Xcoresigma Technology Co., Ltd
"""Ascend causal-conv1d prefill (varlen, continuous batching).

The launch geometry and inner loop mirror ``fla_npu``'s AscendC
``causal_conv1d_fn`` (``causal_conv1d_tiling_planner.h`` plus
``RunSeq``/``RunSeqFnRolling``):

* The host splits every request into contiguous token ranges so that
  ``#ranges x #channel_tiles`` is about one wave over the AIV cores.  A fixed
  per-chunk grid launched several sequential waves on the KDA shapes and
  re-paid the per-program prologue on every one of them.
* A program streams its whole token range in ``BLOCK_T``-token tiles.  The
  ``W`` taps are the same tile re-read at row offsets ``0 .. W-1``: the repeats
  of one iteration overlap almost completely and are served from L2, and the
  tile is large enough that the MTE2 requests amortize.  Tile size dominates
  here -- a per-token loop tops out near 0.5 TB/s on Ascend950DT, while a wide
  tile reaches ~2 TB/s, at the AscendC kernel's own bandwidth.
* The ``W-1`` leading tokens of a range have a window that reaches outside it
  (conv_state history or the previous range), so they use the register rolling
  recurrence of AscendC's fast path (``s_k <- s_{k+1} + w_k x_t``).  The tiled
  loop therefore needs no masked history correction at all.
* The ragged tail is peeled out of the loop and walked in short
  ``BLOCK_T_TAIL``-row tiles whose *start* is scalar-clamped to
  ``tk_end - BLOCK_T_TAIL``.  A per-lane clamped row makes the address
  non-affine and triton-ascend answers that with one DMA per element; keeping
  the start scalar keeps the tail affine, in bounds, and mask-free (the last
  tile re-covers a few rows, which recomputes identical values).

Two input layouts are consumed without a copy, chosen by strides:

* Unit channel stride (``x`` is token-major memory, what the vLLM projection
  produces) keeps the channel axis contiguous and uses a ``[tokens, channels]``
  tile.  This is the fastest shape: rows of ~9 KB.
* Unit token stride (the operator's own ``[dim, tokens]`` contract) keeps the
  token axis contiguous and uses a ``[channels, tokens]`` tile, so the tile load
  *is* the transpose the naive wrapper did with ``transpose(0, 1).contiguous()``
  -- a full-size device copy.  Fusing it measured 1.26-1.79x faster end to end.
  Rows here are only ``BLOCK_T * 2`` bytes, which caps the DMA efficiency, so
  ``_pick_tiles_channel_first`` trades channel width for row length.
* Anything else (neither axis unit-strided) is made contiguous once.

The channel tile is the dominant knob (``_pick_tiles``): it sets the bytes per
row transfer, and it should divide ``dim`` so no program pays a masked tail.
One read per row plus a UB row shift was also implemented and measured ~9%
faster (``extract_slice`` of ``BLOCK_T + W - 1`` rows), but the triton-ascend
software pipeline (``num_stages=2``) mis-versions the extracted views and
silently returns stale rows -- with ``num_stages=1`` the same kernel is
correct and ~35% slower, and an explicitly bound UB buffer does not fix the
pipeliner.  Re-reading from L2 is therefore the fastest correct formulation on
this toolchain.
"""

import numpy as np
import torch
import triton
import triton.language as tl

PAD_SLOT_ID = -1

# AIV cores per device.  NPUUtils reloads an extension on every call, so the
# answer is cached per device and never keyed on tensor data.
_CORE_COUNT_CACHE: dict = {}


def _aivector_core_count(device_index):
    key = -1 if device_index is None else int(device_index)
    if key not in _CORE_COUNT_CACHE:
        from triton.backends.ascend.driver import NPUUtils

        _CORE_COUNT_CACHE[key] = int(NPUUtils().get_aivector_core_num())
    return _CORE_COUNT_CACHE[key]


# Per-device [4, n] int32 upload buffer for the per-range plan: sequence id,
# range index, first token, token count.
_PLAN_WORKSPACE: dict = {}


def _plan_workspace(device, n_tasks):
    key = -1 if device.index is None else int(device.index)
    buf = _PLAN_WORKSPACE.get(key)
    width = max(n_tasks + 1, 1024)
    if buf is None or buf.size(1) < width:
        buf = torch.full((4, width), PAD_SLOT_ID, dtype=torch.int32, device=device)
        _PLAN_WORKSPACE[key] = buf
    return buf


@triton.jit()
def _weight_row(
    w_ptr,
    idx_feats,
    m_feat,
    row: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_w_dim: tl.constexpr,
):
    """One row of the ``(width, dim)`` weight as an fp32 channel vector."""
    return tl.load(
        w_ptr + row * stride_w_width + idx_feats * stride_w_dim, mask=m_feat, other=0.0
    ).to(tl.float32)


@triton.jit()
def _history_row(
    x_ptr,
    cs_ptr,
    idx_feats,
    m_feat,
    seq_start,
    rel,
    has_init,
    cs_slot,
    state_len: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_cs_seq: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    HAS_CONV_STATES: tl.constexpr,
):
    """History row for token ``rel`` before the range start.

    ``rel >= 0`` reads ``x``; otherwise the row comes from the conv_state
    history (when the sequence has one) or stays zero.  The history row for a
    negative ``rel`` is ``rel + state_len``: the state holds the ``state_len``
    tokens immediately before the sequence, so a range that starts inside those
    (a short sequence split across ranges) must index past the first row.
    """
    safe = tl.maximum(rel, 0)
    xv = tl.load(
        x_ptr + (seq_start + safe) * stride_x_token + idx_feats * stride_x_dim,
        mask=(rel >= 0) & m_feat,
        other=0.0,
    )
    if HAS_CONV_STATES:
        irow = tl.minimum(tl.maximum(rel + state_len, 0), state_len - 1)
        iv = tl.load(
            cs_ptr
            + cs_slot * stride_cs_seq
            + irow * stride_cs_tok
            + idx_feats * stride_cs_dim,
            mask=(rel < 0) & has_init & m_feat,
            other=0.0,
        )
        return tl.where(rel >= 0, xv, iv).to(tl.float32)
    return tl.where(rel >= 0, xv, 0.0).to(tl.float32)


@triton.jit()
def _causal_conv1d_fwd_kernel(
    # Pointers to matrices
    x_ptr,  # (seqlen, dim), one contiguous row per token
    w_ptr,  # (width, dim)
    bias_ptr,
    initial_states_ptr,  # (pool, dim, width - 1), unit dim stride
    cache_indices_ptr,  # (batch, n_blocks + padding)
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,  # (task,) -> sequence id, PAD_SLOT_ID past the last task
    token_chunk_offset_ptr,  # (task,) -> range index inside the sequence
    token_start_ptr,  # (task,) -> first token of the range, sequence-local
    token_len_ptr,  # (task,) -> tokens in the range
    block_idx_first_scheduled_token,  # (batch,)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    num_computed_tokens,  # (batch,)
    o_ptr,  # (seqlen, dim)
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # total tokens across the batch
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    HAS_CONV_STATES: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    APC_BLOCK: tl.constexpr,  # prefix-caching block size in tokens
    BLOCK_N: tl.constexpr,  # channels per program
    BLOCK_T: tl.constexpr,  # tokens per staged tile
    BLOCK_T_TAIL: tl.constexpr,  # tokens per clamped tail tile
    NCT: tl.constexpr,  # channel tiles: cdiv(dim, BLOCK_N)
    CHANNEL_FIRST: tl.constexpr,  # tile is [channels, tokens] instead
):
    pid = tl.program_id(0)
    task = pid // NCT
    ct = pid % NCT

    idx_seq = tl.load(batch_ptr + task)
    if idx_seq == pad_slot_id:
        return

    idx_feats = ct * BLOCK_N + tl.arange(0, BLOCK_N)
    m_feat = idx_feats < dim

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seq_len = sequence_end_index - sequence_start_index

    tk_start = tl.load(token_start_ptr + task)
    tk_len = tl.load(token_len_ptr + task)
    chunk_offset = tl.load(token_chunk_offset_ptr + task)
    tk_end = tk_start + tk_len
    if tk_len <= 0:
        return

    state_len = KERNEL_WIDTH - 1

    if IS_APC_ENABLED:
        # Prefix caching: a range is one cache block, and ranges 1..n cache the
        # state at their own block boundary.
        B_size = APC_BLOCK
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)
        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seq_len - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size
        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    if HAS_CONV_STATES:
        if HAS_CACHE_INDICES:
            conv_state_slot = tl.load(
                cache_indices_ptr
                + idx_seq.to(tl.int64) * stride_cache_indices
                + conv_state_init_index
            ).to(tl.int64)
        else:
            # No index table: cache slots are sequence-major and updated in place.
            conv_state_slot = idx_seq.to(tl.int64)
        if USE_PAD_SLOT:  # noqa
            if conv_state_slot == pad_slot_id:
                # not processing as this is not the actual sequence
                return
        conv_state_base = (
            initial_states_ptr
            + conv_state_slot * stride_istate_seq
            + idx_feats * stride_istate_dim
        )
        if HAS_INITIAL:
            has_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        else:
            has_init = False
    else:
        conv_state_slot = 0
        conv_state_base = initial_states_ptr
        has_init = False

    # ------------------------------------------------------------------ #
    # weights and bias
    # ------------------------------------------------------------------ #
    if KERNEL_WIDTH == 2:
        wrows = [
            _weight_row(w_ptr, idx_feats, m_feat, 0, stride_w_width, stride_w_dim),
            _weight_row(w_ptr, idx_feats, m_feat, 1, stride_w_width, stride_w_dim),
        ]
    elif KERNEL_WIDTH == 3:
        wrows = [
            _weight_row(w_ptr, idx_feats, m_feat, 0, stride_w_width, stride_w_dim),
            _weight_row(w_ptr, idx_feats, m_feat, 1, stride_w_width, stride_w_dim),
            _weight_row(w_ptr, idx_feats, m_feat, 2, stride_w_width, stride_w_dim),
        ]
    else:
        wrows = [
            _weight_row(w_ptr, idx_feats, m_feat, 0, stride_w_width, stride_w_dim),
            _weight_row(w_ptr, idx_feats, m_feat, 1, stride_w_width, stride_w_dim),
            _weight_row(w_ptr, idx_feats, m_feat, 2, stride_w_width, stride_w_dim),
            _weight_row(w_ptr, idx_feats, m_feat, 3, stride_w_width, stride_w_dim),
        ]

    if HAS_BIAS:
        bias_v = tl.load(bias_ptr + idx_feats, mask=m_feat, other=0.0).to(tl.float32)
    else:
        bias_v = tl.zeros((BLOCK_N,), tl.float32)

    # ------------------------------------------------------------------ #
    # rolling prologue: the W-1 tokens whose window leaves this range.
    # h_i = x[tk_start - (W-1) + i] and weight row i multiplies h_{i+k}.
    # ------------------------------------------------------------------ #
    rel0 = tk_start - state_len
    h0 = _history_row(
        x_ptr,
        initial_states_ptr,
        idx_feats,
        m_feat,
        sequence_start_index,
        rel0,
        has_init,
        conv_state_slot,
        state_len,
        stride_x_token,
        stride_x_dim,
        stride_istate_seq,
        stride_istate_token,
        stride_istate_dim,
        HAS_CONV_STATES,
    )
    if KERNEL_WIDTH >= 3:
        h1 = _history_row(
            x_ptr,
            initial_states_ptr,
            idx_feats,
            m_feat,
            sequence_start_index,
            rel0 + 1,
            has_init,
            conv_state_slot,
            state_len,
            stride_x_token,
            stride_x_dim,
            stride_istate_seq,
            stride_istate_token,
            stride_istate_dim,
            HAS_CONV_STATES,
        )
    if KERNEL_WIDTH >= 4:
        h2 = _history_row(
            x_ptr,
            initial_states_ptr,
            idx_feats,
            m_feat,
            sequence_start_index,
            rel0 + 2,
            has_init,
            conv_state_slot,
            state_len,
            stride_x_token,
            stride_x_dim,
            stride_istate_seq,
            stride_istate_token,
            stride_istate_dim,
            HAS_CONV_STATES,
        )
    if KERNEL_WIDTH >= 5:
        h3 = _history_row(
            x_ptr,
            initial_states_ptr,
            idx_feats,
            m_feat,
            sequence_start_index,
            rel0 + 3,
            has_init,
            conv_state_slot,
            state_len,
            stride_x_token,
            stride_x_dim,
            stride_istate_seq,
            stride_istate_token,
            stride_istate_dim,
            HAS_CONV_STATES,
        )

    if KERNEL_WIDTH == 2:
        s0 = wrows[0] * h0 + bias_v
    elif KERNEL_WIDTH == 3:
        s0 = wrows[0] * h0 + wrows[1] * h1 + bias_v
        s1 = wrows[0] * h1 + bias_v
    elif KERNEL_WIDTH == 4:
        s0 = wrows[0] * h0 + wrows[1] * h1 + wrows[2] * h2 + bias_v
        s1 = wrows[0] * h1 + wrows[1] * h2 + bias_v
        s2 = wrows[0] * h2 + bias_v
    else:
        s0 = wrows[0] * h0 + wrows[1] * h1 + wrows[2] * h2 + wrows[3] * h3 + bias_v
        s1 = wrows[0] * h1 + wrows[1] * h2 + wrows[2] * h3 + bias_v
        s2 = wrows[0] * h2 + wrows[1] * h3 + bias_v
        s3 = wrows[0] * h3 + bias_v

    for i in tl.static_range(KERNEL_WIDTH - 1):
        row = tk_start + i
        safe = tl.minimum(row, seq_len - 1)
        m_cur = (i < tk_len) & m_feat
        xt = tl.load(
            x_ptr
            + (sequence_start_index + safe) * stride_x_token
            + idx_feats * stride_x_dim,
            mask=m_cur,
            other=0.0,
        ).to(tl.float32)
        y = s0 + wrows[KERNEL_WIDTH - 1] * xt
        if SILU_ACTIVATION:
            y = y / (1 + tl.exp(-y))
        tl.store(
            o_ptr
            + (sequence_start_index + safe) * stride_o_token
            + idx_feats * stride_o_dim,
            y.to(o_ptr.dtype.element_ty),
            mask=m_cur,
        )
        if KERNEL_WIDTH == 2:
            s0 = wrows[0] * xt
        elif KERNEL_WIDTH == 3:
            ns0 = wrows[1] * xt + s1
            ns1 = wrows[0] * xt
            s0, s1 = ns0, ns1
        elif KERNEL_WIDTH == 4:
            ns0 = wrows[2] * xt + s1
            ns1 = wrows[1] * xt + s2
            ns2 = wrows[0] * xt
            s0, s1, s2 = ns0, ns1, ns2
        else:
            ns0 = wrows[3] * xt + s1
            ns1 = wrows[2] * xt + s2
            ns2 = wrows[1] * xt + s3
            ns3 = wrows[0] * xt
            s0, s1, s2, s3 = ns0, ns1, ns2, ns3

    # ------------------------------------------------------------------ #
    # tiled main loop: taps are the same tile at a row/token offset, so this
    # part never touches the rolling state.  Every address here is affine in
    # the loop inducer; a per-lane clamped row makes it non-affine, and
    # triton-ascend answers that with one DMA per element, so only the short
    # fallback tail pays for it.
    # ------------------------------------------------------------------ #
    rr = tl.arange(0, BLOCK_T)
    t_first = tk_start + state_len
    n_full = tl.maximum((tk_end - t_first) // BLOCK_T, 0)

    if CHANNEL_FIRST:
        m_chan = m_feat[:, None]
    else:
        m_chan = m_feat[None, :]
    for it in range(0, n_full):
        rows = t_first + it * BLOCK_T + rr
        if CHANNEL_FIRST:
            acc = bias_v[:, None] + tl.zeros((BLOCK_N, BLOCK_T), tl.float32)
        else:
            acc = bias_v[None, :] + tl.zeros((BLOCK_T, BLOCK_N), tl.float32)
        for j in tl.static_range(KERNEL_WIDTH):
            # Tap j reads x[t - j]; rows - j >= tk_start - j + state_len > 0.
            tok_off = (sequence_start_index + rows - j) * stride_x_token
            if CHANNEL_FIRST:
                x_off = (idx_feats * stride_x_dim)[:, None] + tok_off[None, :]
                wj = wrows[KERNEL_WIDTH - 1 - j][:, None]
            else:
                x_off = tok_off[:, None] + (idx_feats * stride_x_dim)[None, :]
                wj = wrows[KERNEL_WIDTH - 1 - j][None, :]
            xj = tl.load(x_ptr + x_off, mask=m_chan, other=0.0)
            acc += wj * xj.to(tl.float32)
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        if CHANNEL_FIRST:
            o_off = (idx_feats * stride_o_dim)[:, None] + (
                (sequence_start_index + rows) * stride_o_token
            )[None, :]
        else:
            o_off = (sequence_start_index + rows)[:, None] * stride_o_token + (
                idx_feats * stride_o_dim
            )[None, :]
        tl.store(o_ptr + o_off, acc.to(o_ptr.dtype.element_ty), mask=m_chan)

    # Peeled tail tile; fully masked when the range is tile aligned.
    # Peeled tail.  Its rows can run past the range end and even past the
    # tensor, and triton-ascend answers a per-lane clamped (non-affine) address
    # with one DMA per element, so the fast path keeps the tile start scalar and
    # walks the tail in short ``BLOCK_T_TAIL``-row tiles whose start is clamped
    # to ``tk_end - BLOCK_T_TAIL``: every row stays in bounds and affine, and
    # the last tile simply re-covers a few rows the previous one already wrote
    # (the convolution is a pure function of ``x``, so the values are equal).
    rr_t = tl.arange(0, BLOCK_T_TAIL)
    tail_first = t_first + n_full * BLOCK_T
    n_tail = tl.maximum(tl.cdiv(tk_end - tail_first, BLOCK_T_TAIL), 0)
    if tk_end >= BLOCK_T_TAIL + state_len:
        for it in range(0, n_tail):
            tail_start = tl.minimum(
                tail_first + it * BLOCK_T_TAIL, tk_end - BLOCK_T_TAIL
            )
            rows = tail_start + rr_t
            if CHANNEL_FIRST:
                acc = bias_v[:, None] + tl.zeros((BLOCK_N, BLOCK_T_TAIL), tl.float32)
            else:
                acc = bias_v[None, :] + tl.zeros((BLOCK_T_TAIL, BLOCK_N), tl.float32)
            for j in tl.static_range(KERNEL_WIDTH):
                tok_off = (sequence_start_index + rows - j) * stride_x_token
                if CHANNEL_FIRST:
                    x_off = (idx_feats * stride_x_dim)[:, None] + tok_off[None, :]
                    wj = wrows[KERNEL_WIDTH - 1 - j][:, None]
                else:
                    x_off = tok_off[:, None] + (idx_feats * stride_x_dim)[None, :]
                    wj = wrows[KERNEL_WIDTH - 1 - j][None, :]
                xj = tl.load(x_ptr + x_off, mask=m_chan, other=0.0)
                acc += wj * xj.to(tl.float32)
            if SILU_ACTIVATION:
                acc = acc / (1 + tl.exp(-acc))
            if CHANNEL_FIRST:
                o_off = (idx_feats * stride_o_dim)[:, None] + (
                    (sequence_start_index + rows) * stride_o_token
                )[None, :]
            else:
                o_off = (sequence_start_index + rows)[:, None] * stride_o_token + (
                    idx_feats * stride_o_dim
                )[None, :]
            tl.store(o_ptr + o_off, acc.to(o_ptr.dtype.element_ty), mask=m_chan)
    else:
        # A range shorter than one clipped tile: clamp per lane and mask.
        rows = tail_first + rr_t
        if CHANNEL_FIRST:
            m_ld = m_feat[:, None] & (rows < tk_end)[None, :]
            acc = bias_v[:, None] + tl.zeros((BLOCK_N, BLOCK_T_TAIL), tl.float32)
        else:
            m_ld = (rows < tk_end)[:, None] & m_feat[None, :]
            acc = bias_v[None, :] + tl.zeros((BLOCK_T_TAIL, BLOCK_N), tl.float32)
        for j in tl.static_range(KERNEL_WIDTH):
            r_safe = tl.minimum(tl.maximum(rows - j, 0), seq_len - 1)
            tok_off = (sequence_start_index + r_safe) * stride_x_token
            if CHANNEL_FIRST:
                x_off = (idx_feats * stride_x_dim)[:, None] + tok_off[None, :]
                wj = wrows[KERNEL_WIDTH - 1 - j][:, None]
            else:
                x_off = tok_off[:, None] + (idx_feats * stride_x_dim)[None, :]
                wj = wrows[KERNEL_WIDTH - 1 - j][None, :]
            xj = tl.load(x_ptr + x_off, mask=m_ld, other=0.0)
            acc += wj * xj.to(tl.float32)
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        o_rows = tl.minimum(rows, seq_len - 1)
        if CHANNEL_FIRST:
            o_off = (idx_feats * stride_o_dim)[:, None] + (
                (sequence_start_index + o_rows) * stride_o_token
            )[None, :]
        else:
            o_off = (sequence_start_index + o_rows)[:, None] * stride_o_token + (
                idx_feats * stride_o_dim
            )[None, :]
        tl.store(o_ptr + o_off, acc.to(o_ptr.dtype.element_ty), mask=m_ld)

    # ------------------------------------------------------------------ #
    # conv_state writes
    # ------------------------------------------------------------------ #
    if HAS_CONV_STATES:
        tl.debug_barrier()

        if IS_APC_ENABLED:  # noqa
            # Block-boundary caches: range k > 0 stores the state at the start
            # of its own cache block.
            if chunk_offset > 0:  # noqa
                if (chunk_offset - 1) < n_block_to_fill:  # noqa
                    conv_states_output_coord = tl.load(
                        cache_indices_ptr
                        + idx_seq * stride_cache_indices
                        + current_first_index
                        + (chunk_offset - 1)
                    )
                    base_rel = (
                        last_full_block_token_index
                        - (n_block_to_fill - chunk_offset) * APC_BLOCK
                        - state_len
                    )
                    for pos in tl.static_range(KERNEL_WIDTH - 1):
                        rel = base_rel + pos
                        safe = tl.minimum(tl.maximum(rel, 0), seq_len - 1)
                        loaded = tl.load(
                            x_ptr
                            + (sequence_start_index + safe) * stride_x_token
                            + idx_feats * stride_x_dim,
                            mask=(rel >= 0) & m_feat,
                            other=0.0,
                        )
                        tl.store(
                            initial_states_ptr
                            + conv_states_output_coord * stride_istate_seq
                            + pos * stride_istate_token
                            + idx_feats * stride_istate_dim,
                            loaded,
                            mask=m_feat,
                        )

        # The range that starts the sequence also owns the final state. It is
        # the only program that reads this cache slot, so read-then-write here
        # cannot race with the state written by any other program.
        if tk_start == 0:  # noqa
            if HAS_CACHE_INDICES:
                conv_states_output_coord = tl.load(
                    cache_indices_ptr
                    + idx_seq * stride_cache_indices
                    + current_last_index
                ).to(tl.int64)
            else:
                conv_states_output_coord = idx_seq.to(tl.int64)
            for pos in tl.static_range(KERNEL_WIDTH - 1):
                rel = seq_len - state_len + pos
                safe = tl.minimum(tl.maximum(rel, 0), seq_len - 1)
                xv = tl.load(
                    x_ptr
                    + (sequence_start_index + safe) * stride_x_token
                    + idx_feats * stride_x_dim,
                    mask=(rel >= 0) & m_feat,
                    other=0.0,
                )
                irow = tl.minimum(tl.maximum(rel + state_len, 0), state_len - 1)
                iv = tl.load(
                    conv_state_base + irow * stride_istate_token,
                    mask=(rel < 0) & has_init & m_feat,
                    other=0.0,
                )
                tl.store(
                    initial_states_ptr
                    + conv_states_output_coord * stride_istate_seq
                    + pos * stride_istate_token
                    + idx_feats * stride_istate_dim,
                    tl.where(rel >= 0, xv, iv),
                    mask=m_feat,
                )


def _plan_ranges(seqlens, nct, core_num, align, uniform=None):
    """Split each request into contiguous token ranges.

    ``#ranges`` is chosen so ``#ranges * nct`` fits in one wave over the AIV
    cores, the AscendC tiling planner's ``blockDim <= coreNum`` rule: every
    extra wave re-pays the per-program prologue, and a partial wave strands
    most of the cores.
    """
    seqlens = [int(s) for s in seqlens]
    n_seq = len(seqlens)
    counts = [1] * n_seq

    if uniform is not None:
        # Prefix caching: a range is one cache block, all of equal size.
        chunk = max(1, int(uniform))
        counts = [max(1, -(-s // chunk)) for s in seqlens]
    else:
        budget = max(1, core_num // nct)
        while sum(counts) < budget:
            cand = [i for i in range(n_seq) if seqlens[i] > counts[i]]
            if not cand:
                break
            j = max(cand, key=lambda i: (seqlens[i] / counts[i], -i))
            counts[j] += 1

    seq_ids, starts, lens, chunks = [], [], [], []
    for s, seq_len in enumerate(seqlens):
        if seq_len <= 0:
            continue
        k = counts[s]
        chunk = min(seq_len, -(-seq_len // k))
        chunk = min(seq_len, -(-chunk // align) * align)
        for c, a in enumerate(range(0, seq_len, chunk)):
            seq_ids.append(s)
            starts.append(a)
            lens.append(min(chunk, seq_len - a))
            chunks.append(c)
    return (
        np.asarray(seq_ids, dtype=np.int32),
        np.asarray(starts, dtype=np.int32),
        np.asarray(lens, dtype=np.int32),
        np.asarray(chunks, dtype=np.int32),
    )


# UB budget for the staged tiles: ``W`` tap buffers (``BLOCK_T x BLOCK_N``,
# bf16, doubled by the auto multi-buffer pass) plus the fp32 accumulator cost
# ``4*W + 4`` bytes per element.  A 20 B/element tile of 9216 elements (~184 KB)
# is the largest that compiles and runs on Ascend950DT; larger tiles are
# rejected by the compiler or spill.
_UB_BYTES = 184 * 1024
_MAX_BLOCK_N = 4608
_ALIGN_N = 256


def _pick_tiles(dim, width=4):
    """Channel tile and token tile.

    The channel tile is the unit of a row transfer and the main bandwidth
    knob: one 4608-wide tile of a 9216-channel row transfers 9 KB per row, a
    1024-wide split only 2 KB, and the narrow split measured ~2x slower.  The
    token tile then takes whatever the UB budget leaves.

    Preferring a channel tile that divides ``dim`` exactly matters as much as
    width: a ragged last tile makes every program in that column pay mask and
    DMA inefficiency, and dim=2304 / dim=4608 used to land on 2048 + 256 / 512
    tails.  Widths are rounded to 256 so each row transfer stays burst
    aligned; Ascend 950DT KDA shapes (dim = 3*H*128, e.g. 2304 / 4608 / 9216)
    then split into 2304x4 / 4608x2 / 4608x2 with no tail at all.
    """
    budget = _UB_BYTES // (4 * width + 4)
    if dim <= 512:
        # Tiny channel counts: one whole-row tile, tall token tiles are free
        # until the same UB budget caps them.
        block_n = triton.next_power_of_2(max(1, dim))
        block_t = max(2, min(32, budget // block_n))
    else:
        block_n = _MAX_BLOCK_N
        for n_tiles in range(1, 65):
            candidate = -(-dim // n_tiles)
            candidate = -(-candidate // _ALIGN_N) * _ALIGN_N
            # Two token rows must fit: a single-row tile is all DMA overhead.
            if candidate <= _MAX_BLOCK_N and 2 * candidate <= budget:
                block_n = candidate
                break
        block_t = max(2, budget // block_n)
    # Power-of-two tile heights only: a non-power-of-two height (5, 6, ...)
    # compiled but produced wrong rows on triton-ascend.
    block_t = 1 << (block_t.bit_length() - 1)
    return block_n, block_t


def _pick_tiles_channel_first(dim, width, max_seqlen):
    """Tile for a channel-first ``[dim, tokens]`` input.

    The token axis is the unit-strided one, so one channel row is
    ``BLOCK_T * 2`` bytes of contiguous memory and ``BLOCK_T`` is the bandwidth
    knob; ``BLOCK_N`` only sets how many programs share the row.  On the KDA
    shapes a ~32-channel x ~256-token tile measured fastest: wider channel
    tiles shorten the rows faster than they cut the (already cheap) program
    count, and narrower ones make every program pay its prologue/epilogue on
    only a few channels.

    Two bounds keep other batches sane:

    * ``BLOCK_T`` never exceeds the request, so the pipelined full-tile loop
      does the work instead of the clamped tail.
    * short requests raise ``BLOCK_N`` so a program still gets ~4K elements;
      otherwise a batch of 8-token requests would explode into tens of
      thousands of tiny programs.
    """
    budget = _UB_BYTES // (4 * width + 4)
    need = 4096 // max(1, max_seqlen)
    block_n = max(32, triton.next_power_of_2(max(1, need)))
    # Wide channel tiles are cheap to *load* but not to *describe*: the scalar
    # helpers around the tile (weights, history vectors, masks) are padded to
    # the vector fragment shape, and >=1K channels blow the UB budget on their
    # own (2048 channels x 2 tokens needed 462 KB).  Stay moderate.
    block_n = min(block_n, 256, triton.next_power_of_2(max(1, dim)))
    block_t = min(budget // block_n, triton.next_power_of_2(max(1, max_seqlen)))
    block_t = max(2, block_t)
    block_t = 1 << (block_t.bit_length() - 1)
    return block_n, block_t


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if cache_indices are not provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in the
        batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example, query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block containing the initial state is located.
    num_computed_tokens: (batch,), dtype int32
        The number of tokens already completed for each sequence
    block_size_to_align: int
        The block size to align the cached states to
    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    if x.dim() != 2:
        raise NotImplementedError("causal_conv1d_fn expects a 2D [dim, tokens] x")
    if conv_states is None:
        raise NotImplementedError("causal_conv1d_fn requires conv_states")
    if weight.shape[1] < 2 or weight.shape[1] > 5:
        raise NotImplementedError("causal_conv1d_fn supports width 2..5")

    # Store original dtype to cast back at the end
    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    dim, cu_seqlen = x.shape
    weight = weight.transpose(0, 1).contiguous()  # (dim, width) -> (width, dim)
    width, _ = weight.shape

    # Two layouts are consumed without a copy: unit channel stride means the
    # token rows are contiguous (the vLLM projection layout), unit token stride
    # means the channel rows are contiguous (the operator's own [dim, tokens]
    # contract).  Reading the latter directly fuses what used to be an up-front
    # ``transpose(0, 1).contiguous()`` -- a full-size device copy -- into the
    # kernel's tile loads.  Only an input with neither axis unit-strided needs
    # the copy.
    channel_first = x.stride(0) != 1
    if channel_first and x.stride(1) != 1:
        x = x.contiguous()
    if channel_first:
        stride_x_dim, stride_x_token = x.stride(0), x.stride(1)
        out = torch.empty((dim, cu_seqlen), dtype=x.dtype, device=x.device)
        stride_o_dim, stride_o_token = out.stride(0), out.stride(1)
        result = out
    else:
        stride_x_dim, stride_x_token = 1, x.stride(1)
        out = torch.empty((cu_seqlen, dim), dtype=x.dtype, device=x.device)
        stride_o_token, stride_o_dim = out.stride(0), out.stride(1)
        result = out.transpose(0, 1)

    is_apc_enabled = block_idx_last_scheduled_token is not None
    seqlens = query_start_loc.cpu().diff()
    if seqlens.numel() == 0:
        return result.to(original_x_dtype)

    stride_w_dim = weight.stride(1)  # (width, dim): stride(1) indexes dim
    stride_w_width = weight.stride(0)  # (width, dim): stride(0) indexes width
    num_cache_lines = conv_states.size(0)
    if dim != conv_states.shape[1] or width - 1 > conv_states.shape[2]:
        raise AssertionError("conv_states must be (pool, dim, width - 1)")
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    if stride_istate_dim != 1:
        raise AssertionError("conv_states needs a unit dim stride")
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None and query_start_loc.dim() == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1 and dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None
        assert weight.stride(1) == 1
        assert (width, dim) == weight.shape
        assert (x.stride(0) == 1) or (
            x.stride(1) == 1
        ), "causal_conv1d_fn needs one unit-stride axis"

    core_num = _aivector_core_count(x.device.index)
    if channel_first:
        block_n, block_t = _pick_tiles_channel_first(
            dim, width, int(seqlens.max().item()) if seqlens.numel() else 0
        )
    else:
        block_n, block_t = _pick_tiles(dim, width)
    nct = triton.cdiv(dim, block_n)

    if is_apc_enabled:
        # One range per prefix-cache block keeps the block-boundary state math
        # identical to the chunked formulation.
        apc_block = int(block_size_to_align) if block_size_to_align else 128
        ranges = _plan_ranges(seqlens, nct, core_num, 1, uniform=apc_block)
    else:
        apc_block = 0
        ranges = _plan_ranges(seqlens, nct, core_num, block_t)

    seq_ids, token_starts, token_lens, chunk_ids = ranges
    n_tasks = int(seq_ids.size)
    grid = (n_tasks * nct,)

    # One device-side upload per call for all four per-range arrays.  The
    # buffers are cached per device (like the old fixed-size batch_ptr), so a
    # repeated shape costs a single copy kernel instead of four allocations.
    plan = torch.from_numpy(
        np.stack([seq_ids, chunk_ids, token_starts, token_lens]).astype(np.int32)
    )
    ws = _plan_workspace(x.device, n_tasks)
    ws[:, :n_tasks].copy_(plan)
    batch_ptr, token_chunk_offset_ptr, token_start_ptr, token_len_ptr = ws

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        token_start_ptr,
        token_len_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_cache_indices,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        HAS_INITIAL=has_initial_state is not None,
        HAS_CONV_STATES=conv_states is not None,
        HAS_CACHE_INDICES=cache_indices is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=is_apc_enabled,
        USE_PAD_SLOT=pad_slot_id is not None,
        APC_BLOCK=max(apc_block, 1),
        BLOCK_N=block_n,
        BLOCK_T=block_t,
        BLOCK_T_TAIL=min(block_t, 32),
        NCT=nct,
        CHANNEL_FIRST=channel_first,
        num_stages=2,
        multibuffer=True,
    )

    return result.to(original_x_dtype)
