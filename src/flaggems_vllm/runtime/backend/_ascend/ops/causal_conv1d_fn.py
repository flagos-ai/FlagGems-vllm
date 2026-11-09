# Copyright 2026- Xcoresigma Technology Co., Ltd
"""Ascend causal-conv1d prefill (varlen, continuous batching).

The launch geometry and inner loop mirror ``fla_npu``'s AscendC
``causal_conv1d_fn`` (``causal_conv1d_tiling_planner.h`` plus
``RunSeq``/``RunSeqFnRolling``), which is what the previous fixed
per-128-token-chunk grid was missing:

* The host splits every request into contiguous token ranges so that
  ``#ranges x #channel_tiles`` is about one wave over the AIV cores.  The old
  grid launched one program per 128-token chunk -- 9 sequential waves on the
  KDA shapes -- and re-paid the per-program prologue on every one of them.
* A program streams its whole token range in ``BLOCK_T``-token tiles, reading
  each ``x`` row once.  The ``W`` taps are the same tile re-read at row offsets
  ``1 .. W-1``: the raw volume stays at one read per row and the overlap is
  served from L2.  Tile size dominates here -- a per-token loop tops out near
  0.5 TB/s on Ascend950DT, an 8-row x 1024-channel tile reaches ~2 TB/s and is
  at the AscendC kernel's own bandwidth.
* The ``W-1`` leading tokens of a range have a window that reaches outside it
  (conv_state history or the previous range), so they use the register rolling
  recurrence of AscendC's fast path (``s_k <- s_{k+1} + w_k x_t``).  The tiled
  loop therefore needs no masked history correction at all.

Re-reading a tile per tap costs MTE2 traffic, but the tap tiles of one
iteration overlap almost completely, so the repeats hit L2.  That is measured
to beat slicing the loaded tile with ``extract_slice``, which on triton-ascend
either serializes the loop (without multi-buffer) or silently returns stale
rows (multi-buffer plus ``extract_slice``).
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
    NCT: tl.constexpr,  # channel tiles: cdiv(dim, BLOCK_N)
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
    # tiled main loop: one staged read per row, taps are the same tile at a
    # row offset, so this part never touches the rolling state.
    #
    # The full tiles keep an affine address in the loop inducer: clamping the
    # row (as the ragged tail needs) makes the address non-affine and costs
    # ~45% of the kernel, so only the peeled tail tile pays it.
    # ------------------------------------------------------------------ #
    rr = tl.arange(0, BLOCK_T)
    t_first = tk_start + state_len
    n_full = tl.maximum((tk_end - t_first) // BLOCK_T, 0)

    for it in range(0, n_full):
        rows = t_first + it * BLOCK_T + rr
        acc = bias_v[None, :] + tl.zeros((BLOCK_T, BLOCK_N), tl.float32)
        for j in tl.static_range(KERNEL_WIDTH):
            # Tap j reads x[t - j]; rows - j >= tk_start - j + state_len > 0.
            xj = tl.load(
                x_ptr
                + (sequence_start_index + rows - j)[:, None] * stride_x_token
                + (idx_feats * stride_x_dim)[None, :],
                mask=m_feat[None, :],
                other=0.0,
            )
            acc += wrows[KERNEL_WIDTH - 1 - j][None, :] * xj.to(tl.float32)
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(
            o_ptr
            + (sequence_start_index + rows)[:, None] * stride_o_token
            + (idx_feats * stride_o_dim)[None, :],
            acc.to(o_ptr.dtype.element_ty),
            mask=m_feat[None, :],
        )

    # Peeled tail tile; fully masked when the range is tile aligned.
    rows = t_first + n_full * BLOCK_T + rr
    m_ld = (rows[:, None] < tk_end) & m_feat[None, :]
    acc = bias_v[None, :] + tl.zeros((BLOCK_T, BLOCK_N), tl.float32)
    for j in tl.static_range(KERNEL_WIDTH):
        r_safe = tl.minimum(tl.maximum(rows - j, 0), seq_len - 1)
        xj = tl.load(
            x_ptr
            + (sequence_start_index + r_safe)[:, None] * stride_x_token
            + (idx_feats * stride_x_dim)[None, :],
            mask=m_ld,
            other=0.0,
        )
        acc += wrows[KERNEL_WIDTH - 1 - j][None, :] * xj.to(tl.float32)
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    o_rows = tl.minimum(rows, seq_len - 1)
    tl.store(
        o_ptr
        + (sequence_start_index + o_rows)[:, None] * stride_o_token
        + (idx_feats * stride_o_dim)[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=m_ld,
    )

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


def _pick_tiles(dim):
    """Channel tile and token tile.

    The channel tile is the unit of a row transfer, so it wants to be wide; the
    token tile is bounded by UB (``W`` staged tiles x multi-buffer plus the
    fp32 accumulator), and 8 rows x 1024 channels measured fastest on
    Ascend950DT.
    """
    if dim <= 512:
        # Tiny channel counts: one whole-row tile, tall token tiles are free.
        return triton.next_power_of_2(max(1, dim)), 32
    if dim <= 2304:
        # Narrow rows: a 2048-wide tile still covers the whole row (like the
        # AscendC planner's token-first baseDim), which matters because a
        # 1024-wide split of a 2304-channel row drops the copy roofline from
        # 2.3 TB/s to 1.4 TB/s.  UB then caps the token tile at 4.
        return 2048, 4
    return 1024, 8


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
    x = x.transpose(0, 1).contiguous()  # (dim, seqlen) -> (seqlen, dim)
    weight = weight.transpose(0, 1).contiguous()  # (dim, width) -> (width, dim)

    out = torch.empty_like(x)
    cu_seqlen, dim = x.shape
    width, _ = weight.shape

    is_apc_enabled = block_idx_last_scheduled_token is not None
    seqlens = query_start_loc.diff().to("cpu")
    if seqlens.numel() == 0:
        return out.transpose(0, 1).to(original_x_dtype)

    stride_x_dim = x.stride(1)
    stride_x_token = x.stride(0)
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
    stride_o_token = out.stride(0)
    stride_o_dim = out.stride(1)
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
        is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
        assert is_channel_last, "Need to run in channel-last layout"

    core_num = _aivector_core_count(x.device.index)
    block_n, block_t = _pick_tiles(dim)
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
        NCT=nct,
        num_stages=2,
        multibuffer=True,
    )

    return out.transpose(0, 1).to(original_x_dtype)
