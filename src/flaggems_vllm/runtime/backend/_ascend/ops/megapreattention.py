# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Single-launch BF16 pre-attention for an intra-node Ascend TP group.

Requires a CANN device compiler linked against the SHMEM mixed-core bitcode.
"""

import math
import socket

import torch
import torch.distributed as dist
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend.distributed import _extern_call
from triton.language.core import builtin
from triton.runtime import driver


@builtin
def publish_signal(signal, epoch, peer, _semantic=None):
    return _extern_call(
        "libshmem_device",
        "",
        [signal, epoch, 0, peer],
        {(tl.int32, tl.int32, tl.int32, tl.int32): ("aclshmemx_signal_op", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def wait_eq(signal, epoch, _semantic=None):
    return _extern_call(
        "libshmem_device",
        "",
        [signal, 0, epoch],
        {(tl.int32, tl.int32, tl.int32): ("aclshmem_wait_until_int32", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@builtin
def shmem_quiet(_semantic=None):
    return _extern_call(
        "libshmem_device",
        "",
        [],
        {(): ("aclshmem_quiet", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _norm_rope_block(
    x,
    gamma,
    cosine,
    sine,
    EPS: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    D: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    inv = tl.rsqrt(tl.sum(x * x, 2) / D + EPS)
    # Ascend's fused RMSNorm/MRoPE keeps normalization in FP32 until RoPE.
    norm = x * inv[:, :, None] * gamma[None, None, :]
    a = al.extract_slice(norm, [0, 0, 0], [BT, BH, HALF_ROPE], [1, 1, 1])
    b = al.extract_slice(norm, [0, 0, HALF_ROPE], [BT, BH, HALF_ROPE], [1, 1, 1])
    first = a * cosine[:, None, :] - b * sine[:, None, :]
    second = b * cosine[:, None, :] + a * sine[:, None, :]
    out = al.insert_slice(norm, first, [0, 0, 0], [BT, BH, HALF_ROPE], [1, 1, 1])
    return al.insert_slice(
        out, second, [0, 0, HALF_ROPE], [BT, BH, HALF_ROPE], [1, 1, 1]
    )


@triton.jit
def _prepare_rope_tile(
    Rope,
    Pos,
    Prepared,
    row_tile,
    T: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BT: tl.constexpr,
    R: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    ROPE_ROWS: tl.constexpr,
):
    t = row_tile * BT + tl.arange(0, BT)
    pos0 = tl.load(Pos + t * PS1, t < T, other=0)
    pos1 = tl.load(Pos + PS0 + t * PS1, t < T, other=0)
    pos2 = tl.load(Pos + 2 * PS0 + t * PS1, t < T, other=0)
    pos0 = pos0.to(tl.int64)
    pos1 = pos1.to(tl.int64)
    pos2 = pos2.to(tl.int64)
    f = tl.arange(0, R)
    freq = f % (R // 2)
    rope = tl.full((BT, R), 0, tl.bfloat16)
    for row in tl.static_range(BT):
        p0 = al.get_element(pos0, (row,))
        p1 = al.get_element(pos1, (row,))
        p2 = al.get_element(pos2, (row,))
        r0 = tl.load(Rope + p0 * ROPE_STRIDE + f, (p0 >= 0) & (p0 < ROPE_ROWS), other=0)
        if p1 == p0:
            r1 = r0
        else:
            r1 = tl.load(
                Rope + p1 * ROPE_STRIDE + f, (p1 >= 0) & (p1 < ROPE_ROWS), other=0
            )
        if p2 == p0:
            r2 = r0
        else:
            r2 = tl.load(
                Rope + p2 * ROPE_STRIDE + f, (p2 >= 0) & (p2 < ROPE_ROWS), other=0
            )
        mixed = tl.where(
            (freq % 3 == 1) & (freq < 3 * S1),
            r1,
            tl.where((freq % 3 == 2) & (freq < 3 * S2), r2, r0),
        )
        rope = al.insert_slice(rope, mixed[None, :], [row, 0], [1, R], [1, 1])
    tl.store(Prepared + t[:, None] * R + f[None, :], rope, t[:, None] < T)


@triton.jit
def _finish_tile(
    Y,
    QW,
    KW,
    Rope,
    Slots,
    Q,
    K,
    V,
    Gate,
    Cache,
    row_tile,
    head_group,
    T: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    SS: tl.constexpr,
    CACHE_PLANE: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    BT: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    t = row_tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    hq = head_group * BQ + tl.arange(0, BQ)
    hk = head_group * BK + tl.arange(0, BK)
    width: tl.constexpr = 2 * (QH + KH) * D
    slots = tl.load(Slots + t * SS, t < T, other=-1).to(tl.int64)
    f = tl.arange(0, R)
    rope = tl.load(Rope + t[:, None] * R + f[None, :], t[:, None] < T, other=0)
    cosine = al.extract_slice(rope, [0, 0], [BT, R // 2], [1, 1]).to(tl.float32)
    sine = al.extract_slice(rope, [0, R // 2], [BT, R // 2], [1, 1]).to(tl.float32)
    qmask = (t[:, None, None] < T) & (hq[None, :, None] < QH)
    kmask = (t[:, None, None] < T) & (hk[None, :, None] < KH)
    qaddr = t[:, None, None] * width + hq[None, :, None] * (2 * D) + d[None, None, :]
    xq = tl.load(Y + qaddr, qmask, other=0).to(tl.float32)
    gate = tl.load(Y + qaddr + D, qmask, other=0)
    qout = (t[:, None, None] * QH + hq[None, :, None]) * D + d[None, None, :]
    tl.store(Gate + qout, gate, qmask)
    qgamma = (tl.load(QW + d).to(tl.float32) + 1).to(tl.bfloat16).to(tl.float32)
    qvalue = _norm_rope_block(xq, qgamma, cosine, sine, EPS, BT, BQ, D, R // 2)
    tl.store(Q + qout, qvalue, qmask)
    if head_group * BK < KH:
        kaddr = (
            t[:, None, None] * width
            + QH * (2 * D)
            + hk[None, :, None] * D
            + d[None, None, :]
        )
        xk = tl.load(Y + kaddr, kmask, other=0).to(tl.float32)
        value = tl.load(Y + kaddr + KH * D, kmask, other=0)
        kgamma = (tl.load(KW + d).to(tl.float32) + 1).to(tl.bfloat16).to(tl.float32)
        key = _norm_rope_block(xk, kgamma, cosine, sine, EPS, BT, BK, D, R // 2)
        kout = (t[:, None, None] * KH + hk[None, :, None]) * D + d[None, None, :]
        tl.store(K + kout, key, kmask)
        tl.store(V + kout, value, kmask)
        cd = head_group * BK * D + tl.arange(0, BK * D)
        for row in tl.static_range(BT):
            token = row_tile * BT + row
            slot = al.get_element(slots, (row,))
            kr = al.extract_slice(key, [row, 0, 0], [1, BK, D], [1, 1, 1]).reshape(
                (BK * D,)
            )
            vr = al.extract_slice(value, [row, 0, 0], [1, BK, D], [1, 1, 1]).reshape(
                (BK * D,)
            )
            mask = (token < T) & (slot >= 0) & (slot < CACHE_SLOTS) & (cd < KH * D)
            tl.store(Cache + slot * KH * D + cd, kr, mask)
            tl.store(Cache + CACHE_PLANE + slot * KH * D + cd, vr, mask)


@triton.jit
def _project_tile(
    Input,
    W,
    Projected,
    rows,
    t,
    valid,
    head,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    d = tl.arange(0, D)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, D), 0, tl.float32)
    for step in range(tl.cdiv(H, BK)):
        k = step * BK + kk
        a = tl.load(
            Input + rows[:, None] * H + k[None, :],
            valid[:, None] & (k[None, :] < H),
            other=0,
        )
        b = tl.load(
            W + (head.to(rows.dtype) * D + d[None, :]) * H + k[:, None],
            k[:, None] < H,
            other=0,
        )
        acc += tl.dot(a, b)
    tl.store(
        Projected + t[:, None] * WIDTH + head * D + d[None, :],
        acc,
        valid[:, None],
    )


@triton.jit
def _megapreattention_kernel(
    X,
    W,
    Gathered,
    Projected,
    Pos,
    Rope,
    Prepared,
    QNorm,
    KNorm,
    Slots,
    Cache,
    Q,
    K,
    V,
    Gate,
    Signals,
    Epoch,
    Rank,
    LOCAL_T: tl.constexpr,
    TP: tl.constexpr,
    H: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    SS: tl.constexpr,
    PLANE: tl.constexpr,
    CACHE_SLOTS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    ROPE_ROWS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    EPS: tl.constexpr,
    CORES: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    COPY: tl.constexpr,
    BT: tl.constexpr,
    BQ: tl.constexpr,
    BKV: tl.constexpr,
    GROUPS: tl.constexpr,
    PIPELINED: tl.constexpr,
    SIGNAL_STRIDE: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    pid = tl.program_id(0)
    T: tl.constexpr = LOCAL_T * TP
    HEADS: tl.constexpr = 2 * (QH + KH)
    WIDTH: tl.constexpr = HEADS * D
    TILES: tl.constexpr = (LOCAL_T + BM - 1) // BM
    TILE_STRIDE: tl.constexpr = TP * SIGNAL_STRIDE
    ROPE_BLOCK: tl.constexpr = 16
    INDEX: tl.constexpr = tl.int64 if WIDE_OFFSETS else tl.int32
    rank_row = tl.cast(Rank, INDEX) * LOCAL_T

    if PIPELINED:
        with tle.scope(core_mode="vector"):
            if (pid == 0) & (tle.dsa.ascend.sub_vec_id() == 0):
                shmem_quiet()
                generation = tl.load(Epoch, volatile=True)
                tl.store(Epoch, generation + 1)
                shmem_quiet()
        tle.dsa.ascend.sync_block_all("all", 8)

        # Producer and consumer traverse the same row tiles independently. Each
        # tile keeps its original rows in the rank-major workspace until exit.
        with tle.scope(core_mode="vector"):
            if tle.dsa.ascend.sub_vec_id() == 0:
                shmem_quiet()
                write_epoch = tl.load(Epoch, volatile=True)
                # Resolve each destination once per producer, before copying tiles.
                peers = ()
                for peer in tl.static_range(TP):
                    peers += (tle.remote(Gathered, peer),)
                for copy_tile in range(pid, TILES, CORES):
                    tile_row = copy_tile.to(INDEX) * BM
                    base = tile_row * H
                    ready = copy_tile * TILE_STRIDE + Rank * SIGNAL_STRIDE
                    COPY_TILE: tl.constexpr = 8192
                    for copy_block in range(tl.cdiv(BM * H, COPY_TILE)):
                        offset = copy_block * COPY_TILE + tl.arange(0, COPY_TILE)
                        valid = (offset < BM * H) & (base + offset < LOCAL_T * H)
                        data = tl.load(X + base + offset, valid, other=0)
                        for peer in tl.static_range(TP):
                            tl.store(
                                peers[peer] + rank_row * H + base + offset,
                                data,
                                valid,
                            )
                    # All destinations must see the payload before tile readiness.
                    shmem_quiet()
                    for peer in tl.static_range(TP):
                        publish_signal(Signals + ready, write_epoch, peer)
            else:
                for rope_task in range(pid, tl.cdiv(T, ROPE_BLOCK), CORES):
                    _prepare_rope_tile(
                        Rope,
                        Pos,
                        Prepared,
                        rope_task.to(INDEX),
                        T,
                        PS0,
                        PS1,
                        S1,
                        S2,
                        ROPE_BLOCK,
                        R,
                        ROPE_STRIDE,
                        ROPE_ROWS,
                    )

        with tle.scope(core_mode="cube"):
            shmem_quiet()
            read_epoch = tl.load(Epoch, volatile=True)
            m = tl.arange(0, BM)
            FULL_TILES: tl.constexpr = TILES // CORES * CORES
            JOBS_PER_TILE: tl.constexpr = TP * HEADS
            for gemm_tile in range(pid, FULL_TILES, CORES):
                local_rows = gemm_tile.to(INDEX) * BM + m
                valid = local_rows < LOCAL_T
                local_t = rank_row + local_rows
                # Local input already exists. Do those columns while the
                # Vector workers publish peer data, then consume remotes.
                for head in range(HEADS):
                    _project_tile(
                        X,
                        W,
                        Projected,
                        local_rows,
                        local_t,
                        valid,
                        head,
                        H,
                        WIDTH,
                        D,
                        BM,
                        BK,
                    )
                for wait_source in tl.static_range(TP):
                    wait_eq(
                        Signals + gemm_tile * TILE_STRIDE + wait_source * SIGNAL_STRIDE,
                        read_epoch,
                    )
                for task in range((TP - 1) * HEADS):
                    source = (Rank + 1 + task // HEADS) % TP
                    head = task % HEADS
                    t = source.to(INDEX) * LOCAL_T + local_rows
                    _project_tile(
                        Gathered, W, Projected, t, t, valid, head, H, WIDTH, D, BM, BK
                    )
            # Full waves preserve row ownership. Share the final wave's head
            # jobs across all Cube workers so the last few row tiles do not
            # keep the rest of the chip waiting at the post-processing join.
            if FULL_TILES < TILES:
                last_tile = tl.full((), -1, tl.int32)
                for tail_job in range(pid, (TILES - FULL_TILES) * JOBS_PER_TILE, CORES):
                    tile = FULL_TILES + tail_job // JOBS_PER_TILE
                    if tile != last_tile:
                        for wait_source in tl.static_range(TP):
                            wait_eq(
                                Signals
                                + tile * TILE_STRIDE
                                + wait_source * SIGNAL_STRIDE,
                                read_epoch,
                            )
                        last_tile = tile
                    task = tail_job % JOBS_PER_TILE
                    source, head = task // HEADS, task % HEADS
                    local_rows = tile.to(INDEX) * BM + m
                    t = source.to(INDEX) * LOCAL_T + local_rows
                    _project_tile(
                        Gathered,
                        W,
                        Projected,
                        t,
                        t,
                        local_rows < LOCAL_T,
                        head,
                        H,
                        WIDTH,
                        D,
                        BM,
                        BK,
                    )
    else:
        with tle.scope(core_mode="vector"):
            if tle.dsa.ascend.sub_vec_id() == 0:
                for copy_task in range(pid, tl.cdiv(LOCAL_T * H, COPY), CORES):
                    offsets = copy_task.to(INDEX) * COPY + tl.arange(0, COPY)
                    valid = offsets < LOCAL_T * H
                    data = tl.load(X + offsets, valid, other=0)
                    for peer in tl.static_range(TP):
                        peer_input = tle.remote(Gathered, peer)
                        tl.store(peer_input + rank_row * H + offsets, data, valid)
        tle.dsa.ascend.sync_block_all("all", 8)
        # Both core roles participate in the SHMEM mixed barrier.
        with tle.scope(core_mode="cube"):
            tle.distributed_barrier()
        with tle.scope(core_mode="vector"):
            tle.distributed_barrier()
        tle.dsa.ascend.sync_block_all("all", 9)

        with tle.scope(core_mode="vector"):
            rope_worker = pid * 2 + tle.dsa.ascend.sub_vec_id().to(tl.int32)
            for rope_task in range(rope_worker, tl.cdiv(T, ROPE_BLOCK), CORES * 2):
                _prepare_rope_tile(
                    Rope,
                    Pos,
                    Prepared,
                    rope_task.to(INDEX),
                    T,
                    PS0,
                    PS1,
                    S1,
                    S2,
                    ROPE_BLOCK,
                    R,
                    ROPE_STRIDE,
                    ROPE_ROWS,
                )
        with tle.scope(core_mode="cube"):
            m = tl.arange(0, BM)
            for gemm_task in range(pid, tl.cdiv(T, BM) * HEADS, CORES):
                row, head = gemm_task // HEADS, gemm_task % HEADS
                t = row.to(INDEX) * BM + m
                _project_tile(
                    Gathered, W, Projected, t, t, t < T, head, H, WIDTH, D, BM, BK
                )
    tle.dsa.ascend.sync_block_all("all", 10)

    with tle.scope(core_mode="vector"):
        post_worker = pid * 2 + tle.dsa.ascend.sub_vec_id().to(tl.int32)
        for post_task in range(post_worker, tl.cdiv(T, BT) * GROUPS, CORES * 2):
            _finish_tile(
                Projected,
                QNorm,
                KNorm,
                Prepared,
                Slots,
                Q,
                K,
                V,
                Gate,
                Cache,
                (post_task // GROUPS).to(INDEX),
                post_task % GROUPS,
                T,
                QH,
                KH,
                SS,
                PLANE,
                CACHE_SLOTS,
                D,
                R,
                EPS,
                BT,
                BQ,
                BKV,
            )

    # Ranks may enter the next replay at different times. All readers must
    # finish before any rank starts writing the shared input for that replay.
    tle.dsa.ascend.sync_block_all("all", 11)
    with tle.scope(core_mode="cube"):
        tle.distributed_barrier()
    with tle.scope(core_mode="vector"):
        tle.distributed_barrier()
    tle.dsa.ascend.sync_block_all("all", 12)


class MegaPreAttentionWorkspace:
    """Own symmetric scratch for a fixed hidden width and token capacity.

    Only one SHMEM workspace may be active per process.
    All ranks must construct, call and close the workspace collectively, using
    the same number of input rows and call order. Calls sharing a workspace
    must run serially on one stream. Nonnegative cache slots must be unique.
    The rotary capacity is the largest rotary width this workspace can use.
    """

    def __init__(self, hidden, projected_width, group, *, capacity=None, rotary_dim=64):
        import shmem

        assert hidden.ndim == 2 and min(hidden.shape) > 0
        if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16:
            raise NotImplementedError("MegaPreAttention requires Ascend BF16 inputs")
        if not hidden.is_contiguous():
            raise NotImplementedError("the hidden shard must be contiguous")
        rank, tp = group.rank(), group.size()
        if tp not in (2, 4) or dist.get_process_group_ranks(group) != list(
            range(dist.get_world_size())
        ):
            raise NotImplementedError("MegaPreAttention requires a local TP2/4 world")
        self.group = group
        self.rank, self.tp = rank, tp
        local_tokens, self.hidden_width = hidden.shape
        self.capacity = local_tokens if capacity is None else capacity
        if self.capacity < local_tokens:
            raise ValueError("workspace capacity is smaller than the input shard")
        self.device = hidden.device
        self.shmem = shmem
        self.prepared = hidden.new_empty((self.capacity * tp, rotary_dim))
        self.projected = hidden.new_empty((self.capacity * tp, projected_width))
        self.cores = int(
            driver.active.utils.get_device_properties(self.device.index)["num_aicore"]
        )
        # Signal allocation and the launch use the same row tile.
        self._block_rows = 128
        self._signal_stride = 32  # Keep rank signals 128 bytes apart.
        gathered_elements = self.capacity * tp * self.hidden_width
        tiles = triton.cdiv(self.capacity, self._block_rows)
        ready_tiles = tiles if tiles > self.cores else 1
        signal_elements = ready_tiles * tp * self._signal_stride
        port = [None]
        if rank == 0:
            with socket.socket() as server:
                server.bind(("127.0.0.1", 0))
                port[0] = server.getsockname()[1]
        dist.broadcast_object_list(port, group=group, group_src=0, device=self.device)
        shmem.set_conf_store_tls(False, "")
        attr = shmem.InitAttr()
        attr.my_rank, attr.n_ranks = rank, tp
        heap_bytes = (
            gathered_elements * hidden.element_size() + signal_elements * 4 + (32 << 20)
        )
        heap_alignment = 2 << 20
        attr.local_mem_size = triton.cdiv(heap_bytes, heap_alignment) * heap_alignment
        attr.ip_port = f"tcp://127.0.0.1:{port[0]}"
        attr.option_attr.data_op_engine_type = shmem.OpEngineType.MTE
        status = shmem.aclshmem_init(attr)
        if status != 0:
            raise RuntimeError(
                f"SHMEM initialization failed: {status}, heap={attr.local_mem_size}"
            )
        self.gathered = shmem.aclshmem_create_tensor(
            [gathered_elements], dtype=hidden.dtype, device_id=self.device.index
        )
        self.signals = shmem.aclshmem_create_tensor(
            [signal_elements], dtype=torch.int32, device_id=self.device.index
        )
        self.signals.zero_()
        self.epoch = torch.zeros(
            self._signal_stride, dtype=torch.int32, device=self.device
        )
        torch.npu.synchronize(self.device)
        dist.barrier(group=group)

    def close(self):
        if self.gathered is None:
            return
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.group)
        self.shmem.aclshmem_free_tensor(self.signals)
        self.shmem.aclshmem_free_tensor(self.gathered)
        self.gathered = None
        self.shmem.aclshmem_finalize()

    def __call__(
        self,
        hidden,
        weight,
        positions,
        q_norm,
        k_norm,
        rope,
        cache,
        slots,
        group,
        *,
        num_heads,
        num_kv_heads,
        head_dim=256,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        eps=1e-6,
    ):
        if self.gathered is None:
            raise RuntimeError("MegaPreAttention workspace is closed")
        # Cube projection and vector reductions use power-of-two lanes.
        max_head_dim = 256
        if not (
            16 <= head_dim <= max_head_dim
            and head_dim & (head_dim - 1) == 0
            and 2 <= rotary_dim <= head_dim
            and rotary_dim & (rotary_dim - 1) == 0
        ):
            raise NotImplementedError(
                f"head_dim must be a power of two in [16, {max_head_dim}] and rotary_dim "
                "a power of two in [2, head_dim]"
            )
        assert rotary_dim <= self.prepared.shape[1]
        assert group is self.group and hidden.ndim == 2
        local_t, width = hidden.shape
        assert 0 < local_t <= self.capacity and width == self.hidden_width
        if not hidden.is_contiguous() or not weight.is_contiguous():
            raise NotImplementedError("hidden and projection weight must be contiguous")
        assert num_heads > 0 and num_kv_heads > 0
        assert num_heads % self.tp == 0
        assert num_kv_heads % self.tp == 0 or self.tp % num_kv_heads == 0
        qh, kh = num_heads // self.tp, max(1, num_kv_heads // self.tp)
        tokens = local_t * self.tp
        assert weight.shape == (self.projected.shape[1], width)
        assert self.projected.shape[1] == 2 * (qh + kh) * head_dim
        assert positions.shape == (3, tokens) and slots.shape == (tokens,)
        assert positions.dtype in (torch.int32, torch.int64)
        assert slots.dtype in (torch.int32, torch.int64)
        assert positions.device == slots.device == self.device
        assert q_norm.shape == k_norm.shape == (head_dim,)
        assert q_norm.is_contiguous() and k_norm.is_contiguous()
        assert rope.ndim == 2 and rope.shape[0] > 0 and rope.shape[1] == rotary_dim
        assert rope.stride(1) == 1
        assert cache.ndim == 5 and cache.shape[0] == 2
        assert cache.is_contiguous() and cache.shape[-2:] == (kh, head_dim)
        assert len(mrope_section) == 3 and sum(mrope_section) == rotary_dim // 2
        assert all(isinstance(part, int) and part >= 0 for part in mrope_section)
        assert math.isfinite(eps) and eps >= 0
        tensors = (hidden, weight, q_norm, k_norm, rope, cache)
        if any(x.dtype != torch.bfloat16 or x.device != self.device for x in tensors):
            raise NotImplementedError(
                "all floating inputs must be BF16 on the workspace device"
            )
        if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
            raise NotImplementedError("MegaPreAttention is inference-only")
        q = torch.empty((tokens, qh * head_dim), dtype=hidden.dtype, device=self.device)
        k = torch.empty((tokens, kh * head_dim), dtype=hidden.dtype, device=self.device)
        v = torch.empty_like(k)
        gate = torch.empty(
            (tokens, qh, head_dim), dtype=hidden.dtype, device=self.device
        )
        bq = min(4, triton.next_power_of_2(qh))
        bkv = min(4, triton.next_power_of_2(kh))
        bt = min(8, 1 << ((48 // (bq + bkv)).bit_length() - 1))
        groups = max(triton.cdiv(qh, bq), triton.cdiv(kh, bkv))
        # A second row tile per Cube makes rank-striped GEMM worthwhile.
        pipelined = triton.cdiv(local_t, self._block_rows) > self.cores
        # Widen before multiplying the row index when any linear offset needs it.
        wide_offsets = (
            max(
                tokens * width,
                tokens * weight.shape[0],
                weight.numel(),
                tokens * rotary_dim,
                2 * positions.stride(0) + (tokens - 1) * positions.stride(1),
                (tokens - 1) * slots.stride(0),
            )
            > 2**31 - 1
        )
        with torch.npu.device(self.device):
            _megapreattention_kernel[(self.cores, 1, 1)](
                hidden,
                weight,
                self.gathered,
                self.projected,
                positions,
                rope,
                self.prepared,
                q_norm,
                k_norm,
                slots,
                cache,
                q,
                k,
                v,
                gate,
                self.signals,
                self.epoch,
                self.rank,
                local_t,
                self.tp,
                width,
                qh,
                kh,
                positions.stride(0),
                positions.stride(1),
                slots.stride(0),
                cache.stride(0),
                cache.shape[1] * cache.shape[2],
                head_dim,
                rotary_dim,
                rope.stride(0),
                rope.shape[0],
                mrope_section[1],
                mrope_section[2],
                eps,
                CORES=self.cores,
                BM=self._block_rows,
                BK=256,
                COPY=1024,
                BT=bt,
                BQ=bq,
                BKV=bkv,
                GROUPS=groups,
                PIPELINED=pipelined,
                SIGNAL_STRIDE=self._signal_stride,
                WIDE_OFFSETS=wide_offsets,
                num_warps=4,
                num_stages=2 if pipelined else 1,
            )
        return q, k, v, gate
