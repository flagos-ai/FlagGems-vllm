# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Private TLE implementation adapted from FlagTree tutorials 08/09.

Source revision: ff67c2fb6be1d4651258605f6a5edde533d3a61c.
Only kernels and scheduling are reused; communicator ownership is library-local.
On HCU, FlagTree must lower tensor descriptors to ordinary masked pointer
loads/stores; the descriptor kernels then run without NVIDIA TMA hardware.
Autotune exemption: this stateful collective must execute the same launch sequence
on every rank. Use the coordinated benchmark sweep instead of per-rank libtuner.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.runtime import DistributedRtContext


@dataclasses.dataclass
class TleReduceScatter2DContext:

    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    with_gemm_output: bool
    # Scatter, reduction, and P2P share one registered communication buffer.
    comm_buf: torch.Tensor
    gemm_out_buf: Optional[torch.Tensor]
    scatter_buf: torch.Tensor
    rs_per_node_buf: torch.Tensor
    p2p_buf: torch.Tensor
    signal_buf: torch.Tensor
    dist_ctx: DistributedRtContext

    reduction_stream: torch.cuda.Stream
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int
    num_scatter_sms: int

    scatter_signal_buf: torch.Tensor = dataclasses.field(init=False)
    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)
    device_mesh: tle.device_mesh = dataclasses.field(init=False)
    world_mesh: tle.device_mesh = dataclasses.field(init=False)

    def __post_init__(self):
        if self.world_size < 2:
            raise ValueError("TLE reduce-scatter requires at least two GPUs")
        if self.local_world_size < 1 or self.world_size % self.local_world_size:
            raise ValueError(
                "world_size must be divisible by positive local_world_size"
            )
        if self.max_M % self.world_size:
            raise ValueError("max_M must be divisible by world_size")
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        self.device_mesh = tle.device_mesh(tle.MeshConfig(device=self.local_world_size))
        self.world_mesh = tle.device_mesh(
            tle.MeshConfig(node=self.nnodes, device=self.local_world_size)
        )
        if self.signal_buf.numel() < self.world_size:
            raise ValueError("signal_buf must contain one GEMM-ready flag per rank")
        if self.num_scatter_sms < 1:
            raise ValueError("num_scatter_sms must be positive")

        # GEMM-ready flags; tile counters use a separate workspace.
        self.scatter_signal_buf = self.signal_buf[: self.world_size]

    @property
    def num_rs_sms(self) -> int:
        """Return the communication SM budget."""
        if self.nnodes == 1:
            return self.num_scatter_sms
        return (
            self.num_scatter_sms
            + self.num_sync_sms
            + self.num_p2p_sms
            + self.num_reduction_sms
        )

    @property
    def rs_per_node_offset_elems(self) -> int:
        return self.max_M * self.N

    @property
    def p2p_offset_elems(self) -> int:
        per_node_elems = (self.max_M // self.local_world_size) * self.N
        return self.rs_per_node_offset_elems + per_node_elems

    def scatter_view(self, M: int, N: int) -> torch.Tensor:
        nelems = M * N
        if nelems > self.rs_per_node_offset_elems:
            raise ValueError("shape exceeds the registered scatter buffer")
        return self.comm_buf[:nelems].view(M, N)

    def rs_per_node_view(self, M: int, N: int) -> torch.Tensor:
        rows = M // self.local_world_size
        nelems = rows * N
        start = self.rs_per_node_offset_elems
        if start + nelems > self.p2p_offset_elems:
            raise ValueError("shape exceeds the registered per-node buffer")
        return self.comm_buf[start : start + nelems].view(rows, N)

    def p2p_view(self, M: int, N: int) -> torch.Tensor:
        rows = M // self.local_world_size
        nelems = rows * N
        start = self.p2p_offset_elems
        if start + nelems > self.comm_buf.numel():
            raise ValueError("shape exceeds the registered p2p buffer")
        return self.comm_buf[start : start + nelems].view(rows, N)

    def reset_barriers(self):
        self.signal_buf.zero_()


@triton.jit
def _scatter_kernel(
    input_ptr,
    local_scatter_ptr,
    dist_ctx: tl.constexpr,
    ready_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SCATTER_NODE_SLICE_OFFSET_ELEMS: tl.constexpr,
    WAIT_FOR_READY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Scatter ready shards to local GPUs."""

    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_peer = num_tiles_m * num_tiles_n
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    source_slot_offset_elems = LOCAL_RANK * M_per_rank * N
    # Resolve local/remote pointer selection at compile time for the HCU backend.
    for step in tl.static_range(WORLD_SIZE):
        target_rank = (LOCAL_RANK + step + 1) % WORLD_SIZE

        # Wait for this GPU to finish the destination GEMM shard.
        if WAIT_FOR_READY:
            while (
                tl.atomic_add(ready_ptr + target_rank, 0, sem="acquire", scope="gpu")
                == 0
            ):
                pass

        if target_rank == LOCAL_RANK:
            remote_base = local_scatter_ptr + source_slot_offset_elems
        else:
            remote_base = tle.remote(
                dist_ctx,
                space="device",
                dtype=input_ptr.dtype.element_ty,
                shard_id=target_rank,
                offset=SCATTER_NODE_SLICE_OFFSET_ELEMS + source_slot_offset_elems,
            )

        for local_tile in range(pid, tiles_per_peer, num_pid):
            tile_m = local_tile // num_tiles_n
            tile_n = local_tile % num_tiles_n
            input_row = target_rank * M_per_rank + tile_m * BLOCK_M
            input_col = tile_n * BLOCK_N
            input_ptrs = (
                input_ptr
                + (input_row + row_offs[:, None]) * N
                + input_col
                + col_offs[None, :]
            )
            row_mask = (input_row + row_offs[:, None]) < (target_rank + 1) * M_per_rank
            col_mask = (input_col + col_offs[None, :]) < N
            values = tl.load(input_ptrs, mask=row_mask & col_mask, other=0.0)

            scatter_row = tile_m * BLOCK_M
            scatter_ptrs = (
                remote_base
                + (scatter_row + row_offs[:, None]) * N
                + input_col
                + col_offs[None, :]
            )
            scatter_row_mask = (scatter_row + row_offs[:, None]) < M_per_rank
            tl.store(scatter_ptrs, values, mask=scatter_row_mask & col_mask)


@triton.jit
def _device_barrier_kernel(dist_ctx: tl.constexpr, mesh: tl.constexpr):
    """Synchronize local GPUs before reduction."""

    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=dist_ctx,
        space="device",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=0,
    )


@triton.jit
def _world_barrier_kernel(dist_ctx: tl.constexpr, mesh: tl.constexpr):
    tle.distributed_barrier(
        mesh=mesh,
        device_dptr=dist_ctx,
        space="world",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=0,
    )


@triton.jit
def _inter_node_p2p_kernel(
    comm_buf,
    offset,
    local_world_size,
    M_per_rank,
    N,
    ctx: tl.constexpr,
    DTYPE: tl.constexpr,
    NODE_ID: tl.constexpr,
    NNODES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    RS_OFFSET: tl.constexpr,
    P2P_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Send a locally reduced shard to its destination node."""
    remote_node_id = (offset + 1 + NODE_ID) % NNODES
    remote_rank = LOCAL_RANK + remote_node_id * local_world_size
    nelems_per_rank = M_per_rank * N

    # Reduction slots are indexed by destination node; receive slots by source node.
    src_offset = RS_OFFSET + remote_node_id * nelems_per_rank
    dst_offset = P2P_OFFSET + NODE_ID * nelems_per_rank

    remote_dst = tle.remote(
        ctx,
        space="node",
        dtype=DTYPE,
        shard_id=remote_rank,
        coopkind=tle.GroupKind.BLOCK,
        context_id=0,
    )

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < nelems_per_rank
    values = tl.load(
        comm_buf + src_offset + offsets,
        mask=mask,
    )
    tl.store(
        remote_dst + dst_offset + offsets,
        values,
        mask=mask,
    )


def _tl_dtype(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"unsupported TLE node-put dtype: {dtype}")


@triton.jit
def _ring_reduce_tma_kernel(
    local_scatter_ptr,
    output_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr = 1.0,
):
    """Sum shard contributions using TMA loads."""

    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n
    scatter_desc = tl.make_tensor_descriptor(
        local_scatter_ptr,
        shape=[M_per_rank * WORLD_SIZE, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    for tile_id in range(pid, total_tiles, num_pid):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row = tile_m * BLOCK_M
        col = tile_n * BLOCK_N
        source_rank = (LOCAL_RANK + 1) % WORLD_SIZE
        accum = scatter_desc.load([row + source_rank * M_per_rank, col])

        for i in range(1, WORLD_SIZE):
            source_rank = (LOCAL_RANK + i + 1) % WORLD_SIZE
            accum += scatter_desc.load([row + source_rank * M_per_rank, col])
        if SCALE != 1.0:
            accum = (accum.to(tl.float32) * SCALE).to(output_ptr.dtype.element_ty)
        output_desc.store([row, col], accum)


def reduce_scatter_for_each_node(
    input_tensor: torch.Tensor,
    stream: torch.cuda.Stream,
    ctx: TleReduceScatter2DContext,
    ready_flags: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Reduce locally for each destination node and launch transfers."""

    if config is None:
        raise ValueError("explicit collective configuration is required")

    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    M, N = input_tensor.shape
    scatter_buf = ctx.scatter_view(M, N)
    rs_per_node_buf = ctx.rs_per_node_view(M, N)
    p2p_buf = ctx.p2p_view(M, N)
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    scatter_num_warps = config["scatter_num_warps"]
    reduce_num_warps = config["reduce_num_warps"]

    scatter_grid = lambda META: (
        min(
            triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            ctx.num_scatter_sms,
        ),
    )

    def reduce_launch_config(num_sms: int):
        if num_sms == -1:
            return (
                (
                    lambda META: (
                        triton.cdiv(M_per_rank, META["BLOCK_M"])
                        * triton.cdiv(N, META["BLOCK_N"]),
                    )
                ),
                BLOCK_N,
                reduce_num_warps,
            )
        return (
            (
                lambda META: (
                    min(
                        triton.cdiv(M_per_rank, META["BLOCK_M"])
                        * triton.cdiv(N, META["BLOCK_N"]),
                        num_sms,
                    ),
                )
            ),
            BLOCK_N,
            reduce_num_warps,
        )

    # Only fused GEMM waits for per-rank ready flags.

    with torch.cuda.stream(stream):
        for n in range(nnodes):
            # Process remote destinations first and this node last.
            cur_node_id = (node_id + n + 1) % nnodes

            input_intra_node = input_tensor[
                cur_node_id * M_per_node : (cur_node_id + 1) * M_per_node
            ]

            scatter_for_node = scatter_buf[
                cur_node_id * M_per_node : (cur_node_id + 1) * M_per_node
            ]

            rs_per_node_output = rs_per_node_buf[
                cur_node_id * M_per_rank : (cur_node_id + 1) * M_per_rank
            ]

            # Ready flags use global-rank order.

            signal_start = cur_node_id * local_world_size

            if ready_flags is not None:
                ready_for_node = ready_flags[
                    signal_start : signal_start + local_world_size
                ]
            else:
                ready_for_node = input_tensor

            # Select the destination node within the registered window; offsets are in elements.
            scatter_node_slice_offset_elems = cur_node_id * M_per_node * N

            _scatter_kernel[scatter_grid](
                input_intra_node,
                scatter_for_node,
                ready_ptr=ready_for_node,
                M_per_rank=M_per_rank,
                N=N,
                dist_ctx=ctx.dist_ctx,
                LOCAL_RANK=local_rank,
                WORLD_SIZE=local_world_size,
                SCATTER_NODE_SLICE_OFFSET_ELEMS=scatter_node_slice_offset_elems,
                WAIT_FOR_READY=ready_flags is not None,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=scatter_num_warps,
            )

            # Complete local scatter writes before reduction.
            _device_barrier_kernel[(1,)](dist_ctx=ctx.dist_ctx, mesh=ctx.device_mesh)

            # Use the full reduction grid for the final destination node.
            node_reduce_sms = -1 if n == nnodes - 1 else num_reduction_sms

            reduce_grid, reduce_block_n, reduce_warps = reduce_launch_config(
                node_reduce_sms
            )

            reduction_stream.wait_stream(stream)

            with torch.cuda.stream(reduction_stream):
                _ring_reduce_tma_kernel[reduce_grid](
                    scatter_for_node,
                    rs_per_node_output,
                    M_per_rank,
                    N,
                    LOCAL_RANK=local_rank,
                    WORLD_SIZE=local_world_size,
                    BLOCK_M=256,
                    BLOCK_N=reduce_block_n,
                    num_warps=reduce_warps,
                )

                if nnodes > 1:
                    if n == nnodes - 1:
                        p2p_buf[
                            node_id * M_per_rank : (node_id + 1) * M_per_rank
                        ].copy_(rs_per_node_output)
                    else:
                        _inter_node_p2p_kernel[(ctx.num_p2p_sms,)](
                            ctx.comm_buf,
                            n,
                            local_world_size,
                            M_per_rank,
                            N,
                            ctx=ctx.dist_ctx,
                            DTYPE=_tl_dtype(ctx.dtype),
                            NODE_ID=node_id,
                            NNODES=nnodes,
                            LOCAL_RANK=local_rank,
                            RS_OFFSET=ctx.rs_per_node_offset_elems,
                            P2P_OFFSET=ctx.p2p_offset_elems,
                            BLOCK_SIZE=triton.next_power_of_2(M_per_rank * N),
                            num_warps=16,
                        )

    stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return rs_per_node_buf[: M_per_rank * nnodes]
    return p2p_buf[: M_per_rank * nnodes]


def reduce_scatter_multi_node(
    input_tensor: torch.Tensor,
    stream: torch.cuda.Stream,
    ctx: TleReduceScatter2DContext,
    output: torch.Tensor,
    ready_flags: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Combine node contributions into the final output shard."""

    if config is None:
        raise ValueError("explicit collective configuration is required")

    M, N = input_tensor.shape
    M_per_rank = M // ctx.world_size

    rs_result_per_node = reduce_scatter_for_each_node(
        input_tensor, stream, ctx, ready_flags, config=config
    )

    final_grid = lambda META: (
        triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    # With one node, the final reduction copies the local result.
    with torch.cuda.stream(stream):
        if ctx.nnodes > 1:
            # Local reduction/P2P precede this barrier; one CTA owns index 0.
            _world_barrier_kernel[(1,)](dist_ctx=ctx.dist_ctx, mesh=ctx.world_mesh)
        _ring_reduce_tma_kernel[final_grid](
            rs_result_per_node,
            output,
            M_per_rank,
            N,
            LOCAL_RANK=ctx.node_id,
            WORLD_SIZE=ctx.nnodes,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            SCALE=config.get("scale", 1.0),
            num_warps=config["final_num_warps"],
        )
    return output


def reduce_scatter_2d_op(
    input_tensor: torch.Tensor,
    ctx: TleReduceScatter2DContext,
    output: Optional[torch.Tensor] = None,
    ready_flags: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Validate inputs and run hierarchical reduce-scatter."""

    if config is None:
        raise ValueError("explicit collective configuration is required")

    M, N = input_tensor.shape
    if input_tensor.dtype != ctx.dtype or N > ctx.N:
        raise ValueError("input shape/dtype exceeds reduce-scatter context")
    if M > ctx.max_M:
        raise ValueError("M exceeds reduce-scatter context capacity")
    M_per_rank = M // ctx.world_size
    if M_per_rank < 256:
        raise ValueError("M_per_rank must be >= 256 for the TMA reduce kernel")
    if M_per_rank < config["BLOCK_M"]:
        raise ValueError(
            f"M_per_rank ({M_per_rank}) must be >= BLOCK_M ({config['BLOCK_M']})"
        )
    if N < config["BLOCK_N"]:
        raise ValueError(f"N ({N}) must be >= BLOCK_N ({config['BLOCK_N']})")
    if output is None:
        output = torch.empty(
            (M_per_rank, N), dtype=input_tensor.dtype, device=input_tensor.device
        )
    if tuple(output.shape) != (M_per_rank, N):
        raise ValueError("output has an invalid reduce-scatter shape")
    if ready_flags is not None and ready_flags.numel() != ctx.world_size:
        raise ValueError("ready_flags must contain one entry per target rank")

    reduction_stream = ctx.reduction_stream
    scatter_stream = torch.cuda.current_stream()
    if scatter_stream is reduction_stream:
        raise ValueError("scatter_stream and reduction_stream must be distinct")

    with torch.cuda.stream(scatter_stream):
        _device_barrier_kernel[(1,)](dist_ctx=ctx.dist_ctx, mesh=ctx.device_mesh)

    output = reduce_scatter_multi_node(
        input_tensor, scatter_stream, ctx, output, ready_flags, config=config
    )

    with torch.cuda.stream(scatter_stream):
        ctx.reset_barriers()
    return output


@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    ready_ptr,
    counter_ptr,
    RANK: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Compute output tiles and publish per-rank readiness."""

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = RANK // LOCAL_WORLD_SIZE
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    tiles_per_sm = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_sm += 1

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K]
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_M,
            BLOCK_N // 2 if EPILOGUE_SUBTILE else BLOCK_N,
        ],
    )

    M_per_rank = M // WORLD_SIZE
    tiles_m_per_rank = M_per_rank // BLOCK_M
    tiles_per_group = GROUP_M * num_pid_n
    tile_id = start_pid - NUM_SMS
    k_tile = -1
    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_sm):

        k_tile = tl.where(k_tile == k_tiles - 1, 0, k_tile + 1)

        if k_tile == 0:
            tile_id += NUM_SMS
            group_id = tile_id // tiles_per_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            logical_pid_m = first_pid_m + tile_id % group_size_m
            pid_n = (tile_id % tiles_per_group) // group_size_m
            m_rank = logical_pid_m // tiles_m_per_rank
            pid_m_intra_rank = logical_pid_m - m_rank * tiles_m_per_rank
            m_node_id = m_rank // LOCAL_WORLD_SIZE
            m_local_rank = m_rank % LOCAL_WORLD_SIZE
            swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
            swizzle_m_local_rank = (m_local_rank + RANK + 1) % LOCAL_WORLD_SIZE
            swizzle_m_rank = swizzle_m_node_id * LOCAL_WORLD_SIZE + swizzle_m_local_rank
            pid_m = swizzle_m_rank * tiles_m_per_rank + pid_m_intra_rank
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

        a = a_desc.load([offs_am, k_tile * BLOCK_K])
        b = b_desc.load([offs_bn, k_tile * BLOCK_K])
        accumulator = tl.dot(a, b.T, accumulator)

        if k_tile == k_tiles - 1:

            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(c_ptr.dtype.element_ty)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(c_ptr.dtype.element_ty)
                c_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1)
            else:
                c_desc.store(
                    [offs_am, offs_bn],
                    accumulator.to(c_ptr.dtype.element_ty),
                )

            counter_start = offs_am // M_per_rank
            counter_end = (offs_am + BLOCK_M - 1) // M_per_rank
            counter_end = min(counter_end, WORLD_SIZE - 1)

            for counter_id in range(counter_start, counter_end + 1):
                m_start = M_per_rank * counter_id
                m_end = M_per_rank * (counter_id + 1) - 1
                tiled_m_start = m_start // BLOCK_M
                tiled_m_end = m_end // BLOCK_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                tiled_n = tl.cdiv(N, BLOCK_N)

                # Order tile-counter updates before publishing ready.
                # TMA store completion is a separate requirement.
                prior = tl.atomic_add(
                    counter_ptr + counter_id, 1, sem="acq_rel", scope="gpu"
                )

                if prior == tiled_m_size * tiled_n - 1:
                    # The last tile publishes ready; RS clears the flag after use.
                    tl.atomic_xchg(
                        ready_ptr + counter_id, 1, sem="release", scope="gpu"
                    )

            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


def gemm_rs_producer_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    barrier: torch.Tensor,
    workspace: torch.Tensor,
    world_size: int,
    local_world_size: int,
    rank: int,
    num_gemm_sms: int,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 256,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 8,
    STAGES: int = 4,
    NUM_WARPS: int = 8,
    EPILOGUE_SUBTILE: bool = False,
):
    """Launch the persistent GEMM producer."""

    if a.shape[1] != b.shape[1]:
        raise ValueError("incompatible GEMM dimensions")
    if a.dtype != b.dtype:
        raise ValueError("GEMM operands must have the same dtype")
    M, local_K = a.shape
    N = b.shape[0]
    M_per_rank = M // world_size

    if M_per_rank % BLOCK_SIZE_M:
        raise ValueError("M_per_rank must be aligned to BLOCK_SIZE_M")

    grid = lambda META: (
        min(
            num_gemm_sms,
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ),
    )

    return kernel_gemm_rs_producer_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        barrier,
        workspace,
        RANK=rank,
        LOCAL_WORLD_SIZE=local_world_size,
        WORLD_SIZE=world_size,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
        GROUP_M=GROUP_SIZE_M,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        NUM_SMS=num_gemm_sms,
        num_warps=NUM_WARPS,
        num_stages=STAGES,
    )
