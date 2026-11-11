"""SM90 ``Fn`` staging probe using per-thread classic ``cp.async``.

Unlike the descriptor/TMA probe, this path has no pipe token, TMA transaction
barrier, or writer/reader acquire-release pair.  Producer threads each copy one
aligned 16-byte FP32 vector directly into the logical coordinates of an
``nv_mma_shared_layout`` B tile.  Rows 24..31 use ``src-size=0`` and are
therefore zero-filled by the copy instruction.  After every issuing thread has
committed and waited for its own async group, one CTA named barrier publishes
the completed tile to the RS-WGMMA worker partition.

Two geometries mirror the intended integrations:

* K32: 256 producer threads, one 16-byte vector per thread (eager p256);
* K64: 512 producer threads, one 16-byte vector per thread (large Graph).

The module is research-only and is never imported by production dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from .sm90_eager_quad_layout_probe import _shared_address


HIDDEN = 4096
HC_MULT = 4
FLAT_HIDDEN = HIDDEN * HC_MULT
PROJECTION_OUTPUTS = 24
BLOCK_M = 64
BLOCK_N = 32
SUPPORTED_BLOCK_K = (32, 64)
JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_PROJECTION_OUTPUTS = tl.constexpr(PROJECTION_OUTPUTS)
JIT_FLAT_HIDDEN = tl.constexpr(FLAT_HIDDEN)


@triton.jit
def _physical_thread_id_x():
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %tid.x;",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cp_async_cg_16b_zero_fill(shared_address, global_address, src_size):
    """Issue one 16-byte global-to-shared copy with hardware zero fill."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          cp.async.cg.shared.global [$1], [$2], 16, $3;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,l,r",
        args=[shared_address, global_address, src_size],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _cp_async_commit_wait_all():
    return tl.inline_asm_elementwise(
        asm="""
        {
          cp.async.commit_group;
          cp.async.wait_group 0;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _fn_cp_async_writer(
    fn_ptr,
    fn_shared,
    ready,
    flat_hidden_base,
    BLOCK_K: tl.constexpr,
    ROOT_THREADS: tl.constexpr,
):
    physical_thread = _physical_thread_id_x()
    vectors_per_row: tl.constexpr = BLOCK_K // 4
    linear_vector = physical_thread
    row = linear_vector // vectors_per_row
    k_base = (linear_vector % vectors_per_row) * 4
    valid_row = row < JIT_PROJECTION_OUTPUTS
    safe_row = tl.where(valid_row, row, 0)
    global_address = tl.cast(
        fn_ptr
        + safe_row * JIT_FLAT_HIDDEN
        + flat_hidden_base
        + k_base,
        tl.uint64,
    )
    shared_address = _shared_address(
        tle.gpu.local_ptr(fn_shared, (row, k_base))
    )
    _ = _cp_async_cg_16b_zero_fill(
        shared_address,
        global_address,
        tl.where(valid_row, 16, 0),
    )
    _ = _cp_async_commit_wait_all()
    tle.gpu.barrier_wait(ready)


@triton.jit
def _fn_cp_async_wgmma_reader(
    q_ptr,
    c_ptr,
    fn_shared,
    ready,
    BLOCK_K: tl.constexpr,
):
    tle.gpu.barrier_wait(ready)
    m = tl.arange(0, JIT_BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    q_registers = tl.load(q_ptr + m[:, None] * BLOCK_K + k[None, :]).to(
        tl.float32
    )
    accumulator = tl.zeros((JIT_BLOCK_M, JIT_BLOCK_N), tl.float32)
    accumulator = tle.gpu.wgmma(
        q_registers,
        fn_shared,
        accumulator,
        input_precision="tf32",
        trans_b=True,
    )
    accumulator = tle.gpu.wgmma_wait(0, accumulator)
    n = tl.arange(0, JIT_BLOCK_N)
    tl.store(c_ptr + m[:, None] * JIT_BLOCK_N + n[None, :], accumulator)


@triton.jit
def _sm90_fn_cp_async_stage_probe_kernel(
    q_ptr,
    fn_ptr,
    c_ptr,
    flat_hidden_base,
    BLOCK_K: tl.constexpr,
    ROOT_THREADS: tl.constexpr,
):
    tl.static_assert(BLOCK_K == 32 or BLOCK_K == 64)
    tl.static_assert(ROOT_THREADS == JIT_BLOCK_N * BLOCK_K // 4)
    fn_shared = tle.gpu.alloc(
        [JIT_BLOCK_N, BLOCK_K],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    ready = tle.gpu.alloc_barrier(arrive_count=ROOT_THREADS + 128)
    tle.gpu.warp_specialize(
        [
            (
                _fn_cp_async_writer,
                (
                    fn_ptr,
                    fn_shared,
                    ready,
                    flat_hidden_base,
                    BLOCK_K,
                    ROOT_THREADS,
                ),
            ),
            (
                _fn_cp_async_wgmma_reader,
                (q_ptr, c_ptr, fn_shared, ready, BLOCK_K),
            ),
        ],
        [4],
        [168],
    )


@dataclass(frozen=True)
class FnCPAsyncStageProbePlan:
    q: torch.Tensor
    fn: torch.Tensor
    c: torch.Tensor
    flat_hidden_base: int
    block_k: int
    root_warps: int

    def _args(self) -> tuple[object, ...]:
        return (
            self.q,
            self.fn,
            self.c,
            self.flat_hidden_base,
            self.block_k,
            self.root_warps * 32,
        )

    def compile(self):
        return _sm90_fn_cp_async_stage_probe_kernel.warmup(
            *self._args(),
            grid=(1,),
            num_warps=self.root_warps,
            num_stages=1,
            num_ctas=1,
        )

    def run(self) -> None:
        _sm90_fn_cp_async_stage_probe_kernel[(1,)](
            *self._args(),
            num_warps=self.root_warps,
            num_stages=1,
            num_ctas=1,
        )


def prepare_fn_cp_async_stage_probe(
    q: torch.Tensor,
    fn: torch.Tensor,
    *,
    flat_hidden_base: int,
) -> FnCPAsyncStageProbePlan:
    if q.ndim != 2 or tuple(q.shape)[0] != BLOCK_M:
        raise ValueError(f"q must have shape [{BLOCK_M}, K]")
    block_k = int(q.shape[1])
    if block_k not in SUPPORTED_BLOCK_K:
        raise ValueError(f"K must be one of {SUPPORTED_BLOCK_K}")
    if q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError("q must be contiguous BF16")
    if tuple(fn.shape) != (PROJECTION_OUTPUTS, FLAT_HIDDEN):
        raise ValueError(
            f"fn must have shape [{PROJECTION_OUTPUTS}, {FLAT_HIDDEN}]"
        )
    if fn.dtype != torch.float32 or not fn.is_contiguous():
        raise ValueError("fn must be contiguous FP32")
    if q.device != fn.device or q.device.type != "cuda":
        raise ValueError("q and fn must reside on the same CUDA device")
    if flat_hidden_base < 0 or flat_hidden_base + block_k > FLAT_HIDDEN:
        raise ValueError("Fn tile is outside the flat hidden dimension")
    if flat_hidden_base % block_k != 0:
        raise ValueError("Fn tile offset must be block aligned")
    c = torch.empty(
        (BLOCK_M, BLOCK_N), dtype=torch.float32, device=q.device
    )
    root_warps = (BLOCK_N * block_k // 4) // 32
    return FnCPAsyncStageProbePlan(
        q=q,
        fn=fn,
        c=c,
        flat_hidden_base=int(flat_hidden_base),
        block_k=block_k,
        root_warps=root_warps,
    )


__all__ = [
    "FLAT_HIDDEN",
    "FnCPAsyncStageProbePlan",
    "prepare_fn_cp_async_stage_probe",
]
