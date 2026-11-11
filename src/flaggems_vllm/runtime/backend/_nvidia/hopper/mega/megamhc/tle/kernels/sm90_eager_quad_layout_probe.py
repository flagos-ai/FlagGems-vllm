"""Opt-in probe for the eager K64 producer's per-thread BF16x4 stores.

The production experiment needs the 256 producer threads to write one
``M64 x K64`` tile into an MMA-layout shared allocation without first
materializing a two-dimensional Triton tensor.  This probe isolates exactly
that mapping: every physical thread owns four adjacent BF16 quads, writes them
through the address returned by :func:`tle.gpu.local_ptr`, and the complete
tile is then read back through the normal TLE shared-memory layout.

It is deliberately independent from MegaMHC dispatch and never participates
in the graph implementation.
"""

from __future__ import annotations

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from . import sm90_small_family as _small


BLOCK_M = 64
BLOCK_K = 64
NUM_WARPS = 8

JIT_BLOCK_M = tl.constexpr(BLOCK_M)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)
JIT_THREADS = tl.constexpr(NUM_WARPS * 32)
JIT_QUADS_PER_ROW = tl.constexpr(BLOCK_K // 4)


@triton.jit
def _load_global_bf16x4(address):
    """Load one naturally aligned BF16x4 and expand it to four FP32 values."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b32 packed0, packed1;
          .reg .b16 h0, h1, h2, h3;
          ld.global.v2.b32 {packed0, packed1}, [$4];
          mov.b32 {h0, h1}, packed0;
          mov.b32 {h2, h3}, packed1;
          cvt.f32.bf16 $0, h0;
          cvt.f32.bf16 $1, h1;
          cvt.f32.bf16 $2, h2;
          cvt.f32.bf16 $3, h3;
        }
        """,
        constraints="=&f,=&f,=&f,=&f,l",
        args=[address],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_shared_bf16x4(address, value0, value1, value2, value3):
    """Round four FP32 values to BF16 and store one shared uint2."""
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b32 packed0, packed1;
          cvt.rn.bf16x2.f32 packed0, $3, $2;
          cvt.rn.bf16x2.f32 packed1, $5, $4;
          st.shared.v2.b32 [$1], {packed0, packed1};
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,f,f,f,f",
        args=[address, value0, value1, value2, value3],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _shared_address(pointer):
    """Expose a TLE shared pointer as the 32-bit PTX shared address."""
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, $1;",
        constraints="=r,r",
        args=[pointer],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _quad_layout_probe_kernel(input_ptr, output_ptr):
    tile = tle.gpu.alloc(
        [JIT_BLOCK_M, JIT_BLOCK_K],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    ready = tle.gpu.alloc_barrier(arrive_count=JIT_THREADS)

    physical_thread = _small._physical_thread_id_x()
    for iteration in tl.static_range(0, 4):
        linear_quad = physical_thread + iteration * JIT_THREADS
        row = linear_quad // JIT_QUADS_PER_ROW
        column = (linear_quad % JIT_QUADS_PER_ROW) * 4
        global_address = tl.cast(
            input_ptr + row * JIT_BLOCK_K + column, tl.uint64
        )
        value0, value1, value2, value3 = _load_global_bf16x4(
            global_address
        )
        shared_address = _shared_address(
            tle.gpu.local_ptr(tile, (row, column))
        )
        _store_shared_bf16x4(
            shared_address, value0, value1, value2, value3
        )

    tle.gpu.barrier_wait(ready)
    values = tl.load(tle.gpu.local_ptr(tile))
    rows = tl.arange(0, JIT_BLOCK_M)
    columns = tl.arange(0, JIT_BLOCK_K)
    tl.store(
        output_ptr + rows[:, None] * JIT_BLOCK_K + columns[None, :],
        values,
    )


def compile_probe(input_tensor, output_tensor):
    """Compile and load the probe without launching it."""
    return _quad_layout_probe_kernel.warmup(
        input_tensor,
        output_tensor,
        grid=(1,),
        num_warps=NUM_WARPS,
        num_stages=1,
        num_ctas=1,
    )


def run_probe(input_tensor, output_tensor) -> None:
    """Launch the isolated one-CTA layout probe."""
    _quad_layout_probe_kernel[(1,)](
        input_tensor,
        output_tensor,
        num_warps=NUM_WARPS,
        num_stages=1,
        num_ctas=1,
    )
