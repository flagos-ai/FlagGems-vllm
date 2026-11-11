"""Research-only raw-global tensor-map probe for the p384 Fn producer.

PTX permits a tensor-map object to reside in global memory.  FlagTree's
``tensordesc`` value cannot be used as an inline-assembly operand, so this
probe encodes the same CUDA tensor map on the host, copies its 128 bytes to a
device allocation, and consumes that ordinary pointer from inline PTX.  A
successful probe enables the role-aware 12+4-warp p384 loop to issue TMA
without leaving its single lexical assembly scope.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


HIDDEN = 4096
FLAT_HIDDEN = 4 * HIDDEN
PROJECTION_OUTPUTS = 24
BLOCK_N = 32
BLOCK_K = 64
JIT_BLOCK_N = tl.constexpr(BLOCK_N)
JIT_BLOCK_K = tl.constexpr(BLOCK_K)


@triton.jit
def _shared_address(pointer):
    return tl.inline_asm_elementwise(
        asm="cvta.to.shared.u64 $0, $1;",
        constraints="=l,l",
        args=[tl.cast(pointer, tl.uint64)],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    ).to(tl.uint32)


@triton.jit
def _raw_global_tensormap_copy(
    descriptor_address,
    shared_address,
    mbarrier_address,
    flat_hidden_base,
):
    return tl.inline_asm_elementwise(
        asm="""
        {
          .reg .pred leader, complete;
          .reg .u32 tid, coord0, smem1;
          .reg .u64 state;
          mov.u32 tid, %tid.x;
          setp.eq.u32 leader, tid, 0;
          @leader mbarrier.init.shared::cta.b64 [$3], 1;
          bar.sync 0, 128;
          @leader fence.proxy.tensormap::generic.acquire.gpu [$1], 128;
          @leader mbarrier.arrive.expect_tx.shared::cta.b64 state, [$3], 8192;
          mov.u32 coord0, $4;
          @leader cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [$2], [$1, {coord0, 0}], [$3];
          add.u32 coord0, coord0, 32;
          add.u32 smem1, $2, 4096;
          @leader cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [smem1], [$1, {coord0, 0}], [$3];
          @!leader bra DG_TLE_MHC_RAW_TMAP_WAIT_DONE;
        DG_TLE_MHC_RAW_TMAP_WAIT:
          mbarrier.try_wait.shared::cta.b64 complete, [$3], state;
          @!complete bra.uni DG_TLE_MHC_RAW_TMAP_WAIT;
        DG_TLE_MHC_RAW_TMAP_WAIT_DONE:
          bar.sync 0, 128;
          mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r,r",
        args=[
            descriptor_address,
            shared_address,
            mbarrier_address,
            flat_hidden_base,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _sm90_global_tensormap_inline_probe_kernel(
    descriptor_ptr,
    output_ptr,
    flat_hidden_base,
):
    fn_shared = tle.gpu.alloc(
        [JIT_BLOCK_N, JIT_BLOCK_K],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    tma_barrier = tle.gpu.alloc(
        [1],
        dtype=tl.uint64,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    _ = _raw_global_tensormap_copy(
        tl.cast(descriptor_ptr, tl.uint64),
        _shared_address(tle.gpu.local_ptr(fn_shared, (0, 0))),
        _shared_address(tle.gpu.local_ptr(tma_barrier, (0,))),
        flat_hidden_base,
    )
    n = tl.arange(0, JIT_BLOCK_N)
    k = tl.arange(0, JIT_BLOCK_K)
    n_index = tl.broadcast_to(n[:, None], (JIT_BLOCK_N, JIT_BLOCK_K))
    k_index = tl.broadcast_to(k[None, :], (JIT_BLOCK_N, JIT_BLOCK_K))
    values = tl.load(tle.gpu.local_ptr(fn_shared, (n_index, k_index)))
    tl.store(output_ptr + n[:, None] * JIT_BLOCK_K + k[None, :], values)


def _encode_global_tensor_map(fn: torch.Tensor) -> torch.Tensor:
    """Encode the exact [N32,K32] swizzled map and copy it to device global."""
    import triton.runtime.driver as driver_module

    # These constants are emitted by the validated FlagTree descriptor probe:
    # swizzle=3 (128 B), FP32 element size/type=(4,7), box=[32,32].
    tensor_map = driver_module.active.utils.fill_tma_descriptor(
        fn.data_ptr(),
        3,
        4,
        7,
        [32, 32],
        list(fn.shape),
        list(fn.stride()),
        0,
    )
    object_address = id(tensor_map)
    if object_address % 128 != 0:
        raise RuntimeError("PyCUtensorMap allocation lost its 128-byte alignment")
    # PyCUtensorMapObject is ``PyObject_HEAD; alignas(128) CUtensorMap``.
    raw = ctypes.string_at(object_address + 128, 128)
    host = torch.tensor(list(raw), dtype=torch.uint8)
    device = host.to(device=fn.device)
    if device.data_ptr() % 128 != 0:
        raise RuntimeError("device tensor-map storage must be 128-byte aligned")
    return device


@dataclass(frozen=True)
class GlobalTensorMapInlineProbePlan:
    fn: torch.Tensor
    descriptor_storage: torch.Tensor
    output: torch.Tensor
    flat_hidden_base: int

    def compile(self):
        return _sm90_global_tensormap_inline_probe_kernel.warmup(
            self.descriptor_storage,
            self.output,
            self.flat_hidden_base,
            grid=(1,),
            num_warps=4,
            num_stages=1,
            num_ctas=1,
        )

    def run(self) -> None:
        _sm90_global_tensormap_inline_probe_kernel[(1,)](
            self.descriptor_storage,
            self.output,
            self.flat_hidden_base,
            num_warps=4,
            num_stages=1,
            num_ctas=1,
        )


def prepare_global_tensormap_inline_probe(
    fn: torch.Tensor,
    *,
    flat_hidden_base: int,
) -> GlobalTensorMapInlineProbePlan:
    if tuple(fn.shape) != (PROJECTION_OUTPUTS, FLAT_HIDDEN):
        raise ValueError("fn must have shape [24, 16384]")
    if fn.dtype != torch.float32 or not fn.is_contiguous():
        raise ValueError("fn must be contiguous FP32")
    if fn.device.type != "cuda":
        raise ValueError("fn must reside on CUDA")
    if flat_hidden_base < 0 or flat_hidden_base + BLOCK_K > FLAT_HIDDEN:
        raise ValueError("tile is outside Fn")
    if flat_hidden_base % BLOCK_K != 0:
        raise ValueError("tile must be K64 aligned")
    return GlobalTensorMapInlineProbePlan(
        fn=fn,
        descriptor_storage=_encode_global_tensor_map(fn),
        output=torch.empty(
            (BLOCK_N, BLOCK_K), device=fn.device, dtype=torch.float32
        ),
        flat_hidden_base=int(flat_hidden_base),
    )


__all__ = ["prepare_global_tensormap_inline_probe"]
