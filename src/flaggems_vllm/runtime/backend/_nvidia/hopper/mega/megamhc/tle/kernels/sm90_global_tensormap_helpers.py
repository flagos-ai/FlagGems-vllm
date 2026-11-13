"""Host tensor-map helpers for the production eager P384 family.

Formal callers are ``T in {12991, 13005, 13021, 16384}`` at
``H=4096, R=4, P=24``.  They use the fixed manual512, 12-producer/4-math,
four-stage K64 configuration with TMA Fn staging; this file is not a separate
kernel specialization.
"""

from __future__ import annotations

import ctypes

import torch


def _encode_global_tensor_map(fn: torch.Tensor) -> torch.Tensor:
    """Encode the exact [N32,K32] swizzled map and copy it to device global."""
    import triton.runtime.driver as driver_module

    # These constants are emitted by the validated FlagTree descriptor test:
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


__all__ = ["_encode_global_tensor_map"]
