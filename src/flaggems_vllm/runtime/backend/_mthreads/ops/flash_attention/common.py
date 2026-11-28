# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def validate_inputs(q, k, v, cuq, cuk, usedk, table, out, alibi, maxq, maxk):
    """Validate metadata only; length/index values remain the caller's contract."""
    if q.device.type != "musa":
        raise ValueError("MThreads FlashAttention requires MUSA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("MThreads FlashAttention supports FP16 and BF16")
    if q.ndim != 3 or k.ndim != (4 if table is not None else 3):
        raise ValueError("Invalid query or KV rank")
    if k.shape != v.shape or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q/K/V must have matching dtype and K/V shape")
    for tensor in (q, k, v):
        if tensor.device != q.device:
            raise ValueError("Q/K/V must use the same device")
        if tensor.stride(-1) != 1:
            raise NotImplementedError("Last dimension must have stride one")
    if q.shape[-1] != k.shape[-1] or not 0 < q.shape[-1] <= 256 or q.shape[-1] % 8:
        raise ValueError("Head dimensions must match and be multiples of 8 up to 256")
    if k.shape[-2] == 0 or q.shape[1] == 0 or q.shape[1] % k.shape[-2]:
        raise ValueError("KV heads must divide query heads")
    if not isinstance(maxq, int) or not isinstance(maxk, int) or min(maxq, maxk) < 0:
        raise ValueError("Maximum sequence lengths must be nonnegative Python integers")
    if q.shape[0] and maxq == 0:
        raise ValueError("Nonempty query requires a positive maximum length")
    if cuq is None or cuq.ndim != 1 or cuq.numel() < 1:
        raise ValueError("cu_seqlens_q must contain batch_size + 1 entries")
    batch = cuq.numel() - 1
    if q.shape[0] and batch == 0:
        raise ValueError("Nonempty query requires at least one sequence")
    if table is not None:
        if cuk is not None or usedk is None:
            raise ValueError("Paged KV requires seqused_k and no cu_seqlens_k")
        if k.shape[1] == 0 or table.ndim != 2 or table.shape[0] != batch:
            raise ValueError("Invalid page table or page size")
        if table.shape[1] * k.shape[1] < maxk or usedk.shape != (batch,):
            raise ValueError("Page table capacity or seqused_k shape is invalid")
    elif cuk is None or usedk is not None or cuk.shape != (batch + 1,):
        raise ValueError("Nonpaged KV requires cu_seqlens_k and no seqused_k")
    for tensor in (cuq, cuk, usedk, table):
        if tensor is not None and (
            tensor.device != q.device
            or tensor.dtype != torch.int32
            or (tensor.ndim == 1 and not tensor.is_contiguous())
            or tensor.stride(-1) != 1
        ):
            raise ValueError("Sequence metadata must be int32 on the query device")
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or out.stride(-1) != 1
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous with the query shape/dtype/device")
        # Cross-row aliasing can race between CTAs, including paged cache writes.
        for tensor in (q, k, v):
            if (
                out.numel()
                and tensor.numel()
                and (
                    out.untyped_storage().data_ptr()
                    == tensor.untyped_storage().data_ptr()
                )
            ):
                raise ValueError("out must not alias Q/K/V storage")
    if alibi is not None and (
        alibi.shape not in ((q.shape[1],), (batch, q.shape[1]))
        or alibi.dtype != torch.float32
        or alibi.device != q.device
        or alibi.stride(-1) != 1
    ):
        raise ValueError("ALiBi must be FP32 with shape [Hq] or [batch,Hq]")
