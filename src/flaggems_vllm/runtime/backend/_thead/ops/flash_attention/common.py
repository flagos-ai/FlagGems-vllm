# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from triton.compiler.compiler import make_backend

try:
    import triton.experimental.tle as tle_api
    from triton.experimental.tle.language import gpu
except ImportError:
    tle_api = gpu = None


def supports_tle():
    if tle_api is None:
        return False
    features = ("gpu.alloc", "gpu.local_ptr")
    if hasattr(tle_api, "is_tle_supported"):
        return tle_api.is_tle_supported(features)
    target = triton.runtime.driver.active.get_current_target()
    backend = type(make_backend(target)).__module__.split(".")[-2]
    supported = tle_api.get_supported_primitives(backend)
    return all(feature in supported for feature in features)


def validate_inputs(q, k, v, cuq, cuk, usedk, table, out, alibi, maxq, maxk):
    if q.device.type != "cuda":
        raise ValueError("T-Head PPU FlashAttention requires PPU tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("T-Head PPU FlashAttention supports FP16 and BF16")
    if q.ndim != 3 or k.ndim != (4 if table is not None else 3):
        raise ValueError("Invalid query or KV rank")
    if k.shape != v.shape or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Q/K/V must have matching dtype and K/V shape")
    for tensor in (q, k, v):
        if tensor.requires_grad:
            raise NotImplementedError("T-Head PPU FlashAttention v1 is inference-only")
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
            or (tensor.ndim == 1 and (not tensor.is_contiguous()))
            or (tensor.stride(-1) != 1)
        ):
            raise ValueError("Sequence metadata must be int32 on the query device")
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or (out.stride(-1) != 1)
            or (not out.is_contiguous())
        ):
            raise ValueError("out must be contiguous with the query shape/dtype/device")
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
        or (alibi.stride(-1) != 1)
    ):
        raise ValueError("ALiBi must be FP32 with shape [Hq] or [batch,Hq]")


@triton.constexpr_function
def cube_layout(SWIZZLE):
    return gpu.swizzled_shared_layout(
        8, 1, 8 if SWIZZLE else 1, [1, 0], [1, 1], [1, 1], [1, 1], [0, 1]
    )


@triton.jit
def tsm_fragment(base, row, col, TRANS: tl.constexpr):
    ptr = base + tl.zeros((128,), tl.uint64)
    if TRANS:
        b0, b1, b2, b3 = tl.inline_asm_elementwise(
            "ppu.tc01.ldmatrix.sync.aligned.m16n16.x1.swzl.trans.shared.b16 {$0,$1,$2,$3}, [$4], {$5,$6,$7,$8,$9,$10};",
            constraints="=r,=r,=r,=r,l,r,r,r,r,r,r,~{memory}",
            args=(ptr, 0, row, 1, 32, 1, col * 2),
            dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
            is_pure=False,
            pack=1,
        )
        return (b0, b1, b2, b3)
    else:
        b0, b1, b2, b3 = tl.inline_asm_elementwise(
            "ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 {$0,$1,$2,$3}, [$4], {$5,$6,$7,$8,$9,$10};",
            constraints="=r,=r,=r,=r,l,r,r,r,r,r,r,~{memory}",
            args=(ptr, 0, row, 1, 32, 1, col * 2),
            dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
            is_pure=False,
            pack=1,
        )
        return (b0, b1, b2, b3)


@triton.jit
def issue_copy(X, BUF, start, klen, STRIDE: tl.constexpr, SLOT_ROW, HEADCOL):
    zero = tl.full((), 0, tl.int32)
    owner = tl.arange(0, 128)
    for cube in tl.static_range(4):
        dest = (
            gpu.local_ptr(BUF, ((SLOT_ROW // 32 * 4 + cube) * 32, zero))
            .to(tl.uint64)
            .to(tl.uint32)
        )
        tl.inline_asm_elementwise(
            (
                "{ .reg .pred p; setp.eq.u32 p, $13, 0; @p ppu.cp.async.aiu.bulk."
                "tensor.shared.global.padz.swzl.2d.b16 [$1], [$2], {$3,$4,$5,$6,$"
                "7,$8}, {$9,$10,$11,$12}; mov.u32 $0, 0; }"
            ),
            constraints="=r,r,l,r,r,r,r,r,r,r,r,r,r,r,~{memory}",
            args=(
                dest,
                X.to(tl.uint64),
                1,
                klen,
                STRIDE,
                0,
                start,
                HEADCOL + cube * 64,
                1,
                32,
                64,
                1,
                owner,
            ),
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )


@triton.jit
def ld_fragment(BUF, col, token, TRANS: tl.constexpr, ROW):
    zero = tl.full((), 0, tl.int32)
    cube = col // 64
    base = gpu.local_ptr(BUF, ((ROW // 32 * 4 + cube) * 32, zero)).to(tl.uint64)
    base = base + token.to(tl.uint64)
    return tsm_fragment(base, ROW % 32, col % 64, TRANS)


@triton.jit
def commit_copies():
    tl.inline_asm_elementwise(
        "cp.async.commit_group; mov.u32 $0, 0;",
        constraints="=r,~{memory}",
        args=(),
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_copies():
    return tl.inline_asm_elementwise(
        "cp.async.wait_group 0; bar.sync 0; mov.u32 $0, 0;",
        constraints="=r,~{memory}",
        args=(),
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def pack_bf(a, b):
    return (
        a.to(tl.uint16, bitcast=True).to(tl.uint32)
        | b.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
    )


@triton.jit
def mma_reorder(a0, a1, a2, a3, a4, a5, a6, a7, word):
    ap0 = pack_bf(a0, a1).to(tl.float32, bitcast=True)
    ap1 = pack_bf(a2, a3).to(tl.float32, bitcast=True)
    ap2 = pack_bf(a4, a5).to(tl.float32, bitcast=True)
    ap3 = pack_bf(a6, a7).to(tl.float32, bitcast=True)
    w = word.to(tl.float32, bitcast=True)
    z = tl.full((128,), 0.0, tl.float32)
    if a0.dtype == tl.float16:
        instruction: tl.constexpr = (
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15};"
        )
    else:
        instruction: tl.constexpr = (
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15};"
        )
    d0, d1, d2, d3 = tl.inline_asm_elementwise(
        instruction,
        constraints="=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=(ap0, ap1, ap2, ap3, w, z, z, w, z, z, z, z),
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )
    bits0 = d0.to(tl.uint32, bitcast=True)
    v0 = (bits0 & 65535).to(tl.uint16).to(a0.dtype, bitcast=True)
    v1 = (bits0 >> 16).to(tl.uint16).to(a0.dtype, bitcast=True)
    bits1 = d1.to(tl.uint32, bitcast=True)
    v2 = (bits1 & 65535).to(tl.uint16).to(a0.dtype, bitcast=True)
    v3 = (bits1 >> 16).to(tl.uint16).to(a0.dtype, bitcast=True)
    bits2 = d2.to(tl.uint32, bitcast=True)
    v4 = (bits2 & 65535).to(tl.uint16).to(a0.dtype, bitcast=True)
    v5 = (bits2 >> 16).to(tl.uint16).to(a0.dtype, bitcast=True)
    bits3 = d3.to(tl.uint32, bitcast=True)
    v6 = (bits3 & 65535).to(tl.uint16).to(a0.dtype, bitcast=True)
    v7 = (bits3 >> 16).to(tl.uint16).to(a0.dtype, bitcast=True)
    return (v0, v1, v2, v3, v4, v5, v6, v7)


@triton.jit
def mma_packed(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    DTYPE: tl.constexpr = tl.bfloat16,
):
    if DTYPE == tl.float16:
        instruction: tl.constexpr = (
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
            "{$0,$1,$2,$3,$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15}, "
            "{$16,$17,$18,$19,$20,$21,$22,$23};"
        )
    else:
        instruction: tl.constexpr = (
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
            "{$0,$1,$2,$3,$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15}, "
            "{$16,$17,$18,$19,$20,$21,$22,$23};"
        )
    return tl.inline_asm_elementwise(
        instruction,
        constraints="=f,=f,=f,=f,=f,=f,=f,=f,r,r,r,r,r,r,r,r,f,f,f,f,f,f,f,f",
        args=(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7),
        dtype=(
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
            tl.float32,
        ),
        is_pure=False,
        pack=1,
    )


@triton.jit
def q_fragment(BUF, col, token, TRANS: tl.constexpr, ROW):
    zero = tl.zeros((128,), tl.int32)
    cube = col // 64
    base = gpu.local_ptr(BUF, ((ROW // 32 * 4 + cube) * 32, zero)).to(tl.uint64)
    base = base + token.to(tl.uint64)
    return tsm_fragment(base, ROW % 32, col % 64, TRANS)


@triton.jit
def load_q(Q, BUF, qstart, qlen, tile, group, QS: tl.constexpr, QH: tl.constexpr):
    warp = tl.arange(0, 128) // 32
    fragments = ()
    for index in tl.static_range(64):
        fragments += (tl.zeros((128,), tl.uint32),)
    if qlen > 0:
        for pair in tl.static_range(2):
            issue_copy(
                Q,
                BUF,
                qstart + tile * 16,
                qstart + qlen,
                QS,
                0,
                (group * 4 + pair * 2) * QH,
            )
            issue_copy(
                Q,
                BUF,
                qstart + tile * 16,
                qstart + qlen,
                QS,
                32,
                (group * 4 + pair * 2 + 1) * QH,
            )
            commit_copies()
            token = wait_copies() + tl.zeros((128,), tl.int32)
            updated = ()
            for column in tl.static_range(16):
                fragment = q_fragment(BUF, column * 16, token, False, ROW=warp % 2 * 32)
                for word in tl.static_range(4):
                    updated += (
                        tl.where(
                            warp // 2 == pair,
                            fragment[word],
                            fragments[column * 4 + word],
                        ),
                    )
            fragments = updated
            wait_copies()
    return fragments
