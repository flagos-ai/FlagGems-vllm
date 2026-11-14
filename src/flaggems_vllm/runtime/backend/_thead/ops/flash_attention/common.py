# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from triton.experimental.tle.language import gpu


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
    d0, d1, d2, d3 = tl.inline_asm_elementwise(
        (
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.bf16.bf16.bf16.bf16 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15}"
            ";"
        ),
        constraints="=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=(ap0, ap1, ap2, ap3, w, z, z, w, z, z, z, z),
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )
    bits0 = d0.to(tl.uint32, bitcast=True)
    v0 = (bits0 & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    v1 = (bits0 >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    bits1 = d1.to(tl.uint32, bitcast=True)
    v2 = (bits1 & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    v3 = (bits1 >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    bits2 = d2.to(tl.uint32, bitcast=True)
    v4 = (bits2 & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    v5 = (bits2 >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    bits3 = d3.to(tl.uint32, bitcast=True)
    v6 = (bits3 & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    v7 = (bits3 >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return (v0, v1, v2, v3, v4, v5, v6, v7)


@triton.jit
def mma_packed(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, c4, c5, c6, c7):
    return tl.inline_asm_elementwise(
        (
            "ppu.tc01.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 {$"
            "0,$1,$2,$3,$4,$5,$6,$7}, {$8,$9,$10,$11}, {$12,$13,$14,$15}, {$1"
            "6,$17,$18,$19,$20,$21,$22,$23};"
        ),
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
    thread = tl.arange(0, 128)
    warp = thread // 32
    qp0_0 = tl.zeros((128,), tl.uint32)
    qp0_1 = tl.zeros((128,), tl.uint32)
    qp0_2 = tl.zeros((128,), tl.uint32)
    qp0_3 = tl.zeros((128,), tl.uint32)
    qp1_0 = tl.zeros((128,), tl.uint32)
    qp1_1 = tl.zeros((128,), tl.uint32)
    qp1_2 = tl.zeros((128,), tl.uint32)
    qp1_3 = tl.zeros((128,), tl.uint32)
    qp2_0 = tl.zeros((128,), tl.uint32)
    qp2_1 = tl.zeros((128,), tl.uint32)
    qp2_2 = tl.zeros((128,), tl.uint32)
    qp2_3 = tl.zeros((128,), tl.uint32)
    qp3_0 = tl.zeros((128,), tl.uint32)
    qp3_1 = tl.zeros((128,), tl.uint32)
    qp3_2 = tl.zeros((128,), tl.uint32)
    qp3_3 = tl.zeros((128,), tl.uint32)
    qp4_0 = tl.zeros((128,), tl.uint32)
    qp4_1 = tl.zeros((128,), tl.uint32)
    qp4_2 = tl.zeros((128,), tl.uint32)
    qp4_3 = tl.zeros((128,), tl.uint32)
    qp5_0 = tl.zeros((128,), tl.uint32)
    qp5_1 = tl.zeros((128,), tl.uint32)
    qp5_2 = tl.zeros((128,), tl.uint32)
    qp5_3 = tl.zeros((128,), tl.uint32)
    qp6_0 = tl.zeros((128,), tl.uint32)
    qp6_1 = tl.zeros((128,), tl.uint32)
    qp6_2 = tl.zeros((128,), tl.uint32)
    qp6_3 = tl.zeros((128,), tl.uint32)
    qp7_0 = tl.zeros((128,), tl.uint32)
    qp7_1 = tl.zeros((128,), tl.uint32)
    qp7_2 = tl.zeros((128,), tl.uint32)
    qp7_3 = tl.zeros((128,), tl.uint32)
    qp8_0 = tl.zeros((128,), tl.uint32)
    qp8_1 = tl.zeros((128,), tl.uint32)
    qp8_2 = tl.zeros((128,), tl.uint32)
    qp8_3 = tl.zeros((128,), tl.uint32)
    qp9_0 = tl.zeros((128,), tl.uint32)
    qp9_1 = tl.zeros((128,), tl.uint32)
    qp9_2 = tl.zeros((128,), tl.uint32)
    qp9_3 = tl.zeros((128,), tl.uint32)
    qp10_0 = tl.zeros((128,), tl.uint32)
    qp10_1 = tl.zeros((128,), tl.uint32)
    qp10_2 = tl.zeros((128,), tl.uint32)
    qp10_3 = tl.zeros((128,), tl.uint32)
    qp11_0 = tl.zeros((128,), tl.uint32)
    qp11_1 = tl.zeros((128,), tl.uint32)
    qp11_2 = tl.zeros((128,), tl.uint32)
    qp11_3 = tl.zeros((128,), tl.uint32)
    qp12_0 = tl.zeros((128,), tl.uint32)
    qp12_1 = tl.zeros((128,), tl.uint32)
    qp12_2 = tl.zeros((128,), tl.uint32)
    qp12_3 = tl.zeros((128,), tl.uint32)
    qp13_0 = tl.zeros((128,), tl.uint32)
    qp13_1 = tl.zeros((128,), tl.uint32)
    qp13_2 = tl.zeros((128,), tl.uint32)
    qp13_3 = tl.zeros((128,), tl.uint32)
    qp14_0 = tl.zeros((128,), tl.uint32)
    qp14_1 = tl.zeros((128,), tl.uint32)
    qp14_2 = tl.zeros((128,), tl.uint32)
    qp14_3 = tl.zeros((128,), tl.uint32)
    qp15_0 = tl.zeros((128,), tl.uint32)
    qp15_1 = tl.zeros((128,), tl.uint32)
    qp15_2 = tl.zeros((128,), tl.uint32)
    qp15_3 = tl.zeros((128,), tl.uint32)
    if qlen > 0:
        issue_copy(
            Q, BUF, qstart + tile * 16, qstart + qlen, QS, 0, (group * 4 + 0) * QH
        )
        issue_copy(
            Q, BUF, qstart + tile * 16, qstart + qlen, QS, 32, (group * 4 + 1) * QH
        )
        commit_copies()
        token = wait_copies() + tl.zeros((128,), tl.int32)
        a0, a1, a2, a3 = q_fragment(BUF, 0, token, False, ROW=warp % 2 * 32)
        qp0_0 = tl.where(warp // 2 == 0, a0, qp0_0)
        qp0_1 = tl.where(warp // 2 == 0, a1, qp0_1)
        qp0_2 = tl.where(warp // 2 == 0, a2, qp0_2)
        qp0_3 = tl.where(warp // 2 == 0, a3, qp0_3)
        a0, a1, a2, a3 = q_fragment(BUF, 16, token, False, ROW=warp % 2 * 32)
        qp1_0 = tl.where(warp // 2 == 0, a0, qp1_0)
        qp1_1 = tl.where(warp // 2 == 0, a1, qp1_1)
        qp1_2 = tl.where(warp // 2 == 0, a2, qp1_2)
        qp1_3 = tl.where(warp // 2 == 0, a3, qp1_3)
        a0, a1, a2, a3 = q_fragment(BUF, 32, token, False, ROW=warp % 2 * 32)
        qp2_0 = tl.where(warp // 2 == 0, a0, qp2_0)
        qp2_1 = tl.where(warp // 2 == 0, a1, qp2_1)
        qp2_2 = tl.where(warp // 2 == 0, a2, qp2_2)
        qp2_3 = tl.where(warp // 2 == 0, a3, qp2_3)
        a0, a1, a2, a3 = q_fragment(BUF, 48, token, False, ROW=warp % 2 * 32)
        qp3_0 = tl.where(warp // 2 == 0, a0, qp3_0)
        qp3_1 = tl.where(warp // 2 == 0, a1, qp3_1)
        qp3_2 = tl.where(warp // 2 == 0, a2, qp3_2)
        qp3_3 = tl.where(warp // 2 == 0, a3, qp3_3)
        a0, a1, a2, a3 = q_fragment(BUF, 64, token, False, ROW=warp % 2 * 32)
        qp4_0 = tl.where(warp // 2 == 0, a0, qp4_0)
        qp4_1 = tl.where(warp // 2 == 0, a1, qp4_1)
        qp4_2 = tl.where(warp // 2 == 0, a2, qp4_2)
        qp4_3 = tl.where(warp // 2 == 0, a3, qp4_3)
        a0, a1, a2, a3 = q_fragment(BUF, 80, token, False, ROW=warp % 2 * 32)
        qp5_0 = tl.where(warp // 2 == 0, a0, qp5_0)
        qp5_1 = tl.where(warp // 2 == 0, a1, qp5_1)
        qp5_2 = tl.where(warp // 2 == 0, a2, qp5_2)
        qp5_3 = tl.where(warp // 2 == 0, a3, qp5_3)
        a0, a1, a2, a3 = q_fragment(BUF, 96, token, False, ROW=warp % 2 * 32)
        qp6_0 = tl.where(warp // 2 == 0, a0, qp6_0)
        qp6_1 = tl.where(warp // 2 == 0, a1, qp6_1)
        qp6_2 = tl.where(warp // 2 == 0, a2, qp6_2)
        qp6_3 = tl.where(warp // 2 == 0, a3, qp6_3)
        a0, a1, a2, a3 = q_fragment(BUF, 112, token, False, ROW=warp % 2 * 32)
        qp7_0 = tl.where(warp // 2 == 0, a0, qp7_0)
        qp7_1 = tl.where(warp // 2 == 0, a1, qp7_1)
        qp7_2 = tl.where(warp // 2 == 0, a2, qp7_2)
        qp7_3 = tl.where(warp // 2 == 0, a3, qp7_3)
        a0, a1, a2, a3 = q_fragment(BUF, 128, token, False, ROW=warp % 2 * 32)
        qp8_0 = tl.where(warp // 2 == 0, a0, qp8_0)
        qp8_1 = tl.where(warp // 2 == 0, a1, qp8_1)
        qp8_2 = tl.where(warp // 2 == 0, a2, qp8_2)
        qp8_3 = tl.where(warp // 2 == 0, a3, qp8_3)
        a0, a1, a2, a3 = q_fragment(BUF, 144, token, False, ROW=warp % 2 * 32)
        qp9_0 = tl.where(warp // 2 == 0, a0, qp9_0)
        qp9_1 = tl.where(warp // 2 == 0, a1, qp9_1)
        qp9_2 = tl.where(warp // 2 == 0, a2, qp9_2)
        qp9_3 = tl.where(warp // 2 == 0, a3, qp9_3)
        a0, a1, a2, a3 = q_fragment(BUF, 160, token, False, ROW=warp % 2 * 32)
        qp10_0 = tl.where(warp // 2 == 0, a0, qp10_0)
        qp10_1 = tl.where(warp // 2 == 0, a1, qp10_1)
        qp10_2 = tl.where(warp // 2 == 0, a2, qp10_2)
        qp10_3 = tl.where(warp // 2 == 0, a3, qp10_3)
        a0, a1, a2, a3 = q_fragment(BUF, 176, token, False, ROW=warp % 2 * 32)
        qp11_0 = tl.where(warp // 2 == 0, a0, qp11_0)
        qp11_1 = tl.where(warp // 2 == 0, a1, qp11_1)
        qp11_2 = tl.where(warp // 2 == 0, a2, qp11_2)
        qp11_3 = tl.where(warp // 2 == 0, a3, qp11_3)
        a0, a1, a2, a3 = q_fragment(BUF, 192, token, False, ROW=warp % 2 * 32)
        qp12_0 = tl.where(warp // 2 == 0, a0, qp12_0)
        qp12_1 = tl.where(warp // 2 == 0, a1, qp12_1)
        qp12_2 = tl.where(warp // 2 == 0, a2, qp12_2)
        qp12_3 = tl.where(warp // 2 == 0, a3, qp12_3)
        a0, a1, a2, a3 = q_fragment(BUF, 208, token, False, ROW=warp % 2 * 32)
        qp13_0 = tl.where(warp // 2 == 0, a0, qp13_0)
        qp13_1 = tl.where(warp // 2 == 0, a1, qp13_1)
        qp13_2 = tl.where(warp // 2 == 0, a2, qp13_2)
        qp13_3 = tl.where(warp // 2 == 0, a3, qp13_3)
        a0, a1, a2, a3 = q_fragment(BUF, 224, token, False, ROW=warp % 2 * 32)
        qp14_0 = tl.where(warp // 2 == 0, a0, qp14_0)
        qp14_1 = tl.where(warp // 2 == 0, a1, qp14_1)
        qp14_2 = tl.where(warp // 2 == 0, a2, qp14_2)
        qp14_3 = tl.where(warp // 2 == 0, a3, qp14_3)
        a0, a1, a2, a3 = q_fragment(BUF, 240, token, False, ROW=warp % 2 * 32)
        qp15_0 = tl.where(warp // 2 == 0, a0, qp15_0)
        qp15_1 = tl.where(warp // 2 == 0, a1, qp15_1)
        qp15_2 = tl.where(warp // 2 == 0, a2, qp15_2)
        qp15_3 = tl.where(warp // 2 == 0, a3, qp15_3)
        wait_copies()
        issue_copy(
            Q, BUF, qstart + tile * 16, qstart + qlen, QS, 0, (group * 4 + 2) * QH
        )
        issue_copy(
            Q, BUF, qstart + tile * 16, qstart + qlen, QS, 32, (group * 4 + 3) * QH
        )
        commit_copies()
        token = wait_copies() + tl.zeros((128,), tl.int32)
        a0, a1, a2, a3 = q_fragment(BUF, 0, token, False, ROW=warp % 2 * 32)
        qp0_0 = tl.where(warp // 2 == 1, a0, qp0_0)
        qp0_1 = tl.where(warp // 2 == 1, a1, qp0_1)
        qp0_2 = tl.where(warp // 2 == 1, a2, qp0_2)
        qp0_3 = tl.where(warp // 2 == 1, a3, qp0_3)
        a0, a1, a2, a3 = q_fragment(BUF, 16, token, False, ROW=warp % 2 * 32)
        qp1_0 = tl.where(warp // 2 == 1, a0, qp1_0)
        qp1_1 = tl.where(warp // 2 == 1, a1, qp1_1)
        qp1_2 = tl.where(warp // 2 == 1, a2, qp1_2)
        qp1_3 = tl.where(warp // 2 == 1, a3, qp1_3)
        a0, a1, a2, a3 = q_fragment(BUF, 32, token, False, ROW=warp % 2 * 32)
        qp2_0 = tl.where(warp // 2 == 1, a0, qp2_0)
        qp2_1 = tl.where(warp // 2 == 1, a1, qp2_1)
        qp2_2 = tl.where(warp // 2 == 1, a2, qp2_2)
        qp2_3 = tl.where(warp // 2 == 1, a3, qp2_3)
        a0, a1, a2, a3 = q_fragment(BUF, 48, token, False, ROW=warp % 2 * 32)
        qp3_0 = tl.where(warp // 2 == 1, a0, qp3_0)
        qp3_1 = tl.where(warp // 2 == 1, a1, qp3_1)
        qp3_2 = tl.where(warp // 2 == 1, a2, qp3_2)
        qp3_3 = tl.where(warp // 2 == 1, a3, qp3_3)
        a0, a1, a2, a3 = q_fragment(BUF, 64, token, False, ROW=warp % 2 * 32)
        qp4_0 = tl.where(warp // 2 == 1, a0, qp4_0)
        qp4_1 = tl.where(warp // 2 == 1, a1, qp4_1)
        qp4_2 = tl.where(warp // 2 == 1, a2, qp4_2)
        qp4_3 = tl.where(warp // 2 == 1, a3, qp4_3)
        a0, a1, a2, a3 = q_fragment(BUF, 80, token, False, ROW=warp % 2 * 32)
        qp5_0 = tl.where(warp // 2 == 1, a0, qp5_0)
        qp5_1 = tl.where(warp // 2 == 1, a1, qp5_1)
        qp5_2 = tl.where(warp // 2 == 1, a2, qp5_2)
        qp5_3 = tl.where(warp // 2 == 1, a3, qp5_3)
        a0, a1, a2, a3 = q_fragment(BUF, 96, token, False, ROW=warp % 2 * 32)
        qp6_0 = tl.where(warp // 2 == 1, a0, qp6_0)
        qp6_1 = tl.where(warp // 2 == 1, a1, qp6_1)
        qp6_2 = tl.where(warp // 2 == 1, a2, qp6_2)
        qp6_3 = tl.where(warp // 2 == 1, a3, qp6_3)
        a0, a1, a2, a3 = q_fragment(BUF, 112, token, False, ROW=warp % 2 * 32)
        qp7_0 = tl.where(warp // 2 == 1, a0, qp7_0)
        qp7_1 = tl.where(warp // 2 == 1, a1, qp7_1)
        qp7_2 = tl.where(warp // 2 == 1, a2, qp7_2)
        qp7_3 = tl.where(warp // 2 == 1, a3, qp7_3)
        a0, a1, a2, a3 = q_fragment(BUF, 128, token, False, ROW=warp % 2 * 32)
        qp8_0 = tl.where(warp // 2 == 1, a0, qp8_0)
        qp8_1 = tl.where(warp // 2 == 1, a1, qp8_1)
        qp8_2 = tl.where(warp // 2 == 1, a2, qp8_2)
        qp8_3 = tl.where(warp // 2 == 1, a3, qp8_3)
        a0, a1, a2, a3 = q_fragment(BUF, 144, token, False, ROW=warp % 2 * 32)
        qp9_0 = tl.where(warp // 2 == 1, a0, qp9_0)
        qp9_1 = tl.where(warp // 2 == 1, a1, qp9_1)
        qp9_2 = tl.where(warp // 2 == 1, a2, qp9_2)
        qp9_3 = tl.where(warp // 2 == 1, a3, qp9_3)
        a0, a1, a2, a3 = q_fragment(BUF, 160, token, False, ROW=warp % 2 * 32)
        qp10_0 = tl.where(warp // 2 == 1, a0, qp10_0)
        qp10_1 = tl.where(warp // 2 == 1, a1, qp10_1)
        qp10_2 = tl.where(warp // 2 == 1, a2, qp10_2)
        qp10_3 = tl.where(warp // 2 == 1, a3, qp10_3)
        a0, a1, a2, a3 = q_fragment(BUF, 176, token, False, ROW=warp % 2 * 32)
        qp11_0 = tl.where(warp // 2 == 1, a0, qp11_0)
        qp11_1 = tl.where(warp // 2 == 1, a1, qp11_1)
        qp11_2 = tl.where(warp // 2 == 1, a2, qp11_2)
        qp11_3 = tl.where(warp // 2 == 1, a3, qp11_3)
        a0, a1, a2, a3 = q_fragment(BUF, 192, token, False, ROW=warp % 2 * 32)
        qp12_0 = tl.where(warp // 2 == 1, a0, qp12_0)
        qp12_1 = tl.where(warp // 2 == 1, a1, qp12_1)
        qp12_2 = tl.where(warp // 2 == 1, a2, qp12_2)
        qp12_3 = tl.where(warp // 2 == 1, a3, qp12_3)
        a0, a1, a2, a3 = q_fragment(BUF, 208, token, False, ROW=warp % 2 * 32)
        qp13_0 = tl.where(warp // 2 == 1, a0, qp13_0)
        qp13_1 = tl.where(warp // 2 == 1, a1, qp13_1)
        qp13_2 = tl.where(warp // 2 == 1, a2, qp13_2)
        qp13_3 = tl.where(warp // 2 == 1, a3, qp13_3)
        a0, a1, a2, a3 = q_fragment(BUF, 224, token, False, ROW=warp % 2 * 32)
        qp14_0 = tl.where(warp // 2 == 1, a0, qp14_0)
        qp14_1 = tl.where(warp // 2 == 1, a1, qp14_1)
        qp14_2 = tl.where(warp // 2 == 1, a2, qp14_2)
        qp14_3 = tl.where(warp // 2 == 1, a3, qp14_3)
        a0, a1, a2, a3 = q_fragment(BUF, 240, token, False, ROW=warp % 2 * 32)
        qp15_0 = tl.where(warp // 2 == 1, a0, qp15_0)
        qp15_1 = tl.where(warp // 2 == 1, a1, qp15_1)
        qp15_2 = tl.where(warp // 2 == 1, a2, qp15_2)
        qp15_3 = tl.where(warp // 2 == 1, a3, qp15_3)
        wait_copies()
    return (
        qp0_0,
        qp0_1,
        qp0_2,
        qp0_3,
        qp1_0,
        qp1_1,
        qp1_2,
        qp1_3,
        qp2_0,
        qp2_1,
        qp2_2,
        qp2_3,
        qp3_0,
        qp3_1,
        qp3_2,
        qp3_3,
        qp4_0,
        qp4_1,
        qp4_2,
        qp4_3,
        qp5_0,
        qp5_1,
        qp5_2,
        qp5_3,
        qp6_0,
        qp6_1,
        qp6_2,
        qp6_3,
        qp7_0,
        qp7_1,
        qp7_2,
        qp7_3,
        qp8_0,
        qp8_1,
        qp8_2,
        qp8_3,
        qp9_0,
        qp9_1,
        qp9_2,
        qp9_3,
        qp10_0,
        qp10_1,
        qp10_2,
        qp10_3,
        qp11_0,
        qp11_1,
        qp11_2,
        qp11_3,
        qp12_0,
        qp12_1,
        qp12_2,
        qp12_3,
        qp13_0,
        qp13_1,
        qp13_2,
        qp13_3,
        qp14_0,
        qp14_1,
        qp14_2,
        qp14_3,
        qp15_0,
        qp15_1,
        qp15_2,
        qp15_3,
    )
