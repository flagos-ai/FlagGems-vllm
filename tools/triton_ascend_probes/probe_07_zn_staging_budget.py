"""GAP F6: ND -> zN staging for an L1-resident dot operand.

F1 is now solved at the API level: 4D zN operands in L1 are consumable via
`al.dot(..., format_a/b="fractal")` (see probe_01).  But the only way to build
a zN tile from a 2D ND tile is a UB-side layout conversion

    a_tmp = tl.permute(a.reshape(R, K // 16, 16), (1, 0, 2))
    a_nz = a_tmp.reshape(K // 16, R // 16, 16, 16)
    al.copy(bl.to_buffer(a_nz, al.ascend_address_space.UB), l1)

and that conversion materialises a second 4D tile in UB.  In the KDA Gram
kernel each program builds seven [64, 128] bf16 panels and runs six dots, and
the extra UB tiles push the kernel over the 2.03 Mbit budget:

    MODE=plain  (six tl.dot, 7 panels)                       compiles
    MODE=l1a    (4 A panels staged in L1, B stays ND)        ub 3.15 Mbit
    MODE=l1b    (3 shared B panels staged in L1)             ub 3.28 Mbit
    MODE=l1     (all 7 panels staged in L1)                  ub 2.37 Mbit
    MODE=l1 two chunks (14 panels in L1)                     ub 3.94 Mbit

What we need instead is a fused staging copy, e.g. an `al.copy` that accepts
a 2D ND UB tile and a 4D zN L1 destination (`ND2NZ`), or `al.dot` accepting
2D ND operands with an internal L1 staging that reuses the compiler's
existing (UB-cheap) ND->zN path.  Then explicit L1 residency becomes usable
for multi-panel kernels, which is what the cross-chunk pipeline (F4) needs.

Usage:
    PYTHONPATH=src:FlagGems/src python probe_07_zn_staging_budget.py plain
    PYTHONPATH=src:FlagGems/src python probe_07_zn_staging_budget.py l1a
    PYTHONPATH=src:FlagGems/src python probe_07_zn_staging_budget.py l1b
    PYTHONPATH=src:FlagGems/src python probe_07_zn_staging_budget.py l1
"""

import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def zN_A(a, R: tl.constexpr, K: tl.constexpr):
    t = tl.permute(a.reshape(R, K // 16, 16), (1, 0, 2))
    return t.reshape(K // 16, R // 16, 16, 16)


@triton.jit
def zN_B(a, R: tl.constexpr, K: tl.constexpr):
    at = tl.trans(a)
    t = tl.permute(at.reshape(K, R // 16, 16), (1, 0, 2))
    return t.reshape(R // 16, K // 16, 16, 16)


@triton.jit
def kern(
    x,
    y,
    R: tl.constexpr,
    K: tl.constexpr,
    DO_A: tl.constexpr,
    DO_B: tl.constexpr,
    FA: tl.constexpr,
    FB: tl.constexpr,
):
    pid = tl.program_id(0)
    o_i = tl.arange(0, R)
    o_k = tl.arange(0, K)
    p = (pid % 6) * R * K + o_i[:, None] * K + o_k[None, :]
    a = tl.load(x + p)
    b = tl.load(x + R * K + p)
    gc = tl.load(x + 2 * R * K + p).to(tl.float32)
    H: tl.constexpr = R // 2
    in0 = o_i[:, None] < H
    f0 = tl.where(in0, gc, 0.0)
    f1 = tl.where(in0, gc * 1.001, -0.5)
    q0 = (a * tl.math.exp2(f0)).to(tl.bfloat16)
    q1 = (a * tl.math.exp2(f1)).to(tl.bfloat16)
    k0 = (b * tl.math.exp2(-f0)).to(tl.bfloat16)
    k1 = (b * tl.math.exp2(-f1)).to(tl.bfloat16)
    k0c = (b * tl.math.exp2(-f0) * 1.002).to(tl.bfloat16)
    kq0 = (b * tl.math.exp2(f0)).to(tl.bfloat16)
    kq1 = (b * tl.math.exp2(f1)).to(tl.bfloat16)

    if not DO_A and not DO_B:
        aq = tl.dot(q0, tl.trans(k0), out_dtype=tl.float32)
        aq += tl.dot(q1, tl.trans(k1), out_dtype=tl.float32)
        aq += tl.dot(q1, tl.trans(k0c), out_dtype=tl.float32)
        ak = tl.dot(kq0, tl.trans(k0), out_dtype=tl.float32)
        ak += tl.dot(kq1, tl.trans(k1), out_dtype=tl.float32)
        ak += tl.dot(kq1, tl.trans(k0c), out_dtype=tl.float32)
    else:
        if DO_B:
            lk0 = bl.alloc(
                tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
            )
            lk1 = bl.alloc(
                tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
            )
            lkc = bl.alloc(
                tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
            )
            al.copy(bl.to_buffer(zN_B(k0, R, K), al.ascend_address_space.UB), lk0)
            al.copy(bl.to_buffer(zN_B(k1, R, K), al.ascend_address_space.UB), lk1)
            al.copy(bl.to_buffer(zN_B(k0c, R, K), al.ascend_address_space.UB), lkc)
            tk0 = bl.to_tensor(lk0)
            tk1 = bl.to_tensor(lk1)
            tkc = bl.to_tensor(lkc)
        if DO_A:
            lq0 = bl.alloc(
                tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
            )
            lq1 = bl.alloc(
                tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
            )
            lkq0 = bl.alloc(
                tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
            )
            lkq1 = bl.alloc(
                tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
            )
            al.copy(bl.to_buffer(zN_A(q0, R, K), al.ascend_address_space.UB), lq0)
            al.copy(bl.to_buffer(zN_A(q1, R, K), al.ascend_address_space.UB), lq1)
            al.copy(bl.to_buffer(zN_A(kq0, R, K), al.ascend_address_space.UB), lkq0)
            al.copy(bl.to_buffer(zN_A(kq1, R, K), al.ascend_address_space.UB), lkq1)
            tq0 = bl.to_tensor(lq0)
            tq1 = bl.to_tensor(lq1)
            tkq0 = bl.to_tensor(lkq0)
            tkq1 = bl.to_tensor(lkq1)
        a0 = tq0 if DO_A else q0
        a1 = tq1 if DO_A else q1
        ak0 = tkq0 if DO_A else kq0
        ak1 = tkq1 if DO_A else kq1
        b0 = tk0 if DO_B else tl.trans(k0)
        b1 = tk1 if DO_B else tl.trans(k1)
        bc = tkc if DO_B else tl.trans(k0c)
        aq = al.dot(a0, b0, format_a=FA, format_b=FB)
        aq += al.dot(a1, b1, format_a=FA, format_b=FB)
        aq += al.dot(a1, bc, format_a=FA, format_b=FB)
        ak = al.dot(ak0, b0, format_a=FA, format_b=FB)
        ak += al.dot(ak1, b1, format_a=FA, format_b=FB)
        ak += al.dot(ak1, bc, format_a=FA, format_b=FB)
    tl.store(y + o_i[:, None] * R + o_i[None, :], aq + ak)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plain"
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    x = torch.randn(6 * R * K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R * R, device="npu", dtype=torch.float32)
    grid = (12288,)  # one program per (chunk, head) of the KDA Gram grid
    do_a, do_b, fa, fb = {
        "plain": (0, 0, "nd", "nd"),
        "l1a": (1, 0, "fractal", "nd"),
        "l1b": (0, 1, "nd", "fractal"),
        "l1": (1, 1, "fractal", "fractal"),
    }[mode]
    with torch.inference_mode():
        kern[grid](x, y, R, K, DO_A=do_a, DO_B=do_b, FA=fa, FB=fb)
        torch.npu.synchronize()
    print(f"RESULT: PASS probe_07_zn_staging_budget mode={mode}")


if __name__ == "__main__":
    main()
