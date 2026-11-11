"""DEFECT: a multi-dot body inside `al.scope("cube")` computes wrong values.

Isolation performed on Ascend950DT_9582 / triton-ascend 3.6.0:

1. TTIR is ordered correctly: `hivm.hir.sync_block_wait` precedes all seven
   `tt.load`s of the panel workspace.
2. The cube stream loads exactly the panels the vector stream wrote: a debug
   store of the loaded panels matches the workspace bit-for-bit (maxerr=0).
3. The identical six-dot + scale/mask/inverse body *outside* any scope is
   correct against the torch reference.
4. Inside `al.scope("cube")` the same body returns wrong values.  Smaller
   scoped bodies (1-3 dots, few stores) still pass, so this is not a sync or
   load-visibility problem: it is scoped-cube code generation for the larger
   multi-dot body (likely UB/L0 buffer allocation or scheduling across dots).

`enable_mixed_cv=True, disable_auto_inject_block_sync=True` is required just
to compile the scope; the default options crash (defects/d1).

Usage (from the repo root):
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/defects/d5_scoped_cube_multidot.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def cube_body(panels, beta, Aqk, Akk, scale, HALF: tl.constexpr, K: tl.constexpr,
              BT: tl.constexpr):
    o_h = tl.arange(0, HALF)
    o_k = tl.arange(0, K)
    base = o_h[:, None] * K + o_k[None, :]
    qP0 = tl.load(panels + 0 * HALF * K + base)
    qP1 = tl.load(panels + 1 * HALF * K + base)
    kP0 = tl.load(panels + 2 * HALF * K + base)
    kP1 = tl.load(panels + 3 * HALF * K + base)
    kM0 = tl.load(panels + 4 * HALF * K + base)
    kM10 = tl.load(panels + 5 * HALF * K + base)
    kM11 = tl.load(panels + 6 * HALF * K + base)

    r0q = tl.dot(qP0, tl.trans(kM0), out_dtype=tl.float32)
    r10q = tl.dot(qP1, tl.trans(kM10), out_dtype=tl.float32)
    r11q = tl.dot(qP1, tl.trans(kM11), out_dtype=tl.float32)
    r0k = tl.dot(kP0, tl.trans(kM0), out_dtype=tl.float32)
    r10k = tl.dot(kP1, tl.trans(kM10), out_dtype=tl.float32)
    r11k = tl.dot(kP1, tl.trans(kM11), out_dtype=tl.float32)

    b0 = tl.sigmoid(tl.load(beta + o_h).to(tl.float32))
    b1 = tl.sigmoid(tl.load(beta + HALF + o_h).to(tl.float32))
    lo = o_h[:, None] >= o_h[None, :]
    strict = o_h[:, None] > o_h[None, :]
    diag = o_h[:, None] == o_h[None, :]
    aq0 = tl.where(lo, r0q * scale, 0.0)
    aq10 = r10q * scale
    aq11 = tl.where(lo, r11q * scale, 0.0)
    l0 = tl.where(strict, r0k * b0[:, None], 0.0)
    l10 = r10k * b1[:, None]
    l11 = tl.where(strict, r11k * b1[:, None], 0.0)
    i0 = diag.to(tl.float32) - l0
    i11 = diag.to(tl.float32) - l11

    st = Aqk.dtype.element_ty
    r0 = o_h[:, None] * BT + o_h[None, :]
    r0h = o_h[:, None] * BT + HALF + o_h[None, :]
    r1 = (HALF + o_h[:, None]) * BT + o_h[None, :]
    r1h = (HALF + o_h[:, None]) * BT + HALF + o_h[None, :]
    z = tl.zeros((HALF, HALF), tl.float32).to(st)
    tl.store(Aqk + r0, aq0.to(st))
    tl.store(Aqk + r0h, z)
    tl.store(Aqk + r1, aq10.to(st))
    tl.store(Aqk + r1h, aq11.to(st))
    tl.store(Akk + r0, i0.to(st))
    tl.store(Akk + r0h, z)
    tl.store(Akk + r1, (-l10).to(st))
    tl.store(Akk + r1h, i11.to(st))


@triton.jit
def kern_unscoped(panels, beta, Aqk, Akk, scale, HALF: tl.constexpr,
                  K: tl.constexpr, BT: tl.constexpr):
    cube_body(panels, beta, Aqk, Akk, scale, HALF, K, BT)


@triton.jit
def kern_scoped(panels, beta, Aqk, Akk, scale, HALF: tl.constexpr,
                K: tl.constexpr, BT: tl.constexpr):
    o_h = tl.arange(0, HALF)
    with al.scope("cube"):
        cube_body(panels, beta, Aqk, Akk, scale, HALF, K, BT)


def main():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    HALF, K, BT = 32, 128, 64
    # Normal-magnitude panels: the failure is not caused by huge gate factors.
    panels = torch.randn(7, HALF, K, device="npu", dtype=torch.bfloat16) * 0.01
    beta = torch.rand(2 * HALF, device="npu", dtype=torch.float32)
    scale = K**-0.5
    ref = torch.zeros(1, 1, BT, BT, device="npu", dtype=torch.bfloat16)
    refk = torch.zeros_like(ref)
    got = torch.zeros_like(ref)
    gotk = torch.zeros_like(ref)
    with torch.inference_mode():
        kern_unscoped[(1,)](panels, beta, ref, refk, scale, HALF=HALF, K=K, BT=BT)
        kern_scoped[(1,)](
            panels,
            beta,
            got,
            gotk,
            scale,
            HALF=HALF,
            K=K,
            BT=BT,
            enable_mixed_cv=True,
            disable_auto_inject_block_sync=True,
        )
        torch.npu.synchronize()
    # Torch reference for the same six dots: proves the unscoped body is
    # correct and that the discrepancy is introduced by the scope.
    P = [panels[i].float() for i in range(7)]
    lo = torch.arange(HALF, device="npu")[:, None] >= torch.arange(
        HALF, device="npu"
    )[None, :]
    strict = torch.arange(HALF, device="npu")[:, None] > torch.arange(
        HALF, device="npu"
    )[None, :]
    diag = torch.eye(HALF, device="npu", dtype=torch.bool)
    b0 = torch.sigmoid(beta[:HALF])
    b1 = torch.sigmoid(beta[HALF:])
    r0q = P[0] @ P[4].T
    r10q = P[1] @ P[5].T
    r11q = P[1] @ P[6].T
    r0k = P[2] @ P[4].T
    r10k = P[3] @ P[5].T
    r11k = P[3] @ P[6].T
    aq = torch.zeros(BT, BT, device="npu")
    ak = torch.zeros(BT, BT, device="npu")
    aq[:HALF, :HALF] = torch.where(lo, r0q * scale, torch.zeros_like(r0q))
    aq[HALF:, :HALF] = r10q * scale
    aq[HALF:, HALF:] = torch.where(lo, r11q * scale, torch.zeros_like(r11q))
    l0 = torch.where(strict, r0k * b0[:, None], torch.zeros_like(r0k))
    l10 = r10k * b1[:, None]
    l11 = torch.where(strict, r11k * b1[:, None], torch.zeros_like(r11k))
    ak[:HALF, :HALF] = diag.float() - l0
    ak[HALF:, :HALF] = -l10
    ak[HALF:, HALF:] = diag.float() - l11

    ref_err = max(
        (ref.float()[0, 0] - aq).abs().max().item(),
        (refk.float()[0, 0] - ak).abs().max().item(),
    )
    dq = (got.float() - ref.float()).abs().max().item()
    dk = (gotk.float() - refk.float()).abs().max().item()
    verdict = "PASS" if max(dq, dk) < 1e-2 else "FAIL"
    print(
        f"RESULT: {verdict} d5_scoped_cube_multidot "
        f"unscoped_err={ref_err:.6g} aqk_err={dq:.6g} akk_err={dk:.6g}"
    )


if __name__ == "__main__":
    main()
