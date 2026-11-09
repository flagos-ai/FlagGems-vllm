"""GAP F1: a bf16 dot operand that stays resident in L1.

AscendC chunk_kda_fwd_prepare keeps the Q+/K+/K- panels of 4 heads in L1 and
lets every Gram dot read them there.  The Triton front door for that is

    l1 = bl.alloc(bf16, [BT, K], al.ascend_address_space.L1)
    ... fill from UB ...
    a = bl.to_tensor(l1)
    c = tl.dot(a, tl.trans(a))

Observed on Ascend950DT_9582 / triton-ascend 3.6.0.post2:

  MODE=copy  fill UB with bind_buffer + al.copy(ub, l1)    -> FAIL
      'memref.alloc' op error: read before first write
  MODE=dot   ... + bl.to_tensor(l1) + tl.dot               -> FAIL (last mile)
      different memory spaces specified for base memref type
      'memref<64x128xbf16, #hivm.address_space<cbuf>>' and subview memref type
      'memref<1x128xbf16, strided<[128, 1], offset: ?>,
      #hivm.address_space<ub>>'

Also observed while developing the same path (not reproduced by this reduced
probe): `bl.to_buffer(a, al.ascend_address_space.UB)` (the positional address
space form) reports
`'hivm.hir.pointer_cast' op addrs of PointerCastOp should not be empty!`, and
`tl.store` into `bl.to_tensor(ub, writable=True)` reports
`Unsupported ptr type ... in tl.store`.

So the UB->L1 copy API exists and type-checks, but no Triton program can get a
L1-resident tensor today: `bl.to_tensor` is the single blocking step.

Usage:
    PYTHONPATH=src:FlagGems/src python probe_01_l1_resident_dot.py [copy|dot]
"""

import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def kern(x, y, R: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    o_r = tl.arange(0, R)
    o_k = tl.arange(0, K)
    a = tl.load(x + o_r[:, None] * K + o_k[None, :])
    ub = bl.alloc(tl.bfloat16, [R, K], al.ascend_address_space.UB)
    l1 = bl.alloc(tl.bfloat16, [R, K], al.ascend_address_space.L1)
    al.copy(bl.to_buffer(a, bind_buffer=ub), ub)
    al.copy(ub, l1)
    if MODE == "dot":
        at = bl.to_tensor(l1)
        c = tl.dot(at, tl.trans(at), out_dtype=tl.float32)
        tl.store(y + o_r[:, None] * R + o_r[None, :], c)
    else:
        tl.store(y + o_r[:, None] * R + o_r[None, :], 1.0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dot"
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    x = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, R, K, MODE=mode)
        torch.npu.synchronize()
    print(f"RESULT: PASS probe_01_l1_resident_dot mode={mode}")
    if mode == "dot":
        ref = x.float() @ x.float().T
        print("maxerr", (y - ref).abs().max().item())


if __name__ == "__main__":
    main()
