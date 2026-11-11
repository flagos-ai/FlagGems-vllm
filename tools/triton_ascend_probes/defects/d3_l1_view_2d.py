"""DEFECT (workaround available): a 2D ND view of an L1 buffer.

Two spellings are still rejected on Ascend950DT_9582:
  MODE=nd      2D ND L1 buffer + `bl.to_tensor(l1)` + `al.dot(nd, nd)`
               -> different memory spaces ... cbuf vs ub
  MODE=target  `bl.to_tensor(l1_4d, target_shape=[R, K])` + `tl.dot`
               -> LLVM ERROR: unsupported op to get out operands

Workaround (supported/s2_l1_resident_dot.py): keep the operand in its 4D zN
layout and use `al.dot(..., format_a/b="fractal")`.  This file is a bug
report, not a requirement: the capability is reachable through the zN path;
only the ND spelling is broken.

Usage (from the repo root):
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/defects/d3_l1_view_2d.py nd
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/defects/d3_l1_view_2d.py target
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
    if MODE == "target":
        # 4D zN buffer viewed as 2D ND through target_shape.
        a_nz = tl.permute(a.reshape(R, K // 16, 16), (1, 0, 2)).reshape(
            K // 16, R // 16, 16, 16
        )
        l1_4d = bl.alloc(
            tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
        )
        al.copy(bl.to_buffer(a_nz, al.ascend_address_space.UB), l1_4d)
        at = bl.to_tensor(l1_4d, target_shape=[R, K])
        c = tl.dot(at, tl.trans(at), out_dtype=tl.float32)
    else:
        at = bl.to_tensor(l1)
        c = al.dot(at, tl.trans(at), format_a="nd", format_b="nd")
    tl.store(y + o_r[:, None] * R + o_r[None, :], c)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nd"
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    x = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, R, K, MODE=mode)
        torch.npu.synchronize()
    print(f"RESULT: PASS d3_l1_view_2d mode={mode}")


if __name__ == "__main__":
    main()
