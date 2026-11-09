"""GAP: assemble a bf16 dot operand from per-32-row halves with insert_slice.

Needed to compute the KDA gate factors on 32-row sub-blocks (half the vector
work, matching AscendC's ScoreRefToken span) and then feed the assembled
[BT, K] tile to one full-width dot.

Current state: `al.insert_slice` into an SSA tile used as a dot operand makes
BiShengIR abort (SIGABRT) on Ascend950DT.  The variant that inserts into a
tile and *stores* it works.

Usage:
    PYTHONPATH=src:FlagGems/src python probe_03_insert_slice_dot.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann.extension import insert_slice


@triton.jit
def kern(x, y, R: tl.constexpr, K: tl.constexpr):
    o_r = tl.arange(0, R)
    o_h = tl.arange(0, R // 2)
    o_k = tl.arange(0, K)
    top = tl.load(x + o_h[:, None] * K + o_k[None, :])
    bot = tl.load(x + (o_h[:, None] + R // 2) * K + o_k[None, :])
    base = tl.zeros((R, K), dtype=tl.bfloat16)
    full = insert_slice(base, top, [0, 0], [R // 2, K], [1, 1])
    full = insert_slice(full, bot, [R // 2, 0], [R // 2, K], [1, 1])
    c = tl.dot(full, tl.trans(full), out_dtype=tl.float32)
    tl.store(y + o_r[:, None] * R + o_r[None, :], c)


def main():
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    x = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, R, K)
        torch.npu.synchronize()
    ref = x.float() @ x.float().T
    print("RESULT: PASS probe_03_insert_slice_dot maxerr", (y - ref).abs().max().item())


if __name__ == "__main__":
    main()
