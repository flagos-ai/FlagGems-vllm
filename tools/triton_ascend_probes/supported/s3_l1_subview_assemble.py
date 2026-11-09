"""SUPPORTED: assemble a full L1 zN dot operand from 32-row half panels.

The natural spelling is `insert_slice` into an SSA tile, which still crashes
(see defects/d2_insert_slice_dot.py).  This file is the path that works and
that AscendC uses as well (chunk_kda_fwd_prepare_cube.h copies each 32x32
quadrant straight into its zN slot in L1):

    * allocate the full zN panel in L1,
    * `bl.subview` to select the top / bottom 32-row slot,
    * `al.copy` each half's zN tile from UB into its slot,
    * `al.dot(..., format_a/b="fractal")`.

Use it when deciding how to implement per-32-row gate factors: it is the
supported replacement for `insert_slice` and it keeps the factor work at
half height.

Observed on Ascend950DT_9582: PASS, maxerr <= 6.1e-5 (grid 1, R=64, K=128).

Usage (from the repo root):
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/supported/s3_l1_subview_assemble.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def kern(x, y, R: tl.constexpr, K: tl.constexpr):
    o_r = tl.arange(0, R)
    o_h = tl.arange(0, R // 2)
    o_k = tl.arange(0, K)
    top = tl.load(x + o_h[:, None] * K + o_k[None, :])
    bot = tl.load(x + (o_h[:, None] + R // 2) * K + o_k[None, :])

    half: tl.constexpr = (R // 2) // 16
    # top/bot [R/2, K] -> A zN [K//16, half, 16, 16]
    top_nz = tl.permute(top.reshape(R // 2, K // 16, 16), (1, 0, 2)).reshape(
        K // 16, half, 16, 16
    )
    bot_nz = tl.permute(bot.reshape(R // 2, K // 16, 16), (1, 0, 2)).reshape(
        K // 16, half, 16, 16
    )
    # top^T/bot^T [K, R/2] -> B zN [half, K//16, 16, 16]
    top_t_nz = tl.permute(tl.trans(top).reshape(K, half, 16), (1, 0, 2)).reshape(
        half, K // 16, 16, 16
    )
    bot_t_nz = tl.permute(tl.trans(bot).reshape(K, half, 16), (1, 0, 2)).reshape(
        half, K // 16, 16, 16
    )

    l1a = bl.alloc(tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1)
    l1b = bl.alloc(tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1)
    sub_top_a = bl.subview(l1a, [0, 0, 0, 0], [K // 16, half, 16, 16], [1, 1, 1, 1])
    sub_bot_a = bl.subview(l1a, [0, half, 0, 0], [K // 16, half, 16, 16], [1, 1, 1, 1])
    sub_top_b = bl.subview(l1b, [0, 0, 0, 0], [half, K // 16, 16, 16], [1, 1, 1, 1])
    sub_bot_b = bl.subview(l1b, [half, 0, 0, 0], [half, K // 16, 16, 16], [1, 1, 1, 1])

    al.copy(bl.to_buffer(top_nz, al.ascend_address_space.UB), sub_top_a)
    al.copy(bl.to_buffer(bot_nz, al.ascend_address_space.UB), sub_bot_a)
    al.copy(bl.to_buffer(top_t_nz, al.ascend_address_space.UB), sub_top_b)
    al.copy(bl.to_buffer(bot_t_nz, al.ascend_address_space.UB), sub_bot_b)

    c = al.dot(
        bl.to_tensor(l1a),
        bl.to_tensor(l1b),
        format_a="fractal",
        format_b="fractal",
    )
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
    print("RESULT: PASS s3_l1_subview_assemble maxerr", (y - ref).abs().max().item())


if __name__ == "__main__":
    main()
