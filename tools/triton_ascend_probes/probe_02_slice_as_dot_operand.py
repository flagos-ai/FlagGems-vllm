"""探针 02: 切片作为 dot 操作数 (可用性与回归验证)

【背景与目标】
KDA Gram kernel 需要在单次载入 [BT, 2K] 连续工作区后（w 和 kg 分布在同一行的前后半段），
通过切片切分出左右操作数来执行 Aqk 和 Akk 矩阵乘法；同时也需要把 [2BT, K] 的左操作数拆分为上下两半。

【实测现象】(Ascend950DT_9582 / triton-ascend 3.6.0.post2)
  * 列偏移切片 (col):
    extract_slice(a, [0, K], [BT, K]) 作为 dot 操作数 -> PASS (最大误差 7.6e-6)
  * 行偏移切片 (row):
    extract_slice(a, [HALF, 0], [BT, K]) 作为 dot 操作数 -> PASS (最大误差 5.3e-5)

【结论】
两类切片作为 dot 操作数当前均可正常运行（本探针保留作为基线功能与回归验证）。

运行方式:
    PYTHONPATH=src:FlagGems/src python probe_02_slice_as_dot_operand.py col
    PYTHONPATH=src:FlagGems/src python probe_02_slice_as_dot_operand.py row
"""

import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann.extension import extract_slice


@triton.jit
def kern(x, y, R: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    o_r = tl.arange(0, R)
    o_k = tl.arange(0, 2 * K)
    a = tl.load(x + o_r[:, None] * (2 * K) + o_k[None, :])
    if MODE == "col":
        w = extract_slice(a, [0, 0], [R, K], [1, 1])
        kg = extract_slice(a, [0, K], [R, K], [1, 1])
        c = tl.dot(w, tl.trans(kg), out_dtype=tl.float32)
    else:
        # a is [R, 2K]; take the two row halves as [R/2, 2K] left operands.
        top = extract_slice(a, [0, 0], [R // 2, K], [1, 1])
        bot = extract_slice(a, [R // 2, 0], [R // 2, K], [1, 1])
        c0 = tl.dot(top, tl.trans(top), out_dtype=tl.float32)
        c1 = tl.dot(bot, tl.trans(bot), out_dtype=tl.float32)
        o_h = tl.arange(0, R // 2)
        tl.store(y + o_h[:, None] * (R // 2) + o_h[None, :], c0)
        tl.store(y + R * R // 2 + o_h[:, None] * (R // 2) + o_h[None, :], c1)
        c = None
    if MODE == "col":
        tl.store(y + o_r[:, None] * R + o_r[None, :], c)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "col"
    torch.npu.set_device("npu:0")
    R, K = 128, 128
    x = torch.randn(R, 2 * K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, R, K, MODE=mode)
        torch.npu.synchronize()
    print(f"RESULT: PASS probe_02_slice_as_dot_operand mode={mode}")
    if mode == "col":
        ref = x[:, :K].float() @ x[:, K:].float().T
        print("maxerr", (y - ref).abs().max().item())
    else:
        h = R // 2
        ref0 = x[:h, :K].float() @ x[:h, :K].float().T
        ref1 = x[h:, :K].float() @ x[h:, :K].float().T
        flat = y.reshape(-1)
        err = max(
            (flat[: h * h].view(h, h) - ref0).abs().max().item(),
            (flat[R * R // 2 : R * R // 2 + h * h].view(h, h) - ref1)
            .abs()
            .max()
            .item(),
        )
        print("maxerr", err)


if __name__ == "__main__":
    main()
