"""探针 02: 切片作为 dot 操作数（UB 基线 + 显式 L1 版本）

【背景与目标】
KDA Gram kernel 需要在单次载入 [BT, 2K] 连续工作区后（w 和 kg 分布在同一行的
前后半段），通过切片切分出左右操作数来执行 Aqk 和 Akk 矩阵乘法；同时也需要把
[2BT, K] 的左操作数拆分为上下两半。

【模式说明】
  col     : UB tensor + extract_slice + tl.dot                -> PASS（基线）
  row     : UB tensor + extract_slice + tl.dot                -> PASS（基线）
  l1_col  : 显式 L1 缓存 + col 切片语义 + al.dot(fractal)      -> PASS
  l1_row  : 显式 L1 缓存 + 行切片 + al.dot(fractal)            -> PASS

【显式 L1 版本的关键点】
1. L1/Cube 操作数必须使用 zN (fractal) 布局：
     A [M, K]  -> [K//16, M//16, 16, 16]   (A 布局)
     B [K, N]  -> [N//16, K//16, 16, 16]   (B 布局)
   同一个逻辑矩阵作为左/右操作数时物理布局不同，不能共用同一块 L1 面板。
2. 取回 L1 时用 bl.to_tensor(l1)（保持 4D zN），不要传 target_shape。
3. 用 al.dot(a, b, format_a="fractal", format_b="fractal") 直接消费 zN 操作数。

【切片方向的限制】
zN 布局把 M/N 方向放在内层 block 维度：
  - A 布局的 M 方向是第 1 维（非最外层），对它做行切片会得到错误的物理视图；
  - B 布局的 N 方向是第 0 维（最外层），行切片是安全的。
因此 l1_row 采用 "B 面板按行切片 + 每个行块单独的 A 面板" 的方式；如果强行
对 A 布局面板做 M 方向切片，虽然能编译运行，但结果不正确（见 README）。

运行方式:
    PYTHONPATH=src:FlagGems/src python s1_slice_dot.py [col|row|l1_col|l1_row]
"""

import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al
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
        tl.store(y + o_r[:, None] * R + o_r[None, :], c)

    if MODE == "row":
        # a is [R, 2K]; take the two row halves as [R/2, K] left operands.
        top = extract_slice(a, [0, 0], [R // 2, K], [1, 1])
        bot = extract_slice(a, [R // 2, 0], [R // 2, K], [1, 1])
        c0 = tl.dot(top, tl.trans(top), out_dtype=tl.float32)
        c1 = tl.dot(bot, tl.trans(bot), out_dtype=tl.float32)
        o_h = tl.arange(0, R // 2)
        tl.store(y + o_h[:, None] * (R // 2) + o_h[None, :], c0)
        tl.store(y + R * R // 2 + o_h[:, None] * (R // 2) + o_h[None, :], c1)

    if MODE == "l1_col":
        # 从同一块 UB 工作区切出 w / kg，再分别 staging 成 A / B 布局的 L1 面板。
        w = extract_slice(a, [0, 0], [R, K], [1, 1])
        kg = extract_slice(a, [0, K], [R, K], [1, 1])
        kgt = tl.trans(kg)  # [K, R]

        w_tmp = tl.permute(w.reshape(R, K // 16, 16), (1, 0, 2))
        w_nz = w_tmp.reshape(K // 16, R // 16, 16, 16)
        kgt_tmp = tl.permute(kgt.reshape(K, R // 16, 16), (1, 0, 2))
        kgt_nz = kgt_tmp.reshape(R // 16, K // 16, 16, 16)

        l1w = bl.alloc(
            tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
        )
        l1k = bl.alloc(
            tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
        )
        al.copy(bl.to_buffer(w_nz, al.ascend_address_space.UB), l1w)
        al.copy(bl.to_buffer(kgt_nz, al.ascend_address_space.UB), l1k)

        c = al.dot(
            bl.to_tensor(l1w),
            bl.to_tensor(l1k),
            format_a="fractal",
            format_b="fractal",
        )
        tl.store(y + o_r[:, None] * R + o_r[None, :], c)

    if MODE == "l1_row":
        # 整个 [R, K] 先 staging 成 B 布局面板；B 布局的 N 维（第 0 维）是最外层，
        # 因此行切片是安全的。
        a_k = extract_slice(a, [0, 0], [R, K], [1, 1])
        at = tl.trans(a_k)
        at_tmp = tl.permute(at.reshape(K, R // 16, 16), (1, 0, 2))
        at_nz = at_tmp.reshape(R // 16, K // 16, 16, 16)
        l1b = bl.alloc(
            tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
        )
        al.copy(bl.to_buffer(at_nz, al.ascend_address_space.UB), l1b)
        tb = bl.to_tensor(l1b)
        h1: tl.constexpr = (R // 2) // 16
        btop = al.extract_slice(tb, [0, 0, 0, 0], [h1, K // 16, 16, 16], [1, 1, 1, 1])
        bbot = al.extract_slice(tb, [h1, 0, 0, 0], [h1, K // 16, 16, 16], [1, 1, 1, 1])

        # A 布局的 M 维不是最外层，按行切片不可靠；每个行块单独分配 A 面板。
        top = extract_slice(a, [0, 0], [R // 2, K], [1, 1])
        bot = extract_slice(a, [R // 2, 0], [R // 2, K], [1, 1])
        top_tmp = tl.permute(top.reshape(R // 2, K // 16, 16), (1, 0, 2))
        top_nz = top_tmp.reshape(K // 16, (R // 2) // 16, 16, 16)
        bot_tmp = tl.permute(bot.reshape(R // 2, K // 16, 16), (1, 0, 2))
        bot_nz = bot_tmp.reshape(K // 16, (R // 2) // 16, 16, 16)
        l1top = bl.alloc(
            tl.bfloat16, [K // 16, (R // 2) // 16, 16, 16], al.ascend_address_space.L1
        )
        l1bot = bl.alloc(
            tl.bfloat16, [K // 16, (R // 2) // 16, 16, 16], al.ascend_address_space.L1
        )
        al.copy(bl.to_buffer(top_nz, al.ascend_address_space.UB), l1top)
        al.copy(bl.to_buffer(bot_nz, al.ascend_address_space.UB), l1bot)

        c0 = al.dot(bl.to_tensor(l1top), btop, format_a="fractal", format_b="fractal")
        c1 = al.dot(bl.to_tensor(l1bot), bbot, format_a="fractal", format_b="fractal")
        o_h = tl.arange(0, R // 2)
        tl.store(y + o_h[:, None] * (R // 2) + o_h[None, :], c0)
        tl.store(y + R * R // 2 + o_h[:, None] * (R // 2) + o_h[None, :], c1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "col"
    torch.npu.set_device("npu:0")
    R, K = 128, 128
    x = torch.randn(R, 2 * K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, R, K, MODE=mode)
        torch.npu.synchronize()
    print(f"RESULT: PASS s1_slice_dot mode={mode}")

    if mode in ("col", "l1_col"):
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
