"""探针 03: insert_slice 拼装 dot 操作数

【背景与目标】
AscendC 原生实现中，门控因子按 32 行子块计算，再拼接为完整的 [BT, K] 参与
全宽 dot 运算，从而减少一半 AIV 向量计算量。

【三种写法与实测】
  insert     : tl.zeros 基座 + al.insert_slice 拼装 + tl.dot
               -> FAIL: LLVM ERROR: Unknown PIPE!（BiShengIR SIGABRT）
  l1_insert  : 把 insert_slice 写进显式 L1 buffer 的 4D zN tensor，
               再用 al.dot(fractal) 消费
               -> FAIL: 仍为 LLVM ERROR: Unknown PIPE!
               说明"显式指定地址空间"并不能修复 insert_slice -> dot 的 lowering。
  l1_copy    : 不拼装 SSA tensor；用 bl.subview 把 L1 面板切成上下子块，
               对每个子块 al.copy(UB -> L1)，最后由 al.dot(fractal) 读取整块
               -> PASS

【结论】
insert_slice 的结果仍不能直接作为 Cube dot 输入，即使给它显式 L1 地址空间也
不行。可行的替代路径是：在 L1 上预先分配 A / B 两种 zN 布局的面板，用
bl.subview 定位上下子块，再用 al.copy 从 UB 写入；Cube 侧用
bl.to_tensor(l1) + al.dot(..., format_a="fractal", format_b="fractal") 读取。
这条路径不经过 insert_slice，也不产生昂贵的 layout 转换。

运行方式:
    PYTHONPATH=src:FlagGems/src python probe_03_insert_slice_dot.py [l1_copy|insert|l1_insert]
"""

import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al
from triton.language.extra.cann.extension import insert_slice


@triton.jit
def kern(x, y, R: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    o_r = tl.arange(0, R)
    o_h = tl.arange(0, R // 2)
    o_k = tl.arange(0, K)
    top = tl.load(x + o_h[:, None] * K + o_k[None, :])
    bot = tl.load(x + (o_h[:, None] + R // 2) * K + o_k[None, :])

    # top / bot [R/2, K] -> A 布局 zN [K//16, (R/2)//16, 16, 16]
    top_tmp = tl.permute(top.reshape(R // 2, K // 16, 16), (1, 0, 2))
    top_nz = top_tmp.reshape(K // 16, (R // 2) // 16, 16, 16)
    bot_tmp = tl.permute(bot.reshape(R // 2, K // 16, 16), (1, 0, 2))
    bot_nz = bot_tmp.reshape(K // 16, (R // 2) // 16, 16, 16)

    # top^T / bot^T [K, R/2] -> B 布局 zN [(R/2)//16, K//16, 16, 16]
    top_t = tl.trans(top)
    top_t_tmp = tl.permute(top_t.reshape(K, (R // 2) // 16, 16), (1, 0, 2))
    top_t_nz = top_t_tmp.reshape((R // 2) // 16, K // 16, 16, 16)
    bot_t = tl.trans(bot)
    bot_t_tmp = tl.permute(bot_t.reshape(K, (R // 2) // 16, 16), (1, 0, 2))
    bot_t_nz = bot_t_tmp.reshape((R // 2) // 16, K // 16, 16, 16)

    if MODE == "insert":
        # 原始写法：SSA tensor 拼装后直接 tl.dot。
        base = tl.zeros((R, K), dtype=tl.bfloat16)
        full = insert_slice(base, top, [0, 0], [R // 2, K], [1, 1])
        full = insert_slice(full, bot, [R // 2, 0], [R // 2, K], [1, 1])
        c = tl.dot(full, tl.trans(full), out_dtype=tl.float32)
        tl.store(y + o_r[:, None] * R + o_r[None, :], c)

    if MODE == "l1_insert":
        # 把拼装结果放进显式 L1 的 4D zN tensor，再交给 al.dot。
        l1a = bl.alloc(
            tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
        )
        l1b = bl.alloc(
            tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
        )
        full_a = bl.to_tensor(l1a)
        full_b = bl.to_tensor(l1b)
        full_a = insert_slice(
            full_a,
            top_nz,
            [0, 0, 0, 0],
            [K // 16, (R // 2) // 16, 16, 16],
            [1, 1, 1, 1],
        )
        full_a = insert_slice(
            full_a,
            bot_nz,
            [0, (R // 2) // 16, 0, 0],
            [K // 16, (R // 2) // 16, 16, 16],
            [1, 1, 1, 1],
        )
        full_b = insert_slice(
            full_b,
            top_t_nz,
            [0, 0, 0, 0],
            [(R // 2) // 16, K // 16, 16, 16],
            [1, 1, 1, 1],
        )
        full_b = insert_slice(
            full_b,
            bot_t_nz,
            [(R // 2) // 16, 0, 0, 0],
            [(R // 2) // 16, K // 16, 16, 16],
            [1, 1, 1, 1],
        )
        c = al.dot(full_a, full_b, format_a="fractal", format_b="fractal")
        tl.store(y + o_r[:, None] * R + o_r[None, :], c)

    if MODE == "l1_copy":
        # 推荐路径：L1 面板 + subview 定位子块 + al.copy(UB -> L1)。
        l1a = bl.alloc(
            tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1
        )
        l1b = bl.alloc(
            tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1
        )
        sub_top_a = bl.subview(
            l1a,
            [0, 0, 0, 0],
            [K // 16, (R // 2) // 16, 16, 16],
            [1, 1, 1, 1],
        )
        sub_bot_a = bl.subview(
            l1a,
            [0, (R // 2) // 16, 0, 0],
            [K // 16, (R // 2) // 16, 16, 16],
            [1, 1, 1, 1],
        )
        sub_top_b = bl.subview(
            l1b,
            [0, 0, 0, 0],
            [(R // 2) // 16, K // 16, 16, 16],
            [1, 1, 1, 1],
        )
        sub_bot_b = bl.subview(
            l1b,
            [(R // 2) // 16, 0, 0, 0],
            [(R // 2) // 16, K // 16, 16, 16],
            [1, 1, 1, 1],
        )
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "l1_copy"
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    x = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    y = torch.empty(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, R, K, MODE=mode)
        torch.npu.synchronize()
    ref = x.float() @ x.float().T
    print(f"RESULT: PASS probe_03_insert_slice_dot mode={mode}")
    print("maxerr", (y - ref).abs().max().item())


if __name__ == "__main__":
    main()
