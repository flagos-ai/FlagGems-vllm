"""探针 03: insert_slice 拼装 dot 操作数 (GAP F2)

【背景与目标】
AscendC 原生实现中，门控因子按 32 行子块（对齐 ScoreRefToken span）计算，
可减半 AIV 向量计算量，再拼接为完整的 [BT, K] 参与全宽 dot 运算。
在 Triton 中尝试使用 al.insert_slice 分别拼接上半块与下半块：
    full = insert_slice(base, top, [0, 0], [R // 2, K], [1, 1])
    full = insert_slice(full, bot, [R // 2, 0], [R // 2, K], [1, 1])
    c = tl.dot(full, tl.trans(full))

【实测现象】(Ascend950DT_9582 / triton-ascend 3.6.0.post2)
  将 al.insert_slice 拼装出的 SSA tile 作为 dot 操作数时，BiShengIR 发生崩溃：
    LLVM ERROR: Unknown PIPE!
    ... died with <Signals.SIGABRT: 6>.
  注：若 insert_slice 的结果仅用于 tl.store，则能正常执行；只有送入 tl.dot 时触发崩溃。

【结论】
insert_slice 的输出无法直接作为 Cube dot 输入。备选方案 tl.join+permute+reshape 虽然能跑通，
但会引入昂贵的 layout 转换，实测耗时增加约 4 倍，不具备实用性。

运行方式:
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
