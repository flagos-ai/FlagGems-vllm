"""探针 06: 跨 chunk 软件流水与 L1/UB 容量预算 (GAP F4)

【背景与目标】
在单 program 内实现跨 chunk 流水：在 chunk c 的 6 个 Gram dot 运算期间，
提前计算 chunk c+1 的门控 bf16 操作数并下发其 dot 指令，最后统一做后处理。
该模式等价于 AscendC 原生的 AIC/AIV ping-pong，旨在隐藏当前串行阻塞 Prepare/Gram kernel 的 Cube 延迟。

【实测现象】(Ascend950DT_9582 / triton-ascend 3.6.0.post2)
  两个 chunk 的点积操作数无法同时容纳在 L1 (cbuf) 中，编译报错：
    cbuf overflow, requires 6291456 bits while 4194304 bits available!
  若改用顺序循环展开 (CPT=2，每次只处理一块)，虽可通过编译，但耗时反而上升
  （Gram 从 2649 us 恶化至 3311 us @ T=32768），因为两块 dot 之间依然被后处理串行阻塞。

【探针说明】
本探针构建了该模式的最小形态：单 program 内包含两个 chunk，每个 chunk 包含 5 个 bf16 dot 操作数
和 6 个共享操作数的 dot，并在 12 个 dot 全部下发后进行后处理。
（注：本简化探针可通过编译；当与真实算子中的 fp32 因子及累加器共同保活时即触发上述 cbuf 溢出）。

运行方式:
    PYTHONPATH=src:FlagGems/src python probe_06_cross_chunk_l1_budget.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def kern(x, y, BT: tl.constexpr, K: tl.constexpr):
    o_i = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    p = o_i[:, None] * K + o_k[None, :]
    a = tl.load(x + p)
    b = tl.load(x + BT * K + p)
    gc = tl.load(x + 2 * BT * K + p)
    HALF: tl.constexpr = BT // 2
    in_sb0 = o_i[:, None] < HALF
    in_sb1 = ~in_sb0

    # tile 0: the real gated operand set (q0,q1,kq0,kq1,k0,k1,k0corr)
    f00 = tl.where(in_sb0, gc, 0.0)
    f01 = tl.where(in_sb1, gc * 1.001, 0.0)
    f02 = tl.where(in_sb0, gc * 1.002, 0.0)
    f03 = tl.where(in_sb1, gc * 1.003, 0.0)
    f04 = tl.where(in_sb0, gc * 1.004, 0.0)
    g0 = _exp2(f00)
    h0 = _exp2(f01)
    q00 = (a * g0).to(tl.bfloat16)
    q01 = (a * h0).to(tl.bfloat16)
    kq0 = (b * g0).to(tl.bfloat16)
    kq1 = (b * h0).to(tl.bfloat16)
    kg0 = (b * (1.0 / g0)).to(tl.bfloat16)
    kg1 = (b * (1.0 / h0)).to(tl.bfloat16)
    kgc = (b * g0 * 1.002).to(tl.bfloat16)
    d00 = tl.dot(q00, tl.trans(kg0), out_dtype=tl.float32)
    d01 = tl.dot(q01, tl.trans(kg1), out_dtype=tl.float32)
    d0c = tl.dot(q01, tl.trans(kgc), out_dtype=tl.float32)
    e00 = tl.dot(kq0, tl.trans(kg0), out_dtype=tl.float32)
    e01 = tl.dot(kq1, tl.trans(kg1), out_dtype=tl.float32)
    e0c = tl.dot(kq1, tl.trans(kgc), out_dtype=tl.float32)

    # tile 1: same seven operands for the next chunk, all still live
    gc1 = gc * 1.003
    f10 = tl.where(in_sb0, gc1, 0.0)
    f11 = tl.where(in_sb1, gc1 * 1.001, 0.0)
    f12 = tl.where(in_sb0, gc1 * 1.002, 0.0)
    f13 = tl.where(in_sb1, gc1 * 1.003, 0.0)
    f14 = tl.where(in_sb0, gc1 * 1.004, 0.0)
    g1 = _exp2(f10)
    h1 = _exp2(f11)
    q10 = (a * g1).to(tl.bfloat16)
    q11 = (a * h1).to(tl.bfloat16)
    kq10 = (b * g1).to(tl.bfloat16)
    kq11 = (b * h1).to(tl.bfloat16)
    kg10 = (b * (1.0 / g1)).to(tl.bfloat16)
    kg11 = (b * (1.0 / h1)).to(tl.bfloat16)
    kg1c = (b * g1 * 1.002).to(tl.bfloat16)
    d10 = tl.dot(q10, tl.trans(kg10), out_dtype=tl.float32)
    d11 = tl.dot(q11, tl.trans(kg11), out_dtype=tl.float32)
    d1c = tl.dot(q11, tl.trans(kg1c), out_dtype=tl.float32)
    e10 = tl.dot(kq10, tl.trans(kg10), out_dtype=tl.float32)
    e11 = tl.dot(kq11, tl.trans(kg11), out_dtype=tl.float32)
    e1c = tl.dot(kq11, tl.trans(kg1c), out_dtype=tl.float32)

    # postprocess both tiles only after all twelve dots are issued
    r0 = d00 + d01 + d0c + e00 + e01 + e0c
    r1 = d10 + d11 + d1c + e10 + e11 + e1c
    # keep the fp32 factor tiles live until here (the real kernel needs them
    # for the rstd/scale/beta postprocessing of both chunks)
    z0 = (tl.sum(f00, 1) + tl.sum(f01, 1) + tl.sum(f02, 1))[:, None]
    z0 += (tl.sum(f03, 1) + tl.sum(f04, 1))[:, None]
    z1 = (tl.sum(f10, 1) + tl.sum(f11, 1) + tl.sum(f12, 1))[:, None]
    z1 += (tl.sum(f13, 1) + tl.sum(f14, 1))[:, None]
    r0 = r0 + z0 * 1e-30
    r1 = r1 + z1 * 1e-30
    tl.store(y + o_i[:, None] * BT + o_i[None, :], r0)
    tl.store(y + BT * BT + o_i[:, None] * BT + o_i[None, :], r1)


def main():
    torch.npu.set_device("npu:0")
    BT, K = 64, 128
    x = torch.randn(3 * BT * K, device="npu", dtype=torch.float32).bfloat16()
    y = torch.empty(2 * BT * BT, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](x, y, BT, K)
        torch.npu.synchronize()
    print("RESULT: PASS probe_06_cross_chunk_l1_budget")


if __name__ == "__main__":
    main()
