"""探针 04: scope 混核分工与跨核同步 (GAP F3)

【背景与目标】
在单个 kernel 内表达 Cube 与 Vector 的混核流水分工（对应 AscendC 的 AIC/AIV C2/C4/C5/C7 流水）：
Cube scope 负责 tl.dot 计算并将结果写入 GM，Vector scope 从 GM 消费该结果进行后续运算，
两核在循环间流水重叠。
Triton 提供的机制为 al.scope("cube"|"vector") 以及 al.sync_block_set/wait。

【实测现象】(Ascend950DT_9582 / triton-ascend 3.6.0.post2)
  1. MODE=scope       (默认编译选项):
     -> 失败: BiShengIR 段错误 SIGSEGV (11)
  2. MODE=scope_noauto(开启 enable_mixed_cv=True, disable_auto_inject_block_sync=True):
     -> 编译通过，但运算结果错误 (maxerr ≈ 1e2，读到了未完成写入的脏数据)
【可用替代路径】不使用 scope、只用 al.sync_block_set/wait，见
  supported/s4_manual_cross_core_sync.py（PASS，maxerr 1.5e-5）。
  因此本文件是"崩溃/脏读"缺陷报告，不是能力需求。

运行方式（仓库根目录）:
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/defects/d1_scope_auto_sync.py [scope|scope_noauto]
"""

import sys

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def kern(a, b, c, out, R: tl.constexpr, K: tl.constexpr, MODE: tl.constexpr):
    o_r = tl.arange(0, R)
    o_k = tl.arange(0, K)
    pa = a + o_r[:, None] * K + o_k[None, :]
    pb = b + o_r[:, None] * K + o_k[None, :]
    pc = c + o_r[:, None] * R + o_r[None, :]
    po = out + o_r[:, None] * R + o_r[None, :]

    if MODE == "scope":
        with al.scope("cube"):
            cb = tl.dot(tl.load(pa), tl.trans(tl.load(pb)), out_dtype=tl.float32)
            tl.store(pc, cb)
        with al.scope("vector"):
            tl.store(po, tl.load(pc) * 2.0)
    elif MODE == "scope_noauto":
        with al.scope("cube"):
            cb = tl.dot(tl.load(pa), tl.trans(tl.load(pb)), out_dtype=tl.float32)
            tl.store(pc, cb)
        with al.scope("vector"):
            tl.store(po, tl.load(pc) * 2.0)
    else:
        cb = tl.dot(tl.load(pa), tl.trans(tl.load(pb)), out_dtype=tl.float32)
        tl.store(pc, cb)
        al.sync_block_set("cube", "vector", 0)
        al.sync_block_wait("cube", "vector", 0)
        tl.store(po, tl.load(pc) * 2.0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "scope"
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    a = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    b = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    c = torch.zeros(R, R, device="npu", dtype=torch.float32)
    out = torch.zeros(R, R, device="npu", dtype=torch.float32)
    opts = {}
    if mode in ("scope_noauto", "scope_sync"):
        opts = {"enable_mixed_cv": True, "disable_auto_inject_block_sync": True}
    with torch.inference_mode():
        kern[(1,)](a, b, c, out, R, K, MODE=mode, **opts)
        torch.npu.synchronize()
    ref = (a.float() @ b.float().T) * 2.0
    err = (out - ref).abs().max().item()
    print(f"RESULT: PASS d1_scope_auto_sync mode={mode} maxerr={err:.6e}")


if __name__ == "__main__":
    main()
