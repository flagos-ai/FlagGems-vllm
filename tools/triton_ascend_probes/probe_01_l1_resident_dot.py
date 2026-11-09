"""探针 01: 显式 L1 驻留的 dot 操作数 (GAP F1)

【背景与目标】
AscendC 的 chunk_kda_fwd_prepare 将 4 个 head 的 Q+/K+/K- 面板缓存在 L1 中，
供后续多个 Gram dot 操作复用，避免重复走 UB->L1 的自动 staging。
在 Triton 中的理想表达形式为：
    l1 = bl.alloc(bf16, [BT, K], al.ascend_address_space.L1)
    ... 从 UB 写入 l1 ...
    a = bl.to_tensor(l1)
    c = tl.dot(a, tl.trans(a))

【实测现象】(Ascend950DT_9582 / triton-ascend 3.6.0.post2)
  1. MODE=copy  (通过 bind_buffer 写入 UB 并执行 al.copy(ub, l1)):
     -> 失败: 'memref.alloc' op error: read before first write
  2. MODE=dot   (在 copy 基础上增加 bl.to_tensor(l1) + tl.dot):
     -> 失败: different memory spaces specified for base memref type
        'memref<64x128xbf16, #hivm.address_space<cbuf>>' and subview memref type
        'memref<1x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>'

【相关已知异常】
- 位置参数形式 bl.to_buffer(a, al.ascend_address_space.UB) 报错:
  'hivm.hir.pointer_cast' op addrs of PointerCastOp should not be empty!
- 通过 tl.store 写入 bl.to_tensor(ub, writable=True) 视图报错:
  Unsupported ptr type ... in tl.store

【结论】
虽然 UB->L1 的 copy API 在前端能通过类型检查，但由于 bl.to_tensor lowering 时
在 L1 buffer 上错误生成了 UB 地址空间的 subview，导致当前无法在 Triton 中获得 L1 驻留的 Tensor。

运行方式:
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
