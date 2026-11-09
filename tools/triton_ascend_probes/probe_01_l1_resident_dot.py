"""探针 01: 显式 L1 驻留的 dot 操作数

【背景与目标】
AscendC 的 chunk_kda_fwd_prepare 将 Q+/K+/K- 面板缓存在 L1 中，供后续多个
Gram dot 复用，避免重复走 UB->L1 的自动 staging。

triton-ascend 的 L1 驻留用法并不是"把 ND tensor 放进 L1 再用 tl.dot"：
  1. L1/Cube 操作数使用 zN (fractal) 布局，形状为 [N1, M1, 16, 16]
     （bf16 的 block 是 16x16；fp32 是 16x8）。
  2. 用 al.copy(bl.to_buffer(x_nz, UB), l1) 把 UB 中的 zN 数据搬到 L1。
  3. 用 bl.to_tensor(l1) 直接取回 4D zN tensor，不要传 target_shape。
  4. 用 al.dot(a, b, format_a="fractal", format_b="fractal") 让 Cube 直接
     消费 zN 操作数；tl.dot 不理解 4D zN 布局。

对 A [R, K] @ A^T [K, R]：
  - A 的 zN 面板：[K//16, R//16, 16, 16]（A 布局）
  - A^T 的 zN 面板：[R//16, K//16, 16, 16]（B 布局）
两者不能共用同一块物理面板，需要分别 staging。

【关于 copy_from_ub_to_l1】
al.copy_from_ub_to_l1 只是 al.copy 的 deprecated 别名，语义完全相同；它本身
并不能解决旧写法的 lowering 问题。旧写法失败的关键是
  bl.to_tensor(l1, target_shape=[R, K]) + tl.dot(...)
该路径会在 L1 memref 上生成错误的 UB 地址空间 subview / 不支持的
convert_layout。把消费端改成 4D zN + al.dot(fractal) 之后，copy 与
copy_from_ub_to_l1 都能正常工作（后者会打印 deprecation warning）。

运行方式:
    PYTHONPATH=src:FlagGems/src python probe_01_l1_resident_dot.py [dot|copy|old]
      dot  : 推荐写法，al.copy + bl.to_tensor + al.dot(fractal)      -> PASS
      copy : 用 deprecated 的 al.copy_from_ub_to_l1 + al.dot(fractal) -> PASS
      old  : 旧写法 bl.to_tensor(target_shape) + tl.dot              -> FAIL
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

    # A [R, K] -> zN [K//16, R//16, 16, 16]
    a_tmp = tl.permute(a.reshape(R, K // 16, 16), (1, 0, 2))
    a_nz = a_tmp.reshape(K // 16, R // 16, 16, 16)

    # A^T [K, R] -> B 布局 zN [R//16, K//16, 16, 16]
    at = tl.trans(a)
    at_tmp = tl.permute(at.reshape(K, R // 16, 16), (1, 0, 2))
    at_nz = at_tmp.reshape(R // 16, K // 16, 16, 16)

    l1a = bl.alloc(tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1)
    l1b = bl.alloc(tl.bfloat16, [R // 16, K // 16, 16, 16], al.ascend_address_space.L1)

    if MODE == "old":
        # 旧写法：把 L1 buffer 转成 2D ND tensor 后交给 tl.dot。
        # 这条路径仍会在 lowering 阶段失败（cbuf/ub 地址空间冲突）。
        al.copy(bl.to_buffer(a_nz, al.ascend_address_space.UB), l1a)
        at2 = bl.to_tensor(l1a, target_shape=[R, K])
        c = tl.dot(at2, tl.trans(at2), out_dtype=tl.float32)
    else:
        if MODE == "copy":
            # deprecated 别名，仅用于验证 copy 入口本身不是问题。
            al.copy_from_ub_to_l1(
                bl.to_buffer(a_nz, al.ascend_address_space.UB), l1a
            )
            al.copy_from_ub_to_l1(
                bl.to_buffer(at_nz, al.ascend_address_space.UB), l1b
            )
        else:
            al.copy(bl.to_buffer(a_nz, al.ascend_address_space.UB), l1a)
            al.copy(bl.to_buffer(at_nz, al.ascend_address_space.UB), l1b)

        # 保留 4D zN 布局，直接交给 al.dot。
        ta = bl.to_tensor(l1a)
        tb = bl.to_tensor(l1b)
        c = al.dot(ta, tb, format_a="fractal", format_b="fractal")

    tl.store(y + o_r[:, None] * R + o_r[None, :], c)


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
    ref = x.float() @ x.float().T
    print("maxerr", (y - ref).abs().max().item())


if __name__ == "__main__":
    main()
