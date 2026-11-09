"""探针 05: 长 exp2/select 链的向量指令融合 (GAP F5)

【背景与目标】
KDA Prepare/Gram kernel 的耗时瓶颈在于大量的 [64, 128] 逐元素向量运算链
（sigmoid -> cumsum -> exp2 -> masks -> 5~7 次乘法）。
在 AscendC 中，此类计算链会被手动融合为少数几条紧凑的向量指令；
而在 Triton 中，每个 tl op 会被拆解为独立的向量操作并伴随频繁的 UB 读写往返，
导致向量单元利用率受限在 ~10%。

本探针通过在全网格（12288 个 program，对应 T=32768/H=24 网格规模）上
构造依赖链 a = exp2(a * 1e-4) + where(a > 0.5, 1, 0)，评测长链下的边际开销与融合表现。

【实测现象】(Ascend950DT_9582 / triton-ascend 3.6.0.post2)
    额外 op 数 (N)   全网格耗时     单 op 边际耗时
    0               ~255 us        -
    4               ~271 us        ~4 us/op   (短链成功融合)
    8               ~794 us        ~67 us/op  (出现性能断崖)
    12              ~1478 us       ~101 us/op (单 AIV 约 0.53 us/op)

【融合编译选项测试】
vf_fusion_mode 提供了 max-parallel / all-op / ub-aware-op 三种模式：
- all-op: 编译失败；
- max-parallel 与 ub-aware-op: 性能与默认表现一致，未见改善。

【结论】
≤4 的短 op 链可被良好融合；达到 8 个 op 后编译器退化为逐 op 往返 UB，
每增加一个 [64, 128] op 全网格开销递增约 100 us，直接构成了 Gram kernel 2.6 ms 的主要开销来源。

运行方式:
    PYTHONPATH=src:FlagGems/src python g2_vec_chain_fusion.py
    VF_MODE=all-op PYTHONPATH=src:FlagGems/src python g2_vec_chain_fusion.py  # 观察编译失败
"""

import os

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def kern(x, y, R: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(0)
    o_r = tl.arange(0, R)
    o_k = tl.arange(0, K)
    p = (pid % 8) * R * K + o_r[:, None] * K + o_k[None, :]
    a = tl.load(x + p).to(tl.float32)
    for _ in tl.static_range(N):
        # exp2 + select + cast: a representative KDA gate step that the
        # vectorizer cannot fold into one FMA
        a = tl.math.exp2(a * 1e-4) + tl.where(a > 0.5, 1.0, 0.0)
    tl.store(y + p, a)


def bench(n_ops, vf_mode=None, grid=12288):
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    x = torch.randn(8 * R * K, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    opts = {}
    if vf_mode:
        opts["vf_fusion_mode"] = vf_mode
    with torch.inference_mode():
        for _ in range(5):
            kern[(grid,)](x, y, R, K, N=n_ops, **opts)
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            kern[(grid,)](x, y, R, K, N=n_ops, **opts)
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / 10 * 1e3  # us


def main():
    vf = os.environ.get("VF_MODE") or None
    base = None
    for n in (0, 4, 8, 12):
        us = bench(n, vf)
        if n == 0:
            base = us
        slope = (us - base) / n if n else 0.0
        print(
            f"RESULT: PASS g2_vec_chain_fusion N={n:2d} {us:8.1f} us "
            f"marginal={slope:6.1f} us/op" + (f" vf={vf}" if vf else "")
        )


if __name__ == "__main__":
    main()
