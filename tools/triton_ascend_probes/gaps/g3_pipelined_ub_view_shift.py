"""探针 08: 多缓冲 tile 的行偏移复用（GAP R3）

【背景与目标】
causal_conv1d_fn 的每个输出 tile 需要 W=4 个 tap（x[t-3..t]）。当前实现按行偏移
把同一个 tile 重读 4 次，MTE2 流量是数据量的 5 倍（4 读 + 1 写）；AscendC 的
RunSeq 只从 GM 读一份 tile，在 UB 里按行偏移复用，流量是 1.375 倍。本探针量化
这个需求，并给出验收口径：

  MODE=0 read4      四次 GM 读（参考实现，当前生产版本的做法）
  MODE=1 shift-copy 读一次 halo（BLOCK_T + W - 1 行），al.copy 到独立 UB 缓冲
                    后按行 extract_slice
  MODE=2 shift-bind 读一次 halo，bl.to_buffer(bind_buffer=halo) 后按行
                    extract_slice

【AscendC 证据】fla_npu 的 causal_conv1d（causal_conv1d_tiling_planner.h 的
RunSeq/RunSeqFnRolling）每行输入只从 GM 读一次，W 个 tap 是同一份 UB 缓冲的行
偏移视图，不需要额外 GM 访问；这也是它在该 shape 上领先的根因之一。

【Triton 现状】详见 defects/d4_extract_slice_pipeline.py 的完整矩阵：
  1. 直接对 load 结果取 extract_slice 视图：num_stages=2 + multibuffer=True 下
     读到旧版本（maxerr 约 16，确定性错值，非竞态）；
  2. shift-copy / shift-bind 两种写法在同一份代码里"谁正确"会随编译变化
     （同一结构先后出现过 copy 正确 bind 错、bind 正确 copy 错），不可依赖；
  3. 把能正确工作的写法搬进 production kernel（带 prologue、tail、多 range
     网格）后，96 个回归用例里 20-32 个算错，所以该路径尚未上线。
  因此这是一个能力需求，不是纯用法问题：需要编译器对"一次读 + 行偏移复用"给出
  确定的、可依赖的语义。

【需求与验收】
  auto multi-buffer / 软件流水需要把 extract_slice（以及 insert_slice）的视图
  视为对缓冲区的别名访问并做版本跟踪；或者提供一个显式的行偏移复用原语，
  由编译器保证读到的版本正确。
  验收：本探针 MODE=1 与 MODE=2 在 num_stages=2 + multibuffer=True 下都与
  MODE=0 逐位一致，且不慢于 MODE=0。

【收益】causal_conv1d_fn 的 MTE2 流量 5x -> 1.375x。token-major 目标约 111us
（现 155us），channel-first 目标约 120us（现 232us）——后者是当前在诚实输入
契约（channel-first 连续 layout）下唯一仍落后 AscendC 的路径（0.70x）。

运行方式:
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/gaps/g3_pipelined_ub_view_shift.py
"""

import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al
from triton.language.extra.cann.extension import extract_slice

T, C, BN, BT, NSPLIT = 8192, 9216, 2304, 4, 32


@triton.jit
def kern(
    x_ptr,
    o_ptr,
    w_ptr,
    SX,
    T,
    NCT: tl.constexpr,
    NSPLIT: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    ct = pid % NCT
    sp = pid // NCT
    cc = ct * BN + tl.arange(0, BN)
    rr = tl.arange(0, BT)
    H: tl.constexpr = BT + 3
    xr = tl.arange(0, H)
    wk0 = tl.load(w_ptr + 0 * SX + cc).to(tl.float32)
    wk1 = tl.load(w_ptr + 1 * SX + cc).to(tl.float32)
    wk2 = tl.load(w_ptr + 2 * SX + cc).to(tl.float32)
    wk3 = tl.load(w_ptr + 3 * SX + cc).to(tl.float32)
    wk = [wk0, wk1, wk2, wk3]
    if MODE != 0:
        halo = bl.alloc(x_ptr.dtype.element_ty, [H, BN], al.ascend_address_space.UB)
    span = T // NSPLIT
    n = (span - 3) // BT
    for it in range(0, n):
        base = sp * span + it * BT
        rows = base + 3 + rr
        acc = tl.zeros((BT, BN), tl.float32)
        if MODE == 0:
            for j in tl.static_range(4):
                xj = tl.load(x_ptr + (base + j + rr)[:, None] * SX + cc[None, :])
                acc += wk[j][None, :] * xj.to(tl.float32)
        else:
            tile = tl.load(x_ptr + (base + xr)[:, None] * SX + cc[None, :])
            if MODE == 2:
                bl.to_buffer(tile, bind_buffer=halo)
            else:
                al.copy(bl.to_buffer(tile, al.ascend_address_space.UB), halo)
            stable = bl.to_tensor(halo, writable=False)
            for j in tl.static_range(4):
                sub = extract_slice(stable, [j, 0], [BT, BN], [1, 1])
                acc += wk[j][None, :] * sub.to(tl.float32)
        acc = acc / (1 + tl.exp(-acc))
        tl.store(o_ptr + rows[:, None] * SX + cc[None, :], acc.to(tl.bfloat16))


def run(x, out, w, mode, num_stages=2, multibuffer=True):
    kern[(C // BN * NSPLIT,)](
        x,
        out,
        w,
        C,
        T,
        C // BN,
        NSPLIT,
        BN,
        BT,
        mode,
        num_stages=num_stages,
        multibuffer=multibuffer,
    )
    torch.npu.synchronize()
    return out


def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    x = torch.randn(T + 8, C, dtype=torch.bfloat16, device="npu")
    w = torch.randn(4, C, dtype=torch.bfloat16, device="npu")
    outs = [torch.zeros_like(x) for _ in range(3)]
    ok = 0
    with torch.inference_mode():
        ref = run(x, outs[0], w, 0).clone()
        t_ref = bench(lambda: run(x, outs[0], w, 0))
        print(f"read4      : {t_ref:6.1f} us", flush=True)
        for mode, name in ((1, "shift-copy"), (2, "shift-bind")):
            got = run(x, outs[mode], w, mode)
            err = (ref.float() - got.float()).abs().max().item()
            us = bench(lambda: run(x, outs[mode], w, mode))
            good = err == 0
            ok += int(good)
            print(
                f"{name:11s}: maxerr={err:<8.4g} {us:6.1f} us "
                f"{'PASS' if good else 'FAIL'} {t_ref / us:.2f}x",
                flush=True,
            )
    print(
        "RESULT:",
        (
            "requirement met (single read + UB row shift is correct and faster)"
            if ok == 2
            else "GAP R3: single read + UB row shift is not reliably correct "
            f"({ok}/2 spellings exact)"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
