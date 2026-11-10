"""探针 07: 单读 halo + UB 内切片移位返回旧行/脏数据 (DEFECT D4)

【背景与目标】
causal_conv1d_fn 的一个输出 tile 需要 W=4 个 tap（x[t-3..t]）。按行偏移重读
4 次把 MTE2 流量放大到 5x 数据量；理想写法是读一次 halo tile（BT+W-1 行），
再用 extract_slice 在 UB 内按行切片，流量降到 ~1.375x。
本探针给出该写法在 triton-ascend 上的复现矩阵（写法 x 流水配置）。

【写法】
  MODE=0 read4   四次 GM 读（参考实现）
  MODE=1 view    load 一次 + extract_slice(tile) 直接取视图
  MODE=2 bind    bl.alloc(UB) + bl.to_buffer(bind_buffer=halo) + extract_slice
  MODE=3 copy    bl.alloc(UB) + al.copy(to_buffer(tile) -> halo) + extract_slice

【实测现象】(C=9216, T=8192, BLOCK_N=2304, BLOCK_T=4, 32 split；脚本会打印矩阵)
  ns=1, multibuffer=False: view / bind / copy 三种写法都 PASS (maxerr = 0)
  ns=2, multibuffer=True : view 读到旧版本 (maxerr 约 15.8，确定性错值，非竞态)；
                           bind / copy 的 PASS/FAIL 在不同编译与尺寸下不稳定
                           （同一结构另一轮出现过 maxerr 约 17）
  结论：单读 + UB 视图移位在这套编译器上不可靠。关掉流水/多缓冲才稳定正确，
        但那时比四次重读更慢（视图 272-278us vs 重读 235us），收益也消失。
【用法提示】al.copy 要求 bl.alloc 显式传 al.ascend_address_space.UB，
否则报 `src's AddressSpace must be UB`。

【可用替代路径】四次按行偏移重读（MODE=0），靠 L2 吸收 3/4 的重复读，见
src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py。

运行方式:
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/defects/d4_extract_slice_pipeline.py
"""

import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.extension.buffer.language as bl
import triton.language as tl
from triton.language.extra.cann import extension as al
from triton.language.extra.cann.extension import extract_slice

# (C, T, BLOCK_N, BLOCK_T, nsplit): small -> the real causal_conv1d tile
CASES = (
    (256, 512, 64, 4, 2),
    (9216, 8192, 2304, 4, 32),
)


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
    if MODE == 2 or MODE == 3:
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
        elif MODE == 1:
            # naive: view straight out of the (multi-buffered) load
            tile = tl.load(x_ptr + (base + xr)[:, None] * SX + cc[None, :])
            for j in tl.static_range(4):
                sub = extract_slice(tile, [j, 0], [BT, BN], [1, 1])
                acc += wk[j][None, :] * sub.to(tl.float32)
        else:
            tile = tl.load(x_ptr + (base + xr)[:, None] * SX + cc[None, :])
            if MODE == 3:
                bl.to_buffer(tile, bind_buffer=halo)
            else:
                al.copy(bl.to_buffer(tile, al.ascend_address_space.UB), halo)
            stable = bl.to_tensor(halo, writable=False)
            for j in tl.static_range(4):
                sub = extract_slice(stable, [j, 0], [BT, BN], [1, 1])
                acc += wk[j][None, :] * sub.to(tl.float32)
        acc = acc / (1 + tl.exp(-acc))
        tl.store(o_ptr + rows[:, None] * SX + cc[None, :], acc.to(tl.bfloat16))


def run(x, out, w, C, T, nct, nsplit, bn, bt, mode, num_stages, multibuffer=True):
    kern[(nct * nsplit,)](
        x,
        out,
        w,
        C,
        T,
        nct,
        nsplit,
        bn,
        bt,
        mode,
        num_stages=num_stages,
        multibuffer=multibuffer,
    )
    torch.npu.synchronize()
    return out


def bench(args, iters=10):
    for _ in range(3):
        run(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    failures = 0
    with torch.inference_mode():
        for C, T, bn, bt, nsplit in CASES:
            nct = C // bn
            x = torch.randn(T + 8, C, dtype=torch.bfloat16, device="npu")
            w = torch.randn(4, C, dtype=torch.bfloat16, device="npu")
            out = [torch.zeros_like(x) for _ in range(4)]
            print(f"--- C={C} T={T} BN={bn} BT={bt} split={nsplit}", flush=True)
            for ns, mb in ((1, False), (2, True)):
                ref = run(x, out[0], w, C, T, nct, nsplit, bn, bt, 0, ns, mb).clone()
                t_ref = bench((x, out[0], w, C, T, nct, nsplit, bn, bt, 0, ns, mb))
                print(f"  ns={ns} mb={mb} read4    : {t_ref:6.1f} us", flush=True)
                for mode, name in ((1, "view"), (2, "bind"), (3, "copy")):
                    got = run(x, out[mode], w, C, T, nct, nsplit, bn, bt, mode, ns, mb)
                    err = (ref.float() - got.float()).abs().max().item()
                    first = got.clone()
                    again = run(
                        x, out[mode], w, C, T, nct, nsplit, bn, bt, mode, ns, mb
                    )
                    det = torch.equal(first, again)
                    if err != 0 or not det:
                        failures += 1
                    us = bench(
                        (x, out[mode], w, C, T, nct, nsplit, bn, bt, mode, ns, mb)
                    )
                    print(
                        f"  ns={ns} mb={mb} {name:9s}: maxerr={err:<9.4g} "
                        f"det={det} {us:6.1f} us",
                        flush=True,
                    )
            del x, w, out
    print(
        "RESULT:",
        (
            "PASS (no defect)"
            if failures == 0
            else f"FAIL d4_extract_slice_pipeline: {failures} stale/racy single-read cases"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
