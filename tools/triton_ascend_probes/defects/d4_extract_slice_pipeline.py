"""探针 07: 单读 halo + UB 内切片移位返回旧行/脏数据 (DEFECT D4)

【背景与目标】
causal_conv1d_fn 的一个输出 tile 需要 W=4 个 tap（x[t-3..t]）。按行偏移重读
4 次把 MTE2 流量放大到 5x 数据量；理想写法是读一次 halo tile（BT+W-1 行），
再用 extract_slice 在 UB 内按行切片，流量降到 ~1.375x。
本探针给出该写法在 triton-ascend 上的确定性复现。

【写法】
  MODE=0 read4   四次 GM 读（参考实现）
  MODE=1 ssa     load 一次 + extract_slice(tile) 视图
  MODE=3 buf     bl.alloc(UB) + bind_buffer + to_tensor + extract_slice

【实测现象】(Ascend950DT / triton-ascend 3.6.0，输出见脚本末尾的 RESULT)
  1. 小规模（C=256, T=512, BN=64, 2 split）: 结果随写法/流水深度跳变
     （ssa-view 两种 ns 都错，buf-view ns=1 错、ns=2 对）
  2. 真实规模（C=9216, T=8192, BN=2304, 32 split，即 causal_conv1d_fn 的
     实际 tile）: 所有写法、两种 ns 全错，且同一输入两次运行结果不同（竞态）
  3. 某些尺寸组合直接触发设备错误 (device error 507035)，进程退出
  结论：不是公式问题。循环外分配、循环内被当作循环不变量的 UB 视图与
  流水/UB 分配交互，表现为「错值 / 竞态 / 硬故障」三种形态，且会静默返回
  旧行——比崩溃更危险。
【用法提示】al.copy 要求 bl.alloc 显式传 al.ascend_address_space.UB，
否则报 `src's AddressSpace must be UB`。

【可用替代路径】四次按行偏移重读（MODE=0），靠 L2 吸收 3/4 的重复读，见
src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py。

运行方式:
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/defects/d4_extract_slice_pipeline.py
"""

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
    wk0 = tl.load(w_ptr + 0 * T + cc).to(tl.float32)
    wk1 = tl.load(w_ptr + 1 * T + cc).to(tl.float32)
    wk2 = tl.load(w_ptr + 2 * T + cc).to(tl.float32)
    wk3 = tl.load(w_ptr + 3 * T + cc).to(tl.float32)
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
                xj = tl.load(x_ptr + (base + j + rr)[:, None] * T + cc[None, :])
                acc += wk[j][None, :] * xj.to(tl.float32)
        else:
            tile = tl.load(x_ptr + (base + xr)[:, None] * T + cc[None, :])
            if MODE == 3:
                bl.to_buffer(tile, bind_buffer=halo)
            else:
                al.copy(bl.to_buffer(tile, al.ascend_address_space.UB), halo)
            stable = bl.to_tensor(halo, writable=False)
            for j in tl.static_range(4):
                sub = extract_slice(stable, [j, 0], [BT, BN], [1, 1])
                acc += wk[j][None, :] * sub.to(tl.float32)
        acc = acc / (1 + tl.exp(-acc))
        tl.store(o_ptr + rows[:, None] * T + cc[None, :], acc.to(tl.bfloat16))


def run(x, out, w, C, T, nct, nsplit, bn, bt, mode, num_stages):
    kern[(nct * nsplit,)](
        x,
        out,
        w,
        T,
        nct,
        nsplit,
        bn,
        bt,
        mode,
        num_stages=num_stages,
        multibuffer=True,
    )
    torch.npu.synchronize()
    return out


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
            for ns in (1, 2):
                ref = run(x, out[0], w, C, T, nct, nsplit, bn, bt, 0, ns).clone()
                for mode, name in ((1, "ssa-view"), (3, "buf-view")):
                    got = run(x, out[mode], w, C, T, nct, nsplit, bn, bt, mode, ns)
                    err = (ref.float() - got.float()).abs().max().item()
                    first = got.clone()
                    again = run(x, out[mode], w, C, T, nct, nsplit, bn, bt, mode, ns)
                    det = torch.equal(first, again)
                    if err != 0 or not det:
                        failures += 1
                    print(
                        f"  ns={ns} {name:9s}: maxerr={err:<9.4g} deterministic={det}",
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
