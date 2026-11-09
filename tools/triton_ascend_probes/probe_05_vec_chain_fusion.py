"""GAP: fuse a long elementwise chain instead of one UB round-trip per op.

The KDA Prepare/Gram kernels spend most of their time on chains of [64, 128]
elementwise ops (safe-gate sigmoid, cumsum, exp2, masks, 5-7 multiplies).  On
AscendC these are hand-fused into a handful of vector instructions; through
Triton each `tl` op is emitted as a separate vector op with its own UB
read/write, which is what caps the kernels at ~10% of the vector peak.

This probe measures the marginal cost of one dependent `[64, 128]` fp32
elementwise op across the whole grid (12288 programs, i.e. the KDA Prepare
grid at T=32768/H=24), and tries the available fusion switches.

Observed on Ascend950DT_9582 (triton-ascend 3.6.0.post2):
    N extra ops   duration      marginal per grid-wide op
    0             ~0.1 ms       -
    4             ~0.6 ms       ~0.13 ms
    8             ~1.1 ms       ~0.13 ms
    12            ~1.6 ms       ~0.13 ms
i.e. ~0.13 ms per dependent [64,128] op for the whole grid, or ~0.67 us per op
per AIV (192 programs/core).  `vf_fusion_mode` only accepts max-parallel /
all-op / ub-aware-op; `all-op` does not compile and the other two are neutral.

Usage:
    PYTHONPATH=src:FlagGems/src python probe_05_vec_chain_fusion.py
    VF_MODE=all-op ... python probe_05_vec_chain_fusion.py   # shows the failure
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
            f"RESULT: PASS probe_05_vec_chain_fusion N={n:2d} {us:8.1f} us "
            f"marginal={slope:6.1f} us/op" + (f" vf={vf}" if vf else "")
        )


if __name__ == "__main__":
    main()
