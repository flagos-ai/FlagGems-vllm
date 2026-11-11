"""SUPPORTED: scoped Cube/Vector hand-off with explicit sync_block flags.

`al.scope("vector")` / `al.scope("cube")` plus `al.sync_block_set/wait` can
express a producer-consumer pair today, but only with

    enable_mixed_cv=True, disable_auto_inject_block_sync=True

The default compile options still crash (see defects/d1_scope_auto_sync.py).

This probe is the minimal two-item pipeline: the vector stream writes an item
to a GM workspace and signals the cube stream; the cube stream waits, loads
the item, dots it, and signals back.  AIV can produce item 1 while AIC
consumes item 0, so the pattern is the building block for the AscendC-style
4-head CV pipeline.

Observed on Ascend950DT_9582 / triton-ascend 3.6.0: PASS, maxerr <= 6.1e-5.

Usage (from the repo root):
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/supported/s6_scoped_cv_handoff.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def kern(a, c, ws, R: tl.constexpr, K: tl.constexpr):
    o_r = tl.arange(0, R)
    o_k = tl.arange(0, K)
    for s in tl.static_range(2):
        with al.scope("vector"):
            p = tl.load(a + s * R * K + o_r[:, None] * K + o_k[None, :])
            tl.store(ws + s * R * K + o_r[:, None] * K + o_k[None, :], p)
            al.sync_block_set(
                "vector",
                "cube",
                s,
                al.PIPE.PIPE_MTE3,
                al.PIPE.PIPE_MTE2,
            )
        with al.scope("cube"):
            al.sync_block_wait(
                "vector",
                "cube",
                s,
                al.PIPE.PIPE_MTE3,
                al.PIPE.PIPE_MTE2,
            )
            q = tl.load(ws + s * R * K + o_r[:, None] * K + o_k[None, :])
            tl.store(
                c + s * R * R + o_r[:, None] * R + o_r[None, :],
                tl.dot(q, tl.trans(q), out_dtype=tl.float32),
            )
            al.sync_block_set(
                "cube",
                "vector",
                s,
                al.PIPE.PIPE_FIX,
                al.PIPE.PIPE_MTE2,
            )


def main():
    torch.npu.set_device("npu:0")
    torch.manual_seed(0)
    R, K = 64, 128
    a = torch.randn(2, R, K, device="npu", dtype=torch.bfloat16)
    c = torch.zeros(2, R, R, device="npu", dtype=torch.float32)
    ws = torch.zeros(2, R, K, device="npu", dtype=torch.bfloat16)
    with torch.inference_mode():
        kern[(1,)](
            a,
            c,
            ws,
            R,
            K,
            enable_mixed_cv=True,
            disable_auto_inject_block_sync=True,
        )
        torch.npu.synchronize()
    ref = a.float() @ a.float().transpose(1, 2)
    maxerr = (c - ref).abs().max().item()
    verdict = "PASS" if maxerr < 1e-2 else "FAIL"
    print(f"RESULT: {verdict} s6_scoped_cv_handoff maxerr={maxerr:.6g}")


if __name__ == "__main__":
    main()
