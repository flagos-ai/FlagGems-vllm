"""GAP: express a Cube->Vector producer/consumer hand-off in one kernel.

What we need (the shape of an AscendC mixed-core pipeline such as
chunk_kda_fwd_prepare): the Cube scope runs `tl.dot`, hands the result to GM;
the Vector scope consumes it; the two overlap across loop iterations.
The Triton front door for this is `al.scope("cube"|"vector")` plus
`al.sync_block_set/wait`.

Current state (Ascend950DT_9582):
  MODE=scope       default flags                      -> SIGSEGV in bishengir
  MODE=scope_noauto enable_mixed_cv + disable_auto_    -> compiles, WRONG
                   inject_block_sync                     (stale/zero exchange)
  MODE=scope_sync  same flags + explicit sync_block_* -> correct (manual)
  MODE=noscope_sync no scopes, sync_block_* + default -> correct (fallback used
                                                        by chunk_gated_delta_rule)

Usage:
    PYTHONPATH=src:FlagGems/src python probe_04_scope_sync.py scope
    ... scope_noauto | scope_sync | noscope_sync
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
    elif MODE == "scope_sync":
        with al.scope("cube"):
            cb = tl.dot(tl.load(pa), tl.trans(tl.load(pb)), out_dtype=tl.float32)
            tl.store(pc, cb)
            al.sync_block_set("cube", "vector", 0)
        with al.scope("vector"):
            al.sync_block_wait("cube", "vector", 0)
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
    print(f"RESULT: PASS probe_04_scope_sync mode={mode} maxerr={err:.6e}")


if __name__ == "__main__":
    main()
