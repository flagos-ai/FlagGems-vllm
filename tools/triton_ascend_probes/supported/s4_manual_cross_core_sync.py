"""SUPPORTED: Cube -> Vector hand-off with manual cross-core flags.

`al.scope` still crashes in the default configuration (see
defects/d1_scope_auto_sync.py), but the underlying hardware hand-off is
expressible today with `al.sync_block_set/wait` and no `scope` at all.  This
is the pattern chunk_gated_delta_rule_fwd.py already uses and the one to
reach for when a kernel needs a Cube/Vector producer-consumer pair.

Observed on Ascend950DT_9582: PASS, maxerr <= 1.5e-5.

Usage (from the repo root):
    PYTHONPATH=src:FlagGems/src python \
        tools/triton_ascend_probes/supported/s4_manual_cross_core_sync.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.language.extra.cann import extension as al


@triton.jit
def kern(a, b, c, out, R: tl.constexpr, K: tl.constexpr):
    o_r = tl.arange(0, R)
    o_k = tl.arange(0, K)
    pa = a + o_r[:, None] * K + o_k[None, :]
    pb = b + o_r[:, None] * K + o_k[None, :]
    pc = c + o_r[:, None] * R + o_r[None, :]
    po = out + o_r[:, None] * R + o_r[None, :]

    cb = tl.dot(tl.load(pa), tl.trans(tl.load(pb)), out_dtype=tl.float32)
    tl.store(pc, cb)
    al.sync_block_set("cube", "vector", 0)
    al.sync_block_wait("cube", "vector", 0)
    tl.store(po, tl.load(pc) * 2.0)


def main():
    torch.npu.set_device("npu:0")
    R, K = 64, 128
    a = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    b = torch.randn(R, K, device="npu", dtype=torch.bfloat16)
    c = torch.zeros(R, R, device="npu", dtype=torch.float32)
    out = torch.zeros(R, R, device="npu", dtype=torch.float32)
    with torch.inference_mode():
        kern[(1,)](a, b, c, out, R, K)
        torch.npu.synchronize()
    ref = (a.float() @ b.float().T) * 2.0
    print(
        "RESULT: PASS s4_manual_cross_core_sync maxerr",
        (out - ref).abs().max().item(),
    )


if __name__ == "__main__":
    main()
