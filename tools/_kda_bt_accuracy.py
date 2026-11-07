"""Accuracy probe for Ascend chunk_kda at BT in {16, 32, 64}.

The factored decay exp2(g_i) * exp2(-g_j) grows its dynamic range with BT, and
the operands are cast to bf16 (8 mantissa bits) for tl.dot. BT=64 previously
produced NaN. This checks each BT against the AscendC reference on the harness
shapes and reports non-finite counts, so a BT change is never adopted on
timing alone.
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kda_shape_bench import (  # noqa: E402
    ASCENDC_CHUNK_SIZE,
    CHUNK_CASES,
    GATE_LOWER_BOUND,
    HEAD_DIM,
    kda_inputs,
)


def relax_validator():
    _shared = sys.modules["flaggems_vllm.ops.FLA.chunk_kda"]
    _asc = sys.modules["flaggems_vllm.runtime.backend._ascend.FLA.chunk_kda"]
    _orig = _shared._validate_chunk_kda_inputs

    def _patched(*a, **kw):
        kw["chunk_size"] = 16
        return _orig(*a, **kw)

    _shared._validate_chunk_kda_inputs = _patched
    _asc._validate_chunk_kda_inputs = _patched


def main():
    from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
    from flaggems_vllm.runtime.backend._ascend.FLA.chunk_kda import chunk_kda

    sys.path.insert(0, "/workspace/FlagGems/tools")
    from test_kda_vs_vllm_ascend import _import_sides

    ops = _import_sides()
    relax_validator()

    dev = torch.device("npu:0")
    dtype = torch.bfloat16
    bts = [int(x) for x in os.environ.get("BT_LIST", "16,32,64").split(",")]

    print(
        f"{'case':26s} {'BT':>3s} {'out max_abs':>12s} {'state max_abs':>14s} "
        f"{'nonfinite':>10s}"
    )
    for tokens, heads, nseq in CHUNK_CASES:
        per = (tokens // nseq // ASCENDC_CHUNK_SIZE) * ASCENDC_CHUNK_SIZE
        cu_list = [j * per for j in range(nseq)] + [tokens]
        cu = torch.tensor(cu_list, dtype=torch.int64, device=dev)
        q, k, v, raw_g, raw_beta, a_log, dt_bias = kda_inputs(heads, tokens, dev, dtype)
        init_vk = torch.randn(
            nseq, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=dev
        )

        ci_ref = ops["prepare_chunk_indices"](cu, ASCENDC_CHUNK_SIZE)
        want_o, want_s = ops["run_chunk_kda"](
            q,
            k,
            v,
            raw_g,
            raw_beta.sigmoid(),
            init_vk.clone(),
            list(cu_list),
            ci_ref.flatten().cpu().tolist(),
            a_log,
            dt_bias,
            lower_bound=GATE_LOWER_BOUND,
        )
        torch.npu.synchronize()

        name = f"T={tokens} H={heads} seq={nseq}"
        for bt in bts:
            ci = prepare_chunk_indices(cu, bt)
            got_o, got_s = chunk_kda(
                q,
                k,
                v,
                raw_g,
                raw_beta.to(q.dtype),
                scale=HEAD_DIM**-0.5,
                initial_state=init_vk.clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
                state_v_first=True,
                cu_seqlens=cu,
                chunk_indices=ci,
                chunk_size=bt,
                A_log=a_log,
                dt_bias=dt_bias,
            )
            torch.npu.synchronize()
            nf = int((~torch.isfinite(got_o)).sum().item()) + int(
                (~torch.isfinite(got_s)).sum().item()
            )
            do = (got_o.float() - want_o.float()).abs().max().item()
            ds = (got_s.float() - want_s.float()).abs().max().item()
            print(f"{name:26s} {bt:3d} {do:12.3e} {ds:14.3e} {nf:10d}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
