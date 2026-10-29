# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exploratory gate fusion using the existing MetaX packed-decode primitive.

Set GDN_CURRENT_IMPL, GDN_SHAPE_FILE and optionally GDN_RESULT_DIR, as in
benchmark/test_gdn_w8a16_compare.py. This is a separate scope from the core
benchmark: raw a/b/A_log/dt_bias are generated and gate creation is timed.
Both fused variants use the upstream primitive, differing only in launch
configuration. All paths return active final states without mutating inputs;
fused primitives are adapted with a distinct output-state buffer. Identity
cache indices start at one; normalization and speculative decoding are off.
This experiment does not add a new public operator API.
"""
import gc
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch
import triton
from test_gdn_w8a16_compare import load_module


def main():
    from flaggems_vllm.runtime.backend._metax.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode_kernel as fused,
    )

    root = Path(os.environ.get("GDN_RESULT_DIR", "gdn_w8a16_results"))
    root.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("GDN_CURRENT_IMPL")
    if not path:
        raise ValueError("Set GDN_CURRENT_IMPL to the frozen current implementation")
    baseline = load_module("gdn_current_fusion_reference", Path(path))
    bench = load_module(
        "gdn_fusion_inputs", Path(__file__).with_name("test_gdn_w8a16.py")
    )
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads(bench.SHAPES.read_text())
    results = []
    for idx, row in enumerate(rows, 1):
        if len(row["shape"]) != 4:
            continue
        n, dim, hv, dv = row["shape"]
        h, dk = 16, 128
        mixed = torch.randn(n, dim, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(n, hv, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        alog = torch.randn(hv, device="cuda") * 0.1
        bias = torch.randn(hv, device="cuda") * 0.1
        state = torch.randn(n + 1, hv, dv, dk, device="cuda") * 0.1
        before = state.clone()
        ids = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
        cu = torch.arange(n + 1, device="cuda", dtype=torch.int32)
        bounds = list(range(n + 1))

        def args():
            q = mixed[:, : h * dk].view(1, n, h, dk)
            k = mixed[:, h * dk : 2 * h * dk].view(1, n, h, dk)
            v = mixed[:, 2 * h * dk :].view(1, n, hv, dv)
            g = (
                -torch.exp(alog) * torch.nn.functional.softplus(a.float() + bias)
            ).unsqueeze(0)
            beta = b.float().sigmoid().to(b.dtype).float().unsqueeze(0)
            return q, k, v, g, beta, state[1:], cu

        def old():
            return baseline.gdn_w8a16(*args())

        def ref():
            return bench.torch_baseline(args(), bounds)

        def new(bv, warps):
            o = torch.empty(n, hv, dv, device="cuda", dtype=mixed.dtype)
            f = torch.empty_like(state)
            fused[(triton.cdiv(dv, bv), n * hv)](
                mixed,
                a,
                b,
                alog,
                bias,
                o,
                state,
                f,
                ids,
                dk**-0.5,
                mixed.stride(0),
                a.stride(0),
                b.stride(0),
                state.stride(0),
                f.stride(0),
                ids.stride(0),
                h,
                hv,
                dk,
                dv,
                128,
                bv,
                20.0,
                False,
                True,
                num_warps=warps,
                num_stages=1,
            )
            return o.unsqueeze(0), f[1:]

        native_bv = 32 if (dv // 32) * n * hv >= 512 else 16
        fns = {
            "current_full": old,
            "pytorch_full": ref,
            "optimized_fused": partial(new, 32 if n >= 3 else 16, 4),
            "metax_fused": partial(new, native_bv, 1),
        }
        expected = ref()
        graphs = {}
        outs = {}
        valid = {}
        samples = {k: [] for k in fns}
        for name, fn in fns.items():
            out = fn()
            valid[name] = True
            for aa, bb in zip(out, expected):
                try:
                    torch.testing.assert_close(
                        aa.float(), bb.float(), atol=0.002, rtol=0.02
                    )
                except AssertionError:
                    valid[name] = False
            graphs[name], outs[name] = bench.capture(fn)
            torch.testing.assert_close(state, before, atol=0, rtol=0)
        keys = list(graphs)
        for sample in range(31):
            for name in keys[sample % 4 :] + keys[: sample % 4]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(3):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / 3)
        med = {k: statistics.median(v) for k, v in samples.items()}
        results.append(dict(index=idx, shape=row["shape"], us=med, valid=valid))
        print(idx, med, valid, flush=True)
        (root / "fusion_comparison.json").write_text(json.dumps(results, indent=2))
        del graphs, outs, fns, expected, out
        gc.collect()
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
