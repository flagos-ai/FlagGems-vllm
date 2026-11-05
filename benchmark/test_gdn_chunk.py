# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Host+device latency for existing GDN metadata fallback vs caller skew hint.

Fresh GPU boundary identities model a new scheduler step; their upload and
the caller's existing metadata construction are outside each measured scope.
This isolates the hint path; it is not a reproduction of historical whole-model speedup.
"""
import importlib
import json
import statistics
import time

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.chunk_gated_delta_rule,
    pytest.mark.skipif(
        flaggems_vllm.runtime.device.vendor_name != "metax", reason="MetaX GDN metadata"
    ),
]


@pytest.mark.parametrize("tokens,nreq", [(256, 2), (2048, 4), (8192, 4)])
def test_metax_gdn_metadata_benchmark(tokens, nreq):
    op = importlib.import_module("flaggems_vllm.runtime.backend._metax.fused.gdn_chunk")
    index = importlib.import_module("flaggems_vllm.ops.FLA.index")
    lengths = [64] * (nreq - 1) + [tokens - 64 * (nreq - 1)]
    counts = [(x + 63) // 64 for x in lengths]
    skew = max(counts) / (sum(counts) / nreq)
    hint = 0 if skew <= 1.15 else 1 if skew <= 1.5 else 2 if skew <= 2.5 else 3
    boundaries = [0]
    for x in lengths:
        boundaries.append(boundaries[-1] + x)
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    q = torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, tokens, 6, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    g = -torch.rand(1, tokens, 6, device="cuda") * 0.1
    beta = torch.rand(1, tokens, 6, device="cuda", dtype=torch.bfloat16)
    state = torch.zeros(nreq, 6, 128, 128, device="cuda")
    ci = index.prepare_chunk_indices(cu, 64)
    co = index.prepare_chunk_offsets(cu, 64)

    def run(h, local_cu, local_co):
        return op.chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            128**-0.5,
            state,
            True,
            cu_seqlens=local_cu,
            chunk_indices=ci,
            chunk_offsets=local_co,
            chunk_skew_bucket=h,
        )

    for _ in range(2):
        run(None, cu, co)
        run(hint, cu, co)
    values = {"fallback": [], "caller": []}
    for name in ["fallback", "caller", "caller", "fallback"]:
        samples = []
        for _ in range(8):
            local_cu = cu.clone()
            local_co = co.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run(None if name == "fallback" else hint, local_cu, local_co)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        values[name].append(statistics.median(samples))
    med = {name: statistics.median(x) for name, x in values.items()}
    print(
        json.dumps(
            {
                "tokens": tokens,
                "requests": nreq,
                "dtype": "bf16",
                "latency_ms": med,
                "speedup": med["fallback"] / med["caller"],
                "scope": "full GDN call with fresh metadata identities; optional hint comparison",
            }
        ),
        flush=True,
    )
    assert med["fallback"] / med["caller"] >= 0.9
