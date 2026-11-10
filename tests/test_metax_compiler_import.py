# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Importing the MetaX backend must not replace compiler entry points."""
import subprocess
import sys
from pathlib import Path

import pytest

import flaggems_vllm as ops


@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX compiler contract")
def test_backend_import_preserves_compiler_entrypoint():
    source_root = str(Path(ops.__file__).resolve().parents[1])
    code = (
        f"import sys; expected_root = {source_root!r}; sys.path.insert(0, expected_root)\n"
        + """import triton.backends.metax.compiler as compiler
before = compiler.metax.translate_llvmir_to_mcfatbin
import flaggems_vllm
assert flaggems_vllm.__file__.startswith(expected_root + "/")
assert compiler.metax.translate_llvmir_to_mcfatbin is before
"""
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX compiler contract")
@pytest.mark.parametrize("tokens", [64, 128])
def test_gdn_after_compiler_isolation_reference_and_graph(tokens):
    import torch
    import torch.nn.functional as functional

    from .test_FLA.test_chunk_gated_delta_rule_fwd import (
        naive_chunk_gated_delta_rule_fwd,
    )

    generator = torch.Generator(device="cuda").manual_seed(1024)
    q = functional.normalize(
        torch.randn(1, tokens, 2, 64, device="cuda", generator=generator), dim=-1
    ).bfloat16()
    k = functional.normalize(
        torch.randn(1, tokens, 2, 64, device="cuda", generator=generator), dim=-1
    ).bfloat16()
    v = (
        torch.randn(1, tokens, 2, 64, device="cuda", generator=generator) * 0.1
    ).bfloat16()
    g = functional.logsigmoid(
        torch.randn(1, tokens, 2, device="cuda", generator=generator)
    ).bfloat16()
    beta = torch.rand(1, tokens, 2, device="cuda", generator=generator).bfloat16()
    result = []

    def call():
        values = ops.chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=64**-0.5,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        result[:] = [values[1]]

    def check():
        expected, _ = naive_chunk_gated_delta_rule_fwd(
            q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), 64**-0.5, None
        )
        # Match the established CUDA/native test tolerance; do not change its gate.
        torch.testing.assert_close(
            result[0].cpu().float(), expected, rtol=1e-1, atol=2e-1
        )
        assert bool(torch.isfinite(result[0]).all())

    for _ in range(3):
        call()
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.125, -0.25):
        v.fill_(value)
        result[0].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()
