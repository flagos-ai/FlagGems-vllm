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
        # The bounded normalized-input case admits a stricter BF16 check than
        # the upstream unnormalized recurrence stress cases.
        torch.testing.assert_close(
            result[0].cpu().float(), expected, rtol=2e-2, atol=2e-3
        )
        assert bool(torch.isfinite(result[0]).all())

    for _ in range(3):
        call()
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    previous = result[0].clone()
    for value in (0.125, -0.25):
        v.fill_(value)
        result[0].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()
        assert not torch.equal(result[0], previous)
        previous = result[0].clone()


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX compiler contract")
def test_recompute_native_optout_preserves_default_selection(monkeypatch):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("flaggems_vllm.ops.FLA.wy_fast")
    calls = []

    class Launch:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def run(**kwargs):
                calls.append(self.name)

            return run

    monkeypatch.setattr(module, "_use_tle_recompute_w_u", lambda _: True)
    monkeypatch.setattr(module, "recompute_w_u_fwd_kernel", Launch("native"))
    monkeypatch.setattr(
        module, "recompute_w_u_fwd_tle_kernel", Launch("tle"), raising=False
    )
    monkeypatch.setattr(module.torch, "empty_like", lambda _: object())
    k = SimpleNamespace(shape=(1, 64, 2, 64), new_empty=lambda *a: object())
    v = SimpleNamespace(shape=(1, 64, 2, 64))
    a = SimpleNamespace(shape=(1, 64, 2, 64))
    module.recompute_w_u_fwd(k, v, None, None, a, None, use_tle=False)
    assert calls == ["native"]
    calls.clear()
    module.recompute_w_u_fwd(k, v, None, None, a, None)
    assert calls == ["tle"]


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX compiler contract")
def test_gdn_tle_load_capability_guard(monkeypatch):
    import importlib

    tle = importlib.import_module("triton.experimental.tle")
    gdn = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.fused.gdn_chunk"
    )
    monkeypatch.setattr(
        tle, "get_supported_primitives", lambda backend: frozenset({"load"})
    )
    assert gdn._supports_tle_load()
    monkeypatch.setattr(tle, "get_supported_primitives", lambda backend: frozenset())
    assert not gdn._supports_tle_load()
    monkeypatch.delattr(tle, "get_supported_primitives")
    assert not gdn._supports_tle_load()
