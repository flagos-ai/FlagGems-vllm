# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from . import conftest as cfg
from ._gemma4_routing_reference import gemma4_fused_routing_kernel_triton as reference

op = flaggems_vllm.gemma4_fused_routing_kernel_triton
pytestmark = [
    pytest.mark.gemma4_fused_routing_kernel_triton,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
SHAPES = [
    (0, 64, 4),
    (1, 128, 8),
    (17, 65, 3),
    (512, 64, 4),
    (1024, 128, 8),
    (512, 256, 8),
    (3, 32, 32),
    (2, 1024, 1),
]


@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32]
        if cfg.QUICK_MODE
        else [torch.float32, torch.float16, torch.bfloat16]
    ),
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("strided", [False, True])
def test_routing(dtype, shape, strided):
    t, e, k = shape
    torch.manual_seed(117)
    g = (
        torch.randn((t, e * 2), device="cuda", dtype=dtype)[:, ::2]
        if strided
        else torch.randn((t, e), device="cuda", dtype=dtype)
    )
    s = (
        torch.randn(e * 2, device="cuda", dtype=dtype)[::2]
        if strided
        else torch.randn(e, device="cuda", dtype=dtype)
    )
    pad = (torch.arange(t * 2, device="cuda") % 5 == 0)[::2] if strided else None
    w, i = op(g, k, s, is_padding=pad)
    assert w.dtype == torch.float32 and i.dtype == torch.int32
    assert w.shape == i.shape == (t, k)
    if not t:
        return
    # The pinned vLLM reference expects contiguous padding.
    rw, ri = reference(g, k, s, is_padding=None if pad is None else pad.contiguous())
    torch.testing.assert_close(i, ri, atol=0, rtol=0)
    torch.testing.assert_close(w, rw, atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize(
    "t,e,k", [(7, 65, 3), (512, 64, 4), (1024, 128, 8), (512, 256, 8)]
)
@pytest.mark.parametrize("mode", ["ties", "signed_zero", "nonfinite", "nan_payload"])
def test_edge_values(t, e, k, mode):
    g = torch.zeros((t, e), device="cuda")
    if mode == "ties":
        g.copy_(torch.randn_like(g).round())
    elif mode == "signed_zero":
        g[:, ::2] = -0.0
    elif mode == "nan_payload":
        g.copy_(
            torch.tensor([-1], dtype=torch.int32, device="cuda")
            .view(torch.float32)
            .expand_as(g)
        )
    else:
        g[0] = float("inf")
        g[1] = -float("inf")
        g[2, 0] = float("nan")
    s = torch.randn(e, device="cuda")
    p = torch.arange(t, device="cuda") % 7 == 0
    w, i = op(g, k, s, is_padding=p)
    rw, ri = reference(g, k, s, is_padding=p)
    torch.testing.assert_close(i, ri, atol=0, rtol=0)
    torch.testing.assert_close(w, rw, atol=1e-6, rtol=1e-5, equal_nan=True)


def test_validation_and_graph_cache():
    g = torch.randn((512, 64), device="cuda")
    s = torch.ones(64, device="cuda")
    for k in [0, 65]:
        with pytest.raises(ValueError):
            op(g, k, s)
    with pytest.raises(ValueError):
        op(g, 4, s[:2])
    with pytest.raises(NotImplementedError):
        op(g.int(), 4, s)
    with pytest.raises(NotImplementedError):
        op(torch.empty((1, 2048), device="cuda"), 4, torch.ones(2048, device="cuda"))
    expected = op(g, 4, s)
    import importlib

    module = importlib.import_module(op.__module__)
    entry = getattr(module, "_gemma4_hopper_kernel", None)
    if entry is None:
        entry = module._gemma4_native_kernel
    size = len(entry.kernel_cache[torch.cuda.current_device()])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(g, 4, s)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert len(entry.kernel_cache[torch.cuda.current_device()]) == size
    assert entry.fn.best_config.kwargs["BLOCK_T"] in [1, 4, 8, 16]


@pytest.mark.parametrize("shape", [(17, 65, 3), (512, 128, 4), (3, 32, 32)])
def test_generic_and_missing_tle(shape, monkeypatch):
    import importlib

    generic = importlib.import_module(
        "flaggems_vllm.ops.gemma4_fused_routing_kernel_triton"
    )
    backend = importlib.import_module(op.__module__)
    if hasattr(backend, "_HAS_LAYOUT"):
        monkeypatch.setattr(backend, "_HAS_LAYOUT", False)
    t, e, k = shape
    g = torch.randn((t, e), device="cuda")
    s = torch.randn(e, device="cuda")
    rw, ri = reference(g, k, s)
    for function in [generic.gemma4_fused_routing_kernel_triton, op]:
        w, i = function(g, k, s)
        torch.testing.assert_close(i, ri, atol=0, rtol=0)
        torch.testing.assert_close(w, rw, atol=1e-6, rtol=1e-5)
