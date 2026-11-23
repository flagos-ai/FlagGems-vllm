# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib
import math

import pytest
import torch

import flaggems_vllm as gems
from tests.accuracy_utils import gems_assert_close
from tests.test_flash_attn_varlen_func_w8a8_int8 import _run_case

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(gems.vendor_name != "metax", reason="MetaX mixed split"),
]


@pytest.fixture
def dispatch_audit(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.flash_attention.launcher"
    )
    calls = []
    original = module.flash_varlen_int8_mixed_short

    class Audit:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                calls.append(grid)
                return original[grid](*args, **kwargs)

            return call

    monkeypatch.setattr(module, "flash_varlen_int8_mixed_short", Audit())
    return calls


@pytest.mark.parametrize(
    "shortq,longk",
    [(1, 8192), (12, 8193), (31, 65536), (32, 8193), (33, 8193), (12, 8191)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_mixed_short_split_reference(shortq, longk, dtype, causal, dispatch_audit):
    _run_case(
        [0, shortq, 1025],
        [0, longk, 1057],
        dim=256,
        heads=4,
        kvheads=1,
        paged=True,
        causal=causal,
        dtype=dtype,
        broadcast_scales=True,
        return_lse=True,
    )
    assert bool(dispatch_audit) == (longk >= 8192)


@pytest.mark.parametrize("scale", [1.7, 16.0, -16.0])
@pytest.mark.parametrize("shortq", [1, 32])
def test_mixed_short_split_cancellation(scale, shortq, dispatch_audit):
    dev = gems.device
    length = 8193
    page = 16
    pages = (length + page - 1) // page
    # Two distant relevant keys exercise different KV partitions for every
    # short query. A separate zero-valued prefill request checks output scatter.
    q = torch.full((shortq + 1024, 4, 256), 127, dtype=torch.int8, device=dev)
    q[:, :, 0] = 1
    k = torch.full((2 * pages, page, 1, 256), -127, dtype=torch.int8, device=dev)
    v = torch.zeros_like(k)
    k[pages:] = 0
    k[0, 0] = 0
    k[256, 0] = 0
    k[256, 0, 0, 0] = 1
    v[0, 0] = 127
    v[256, 0] = -127
    cu = torch.tensor([0, shortq, shortq + 1024], dtype=torch.int32, device=dev)
    used = torch.tensor([length, 1024], dtype=torch.int32, device=dev)
    table = torch.arange(2 * pages, dtype=torch.int32, device=dev).reshape(2, pages)
    qs = torch.ones((2, 4, 1), device=dev).expand(2, 4, 8)
    gap = 0.000244
    ks = torch.full((2, 1, 1), gap * math.sqrt(256), device=dev).expand(2, 1, 65)
    vs = torch.full((2, 1, 1), scale, device=dev).expand(2, 1, 65)
    actual, lse = gems.flash_attn_varlen_func(
        q,
        k,
        v,
        1024,
        cu,
        length,
        seqused_k=used,
        block_table=table,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=True,
        return_softmax_lse=True,
    )
    expected = torch.full_like(
        actual[:shortq].float(), -127 * scale * math.tanh(gap / 2)
    )
    gems_assert_close(
        actual[:shortq].float(), expected, dtype=torch.float32, atol=0.025
    )
    expected_lse = torch.full_like(lse[:, :shortq], math.log1p(math.exp(gap)))
    gems_assert_close(lse[:, :shortq], expected_lse, dtype=torch.float32, atol=0.0015)
    assert dispatch_audit, "cancellation must exercise the split producer"
    gems_assert_close(
        actual[shortq:].float(),
        torch.zeros_like(actual[shortq:].float()),
        dtype=torch.float32,
        atol=0.025,
    )
    gems_assert_close(
        lse[:, shortq:],
        torch.arange(1, 1025, device=dev, dtype=torch.float32)
        .log()[None, :]
        .expand_as(lse[:, shortq:]),
        dtype=torch.float32,
        atol=0.0015,
    )
