# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Additional integration checks for cached attention."""

import pytest
import torch

import flaggems_vllm
from tests.test_flash_attn_varlen_mthreads import make_case, reference

pytest_plugins = ("tests.mthreads_attention_test_utils",)

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "mthreads", reason="MThreads backend only"
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.0625])
def test_nonpositive_scale(monkeypatch, dtype, scale):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import cached

    if cached.launch_prefill is None:
        pytest.skip("Triton 3.6 TLE is required")
    qlens, klens = [65, 129, 318], [0, 97, 193]
    args = make_case(dtype, True, qlens, klens, heads=(4, 1), dim=256)
    args.update(causal=True, softmax_scale=scale, out=torch.empty_like(args["q"]))
    calls = []
    launch = cached.launch_prefill

    def record(*a, **kw):
        compiled = launch(*a, **kw)
        calls.append(compiled)
        assert compiled.n_spills == 0
        return compiled

    monkeypatch.setattr(cached, "launch_prefill", record)
    expected, expected_lse = reference(args, qlens, klens)
    actual, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    assert actual is args["out"] and calls
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0.002)
