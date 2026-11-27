# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import importlib

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.chunk_gated_delta_rule,
    pytest.mark.skipif(
        flaggems_vllm.runtime.device.vendor_name != "metax",
        reason="MetaX metadata regression",
    ),
]


@pytest.mark.parametrize(
    "lengths,hint",
    [([64, 64], 0), ([64, 128], 1), ([64, 256], 2), ([64, 64, 64, 1024], 3)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_caller_metadata_preserves_output_and_state(monkeypatch, lengths, hint, dtype):
    op = importlib.import_module("flaggems_vllm.runtime.backend._metax.fused.gdn_chunk")
    wy = importlib.import_module("flaggems_vllm.ops.FLA.wy_fast")
    index = importlib.import_module("flaggems_vllm.ops.FLA.index")
    t = sum(lengths)
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.randn(1, t, 2, 128, device="cuda", dtype=dtype) * 0.05
    k = torch.randn_like(q) * 0.05
    v = torch.randn(1, t, 6, 128, device="cuda", dtype=dtype) * 0.05
    g = -torch.rand(1, t, 6, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand(1, t, 6, device="cuda", dtype=dtype)
    state = (
        torch.randn(len(lengths), 6, 128, 128, device="cuda", dtype=torch.float32)
        * 0.01
    )
    ci = index.prepare_chunk_indices(cu, 64)
    co = index.prepare_chunk_offsets(cu, 64)
    args = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=128**-0.5,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=ci,
        chunk_offsets=co,
    )
    expected = op.chunk_gated_delta_rule_fwd(**args)

    def forbidden(*a, **kw):
        raise AssertionError("caller metadata was recomputed or device skew read back")

    monkeypatch.setattr(wy, "prepare_chunk_indices", forbidden)
    monkeypatch.setattr(op, "_skew_bucket", forbidden)
    actual = op.chunk_gated_delta_rule_fwd(**args, chunk_skew_bucket=hint)
    for i in [0, 1, 2, 3]:
        if isinstance(expected[i], torch.Tensor):
            torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
    assert actual[3].dtype == torch.float32


@pytest.mark.parametrize("hint", [-1, 4, True, 1.0])
def test_invalid_hint_is_rejected(hint):
    op = importlib.import_module("flaggems_vllm.runtime.backend._metax.fused.gdn_chunk")
    q = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="chunk_skew_bucket"):
        op.chunk_gated_delta_rule_fwd_h_vmajor(
            q, q, q, chunk_size=64, chunk_skew_bucket=hint
        )
