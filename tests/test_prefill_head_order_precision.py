# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close
from tests.test_flash_attn_varlen_func_w8a8_int8 import _lse_atol, _run_case

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX head ordering"
    ),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mixed", [False, True])
def test_long_prefill_output_and_lse(dtype, mixed):
    _run_case(
        [1, 1024] if mixed else [1024],
        [16384, 16384] if mixed else [16384],
        dim=256,
        heads=4,
        kvheads=2,
        paged=True,
        causal=True,
        dtype=dtype,
        broadcast_scales=True,
        return_lse=True,
    )


@pytest.mark.parametrize(
    "scale",
    [-0.125, 0.0, (0.125, 1.7), (0.0625, 1.7)],
    ids=["negative", "zero", "both-residual-heads", "mixed-precision-branches"],
)
def test_long_prefill_residual_head_selection(scale):
    # H=8/GQA=4 exercises the reordered dispatch, including different residual
    # decisions for the two KV heads. At nk=16384, scales 0.0625 and 1.7
    # straddle the existing compensation bound; 0.125 also selects residual.
    # Opposite V rows expose lost compensation.
    device = flaggems_vllm.device
    nq, nk, heads, kvheads, dim = 1024, 16384, 8, 2, 256
    gap = 0.000244
    q = torch.full((nq, heads, dim), 127, dtype=torch.int8, device=device)
    q[:, :, 0] = 1
    k = torch.full((nk // 16, 16, kvheads, dim), -127, dtype=torch.int8, device=device)
    k[0, :2] = 0
    k[0, 1, :, 0] = 1
    v = torch.zeros_like(k)
    v[0, 0] = 127
    v[0, 1] = -127
    qs = torch.ones((1, heads, 1), device=device).expand(1, heads, nq // 128)
    ks = torch.full((1, kvheads, 1), gap * math.sqrt(dim), device=device).expand(
        1, kvheads, nk // 128
    )
    vscale = torch.as_tensor(scale, dtype=torch.float32, device=device).expand(kvheads)
    vs = vscale[None, :, None].expand_as(ks)
    cuq = torch.tensor([0, nq], dtype=torch.int32, device=device)
    used = torch.tensor([nk], dtype=torch.int32, device=device)
    table = torch.arange(nk // 16, dtype=torch.int32, device=device)[None, :]
    # A skipped head must fail even if the allocator reuses an earlier correct output.
    out = torch.full(q.shape, float("nan"), dtype=torch.bfloat16, device=device)
    actual, lse = flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        nq,
        cuq,
        nk,
        seqused_k=used,
        block_table=table,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=True,
        out=out,
        return_softmax_lse=True,
    )
    expected = (-127 * vscale.repeat_interleave(heads // kvheads) * math.tanh(gap / 2))[
        None, :, None
    ].expand_as(actual)
    gems_assert_close(actual.float(), expected, dtype=torch.float32, atol=0.025)
    expected_lse = torch.full_like(lse, math.log1p(math.exp(gap)))
    gems_assert_close(lse, expected_lse, dtype=torch.float32, atol=_lse_atol())
    assert actual.data_ptr() == out.data_ptr()
