# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.vit_attention,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def reference(q, k, v, scale):
    return (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(-3, -2).float(),
            k.transpose(-3, -2).float(),
            v.transpose(-3, -2).float(),
            scale=scale,
            enable_gqa=True,
        )
        .transpose(-3, -2)
        .to(q.dtype)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [64, 80])
@pytest.mark.parametrize("scale", [None, 0.0, -0.125])
@pytest.mark.parametrize("strided", [False, True])
def test_vit_attention(dtype, dim, scale, strided):
    torch.manual_seed(41)
    q = torch.randn((2, 17, 4, dim * (2 if strided else 1)), device="cuda", dtype=dtype)
    k = torch.randn((2, 23, 2, dim * (2 if strided else 1)), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    if strided:
        q, k, v = q[..., ::2], k[..., ::2], v[..., ::2]
    actual = flaggems_vllm.vit_flash_attn_wrapper(q, k, v, 2, scale=scale)
    torch.testing.assert_close(
        actual,
        reference(q, k, v, scale),
        atol=0.016 if dtype == torch.bfloat16 else 0.003,
        rtol=0.016 if dtype == torch.bfloat16 else 0.003,
    )


def test_vit_packed_changed_boundaries_graph():
    q = torch.randn((12, 2, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 5, 12], device="cuda", dtype=torch.int32)

    def run():
        return flaggems_vllm.vision_flash_attn_varlen(q, k, v, cu, cu)

    run()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run()
    cu.copy_(torch.tensor([0, 3, 12], device="cuda", dtype=torch.int32))
    q.mul_(-1)
    g.replay()
    expected = torch.cat(
        [
            reference(
                q[a:b].unsqueeze(0), k[a:b].unsqueeze(0), v[a:b].unsqueeze(0), None
            )[0]
            for a, b in ((0, 3), (3, 12))
        ]
    )
    torch.testing.assert_close(out, expected, atol=0.016, rtol=0.016)


def test_vit_empty_and_invalid():
    q = torch.empty((1, 0, 2, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
    assert flaggems_vllm.vit_flash_attn_wrapper(q, k, k, 1).shape == q.shape
    q = torch.randn((1, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((1, 0, 2, 64), device="cuda", dtype=torch.bfloat16)
    assert bool((flaggems_vllm.vit_flash_attn_wrapper(q, k, k, 1) == 0).all())
    with pytest.raises(ValueError):
        flaggems_vllm.vit_flash_attn_wrapper(q, k, k, 2)
