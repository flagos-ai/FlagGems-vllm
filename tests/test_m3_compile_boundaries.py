# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("m", [0, 4])
def test_m3_compiled_router_and_routing(m):
    x = torch.randn((m, 6144), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 6144), device="cuda") * 0.01
    w = torch.empty((m, 8), device="cuda")
    ids = torch.empty((m, 8), device="cuda", dtype=torch.int32)
    src = torch.empty_like(ids)

    def forward(x, weight, w, ids, src):
        logits = flaggems_vllm.fp32_router_gemm(x, weight)
        flaggems_vllm.topk_sigmoid(w, ids, src, logits, True)
        return logits

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    out = compiled(x, weight, w, ids, src)
    torch.testing.assert_close(
        out, flaggems_vllm.fp32_router_gemm(x, weight), rtol=0, atol=0
    )
    assert ids.shape == (m, 8)


def test_m3_compiled_mutation_and_vision():
    x = torch.randn((3, 1024), device="cuda", dtype=torch.bfloat16)
    original = x.clone()
    w = torch.zeros((128,), device="cuda", dtype=torch.bfloat16)
    cs = torch.randn((16, 64), device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(3, device="cuda", dtype=torch.int64)
    qo = torch.empty((3, 512), device="cuda", dtype=torch.bfloat16)

    def forward(x, qo):
        flaggems_vllm.fused_minimax_m3_qknorm_rope_kv_insert(
            x, w, w, cs, pos, 4, 2, 64, 1e-6, q_out=qo
        )
        return qo

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    out = compiled(x, qo).clone()
    x.copy_(original)
    forward(x, qo)
    assert torch.equal(out, qo)
    q = torch.randn((1, 17, 2, 64), device="cuda", dtype=torch.bfloat16)
    vision = torch.compile(
        lambda q: flaggems_vllm.vit_flash_attn_wrapper(q, q, q, 1),
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        vision(q), flaggems_vllm.vit_flash_attn_wrapper(q, q, q, 1), rtol=0, atol=0
    )
