# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch
import triton.language as tl

pytestmark = [
    pytest.mark.fused_moe,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("m", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("gate", [True, False])
def test_w8a8_expert_schedule(m, gate, monkeypatch):
    if not torch.version.cuda or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("The TMA schedules target Hopper")
    pytest.importorskip("triton.experimental.tle.language")
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    from flaggems_vllm.ops import _hopper_w8a8_moe as kernel

    fused = importlib.import_module("flaggems_vllm.ops.fused_moe")
    n, div = (3072, 16) if gate else (1536, 1)
    bm = kernel._CONFIGS[(m, "gate_up" if gate else "down")][0]
    topk = 16
    torch.manual_seed(20261119)
    a = torch.randint(
        -128, 128, (m * topk // div, 1536), dtype=torch.int8, device="cuda"
    )
    w = torch.randint(-128, 128, (64, n, 1536), dtype=torch.int8, device="cuda")
    sa = torch.rand((a.shape[0], 1), device="cuda") * 0.002 + 0.001
    sb = torch.rand((64, n, 1), device="cuda") * 0.002 + 0.001
    probs = torch.rand((m, topk), device="cuda")
    routes = torch.randint(0, 512, (m, topk), device="cuda", dtype=torch.int32)
    expert_map = torch.full((512,), -1, device="cuda", dtype=torch.int32)
    expert_map[:64] = torch.arange(64, device="cuda", dtype=torch.int32)

    def align(route):
        return moe_align_block_size(
            route, bm, 512, expert_map, ignore_invalid_experts=True
        )

    metadata = align(routes)
    out = torch.empty((m, topk, n), device="cuda", dtype=torch.bfloat16)
    cfg = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    }
    invocations = []
    invoke = kernel.try_invoke

    def record(*args, **kwargs):
        hit = invoke(*args, **kwargs)
        invocations.append(hit)
        return hit

    monkeypatch.setattr(kernel, "try_invoke", record)

    def launch():
        fused.invoke_fused_moe_triton_kernel(
            a,
            w,
            out,
            sa,
            sb,
            probs,
            *metadata,
            not gate,
            div,
            cfg,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )

    out.fill_(-321)
    launch()
    assert invocations == [True]

    def check(route):
        flat = route.flatten()
        live = ((flat >= 0) & (flat < 64)).nonzero().flatten()
        sampled = live[:: max(1, len(live) // 16)][:16]
        experts = flat[sampled].long()
        dot = (
            torch.bmm(
                w[experts].double(),
                a[sampled // div].double().unsqueeze(-1),
            )
            .squeeze(-1)
            .float()
        )
        ref = dot * sa[sampled // div] * sb[experts].squeeze(-1)
        if not gate:
            ref = ref * probs.flatten()[sampled, None]
        torch.testing.assert_close(
            out.view(-1, n)[sampled], ref.bfloat16(), rtol=0, atol=0
        )
        untouched = ((flat < 0) | (flat >= 64)).nonzero().flatten()[::1024]
        assert torch.all(out.view(-1, n)[untouched] == -321)

    check(routes)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    assert invocations == [True, True]
    other = routes.roll(13, 0)
    other[::17, ::4] = -1
    for route in (other, routes):
        for destination, source in zip(metadata, align(route)):
            destination.copy_(source)
        out.fill_(-321)
        graph.replay()
        check(route)
