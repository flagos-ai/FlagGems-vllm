# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace

import flaggems_vllm
import pytest
import torch

pytestmark = [
    pytest.mark.fused_marlin_moe_w4a16_int4,
    pytest.mark.skipif(
        flaggems_vllm.device != "npu", reason="Ascend-specific validation"
    ),
]


@pytest.fixture(scope="module")
def utils():
    module = importlib.import_module("benchmark.test_fused_marlin_moe_w4a16_int4")
    return SimpleNamespace(
        weights=module._ascend_weights,
        inputs=module._ascend_inputs,
        baseline=module._ascend_baseline,
        reference=module._ascend_reference,
        gems_call=module._ascend_gems_call,
    )


@pytest.fixture(scope="module")
def ww(utils):
    return utils.weights(4, 256, 128, torch.bfloat16)


def test_public_registration():
    assert flaggems_vllm.fused_marlin_moe_w4a16_int4.__module__.startswith(
        "flaggems_vllm.runtime.backend._ascend"
    )
    assert "fused_marlin_moe_w4a16_int4" in flaggems_vllm.ops.__all__


@pytest.mark.parametrize("m", [0, 1, 2, 3, 7, 8, 17, 32, 33, 40, 257])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_accuracy(utils, ww, m, id_dtype):
    x, p, ids = utils.inputs(m, 4, 256, 2)
    ids = ids.to(id_dtype)
    got = utils.gems_call(x, ww, p, ids)
    if m == 0:
        assert got.shape == x.shape and got.dtype == x.dtype
        return
    expected = utils.baseline(x, ww, p, ids)
    torch.testing.assert_close(got, expected, rtol=0.01, atol=0.001)
    if m <= 8:
        torch.testing.assert_close(
            got.cpu(), utils.reference(x, ww, p, ids), rtol=0.01, atol=0.001
        )


@pytest.mark.parametrize(
    "mode", ["single_expert", "duplicate_routes", "zero_probabilities"]
)
def test_routing_edges(utils, ww, mode):
    x, p, ids = utils.inputs(33, 4, 256, 2)
    if mode == "single_expert":
        ids.fill_(0)
    elif mode == "duplicate_routes":
        ids[:, 1] = ids[:, 0]
    else:
        p.zero_()
    torch.testing.assert_close(
        utils.gems_call(x, ww, p, ids),
        utils.baseline(x, ww, p, ids),
        rtol=0.02,
        atol=0.02,
    )


def test_mutation_invalidates_packed_cache(utils):
    w = utils.weights(4, 256, 128, torch.bfloat16)
    x, p, ids = utils.inputs(2, 4, 256, 2)
    ids.fill_(0)
    utils.gems_call(x, w, p, ids)
    w[0][0][0].bitwise_xor_(0x10)
    w[0][1][0].mul_(1.25)
    got = utils.gems_call(x, w, p, ids)
    torch.testing.assert_close(
        got.cpu(), utils.reference(x, w, p, ids), rtol=0.01, atol=0.001
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 32},
        {"activation": "relu"},
        {"inplace": True},
        {"apply_router_weight_on_input": True},
        {"global_num_experts": 8},
    ],
)
def test_unsupported_options(utils, ww, kwargs):
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    x, p, ids = utils.inputs(1, 4, 256, 2)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_marlin_moe_w4a16_int4(
            x,
            ww[0][0],
            ww[1][0],
            None,
            None,
            ww[0][1],
            ww[1][1],
            p,
            ids,
            QUANT_TYPE_UINT4B8,
            **kwargs,
        )


def test_fp16_explicitly_unsupported(utils, ww):
    x, p, ids = utils.inputs(1, 4, 256, 2)
    with pytest.raises(NotImplementedError):
        utils.gems_call(x.half(), ww, p, ids)


def test_noncontiguous_ids_rejected(utils, ww):
    x, p, ids = utils.inputs(2, 4, 256, 2)
    ids = ids.long().T
    with pytest.raises(NotImplementedError):
        utils.gems_call(x, ww, p, ids)


@pytest.mark.parametrize("scale", [0.03, 2**-14, 2**-20, 0.0, -0.03, 4096.0, 8192.0])
def test_exact_half_fast_path_and_fp32_fallback(utils, scale):
    import torch_npu
    from flaggems_vllm.runtime.backend._ascend.ops.fused_marlin_moe_w4a16_int4 import (
        _gemm as gemm,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.fused_marlin_moe_w4a16_int4 import (
        _prepare_weights as prepare,
    )

    weights = utils.weights(4, 256, 128, torch.bfloat16)
    w, s, native_w, _, _ = weights[0]
    s.fill_(scale)
    _, _, safe = prepare(w, s)
    expected_safe = scale == 0 or (2**-14 <= abs(float(s.flatten()[0])) <= 4096)
    assert bool(safe.bool().all()) == expected_safe
    x = torch.randn((32, 256), device="npu", dtype=torch.bfloat16) * 0.1
    expert_ids = torch.tensor([0, 1], device="npu", dtype=torch.int32)
    output = torch.empty((32, 256), device="npu", dtype=x.dtype)
    gemm(x, w, s, expert_ids, output, 16, 256)
    native_s = s.transpose(1, 2).contiguous()
    counts = torch.tensor([16, 16, 0, 0], device="npu", dtype=torch.int64)
    expected = torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[native_w],
        antiquant_scale=[native_s],
        antiquant_offset=[torch.zeros_like(native_s)],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=counts,
        output_dtype=x.dtype,
    )[0]
    torch.testing.assert_close(
        output, expected, rtol=0.01, atol=max(1e-6, abs(scale) * 0.001)
    )


@pytest.mark.parametrize("n,active", [(128, 4080), (256, 4080), (256, 4096)])
def test_batched_silu(utils, n, active):
    import torch_npu
    from flaggems_vllm.runtime.backend._ascend.ops.fused_marlin_moe_w4a16_int4 import (
        _silu as silu,
    )

    x = torch.randn((4096, 2 * n), device="npu", dtype=torch.bfloat16)
    out = torch.empty((4096, n), device="npu", dtype=x.dtype)
    offsets = torch.tensor([0, active], device="npu", dtype=torch.int32)
    silu(x, out, offsets)
    expected = torch_npu.npu_swiglu(x[:active])
    torch.testing.assert_close(out[:active], expected, rtol=0.01, atol=0.002)


@pytest.mark.parametrize("topk", [1, 2, 6, 8])
@pytest.mark.parametrize("rows", [3, 257])
def test_prefetched_combine(topk, rows):
    from flaggems_vllm.runtime.backend._ascend.ops.fused_marlin_moe_w4a16_int4 import (
        _combine as combine,
    )

    x = torch.randn((rows * topk, 4096), device="npu", dtype=torch.bfloat16)
    p = torch.randn((rows, topk), device="npu")
    inv = torch.randperm(rows * topk, device="npu").int()
    out = torch.empty((rows, 4096), device="npu", dtype=x.dtype)
    for _ in range(3):
        combine(x, p, out, inv)
        expected = (
            (x[inv.long()].float().reshape(rows, topk, 4096) * p[:, :, None])
            .sum(1)
            .bfloat16()
        )
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.002)
        x.mul_(-0.75)


@pytest.mark.parametrize("m", [1, 8, 32])
def test_composite_graph_replay_with_changed_inputs(utils, ww, m):
    x, p, ids = utils.inputs(m, 4, 256, 2)
    for _ in range(3):
        utils.gems_call(x, ww, p, ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = utils.gems_call(x, ww, p, ids)
    for seed in [11, 19, 23]:
        xx, pp, ii = utils.inputs(m, 4, 256, 2, seed)
        x.copy_(xx)
        p.copy_(pp)
        ids.copy_(ii)
        graph.replay()
        expected = utils.baseline(x, ww, p, ids)
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.001)


def test_composite_max_activation_width(utils):
    weights = utils.weights(4, 128, 4096, torch.bfloat16)
    x, p, ids = utils.inputs(1, 4, 128, 2)
    got = utils.gems_call(x, weights, p, ids)
    expected = utils.baseline(x, weights, p, ids)
    torch.testing.assert_close(got, expected, rtol=0.01, atol=0.001)


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("k,n", [(384, 128), (256, 384)])
def test_non_power_of_two_geometry(utils, m, k, n):
    weights = utils.weights(4, k, n, torch.bfloat16)
    x, p, ids = utils.inputs(m, 4, k, 2)
    got = utils.gems_call(x, weights, p, ids)
    expected = utils.baseline(x, weights, p, ids)
    torch.testing.assert_close(got, expected, rtol=0.01, atol=0.001)


@pytest.mark.parametrize("m", [33, 128, 129])
def test_sparse_and_dense_packing(utils, m):
    # torch_npu 2.10/CANN 9 can reuse the 4-expert routing output shape when
    # only expert_num changes. Use the independent semantic reference here;
    # the performance baseline keeps E=256 in a separate process.
    weights = utils.weights(16, 256, 128, torch.bfloat16)
    x, p, ids = utils.inputs(m, 16, 256, 2)
    torch.testing.assert_close(
        utils.gems_call(x, weights, p, ids).cpu(),
        utils.reference(x, weights, p, ids),
        rtol=0.01,
        atol=0.001,
    )


def test_sparse_packing_graph_route_change(utils):
    weights = utils.weights(16, 256, 128, torch.bfloat16)
    x, p, ids = utils.inputs(33, 16, 256, 2)
    for _ in range(3):
        utils.gems_call(x, weights, p, ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = utils.gems_call(x, weights, p, ids)
    ids.fill_(0)
    x.mul_(-0.5)
    graph.replay()
    torch.testing.assert_close(
        out.cpu(), utils.reference(x, weights, p, ids), rtol=0.01, atol=0.001
    )


@pytest.mark.parametrize("e,k,n", [(4, 7168, 128), (4, 128, 14336), (512, 128, 128)])
@pytest.mark.parametrize("m", [1, 40])
def test_extended_geometry(utils, e, k, n, m):
    weights = utils.weights(e, k, n, torch.bfloat16)
    x, p, ids = utils.inputs(m, e, k, 2)
    # Include the highest expert in both small and grouped dispatch paths.
    ids[:, 0] = e - 1
    torch.testing.assert_close(
        utils.gems_call(x, weights, p, ids).cpu(),
        utils.reference(x, weights, p, ids),
        rtol=0.02,
        atol=0.02,
    )


def test_small_wide_k_workspace(utils):
    # First GEMM uses BN1=256, while the second uses BN2=128 for K=4224.
    weights = utils.weights(4, 4224, 128, torch.bfloat16)
    x, p, ids = utils.inputs(1, 4, 4224, 2)
    torch.testing.assert_close(
        utils.gems_call(x, weights, p, ids).cpu(),
        utils.reference(x, weights, p, ids),
        rtol=0.02,
        atol=0.02,
    )
