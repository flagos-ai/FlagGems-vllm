# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.fused_marlin_moe_w4a16_int4,
    pytest.mark.skipif(
        flaggems_vllm.device != "npu", reason="Ascend-specific validation"
    ),
]


def make_uint4_weights(expert_count, hidden_size, intermediate_size, dtype):
    import torch_npu

    torch.manual_seed(7)
    result = []
    for ni, ki in [
        (2 * intermediate_size, hidden_size),
        (hidden_size, intermediate_size),
    ]:
        w = torch.randint(
            0, 256, (expert_count, ni, ki // 2), device="npu", dtype=torch.uint8
        )
        s = torch.rand((expert_count, ni, ki // 128), device="npu", dtype=dtype) * 0.03
        native = []
        for ei in range(expert_count):
            q = w[ei].to(torch.int32)
            q = torch.stack((q & 15, q >> 4), dim=-1).reshape(ni, ki) - 8
            native.append(torch_npu.npu_convert_weight_to_int4pack(q.T.contiguous()))
        wp = torch.stack(native)
        sn = s.transpose(1, 2).contiguous()
        result.append((w, s, wp, sn, torch.zeros_like(sn)))
    return result


def ascend_grouped_matmul(hidden_states, weights, router_weights, expert_ids):
    import torch_npu

    e = weights[0][0].shape[0]
    a, idx, counts, _ = torch_npu.npu_moe_init_routing_v2(
        hidden_states,
        expert_ids.to(torch.int32),
        expert_num=e,
        active_num=hidden_states.shape[0] * expert_ids.shape[1],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        active_expert_range=[0, e],
        quant_mode=-1,
    )
    for j in range(2):
        _, _, w, s, z = weights[j]
        a = torch_npu.npu_grouped_matmul(
            x=[a],
            weight=[w],
            antiquant_scale=[s],
            antiquant_offset=[z],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=hidden_states.dtype,
        )[0]
        if j == 0:
            a = torch_npu.npu_swiglu(a)
    return torch_npu.npu_moe_token_unpermute(a, idx, probs=router_weights)


def swiglu_reference(hidden_states, weights, router_weights, expert_ids):
    hidden_states = hidden_states.cpu()
    router_weights = router_weights.cpu()
    expert_ids = expert_ids.cpu()
    y = torch.zeros_like(hidden_states, dtype=torch.float32)
    decoded = []
    for w, s, *_ in weights:
        w = w.cpu().to(torch.int32)
        s = s.cpu()
        q = (
            torch.stack((w & 15, w >> 4), dim=-1).reshape(
                *w.shape[:-1], w.shape[-1] * 2
            )
            - 8
        )
        decoded.append(
            (q.float() * s.float().repeat_interleave(128, dim=-1))
            .to(hidden_states.dtype)
            .float()
        )
    for token in range(hidden_states.shape[0]):
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route])
            gate_up = (
                (hidden_states[token].float() @ decoded[0][expert].T)
                .to(hidden_states.dtype)
                .float()
            )
            gate, up = gate_up.chunk(2)
            activated = (
                (torch.nn.functional.silu(gate) * up).to(hidden_states.dtype).float()
            )
            down = (activated @ decoded[1][expert].T).to(hidden_states.dtype).float()
            y[token] += down * router_weights[token, route]
    return y.to(hidden_states.dtype)


def make_inputs(num_tokens, expert_count, hidden_size, top_k, seed=7):
    torch.manual_seed(seed + num_tokens)
    hidden_states = (
        torch.randn((num_tokens, hidden_size), device="npu", dtype=torch.bfloat16) * 0.1
    )
    expert_ids = (
        torch.rand((num_tokens, expert_count), device="npu")
        .topk(top_k, -1)
        .indices.to(torch.int32)
    )
    router_weights = torch.softmax(torch.randn((num_tokens, top_k), device="npu"), -1)
    return hidden_states, router_weights, expert_ids


def call_fused_marlin_moe(hidden_states, weights, router_weights, expert_ids):
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    return flaggems_vllm.fused_marlin_moe(
        hidden_states,
        weights[0][0],
        weights[1][0],
        None,
        None,
        weights[0][1],
        weights[1][1],
        router_weights,
        expert_ids,
        QUANT_TYPE_UINT4B8,
    )


@pytest.fixture(scope="module")
def utils():
    return SimpleNamespace(
        weights=make_uint4_weights,
        inputs=make_inputs,
        baseline=ascend_grouped_matmul,
        reference=swiglu_reference,
        gems_call=call_fused_marlin_moe,
    )


@pytest.fixture(scope="module")
def ww(utils):
    return utils.weights(4, 256, 128, torch.bfloat16)


def test_public_registration():
    assert flaggems_vllm.fused_marlin_moe.__module__.startswith(
        "flaggems_vllm.runtime.backend._ascend"
    )


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
        flaggems_vllm.fused_marlin_moe(
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
        grouped_int4_gemm as gemm,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.fused_marlin_moe_w4a16_int4 import (
        prepare_weights as prepare,
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
        apply_swiglu as silu,
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
        combine_expert_outputs as combine,
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
