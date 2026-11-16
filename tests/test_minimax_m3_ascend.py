# SPDX-License-Identifier: Apache-2.0
"""Independent numerical references for the Ascend M3 operator boundary.

All floating tolerances below are provisional BF16 screening tolerances, NOT
approved acceptance thresholds. They have not been calibrated on target NPU.
Before formal acceptance, compare against the existing independent native gate
and record agreed tolerances; a pass here alone does not establish F2 acceptance.
Exact cache bytes, identity tails, mutation and integer contracts use exact checks.
"""

import math

import pytest
import torch

import flaggems_vllm as ops

pytestmark = [
    pytest.mark.minimax_m3_ascend,
    pytest.mark.skipif(
        ops.runtime.device.vendor_name != "ascend", reason="Ascend-only operators"
    ),
]


def norm_rope_reference(x, weight, cs, pos, r, eps):
    gamma = (1 + weight.float()).to(x.dtype)
    xf = x.float()
    normalized = (
        xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * gamma.float()
    ).to(x.dtype)
    y = normalized.float()
    cos, sin = (
        cs[pos, : r // 2].float().unsqueeze(1),
        cs[pos, r // 2 :].float().unsqueeze(1),
    )
    a, b = y[..., : r // 2], y[..., r // 2 : r]
    return torch.cat((a * cos - b * sin, b * cos + a * sin, y[..., r:]), -1).to(x.dtype)


@pytest.mark.parametrize("n", [0, 1, 7])
@pytest.mark.parametrize("r", [32, 64, 128])
def test_fused_qkv_outputs_and_mutation(n, r):
    device = ops.device
    nh, nk, ni = 4, 2, 2
    heads = nh + 2 * nk + ni + 1
    original = torch.randn((n, heads, 128), device=device, dtype=torch.bfloat16)
    x = original.flatten(1).clone()
    weights = [torch.randn(128, device=device, dtype=x.dtype) * 0.1 for _ in range(4)]
    cs = torch.randn((16, r), device=device, dtype=x.dtype)
    pos = torch.arange(n, device=device, dtype=torch.int64)
    qout = torch.empty((n, nh * 128), device=device, dtype=x.dtype)
    iqout = torch.empty((n, ni * 128), device=device, dtype=x.dtype)
    ops.fused_minimax_m3_qknorm_rope_kv_insert_ascend(
        x,
        weights[0],
        weights[1],
        cs,
        pos,
        nh,
        nk,
        r,
        1e-6,
        weights[2],
        weights[3],
        ni,
        q_out=qout,
        index_q_out=iqout,
    )
    result = x.view_as(original)
    # Dedicated outputs suppress mutation of their original packed Q lanes.
    assert torch.equal(result[:, :nh], original[:, :nh])
    assert torch.equal(
        result[:, nh + nk : nh + 2 * nk], original[:, nh + nk : nh + 2 * nk]
    )
    groups = [
        (qout.view(n, nh, 128), original[:, :nh], weights[0]),
        (result[:, nh : nh + nk], original[:, nh : nh + nk], weights[1]),
        (iqout.view(n, ni, 128), original[:, nh + 2 * nk : -1], weights[2]),
        (result[:, -1:], original[:, -1:], weights[3]),
    ]
    for actual, source, weight in groups:
        expected = norm_rope_reference(source, weight, cs, pos, r, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("plane_major", [True, False])
def test_cache_negative_slots_and_ambiguous_two_blocks(plane_major):
    device = ops.device
    shape = (2, 2, 128, 2, 128)  # both leading axes equal 2: no shape inference.
    cache = torch.full(shape, 9, device=device, dtype=torch.bfloat16)
    key = torch.randn((3, 2, 128), device=device, dtype=cache.dtype)
    value = torch.randn_like(key)
    slots = torch.tensor([0, -1, 129], device=device, dtype=torch.int64)
    expected = cache.clone()
    for token, slot in ((0, 0), (2, 129)):
        block, offset = divmod(slot, 128)
        if plane_major:
            expected[0, block, offset], expected[1, block, offset] = (
                key[token],
                value[token],
            )
        else:
            expected[block, 0, offset], expected[block, 1, offset] = (
                key[token],
                value[token],
            )
    ops.insert_minimax_m3_kv_cache_ascend(
        cache, slots, key, value, 128, plane_major=plane_major
    )
    assert torch.equal(cache, expected)
    index = torch.full((2, 128, 128), 9, device=device, dtype=cache.dtype)
    ikey = key[:, :1]
    ops.insert_minimax_m3_index_cache_ascend(index, slots, ikey, 128)
    assert torch.equal(index[0, 0], ikey[0, 0])
    assert torch.equal(index[1, 1], ikey[2, 0])
    assert torch.all(index[1, -1] == 9)
    ops.insert_minimax_m3_index_cache_ascend(index, slots[:0], ikey[:0], 128)


@pytest.mark.parametrize("neox", [True, False])
def test_partial_rotary_vision_and_tail(neox):
    x = torch.randn((2, 3, 4, 80), device=ops.device, dtype=torch.bfloat16)
    cs = torch.randn((3, 39), device=ops.device, dtype=x.dtype)
    sn = torch.randn_like(cs)
    y = ops.apply_minimax_m3_rotary_ascend(x, cs, sn, neox)
    cos, sin = cs.unsqueeze(0).unsqueeze(-2), sn.unsqueeze(0).unsqueeze(-2)
    if neox:
        a, b = x[..., :39].float(), x[..., 39:78].float()
        rotated = torch.cat(
            (a * cos.float() - b * sin.float(), b * cos.float() + a * sin.float()), -1
        )
    else:
        a, b = x[..., :78:2], x[..., 1:78:2]
        rotated = torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2)
    expected = torch.cat((rotated.to(x.dtype), x[..., 78:]), -1)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


def test_index_score_causal_boundary_and_topk():
    device = ops.device
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    cache = torch.randn((2, 128, 128), device=device, dtype=q.dtype)
    table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 3], device=device, dtype=torch.int32)
    seq, prefix = (
        torch.tensor([130], device=device, dtype=torch.int32),
        torch.tensor([127], device=device, dtype=torch.int32),
    )
    score = ops.minimax_m3_index_score_ascend(
        q, cache, table, cu, seq, prefix, 3, 130, 2
    )
    for token in range(3):
        for block in range(math.ceil((128 + token) / 128)):
            end = min(128, 128 + token - block * 128)
            expected = (q[token].float() @ cache[block, :end].float().T).max(-1).values
            torch.testing.assert_close(
                score[:, token, block], expected, rtol=0.01, atol=0.05
            )
    out = ops.minimax_m3_index_topk_ascend(score, cu, prefix, 3, 3, 1, 1)
    assert out.dtype == torch.int32
    assert set(out[0, 0].cpu().tolist()) == {0, -1}
    assert set(out[0, 2].cpu().tolist()) == {0, 1, -1}


def test_native_sparse_attention_against_full_visible_reference():
    device = ops.device
    q = torch.randn((1, 4, 128), device=device, dtype=torch.bfloat16)
    cache = torch.randn((2, 2, 128, 2, 128), device=device, dtype=q.dtype)
    topk = torch.tensor([[[0, 1]], [[0, 1]]], device=device, dtype=torch.int32)
    table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    seq = torch.tensor([129], device=device, dtype=torch.int64)
    qlen = torch.tensor([1], device=device, dtype=torch.int64)
    out = torch.empty_like(q)
    ops.attend_minimax_m3_sparse_ascend(
        q, cache, topk, table, seq, qlen, 2, 128**-0.5, 128, out, plane_major=True
    )
    key, value = cache[0].flatten(0, 1)[:129], cache[1].flatten(0, 1)[:129]
    expected = torch.empty_like(q)
    for head in range(4):
        scores = q[0, head].float() @ key[:, head // 2].float().T * 128**-0.5
        expected[0, head] = scores.softmax(-1) @ value[:, head // 2].float()
    torch.testing.assert_close(out, expected, rtol=0.03, atol=0.03)


def test_cache_graph_replay_changes_inputs_and_preserves_sentinel():
    cache = torch.full((2, 3, 128, 1, 128), 13, device=ops.device, dtype=torch.bfloat16)
    slots = torch.tensor([0, -1], device=ops.device, dtype=torch.int64)
    key = torch.ones((2, 1, 128), device=ops.device, dtype=cache.dtype)
    value = key.clone()
    run = lambda: ops.insert_minimax_m3_kv_cache_ascend(
        cache, slots, key, value, 128, plane_major=True
    )
    run()  # compile outside capture
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    key.fill_(5)
    value.fill_(7)
    slots.copy_(torch.tensor([129, -1], device=ops.device, dtype=slots.dtype))
    graph.replay()
    assert torch.all(cache[0, 1, 1] == 5)
    assert torch.all(cache[1, 1, 1] == 7)
    assert torch.all(cache[:, 2] == 13)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemma_weight_initialization_is_detached(dtype):
    w = torch.randn(128, device=ops.device, dtype=dtype, requires_grad=True)
    result = ops.prepare_minimax_m3_gemma_weight_ascend(w)
    assert not result.requires_grad
    assert result.data_ptr() != w.data_ptr()
    assert torch.equal(result, (1 + w).detach())


def test_decode_two_requests_matches_block_score_reference():
    q = torch.randn((2, 2, 128), device=ops.device, dtype=torch.bfloat16)
    cache = torch.randn((6, 128, 128), device=ops.device, dtype=q.dtype)
    table = torch.tensor([[0, 1, 2], [3, 4, 5]], device=ops.device, dtype=torch.int32)
    lengths = [130, 257]
    seq = torch.tensor(lengths, device=ops.device, dtype=torch.int32)
    result = ops.minimax_m3_index_decode_ascend(
        q, cache, table, seq, 384, 2, 0, 0, 2, 1
    )
    for batch, length in enumerate(lengths):
        values = []
        for block in range(math.ceil(length / 128)):
            visible = min(128, length - block * 128)
            values.append(
                (q[batch].float() @ cache[batch * 3 + block, :visible].float().T)
                .max(-1)
                .values
            )
        expected = torch.stack(values, -1).topk(2, -1).indices
        for head in range(2):
            assert set(result[head, batch].cpu().tolist()) == set(
                expected[head].cpu().tolist()
            )


def test_fused_graph_replay_uses_changed_positions_and_query():
    original = torch.randn((1, 1024), device=ops.device, dtype=torch.bfloat16)
    qkv = original.clone()
    w = torch.randn(128, device=ops.device, dtype=qkv.dtype) * 0.1
    cs = torch.randn((4, 64), device=ops.device, dtype=qkv.dtype)
    positions = torch.tensor([0], device=ops.device, dtype=torch.int64)
    qout = torch.empty((1, 512), device=ops.device, dtype=qkv.dtype)

    def run():
        ops.fused_minimax_m3_qknorm_rope_kv_insert_ascend(
            qkv, w, w, cs, positions, 4, 2, 64, 1e-6, q_out=qout
        )

    run()
    qkv.copy_(original)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    changed = torch.randn_like(qkv)
    qkv.copy_(changed)
    positions.copy_(torch.tensor([2], device=ops.device, dtype=positions.dtype))
    graph.replay()
    expected = norm_rope_reference(
        changed.view(1, 8, 128)[:, :4], w, cs, positions, 64, 1e-6
    )
    torch.testing.assert_close(qout.view(1, 4, 128), expected, rtol=0.02, atol=0.02)


def test_sparse_chain_graph_updates_metadata_cache_and_output():
    """Actual capture/replay of QKV->cache->index selection->native attend."""
    device = ops.device
    raw = torch.randn((1, 11 * 128), device=device, dtype=torch.bfloat16)
    qkv = raw.clone()
    weights = [torch.randn(128, device=device, dtype=raw.dtype) * 0.1 for _ in range(4)]
    cs = torch.randn((256, 64), device=device, dtype=raw.dtype)
    pos = torch.tensor([128], device=device, dtype=torch.int64)
    slots = pos.clone()
    main = torch.randn((2, 3, 128, 2, 128), device=device, dtype=raw.dtype)
    index = torch.randn((3, 128, 128), device=device, dtype=raw.dtype)
    # Unreachable third physical block is a persistent sentinel.
    main[:, 2].fill_(13)
    index[2].fill_(17)
    qout = torch.empty((1, 512), device=device, dtype=raw.dtype)
    iqout = torch.empty((1, 256), device=device, dtype=raw.dtype)
    output = torch.empty((1, 4, 128), device=device, dtype=raw.dtype)
    table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    seq = torch.tensor([129], device=device, dtype=torch.int64)
    qlen = torch.tensor([1], device=device, dtype=torch.int64)
    ops.load_minimax_m3_ascend_native()  # load before any graph/cache writes

    def run():
        ops.fused_minimax_m3_qknorm_rope_kv_insert_ascend(
            qkv,
            weights[0],
            weights[1],
            cs,
            pos,
            4,
            2,
            64,
            1e-6,
            weights[2],
            weights[3],
            2,
            slots,
            slots,
            main,
            index,
            128,
            qout,
            iqout,
            plane_major=True,
        )
        selected = ops.minimax_m3_index_decode_ascend(
            iqout.view(1, 2, 128), index, table, seq, 256, 2, 0, 0, 2, 1
        )
        ops.attend_minimax_m3_sparse_ascend(
            qout.view(1, 4, 128),
            main,
            selected,
            table,
            seq,
            qlen,
            2,
            128**-0.5,
            128,
            output,
            plane_major=True,
        )

    run()  # compilation/autotuning occurs outside capture
    qkv.copy_(raw)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    changed = torch.randn_like(qkv)
    qkv.copy_(changed)
    pos.copy_(torch.tensor([127], device=device, dtype=pos.dtype))
    slots.copy_(pos)
    seq.copy_(torch.tensor([128], device=device, dtype=seq.dtype))
    graph.replay()
    expected_q = norm_rope_reference(
        changed.view(1, 11, 128)[:, :4], weights[0], cs, pos, 64, 1e-6
    )
    torch.testing.assert_close(
        qout.view_as(expected_q), expected_q, rtol=0.02, atol=0.02
    )
    key, value = main[0, 0], main[1, 0]
    expected = torch.empty_like(output)
    for head in range(4):
        logits = expected_q[0, head].float() @ key[:, head // 2].float().T * 128**-0.5
        expected[0, head] = logits.softmax(-1) @ value[:, head // 2].float()
    torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
    assert torch.all(main[:, 2] == 13)
    assert torch.all(index[2] == 17)
