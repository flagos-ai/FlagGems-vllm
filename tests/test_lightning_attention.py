# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import json

import pytest
import torch

import flaggems_vllm
from tests.lightning_attention_cases import (
    DECODE_CASES,
    PREFIX_CASES,
    case_id,
    make_decode_inputs,
    make_initial,
    make_qkv,
    make_slope,
    make_state,
    native_prefix_reason,
)
from tests.lightning_attention_reference import decode_reference, prefix_reference

pytestmark = pytest.mark.skipif(
    not hasattr(flaggems_vllm, "lightning_attention_decode"),
    reason="requires Hopper with FlagTree TLE",
)


def assert_relative(actual, expected, limit):
    assert actual.shape == expected.shape
    if not actual.numel():
        return 0.0
    expected = expected.double()
    actual = actual.to(device=expected.device, dtype=torch.float64)
    error = (
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    ).item()
    assert error <= limit, f"relative L2 = {error:.7g}, limit={limit}"
    return error


def test_matrix_contains_distinct_cases():
    for cases in (PREFIX_CASES, DECODE_CASES):
        assert len(cases) == len({case_id(c) for c in cases}) == 80
        assert (
            len({tuple(v for k, v in vars(c).items() if k != "group") for c in cases})
            == 80
        )


@pytest.mark.parametrize("case", PREFIX_CASES, ids=case_id)
def test_lightning_prefix(case, record_property):
    torch.manual_seed(927)
    device = flaggems_vllm.device
    q, k, v = make_qkv(case, device)
    slope = make_slope(case, device)
    initial = make_initial(case, device)
    old = None if initial is None else initial.cpu()
    y, states = flaggems_vllm.lightning_attention(q, k, v, slope, kv_history=initial)
    ref, ref_s = prefix_reference(q.cpu(), k.cpu(), v.cpu(), slope.cpu(), old)
    assert y.dtype == case.dtype and states.dtype == torch.float32
    limit = 0.007 if case.dtype == torch.bfloat16 else 0.0025
    output_error = assert_relative(y, ref, limit)
    snapshot_error = max(
        assert_relative(x, z, 0.0025) for x, z in zip(states.unbind(2), ref_s.unbind(2))
    )
    # An isolated error at a hand-off can disappear in a whole-sequence norm.
    boundaries = sorted(
        {t for t in (0, 63, 64, 255, 256, 1471, 1472, case.t - 1) if 0 <= t < case.t}
    )
    if boundaries:
        assert_relative(y[:, :, boundaries], ref[:, :, boundaries], limit)
    if initial is not None:
        torch.testing.assert_close(initial.cpu(), old, rtol=0, atol=0)
    metrics = dict(output=output_error, max_snapshot=snapshot_error)
    # With native BF16 unsupported, compare the same quantized values promoted to FP32.
    reason = native_prefix_reason(case)
    if reason in (None, "native prefix needs FP32 V with FP32 decay"):
        native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")

        native_y, native_s = native.lightning_attention(
            q.float(), k.float(), v.float(), slope, kv_history=initial
        )
        metrics["native_output"] = assert_relative(
            y, native_y, 0.012 if case.dtype == torch.bfloat16 else 0.0025
        )
        metrics["native_history"] = assert_relative(states, native_s, 0.003)
    record_property("relative_l2", json.dumps(metrics))


@pytest.mark.parametrize("case", DECODE_CASES, ids=case_id)
def test_lightning_decode(case, record_property):
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")

    torch.manual_seed(928)
    device = flaggems_vllm.device
    q, k, v = make_qkv(case, device)
    slope, slots, initial = make_decode_inputs(case, device)
    pool = make_state(case, len(initial), device)
    # Check padding bytes of pitched/interleaved views independently of logical values.
    raw = pool.as_strided(
        (pool.untyped_storage().nbytes() // pool.element_size(),), (1,), 0
    )
    raw.fill_(-17.5)
    pool.copy_(initial)
    guard = (raw == -17.5) if raw.numel() > pool.numel() else None
    expected = initial.contiguous().clone()
    baseline = initial.contiguous().clone()
    native_qkv = tuple(x.contiguous() for x in (q, k, v))
    live = slots >= 0
    unused = torch.ones(len(pool), device=device, dtype=torch.bool)
    unused[slots[live].long()] = False
    metrics = dict(output=0.0, state=0.0, native_output=0.0, native_state=0.0)
    for _ in range(3):
        y = flaggems_vllm.lightning_attention_decode(
            q, k, v, pool, slope, slots, case.block
        )
        ref = decode_reference(q, k, v, expected, slope, slots, dtype=torch.float64)
        native_y = native.linear_decode_forward_triton(
            *native_qkv, baseline, slope.contiguous(), slots.contiguous(), case.block
        )
        assert y.shape == (case.b, case.h * case.d) and y.dtype == case.dtype
        # Native padding rows have unspecified output. TLE zeros those rows.
        assert torch.count_nonzero(y[~live]).item() == 0
        measured = {
            "output": assert_relative(
                y, ref, 0.002 if case.dtype == torch.bfloat16 else 2e-6
            ),
            "state": assert_relative(pool, expected, 2e-6),
            "native_output": assert_relative(
                y[live], native_y[live], 0.008 if case.dtype == torch.bfloat16 else 2e-6
            ),
            "native_state": assert_relative(
                pool, baseline, 0.004 if case.dtype == torch.bfloat16 else 2e-6
            ),
        }
        for key, value in measured.items():
            metrics[key] = max(metrics[key], value)
    torch.testing.assert_close(pool[unused], initial[unused], atol=0, rtol=0)
    if guard is not None:
        assert torch.all(raw[guard] == -17.5).item()
    record_property("relative_l2", json.dumps(metrics))


def test_vllm_read_at_history_boundary():
    torch.manual_seed(929)
    device = flaggems_vllm.device
    q = torch.zeros(1, 1, 1, 64, device=device)
    q[..., 0] = 1
    zero = torch.zeros_like(q)
    state = torch.zeros(1, 1, 64, 64, device=device)
    state[0, 0, 0, 0] = 4
    slope = -torch.tensor([0.5], device=device).log()
    y, history = flaggems_vllm.lightning_attention(
        q, zero, zero, slope, kv_history=state
    )
    single = flaggems_vllm.lightning_attention_decode(
        q, zero, zero, state, slope, torch.tensor([0], device=device)
    )
    assert (y.flatten()[0].item(), single[0, 0].item()) == (4, 2)
    torch.testing.assert_close(history[:, :, -1], state)


def test_vllm_minimax_prefix_and_decode(monkeypatch):
    layer = pytest.importorskip(
        "vllm.model_executor.layers.mamba.linear.minimax_linear_attn"
    )
    torch.manual_seed(930)
    device = flaggems_vllm.device
    t, h, d = 321, 3, 64
    q, k, v = [torch.randn(h, t, d, device=device) for _ in range(3)]
    s = torch.linspace(0.001, 0.2, h, device=device)
    initial = torch.randn(h, d, d, device=device)
    expected, actual = initial.clone(), initial.clone()
    prefix = layer.MiniMaxText01LinearKernel.jit_linear_forward_prefix
    y0 = prefix(q, k, v, expected, s, 256)
    monkeypatch.setattr(layer, "lightning_attention", flaggems_vllm.lightning_attention)
    y1 = prefix(q, k, v, actual, s, 256)
    assert y1.shape == (t, h * d)
    assert_relative(y1, y0, 0.0025)
    assert_relative(actual, expected, 0.0025)
    q, k, v = [x[:, 0].unsqueeze(0) for x in (q, k, v)]
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    z0 = layer.linear_attention_decode(q, k, v, expected.unsqueeze(0), s, slots)
    monkeypatch.setattr(
        layer, "linear_decode_forward_triton", flaggems_vllm.lightning_attention_decode
    )
    z1 = layer.linear_attention_decode(q, k, v, actual.unsqueeze(0), s, slots)
    assert_relative(z1, z0, 0.0025)
    assert_relative(actual, expected, 0.0025)


def test_block_boundary_and_chunked_continuation():
    device = flaggems_vllm.device
    q = torch.zeros(1, 1, 257, 64, device=device)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    q[..., 0] = 1
    k[0, 0, 255, 0] = 1
    v[0, 0, 255, 0] = 8
    s = -torch.tensor([0.5], device=device).log()
    y, states = flaggems_vllm.lightning_attention(q, k, v, s)
    assert (
        y[0, 0, 256, 0].item(),
        states[0, 0, 1, 0, 0].item(),
        states[0, 0, 2, 0, 0].item(),
    ) == (8, 8, 4)
    first, h1 = flaggems_vllm.lightning_attention(
        q[:, :, :256], k[:, :, :256], v[:, :, :256], s
    )
    last, h2 = flaggems_vllm.lightning_attention(
        q[:, :, 256:], k[:, :, 256:], v[:, :, 256:], s, kv_history=h1[:, :, -1]
    )
    torch.testing.assert_close(torch.cat([first, last], dim=2), y)
    torch.testing.assert_close(h2[:, :, -1], states[:, :, -1])


def test_graph_reads_current_slots_and_state():
    torch.manual_seed(931)
    b, h, d = 3, 2, 64
    device = flaggems_vllm.device
    q, k, v = [torch.randn(b, h, 1, d, device=device) for _ in range(3)]
    cache = torch.randn(5, h, d, d, device=device)
    s = torch.linspace(0.02, 0.12, h, device=device)
    slots = torch.tensor([0, 1, 2], device=device)
    flaggems_vllm.lightning_attention_decode(q, k, v, cache, s, slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = flaggems_vllm.lightning_attention_decode(q, k, v, cache, s, slots)
    for mapping in ([3, -1, 0], [2, 1, 4]):
        slots.copy_(torch.tensor(mapping, device=device))
        expected = cache.clone()
        ref = decode_reference(q, k, v, expected, s, slots)
        graph.replay()
        assert_relative(y, ref, 2e-6)
        assert_relative(cache, expected, 2e-6)


def test_decode_rejects_mismatched_input_before_writing():
    q = torch.zeros(1, 2, 1, 64, device=flaggems_vllm.device)
    s = torch.zeros(2, device=q.device)
    cache = torch.ones(3, 2, 64, 64, device=q.device)
    original = cache.clone()
    with pytest.raises(ValueError, match="one token"):
        flaggems_vllm.lightning_attention_decode(
            q,
            q,
            q[..., :32],
            cache,
            s,
            torch.zeros(1, dtype=torch.int64, device=q.device),
        )
    torch.testing.assert_close(cache, original, atol=0, rtol=0)


@pytest.mark.parametrize(
    "t,d,has_initial",
    [
        (7, 128, True),
        (31, 64, False),
        (62, 128, True),
        (64, 64, True),
        (67, 128, True),
        (129, 128, True),
        (257, 64, False),
    ],
)
def test_prefix_graph_reads_current_operands(t, d, has_initial):
    device = flaggems_vllm.device
    shape = (1, 3, t, d)
    torch.manual_seed(933)
    q, k, v = [torch.randn(shape, device=device) for _ in range(3)]
    initial = torch.randn(1, 3, d, d, device=device) if has_initial else None
    slope = torch.rand(3, device=device)

    def run():
        return flaggems_vllm.lightning_attention(q, k, v, slope, kv_history=initial)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, history = run()
    for _ in range(6):
        for x in (q, k, v):
            x.normal_()
        slope.uniform_(0.0, 0.25)
        if initial is not None:
            initial.normal_()
        graph.replay()
        expected, states = prefix_reference(
            q.cpu(),
            k.cpu(),
            v.cpu(),
            slope.cpu(),
            initial.cpu() if has_initial else None,
        )
        assert_relative(y, expected, 0.0025)
        assert_relative(history, states, 0.0025)
