# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm
from tests.lightning_attention_reference import decode_reference, prefix_reference

pytestmark = pytest.mark.skipif(
    not hasattr(flaggems_vllm, "lightning_attention_decode"),
    reason="requires Hopper with FlagTree TLE",
)


def assert_relative(actual, expected, limit):
    expected = expected.double().cpu()
    actual = actual.double().cpu()
    if actual.numel() == 0:
        assert actual.shape == expected.shape
        return
    error = torch.linalg.vector_norm(actual - expected)
    scale = torch.linalg.vector_norm(expected).clamp_min(1.0e-12)
    assert error / scale <= limit, f"relative L2 = {(error / scale).item():.6g}"


@pytest.mark.parametrize(
    "b,h,t,d,e,has_initial,strided,dtype",
    [
        (1, 2, 0, 64, 64, True, False, torch.float32),
        (2, 3, 1, 64, 64, True, False, torch.float32),
        (2, 3, 4, 128, 128, False, True, torch.float32),
        (1, 2, 63, 64, 64, False, True, torch.float32),
        (1, 2, 64, 128, 64, True, False, torch.float32),
        (1, 2, 65, 64, 64, True, False, torch.bfloat16),
        (2, 3, 255, 64, 64, True, True, torch.float32),
        (1, 2, 256, 128, 128, False, False, torch.float32),
        (1, 2, 257, 64, 64, True, True, torch.float32),
        (2, 2, 320, 64, 32, True, False, torch.float32),
        (1, 2, 513, 128, 128, True, False, torch.bfloat16),
        (1, 4, 2049, 128, 128, True, False, torch.float32),
    ],
)
def test_lightning_prefix(b, h, t, d, e, has_initial, strided, dtype):
    torch.manual_seed(927)
    device = flaggems_vllm.device

    def make(width):
        x = torch.randn(
            b, t, h, width * 2 if strided else width, dtype=dtype, device=device
        )
        return (
            x[..., ::2].transpose(1, 2) if strided else x.transpose(1, 2).contiguous()
        )

    q, k, v = make(d), make(d), make(e)
    slope = torch.linspace(0, 0.25, h, device=device)
    s0 = torch.randn(b, h, d, e * 2, device=device)[..., ::2] if has_initial else None
    original = s0.clone() if has_initial else None
    y, states = flaggems_vllm.lightning_attention(q, k, v, slope, kv_history=s0)
    expected = prefix_reference(
        q.cpu(), k.cpu(), v.cpu(), slope.cpu(), s0.cpu() if has_initial else None
    )
    assert y.shape == expected[0].shape
    assert states.shape == expected[1].shape
    assert y.dtype == dtype and states.dtype == torch.float32
    assert_relative(y, expected[0], 0.007 if dtype == torch.bfloat16 else 0.0025)
    for block, expected_state in zip(states.unbind(2), expected[1].unbind(2)):
        assert_relative(block, expected_state, 0.0025)
    if has_initial:
        torch.testing.assert_close(s0, original, rtol=0, atol=0)
    if t:
        native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
        # Bailing passes FP32. The native 0.24 kernel rejects BF16 V with FP32 decay.
        native_qkv = (
            (q.float(), k.float(), v.float()) if dtype == torch.bfloat16 else (q, k, v)
        )
        baseline_y, baseline_s = native.lightning_attention(
            *native_qkv, slope, kv_history=s0
        )
        assert_relative(y, baseline_y, 0.012 if dtype == torch.bfloat16 else 0.0025)
        assert_relative(states, baseline_s, 0.003)


@pytest.mark.parametrize(
    "b,d,dtype",
    [(1, 64, torch.float32), (5, 128, torch.float32), (5, 64, torch.bfloat16)],
)
def test_lightning_decode(b, d, dtype):
    torch.manual_seed(928)
    device = flaggems_vllm.device
    h = 4
    q, k, v = [torch.randn(b, h, 1, d, dtype=dtype, device=device) for _ in range(3)]
    s = torch.linspace(0.005, 0.25, h, device=device)
    full = torch.randn(b + 3, h, d, d * 2, device=device)
    original = full.clone()
    pool = full[..., ::2]
    slots = torch.tensor([2, -1, 0, 4, 1][:b], dtype=torch.int32, device=device)
    expected_pool = pool.clone()
    native_pool = pool.clone()
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
    for _ in range(3):
        y = flaggems_vllm.lightning_attention_decode(q, k, v, pool, s, slots)
        expected = decode_reference(q, k, v, expected_pool, s, slots)
        baseline = native.linear_decode_forward_triton(q, k, v, native_pool, s, slots)
        assert y.shape == (b, h * d)
        assert_relative(y, expected, 0.002 if dtype == torch.bfloat16 else 0.000002)
        assert_relative(
            y[slots >= 0],
            baseline[slots >= 0],
            0.008 if dtype == torch.bfloat16 else 0.000002,
        )
        assert_relative(pool, expected_pool, 2e-6)
        assert_relative(pool, native_pool, 0.004 if dtype == torch.bfloat16 else 2e-6)
    torch.testing.assert_close(full[..., 1::2], original[..., 1::2], rtol=0, atol=0)
    unused = [n for n in range(pool.shape[0]) if n not in slots.tolist()]
    torch.testing.assert_close(pool[unused], original[unused, ..., ::2], atol=0, rtol=0)


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
