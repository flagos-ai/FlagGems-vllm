# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional tests for Hopper TLE MiniMax Lightning Attention kernels."""

import gc
import json
from dataclasses import dataclass

import pytest
import torch

import flaggems_vllm

pytestmark = pytest.mark.skipif(
    not hasattr(flaggems_vllm, "lightning_attention_decode"),
    reason="requires Hopper with FlagTree TLE",
)


@dataclass(frozen=True)
class PrefixCase:
    group: str
    b: int
    h: int
    t: int
    d: int
    e: int
    dtype: torch.dtype = torch.float32
    state_dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class DecodeCase:
    group: str
    b: int
    h: int
    d: int
    dtype: torch.dtype = torch.float32
    block: int = 32
    state_dtype: torch.dtype = torch.float32
    cache_rows: int = 10749


# Measured MiniMax-Text-01 TP8 shapes on vLLM: frequent sequence lengths, chunk
# boundaries, and decode batch sizes spanning Hopper SM-wave dispatch thresholds.
MINIMAX_PREFIX_LENGTHS = (
    2,
    16,
    31,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    512,
    1023,
    1024,
    1025,
    1536,
    2048,
    4095,
    4096,
    4097,
    8191,
    8192,
    12185,
    16383,
    16384,
)
MINIMAX_DECODE_BATCHES = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    14,
    15,
    16,
    17,
    24,
    25,
    31,
    32,
    33,
    48,
    63,
    64,
    65,
    96,
    127,
    128,
)
# TP rank 0's measured MiniMax-Text-01 slopes; all ranks share the [H, 1, 1] layout.
_MINIMAX_TP8_SLOPES = (
    0.91701323,
    0.84090483,
    0.77111316,
    0.70711386,
    0.64842629,
    0.59460950,
    0.54525936,
    0.50000501,
)

PREFIX_CASES = [
    PrefixCase("minimax-tp8", 1, 8, t, 128, 128, state_dtype=torch.bfloat16)
    for t in MINIMAX_PREFIX_LENGTHS
]
DECODE_CASES = [
    DecodeCase("minimax-tp8", b, 8, 128, state_dtype=torch.bfloat16)
    for b in MINIMAX_DECODE_BATCHES
]


def case_id(case):
    tag = "p" if isinstance(case, PrefixCase) else "d"
    t = getattr(case, "t", 1)
    e = getattr(case, "e", case.d)
    return f"{tag}-{case.group}-b{case.b}h{case.h}t{t}d{case.d}e{e}"


def make_qkv(case, device):
    t = getattr(case, "t", 1)
    e = getattr(case, "e", case.d)
    return tuple(
        torch.randn(case.b, case.h, t, width, device=device, dtype=case.dtype).mul_(
            0.125
        )
        for width in (case.d, case.d, e)
    )


def make_slope(case, device):
    if case.group == "minimax-tp8":
        return torch.tensor(
            _MINIMAX_TP8_SLOPES, dtype=torch.float32, device=device
        ).view(case.h, 1, 1)
    return torch.linspace(0.002, 0.12, case.h, device=device)


def make_state(case, rows, device):
    e = getattr(case, "e", case.d)
    return torch.empty(rows, case.h, case.d, e, device=device, dtype=case.state_dtype)


def make_initial(case, device):
    return make_state(case, case.b, device).normal_(std=0.125)


def make_decode_inputs(case, device):
    # Exercise the far end of the model-sized cache allocation (above 2 GiB).
    slots = torch.arange(
        case.cache_rows - case.b, case.cache_rows, dtype=torch.int32, device=device
    )
    slope = make_slope(case, device)
    initial = make_state(case, case.cache_rows, device).normal_(std=0.125)
    return slope, slots, initial


def prefix_reference(q, k, v, slope, initial=None, dtype=torch.float64):
    """Dense causal reference inside each native 256-token history block."""
    q, k, v = (x.to(dtype) for x in (q, k, v))
    b, h, t, d = q.shape
    e = v.shape[-1]
    slope = slope.reshape(1, h, 1, 1).to(dtype)
    state = q.new_zeros((b, h, d, e)) if initial is None else initial.to(dtype)
    output, snapshots = [], []
    for start in range(0, t, 256):
        stop = min(start + 256, t)
        row = torch.arange(stop - start, device=q.device)
        age = row[:, None] - row[None, :]
        weight = torch.exp(-slope * age.clamp_min(0)) * (age >= 0)
        qs, ks, vs = (x[:, :, start:stop] for x in (q, k, v))
        inner = ((qs @ ks.transpose(-1, -2)) * weight) @ vs
        # vLLM's prefix read starts at power zero for a new 256-token block.
        output.append(inner + (qs @ state) * torch.exp(-slope * row[:, None]))
        snapshots.append(state)
        tail = torch.exp(-slope * (len(row) - 1 - row[:, None])) * vs
        state = torch.exp(-slope * len(row)) * state + ks.transpose(-1, -2) @ tail
    snapshots.append(state)
    y = torch.cat(output, dim=2) if output else q.new_empty((b, h, 0, e))
    return y, torch.stack(snapshots, dim=2)


def decode_reference(q, k, v, cache, slope, slots, dtype=torch.float32):
    """Compute live rows in requested precision and round persistent cache each step."""
    active = slots >= 0
    idx = slots[active].long()
    old = cache[idx].to(dtype)
    state = torch.exp(-slope.to(dtype).reshape(1, -1, 1, 1)) * old
    state = state + k[active, :, 0, :, None].to(dtype) * v[active, :, 0, None, :].to(
        dtype
    )
    result = torch.zeros(
        (q.shape[0], q.shape[1] * q.shape[-1]), device=q.device, dtype=q.dtype
    )
    result[active] = (
        (q[active].to(dtype) @ state).reshape(-1, result.shape[-1]).to(q.dtype)
    )
    cache[idx] = state.to(cache.dtype)
    return result


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
        assert {c.group for c in cases} == {"minimax-tp8"}
        assert len(cases) == len({case_id(c) for c in cases}) == 25
        assert len(
            {tuple(v for k, v in vars(c).items() if k != "group") for c in cases}
        ) == len(cases)


@pytest.mark.parametrize("case", PREFIX_CASES, ids=case_id)
def test_lightning_prefix(case, record_property):
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
    torch.manual_seed(927)
    device = flaggems_vllm.device
    q, k, v = make_qkv(case, device)
    slope = make_slope(case, device)
    initial = make_initial(case, device)
    old = initial.cpu()
    y, states = flaggems_vllm.lightning_attention(q, k, v, slope, kv_history=initial)
    ref, ref_s = prefix_reference(q.cpu(), k.cpu(), v.cpu(), slope.cpu(), old)
    assert y.dtype == case.dtype and states.dtype == torch.float32
    limit = 0.007 if case.dtype == torch.bfloat16 else 0.0025
    output_error = assert_relative(y, ref, limit)
    snapshot_error = max(
        assert_relative(x, z, 0.0025) for x, z in zip(states.unbind(2), ref_s.unbind(2))
    )
    # Isolated boundary hand-off errors could otherwise be diluted in a full norm.
    boundaries = sorted(
        {t for t in (0, 63, 64, 255, 256, 1471, 1472, case.t - 1) if 0 <= t < case.t}
    )
    assert_relative(y[:, :, boundaries], ref[:, :, boundaries], limit)
    torch.testing.assert_close(initial.cpu(), old, rtol=0, atol=0)

    native_y, native_s = native.lightning_attention(
        q.float(), k.float(), v.float(), slope, kv_history=initial
    )
    metrics = dict(
        output=output_error,
        max_snapshot=snapshot_error,
        native_output=assert_relative(
            y, native_y, 0.012 if case.dtype == torch.bfloat16 else 0.0025
        ),
        native_history=assert_relative(states, native_s, 0.003),
    )
    record_property("relative_l2", json.dumps(metrics))


@pytest.mark.parametrize("case", DECODE_CASES, ids=case_id)
def test_lightning_decode(case, record_property):
    native = pytest.importorskip("vllm.model_executor.layers.lightning_attn")
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(928)
    device = flaggems_vllm.device
    q, k, v = make_qkv(case, device)
    slope, slots, pool = make_decode_inputs(case, device)
    live = slots >= 0
    active_ids = slots[live].long()
    unused = torch.ones(len(pool), dtype=torch.bool)
    unused[active_ids.cpu()] = False

    # Keep full 2.6 GiB initial snapshot on CPU and reuse a single GPU cache pool
    # across TLE and native runs to avoid >10 GiB peak VRAM allocations.
    initial_cpu = pool.cpu()
    expected_active = pool[active_ids].clone()
    compact_slots = torch.full_like(slots, -1)
    compact_slots[live] = torch.arange(
        active_ids.numel(), device=device, dtype=slots.dtype
    )

    tle_outs = []
    for _ in range(3):
        tle_outs.append(
            flaggems_vllm.lightning_attention_decode(
                q, k, v, pool, slope, slots, case.block
            )
        )
    tle_active = pool[active_ids].clone()
    assert torch.equal(pool.cpu()[unused], initial_cpu[unused])

    pool.copy_(initial_cpu)
    # Each decode writes persistent state at cache precision; BF16 rounding can
    # propagate slightly across multi-step updates.
    state_limit = 0.0002 if pool.dtype == torch.bfloat16 else 2e-6
    out_limit = 0.002 if case.dtype == torch.bfloat16 else state_limit
    metrics = dict(output=0.0, state=0.0, native_output=0.0, native_state=0.0)
    for step_idx, y in enumerate(tle_outs):
        ref = decode_reference(
            q, k, v, expected_active, slope, compact_slots, dtype=torch.float64
        )
        native_y = native.linear_decode_forward_triton(
            q, k, v, pool, slope.contiguous(), slots.contiguous(), case.block
        )
        assert y.shape == (case.b, case.h * case.d) and y.dtype == case.dtype
        assert torch.count_nonzero(y[~live]).item() == 0
        metrics["output"] = max(metrics["output"], assert_relative(y, ref, out_limit))
        metrics["native_output"] = max(
            metrics["native_output"],
            assert_relative(
                y[live],
                native_y[live],
                0.008 if case.dtype == torch.bfloat16 else state_limit,
            ),
        )
        if step_idx == 2:
            metrics["state"] = assert_relative(tle_active, expected_active, state_limit)
            metrics["native_state"] = assert_relative(
                tle_active,
                pool[active_ids],
                0.004 if case.dtype == torch.bfloat16 else state_limit,
            )
    record_property("relative_l2", json.dumps(metrics))


@pytest.mark.parametrize(
    "t,d,e,dtype", [(0, 64, 64, torch.float32), (67, 128, 17, torch.bfloat16)]
)
def test_prefix_empty_and_strided_layouts(t, d, e, dtype):
    torch.manual_seed(926)
    device = flaggems_vllm.device
    b, h = 2, 3
    q = torch.randn(b, t, h, d * 2, device=device, dtype=dtype)[..., ::2].transpose(
        1, 2
    )
    k = torch.randn(b, t, h, d, device=device, dtype=dtype).transpose(1, 2)
    v = torch.randn(b, h, t, e + 3, device=device, dtype=dtype)[..., 1 : 1 + e]
    s0 = torch.randn(b, h, e, d, device=device).transpose(-1, -2)
    slope = torch.linspace(0.01, 0.2, h, device=device)
    y, states = flaggems_vllm.lightning_attention(q, k, v, slope, kv_history=s0)
    ref_y, ref_s = prefix_reference(q.cpu(), k.cpu(), v.cpu(), slope.cpu(), s0.cpu())
    assert_relative(y, ref_y, 0.007 if dtype == torch.bfloat16 else 0.0025)
    assert_relative(states, ref_s, 0.0025)


@pytest.mark.parametrize("state_mode", ["transposed", "pitched"])
def test_decode_strided_cache_and_padding_slots(state_mode):
    torch.manual_seed(925)
    device = flaggems_vllm.device
    b, h, d = 5, 4, 128
    q, k, v = [
        torch.randn(b, 1, h, d, device=device, dtype=torch.bfloat16).transpose(1, 2)
        for _ in range(3)
    ]
    slope = torch.linspace(0.01, 0.2, h, device=device)
    slots = torch.tensor([3, -1, 0, -1, 5], dtype=torch.int32, device=device)
    if state_mode == "transposed":
        cache = torch.randn(7, h, d, d, device=device).transpose(-1, -2)
    else:
        full = torch.randn(7, h, d, d + 4, device=device)
        cache = full[..., 1 : 1 + d]
    expected = cache.contiguous().clone()
    y = flaggems_vllm.lightning_attention_decode(q, k, v, cache, slope, slots, 16)
    ref = decode_reference(q, k, v, expected, slope, slots, dtype=torch.float64)
    assert torch.count_nonzero(y[slots < 0]).item() == 0
    assert_relative(y, ref, 0.002)
    assert_relative(
        cache[slots[slots >= 0].long()], expected[slots[slots >= 0].long()], 2e-6
    )


def test_unsupported_head_dim_raises():
    device = flaggems_vllm.device
    q = torch.zeros(1, 2, 1, 32, device=device)
    slope = torch.zeros(2, device=device)
    cache = torch.zeros(2, 2, 32, 32, device=device)
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError, match="64 or 128"):
        flaggems_vllm.lightning_attention(q, q, q, slope)
    with pytest.raises(NotImplementedError, match="64 or 128"):
        flaggems_vllm.lightning_attention_decode(q, q, q, cache, slope, slots)


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


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_vllm_minimax_prefix_and_decode(monkeypatch, state_dtype):
    layer = pytest.importorskip(
        "vllm.model_executor.layers.mamba.linear.minimax_linear_attn"
    )
    torch.manual_seed(930)
    device = flaggems_vllm.device
    t, h, d = 321, 3, 64
    q, k, v = [torch.randn(h, t, d, device=device) for _ in range(3)]
    s = torch.linspace(0.001, 0.2, h, device=device)
    initial = torch.randn(h, d, d, device=device, dtype=state_dtype)
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


@pytest.mark.parametrize("decode", [False, True])
def test_tensor_device_with_another_current_device(decode):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible Hopper devices")
    device = torch.device("cuda", 1)
    t = 1 if decode else 257
    torch.manual_seed(935)
    q, k, v = [torch.randn(1, 4, t, 128, device=device).mul_(0.125) for _ in range(3)]
    state = torch.randn(1, 4, 128, 128, device=device).mul_(0.125)
    slope = torch.linspace(0.002, 0.1, 4, device=device)
    with torch.cuda.device(0):
        if decode:
            expected = state.clone()
            slots = torch.zeros(1, dtype=torch.int64, device=device)
            reference = decode_reference(
                q, k, v, expected, slope, slots, dtype=torch.float64
            )
            actual = flaggems_vllm.lightning_attention_decode(
                q, k, v, state, slope, slots
            )
            assert_relative(actual, reference, 2e-6)
            assert_relative(state, expected, 2e-6)
        else:
            reference, expected = prefix_reference(
                q.cpu(), k.cpu(), v.cpu(), slope.cpu(), state.cpu()
            )
            actual, history = flaggems_vllm.lightning_attention(
                q, k, v, slope, kv_history=state
            )
            assert_relative(actual, reference, 0.0025)
            assert_relative(history, expected, 0.0025)
        assert actual.device == device
        assert torch.cuda.current_device() == 0


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_decode_sm_boundary(offset):
    device = torch.device(flaggems_vllm.device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    # At D=128 and BLOCK_SIZE=32, 4 CTAs launch per head; 2 * 4 * B >= sms switches
    # from narrow 2-warp tiles to 32-column 4-warp tiles around B = ceil(sms / 8).
    b = max(1, (sms + 7) // 8 + offset)
    torch.manual_seed(936)
    q, k, v = [torch.randn(b, 1, 1, 128, device=device).mul_(0.125) for _ in range(3)]
    cache = torch.randn(b, 1, 128, 128, device=device).mul_(0.125)
    expected = cache.clone()
    slots = torch.arange(b, device=device).flip(0)
    slope = torch.tensor([0.01], device=device)
    reference = decode_reference(q, k, v, expected, slope, slots, dtype=torch.float64)
    actual = flaggems_vllm.lightning_attention_decode(q, k, v, cache, slope, slots)
    assert_relative(actual, reference, 2e-6)
    assert_relative(cache, expected, 2e-6)


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_prefix_sm_boundary(offset):
    device = torch.device(flaggems_vllm.device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    # At T=129 and E=128, a 64-column value tile launches six CTAs per head.
    h = (sms + 3) // 4 + offset
    torch.manual_seed(937)
    q, k, v = [torch.randn(1, h, 129, 128, device=device).mul_(0.125) for _ in range(3)]
    initial = torch.randn(1, h, 128, 128, device=device).mul_(0.125)
    slope = torch.linspace(0.002, 0.12, h, device=device)
    reference, expected = prefix_reference(
        q.cpu(), k.cpu(), v.cpu(), slope.cpu(), initial.cpu()
    )
    actual, history = flaggems_vllm.lightning_attention(
        q, k, v, slope, kv_history=initial
    )
    assert_relative(actual, reference, 0.0025)
    assert_relative(history, expected, 0.0025)
