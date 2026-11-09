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

"""MetaX vLLM filtering: public dispatch, exact masks, aliasing and graphs."""

import importlib
import inspect

import pytest
import torch

import flaggems_vllm
from baseline.apply_top_k_top_p import apply_top_k_top_p as reference
from flaggems_vllm.ops.top_k_top_p import top_k_top_p as unsupported
from flaggems_vllm.testing import RESOLUTION

from . import conftest as cfg

pytestmark = [
    pytest.mark.top_k_top_p,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "metax", reason="MetaX-specific implementation"
    ),
]

SHAPES = (
    [(1, 17), (3, 257)]
    if cfg.QUICK_MODE
    else [(1, 1), (1, 17), (3, 257), (7, 8193), (8, 32000), (64, 151936), (1, 262144)]
)


def make_logits(shape, bf16_grid=False):
    generator = torch.Generator().manual_seed(1234)
    logits = torch.randn(shape, generator=generator, dtype=torch.float32)
    if bf16_grid:
        logits = logits.bfloat16().float()
    return logits.to(flaggems_vllm.device)


def filters(m, n, mode):
    device = flaggems_vllm.device
    k = (
        None
        if mode == "p"
        else torch.full((m,), min(20, n), device=device, dtype=torch.int32)
    )
    p = None if mode == "k" else torch.full((m,), 0.8, device=device)
    return k, p


def assert_exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def assert_top_p_mask(actual, logits, p):
    """Check the nucleus decision against an independent CPU FP64 CDF.

    Parallel FP32 softmax/cumsum reductions need not round like torch's
    reduction. Mask differences are permitted only inside the repository's
    FP32 resolution around the cutoff; retained values remain bitwise exact.
    """
    assert not torch.isnan(actual).any()
    kept = torch.isfinite(actual)
    assert_exact(actual[kept], logits[kept])
    assert torch.isneginf(actual[~kept]).all()
    values, order = logits.cpu().double().sort(dim=-1, stable=True)
    cumulative = values.softmax(dim=-1).cumsum(dim=-1)
    sorted_kept = kept.cpu().gather(1, order)
    cutoff = (1 - p.cpu()).double()[:, None]
    tolerance = RESOLUTION[torch.float32]
    must_keep = cumulative > cutoff + tolerance
    must_remove = cumulative < cutoff - tolerance
    must_keep[:, -1] = True
    must_remove[:, -1] = False
    assert sorted_kept[must_keep].all()
    assert not sorted_kept[must_remove].any()
    # Stable threshold filtering retains a suffix, including tied values.
    assert (sorted_kept[:, 1:].int() - sorted_kept[:, :-1].int() >= 0).all()


def test_public_dispatch(monkeypatch):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.top_k_top_p"
    )
    assert inspect.getmodule(flaggems_vllm.top_k_top_p) is backend
    assert not hasattr(flaggems_vllm, "apply_top_k_top_p")
    assert {"top_k_top_p"}.isdisjoint({item[0] for item in flaggems_vllm._FULL_CONFIG})
    called = []
    kernel = backend._count_bf16

    class CountLaunch:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def invoke(*args, **kwargs):
                called.append(True)
                return launch(*args, **kwargs)

            return invoke

    monkeypatch.setattr(backend, "_count_bf16", CountLaunch())
    logits = make_logits((3, 257))
    k, p = filters(3, 257, "both")
    flaggems_vllm.top_k_top_p(logits, p, k)
    torch.cuda.synchronize()
    assert called == [True]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bf16_grid", [False, True])
@pytest.mark.parametrize("mode", ["k", "p", "both"])
def test_filter(shape, bf16_grid, mode):
    logits = make_logits(shape, bf16_grid)
    k, p = filters(*shape, mode)
    before = logits.clone()
    expected = reference(before.clone(), k, p)
    actual = flaggems_vllm.top_k_top_p(logits, p, k)
    assert actual is not logits
    assert_exact(logits, before)
    if mode == "p":
        assert_top_p_mask(actual, before, p)
    else:
        assert_exact(actual, expected)


@pytest.mark.parametrize("p_value", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("k_value", [1, 17])
def test_ties_and_boundaries(p_value, k_value):
    logits = torch.zeros((2, 17), device=flaggems_vllm.device)
    k = torch.full((2,), k_value, dtype=torch.int64, device=logits.device)
    p = torch.full((2,), p_value, device=logits.device)
    expected = reference(logits.clone(), k, p)
    assert_exact(flaggems_vllm.top_k_top_p(logits, p, k), expected)


def test_heterogeneous_rows_and_storage_offset():
    source = make_logits((4, 257), True)
    source[1, 0] += 0.000123  # A single non-BF16 value selects the radix path.
    storage = torch.full((source.numel() + 2,), 123.0, device=source.device)
    logits = storage[1:-1].view_as(source)
    logits.copy_(source)
    k = torch.tensor([1, 20, 257, 3], dtype=torch.int64, device=source.device)
    p = torch.tensor([0.0, 0.8, 1.0, 0.5], device=source.device)
    before_k, before_p = k.clone(), p.clone()
    expected = reference(source.clone(), k, p)
    assert_exact(flaggems_vllm.top_k_top_p(logits, p, k), expected)
    assert torch.equal(k, before_k) and torch.equal(p, before_p)
    assert storage[0].item() == storage[-1].item() == 123.0


@pytest.mark.parametrize(
    "values",
    [
        [-float("inf"), -3.0, -0.0, 0.0, 1.0, 2.0],
        [-float("inf")] * 6,
        [float("nan"), -2.0, -1.0, 0.0, 1.0, float("inf")],
    ],
)
def test_special_values(values):
    logits = torch.tensor([values], device=flaggems_vllm.device)
    p = torch.ones((1,), device=logits.device)
    expected = reference(logits.clone(), None, p)
    assert_exact(flaggems_vllm.top_k_top_p(logits, p), expected)


def test_empty_and_identity():
    logits = make_logits((0, 17))
    p = torch.empty((0,), device=logits.device)
    assert flaggems_vllm.top_k_top_p(logits, p) is not logits
    logits = make_logits((2, 17))
    before = logits.clone()
    with pytest.raises(ValueError, match="cannot both be None"):
        flaggems_vllm.top_k_top_p(logits)
    assert_exact(logits, before)
    with pytest.raises(ValueError, match="cannot both be None"):
        unsupported(logits)
    with pytest.raises(NotImplementedError):
        unsupported(logits, p=torch.ones((2,), device=logits.device))


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "dtype",
        "stride",
        "batch",
        "vocab",
        "empty_vocab",
        "k_dtype",
        "p_dtype",
        "k_shape",
        "p_stride",
        "cpu",
    ],
)
def test_unsupported_metadata(case):
    logits = make_logits((2, 17))
    k, p = filters(2, 17, "both")
    if case == "rank":
        logits = logits[0]
    elif case == "dtype":
        logits = logits.half()
    elif case == "stride":
        logits = logits[:, ::2]
    elif case == "batch":
        logits = make_logits((65, 17))
    elif case == "vocab":
        logits = make_logits((2, 262145))
    elif case == "empty_vocab":
        logits = make_logits((2, 0))
    elif case == "k_dtype":
        k = k.float()
    elif case == "p_dtype":
        p = p.half()
    elif case == "k_shape":
        k = k[:1]
    elif case == "p_stride":
        p = torch.ones((4,), device=logits.device)[::2]
    elif case == "cpu":
        logits = logits.cpu()
    before = logits.clone()
    with pytest.raises(NotImplementedError):
        flaggems_vllm.top_k_top_p(logits, p, k)
    assert_exact(logits, before)


@pytest.mark.parametrize("bf16_grid", [True, False])
@pytest.mark.parametrize("shape", [(1, 4096), (1, 151936), (3, 8193)])
def test_graph_changed_inputs(bf16_grid, shape):
    logits = make_logits(shape, bf16_grid)
    original = logits.clone()
    k, p = filters(*shape, "both")
    flaggems_vllm.top_k_top_p(logits, p, k)
    logits.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flaggems_vllm.top_k_top_p(logits, p, k)
    for shift in (1, 7, 31):
        fresh = original.roll(shift, dims=1)
        k.fill_(min(20 + shift, shape[1]))
        p.fill_(0.5 if shift == 7 else 0.9)
        expected = reference(fresh.clone(), k, p)
        result.fill_(float("nan"))  # A replay must overwrite the output.
        logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert result is not logits
        assert_exact(logits, fresh)
        assert_exact(result, expected)


def test_shared_signature_input_preserved_and_hints():
    logits = make_logits((2, 257))
    before = logits.clone()
    k, p = filters(2, 257, "both")
    result = flaggems_vllm.top_k_top_p(logits, p=p, k=k)
    assert result is not logits
    assert_exact(logits, before)
    assert_exact(result, reference(before.clone(), k, p))
    with pytest.raises(ValueError, match="cannot both be None"):
        flaggems_vllm.top_k_top_p(before)
    for hints in ({"topk_len_override": 20}, {"p_is_one_override": False}):
        with pytest.raises(NotImplementedError, match="only supported by Ascend"):
            flaggems_vllm.top_k_top_p(before, p=p, k=k, **hints)
        assert_exact(before, make_logits((2, 257)))


@pytest.mark.parametrize("n", [1, 17, 4095, 4096, 4097, 8193, 65537, 151936, 262144])
@pytest.mark.parametrize("bf16_grid", [True, False])
def test_single_row_sort_bits(n, bf16_grid):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.top_k_top_p"
    )
    logits = make_logits((1, n), bf16_grid)
    # Signed zeros and different NaN payloads must preserve the radix key order.
    special_bits = torch.tensor(
        [
            -8388608,
            -1082130432,
            -2147483648,
            0,
            1065353216,
            2139095040,
            2143289345,
            2143289346,
            -4194303,
        ],
        dtype=torch.int32,
    )
    if n >= special_bits.numel():
        logits[0, : special_bits.numel()].copy_(special_bits.view(torch.float32))
    before = logits.view(torch.int32).clone()
    bits = before.cpu().long() & 0xFFFFFFFF
    keys = torch.where(
        (bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000
    )
    order = keys.argsort(dim=-1, stable=True)
    expected_bits = before.cpu().gather(1, order)
    result = backend._sorted_values(logits)
    assert torch.equal(result.view(torch.int32).cpu(), expected_bits)
    assert torch.equal(logits.view(torch.int32), before)


def test_single_row_dispatch(monkeypatch):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.top_k_top_p"
    )
    original = backend._sorted_values_single_row
    calls = []

    def observed(logits):
        calls.append(logits.shape)
        return original(logits)

    monkeypatch.setattr(backend, "_sorted_values_single_row", observed)
    logits = make_logits((1, 4097))
    k, p = filters(1, 4097, "both")
    assert_exact(
        flaggems_vllm.top_k_top_p(logits, p, k), reference(logits.clone(), k, p)
    )
    assert calls == [logits.shape]


@pytest.mark.parametrize(
    "shape", [(31, 131072), (32, 131071), (32, 131072), (33, 131073), (64, 151936)]
)
@pytest.mark.parametrize("bf16_grid", [True, False])
@pytest.mark.parametrize("mode", ["k", "p", "both"])
def test_prefix_guard_filter(shape, bf16_grid, mode, monkeypatch):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.top_k_top_p"
    )
    kernel = backend._threshold_prefix_stats
    calls = []

    class ObserveLaunch:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def invoke(*args, **kwargs):
                calls.append(grid)
                return launch(*args, **kwargs)

            return invoke

    monkeypatch.setattr(backend, "_threshold_prefix_stats", ObserveLaunch())
    logits = make_logits(shape, bf16_grid)
    before = logits.clone()
    k, p = filters(*shape, mode)
    actual = flaggems_vllm.top_k_top_p(logits, p, k)
    if mode == "p":
        assert_top_p_mask(actual, before, p)
    else:
        assert_exact(actual, reference(before.clone(), k, p))
    assert_exact(logits, before)
    assert actual.data_ptr() != logits.data_ptr()
    uses_prefix = mode != "k" and shape[0] >= 32 and shape[1] >= 131072
    assert len(calls) == int(uses_prefix)


@pytest.mark.parametrize("m,g", [(1, 1), (1, 16), (3, 33), (32, 512), (64, 1024)])
def test_prefix_stats_cpu_int64(m, g):
    import triton

    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.top_k_top_p"
    )
    generator = torch.Generator().manual_seed(1069)
    less = torch.randint(0, 257, (m, g), generator=generator, dtype=torch.int32)
    equal = torch.randint(0, 257, (m, g), generator=generator, dtype=torch.int32)
    total = torch.empty((m,), dtype=torch.int32, device=flaggems_vllm.device)
    prefix = torch.empty((m, g), dtype=torch.int32, device=flaggems_vllm.device)
    backend._threshold_prefix_stats[(m,)](
        less.to(total.device),
        equal.to(total.device),
        total,
        prefix,
        g,
        triton.next_power_of_2(g),
        num_warps=4,
    )
    assert torch.equal(total.cpu().long(), less.long().sum(-1))
    assert torch.equal(prefix.cpu().long(), equal.long().cumsum(-1) - equal.long())


@pytest.mark.parametrize(
    "shape", [(31, 131072), (32, 131071), (32, 131072), (33, 131073)]
)
@pytest.mark.parametrize("bf16_grid", [True, False])
def test_prefix_guard_graph_changed_inputs(shape, bf16_grid):
    logits = make_logits(shape, bf16_grid)
    original = logits.clone()
    k, p = filters(*shape, "both")
    before_rng = torch.cuda.get_rng_state().clone()
    flaggems_vllm.top_k_top_p(logits, p, k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flaggems_vllm.top_k_top_p(logits, p, k)
    for shift in (1, 7, 31):
        fresh = original.roll(shift, dims=1)
        k.fill_(min(20 + shift, shape[1]))
        p.fill_(0.5 if shift == 7 else 0.9)
        result.fill_(float("nan"))
        expected = reference(fresh.clone(), k, p)
        logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(result, expected)
        assert_exact(logits, fresh)
    assert torch.equal(before_rng, torch.cuda.get_rng_state())


@pytest.mark.parametrize(
    "values",
    [
        [-0.0, 0.0, 0.0, -0.0],
        [0.0, -0.0, -0.0, 0.0],
        [-0.0, 0.0, -0.0, 0.0],
        [0.0, -0.0, 0.0, -0.0],
    ],
)
@pytest.mark.parametrize("p_value", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("mode", ["p", "both"])
def test_signed_zero_nucleus_boundary(values, p_value, mode):
    logits = torch.tensor([values], device=flaggems_vllm.device)
    before = logits.view(torch.int32).clone()
    p = torch.tensor([p_value], device=logits.device)
    k = None if mode == "p" else torch.tensor([2], device=logits.device)
    expected = reference(logits.clone(), k, p)
    actual = flaggems_vllm.top_k_top_p(logits, p=p, k=k)
    # Quartile probabilities are exact. Keep the suffix in input-token order,
    # even when the signed-zero sort bit patterns are interleaved.
    removed = min(int(4 * (1 - p_value)), 3)
    assert torch.equal(torch.isfinite(actual)[0].cpu(), torch.arange(4) >= removed)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(logits.view(torch.int32), before)


def make_signed_zero_logits(shape, bf16_grid):
    m, n = shape
    logits = torch.linspace(-40.0, -30.0, n).repeat(m, 1)
    if bf16_grid:
        logits = logits.bfloat16().float()
    positions = [0, (n - 1) // 3, 2 * (n - 1) // 3, n - 1]
    patterns = [[-0.0, 0.0, 0.0, -0.0], [0.0, -0.0, -0.0, 0.0]]
    for row in range(m):
        logits[row, positions] = torch.tensor(patterns[row % len(patterns)])
    return logits.to(flaggems_vllm.device)


SIGNED_ZERO_SHAPES = [
    (1, 4097),
    (3, 257),
    (31, 131072),
    (32, 131071),
    (32, 131072),
    (64, 151936),
]


@pytest.mark.parametrize("shape", SIGNED_ZERO_SHAPES)
@pytest.mark.parametrize("bf16_grid", [True, False])
@pytest.mark.parametrize("mode", ["p", "both"])
@pytest.mark.parametrize("p_value", [0.0, 0.5, 1.0])
def test_signed_zero_paths(shape, bf16_grid, mode, p_value):
    logits = make_signed_zero_logits(shape, bf16_grid)
    before = logits.view(torch.int32).clone()
    k, p = filters(*shape, mode)
    p.fill_(p_value)
    before_k = None if k is None else k.clone()
    before_p = p.clone()
    expected = reference(logits.clone(), k, p)
    actual = flaggems_vllm.top_k_top_p(logits, p=p, k=k)
    assert actual.data_ptr() != logits.data_ptr()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(logits.view(torch.int32), before)
    assert torch.equal(p, before_p)
    assert k is None or torch.equal(k, before_k)


@pytest.mark.parametrize("shape", [(1, 4097), (3, 257), (32, 131072)])
@pytest.mark.parametrize("bf16_grid", [True, False])
@pytest.mark.parametrize("mode", ["p", "both"])
def test_signed_zero_graph_changed_inputs(shape, bf16_grid, mode):
    logits = make_signed_zero_logits(shape, bf16_grid)
    original = logits.clone()
    k, p = filters(*shape, mode)
    before_rng = torch.cuda.get_rng_state().clone()
    flaggems_vllm.top_k_top_p(logits, p=p, k=k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = flaggems_vllm.top_k_top_p(logits, p=p, k=k)
    for shift, p_value in [(1, 0.25), (7, 0.5), (31, 0.75)]:
        fresh = original.roll(shift, dims=1)
        p.fill_(p_value)
        if k is not None:
            k.fill_(min(20 + shift, shape[1]))
        expected = reference(fresh.clone(), k, p)
        result.fill_(float("nan"))
        logits.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(result.view(torch.int32), expected.view(torch.int32))
        assert torch.equal(logits.view(torch.int32), fresh.view(torch.int32))
    assert torch.equal(before_rng, torch.cuda.get_rng_state())
