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

import pytest
import torch

import flaggems_vllm

from .conftest import QUICK_MODE

torch_npu = pytest.importorskip("torch_npu")

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
FULL_CASES = (
    # cache_shape, n rows, block_size
    # ---- production shapes (kernel_details) ----
    ((6220, 4, 256), 8192, 4),  # compressor-state prefill write (production exact)
    ((6220, 4, 256), 8180, 4),  # spec-merge tail rows (8192-12 draft)
    ((6220, 4, 256), 16, 4),  # decode step (16 rows)
    ((6220, 4, 256), 56, 4),  # chunked-prefill tail rows
    # ---- stress shapes ----
    ((12440, 4, 256), 8192, 4),  # stress: doubled slot pool
    ((18660, 128, 1, 512), 8192, 128),  # stress: full-size MLA KV layout (2.4GiB bf16)
    ((64, 4, 256), 64, 4),  # stress: small decode-shaped cache
    ((6220, 4, 256), 16384, 4),  # stress: rows beyond one block-table generation
    (
        (1166, 128, 1, 512),
        512,
        128,
    ),  # (cache,n,bs) MLA KV layout, shrunk from 18660 (9.1GiB)
    ((32, 128, 1, 512), 56, 128),  # tiny MLA table, chunk-tail rows
)
CASES = (FULL_CASES[0], FULL_CASES[2]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized paged scatter targets Ascend",
)


def _ref_scatter(cache, slots, values, block_size):
    """Independent CPU logical reference for unique valid slots."""
    out = cache.cpu().clone()
    rows = values.cpu()
    for i, slot in enumerate(slots.cpu().tolist()):
        if 0 <= slot < out.shape[0] * block_size:
            if block_size == 1:
                out[slot].copy_(rows[i].reshape(out[slot].shape))
            else:
                dst = out[slot // block_size, slot % block_size]
                dst.copy_(rows[i].reshape(dst.shape))
    return out


def _make_inputs(cache_shape, n, block_size, dtype, seed=0):
    torch.manual_seed(seed)
    cache = torch.zeros(cache_shape, dtype=dtype, device="npu")
    values = torch.randn(n, cache_shape[-1], dtype=torch.float32, device="npu").to(
        dtype
    )
    # NOTE: NPU randperm over the full slot space (>2M elems) faults the vector
    # cores; build unique slots on CPU instead.
    total = cache_shape[0] * block_size
    slots = torch.randperm(total, device="cpu")[:n].to(torch.int64).to("npu")
    # inject invalid slots (pad rows)
    n_pad = max(1, n // 16)
    slots[torch.randperm(n, device="cpu")[:n_pad].to("npu")] = -1
    return cache, slots, values


@pytest.mark.paged_scatter_triton
@pytest.mark.parametrize("cache_shape,n,block_size", CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_paged_scatter_accuracy(cache_shape, n, block_size, dtype):
    cache, slots, values = _make_inputs(cache_shape, n, block_size, dtype)
    ref = _ref_scatter(cache, slots, values, block_size)
    new = cache.clone()
    assert flaggems_vllm.paged_scatter_triton(new, slots, values, block_size)
    torch.npu.synchronize()
    assert torch.equal(new.cpu(), ref)


@pytest.mark.paged_scatter_triton
def test_paged_scatter_all_invalid_is_noop():
    cache = torch.randn(64, 4, 256, dtype=torch.bfloat16, device="npu")
    before = cache.clone()
    slots = torch.full((128,), -1, dtype=torch.int64, device="npu")
    values = torch.randn(128, 256, dtype=torch.bfloat16, device="npu")
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    torch.npu.synchronize()
    assert torch.equal(cache, before)


def _backing_case(cache_dtype, value_dtype, slot_dtype=torch.int64):
    backing = torch.full((160,), -17, dtype=cache_dtype, device="npu")
    cache = backing.as_strided((3, 4, 5), (48, 12, 2), 3)
    slot_data = [-(2**63), 0, 2**32, 5, -(2**32), 11, 2**63 - 1]
    if slot_dtype == torch.int32:
        slot_data = [-1, 0, 12, 5, -2, 11, 100]
    slot_backing = torch.zeros(14, dtype=slot_dtype, device="npu")
    slots = slot_backing[::2]
    slots.copy_(torch.tensor(slot_data, dtype=slot_dtype, device="npu"))
    values_backing = torch.zeros((7, 13), dtype=value_dtype, device="npu")
    values = values_backing[:, 1:11:2]
    values.copy_(torch.arange(35, device="npu").reshape(7, 5).to(value_dtype))
    return backing, cache, slots, values


def _expected_backing(backing, cache, slots, values, block_size):
    expected = backing.cpu().clone()
    expected_cache = expected.as_strided(
        cache.shape, cache.stride(), cache.storage_offset()
    )
    logical = _ref_scatter(cache, slots, values, block_size)
    expected_cache.copy_(logical)
    return expected


def _assert_bits_equal(actual, expected):
    actual = actual.cpu().contiguous()
    expected = expected.cpu().contiguous()
    if actual.dtype.is_floating_point:
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        # NaN payload bits are not part of the API; signed zeros are.
        actual[torch.isnan(actual)] = 0
        expected[torch.isnan(expected)] = 0
    if actual.dtype == torch.float32:
        actual, expected = actual.view(torch.int32), expected.view(torch.int32)
    elif actual.dtype in (torch.float16, torch.bfloat16):
        actual, expected = actual.view(torch.int16), expected.view(torch.int16)
    assert torch.equal(actual, expected)


@pytest.mark.paged_scatter_triton
@pytest.mark.parametrize(
    "cache_dtype,value_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float32, torch.float32),
        (torch.int8, torch.int8),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_strided_backing_and_extreme_slots(cache_dtype, value_dtype, slot_dtype):
    backing, cache, slots, values = _backing_case(cache_dtype, value_dtype, slot_dtype)
    if value_dtype != torch.int8:
        values[1].copy_(
            torch.tensor(
                [float("nan"), -0.0, 0.0, 1.003, -1.003],
                dtype=value_dtype,
                device="npu",
            )
        )
    expected = _expected_backing(backing, cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_supports(cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    torch.npu.synchronize()
    _assert_bits_equal(backing, expected)


@pytest.mark.paged_scatter_triton
def test_metadata_and_empty_never_launch(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.paged_scatter"
    )

    class NoLaunch:
        def __getitem__(self, grid):
            pytest.fail("metadata or empty input launched a kernel")

    monkeypatch.setattr(module, "_paged_scatter_kernel", NoLaunch())
    _, cache, slots, values = _backing_case(torch.float32, torch.float32)
    assert flaggems_vllm.paged_scatter_supports(cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_supports(cache, slots[:0], values[:0], 4)
    assert flaggems_vllm.paged_scatter_triton(cache, slots[:0], values[:0], 4)


@pytest.mark.paged_scatter_triton
def test_reject_before_mutation():
    backing, cache, slots, values = _backing_case(torch.float32, torch.float32)
    before = backing.cpu().clone()
    cases = [
        (cache, slots, values.to(torch.int32)),
        (cache.as_strided((3, 4, 5), (1, 1, 1)), slots, values),
        (cache, slots, cache.as_strided((7, 5), (5, 1))),
    ]
    for bad_cache, bad_slots, bad_values in cases:
        assert not flaggems_vllm.paged_scatter_supports(
            bad_cache, bad_slots, bad_values, 4
        )
        assert not flaggems_vllm.paged_scatter_triton(
            bad_cache, bad_slots, bad_values, 4
        )
        _assert_bits_equal(backing, before)
    for bad_slots, bad_values in [(slots[:-1], values), (slots, values[:, :-1])]:
        with pytest.raises(ValueError):
            flaggems_vllm.paged_scatter_triton(cache, bad_slots, bad_values, 4)
        _assert_bits_equal(backing, before)
    with pytest.raises(ValueError):
        flaggems_vllm.paged_scatter_triton(cache, slots.cpu(), values, 4)
    _assert_bits_equal(backing, before)


@pytest.mark.paged_scatter_triton
def test_graph_replay_updates_inputs_and_resets_backing():
    backing, cache, slots, values = _backing_case(torch.float32, torch.float32)
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    replay_slots = [
        [11, -1, 5, 0, 12, 2**32, -(2**32)],
        [-1, 12, 2**32, -(2**32), -(2**63), 2**63 - 1, -7],
        [1, 2, 3, 4, 5, 6, 7],
    ]
    for iteration, slot_data in enumerate(replay_slots):
        backing.fill_(-23 - iteration)
        slots.copy_(torch.tensor(slot_data, dtype=torch.int64, device="npu"))
        values.fill_(iteration + 3)
        values[0].copy_(
            torch.tensor(
                [float("nan"), -0.0, 0.0, float("inf"), -float("inf")],
                device="npu",
            )
        )
        expected = _expected_backing(backing, cache, slots, values, 4)
        graph.replay()
        torch.npu.synchronize()
        _assert_bits_equal(backing, expected)


@pytest.mark.paged_scatter_triton
@pytest.mark.parametrize("block_size", [1, 3])
def test_row_and_non_power_of_two_pages(block_size):
    shape = (12, 7) if block_size == 1 else (4, block_size, 7)
    cache = torch.full(shape, -1, dtype=torch.float32, device="npu")
    slots = torch.tensor([0, 11, -1, 12], dtype=torch.int64, device="npu")
    values = torch.arange(28, dtype=torch.float32, device="npu").reshape(4, 7)
    expected = _ref_scatter(cache, slots, values, block_size)
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, block_size)
    torch.npu.synchronize()
    _assert_bits_equal(cache, expected)


@pytest.mark.paged_scatter_triton
def test_launch_failure_propagates_without_retry(monkeypatch):
    import importlib

    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.paged_scatter"
    )
    calls = []

    class FailingLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(grid)
                raise RuntimeError("injected launch failure")

            return launch

    monkeypatch.setattr(module, "_paged_scatter_kernel", FailingLaunch())
    backing, cache, slots, values = _backing_case(torch.float32, torch.float32)
    before = backing.cpu().clone()
    with pytest.raises(RuntimeError, match="injected launch failure"):
        flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    assert len(calls) == 1
    _assert_bits_equal(backing, before)


@pytest.mark.paged_scatter_triton
def test_transposed_pages_preserve_backing():
    backing = torch.full((4, 33, 7), -17, dtype=torch.bfloat16, device="npu")
    cache = backing.transpose(0, 1)
    slots = torch.tensor([0, 3, 4, 131, -1, 132], device="npu")
    values = torch.arange(42, dtype=torch.float32, device="npu").reshape(6, 7)
    expected = _expected_backing(backing, cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_supports(cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    torch.npu.synchronize()
    _assert_bits_equal(backing, expected)


@pytest.mark.paged_scatter_triton
def test_multidimensional_values_reshape_copy():
    cache = torch.full((3, 4, 6), -17, dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, 11, -1], device="npu")
    values = torch.arange(18, dtype=torch.float32, device="npu").reshape(3, 2, 3)
    values = values.transpose(1, 2)
    assert not values.is_contiguous()
    expected = _ref_scatter(cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_supports(cache, slots, values, 4)
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    torch.npu.synchronize()
    _assert_bits_equal(cache, expected)
