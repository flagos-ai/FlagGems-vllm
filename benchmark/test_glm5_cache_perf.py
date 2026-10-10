# SPDX-License-Identifier: Apache-2.0
"""Cache mutation active set; resets are outside CUDA event intervals."""

import json
import math

import pytest
import torch

import flaggems_vllm as gems

from . import conftest
from .test_glm5_ops_perf import _effective_mode, _hadamard_reference, _latency_us

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

ACTIVE = {
    "kpool_compress": [1, 128, 17, 256],
    "kpool_tail_seed": [1, 128, 17, 256],
    "kpool_decode": [1, 128, 17, 256],
    "paged_mqa_logits": [1, 8, 3, 64],
    "gather_state_rows": [1, 128, 17, 256],
    "scatter_state_rows": [1, 128, 17, 256],
    "zero_state_rows": [1, 128, 17, 256],
}


def _pool_reference(keys, scores, ape):
    probabilities = torch.softmax(scores.float() + ape, dim=1)
    pooled = (keys.float() * probabilities).sum(1).bfloat16()
    rotated = _hadamard_reference(pooled).float()
    scale = torch.pow(
        2.0,
        torch.ceil(torch.log2(rotated.abs().amax(-1).clamp_min(1e-4) / 448.0)),
    )
    values = (rotated / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.view(torch.uint8), scale


def _cache_views(cache):
    flat = cache.view(cache.shape[0], -1)
    return flat[:, : 32 * 128].view(-1, 32, 128), flat[:, 32 * 128 :].view(
        torch.float32
    )


def _make_case(family, rows, shape_class):
    torch.manual_seed(20261008 + list(ACTIVE).index(family))
    device = gems.device
    if family.endswith("state_rows"):
        initial = torch.randn(rows + 2, 2, 128, 128, device=device)
        state = initial.clone()
        slots = torch.arange(rows - 1, -1, -1, dtype=torch.int32, device=device)
        values = torch.randn(rows, 2, 128, 128, device=device)

        def baseline():
            if family == "gather_state_rows":
                return state[slots.long()]
            state[slots.long()] = values if family == "scatter_state_rows" else 0
            return state

        def candidate():
            if family == "gather_state_rows":
                return gems.gather_state_rows(state, slots)
            if family == "scatter_state_rows":
                gems.scatter_state_rows(state, slots, values)
            else:
                gems.zero_state_rows(state, slots)
            return state

        def reset():
            state.copy_(initial)

        reset.mutates = family != "gather_state_rows"
        return baseline, candidate, reset, 0

    if family == "paged_mqa_logits":
        length = {"small": 32, "hot": 2051, "odd": 65, "large": 2051}[shape_class]
        pages = math.ceil(length / 32)
        keys = torch.randn(rows, pages, 32, 128, device=device).to(torch.float8_e4m3fn)
        scales = torch.rand(rows, pages, 32, device=device) * 0.05 + 0.01
        cache = torch.empty(rows * pages, 32, 1, 132, dtype=torch.uint8, device=device)
        flat = cache.view(rows * pages, -1)
        flat[:, : 32 * 128].copy_(keys.view(torch.uint8).reshape(rows * pages, -1))
        flat[:, 32 * 128 :].copy_(scales.reshape(rows * pages, 32).view(torch.uint8))
        q = torch.randn(rows, 1, 32, 128, device=device).to(torch.float8_e4m3fn)
        weights = torch.randn(rows, 32, device=device)
        lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
        table = torch.arange(rows * pages, dtype=torch.int32, device=device).view(
            rows, pages
        )

        def baseline():
            k = keys.float().reshape(rows, -1, 128)[:, :length]
            s = scales.reshape(rows, -1)[:, :length]
            dots = q[:, 0].float() @ k.mT
            return (torch.relu(dots * s[:, None, :]) * weights[:, :, None]).sum(1)

        return (
            baseline,
            lambda: gems.paged_mqa_logits(
                (q, None), cache, weights, lengths, table, None, length, True
            ),
            lambda: None,
            2e-5,
        )

    pool = 4
    cache = torch.zeros(math.ceil(rows / 32), 32, 132, dtype=torch.uint8, device=device)
    tail_initial = torch.randn(rows, 2, pool, 128, dtype=torch.bfloat16, device=device)
    tail = tail_initial.clone()
    initial_cache = cache.clone()
    keys = torch.randn(rows, pool, 128, dtype=torch.bfloat16, device=device)
    scores = torch.randn_like(keys)
    ape = torch.randn(pool, 128, device=device)
    locations = torch.arange(rows, dtype=torch.int64, device=device)
    tail_slots = torch.arange(rows * pool, dtype=torch.int64, device=device)

    def reset():
        cache.copy_(initial_cache)
        tail.copy_(tail_initial)

    reset.mutates = True

    def write(keys, scores):
        values, scales = _pool_reference(keys, scores, ape)
        cache_values, cache_scales = _cache_views(cache)
        cache_values[locations // 32, locations % 32] = values
        cache_scales[locations // 32, locations % 32] = scales

    def baseline():
        if family == "kpool_tail_seed":
            tail[:, 0] = keys
            tail[:, 1] = scores
            return tail
        write(keys, scores)
        if family == "kpool_decode":
            tail[:, 0] = keys
            tail[:, 1] = scores
            return cache, tail
        return cache

    def candidate():
        if family == "kpool_tail_seed":
            gems.kpool_seed_tail_cache(
                tail, keys.flatten(0, 1), scores.flatten(0, 1), tail_slots, pool
            )
            return tail
        gems.kpool_compress_and_write_cache(cache, keys, scores, ape, locations, pool)
        return cache

    # Freeze decode metadata before timing, just as the framework owns it.
    if family == "kpool_decode":
        positions = (
            torch.arange(pool, dtype=torch.int32, device=device)
            .expand(rows, -1)
            .contiguous()
        )
        slots = tail_slots.view(rows, pool).int()
        cache_slots = locations[:, None].expand(-1, pool).int()

        def candidate():
            gems.kpool_decode_update_and_maybe_write_cache_batched(
                cache, tail, slots, keys, scores, ape, cache_slots, positions, pool
            )
            return cache, tail

    return baseline, candidate, reset, 0


@torch.no_grad()
@pytest.mark.parametrize("family", tuple(ACTIVE))
@pytest.mark.parametrize("shape_class", ["small", "hot", "odd", "large"])
def test_glm5_cache_performance(family, shape_class, request, record_property):
    if request.config.getoption("level") == "core" and shape_class in {"odd", "large"}:
        pytest.skip("comprehensive active set case")
    rows = ACTIVE[family][["small", "hot", "odd", "large"].index(shape_class)]
    baseline, candidate, reset, tolerance = _make_case(family, rows, shape_class)
    reset()
    expected = baseline()
    expected = (
        tuple(x.clone() for x in expected)
        if isinstance(expected, tuple)
        else expected.clone()
    )
    reset()
    actual = candidate()
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    warmup = float(request.config.getoption("warmup"))
    budget = float(request.config.getoption("iter"))
    mode = request.config.getoption("mode")
    base_us = _latency_us(baseline, reset, warmup, budget, mode)
    kernel_us = _latency_us(candidate, reset, warmup, budget, mode)
    result = {
        "family": family,
        "shape_class": shape_class,
        "shape": [rows],
        "requested_mode": mode,
        "effective_mode": _effective_mode(mode, reset),
        "dtype": "fp32" if family.endswith("state_rows") else "bfloat16/fp8",
        "latency_base_us": base_us,
        "latency_us": kernel_us,
        "SpeedUp": base_us / kernel_us,
        "aggregation": "median",
        "direction": "higher SpeedUp is better",
    }
    record_property("glm5_perf", json.dumps(result))
    conftest.update_result(family, result)
    print(json.dumps(result))
