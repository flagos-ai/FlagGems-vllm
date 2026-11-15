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

"""Independent contract tests for the stateless Murmur3 Gumbel sampler."""

import importlib
import json
import math
import os

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.gumbel_sample import (
    _combine_kernel,
    _gumbel_from_uniform,
    _masked_argmax,
    _rng_token_offsets,
    _sample_scores,
    _source_offset,
    _uniform32_from_random32,
    _uniform64_from_random53,
    gumbel_sample,
    murmur3_hash32,
    murmur3_uniform32,
    murmur3_uniform64,
)

device = flaggems_vllm.device

KAT_INPUTS = [
    (0, 0, 0, 0x8134CDF8, 0xD28AD962),
    (1, 2, 3, 0x23B53F91, 0xE3AAF818),
    (-1, 17, 1023, 0xC8C13C14, 0xD223C51C),
    (0x123456789ABCDEF, 123456789, 199999, 0x5DDE974F, 0x0547D2FE),
    (-(1 << 63), 1 << 30, 4096, 0xF1A9355B, 0x3AED160F),
    ((1 << 63) - 1, (1 << 30) + 7, 0xFFFFFFFF, 0xD952F9AA, 0x49E8EA7F),
]
MAPPING = [
    4,
    1,
    4,
    0,
    3,
    1,
    2,
    4,
    0,
    2,
    3,
    1,
    4,
    0,
    2,
    3,
    1,
    0,
    4,
    2,
    3,
    1,
    4,
    0,
    2,
    3,
    1,
    0,
    4,
    2,
    3,
    1,
    4,
    0,
    2,
    3,
    1,
]
POSITIONS = [
    0,
    17,
    1023,
    1 << 30,
    (1 << 32) + 17,
] + [100_003 + 17 * i for i in range(32)]
TEMPERATURES = [0.25, 0.5, 0.8, 1.0, 1.75]
SEEDS = [0, 1, -1, 0x123456789ABCDEF, -(1 << 63)]
STAT_ALPHA = 2e-7


def _u32(value):
    return value & 0xFFFFFFFF


def _rotl32(value, shift):
    value = _u32(value)
    return _u32((value << shift) | (value >> (32 - shift)))


def _mix32(h, key):
    key = _u32(key * 0xCC9E2D51)
    key = _rotl32(key, 15)
    key = _u32(key * 0x1B873593)
    h = _u32(h ^ key)
    h = _rotl32(h, 13)
    return _u32(h * 5 + 0xE6546B64)


def _fmix32(h):
    h = _u32(h ^ (h >> 16))
    h = _u32(h * 0x85EBCA6B)
    h = _u32(h ^ (h >> 13))
    h = _u32(h * 0xC2B2AE35)
    return _u32(h ^ (h >> 16))


def cpu_murmur3_hash32(seed, pos, offset, domain=0):
    seed_bits = seed & ((1 << 64) - 1)
    h = _u32(domain)
    h = _mix32(h, seed_bits)
    h = _mix32(h, seed_bits >> 32)
    h = _mix32(h, pos)
    h = _mix32(h, offset)
    return _fmix32(h ^ 16)


def cpu_uniform32(seed, pos, offset):
    random32 = cpu_murmur3_hash32(seed, pos, offset)
    hi16 = random32 >> 16
    lo16 = random32 & 0xFFFF
    return np.float32(
        np.float32(hi16) * np.float32(2.0**-16)
        + (np.float32(lo16) + np.float32(0.5)) * np.float32(2.0**-32)
    )


def cpu_uniform64(seed, pos, offset):
    lo = cpu_murmur3_hash32(seed, pos, offset)
    hi = cpu_murmur3_hash32(seed, pos, offset, 0x9E3779B9)
    random53 = ((hi << 32) | lo) >> 11
    uniform = (random53 + 0.5) * 2.0**-53
    return uniform - 2.0**-53 if uniform == 1.0 else uniform


def _fp32_gumbel(uniform):
    return -math.log(-math.log1p(-float(uniform)))


def _lcg_values(count, low=-500, high=500):
    state = 1234
    values = np.empty(count, dtype=np.float32)
    span = high - low + 1
    for index in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values[index] = (state % span + low) / 100.0
    return values


def _tensor(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype, device=device)


def _int_tensor(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device=device)


def _strided_vector(values, dtype):
    backing = torch.empty(len(values) * 2, dtype=dtype, device=device)
    backing[::2] = torch.tensor(values, dtype=dtype, device=device)
    return backing[::2]


def _exact_dtypes():
    dtypes = [torch.float16, torch.float32]
    if flaggems_vllm.runtime.device.support_bf16:
        dtypes.insert(1, torch.bfloat16)
    return dtypes


def _call(logits, mapping, temperature, seed, pos, **kwargs):
    return gumbel_sample(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        kwargs.pop("apply_temperature", True),
        kwargs.pop("is_drafting", False),
        **kwargs,
    )


def _assert_ids(output, vocab_size):
    assert output.dtype == torch.int64
    assert output.ndim == 1
    assert torch.all((output >= 0) & (output < vocab_size))


def _float_bits(tensor):
    return tensor.view(torch.int32 if tensor.dtype == torch.float32 else torch.int16)


def _fp64_probe():
    requested_skip = os.environ.get("GUMBEL_SKIP_FP64")
    if requested_skip:
        pytest.skip(f"GUMBEL_SKIP_FP64 evidence: {requested_skip}")

    logits = _tensor([[0.0, 1.0]], torch.float32)
    mapping = _int_tensor([0], torch.int32)
    temperature = _tensor([1.0])
    seed = _int_tensor([0])
    pos = _int_tensor([0])
    if _is_ascend_triton_351():
        reason = "verified Ascend Triton 3.5.1 SIMD FP64 primitives do not compile"
        with pytest.raises(NotImplementedError, match="Ascend Triton 3.5.1 SIMD"):
            _call(logits, mapping, temperature, seed, pos, use_fp64=True)
        pytest.skip(reason)
    if (
        flaggems_vllm.vendor_name == "iluvatar"
        and triton.__version__.split("+", 1)[0] == "3.1.0"
    ):
        reason = "verified Iluvatar Triton 3.1.0 FP64 log/reduction is unsupported"
        with pytest.raises(NotImplementedError, match="Iluvatar Triton 3.1.0"):
            _call(logits, mapping, temperature, seed, pos, use_fp64=True)
        pytest.skip(reason)
    output = _call(logits, mapping, temperature, seed, pos, use_fp64=True)
    _assert_ids(output, 2)


def _is_ascend_triton_351():
    return (
        flaggems_vllm.vendor_name == "ascend"
        and triton.__version__.split("+", 1)[0] == "3.5.1"
    )


def _require_shared_compiled_launcher():
    module = importlib.import_module("flaggems_vllm.ops.gumbel_sample")
    if not getattr(module, "_COMPILED_FAST_LAUNCH", False):
        pytest.skip("compiled launcher disabled by the active backend capability gate")


def _current_driver_stream():
    from triton.runtime import driver

    return driver.active.get_current_stream(driver.active.get_current_device())


def _device_rng_state():
    device_api = flaggems_vllm.runtime.torch_device_fn
    getter = getattr(device_api, "get_rng_state", None)
    if getter is None:
        pytest.fail("runtime device API does not expose get_rng_state")
    with device_api.device(device):
        return getter().clone()


@triton.jit
def _candidate_rng_kernel(
    seeds_ptr,
    pos_ptr,
    offset_ptr,
    hash_ptr,
    domain_hash_ptr,
    uniform_ptr,
    noise_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    mask = index < numel
    seed = tl.load(seeds_ptr + index, mask=mask)
    pos = tl.load(pos_ptr + index, mask=mask)
    offset = tl.load(offset_ptr + index, mask=mask)
    hashed = murmur3_hash32(seed, pos, offset)
    domain_hashed = murmur3_hash32(seed, pos, offset, domain=0x9E3779B9)
    uniform = murmur3_uniform32(seed, pos, offset)
    noise = _gumbel_from_uniform(uniform, USE_FP64=False)
    tl.store(hash_ptr + index, hashed.to(tl.int64), mask=mask)
    tl.store(domain_hash_ptr + index, domain_hashed.to(tl.int64), mask=mask)
    tl.store(uniform_ptr + index, uniform, mask=mask)
    tl.store(noise_ptr + index, noise, mask=mask)


@triton.jit
def _candidate_uniform32_endpoint_kernel(
    random32_ptr,
    uniform32_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    mask = index < numel
    random32 = tl.load(random32_ptr + index, mask=mask).to(tl.uint32)
    tl.store(uniform32_ptr + index, _uniform32_from_random32(random32), mask=mask)


@triton.jit
def _candidate_uniform64_endpoint_kernel(
    random53_ptr,
    uniform64_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    mask = index < numel
    random53 = tl.load(random53_ptr + index, mask=mask).to(tl.uint64)
    tl.store(uniform64_ptr + index, _uniform64_from_random53(random53), mask=mask)


@triton.jit
def _candidate_rng64_kernel(
    seeds_ptr,
    pos_ptr,
    offset_ptr,
    uniform_ptr,
    noise_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    mask = index < numel
    seed = tl.load(seeds_ptr + index, mask=mask)
    pos = tl.load(pos_ptr + index, mask=mask)
    offset = tl.load(offset_ptr + index, mask=mask)
    uniform = murmur3_uniform64(seed, pos, offset)
    noise = _gumbel_from_uniform(uniform, USE_FP64=True)
    tl.store(uniform_ptr + index, uniform, mask=mask)
    tl.store(noise_ptr + index, noise, mask=mask)


@triton.jit
def _candidate_score_argmax_kernel(
    logits_ptr,
    noise_ptr,
    temperature_ptr,
    scores_ptr,
    winner_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    mask = index < numel
    logits = tl.load(logits_ptr + index, mask=mask, other=float("-inf"))
    noise = tl.load(noise_ptr + index, mask=mask, other=0.0)
    temperature = tl.load(temperature_ptr)
    scores = _sample_scores(logits, noise, temperature, APPLY_TEMPERATURE, USE_FP64)
    _, winner = _masked_argmax(scores, index, mask)
    tl.store(scores_ptr + index, scores, mask=mask)
    tl.store(winner_ptr, winner.to(tl.int64))


@triton.jit
def _candidate_source_offsets_kernel(output, stride_col: tl.constexpr):
    index = tl.arange(0, 4)
    offset = _source_offset(tl.full((), 0, tl.int32), index, 0, stride_col)
    tl.store(output + index, offset, index < 3)


@pytest.mark.gumbel_sample
def test_strided_source_offsets_exceed_int32():
    offsets = torch.empty(3, dtype=torch.int64, device=device)
    _candidate_source_offsets_kernel[(1,)](offsets, stride_col=1 << 30)
    assert offsets.cpu().tolist() == [0, 1 << 30, 1 << 31]


@triton.jit
def _candidate_generated_keys_kernel(
    seeds, positions, output, block_id: tl.constexpr, USE_SCAN: tl.constexpr
):
    lane = tl.arange(0, 1024)
    indices = block_id * 1024 + lane
    keys = _rng_token_offsets(indices, tl.full((), block_id, tl.int32), 1024, USE_SCAN)
    seed = tl.load(seeds)
    pos = tl.load(positions)
    first = murmur3_hash32(seed, pos, keys)
    second = murmur3_hash32(seed, pos, keys, domain=0x9E3779B9)
    tl.store(output + lane, first.to(tl.int64), lane < 8)
    tl.store(output + 8 + lane, second.to(tl.int64), lane < 8)


@pytest.mark.gumbel_sample
def test_generated_token_keys_match_independent_scalar_seed_oracle():
    seed, pos = 0x123456789ABCDEF, (1 << 32) + 17
    seeds, positions = _int_tensor([seed]), _int_tensor([pos])
    output = torch.empty(16, dtype=torch.int64, device=device)
    use_scan = (
        flaggems_vllm.vendor_name == "iluvatar"
        and triton.__version__.split("+", 1)[0] == "3.1.0"
    )
    for block_id in (0, 1, 195, (1 << 21) - 1):
        _candidate_generated_keys_kernel[(1,)](
            seeds, positions, output, block_id=block_id, USE_SCAN=use_scan
        )
        expected = [
            cpu_murmur3_hash32(seed, pos, block_id * 1024 + i, domain)
            for domain in (0, 0x9E3779B9)
            for i in range(8)
        ]
        assert output.cpu().tolist() == expected


@pytest.mark.gumbel_sample
def test_cpu_murmur3_known_answers_and_position_wrap():
    for seed, pos, offset, expected, expected_domain in KAT_INPUTS:
        assert cpu_murmur3_hash32(seed, pos, offset) == expected
        assert cpu_murmur3_hash32(seed, pos, offset, 0x9E3779B9) == expected_domain
        assert cpu_murmur3_hash32(seed, pos + (1 << 32), offset) == expected


@pytest.mark.gumbel_sample
def test_candidate_murmur_uniform_and_noise_helpers():
    seeds = _int_tensor([row[0] for row in KAT_INPUTS])
    positions = _int_tensor([row[1] for row in KAT_INPUTS])
    offsets = _int_tensor([row[2] for row in KAT_INPUTS])
    hashes = torch.empty_like(seeds)
    domain_hashes = torch.empty_like(seeds)
    uniforms = torch.empty(len(KAT_INPUTS), dtype=torch.float32, device=device)
    noises = torch.empty_like(uniforms)
    _candidate_rng_kernel[(1,)](
        seeds,
        positions,
        offsets,
        hashes,
        domain_hashes,
        uniforms,
        noises,
        numel=len(KAT_INPUTS),
        BLOCK=8,
    )

    assert hashes.cpu().tolist() == [row[3] for row in KAT_INPUTS]
    assert domain_hashes.cpu().tolist() == [row[4] for row in KAT_INPUTS]
    expected_uniform = [cpu_uniform32(*row[:3]) for row in KAT_INPUTS]
    expected_noise = [_fp32_gumbel(value) for value in expected_uniform]
    np.testing.assert_allclose(
        uniforms.cpu().numpy(), expected_uniform, rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        noises.cpu().numpy(), expected_noise, rtol=3e-6, atol=3e-6
    )


@pytest.mark.gumbel_sample
def test_candidate_uniform32_endpoints():
    random32 = _int_tensor([0, 0xFFFFFFFF])
    uniform32 = torch.empty(2, dtype=torch.float32, device=device)
    _candidate_uniform32_endpoint_kernel[(1,)](
        random32,
        uniform32,
        numel=2,
        BLOCK=2,
    )
    assert uniform32.cpu().tolist() == [2.0**-33, 1.0]


@pytest.mark.gumbel_sample
def test_candidate_uniform64_endpoints():
    _fp64_probe()
    random53 = _int_tensor([0, (1 << 53) - 1])
    uniform64 = torch.empty(2, dtype=torch.float64, device=device)
    _candidate_uniform64_endpoint_kernel[(1,)](
        random53,
        uniform64,
        numel=2,
        BLOCK=2,
    )
    assert uniform64.cpu().tolist() == [2.0**-54, 1.0 - 2.0**-53]


@pytest.mark.gumbel_sample
def test_candidate_fp64_rng_helper_math():
    _fp64_probe()
    seeds = _int_tensor([row[0] for row in KAT_INPUTS])
    positions = _int_tensor([row[1] for row in KAT_INPUTS])
    offsets = _int_tensor([row[2] for row in KAT_INPUTS])
    uniforms = torch.empty(len(KAT_INPUTS), dtype=torch.float64, device=device)
    noises = torch.empty_like(uniforms)
    _candidate_rng64_kernel[(1,)](
        seeds, positions, offsets, uniforms, noises, numel=len(KAT_INPUTS), BLOCK=8
    )
    expected_uniform = [cpu_uniform64(*row[:3]) for row in KAT_INPUTS]
    expected_noise = [-math.log(-math.log(value)) for value in expected_uniform]
    np.testing.assert_allclose(uniforms.cpu().numpy(), expected_uniform, rtol=0, atol=0)
    np.testing.assert_allclose(
        noises.cpu().numpy(), expected_noise, rtol=2e-12, atol=2e-12
    )
    logits = _tensor([1.0, -1.0])
    fixed_noise = _tensor([0.25, -0.5], torch.float64)
    scores = torch.empty(2, dtype=torch.float64, device=device)
    winner = torch.empty(1, dtype=torch.int64, device=device)
    _candidate_score_argmax_kernel[(1,)](
        logits,
        fixed_noise,
        _tensor([0.75]),
        scores,
        winner,
        numel=2,
        BLOCK=2,
        APPLY_TEMPERATURE=True,
        USE_FP64=True,
    )
    expected = (np.array([1.0, -1.0], dtype=np.float32) / np.float32(0.75)).astype(
        np.float64
    ) + np.array([0.25, -0.5])
    np.testing.assert_allclose(scores.cpu().numpy(), expected, rtol=0, atol=2e-7)
    assert winner.item() == 0


@pytest.mark.gumbel_sample
def test_candidate_fixed_noise_score_and_masked_argmax_helpers():
    logits = _tensor([-1.0, 1.0, 1.0, -2.0, 0.5, -3.0, 2.0])
    noise = _tensor([0.0, 0.0, 0.0, 2.0, -1.0, 3.0, -3.0])
    temperature = _tensor([0.5])
    scores = torch.empty(7, dtype=torch.float32, device=device)
    winner = torch.empty(1, dtype=torch.int64, device=device)
    _candidate_score_argmax_kernel[(1,)](
        logits,
        noise,
        temperature,
        scores,
        winner,
        numel=7,
        BLOCK=8,
        APPLY_TEMPERATURE=True,
        USE_FP64=False,
    )
    expected = logits.cpu().numpy().astype(np.float64) / 0.5 + noise.cpu().numpy()
    np.testing.assert_allclose(scores.cpu().numpy(), expected, rtol=0, atol=2e-7)
    assert winner.item() == int(np.argmax(expected))

    _candidate_score_argmax_kernel[(1,)](
        logits,
        noise,
        temperature,
        scores,
        winner,
        numel=7,
        BLOCK=8,
        APPLY_TEMPERATURE=False,
        USE_FP64=False,
    )
    expected = logits.cpu().numpy().astype(np.float64) + noise.cpu().numpy()
    np.testing.assert_allclose(scores.cpu().numpy(), expected, rtol=0, atol=2e-7)
    assert winner.item() == int(np.argmax(expected))


def _deterministic_inputs(vocab_size, dtype=torch.float32):
    logits = _lcg_values(37 * vocab_size).reshape(37, vocab_size)
    logits = _tensor(logits, dtype)
    mapping = _int_tensor(MAPPING, torch.int32)
    temperature = _tensor(TEMPERATURES)
    seed = _int_tensor(SEEDS)
    pos = _int_tensor(POSITIONS)
    return logits, mapping, temperature, seed, pos


def _cpu_winners(
    logits,
    mapping,
    temperature,
    seed,
    pos,
    use_fp64=False,
    is_drafting=False,
    apply_temperature=True,
    cache_columns=None,
):
    values = logits.detach().cpu().to(torch.float32).numpy()
    mapping = mapping.detach().cpu().numpy().astype(np.int64)
    temperature = temperature.detach().cpu().numpy()
    seed = seed.detach().cpu().numpy().astype(np.int64)
    pos = pos.detach().cpu().numpy().astype(np.int64)
    winners, margins = [], []
    for row, request in enumerate(mapping):
        if request < 0 or temperature[request] == 0:
            scores = values[row].astype(np.float64)
        else:
            local_seed = int(seed[request])
            if cache_columns is not None:
                local_seed ^= int(cache_columns[row]) << 32
            local_pos = int(pos[row]) + ((1 << 30) if is_drafting else 0)
            # The source divides in FP32 before optional FP64 noise addition.
            scores = values[row].astype(np.float32)
            if apply_temperature:
                scores = (scores / np.float32(temperature[request])).astype(np.float32)
            if use_fp64:
                scores = scores.astype(np.float64)
            for index in range(scores.size):
                uniform = (
                    cpu_uniform64(local_seed, local_pos, index)
                    if use_fp64
                    else cpu_uniform32(local_seed, local_pos, index)
                )
                scores[index] += (
                    -math.log(-math.log(uniform)) if use_fp64 else _fp32_gumbel(uniform)
                )
        order = np.argsort(scores)[::-1]
        winner = int(order[0])
        runner_up = int(order[1]) if scores.size > 1 else winner
        winners.append(winner)
        margins.append(float(scores[winner] - scores[runner_up]))
    return np.asarray(winners), np.asarray(margins)


@pytest.mark.gumbel_sample
def test_cpu_fp64_oracle_rounds_temperature_division_in_fp32():
    winners, _ = _cpu_winners(
        torch.tensor([[0.0, -0.5450239181518555]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0.35], dtype=torch.float32),
        torch.tensor([7], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        use_fp64=True,
    )
    assert winners.tolist() == [0]


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("vocab_size", [1, 7, 999, 1024, 1025, 4096, 4097])
def test_greedy_boundaries_and_negative_request_mapping(vocab_size):
    rows = torch.full((6, vocab_size + 1), -float("inf"), device=device)[:, 1:]
    rows[0].fill_(0)
    rows[1, 0] = 4
    rows[1, min(vocab_size - 1, 3)] = 4
    rows[2, 0] = float("inf")
    rows[2, vocab_size - 1] = float("inf")
    rows[3].fill_(-float("inf"))
    rows[4, vocab_size - 1] = 1
    rows[5, min(vocab_size - 1, 2)] = 3
    mapping = _int_tensor([0, 1, 2, 3, -1, -1], torch.int32)
    temperature = _tensor([0.0, 0.0, 0.0, 0.0])
    seed = _int_tensor([7, 8, 9, 10])
    pos = _int_tensor([0, 1, 2, 3, 4, 5])
    output = _call(rows, mapping, temperature, seed, pos)
    expected = _int_tensor([0, 0, 0, 0, vocab_size - 1, min(vocab_size - 1, 2)])
    assert torch.equal(output, expected)


@pytest.mark.gumbel_sample
def test_large_greedy_replay_covers_every_row():
    vocab = 151936
    cpu_logits = ((torch.arange(vocab, device="cpu") * 37) % 101).float() / 10
    row = cpu_logits.to(device)
    expected_id = int(cpu_logits.argmax().item())
    for n in (64, 128, 65):
        logits = row.expand(n, vocab)
        mapping = torch.zeros(n, dtype=torch.int32, device=device)
        positions = torch.arange(n, dtype=torch.int64, device=device)
        args = (logits, mapping, _tensor([0.0]), _int_tensor([7]), positions)
        first = _call(*args)
        replay = _call(*args)
        assert first.cpu().tolist() == [expected_id] * n
        assert replay.cpu().tolist() == [expected_id] * n


@pytest.mark.gumbel_sample
def test_combine_covers_all_rows_after_grid_changes():
    blocks = 149
    # Match the production launch configuration when sharing its libentry cache.
    launch = {}
    if (
        flaggems_vllm.vendor_name == "ascend"
        and triton.__version__.split("+", 1)[0] == "3.5.1"
    ):
        launch["enable_auto_blockify"] = False
    for n in (16, 128, 65):
        rows = torch.arange(n, dtype=torch.int64, device="cpu")
        winners = rows % blocks
        ids = (torch.arange(blocks, device="cpu")[None, :] * 1024 + rows[:, None]).to(
            device
        )
        values = torch.zeros((n, blocks), dtype=torch.float32, device=device)
        values[rows.to(device), winners.to(device)] = 1
        expected = (winners * 1024 + rows).tolist()
        for _ in range(2):
            sampled = torch.full((n,), -777, dtype=torch.int64, device=device)
            _combine_kernel[(n,)](
                ids,
                blocks,
                values,
                blocks,
                sampled,
                blocks,
                PADDED_NUM_BLOCKS=triton.next_power_of_2(blocks),
                num_warps=4,
                num_stages=1,
                **launch,
            )
            assert sampled.cpu().tolist() == expected


@pytest.mark.gumbel_sample
def test_bounded_launch_preserves_global_rng_and_cache_rows(monkeypatch):
    if not (
        flaggems_vllm.vendor_name == "ascend"
        and triton.__version__.split("+", 1)[0] == "3.5.1"
    ):
        pytest.skip("bounded launcher applies to the verified Ascend 3.5.1 stack")
    module = importlib.import_module("flaggems_vllm.ops.gumbel_sample")
    # Force both row and block offsets without allocating a 67M-token vocabulary.
    monkeypatch.setattr(module, "_ASCEND_GRID_LIMIT", 2)
    logits, mapping, temperature, seed, pos = _deterministic_inputs(4097)
    logits, mapping, pos = logits[:4], mapping[:4], pos[:4]
    columns = torch.arange(4, dtype=torch.int64, device=device)
    cache = torch.full((temperature.numel(), 4, 4097), -777.0, device=device)
    args = (logits, mapping, temperature, seed, pos)
    kwargs = dict(logits_cache=cache, logits_cache_col=columns, is_drafting=True)
    first = _call(*args, **kwargs)
    replay = _call(*args, **kwargs)
    expected, margins = _cpu_winners(
        *args, is_drafting=True, cache_columns=np.arange(4)
    )
    valid = margins > 2e-3
    assert valid.any()
    assert first.cpu().numpy()[valid].tolist() == expected[valid].tolist()
    assert torch.equal(first, replay)
    expected_cache = torch.full_like(cache, -777.0)
    for row, request in enumerate(mapping.cpu().tolist()):
        expected_cache[request, row] = logits[row]
    assert torch.equal(cache, expected_cache)


@pytest.mark.gumbel_sample
def test_bounded_large_tile_preserves_global_rng(monkeypatch):
    if not _is_ascend_triton_351():
        pytest.skip(
            "large-tile bounded launch applies to the verified Ascend 3.5.1 stack"
        )
    module = importlib.import_module("flaggems_vllm.ops.gumbel_sample")
    # Split two B4096 blocks and four rows across both global offset dimensions.
    monkeypatch.setattr(module, "_ASCEND_GRID_LIMIT", 1)
    logits, mapping, temperature, seed, pos = _deterministic_inputs(4097)
    args = (logits[:4], mapping[:4], temperature, seed, pos[:4])
    first = _call(*args, is_drafting=True)
    replay = _call(*args, is_drafting=True)
    expected, margins = _cpu_winners(*args, is_drafting=True)
    valid = margins > 2e-3
    assert valid.any()
    assert first.cpu().numpy()[valid].tolist() == expected[valid].tolist()
    assert torch.equal(first, replay)


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("dtype", _exact_dtypes())
@pytest.mark.parametrize("vocab_size", [7, 1025, 4096, 4097])
def test_positive_temperature_matches_independent_oracle(dtype, vocab_size):
    logits, mapping, temperature, seed, pos = _deterministic_inputs(vocab_size, dtype)
    output = _call(logits, mapping, temperature, seed, pos)
    expected, margins = _cpu_winners(logits, mapping, temperature, seed, pos)
    valid = margins > 2e-3
    assert valid.any()
    valid_indices = _int_tensor(np.flatnonzero(valid))
    assert torch.equal(
        output[valid_indices].cpu(), torch.tensor(expected[valid], dtype=torch.int64)
    )
    _assert_ids(output, vocab_size)


@pytest.mark.gumbel_sample
def test_fp64_positive_temperature_matches_independent_oracle():
    _fp64_probe()
    logits, mapping, temperature, seed, pos = _deterministic_inputs(1025)
    output = _call(logits, mapping, temperature, seed, pos, use_fp64=True)
    expected, margins = _cpu_winners(
        logits, mapping, temperature, seed, pos, use_fp64=True
    )
    valid = margins > 2e-8
    assert valid.any()
    valid_indices = _int_tensor(np.flatnonzero(valid))
    assert torch.equal(
        output[valid_indices].cpu(), torch.tensor(expected[valid], dtype=torch.int64)
    )


@pytest.mark.gumbel_sample
def test_partition_permutation_and_temperature_predivision_equivalence():
    logits, mapping, temperature, seed, pos = _deterministic_inputs(1025)
    whole = _call(logits, mapping, temperature, seed, pos)
    pieces = [
        _call(logits[start:end], mapping[start:end], temperature, seed, pos[start:end])
        for start, end in ((0, 11), (11, 23), (23, 37))
    ]
    assert torch.equal(whole, torch.cat(pieces))

    permutation = _int_tensor(
        [
            36,
            2,
            17,
            5,
            11,
            29,
            0,
            34,
            9,
            16,
            31,
            4,
            23,
            13,
            26,
            1,
            18,
            32,
            8,
            22,
            15,
            35,
            7,
            20,
            3,
            28,
            14,
            33,
            6,
            25,
            12,
            30,
            10,
            24,
            19,
            21,
            27,
        ]
    )
    permuted = _call(
        logits[permutation],
        mapping[permutation],
        temperature,
        seed,
        pos[permutation],
    )
    restored = torch.empty_like(whole)
    restored[permutation] = permuted
    assert torch.equal(whole, restored)

    divided = logits / temperature[mapping].unsqueeze(1)
    scaled = _call(
        divided,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
    )
    expected, margins = _cpu_winners(logits, mapping, temperature, seed, pos)
    valid = _int_tensor(np.nonzero(margins > 2e-3)[0])
    assert torch.equal(whole[valid], scaled[valid])
    expected_valid = torch.tensor(expected[valid.cpu().numpy()], dtype=torch.int64)
    assert torch.equal(whole[valid].cpu(), expected_valid)


@pytest.mark.gumbel_sample
def test_shared_compiled_launcher_revalidates_after_warmup():
    _require_shared_compiled_launcher()
    logits, mapping, temperature, seed, pos = _deterministic_inputs(7)
    _call(logits, mapping, temperature, seed, pos)

    logits.requires_grad_()
    with pytest.raises(NotImplementedError, match="inference only"):
        _call(logits, mapping, temperature, seed, pos)
    logits.requires_grad_(False)

    with pytest.raises(NotImplementedError, match="last stride"):
        _call(logits[:, ::2], mapping, temperature, seed, pos)

    resized = logits.clone()
    _call(resized, mapping, temperature, seed, pos)
    resized.resize_(resized.shape[0], 0)
    with pytest.raises(ValueError, match="vocab size"):
        _call(resized, mapping, temperature, seed, pos)

    cache = torch.full((5, 1, 7), -13.0, device=device)
    cache_before = cache.clone()
    with pytest.raises(ValueError, match="exclusive"):
        _call(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            logits_cache=cache,
            logits_cache_col=_int_tensor(0, torch.int32),
            logits_cache_source=logits,
            bias=torch.zeros_like(logits),
        )
    assert torch.equal(cache, cache_before)


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("vocab_size", [1025, 4096, 4097])
def test_shared_compiled_launcher_isolates_pointer_scalar_and_stateful_modes(
    vocab_size,
):
    _require_shared_compiled_launcher()
    warm_logits = _tensor(_lcg_values(4 * vocab_size).reshape(4, vocab_size))
    _call(
        warm_logits,
        _int_tensor([0, 1, 2, 3], torch.int32),
        _tensor([0.25, 0.5, 0.8, 1.0]),
        _int_tensor([11, 13, 17, 19]),
        _int_tensor([3, 5, 7, 11]),
    )

    backing = torch.full((6, vocab_size + 1), -19.0, device=device)
    logits = backing[1:, 1:]
    logits.copy_(_tensor(_lcg_values(5 * vocab_size).reshape(5, vocab_size)))
    assert logits.stride(1) == 1 and logits.data_ptr() % 16 != 0
    mapping = _int_tensor([4, 2, 0, 3, 1], torch.int64)
    temperature = _tensor([0.35, 0.5, 0.8, 1.0, 1.75])
    seed = _int_tensor([101, 103, 107, 109, 113])
    pos = _int_tensor([17, 29, 31, 43, 47])

    no_cache = _call(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
    )
    expected, margins = _cpu_winners(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
    )
    valid = _int_tensor(np.flatnonzero(margins > 2e-3))
    assert valid.numel() == logits.shape[0]
    assert torch.equal(
        no_cache[valid].cpu(), torch.tensor(expected[valid.cpu().numpy()])
    )

    aligned_logits = _tensor(_lcg_values(5 * vocab_size).reshape(5, vocab_size))
    aligned = _call(
        aligned_logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
    )
    assert aligned_logits.data_ptr() % 16 == 0
    assert torch.equal(aligned, no_cache)
    assert aligned.data_ptr() != no_cache.data_ptr()

    replacement_values = np.flip(
        _lcg_values(5 * vocab_size).reshape(5, vocab_size), axis=1
    ).copy()
    replacement_logits = _tensor(replacement_values)
    replacement = _call(
        replacement_logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
    )
    expected_replacement, replacement_margins = _cpu_winners(
        replacement_logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
    )
    valid_replacement = _int_tensor(np.flatnonzero(replacement_margins > 2e-3))
    assert replacement_logits.data_ptr() % 16 == 0
    assert replacement_logits.data_ptr() != aligned_logits.data_ptr()
    assert valid_replacement.numel() == logits.shape[0]
    assert replacement.data_ptr() not in {no_cache.data_ptr(), aligned.data_ptr()}
    assert torch.equal(
        replacement[valid_replacement].cpu(),
        torch.tensor(expected_replacement[valid_replacement.cpu().numpy()]),
    )

    bias = torch.zeros_like(logits)
    bias[:, 0] = 0.125
    biased = _call(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
        bias=bias,
    )
    expected_biased, biased_margins = _cpu_winners(
        logits + bias,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
    )
    valid_biased = _int_tensor(np.flatnonzero(biased_margins > 2e-3))
    assert valid_biased.numel() == logits.shape[0]
    assert torch.equal(
        biased[valid_biased].cpu(),
        torch.tensor(expected_biased[valid_biased.cpu().numpy()]),
    )

    if _is_ascend_triton_351():
        with pytest.raises(NotImplementedError, match="Ascend Triton 3.5.1 SIMD"):
            _call(
                logits,
                mapping,
                temperature,
                seed,
                pos,
                apply_temperature=False,
                is_drafting=True,
                use_fp64=True,
            )
    else:
        fp64 = _call(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            apply_temperature=False,
            is_drafting=True,
            use_fp64=True,
        )
        expected_fp64, fp64_margins = _cpu_winners(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            apply_temperature=False,
            is_drafting=True,
            use_fp64=True,
        )
        valid_fp64 = _int_tensor(np.flatnonzero(fp64_margins > 2e-8))
        assert valid_fp64.numel() == logits.shape[0]
        assert torch.equal(
            fp64[valid_fp64].cpu(),
            torch.tensor(expected_fp64[valid_fp64.cpu().numpy()]),
        )

    cache = torch.full((5, 2, vocab_size + 2), -23.0, device=device)
    cached = _call(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=_int_tensor(1, torch.int32),
    )
    expected_cached, cached_margins = _cpu_winners(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        apply_temperature=False,
        is_drafting=True,
        cache_columns=np.ones(logits.shape[0], dtype=np.int64),
    )
    valid_cached = _int_tensor(np.flatnonzero(cached_margins > 2e-3))
    assert valid_cached.numel() == logits.shape[0]
    assert torch.equal(
        cached[valid_cached].cpu(),
        torch.tensor(expected_cached[valid_cached.cpu().numpy()]),
    )
    expected_cache = torch.full_like(cache, -23.0)
    expected_cache[mapping, 1, :vocab_size] = logits
    assert torch.equal(cache, expected_cache)


@pytest.mark.gumbel_sample
def test_shared_compiled_launcher_uses_callers_current_stream(monkeypatch):
    _require_shared_compiled_launcher()
    module = importlib.import_module("flaggems_vllm.ops.gumbel_sample")
    vocab_size = 4097
    _call(
        _tensor(_lcg_values(4 * vocab_size).reshape(4, vocab_size)),
        _int_tensor([0, 1, 2, 3], torch.int32),
        _tensor([0.0, 0.0, 0.0, 0.0]),
        _int_tensor([0, 1, 2, 3]),
        _int_tensor([0, 1, 2, 3]),
    )

    device_api = flaggems_vllm.runtime.torch_device_fn
    default_stream = device_api.current_stream()
    default_driver_stream = _current_driver_stream()
    events = []

    def record_enter(metadata):
        events.append(("enter", metadata.get()["stream"]))

    def record_exit(metadata):
        events.append(("exit", metadata.get()["stream"]))

    streams = [device_api.Stream(), device_api.Stream(), default_stream]
    observed_streams = []
    for stream_index, stream in enumerate(streams):
        stream.wait_stream(default_stream)
        for cold_plan in (True, False):
            if cold_plan:
                # Keep LibEntry warm so NVIDIA autotuning is outside hook counts.
                module._nocache_plans.clear()
            with device_api.stream(stream):
                expected_stream = _current_driver_stream()
                if cold_plan:
                    observed_streams.append(expected_stream)
                logits = torch.full((4, vocab_size), -float("inf"), device=device)
                shift = stream_index * 2 + int(not cold_plan)
                winners = [
                    (token + shift) % vocab_size for token in (3, 257, 777, 4096)
                ]
                for row, winner in enumerate(winners):
                    logits[row, winner] = 1.0
                mapping = _int_tensor([0, 1, 2, 3], torch.int32)
                temperature = _tensor([0.0, 0.0, 0.0, 0.0])
                seed = _int_tensor([0, 1, 2, 3])
                pos = _int_tensor([0, 1, 2, 3])
                start = len(events)
                with monkeypatch.context() as hooks:
                    hooks.setattr(
                        triton.knobs.runtime, "launch_enter_hook", record_enter
                    )
                    hooks.setattr(triton.knobs.runtime, "launch_exit_hook", record_exit)
                    output = _call(logits, mapping, temperature, seed, pos)
            stream.synchronize()
            assert output.cpu().tolist() == winners
            assert events[start:] == [
                (kind, expected_stream) for kind in ("enter", "exit") * 2
            ]

    assert len(set(observed_streams)) == 3
    assert observed_streams[-1] == default_driver_stream
    before_without_hooks = len(events)
    logits.fill_(-float("inf"))
    logits[:, 19] = 1.0
    output = _call(logits, mapping, temperature, seed, pos)
    assert output.cpu().tolist() == [19] * 4
    assert len(events) == before_without_hooks


def _require_ascend_streaming():
    if not _is_ascend_triton_351():
        pytest.skip("streaming row launch is specific to verified Ascend 3.5.1")
    module = importlib.import_module("flaggems_vllm.ops.gumbel_sample")
    assert module._COMPILED_FAST_LAUNCH
    assert hasattr(module, "_sample_streaming_row")
    return module


@triton.jit
def _candidate_finite_noise_kernel(
    uniform_ptr,
    noise_ptr,
    NUMEL: tl.constexpr,
    FINITE_INPUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.arange(0, BLOCK)
    uniform = tl.load(uniform_ptr + index, index < NUMEL, other=0.5)
    noise = _gumbel_from_uniform(uniform, False, FINITE_INPUT=FINITE_INPUT)
    tl.store(noise_ptr + index, noise, index < NUMEL)


@pytest.mark.gumbel_sample
def test_ascend_finite_noise_endpoints():
    _require_ascend_streaming()
    quarter = np.float32(0.25)
    values = [
        2.0**-33,
        float(np.nextafter(quarter, np.float32(0.0))),
        0.25,
        float(np.nextafter(quarter, np.float32(1.0))),
        0.5,
        float(np.nextafter(np.float32(1.0), np.float32(0.0))),
        1.0,
    ] + [float(cpu_uniform32(*row[:3])) for row in KAT_INPUTS]
    uniform = _tensor(values)
    original = torch.empty_like(uniform)
    finite = torch.empty_like(uniform)
    for finite_input, output in ((False, original), (True, finite)):
        _candidate_finite_noise_kernel[(1,)](
            uniform,
            output,
            len(values),
            finite_input,
            triton.next_power_of_2(len(values)),
        )
    assert torch.equal(original.view(torch.int32), finite.view(torch.int32))
    # The hash-to-FP32 conversion may round to 1; its clamp is part of the contract.
    expected = [_fp32_gumbel(min(value, 1.0 - 2.0**-24)) for value in values]
    np.testing.assert_allclose(finite.cpu().numpy(), expected, rtol=3e-6, atol=3e-6)


def _streaming_sparse_fixture(apply_temperature, is_drafting):
    n, vocab = 65, 32769
    tokens = [0, 2047, 2048, 4095, 4096, 32767, 32768]
    seeds = [0, 17, -1, -9223372036854775807, 0x123456789ABCDEF]
    temperatures = [0.0, 0.2, 0.8, 2.0, 1.0]
    requests = [-1 if row % 13 == 0 else (row * 3) % 5 for row in range(n)]
    positions = [(1 << 32) + row * 8191 - 19 for row in range(n)]
    # Padding is deliberately attractive; invalid lanes must never win.
    storage = torch.full((n, vocab + 4096), float("inf"), dtype=torch.float32)
    logits_cpu = storage[:, :vocab]
    logits_cpu.fill_(-float("inf"))
    expected = []
    for row, request in enumerate(requests):
        values = np.asarray([1.5, -0.3, 0.8, 2.1, 0.5, -2.0, 1.0], np.float32)
        values += np.asarray(
            [((row + col * 3) % 7) * 0.05 for col in range(len(tokens))],
            np.float32,
        )
        logits_cpu[row, tokens] = torch.from_numpy(values)
        scores = values.copy()
        if request >= 0 and temperatures[request] != 0.0:
            if apply_temperature:
                scores = (scores / np.float32(temperatures[request])).astype(np.float32)
            position = positions[row] + ((1 << 30) if is_drafting else 0)
            for column, token in enumerate(tokens):
                scores[column] += _fp32_gumbel(
                    cpu_uniform32(seeds[request], position, token)
                )
        order = np.argsort(-scores, kind="stable")
        # All rows were frozen above the existing stable-margin threshold.
        assert float(scores[order[0]] - scores[order[1]]) > 2e-3
        expected.append(tokens[int(order[0])])
    logits = storage.to(device)[:, :vocab]
    args = (
        logits,
        _strided_vector(requests, torch.int32),
        _strided_vector(temperatures, torch.float32),
        _strided_vector(seeds, torch.int64),
        _strided_vector(positions, torch.int64),
    )
    return args, expected


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("apply_temperature", [False, True])
@pytest.mark.parametrize("is_drafting", [False, True])
def test_ascend_streaming_sparse_global_keys(
    apply_temperature, is_drafting, monkeypatch
):
    module = _require_ascend_streaming()
    args, expected = _streaming_sparse_fixture(apply_temperature, is_drafting)
    calls = []
    original = module._sample_streaming_row

    def record(*call_args, **kwargs):
        calls.append(tuple(call_args[0].shape))
        return original(*call_args, **kwargs)

    monkeypatch.setattr(module, "_sample_streaming_row", record)
    flags = dict(apply_temperature=apply_temperature, is_drafting=is_drafting)
    cpu_rng_before = torch.random.get_rng_state().clone()
    device_rng_before = _device_rng_state()
    first = _call(*args, **flags)
    second = _call(*args, **flags)
    assert torch.equal(cpu_rng_before, torch.random.get_rng_state())
    assert torch.equal(device_rng_before, _device_rng_state())
    assert calls == [(65, 32769), (65, 32769)]
    assert first.cpu().tolist() == expected
    assert second.cpu().tolist() == expected
    assert first.data_ptr() != second.data_ptr()
    # Crossing the N gate must preserve the original global row RNG keys.
    partitioned = []
    for start, end in ((0, 32), (32, 65)):
        part = (
            args[0][start:end],
            args[1][start:end],
            args[2],
            args[3],
            args[4][start:end],
        )
        partitioned.extend(_call(*part, **flags).cpu().tolist())
    assert partitioned == expected
    assert calls == [(65, 32769), (65, 32769)]


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("apply_temperature", [False, True])
def test_ascend_streaming_boundaries(apply_temperature, monkeypatch):
    module = _require_ascend_streaming()
    n, vocab = 65, 32769
    storage = torch.full((n, vocab + 4096), float("inf"), device=device)
    logits = storage[:, :vocab]
    logits.fill_(-float("inf"))
    logits[1, 2048] = logits[1, 32768] = float("inf")
    logits[2, 4096] = logits[2, 32768] = 2.0
    logits[3:, 32768] = 1.0
    expected = [0, 2048, 4096] + [32768] * (n - 3)
    requests = [1] * n
    requests[2] = 0
    requests[4] = -1
    args = (
        logits,
        _int_tensor(requests, torch.int32),
        _tensor([0.0, 0.8]),
        _int_tensor([0, -1]),
        _int_tensor(list(range(n))),
    )
    calls = []
    original = module._sample_streaming_row

    def record(*call_args, **kwargs):
        calls.append(tuple(call_args[0].shape))
        return original(*call_args, **kwargs)

    monkeypatch.setattr(module, "_sample_streaming_row", record)
    for _ in range(2):
        assert (
            _call(*args, apply_temperature=apply_temperature).cpu().tolist() == expected
        )
    assert calls == [(n, vocab)] * 2


@pytest.mark.gumbel_sample
def test_ascend_streaming_uses_callers_current_stream(monkeypatch):
    module = _require_ascend_streaming()
    n, vocab = 65, 32769
    _call(
        torch.zeros((n, vocab), device=device),
        _int_tensor([0] * n, torch.int32),
        _tensor([0.8]),
        _int_tensor([17]),
        _int_tensor(list(range(n))),
        apply_temperature=False,
    )
    device_api = flaggems_vllm.runtime.torch_device_fn
    default_stream = device_api.current_stream()
    default_driver_stream = _current_driver_stream()
    events = []

    def record_enter(metadata):
        events.append(("enter", metadata.get()["stream"]))

    def record_exit(metadata):
        events.append(("exit", metadata.get()["stream"]))

    observed_streams = []
    for stream_index, stream in enumerate(
        [device_api.Stream(), device_api.Stream(), default_stream]
    ):
        stream.wait_stream(default_stream)
        for cold_plan in (True, False):
            if cold_plan:
                module._streaming_plans.clear()
            with device_api.stream(stream):
                expected_stream = _current_driver_stream()
                if cold_plan:
                    observed_streams.append(expected_stream)
                logits = torch.full((n, vocab), -float("inf"), device=device)
                winner = 2047 + stream_index * 2 + int(not cold_plan)
                logits[:, winner] = 1.0
                mapping = _int_tensor([0] * n, torch.int32)
                temperature = _tensor([0.8])
                seed = _int_tensor([17])
                pos = _int_tensor(list(range(n)))
                start = len(events)
                with monkeypatch.context() as hooks:
                    hooks.setattr(
                        triton.knobs.runtime, "launch_enter_hook", record_enter
                    )
                    hooks.setattr(triton.knobs.runtime, "launch_exit_hook", record_exit)
                    output = _call(
                        logits, mapping, temperature, seed, pos, apply_temperature=False
                    )
            stream.synchronize()
            assert output.cpu().tolist() == [winner] * n
            assert events[start:] == [
                ("enter", expected_stream),
                ("exit", expected_stream),
            ]
    assert len(set(observed_streams)) == 3
    assert observed_streams[-1] == default_driver_stream
    before_without_hooks = len(events)
    logits.fill_(-float("inf"))
    logits[:, vocab - 1] = 1.0
    assert (
        _call(logits, mapping, temperature, seed, pos, apply_temperature=False)
        .cpu()
        .tolist()
        == [vocab - 1] * n
    )
    assert len(events) == before_without_hooks


@pytest.mark.gumbel_sample
def test_ascend_streaming_dispatch_boundaries_and_exclusions(monkeypatch):
    module = _require_ascend_streaming()
    calls = []
    original = module._sample_streaming_row

    def record(*call_args, **kwargs):
        calls.append(tuple(call_args[0].shape))
        return original(*call_args, **kwargs)

    monkeypatch.setattr(module, "_sample_streaming_row", record)
    cases = [
        (63, 32768, "plain", False),
        (64, 32767, "plain", False),
        (64, 32768, "plain", True),
        (65, 32769, "plain", True),
        (64, 32768, "fp16", False),
        (64, 32768, "bias", False),
        (64, 32768, "cache", False),
    ]
    for n, vocab, mode, expected_dispatch in cases:
        dtype = torch.float16 if mode == "fp16" else torch.float32
        logits = torch.full((n, vocab), -float("inf"), dtype=dtype, device=device)
        logits[:, 7] = 1.0
        args = (
            logits,
            _int_tensor(list(range(n)), torch.int32),
            _tensor([0.0] * n),
            _int_tensor(list(range(n))),
            _int_tensor(list(range(n))),
        )
        kwargs = {"apply_temperature": False}
        if mode == "bias":
            kwargs["bias"] = torch.zeros_like(logits)
        if mode == "cache":
            kwargs["logits_cache"] = torch.full((n, 1, vocab), 42.0, device=device)
            kwargs["logits_cache_col"] = _int_tensor(0, torch.int32)
        start = len(calls)
        assert _call(*args, **kwargs).cpu().tolist() == [7] * n
        assert calls[start:] == ([(n, vocab)] if expected_dispatch else [])
        if mode == "cache":
            assert torch.equal(kwargs["logits_cache"][:, 0], logits)


@pytest.mark.gumbel_sample
def test_masked_single_valid_and_draft_stream():
    logits, mapping, temperature, seed, pos = _deterministic_inputs(1025)
    logits[:, 1::2] = -float("inf")
    logits[0].fill_(-float("inf"))
    logits[0, 777] = 0.0
    target = _call(logits, mapping, temperature, seed, pos)
    draft = _call(logits, mapping, temperature, seed, pos, is_drafting=True)
    expected_target, target_margin = _cpu_winners(
        logits, mapping, temperature, seed, pos
    )
    expected_draft, draft_margin = _cpu_winners(
        logits, mapping, temperature, seed, pos, is_drafting=True
    )
    assert target[0].item() == 777 and draft[0].item() == 777
    target_valid = target_margin > 2e-3
    draft_valid = draft_margin > 2e-3
    target_indices = _int_tensor(np.flatnonzero(target_valid))
    draft_indices = _int_tensor(np.flatnonzero(draft_valid))
    assert torch.equal(
        target[target_indices].cpu(), torch.tensor(expected_target[target_valid])
    )
    assert torch.equal(
        draft[draft_indices].cpu(), torch.tensor(expected_draft[draft_valid])
    )
    assert torch.equal(
        draft,
        _call(logits, mapping, temperature, seed, pos, is_drafting=True),
    )
    assert torch.all(target[1:] % 2 == 0)
    assert torch.all(draft[1:] % 2 == 0)


@pytest.mark.gumbel_sample
def test_nan_characterization(record_property):
    logits, mapping, temperature, seed, pos = _deterministic_inputs(7)
    logits[0, 1] = float("nan")
    first = _call(logits, mapping, temperature, seed, pos)
    second = _call(logits, mapping, temperature, seed, pos)
    record_property("nan_row_output", int(first[0].item()))
    record_property("nan_row_replay_output", int(second[0].item()))
    record_property("nan_row_replay_equal", bool(first[0].item() == second[0].item()))
    assert torch.equal(first[1:], second[1:])
    _assert_ids(first[1:], 7)


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("dtype", _exact_dtypes())
@pytest.mark.parametrize("per_token_col", [False, True])
def test_cache_pre_temperature_bitwise_and_row_mapping(dtype, per_token_col):
    num_requests, vocab_size, steps = 8, 4099, 3
    logits = _tensor(
        _lcg_values(num_requests * vocab_size).reshape(num_requests, vocab_size), dtype
    )
    mapping = _int_tensor(range(num_requests), torch.int32)
    temperature = _tensor([0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 0.7, 1.0])
    seed = _int_tensor(range(num_requests))
    pos = _int_tensor(range(num_requests))
    cache = torch.zeros(
        (num_requests + 1, steps, vocab_size), dtype=dtype, device=device
    )
    row_mapping = _int_tensor([0, 1, -1, 3, 4, 5, 6, 7], torch.int32)
    columns = (
        _int_tensor([index % steps for index in range(num_requests)], torch.int32)
        if per_token_col
        else _int_tensor(1, torch.int32)
    )
    _call(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=columns,
        logits_cache_row_mapping=row_mapping,
    )
    for row in range(num_requests):
        if row == 2:
            assert not cache[row].any()
            continue
        column = int(columns[row].item()) if per_token_col else 1
        assert torch.equal(_float_bits(cache[row, column]), _float_bits(logits[row]))
    if not per_token_col:
        assert not cache[:, 0].any() and not cache[:, 2].any()
    assert not cache[-1].any()


@pytest.mark.gumbel_sample
def test_cache_column_seed_mixing_with_strided_metadata_and_source():
    num_requests, vocab_size, steps, sentinel = 4, 1031, 4, -17.25
    mapping = _strided_vector([2, 0, 3, 1], torch.int32)
    temperature = _strided_vector([0.5, 1.0, 1.5, 2.0], torch.float32)
    seed = _strided_vector([101, 202, 303, 404], torch.int64)
    pos = _strided_vector([11, 23, 37, 41], torch.int64)
    columns = _strided_vector([3, 1, 0, 2], torch.int32)
    cache_rows = _strided_vector([4, 6, 1, 5], torch.int32)

    source_backing = torch.full(
        (num_requests, vocab_size * 2), sentinel, dtype=torch.float32, device=device
    )
    source = source_backing[:, ::2]
    source.copy_(
        _tensor(
            _lcg_values(num_requests * vocab_size).reshape(num_requests, vocab_size)
        )
    )
    logits = source + 0.25
    cache = torch.full(
        (7, steps, vocab_size + 2), sentinel, dtype=torch.float32, device=device
    )
    expected_cache = cache.clone()
    for row, request in enumerate(mapping.cpu().tolist()):
        cache_row = int(cache_rows[request].item())
        column = int(columns[row].item())
        expected_cache[cache_row, column, :vocab_size] = source[row]

    output = _call(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=columns,
        logits_cache_source=source,
        logits_cache_row_mapping=cache_rows,
    )
    expected, margins = _cpu_winners(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        is_drafting=True,
        cache_columns=columns.cpu().numpy(),
    )
    valid = _int_tensor(np.flatnonzero(margins > 2e-3))
    assert valid.numel() == num_requests
    assert torch.equal(output[valid].cpu(), torch.tensor(expected[valid.cpu().numpy()]))
    assert torch.equal(_float_bits(cache), _float_bits(expected_cache))
    assert torch.all(source_backing[:, 1::2] == sentinel)


@pytest.mark.gumbel_sample
@pytest.mark.parametrize("dtype", _exact_dtypes())
def test_bias_cache_and_sampling_with_input_dtype_rounding(dtype):
    num_requests, vocab_size = 3, 1031
    raw_logits = np.tile(np.linspace(999.0, 1001.0, vocab_size), (num_requests, 1))
    raw_logits += np.arange(num_requests, dtype=np.float32).reshape(-1, 1) * 0.03125
    logits = _tensor(raw_logits, dtype)
    bias = _tensor(np.full_like(raw_logits, 0.3001), dtype)
    mapping = _int_tensor([2, 0, 1], torch.int32)
    temperature = _tensor([0.5, 1.0, 1.5])
    seed = _int_tensor([19, 29, 31])
    pos = _int_tensor([7, 11, 13])
    cache = torch.full((num_requests, 1, vocab_size), -1.0, dtype=dtype, device=device)
    output = _call(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        logits_cache=cache,
        logits_cache_col=_int_tensor(0, torch.int32),
        bias=bias,
    )
    rounded = (logits.float() + bias.float()).to(dtype)
    if dtype != torch.float32:
        assert torch.any(rounded.float() != logits.float() + bias.float())
    expected_cache = torch.full_like(cache, -1.0)
    expected_cache[mapping, 0] = rounded
    assert torch.equal(_float_bits(cache), _float_bits(expected_cache))
    expected, margins = _cpu_winners(rounded, mapping, temperature, seed, pos)
    valid = _int_tensor(np.flatnonzero(margins > 2e-3))
    assert valid.numel() == num_requests
    assert torch.equal(output[valid].cpu(), torch.tensor(expected[valid.cpu().numpy()]))


@pytest.mark.gumbel_sample
def test_invalid_metadata_and_cache_alias_are_rejected():
    logits = _tensor(np.zeros((2, 8), dtype=np.float32))
    mapping = _int_tensor([0, 1], torch.int32)
    temperature = _tensor([1.0, 1.0])
    seed = _int_tensor([0, 1])
    pos = _int_tensor([0, 1])
    cache = torch.zeros((2, 1, 7), device=device)
    with pytest.raises((ValueError, NotImplementedError, AssertionError)):
        _call(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            logits_cache=cache,
            logits_cache_col=_int_tensor(0, torch.int32),
        )
    with pytest.raises((ValueError, NotImplementedError, AssertionError)):
        _call(logits, mapping, temperature, seed, pos, logits_cache_source=logits)
    with pytest.raises((ValueError, NotImplementedError, AssertionError)):
        _call(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            logits_cache=cache,
            logits_cache_col=_int_tensor(0, torch.int32),
            bias=torch.zeros_like(logits),
            logits_cache_source=logits,
        )
    with pytest.raises((ValueError, NotImplementedError, AssertionError)):
        _call(logits[:, ::2], mapping, temperature, seed, pos)
    alias = logits.view(2, 1, 8)
    with pytest.raises((ValueError, NotImplementedError, AssertionError)):
        _call(
            logits,
            mapping,
            temperature,
            seed,
            pos,
            logits_cache=alias,
            logits_cache_col=_int_tensor(0, torch.int32),
        )
    with pytest.raises((ValueError, NotImplementedError, AssertionError)):
        _call(
            torch.empty((1, 0), device=device),
            _int_tensor([0], torch.int32),
            _tensor([1.0]),
            _int_tensor([0]),
            _int_tensor([0]),
        )


@pytest.mark.gumbel_sample
def test_empty_input_and_default_rng_state_are_unchanged():
    empty = torch.empty((0, 7), dtype=torch.float32, device=device)
    output = _call(
        empty,
        _int_tensor([], torch.int32),
        _tensor([1.0]),
        _int_tensor([0]),
        _int_tensor([]),
    )
    assert output.shape == (0,) and output.dtype == torch.int64
    output64 = _call(
        empty,
        _int_tensor([], torch.int32),
        _tensor([1.0]),
        _int_tensor([0]),
        _int_tensor([]),
        use_fp64=True,
    )
    assert output64.shape == (0,) and output64.dtype == torch.int64

    if (
        flaggems_vllm.vendor_name == "iluvatar"
        and triton.__version__.split("+", 1)[0] == "3.1.0"
    ):
        small = _tensor([[0.0, 1.0]])
        cache = torch.full((1, 1, 2), 42.0, dtype=torch.float32, device=device)
        before = cache.clone()
        with pytest.raises(NotImplementedError, match="FP64 log/reduction"):
            _call(
                small,
                _int_tensor([0], torch.int32),
                _tensor([1.0]),
                _int_tensor([0]),
                _int_tensor([0]),
                use_fp64=True,
                logits_cache=cache,
                logits_cache_col=_int_tensor([0], torch.int32),
            )
        assert torch.equal(cache, before)

    logits, mapping, temperature, seed, pos = _deterministic_inputs(7)
    cpu_before = torch.random.get_rng_state().clone()
    device_before = _device_rng_state()
    _call(logits, mapping, temperature, seed, pos)
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    assert torch.equal(device_before, _device_rng_state())


def _histogram(logits_1d, num_samples, temperature=1.0, use_fp64=False, chunk=4096):
    vocab_size = logits_1d.numel()
    histogram = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    mapping = torch.zeros(chunk, dtype=torch.int32, device=device)
    temp = _tensor([temperature])
    seed = _int_tensor([0xABCD])
    for start in range(0, num_samples, chunk):
        size = min(chunk, num_samples - start)
        logits = logits_1d.unsqueeze(0).expand(size, vocab_size)
        output = _call(
            logits,
            mapping[:size],
            temp,
            seed,
            torch.arange(start, start + size, dtype=torch.int64, device=device),
            use_fp64=use_fp64,
        )
        histogram += torch.bincount(output, minlength=vocab_size)
    return histogram


def _assert_chunk_invariant(logits, count, temperature=1.0, use_fp64=False):
    first = _histogram(logits, count, temperature, use_fp64, chunk=4096)
    second = _histogram(logits, count, temperature, use_fp64, chunk=1024)
    assert torch.equal(first, second)
    return first.cpu().numpy()


def _scipy_stats():
    try:
        import scipy.stats
    except ImportError as error:
        pytest.fail(f"SciPy is required by the frozen statistical protocol: {error}")
    return scipy.stats


def _distribution_evidence(
    hypothesis,
    observed,
    expected,
    statistic,
    pvalue,
    *,
    chunk,
    dtype,
    use_fp64,
    intervals=None,
):
    return {
        "hypothesis": hypothesis,
        "n": int(np.asarray(observed).sum()),
        "observed": np.asarray(observed).tolist(),
        "expected": np.asarray(expected).tolist(),
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "chunk": chunk,
        "dtype": str(dtype),
        "use_fp64": use_fp64,
        "vendor": flaggems_vllm.runtime.device.vendor_name,
        "device": str(device),
        "intervals": intervals,
    }


def _chi_square(
    observed,
    expected,
    hypothesis,
    record_property,
    *,
    dtype=torch.float32,
    use_fp64=False,
    intervals=None,
):
    scipy_stats = _scipy_stats()
    statistic, pvalue = scipy_stats.chisquare(observed, expected)
    evidence = _distribution_evidence(
        hypothesis,
        observed,
        expected,
        statistic,
        pvalue,
        chunk=4096,
        dtype=dtype,
        use_fp64=use_fp64,
        intervals=intervals,
    )
    record_property(f"{hypothesis}_evidence", json.dumps(evidence))
    assert pvalue > STAT_ALPHA, json.dumps(evidence)
    return statistic, pvalue


def _weights():
    return np.asarray(
        [1 + ((37 * index + 17) % 127) for index in range(257)], dtype=np.float64
    )


def _quantized_probabilities(logits, temperature=1.0):
    quantized = logits.detach().cpu().numpy().astype(np.float64)
    probability = np.exp(quantized / temperature)
    return probability / probability.sum()


@pytest.mark.gumbel_sample
def test_stat_h1_flat_distribution(record_property):
    logits = torch.zeros(257, dtype=torch.float32, device=device)
    observed = _assert_chunk_invariant(logits, 200_000)
    expected = np.full(257, 200_000 / 257.0)
    statistic, pvalue = _chi_square(observed, expected, "H1", record_property)
    record_property("H1_statistic", statistic)
    record_property("H1_pvalue", pvalue)


@pytest.mark.gumbel_sample
def test_stat_h2_nonuniform_distribution(record_property):
    weights = _weights()
    logits = _tensor(np.log(weights).astype(np.float32))
    probability = _quantized_probabilities(logits)
    observed = _assert_chunk_invariant(logits, 300_000)
    expected = probability * 300_000
    statistic, pvalue = _chi_square(observed, expected, "H2", record_property)
    record_property("H2_statistic", statistic)
    record_property("H2_pvalue", pvalue)


def _merged_low_temperature_bins(probabilities, num_samples):
    intervals, start, accumulated = [], 0, 0.0
    for index, probability in enumerate(probabilities):
        accumulated += num_samples * probability
        if accumulated >= 20.0 and index + 1 < len(probabilities):
            intervals.append((start, index + 1))
            start, accumulated = index + 1, 0.0
    intervals.append((start, len(probabilities)))
    if len(intervals) > 1:
        last_start, last_end = intervals[-1]
        if probabilities[last_start:last_end].sum() * num_samples < 20.0:
            previous_start, _ = intervals[-2]
            intervals[-2:] = [(previous_start, last_end)]
    assert len(intervals) > 1
    return intervals


@pytest.mark.gumbel_sample
def test_stat_h3_low_temperature_merged_distribution(record_property):
    weights = _weights()
    logits = _tensor(np.log(weights).astype(np.float32))
    probability = _quantized_probabilities(logits, temperature=0.35)
    intervals = _merged_low_temperature_bins(probability, 300_000)
    observed = _assert_chunk_invariant(logits, 300_000, temperature=0.35)
    observed_bins = np.asarray([observed[start:end].sum() for start, end in intervals])
    expected_bins = np.asarray(
        [probability[start:end].sum() * 300_000 for start, end in intervals]
    )
    assert expected_bins.min() >= 20.0
    statistic, pvalue = _chi_square(
        observed_bins,
        expected_bins,
        "H3",
        record_property,
        intervals=intervals,
    )
    record_property("H3_intervals", json.dumps(intervals))
    record_property("H3_min_expected", float(expected_bins.min()))
    record_property("H3_statistic", statistic)
    record_property("H3_pvalue", pvalue)


@pytest.mark.gumbel_sample
def test_stat_h4_gap18_aggregate_tail_distribution(record_property):
    counts = np.empty(200_000, dtype=np.int64)
    counts[0] = round(math.exp(18.0))
    state = 1234
    for index in range(1, counts.size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        counts[index] = 1 + state % 3
    logits = _tensor(np.log(counts).astype(np.float32))
    observed = _assert_chunk_invariant(logits, 100_000)
    scipy_stats = _scipy_stats()
    tail_count = int(observed[1:].sum())
    tail_probability = float(counts[1:].sum() / counts.sum())
    pvalue = scipy_stats.binomtest(tail_count, 100_000, tail_probability).pvalue
    evidence = _distribution_evidence(
        "H4",
        [100_000 - tail_count, tail_count],
        [100_000 * (1.0 - tail_probability), 100_000 * tail_probability],
        tail_count,
        pvalue,
        chunk=4096,
        dtype=logits.dtype,
        use_fp64=False,
    )
    record_property("H4_evidence", json.dumps(evidence))
    assert pvalue > STAT_ALPHA, json.dumps(evidence)
    record_property("H4_tail_count", tail_count)
    record_property("H4_tail_expected", 100_000 * tail_probability)
    record_property("H4_pvalue", pvalue)


@pytest.mark.gumbel_sample
def test_stat_h5_fp64_nonuniform_distribution(record_property):
    _fp64_probe()
    weights = _weights()
    logits = _tensor(np.log(weights).astype(np.float32))
    probability = _quantized_probabilities(logits)
    observed = _assert_chunk_invariant(logits, 300_000, use_fp64=True)
    expected = probability * 300_000
    statistic, pvalue = _chi_square(
        observed,
        expected,
        "H5",
        record_property,
        use_fp64=True,
    )
    record_property("H5_statistic", statistic)
    record_property("H5_pvalue", pvalue)
