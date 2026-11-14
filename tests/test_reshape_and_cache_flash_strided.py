# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from functools import partial

import pytest
import torch

import flaggems_vllm

pytestmark = pytest.mark.skipif(
    flaggems_vllm.device != "ptpu",
    reason="The strided Flash cache writer targets Sunrise PTPU",
)

_NUM_BLOCKS, _BLOCK_SIZE, _HEADS, _HEAD_DIM = 2, 16, 2, 24


@pytest.fixture
def device():
    device = torch.device("ptpu:0")
    torch.ptpu.set_device(device)
    return device


def make_cache(device, layout, dtype):
    if layout == "NHD":
        logical = torch.zeros(
            _NUM_BLOCKS, _BLOCK_SIZE, _HEADS, 2 * _HEAD_DIM, device=device, dtype=dtype
        )
    else:
        logical = torch.zeros(
            _NUM_BLOCKS, _HEADS, _BLOCK_SIZE, 2 * _HEAD_DIM, device=device, dtype=dtype
        ).transpose(1, 2)
    return logical, logical[..., :_HEAD_DIM], logical[..., _HEAD_DIM:]


def expected_cache(key, value, slots):
    expected = torch.zeros(
        _NUM_BLOCKS, _BLOCK_SIZE, _HEADS, 2 * _HEAD_DIM, dtype=key.dtype
    )
    for token, slot in enumerate(slots):
        if slot >= 0:
            block, offset = divmod(slot, _BLOCK_SIZE)
            expected[block, offset, :, :_HEAD_DIM] = key[token]
            expected[block, offset, :, _HEAD_DIM:] = value[token]
    return expected


@pytest.mark.reshape_and_cache_flash_strided
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_combined_cache_writes_exact_slots(device, layout):
    dtype = torch.bfloat16
    key = torch.arange(3 * _HEADS * _HEAD_DIM, dtype=dtype).reshape(
        3, _HEADS, _HEAD_DIM
    )
    value_storage = torch.empty(3, _HEAD_DIM, _HEADS, device=device, dtype=dtype)
    value = value_storage.transpose(1, 2)
    value.copy_(key + 100)
    cache, key_cache, value_cache = make_cache(device, layout, dtype)
    slots = [3, -1, 17]
    flaggems_vllm.reshape_and_cache_flash_strided(
        key.to(device),
        value,
        key_cache,
        value_cache,
        torch.tensor(slots, device=device, dtype=torch.int64),
    )
    torch.ptpu.synchronize()
    torch.testing.assert_close(
        cache.cpu(), expected_cache(key, key + 100, slots), rtol=0, atol=0
    )


@pytest.mark.reshape_and_cache_flash_strided
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_graph_replay_uses_current_inputs_and_slots(device, layout):
    runtime = torch.ptpu
    dtype = torch.bfloat16
    cache, key_cache, value_cache = make_cache(device, layout, dtype)
    key = torch.ones(2, _HEADS, _HEAD_DIM, device=device, dtype=dtype)
    value = torch.ones_like(key)
    slots = torch.tensor([3, -1], device=device, dtype=torch.int64)
    write = partial(
        flaggems_vllm.reshape_and_cache_flash_strided,
        key,
        value,
        key_cache,
        value_cache,
        slots,
    )
    stream = runtime.Stream(device=device)
    stream.wait_stream(runtime.current_stream(device))
    with runtime.stream(stream):
        write()
    stream.synchronize()
    graph = runtime.PTPUGraph()
    with runtime.graph(graph, stream=stream):
        write()
    runtime.synchronize()

    for step, mapping in enumerate(([3, -1], [17, 9], [-1, 30]), start=1):
        cache.zero_()
        key.fill_(step)
        value.fill_(step + 10)
        slots.copy_(torch.tensor(mapping, dtype=slots.dtype, device=device))
        runtime.synchronize()
        with runtime.stream(stream):
            graph.replay()
        runtime.synchronize()
        expected_key = torch.full(key.shape, step, dtype=dtype)
        expected_value = torch.full(value.shape, step + 10, dtype=dtype)
        torch.testing.assert_close(
            cache.cpu(),
            expected_cache(expected_key, expected_value, mapping),
            rtol=0,
            atol=0,
        )


@pytest.mark.reshape_and_cache_flash_strided
def test_empty_slot_mapping_preserves_cache(device):
    cache, key_cache, value_cache = make_cache(device, "HND", torch.bfloat16)
    key = torch.zeros(0, _HEADS, _HEAD_DIM, device=device, dtype=cache.dtype)
    slots = torch.empty(0, device=device, dtype=torch.int32)
    flaggems_vllm.reshape_and_cache_flash_strided(
        key, key, key_cache, value_cache, slots
    )
    assert torch.count_nonzero(cache.cpu()) == 0
