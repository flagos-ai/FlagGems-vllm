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


def _reference(cache, kv, rope, slots):
    expected = cache.cpu().clone()
    flat = expected.reshape(-1, expected.shape[-1])
    rows = torch.cat((kv.cpu(), rope.cpu()), dim=-1)
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            flat[slot] = rows[row]
    return expected


def _inputs(dtype, slot_dtype, rope_width, singleton_head):
    device = flaggems_vllm.device
    # Offset and larger row strides match packed projection views.
    kv = torch.randn(33, 1024, dtype=dtype, device=device)[1:, 128:640]
    rope = torch.randn(33, 128, dtype=dtype, device=device)[1:, 8 : 8 + rope_width]
    slots = torch.tensor(
        [31, -1, 8, 0, 5, 18, -1, 23, 2, 40, 7, 11, 9, 14, 39, 4, 1],
        dtype=slot_dtype,
        device=device,
    )
    cache = torch.full((8, 8, 512 + rope_width), -17, dtype=dtype, device=device)
    if singleton_head:
        cache = cache.unsqueeze(2)
    return kv, rope, cache, slots, torch.ones((), device=device)


@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rope_width", [0, 64])
@pytest.mark.parametrize("singleton_head", [False, True])
@torch.inference_mode()
def test_graph_padded_rows(dtype, slot_dtype, rope_width, singleton_head):
    kv, rope, cache, slots, scale = _inputs(
        dtype, slot_dtype, rope_width, singleton_head
    )
    expected = _reference(cache, kv, rope, slots)
    flaggems_vllm.concat_and_cache_mla(kv, rope, cache, slots, "auto", scale)
    torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)


@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_empty_slots_with_padded_rows(dtype):
    kv, rope, cache, slots, scale = _inputs(dtype, torch.int64, 64, False)
    before = cache.cpu().clone()
    flaggems_vllm.concat_and_cache_mla(kv, rope, cache, slots[:0], "auto", scale)
    torch.testing.assert_close(cache.cpu(), before, rtol=0, atol=0)


@pytest.mark.concat_and_cache_mla
@pytest.mark.parametrize("invalid", ["too_many_slots", "rope_rows", "matrix_slots"])
@torch.inference_mode()
def test_invalid_padded_row_contract(invalid):
    kv, rope, cache, slots, scale = _inputs(torch.bfloat16, torch.int64, 64, False)
    if invalid == "too_many_slots":
        kv, rope = kv[:8], rope[:8]
    elif invalid == "rope_rows":
        rope = rope[:-1]
    else:
        slots = slots.unsqueeze(0)
    before = cache.cpu().clone()
    with pytest.raises(ValueError, match="one input row per slot"):
        flaggems_vllm.concat_and_cache_mla(kv, rope, cache, slots, "auto", scale)
    torch.testing.assert_close(cache.cpu(), before, rtol=0, atol=0)


@pytest.mark.concat_and_cache_mla
@torch.inference_mode()
def test_padded_rows_graph_replay_changed_inputs():
    kv, rope, cache, slots, scale = _inputs(torch.bfloat16, torch.int64, 64, True)
    # Prewarm before capture, then reset sentinel independently on every replay.
    flaggems_vllm.concat_and_cache_mla(kv, rope, cache, slots, "auto", scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.concat_and_cache_mla(kv, rope, cache, slots, "auto", scale)
    for step in range(2):
        kv.add_(step + 1)
        rope.sub_(step + 1)
        new_slots = slots.clone()
        valid = new_slots >= 0
        new_slots[valid] = (new_slots[valid] + 7) % 64
        slots.copy_(new_slots)
        cache.fill_(-17)
        expected = _reference(cache, kv, rope, slots)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
