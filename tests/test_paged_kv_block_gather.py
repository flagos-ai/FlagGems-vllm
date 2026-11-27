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

device = flaggems_vllm.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 16, 64, 8), (8, 4, 17, 5), (2, 32, 3, 1)])
@pytest.mark.parametrize("strided", [False, True])
def test_paged_kv_block_gather(dtype, shape, strided):
    blocks, block_size, head_dim, count = shape
    source = torch.arange(
        blocks * block_size * head_dim, dtype=torch.float32, device=device
    ).reshape(blocks, block_size, head_dim)
    cache = source.to(dtype)
    if strided:
        cache = torch.stack((cache, cache), dim=-1)[..., 0]
    table = torch.randint(0, blocks, (3, count), dtype=torch.int64, device=device)
    if strided:
        table = torch.stack((table, table), dim=-1)[..., 0]
    actual = flaggems_vllm.paged_kv_block_gather(cache, table)
    expected = cache[table.long()]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("table_shape", [(0, 4), (3, 0)])
def test_paged_kv_block_gather_empty(table_shape):
    cache = torch.empty((4, 8, 16), dtype=torch.float16, device=device)
    table = torch.empty(table_shape, dtype=torch.int32, device=device)
    actual = flaggems_vllm.paged_kv_block_gather(cache, table)
    assert actual.shape == (*table_shape, 8, 16)


def test_paged_kv_block_gather_invalid():
    cache = torch.arange(32, dtype=torch.float16, device=device).view(2, 4, 4)
    table = torch.tensor([[0, -1, 1, 8]], dtype=torch.int32, device=device)
    expected = torch.zeros((1, 4, 4, 4), dtype=cache.dtype, device=device)
    expected[0, 0] = cache[0]
    expected[0, 2] = cache[1]
    torch.testing.assert_close(
        flaggems_vllm.paged_kv_block_gather(cache, table), expected, rtol=0, atol=0
    )


def test_paged_kv_block_gather_rejects_bad_layout():
    cache = torch.empty((4, 8, 16), device=device)
    with pytest.raises(ValueError, match="int32 or int64"):
        flaggems_vllm.paged_kv_block_gather(
            cache, torch.empty((2, 2), dtype=torch.float32, device=device)
        )
