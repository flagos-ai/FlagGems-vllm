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
    """Official sentinel chain, lifted verbatim from the pre-triton path in
    indexer_kpool_mla_v1.py / glm5_next.py (where/div/mod -> scatter_nd with
    the row-0 zero-state guard)."""
    if block_size == 1:
        valid = (slots >= 0) & (slots < cache.shape[0])
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        row_zero = cache[0].clone()
        safe_values = torch.where(valid.view(-1, 1), values, row_zero.unsqueeze(0))
        rzm = valid & (slots == 0)
        uz = torch.where(rzm.view(-1, 1), values, torch.zeros_like(values)).sum(dim=0)
        ez = torch.where(rzm.any(), uz, row_zero)
        out = cache.clone()
        out[safe] = safe_values
        out[0].copy_(ez)
        return out
    values = values.reshape(values.shape[0], *cache.shape[2:])
    valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
    safe = torch.where(valid, slots, torch.zeros_like(slots))
    block_ids = torch.div(safe, block_size, rounding_mode="floor")
    offsets = torch.remainder(safe, block_size)
    row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
    row_zero = cache[0, 0].clone()
    safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
    rzm = valid & (slots == 0)
    uz = torch.where(
        rzm.view(-1, *([1] * (values.ndim - 1))), values, torch.zeros_like(values)
    ).sum(dim=0)
    ez = torch.where(rzm.any(), uz, row_zero)
    out = cache.clone()
    out[block_ids, offsets] = safe_values
    out[0, 0].copy_(ez)
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


@pytest.mark.paged_scatter
@pytest.mark.parametrize("cache_shape,n,block_size", CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_paged_scatter_accuracy(cache_shape, n, block_size, dtype):
    cache, slots, values = _make_inputs(cache_shape, n, block_size, dtype)
    ref = _ref_scatter(cache, slots, values, block_size)
    new = cache.clone()
    assert flaggems_vllm.paged_scatter_triton(new, slots, values, block_size)
    torch.npu.synchronize()
    assert torch.equal(new, ref)


@pytest.mark.paged_scatter
def test_paged_scatter_all_invalid_is_noop():
    cache = torch.randn(64, 4, 256, dtype=torch.bfloat16, device="npu")
    before = cache.clone()
    slots = torch.full((128,), -1, dtype=torch.int64, device="npu")
    values = torch.randn(128, 256, dtype=torch.bfloat16, device="npu")
    assert flaggems_vllm.paged_scatter_triton(cache, slots, values, 4)
    torch.npu.synchronize()
    assert torch.equal(cache, before)
