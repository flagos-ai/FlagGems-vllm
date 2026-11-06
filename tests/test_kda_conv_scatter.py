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

PAD_SLOT_ID = -1

# (S, N, state_len, conv_dim): (6, 1536) is the exact production layout
# (kimi_kda_state_shape: conv_kernel 4 - 1 + num_spec 3 = 6 rows,
# 3 * 64 heads * 128 / TP16 = 1536 cols); the last case is an edge shape.
# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
FULL_CASES = (
    # S slots, N seqs, W state_len, C conv_dim
    # ---- production shapes (kernel_details) ----
    (8256, 4, 6, 1536),  # (S,N,W,C) 4 seqs, production 6x1536 layout
    (8256, 8, 6, 1536),  # 8 seqs, recorded mid point
    (8256, 12, 6, 1536),  # 12 seqs, recorded mid point
    (8256, 16, 6, 1536),  # 16 seqs, recorded upper end
    # ---- stress shapes ----
    (8256, 32, 6, 1536),  # stress: 32 seqs, max_num_seqs boundary
    (16384, 16, 6, 1536),  # stress: doubled slot pool
    (8256, 16, 6, 3072),  # stress: conv_dim=3072 (TP8 sharding)
    (8256, 64, 6, 1536),  # stress: 64 seqs, extreme depth
    (8256, 1, 6, 1536),  # stress: single sequence
    (2048, 128, 6, 1536),  # 128 seqs (max_num_seqs boundary)
)
CASES = (FULL_CASES[0], FULL_CASES[1]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized KDA conv scatter targets Ascend",
)


def _make_inputs(case, seed=0, dtype=torch.bfloat16, pad_ratio=0.25):
    S, N, W, C = case
    torch.manual_seed(seed)
    conv_state = torch.randn(S, W, C, dtype=torch.float32, device="npu").to(dtype)
    idx = torch.arange(N, dtype=torch.int32, device="npu")
    n_pad = max(1, int(N * pad_ratio))
    idx[torch.randperm(N, device="cpu")[:n_pad].to("npu")] = PAD_SLOT_ID
    return conv_state, idx.view(1, N)


def _ref_scatter(conv_state, staged, flat):
    """Official sentinel chain, lifted from kimi_kda.py's non-triton restore:
    arange/masked_fill -> where/zeros_like/sum row-0 guard -> index_copy_."""
    valid = flat != PAD_SLOT_ID
    safe = flat.masked_fill(~valid, 0).to(torch.long)
    mask_shape = (valid.numel(),) + (1,) * (staged.ndim - 1)
    valid_zero = valid & (flat == 0)
    uz = torch.where(valid_zero.view(mask_shape), staged, torch.zeros_like(staged)).sum(
        dim=0
    )
    ez = torch.where(valid_zero.any(), uz, conv_state[0])
    restore = torch.where(valid.view(mask_shape), staged, ez.unsqueeze(0))
    conv_state.index_copy_(0, safe, restore)


@pytest.mark.kda_conv_scatter
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_conv_scatter_accuracy(case, dtype):
    conv_state, idx = _make_inputs(case, seed=1, dtype=dtype)
    staged = torch.randn_like(conv_state[: idx.numel()]).to(dtype)

    ref_state = conv_state.clone()
    _ref_scatter(ref_state, staged, idx.flatten())

    new_state = conv_state.clone()
    flaggems_vllm.scatter_conv_state(new_state, idx.flatten(), staged)
    torch.npu.synchronize()
    assert torch.equal(new_state, ref_state)
