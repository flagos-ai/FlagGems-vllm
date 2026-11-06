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
    (8256, 16, 12, 1536),  # stress: doubled state length (W=12)
)
CASES = (FULL_CASES[0], FULL_CASES[1]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized KDA conv gather targets Ascend",
)


def _make_inputs(case, seed=0, dtype=torch.bfloat16, pad_ratio=0.25):
    S, N, W, C = case
    torch.manual_seed(seed)
    conv_state = torch.randn(S, W, C, dtype=torch.float32, device="npu").to(dtype)
    idx = torch.arange(N, dtype=torch.int32, device="npu")
    n_pad = max(1, int(N * pad_ratio))
    idx[torch.randperm(N, device="cpu")[:n_pad].to("npu")] = PAD_SLOT_ID
    return conv_state, idx.view(1, N)


def _ref_gather(conv_state, cache_indices):
    """Official chain, lifted verbatim from kimi_kda.py's non-triton branch:
    index_select -> contiguous -> arange -> masked_fill."""
    flat = cache_indices.flatten()
    valid = flat != PAD_SLOT_ID
    safe = flat.masked_fill(~valid, 0).to(torch.long)
    staged = conv_state.index_select(0, safe).contiguous()
    local = torch.arange(flat.numel(), dtype=cache_indices.dtype, device=flat.device)
    local = local.masked_fill(~valid, PAD_SLOT_ID)
    return staged, local.view_as(cache_indices)


# Shape provenance: (state_len=6, conv_dim=1536) is the exact production
# conv-state layout (conv kernel 4, MTP spec 3, 3*64*128/TP16); S/N are
# reduced batch dimensions; the last case is an edge shape.


@pytest.mark.kda_conv_gather
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_conv_gather_accuracy(case, dtype):
    conv_state, idx = _make_inputs(case, dtype=dtype)
    ref_staged, ref_local = _ref_gather(conv_state, idx)
    staged, local = flaggems_vllm.gather_conv_state(conv_state, idx)
    torch.npu.synchronize()
    torch.testing.assert_close(staged, ref_staged, rtol=0, atol=0)
    assert torch.equal(local, ref_local)
