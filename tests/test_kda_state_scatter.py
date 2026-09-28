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
# Shape provenance (production): H=4 = 64 linear-attn heads / TP16, V=K=128
# = head_dim (linear_attn_config); S=8256 and N in {4,8,12,16} per the
# kernel_details Index records, S shrunk here for footprint.
# Stress variants: (256,4,16,...) exercises a deeper H grid axis;
# (512,8,4,64,32) drives BK=next_pow2(32)<64 so the launcher takes the
# BV=next_pow2(V) tiling branch instead of the production BV=64.
FULL_CASES = (
    # S slots, N seqs, H heads, V, K (head dims)
    # ---- production shapes (kernel_details) ----
    (8256, 4, 4, 128, 128),  # (S,N,H,V,K) 4 seqs, production pool/layout
    (8256, 8, 4, 128, 128),  # 8 seqs, recorded mid point
    (8256, 12, 4, 128, 128),  # 12 seqs, recorded mid point
    (8256, 16, 4, 128, 128),  # 16 seqs, recorded upper end
    # ---- stress shapes ----
    (8256, 32, 4, 128, 128),  # stress: 32 seqs, max_num_seqs boundary
    (8256, 64, 4, 128, 128),  # stress: 64 seqs, extreme depth
    (16384, 16, 4, 128, 128),  # stress: doubled slot pool
    (8256, 16, 8, 128, 128),  # stress: H=8, deeper grid axis
    (8256, 16, 4, 64, 32),  # stress: K=32 -> BK<64 tiling branch
    (8256, 1, 4, 128, 128),  # stress: single sequence
)
CASES = (FULL_CASES[0], FULL_CASES[1]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized KDA state scatter targets Ascend",
)


def _ref_scatter(cache, state_indices, values):
    """Official chain, lifted verbatim from kimi_kda.py's non-triton branch
    (transpose -> cast -> index_put)."""
    out = cache.clone()
    out[state_indices] = values.transpose(-1, -2).contiguous().to(out.dtype)
    return out


@pytest.mark.kda_state_scatter
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_kda_state_scatter_accuracy(case, dtype):
    S, N, H, V, K = case
    torch.manual_seed(1)
    cache = torch.randn(S, H, V, K, dtype=torch.float32, device="npu").to(dtype)
    state_indices = torch.randperm(S, device="cpu")[:N].to("npu")
    values = torch.randn(N, H, K, V, dtype=torch.float32, device="npu").to(dtype)

    ref_cache = _ref_scatter(cache, state_indices, values)
    new_cache = cache.clone()
    flaggems_vllm.scatter_kda_state(new_cache, state_indices, values)
    torch.npu.synchronize()
    assert torch.equal(new_cache, ref_cache)


@pytest.mark.kda_state_scatter
def test_kda_state_roundtrip_bitwise():
    """Gather and scatter transposes must be each other's inverse.

    gather emits the AscendC layout [N, H, K, V] (has_initial_state=0 rows
    zeroed); writing that back through scatter must land bit-for-bit where
    the equivalent torch advanced-indexing write lands.
    """
    S, N, H, V, K = 8256, 16, 4, 128, 128  # production layout
    torch.manual_seed(2)
    cache = torch.randn(S, H, V, K, dtype=torch.float32, device="npu")
    idx = torch.randperm(S, device="cpu")[:N].to("npu")
    has = (torch.rand(N) > 0.3).to(torch.int32).to("npu")

    gathered = flaggems_vllm.gather_kda_state(cache, idx, has)
    out = cache.clone()
    flaggems_vllm.scatter_kda_state(out, idx, gathered)

    ref_out = cache.clone()
    ref_out[idx] = gathered.transpose(-1, -2).contiguous()
    torch.npu.synchronize()
    assert torch.equal(out, ref_out)
