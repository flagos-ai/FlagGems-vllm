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
    # n (num_experts)
    # ---- production shapes (kernel_details) ----
    18,  # 288 routed experts / EP16 (production per-rank count)
    288,  # unsharded (EP1) layout of the same expert count
    # ---- stress shapes ----
    1,  # single expert
    2,  # smallest nonzero tile
    17,  # one past the production count 18
    19,  # non-pow2 near production count
    32,  # pow2 boundary
    36,  # 288/8 (TP8-style shard)
    96,  # 288/3 (TP3-style shard)
    576,  # 2x unsharded
)
CASES = (1, 18, 288) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized group-list cumsum targets Ascend",
)


def _official(group_list):
    # torch.cumsum on integer dtypes dispatches to AI_CPU on Ascend (~82us
    # at the EP int64[18] shape); this is the exact replaced baseline.
    return group_list.cumsum(dim=0)


# Shape provenance: n=18 is the exact production per-rank expert count
# (288 routed experts / EP16); 288 covers the unsharded layout; the rest are
# power-of-two boundaries and tails.


@pytest.mark.group_list_cumsum
@pytest.mark.parametrize("n", CASES)
@pytest.mark.parametrize("dtype", (torch.int64, torch.int32))
def test_group_list_cumsum_accuracy(n, dtype):
    torch.manual_seed(n)
    x = torch.randint(0, 4096, (n,), dtype=torch.int64, device="npu").to(dtype)
    expected = _official(x).to(dtype)  # torch promotes int32 -> int64
    actual = flaggems_vllm.group_list_cumsum(x)
    torch.npu.synchronize()
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual, expected, check_dtype=True)


@pytest.mark.group_list_cumsum
def test_group_list_cumsum_zero_and_saturating():
    x0 = torch.zeros(18, dtype=torch.int64, device="npu")
    torch.testing.assert_close(
        flaggems_vllm.group_list_cumsum(x0), _official(x0), check_dtype=True
    )
    xm = torch.full((18,), 65536 // 18 + 1, dtype=torch.int64, device="npu")
    torch.testing.assert_close(
        flaggems_vllm.group_list_cumsum(xm), _official(xm), check_dtype=True
    )
