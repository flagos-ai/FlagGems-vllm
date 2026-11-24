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
from baseline.hc_collapse import all_cases, build_inputs

pytestmark = [
    pytest.mark.hc_collapse,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "ascend", reason="Ascend implementation"
    ),
]


@pytest.mark.parametrize("binding", all_cases())
def test_hc_collapse(binding):
    torch.manual_seed(0)
    (x, p), _ = build_inputs(binding, torch.bfloat16, flaggems_vllm.device)
    expected = (p.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
    actual = flaggems_vllm.hc_collapse(x, p)
    assert actual.shape == expected.shape and actual.dtype == x.dtype
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.01)
    assert torch.isfinite(actual).all().item()
    if x.shape[0]:
        torch.testing.assert_close(actual[0], x[0, -1], atol=0, rtol=0)
    if x.shape[0] > 1 and x.shape[1] == 4:
        torch.testing.assert_close(
            actual[1], torch.zeros_like(actual[1]), atol=0, rtol=0
        )


@pytest.mark.parametrize("shape", [(2, 0, 65), (2, 4, 0), (1, 1, 1), (3, 3, 17)])
def test_hc_collapse_extra_edges(shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device=flaggems_vllm.device)
    p = torch.rand(shape[:2], dtype=torch.float32, device=x.device)
    expected = (p[..., None] * x.float()).sum(1).to(x.dtype)
    torch.testing.assert_close(
        flaggems_vllm.hc_collapse(x, p), expected, atol=0.016, rtol=0.01
    )


def test_hc_collapse_invalid_inputs():
    x = torch.empty((2, 4, 65), dtype=torch.bfloat16, device=flaggems_vllm.device)
    with pytest.raises(ValueError):
        flaggems_vllm.hc_collapse(x, torch.empty((2, 3), device=x.device))
    with pytest.raises(NotImplementedError):
        flaggems_vllm.hc_collapse(
            x, torch.empty((2, 4), dtype=torch.bfloat16, device=x.device)
        )


def test_hc_collapse_special_values():
    x = torch.zeros((1, 4, 65), dtype=torch.bfloat16, device=flaggems_vllm.device)
    p = torch.ones((1, 4), dtype=torch.float32, device=x.device)
    x[0, 0, 0] = float("nan")
    x[0, 0, 1] = float("inf")
    x[0, 0, 2] = -float("inf")
    expected = (p[..., None] * x.float()).sum(1).to(x.dtype)
    torch.testing.assert_close(
        flaggems_vllm.hc_collapse(x, p), expected, equal_nan=True
    )
