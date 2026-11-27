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

import itertools

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils

M_VALUES = [1, 33, 64, 222]
TOP_KS = [2, 6]
K_VALUES = [128, 511, 1024]
MOE_SHAPES = list(itertools.product(M_VALUES, TOP_KS, K_VALUES))


@pytest.mark.moe_sum
@pytest.mark.parametrize("shape", MOE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_moe_sum(shape, dtype):
    m, topk, k = shape
    inp1 = torch.randn((m, topk, k), dtype=dtype, device=flaggems_vllm.device)
    res_out = torch.empty((m, k), dtype=dtype, device=flaggems_vllm.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_out = torch.sum(ref_inp1, dim=1)

    flaggems_vllm.moe_sum(inp1, res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.moe_sum
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_moe_sum_excludes_invalid_routes(mapped, index_dtype):
    if flaggems_vllm.vendor_name != "nvidia":
        pytest.skip("CUDA routed reduction test")
    device = flaggems_vllm.device
    values = torch.ones((2, 4, 13), dtype=torch.bfloat16, device=device)
    output = torch.empty((2, 13), dtype=values.dtype, device=device)
    ids = torch.empty((2, 8), dtype=index_dtype, device=device)[:, ::2]
    expert_map = torch.tensor([-1, 1, 0], dtype=torch.int32, device=device)
    active_map = expert_map if mapped else None

    def prepare(state):
        rows = ((-1, 0, 1, 2), (-1, -1, -1, -1))
        if state:
            rows = ((-1, -1, -1, -1), (2, 1, -1, 0))
        ids.copy_(torch.tensor(rows, dtype=index_dtype, device=device))
        valid = ids >= 0
        if mapped:
            valid &= ids != 0
        values.copy_(torch.where(valid[..., None], 2.0, float("nan")))
        output.fill_(float("nan"))
        return (2 * valid.sum(-1, keepdim=True)).expand_as(output).to(output.dtype)

    expected = prepare(False)
    flaggems_vllm.moe_sum(values, output, ids, active_map)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    if (
        hasattr(torch.ops._moe_C, "moe_sum")
        and len(torch.ops._moe_C.moe_sum.default._schema.arguments) == 4
    ):
        native = torch.empty_like(output)
        torch.ops._moe_C.moe_sum(values, native, ids, active_map)
        torch.testing.assert_close(native, expected, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.moe_sum(values, output, ids, active_map)
    for state in (False, True, False):
        expected = prepare(state)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
