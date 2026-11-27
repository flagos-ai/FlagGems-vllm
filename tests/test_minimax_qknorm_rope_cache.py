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

import flaggems_vllm as gems

pytestmark = [
    pytest.mark.minimax_qknorm_rope_cache,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    tokens, width = shape
    pages = max(2, (tokens + 127) // 128)
    packed = torch.randn(shape, dtype=dtype, device=device)
    trig = torch.randn((tokens + 1, 64), dtype=dtype, device=device)
    pos = torch.arange(tokens, device=device)
    weights = [torch.randn(128, device=device, dtype=dtype) for _ in range(4)]
    cache = torch.zeros((pages, 128, 1, 128), dtype=dtype, device=device)
    index_cache = torch.zeros((pages, 128, 128), dtype=dtype, device=device)
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    return (
        packed,
        trig,
        pos,
        *weights,
        cache,
        torch.zeros_like(cache),
        index_cache,
        slots,
        slots,
        tokens,
        16,
        1,
        1,
        1e-6,
    ), {}


def _reference():
    from vllm_ascend.ops.triton.msa.minimax_qknorm_rope_cache import (
        minimax_qknorm_rope_cache,
    )

    return minimax_qknorm_rope_cache


@pytest.mark.parametrize("shape", [(1, 2560), (1024, 2560), (37, 2560)])
def test_minimax_qknorm_rope_cache(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, gems.device)
    ref_args = list(args)
    for i in [7, 8, 9]:
        ref_args[i] = args[i].clone()
    expected = _reference()(*ref_args, **kwargs)
    actual = gems.minimax_qknorm_rope_cache(*args, **kwargs)
    torch.testing.assert_close(actual, expected)
    for i in [7, 8, 9]:
        torch.testing.assert_close(args[i], ref_args[i])


def test_minimax_shape_guard():
    args, kw = _inputs((1, 2560), torch.bfloat16, gems.device)
    with pytest.raises(ValueError, match="combined packed width"):
        gems.minimax_qknorm_rope_cache(args[0][:, :128].contiguous(), *args[1:])
