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
from vllm.v1.worker.gpu.input_batch import (
    get_num_sampled_and_rejected as vllm_op,
)

import flaggems_vllm

pytestmark = pytest.mark.get_num_sampled_and_rejected
SIZES = (1, 7, 32, 128, 512)


def check_case(n, seed, chunked_percent=None, delta=None):
    rng = torch.Generator().manual_seed(seed)
    counts = torch.randint(1, 9, (n,), generator=rng, dtype=torch.int32)
    sampled = (torch.rand(n, generator=rng) * counts).int() + 1
    prefill = torch.randint(2, 128, (n,), generator=rng, dtype=torch.int32)
    mapping = torch.arange(n, dtype=torch.int32)
    if seed % 3 == 1:
        mapping = torch.randperm(n, generator=rng, dtype=torch.int32)
    elif seed % 3 == 2:
        mapping = torch.randint(n, (n,), generator=rng, dtype=torch.int32)
    if delta is None:
        delta = torch.randint(-1, 2, (n,), generator=rng, dtype=torch.int32)
    if chunked_percent is not None:
        delta = torch.ones(n, dtype=torch.int32)
        delta[: n * chunked_percent // 100] = -1
    seq = prefill[mapping.long()] + delta
    cu = torch.cat((counts.new_zeros(1), counts.cumsum(0, dtype=torch.int32)))
    inputs = [x.to(flaggems_vllm.device) for x in (seq, cu, mapping, prefill)]
    initial = sampled.to(flaggems_vllm.device)
    expected = []
    for op in (vllm_op, flaggems_vllm.get_num_sampled_and_rejected):
        mutable = initial.clone()
        args = [x.clone() for x in inputs]
        result, rejected = op(mutable, *args)
        assert result is mutable
        for actual, original in zip(args, inputs):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        expected.append((result, rejected))
    for native, gems in zip(*expected):
        torch.testing.assert_close(gems, native, rtol=0, atol=0)
    torch.testing.assert_close(expected[0][0].cpu(), sampled.masked_fill(delta < 0, 0))
    torch.testing.assert_close(
        expected[0][1].cpu(), (counts - sampled).masked_fill(delta < 0, 0)
    )


@pytest.mark.parametrize("n", SIZES)
def test_get_num_sampled_and_rejected(n):
    check_case(n, n + 50, chunked_percent=50)


@pytest.mark.parametrize("n", [32, 128])
@pytest.mark.parametrize("chunked_percent", [0, 100])
def test_get_num_sampled_and_rejected_uniform(n, chunked_percent):
    check_case(n, n + chunked_percent, chunked_percent)


def test_get_num_sampled_and_rejected_boundary():
    delta = torch.tensor([-1, 0, 1], dtype=torch.int32)
    check_case(3, seed=0, delta=delta)


def test_get_num_sampled_and_rejected_random():
    for seed in range(100):
        check_case(SIZES[seed % len(SIZES)], seed)


@pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="T-Head dispatch")
def test_get_num_sampled_and_rejected_thead_dispatch(caplog):
    from importlib import import_module

    backend = import_module(
        "flaggems_vllm.runtime.backend._thead.ops.input_batch"
    )
    entry = flaggems_vllm.get_num_sampled_and_rejected
    assert entry is backend.get_num_sampled_and_rejected
    with caplog.at_level("DEBUG", logger=entry.__module__):
        check_case(7, seed=0, chunked_percent=50)
    assert "GEMS_THEAD GET_NUM_SAMPLED_AND_REJECTED" in caplog.text
