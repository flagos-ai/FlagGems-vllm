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

"""Collector keys, provenance and unsupported backend contract."""

import pytest
import torch

import flaggems_vllm as gems
from baseline import matmul_persistent as workload


@pytest.mark.matmul_persistent
def test_workload_contract():
    assert workload.native() is workload.matmul_batch_invariant
    assert workload.native() is not gems.matmul_persistent
    assert len(workload.grid()) == 12
    assert len({tuple(workload.key_shape(b)) for b in workload.grid()}) == 12
    for binding in workload.grid():
        assert workload.binding_from_shape(workload.key_shape(binding)) == binding
        for dtype in workload.DTYPES:
            cfg = workload.config(binding, dtype)
            assert cfg["inputs"]["a"]["shape"] == list(binding["a_shape"])
            assert cfg["outputs"]["c"]["dtype"] == str(dtype)


@pytest.mark.matmul_persistent
def test_cpu_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="NVIDIA CUDA|Ascend NPU"):
        gems.matmul_persistent(torch.empty(2, 3), torch.empty(3, 4))


@pytest.mark.matmul_persistent
@pytest.mark.parametrize(
    "descriptor",
    [
        [],
        [1, 32, 2, 32, 16],
        [2, 32, 64, 2, 32, 16],
        [2, 32, 64, 2, 64],
        [True, 32, 64],
    ],
)
def test_invalid_workload_descriptor(descriptor):
    with pytest.raises((ValueError, RuntimeError)):
        workload.binding_from_shape(descriptor)
