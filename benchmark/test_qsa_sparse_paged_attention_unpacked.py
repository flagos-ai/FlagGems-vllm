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

"""Measure the unpacked public ABI with the established QSA workload/reference."""

from functools import partial

import pytest
import torch

from flaggems_vllm import qsa_sparse_paged_attention_unpacked, vendor_name

from .qwen4_references import sparse_reference
from .test_qsa_attention import SparseBenchmark, inputs


@pytest.mark.qsa_sparse_paged_attention_unpacked
@pytest.mark.skipif(
    vendor_name != "thead", reason="PPU unpacked API compatibility benchmark"
)
@pytest.mark.parametrize("splits", [1, 8])
def test_unpacked_sparse_attention_perf(splits):
    SparseBenchmark(
        input_fn=partial(inputs, splits=splits),
        op_name="qsa_sparse_paged_attention_unpacked",
        torch_op=sparse_reference,
        gems_op=qsa_sparse_paged_attention_unpacked,
        dtypes=[torch.bfloat16],
    ).run()
