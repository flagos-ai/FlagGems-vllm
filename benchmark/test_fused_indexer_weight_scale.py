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

import os

import pytest
import torch

import flaggems_vllm
from tests import test_fused_indexer_weight_scale as indexer_tests
from tests.accuracy_utils import gems_assert_close

from .base import GenericBenchmark
from .test_blas_perf_parallel import PARALLEL_WORKER_ENV, ParallelBenchmarkMixin

native_weight_scale = indexer_tests.native_weight_scale

# All unique GLM-5.3-Flash shapes captured with the native model structure and
# dummy weights. Published shapes/counts: https://github.com/flagos-ai/FlagGems-vllm/pull/1048
# Across four FlagOSTune scenarios, sum(count)=1990208.
REAL_TOKEN_COUNTS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    20,
    24,
    26,
    28,
    29,
    32,
    33,
    34,
    35,
    36,
    43,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    64,
    65,
    66,
    68,
    71,
    72,
    74,
    76,
    152,
    1036,
    2369,
    3700,
    4108,
    4652,
    8998,
    9105,
    12325,
    13845,
    16384,
)


def input_fn(shape, dtype, device):
    torch.manual_seed(0)
    tokens, heads = shape
    weights, query = indexer_tests.make_inputs(tokens, heads, device, dtype=dtype)
    yield weights, query, 128**-0.5 * heads**-0.5


class IndexerWeightScaleBenchmark(ParallelBenchmarkMixin, GenericBenchmark):
    DEFAULT_SHAPES = [(1, 32), (35, 32), (16384, 32)]
    DEFAULT_SHAPE_DESC = "tokens, heads"
    RECORD_PARALLEL_RESULTS = True

    def set_more_shapes(self):
        if os.environ.get(PARALLEL_WORKER_ENV):
            return []
        return [(tokens, 32) for tokens in REAL_TOKEN_COUNTS]

    def get_input_iter(self, dtype):
        for inputs in super().get_input_iter(dtype):
            # Compile and validate before timing, including before graph capture.
            expected = self.torch_op(*inputs)
            output = self.gems_op(*inputs)
            gems_assert_close(output, expected, torch.float32, atol=1e-6)
            yield inputs

    def _get_tuning_fixed_overhead(self):
        return 1_000_000


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.skipif(
    indexer_tests.API_NAME not in vars(flaggems_vllm),
    reason="backend has no indexer weight scaling",
)
def test_fused_indexer_weight_scale(native_weight_scale):
    IndexerWeightScaleBenchmark(
        op_name="fused_indexer_weight_scale",
        torch_op=native_weight_scale,
        gems_op=flaggems_vllm.fused_indexer_weight_scale,
        input_fn=input_fn,
        dtypes=[torch.float32],
    ).run()
