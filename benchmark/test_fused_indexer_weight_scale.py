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

from . import base
from .test_blas_perf_parallel import PARALLEL_WORKER_ENV, ParallelBenchmarkMixin

native_weight_scale = indexer_tests.native_weight_scale


def input_fn(shape, dtype, device):
    torch.manual_seed(0)
    tokens, heads = shape
    weights, q_scale = indexer_tests.make_inputs(tokens, heads, device, dtype=dtype)
    yield weights, q_scale, 128**-0.5 * heads**-0.5


class IndexerWeightScaleBenchmark(ParallelBenchmarkMixin, base.GenericBenchmark):
    RECORD_PARALLEL_RESULTS = True
    DEFAULT_SHAPES = [(1, 32), (35, 32), (16384, 32)]
    DEFAULT_SHAPE_DESC = "tokens, heads"

    def set_more_shapes(self):
        if os.environ.get(PARALLEL_WORKER_ENV):
            return []
        # Include captured long-tail calls in comprehensive runs.
        return [
            (4, 32),  # count=1079584
            (8, 32),  # count=177584
            (35, 32),  # count=158752
            (65, 32),  # count=158576
            (18, 32),  # count=90288
            (46, 32),  # count=90112
            (13, 32),  # count=86064
            (51, 32),  # count=85888
            (24, 32),  # count=20064
            (76, 32),  # count=19712
            (16384, 32),  # count=9328
            (1, 32),  # count=1408
            (3, 32),  # count=1232
            (2, 32),  # count=1056
            (17, 32),  # count=704
            (14, 32),  # count=528
            (15, 32),  # count=528
            (16, 32),  # count=528
            (49, 32),  # count=528
            (50, 32),  # count=528
            (7, 32),  # count=528
            (20, 32),  # count=352
            (32, 32),  # count=352
            (34, 32),  # count=352
            (47, 32),  # count=352
            (48, 32),  # count=352
            (5, 32),  # count=352
            (6, 32),  # count=352
            (1036, 32),  # count=176
            (11, 32),  # count=176
            (12325, 32),  # count=176
            (13845, 32),  # count=176
            (152, 32),  # count=176
            (2369, 32),  # count=176
            (26, 32),  # count=176
            (28, 32),  # count=176
            (29, 32),  # count=176
            (33, 32),  # count=176
            (36, 32),  # count=176
            (3700, 32),  # count=176
            (4108, 32),  # count=176
            (43, 32),  # count=176
            (45, 32),  # count=176
            (4652, 32),  # count=176
            (64, 32),  # count=176
            (66, 32),  # count=176
            (68, 32),  # count=176
            (71, 32),  # count=176
            (72, 32),  # count=176
            (74, 32),  # count=176
            (8998, 32),  # count=176
            (9105, 32),  # count=176
        ]

    def get_input_iter(self, dtype):
        for index, inputs in enumerate(super().get_input_iter(dtype)):
            expected = self.torch_op(*inputs)
            output = self.gems_op(*inputs)
            gems_assert_close(output, expected, torch.float32, atol=1e-6)
            if index or flaggems_vllm.vendor_name != "mthreads":
                yield inputs
                continue
            # Prime first-use setup equally for the compiled MThreads baseline.
            self.get_latency(self.torch_op, *inputs)
            self.get_latency(self.gems_op, *inputs)
            yield inputs


@pytest.mark.fused_indexer_weight_scale
@pytest.mark.skipif(
    indexer_tests.API_NAME not in vars(flaggems_vllm),
    reason="backend has no indexer weight scaling",
)
def test_fused_indexer_weight_scale(native_weight_scale):
    bench = IndexerWeightScaleBenchmark(
        op_name="fused_indexer_weight_scale",
        torch_op=native_weight_scale,
        input_fn=input_fn,
        dtypes=[torch.float32],
    )
    bench.set_gems(flaggems_vllm.fused_indexer_weight_scale)
    bench.run()
