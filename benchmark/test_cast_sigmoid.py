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

"""cast_sigmoid benchmark.

Baseline is ``torch.sigmoid`` (same-dtype output), i.e. the plain element-wise
sigmoid the operator replaces. Shapes (small / medium / large) come from
``benchmark/core_shapes.yaml`` through ``op_name``.
"""

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close

from . import base, consts, utils


class CastSigmoidBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # core_shapes.yaml already spans small/medium/large, and the generic
        # 2**28 shapes only stress the allocator for a bandwidth-bound kernel.
        return []

    def get_input_iter(self, dtype):
        verified = False
        for inputs in super().get_input_iter(dtype):
            if not verified:
                # Guard against a silently broken kernel benchmarking fast.
                gems_assert_close(self.gems_op(*inputs), torch.sigmoid(*inputs), dtype)
                verified = True
            yield inputs


@pytest.mark.cast_sigmoid
def test_cast_sigmoid():
    bench = CastSigmoidBenchmark(
        input_fn=utils.unary_input_fn,
        op_name="cast_sigmoid",
        torch_op=torch.sigmoid,
        gems_op=flaggems_vllm.cast_sigmoid,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
