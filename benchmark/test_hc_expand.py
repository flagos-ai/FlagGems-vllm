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

import flaggems_vllm
from baseline.hc_expand import (
    DTYPES,
    OP_NAME,
    build_inputs,
    native,
    torch_reference,
)

from . import base

if flaggems_vllm.vendor_name == "ascend":
    PERFORMANCE_BASELINE = torch_reference
else:
    PERFORMANCE_BASELINE = native()

pytestmark = [
    pytest.mark.hc_expand,
    pytest.mark.skipif(
        PERFORMANCE_BASELINE is None,
        reason="hc_expand performance baseline unavailable",
    ),
]


class HCExpandBenchmark(base.Benchmark):
    def __init__(self, dtype):
        super().__init__(OP_NAME, PERFORMANCE_BASELINE, [dtype], is_inplace=False)
        self.layouts = ("contiguous", "strided")
        self.current_layout = None
        self.set_gems(flaggems_vllm.hc_expand)

    def get_input_iter(self, dtype):
        for layout in self.layouts:
            self.current_layout = layout
            for num_tokens, hidden_size, hc_mult in self.shapes:
                binding = {
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "hc_mult": hc_mult,
                }
                args, _ = build_inputs(
                    binding,
                    dtype,
                    self.device,
                    layout=layout,
                    special_values=False,
                )
                yield args

    def record_shapes(self, *args, **kwargs):
        return [self.current_layout, *super().record_shapes(*args, **kwargs)]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_hc_expand(dtype):
    bench = HCExpandBenchmark(dtype)
    bench.run()
