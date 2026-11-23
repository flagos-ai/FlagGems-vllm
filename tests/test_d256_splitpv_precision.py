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

from tests.test_d256_port_precision import (
    test_probability_quantization_accuracy_d256 as _cancel_case,
)
from tests.test_flash_attn_varlen_func_w8a8_int8 import _run_case


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([1024], [2048]),
        ([1031], [2051]),
        ([1024], [8192]),
        ([1024], [8193]),
        ([1, 1024], [4096, 4096]),
    ],
)
def test_splitpv_output_lse(qlens, klens, causal):
    _run_case(
        qlens,
        klens,
        dim=256,
        heads=4,
        kvheads=2,
        paged=True,
        causal=causal,
        broadcast_scales=True,
        return_lse=True,
    )


@pytest.mark.parametrize("scale", [0.125, 0.25])
def test_splitpv_cancellation(scale):
    _cancel_case(0, 1024, 2048, True, scale)
