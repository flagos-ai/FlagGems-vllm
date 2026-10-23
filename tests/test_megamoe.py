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

"""Tests for the public Hopper MegaMoE entry point."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.ops import megamoe as hopper_megamoe
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

pytestmark = pytest.mark.megamoe


def _is_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamoe_fp8_is_exported_on_hopper():
    assert callable(hopper_megamoe.megamoe_fp8)
    assert flaggems_vllm.megamoe_fp8 is hopper_megamoe.megamoe_fp8
