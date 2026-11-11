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

"""Tests for the public Hopper MegaMHC entry points."""

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime.backend._nvidia.hopper.ops import megamhc as hopper_megamhc
from flaggems_vllm.utils.triton_version_utils import has_triton_tle


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.megamhc
@pytest.mark.skipif(
    not (_is_hopper() and has_triton_tle(3, 6, 0)),
    reason="requires Hopper with FlagTree TLE",
)
def test_megamhc_is_exported_on_hopper():
    assert flaggems_vllm.create_mega_mhc_context is (
        hopper_megamhc.create_mega_mhc_context
    )
    assert flaggems_vllm.mega_mhc is hopper_megamhc.mega_mhc
