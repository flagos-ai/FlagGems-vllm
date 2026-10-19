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
from importlib.util import find_spec


def initialize_ascend_custom_ops():
    """Load available vLLM-Ascend references before FlagGems queries the NPU.

    CANN discovers vendor implementations during device initialization. Loading
    vLLM only inside a test can leave its registered op using a built-in CANN
    implementation. vLLM owns vendor setup; no installation path is required here.
    """
    vendor = next(
        (
            os.environ[key].lower()
            for key in (
                "GEMS_VENDOR",
                "FLAGGEMS_VENDOR",
                "GEMS_BACKEND",
                "FLAGGEMS_BACKEND",
            )
            if key in os.environ
        ),
        None,
    )
    if "PPU_SDK" in os.environ or vendor not in (None, "ascend"):
        return
    if find_spec("torch_npu") is None or find_spec("vllm_ascend") is None:
        return

    from vllm_ascend.utils import enable_custom_op

    # Unsupported devices may return False; individual reference tests retain
    # their existing failure/skip policy rather than requiring ops for all tests.
    enable_custom_op()
