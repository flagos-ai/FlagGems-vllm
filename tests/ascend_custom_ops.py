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

"""Opt-in pytest plugin for tests using vLLM-Ascend custom-op references.

From the repository root:
    python -m pytest -p tests.ascend_custom_ops tests/test_sparse_flash_attention.py

Do not load this plugin from conftest.py: pytest must register it before loading
initial conftests, whose FlagGems imports can initialize the device runtime.
"""

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests():
    """Let vLLM configure its vendor ops before CANN device initialization."""
    try:
        from vllm_ascend.utils import enable_custom_op
    except ImportError as exc:
        raise pytest.UsageError(
            "The tests.ascend_custom_ops plugin requires Torch-NPU and vLLM-Ascend"
        ) from exc
    if not enable_custom_op():
        raise pytest.UsageError(
            "The installed vLLM-Ascend custom ops could not be enabled on this device"
        )
