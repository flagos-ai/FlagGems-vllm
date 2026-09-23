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

import importlib

import pytest
import torch

import flaggems_vllm
from tests import conftest as test_config
from tests.lightning_indexer_utils import (
    CASES,
    HAS_VLLM_ASCEND_REFERENCE,
    QUICK_CASE_IDS,
    assert_index_multisets_equal,
    get_reference,
    is_torch_fallback_feasible,
    make_inputs,
)

_IS_ASCEND = (
    flaggems_vllm.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)
_LIGHTNING_INDEXER = getattr(flaggems_vllm, "lightning_indexer", None)
_LIGHTNING_INDEXER_MODULE = importlib.import_module(
    "flaggems_vllm.runtime.backend._ascend.ops.lightning_indexer"
)

pytestmark = [
    pytest.mark.lightning_indexer,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
@torch.inference_mode()
def test_lightning_indexer_matches_reference(case):
    if test_config.QUICK_MODE and case.case_id not in QUICK_CASE_IDS:
        pytest.skip("excluded by --quick")
    if not HAS_VLLM_ASCEND_REFERENCE and not is_torch_fallback_feasible(case):
        pytest.skip("long prefill Torch fallback is intentionally bounded")
    assert callable(_LIGHTNING_INDEXER)

    inputs = make_inputs(case, flaggems_vllm.device)
    reference, _ = get_reference()
    expected = reference(**inputs)
    actual = _LIGHTNING_INDEXER(**inputs)
    torch.npu.synchronize()
    assert_index_multisets_equal(actual, expected)


@torch.inference_mode()
def test_lightning_indexer_rejects_unsupported_layout():
    assert callable(_LIGHTNING_INDEXER)
    inputs = make_inputs(CASES[0], flaggems_vllm.device)
    inputs["layout_query"] = "BSND"
    with pytest.raises(NotImplementedError, match="TND query"):
        _LIGHTNING_INDEXER(**inputs)


def test_lightning_indexer_requires_pr1065_custom_ops(monkeypatch):
    monkeypatch.setattr(_LIGHTNING_INDEXER_MODULE, "tle", None)
    monkeypatch.setattr(
        _LIGHTNING_INDEXER_MODULE,
        "_PR1065_IMPORT_ERROR",
        ImportError("missing FlagTree PR #1065"),
    )
    with pytest.raises(RuntimeError, match="FlagTree PR #1065 CustomOps"):
        _LIGHTNING_INDEXER_MODULE.lightning_indexer(None, None, None)
