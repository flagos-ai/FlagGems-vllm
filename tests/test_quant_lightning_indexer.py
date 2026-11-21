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
from tests.quant_lightning_indexer_utils import (
    CASES,
    QUICK_CASE_IDS,
    assert_matches,
    call_candidate,
    call_official,
    make_inputs,
    prepare_metadata,
    register_official_ops,
)
 
_IS_ASCEND = (
    flaggems_vllm.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)
_QUANT_LIGHTNING_INDEXER = getattr(flaggems_vllm, "quant_lightning_indexer", None)
_QUANT_MODULE = importlib.import_module(
    "flaggems_vllm.runtime.backend._ascend.ops.quant_lightning_indexer"
)
 
pytestmark = [
    pytest.mark.quant_lightning_indexer,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]
 
@pytest.mark.parametrize("case", [pytest.param(case, id=case.case_id) for case in CASES])
@torch.inference_mode()
def test_quant_lightning_indexer_matches_official(case):
    if test_config.QUICK_MODE and case.case_id not in QUICK_CASE_IDS:
        pytest.skip("excluded by --quick")
    assert callable(_QUANT_LIGHTNING_INDEXER)
    register_official_ops()
    inputs = make_inputs(case, flaggems_vllm.device, seed=20260915)
    prepare_metadata(inputs, case)
    expected = call_official(inputs)
    actual = call_candidate(_QUANT_MODULE, case, inputs)
    torch.npu.synchronize()
    assert_matches(expected, actual)
 
 
@torch.inference_mode()
def test_quant_lightning_indexer_rejects_unsupported_layout():
    register_official_ops()
    inputs = make_inputs(CASES[0], flaggems_vllm.device)
    prepare_metadata(inputs, CASES[0])
    with pytest.raises(NotImplementedError, match="TND query"):
        _QUANT_LIGHTNING_INDEXER(
            query=inputs["query"],
            key=inputs["key"],
            weights=inputs["weights"],
            query_dequant_scale=inputs["query_dequant_scale"],
            key_dequant_scale=inputs["key_dequant_scale"],
            actual_seq_lengths_query=inputs["source_actual_seq_lengths_query"],
            actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
            block_table=inputs["block_table"],
            metadata=inputs["metadata"],
            local_query_lengths=CASES[0].local_q_lengths,
            layout_query="BSND",
        )
 
 
@torch.inference_mode()
def test_quant_lightning_indexer_rejects_return_value():
    register_official_ops()
    inputs = make_inputs(CASES[0], flaggems_vllm.device)
    prepare_metadata(inputs, CASES[0])
    with pytest.raises(NotImplementedError, match="return_value=True"):
        _QUANT_LIGHTNING_INDEXER(
            query=inputs["query"],
            key=inputs["key"],
            weights=inputs["weights"],
            query_dequant_scale=inputs["query_dequant_scale"],
            key_dequant_scale=inputs["key_dequant_scale"],
            actual_seq_lengths_query=inputs["source_actual_seq_lengths_query"],
            actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
            block_table=inputs["block_table"],
            metadata=inputs["metadata"],
            local_query_lengths=CASES[0].local_q_lengths,
            return_value=True,
        )
 
 
def test_quant_lightning_indexer_requires_pr1065_custom_ops(monkeypatch):
    monkeypatch.setattr(_QUANT_MODULE, "tle", None)
    monkeypatch.setattr(
        _QUANT_MODULE,
        "_PR1065_IMPORT_ERROR",
        ImportError("missing FlagTree PR #1065"),
    )
    with pytest.raises(RuntimeError, match="FlagTree PR #1065 CustomOps"):
        _QUANT_MODULE.quant_lightning_indexer(None, None, None)