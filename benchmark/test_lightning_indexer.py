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
import torch

import flaggems_vllm
from tests.lightning_indexer_utils import (
    CASES,
    HAS_VLLM_ASCEND_REFERENCE,
    get_reference,
    is_torch_fallback_feasible,
    make_inputs,
)

from . import base

_IS_ASCEND = (
    flaggems_vllm.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)
_LIGHTNING_INDEXER = getattr(flaggems_vllm, "lightning_indexer", None)


def _reference_call(_case_index, **inputs):
    reference, _ = get_reference()
    return reference(**inputs)


def _candidate_call(_case_index, **inputs):
    return _LIGHTNING_INDEXER(**inputs)


class LightningIndexerBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            op_name="lightning_indexer",
            torch_op=_reference_call,
            dtypes=[torch.bfloat16],
        )
        self.set_gems(_candidate_call)
        self.shape_desc = (
            "case, T, Hq, D, key_blocks, block_size, Hk, q_ends, key_lengths"
        )

    def set_shapes(self, shape_file_path=None):
        del shape_file_path
        if HAS_VLLM_ASCEND_REFERENCE:
            self.shapes = list(CASES)
        else:
            self.shapes = [case for case in CASES if is_torch_fallback_feasible(case)]

    def get_input_iter(self, dtype):
        del dtype
        for case_index, case in enumerate(self.shapes):
            yield case_index, make_inputs(case, self.device)

    def record_shapes(self, case_index, **inputs):
        case = self.shapes[case_index]
        return {
            "case": case.case_id,
            "scene": case.scene,
            "query": list(inputs["query"].shape),
            "key": list(inputs["key"].shape),
            "weights": list(inputs["weights"].shape),
            "local_q_lengths": list(case.local_q_lengths),
            "key_lengths": list(case.key_lengths),
            "sparse_count": inputs["sparse_count"],
            "sparse_mode": inputs["sparse_mode"],
        }


@pytest.mark.lightning_indexer
@pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")
def test_lightning_indexer_benchmark():
    assert callable(_LIGHTNING_INDEXER)
    LightningIndexerBenchmark().run()
