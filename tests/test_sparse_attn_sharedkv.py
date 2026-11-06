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

"""Accuracy of the six validated DSV4 SparseAttnSharedKV shapes."""

import pytest
import torch

import flaggems_vllm
from tests.sparse_attn_sharedkv_utils import (
    ATOL,
    CASES,
    RTOL,
    make_inputs,
    official_operator,
    output_tensor,
    shape_label,
)

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.sparse_attn_sharedkv,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]
npu_only = pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")


@npu_only
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_sparse_attn_sharedkv_accuracy(case):
    official = official_operator()
    if official is None:
        pytest.skip("requires the vLLM-Ascend SparseAttnSharedKV reference")
    inputs = make_inputs(case, flaggems_vllm.device)
    expected = output_tensor(official(**inputs))
    actual, lse = flaggems_vllm.sparse_attn_sharedkv(
        **inputs,
        host_q_lens=(case.q_len,),
        host_kv_lens=(case.kv_len,),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), atol=ATOL, rtol=RTOL
    )
    assert lse.numel() == 0 and lse.dtype == torch.float32
    assert actual.shape == inputs["q"].shape and actual.dtype == torch.bfloat16


def test_sparse_attn_sharedkv_rejects_cpu():
    module = __import__(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv",
        fromlist=["sparse_attn_sharedkv"],
    )
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        module.sparse_attn_sharedkv(torch.empty((1, 64, 512)))


@npu_only
@torch.inference_mode()
def test_sparse_attn_sharedkv_rejects_unsupported_layout():
    case = CASES[0]
    inputs = make_inputs(case, flaggems_vllm.device)
    inputs["layout_q"] = "BSND"
    with pytest.raises(NotImplementedError, match="TND"):
        flaggems_vllm.sparse_attn_sharedkv(
            **inputs,
            host_q_lens=(case.q_len,),
            host_kv_lens=(case.kv_len,),
        )


@npu_only
@torch.inference_mode()
def test_sparse_attn_sharedkv_rejects_partial_host_lengths():
    case = CASES[0]
    inputs = make_inputs(case, flaggems_vllm.device)
    with pytest.raises(ValueError, match="together"):
        flaggems_vllm.sparse_attn_sharedkv(**inputs, host_q_lens=(case.q_len,))


def test_sparse_attn_sharedkv_case_labels():
    for case in CASES:
        label = shape_label(case)
        assert case.name in label or case.mode in label
