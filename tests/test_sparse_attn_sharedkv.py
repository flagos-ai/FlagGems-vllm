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
pytestmark = [pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")]
npu_only = pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")


@npu_only
@pytest.mark.sparse_attn_sharedkv
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


@pytest.mark.sparse_attn_sharedkv
def test_sparse_attn_sharedkv_rejects_cpu():
    module = __import__(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv",
        fromlist=["sparse_attn_sharedkv"],
    )
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        module.sparse_attn_sharedkv(torch.empty((1, 64, 512)))


@npu_only
@pytest.mark.sparse_attn_sharedkv
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
@pytest.mark.sparse_attn_sharedkv
@torch.inference_mode()
def test_sparse_attn_sharedkv_rejects_partial_host_lengths():
    case = CASES[0]
    inputs = make_inputs(case, flaggems_vllm.device)
    with pytest.raises(ValueError, match="together"):
        flaggems_vllm.sparse_attn_sharedkv(**inputs, host_q_lens=(case.q_len,))


@pytest.mark.sparse_attn_sharedkv
def test_sparse_attn_sharedkv_case_labels():
    for case in CASES:
        label = shape_label(case)
        assert case.name in label or case.mode in label


def _legacy_small_case():
    generator = torch.Generator().manual_seed(932)
    return dict(
        q=(torch.randn(1, 64, 512, generator=generator) * 0.2).bfloat16(),
        ori_kv=(torch.randn(2, 128, 1, 512, generator=generator) * 0.2).bfloat16(),
        ori_block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_kv=torch.tensor([17], dtype=torch.int32),
        sinks=torch.linspace(-1, 1, 64),
        softmax_scale=0.0,
    )


def _legacy_small_oracle(cpu):
    length = int(cpu["seqused_kv"][0])
    rows = []
    for logical in range(max(0, length - 128), length):
        page = int(cpu["ori_block_table"][0, logical // 128])
        rows.append(cpu["ori_kv"][page, logical % 128, 0].double())
    kv = torch.stack(rows)
    logits = cpu["q"][0].double() @ kv.T * 512**-0.5
    denominator = torch.logaddexp(
        torch.logsumexp(logits, dim=-1), cpu["sinks"].double()
    )
    return (torch.exp(logits - denominator[:, None]) @ kv)[None]


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_legacy_small_oracle_graph():
    cpu = _legacy_small_case()
    args = {
        key: value.to("npu") if isinstance(value, torch.Tensor) else value
        for key, value in cpu.items()
    }
    for _ in range(2):
        actual, lse = flaggems_vllm.sparse_attn_sharedkv(**args)
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual.cpu().double(), _legacy_small_oracle(cpu), rtol=2e-2, atol=2e-3
    )
    assert lse.numel() == 0
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, lse = flaggems_vllm.sparse_attn_sharedkv(**args)
    for length in (133, 29):
        cpu["q"] *= 0.8
        cpu["ori_kv"] *= 0.9
        cpu["seqused_kv"][0] = length
        cpu["ori_block_table"][0] = cpu["ori_block_table"][0].flip(0)
        cpu["sinks"] += 0.1
        for key in ("q", "ori_kv", "seqused_kv", "ori_block_table", "sinks"):
            args[key].copy_(cpu[key])
        actual.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual.cpu().double(), _legacy_small_oracle(cpu), rtol=2e-2, atol=2e-3
        )
        assert torch.isfinite(actual).all()


@pytest.mark.sparse_attn_sharedkv
@npu_only
@pytest.mark.parametrize(
    "invalid, message",
    [
        ("page_negative", "physical page id"),
        ("page_upper", "physical page id"),
        ("query_start", "start at 0"),
        ("query_decreasing", "monotonically"),
        ("query_upper", "within"),
        ("query_end", "must equal"),
        ("kv_negative", "non-negative"),
        ("kv_upper", "exceeds ori_block_table capacity"),
        ("compressed_upper", "compressed KV length"),
    ],
)
def test_sparse_attn_sharedkv_legacy_invalid_host_metadata(invalid, message):
    cpu = _legacy_small_case()
    cpu["ori_block_table"] = torch.zeros((1, 65), dtype=torch.int32)
    cpu["seqused_kv"] = torch.tensor([8193], dtype=torch.int32)
    cpu["metadata"] = torch.zeros(1024, dtype=torch.int32)
    if invalid.startswith("page_"):
        cpu["ori_block_table"][0, -1] = -1 if invalid == "page_negative" else 2
    elif invalid.startswith("query_"):
        values = {
            "query_start": [1, 1],
            "query_decreasing": [0, -1],
            "query_upper": [0, 2],
            "query_end": [0, 0],
        }
        cpu["cu_seqlens_q"] = torch.tensor(values[invalid], dtype=torch.int32)
    elif invalid == "kv_negative":
        cpu["seqused_kv"][0] = -1
    elif invalid == "kv_upper":
        cpu["seqused_kv"][0] = 65 * 128 + 1
    else:
        cpu["cmp_ratio"] = 4
        cpu["cmp_kv"] = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16)
        cpu["cmp_block_table"] = torch.zeros((1, 1), dtype=torch.int32)
    args = {
        key: value.to("npu") if isinstance(value, torch.Tensor) else value
        for key, value in cpu.items()
    }
    before = args["ori_kv"].clone()
    with pytest.raises(ValueError, match=message):
        flaggems_vllm.sparse_attn_sharedkv(
            **args, host_q_lens=(1,), host_kv_lens=(8193,)
        )
    torch.testing.assert_close(args["ori_kv"], before, rtol=0, atol=0)
