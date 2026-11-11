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

"""Check compute_topk_scores against a PyTorch oracle."""

import importlib
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm

CUSTOM_IDS = ((11, 17), (), (4, 8, 9), ())
REQUEST_MAPPING = (2, 0, 3, 1)


def _require_supported_dtype(dtype):
    if dtype == torch.bfloat16 and not flaggems_vllm.runtime.device.support_bf16:
        pytest.skip("bfloat16 is unavailable on this backend")


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(flaggems_vllm.device)


@pytest.fixture(scope="module")
def candidate_logprob():
    return importlib.import_module("flaggems_vllm.ops.logprob")


def _custom_state(candidate, device, custom_ids):
    state = candidate.LogprobTokenIdsState(len(custom_ids), device)
    for req_idx, ids in enumerate(custom_ids):
        state.add_request(req_idx, SimpleNamespace(logprob_token_ids=list(ids)))
    state.apply_staged_writes()
    return state


def _pytorch_reference(logits, sampled, k, custom_ids, mapping, logits_mode):
    batch_size, vocab_size = logits.shape
    max_custom = max((len(ids) for ids in custom_ids), default=0)
    num_cols = max(k, max_custom)
    ids = torch.zeros(
        (batch_size, 1 + num_cols), dtype=sampled.dtype, device=logits.device
    )
    valid = torch.zeros_like(ids, dtype=torch.bool)
    ids[:, 0] = sampled
    valid[:, 0] = True
    topk_ids = torch.topk(logits, k, dim=-1).indices if k else None

    for row in range(batch_size):
        requested = custom_ids[mapping[row]] if custom_ids else ()
        selected = requested if requested else topk_ids[row] if k else ()
        count = len(selected)
        if count:
            ids[row, 1 : 1 + count] = torch.as_tensor(
                selected, dtype=ids.dtype, device=ids.device
            )
            valid[row, 1 : 1 + count] = True

    values = (
        logits.float() if logits_mode else torch.log_softmax(logits.float(), dim=-1)
    )
    scores = values.gather(-1, ids).masked_fill(~valid, float("-inf"))
    sampled_logits = logits.gather(-1, sampled[:, None])
    ranks = (logits >= sampled_logits).sum(dim=-1)
    # Match the existing operator's padded -inf rank comparisons.
    padded_tail = (-vocab_size) % 8192
    ranks += (sampled_logits.squeeze(-1) == float("-inf")) * padded_tail
    return ids, scores, ranks


@pytest.mark.parametrize(
    ("scenario", "vocab_size", "k", "dtype", "logits_mode"),
    [
        ("distinct", 1031, 5, torch.float32, False),
        ("ties", 1031, 5, torch.float32, False),
        ("ties", 2049, 1025, torch.float32, True),
        ("all_equal", 32000, 32000, torch.bfloat16, True),
        ("ties", 32000, 32000, torch.float16, False),
        ("all_neg_inf", 32000, 32000, torch.float32, False),
        ("distinct", 33, 33, torch.float32, False),
        ("distinct", 513, 513, torch.float32, True),
        ("distinct", 1024, 1024, torch.float32, False),
        ("distinct", 2049, 2049, torch.float32, True),
        ("signed_zero", 32000, 32000, torch.float32, True),
        ("infinities", 32000, 32000, torch.float32, False),
        ("nan", 1031, 5, torch.float32, False),
        ("nan", 32000, 32000, torch.float32, False),
        ("nan_finite_sampled", 32000, 32000, torch.float32, True),
        ("negative_nan", 2049, 1025, torch.float32, False),
        ("negative_nan", 32000, 32000, torch.float32, True),
    ],
)
@pytest.mark.compute_topk_scores
def test_topk_value_and_score_semantics(
    candidate_logprob, device, scenario, vocab_size, k, dtype, logits_mode
):
    """Tied IDs may differ, but selected values and score alignment must agree."""
    _require_supported_dtype(dtype)
    generator = torch.Generator(device=device).manual_seed(2026)
    if scenario == "distinct":
        logits = torch.randperm(vocab_size, device=device).float()[None, :]
    elif scenario == "ties":
        logits = torch.randint(
            -3, 4, (1, vocab_size), device=device, generator=generator
        ).to(dtype)
    elif scenario == "all_equal":
        logits = torch.zeros((1, vocab_size), device=device, dtype=dtype)
    elif scenario == "all_neg_inf":
        logits = torch.full((1, vocab_size), float("-inf"), device=device, dtype=dtype)
    elif scenario == "signed_zero":
        logits = torch.zeros((1, vocab_size), device=device)
        logits[:, ::2] = -0.0
    else:
        logits = torch.randn(
            (1, vocab_size), device=device, dtype=dtype, generator=generator
        )
        if scenario == "infinities":
            logits[0, :3] = torch.tensor(
                [float("inf"), float("-inf"), float("inf")], device=device
            )
        else:
            logits[0, :3] = float("nan")
            if scenario == "negative_nan":
                logits[0, :3] = -logits[0, :3]
    sampled_id = 10 if scenario == "nan_finite_sampled" else 1
    sampled = torch.tensor([sampled_id], device=device, dtype=torch.int64)
    result = candidate_logprob.compute_topk_scores(
        logits, k, sampled, logits_mode=logits_mode
    )
    ids = result.logprob_token_ids[:, 1:]
    assert ids.shape == (1, k)
    assert bool(((ids >= 0) & (ids < vocab_size)).all())
    assert torch.unique(ids).numel() == k
    torch.testing.assert_close(result.logprob_token_ids[:, 0], sampled)
    selected_values = logits.gather(1, ids)
    expected_values = torch.topk(logits, k, dim=1).values
    torch.testing.assert_close(
        selected_values, expected_values, rtol=0, atol=0, equal_nan=True
    )
    expected_scores = (
        logits.float() if logits_mode else torch.log_softmax(logits.float(), dim=-1)
    ).gather(1, result.logprob_token_ids)
    torch.testing.assert_close(
        result.logprobs, expected_scores, rtol=2e-4, atol=2e-3, equal_nan=True
    )
    expected_rank = (logits >= logits.gather(1, sampled[:, None])).sum(dim=1)
    if logits[0, sampled_id] == float("-inf"):
        expected_rank += (-vocab_size) % 8192
    torch.testing.assert_close(result.selected_token_ranks, expected_rank)


@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.compute_topk_scores
def test_full_vocab_batch64_values_and_ranks(candidate_logprob, device, logits_mode):
    """The large-batch full-vocabulary route preserves every selected value."""
    logits = torch.randn((64, 32000), device=device)
    sampled = torch.arange(64, device=device, dtype=torch.int64)
    result = candidate_logprob.compute_topk_scores(
        logits, 32000, sampled, logits_mode=logits_mode
    )
    selected = result.logprob_token_ids[:, 1:]
    torch.testing.assert_close(
        logits.gather(1, selected),
        torch.topk(logits, 32000, dim=1).values,
        rtol=0,
        atol=0,
    )
    assert all(torch.unique(row).numel() == 32000 for row in selected)
    expected_rank = (logits >= logits.gather(1, sampled[:, None])).sum(dim=1)
    torch.testing.assert_close(result.selected_token_ranks, expected_rank)
    expected_scores = (
        logits if logits_mode else torch.log_softmax(logits, dim=-1)
    ).gather(1, result.logprob_token_ids)
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)


@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.compute_topk_scores
def test_large_partial_topk_rows(candidate_logprob, device, logits_mode):
    """Large K below V must use the K-column stride for every output row."""
    logits = torch.randn((2, 32000), device=device)
    sampled = torch.tensor([0, 1], device=device)
    result = candidate_logprob.compute_topk_scores(
        logits, 20000, sampled, logits_mode=logits_mode
    )
    ids = result.logprob_token_ids
    assert ids.shape == (2, 20001)
    torch.testing.assert_close(ids[:, 0], sampled)
    torch.testing.assert_close(
        logits.gather(1, ids[:, 1:]),
        torch.topk(logits, 20000, dim=1).values,
        rtol=0,
        atol=0,
    )
    expected_scores = (
        logits if logits_mode else torch.log_softmax(logits, dim=-1)
    ).gather(1, ids)
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)


@pytest.mark.compute_topk_scores
def test_topk_above_vocab_rejected(candidate_logprob, device):
    logits = torch.zeros((1, 7), device=device)
    sampled = torch.zeros(1, dtype=torch.int64, device=device)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        candidate_logprob.compute_topk_scores(logits, 8, sampled)


@pytest.mark.parametrize(
    ("batch_size", "vocab_size", "k", "dtype", "custom_ids", "logits_mode"),
    [
        pytest.param(1, 7, 0, torch.float32, (), False, id="sample-only"),
        pytest.param(3, 1031, 5, torch.float32, (), False, id="topk-tail"),
        pytest.param(4, 2049, 1, torch.float16, (), False, id="fp16"),
        pytest.param(2, 2049, 1025, torch.float32, (), False, id="gather-chunks"),
        pytest.param(4, 1031, 5, torch.float32, CUSTOM_IDS, False, id="mixed-custom"),
        pytest.param(4, 1031, 0, torch.float16, CUSTOM_IDS, False, id="custom-k0"),
        pytest.param(4, 1031, 5, torch.float32, (), True, id="raw-logits"),
        pytest.param(4, 1031, 5, torch.float32, CUSTOM_IDS, True, id="custom-logits"),
        pytest.param(1, 151937, 5, torch.float32, (), False, id="large-vocab-tail"),
        pytest.param(17, 17, 17, torch.float32, (), False, id="k-equals-vocab"),
        pytest.param(4, 1031, 1, torch.float32, CUSTOM_IDS, False, id="custom-over-k"),
        pytest.param(1, 1031, 0, torch.float32, (), True, id="raw-logits-k0"),
        pytest.param(
            64,
            151936,
            5,
            torch.float32,
            (),
            True,
            id="large-logits-b64",
        ),
        pytest.param(4, 2049, 5, torch.bfloat16, (), False, id="bf16"),
    ],
)
@pytest.mark.compute_topk_scores
def test_compute_topk_scores_matches_pytorch(
    candidate_logprob,
    device,
    batch_size,
    vocab_size,
    k,
    dtype,
    custom_ids,
    logits_mode,
):
    _require_supported_dtype(dtype)
    torch.manual_seed(2026)
    logits = torch.randn(
        (batch_size, vocab_size), device=device, dtype=torch.float32
    ).to(dtype)
    logits[0, -1] = 12
    if batch_size > 1:
        logits[1, 3] = float("-inf")
    sampled = torch.tensor(
        [(7 + row * 31) % vocab_size for row in range(batch_size)],
        dtype=torch.int64,
        device=device,
    )

    mapping = REQUEST_MAPPING if custom_ids else ()
    state = _custom_state(candidate_logprob, device, custom_ids) if custom_ids else None
    mapping_tensor = (
        torch.tensor(mapping, dtype=torch.int32, device=device) if custom_ids else None
    )
    result = candidate_logprob.compute_topk_scores(
        logits,
        k,
        sampled,
        logprob_token_ids_state=state,
        expanded_idx_mapping=mapping_tensor,
        max_per_req_token_ids=max((len(ids) for ids in custom_ids), default=0),
        logits_mode=logits_mode,
    )
    expected_ids, expected_scores, expected_ranks = _pytorch_reference(
        logits, sampled, k, custom_ids, mapping, logits_mode
    )

    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    assert result.logprob_token_ids.dtype == sampled.dtype
    assert result.logprobs.dtype == torch.float32
    assert result.selected_token_ranks.dtype == torch.int64
    torch.testing.assert_close(
        result.selected_token_ranks, expected_ranks, rtol=0, atol=0
    )
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)


@pytest.mark.compute_topk_scores
def test_rank_counts_ties(candidate_logprob, device):
    logits = torch.tensor([[5.0, 5.0, 3.0, -1.0, -2.0, -3.0, -4.0]], device=device)
    sampled = torch.tensor([0], device=device)
    result = candidate_logprob.compute_topk_scores(logits, 0, sampled)
    expected_ids, expected_scores, expected_ranks = _pytorch_reference(
        logits, sampled, 0, (), (), False
    )

    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        result.selected_token_ranks, expected_ranks, rtol=0, atol=0
    )
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)


@pytest.mark.compute_topk_scores
def test_full_vocab_with_custom_ids(candidate_logprob, device):
    logits_mode = False
    torch.manual_seed(2026)
    vocab_size = 2049
    logits = torch.randn((4, vocab_size), device=device)
    sampled = torch.tensor([7, 11, 17, 23], device=device)
    state = _custom_state(candidate_logprob, device, CUSTOM_IDS)
    mapping = torch.tensor(REQUEST_MAPPING, dtype=torch.int32, device=device)
    result = candidate_logprob.compute_topk_scores(
        logits,
        vocab_size,
        sampled,
        logprob_token_ids_state=state,
        expanded_idx_mapping=mapping,
        max_per_req_token_ids=3,
        logits_mode=logits_mode,
    )
    expected_ids, expected_scores, expected_ranks = _pytorch_reference(
        logits, sampled, vocab_size, CUSTOM_IDS, REQUEST_MAPPING, logits_mode
    )
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        result.selected_token_ranks, expected_ranks, rtol=0, atol=0
    )
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)


@pytest.mark.parametrize(
    ("vocab_size", "logits_mode"),
    [(1031, False), (8192, True), (151937, False)],
)
@pytest.mark.compute_topk_scores
def test_negative_infinity_sampled_rank(
    candidate_logprob, device, vocab_size, logits_mode
):
    logits = torch.zeros((1, vocab_size), device=device)
    logits[0, 0] = float("-inf")
    sampled = torch.tensor([0], device=device)
    result = candidate_logprob.compute_topk_scores(
        logits, 0, sampled, logits_mode=logits_mode
    )
    expected_ids, expected_scores, expected_ranks = _pytorch_reference(
        logits, sampled, 0, (), (), logits_mode
    )

    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        result.selected_token_ranks, expected_ranks, rtol=0, atol=0
    )
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=0, atol=0)


@pytest.mark.compute_topk_scores
def test_extreme_finite_logits(candidate_logprob, device):
    logits_mode = False
    logits = torch.tensor([[1000.0, -1000.0, 999.0, -999.0, 0.0]], device=device)
    sampled = torch.tensor([1], device=device)
    result = candidate_logprob.compute_topk_scores(
        logits, 2, sampled, logits_mode=logits_mode
    )
    expected_ids, expected_scores, expected_ranks = _pytorch_reference(
        logits, sampled, 2, (), (), logits_mode
    )
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        result.selected_token_ranks, expected_ranks, rtol=0, atol=0
    )
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)


@pytest.mark.parametrize("cu_num_logits_kind", ["none", "list", "tensor"])
@pytest.mark.compute_topk_scores
def test_cu_num_logits_output_fields(candidate_logprob, device, cu_num_logits_kind):
    logits = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    sampled = torch.tensor([2], device=device)
    cu_num_logits = {
        "none": None,
        "list": [1],
        "tensor": torch.tensor([1], device=device),
    }[cu_num_logits_kind]
    result = candidate_logprob.compute_topk_scores(
        logits, 0, sampled, cu_num_logits=cu_num_logits
    )
    expected_ids, expected_scores, expected_ranks = _pytorch_reference(
        logits, sampled, 0, (), (), False
    )
    torch.testing.assert_close(result.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        result.selected_token_ranks, expected_ranks, rtol=0, atol=0
    )
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=2e-4, atol=2e-3)
    if cu_num_logits_kind == "tensor":
        assert result.cu_num_generated_tokens is None
        assert result.cu_num_generated_tokens_tensor is cu_num_logits
    else:
        assert result.cu_num_generated_tokens is cu_num_logits
        assert result.cu_num_generated_tokens_tensor is None
