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

"""Benchmark the complete compute_topk_scores call."""

import importlib
import importlib.util
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import runtime

from . import base

try:
    from vllm.v1.worker.gpu.sample import logprob as baseline_logprob
except ImportError:
    if importlib.util.find_spec("vllm") is not None:
        raise
    baseline_logprob = None


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    vocab_size: int
    k: int
    custom_count: int = 0
    logits_mode: bool = False
    dtype: torch.dtype = torch.float32


CASES = (
    Case("decode-b1", 1, 151936, 5),
    Case("decode-b16", 16, 151936, 5),
    Case("decode-b64", 64, 151936, 5),
    Case("decode-b256", 256, 151936, 5),
    Case("decode-b16-fp16", 16, 151936, 5, dtype=torch.float16),
    Case("decode-b16-bf16", 16, 151936, 5, dtype=torch.bfloat16),
    Case("medium-k-b1", 1, 151936, 20),
    Case("medium-k-b64", 64, 151936, 20),
    Case("sample-only-b1", 1, 151936, 0),
    Case("sample-only-b64", 64, 151936, 0),
    Case("custom-b16", 16, 151936, 5, custom_count=8),
    Case("custom-b64", 64, 151936, 5, custom_count=8),
    Case("logits-b1", 1, 151936, 5, logits_mode=True),
    Case("logits-b64", 64, 151936, 5, logits_mode=True),
    Case("logits-k0-b1", 1, 151936, 0, logits_mode=True),
    Case("small-vocab-b1", 1, 32000, 5),
    Case("small-vocab-b64", 64, 32000, 5),
    Case("wide-k", 1, 32000, 1025),
    Case("full-vocab-logprob", 1, 32000, 32000),
    Case("full-vocab-logprob-b64", 64, 32000, 32000),
    Case("full-vocab-logits", 1, 32000, 32000, logits_mode=True),
    Case("full-vocab-logits-b64", 64, 32000, 32000, logits_mode=True),
)


@pytest.fixture(scope="module", autouse=True)
def report_baseline_unavailable():
    if baseline_logprob is None:
        print("vLLM baseline unavailable; baseline latency and speedup are N/A.")


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(flaggems_vllm.device)


@pytest.fixture(scope="module")
def candidate_logprob():
    return importlib.import_module(flaggems_vllm.compute_topk_scores.__module__)


def _make_inputs(candidate, case: Case, device: torch.device, dtype: torch.dtype):
    torch.manual_seed(2026)
    logits = torch.randn(
        (case.batch_size, case.vocab_size), device=device, dtype=torch.float32
    ).to(dtype)
    sampled = torch.tensor(
        [(7 + row * 31) % case.vocab_size for row in range(case.batch_size)],
        dtype=torch.int64,
        device=device,
    )
    if not case.custom_count:
        return logits, sampled, None, None

    state = candidate.LogprobTokenIdsState(case.batch_size, device)
    requested_ids = list(range(11, 11 + case.custom_count))
    for row in range(case.batch_size):
        ids = requested_ids if row % 2 else []
        state.add_request(row, SimpleNamespace(logprob_token_ids=ids))
    state.apply_staged_writes()
    mapping = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    return logits, sampled, state, mapping


class LogprobBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "B, V, K, custom_count, logits_mode"

    def __init__(self, case, candidate, torch_op, device):
        self.case = case
        self.candidate = candidate
        super().__init__(
            op_name="compute_topk_scores",
            torch_op=torch_op,
            gems_op=candidate.compute_topk_scores,
            dtypes=[case.dtype],
        )
        self.device = device

    def set_shapes(self, shape_file_path=None):
        case = self.case
        self.shapes = [
            (
                case.batch_size,
                case.vocab_size,
                case.k,
                case.custom_count,
                case.logits_mode,
            )
        ]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, dtype):
        case = self.case
        logits, sampled, state, mapping = _make_inputs(
            self.candidate, case, self.device, dtype
        )
        yield logits, case.k, sampled, {
            "logprob_token_ids_state": state,
            "expanded_idx_mapping": mapping,
            "max_per_req_token_ids": case.custom_count,
            "logits_mode": case.logits_mode,
        }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.compute_topk_scores
def test_compute_topk_scores_perf(candidate_logprob, device, case):
    dtype = case.dtype
    if dtype == torch.bfloat16 and not runtime.device.support_bf16:
        pytest.skip("bfloat16 is unavailable on this backend")
    benchmark = LogprobBenchmark(
        case,
        candidate_logprob,
        baseline_logprob.compute_topk_scores if baseline_logprob is not None else None,
        device,
    )
    benchmark.run()
