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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

QUERY_HEADS = 64
HEAD_DIM = 128
KEY_BLOCKS = 4422
BLOCK_SIZE = 128
BLOCK_TABLE_WIDTH = 260
TOP_K = 512
QUICK_CASE_IDS = frozenset(("R1", "R3", "R6", "R8"))


@dataclass(frozen=True)
class LightningIndexerCase:
    case_id: str
    scene: str
    query_t: int
    local_q_lengths: tuple[int, ...]
    global_q_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]


CASES = (
    LightningIndexerCase("R1", "short prefill", 1, (1,), (8,), (8,)),
    LightningIndexerCase("R2", "128 prefill", 16, (16,), (128,), (128,)),
    LightningIndexerCase("R3", "2048 prefill", 256, (256,), (2048,), (2048,)),
    LightningIndexerCase("R4", "8192 prefill", 1024, (1024,), (8192,), (8192,)),
    LightningIndexerCase(
        "R5", "single MTP decode K=130", 1, (1, 0, 0, 0), (2, 2, 2, 2), (130, 0, 0, 0)
    ),
    LightningIndexerCase(
        "R6", "single MTP decode K=2050", 1, (1, 0, 0, 0), (2, 2, 2, 2), (2050, 0, 0, 0)
    ),
    LightningIndexerCase(
        "R7", "single MTP decode K=8194", 1, (1, 0, 0, 0), (2, 2, 2, 2), (8194, 0, 0, 0)
    ),
    LightningIndexerCase(
        "R8", "B3 chunked prefill", 33, (1, 16, 16), (2, 128, 128), (130, 128, 128)
    ),
    LightningIndexerCase(
        "R9",
        "B8 chunked prefill",
        81,
        (1, 0, 0, 16, 16, 16, 16, 16),
        (2, 2, 2, 128, 128, 128, 128, 128),
        (132, 130, 130, 128, 128, 128, 128, 128),
    ),
    LightningIndexerCase(
        "R10",
        "B8 decode all valid",
        2,
        (1, 1, 0, 0, 0, 0, 0, 0),
        (2, 2, 2, 2, 2, 2, 2, 2),
        (134, 132, 132, 130, 130, 130, 130, 130),
    ),
    LightningIndexerCase(
        "R11",
        "B8 decode partially valid",
        2,
        (1, 1, 0, 0, 0, 0, 0, 0),
        (2, 2, 2, 2, 2, 2, 2, 2),
        (136, 136, 136, 136, 136, 0, 0, 0),
    ),
    LightningIndexerCase(
        "R12",
        "single decode with mixed short and long requests",
        1,
        (1, 0),
        (1, 0),
        (130, 2050),
    ),
)


def _load_vllm_ascend_reference():
    try:
        import torch_npu
        import vllm_ascend  # noqa: F401

        return torch_npu.npu_lightning_indexer, None
    except Exception as exc:
        return None, exc


VLLM_ASCEND_REFERENCE, VLLM_ASCEND_IMPORT_ERROR = _load_vllm_ascend_reference()
HAS_VLLM_ASCEND_REFERENCE = VLLM_ASCEND_REFERENCE is not None


def _cumulative(lengths: tuple[int, ...]) -> list[int]:
    result = []
    total = 0
    for length in lengths:
        total += int(length)
        result.append(total)
    return result


def make_inputs(case: LightningIndexerCase, device: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(3)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(3)

    query = torch.empty(
        (case.query_t, QUERY_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-8, 8)
    key = torch.empty(
        (KEY_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-8, 8)
    weights = torch.empty(
        (case.query_t, QUERY_HEADS),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-1, 1)
    query_lengths = torch.tensor(
        _cumulative(case.local_q_lengths),
        dtype=torch.int32,
        device=device,
    )
    key_lengths = torch.tensor(case.key_lengths, dtype=torch.int32, device=device)
    block_table = torch.zeros(
        (len(case.key_lengths), BLOCK_TABLE_WIDTH),
        dtype=torch.int32,
        device=device,
    )

    next_block = 0
    for request, key_length in enumerate(case.key_lengths):
        block_count = math.ceil(key_length / BLOCK_SIZE)
        if next_block + block_count > KEY_BLOCKS:
            raise ValueError("block table exceeds the allocated key cache")
        if block_count:
            block_table[request, :block_count] = torch.arange(
                next_block,
                next_block + block_count,
                dtype=torch.int32,
                device=device,
            )
        next_block += block_count

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "actual_seq_lengths_query": query_lengths,
        "actual_seq_lengths_key": key_lengths,
        "block_table": block_table,
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "sparse_count": TOP_K,
        "sparse_mode": 3,
    }


def torch_reference_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_table: torch.Tensor,
    layout_query: str = "TND",
    layout_key: str = "PA_BSND",
    sparse_count: int = TOP_K,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
):
    if layout_query != "TND" or layout_key != "PA_BSND" or sparse_mode != 3:
        raise NotImplementedError("Torch reference only covers TND/PA_BSND mode 3")
    if pre_tokens != 9223372036854775807 or next_tokens != 9223372036854775807:
        raise NotImplementedError("Torch reference only covers default token windows")

    output = torch.full(
        (query.shape[0], 1, sparse_count),
        -1,
        dtype=torch.int32,
        device=query.device,
    )
    query_ends = [int(value) for value in actual_seq_lengths_query.cpu().tolist()]
    key_lengths = [int(value) for value in actual_seq_lengths_key.cpu().tolist()]
    query_begin = 0
    for request, query_end in enumerate(query_ends):
        key_length = key_lengths[request]
        block_count = math.ceil(key_length / key.shape[1])
        if block_count:
            physical_blocks = block_table[request, :block_count].to(torch.long)
            logical_key = key.index_select(0, physical_blocks).reshape(
                -1, key.shape[-1]
            )
            logical_key = logical_key[:key_length].to(torch.float32)
        else:
            logical_key = key.new_empty((0, key.shape[-1]), dtype=torch.float32)

        for token in range(query_begin, query_end):
            active_key_length = key_length - (query_end - token) + 1
            if active_key_length <= 0:
                continue
            if active_key_length <= sparse_count:
                output[token, 0, :active_key_length] = torch.arange(
                    active_key_length,
                    dtype=torch.int32,
                    device=query.device,
                )
                continue

            qk = torch.matmul(
                query[token].to(torch.float32),
                logical_key[:active_key_length].transpose(0, 1),
            )
            scores = (
                torch.relu(qk) * weights[token].to(torch.float32).unsqueeze(1)
            ).sum(dim=0)
            output[token, 0] = torch.topk(scores, sparse_count).indices.to(torch.int32)
        query_begin = query_end
    return output, torch.empty((0,), dtype=query.dtype, device=query.device)


def get_reference() -> tuple[Callable, str]:
    if VLLM_ASCEND_REFERENCE is not None:
        return VLLM_ASCEND_REFERENCE, "vllm-ascend/torch_npu.npu_lightning_indexer"
    return torch_reference_lightning_indexer, "torch-composite"


def is_torch_fallback_feasible(case: LightningIndexerCase) -> bool:
    return max(case.key_lengths) <= TOP_K or case.query_t == 1


def extract_indices(output) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def assert_index_multisets_equal(actual, expected) -> None:
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise AssertionError("candidate must return an (indices, values) pair")
    if not isinstance(expected, (tuple, list)) or len(expected) != 2:
        raise AssertionError("reference must return an (indices, values) pair")

    actual_values = actual[1]
    expected_values = expected[1]
    if (
        not isinstance(actual_values, torch.Tensor)
        or actual_values.shape != expected_values.shape
        or actual_values.dtype != expected_values.dtype
        or actual_values.device != expected_values.device
    ):
        raise AssertionError(
            "value output mismatch: "
            f"{type(actual_values)}/{getattr(actual_values, 'shape', None)}/"
            f"{getattr(actual_values, 'dtype', None)}/"
            f"{getattr(actual_values, 'device', None)} != "
            f"{expected_values.shape}/{expected_values.dtype}/{expected_values.device}"
        )

    actual = extract_indices(actual)
    expected = extract_indices(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"output mismatch: {actual.shape}/{actual.dtype} != "
            f"{expected.shape}/{expected.dtype}"
        )
    actual_sorted = torch.sort(actual.detach().cpu(), dim=-1).values
    expected_sorted = torch.sort(expected.detach().cpu(), dim=-1).values
    torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0, atol=0)
