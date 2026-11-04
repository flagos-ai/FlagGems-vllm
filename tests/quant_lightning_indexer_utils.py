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
 
from dataclasses import dataclass
from typing import Any
 
import torch
 
 
QUERY_HEADS = 64
HEAD_DIM = 128
KEY_BLOCKS = 4422
BLOCK_SIZE = 128
BLOCK_TABLE_WIDTH = 260
TOP_K = 512
INT64_MAX = (1 << 63) - 1
SEED = 20260915
QUERY_STRIDE = (QUERY_HEADS * HEAD_DIM, HEAD_DIM, 1)
KEY_STRIDE = (16640, 128, 128, 1)
WEIGHTS_STRIDE = (QUERY_HEADS, 1)
QUERY_SCALE_STRIDE = (QUERY_HEADS, 1)
KEY_SCALE_STRIDE = (8320, 1, 1)
QUICK_CASE_IDS = frozenset(("R1", "R3", "R6", "R8"))
 
 
@dataclass(frozen=True)
class QuantLightningIndexerCase:
    case_id: str
    scene: str
    query_t: int
    source_q_lengths: tuple[int, ...]
    local_q_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]
 
 
CASES = (
    QuantLightningIndexerCase("R1", "short prefill", 1, (8,), (1,), (8,)),
    QuantLightningIndexerCase("R2", "128 prefill", 16, (128,), (16,), (128,)),
    QuantLightningIndexerCase("R3", "2048 prefill", 256, (2048,), (256,), (2048,)),
    QuantLightningIndexerCase("R4", "8192 prefill", 1024, (8192,), (1024,), (8192,)),
    QuantLightningIndexerCase(
        "R5",
        "single MTP decode K=130",
        1,
        (2, 4, 6, 8),
        (1, 0, 0, 0),
        (130, 0, 0, 0),
    ),
    QuantLightningIndexerCase(
        "R6",
        "single MTP decode K=2050",
        1,
        (2, 4, 6, 8),
        (1, 0, 0, 0),
        (2050, 0, 0, 0),
    ),
    QuantLightningIndexerCase(
        "R7",
        "single MTP decode K=8194",
        1,
        (2, 4, 6, 8),
        (1, 0, 0, 0),
        (8194, 0, 0, 0),
    ),
    QuantLightningIndexerCase(
        "R8",
        "B3 chunked prefill",
        33,
        (2, 130, 258),
        (1, 16, 16),
        (130, 128, 128),
    ),
    QuantLightningIndexerCase(
        "R9",
        "B8 chunked prefill",
        81,
        (2, 4, 6, 134, 262, 390, 518, 646),
        (1, 0, 0, 16, 16, 16, 16, 16),
        (132, 130, 130, 128, 128, 128, 128, 128),
    ),
    QuantLightningIndexerCase(
        "R10",
        "B8 decode all valid",
        2,
        (2, 4, 6, 8, 10, 12, 14, 16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        (134, 132, 132, 130, 130, 130, 130, 130),
    ),
    QuantLightningIndexerCase(
        "R11",
        "B8 decode partially valid",
        2,
        (2, 4, 6, 8, 10, 12, 14, 16),
        (1, 1, 0, 0, 0, 0, 0, 0),
        (136, 136, 136, 136, 136, 0, 0, 0),
    ),
    QuantLightningIndexerCase(
        "R12",
        "single decode with mixed short and long requests",
        1,
        (1, 0),
        (1, 0),
        (130, 2050),
    ),
)
 
 
def _cumulative(lengths: tuple[int, ...]) -> list[int]:
    result: list[int] = []
    total = 0
    for length in lengths:
        total += int(length)
        result.append(total)
    return result
 
 
def _pages_for_raw_length(raw_length: int) -> int:
    return (int(raw_length) + 4 * BLOCK_SIZE - 1) // (4 * BLOCK_SIZE)
 
 
def _copy_to_device(
    source: torch.Tensor,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    device: torch.device | str,
) -> torch.Tensor:
    result = torch.empty_strided(shape, stride, dtype=source.dtype, device=device)
    result.copy_(source)
    return result
 
 
def _make_block_table(case: QuantLightningIndexerCase) -> tuple[torch.Tensor, list[int]]:
    table = torch.zeros((len(case.key_lengths), BLOCK_TABLE_WIDTH), dtype=torch.int32)
    active_blocks: list[int] = []
    next_block = 0
    for row, raw_length in enumerate(case.key_lengths):
        page_count = _pages_for_raw_length(raw_length)
        if next_block + page_count > KEY_BLOCKS:
            raise ValueError(f"{case.case_id} block table exceeds {KEY_BLOCKS} key pages")
        if page_count:
            table[row, :page_count] = torch.arange(
                next_block, next_block + page_count, dtype=torch.int32
            )
        active_blocks.append(page_count)
        next_block += page_count
    return table, active_blocks
 
 
def make_inputs(
    case: QuantLightningIndexerCase,
    device: torch.device | str,
    seed: int = SEED,
) -> dict[str, Any]:
    if sum(case.local_q_lengths) != case.query_t:
        raise ValueError(f"{case.case_id} local query lengths do not sum to T")
    if len(case.source_q_lengths) != len(case.key_lengths):
        raise ValueError(f"{case.case_id} source and key batch sizes differ")
 
    query_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    scale_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    key_generator = torch.Generator(device="cpu").manual_seed(SEED + 100)
    query = torch.randint(
        -8,
        9,
        (case.query_t, QUERY_HEADS, HEAD_DIM),
        dtype=torch.int8,
        generator=query_generator,
    )
    weights = torch.rand(
        (case.query_t, QUERY_HEADS), dtype=torch.float32, generator=scale_generator
    ).mul_(0.99).add_(0.01).to(torch.float16)
    query_scale = torch.rand(
        (case.query_t, QUERY_HEADS), dtype=torch.float32, generator=scale_generator
    ).mul_(0.09).add_(0.01).to(torch.float16)
    key = torch.randint(
        -8,
        9,
        (KEY_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.int8,
        generator=key_generator,
    )
    key_scale = torch.rand(
        (KEY_BLOCKS, BLOCK_SIZE, 1),
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(SEED + 101),
    ).mul_(0.09).add_(0.01).to(torch.float16)
    block_table, active_blocks = _make_block_table(case)
    local_query_ends = _cumulative(case.local_q_lengths)
    return {
        "query": _copy_to_device(query, query.shape, QUERY_STRIDE, device),
        "key": _copy_to_device(key, key.shape, KEY_STRIDE, device),
        "weights": _copy_to_device(weights, weights.shape, WEIGHTS_STRIDE, device),
        "query_dequant_scale": _copy_to_device(
            query_scale, query_scale.shape, QUERY_SCALE_STRIDE, device
        ),
        "key_dequant_scale": _copy_to_device(
            key_scale, key_scale.shape, KEY_SCALE_STRIDE, device
        ),
        "source_actual_seq_lengths_query": torch.tensor(
            case.source_q_lengths, dtype=torch.int32, device=device
        ),
        "actual_seq_lengths_query": torch.tensor(
            local_query_ends, dtype=torch.int32, device=device
        ),
        "actual_seq_lengths_key": torch.tensor(
            case.key_lengths, dtype=torch.int32, device=device
        ),
        "block_table": block_table.to(device),
        "metadata": torch.empty((1024,), dtype=torch.int32, device=device),
        "active_blocks": active_blocks,
    }
 
 
def _namespace_op(name: str) -> Any:
    namespace = getattr(torch.ops, "_C_ascend", None)
    return getattr(namespace, name, None) if namespace is not None else None
 
 
def register_official_ops() -> str:
    if _namespace_op("npu_vllm_quant_lightning_indexer") is not None:
        return "preloaded"
    errors: list[str] = []
    try:
        from vllm_ascend.utils import enable_custom_op
 
        if not enable_custom_op():
            raise RuntimeError("enable_custom_op returned false")
        import vllm_ascend.ops  # noqa: F401
 
        return "vllm_ascend"
    except Exception as exc:  # pragma: no cover - target-environment branch
        errors.append(f"vllm_ascend: {type(exc).__name__}: {exc}")
    try:
        import custom_ops  # noqa: F401
 
        return "custom_ops"
    except Exception as exc:  # pragma: no cover - target-environment branch
        errors.append(f"custom_ops: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not register official quant LightningIndexer ops: " + "; ".join(errors))
 
 
def _schema(op: Any) -> str:
    try:
        return str(op.default._schema)
    except Exception:
        return ""
 
 
def _schema_has(schema: str, name: str) -> bool:
    return f"{name}=" in schema or f", {name}" in schema
 
 
def official_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    op = _namespace_op("npu_vllm_quant_lightning_indexer")
    if op is None:
        raise RuntimeError("official quant LightningIndexer op is unavailable")
    schema = _schema(op)
    kwargs: dict[str, Any] = {
        "query_quant_mode": 0,
        "key_quant_mode": 0,
        "actual_seq_lengths_query": inputs["actual_seq_lengths_query"],
        "actual_seq_lengths_key": inputs["actual_seq_lengths_key"],
        "block_table": inputs["block_table"],
        "metadata": inputs["metadata"],
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "sparse_count": TOP_K,
        "sparse_mode": 3,
        "pre_tokens": INT64_MAX,
        "next_tokens": INT64_MAX,
        "cmp_ratio": 4,
    }
    if _schema_has(schema, "return_values"):
        kwargs["return_values"] = False
    elif _schema_has(schema, "return_value") or not schema:
        kwargs["return_value"] = False
    if _schema_has(schema, "stride") or not schema:
        kwargs["stride"] = int(inputs["key"].stride(0))
    if _schema_has(schema, "scale_stride") or not schema:
        kwargs["scale_stride"] = int(inputs["key_dequant_scale"].stride(0))
    return kwargs
 
 
def call_official(inputs: dict[str, Any]) -> Any:
    op = _namespace_op("npu_vllm_quant_lightning_indexer")
    if op is None:
        raise RuntimeError("official quant LightningIndexer op is unavailable")
    return op(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=inputs["key_dequant_scale"],
        **official_kwargs(inputs),
    )
 
 
def call_candidate(module: Any, case: QuantLightningIndexerCase, inputs: dict[str, Any]) -> Any:
    return module.quant_lightning_indexer(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_dequant_scale"],
        key_dequant_scale=inputs["key_dequant_scale"],
        actual_seq_lengths_query=inputs["source_actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        block_table=inputs["block_table"],
        metadata=inputs["metadata"],
        local_query_lengths=case.local_q_lengths,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOP_K,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=4,
        return_value=False,
        stride=int(inputs["key"].stride(0)),
        scale_stride=int(inputs["key_dequant_scale"].stride(0)),
    )
 
 
def _indices(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output
 
 
def sorted_indices(output: Any) -> torch.Tensor:
    indices = _indices(output)
    if tuple(indices.shape[1:]) != (1, TOP_K):
        raise AssertionError(f"expected [T, 1, {TOP_K}], got {tuple(indices.shape)}")
    return torch.sort(indices.detach().cpu(), dim=-1).values
 
 
def assert_matches(official_output: Any, candidate_output: Any) -> None:
    official_indices = _indices(official_output)
    candidate_indices = _indices(candidate_output)
    assert tuple(candidate_indices.shape) == tuple(official_indices.shape)
    assert official_indices.dtype == candidate_indices.dtype == torch.int32
    torch.testing.assert_close(
        sorted_indices(candidate_output),
        sorted_indices(official_output),
        atol=0,
        rtol=0,
    )
    assert len(candidate_output) == 2
    assert candidate_output[1].dtype == torch.float32
    assert candidate_output[1].numel() == 0
 
 
def prepare_metadata(inputs: dict[str, Any], case: QuantLightningIndexerCase) -> None:
    metadata_op = _namespace_op("npu_vllm_quant_lightning_indexer_metadata")
    if metadata_op is None:
        raise RuntimeError("official quant LightningIndexer metadata op is unavailable")
    inputs["metadata"] = metadata_op(
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_key=inputs["actual_seq_lengths_key"],
        num_heads_q=QUERY_HEADS,
        num_heads_k=1,
        head_dim=HEAD_DIM,
        query_quant_mode=0,
        key_quant_mode=0,
        batch_size=len(case.key_lengths),
        max_seqlen_q=max(case.local_q_lengths),
        max_seqlen_k=max(case.key_lengths),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOP_K,
        sparse_mode=3,
        pre_tokens=INT64_MAX,
        next_tokens=INT64_MAX,
        cmp_ratio=4,
        device=str(inputs["query"].device),
    )
