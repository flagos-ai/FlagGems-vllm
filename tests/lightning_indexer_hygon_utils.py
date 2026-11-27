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

"""Reference implementations and inference cases for Hygon LightningIndexer."""

from __future__ import annotations

import itertools
import math

import torch

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_CASE_DEFAULTS = dict(
    heads=64,
    dim=128,
    key_blocks=512,
    table_width=260,
    block_size=128,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    return_value=False,
    query_dtype="bfloat16",
    key_dtype="bfloat16",
    weights_dtype="bfloat16",
)

# 30 LLM real-inference TND LightningIndexer cases covering DeepSeek-V4-Flash
# (TopK512), DeepSeek-V3.2 (TopK2048, H64), GLM-5/5.2/5.3 (TopK2048, H32),
# TP/CP sharding (H8/H16, TND ragged, return_value=True, sparse_mode=0/3),
# FP16/BF16, FP32 weights, and long-context TopK256..8192.
_RAW_CASES_30 = {
    "R1": dict(
        scene="DeepSeek-V4 short prefill Top512",
        query_t=1,
        local_q_lengths=[1],
        key_lengths=[8],
    ),
    "R2": dict(
        scene="DeepSeek-V4 128 prefill Top512",
        query_t=16,
        local_q_lengths=[16],
        key_lengths=[128],
    ),
    "R3": dict(
        scene="DeepSeek-V4 2048 prefill Top512",
        query_t=256,
        local_q_lengths=[256],
        key_lengths=[2048],
    ),
    "R4": dict(
        scene="DeepSeek-V4 8192 prefill Top512",
        query_t=1024,
        local_q_lengths=[1024],
        key_lengths=[8192],
    ),
    "R5": dict(
        scene="DeepSeek-V4 MTP decode K=130 Top512",
        query_t=1,
        local_q_lengths=[1, 0, 0, 0],
        key_lengths=[130, 0, 0, 0],
    ),
    "R6": dict(
        scene="DeepSeek-V4 MTP decode K=2050 Top512",
        query_t=1,
        local_q_lengths=[1, 0, 0, 0],
        key_lengths=[2050, 0, 0, 0],
    ),
    "R7": dict(
        scene="DeepSeek-V4 MTP decode K=8194 Top512",
        query_t=1,
        local_q_lengths=[1, 0, 0, 0],
        key_lengths=[8194, 0, 0, 0],
    ),
    "R8": dict(
        scene="DeepSeek-V4 B3 chunked prefill Top512",
        query_t=33,
        local_q_lengths=[1, 16, 16],
        key_lengths=[130, 128, 128],
    ),
    "R9": dict(
        scene="DeepSeek-V4 B8 chunked prefill Top512",
        query_t=81,
        local_q_lengths=[1, 0, 0, 16, 16, 16, 16, 16],
        key_lengths=[132, 130, 130, 128, 128, 128, 128, 128],
    ),
    "R10": dict(
        scene="DeepSeek-V4 B8 decode partially valid Top512",
        query_t=2,
        local_q_lengths=[1, 1, 0, 0, 0, 0, 0, 0],
        key_lengths=[136, 136, 136, 136, 136, 0, 0, 0],
    ),
    "R11": dict(
        scene="DeepSeek-V3.2 decode K=4096 page64 Top2048",
        query_t=1,
        heads=64,
        block_size=64,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[1],
        key_lengths=[4096],
    ),
    "R12": dict(
        scene="DeepSeek-V3.2 decode K=16384 page128 Top2048",
        query_t=1,
        heads=64,
        block_size=128,
        sparse_count=2048,
        weights_dtype="bfloat16",
        local_q_lengths=[1],
        key_lengths=[16384],
    ),
    "R13": dict(
        scene="DeepSeek-V3.2 decode K=32768 page128 Top2048",
        query_t=1,
        heads=64,
        block_size=128,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[1],
        key_lengths=[32768],
    ),
    "R14": dict(
        scene="DeepSeek-V3.2 B8 batched decode page64 Top2048",
        query_t=8,
        heads=64,
        block_size=64,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[1] * 8,
        key_lengths=[1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192],
    ),
    "R15": dict(
        scene="DeepSeek-V3.2 B4 MTP target-verify Top2048",
        query_t=8,
        heads=64,
        block_size=128,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[2, 2, 2, 2],
        key_lengths=[2050, 4098, 6146, 8194],
    ),
    "R16": dict(
        scene="DeepSeek-V3.2 B3 chunked prefill Top2048",
        query_t=193,
        heads=64,
        block_size=128,
        sparse_count=2048,
        weights_dtype="bfloat16",
        local_q_lengths=[1, 64, 128],
        key_lengths=[4097, 2048, 4096],
    ),
    "R17": dict(
        scene="DeepSeek-V3.2 ragged TND prefill Q512 K4096 Top2048",
        query_t=512,
        heads=64,
        layout_key="TND",
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[512],
        key_lengths=[4096],
    ),
    "R18": dict(
        scene="DeepSeek-V3.2 B2 ragged TND extend prefill Top2048",
        query_t=384,
        heads=64,
        layout_key="TND",
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[128, 256],
        key_lengths=[2048, 4096],
    ),
    "R19": dict(
        scene="GLM-5 H32 decode K=4096 page64 Top2048",
        query_t=1,
        heads=32,
        block_size=64,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[1],
        key_lengths=[4096],
    ),
    "R20": dict(
        scene="GLM-5 H32 B8 batched decode page64 Top2048",
        query_t=8,
        heads=32,
        block_size=64,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[1] * 8,
        key_lengths=[512, 1024, 2048, 2560, 3072, 4096, 6144, 8192],
    ),
    "R21": dict(
        scene="GLM-5 H32 B4 MTP q4 verify page128 Top2048",
        query_t=16,
        heads=32,
        block_size=128,
        sparse_count=2048,
        weights_dtype="bfloat16",
        local_q_lengths=[4, 4, 4, 4],
        key_lengths=[2052, 4100, 6148, 8196],
    ),
    "R22": dict(
        scene="GLM-5 H32 B4 chunked prefill page64 Top2048",
        query_t=194,
        heads=32,
        block_size=64,
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[1, 1, 64, 128],
        key_lengths=[2049, 4097, 2048, 4096],
    ),
    "R23": dict(
        scene="GLM-5 H32 B2 ragged TND prefill Top2048",
        query_t=384,
        heads=32,
        layout_key="TND",
        sparse_count=2048,
        weights_dtype="float32",
        local_q_lengths=[128, 256],
        key_lengths=[1536, 3072],
    ),
    "R24": dict(
        scene="GLM-5 TP2 / CANN A2 B18 Q3 H16 K3016-3072 Top2048",
        query_t=54,
        heads=16,
        block_size=128,
        sparse_count=2048,
        weights_dtype="bfloat16",
        local_q_lengths=[3] * 18,
        key_lengths=[3016] * 17 + [3072],
    ),
    "R25": dict(
        scene="TP8 H8 FP16 B8 decode page64 Top1024",
        query_t=8,
        heads=8,
        block_size=64,
        sparse_count=1024,
        query_dtype="float16",
        key_dtype="float16",
        weights_dtype="float32",
        local_q_lengths=[1] * 8,
        key_lengths=[2048, 4096] * 4,
    ),
    "R26": dict(
        scene="FP16 H32 B2 prefill page128 Top256",
        query_t=96,
        heads=32,
        block_size=128,
        sparse_count=256,
        query_dtype="float16",
        key_dtype="float16",
        weights_dtype="float16",
        local_q_lengths=[32, 64],
        key_lengths=[1024, 2048],
    ),
    "R27": dict(
        scene="Prefix KV chunk mode0 H64 B2 Top4096",
        query_t=32,
        heads=64,
        block_size=128,
        sparse_count=4096,
        sparse_mode=0,
        weights_dtype="float32",
        local_q_lengths=[16, 16],
        key_lengths=[6144, 8192],
    ),
    "R28": dict(
        scene="CP prefix shard TND return_value mode0 H64 Top1024",
        query_t=48,
        heads=64,
        layout_key="TND",
        sparse_count=1024,
        sparse_mode=0,
        return_value=True,
        weights_dtype="float32",
        local_q_lengths=[16, 32],
        key_lengths=[2048, 4096],
    ),
    "R29": dict(
        scene="CP causal shard FP16 TND return_value mode3 H32 Top2048",
        query_t=48,
        heads=32,
        layout_key="TND",
        sparse_count=2048,
        sparse_mode=3,
        return_value=True,
        query_dtype="float16",
        key_dtype="float16",
        weights_dtype="float32",
        local_q_lengths=[16, 32],
        key_lengths=[2048, 4096],
    ),
    "R30": dict(
        scene="Long-context B2 decode H64 K12k-16k Top8192",
        query_t=2,
        heads=64,
        block_size=128,
        sparse_count=8192,
        sparse_mode=3,
        weights_dtype="float32",
        local_q_lengths=[1, 1],
        key_lengths=[12288, 16384],
    ),
}

CASES_30 = {name: _CASE_DEFAULTS | spec for name, spec in _RAW_CASES_30.items()}


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return _DTYPE_BY_NAME[name]


def build_case_inputs(case: dict, device) -> dict:
    """Build deterministic device inputs for one 30-case LightningIndexer workload."""
    generator = torch.Generator().manual_seed(3)
    t = case["query_t"]
    h = case["heads"]
    d = case["dim"]
    q_dtype = parse_dtype(case["query_dtype"])
    k_dtype = parse_dtype(case["key_dtype"])
    w_dtype = parse_dtype(case["weights_dtype"])
    layout_query = case["layout_query"]
    layout_key = case["layout_key"]
    sparse_count = int(case["sparse_count"])
    sparse_mode = int(case["sparse_mode"])
    return_value = bool(case["return_value"])

    query = (
        torch.empty((t, h, d), dtype=torch.float32)
        .uniform_(-8, 8, generator=generator)
        .to(q_dtype)
    )
    weights = (
        torch.empty((t, h), dtype=torch.float32)
        .uniform_(-1, 1, generator=generator)
        .to(w_dtype)
    )
    ends = list(itertools.accumulate(case["local_q_lengths"]))
    key_lengths = [int(x) for x in case["key_lengths"]]

    if layout_key == "PA_BSND":
        block_size = int(case["block_size"])
        needed_pages = sum(
            (k_len + block_size - 1) // block_size for k_len in key_lengths
        )
        max_req_pages = max(
            ((k_len + block_size - 1) // block_size for k_len in key_lengths),
            default=1,
        )
        pages = max(1, int(case["key_blocks"]), needed_pages)
        table_width = max(1, int(case["table_width"]), max_req_pages)
        key = (
            torch.empty((pages, block_size, 1, d), dtype=torch.float32)
            .uniform_(-8, 8, generator=generator)
            .to(k_dtype)
        )
        table = torch.zeros((len(ends), table_width), dtype=torch.int32)
        cursor = 0
        for request, length in enumerate(key_lengths):
            count = (length + block_size - 1) // block_size
            if count:
                table[request, :count] = torch.arange(
                    cursor, cursor + count, dtype=torch.int32
                )
                cursor += count
        seq_k = torch.tensor(key_lengths, dtype=torch.int32, device=device)
        block_table = table.to(device)
    else:
        total_k = sum(key_lengths)
        raw_k = (
            torch.empty((max(1, total_k), 1, d), dtype=torch.float32)
            .uniform_(-8, 8, generator=generator)
            .to(k_dtype)
        )
        key = raw_k[:total_k]
        k_ends = list(itertools.accumulate(key_lengths))
        seq_k = torch.tensor(k_ends, dtype=torch.int32, device=device)
        block_table = None

    return {
        "query": query.to(device),
        "key": key.to(device),
        "weights": weights.to(device),
        "actual_seq_lengths_query": torch.tensor(
            ends, dtype=torch.int32, device=device
        ),
        "actual_seq_lengths_key": seq_k,
        "block_table": block_table,
        "layout_query": layout_query,
        "layout_key": layout_key,
        "sparse_count": sparse_count,
        "sparse_mode": sparse_mode,
        "return_value": return_value,
    }


def pytorch_indexer(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table=None,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    return_value=False,
):
    """Independent PyTorch reference covering the full non-BSND TND specification."""
    del pre_tokens, next_tokens
    if layout_query != "TND" or layout_key not in {"PA_BSND", "TND"}:
        raise NotImplementedError(
            "PyTorch reference supports TND query and PA_BSND/TND key"
        )

    ends = actual_seq_lengths_query.cpu().tolist()
    raw_k = actual_seq_lengths_key.cpu().tolist()
    output = torch.full(
        (query.shape[0], 1, sparse_count), -1, dtype=torch.int32, device=query.device
    )
    values = (
        torch.full(
            (query.shape[0], 1, sparse_count),
            -float("inf"),
            dtype=query.dtype,
            device=query.device,
        )
        if return_value
        else torch.empty((0,), dtype=query.dtype, device=query.device)
    )
    start = 0
    k_start = 0
    for request, (end, k_meta) in enumerate(zip(ends, raw_k)):
        rows = end - start
        length = k_meta if layout_key == "PA_BSND" else (k_meta - k_start)
        if not rows or length <= 0:
            start = end
            if layout_key == "TND":
                k_start = k_meta
            continue

        if layout_key == "PA_BSND":
            pages = block_table[
                request, : (length + key.shape[1] - 1) // key.shape[1]
            ].long()
            cache = (
                key.index_select(0, pages).reshape(-1, key.shape[-1])[:length].float()
            )
        else:
            cache = key[k_start:k_meta, 0, :].float()
            k_start = k_meta

        q = query[start:end].float()
        w = weights[start:end].float()
        qk = torch.einsum("shd,pd->shp", q, cache)
        scores = (torch.relu(qk) * w[:, :, None]).sum(dim=1)
        if sparse_mode == 3:
            valid = (length - rows + torch.arange(rows, device=query.device) + 1).clamp(
                min=0
            )
        else:
            valid = torch.full((rows,), length, dtype=torch.long, device=query.device)
        mask = torch.arange(length, device=query.device)[None, :] >= valid[:, None]
        scores.masked_fill_(mask, -float("inf"))
        k = min(sparse_count, length)
        if return_value:
            top_vals, indices = torch.topk(scores, k, dim=-1, largest=True, sorted=True)
            invalid = torch.arange(k, device=query.device)[None, :] >= valid[:, None]
            indices = indices.masked_fill(invalid, -1)
            top_vals = top_vals.masked_fill(invalid, -float("inf"))
            output[start:end, 0, :k] = indices.int()
            values[start:end, 0, :k] = top_vals.to(query.dtype)
        else:
            indices = torch.topk(scores, k, dim=-1, sorted=False).indices
            short_rows = valid <= sparse_count
            if bool(short_rows.any()):
                seq_idx = torch.arange(k, device=query.device)[None, :].expand_as(
                    indices
                )
                indices = torch.where(short_rows[:, None], seq_idx, indices)
            indices.masked_fill_(indices >= valid[:, None], -1)
            output[start:end, 0, :k] = indices.int()
        start = end
    return output, values


def _compute_single_row_scores(inputs_dict: dict, row: int) -> tuple[int, torch.Tensor]:
    query = inputs_dict["query"]
    key = inputs_dict["key"]
    weights = inputs_dict["weights"]
    q_ends = [
        int(v) for v in inputs_dict["actual_seq_lengths_query"].detach().cpu().tolist()
    ]
    k_meta = [
        int(v) for v in inputs_dict["actual_seq_lengths_key"].detach().cpu().tolist()
    ]
    layout_key = inputs_dict.get("layout_key", "PA_BSND")
    sparse_mode = int(inputs_dict.get("sparse_mode", 3))

    req = 0
    k_begin = 0
    while req < len(q_ends) - 1 and row >= q_ends[req]:
        if layout_key == "TND":
            k_begin = k_meta[req]
        req += 1
    q_end = q_ends[req]
    k_len = k_meta[req] if layout_key == "PA_BSND" else (k_meta[req] - k_begin)
    valid = max(0, (k_len - (q_end - row) + 1) if sparse_mode == 3 else k_len)
    if valid == 0:
        return 0, torch.empty((0,), dtype=torch.float32)

    if layout_key == "PA_BSND":
        block_table = inputs_dict["block_table"]
        block_count = math.ceil(valid / key.shape[1])
        pages = block_table[req, :block_count].detach().cpu().long()
        k_cpu = (
            key.detach()
            .cpu()
            .index_select(0, pages)
            .reshape(-1, key.shape[-1])[:valid]
            .float()
        )
    else:
        k_cpu = key.detach().cpu()[k_begin : k_begin + valid, 0, :].float()

    q_cpu = query[row].detach().cpu().float()
    w_cpu = weights[row].detach().cpu().float()
    qk = torch.matmul(q_cpu, k_cpu.transpose(0, 1))
    scores = (torch.relu(qk) * w_cpu.unsqueeze(1)).sum(dim=0)
    return valid, scores


def assert_index_multisets_equal(
    actual, expected, *, inputs_dict: dict | None = None
) -> None:
    """Check (indices, values) outputs with 1-ULP cutoff tie-break tolerance."""
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise AssertionError("candidate must return an (indices, values) pair")
    if not isinstance(expected, (tuple, list)) or len(expected) != 2:
        raise AssertionError("reference must return an (indices, values) pair")

    actual_values, expected_values = actual[1], expected[1]
    if (
        not isinstance(actual_values, torch.Tensor)
        or actual_values.shape != expected_values.shape
        or actual_values.dtype != expected_values.dtype
        or actual_values.device != expected_values.device
    ):
        raise AssertionError("sparse_values metadata mismatch")

    if actual_values.numel() > 0:
        act_v = actual_values.detach().cpu().float()
        exp_v = expected_values.detach().cpu().float()
        act_neginf = torch.isneginf(act_v)
        exp_neginf = torch.isneginf(exp_v)
        if not torch.equal(act_neginf, exp_neginf):
            raise AssertionError("sparse_values -inf padding mask mismatch")
        finite = ~exp_neginf
        if bool(finite.any()):
            tol = 2e-2 if actual_values.dtype == torch.bfloat16 else 5e-3
            torch.testing.assert_close(act_v[finite], exp_v[finite], rtol=tol, atol=tol)

    actual_idx, expected_idx = actual[0], expected[0]
    if actual_idx.shape != expected_idx.shape or actual_idx.dtype != expected_idx.dtype:
        raise AssertionError("sparse_indices metadata mismatch")

    actual_sorted = torch.sort(actual_idx.detach().cpu(), dim=-1).values
    expected_sorted = torch.sort(expected_idx.detach().cpu(), dim=-1).values
    if torch.equal(actual_sorted, expected_sorted) or inputs_dict is None:
        torch.testing.assert_close(actual_sorted, expected_sorted, rtol=0, atol=0)
        return

    rows, _, sparse_count = actual_sorted.shape
    for r in range(rows):
        a_row = actual_sorted[r, 0]
        e_row = expected_sorted[r, 0]
        if torch.equal(a_row, e_row):
            continue
        valid, scores = _compute_single_row_scores(inputs_dict, r)
        take = min(sparse_count, valid)
        pad = sparse_count - take
        if pad > 0:
            if not bool((a_row[:pad] == -1).all()) or not bool(
                (e_row[:pad] == -1).all()
            ):
                torch.testing.assert_close(a_row, e_row, rtol=0, atol=0)
        a_valid = a_row[pad:]
        e_valid = e_row[pad:]
        if (
            bool((a_valid < 0).any())
            or bool((a_valid >= valid).any())
            or torch.unique(a_valid).numel() != take
        ):
            torch.testing.assert_close(a_row, e_row, rtol=0, atol=0)
        a_scores = torch.sort(
            scores.index_select(0, a_valid.long()), descending=True
        ).values
        e_scores = torch.sort(
            scores.index_select(0, e_valid.long()), descending=True
        ).values
        torch.testing.assert_close(a_scores, e_scores, rtol=1e-3, atol=1e-2)


def _paged_inputs(
    query_lengths,
    key_lengths,
    heads,
    device,
    *,
    page=128,
    q_dtype=torch.bfloat16,
    w_dtype=torch.bfloat16,
    strided=False,
    table_width=None,
):
    torch.manual_seed(410)
    tokens = sum(query_lengths)
    dim = 128
    nblocks = max(1, sum(math.ceil(length / page) for length in key_lengths))
    q = torch.randn((tokens, heads, dim), dtype=torch.float32).to(q_dtype).to(device)
    key = (
        torch.randn((nblocks, page, 1, dim), dtype=torch.float32).to(q_dtype).to(device)
    )
    weights = torch.randn((tokens, heads), dtype=torch.float32).to(w_dtype).to(device)
    if strided:
        q_storage = torch.empty((tokens, heads, dim * 2), dtype=q.dtype, device=device)
        q_storage[..., ::2] = q
        q = q_storage[..., ::2]
        w_storage = torch.empty((tokens, heads * 2), dtype=weights.dtype, device=device)
        w_storage[..., ::2] = weights
        weights = w_storage[..., ::2]
    width = table_width or max(1, *(math.ceil(length / page) for length in key_lengths))
    table = torch.zeros((len(key_lengths), width), dtype=torch.int32)
    next_page = 0
    for row, length in enumerate(key_lengths):
        count = math.ceil(length / page)
        if count:
            table[row, :count] = torch.arange(next_page, next_page + count).flip(0)
            next_page += count
    query_ends = list(itertools.accumulate(query_lengths))
    return dict(
        query=q,
        key=key,
        weights=weights,
        actual_seq_lengths_query=torch.tensor(
            query_ends, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_key=torch.tensor(
            key_lengths, dtype=torch.int32, device=device
        ),
        block_table=table.to(device),
    )
