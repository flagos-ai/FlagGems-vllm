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

"""Shared case registry and torch golden reference for the compressor operator.

The case registry, input builders and torch golden reference are migrated
from the source operator repository.

The golden implementation is test-only; it must never be imported by
production code.
"""

from __future__ import annotations

from itertools import accumulate
from typing import Iterable, Optional

import torch

# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

CASE_SPECS = {
    "Prefill0": (1, 4096, 8192, 512, 128, 64, 4, 2, 0),
    "Prefill1": (1, 4096, 8192, 128, 128, 64, 4, 2, 0),
    "Prefill2": (1, 4096, 8192, 512, 128, 64, 128, 1, 0),
    "Prefill0_b2": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "Prefill1_b2": (2, 4096, 4096, 128, 128, 64, 4, 2, 0),
    "Prefill2_b2": (2, 4096, 4096, 512, 128, 64, 128, 1, 0),
    "Prefill0_b2_rank3": (2, 4096, 4096, 512, 128, 64, 4, 2, 0),
    "Prefill1_b2_rank3": (2, 4096, 4096, 128, 128, 64, 4, 2, 0),
    "Prefill2_b2_rank3": (2, 4096, 4096, 512, 128, 64, 128, 1, 0),
    "prefill_b8_concurrency": (8, 4096, 1024, 512, 128, 64, 4, 2, 0),
    "prefill_b32_concurrency": (32, 4096, 320, 512, 128, 64, 4, 2, 0),
    "prefill_b2_chunk_tail": (
        2,
        4096,
        (2944, 7296),
        512,
        128,
        64,
        4,
        2,
        (128128, 117888),
    ),
    "decode0_start8195": (1, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "decode1_start8195": (1, 4096, 1, 128, 128, 64, 4, 2, 8195),
    "decode2_start8195": (1, 4096, 1, 512, 128, 64, 128, 1, 8195),
    "decode3": (8, 4096, 3, 512, 128, 64, 4, 2, 8193),
    "decode4_start8193": (8, 4096, 3, 128, 128, 64, 4, 2, 8193),
    "decode5": (8, 4096, 3, 512, 128, 64, 128, 1, 8193),
    "decode0_b32": (32, 4096, 1, 512, 128, 64, 4, 2, 8195),
    "decode3_b32": (32, 4096, 3, 512, 128, 64, 4, 2, 8193),
    "decode5_b32": (32, 4096, 3, 512, 128, 64, 128, 1, 8193),
    "prefill_b1_nonzero_c2": (1, 4096, 1024, 128, 128, 64, 4, 2, 8195),
    "prefill_b1_nonzero_c1": (1, 4096, 1024, 512, 128, 64, 128, 1, 8193),
    "prefill_b2_nonzero_c1": (
        2,
        4096,
        (1024, 1152),
        512,
        128,
        64,
        128,
        1,
        (8193, 12231),
    ),
}

PRODUCTION_CASES = tuple(list(CASE_SPECS)[:21])
REGRESSION_CASES = tuple(list(CASE_SPECS)[21:])
ALL_CASES = PRODUCTION_CASES + REGRESSION_CASES
SCALAR_KEYS = (
    "rope_head_dim",
    "cmp_ratio",
    "coff",
    "norm_eps",
    "rotary_mode",
    "cache_mode",
)


def _per_request_values(value, batch, name):
    values = (value,) * batch if isinstance(value, int) else tuple(value)
    if len(values) != batch:
        raise ValueError(f"{name} must contain {batch} values, got {len(values)}")
    return values


def case_layout(case_name):
    batch, hidden, seq_spec, head_dim, block_size, rope_dim, ratio, coff, start_spec = (
        CASE_SPECS[case_name]
    )
    return (
        batch,
        hidden,
        _per_request_values(seq_spec, batch, "sequence lengths"),
        head_dim,
        block_size,
        rope_dim,
        ratio,
        coff,
        _per_request_values(start_spec, batch, "start positions"),
    )


def _uniform(shape, low, high, dtype, device):
    return (
        torch.rand(shape, dtype=torch.float32, device=device) * (high - low) + low
    ).to(dtype)


def _full_block_table(batch, maximum_position, block_size):
    max_blocks = (maximum_position + block_size - 1) // block_size
    table = torch.arange(1, batch * max_blocks + 1, dtype=torch.int32).reshape(
        batch, max_blocks
    )
    return table, batch * max_blocks + 1


def make_inputs(case_name, device="npu", seed=42):
    batch, hidden, seq_lens, head_dim, block_size, rope_dim, ratio, coff, starts = (
        case_layout(case_name)
    )
    dtype = torch.bfloat16
    total_tokens = sum(seq_lens)
    projection_dim = coff * head_dim
    torch.manual_seed(seed)
    cumulative = torch.tensor((0, *accumulate(seq_lens)), dtype=torch.int32)
    start_pos = torch.tensor(starts, dtype=torch.int32)
    maximum_position = max(start + length for start, length in zip(starts, seq_lens))
    block_table, state_blocks = _full_block_table(batch, maximum_position, block_size)
    rank3 = case_name.endswith("_rank3")
    out_per_batch = (seq_lens[0] + ratio - 1) // ratio
    rope_rows = (
        batch * out_per_batch
        if rank3
        else min(total_tokens, total_tokens // ratio + batch)
    )

    x = _uniform((total_tokens, hidden), -10, 10, dtype, device)
    rope_sin = _uniform((rope_rows, rope_dim), -1, 1, dtype, device)
    rope_cos = _uniform((rope_rows, rope_dim), -1, 1, dtype, device)
    if rank3:
        x = x.reshape(batch, seq_lens[0], hidden)
        rope_sin = rope_sin.reshape(batch, out_per_batch, rope_dim)
        rope_cos = rope_cos.reshape(batch, out_per_batch, rope_dim)

    return {
        "x": x,
        "wkv": _uniform((projection_dim, hidden), -10, 10, dtype, device),
        "wgate": _uniform((projection_dim, hidden), -10, 10, dtype, device),
        "state_cache": _uniform(
            (state_blocks, block_size, 2 * projection_dim),
            -10,
            10,
            torch.float32,
            device,
        ),
        "ape": _uniform((ratio, projection_dim), -10, 10, torch.float32, device),
        "norm_weight": _uniform((head_dim,), -10, 10, dtype, device),
        "rope_sin": rope_sin,
        "rope_cos": rope_cos,
        "state_block_table": block_table.to(device=device),
        "cu_seqlens": None if rank3 else cumulative.to(device=device),
        "seqused": None,
        "start_pos": start_pos.to(device=device),
        "rope_head_dim": rope_dim,
        "cmp_ratio": ratio,
        "coff": coff,
        "norm_eps": 1e-6,
        "rotary_mode": 2,
        "cache_mode": 1,
    }


def split_operator_inputs(inputs):
    tensors = dict(inputs)
    config = {key: tensors.pop(key) for key in SCALAR_KEYS}
    return tensors, config


def clone_inputs(inputs):
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def valid_output_rows(inputs):
    starts = inputs["start_pos"].cpu().tolist()
    if inputs["x"].dim() == 3:
        batch, seq_len, _ = inputs["x"].shape
        used = [seq_len] * batch
    else:
        cumulative = inputs["cu_seqlens"].cpu().tolist()
        used = [
            cumulative[index + 1] - cumulative[index]
            for index in range(len(cumulative) - 1)
        ]
    return sum(
        _compressed_rows(start, count, inputs["cmp_ratio"])
        for start, count in zip(starts, used)
    )


def _compressed_rows(start, count, ratio):
    return max((int(start) + max(int(count), 0)) // ratio - int(start) // ratio, 0)


# ---------------------------------------------------------------------------
# Torch golden reference (test-only)
# ---------------------------------------------------------------------------


def _as_int_list(value, length: Optional[int] = None, default: int = 0):
    if value is None:
        return [] if length is None else [default] * length
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [int(value)] if length is None else [int(value)] * length
    values = [int(item) for item in values]
    if length is not None:
        values = (values + [default] * max(length - len(values), 0))[:length]
    return values


def _compressed_group_count(start: int, used: int, cmp_ratio: int) -> int:
    return max(
        (int(start) + max(int(used), 0)) // cmp_ratio - int(start) // cmp_ratio, 0
    )


def _batch_metadata(
    x, rope_sin, state_block_table, cu_seqlens, seqused, start_pos, cmp_ratio
):
    if x.dim() == 3:
        batch, seq_len, _ = x.shape
        token_bases = [index * seq_len for index in range(batch)]
        seq_lens = [seq_len] * batch
        seq_used = (
            _as_int_list(seqused, batch, seq_len) if seqused is not None else seq_lens
        )
        starts = _as_int_list(start_pos, batch, 0)
        out_per_batch = (seq_len + cmp_ratio - 1) // cmp_ratio
        return token_bases, seq_used, starts, batch * out_per_batch, out_per_batch

    total_tokens = x.shape[0]
    if cu_seqlens is not None:
        cumulative = _as_int_list(cu_seqlens)
        batch = max(len(cumulative) - 1, 0)
        token_bases = cumulative[:-1]
        seq_lens = [
            max(cumulative[index + 1] - cumulative[index], 0) for index in range(batch)
        ]
    elif state_block_table is not None:
        batch = int(state_block_table.shape[0])
        if batch != 1:
            raise ValueError(
                "cu_seqlens is required for rank-2 x when batch size is greater than one"
            )
        token_bases, seq_lens = [0], [total_tokens]
    else:
        batch, token_bases, seq_lens = 1, [0], [total_tokens]
    seq_used = _as_int_list(seqused, batch, 0) if seqused is not None else seq_lens
    starts = _as_int_list(start_pos, batch, 0)
    return token_bases, seq_used, starts, int(rope_sin.shape[0]), 0


def _cache_block_id(state_block_table, batch_index, block_index, state_blocks):
    if block_index < 0:
        return None
    if state_block_table is None:
        return block_index if block_index < state_blocks else None
    if block_index >= state_block_table.shape[1]:
        return None
    cache_block = int(state_block_table[batch_index, block_index].item())
    return cache_block if 0 < cache_block < state_blocks else None


def _apply_rope(x, rope_sin, rope_cos, rope_head_dim, rotary_mode):
    if rope_head_dim <= 0:
        return x
    output = x.clone()
    start = x.shape[-1] - rope_head_dim
    rope = x[..., start:]
    sin = rope_sin.to(torch.float32)
    cos = rope_cos.to(torch.float32)
    if rotary_mode == 1:
        half = rope_head_dim // 2
        rotated = torch.cat((-rope[..., half:], rope[..., :half]), dim=-1)
    else:
        even, odd = rope[..., 0::2], rope[..., 1::2]
        rotated = torch.empty_like(rope)
        rotated[..., 0::2] = -odd
        rotated[..., 1::2] = even
    output[..., start:] = rope * cos + rotated * sin
    return output


def compressor_torch(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    rope_head_dim: int = 64,
    cmp_ratio: int = 128,
    coff: int = 1,
    norm_eps: float = 1e-6,
    rotary_mode: int = 2,
    cache_mode: int = 1,
) -> torch.Tensor:
    """Readable PyTorch reference; ``state_cache`` is updated in place."""
    del cache_mode
    hidden_size = x.shape[-1]
    head_dim = int(norm_weight.shape[0])
    projection_dim = coff * head_dim
    token_bases, seq_used, starts, flat_rows, out_per_batch = _batch_metadata(
        x, rope_sin, state_block_table, cu_seqlens, seqused, start_pos, cmp_ratio
    )

    x_2d = x.reshape(-1, hidden_size).to(torch.float32)
    kv_projection = x_2d @ wkv.to(torch.float32).t()
    score_projection = x_2d @ wgate.to(torch.float32).t()
    block_size, state_blocks = int(state_cache.shape[1]), int(state_cache.shape[0])

    for batch_index, (token_base, used, start) in enumerate(
        zip(token_bases, seq_used, starts)
    ):
        for token_index in range(max(int(used), 0)):
            position = int(start) + token_index
            cache_block = _cache_block_id(
                state_block_table, batch_index, position // block_size, state_blocks
            )
            if cache_block is None:
                continue
            block_offset = position % block_size
            state_cache[cache_block, block_offset, :projection_dim] = kv_projection[
                token_base + token_index
            ]
            state_cache[
                cache_block, block_offset, projection_dim : 2 * projection_dim
            ] = score_projection[token_base + token_index] + ape[
                position % cmp_ratio, :projection_dim
            ].to(
                torch.float32
            )

    if x.dim() == 3:
        output = torch.zeros(
            (len(token_bases), out_per_batch, head_dim),
            device=x.device,
            dtype=torch.float32,
        )
        output_bases = [
            batch_index * out_per_batch for batch_index in range(len(token_bases))
        ]
        group_counts = [
            min(_compressed_group_count(start, used, cmp_ratio), out_per_batch)
            for start, used in zip(starts, seq_used)
        ]
    else:
        output = torch.zeros(
            (flat_rows, head_dim), device=x.device, dtype=torch.float32
        )
        output_bases, group_counts, cursor = [], [], 0
        for start, used in zip(starts, seq_used):
            output_bases.append(cursor)
            groups = max(
                min(
                    _compressed_group_count(start, used, cmp_ratio), flat_rows - cursor
                ),
                0,
            )
            group_counts.append(groups)
            cursor += groups

    output_flat = output.reshape(-1, head_dim)
    rope_sin_flat = rope_sin.reshape(-1, rope_head_dim)
    rope_cos_flat = rope_cos.reshape(-1, rope_head_dim)
    large_negative = torch.finfo(torch.float32).min
    for batch_index, (output_base, groups, start) in enumerate(
        zip(output_bases, group_counts, starts)
    ):
        for group_index in range(groups):
            group_start = (
                (int(start) + group_index * cmp_ratio) // cmp_ratio
            ) * cmp_ratio
            kv_rows, score_rows = [], []
            for group_offset in range(cmp_ratio * coff):
                if coff == 1:
                    position, dim_start = group_start + group_offset, 0
                elif group_offset < cmp_ratio:
                    position, dim_start = group_start - cmp_ratio + group_offset, 0
                else:
                    position, dim_start = (
                        group_start + group_offset - cmp_ratio,
                        head_dim,
                    )
                cache_block = _cache_block_id(
                    state_block_table, batch_index, position // block_size, state_blocks
                )
                if cache_block is None or position < 0:
                    kv_rows.append(
                        torch.zeros(head_dim, device=x.device, dtype=torch.float32)
                    )
                    score_rows.append(
                        torch.full(
                            (head_dim,),
                            large_negative,
                            device=x.device,
                            dtype=torch.float32,
                        )
                    )
                    continue
                block_offset = position % block_size
                kv_rows.append(
                    state_cache[
                        cache_block, block_offset, dim_start : dim_start + head_dim
                    ].to(torch.float32)
                )
                score_rows.append(
                    state_cache[
                        cache_block,
                        block_offset,
                        projection_dim
                        + dim_start : projection_dim
                        + dim_start
                        + head_dim,
                    ].to(torch.float32)
                )
            weights = torch.softmax(torch.stack(score_rows), dim=0)
            compressed = torch.sum(weights * torch.stack(kv_rows), dim=0)
            normalized = compressed * torch.rsqrt(
                torch.mean(compressed * compressed) + norm_eps
            )
            normalized = normalized * norm_weight.to(torch.float32)
            row = output_base + group_index
            output_flat[row] = _apply_rope(
                normalized,
                rope_sin_flat[row],
                rope_cos_flat[row],
                rope_head_dim,
                rotary_mode,
            )
    return output.to(dtype=x.dtype)


__all__ = [
    "ALL_CASES",
    "CASE_SPECS",
    "PRODUCTION_CASES",
    "REGRESSION_CASES",
    "clone_inputs",
    "compressor_torch",
    "make_inputs",
    "split_operator_inputs",
    "valid_output_rows",
]
