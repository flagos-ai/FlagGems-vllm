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

import math

import flaggems_vllm
import pytest
import torch
import triton
from flaggems_vllm.ops.attn_res import (
    _HAS_TLE_LOAD,
    _attn_res_post_kernel,
    _prune_post_configs,
    _select_fixed_launch_config,
    _token_count_bucket,
)

from .accuracy_utils import gems_assert_close

EPS = 1e-5
HIDDEN_SIZE = 7168
MAX_BLOCKS = 8


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


requires_hopper = pytest.mark.skipif(not _is_hopper(), reason="requires CUDA SM90")


def _randn_with_row_padding(
    *shape: int,
    padding: int = 0,
    offset: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    if not 0 <= offset <= padding:
        raise ValueError("offset must be in [0, padding]")
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device=flaggems_vllm.device,
        dtype=torch.bfloat16,
    )
    storage.mul_(scale)
    return storage[..., offset : offset + shape[-1]]


def _make_inputs(
    num_tokens: int,
    has_delta: bool,
    apply_output_norm: bool,
    row_padding: int = 0,
    row_offset: int = 0,
    hidden_size: int = HIDDEN_SIZE,
) -> tuple[torch.Tensor, ...]:
    prefix = _randn_with_row_padding(
        num_tokens,
        hidden_size,
        padding=row_padding,
        offset=row_offset,
    )
    delta = (
        _randn_with_row_padding(
            num_tokens,
            hidden_size,
            padding=row_padding,
            offset=row_offset,
        )
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens,
        MAX_BLOCKS,
        hidden_size,
        padding=row_padding,
        offset=row_offset,
    )
    norm_weight = 1 + _randn_with_row_padding(hidden_size, scale=0.1)
    qk_weight = _randn_with_row_padding(
        hidden_size,
        scale=1 / math.sqrt(hidden_size),
    )
    output_norm_weight = (
        1 + _randn_with_row_padding(hidden_size, scale=0.1)
        if apply_output_norm
        else None
    )
    return prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight


def _reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    if delta is not None:
        updated_prefix = (prefix.float() + delta.float()).to(prefix.dtype)
        prefix.copy_(updated_prefix)
    else:
        updated_prefix = prefix

    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(updated_prefix)

    values = torch.cat(
        (blocks[:, :num_blocks], updated_prefix.unsqueeze(1)),
        dim=1,
    ).float()
    reciprocal_std = torch.rsqrt(values.square().mean(dim=-1) + eps)
    logits = (
        values * reciprocal_std.unsqueeze(-1) * norm_weight.float() * qk_weight.float()
    ).sum(dim=-1)
    probabilities = logits.softmax(dim=-1)
    output = (probabilities.unsqueeze(-1) * values).sum(dim=1)
    if output_norm_weight is not None:
        output = (
            output
            * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + output_norm_eps)
            * output_norm_weight.float()
        )
    return output.to(prefix.dtype)


def _run_and_check(
    num_tokens: int,
    num_blocks: int,
    block_write_idx: int,
    has_delta: bool,
    apply_output_norm: bool,
    row_padding: int = 0,
    row_offset: int = 0,
    hidden_size: int = HIDDEN_SIZE,
) -> None:
    torch.manual_seed(2026)
    args = _make_inputs(
        num_tokens,
        has_delta,
        apply_output_norm,
        row_padding,
        row_offset,
        hidden_size,
    )
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = args
    expected_prefix = prefix.clone()
    expected_blocks = blocks.clone()
    expected = _reference(
        expected_prefix,
        delta,
        expected_blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        2 * EPS,
    )

    actual = flaggems_vllm.attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        2 * EPS,
    )

    gems_assert_close(actual, expected, torch.bfloat16, atol=4e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.attn_res
@requires_hopper
@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "block_write_idx",
        "has_delta",
        "apply_output_norm",
    ),
    [
        pytest.param(1, 0, 0, False, True, id="block0-write"),
        pytest.param(3, 4, 4, True, True, id="write"),
        pytest.param(1, 2, 2, True, True, id="write-b2"),
        pytest.param(17, 5, -1, False, True, id="post"),
        pytest.param(64, 1, -1, False, True, id="post-n64-block1"),
        pytest.param(64, 4, -1, True, True, id="common"),
        pytest.param(64, 3, -1, True, True, id="common-b3"),
        pytest.param(
            5,
            6,
            -1,
            True,
            True,
            id="common-partial-source-tile",
        ),
        pytest.param(256, 4, -1, True, True, id="persistent-common"),
        pytest.param(256, 3, 3, True, True, id="persistent-write"),
        pytest.param(256, 3, -1, False, True, id="post-block3"),
        pytest.param(
            1,
            1,
            -1,
            False,
            False,
            id="post-block1-without-output-norm",
        ),
        pytest.param(5, 4, 2, True, True, id="rewrite-attended-block"),
        pytest.param(
            5,
            2,
            1,
            True,
            True,
            id="rewrite-attended-block-b2",
        ),
        pytest.param(
            256,
            4,
            -1,
            False,
            False,
            id="read-only-without-output-norm",
        ),
        pytest.param(512, 8, -1, True, False, id="final"),
    ],
)
def test_attn_res_semantic_variants(
    num_tokens: int,
    num_blocks: int,
    block_write_idx: int,
    has_delta: bool,
    apply_output_norm: bool,
):
    _run_and_check(
        num_tokens,
        num_blocks,
        block_write_idx,
        has_delta,
        apply_output_norm,
    )


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_small_source_routes():
    for num_tokens in (1, 64):
        for num_blocks in (2, 3):
            _run_and_check(num_tokens, num_blocks, num_blocks, True, True)
            _run_and_check(num_tokens, num_blocks, -1, False, True)
            _run_and_check(num_tokens, num_blocks, -1, True, True)


@pytest.mark.attn_res
@requires_hopper
@pytest.mark.parametrize("num_blocks", range(MAX_BLOCKS + 1))
def test_attn_res_all_block_counts(num_blocks: int):
    _run_and_check(1, num_blocks, -1, False, True)


@pytest.mark.attn_res
@requires_hopper
@pytest.mark.parametrize("hidden_size", (1024, 4096))
def test_attn_res_hidden_size(hidden_size: int):
    _run_and_check(3, 2, -1, False, True, hidden_size=hidden_size)
    _run_and_check(3, 4, -1, True, True, hidden_size=hidden_size)


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_padded_rows():
    _run_and_check(7, 3, 3, True, True, row_padding=11)
    _run_and_check(7, 3, -1, False, True, row_padding=11)


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_misaligned_row_base():
    _run_and_check(
        7,
        3,
        -1,
        False,
        True,
        row_padding=16,
        row_offset=1,
    )


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_empty_tokens():
    args = _make_inputs(0, True, False)
    output = flaggems_vllm.attn_res(*args, MAX_BLOCKS, -1, EPS, EPS)
    assert output.shape == (0, HIDDEN_SIZE)
    assert output.is_contiguous()


@pytest.mark.attn_res
@pytest.mark.parametrize("num_tokens", (1, 64, 255, 256, 7680, 16384))
@pytest.mark.parametrize("num_blocks", range(MAX_BLOCKS + 1))
def test_attn_res_h100_fixed_routing(num_tokens: int, num_blocks: int):
    config = _select_fixed_launch_config(num_tokens, num_blocks)
    assert config.use_persistent_token_loop == (num_tokens >= 256 and num_blocks >= 3)
    assert config.use_compact_source_reduction == (
        num_tokens < 256 and 1 < num_blocks <= 3
    )
    if config.use_persistent_token_loop or num_blocks <= 1:
        expected_source_tile_size = 1
    elif num_blocks <= 3:
        expected_source_tile_size = 4
    elif num_blocks < MAX_BLOCKS:
        expected_source_tile_size = 8
    else:
        expected_source_tile_size = 4
    assert config.source_tile_size == expected_source_tile_size
    assert config.enable_pdl == (num_tokens < 256 and 1 < num_blocks <= 3)


@pytest.mark.attn_res
@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "expected_strategies"),
    (
        (1, 1, ((False, True, 2, False),)),
        (
            1,
            2,
            (
                (False, True, 2, False),
                (True, False, 4, True),
            ),
        ),
        (255, 7, ((False, False, 8, False),)),
        (
            256,
            1,
            (
                (False, False, 2, False),
                (False, True, 2, False),
                (False, False, 4, False),
                (False, False, 8, False),
            ),
        ),
        (
            1,
            8,
            (
                (False, False, 2, False),
                (False, True, 2, False),
                (False, False, 4, False),
                (False, False, 8, False),
            ),
        ),
        (7680, 6, ((False, False, 2, False),)),
        (7680, 7, ((False, True, 2, False),)),
        (16384, 8, ((False, False, 2, False),)),
    ),
)
def test_attn_res_post_config_pruning(
    num_tokens: int,
    num_blocks: int,
    expected_strategies: tuple[tuple[bool, bool, int, bool], ...],
):
    configs = [
        triton.Config(
            {
                "USE_COMPACT_SOURCE_REDUCTION": False,
                "USE_SOURCE_POINTER_TUPLE": False,
                "USE_TLE_ASYNC_LOAD": False,
                "SOURCE_TILE_SIZE": 2,
                "launch_pdl": False,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "USE_COMPACT_SOURCE_REDUCTION": False,
                "USE_SOURCE_POINTER_TUPLE": True,
                "USE_TLE_ASYNC_LOAD": False,
                "SOURCE_TILE_SIZE": 2,
                "launch_pdl": False,
            },
            num_stages=1,
        ),
        triton.Config(
            {
                "USE_COMPACT_SOURCE_REDUCTION": False,
                "USE_SOURCE_POINTER_TUPLE": True,
                "USE_TLE_ASYNC_LOAD": False,
                "SOURCE_TILE_SIZE": 2,
                "launch_pdl": False,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "USE_COMPACT_SOURCE_REDUCTION": False,
                "USE_SOURCE_POINTER_TUPLE": False,
                "USE_TLE_ASYNC_LOAD": False,
                "SOURCE_TILE_SIZE": 4,
                "launch_pdl": False,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "USE_COMPACT_SOURCE_REDUCTION": False,
                "USE_SOURCE_POINTER_TUPLE": False,
                "USE_TLE_ASYNC_LOAD": False,
                "SOURCE_TILE_SIZE": 8,
                "launch_pdl": False,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "USE_COMPACT_SOURCE_REDUCTION": True,
                "USE_SOURCE_POINTER_TUPLE": False,
                "USE_TLE_ASYNC_LOAD": False,
                "SOURCE_TILE_SIZE": 4,
                "launch_pdl": True,
            },
            num_warps=8,
            num_stages=2,
        ),
    ]
    pruned = _prune_post_configs(
        configs,
        {},
        TOKEN_COUNT_BUCKET=_token_count_bucket(num_tokens),
        NUM_BLOCKS=num_blocks,
    )
    actual_strategies = {
        (
            config.kwargs["USE_COMPACT_SOURCE_REDUCTION"],
            config.kwargs["USE_SOURCE_POINTER_TUPLE"],
            config.kwargs["SOURCE_TILE_SIZE"],
            config.kwargs["launch_pdl"],
        )
        for config in pruned
    }
    assert actual_strategies == set(expected_strategies)


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_post_tune_config_integration():
    configs = flaggems_vllm.runtime.get_tuned_config("attn_res_post")
    assert configs
    required_meta = {
        "USE_COMPACT_SOURCE_REDUCTION",
        "USE_SOURCE_POINTER_TUPLE",
        "USE_TLE_ASYNC_LOAD",
        "SOURCE_TILE_SIZE",
        "launch_pdl",
    }
    assert all(required_meta <= config.kwargs.keys() for config in configs)
    assert any(config.kwargs["USE_TLE_ASYNC_LOAD"] for config in configs)
    assert all(
        not config.kwargs["USE_TLE_ASYNC_LOAD"]
        or config.kwargs["USE_SOURCE_POINTER_TUPLE"]
        for config in configs
    )

    for num_tokens, num_blocks in ((1, 1), (1, 2), (255, 7), (1, 8), (256, 1)):
        pruned = _prune_post_configs(
            configs,
            {},
            TOKEN_COUNT_BUCKET=_token_count_bucket(num_tokens),
            NUM_BLOCKS=num_blocks,
        )
        assert pruned
        assert _HAS_TLE_LOAD or not any(
            config.kwargs["USE_TLE_ASYNC_LOAD"] for config in pruned
        )


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_post_reuses_libentry_cache_for_recreated_source_views():
    args = _make_inputs(1, False, True)
    cache = _attn_res_post_kernel.kernel_cache[torch.cuda.current_device()]

    flaggems_vllm.attn_res(*args, 1, -1, EPS, EPS)
    torch.cuda.synchronize()
    cached_entries = len(cache)

    for _ in range(2):
        flaggems_vllm.attn_res(*args, 1, -1, EPS, EPS)
        torch.cuda.synchronize()

    assert len(cache) == cached_entries


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_rejects_unsupported_dtype():
    prefix = torch.empty((1, HIDDEN_SIZE), device="cuda", dtype=torch.float32)
    blocks = torch.empty((1, MAX_BLOCKS, HIDDEN_SIZE), device="cuda")
    weight = torch.empty((HIDDEN_SIZE,), device="cuda")
    with pytest.raises(NotImplementedError, match="bfloat16"):
        flaggems_vllm.attn_res(
            prefix,
            None,
            blocks,
            weight,
            weight,
            weight,
            1,
            -1,
            EPS,
            EPS,
        )


@pytest.mark.attn_res
@requires_hopper
@pytest.mark.parametrize(
    ("num_blocks", "block_write_idx", "message"),
    [
        (-1, -1, "num_blocks"),
        (MAX_BLOCKS + 1, -1, "num_blocks"),
        (1, -2, "block_write_idx"),
        (1, MAX_BLOCKS, "block_write_idx"),
    ],
)
def test_attn_res_rejects_invalid_indices(
    num_blocks: int,
    block_write_idx: int,
    message: str,
):
    args = _make_inputs(1, False, True)
    with pytest.raises(ValueError, match=message):
        flaggems_vllm.attn_res(
            *args,
            num_blocks,
            block_write_idx,
            EPS,
            EPS,
        )


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_rejects_noncontiguous_hidden_dimension():
    prefix = torch.empty(
        (1, HIDDEN_SIZE, 2),
        device="cuda",
        dtype=torch.bfloat16,
    )[:, :, 0]
    blocks = torch.empty(
        (1, MAX_BLOCKS, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.empty((HIDDEN_SIZE,), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="last dimension of prefix"):
        flaggems_vllm.attn_res(
            prefix,
            None,
            blocks,
            weight,
            weight,
            weight,
            1,
            -1,
            EPS,
            EPS,
        )


@pytest.mark.attn_res
@requires_hopper
@pytest.mark.parametrize("tensor_name", ("prefix", "delta", "blocks"))
def test_attn_res_rejects_overlapping_layout(tensor_name: str):
    args = list(_make_inputs(2, True, True))
    tensor_index = {"prefix": 0, "delta": 1, "blocks": 2}[tensor_name]
    if tensor_name == "blocks":
        args[tensor_index] = args[tensor_index][:, :1, :].expand(-1, MAX_BLOCKS, -1)
    else:
        args[tensor_index] = args[tensor_index][:1].expand(2, -1)

    with pytest.raises(
        ValueError,
        match=rf"{tensor_name} must have a non-overlapping row-major layout",
    ):
        flaggems_vllm.attn_res(*args, 1, -1, EPS, EPS)


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_rejects_mutating_alias():
    args = list(_make_inputs(2, True, True))
    args[1] = args[0]
    with pytest.raises(ValueError, match="prefix must not overlap delta when mutated"):
        flaggems_vllm.attn_res(*args, 1, -1, EPS, EPS)


@pytest.mark.attn_res
@requires_hopper
def test_attn_res_rejects_autograd_inputs():
    args = list(_make_inputs(1, False, True))
    args[0].requires_grad_(True)
    with pytest.raises(NotImplementedError, match="forward-only"):
        flaggems_vllm.attn_res(*args, 1, -1, EPS, EPS)
