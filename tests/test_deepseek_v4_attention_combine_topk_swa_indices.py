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

import importlib

import pytest
import torch

import flaggems_vllm
import flaggems_vllm.testing as fg_testing

# Bind through the top-level entry rather than the ops submodule: when a vendor
# ships a specialized implementation, the runtime rebinds it on the package at
# import time, so importing from flaggems_vllm.ops.<module> would bypass vendor
# dispatch (same reasoning as persistent_topk).
combine_topk_swa_indices = flaggems_vllm.combine_topk_swa_indices

pytestmark = pytest.mark.combine_topk_swa_indices

# Device-agnostic entry point: the runtime resolves the vendor device name --
# "cuda" on NVIDIA / MetaX / Hygon / T-Head, "musa" on MThreads, "npu" on
# Ascend. Never hard-code "cuda" here.
device = flaggems_vllm.device
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()


def _has_hq4_cuda() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    return torch.cuda.get_device_capability()[0] >= 8


_HAS_HQ4_CUDA = _has_hq4_cuda()

# vLLM >= 0.23 relocated this op to vllm.models.deepseek_v4.common.ops (the
# definition lives in its .cache_utils submodule); older releases exposed it as
# vllm.v1.attention.ops.deepseek_v4_ops.
try:
    from vllm.models.deepseek_v4.common.ops import (
        combine_topk_swa_indices as vllm_combine_topk_swa_indices,
    )

    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = True
except Exception:
    vllm_combine_topk_swa_indices = None
    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = False


@pytest.mark.parametrize(
    (
        "topk_values",
        "query_start_values",
        "seq_len_values",
        "gather_len_values",
        "window_size",
        "compress_ratio",
        "topk",
        "M",
        "N",
    ),
    [
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [0, 2, 3],
            [8, 10],
            [8, 10],
            4,
            2,
            4,
            64,
            16,
        ),
        (
            [
                [100, 101, 102, 103],
                [110, 111, 112, 113],
                [120, 121, 122, 123],
                [130, 131, 132, 133],
                [140, 141, 142, 143],
            ],
            [0, 3, 5],
            [6, 4],
            [4, 3],
            3,
            2,
            4,
            20,
            8,
        ),
    ],
)
@pytest.mark.skipif(not _HAS_DEVICE, reason=f"requires an available {device} device")
def test_combine_topk_swa_indices_accuracy(
    topk_values,
    query_start_values,
    seq_len_values,
    gather_len_values,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    topk_indices = torch.tensor(topk_values, device=device, dtype=torch.int32)
    query_start_loc = torch.tensor(query_start_values, device=device, dtype=torch.int32)
    seq_lens = torch.tensor(seq_len_values, device=device, dtype=torch.int32)
    gather_lens = torch.tensor(gather_len_values, device=device, dtype=torch.int32)

    actual, actual_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )

    expected = torch.full_like(actual, -1)
    expected_lens = torch.empty_like(actual_lens)
    for batch in range(seq_lens.numel()):
        start = int(query_start_loc[batch].item()) - int(query_start_loc[0].item())
        end = int(query_start_loc[batch + 1].item()) - int(query_start_loc[0].item())
        query_len = end - start
        seq_len = int(seq_lens[batch].item())
        gather_len = int(gather_lens[batch].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        for token_idx in range(start, end):
            token_in_query = token_idx - start
            pos = start_pos + token_in_query
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            if topk_len:
                expected[token_idx, :topk_len] = (
                    topk_indices[token_idx, :topk_len] + M * batch
                )
            for j in range(swa_len):
                expected[token_idx, topk_len + j] = (
                    M * batch + N + j + pos - swa_len + 1 - gather_start
                )
            expected_lens[token_idx] = topk_len + swa_len

    fg_testing.assert_equal(actual, expected)
    fg_testing.assert_equal(actual_lens, expected_lens)


@pytest.mark.skipif(
    not (_HAS_DEVICE and _HAS_VLLM_COMBINE_TOPK_SWA_INDICES),
    reason=f"requires an available {device} device and vLLM's implementation",
)
def test_combine_topk_swa_indices_vllm_accuracy():
    topk_indices = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([8, 10], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([8, 10], device=device, dtype=torch.int32)
    args = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        4,
        2,
        4,
        64,
        16,
    )

    actual, actual_lens = combine_topk_swa_indices(*args)
    expected, expected_lens = vllm_combine_topk_swa_indices(*args)

    fg_testing.assert_equal(actual, expected)
    fg_testing.assert_equal(actual_lens, expected_lens)


def test_combine_topk_swa_indices_hq4_flag_isolation(monkeypatch):
    """Metadata options cannot change the default producer dispatch."""
    module = importlib.import_module(
        "flaggems_vllm.ops.deepseek_v4_attention_combine_topk_swa_indices"
    )
    sentinel = object()
    seen = []

    def fake_default(*args):
        seen.append(args)
        return sentinel

    monkeypatch.setattr(module, "_combine_topk_swa_indices_default", fake_default)
    topk_indices = torch.empty((1, 4), dtype=torch.int32)
    query_start = torch.empty((2,), dtype=torch.int32)
    seq_lens = torch.empty((1,), dtype=torch.int32)
    gather_lens = torch.empty((1,), dtype=torch.int32)
    assert (
        module.combine_topk_swa_indices(
            topk_indices,
            query_start,
            seq_lens,
            gather_lens,
            4,
            2,
            4,
            64,
            16,
        )
        is sentinel
    )
    assert len(seen) == 1

    with pytest.raises(ValueError, match="enable_hq4_sparse_prefill=True"):
        module.combine_topk_swa_indices(
            topk_indices,
            query_start,
            seq_lens,
            gather_lens,
            4,
            2,
            4,
            64,
            16,
            return_pair_metadata=True,
        )
    assert len(seen) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_combine_topk_swa_indices_hq4_exact_1024_metadata():
    """Exercise the grouped producer kernel used at common prefill sizes."""
    tokens = 1024
    torch.manual_seed(5982)
    source = torch.randperm(2048, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    query_start = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([4096 + tokens], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([tokens + 127], device="cuda", dtype=torch.int32)
    args = (
        source,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
    )
    combined, lengths, pairs, quads = combine_topk_swa_indices(
        *args,
        enable_hq4_sparse_prefill=True,
        return_pair_metadata=True,
        return_quad_metadata=True,
    )
    expected_combined, expected_lengths = combine_topk_swa_indices(*args)
    fg_testing.assert_equal(combined, expected_combined)
    fg_testing.assert_equal(lengths, expected_lengths)

    first_position = 4096 + torch.arange(0, tokens, 2, device="cuda", dtype=torch.int32)
    first_topk = (first_position + 1) // 4
    second_topk = (first_position + 2) // 4
    pair_mode = torch.where(first_topk == second_topk, 2, 4)
    expected_pairs = (first_topk << 3) | pair_mode
    expected_quads = torch.ones(((tokens + 3) // 4,), device="cuda", dtype=torch.int32)
    fg_testing.assert_equal(pairs, expected_pairs)
    fg_testing.assert_equal(quads, expected_quads)


@pytest.mark.skipif(
    not _HAS_HQ4_CUDA,
    reason="requires an NVIDIA CUDA GPU with native BF16 support",
)
def test_combine_topk_swa_indices_hq4_multi_request_specialized_metadata():
    """Cover specialized groups crossing request boundaries and an empty request."""
    request_lengths = [5, 0, 9, 1013]
    contexts = [0, 0, 127, 8192]
    offsets = [0]
    for length in request_lengths:
        offsets.append(offsets[-1] + length)
    tokens = offsets[-1]
    assert tokens >= 1024

    source = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    query_start = torch.tensor(
        [offset + 7 for offset in offsets], device="cuda", dtype=torch.int32
    )
    seq_lens = torch.tensor(
        [length + context for length, context in zip(request_lengths, contexts)],
        device="cuda",
        dtype=torch.int32,
    )
    gather_lens = torch.tensor(
        [
            length + min(context, 127)
            for length, context in zip(request_lengths, contexts)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    args = (
        source,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
    )
    combined, lengths, pairs, quads = combine_topk_swa_indices(
        *args,
        enable_hq4_sparse_prefill=True,
        return_pair_metadata=True,
        return_quad_metadata=True,
    )
    expected_combined, expected_lengths = combine_topk_swa_indices(*args)
    fg_testing.assert_equal(combined, expected_combined)
    fg_testing.assert_equal(lengths, expected_lengths)

    expected_pairs = [0] * ((tokens + 1) // 2)
    expected_quads = [0] * ((tokens + 3) // 4)
    for request, context in enumerate(contexts):
        start, end = offsets[request : request + 2]
        first_pair_row = ((start + 1) // 2) * 2
        for row in range(first_pair_row, end - 1, 2):
            pos = context + row - start
            first_topk = min((pos + 1) // 4, 2048)
            second_topk = min((pos + 2) // 4, 2048)
            first_swa = min(pos + 1, 128)
            second_swa = min(pos + 2, 128)
            topk_grows = second_topk == first_topk + 1
            if first_topk + first_swa > 0 and first_swa < 128:
                mode = 3 if topk_grows else 1
            elif first_swa == second_swa == 128:
                mode = 4 if topk_grows else 2
            else:
                mode = 0
            if mode:
                expected_pairs[row // 2] = (first_topk << 3) | mode

        first_quad_row = ((start + 3) // 4) * 4
        for row in range(first_quad_row, end - 3, 4):
            pos = context + row - start
            expected_quads[row // 4] = int(min(pos + 1, 128) > 0)

    fg_testing.assert_equal(
        pairs, torch.tensor(expected_pairs, device="cuda", dtype=torch.int32)
    )
    fg_testing.assert_equal(
        quads, torch.tensor(expected_quads, device="cuda", dtype=torch.int32)
    )
