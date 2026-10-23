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
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_no_tle,
    moe_align_block_size_singleton,
    moe_align_block_size_small_grouped,
)

from . import accuracy_utils as utils


@pytest.mark.moe_align_block_size_triton
@pytest.mark.parametrize(
    "has_tle,is_cuda,num_experts,has_query,unsupported,tle_failure,expected_tle_calls",
    [
        (True, True, 512, True, None, False, 2),
        (True, True, 1024, True, None, False, 2),
        (True, True, 512, True, "device_mesh", False, 0),
        (True, True, 512, True, "distributed_barrier", False, 0),
        (True, True, 512, True, "gpu.alloc", False, 0),
        (True, True, 512, True, "gpu.local_ptr", False, 0),
        (True, True, 512, False, None, False, 0),
        (True, True, 512, True, None, True, 2),
        (False, True, 512, True, None, False, 0),
        (True, False, 512, True, None, False, 0),
        (True, True, 1025, True, None, False, 0),
    ],
)
def test_moe_align_tle_backend_dispatch(
    monkeypatch,
    has_tle,
    is_cuda,
    num_experts,
    has_query,
    unsupported,
    tle_failure,
    expected_tle_calls,
):
    """Check backend selection without allocating tensors or launching kernels."""
    module = importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(self.name)
                if self.name == "tle" and tle_failure:
                    raise RuntimeError("ordinary compilation failure")

            return launch

    topk_ids = SimpleNamespace(
        numel=lambda: 1280,
        ndim=2,
        shape=(128, 10),
        is_cuda=is_cuda,
        device="cuda" if is_cuda else "cpu",
    )
    sorted_ids = SimpleNamespace(numel=lambda: 1280 + num_experts * 15)
    expert_ids = SimpleNamespace(numel=lambda: (1280 + num_experts * 15) // 16)
    queries = []

    def is_tle_supported(api):
        queries.append(api)
        return api != unsupported

    language = SimpleNamespace()
    if has_query:
        language.is_tle_supported = is_tle_supported
    monkeypatch.setattr(module, "tle", language if has_tle else None)
    monkeypatch.setattr(module, "HAS_TLE", has_tle)
    monkeypatch.setattr(module, "_supports_tle_cluster_remote", lambda: False)
    monkeypatch.setattr(module, "_block_mesh", lambda _: None)
    monkeypatch.setattr(module, "_pick_tle_atomic_fused_num_blocks", lambda *args: 1)
    monkeypatch.setattr(module.torch, "zeros", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "moe_align_block_size_tle_atomic_fused_coop", Kernel("tle")
    )
    for stage in ["stage1", "stage2", "stage2_vec", "stage3", "stage4"]:
        monkeypatch.setattr(module, f"moe_align_block_size_{stage}", Kernel(stage))

    # Repeated unsupported calls must skip TLE without attempting compilation.
    for _ in range(2):
        module.moe_align_block_size_triton(
            topk_ids, num_experts, 16, sorted_ids, expert_ids, None
        )
    assert calls.count("tle") == expected_tle_calls
    if has_tle and is_cuda and num_experts <= 1024 and has_query:
        assert queries.count("device_mesh") == 2
        if unsupported:
            assert queries.count(unsupported) == 2
    else:
        assert queries == []
    if expected_tle_calls and not tle_failure:
        assert calls == ["tle", "tle"]
    else:
        stages = [
            "stage1",
            "stage2_vec" if num_experts & (num_experts - 1) == 0 else "stage2",
            "stage3",
            "stage4",
        ]
        assert [call for call in calls if call != "tle"] == stages * 2


@pytest.mark.moe_align_block_size_triton
@pytest.mark.parametrize(
    "unsupported,error,cluster_calls",
    [
        ("remote", None, 0),
        ("shard_id", None, 0),
        (None, None, 2),
        (None, "ordinary compilation failure", 2),
    ],
)
def test_moe_align_cluster_capability_is_independent(
    monkeypatch, unsupported, error, cluster_calls
):
    module = importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(self.name)
                if self.name == "cluster" and error:
                    raise RuntimeError(error)

            return launch

    monkeypatch.setattr(module, "HAS_TLE", True)
    monkeypatch.setattr(
        module, "tle", SimpleNamespace(is_tle_supported=lambda api: api != unsupported)
    )
    monkeypatch.setattr(module, "_supports_tle_cluster_remote", lambda: True)
    monkeypatch.setattr(module, "_block_mesh", lambda _: None)
    monkeypatch.setattr(module, "_block_cluster_mesh_8", lambda: None)
    monkeypatch.setattr(module, "_pick_tle_atomic_fused_num_blocks", lambda *args: 1)
    monkeypatch.setattr(module.torch, "zeros", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module, "moe_align_block_size_tle_cluster_fused", Kernel("cluster")
    )
    monkeypatch.setattr(
        module, "moe_align_block_size_tle_atomic_fused_coop", Kernel("atomic")
    )
    topk = SimpleNamespace(
        numel=lambda: 1280, shape=(128, 10), ndim=2, is_cuda=True, device="cuda:0"
    )
    sorted_ids = SimpleNamespace(numel=lambda: 8960)
    expert_ids = SimpleNamespace(numel=lambda: 560)
    for _ in range(2):
        module.moe_align_block_size_triton(topk, 512, 16, sorted_ids, expert_ids, None)
    assert calls.count("cluster") == cluster_calls
    assert calls.count("atomic") == (0 if unsupported is None and error is None else 2)


@pytest.mark.moe_align_block_size_triton
@pytest.mark.parametrize("num_experts", [10, 250, 512])
@pytest.mark.parametrize(
    "num_routes,num_sorted,num_blocks",
    [(0, 0, 0), (0, 150, 9), (1, 16, 1), (1280, 8960, 560)],
)
def test_moe_align_nonzero_launch_extents(
    monkeypatch, num_experts, num_routes, num_sorted, num_blocks
):
    """Empty compact buffers must not produce arange(0, 0) launch constants."""
    module = importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(args)

            return launch

    for stage in ["stage1", "stage2", "stage2_vec", "stage3", "stage4"]:
        monkeypatch.setattr(module, f"moe_align_block_size_{stage}", Kernel())
    module._moe_align_block_size_triton_4stage(
        SimpleNamespace(numel=lambda: num_routes),
        num_experts,
        16,
        SimpleNamespace(numel=lambda: num_sorted),
        SimpleNamespace(numel=lambda: num_blocks),
        None,
        workspace=(None, None),
    )
    assert len(calls) == 4
    extents = [calls[0][4], calls[0][-2], calls[0][-1]]
    expected = [
        1 << (max(1, (n + num_experts - 1) // num_experts) - 1).bit_length()
        for n in [num_routes, num_sorted, num_blocks]
    ]
    assert extents == expected


@pytest.mark.moe_align_block_size_triton
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-compatible graph required"
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_experts,block_size", [(10, 16), (512, 64)])
def test_moe_align_empty_compact_graph(dtype, num_experts, block_size):
    """Exercise all three zero extents through the existing compact-buffer API."""
    topk_ids = torch.empty((0, 10), dtype=dtype, device=flaggems_vllm.device)

    def run():
        return moe_align_block_size_no_tle(topk_ids, block_size, num_experts)

    def check(result):
        assert result[0].shape == (0,)
        assert result[1].shape == (0,)
        assert result[2].shape == (1,)
        assert all(t.dtype == torch.int32 for t in result)
        assert all(t.device == topk_ids.device for t in result)
        assert result[2].item() == 0

    check(run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for _ in range(3):
        result[2].fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        check(result)


# Modified from: https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_moe_align_block_size.py
def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    # if topk_ids.numel() < num_experts:
    #     max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(
        topk_ids.numel(), device=topk_ids.device, dtype=torch.int32
    )
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(
        num_experts, dtype=torch.int64, device=topk_ids.device
    )
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = (
                (original_count + block_size - 1) // block_size
            ) * block_size

    in_sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )

    # max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    max_num_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            in_sorted_token_ids[current_pos : current_pos + num_expert_tokens] = (
                expert_tokens
            )

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size

            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block : current_block + expert_blocks_needed] = (
                expert_id_new
            )

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    in_num_tokens_post_pad = torch.tensor(
        [total_padded_tokens], dtype=torch.int32, device=topk_ids.device
    )
    sorted_token_ids.copy_(in_sorted_token_ids)
    experts_ids.copy_(expert_ids)
    num_tokens_post_pad.copy_(in_num_tokens_post_pad)

    return in_sorted_token_ids, expert_ids, num_tokens_post_pad


def _group_tokens_by_expert(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
) -> dict:
    num_blocks = valid_length // block_size
    expert_tokens: dict[int, list[int]] = {}

    for block_idx in range(num_blocks):
        expert_id = expert_ids[block_idx].item()
        block_start = block_idx * block_size
        block_end = min(block_start + block_size, valid_length)

        block_tokens = sorted_ids[block_start:block_end]
        valid_tokens = block_tokens[block_tokens < total_tokens]

        if expert_id not in expert_tokens:
            expert_tokens[expert_id] = []
        expert_tokens[expert_id].extend(valid_tokens.tolist())
    return expert_tokens


def _verify_expert_level_sorting(
    actual_sorted_ids: torch.Tensor,
    golden_sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    block_size: int,
    valid_length: int,
    total_tokens: int,
):
    """
    Verify that actual_sorted_ids follows the correct expert-level sorting.
    The kernel implementation may or may not preserve original token order in
    topk_ids in the final sorted_ids, but this does not impact correctness.
    """
    golden_expert_tokens = _group_tokens_by_expert(
        golden_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    actual_expert_tokens = _group_tokens_by_expert(
        actual_sorted_ids, expert_ids, block_size, valid_length, total_tokens
    )

    assert set(golden_expert_tokens.keys()) == set(actual_expert_tokens.keys()), (
        f"Expert IDs mismatch: golden={set(golden_expert_tokens.keys())}, "
        f"actual={set(actual_expert_tokens.keys())}"
    )

    for expert_id in golden_expert_tokens:
        golden_tokens = torch.tensor(
            golden_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        actual_tokens = torch.tensor(
            actual_expert_tokens[expert_id], device=actual_sorted_ids.device
        )
        assert torch.equal(
            torch.sort(golden_tokens)[0], torch.sort(actual_tokens)[0]
        ), (
            f"Expert {expert_id} token mismatch: "
            f"golden={golden_expert_tokens[expert_id]}, "
            f"actual={actual_expert_tokens[expert_id]}"
        )


def _synchronize():
    if flaggems_vllm.vendor_name == "ascend":
        torch.npu.synchronize()
    elif flaggems_vllm.vendor_name == "sunrise":
        from flaggems_vllm.runtime import torch_device_fn

        torch_device_fn.synchronize()
    else:
        torch.cuda.synchronize()


# ref: https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_moe.py
@pytest.mark.moe_align_block_size_triton
@pytest.mark.parametrize("num_experts", [10, 128, 250, 512])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize(
    "topk_ids_shape",
    [
        (0, 10),
        (1, 10),
        (1024, 10),
        (6152, 10),
        (11575, 10),
        (16384, 10),
    ],
)
def test_accuracy_moe_align_block_size(num_experts, block_size, topk_ids_shape):
    device = flaggems_vllm.device
    dtype = torch.int32
    topk_ids = torch.randint(0, num_experts, topk_ids_shape, dtype=dtype, device=device)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    topk_ids_vllm = topk_ids.clone()
    sorted_ids_vllm = sorted_ids.clone()
    expert_ids_vllm = expert_ids.clone()
    num_tokens_post_pad_vllm = num_tokens_post_pad.clone()

    flaggems_vllm.moe_align_block_size_triton(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )

    torch_moe_align_block_size(
        topk_ids=topk_ids_vllm,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids_vllm,
        experts_ids=expert_ids_vllm,
        num_tokens_post_pad=num_tokens_post_pad_vllm,
    )

    if flaggems_vllm.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        from flaggems_vllm.runtime import torch_device_fn

        torch_device_fn.synchronize()

    _verify_expert_level_sorting(
        sorted_ids,
        sorted_ids_vllm,
        expert_ids_vllm,
        block_size,
        num_tokens_post_pad.item(),
        topk_ids.numel(),
    )
    utils.gems_assert_close(
        expert_ids, utils.to_reference(expert_ids_vllm), dtype=dtype
    )
    utils.gems_assert_close(
        num_tokens_post_pad, utils.to_reference(num_tokens_post_pad_vllm), dtype=dtype
    )


@pytest.mark.moe_align_block_size_triton
@pytest.mark.parametrize(
    ("num_experts", "block_size", "topk_ids_shape"),
    [
        (512, 64, (16384, 10)),
        (512, 64, (6152, 10)),
        (512, 64, (4727, 10)),
        (512, 64, (1905, 10)),
        (512, 64, (11575, 10)),
        (512, 64, (1032, 10)),
        (512, 64, (4201, 10)),
        (512, 64, (2056, 10)),
        (512, 64, (7561, 10)),
        (512, 64, (4104, 10)),
        (512, 64, (14281, 10)),
    ],
)
def test_accuracy_moe_align_block_size_triton(num_experts, block_size, topk_ids_shape):
    device = flaggems_vllm.device
    dtype = torch.int32
    topk_ids = torch.randint(0, num_experts, topk_ids_shape, dtype=dtype, device=device)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=dtype, device=device)
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=dtype, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    topk_ids_ref = topk_ids.clone()
    sorted_ids_ref = sorted_ids.clone()
    expert_ids_ref = expert_ids.clone()
    num_tokens_post_pad_ref = num_tokens_post_pad.clone()

    flaggems_vllm.moe_align_block_size_triton(
        topk_ids=topk_ids,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_post_pad=num_tokens_post_pad,
    )

    torch_moe_align_block_size(
        topk_ids=topk_ids_ref,
        num_experts=num_experts,
        block_size=block_size,
        sorted_token_ids=sorted_ids_ref,
        experts_ids=expert_ids_ref,
        num_tokens_post_pad=num_tokens_post_pad_ref,
    )

    _synchronize()

    _verify_expert_level_sorting(
        sorted_ids,
        sorted_ids_ref,
        expert_ids_ref,
        block_size,
        num_tokens_post_pad.item(),
        topk_ids.numel(),
    )
    utils.gems_assert_close(expert_ids, utils.to_reference(expert_ids_ref), dtype=dtype)
    utils.gems_assert_close(
        num_tokens_post_pad, utils.to_reference(num_tokens_post_pad_ref), dtype=dtype
    )


@pytest.mark.moe_align_block_size
@pytest.mark.parametrize("fast_path", ["singleton", "small_grouped"])
def test_accuracy_moe_align_block_size_fast_paths(fast_path):
    device = flaggems_vllm.device
    block_size = 8
    num_experts = 8

    if fast_path == "singleton":
        topk_ids = torch.tensor([[1, 3]], dtype=torch.int32, device=device)
        actual = moe_align_block_size_singleton(topk_ids, block_size)
    else:
        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            dtype=torch.int32,
            device=device,
        )
        actual = moe_align_block_size_small_grouped(topk_ids, num_experts, block_size)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    expected = (
        torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device),
        torch.empty(
            max_num_tokens_padded // block_size,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(1, dtype=torch.int32, device=device),
    )
    torch_moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        expected[0],
        expected[1],
        expected[2],
    )

    num_tokens = actual[2].item()
    num_blocks = num_tokens // block_size
    torch.testing.assert_close(actual[0][:num_tokens], expected[0][:num_tokens])
    torch.testing.assert_close(actual[1][:num_blocks], expected[1][:num_blocks])
    torch.testing.assert_close(actual[2], expected[2])
