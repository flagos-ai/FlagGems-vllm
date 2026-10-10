# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm as gems


@pytest.mark.parametrize("shape", [(4, 2, 3, 5), (8, 2, 128, 128)])
def test_state_rows_preserve_cache_and_padding(shape):
    state = torch.randn(shape, device=gems.device)
    before = state.clone()
    slots = torch.tensor([2, -1, 0], dtype=torch.int32, device=gems.device)
    actual = gems.gather_state_rows(state, slots)
    expected = torch.stack((state[2], torch.zeros_like(state[0]), state[0]))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    values = torch.randn_like(actual)
    gems.scatter_state_rows(state, slots, values)
    before[2] = values[0]
    before[0] = values[2]
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    gems.zero_state_rows(state, slots)
    before[2].zero_()
    before[0].zero_()
    torch.testing.assert_close(state, before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_state_rows_preserve_physical_slot_padding(dtype):
    storage = torch.randn(5, 2 * 3 * 5 + 8, dtype=dtype, device=gems.device)
    state = torch.as_strided(storage, (5, 2, 3, 5), (38, 15, 5, 1))
    before = storage.clone()
    expected = torch.as_strided(before, state.shape, state.stride())
    slots = torch.tensor([3, -1, 1], dtype=torch.int32, device=gems.device)
    gathered = gems.gather_state_rows(state, slots)
    torch.testing.assert_close(gathered[0], state[3], atol=0, rtol=0)
    assert torch.count_nonzero(gathered[1]) == 0
    values = torch.randn_like(gathered)
    gems.scatter_state_rows(state, slots, values)
    expected[3], expected[1] = values[0], values[2]
    torch.testing.assert_close(storage, before, atol=0, rtol=0)
    gems.zero_state_rows(state, slots)
    expected[3].zero_()
    expected[1].zero_()
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


def test_state_rows_reject_overlapping_slots_before_mutation():
    storage = torch.randn(100, device=gems.device)
    state = torch.as_strided(storage, (4, 3, 5), (7, 5, 1))
    before = storage.clone()
    slots = torch.tensor([0, 1], dtype=torch.int32, device=gems.device)
    with pytest.raises(ValueError, match="overlap"):
        gems.zero_state_rows(state, slots)
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_copy_to_stable_buffer_and_graph(dtype):
    src = torch.randn(3, 5, device=gems.device, dtype=dtype).t()
    out = torch.empty(src.shape, dtype=dtype, device=gems.device)
    gems.copy_to(src, out)
    torch.testing.assert_close(out, src, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gems.copy_to(src, out)
    ptr = out.data_ptr()
    src.copy_(torch.randn_like(src))
    out.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out, src, atol=0, rtol=0)
    assert out.data_ptr() == ptr


def test_tle_missing_pipeline_api_is_rejected_before_execution():
    from types import SimpleNamespace

    from flaggems_vllm.ops.flashmla_sparse import _has_tle_pipeline_api

    fields = ("alloc", "copy", "local_ptr", "smem", "warp_specialize")
    gpu = SimpleNamespace(**dict.fromkeys(fields, object()))
    supported = SimpleNamespace(pipe=object(), gpu=gpu)
    assert _has_tle_pipeline_api(supported)
    assert not _has_tle_pipeline_api(SimpleNamespace(gpu=gpu))
    for field in fields:
        missing = SimpleNamespace(
            **{name: object() for name in fields if name != field}
        )
        assert not _has_tle_pipeline_api(SimpleNamespace(pipe=object(), gpu=missing))


@pytest.mark.parametrize("length", [0, 1, 127, 128])
def test_sparse_mla_public_entry_with_optional_pipeline_abi(length):
    from .test_flash_mla_sparse_fwd import FlashmlaSparseTestKit

    torch.manual_seed(881)
    q = torch.randn(3, 64, 512, dtype=torch.bfloat16, device=gems.device)
    kv = torch.randn(128, 1, 512, dtype=q.dtype, device=q.device)
    indices = (
        torch.arange(128, dtype=torch.int32, device=q.device)
        .expand(3, 1, -1)
        .contiguous()
    )
    lengths = torch.full((3,), length, dtype=torch.int32, device=q.device)
    expected = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        3, 128, 64, 1, 512, 128, q, kv, indices, 512**-0.5, 512, None, lengths
    )
    actual = gems.flash_mla_sparse_fwd(q, kv, indices, 512**-0.5, topk_length=lengths)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual[2], expected[2], atol=1e-4, rtol=1e-4)
