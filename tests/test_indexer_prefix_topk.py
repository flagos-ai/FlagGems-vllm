# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent bit-key reference and mutable-input graph contracts."""

import numpy as np
import pytest
import torch

import flaggems_vllm


def reference(values, lengths, count):
    bits = values.view(np.uint32).copy()
    bits[values == 0.0] = 0
    masks = np.where(bits & 0x80000000, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    keys = bits ^ masks
    result = np.full((values.shape[0], count), -1, dtype=np.int32)
    for row, length in enumerate(lengths):
        end = min(max(int(length), 0), values.shape[1])
        order = np.lexsort((np.arange(end), -keys[row, :end].astype(np.int64)))[:count]
        finite = np.isfinite(values[row, order])
        result[row, : len(order)] = np.where(finite, order, -1)
    return result


@pytest.mark.indexer_prefix_topk
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="NVIDIA CUDA required",
)
@pytest.mark.parametrize("width", [16384, 32768, 65536, 204800])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prefix_topk_reference_and_changed_input_graph(width, length_dtype):
    from flag_gems.utils.triton_version_utils import HAS_TLE

    if not HAS_TLE:
        pytest.skip("the optional prefix algorithm requires TLE")
    count = 2048
    rng = np.random.default_rng(2026 + width)
    cpu = rng.standard_normal((6, width), dtype=np.float32)
    lengths = np.array([0, 1023, 2048, 4096, width - 1, width], dtype=np.int64)
    cpu[3, :4096] = 0.0
    cpu[3, 1:4096:2] = -0.0
    cpu[4, :3] = [np.inf, -np.inf, np.nan]
    for row, end in enumerate(lengths):
        cpu[row, end:] = -np.inf
    scores = torch.from_numpy(cpu).cuda()
    device_lengths = torch.as_tensor(lengths, dtype=length_dtype, device="cuda")
    output = torch.full((6, count), -17, dtype=torch.int32, device="cuda")
    assert flaggems_vllm.supports_indexer_prefix_topk(
        scores, device_lengths, count, output
    )
    actual = flaggems_vllm.indexer_prefix_topk(scores, device_lengths, count, output)
    assert actual is output
    np.testing.assert_array_equal(actual.cpu().numpy(), reference(cpu, lengths, count))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        flaggems_vllm.indexer_prefix_topk(scores, device_lengths, count, output)
        with torch.cuda.graph(graph, stream=stream):
            flaggems_vllm.indexer_prefix_topk(scores, device_lengths, count, output)
    torch.cuda.current_stream().wait_stream(stream)
    for data in (cpu[:, ::-1].copy(), cpu):
        changed = data.copy()
        for row, end in enumerate(lengths):
            changed[row, end:] = -np.inf
        scores.copy_(torch.from_numpy(changed))
        output.fill_(-17)
        graph.replay()
        torch.cuda.synchronize()
        np.testing.assert_array_equal(
            output.cpu().numpy(), reference(changed, lengths, count)
        )


@pytest.mark.indexer_prefix_topk
@pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="CUDA descriptors required",
)
@torch.inference_mode()
def test_prefix_topk_rejects_one_changed_descriptor_before_launch():
    scores = torch.zeros((1, 4096), device="cuda")
    lengths = torch.full((1,), 4096, dtype=torch.int32, device="cuda")
    output = torch.full((1, 2048), 17, dtype=torch.int32, device="cuda")
    if not flaggems_vllm.supports_indexer_prefix_topk(scores, lengths, 2048, output):
        pytest.skip("baseline descriptor needs the optional TLE backend")
    assert not flaggems_vllm.supports_indexer_prefix_topk(
        scores.half(), lengths, 2048, output
    )
    assert torch.equal(output, torch.full_like(output, 17))
