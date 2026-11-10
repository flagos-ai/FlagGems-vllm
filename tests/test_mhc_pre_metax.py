# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
import pytest
import torch

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(
    ops.vendor_name != "metax", reason="MetaX specialization"
)


def check_replays(call, update, check, outputs=()):
    # No compilation, allocation growth or tuning is permitted during capture.
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        update(value)
        for output in outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.mhc_pre
@pytest.mark.parametrize(
    "rows,hidden",
    [(1, 4096), (32, 4096), (64, 4096), (65, 4096), (128, 4096), (129, 4096), (3, 128)],
)
def test_public_mhc_pre_reference_and_new_input_replay(rows, hidden):
    from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre_ref

    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(rows, 4, hidden)
    result = []

    def call():
        result[:] = ops.mhc_pre(**data)

    def check():
        cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        expected = mhc_pre_ref(**cpu)
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-2, atol=1e-2)

    call()
    check()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        data["residual"].fill_(value)
        for output in result:
            output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.mhc_pre
def test_mhc_pre_empty_batch_preserves_shape():
    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(0, 4, 4096)
    post, comb, layer = ops.mhc_pre(**data)
    assert post.shape == (0, 4, 1) and comb.shape == (0, 4, 4)
    assert layer.shape == (0, 4096) and layer.dtype == torch.bfloat16
