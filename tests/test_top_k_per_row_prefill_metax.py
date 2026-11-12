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


@pytest.mark.top_k_per_row_prefill
@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX layout regression")
@pytest.mark.parametrize(
    "width,padding,column_step",
    [(513, 63, 1), (8193, 63, 1), (9026, 62, 1), (16384, 64, 1), (4095, 7, 2)],
)
def test_metax_prefill_topk_pitched_layout(width, padding, column_step):
    """Pitched logits preserve exact top-k values and output/input canaries."""
    generator = torch.Generator().manual_seed(103)
    rows, k = 4, 512
    cpu = torch.randn((rows, width * column_step + padding + 1), generator=generator)
    storage = cpu.to("cuda")
    logits = storage[:, 1 : 1 + width * column_step : column_step]
    expected_input = cpu[:, 1 : 1 + width * column_step : column_step]
    starts_cpu = torch.tensor(
        [min(v, width - k) for v in (0, 1, 3, 7)], dtype=torch.int32
    )
    ends_cpu = torch.full((rows,), width, dtype=torch.int32)
    starts, ends = starts_cpu.to("cuda"), ends_cpu.to("cuda")
    output = torch.full((rows, k + 8), -12345, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        logits, starts, ends, output[:, :k], rows, logits.stride(0), logits.stride(1), k
    )
    actual = output.cpu()
    assert torch.equal(storage.cpu(), cpu)
    assert torch.equal(actual[:, k:], torch.full_like(actual[:, k:], -12345))
    for row in range(rows):
        indices = actual[row, :k].long() + int(starts_cpu[row])
        assert indices.unique().numel() == k
        assert int(indices.min()) >= int(starts_cpu[row]) and int(indices.max()) < width
        values = expected_input[row, indices].sort().values
        reference = (
            torch.topk(expected_input[row, int(starts_cpu[row]) :], k)
            .values.sort()
            .values
        )
        assert torch.equal(values, reference)
