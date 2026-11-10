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


@pytest.mark.parametrize("width", [1024, 2048, 3072])
@pytest.mark.top_k_per_row_decode
def test_topk_threshold_ties_and_invalid_padding(width):
    # Historical >=threshold positional truncation dropped the high last item.
    logits = torch.zeros(6, width, device="cuda")
    valid = torch.tensor([0, 1, 511, 512, 513, width], dtype=torch.int32, device="cuda")
    logits[:, -1] = 10
    result = torch.empty(6, 512, dtype=torch.int32, device="cuda")

    def call():
        ops.top_k_per_row_decode(
            logits,
            1,
            valid,
            result,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            512,
        )

    def check():
        for row, length in enumerate(valid.cpu().tolist()):
            values = logits[row, :length].cpu()
            selected = (
                torch.argsort(values, descending=True, stable=True)[:512].sort().values
            )
            expected = torch.full((512,), -1, dtype=torch.int32)
            expected[: selected.numel()] = selected.to(torch.int32)
            torch.testing.assert_close(result[row].cpu(), expected, atol=0, rtol=0)

    call()
    check()
    check_replays(
        call, lambda value: logits[:, -1].fill_(value), check, outputs=(result,)
    )
