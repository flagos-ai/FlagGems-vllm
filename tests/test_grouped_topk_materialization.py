# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The materialized sigmoid contract differs from the default tanh route."""
import pytest
import torch

from flaggems_vllm import grouped_topk, runtime

from .grouped_topk_materialization_utils import assert_exact, candidate, reference

pytestmark = pytest.mark.skipif(
    runtime.device.vendor_name != "metax",
    reason="MetaX-qualified explicit mode",
)


@pytest.mark.parametrize("m", [1, 3, 12, 64, 513, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "pattern", ["random", "ties", "boundary", "nan", "infinity", "few_finite"]
)
def test_materialized_sigmoid_exact(m, dtype, pattern):
    torch.manual_seed(7341 + m)
    scores = torch.randn(m, 896, device="cuda", dtype=dtype)
    bias = torch.randn(896, device="cuda", dtype=dtype) * 0.1
    if pattern == "ties":
        scores.zero_()
        bias.zero_()
    elif pattern == "boundary":
        scores.copy_((torch.arange(896, device="cuda") % 32).expand(m, -1).to(dtype))
        bias.zero_()
    elif pattern == "nan":
        scores[:, 0] = float("nan")
    elif pattern == "infinity":
        bias[0] = float("inf")
    elif pattern == "few_finite":
        bias.fill_(-float("inf"))
        bias[:15] = 0
    assert_exact(candidate(scores, bias), reference(scores, bias))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("m", [7, 64])
def test_materialized_sigmoid_mixed_stride_graph(m, dtype):
    scores = torch.randn(m, 912, device="cuda", dtype=dtype)[:, :896]
    bias = torch.randn(896, device="cuda", dtype=torch.float32)
    candidate(scores, bias)
    candidate(scores, bias)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = candidate(scores, bias)
    for _ in range(3):
        scores.normal_()
        bias.normal_(0, 0.1)
        graph.replay()
        assert_exact(output, reference(scores, bias))


@pytest.mark.parametrize("m,e", [(0, 896), (4097, 896), (8, 256)])
def test_materialized_sigmoid_rejects_unqualified_shape(m, e):
    scores = torch.zeros(m, e, device="cuda")
    bias = torch.zeros(e, device="cuda")
    with pytest.raises(NotImplementedError):
        candidate(scores, bias)


def test_materialized_sigmoid_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown sigmoid"):
        grouped_topk(
            torch.zeros(1, 896, device="cuda"),
            1,
            1,
            16,
            True,
            1.0,
            torch.zeros(896, device="cuda"),
            1,
            sigmoid_mode="unknown",
        )
