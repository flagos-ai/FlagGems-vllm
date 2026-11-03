# SPDX-License-Identifier: Apache-2.0
"""FP8 page addressing, numerical parity and changing-input graph replay."""

import pytest
import torch

from flaggems_vllm import paged_mqa_logits

pytestmark = pytest.mark.gpu


def make_case(page_size, padding=0, pool_pages=128, batch=3):
    torch.manual_seed(11)
    device = "cuda"
    width, heads, max_len = 128, 32, 2051
    stride = page_size * (width + 4) + padding
    raw = torch.zeros(pool_pages, stride, dtype=torch.uint8, device=device)
    cache = torch.as_strided(
        raw, (pool_pages, page_size, 1, width + 4), (stride, width + 4, width + 4, 1)
    )
    keys = torch.randn(pool_pages, page_size, width, device=device).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pool_pages, page_size, device=device) * 0.05 + 0.01
    raw[:, : page_size * width].copy_(keys.view(torch.uint8).reshape(pool_pages, -1))
    raw[:, page_size * width : page_size * (width + 4)].copy_(scales.view(torch.uint8))
    query = torch.randn(batch, 1, heads, width, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, device=device)
    table = torch.stack(
        [torch.randperm(pool_pages, device=device)[:65] for _ in range(batch)]
    ).int()
    lens = torch.full((batch,), 65, dtype=torch.int32, device=device)
    return query, cache, weights, lens, table, keys, scales, max_len


def candidate(case):
    q, cache, weights, lens, table, _, _, max_len = case
    return paged_mqa_logits(
        (q, None),
        cache,
        weights,
        lens,
        table,
        None,
        max_len,
        True,
    )


def reference(case):
    q, _, weights, lens, table, keys, scales, max_len = case
    result = torch.full((len(lens), max_len), -torch.inf, device=q.device)
    for row, length in enumerate(lens.tolist()):
        idx = table[row].long()
        k = keys.float()[idx].reshape(-1, 128)[:length]
        s = scales[idx].reshape(-1)[:length]
        dots = q[row, 0].float() @ k.T
        result[row, :length] = (torch.relu(dots * s) * weights[row, :, None]).sum(0)
    return result


@pytest.mark.parametrize("page_size,padding", [(32, 0), (32, 256), (64, 0), (64, 256)])
def test_page_stride_eager_and_changing_graph_inputs(page_size, padding):
    case = make_case(page_size, padding)
    for length in (0, 1, page_size - 1, page_size, page_size + 1, 2049):
        case[3].copy_(torch.tensor([length, length // 2, 0], device="cuda"))
        torch.testing.assert_close(
            candidate(case), reference(case), atol=2e-5, rtol=2e-5
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate(case)
    # Rewrite metadata in place: empty/padded row becomes live, and logical
    # pages resolve to different physical pages after the capture.
    case[3].copy_(torch.tensor([33, 2051, 64], device="cuda"))
    case[4].copy_(case[4].flip(1))
    expected = reference(case)
    for iteration in range(10):
        case[0].copy_(
            (case[0].float() * (-1 if iteration % 2 else 0.5)).to(case[0].dtype)
        )
        captured.fill_(torch.nan)
        expected = reference(case)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=2e-5, rtol=2e-5)
