# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch
import triton

import flaggems_vllm as gems
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

pytestmark = pytest.mark.skipif(
    gems.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0)
    or not has_triton_tle_attrs(("gpu.alloc", "gpu.wgmma", "gpu.wgmma_wait"), 3, 6, 0),
    reason="requires SM90 and FlagTree TLE WGMMA",
)


def _case(prefill, request_lengths, query_counts, *, dyadic=True):
    """Use native chunk metadata and shuffled physical page backing."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(284)
    page_counts = [triton.cdiv(n, 64) for n in request_lengths]
    physical = torch.randperm(sum(page_counts), device=device, generator=generator)
    # A column slice keeps padded row stride, as in vLLM's block table.
    table = torch.full(
        (len(request_lengths), max(page_counts) + 3),
        -1,
        dtype=torch.int32,
        device=device,
    )[:, : max(page_counts)]
    kv = torch.empty((len(physical), 64, 132), dtype=torch.uint8, device=device)
    keys = kv.view(len(physical), -1)[:, :8192].view(torch.float8_e4m3fn)
    scales = kv.view(len(physical), -1)[:, 8192:].view(torch.float32)
    rows = sum(query_counts)
    q_shape = (
        (rows, 64, 128) if prefill else (len(request_lengths), query_counts[0], 64, 128)
    )
    if dyadic:
        keys.copy_(
            torch.randint(-8, 9, keys.shape, device=device, generator=generator) / 8
        )
        q = (torch.randint(-8, 9, q_shape, device=device, generator=generator) / 8).to(
            torch.float8_e4m3fn
        )
        scales.copy_(
            torch.randint(1, 5, scales.shape, device=device, generator=generator) / 4
        )
        weights = (
            torch.randint(-8, 9, (rows, 72), device=device, generator=generator)[:, :64]
            / 64
        )
    else:
        keys.copy_(
            torch.randn(
                keys.shape, device=device, dtype=torch.bfloat16, generator=generator
            )
        )
        q = torch.randn(
            q_shape, device=device, dtype=torch.bfloat16, generator=generator
        ).to(torch.float8_e4m3fn)
        scales.copy_(
            torch.rand(scales.shape, device=device, generator=generator) * 0.8 + 0.2
        )
        weights = (
            torch.randn((rows, 64), device=device, generator=generator)
            / (64 * 128) ** 0.5
        )
    starts, ends, lens, row_requests = [], [], [], []
    k_offset, page_offset = 0, 0
    offsets = [0]
    for request, (n, num_queries, count) in enumerate(
        zip(request_lengths, query_counts, page_counts)
    ):
        table[request, :count] = physical[page_offset : page_offset + count]
        visible = list(range(n - num_queries + 1, n + 1))
        starts.extend([k_offset] * num_queries)
        ends.extend(k_offset + length for length in visible)
        lens.extend(visible)
        row_requests.extend([request] * num_queries)
        k_offset += n
        page_offset += count
        offsets.append(k_offset)

    def ints(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    out_storage = torch.full((rows + 4, 2048), -777, device=device, dtype=torch.int32)
    out = out_storage[2 : rows + 2]
    workspace = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in gems.flash_topk_workspace_specs(rows, max(request_lengths))
    )
    return dict(
        q=q,
        weights=weights,
        kv=kv,
        table=table,
        starts=ints(starts),
        ends=ints(ends),
        lens=ints(lens),
        token_to_seq=ints(
            [req for req, length in enumerate(request_lengths) for _ in range(length)]
        ),
        offsets=ints(offsets),
        max_context=max(request_lengths),
        out=out,
        out_storage=out_storage,
        workspace=workspace,
        row_requests=row_requests,
        request_lengths=request_lengths,
    )


def _call(prefill, case):
    q = case["q"], None
    kv, weights, table = (case[k] for k in ("kv", "weights", "table"))
    args = (case["max_context"], case["out"])
    if prefill:
        return gems.flash_topk_prefill(
            q,
            kv,
            weights,
            table,
            case["starts"],
            case["ends"],
            case["token_to_seq"],
            *args,
            workspace=case["workspace"],
        )
    lengths = case["lens"].view(case["q"].shape[:2])
    return gems.flash_topk_decode(
        q,
        kv.unsqueeze(-2),
        weights,
        table,
        lengths,
        *args,
        workspace=case["workspace"],
    )


def _check(case):
    # CPU unpacks each physical page independently of the GPU kernel. Dyadic
    # inputs keep each multiply/reduction exact for these small test values.
    kv = case["kv"].cpu().view(-1, 8448)
    table = case["table"].cpu()
    keys = []
    scales = []
    for i, n in enumerate(case["request_lengths"]):
        pages = kv[table[i, : triton.cdiv(n, 64)].long()]
        keys.append(
            pages[:, :8192].contiguous().view(torch.float8_e4m3fn).view(-1, 128).float()
        )
        scales.append(pages[:, 8192:].contiguous().view(torch.float32).view(-1))
    q = case["q"].cpu().reshape(-1, 64, 128).float()
    weights = case["weights"].cpu()
    lengths = case["lens"].cpu()
    result = case["out"].cpu()
    for row, request in enumerate(case["row_requests"]):
        length = int(lengths[row])
        ids = result[row]
        valid = ids[ids >= 0].long()
        count = min(length, 2048)
        assert len(valid) == count and valid.unique().numel() == count
        assert (valid < length).all()
        assert (ids[count:] == -1).all()
        if length > 2048:
            scores = ((keys[request][:length] @ q[row].T).relu() * weights[row]).sum(1)
            scores = scores * scales[request][:length]
            actual_scores = case["workspace"][0][row, :length].cpu()
            torch.testing.assert_close(actual_scores, scores, atol=0, rtol=0)
            cutoff = scores.topk(2048).values[-1]
            assert (scores[valid] >= cutoff).all()
            assert (scores[valid] > cutoff).sum() == (scores > cutoff).sum()
        else:
            torch.testing.assert_close(valid, torch.arange(length))
    assert (case["out_storage"][:2] == -777).all()
    assert (case["out_storage"][-2:] == -777).all()


@pytest.mark.flash_topk_prefill
@pytest.mark.parametrize(
    "lengths,counts",
    [
        ([1, 64, 127], [1, 2, 1]),
        ([2113, 2321], [33, 34]),
        ([2113, 8321], [1, 2]),
    ],
)
def test_flash_topk_prefill(lengths, counts):
    assert gems.flash_topk_prefill.__module__.endswith("hopper.ops.flash_topk")
    case = _case(True, lengths, counts)
    result = _call(True, case)
    assert result is case["out"]
    _check(case)


@pytest.mark.flash_topk_decode
@pytest.mark.parametrize(
    "lengths,counts",
    [([64, 128], [1, 1]), ([2049, 4115], [2, 2]), ([8193, 4097], [1, 1])],
)
def test_flash_topk_decode(lengths, counts):
    assert gems.flash_topk_decode.__module__.endswith("hopper.ops.flash_topk")
    case = _case(False, lengths, counts)
    case["lens"][0] = 0
    result = _call(False, case)
    assert result is case["out"]
    _check(case)


@pytest.mark.flash_topk_decode
@pytest.mark.flash_topk_prefill
@pytest.mark.parametrize("prefill", [False, True])
def test_flash_topk_ties_and_graph_replay(prefill):
    case = _case(prefill, [2114, 2135], [2, 2])
    _call(prefill, case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call(prefill, case)

    # Change both the causal intervals and score distribution. A stale
    # selection cannot pass the independently recomputed cutoff.
    case["lens"].sub_(3)
    case["ends"].sub_(3)
    scales = case["kv"].view(-1, 8448)[:, 8192:].view(torch.float32)
    scales.copy_(scales.flip(1))
    case["out"].fill_(-777)
    graph.replay()
    _check(case)

    # A fully tied row larger than the cutoff must still return unique IDs.
    case["weights"].zero_()
    case["out"].fill_(-777)
    graph.replay()
    _check(case)


@pytest.mark.flash_topk_decode
def test_flash_topk_invalid_output_before_launch():
    case = _case(False, [2049], [1])
    output = case["out"]
    case["out"] = output[:, :1024]
    with pytest.raises(ValueError, match="out must"):
        _call(False, case)
    assert (output == -777).all()

    case["out"] = output
    scratch = case["workspace"]
    case["workspace"] = (scratch[0], case["lens"], scratch[2])
    with pytest.raises(ValueError, match="must not overlap"):
        _call(False, case)
    assert (output == -777).all()


@pytest.mark.flash_topk_decode
@pytest.mark.flash_topk_prefill
@pytest.mark.parametrize("prefill", [False, True])
def test_flash_topk_invalid_device_metadata_does_not_read_out_of_bounds(prefill):
    case = _case(prefill, [2049, 2081], [1, 1])
    if prefill:
        case["ends"][0] = case["token_to_seq"].numel() + 1
        case["token_to_seq"][case["starts"][1]] = len(case["request_lengths"])
    else:
        case["lens"].fill_(case["max_context"] + 1024)
    _call(prefill, case)
    assert (case["out"] == -1).all()
    assert (case["out_storage"][:2] == -777).all()
    assert (case["out_storage"][-2:] == -777).all()
