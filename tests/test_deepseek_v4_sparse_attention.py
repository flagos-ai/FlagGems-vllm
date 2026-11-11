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

# SPDX-License-Identifier: Apache-2.0
# Derived from vllm-plugin-FL sparse_attn_sharedkv.py and its tests (Apache-2.0).
"""Independent math, sparse-page safety and live graph contracts."""

import math

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.deepseek_v4_sparse_attention,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend"),
]


def _oracle(args):
    q, swa, cmp, st, ct, reqs, positions, sl, cl, indices, window = args[:11]
    ratio, scale, sink, value_dim = args[11:]
    output = torch.zeros((*q.shape[:2], value_dim), dtype=torch.float64)
    for token, (request, position) in enumerate(zip(reqs.tolist(), positions.tolist())):
        if request < 0 or position < 0 or request >= sl.numel():
            continue
        rows = []

        def append(cache, table, logical, length, request=request, rows=rows):
            if logical < 0 or logical >= length:
                return
            block, offset = divmod(logical, cache.shape[1])
            if block >= table.shape[1]:
                return
            page = int(table[request, block])
            if 0 <= page < cache.shape[0]:
                rows.append(cache[page, offset].double())

        for logical in range(position - window + 1, position + 1):
            append(swa, st, logical, int(sl[request]))
        if cmp is not None:
            limit = min((position + 1) // ratio, int(cl[request]))
            for logical in indices[token].tolist():
                append(cmp, ct, logical, limit)
        if not rows:
            continue
        kv = torch.stack(rows)
        logits = q[token].double() @ kv.T * scale
        denominator = torch.logsumexp(logits, dim=-1)
        if sink is not None:
            denominator = torch.logaddexp(denominator, sink.double())
        output[token] = torch.exp(logits - denominator[:, None]) @ kv[:, :value_dim]
    return output


def _case(ratio, dtype, *, dim=64, tokens=5, sink=True, heads=4):
    generator = torch.Generator().manual_seed(731 + ratio)
    q = torch.randn(tokens, heads, dim, generator=generator).to(dtype)
    swa = torch.randn(9, 16, dim, generator=generator).to(dtype)
    compact = (
        None if ratio == 0 else torch.randn(9, 16, dim, generator=generator).to(dtype)
    )
    # NaN pages are deliberately masked by logical bounds or a -1 page table.
    swa[-1] = math.nan
    if compact is not None:
        compact[-1] = math.nan
    table = torch.tensor([[0, 2, 1, -1, 8, 99], [3, 4, 5, 6, 7, 8]], dtype=torch.int64)
    reqs = torch.tensor(
        ([0, 1, -1, 0, 9] * math.ceil(tokens / 5))[:tokens], dtype=torch.int64
    )
    positions = torch.tensor(
        ([41, 36, 12, -1, 5] * math.ceil(tokens / 5))[:tokens], dtype=torch.int64
    )
    if ratio == 128 and tokens:
        positions[0] = 511
    lengths = torch.tensor([43, 39])
    indices = torch.tensor(
        [[0, 1, 2, 2, 3, -1, 2**40, 80, 100]] * tokens, dtype=torch.int64
    ).reshape(tokens, 9)
    args = [
        q,
        swa,
        compact,
        table,
        table.clone() if compact is not None else None,
        reqs,
        positions,
        lengths,
        torch.tensor([5, 4]) if compact is not None else None,
        indices if compact is not None else None,
        17,
        ratio,
        dim**-0.5,
        torch.linspace(-2, 2, heads) if sink else None,
        dim - 3,
    ]
    device = []
    for tensor in args:
        if not isinstance(tensor, torch.Tensor):
            device.append(tensor)
            continue
        # Exercise padded innermost strides and nonzero storage offsets on NPU,
        # including all metadata. A simple .to(device) would erase these strides.
        shape = (*tensor.shape[:-1], tensor.shape[-1] * 2 + 3)
        backing = torch.empty(shape, device="npu", dtype=tensor.dtype)
        view = backing[..., 1 : 1 + 2 * tensor.shape[-1] : 2]
        view.copy_(tensor)
        assert view.storage_offset() > 0
        device.append(view)
    return args, device


def _call(args):
    return flaggems_vllm.deepseek_v4_sparse_attention(*args)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sink", [False, True])
def test_sparse_attention_independent_fp64(ratio, dtype, sink):
    cpu, device = _case(ratio, dtype, sink=sink)
    result = _call(device).cpu().double()
    tolerance = {
        torch.bfloat16: (8e-3, 8e-3),
        torch.float16: (1e-3, 1e-3),
        torch.float32: (2e-5, 2e-6),
    }
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(result, _oracle(cpu), rtol=rtol, atol=atol)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[2:]) == 0


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_sparse_attention_graph_reloads_metadata(ratio):
    cpu, device = _case(ratio, torch.bfloat16, dim=512)
    for _ in range(2):
        _call(device)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _call(device)
    # This changes valid tokens, selected pages, request lengths, and the sink
    # without changing shapes or creating a second capture.
    cpu[0] = cpu[0] * 0.7
    device[0].copy_(cpu[0])
    for index, replacement in (
        (5, torch.tensor([1, 0, 0, 1, -1])),
        (6, torch.tensor([33, 17, 12, 15, -1])),
        (7, torch.tensor([15, 35])),
        (13, torch.tensor([-1.0, 0.0, 1.0, 2.0])),
    ):
        cpu[index] = replacement
        device[index].copy_(replacement)
    cpu[3][0, 0] = 4
    device[3].copy_(cpu[3])
    if ratio:
        cpu[9][:] = -1
        cpu[9][:, 0] = 0
        device[9].copy_(cpu[9])
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu().double(), _oracle(cpu), rtol=8e-3, atol=8e-3
    )


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_attention_contiguous_cube(ratio, dtype):
    cpu, device = _case(ratio, dtype, dim=512)
    device = [
        tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in device
    ]
    result = _call(device).cpu().double()
    tolerance = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(result, _oracle(cpu), rtol=tolerance, atol=tolerance)


def test_sparse_attention_empty_and_all_masked():
    for tokens in (0, 5):
        cpu, device = _case(4, torch.float32, tokens=tokens, sink=False)
        cpu[5].fill_(-1)
        device[5].fill_(-1)
        result = _call(device).cpu()
        assert result.shape == (tokens, 4, 61)
        assert torch.count_nonzero(result) == 0


def test_sparse_attention_large_shared_storage_address():
    # A tiny view can span more than INT32_MAX elements in shared KV storage.
    # Initialize only the two live rows, not the multi-GB backing allocation.
    elements = 2**31 + 16 * 32 + 3
    free, _ = torch.npu.mem_get_info()
    if free < elements * 2 + 1024**3:
        pytest.skip("Requires room for a shared cache allocation above 2^31 elements")
    backing = torch.empty(elements, dtype=torch.float16, device="npu")
    cache = backing.as_strided((2, 16, 32), (2**31, 32, 1), storage_offset=3)
    cache[0].fill_(1.0)
    cache[1].fill_(2.0)
    query = torch.ones((1, 4, 32), dtype=torch.float16, device="npu")
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    args = [
        query,
        cache,
        None,
        table,
        None,
        torch.tensor([0], dtype=torch.int64, device="npu"),
        torch.tensor([16], dtype=torch.int64, device="npu"),
        torch.tensor([17], dtype=torch.int64, device="npu"),
        None,
        None,
        1,
        0,
        32**-0.5,
        None,
        32,
    ]
    result = _call(args).cpu()
    torch.testing.assert_close(result, torch.full_like(result, 2.0), rtol=0, atol=0)


def _packed_device_inputs(cpu):
    """Contiguous query with offset, padded KV rows, strided metadata."""
    device = []
    for index, tensor in enumerate(cpu):
        if not isinstance(tensor, torch.Tensor):
            device.append(tensor)
            continue
        if index == 0:
            backing = torch.empty(
                (tensor.shape[0] + 2, *tensor.shape[1:]),
                device="npu",
                dtype=tensor.dtype,
            )
            view = backing[1 : 1 + tensor.shape[0]]
        elif index in (1, 2):
            backing = torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] + 3),
                device="npu",
                dtype=tensor.dtype,
            )
            view = backing[..., 1 : 1 + tensor.shape[-1]]
        else:
            backing = torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] * 2 + 3),
                device="npu",
                dtype=tensor.dtype,
            )
            view = backing[..., 1 : 1 + 2 * tensor.shape[-1] : 2]
        view.copy_(tensor)
        assert view.storage_offset() > 0
        device.append(view)
    assert device[0].is_contiguous()
    assert device[1].stride(-1) == 1 and not device[1].is_contiguous()
    return device


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_attention_prefill_masks_offsets(ratio, dtype):
    cpu, _ = _case(ratio, dtype, dim=512, tokens=33, heads=8, sink=ratio != 0)
    device = _packed_device_inputs(cpu)
    result = _call(device).cpu().double()
    tolerance = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(result, _oracle(cpu), rtol=tolerance, atol=tolerance)
    assert torch.isfinite(result).all()


def _model_prefill_case(ratio, dtype, tokens=129):
    generator = torch.Generator().manual_seed(918 + ratio)
    dim, heads = 512, 8
    q = (torch.randn(tokens, heads, dim, generator=generator) * 0.3).to(dtype)
    swa = (torch.randn(65, 128, dim, generator=generator) * 0.3).to(dtype)
    compact = (
        None
        if ratio == 0
        else (torch.randn(65, 32, dim, generator=generator) * 0.3).to(dtype)
    )
    swa[0] = math.nan
    if compact is not None:
        compact[0] = math.nan
    table = torch.arange(1, 65, dtype=torch.int64).repeat(2, 1)
    table[1] = table[1].roll(7)
    table[1, 55] = 99
    compact_table = None if compact is None else table.flip(-1).clone()
    if compact_table is not None:
        compact_table[0, 0] = -1
    lengths = torch.tensor([8192, 7168], dtype=torch.int64)
    requests = torch.arange(tokens, dtype=torch.int64) % 2
    positions = lengths[requests] - tokens + torch.arange(tokens)
    topk = 512 if ratio == 4 else 64
    indices = None
    if ratio:
        indices = torch.full((tokens, topk), -1, dtype=torch.int64)
        for token in range(tokens):
            limit = int((positions[token] + 1) // ratio)
            count = min(topk, limit)
            indices[token, :count] = torch.randperm(limit, generator=generator)[:count]
        indices[:, -1] = 2**40
    # Keep the final query valid: T129 must exercise a real one-token chunk
    # after the full workspace was used, in addition to masked rows.
    requests[-2] = -1
    positions[3] = -1
    return [
        q,
        swa,
        compact,
        table,
        compact_table,
        requests,
        positions,
        lengths,
        None if ratio == 0 else lengths // ratio,
        indices,
        128,
        ratio,
        dim**-0.5,
        torch.linspace(-1, 1, heads),
        511,
    ]


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_attention_prefill_random_indices_chunk_tail(dtype, ratio):
    cpu = _model_prefill_case(ratio, dtype)
    device = _packed_device_inputs(cpu)
    result = _call(device).cpu().double()
    expected = _oracle(cpu)
    rtol, atol = (2e-3, 2e-5) if dtype == torch.float16 else (8e-3, 2e-4)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_attention_prefill_graph_workspace_reuse(ratio, dtype):
    cpu = _model_prefill_case(ratio, dtype)
    device = _packed_device_inputs(cpu)
    for _ in range(2):
        _call(device)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _call(device)
    for _ in range(2):
        cpu[0] = cpu[0] * 0.8
        device[0].copy_(cpu[0])
        cpu[5][0:2] = torch.tensor([1, 0])
        cpu[6] -= 7
        cpu[7] -= 3
        cpu[3][0, -1] = 1
        for index in (3, 5, 6, 7):
            device[index].copy_(cpu[index])
        if ratio:
            cpu[8] -= 1
            cpu[9] = cpu[9].roll(13, dims=-1)
            cpu[4][1, 0] = -1
            for index in (4, 8, 9):
                device[index].copy_(cpu[index])
        expected = _oracle(cpu)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        rtol, atol = (2e-3, 2e-5) if dtype == torch.float16 else (8e-3, 2e-4)
        torch.testing.assert_close(
            output.cpu().double(), expected, rtol=rtol, atol=atol
        )


def test_sparse_attention_prefill_wide_address():
    elements = 2**31 + 16 * 512 + 3
    free, _ = torch.npu.mem_get_info()
    if free < elements * 2 + 1024**3:
        pytest.skip("Requires room for shared KV above 2^31 elements")
    backing = torch.empty(elements, dtype=torch.float16, device="npu")
    cache = backing.as_strided((2, 16, 512), (2**31, 512, 1), storage_offset=3)
    cache[0].fill_(1.0)
    cache[1].fill_(2.0)
    tokens = 33
    args = [
        torch.zeros((tokens, 8, 512), dtype=torch.float16, device="npu"),
        cache,
        None,
        torch.tensor([[0, 1]], dtype=torch.int32, device="npu"),
        None,
        torch.zeros(tokens, dtype=torch.int64, device="npu"),
        torch.full((tokens,), 31, dtype=torch.int64, device="npu"),
        torch.tensor([32], dtype=torch.int64, device="npu"),
        None,
        None,
        128,
        0,
        512**-0.5,
        None,
        512,
    ]
    output = _call(args).cpu()
    torch.testing.assert_close(output, torch.full_like(output, 1.5), rtol=0, atol=0)


@pytest.mark.parametrize(
    "index,replacement,error",
    [
        (10, 0, ValueError),
        (11, 8, ValueError),
        (14, 513, ValueError),
        (5, torch.zeros(4, dtype=torch.int64), ValueError),
        (9, torch.zeros(5, 3, dtype=torch.float32), ValueError),
        (13, torch.zeros(3), ValueError),
        (0, torch.zeros(5, 4, 64, dtype=torch.int32), TypeError),
        (1, torch.zeros(9, 16, 64, dtype=torch.float32), TypeError),
    ],
)
def test_sparse_attention_rejects_invalid_contract(index, replacement, error):
    _, args = _case(4, torch.bfloat16)
    args[index] = (
        replacement.to("npu") if isinstance(replacement, torch.Tensor) else replacement
    )
    with pytest.raises(error):
        _call(args)


def test_sparse_attention_cpu_query_rejected():
    cpu, _ = _case(4, torch.bfloat16)
    with pytest.raises(ValueError, match="NPU query"):
        _call(cpu)


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_sparse_attention_defaults_and_zero_topk(ratio):
    cpu, device = _case(ratio, torch.float32, sink=False)
    cpu[12] = device[12] = None
    cpu[14] = device[14] = None
    if ratio:
        cpu[9] = cpu[9][:, :0]
        device[9] = device[9][:, :0]
    expected_args = list(cpu)
    expected_args[12] = cpu[0].shape[-1] ** -0.5
    expected_args[14] = cpu[0].shape[-1]
    result = _call(device)
    assert result.dtype == device[0].dtype
    assert result.device == device[0].device
    assert result.untyped_storage().data_ptr() != device[0].untyped_storage().data_ptr()
    torch.testing.assert_close(
        result.cpu().double(), _oracle(expected_args), rtol=2e-5, atol=2e-6
    )
