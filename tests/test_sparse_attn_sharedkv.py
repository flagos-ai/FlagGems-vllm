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

"""Accuracy of the six validated DSV4 SparseAttnSharedKV shapes."""

import math

import pytest
import torch

import flaggems_vllm
from tests.sparse_attn_sharedkv_utils import (
    ATOL,
    CASES,
    RTOL,
    make_inputs,
    official_operator,
    output_tensor,
    shape_label,
)

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
npu_only = pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@torch.inference_mode()
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_accuracy(case):
    official = official_operator()
    if official is None:
        pytest.skip("requires the vLLM-Ascend SparseAttnSharedKV reference")
    inputs = make_inputs(case, flaggems_vllm.device)
    expected = output_tensor(official(**inputs))
    actual, lse = flaggems_vllm.sparse_attn_sharedkv(
        **inputs,
        host_q_lens=(case.q_len,),
        host_kv_lens=(case.kv_len,),
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), atol=ATOL, rtol=RTOL
    )
    assert lse.numel() == 0 and lse.dtype == torch.float32
    assert actual.shape == inputs["q"].shape and actual.dtype == torch.bfloat16


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_rejects_cpu():
    module = __import__(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_attn_sharedkv",
        fromlist=["sparse_attn_sharedkv"],
    )
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        module.sparse_attn_sharedkv(torch.empty((1, 64, 512)))


@torch.inference_mode()
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_rejects_unsupported_layout():
    case = CASES[0]
    inputs = make_inputs(case, flaggems_vllm.device)
    inputs["layout_q"] = "BSND"
    with pytest.raises(NotImplementedError, match="TND"):
        flaggems_vllm.sparse_attn_sharedkv(
            **inputs,
            host_q_lens=(case.q_len,),
            host_kv_lens=(case.kv_len,),
        )


@torch.inference_mode()
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_rejects_partial_host_lengths():
    case = CASES[0]
    inputs = make_inputs(case, flaggems_vllm.device)
    with pytest.raises(ValueError, match="together"):
        flaggems_vllm.sparse_attn_sharedkv(**inputs, host_q_lens=(case.q_len,))


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_case_labels():
    for case in CASES:
        label = shape_label(case)
        assert case.name in label or case.mode in label


# Independent explicit-metadata math and graph contracts.
def _explicit_oracle(args):
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


def _explicit_case(ratio, dtype, *, dim=64, tokens=5, sink=True, heads=4):
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


def _explicit_call(args):
    (
        q,
        ori,
        cmp,
        ot,
        ct,
        requests,
        positions,
        ol,
        cl,
        indices,
        window,
        ratio,
        scale,
        sinks,
        value_dim,
    ) = args
    out, lse = flaggems_vllm.sparse_attn_sharedkv(
        q,
        ori_kv=ori,
        cmp_kv=cmp,
        ori_block_table=ot,
        cmp_block_table=ct,
        request_ids=requests,
        positions=positions,
        ori_lengths=ol,
        cmp_lengths=cl,
        cmp_sparse_indices=indices,
        ori_win_left=window - 1,
        cmp_ratio=ratio,
        softmax_scale=scale,
        sinks=sinks,
        value_dim=value_dim,
    )
    assert lse.shape == (0,) and lse.dtype == torch.float32 and lse.device == q.device
    return out


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_independent_fp64(ratio, dtype, sink):
    cpu, device = _explicit_case(ratio, dtype, sink=sink)
    result = _explicit_call(device).cpu().double()
    tolerance = {
        torch.bfloat16: (8e-3, 8e-3),
        torch.float16: (1e-3, 1e-3),
        torch.float32: (2e-5, 2e-6),
    }
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(result, _explicit_oracle(cpu), rtol=rtol, atol=atol)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[2:]) == 0


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_graph_reloads_metadata(ratio):
    cpu, device = _explicit_case(ratio, torch.bfloat16, dim=512)
    for _ in range(2):
        _explicit_call(device)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _explicit_call(device)
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
        output.cpu().double(), _explicit_oracle(cpu), rtol=8e-3, atol=8e-3
    )


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_contiguous_cube(ratio, dtype):
    cpu, device = _explicit_case(ratio, dtype, dim=512)
    device = [
        tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in device
    ]
    result = _explicit_call(device).cpu().double()
    tolerance = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(
        result, _explicit_oracle(cpu), rtol=tolerance, atol=tolerance
    )


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_empty_and_all_masked():
    for tokens in (0, 5):
        cpu, device = _explicit_case(4, torch.float32, tokens=tokens, sink=False)
        cpu[5].fill_(-1)
        device[5].fill_(-1)
        result = _explicit_call(device).cpu()
        assert result.shape == (tokens, 4, 61)
        assert torch.count_nonzero(result) == 0


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_large_shared_storage_address():
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
    result = _explicit_call(args).cpu()
    torch.testing.assert_close(result, torch.full_like(result, 2.0), rtol=0, atol=0)


def _explicit_packed_device_inputs(cpu):
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
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_prefill_masks_offsets(ratio, dtype):
    cpu, _ = _explicit_case(ratio, dtype, dim=512, tokens=33, heads=8, sink=ratio != 0)
    device = _explicit_packed_device_inputs(cpu)
    result = _explicit_call(device).cpu().double()
    tolerance = 1e-3 if dtype == torch.float16 else 8e-3
    torch.testing.assert_close(
        result, _explicit_oracle(cpu), rtol=tolerance, atol=tolerance
    )
    assert torch.isfinite(result).all()


def _explicit_model_prefill_case(ratio, dtype, tokens=129):
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
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_prefill_random_indices_chunk_tail(dtype, ratio):
    cpu = _explicit_model_prefill_case(ratio, dtype)
    device = _explicit_packed_device_inputs(cpu)
    result = _explicit_call(device).cpu().double()
    expected = _explicit_oracle(cpu)
    rtol, atol = (2e-3, 2e-5) if dtype == torch.float16 else (8e-3, 2e-4)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_prefill_graph_workspace_reuse(ratio, dtype):
    cpu = _explicit_model_prefill_case(ratio, dtype)
    device = _explicit_packed_device_inputs(cpu)
    for _ in range(2):
        _explicit_call(device)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _explicit_call(device)
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
        expected = _explicit_oracle(cpu)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        rtol, atol = (2e-3, 2e-5) if dtype == torch.float16 else (8e-3, 2e-4)
        torch.testing.assert_close(
            output.cpu().double(), expected, rtol=rtol, atol=atol
        )


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_prefill_wide_address():
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
    output = _explicit_call(args).cpu()
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
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_rejects_invalid_contract(
    index, replacement, error
):
    _, args = _explicit_case(4, torch.bfloat16)
    args[index] = (
        replacement.to("npu") if isinstance(replacement, torch.Tensor) else replacement
    )
    with pytest.raises(error):
        _explicit_call(args)


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_cpu_query_rejected():
    cpu, _ = _explicit_case(4, torch.bfloat16)
    with pytest.raises(ValueError, match="NPU query"):
        _explicit_call(cpu)


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_explicit_defaults_and_zero_topk(ratio):
    cpu, device = _explicit_case(ratio, torch.float32, sink=False)
    cpu[12] = device[12] = None
    cpu[14] = device[14] = None
    if ratio:
        cpu[9] = cpu[9][:, :0]
        device[9] = device[9][:, :0]
    expected_args = list(cpu)
    expected_args[12] = cpu[0].shape[-1] ** -0.5
    expected_args[14] = cpu[0].shape[-1]
    result = _explicit_call(device)
    assert result.dtype == device[0].dtype
    assert result.device == device[0].device
    assert result.untyped_storage().data_ptr() != device[0].untyped_storage().data_ptr()
    torch.testing.assert_close(
        result.cpu().double(), _explicit_oracle(expected_args), rtol=2e-5, atol=2e-6
    )


@pytest.mark.sparse_attn_sharedkv
@npu_only
@pytest.mark.parametrize(
    "extra",
    [
        {"cu_seqlens_q": None},
        {"metadata": None},
        {"ori_sparse_indices": None},
        {"host_q_lens": (5,), "host_kv_lens": (43,)},
        {"max_model_len": 43},
        {"ori_win_left": -1},
        {"ori_win_right": 1},
        {"ori_mask_mode": 0},
        {"cmp_mask_mode": 0},
        {"layout_q": "BSND"},
        {"layout_kv": "ND"},
        {"return_softmax_lse": True},
    ],
)
def test_sparse_attn_sharedkv_explicit_rejects_mixed_contract(extra):
    _, args = _explicit_case(0, torch.bfloat16)
    kwargs = dict(
        ori_kv=args[1],
        ori_block_table=args[3],
        request_ids=args[5],
        positions=args[6],
        ori_lengths=args[7],
    )
    kwargs.update(extra)
    with pytest.raises((ValueError, NotImplementedError)):
        flaggems_vllm.sparse_attn_sharedkv(args[0], **kwargs)


@pytest.mark.sparse_attn_sharedkv
@npu_only
@pytest.mark.parametrize("missing", ["request_ids", "positions", "ori_lengths"])
def test_sparse_attn_sharedkv_explicit_rejects_partial_metadata(missing):
    _, args = _explicit_case(0, torch.bfloat16)
    kwargs = dict(
        ori_kv=args[1],
        ori_block_table=args[3],
        request_ids=args[5],
        positions=args[6],
        ori_lengths=args[7],
    )
    kwargs.pop(missing)
    with pytest.raises(ValueError, match="requires request_ids"):
        flaggems_vllm.sparse_attn_sharedkv(args[0], **kwargs)


@pytest.mark.sparse_attn_sharedkv
@npu_only
@pytest.mark.parametrize("scale", [None, 0.0])
def test_sparse_attn_sharedkv_explicit_scale_default_and_zero(scale):
    cpu, args = _explicit_case(0, torch.float32)
    kwargs = dict(
        ori_kv=args[1],
        ori_block_table=args[3],
        request_ids=args[5],
        positions=args[6],
        ori_lengths=args[7],
        ori_win_left=16,
        sinks=args[13],
        value_dim=61,
    )
    if scale is not None:
        kwargs["softmax_scale"] = scale
    actual, lse = flaggems_vllm.sparse_attn_sharedkv(args[0], **kwargs)
    cpu[12] = 64**-0.5 if scale is None else scale
    torch.testing.assert_close(
        actual.cpu().double(), _explicit_oracle(cpu), rtol=2e-5, atol=2e-6
    )
    assert lse.numel() == 0


def _legacy_small_case():
    generator = torch.Generator().manual_seed(932)
    return dict(
        q=(torch.randn(1, 64, 512, generator=generator) * 0.2).bfloat16(),
        ori_kv=(torch.randn(2, 128, 1, 512, generator=generator) * 0.2).bfloat16(),
        ori_block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_kv=torch.tensor([17], dtype=torch.int32),
        sinks=torch.linspace(-1, 1, 64),
        softmax_scale=0.0,
    )


def _legacy_small_oracle(cpu):
    length = int(cpu["seqused_kv"][0])
    rows = []
    for logical in range(max(0, length - 128), length):
        page = int(cpu["ori_block_table"][0, logical // 128])
        rows.append(cpu["ori_kv"][page, logical % 128, 0].double())
    kv = torch.stack(rows)
    logits = cpu["q"][0].double() @ kv.T * 512**-0.5
    denominator = torch.logaddexp(
        torch.logsumexp(logits, dim=-1), cpu["sinks"].double()
    )
    return (torch.exp(logits - denominator[:, None]) @ kv)[None]


@pytest.mark.sparse_attn_sharedkv
@npu_only
def test_sparse_attn_sharedkv_legacy_small_oracle_graph():
    cpu = _legacy_small_case()
    args = {
        key: value.to("npu") if isinstance(value, torch.Tensor) else value
        for key, value in cpu.items()
    }
    for _ in range(2):
        actual, lse = flaggems_vllm.sparse_attn_sharedkv(**args)
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual.cpu().double(), _legacy_small_oracle(cpu), rtol=2e-2, atol=2e-3
    )
    assert lse.numel() == 0
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual, lse = flaggems_vllm.sparse_attn_sharedkv(**args)
    for length in (133, 29):
        cpu["q"] *= 0.8
        cpu["ori_kv"] *= 0.9
        cpu["seqused_kv"][0] = length
        cpu["ori_block_table"][0] = cpu["ori_block_table"][0].flip(0)
        cpu["sinks"] += 0.1
        for key in ("q", "ori_kv", "seqused_kv", "ori_block_table", "sinks"):
            args[key].copy_(cpu[key])
        actual.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual.cpu().double(), _legacy_small_oracle(cpu), rtol=2e-2, atol=2e-3
        )
        assert torch.isfinite(actual).all()
