# SPDX-License-Identifier: Apache-2.0
"""Independent numerical and preflight contracts for GLM data movement."""

import importlib

import pytest
import torch

import flaggems_vllm as gems
from flaggems_vllm.ops.data_movement import fill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
DEVICE = gems.device


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32]
)
@pytest.mark.parametrize("layout", ["slice", "transpose", "empty"])
def test_contiguous_copy_layout_and_cast(dtype, layout):
    x = torch.arange(120, device=DEVICE).reshape(3, 5, 8).to(dtype)
    x = {"slice": x[:, ::2, ::2], "transpose": x.transpose(0, 2), "empty": x[:0]}[
        layout
    ]
    actual = gems.contiguous_copy(x, torch.float32)
    assert actual.dtype == torch.float32
    assert actual.shape == x.shape and actual.is_contiguous()
    torch.testing.assert_close(actual, x.float(), rtol=0, atol=0)


def test_contiguous_copy_identity_returns_same_tensor():
    x = torch.randn(3, 7, device=DEVICE)
    assert gems.contiguous_copy(x) is x


@pytest.mark.parametrize("shape", [(0,), (2, 0, 7), (3, 11)])
def test_fill_empty_and_regular(shape):
    out = torch.empty(shape, dtype=torch.float32, device=DEVICE)
    assert fill(out, -7.5) is out
    torch.testing.assert_close(out, torch.full_like(out, -7.5), rtol=0, atol=0)


def test_fill_noncontiguous_rejects_without_mutation():
    out = torch.full((3, 5), 23.0, device=DEVICE).t()
    before = out.clone()
    with pytest.raises(ValueError, match="contiguous"):
        fill(out, 0)
    torch.testing.assert_close(out, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gather_strided_rows_columns_and_indices(dtype, index_dtype):
    x = torch.arange(9 * 14, device=DEVICE).reshape(9, 14).to(dtype)[::2, ::2]
    indices = torch.tensor(
        [[4, 2, 0, 1], [3, 1, 4, 0]], dtype=index_dtype, device=DEVICE
    )[:, ::2]
    actual = gems.gather_rows(x, indices)
    torch.testing.assert_close(actual, x[indices.long()], rtol=0, atol=0)
    assert actual.shape == (2, 2, 7)


@pytest.mark.parametrize(
    "shape,index_shape", [((0, 7), (0,)), ((3, 0), (2, 0)), ((3, 0), (2,))]
)
def test_gather_empty(shape, index_shape):
    x = torch.empty(shape, device=DEVICE)
    indices = torch.zeros(index_shape, dtype=torch.int64, device=DEVICE)
    assert gems.gather_rows(x, indices).shape == (*index_shape, shape[-1])


@pytest.mark.parametrize("invalid", ["rank", "dtype", "device"])
def test_gather_preflight_rejects_before_launch(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.data_movement")
    calls = []

    class RejectLaunch:
        def __getitem__(self, grid):
            calls.append(grid)
            raise AssertionError("invalid gather launched")

    monkeypatch.setattr(module, "_gather_kernel", RejectLaunch())
    x = torch.randn(4, 8, device=DEVICE)
    indices = torch.tensor([1, 3], dtype=torch.int64, device=DEVICE)
    if invalid == "rank":
        x = x.unsqueeze(0)
    elif invalid == "dtype":
        indices = indices.float()
    else:
        indices = indices.cpu()
    with pytest.raises(ValueError):
        gems.gather_rows(x, indices)
    assert not calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_decode_scatter_multidimensional_strided_tokens(dtype, empty):
    n = 0 if empty else 4
    x = torch.arange(n * 2 * 10, device=DEVICE).reshape(n, 2, 10).to(dtype)[:, :, ::2]
    req = torch.tensor([2, 0, 2, 1][:n], dtype=torch.int32, device=DEVICE)
    pos = torch.tensor([3, 0, 1, 2][:n], dtype=torch.int64, device=DEVICE)
    actual = gems.scatter_decode_tokens(x, -3, 3, 4, (req, pos))
    expected = torch.full((3, 4, 2, 5), -3, dtype=dtype, device=DEVICE)
    expected[req.long(), pos] = x
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["shape", "device", "dtype"])
def test_decode_scatter_metadata_preflight(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.data_movement")
    calls = []
    monkeypatch.setattr(module, "fill", lambda *args: calls.append("fill"))
    x = torch.randn(2, 5, device=DEVICE)
    req = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)
    pos = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)
    if invalid == "shape":
        req = req[:1]
    elif invalid == "device":
        req = req.cpu()
    else:
        req = req.float()
    with pytest.raises(ValueError):
        gems.scatter_decode_tokens(x, 0, 2, 1, (req, pos))
    assert not calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rope_dim", [0, 7])
@pytest.mark.parametrize("tokens", [0, 3])
def test_concat_query_strided_inputs_and_zero_rope(dtype, rope_dim, tokens):
    a = torch.randn(tokens, 2, 18, device=DEVICE, dtype=dtype)[..., ::2]
    b = torch.randn(tokens, 2, rope_dim * 2, device=DEVICE, dtype=dtype)[..., ::2]
    out = torch.full(
        (tokens, 2, 9 + rope_dim), float("nan"), device=DEVICE, dtype=dtype
    )
    assert gems.concat_mla_q(a, b, out) is None
    expected = torch.cat((a, b), dim=-1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(gems.concat_query(a, b), expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "out_stride"])
def test_concat_preflight_does_not_modify_output(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.data_movement")
    calls = []

    class RejectLaunch:
        def __getitem__(self, grid):
            calls.append(grid)
            raise AssertionError("invalid concatenation launched")

    monkeypatch.setattr(module, "_concat_query_kernel", RejectLaunch())
    a = torch.ones(3, 2, 9, device=DEVICE)
    b = torch.ones(3, 2, 7, device=DEVICE)
    out = torch.full((3, 2, 16), 123.0, device=DEVICE)
    if invalid == "shape":
        b = b[:2]
    elif invalid == "dtype":
        b = b.half()
    elif invalid == "device":
        b = b.cpu()
    else:
        out = torch.full((3, 2, 32), 123.0, device=DEVICE)[..., ::2]
    before = out.clone()
    with pytest.raises(ValueError):
        gems.concat_mla_q(a, b, out)
    assert not calls
    torch.testing.assert_close(out, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "tokens,heads,dim,padded",
    [(3, 2, 9, 5), (0, 2, 9, 5), (3, 0, 9, 5), (3, 2, 0, 5), (3, 2, 9, 2)],
)
def test_attention_head_padding(tokens, heads, dim, padded):
    q = torch.randn(tokens, heads, dim * 2, device=DEVICE)[..., ::2]
    actual = gems.pad_attention_heads(q, padded)
    expected = torch.zeros(tokens, padded, dim, device=DEVICE)
    expected[:, :heads] = q
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_attention_head_padding_rejects_reduction():
    with pytest.raises(ValueError):
        gems.pad_attention_heads(torch.ones(3, 4, 7, device=DEVICE), 3)
