# SPDX-License-Identifier: Apache-2.0
"""Indexer scales retain their producer dtype, rank and token/head strides."""

import importlib

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("tokens", [128, 512, 2051])
def test_prefill_mqa_squeezed_key_scale(tokens):
    keys = torch.ones(tokens, 128, device=gems.device).to(torch.float8_e4m3fn)
    scales = torch.arange(1, tokens + 1, dtype=torch.float32, device=gems.device)
    query = torch.ones(2, 3, 128, device=gems.device).to(torch.float8_e4m3fn)
    weights = torch.ones(2, 3, device=gems.device)
    starts = torch.tensor([0, 1], dtype=torch.int32, device=gems.device)
    ends = torch.tensor([tokens, tokens - 1], dtype=torch.int32, device=gems.device)
    actual = gems.fp8_fp4_mqa_logits(
        (query, None), (keys, scales), weights, starts, ends
    )
    expected = (384 * scales).expand(2, -1).clone()
    expected[1, 0] = expected[1, -1] = -torch.inf
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("invalid_shape", [(3,), (128, 2)])
def test_invalid_key_scale_groups_rejected_before_numerics(invalid_shape, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fp8_fp4_mqa_logits")
    calls = []

    class RejectLaunch:
        def __getitem__(self, grid):
            calls.append(grid)
            raise AssertionError("invalid scale shape launched")

    monkeypatch.setattr(module, "_fp8_fp4_mqa_logits_kernel", RejectLaunch())
    query = torch.ones(2, 3, 128, device=gems.device).to(torch.float8_e4m3fn)
    keys = torch.ones(128, 128, device=gems.device).to(torch.float8_e4m3fn)
    scales = torch.ones(invalid_shape, device=gems.device)
    weights = torch.ones(2, 3, device=gems.device)
    starts = torch.zeros(2, dtype=torch.int32, device=gems.device)
    ends = torch.full((2,), 128, dtype=torch.int32, device=gems.device)
    with pytest.raises(ValueError, match="[Ss]cale"):
        gems.fp8_fp4_mqa_logits((query, None), (keys, scales), weights, starts, ends)
    assert not calls


def _mxfp4_inputs(packed_dtype, scale_rank, strided):
    m, h, n, dim = 3, 3, 37, 128
    device = gems.device
    token = torch.arange(m, device=device)[:, None, None]
    head = torch.arange(h, device=device)[None, :, None]
    feature = torch.arange(dim, device=device)[None, None, :]
    nibbles = ((feature + 3 * token + 5 * head) % 16).to(torch.uint8)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    if strided:
        backing = torch.full(
            (m * 2, h * 2, dim // 2), 255, device=device, dtype=torch.uint8
        )
        query = backing[::2, ::2]
        query.copy_(packed)
    else:
        query = packed
    query = query.view(packed_dtype)

    # Four distinct block exponents; vary tokens/heads as well. The high byte
    # crosses 127 so the packed int32 and signed Q exercise sign extension.
    block = torch.arange(dim // 32, device=device)
    exponent = 125 + block + ((token[..., 0] + head[..., 0]) % 3)[..., None]
    packed_scales = (exponent.to(torch.int64) << (8 * block)).sum(-1).to(torch.int32)
    if strided:
        backing_scales = torch.full(
            (m * 2, h * 2, 2), 0x01010101, device=device, dtype=torch.int32
        )
        scales = backing_scales[::2, ::2, :1]
        scales[..., 0].copy_(packed_scales)
    else:
        scales = packed_scales[..., None]
    if scale_rank == 2:
        scales = scales[..., 0]

    keys = ((torch.arange(n * dim, device=device).reshape(n, dim) % 7) - 3).to(
        torch.float8_e4m3fn
    )
    key_scales = (1 + torch.arange(n, device=device) % 4).float() / 4
    weights = (torch.arange(m * h, device=device).reshape(m, h).float() - 3) / 4
    starts = torch.tensor([0, 1, 4], device=device, dtype=torch.int32)
    ends = torch.tensor([n, n - 1, n - 3], device=device, dtype=torch.int32)
    return query, scales, keys, key_scales, weights, starts, ends


def _reference_mxfp4_logits(query, scales, keys, key_scales, weights, starts, ends):
    """Independent E2M1 LUT and UE8M0 decode, with no producer/dequant helper."""
    m, h, packed_dim = query.shape
    dim = packed_dim * 2
    raw = query.to(torch.uint8).long()
    nibbles = torch.stack((raw & 15, (raw >> 4) & 15), dim=-1).reshape(m, h, dim)
    magnitudes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=query.device, dtype=torch.float32
    )
    values = magnitudes[nibbles & 7] * torch.where((nibbles & 8) != 0, -1, 1)
    blocks = torch.arange(dim // 32, device=query.device)
    exponent = ((scales.reshape(m, h).long()[..., None] >> (8 * blocks)) & 255) - 127
    block_scales = torch.exp2(exponent.float()).repeat_interleave(32, dim=-1)
    decoded = values * block_scales
    scores = torch.einsum("mhd,nd->mhn", decoded, keys.float())
    logits = ((scores * key_scales).clamp_min(0) * weights[..., None]).sum(1)
    column = torch.arange(keys.shape[0], device=query.device)[None, :]
    return logits.masked_fill(
        (column < starts[:, None]) | (column >= ends[:, None]), -torch.inf
    )


@pytest.mark.parametrize("packed_dtype", [torch.uint8, torch.int8])
@pytest.mark.parametrize("scale_rank", [2, 3])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_mxfp4_mqa_input_contract_numerics(packed_dtype, scale_rank, strided):
    query, scales, keys, key_scales, weights, starts, ends = _mxfp4_inputs(
        packed_dtype, scale_rank, strided
    )
    assert (query.view(torch.int8) < 0).any()
    assert (scales < 0).any()
    if strided:
        assert not query.is_contiguous()
        assert not scales.is_contiguous()
    expected = _reference_mxfp4_logits(
        query, scales, keys, key_scales, weights, starts, ends
    )
    actual = gems.fp8_fp4_mqa_logits(
        (query, scales), (keys, key_scales), weights, starts, ends
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("packed_dtype", [torch.uint8, torch.int8])
@pytest.mark.parametrize("scale_rank", [2, 3])
def test_mxfp4_mqa_forwards_original_objects_and_strides(
    packed_dtype, scale_rank, monkeypatch
):
    module = importlib.import_module("flaggems_vllm.ops.fp8_fp4_mqa_logits")
    calls = []

    class CaptureLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(args)

            return launch

    monkeypatch.setattr(module, "_fp8_fp4_mqa_logits_mxfp4_kernel", CaptureLaunch())
    query, scales, keys, key_scales, weights, starts, ends = _mxfp4_inputs(
        packed_dtype, scale_rank, strided=True
    )
    gems.fp8_fp4_mqa_logits(
        (query, scales), (keys, key_scales), weights, starts, ends, clean_logits=False
    )
    assert len(calls) == 1
    args = calls[0]
    assert args[0] is query
    assert args[1] is scales
    assert args[10:13] == query.stride()
    assert args[13:15] == scales.stride()[:2]


@pytest.mark.parametrize(
    "invalid",
    [
        "query_dtype",
        "query_rank",
        "query_dim",
        "scale_dtype",
        "scale_2d_shape",
        "scale_3d_shape",
        "scale_rank",
        "query_feature_stride",
        "scale_device",
    ],
)
def test_mxfp4_invalid_input_rejected_before_launch(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.fp8_fp4_mqa_logits")
    calls = []

    class CaptureLaunch:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(args)

            return launch

    monkeypatch.setattr(module, "_fp8_fp4_mqa_logits_mxfp4_kernel", CaptureLaunch())
    query, scales, keys, key_scales, weights, starts, ends = _mxfp4_inputs(
        torch.uint8, 2, strided=True
    )
    gems.fp8_fp4_mqa_logits(
        (query, scales), (keys, key_scales), weights, starts, ends, clean_logits=False
    )
    assert len(calls) == 1
    calls.clear()
    if invalid == "query_dtype":
        query = query.to(torch.int16)
    elif invalid == "query_rank":
        query = query[0]
    elif invalid == "query_dim":
        query = query[..., :-1]
    elif invalid == "scale_dtype":
        scales = scales.to(torch.int64)
    elif invalid == "scale_2d_shape":
        scales = scales[:, :-1]
    elif invalid == "scale_3d_shape":
        scales = scales[..., None].expand(-1, -1, 2)
    elif invalid == "scale_rank":
        scales = scales[None, :, :, None]
    elif invalid == "query_feature_stride":
        query = torch.empty(
            *query.shape[:-1],
            query.shape[-1] * 2,
            device=query.device,
            dtype=query.dtype,
        )[..., ::2]
    elif invalid == "scale_device":
        scales = scales.cpu()
    with pytest.raises(ValueError):
        gems.fp8_fp4_mqa_logits(
            (query, scales),
            (keys, key_scales),
            weights,
            starts,
            ends,
            clean_logits=False,
        )
    assert not calls


def test_mxfp4_real_producer_consumer_sm100():
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("real MXFP4 producer PTX conversion requires SM100 or newer")
    torch.manual_seed(0)
    m, h, dim = 3, 3, 128
    device = gems.device
    query = torch.randn(m, h, dim, device=device, dtype=torch.bfloat16)
    query *= torch.tensor([0.25, 0.5, 1, 2], device=device).repeat_interleave(32)
    positions = torch.tensor([0, 2, 5], device=device, dtype=torch.int64)
    angle = (
        torch.arange(8, device=device)[:, None]
        * torch.arange(1, 33, device=device)[None, :]
        / 32
    )
    cache = torch.cat((angle.cos(), angle.sin()), dim=1).to(torch.bfloat16)
    weights = torch.randn(m, h, device=device, dtype=torch.bfloat16)
    softmax_scale, head_scale = dim**-0.5, h**-0.5
    q_quant, weights_out = gems.fused_indexer_q_rope_quant(
        positions, query, cache, weights, softmax_scale, head_scale, use_fp4=True
    )
    packed, scales = q_quant
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.int32
    assert scales.shape == (m, h)
    expected_weights = weights.float() * softmax_scale * head_scale
    torch.testing.assert_close(weights_out, expected_weights, rtol=0, atol=0)
    _, _, keys, key_scales, _, starts, ends = _mxfp4_inputs(torch.uint8, 2, False)
    actual = gems.fp8_fp4_mqa_logits(
        q_quant, (keys, key_scales), weights_out, starts, ends
    )
    expected = _reference_mxfp4_logits(
        packed, scales, keys, key_scales, expected_weights, starts, ends
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
