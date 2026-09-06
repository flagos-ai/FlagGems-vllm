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

"""PPU MXFP4 correctness, packing and public dispatch tests."""

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP4_E2M1
from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_w4a16_mxfp4 import (
    _decode_e2m1,
    _pack_mxfp4,
    _ppu_dequant_mxfp4,
)

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU specialization"
)


def make_inputs(m=4, e=4, k=128, n=256, topk=2, dtype=torch.bfloat16):
    torch.manual_seed(42)
    a = torch.randn((m, k), dtype=dtype, device="cuda") * 0.1
    weights, scales, refs = [], [], []
    for out_dim, in_dim in ((2 * n, k), (k, n)):
        q = torch.randint(0, 16, (e, out_dim, in_dim), device="cuda", dtype=torch.uint8)
        s = torch.randint(
            120, 125, (e, out_dim, in_dim // 32), device="cuda", dtype=torch.uint8
        )
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        ref = (
            lut[q.long()] * torch.exp2(s.float() - 127).repeat_interleave(32, -1)
        ).to(dtype)
        weights.append(q[..., ::2] | (q[..., 1::2] << 4))
        scales.append(s)
        refs.append(ref)
    ids = torch.argsort(torch.rand((m, e), device="cuda"), dim=-1)[
        :, :topk
    ].contiguous()
    tw = torch.softmax(torch.randn((m, topk), device="cuda"), dim=-1)
    args = dict(
        hidden_states=a,
        w1=weights[0],
        w2=weights[1],
        bias1=None,
        bias2=None,
        w1_scale=scales[0],
        w2_scale=scales[1],
        topk_weights=tw,
        topk_ids=ids,
        quant_type_id=QUANT_TYPE_FP4_E2M1,
        group_size=32,
    )
    return args, refs


def reference(args, refs):
    a, ids, tw = args["hidden_states"], args["topk_ids"], args["topk_weights"]
    result = torch.zeros_like(a, dtype=torch.float32)
    # Independent dequantized FP32 expert-wise reference.
    for e in range(refs[0].shape[0]):
        rows, routes = torch.where(ids == e)
        gu = a[rows].float() @ refs[0][e].float().T
        if args.get("apply_router_weight_on_input", False):
            gu = gu * tw[rows, routes, None]
        gate, up = gu.chunk(2, dim=-1)
        inter = (torch.nn.functional.silu(gate) * up).to(a.dtype)
        down = inter.float() @ refs[1][e].float().T
        if not args.get("apply_router_weight_on_input", False):
            down = down * tw[rows, routes, None]
        result.index_add_(0, rows, down)
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 128, 256, 2),
        (2, 4, 128, 256, 1),
        (4, 4, 256, 128, 2),
        (33, 8, 128, 256, 4),
        (128, 4, 512, 256, 2),
        (1, 4, 32, 96, 2),
        (33, 4, 160, 96, 2),
        (1, 16, 4096, 512, 6),
        (256, 16, 1024, 512, 4),
        (4096, 4, 128, 64, 2),
    ],
)
def test_mxfp4(shape, dtype, router):
    args, refs = make_inputs(*shape, dtype=dtype)
    args["apply_router_weight_on_input"] = router
    actual = flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)
    expected = reference(args, refs)
    err = (actual.float() - expected).abs().mean() / expected.abs().mean().clamp_min(
        1e-8
    )
    assert err < 0.04, err.item()
    assert actual.shape == args["hidden_states"].shape
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("k", [32, 96, 128, 160, 256])
def test_pack_and_invalidation(k):
    w = torch.randint(0, 256, (2, 96, k // 2), device="cuda", dtype=torch.uint8)
    actual = _pack_mxfp4(w)
    assert _pack_mxfp4(w) is actual
    q = torch.stack((w & 15, w >> 4), -1).flatten(-2).to(torch.int64)
    q = torch.nn.functional.pad(q, (0, triton.cdiv(k, 128) * 128 - k))
    tiled = q.reshape(2, 96, -1, 8, 16).transpose(-1, -2)
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device="cuda")
    expected = (tiled << shifts).sum(-1).flatten(-2).transpose(1, 2).to(torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    w.zero_()
    updated = _pack_mxfp4(w)
    assert updated is not actual
    assert torch.count_nonzero(updated).item() == 0


@triton.jit
def _decode_test(Q, S, Out, B: tl.constexpr, T: tl.constexpr = tl.float32):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q = tl.load(Q + i).to(tl.int32)
    s = tl.load(S + i).to(tl.int32)
    bits = tl.where(s == 0, 0x00400000, s << 23)
    bits = tl.where(s == 255, 0x7FC00000, bits)
    x = _decode_e2m1(q, bits.to(tl.float32, bitcast=True), T)
    tl.store(Out + i, x)


def test_e2m1_all_codes_and_e8m0_extremes():
    q = torch.arange(16, dtype=torch.int32, device="cuda").repeat(8)
    s = torch.tensor(
        [0, 1, 120, 126, 127, 128, 254, 255], device="cuda", dtype=torch.int32
    ).repeat_interleave(16)
    out = torch.empty(128, device="cuda")
    _decode_test[(1,)](q, s, out, 128)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    scale = torch.exp2(s.double() - 127).float()
    scale[s == 255] = float("nan")
    expected = lut[q.long()] * scale
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("m", [0, 1, 33])
@pytest.mark.parametrize("mode", ["output", "inplace", "e8m0"])
def test_output_and_empty(m, mode):
    args, refs = make_inputs(m=m)
    expected = reference(args, refs)
    if mode == "output":
        args["output"] = torch.empty_like(args["hidden_states"])
    elif mode == "inplace":
        args["inplace"] = True
    else:
        for key in ("w1_scale", "w2_scale"):
            args[key] = args[key].view(torch.float8_e8m0fnu)
    actual = flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)
    if mode in ("output", "inplace"):
        assert actual is args["output" if mode == "output" else "hidden_states"]
    if m:
        err = (actual.float() - expected).abs().mean() / expected.abs().mean()
        assert err < 0.04


@pytest.mark.parametrize(
    "change",
    [
        {"group_size": 128},
        {"activation": "relu"},
        {"is_k_full": False},
        {"activation_func": lambda x: x},
        {"global_num_experts": 9},
    ],
)
def test_unsupported(change):
    args, _ = make_inputs()
    args.update(change)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)


def test_bad_metadata():
    args, _ = make_inputs()
    args["topk_ids"] = args["topk_ids"].to(torch.float32)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)


@pytest.mark.parametrize("m", [1, 33])
def test_cuda_graph(m):
    args, refs = make_inputs(m=m)
    expected = reference(args, refs)
    # Include alignment and all dispatch paths in the captured operator.
    for _ in range(2):
        flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)
    graph.replay()
    torch.cuda.synchronize()
    err = (actual.float() - expected).abs().mean() / expected.abs().mean()
    assert err < 0.04


@pytest.mark.parametrize("m", [1, 33])
def test_strided_weights_scales_and_skew(m):
    args, refs = make_inputs(m=m)
    for key in ("w1", "w2", "w1_scale", "w2_scale"):
        # Same logical values with a genuinely strided physical layout.
        args[key] = args[key].transpose(1, 2).contiguous().transpose(1, 2)
    args["topk_ids"] = torch.zeros_like(args["topk_ids"], dtype=torch.int32)
    args["topk_weights"] = args["topk_weights"].to(torch.bfloat16)
    actual = flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)
    expected = reference(args, refs)
    err = (actual.float() - expected).abs().mean() / expected.abs().mean()
    assert err < 0.04


@pytest.mark.parametrize(
    "case",
    [
        "scale_shape",
        "routing_shape",
        "output_shape",
        "output_alias",
        "inplace_output",
        "grad",
        "strided_activation",
    ],
)
def test_reject_bad_contract(case):
    args, _ = make_inputs()
    if case == "scale_shape":
        args["w1_scale"] = args["w1_scale"][..., :1]
    elif case == "routing_shape":
        args["topk_weights"] = args["topk_weights"][:1]
    elif case == "output_shape":
        args["output"] = torch.empty(1, device="cuda")
    elif case == "output_alias":
        args["output"] = args["w1"].view(torch.bfloat16).flatten()[:512].view(4, 128)
    elif case == "inplace_output":
        args["inplace"] = True
        args["output"] = torch.empty_like(args["hidden_states"])
    elif case == "grad":
        args["hidden_states"].requires_grad_(True)
    elif case == "strided_activation":
        args["hidden_states"] = args["hidden_states"].T.contiguous().T
    with pytest.raises((ValueError, NotImplementedError)):
        flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)


@pytest.mark.parametrize(
    "dtype, tltype", [(torch.float16, tl.float16), (torch.bfloat16, tl.bfloat16)]
)
@pytest.mark.parametrize("use_cache", [False, True])
def test_decode_all_scale_code_combinations(dtype, tltype, use_cache):
    # CPU FP64 oracle: compare all finite output bits, including signed zero.
    q = torch.arange(16).repeat(256)
    s = torch.arange(256).repeat_interleave(16)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    scale = torch.exp2(s.double() - 127)
    scale[s == 255] = float("nan")
    expected = (lut[q] * scale).to(dtype)
    out = torch.empty(4096, device="cuda", dtype=dtype)
    if use_cache:
        from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_w4a16_mxfp4 import (
            _pack_e8m0,
        )

        codes = torch.arange(256, device="cuda", dtype=torch.uint8).repeat(1, 16, 1)
        packed = _pack_e8m0(codes).view(-1)
        _decode_cached_test[(16,)](q.int().cuda(), packed, out, tltype)
    else:
        _decode_test[(16,)](q.int().cuda(), s.int().cuda(), out, 256, tltype)
    got = out.cpu()
    valid = ~torch.isnan(expected)
    assert torch.equal(got.view(torch.int16)[valid], expected.view(torch.int16)[valid])
    assert torch.equal(torch.isnan(got), torch.isnan(expected))


@pytest.mark.parametrize("as_float8", [False, True])
def test_scale_pack_extremes_and_invalidation(as_float8):
    from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_w4a16_mxfp4 import (
        _pack_e8m0,
    )

    raw = torch.arange(256, device="cuda", dtype=torch.uint8).repeat(2, 3, 1)
    raw = raw.transpose(1, 2).contiguous().transpose(1, 2)
    scales = raw.view(torch.float8_e8m0fnu) if as_float8 else raw
    packed = _pack_e8m0(scales)
    assert _pack_e8m0(scales) is packed
    expected = torch.exp2(raw.cpu().double() - 127).float().transpose(1, 2)
    expected[raw.cpu().transpose(1, 2) == 255] = float("nan")
    torch.testing.assert_close(
        packed.cpu().float(), expected, rtol=0, atol=0, equal_nan=True
    )
    raw.fill_(127)
    updated = _pack_e8m0(scales)
    assert updated is not packed
    assert torch.equal(updated, torch.ones_like(updated))


@pytest.mark.parametrize("m", [1, 33])
def test_cold_cuda_graph_cache_ownership(m):
    from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_w4a16_mxfp4 import (
        _PACK_CACHE,
        _SCALE_CACHE,
    )

    warm, _ = make_inputs(m=m)
    flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**warm)
    args, refs = make_inputs(m=m)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flaggems_vllm.fused_marlin_moe_w4a16_mxfp4(**args)
    for key in ("w1", "w2"):
        assert _PACK_CACHE.get(args[key]) is None
    for key in ("w1_scale", "w2_scale"):
        assert _SCALE_CACHE.get(args[key]) is None
    graph.replay()
    expected = reference(args, refs)
    err = (actual.float() - expected).abs().mean() / expected.abs().mean()
    assert err < 0.04


@triton.jit
def _decode_cached_test(Q, S, Out, T: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    q = tl.load(Q + i).to(tl.int32)
    scale = tl.load(S + i)
    value = _decode_e2m1(q, scale, T)
    tl.store(Out + i, value)


@triton.jit
def _decode_packed_tile_test(W, S, Out, T: tl.constexpr):
    tile = tl.program_id(0)
    ns = tl.arange(0, 32)
    rows = tl.arange(0, 16)
    packed = tl.load(W + (tile * 16 + rows[:, None]) * 32 + ns[None, :])
    decoded = _ppu_dequant_mxfp4(
        packed, S, 0, tile * 128, ns, 256 * 32, 32, 1, 32, 8192, T
    )
    ks = tile * 128 + tl.arange(0, 128)
    tl.store(Out + ks[:, None] * 32 + ns[None, :], decoded)


@pytest.mark.parametrize(
    "dtype,tltype", [(torch.float16, tl.float16), (torch.bfloat16, tl.bfloat16)]
)
def test_full_packed_tile_decode(dtype, tltype):
    from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_w4a16_mxfp4 import (
        _pack_e8m0,
    )

    # Vary the code across both the 16-row subtile and the eight packed
    # nibbles. Every scale byte is exercised with every E2M1 code.
    ks = torch.arange(8192)
    ns = torch.arange(32)
    q = ((ks // 16 + ks % 16)[None, :] + ns[:, None]) % 16
    weight = (q[:, ::2] | (q[:, 1::2] << 4)).to(torch.uint8).unsqueeze(0).cuda()
    codes = torch.arange(256, dtype=torch.uint8).repeat(1, 32, 1).cuda()
    packed = _pack_mxfp4(weight)
    scales = _pack_e8m0(codes)
    actual = torch.empty((8192, 32), device="cuda", dtype=dtype)
    _decode_packed_tile_test[(64,)](packed, scales, actual, tltype)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    scale = torch.exp2(torch.arange(256, dtype=torch.float64) - 127)
    scale[255] = float("nan")
    expected = (lut[q] * scale.repeat_interleave(32)[None, :]).T.to(dtype)
    got = actual.cpu()
    valid = ~torch.isnan(expected)
    assert torch.equal(got.view(torch.int16)[valid], expected.view(torch.int16)[valid])
    assert torch.equal(torch.isnan(got), torch.isnan(expected))


def test_public_operator_registration():
    import flaggems_vllm.ops as public_ops

    op = flaggems_vllm.fused_marlin_moe_w4a16_mxfp4
    assert op.__name__ == "fused_marlin_moe_w4a16_mxfp4"
    assert (
        op.__module__
        == "flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_w4a16_mxfp4"
    )
    assert "fused_marlin_moe_w4a16_mxfp4" in public_ops.__all__
    assert ("fused_marlin_moe_w4a16_mxfp4", op) in flaggems_vllm._FULL_CONFIG
