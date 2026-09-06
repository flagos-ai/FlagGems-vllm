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

"""THead FP8 W8A16 decoding, dispatch and MoE correctness."""
import math

import pytest
import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_FP8_E4M3
from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_fp8 import (
    _PACK_CACHE,
    _SCALE_CACHE,
    _decode_e4m3,
    _pack_fp8,
    _pack_fp8_scale,
    _ppu_dequant_fp8,
)

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "thead", reason="PPU FP8 specialization"
)


def e4m3_lut():
    result = []
    for q in range(256):
        exponent = (q >> 3) & 15
        mantissa = q & 7
        value = (
            math.ldexp(mantissa, -9)
            if exponent == 0
            else math.ldexp(1 + mantissa / 8, exponent - 7)
        )
        if exponent == 15 and mantissa == 7:
            value = float("nan")
        result.append(-value if q & 128 else value)
    return torch.tensor(result, dtype=torch.float64)


def make_inputs(m=4, e=4, k=128, n=256, topk=2, dtype=torch.bfloat16, group_size=128):
    torch.manual_seed(37)
    a = torch.randn((m, k), dtype=dtype, device="cuda") * 0.1
    weights = []
    scales = []
    refs = []
    lut = e4m3_lut().float().cuda()
    for no, ki in ((2 * n, k), (k, n)):
        q = torch.randint(0, 256, (e, no, ki), dtype=torch.uint8, device="cuda")
        q = torch.where((q & 127) == 127, q - 1, q)
        ng = 1 if group_size == -1 else ki // group_size
        scale = (torch.rand((e, no, ng), device="cuda") * 0.001 + 0.0005).to(dtype)
        expanded = (
            scale.float().expand(e, no, ki)
            if group_size == -1
            else scale.float().repeat_interleave(group_size, -1)
        )
        ref = (lut[q.long()] * expanded).to(dtype)
        weights.append(q.view(torch.float8_e4m3fn))
        scales.append(scale)
        refs.append(ref)
    ids = torch.rand((m, e), device="cuda").topk(topk, dim=-1).indices
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
        quant_type_id=QUANT_TYPE_FP8_E4M3,
        group_size=group_size,
    )
    return args, refs


def reference(args, refs):
    a, ids, tw = args["hidden_states"], args["topk_ids"], args["topk_weights"]
    result = torch.zeros_like(a, dtype=torch.float32)
    for e in range(refs[0].shape[0]):
        rows, routes = torch.where(ids == e)
        gu = a[rows].float() @ refs[0][e].float().T
        if args.get("apply_router_weight_on_input", False):
            gu *= tw[rows, routes, None]
        gate, up = gu.chunk(2, dim=-1)
        inter = (torch.nn.functional.silu(gate) * up).to(a.dtype)
        down = inter.float() @ refs[1][e].float().T
        if not args.get("apply_router_weight_on_input", False):
            down *= tw[rows, routes, None]
        result.index_add_(0, rows, down)
    return result


def check(args, refs):
    expected = reference(args, refs)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    if actual.numel():
        error = (
            actual.float() - expected
        ).abs().mean() / expected.abs().mean().clamp_min(1e-12)
        assert error < 0.04, error.item()
        assert torch.isfinite(actual).all()
    assert (
        actual.shape == args["hidden_states"].shape
        and actual.dtype == args["hidden_states"].dtype
    )
    return actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("m", [1, 2, 4, 33, 128])
def test_fp8_moe(m, dtype, group_size, router):
    args, refs = make_inputs(m=m, dtype=dtype, group_size=group_size)
    args["apply_router_weight_on_input"] = router
    check(args, refs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4, 32, 96, 2),
        (33, 4, 160, 96, 2),
        (1, 16, 4096, 256, 6),
        (4096, 4, 128, 64, 2),
    ],
)
def test_fp8_tails_and_large(shape, dtype):
    args, refs = make_inputs(*shape, dtype=dtype, group_size=-1)
    check(args, refs)


@triton.jit
def _decode_test(Q, S, Out, T: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = _decode_e4m3(tl.load(Q + i).to(tl.int32), tl.load(S + i), T)
    tl.store(Out + i, value)


@pytest.mark.parametrize(
    "dtype,tltype", [(torch.float16, tl.float16), (torch.bfloat16, tl.bfloat16)]
)
def test_all_encodings(dtype, tltype):
    q = torch.arange(256).repeat(5)
    scales = torch.tensor(
        [0.0, 2**-16, 1.0, 2**16, float("nan")], dtype=torch.float64
    ).repeat_interleave(256)
    expected = (e4m3_lut()[q] * scales).to(dtype)
    actual = torch.empty(1280, device="cuda", dtype=dtype)
    _decode_test[(5,)](q.int().cuda(), scales.float().cuda(), actual, tltype)
    got = actual.cpu()
    valid = ~torch.isnan(expected)
    assert torch.equal(got.view(torch.int16)[valid], expected.view(torch.int16)[valid])
    assert torch.equal(torch.isnan(got), torch.isnan(expected))


@pytest.mark.parametrize("k", [32, 96, 128, 160, 256])
def test_pack_and_versions(k):
    raw = torch.randint(0, 256, (2, 96, k), dtype=torch.uint8, device="cuda")
    w = raw.view(torch.float8_e4m3fn)
    packed = _pack_fp8(w)
    assert _pack_fp8(w) is packed
    padded = torch.nn.functional.pad(raw.long(), (0, triton.cdiv(k, 128) * 128 - k))
    tiles = padded.reshape(2, 96, -1, 4, 32).transpose(-1, -2)
    shifts = torch.tensor([0, 8, 16, 24], device="cuda")
    expected = (tiles << shifts).sum(-1).flatten(-2).transpose(1, 2).to(torch.int32)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    raw.zero_()
    assert _pack_fp8(w) is not packed
    assert torch.count_nonzero(_pack_fp8(w)) == 0


@pytest.mark.parametrize("m", [0, 1, 33])
@pytest.mark.parametrize("mode", ["inplace", "output", "strided"])
def test_output_layout(m, mode):
    args, refs = make_inputs(m=m)
    if mode == "inplace":
        args["inplace"] = True
    elif mode == "output":
        args["output"] = torch.empty_like(args["hidden_states"])
    else:
        for key in ["w1", "w2", "w1_scale", "w2_scale"]:
            args[key] = args[key].transpose(1, 2).contiguous().transpose(1, 2)
    actual = check(args, refs)
    if mode != "strided":
        assert actual is args["hidden_states" if mode == "inplace" else "output"]


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("cold", [False, True])
def test_graph(m, cold):
    warm, _ = make_inputs(m=m)
    flaggems_vllm.fused_marlin_moe(**warm)
    args, refs = make_inputs(m=m)
    expected = reference(args, refs)
    if not cold:
        flaggems_vllm.fused_marlin_moe(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flaggems_vllm.fused_marlin_moe(**args)
    if cold:
        for key in ["w1", "w2"]:
            assert _PACK_CACHE.get(args[key]) is None
        for key in ["w1_scale", "w2_scale"]:
            assert _SCALE_CACHE.get(args[key]) is None
    graph.replay()
    error = (actual.float() - expected).abs().mean() / expected.abs().mean()
    assert error < 0.04


@pytest.mark.parametrize(
    "case",
    ["group", "dtype", "scale", "callback", "map", "workspace", "grad", "output"],
)
def test_rejections(case):
    args, _ = make_inputs()
    if case == "group":
        args["group_size"] = 16
    elif case == "dtype":
        args["w1"] = args["w1"].view(torch.float8_e5m2)
    elif case == "scale":
        args["w1_scale"] = args["w1_scale"].to(torch.uint8)
    elif case == "callback":
        args["activation_func"] = lambda x: x
    elif case == "map":
        args["expert_map"] = torch.arange(4, device="cuda")
    elif case == "workspace":
        args["workspace"] = torch.empty(1, device="cuda")
    elif case == "grad":
        args["hidden_states"].requires_grad_(True)
    elif case == "output":
        args["output"] = torch.empty(1, device="cuda")
    with pytest.raises((NotImplementedError, ValueError)):
        flaggems_vllm.fused_marlin_moe(**args)


def test_native_type_id():
    from vllm.scalar_type import scalar_types

    assert QUANT_TYPE_FP8_E4M3 == scalar_types.float8_e4m3fn.id


def test_scale_versions():
    s = torch.rand((2, 128, 4), device="cuda", dtype=torch.bfloat16)
    packed = _pack_fp8_scale(s)
    torch.testing.assert_close(packed, s.transpose(1, 2).float(), rtol=0, atol=0)
    s.fill_(2)
    assert _pack_fp8_scale(s) is not packed
    assert torch.equal(_pack_fp8_scale(s), torch.full_like(packed, 2))


@triton.jit
def _decode_full_tile(
    W, S, Out, K: tl.constexpr, GROUP_SIZE: tl.constexpr, T: tl.constexpr
):
    tile = tl.program_id(0)
    ns = tl.arange(0, 32)
    rows = tile * 32 + tl.arange(0, 32)
    packed = tl.load(W + rows[:, None] * 32 + ns[None, :])
    groups: tl.constexpr = 1 if GROUP_SIZE == -1 else K // GROUP_SIZE
    value = _ppu_dequant_fp8(
        packed, S, 0, tile * 128, ns, groups * 32, 32, 1, 32, K, T, GROUP_SIZE
    )
    ks = tile * 128 + tl.arange(0, 128)
    tl.store(Out + ks[:, None] * 32 + ns[None, :], value)


@pytest.mark.parametrize(
    "dtype,tltype", [(torch.float16, tl.float16), (torch.bfloat16, tl.bfloat16)]
)
@pytest.mark.parametrize("k,group_size", [(512, 32), (512, 64), (512, 128), (96, -1)])
def test_packed_tile_special_values(k, group_size, dtype, tltype):
    ns = torch.arange(32)
    ks = torch.arange(k)
    q = ((ns * 11)[:, None] + ks[None, :]) % 256
    w = q.to(torch.uint8).unsqueeze(0).cuda().view(torch.float8_e4m3fn)
    ng = 1 if group_size == -1 else k // group_size
    powers = torch.tensor(
        [0.0, 2**-16, 1.0, 2**16, float("inf"), float("nan")], dtype=torch.float64
    )
    scale = (
        powers[(ns[:, None] + torch.arange(ng)[None, :]) % 6]
        .unsqueeze(0)
        .float()
        .cuda()
    )
    wp = _pack_fp8(w)
    sp = _pack_fp8_scale(scale)
    kp = triton.cdiv(k, 128) * 128
    actual = torch.empty((kp, 32), device="cuda", dtype=dtype)
    _decode_full_tile[(triton.cdiv(k, 128),)](wp, sp, actual, k, group_size, tltype)
    expanded = (
        scale.cpu().double()[0].expand(32, k)
        if group_size == -1
        else scale.cpu().double()[0].repeat_interleave(group_size, -1)
    )
    expected = torch.zeros((kp, 32), dtype=dtype)
    expected[:k] = (e4m3_lut()[q] * expanded).T.to(dtype)
    got = actual.cpu()
    valid = ~torch.isnan(expected)
    assert torch.equal(got.view(torch.int16)[valid], expected.view(torch.int16)[valid])
    assert torch.equal(torch.isnan(got), torch.isnan(expected))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_native_marlin_groups(dtype, group_size):
    from vllm.scalar_type import scalar_types

    import benchmark.test_fused_marlin_moe_w8a16_fp8 as bench

    args, _ = make_inputs(m=4, dtype=dtype, group_size=group_size)
    saved = bench.GROUP_SIZE
    try:
        bench.GROUP_SIZE = group_size
        w1, s1 = bench._to_vllm_marlin(args["w1"], args["w1_scale"], 128, 512)
        w2, s2 = bench._to_vllm_marlin(args["w2"], args["w2_scale"], 256, 128)
    finally:
        bench.GROUP_SIZE = saved
    try:
        expected = bench.vllm_fused_marlin_moe(
            args["hidden_states"],
            w1,
            w2,
            None,
            None,
            s1,
            s2,
            None,
            args["topk_weights"],
            args["topk_ids"],
            scalar_types.float8_e4m3fn.id,
        )
    except RuntimeError as error:
        if group_size in (32, 64) and "Invalid thread config" in str(error):
            pytest.skip(
                "Installed native FP8 Marlin has no configuration for this "
                f"group_size={group_size}, dtype={dtype}, small shape. "
                "Independent FP32-reference tests cover this supported PPU path."
            )
        raise
    actual = flaggems_vllm.fused_marlin_moe(**args)
    error = (
        actual.float() - expected.float()
    ).abs().mean() / expected.float().abs().mean()
    assert error < 0.04, error.item()


def test_safety_flags_and_updates():
    from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_fp8 import (
        _pack_fp8_cache,
        _pack_fp8_scale_cache,
    )

    args, _ = make_inputs()
    w = args["w1"]
    _, safe = _pack_fp8_cache(w)
    assert safe.cpu().tolist() == [1, 1, 1, 1, 1]
    w.view(torch.uint8)[1, 0, 0] = 127
    _, safe = _pack_fp8_cache(w)
    assert safe.cpu().tolist() == [1, 0, 1, 1, 0]
    w.view(torch.uint8)[1, 0, 0] = 0
    assert _pack_fp8_cache(w)[1].cpu().tolist() == [1, 1, 1, 1, 1]
    scale = args["w1_scale"].float()
    bound = torch.tensor(0x7B7FFFFF, dtype=torch.int32).view(torch.float32).item()
    scale.fill_(bound)
    assert _pack_fp8_scale_cache(scale)[1].cpu().tolist() == [1, 1, 1, 1, 1]
    scale[2].fill_(float.fromhex("0x1p120"))
    assert _pack_fp8_scale_cache(scale)[1].cpu().tolist() == [1, 1, 0, 1, 0]
    scale[2].fill_(float("inf"))
    scale[3].fill_(float("nan"))
    assert _pack_fp8_scale_cache(scale)[1].cpu().tolist() == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("m", [1, 2, 4, 33])
@pytest.mark.parametrize("projection", [1, 2])
def test_mixed_fast_and_general_experts(m, projection):
    args, refs = make_inputs(m=m)
    weight = args["w1" if projection == 1 else "w2"]
    key = "w1_scale" if projection == 1 else "w2_scale"
    scale = args[key].float()
    # Expert 0 needs a full path: folding this finite scale overflows.
    # Zero weights must still decode to zero rather than 0*infinity -> NaN.
    weight.view(torch.uint8)[0].zero_()
    scale[0].fill_(1e38)
    # Expert 1 remains fast, expert 2 has a genuine FP8 NaN, and expert 3
    # has NaN scales (which are safe to fold and must still propagate).
    weight.view(torch.uint8)[2, 0, 0] = 127
    scale[3].fill_(float("nan"))
    args[key] = scale
    args["topk_ids"] = (torch.arange(m * 2, device="cuda").reshape(m, 2) % 4).long()
    for j, (wk, sk) in enumerate([("w1", "w1_scale"), ("w2", "w2_scale")]):
        q = args[wk].view(torch.uint8).long()
        expanded = args[sk].float().repeat_interleave(128, -1)
        refs[j] = (e4m3_lut().float().cuda()[q] * expanded).to(
            args["hidden_states"].dtype
        )
    expected = reference(args, refs)
    actual = flaggems_vllm.fused_marlin_moe(**args).float()
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isinf(actual), torch.isinf(expected))
    finite = torch.isfinite(expected)
    if finite.any():
        error = (actual[finite] - expected[finite]).abs().mean() / expected[
            finite
        ].abs().mean().clamp_min(1e-12)
        assert error < 0.04


@triton.jit
def _decode_guarded_tile(W, S, WSafe, SSafe, Out, E: tl.constexpr, T: tl.constexpr):
    expert = tl.program_id(0)
    ns = tl.arange(0, 32)
    ks = tl.arange(0, 32)
    packed = tl.load(W + expert * 32 * 32 + ks[:, None] * 32 + ns[None, :])
    safe = tl.load(WSafe + expert) & tl.load(SSafe + expert)
    if safe:
        value = _ppu_dequant_fp8(
            packed, S, expert, 0, ns, 32, 32, 1, 32, 128, T, 128, True
        )
    else:
        value = _ppu_dequant_fp8(
            packed, S, expert, 0, ns, 32, 32, 1, 32, 128, T, 128, False
        )
    rows = tl.arange(0, 128)
    tl.store(Out + expert * 128 * 32 + rows[:, None] * 32 + ns[None, :], value)


@pytest.mark.parametrize(
    "dtype,tltype", [(torch.float16, tl.float16), (torch.bfloat16, tl.bfloat16)]
)
def test_guarded_decode_scale_extremes(dtype, tltype):
    from flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe_fp8 import (
        _pack_fp8_cache,
        _pack_fp8_scale_cache,
    )

    scales = torch.tensor(
        [
            0.0,
            -0.0,
            2**-149,
            2**-140,
            2**-130,
            2**-127,
            2**-126,
            1.0,
            -1.0,
            2**119,
            2**120,
            1e38,
            -1e38,
            float("inf"),
            -float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    e = scales.numel()
    q = (torch.arange(128)[None, :] + torch.arange(32)[:, None] * 9) % 256
    q = torch.where((q & 127) == 127, q - 1, q)
    w = q.to(torch.uint8).unsqueeze(0).repeat(e, 1, 1).cuda().view(torch.float8_e4m3fn)
    scale = scales[:, None, None].expand(e, 32, 1).contiguous().cuda()
    packed, ws = _pack_fp8_cache(w)
    packed_scale, ss = _pack_fp8_scale_cache(scale)
    actual = torch.empty((e, 128, 32), device="cuda", dtype=dtype)
    _decode_guarded_tile[(e,)](packed, packed_scale, ws, ss, actual, e, tltype)
    expected = (
        (e4m3_lut()[q][None, :, :] * scales.double()[:, None, None])
        .float()
        .to(dtype)
        .transpose(1, 2)
    )
    got = actual.cpu()
    valid = ~torch.isnan(expected)
    assert torch.equal(got.view(torch.int16)[valid], expected.view(torch.int16)[valid])
    assert torch.equal(torch.isnan(got), torch.isnan(expected))


@pytest.mark.parametrize("m", [4, 33])
def test_cold_graph_switches_safety_paths(m):
    warm, _ = make_inputs(m=m)
    flaggems_vllm.fused_marlin_moe(**warm)
    args, refs = make_inputs(m=m)
    args["topk_ids"] = (
        (torch.arange(m, device="cuda")[:, None] % 4).expand(m, 2).contiguous()
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flaggems_vllm.fused_marlin_moe(**args)
    graph.replay()
    torch.cuda.synchronize()
    # Cold capture includes packing and flag generation, so changing the
    # captured raw inputs can switch the chosen kernel on the next replay.
    args["w1"].view(torch.uint8)[0].zero_()
    args["w1_scale"][0].fill_(1e38)
    args["w1"].view(torch.uint8)[1, 0, 0] = 127
    graph.replay()
    for j, (wk, sk) in enumerate([("w1", "w1_scale"), ("w2", "w2_scale")]):
        q = args[wk].view(torch.uint8).long()
        refs[j] = (
            e4m3_lut().float().cuda()[q] * args[sk].float().repeat_interleave(128, -1)
        ).to(args["hidden_states"].dtype)
    expected = reference(args, refs)
    got = actual.float()
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    assert torch.equal(torch.isinf(got), torch.isinf(expected))
    finite = torch.isfinite(expected)
    assert (
        (got[finite] - expected[finite]).abs().mean()
        / expected[finite].abs().mean().clamp_min(1e-12)
    ) < 0.04
