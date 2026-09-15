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

import inspect
import itertools

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8_int8,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "thead", reason="PPU-only API"),
]


def _inputs(lengths, heads, dim, broadcast_scales=False):
    torch.manual_seed(sum(lengths) + heads + dim)
    quant = torch.randint(
        -127, 128, (sum(lengths), heads, dim), device="cuda", dtype=torch.int8
    )
    scales = (
        torch.rand((len(lengths), heads, (max(lengths) + 127) // 128), device="cuda")
        * 0.015
        + 0.002
    )
    if broadcast_scales:
        scales = scales[:, :, :1].expand_as(scales)
    ref = torch.empty(quant.shape, device="cuda", dtype=torch.float32)
    offset = 0
    for b, length in enumerate(lengths):
        for start in range(0, length, 128):
            end = min(start + 128, length)
            ref[offset + start : offset + end] = (
                quant[offset + start : offset + end].float()
                * scales[b, :, start // 128, None]
            )
        offset += length
    cu = torch.tensor(
        [0] + list(itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
    )
    return quant, scales, ref, cu


def _reference(q, k, v, qlens, klens, causal, window=(-1, -1), cap=0, alibi=None):
    outputs, lses = [], []
    oq = ok = 0
    for b, (nq, nk) in enumerate(zip(qlens, klens)):
        qi = q[oq : oq + nq].transpose(0, 1)
        ki = (
            k[ok : ok + nk]
            .transpose(0, 1)
            .repeat_interleave(q.shape[1] // k.shape[1], 0)
        )
        vi = (
            v[ok : ok + nk]
            .transpose(0, 1)
            .repeat_interleave(q.shape[1] // v.shape[1], 0)
        )
        scores = qi @ ki.transpose(1, 2) * q.shape[-1] ** -0.5
        if cap:
            scores = cap * (scores / cap).tanh()
        m = torch.arange(nq, device=q.device) + nk - nq
        n = torch.arange(nk, device=q.device)
        if alibi is not None:
            slope = alibi if alibi.ndim == 1 else alibi[b]
            scores -= slope[:, None, None] * (m[:, None] - n).abs()
        mask = torch.ones((nq, nk), device=q.device, dtype=torch.bool)
        if causal:
            mask &= n <= m[:, None]
        if window[0] >= 0:
            mask &= n >= m[:, None] - window[0]
        if window[1] >= 0:
            mask &= n <= m[:, None] + window[1]
        scores = scores.masked_fill(~mask, -torch.inf)
        outputs.append((scores.softmax(-1).nan_to_num() @ vi).transpose(0, 1))
        lse = scores.logsumexp(-1)
        lses.append(lse.masked_fill(~mask.any(-1), torch.inf))
        oq += nq
        ok += nk
    return torch.cat(outputs), torch.cat(lses, dim=1)


def _run_case(
    qlens,
    klens,
    dim=64,
    heads=4,
    kvheads=4,
    causal=False,
    dtype=torch.bfloat16,
    paged=False,
    strided=False,
    window=(-1, -1),
    cap=0,
    alibi=None,
    broadcast_scales=False,
):
    q, qs, qr, cuq = _inputs(qlens, heads, dim, broadcast_scales)
    k, ks, kr, cuk = _inputs(klens, kvheads, dim, broadcast_scales)
    v, vs, vr, _ = _inputs(klens, kvheads, dim, broadcast_scales)
    # Use different V data/scales to detect accidentally reusing K or its scale.
    v = -v
    vs = (vs[:, :, :1] * 1.7).expand_as(vs) if broadcast_scales else vs * 1.7
    vr = -vr * 1.7
    kwargs = dict(cu_seqlens_k=cuk)
    if paged:
        page_size = 16
        pages = (max(klens) + page_size - 1) // page_size
        table = (
            torch.randperm(len(klens) * pages, device="cuda")
            .to(torch.int32)
            .reshape(len(klens), pages)
        )
        kc = torch.empty(
            (table.numel(), page_size, kvheads, dim), device="cuda", dtype=torch.int8
        )
        vc = torch.empty_like(kc)
        offset = 0
        for b, length in enumerate(klens):
            for start in range(0, length, page_size):
                count = min(page_size, length - start)
                kc[table[b, start // page_size].long(), :count] = k[
                    offset + start : offset + start + count
                ]
                vc[table[b, start // page_size].long(), :count] = v[
                    offset + start : offset + start + count
                ]
            offset += length
        k, v = kc, vc
        kwargs = dict(
            seqused_k=torch.tensor(klens, device="cuda", dtype=torch.int32),
            block_table=table,
        )
    out = torch.empty(q.shape, device="cuda", dtype=dtype)
    if strided:

        def padded(x):
            storage = torch.empty(
                (*x.shape[:-2], x.shape[-2] * 2, x.shape[-1]),
                device=x.device,
                dtype=x.dtype,
            )
            result = storage[..., ::2, :]
            result.copy_(x)
            return result

        q, k, v, out = [padded(x) for x in (q, k, v, out)]
        qs, ks, vs = [
            x.transpose(1, 2).contiguous().transpose(1, 2) for x in (qs, ks, vs)
        ]
    result, lse = flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        max(qlens),
        cuq,
        max(klens),
        **kwargs,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=causal,
        return_softmax_lse=True,
        out=out,
        window_size=window,
        softcap=cap,
        alibi_slopes=alibi,
    )
    ref, ref_lse = _reference(qr, kr, vr, qlens, klens, causal, window, cap, alibi)
    assert result is out
    torch.testing.assert_close(result.float(), ref, atol=0.025, rtol=0.025)
    torch.testing.assert_close(lse, ref_lse, atol=2e-5, rtol=2e-5)
    return qr, kr, vr, cuq, cuk, result


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [([1], [1]), ([17, 129], [33, 257]), ([131, 0, 3], [1, 0, 0]), ([128], [256])],
)
def test_packed(dim, dtype, causal, qlens, klens):
    _run_case(qlens, klens, dim=dim, dtype=dtype, causal=causal)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_gqa(dim, causal):
    _run_case([17, 129], [145, 257], dim=dim, kvheads=2, causal=causal, paged=True)


@pytest.mark.parametrize("paged", [False, True])
def test_strided(paged):
    _run_case([17, 129], [145, 257], kvheads=1, paged=paged, strided=True)


@pytest.mark.parametrize(
    "window,cap,alibi",
    [
        ((31, 0), 0, None),
        ((-1, 17), 3, None),
        ((32, 9), 0, "head"),
        ((-1, -1), 4, "batch"),
    ],
)
def test_score_modifiers(window, cap, alibi):
    slopes = (
        None
        if alibi is None
        else torch.rand((4,) if alibi == "head" else (2, 4), device="cuda") * 0.1
    )
    _run_case([17, 129], [145, 257], window=window, cap=cap, alibi=slopes)


def test_bf16_baseline():
    q, k, v, cuq, cuk, result = _run_case([17, 129], [33, 257], causal=True)
    baseline = flaggems_vllm.flash_attn_varlen_func(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), 129, cuq, 257, cuk, causal=True
    )
    torch.testing.assert_close(result, baseline, atol=0.03, rtol=0.03)


def test_export_signature_and_empty():
    op = flaggems_vllm.flash_attn_varlen_func_w8a8_int8
    assert inspect.signature(op) == inspect.signature(
        flaggems_vllm.flash_attn_varlen_func
    )
    assert flaggems_vllm.ops.flash_attn_varlen_func_w8a8_int8 is op
    assert op.__module__.startswith("flaggems_vllm.runtime.backend._thead.")
    assert op in [entry[1] for entry in flaggems_vllm._FULL_CONFIG]
    q = torch.empty((0, 4, 64), device="cuda", dtype=torch.int8)
    cu = torch.zeros(2, device="cuda", dtype=torch.int32)
    out, lse = op(q, q, q, 0, cu, 0, cu, return_softmax_lse=True)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert lse.shape == (4, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dropout_p=0.1),
        dict(return_attn_probs=True),
        dict(num_splits=1),
        dict(cp_world_size=2),
        dict(fa_version=3),
    ],
)
def test_unsupported(kwargs):
    q = torch.empty((1, 4, 64), device="cuda", dtype=torch.int8)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.flash_attn_varlen_func_w8a8_int8(q, q, q, 1, cu, 1, cu, **kwargs)


@pytest.mark.parametrize("zero_scale", [False, True])
def test_default_output_and_broadcast_scales(zero_scale):
    q = torch.full((3, 2, 64), -128, device="cuda", dtype=torch.int8)
    k = torch.full((129, 2, 64), 127, device="cuda", dtype=torch.int8)
    v = -k
    cuq = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 129], device="cuda", dtype=torch.int32)
    scale = torch.full((1, 2, 1), 0.0 if zero_scale else 0.01, device="cuda")
    result = flaggems_vllm.flash_attn_varlen_func_w8a8_int8(
        q,
        k,
        v,
        3,
        cuq,
        129,
        cuk,
        q_descale=scale,
        k_descale=scale.expand(1, 2, 2),
        v_descale=scale.expand(1, 2, 2),
    )
    assert result.dtype == torch.bfloat16
    expected = torch.full_like(result, 0.0 if zero_scale else -1.27)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("entry", ["top_level", "ops"])
def test_public_float_route(dtype, entry):
    from flaggems_vllm.ops.attention import flash_attn_varlen_func as shared

    q, _, _, cuq = _inputs([17, 129], 4, 64)
    k, _, _, cuk = _inputs([33, 257], 4, 64)
    q, k = q.to(dtype) * 0.01, k.to(dtype) * 0.01
    v = -k
    op = (
        flaggems_vllm if entry == "top_level" else flaggems_vllm.ops
    ).flash_attn_varlen_func
    assert op is shared
    assert inspect.signature(op) == inspect.signature(shared)
    out = torch.empty_like(q)
    actual, lse = op(
        q, k, v, 129, cuq, 257, cuk, causal=True, out=out, return_softmax_lse=True
    )
    expected, expected_lse = _reference(
        q.float(), k.float(), v.float(), [17, 129], [33, 257], True
    )
    assert actual is out
    torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)


def test_public_int8_matches_specialized():
    from flaggems_vllm.ops.attention import flash_attn_varlen_func as shared

    q, qs, _, cuq = _inputs([17, 129], 4, 64)
    k, ks, _, cuk = _inputs([33, 257], 4, 64)
    v, vs = -k, ks * 1.7
    kwargs = dict(
        q_descale=qs, k_descale=ks, v_descale=vs, causal=True, return_softmax_lse=True
    )
    expected = flaggems_vllm.flash_attn_varlen_func_w8a8_int8(
        q, k, v, 129, cuq, 257, cuk, **kwargs
    )
    for op in (
        shared,
        flaggems_vllm.flash_attn_varlen_func,
        flaggems_vllm.ops.flash_attn_varlen_func,
    ):
        actual = op(q, k, v, 129, cuq, 257, cuk, **kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("qlens,klens", [([512], [1024]), ([513, 1025], [257, 2049])])
def test_long_sequences(dim, causal, qlens, klens):
    _run_case(qlens, klens, dim=dim, causal=causal)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_probability_quantization_accuracy(seed):
    torch.manual_seed(seed)
    tensors, descales, references = [], [], []
    for heads in (16, 8, 8):
        x = torch.randn((512, heads, 128), device="cuda", dtype=torch.bfloat16)
        scale = x.float().abs().amax((0, 2)) / 127
        quant = (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        tensors.append(quant)
        descales.append(scale[None, :, None].expand(1, heads, 4))
        references.append(quant.float() * scale[:, None])
    cu = torch.tensor([0, 512], dtype=torch.int32, device="cuda")
    out, lse = flaggems_vllm.flash_attn_varlen_func(
        *tensors,
        512,
        cu,
        512,
        cu,
        causal=True,
        return_softmax_lse=True,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
    )
    expected, expected_lse = _reference(*references, [512], [512], True)
    torch.testing.assert_close(out.float(), expected, atol=0.025, rtol=0.025)
    torch.testing.assert_close(lse, expected_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("heads,kvheads", [(8, 2), (16, 1), (6, 2)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [(-1, -1), (17, 3)])
def test_paged_short_query_gqa(dim, heads, kvheads, causal, window):
    _run_case(
        [1, 4, 8, 16],
        [1, 17, 129, 257],
        dim=dim,
        heads=heads,
        kvheads=kvheads,
        causal=causal,
        paged=True,
        window=window,
        cap=5.0,
        alibi=torch.linspace(0.01, 0.1, heads, device="cuda"),
    )


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("heads,kvheads", [(8, 2), (16, 1)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("broadcast_scales", [False, True])
def test_paged_long_query_gqa(dim, heads, kvheads, causal, broadcast_scales):
    _run_case(
        [129, 513],
        [257, 1025],
        dim=dim,
        heads=heads,
        kvheads=kvheads,
        causal=causal,
        paged=True,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_mixed_query_grid(dim, causal):
    _run_case(
        [1, 2, 513, 1],
        [129, 257, 1025, 17],
        dim=dim,
        heads=8,
        kvheads=2,
        causal=causal,
        paged=True,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_long_query_strided(dtype):
    _run_case(
        [129, 257],
        [257, 513],
        dim=128,
        heads=8,
        kvheads=2,
        causal=True,
        paged=True,
        strided=True,
        dtype=dtype,
    )


def test_paged_long_query_empty_kv():
    _run_case(
        [129, 0, 3], [0, 0, 0], dim=128, heads=8, kvheads=2, causal=True, paged=True
    )
