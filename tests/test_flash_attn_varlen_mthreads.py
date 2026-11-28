# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent FP32 O/LSE and graph checks for the MThreads v1 paths."""

import itertools

import numpy as np
import pytest
import torch

import flaggems_vllm

pytest_plugins = ("tests.mthreads_attention_test_utils",)

pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "mthreads", reason="MThreads backend only"
    ),
]


def make_case(dtype, paged, qlens, klens, heads=(8, 2), dim=64, strided=False, page=16):
    device = "musa"
    torch.manual_seed(20260929)
    hq, hk = heads
    q = torch.randn(
        (sum(qlens) * (2 if strided else 1), hq, dim), device=device, dtype=dtype
    )
    if strided:
        q = q[::2]
    cuq = torch.tensor(
        [0, *itertools.accumulate(qlens)], device=device, dtype=torch.int32
    )
    cuk = torch.tensor(
        [0, *itertools.accumulate(klens)], device=device, dtype=torch.int32
    )
    if paged:
        width = max(1, (max(klens, default=0) + page - 1) // page)
        shape = (max(1, len(qlens) * width + 3), page, hk, dim)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)
        if strided:
            k = torch.stack([k, k], dim=1)[:, 0]
            v = torch.stack([v, v, v], dim=1)[:, 1]
        table = torch.randperm(shape[0], device=device)[: len(qlens) * width]
        table = table.reshape(len(qlens), width).to(torch.int32)
        used = torch.tensor(klens, device=device, dtype=torch.int32)
        lengths = dict(block_table=table, seqused_k=used)
    else:
        k = torch.randn((sum(klens), hk, dim), device=device, dtype=dtype)
        v = torch.randn_like(k)
        if strided:
            k = torch.stack([k, k], dim=1)[:, 0]
            v = torch.stack([v, v, v], dim=1)[:, 1]
        lengths = dict(cu_seqlens_k=cuk)
    args = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cuq,
        max_seqlen_q=max(qlens, default=0),
        max_seqlen_k=max(klens, default=0),
        return_softmax_lse=True,
        **lengths,
    )
    return args


def reference(args, qlens, klens):
    # CPU FP32 keeps this oracle independent of MUSA reduction edge cases.
    q, k, v = (args[x].cpu() for x in ("q", "k", "v"))
    out = torch.zeros_like(q)
    lse = torch.full((q.shape[1], q.shape[0]), float("inf"), device=q.device)
    qb = kb = 0
    groups = q.shape[1] // k.shape[-2]
    for b, (ql, kl) in enumerate(zip(qlens, klens)):
        if ql and kl:
            if "block_table" in args:
                indices = args["block_table"][b].cpu().long()
                ki = k[indices].reshape(-1, k.shape[-2], k.shape[-1])[:kl]
                vi = v[indices].reshape(-1, v.shape[-2], v.shape[-1])[:kl]
            else:
                ki, vi = k[kb : kb + kl], v[kb : kb + kl]
            ki = ki.float().repeat_interleave(groups, dim=1)
            vi = vi.float().repeat_interleave(groups, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q[qb : qb + ql].float(), ki)
            scores *= args.get("softmax_scale", q.shape[-1] ** -0.5)
            cap = args.get("softcap", 0.0)
            if cap:
                scores = torch.from_numpy(cap * np.tanh(scores.numpy() / cap))
            pos = torch.arange(ql, device=q.device)[:, None] + kl - ql
            col = torch.arange(kl, device=q.device)[None, :]
            slopes = args.get("alibi_slopes")
            if slopes is not None:
                slopes = slopes.cpu()
                slopes = slopes[b] if slopes.ndim == 2 else slopes
                scores -= slopes[:, None, None] * (pos - col).abs()[None]
            valid = torch.ones((ql, kl), device=q.device, dtype=torch.bool)
            if args.get("causal", False):
                valid &= col <= pos
            left, right = args.get("window_size", (-1, -1))
            if left >= 0:
                valid &= col >= pos - left
            if right >= 0:
                valid &= col <= pos + right
            scores.masked_fill_(~valid[None], -float("inf"))
            # This MUSA torch build traps in CPU logsumexp. Use a stable NumPy
            # softmax oracle and explicitly avoid arithmetic on all-masked rows.
            values = scores.numpy().astype(np.float64)
            maximum = values.max(axis=-1, keepdims=True)
            maximum[~np.isfinite(maximum)] = 0
            weights = np.exp(values - maximum)
            denominator = weights.sum(axis=-1, keepdims=True)
            safe_denominator = np.maximum(denominator, np.finfo(np.float64).tiny)
            probs = torch.from_numpy((weights / safe_denominator).astype(np.float32))
            normalizer = torch.from_numpy(
                np.where(denominator > 0, maximum + np.log(safe_denominator), np.inf)
                .squeeze(-1)
                .astype(np.float32)
            )
            out[qb : qb + ql] = torch.einsum("hqk,khd->qhd", probs, vi).to(q.dtype)
            lse[:, qb : qb + ql] = torch.where(
                normalizer == -float("inf"), float("inf"), normalizer
            )
        qb += ql
        kb += kl
    return out, lse


CASES = [
    ([1, 1], [5, 11], (32, 1), 8),
    ([1, 1, 0], [137, 0, 33], (8, 2), 256),
    ([0, 3, 17], [0, 71, 11], (4, 1), 192),
    ([9, 1], [35, 17], (4, 4), 80),
    ([33, 2], [97, 0], (8, 2), 128),
    ([0, 0], [0, 1], (4, 1), 64),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("paged", [True, False])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", ["causal", "window", "softcap_alibi"])
def test_reference(dtype, paged, case, mode):
    qlens, klens, heads, dim = case
    args = make_case(dtype, paged, qlens, klens, heads, dim, strided=True)
    args["causal"] = mode != "window"
    if mode == "window":
        args["window_size"] = (-1, 2)
    if mode == "softcap_alibi":
        args["softcap"] = 10.0
        args["window_size"] = (19, 0)
        args["alibi_slopes"] = torch.rand((len(qlens), heads[0]), device="musa") * 0.2
    expected, expected_lse = reference(args, qlens, klens)
    args["out"] = torch.empty(args["q"].shape, device="musa", dtype=dtype)
    actual, actual_lse = flaggems_vllm.flash_attn_varlen_func(**args)
    assert actual is args["out"]
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("qlens", [[1, 0, 1], [5, 33]])
def test_graph_replay(qlens):
    klens = [117] * len(qlens)
    args = make_case(torch.bfloat16, True, qlens, klens, dim=256)
    args["causal"] = True
    fn = flaggems_vllm.ops.flash_attn_varlen_func
    assert fn is flaggems_vllm.flash_attn_varlen_func
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            fn(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        actual, lse = fn(**args)
    for _ in range(2):
        args["q"].normal_()
        expected, expected_lse = reference(args, qlens, klens)
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize(
    "option,value",
    [
        ("dropout_p", 0.1),
        ("return_attn_probs", True),
        ("num_splits", 2),
        ("cp_world_size", 2),
        ("fa_version", 3),
    ],
)
def test_unsupported(option, value):
    args = make_case(torch.bfloat16, True, [1], [1])
    args[option] = value
    with pytest.raises(NotImplementedError):
        flaggems_vllm.flash_attn_varlen_func(**args)


def test_out_alias_rejected():
    args = make_case(torch.bfloat16, True, [1], [1])
    args["out"] = args["q"]
    with pytest.raises(ValueError, match="alias"):
        flaggems_vllm.flash_attn_varlen_func(**args)


def test_empty_batch():
    args = make_case(torch.bfloat16, False, [], [], dim=64)
    out, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    assert out.shape == (0, 8, 64)
    assert lse.shape == (8, 0)


def test_one_dimensional_alibi():
    args = make_case(torch.bfloat16, True, [1, 1], [47, 91], dim=64)
    args["alibi_slopes"] = torch.rand((8,), device="musa") * 0.2
    expected, expected_lse = reference(args, [1, 1], [47, 91])
    actual, lse = flaggems_vllm.flash_attn_varlen_func(**args)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("paged", [True, False])
@pytest.mark.parametrize("splits", [2, 3, 8, 32])
@pytest.mark.parametrize("packed", ["direct", "gqa8", "decode"])
def test_splitkv_reference(monkeypatch, dtype, paged, splits, packed):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.scheduling import (
        AttentionPlan,
    )

    decode = packed == "decode"
    qlens = [1, 0, 1, 1] if decode else [3, 0, 17, 2]
    klens = [11, 9, 137, 0] if decode else [2, 9, 137, 0]
    heads = (16, 2) if packed == "gqa8" else (8, 2)
    args = make_case(dtype, paged, qlens, klens, heads=heads, dim=80, strided=True)
    args.update(causal=True, window_size=(35, 0), softcap=10.0)
    args["alibi_slopes"] = torch.rand((heads[0],), device="musa") * 0.2
    plan = AttentionPlan(
        "split_decode" if decode else "split_kv",
        16 if decode else 32,
        32,
        num_splits=splits,
        head_pack=8 if packed == "gqa8" else 1,
    )
    monkeypatch.setattr(launcher, "select_plan", lambda *a, **kw: plan)
    expected, expected_lse = reference(args, qlens, klens)
    actual, actual_lse = flaggems_vllm.flash_attn_varlen_func(**args)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


def test_splitkv_graph_length_changes(monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.scheduling import (
        AttentionPlan,
    )

    qlens, klens = [1, 0, 2, 3], [157, 71, 89, 191]
    args = make_case(torch.bfloat16, True, qlens, klens, dim=256)
    args.update(causal=True, max_seqlen_q=3)
    monkeypatch.setattr(
        launcher,
        "select_plan",
        lambda *a, **kw: AttentionPlan("split_kv", 32, 32, num_splits=8),
    )
    fn = flaggems_vllm.flash_attn_varlen_func
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            fn(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        actual, actual_lse = fn(**args)
    for new_qlens, new_klens in [
        ([2, 1, 0, 3], [0, 1, 17, 190]),
        ([0, 3, 2, 1], [0, 0, 2, 0]),
    ]:
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, *itertools.accumulate(new_qlens)], device="musa", dtype=torch.int32
            )
        )
        args["seqused_k"].copy_(
            torch.tensor(new_klens, device="musa", dtype=torch.int32)
        )
        args["q"].normal_()
        expected, expected_lse = reference(args, new_qlens, new_klens)
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("paged", [True, False])
@pytest.mark.parametrize("family", ["direct", "compact"])
@pytest.mark.parametrize("pack", [2, 4, 8])
def test_portable_tile_mapping(monkeypatch, dtype, paged, family, pack):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.scheduling import (
        AttentionPlan,
    )

    qlens, klens = [0, 1, 7, 33, 2], [0, 137, 0, 19, 65]
    args = make_case(dtype, paged, qlens, klens, heads=(16, 2), dim=80, strided=True)
    args.update(causal=True, window_size=(17, 0))
    plan = AttentionPlan(family, 32, 32, head_pack=pack)
    monkeypatch.setattr(launcher, "select_plan", lambda *a, **kw: plan)
    expected, expected_lse = reference(args, qlens, klens)
    actual, actual_lse = flaggems_vllm.flash_attn_varlen_func(**args)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


def test_compact_graph_length_changes(monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.scheduling import (
        AttentionPlan,
    )

    qlens, klens = [1, 0, 64, 31], [157, 71, 89, 191]
    args = make_case(torch.bfloat16, True, qlens, klens, dim=128)
    args.update(causal=True, max_seqlen_q=64)
    monkeypatch.setattr(
        launcher,
        "select_plan",
        lambda *a, **kw: AttentionPlan("compact", 32, 32, head_pack=4),
    )
    fn = flaggems_vllm.flash_attn_varlen_func
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            fn(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        actual, actual_lse = fn(**args)
    for qlens, klens in [
        ([2, 63, 0, 31], [0, 100, 17, 190]),
        ([0, 32, 64, 0], [0, 0, 2, 0]),
    ]:
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, *itertools.accumulate(qlens)], device="musa", dtype=torch.int32
            )
        )
        args["seqused_k"].copy_(torch.tensor(klens, device="musa", dtype=torch.int32))
        args["q"].normal_()
        expected, expected_lse = reference(args, qlens, klens)
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page", [16, 32])
def test_cached_reference(monkeypatch, dtype, causal, page):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import launcher

    qlens, klens = [0, 65, 129, 318], [0, 0, 97, 513]
    args = make_case(dtype, True, qlens, klens, heads=(4, 1), dim=256, page=page)
    args["causal"] = causal
    args["out"] = torch.empty_like(args["q"])
    calls = []
    original = launcher.launch_cached

    def recording(*a, **kw):
        compiled = original(*a, **kw)
        calls.append(compiled)
        assert compiled.n_spills == 0
        return compiled

    monkeypatch.setattr(launcher, "launch_cached", recording)
    expected, expected_lse = reference(args, qlens, klens)
    actual, actual_lse = flaggems_vllm.flash_attn_varlen_func(**args)
    assert calls and actual is args["out"]
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cached_graph_updates(dtype):
    qlens, klens = [0, 65, 129, 318], [0, 0, 97, 513]
    args = make_case(dtype, True, qlens, klens, heads=(4, 1), dim=256)
    args["causal"] = True
    fn = flaggems_vllm.flash_attn_varlen_func
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            fn(**args)
    torch.musa.synchronize()
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        actual, actual_lse = fn(**args)
    for qlens, klens in [
        ([65, 0, 129, 318], [0, 3, 513, 97]),
        ([129, 65, 0, 318], [1, 0, 0, 0]),
    ]:
        args["cu_seqlens_q"].copy_(
            torch.tensor(
                [0, *itertools.accumulate(qlens)], device="musa", dtype=torch.int32
            )
        )
        args["seqused_k"].copy_(torch.tensor(klens, device="musa", dtype=torch.int32))
        args["q"].normal_()
        args["k"].normal_()
        args["v"].normal_()
        args["block_table"].copy_(args["block_table"].flip(1))
        expected, expected_lse = reference(args, qlens, klens)
        graph.replay()
        torch.musa.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            actual_lse.cpu(), expected_lse, atol=0.002, rtol=0.002
        )
