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

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import runtime

from .test_paged_prefix_window_attention import make_case

pytestmark = [
    pytest.mark.paged_causal_attention,
    pytest.mark.skipif(
        runtime.device.vendor_name != "metax" or not torch.cuda.is_available(),
        reason="MetaX CUDA runtime required",
    ),
]


def inputs(dtype=torch.bfloat16, layout="packed", block=16, kv_heads=10):
    args, logical, _, _ = make_case(
        dtype=dtype,
        prefixes=(73, 91),
        generated=129,
        holes=False,
        layout=layout,
        block=block,
        kv_heads=kv_heads,
    )
    queries = [7, 13]
    torch.manual_seed(81293)
    qcpu = torch.randn((sum(queries), 10, 128)).to(dtype)
    storage = torch.empty((sum(queries), 10, 256), dtype=dtype, device="cuda")
    q = storage[..., :128]
    q.copy_(qcpu)
    result = dict(
        q=q,
        k_cache=args["k_cache"],
        v_cache=args["v_cache"],
        query_start_loc=torch.tensor([0, 7, 20], dtype=torch.int32, device="cuda"),
        seq_lens=args["seq_lens"],
        block_table=args["block_table"],
        max_query_len=13,
        max_seq_len=int(args["seq_lens"].max().item()),
    )
    return result, logical, qcpu, queries


def reference(logical, q, queries, lengths):
    values = []
    start = 0
    for b, (count, length) in enumerate(zip(queries, lengths)):
        kv = logical[b, :length].float()
        repeats = q.shape[1] // kv.shape[1]
        kv = kv.repeat_interleave(repeats, dim=1)
        scores = torch.einsum(
            "qhd,khd->hqk", q[start : start + count].float(), kv[..., :128]
        ) / math.sqrt(128)
        mask = (
            torch.arange(length)[None, :]
            <= torch.arange(length - count, length)[:, None]
        )
        scores.masked_fill_(~mask[None, :, :], -float("inf"))
        values.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), kv[..., 128:]))
        start += count
    return torch.cat(values).to(q.dtype)


@pytest.mark.paged_causal_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["packed", "nhd"])
@pytest.mark.parametrize("block", [16, 32, 64])
@pytest.mark.parametrize("kv_heads", [2, 10])
def test_ragged_chunked_layouts(dtype, layout, block, kv_heads):
    args, logical, qcpu, queries = inputs(dtype, layout, block, kv_heads)
    storage = torch.full((20, 10, 256), float("nan"), device="cuda", dtype=dtype)
    out = storage[..., :128]
    args["out"] = out
    before = [
        t.clone()
        for t in [
            args["q"],
            args["k_cache"],
            args["v_cache"],
            args["seq_lens"],
            args["block_table"],
        ]
    ]
    expected = reference(logical, qcpu, queries, args["seq_lens"].cpu().tolist())
    actual = flaggems_vllm.paged_causal_attention(**args)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.cpu(), expected, atol=0.003, rtol=0.02)
    assert torch.isfinite(actual).all() and torch.isnan(storage[..., 128:]).all()
    for old, new in zip(
        before,
        [
            args["q"],
            args["k_cache"],
            args["v_cache"],
            args["seq_lens"],
            args["block_table"],
        ],
    ):
        torch.testing.assert_close(old, new, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_graph_new_queries_lengths_and_pages(dtype):
    args, logical, qcpu, queries = inputs(dtype)
    args["out"] = torch.full_like(args["q"], float("nan"))
    flaggems_vllm.paged_causal_attention(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.paged_causal_attention(**args)
    initial = args["seq_lens"].cpu().tolist()
    for epoch in (1, 2):
        qnew = qcpu * epoch
        args["q"].copy_(qnew.cuda())
        lengths = [n - epoch for n in initial]
        args["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        # Reuse a physical page: exchange both data and block-table references.
        old = int(args["block_table"][0, 0].item())
        other = int(args["block_table"][1, 0].item())
        for cache in [args["k_cache"], args["v_cache"]]:
            saved = cache[old].clone()
            cache[old].copy_(cache[other])
            cache[other].copy_(saved)
        args["block_table"][0, 0] = other
        args["block_table"][1, 0] = old
        args["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(logical, qnew, queries, lengths)
        torch.testing.assert_close(args["out"].cpu(), expected, atol=0.003, rtol=0.02)


@pytest.mark.parametrize(
    "bad",
    [
        "dtype",
        "dim",
        "page",
        "starts",
        "lengths",
        "max_q",
        "out_dtype",
        "alias",
        "grad",
    ],
)
def test_reject_before_native_call(monkeypatch, bad):
    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_causal_attention"
    )
    args, _, _, _ = inputs()

    def forbidden():
        raise AssertionError("Native API was selected for invalid input")

    monkeypatch.setattr(mod, "_varlen_api", forbidden)
    if bad == "dtype":
        for key in ["q", "k_cache", "v_cache"]:
            args[key] = args[key].float()
    elif bad == "dim":
        for key in ["q", "k_cache", "v_cache"]:
            args[key] = args[key][..., :64]
    elif bad == "page":
        for key in ["k_cache", "v_cache"]:
            args[key] = args[key][:, :8]
    elif bad == "starts":
        args["query_start_loc"] = args["query_start_loc"].long()
    elif bad == "lengths":
        args["seq_lens"] = args["seq_lens"].long()
    elif bad == "max_q":
        args["max_query_len"] = True
    elif bad == "out_dtype":
        args["out"] = torch.empty_like(args["q"], dtype=torch.float32)
    elif bad == "alias":
        args["out"] = args["q"]
    elif bad == "grad":
        args["q"].requires_grad_(True)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        flaggems_vllm.paged_causal_attention(**args)


def test_native_execution_failure_propagates(monkeypatch):
    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_causal_attention"
    )
    args, _, _, _ = inputs()
    calls = []

    def fail(*a, **kw):
        calls.append(True)
        raise RuntimeError("test native failure")

    monkeypatch.setattr(mod, "_varlen_api", lambda: fail)
    with pytest.raises(RuntimeError, match="test native failure"):
        flaggems_vllm.paged_causal_attention(**args)
    assert calls == [True]


def test_empty_returns_output_without_native(monkeypatch):
    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_causal_attention"
    )
    monkeypatch.setattr(
        mod, "_varlen_api", lambda: pytest.fail("Empty batch called native API")
    )
    q = torch.empty((0, 10, 128), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty((0, 16, 10, 128), device="cuda", dtype=q.dtype)
    out = torch.empty_like(q)
    value = flaggems_vllm.paged_causal_attention(
        q,
        cache,
        cache,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty((0, 0), dtype=torch.int32, device="cuda"),
        0,
        0,
        out=out,
    )
    assert value is out


def test_other_vendor_rejected_before_import(monkeypatch):
    monkeypatch.setattr(runtime, "device", SimpleNamespace(vendor_name="nvidia"))
    assert not flaggems_vllm.is_paged_causal_attention_available()
    with pytest.raises(NotImplementedError, match="requires MetaX"):
        flaggems_vllm.paged_causal_attention(None, None, None, None, None, None, 0, 0)


def test_missing_native_abi_is_unavailable(monkeypatch):
    from flash_attn.flash_attn_interface import flash_attn_cuda

    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_causal_attention"
    )
    mod._varlen_api.cache_clear()
    monkeypatch.setattr(flash_attn_cuda, "varlen_fwd", None)
    assert not mod.is_paged_causal_attention_available()
    mod._varlen_api.cache_clear()
