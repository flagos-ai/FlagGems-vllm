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

pytestmark = [
    pytest.mark.paged_prefix_window_attention,
    pytest.mark.skipif(
        runtime.device.vendor_name != "metax" or not torch.cuda.is_available(),
        reason="MetaX CUDA runtime required",
    ),
]


def make_case(
    dtype=torch.bfloat16,
    generated=129,
    prefixes=(273, 301),
    window=128,
    block=16,
    layout="packed",
    onehot=False,
    reuse=False,
    heads=10,
    kv_heads=10,
    holes=True,
):
    torch.set_num_threads(4)
    batch, dim = len(prefixes), 128
    lengths = [p + generated + 1 for p in prefixes]
    pages = (max(lengths) + block - 1) // block
    generator = torch.Generator().manual_seed(719)
    shape = (batch, pages * block, kv_heads, 2 * dim)
    logical = (
        torch.zeros(shape) if onehot else torch.randn(shape, generator=generator)
    ).to(dtype)
    q_cpu = (
        torch.zeros(batch, heads, dim)
        if onehot
        else torch.randn(batch, heads, dim, generator=generator)
    ).to(dtype)
    if onehot:
        assert heads == kv_heads
        for b, length in enumerate(lengths):
            for pos in range(length):
                logical[b, pos, pos // dim, dim + pos % dim] = b + 1
    permutation = torch.randperm(batch * pages, generator=generator) + 1
    table_cpu = permutation.reshape(batch, pages).to(torch.int32)
    logical_pages = logical.reshape(batch * pages, block, kv_heads, 2 * dim)
    if layout == "packed":
        owner = torch.full(
            (1 + batch * pages, kv_heads, block, 2 * dim),
            30,
            device="cuda",
            dtype=dtype,
        )
        owner.index_copy_(0, permutation.cuda(), logical_pages.cuda().transpose(1, 2))
        k, v = owner[..., :dim].transpose(1, 2), owner[..., dim:].transpose(1, 2)
    else:
        owner = torch.full(
            (1 + batch * pages, block, kv_heads, 2 * dim),
            30,
            device="cuda",
            dtype=dtype,
        )
        owner.index_copy_(0, permutation.cuda(), logical_pages.cuda())
        k, v = owner[..., :dim], owner[..., dim:]
    freed = []
    if holes:
        for b, (prefix, length) in enumerate(zip(prefixes, lengths)):
            start, end = (prefix + block - 1) // block, max(
                prefix, length - window
            ) // block
            freed.extend(table_cpu[b, start:end].tolist())
            table_cpu[b, start:end] = 0
    if reuse:
        assert freed
        index = (lengths[-1] - 1) // block
        source, destination = int(table_cpu[-1, index]), freed[0]
        owner[destination].copy_(owner[source])
        table_cpu[-1, index] = destination
    q = q_cpu.cuda()
    seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    cuts = torch.tensor(
        [min(p, max(l - window, 0)) for p, l in zip(prefixes, lengths)],
        dtype=torch.int32,
        device="cuda",
    )
    args = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        seq_lens=seq,
        block_table=table_cpu.cuda(),
        prefix_lens=cuts,
        window_size=window,
    )
    return args, logical, q_cpu, owner


def reference(logical, q, prefixes, lengths, window):
    result = []
    group = q.shape[1] // logical.shape[2]
    for b, (prefix, length) in enumerate(zip(prefixes, lengths)):
        ids = torch.cat(
            (
                torch.arange(min(prefix, length)),
                torch.arange(max(prefix, length - window), length),
            )
        )
        kv = logical[b, ids].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,lhd->hl", q[b].float(), kv[..., :128]) / math.sqrt(
            128
        )
        result.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), kv[..., 128:]))
    return torch.stack(result)


def check(args, logical, q_cpu, prefixes, generated, onehot=False):
    expected = reference(
        logical,
        q_cpu,
        prefixes,
        [p + generated + 1 for p in prefixes],
        args["window_size"],
    )
    result = flaggems_vllm.paged_prefix_window_attention(**args)
    actual = result.float().cpu()
    if onehot:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-2)
        for row, prefix in enumerate(prefixes):
            assert torch.all(actual[row].flatten()[:prefix] > 0)
    else:
        delta = actual - expected
        nrms = delta.square().mean().sqrt() / expected.square().mean().sqrt()
        assert nrms < 0.008
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.01)
    return result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.paged_prefix_window_attention
@pytest.mark.parametrize("generated", [0, 1, 126, 127, 128, 129, 223, 1023])
@pytest.mark.parametrize("layout", ["packed", "NHD"])
@torch.inference_mode()
def test_values_and_readonly_inputs(dtype, generated, layout):
    args, logical, q_cpu, owner = make_case(
        dtype=dtype, generated=generated, layout=layout
    )
    before = owner.clone()
    check(args, logical, q_cpu, (273, 301), generated)
    torch.testing.assert_close(owner, before, atol=0, rtol=0)
    torch.testing.assert_close(args["q"].cpu(), q_cpu, atol=0, rtol=0)


@pytest.mark.parametrize(
    "generated,reuse",
    [(127, False), (128, False), (129, False), (223, False), (223, True), (257, True)],
)
@torch.inference_mode()
def test_every_prefix_position_and_retired_pages(generated, reuse):
    args, logical, q_cpu, _ = make_case(generated=generated, onehot=True, reuse=reuse)
    check(args, logical, q_cpu, (273, 301), generated, onehot=True)


@pytest.mark.parametrize("prefix", [0, 1, 127, 128])
@pytest.mark.parametrize("window", [16, 128, 257])
@torch.inference_mode()
def test_small_prefix_and_tail_overlap(prefix, window):
    args, logical, q_cpu, _ = make_case(prefixes=(prefix,), generated=1, window=window)
    check(args, logical, q_cpu, (prefix,), 1)


@pytest.mark.parametrize("block", [16, 32, 64, 256])
@pytest.mark.parametrize("heads,kv_heads", [(10, 10), (8, 2)])
@torch.inference_mode()
def test_page_sizes_and_gqa(block, heads, kv_heads):
    args, logical, q_cpu, _ = make_case(block=block, heads=heads, kv_heads=kv_heads)
    check(args, logical, q_cpu, (273, 301), 129)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_graph_updated_queries_lengths_and_output_storage(dtype):
    args, logical, q_cpu, _ = make_case(dtype=dtype, generated=256, holes=False)
    storage = torch.full((2, 10, 256), float("nan"), device="cuda", dtype=dtype)
    args["out"] = storage[..., :128]
    result = flaggems_vllm.paged_prefix_window_attention(**args)
    assert result.data_ptr() == args["out"].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flaggems_vllm.paged_prefix_window_attention(**args)
    for epoch, generated in enumerate([0, 129, 223], 1):
        q = (q_cpu.float() * epoch).to(dtype)
        lengths = [p + generated + 1 for p in (273, 301)]
        args["q"].copy_(q)
        args["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        args["prefix_lens"].copy_(
            torch.tensor(
                [min(p, max(l - 128, 0)) for p, l in zip((273, 301), lengths)],
                dtype=torch.int32,
                device="cuda",
            )
        )
        storage.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(logical, q, (273, 301), lengths, 128)
        torch.testing.assert_close(
            args["out"].float().cpu(), expected, atol=0.002, rtol=0.01
        )
        assert torch.isnan(storage[..., 128:]).all()


def test_bad_arguments_rejected_before_vendor_call(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_prefix_window_attention"
    )
    args, _, _, _ = make_case(prefixes=(273,))
    calls = []
    monkeypatch.setattr(module, "_kv_api", lambda: calls.append(True))
    with pytest.raises(ValueError, match="window_size"):
        flaggems_vllm.paged_prefix_window_attention(**{**args, "window_size": 0})
    with pytest.raises(ValueError, match="overlap"):
        flaggems_vllm.paged_prefix_window_attention(**args, out=args["q"])
    with pytest.raises(ValueError, match="Lengths"):
        flaggems_vllm.paged_prefix_window_attention(
            **{**args, "prefix_lens": args["prefix_lens"].long()}
        )
    assert not calls


def test_execution_error_is_not_retried(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_prefix_window_attention"
    )
    args, _, _, _ = make_case(prefixes=(273,))
    calls = []

    def failing(*unused):
        calls.append(True)
        raise RuntimeError("vendor execution failed")

    monkeypatch.setattr(module, "_kv_api", lambda: failing)
    with pytest.raises(RuntimeError, match="vendor execution failed"):
        flaggems_vllm.paged_prefix_window_attention(**args)
    assert len(calls) == 1


def test_empty_batch_does_not_call_vendor(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_prefix_window_attention"
    )

    def forbidden():
        raise AssertionError("An empty batch must not load the vendor API")

    monkeypatch.setattr(module, "_kv_api", forbidden)
    q = torch.empty((0, 10, 128), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty((0, 16, 10, 128), device="cuda", dtype=q.dtype)
    lengths = torch.empty((0,), device="cuda", dtype=torch.int32)
    table = torch.empty((0, 0), device="cuda", dtype=torch.int32)
    out = flaggems_vllm.paged_prefix_window_attention(
        q, cache, cache, lengths, table, lengths, 128
    )
    assert out.shape == q.shape


def test_vendor_abi_checked_before_selection(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.paged_prefix_window_attention"
    )
    interface = importlib.import_module("flash_attn.flash_attn_interface")

    def incompatible():
        raise AssertionError("An incompatible ABI must not execute")

    module._kv_api.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(
            interface, "flash_attn_cuda", SimpleNamespace(fwd_kvcache=incompatible)
        )
        assert not module.is_paged_prefix_window_attention_available()
        with pytest.raises(RuntimeError, match="21 arguments"):
            module._kv_api()
    module._kv_api.cache_clear()


def test_unsupported_vendor_rejected_before_optional_import(monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.paged_prefix_window_attention")
    monkeypatch.setattr(
        module, "runtime", SimpleNamespace(device=SimpleNamespace(vendor_name="nvidia"))
    )
    assert not module.is_paged_prefix_window_attention_available()
    with pytest.raises(NotImplementedError, match="MetaX"):
        module.paged_prefix_window_attention(None, None, None, None, None, None, 128)
