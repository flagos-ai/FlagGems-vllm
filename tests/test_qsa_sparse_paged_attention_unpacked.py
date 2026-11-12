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

"""Stable unpacked QSA ABI, including the existing PPU vendor dispatch."""

import importlib
import inspect

import pytest
import torch

import flaggems_vllm as ops

pytestmark = pytest.mark.qsa_sparse_paged_attention_unpacked


def test_unpacked_qsa_public_abi():
    marker = object()
    inspect.signature(ops.qsa_sparse_paged_attention_unpacked).bind(
        *([marker] * 6),
        softmax_scale=0.2,
        out=marker,
        gate=marker,
        split_workspace=None,
    )


def test_unpacked_qsa_is_library_only():
    assert "qsa_sparse_paged_attention_unpacked" in ops._LIBRARY_ONLY_OPS
    assert all(
        name != "qsa_sparse_paged_attention_unpacked" for name, _ in ops._FULL_CONFIG
    )


def test_unpacked_qsa_generic_abi_is_preserved():
    generic = importlib.import_module("flaggems_vllm.ops.qwen4.qsa_attention")
    assert (
        ops._ops_module.qsa_sparse_paged_attention_unpacked
        is generic.qsa_sparse_paged_attention
    )
    if ops.runtime.device.vendor_name == "thead":
        # Both public names must retain the already-tested vendor implementation.
        assert ops.qsa_sparse_paged_attention_unpacked is ops.qsa_sparse_paged_attention


@pytest.mark.skipif(
    ops.runtime.device.vendor_name != "thead", reason="PPU vendor-route regression"
)
@pytest.mark.parametrize("group_size", [1, 3, 17])
@pytest.mark.parametrize("width", [32, 256])
@pytest.mark.parametrize("with_gate", [False, True])
def test_unpacked_qsa_public_changed_input_graph(
    monkeypatch, group_size, width, with_gate
):
    generic = importlib.import_module("flaggems_vllm.ops.qwen4.qsa_attention")
    original = generic._qsa_sparse_paged_gqa_kernel
    launches = []

    class ObservedLauncher:
        def __getitem__(self, grid):
            launch = original[grid]

            def invoke(*args, **kwargs):
                launches.append(kwargs["BLOCK_M"])
                return launch(*args, **kwargs)

            return invoke

    monkeypatch.setattr(generic, "_qsa_sparse_paged_gqa_kernel", ObservedLauncher())
    generator = torch.Generator().manual_seed(1134)
    host = [
        torch.randn(shape, generator=generator).to(torch.bfloat16)
        for shape in [
            (3, group_size, width),
            (2, 4, 1, width),
            (2, 4, 1, width),
            (3, group_size, width),
        ]
    ]
    inputs = []
    for cpu in host:
        storage = torch.empty(
            (*cpu.shape[:-1], width + 8), dtype=cpu.dtype, device="cuda"
        )
        view = storage[..., 3 : 3 + width]
        view.copy_(cpu)
        assert view.storage_offset() == 3 and not view.is_contiguous()
        inputs.append(view)
    query, keys, values, gate = inputs
    indices_host = torch.tensor([[0, 3, -1, 6]] * 3, dtype=torch.int32)
    table_host = torch.tensor([[1, 0]], dtype=torch.int32)
    indices, table = indices_host.cuda(), table_host.cuda()
    requests = torch.zeros(3, dtype=torch.int32, device=query.device)
    out_storage = torch.full(
        (3, group_size, width + 8), 17, dtype=query.dtype, device=query.device
    )
    out = out_storage[..., 2 : 2 + width]

    def invoke():
        return ops.qsa_sparse_paged_attention_unpacked(
            query,
            keys,
            values,
            indices,
            table,
            requests,
            softmax_scale=0.2,
            out=out,
            gate=gate if with_gate else None,
        )

    def check(result):
        expected = torch.empty_like(host[0], dtype=torch.float32)
        for row in range(3):
            selected_keys, selected_values = [], []
            for token in indices_host[row].tolist():
                if token < 0:
                    continue
                block = int(table_host[0, token // 4])
                selected_keys.append(host[1][block, token % 4, 0].float())
                selected_values.append(host[2][block, token % 4, 0].float())
            k, v = torch.stack(selected_keys), torch.stack(selected_values)
            for head in range(group_size):
                weights = (host[0][row, head].float() @ k.T * 0.2).softmax(-1)
                expected[row, head] = weights @ v
        if with_gate:
            expected = (expected.to(torch.bfloat16) * host[3].sigmoid()).float()
        assert result is out
        torch.testing.assert_close(
            result.cpu().float(), expected, atol=0.025, rtol=0.025
        )
        guards = out_storage.cpu()
        assert torch.all(guards[..., :2] == 17)
        assert torch.all(guards[..., width + 2 :] == 17)

    check(invoke())
    assert launches == [max(16, 1 << (group_size - 1).bit_length())]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = invoke()
    for step in range(2):
        for index, (cpu, device) in enumerate(zip(host, inputs)):
            cpu.mul_(-0.6).add_((step + index + 1) * 0.1)
            device.copy_(cpu)
        indices_host[:, 1] = step + 1
        indices.copy_(indices_host)
        if step == 1:
            table_host.copy_(table_host.flip(1))
            table.copy_(table_host)
        out.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        check(result)
