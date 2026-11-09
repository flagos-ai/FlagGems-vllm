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

"""MegaMoE tests for NVIDIA Hopper and MetaX backends."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

_MEGAMOE_BACKEND = os.environ.get("DNN_VENDOR", "").lower()
if not _MEGAMOE_BACKEND:
    try:
        _MEGAMOE_BACKEND = "metax" if "metax" in version("torch").lower() else "nvidia"
    except PackageNotFoundError:
        _MEGAMOE_BACKEND = "nvidia"


if _MEGAMOE_BACKEND == "metax":
    import ctypes
    import importlib
    import inspect
    import json
    import os
    import subprocess
    import sys
    from contextlib import ExitStack
    from dataclasses import FrozenInstanceError
    from pathlib import Path
    from types import SimpleNamespace

    import pytest
    import torch
    import triton
    import triton.language as tl

    from benchmark import test_megamoe as benchmark_module
    from benchmark.test_megamoe import create_workspace
    from flaggems_vllm import megamoe
    from flaggems_vllm.runtime import device
    from flaggems_vllm.runtime.backend._metax.mega.megamoe.kernel import (
        _return_packet,
        _symmetric_int8,
        plan_workspace,
    )
    from flaggems_vllm.testing.megamoe import (
        assert_moe_output,
        make_inputs,
        reference,
        validate,
    )

    pytestmark = [
        pytest.mark.megamoe,
        pytest.mark.skipif(device.vendor_name != "metax", reason="MetaX EP kernel"),
    ]

    @pytest.fixture(scope="module")
    def shmem():
        # Even a local SDK device call requires an initialized SHMEM device state.
        torch.empty(1, device="cuda")
        importlib.import_module("vllm._custom_ops")
        library = ctypes.CDLL(os.environ["MEGAMOE_HOST_LIB"])
        library.mxshmem_broadcast_bytes_wrapper.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.mxshmem_all_success_wrapper.argtypes = [ctypes.c_int]
        library.mxshmem_all_success_wrapper.restype = ctypes.c_int
        library.mxshmem_init_wrapper()
        yield library
        torch.cuda.synchronize()
        library.mxshmem_finalize_wrapper()

    @pytest.fixture
    def workspace(shmem):
        with ExitStack() as stack:

            def build(*args, **kwargs):
                return stack.enter_context(
                    create_workspace(*args, shmem=shmem, **kwargs)
                )

            yield build

    @pytest.mark.parametrize(
        "t,h,i,experts,top_k",
        [
            (1, 128, 64, 4, 2),
            (7, 80, 70, 5, 3),
            (33, 256, 128, 8, 4),
        ],
    )
    def test_tensor_contract(t, h, i, experts, top_k, workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(t, h, i, experts, top_k)
        ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        y = megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(y.device)
        assert_moe_output(y, expected)

    def test_dynamic_routes_and_token_counts(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(65, 128, 64, 8, 2)
        ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        for t, expert in [(65, 7), (3, 0), (33, 6), (0, 0), (65, 1)]:
            current = routes[:t].clone()
            current[:, 0] = expert
            current[:, 1] = -1
            y = megamoe(
                x[:t],
                w1,
                w2,
                weights[:t],
                current,
                s1,
                s2,
                workspace=ws,
                a1_scale=xs[:t],
            )
            expected = reference(
                x[:t], xs[:t], w1, w2, s1, s2, current, weights[:t]
            ).to(y.device)
            assert_moe_output(y, expected)

    def test_noncontiguous_inputs_and_expert_map(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(9, 128, 64, 4, 2)
        x = x.T.contiguous().T
        routes = routes.T.contiguous().T
        weights = weights.T.contiguous().T
        permutation = torch.tensor([2, 0, 3, 1], device=x.device)
        mapping = torch.argsort(permutation).int()
        ws = workspace(
            x,
            w1[permutation],
            w2[permutation],
            weights,
            routes,
            s1[permutation],
            s2[permutation],
            a1_scale=xs,
            expert_map=mapping,
        )
        y = megamoe(
            x,
            w1[permutation],
            w2[permutation],
            weights,
            routes,
            s1[permutation],
            s2[permutation],
            workspace=ws,
            a1_scale=xs,
            expert_map=mapping,
        )
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(y.device)
        assert_moe_output(y, expected)

    def test_single_launch_and_bf16_input(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
        hidden = (x.float() * xs[:, None]).bfloat16()
        ws = workspace(hidden, w1, w2, weights, routes, s1, s2)
        megamoe(hidden, w1, w2, weights, routes, s1, s2, workspace=ws)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            y = megamoe(hidden, w1, w2, weights, routes, s1, s2, workspace=ws)
            torch.cuda.synchronize()
        assert (
            len(
                [
                    e
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
            )
            == 1
        )
        cpu = hidden.float().cpu()
        amax = cpu.abs().amax(1)
        scale = torch.where(amax > 0, torch.div(amax, torch.tensor(127.0)), 1.0)
        q = torch.div(cpu, scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        expected = reference(q, scale, w1, w2, s1, s2, routes, weights).to(y.device)
        assert_moe_output(y, expected)

    def test_graph_replay_with_changing_routes(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
        for expert in (0, 3, 1, 2, 0):
            routes[:, 0] = expert
            routes[:, 1] = -1
            graph.replay()
            expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
            assert_moe_output(out, expected)

    def test_dense_workspace_occupancy_changes_and_graph(workspace, monkeypatch):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(19, 80, 70, 3, 2)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        harness = importlib.import_module("benchmark.test_megamoe")
        query = harness.resident_blocks
        strided_w1 = w1.transpose(1, 2).contiguous().transpose(1, 2)

        def forbidden_query(*args):
            raise AssertionError("occupancy queried during an operator call")

        for limit, tokens in ((1, 19), (2, 3), (1, 0), (3, 19)):
            monkeypatch.setattr(harness, "resident_blocks", query)
            current_w1 = w1 if limit == 1 else strided_w1
            ws = workspace(
                x[:tokens],
                current_w1,
                w2,
                weights[:tokens],
                routes[:tokens],
                s1,
                s2,
                a1_scale=xs[:tokens],
                max_tokens=sms * 64,
                ctas_per_sm=limit,
            )
            assert ws.workers <= limit * sms
            monkeypatch.setattr(harness, "resident_blocks", forbidden_query)
            y = megamoe(
                x[:tokens],
                current_w1,
                w2,
                weights[:tokens],
                routes[:tokens],
                s1,
                s2,
                workspace=ws,
                a1_scale=xs[:tokens],
            )
            expected = reference(
                x[:tokens],
                xs[:tokens],
                w1,
                w2,
                s1,
                s2,
                routes[:tokens],
                weights[:tokens],
            ).to(y.device)
            assert_moe_output(y, expected)
        monkeypatch.setattr(harness, "resident_blocks", query)
        ws = workspace(
            x, w1, w2, weights, routes, s1, s2, a1_scale=xs, max_tokens=sms * 64
        )
        monkeypatch.setattr(harness, "resident_blocks", forbidden_query)
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
            torch.cuda.synchronize()
        assert (
            len(
                [
                    event
                    for event in prof.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                ]
            )
            == 1
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
        for expert in (2, 0, 1):
            routes[:, 0] = expert
            routes[:, 1] = -1
            graph.replay()
            expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
            assert_moe_output(out, expected)

    def test_zero_rows_out_and_unsupported_input(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
        x.zero_()
        out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        result = megamoe(
            x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
        )
        assert result is out
        assert torch.count_nonzero(out) == 0
        with pytest.raises(ValueError, match="a1_scale"):
            megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws)
        with pytest.raises(NotImplementedError):
            megamoe(x.float(), w1, w2, weights, routes, s1, s2, workspace=ws)

    def test_public_backend_exports(workspace):
        import flaggems_vllm
        from flaggems_vllm.runtime.backend._metax.ops.megamoe import (
            megamoe as backend_entry,
        )

        assert flaggems_vllm.megamoe is backend_entry
        assert "workspace" in inspect.signature(backend_entry).parameters
        assert not hasattr(flaggems_vllm, "create_megamoe_workspace")
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(3, 128, 64, 4, 2)
        ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        y = flaggems_vllm.megamoe(
            x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs
        )
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(y.device)
        assert_moe_output(y, expected)

    def test_batched_partials_and_empty_graph(workspace):
        # A small resident grid forces several ring generations on modest tensors.
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(2049, 80, 70, 3, 2)
        x = x.T.contiguous().T
        w1 = w1.transpose(1, 2).contiguous().transpose(1, 2)
        w2 = w2.transpose(1, 2).contiguous().transpose(1, 2)
        routes = routes.T.contiguous().T
        weights = weights.T.contiguous().T
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        ws = workspace(
            x,
            w1,
            w2,
            weights,
            routes,
            s1,
            s2,
            a1_scale=xs,
            max_tokens=max(2049, sms * 32),
            workers=8,
            local_output=True,
        )
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
        out = torch.empty_like(x, dtype=torch.bfloat16)

        def run():
            megamoe(
                x,
                w1,
                w2,
                weights,
                routes,
                s1,
                s2,
                workspace=ws,
                a1_scale=xs,
                out=out,
            )

        run()
        assert_moe_output(out, expected)
        assert bool((ws.buffers[0][24:28] > 0).all()), "pipeline was not exercised"
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        saved_routes = routes.clone()
        for empty in (True, False, True, False):
            routes.copy_(torch.full_like(routes, -1) if empty else saved_routes)
            graph.replay()
            target = torch.zeros_like(expected) if empty else expected
            assert_moe_output(out, target)

    def test_workspace_reuse_without_caching_inputs(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 80, 70, 4, 2)
        ws = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        pointers = tuple(buffer.data_ptr() for buffer in ws.buffers)
        y = megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)
        changed_w1 = w1.neg()
        changed_routes = routes[:5].flip(1)
        actual = megamoe(
            x[:5],
            changed_w1,
            w2,
            weights[:5],
            changed_routes,
            s1,
            s2,
            workspace=ws,
            a1_scale=xs[:5],
        )
        expected = reference(
            x[:5], xs[:5], changed_w1, w2, s1, s2, changed_routes, weights[:5]
        ).to(x.device)
        assert_moe_output(actual, expected)
        assert pointers == tuple(buffer.data_ptr() for buffer in ws.buffers)
        assert y.shape == x.shape

    def test_workspace_is_explicit_and_caller_owned(workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(3, 128, 64, 4, 2)
        first = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        second = workspace(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        assert first.buffers[0].data_ptr() != second.buffers[0].data_ptr()
        with pytest.raises(FrozenInstanceError):
            first.workers = 1
        with pytest.raises(TypeError, match="workspace"):
            megamoe(x, w1, w2, weights, routes, s1, s2, a1_scale=xs)
        for ws in (first, second, first):
            actual = megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs
            )
            expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
            assert_moe_output(actual, expected)

    def test_workspace_capacity_and_fixed_launch(workspace, monkeypatch):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(8, 80, 70, 4, 2)
        ws = workspace(
            x[:7],
            w1,
            w2,
            weights[:7],
            routes[:7],
            s1,
            s2,
            a1_scale=xs[:7],
            max_tokens=7,
            workers=3,
            local_output=True,
        )
        harness = importlib.import_module("benchmark.test_megamoe")

        def forbidden_query(*args):
            raise AssertionError("occupancy queried during an operator call")

        monkeypatch.setattr(harness, "resident_blocks", forbidden_query)
        actual = megamoe(
            x[:7],
            w1,
            w2,
            weights[:7],
            routes[:7],
            s1,
            s2,
            workspace=ws,
            a1_scale=xs[:7],
        )
        expected = reference(x[:7], xs[:7], w1, w2, s1, s2, routes[:7], weights[:7]).to(
            x.device
        )
        assert_moe_output(actual, expected)
        assert ws.workers == 3
        with pytest.raises(ValueError, match="capacity"):
            megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs)

    @triton.jit
    def _quantization_probe(X, Q, S, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x = tl.load(X + row * WIDTH + cols, cols < WIDTH, other=0).to(tl.float32)
        q, scale = _symmetric_int8(x)
        tl.store(Q + row * WIDTH + cols, q, cols < WIDTH)
        tl.store(S + row, scale)

    @pytest.mark.parametrize("width", [9, 129])
    def test_formula_quantization_zero_ties_and_tail(width):
        cpu = torch.zeros((3, width), dtype=torch.bfloat16)
        cpu[1, :9] = torch.tensor([127, -127, 126.5, -126.5, 0.5, -0.5, 1.5, -1.5, 0])
        cpu[2, :4] = torch.tensor([1.828125, 0.9140625, -1.828125, -0.9140625])
        values = cpu.float()
        amax = values.abs().amax(1)
        expected_scale = torch.where(
            amax > 0, torch.div(amax, torch.tensor(127.0)), 1.0
        )
        expected_q = (
            torch.div(values, expected_scale[:, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )
        x = cpu.cuda()
        q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        scale = torch.empty((3,), device=x.device, dtype=torch.float32)
        _quantization_probe[(3,)](x, q, scale, width, triton.next_power_of_2(width))
        torch.testing.assert_close(q.cpu(), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)

    @pytest.mark.parametrize("workers", [1, 8])
    def test_rank_reduction_replay(workspace, workers):
        # A single producer still exercises both the serial and pipelined protocol.
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(17, 80, 70, 5, 3)
        ws = workspace(
            x,
            w1,
            w2,
            weights,
            routes,
            s1,
            s2,
            a1_scale=xs,
            max_tokens=33,
            workers=workers,
            rank_reduce=True,
        )
        out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
        for expert in (4, -1, 0, 3, -1, 1):
            routes.fill_(-1)
            routes[:, 0] = expert
            x.neg_()
            for _ in range(5):
                graph.replay()
            expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
            assert_moe_output(out, expected)
            if expert == -1:
                assert torch.count_nonzero(out) == 0
        for tokens in (0, 3, 17):
            y = megamoe(
                x[:tokens],
                w1,
                w2,
                weights[:tokens],
                routes[:tokens],
                s1,
                s2,
                workspace=ws,
                a1_scale=xs[:tokens],
            )
            expected = reference(
                x[:tokens],
                xs[:tokens],
                w1,
                w2,
                s1,
                s2,
                routes[:tokens],
                weights[:tokens],
            ).to(x.device)
            assert_moe_output(y, expected)

    def test_offset_peer_allocation(workspace):
        # Four-byte aligned peers are legal, but must not take the 32-byte IO path.
        storage = []

        def allocate(nbytes):
            allocation = torch.full(
                (nbytes + 64,), 0x5A, device="cuda", dtype=torch.uint8
            )
            target = allocation[4 : 4 + nbytes]
            target.zero_()
            storage.append((allocation, nbytes))
            return torch.tensor([target.data_ptr()], device="cuda", dtype=torch.int64)

        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(7, 128, 64, 4, 2)
        ws = workspace(
            x, w1, w2, weights, routes, s1, s2, a1_scale=xs, allocate_symmetric=allocate
        )
        assert ws.peer_alignment == 1
        out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
        routes[:, 1] = -1
        graph.replay()
        expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
        assert_moe_output(out, expected)
        for allocation, nbytes in storage:
            assert bool((allocation[:4] == 0x5A).all())
            assert bool((allocation[4 + nbytes :] == 0x5A).all())

    @pytest.mark.parametrize(
        "t,h,i,e,k,capacity",
        [
            (193, 640, 192, 8, 3, 257),
            (257, 1152, 384, 12, 4, 512),
            (129, 1536, 384, 16, 4, 768),
        ],
    )
    def test_resource_schedule_column_tail_graph(t, h, i, e, k, capacity, workspace):
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(t, h, i, e, k)
        ws = workspace(
            x,
            w1,
            w2,
            weights,
            routes,
            s1,
            s2,
            a1_scale=xs,
            max_tokens=capacity,
            workers=8,
            rank_reduce=True,
        )
        out = torch.empty_like(x, dtype=torch.bfloat16)
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
        for expert in (0, e - 1, -1, 2):
            routes.fill_(-1)
            routes[:, 0] = expert
            x.neg_()
            graph.replay()
            expected = reference(x, xs, w1, w2, s1, s2, routes, weights).to(x.device)
            assert_moe_output(out, expected)
            assert torch.isfinite(out).all()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            graph.replay()
            torch.cuda.synchronize()
        events = [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(events) == 1

    def test_resource_schedule_unroll_remainders(workspace):
        test_resource_schedule_column_tail_graph(129, 1408, 640, 16, 4, 768, workspace)

    @pytest.mark.parametrize("workers", [1, 2])
    def test_local_output_worker_boundary_graph(workspace, workers):
        # Twelve M16 blocks exceed both the pipeline threshold and one scratch batch.
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(96, 128, 64, 4, 2)
        routes[:, 0] = 0
        routes[:, 1] = 1
        ws = workspace(
            x,
            w1,
            w2,
            weights,
            routes,
            s1,
            s2,
            a1_scale=xs,
            workers=workers,
            local_output=True,
            rank_reduce=False,
            block_m=16,
        )
        assert ws.workers == workers and ws.local_output and not ws.rank_reduce
        out = torch.empty_like(x, dtype=torch.bfloat16)

        def check(actual, current_x, current_xs, current_routes, current_weights):
            expected = reference(
                current_x,
                current_xs,
                w1,
                w2,
                s1,
                s2,
                current_routes,
                current_weights,
            ).to(x.device)
            assert_moe_output(actual, expected)
            assert torch.isfinite(actual).all()

        ws.expert_partials.fill_(float("nan"))
        megamoe(x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out)
        check(out, x, xs, routes, weights)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )
        for first, second in ((0, 1), (2, 3), (-1, -1), (1, 0)):
            routes[:, 0] = first
            routes[:, 1] = second
            x.neg_()
            ws.expert_partials.fill_(float("nan"))
            graph.replay()
            check(out, x, xs, routes, weights)
        for tokens in (0, 16, 32, 96):
            ws.expert_partials.fill_(float("nan"))
            actual = megamoe(
                x[:tokens],
                w1,
                w2,
                weights[:tokens],
                routes[:tokens],
                s1,
                s2,
                workspace=ws,
                a1_scale=xs[:tokens],
            )
            check(actual, x[:tokens], xs[:tokens], routes[:tokens], weights[:tokens])
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            graph.replay()
            torch.cuda.synchronize()
        assert (
            len(
                [
                    e
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
            )
            == 1
        )

    @triton.jit
    def _return_packet_probe(
        OUT,
        WORLD: tl.constexpr,
        CAPACITY: tl.constexpr,
        STRIPED: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        task = tl.arange(0, BLOCK)
        packet = _return_packet(task, WORLD, CAPACITY, STRIPED)
        tl.store(OUT + task, packet, task < triton.cdiv(WORLD * CAPACITY, 16))

    @pytest.mark.parametrize(
        "world,capacity,striped",
        [(2, 32, True), (4, 64, True), (8, 16, True), (3, 33, False)],
    )
    def test_return_packet_order(world, capacity, striped):
        packets = triton.cdiv(world * capacity, 16)
        output = torch.empty((packets,), device="cuda", dtype=torch.int32)
        _return_packet_probe[(1,)](
            output, world, capacity, striped, triton.next_power_of_2(packets)
        )
        if striped:
            expected = [
                rank * (capacity // 16) + packet
                for packet in range(capacity // 16)
                for rank in range(world)
            ]
        else:
            expected = list(range(packets))
        actual = output.cpu().tolist()
        assert actual == expected
        assert sorted(actual) == list(range(packets))

    def test_output_accuracy_policy_rejects_invalid_results():
        expected = torch.ones((4, 4), dtype=torch.bfloat16)
        assert_moe_output(expected.clone(), expected)
        assert_moe_output(expected + 1 / 256, expected)
        for value in (
            expected + 1 / 128,
            expected * float("nan"),
            expected * float("inf"),
        ):
            with pytest.raises(AssertionError):
                assert_moe_output(value, expected)
        zero = torch.zeros_like(expected)
        assert_moe_output(zero, zero)
        with pytest.raises(AssertionError):
            assert_moe_output(torch.full_like(zero, 1e-30), zero)
        empty = expected[:0]
        assert_moe_output(empty, empty)
        with pytest.raises(AssertionError):
            assert_moe_output(expected[:1], expected)
        with pytest.raises(AssertionError):
            assert_moe_output(expected.float(), expected)

    @pytest.mark.parametrize("rank_reduce", [False, True])
    def test_strict_formula_and_dispatch_graph(workspace, shmem, rank_reduce):
        # Run the migrated full53 oracle on fresh data, not recorded PASS summaries.
        x, xs, w1, w2, s1, s2, routes, weights = make_inputs(17, 128, 128, 4, 2)
        ws = workspace(
            x,
            w1,
            w2,
            weights,
            routes,
            s1,
            s2,
            a1_scale=xs,
            workers=8,
            rank_reduce=rank_reduce,
        )
        out = torch.empty_like(x, dtype=torch.bfloat16)

        def run():
            megamoe(
                x, w1, w2, weights, routes, s1, s2, workspace=ws, a1_scale=xs, out=out
            )

        def check(phase):
            expected = reference(x, xs, w1, w2, s1, s2, routes, weights)
            assert_moe_output(out, expected)
            shared = {
                "input_int8": x.cpu().unsqueeze(0),
                "input_scales": xs.cpu().unsqueeze(0),
                "topk_idx": routes.cpu().unsqueeze(0),
                "topk_weights": weights.cpu().unsqueeze(0),
            }
            assert validate(
                ws,
                w1,
                w2,
                s1,
                s2,
                out,
                routes,
                {"output": expected},
                shared,
                shmem,
                True,
                phase,
            )

        run()
        check("eager")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.neg_()
        routes[:, 0] = 3
        routes[:, 1] = -1
        graph.replay()
        check("graph_changed_inputs_and_routes")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            graph.replay()
            torch.cuda.synchronize()
        events = [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(events) == 1

    def test_dispatch_scales_with_work_and_resources():
        plans = [
            plan_workspace(
                capacity=1024 * scale,
                top_k=4,
                local_experts=16 * scale,
                hidden=768,
                intermediate=256,
                world_size=2,
                multiprocessors=32 * scale,
            )
            for scale in (1, 2, 3, 4, 7)
        ]
        first = plans[0]
        for scale, plan in zip((1, 2, 3, 4, 7), plans):
            assert (plan.bm, plan.bn, plan.bk, plan.stages) == (
                first.bm,
                first.bn,
                first.bk,
                first.stages,
            )
            assert (plan.local_output, plan.rank_reduce) == (
                first.local_output,
                first.rank_reduce,
            )
            assert plan.workers == first.workers * scale

    def test_dispatch_preserves_explicit_configuration():
        arguments = dict(
            capacity=257,
            top_k=3,
            local_experts=7,
            hidden=640,
            intermediate=192,
            world_size=2,
            multiprocessors=13,
            workers=3,
            local_output=False,
            rank_reduce=False,
            block_m=32,
            block_n=64,
            block_k=64,
            stages=3,
        )
        plan = plan_workspace(**arguments)
        assert (plan.bm, plan.bn, plan.bk, plan.stages, plan.workers) == (
            32,
            64,
            64,
            3,
            3,
        )
        assert not plan.local_output
        assert not plan.rank_reduce
        with pytest.raises(ValueError, match="positive CTA budget"):
            plan_workspace(**dict(arguments, workers=0))

    @pytest.fixture
    def compare_arguments(tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        (data / "manifest.json").write_text(
            json.dumps(
                {
                    "shape": {
                        "hidden_size": 2048,
                        "moe_intermediate_size": 512,
                        "num_experts": 256,
                        "num_experts_per_tok": 8,
                    }
                }
            )
        )
        for name in ("host.so", "sync.bc", "shmem.bc"):
            (tmp_path / name).write_bytes(b"planning-only")
        monkeypatch.setenv("MACA_PATH", "/opt/maca")
        monkeypatch.setenv("MEGAMOE_DEVICE_LIB", str(tmp_path / "sync.bc"))
        monkeypatch.setenv("MXSHMEM_DEVICE_LIB", str(tmp_path / "shmem.bc"))
        return [
            "--data",
            str(data),
            "--model",
            str(tmp_path / "model"),
            "--host-lib",
            str(tmp_path / "host.so"),
            "--devices",
            "0,1,2,3,4,5,6,7",
            "--output",
            str(tmp_path / "results"),
        ]

    def test_official_benchmark_plans_full_graph_matrix(compare_arguments):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark.test_megamoe",
                *compare_arguments,
                "--dry-run",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        output = Path(compare_arguments[-1])
        manifest = json.loads((output / "manifest.json").read_text())
        plan = json.loads((output / "launch-plan.json").read_text())
        assert manifest["full_53_shape_ep248"]
        assert manifest["modes"] == ["graph"]
        assert (manifest["warmup"], manifest["iterations"], manifest["repeats"]) == (
            20,
            100,
            5,
        )
        assert len(manifest["tokens"]) == 53
        assert [(v["ep"], v["role"]) for v in plan["launches"]] == [
            (ep, role) for ep in (2, 4, 8) for role in ("baseline", "candidate")
        ]
        for launch in plan["launches"]:
            command = launch["command"]
            assert "--graph-profile" in command
            assert "artifacts/" not in " ".join(command)
            worker = (
                "graph_baseline.py"
                if launch["role"] == "baseline"
                else "graph_candidate.py"
            )
            assert command.count("--worker") == 1
            assert command[command.index("--worker") + 1] == worker.removesuffix(
                ".py"
            ).replace("_", "-")
            assert "--rank-launcher" in command
            assert any(value.endswith("benchmark/test_megamoe.py") for value in command)
            if launch["role"] == "candidate":
                assert "--check-formula" in command
        assert plan["dry_run"] and not (output / "summary.json").exists()

    @pytest.mark.parametrize(
        "extra",
        [
            ["--devices", "0,0,1,2,3,4,5,6"],
            ["--devices", "0,1"],
            ["--tokens", "3"],
        ],
    )
    def test_matrix_rejects_invalid_selection(compare_arguments, extra):
        with pytest.raises(SystemExit) as error:
            benchmark_module.main([*compare_arguments, *extra, "--dry-run"])
        assert error.value.code == 2

    @pytest.mark.parametrize("bad_names", [[], ["_megamoe", "copy"], ["gemm"]])
    def test_profiles_require_one_complete_megamoe_kernel(tmp_path, bad_names):
        for role in ("baseline", "candidate"):
            folder = tmp_path / f"audit-{role}"
            folder.mkdir()
            for rank in range(2):
                record = {
                    "tokens": 1,
                    "profile_kernel_names": (
                        ["native"] if role == "baseline" else ["_megamoe"]
                    ),
                }
                (folder / f"np2-rank{rank}.jsonl").write_text(
                    (json.dumps(record) + "\n") * (5 if role == "baseline" else 1)
                )
        benchmark_module.validate_kernel_profiles(tmp_path, 2, [1])
        (tmp_path / "audit-candidate/np2-rank1.jsonl").write_text(
            json.dumps({"tokens": 1, "profile_kernel_names": bad_names}) + "\n"
        )
        with pytest.raises(AssertionError):
            benchmark_module.validate_kernel_profiles(tmp_path, 2, [1])

    def _paired_benchmark(monkeypatch, tmp_path, complete):
        bench = importlib.import_module("benchmark.test_megamoe")
        monkeypatch.setattr(
            bench,
            "Config",
            SimpleNamespace(
                mode=bench.BenchMode.KERNEL,
                warm_up=bench.DEFAULT_WARMUP_TIME,
                repetition=bench.DEFAULT_ITER_TIME,
            ),
        )
        for key in ("MEGAMOE_DATA_DIR", "MEGAMOE_MODEL_DIR", "MEGAMOE_HOST_LIB"):
            monkeypatch.setenv(key, str(tmp_path))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        monkeypatch.setenv("MEGAMOE_BENCHMARK_OUTPUT", str(tmp_path / "results"))
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            report = {"complete": complete, "results": {"graph": {}}}
            for ep in (2, 4, 8):
                report["results"]["graph"][str(ep)] = {
                    "complete": complete,
                    "rows": [
                        {
                            "tokens_per_rank": 1,
                            "vllm_ms": 2.0,
                            "megamoe_ms": 1.0,
                            "correctness": True,
                            "baseline_valid": True,
                            "timing_valid": True,
                        }
                    ],
                }
            (output / "summary.json").write_text(json.dumps(report))

        monkeypatch.setattr(bench.subprocess, "run", execute)
        instance = bench.MegaMoEBenchmark()
        instance.shapes = [(1, 2048, 512, 256, 8, ep) for ep in (2, 4, 8)]
        return bench, instance, calls

    def test_framework_reports_distinct_sides_from_one_pair(monkeypatch, tmp_path):
        bench, instance, calls = _paired_benchmark(monkeypatch, tmp_path, True)
        for ep in (2, 4, 8):
            shape = (1, 2048, 512, 256, 8, ep)
            assert instance.get_latency(bench._baseline_latency, *shape) == 2.0
            assert instance.get_latency(bench._candidate_latency, *shape) == 1.0
        assert len(calls) == 1

    def test_framework_rejects_incomplete_matrix(monkeypatch, tmp_path):
        bench, instance, calls = _paired_benchmark(monkeypatch, tmp_path, False)
        with pytest.raises(RuntimeError, match="Incomplete paired benchmark"):
            instance.get_latency(bench._candidate_latency, 1, 2048, 512, 256, 8, 2)
        assert len(calls) == 1

    def test_summary_outputs_are_written_atomically(tmp_path):
        row = dict(
            tokens_per_rank=1,
            vllm_ms=2.0,
            megamoe_ms=1.0,
            speedup=2.0,
            correctness=True,
            baseline_valid=True,
            timing_valid=True,
        )
        result = {"ranks": 2, "complete": True, "rows": [row]}
        benchmark_module._summary_write_report(tmp_path, result)
        assert json.loads((tmp_path / "summary-np2.json").read_text()) == result
        csv = (tmp_path / "summary-np2.csv").read_text()
        assert "tokens_per_rank,vllm_ms,megamoe_ms,speedup" in csv
        assert "1,2.0,1.0,2.0,True,True,True" in csv
        assert not list(tmp_path.glob("*.tmp"))

else:
    import os
    import subprocess
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    _IS_ACCURACY_WORKER = "--accuracy-worker" in sys.argv
    if _IS_ACCURACY_WORKER:
        _local_rank = os.environ.get("LOCAL_RANK", "0")
        _cache_root = Path(
            os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")
        )
        os.environ["TRITON_CACHE_DIR"] = str(
            _cache_root.parent / f"{_cache_root.name}-megamoe-test-rank-{_local_rank}"
        )

    import pytest  # noqa: E402
    import torch  # noqa: E402

    if _IS_ACCURACY_WORKER:
        torch.cuda.set_device(int(_local_rank))

    # The worker cache and CUDA device must be selected before these imports.
    # isort: off
    import flaggems_vllm  # noqa: E402
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (  # noqa: E402
        megamoe as hopper_megamoe,
    )
    from flaggems_vllm.testing.megamoe import StandaloneNvshmem  # noqa: E402
    from flaggems_vllm.testing.megamoe import (  # noqa: E402
        configure_descriptor_allocator,
        interleave_l1_gate_up_rows,
        make_context,
        pool_tokens,
    )
    from flaggems_vllm.utils.triton_version_utils import has_triton_tle  # noqa: E402

    # isort: on

    pytestmark = pytest.mark.megamoe

    def _is_hopper():
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9

    @pytest.mark.skipif(
        not (_is_hopper() and has_triton_tle(3, 6, 0)),
        reason="requires Hopper with FlagTree TLE",
    )
    def test_megamoe_fp8_is_exported_on_hopper():
        assert callable(hopper_megamoe.megamoe_fp8)
        assert flaggems_vllm.megamoe_fp8 is hopper_megamoe.megamoe_fp8

    @dataclass(frozen=True)
    class _AccuracyConfig:
        num_ranks: int = 2
        tokens: int = 2
        hidden: int = 128
        intermediate: int = 128
        num_experts: int = 2
        topk: int = 1
        stages: int = 2
        max_recv: int = 128

    def _make_accuracy_inputs(config: _AccuracyConfig, rank: int, device):
        generator = torch.Generator().manual_seed(20261105)
        fp8 = torch.float8_e4m3fn

        inputs = torch.randn(
            config.num_ranks,
            config.tokens,
            config.hidden,
            generator=generator,
        ).mul_(0.25)
        input_scales = (
            torch.rand(
                config.num_ranks,
                config.tokens,
                config.hidden // 128,
                generator=generator,
            )
            .mul_(0.25)
            .add_(0.75)
        )
        w1_logical = torch.randn(
            config.num_experts,
            2 * config.intermediate,
            config.hidden,
            generator=generator,
        ).mul_(0.125)
        w1_scales = (
            torch.rand(
                config.num_experts,
                config.intermediate // 64,
                config.hidden // 128,
                generator=generator,
            )
            .mul_(0.25)
            .add_(0.75)
        )
        w2 = torch.randn(
            config.num_experts,
            config.hidden,
            config.intermediate,
            generator=generator,
        ).mul_(0.125)
        w2_scales = (
            torch.rand(
                config.num_experts,
                config.hidden // 128,
                config.intermediate // 128,
                generator=generator,
            )
            .mul_(0.25)
            .add_(0.75)
        )

        inputs = inputs.to(device=device, dtype=fp8)
        input_scales = input_scales.to(device)
        w1_logical = w1_logical.to(device=device, dtype=fp8)
        w1_scales = w1_scales.to(device)
        w2 = w2.to(device=device, dtype=fp8)
        w2_scales = w2_scales.to(device)
        w1_packed = interleave_l1_gate_up_rows(w1_logical)

        topk_all = torch.empty(
            config.num_ranks,
            config.tokens,
            config.topk,
            dtype=torch.int64,
        )
        for source_rank in range(config.num_ranks):
            topk_all[source_rank].fill_(1 - source_rank)
        topk_weights = torch.tensor(
            [[0.75], [1.25]], dtype=torch.float32, device=device
        )

        local_slice = slice(rank, rank + 1)
        operands = (
            inputs[rank].contiguous(),
            input_scales[rank].contiguous(),
            topk_all[rank].to(device),
            topk_weights,
            w1_packed[local_slice].contiguous(),
            w1_scales[local_slice].contiguous(),
            w2[local_slice].contiguous(),
            w2_scales[local_slice].contiguous(),
        )
        reference_inputs = (
            inputs[rank],
            input_scales[rank],
            topk_all[rank].to(device),
            topk_weights,
            w1_logical,
            w1_scales,
            w2,
            w2_scales,
        )
        return operands, reference_inputs, topk_all

    def _reference_megamoe(
        input_fp8,
        input_scales,
        topk_ids,
        topk_weights,
        w1,
        w1_scales,
        w2,
        w2_scales,
    ):
        tokens, hidden = input_fp8.shape
        intermediate = w2.shape[-1]
        output = torch.zeros(
            tokens, hidden, dtype=torch.float32, device=input_fp8.device
        )

        for token in range(tokens):
            for route in range(topk_ids.shape[1]):
                expert = int(topk_ids[token, route])
                l1 = torch.zeros(
                    2 * intermediate, dtype=torch.float32, device=input_fp8.device
                )
                for n_block in range((2 * intermediate) // 128):
                    n_start = n_block * 128
                    for k_block in range(hidden // 128):
                        k_start = k_block * 128
                        product = (
                            input_fp8[token, k_start : k_start + 128].float()
                            @ w1[
                                expert,
                                n_start : n_start + 128,
                                k_start : k_start + 128,
                            ]
                            .float()
                            .T
                        )
                        l1[n_start : n_start + 128] += product * (
                            input_scales[token, k_block]
                            * w1_scales[expert, n_block, k_block]
                        )

                gate, up = l1.chunk(2)
                swiglu = gate / (1.0 + torch.exp(-gate)) * up
                route_weight = topk_weights[token, route]
                segments = swiglu.reshape(intermediate // 64, 64)
                amax = segments.abs().amax(dim=1) * route_weight.abs()
                scaled = amax / 448.0
                segment_scales = torch.where(
                    scaled > 0,
                    torch.exp2(torch.ceil(torch.log2(scaled))),
                    torch.ones_like(scaled),
                )
                quantized = (segments * (route_weight / segment_scales)[:, None]).to(
                    torch.float8_e4m3fn
                )

                partial = torch.zeros(
                    hidden, dtype=torch.float32, device=input_fp8.device
                )
                for n_block in range(hidden // 128):
                    n_start = n_block * 128
                    for k_block in range(intermediate // 128):
                        k_start = k_block * 128
                        weight_scale = w2_scales[expert, n_block, k_block]
                        low = (
                            quantized[2 * k_block].float()
                            @ w2[
                                expert,
                                n_start : n_start + 128,
                                k_start : k_start + 64,
                            ]
                            .float()
                            .T
                        )
                        high = (
                            quantized[2 * k_block + 1].float()
                            @ w2[
                                expert,
                                n_start : n_start + 128,
                                k_start + 64 : k_start + 128,
                            ]
                            .float()
                            .T
                        )
                        partial[n_start : n_start + 128] += weight_scale * (
                            low * segment_scales[2 * k_block]
                            + high * segment_scales[2 * k_block + 1]
                        )
                output[token] += partial.to(torch.bfloat16).float()

        return output.to(torch.bfloat16)

    def _run_accuracy_worker() -> None:
        config = _AccuracyConfig()
        with StandaloneNvshmem() as runtime:
            assert runtime.world_size == config.num_ranks
            configure_descriptor_allocator(runtime.device)
            operands, reference_inputs, topk_all = _make_accuracy_inputs(
                config, runtime.rank, runtime.device
            )
            pool_size = pool_tokens(topk_all, config)
            context = make_context(config, runtime, pool_size)
            output = hopper_megamoe.megamoe_fp8(
                *operands,
                context,
                stages=config.stages,
            )
            torch.cuda.synchronize(runtime.device)
            reference = _reference_megamoe(*reference_inputs)

            relative_l2 = (
                output.float() - reference.float()
            ).norm() / reference.float().norm().clamp_min(1e-12)
            cosine = torch.nn.functional.cosine_similarity(
                output.float().flatten(), reference.float().flatten(), dim=0
            )
            assert float(relative_l2) < 0.02, relative_l2
            assert float(cosine) > 0.999, cosine

    @pytest.mark.skipif(
        not (_is_hopper() and has_triton_tle(3, 6, 0)),
        reason="requires Hopper with FlagTree TLE",
    )
    def test_megamoe_fp8_accuracy():
        if torch.cuda.device_count() < 2:
            pytest.skip("requires two visible Hopper GPUs")

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--rdzv-backend",
            "c10d",
            "--rdzv-endpoint",
            "127.0.0.1:0",
            "--nproc-per-node",
            "2",
            str(Path(__file__).resolve()),
            "--accuracy-worker",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode:
            pytest.fail(completed.stdout + completed.stderr)

    if __name__ == "__main__" and _IS_ACCURACY_WORKER:
        _run_accuracy_worker()
