# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib

import pytest
import torch
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend
from triton.runtime.jit import compute_cache_key, create_function_from_signature

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX regression")


@pytest.mark.parametrize(
    "module_name,kernel_name,parameter",
    [
        pytest.param(
            "flaggems_vllm.ops.top_k_per_row_prefill",
            "non_tle_top_k_per_row_prefill",
            "NUM_ROWS",
            marks=pytest.mark.top_k_per_row_prefill,
        )
    ],
)
def test_unused_batch_native_binding_reuses_specialization(
    module_name, kernel_name, parameter
):
    """Exercise the installed native binder, without launching a kernel."""
    kernel = getattr(importlib.import_module(module_name), kernel_name)
    backend = make_backend(GPUTarget("maca", 80, 64))
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    values = dict.fromkeys(kernel.arg_names, None)
    keys = set()
    for batch in (*range(1, 65), 1024, 4032, 4034, 4094, 4096):
        values[parameter] = batch
        _, specialization, options = binder(**values)
        keys.add(str(compute_cache_key({}, specialization, options)))
    assert len(keys) == 1


def _cache_keys(kernel):
    return set(kernel.device_caches[torch.cuda.current_device()][0])


@pytest.mark.top_k_per_row_prefill
def test_public_prefill_topk_new_rows_reuse_kernel_and_graph(record_property):
    module = importlib.import_module("flaggems_vllm.ops.top_k_per_row_prefill")
    kernel = module.non_tle_top_k_per_row_prefill
    assert ops.top_k_per_row_prefill.__module__.endswith(
        "_metax.ops.top_k_per_row_prefill"
    )
    width, topk = (4096, 512)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.cuda.manual_seed(104)
        warm_keys = None
        observed = []
        for rows in (1, 3, 7, 17, 31, 63):
            storage = torch.randn(rows, width + 64, device="cuda")
            logits = storage[:, :width]
            starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 8
            ends = torch.full((rows,), width, device="cuda", dtype=torch.int32)
            output = torch.full(
                (rows, topk + 8), -12345, device="cuda", dtype=torch.int32
            )

            def call():
                ops.top_k_per_row_prefill(
                    logits,
                    starts,
                    ends,
                    output[:, :topk],
                    rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk,
                )

            def check():
                cpu, ids, begin = (logits.cpu(), output.cpu(), starts.cpu())
                assert (ids[:, topk:] == -12345).all()
                for row in range(rows):
                    selected = ids[row, :topk].long() + int(begin[row])
                    assert selected.unique().numel() == topk
                    assert int(selected.min()) >= int(begin[row])
                    assert int(selected.max()) < width
                    expected = torch.topk(cpu[row, int(begin[row]) :], topk).values
                    torch.testing.assert_close(
                        cpu[row, selected].sort().values,
                        expected.sort().values,
                        atol=0,
                        rtol=0,
                    )

            call()
            torch.cuda.synchronize()
            check()
            if warm_keys is None:
                warm_keys = _cache_keys(kernel)
                assert warm_keys, "The public Top-k path did not hit its kernel"
            else:
                assert _cache_keys(kernel) == warm_keys
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for _ in range(2):
                logits.neg_()
                output[:, :topk].fill_(-999)
                graph.replay()
                torch.cuda.synchronize()
                check()
            assert _cache_keys(kernel) == warm_keys
            observed.append((rows, len(warm_keys)))
        record_property("native_cache_keys_by_rows", repr(observed))
