# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in full BF16 shape validation, independent of benchmark timing.

MTHREADS_FA_SHAPE_SET=full python -m pytest -q \
    tests/test_flash_attn_varlen_mthreads_full.py

Use tp1-full/tp2-full/tp4-full, or MTHREADS_FA_CASE_IDS, to select a subset. The native
reference package is validation-only. Production kernels are never replaced.
"""
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import flaggems_vllm
from benchmark.qwen36_flash_attn_cases import load_cases
from benchmark.test_flash_attn_varlen_func import (
    FlashAttnVarlenBenchmark,
    flash_attn_varlen_mthreads,
)

SHAPE_SET = os.environ.get("MTHREADS_FA_SHAPE_SET", "default")
CASES = load_cases(SHAPE_SET) if SHAPE_SET != "default" else []
pytestmark = [
    pytest.mark.flash_attn_varlen_func,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "mthreads", reason="MThreads only"),
]


def compare(actual, expected, atol, rtol):
    a = actual.detach().float().cpu().numpy().reshape(-1)
    b = expected.detach().float().cpu().numpy().reshape(-1)
    assert a.shape == b.shape
    bad, maximum = 0, 0.0
    for offset in range(0, len(a), 1_000_000):
        x, y = a[offset : offset + 1_000_000], b[offset : offset + 1_000_000]
        finite = np.isfinite(x) & np.isfinite(y)
        difference = np.abs(x[finite] - y[finite])
        maximum = max(maximum, float(difference.max(initial=0)))
        bad += int(np.count_nonzero(difference > atol + rtol * np.abs(y[finite])))
        bad += int(np.count_nonzero(~finite & (x != y)))
    assert bad == 0, dict(bad=bad, elements=len(a), maximum_abs=maximum)
    return dict(elements=len(a), maximum_abs=maximum, bad=bad)


def compile_resources(args, kwargs, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
        compile_only,
    )

    with compile_only() as compiled:
        flaggems_vllm.flash_attn_varlen_func(*args, **kwargs)
    resources = []
    for kernel in compiled:
        kernel._init_handles()
        resources.append(
            dict(
                name=kernel.name,
                registers=kernel.n_regs,
                spills=kernel.n_spills,
                shared=kernel.metadata.shared,
            )
        )
    assert resources, "No production Triton kernels observed"
    return resources


def native_reference(args, kwargs):
    expected = tuple(
        value.detach().cpu().clone()
        for value in flash_attn_varlen_mthreads(*args, **kwargs)
    )
    cuq, usedk = args[4].cpu().tolist(), args[7].cpu().tolist()
    for begin, end, klen in zip(cuq, cuq[1:], usedk):
        empty = (
            max(0, end - begin - klen) if args[11] else (end - begin if not klen else 0)
        )
        if empty:
            assert torch.isinf(expected[1][:, begin : begin + empty]).all()
            assert (expected[0][begin : begin + empty] == 0).all()
            expected[1][:, begin : begin + empty] = float("inf")
    return expected


@pytest.mark.parametrize("case", CASES, ids=[case["case_id"] for case in CASES])
def test_full_shape(case, monkeypatch):
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention import autotune
    from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.execution import (
        capture_call,
        use_config,
    )

    started = time.monotonic()
    record = dict(case_id=case["case_id"], tp=case["tp"], category=case["category"])
    folder = os.environ.get("MTHREADS_FA_VALIDATION_OUT")
    output_dir = Path(folder) if folder else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    def phase(value):
        record["phase"] = value
        if output_dir:
            (output_dir / "progress.json").write_text(json.dumps(record, default=str))

    try:
        phase("input")
        bench = FlashAttnVarlenBenchmark(
            "flash_attn_varlen_func", flash_attn_varlen_mthreads
        )
        packed = bench.trace_input_fn(case, torch.bfloat16, "musa")
        args, kwargs = list(packed[:-1]), packed[-1]
        q, k, v = args[:3]
        assert list(q.shape) == case["q_shape"]
        assert list(k.shape) == case["k_shape"] == list(v.shape)
        assert list(args[17].shape) == case["block_table_shape"]
        assert args[3] == case["max_seqlen_q"]
        assert args[5] == case["max_seqlen_k"]
        assert kwargs["num_splits"] == case["num_splits"] == 0
        if case["tp"] == "TP2":
            assert k.stride() == v.stride() == (16384, 256, 256, 1)
            assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
            assert k.storage_offset() == 0 and v.storage_offset() == 8192
        args[18] = True
        phase("resources")
        with capture_call() as call_info:
            record["resources"] = compile_resources(args, kwargs, monkeypatch)
        record["seed_resources"] = record["resources"]
        if any(row["spills"] >= 40 for row in record["resources"]):
            from flaggems_vllm.runtime import get_tuned_config
            from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.policy import (
                candidates,
            )

            qualified = []
            for candidate in candidates(
                call_info["problem"],
                call_info["default"],
                get_tuned_config("flash_attn_varlen_mthreads"),
            ):
                try:
                    resources = autotune.compile_candidate(
                        lambda: flaggems_vllm.flash_attn_varlen_func(*args, **kwargs),
                        candidate,
                    )
                except autotune.RESOURCE_ERRORS:
                    continue
                qualified.append(resources)
            assert qualified, record["seed_resources"]
        phase("native_reference")
        expected = native_reference(args, kwargs)
        # Keep the existing +inf LSE contract for causal all-masked rows.
        empty_rows = 0
        for begin, end, klen in zip(
            case["cu_seqlens_q"],
            case["cu_seqlens_q"][1:],
            case["seqused_k"],
        ):
            empty = max(0, end - begin - klen)
            if empty:
                assert torch.isinf(expected[1][:, begin : begin + empty]).all()
                assert (expected[0][begin : begin + empty] == 0).all()
                expected[1][:, begin : begin + empty] = float("inf")
                empty_rows += empty
        record["all_masked_rows"] = empty_rows
        phase("eager")

        def run():
            return flaggems_vllm.flash_attn_varlen_func(*args, **kwargs)

        actual = run()
        torch.musa.synchronize()
        selected = autotune._MEMORY.get(call_info.get("key"))
        record["tuned_config"] = (
            selected.values() if selected is not None else "single_kv_or_v16_seed"
        )
        if selected is not None:
            with use_config(selected):
                record["resources"] = compile_resources(args, kwargs, monkeypatch)
        assert all(row["spills"] < 40 for row in record["resources"]), record[
            "resources"
        ]
        assert actual[0] is args[19]
        record["O"] = compare(actual[0], expected[0], 0.02, 0.02)
        record["LSE"] = compare(actual[1], expected[1], 0.002, 0.002)
        phase("graph")
        stream = torch.musa.Stream()
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            run()
        torch.musa.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph, stream=stream):
            actual = run()
        for replay in range(2):
            q.add_(0.0625 if replay == 0 else -0.125)
            k.mul_(0.9375 if replay == 0 else 1.0625)
            v.add_(0.03125 if replay == 0 else -0.0625)
            lengths = case["seqused_k"]
            args[7].copy_(
                torch.tensor(
                    [max(0, n - 1) if replay == 0 else n for n in lengths],
                    dtype=args[7].dtype,
                    device=q.device,
                )
            )
            qlengths = [
                b - a for a, b in zip(case["cu_seqlens_q"], case["cu_seqlens_q"][1:])
            ]
            if replay == 0:
                qlengths.reverse()
            cumulative = [0]
            for length in qlengths:
                cumulative.append(cumulative[-1] + length)
            args[4].copy_(
                torch.tensor(cumulative, dtype=args[4].dtype, device=q.device)
            )
            if args[17].shape[1] > 1:
                args[17].copy_(args[17].roll(1, 1))
            expected = native_reference(args, kwargs)
            actual[0].fill_(float("nan"))
            actual[1].fill_(float("nan"))
            graph.replay()
            torch.musa.synchronize()
            compare(actual[0], expected[0], 0.02, 0.02)
            compare(actual[1], expected[1], 0.002, 0.002)
        record.update(
            status="passed", graph_replays=2, modified_qkv_metadata_graph_replays=2
        )
        if case["tp"] == "TP2":
            assert (bench._tp2_input_factory.output_guard == 13).all().item()
            record["modified_q_graph_replays"] = 2
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        if output_dir:
            with (output_dir / "cases.jsonl").open("a") as stream:
                stream.write(json.dumps(record, default=str) + "\n")
        print("FULL_SHAPE_RESULT", json.dumps(record, default=str), flush=True)
        # Failed pytest frames must not retain multi-GiB KV caches.
        packed = args = kwargs = q = k = v = actual = expected = graph = run = None
        bench = None
        stream = None
        gc.collect()
        torch.musa.empty_cache()
