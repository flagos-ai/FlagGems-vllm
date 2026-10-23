# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Four-way GDN core comparison on MetaX; real shapes remain external.

Freeze the original implementation before comparing:
    git show 165c2aa:src/flaggems_vllm/ops/gdn_w8a16.py > /tmp/gdn_current.py
    GDN_CURRENT_IMPL=/tmp/gdn_current.py GDN_SHAPE_FILE=/path/to/shapes.json \
        python benchmark/test_gdn_w8a16_compare.py

GDN_COMPARE_GROUP selects all (default), decode, or prefill. The native
MetaX adapters disable QK normalization and do not modify the input state.
Recurrent returns per-token states: selection of final states is timed.
Set GDN_MEASURE_MEMORY=1 to record peak incremental live allocations after
warmup, including returned tensors (reported separately). Reserved allocator
cache is excluded. Chunk metadata is prepared before capture for the native API.
The optimized implementation constructs its chunk mapping on device.
"""

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch

from flaggems_vllm import gdn_w8a16


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_native(args, bounds, prefill):
    from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets
    from flaggems_vllm.runtime.backend._metax.fused.gdn_chunk import (
        chunk_gated_delta_rule_fwd,
    )
    from flaggems_vllm.runtime.backend._metax.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )

    q, k, v, g, beta, state, cu = args
    n, tokens = len(bounds) - 1, q.shape[1]
    scale = q.shape[-1] ** -0.5
    if prefill:
        indices = prepare_chunk_indices(cu, 64)
        offsets = prepare_chunk_offsets(cu, 64)

        def call():
            result = chunk_gated_delta_rule_fwd(
                q, k, v, g, beta, scale, state, True, cu, indices, offsets
            )
            return result[1], result[3]

    else:
        max_length = max(b - a for a, b in zip(bounds, bounds[1:]))
        indices = torch.arange(n, device=q.device, dtype=torch.int32)[:, None].repeat(
            1, max_length
        )
        last = torch.tensor(
            [end - 1 for end in bounds[1:]], device=q.device, dtype=torch.int64
        )

        def call():
            out, final = fused_recurrent_gated_delta_rule_fwd(
                q, k, v, g, beta, scale, state, False, cu, indices
            )
            return out, final if n == tokens else torch.index_select(final, 0, last)

    return call


def measure_peak_allocation(function):
    """Incremental live allocation after warmup; excludes cached/reserved bytes."""
    warm = function()
    torch.cuda.synchronize()
    del warm
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - allocated
    result_bytes = sum(x.numel() * x.element_size() for x in result)
    return dict(peak_bytes=peak, result_bytes=result_bytes)


def compare_case(row, current, bench):
    from functools import partial

    args, bounds = bench.make_inputs(row)
    before = args[5].clone()
    expected = bench.torch_baseline(args, bounds)
    functions = {
        "current": partial(current.gdn_w8a16, *args),
        "optimized": partial(gdn_w8a16, *args),
        "pytorch": partial(bench.torch_baseline, args, bounds),
        "metax": prepare_native(args, bounds, row["op"] == "chunk_gated_delta_rule"),
    }
    graphs, outputs, valid, errors = {}, {}, {}, {}
    samples = {name: [] for name in functions}
    for name, function in functions.items():
        actual = function()
        valid[name] = True
        errors[name] = [
            float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)
        ]
        graphs[name], outputs[name] = bench.capture(function)
        for result in (actual, outputs[name]):
            for a, b in zip(result, expected):
                try:
                    torch.testing.assert_close(
                        a.float(), b.float(), atol=0.002, rtol=0.02
                    )
                except AssertionError:
                    valid[name] = False
        torch.testing.assert_close(args[5], before, atol=0, rtol=0)
    names = list(graphs)
    for sample in range(31):
        order = names[sample % 4 :] + names[: sample % 4]
        if sample % 2:
            order.reverse()
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(3):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / 3)
    memory = {}
    if os.environ.get("GDN_MEASURE_MEMORY", "0") == "1":
        memory = {
            name: measure_peak_allocation(functions[name])
            for name in ("current", "optimized", "metax")
        }
    return dict(
        memory=memory,
        op=row["op"],
        shape=row["shape"],
        count=row["count"],
        us={name: statistics.median(times) for name, times in samples.items()},
        valid=valid,
        errors=errors,
        samples_us=samples,
    )


def main():
    import gc
    import hashlib

    path = os.environ.get("GDN_CURRENT_IMPL")
    if not path:
        raise ValueError("Set GDN_CURRENT_IMPL to the frozen original implementation")
    current = load_module("gdn_current_reference", Path(path))
    bench = load_module(
        "gdn_benchmark_inputs", Path(__file__).with_name("test_gdn_w8a16.py")
    )
    rows = json.loads(bench.SHAPES.read_text())
    group = os.environ.get("GDN_COMPARE_GROUP", "all")
    if group not in ("all", "decode", "prefill"):
        raise ValueError("GDN_COMPARE_GROUP must be all, decode, or prefill")
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    bench.ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    metadata = {
        "baseline_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "timing": "CUDA Graph median us, 31 alternating samples, 3 replays each",
        "scope": "precomputed gates, immutable initial state, final state only",
        "precision": "FP32 state; optimized MetaX chunks use tf32x3 FP32 dot operands",
        "sequence_boundaries": "floor(i*T/N); same raw random tensors for all paths",
    }
    for index, row in enumerate(rows, 1):
        prefill = row["op"] == "chunk_gated_delta_rule"
        if group != "all" and prefill != (group == "prefill"):
            continue
        result = dict(index=index, **compare_case(row, current, bench))
        results.append(result)
        (bench.ROOT / "comparison.json").write_text(
            json.dumps(dict(metadata=metadata, results=results), indent=2)
        )
        print(index, result["us"], result["valid"], flush=True)
        if not all(
            result["valid"][name] for name in ("current", "optimized", "pytorch")
        ):
            raise AssertionError("Core correctness check failed; see comparison.json")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
