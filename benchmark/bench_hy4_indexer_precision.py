# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Matched HY4 INT8/FP8 microbenchmarks, including graph-captured TopK chains.

Run with USE_FLAGGEMS=0 so the Torch baseline uses native ATen. Public FlagGems
and FlagGems-vllm functions are still called explicitly. No clocks are changed.
"""

import argparse
import gc
import hashlib
import importlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import flag_gems
import torch

import flaggems_vllm as gems

HEADS, DIM, PAGE, TOPK = 32, 128, 64, 2048


def quant(x, precision, eps=1e-10):
    values = x.float().reshape(-1, DIM)
    maximum = 127 if precision == "int8" else 448
    scales = values.abs().amax(-1, keepdim=True).clamp_min(eps) / maximum
    scales = torch.exp2(torch.ceil(torch.log2(scales)))
    values = (values / scales).clamp(-maximum, maximum)
    if precision == "int8":
        values = values.round()
    dtype = torch.int8 if precision == "int8" else torch.float8_e4m3fn
    return values.to(dtype).reshape(x.shape), scales


def query(x, precision):
    if precision == "int8":
        return gems.per_token_group_quant_int8(x, DIM, use_ue8m0=True)
    return flag_gems.per_token_group_quant_fp8(
        x, DIM, dtype=torch.float8_e4m3fn, scale_ue8m0=True
    )


def cache_parts(cache, precision):
    flat = cache.view(cache.shape[0], -1)
    dtype = torch.int8 if precision == "int8" else torch.float8_e4m3fn
    return (
        flat[:, : PAGE * DIM].view(dtype).view(-1, PAGE, DIM),
        flat[:, PAGE * DIM :].view(torch.float32),
    )


def write_cache(k, cache, slots, precision):
    fn = (
        gems.indexer_k_quant_and_cache_int8
        if precision == "int8"
        else gems.indexer_k_quant_and_cache
    )
    fn(k, cache, slots, DIM, "ue8m0")


def torch_cache(k, cache, slots):
    values, scales = cache_parts(cache, "int8")
    q, s = quant(k, "int8", 1e-4)
    values[slots // PAGE, slots % PAGE] = q
    scales[slots // PAGE, slots % PAGE] = s[:, 0]
    return cache


def torch_logits(q, keys, scales, weights):
    # FP32 exactly represents the D=128 INT8 dot; TF32 is disabled in main().
    dots = torch.matmul(q.float(), keys.float().transpose(-1, -2))
    return ((dots * scales.unsqueeze(-2)).relu() * weights.unsqueeze(-1)).sum(-2)


def measure(functions, warmup, inner, rounds, seed):
    graphs = {}
    keep = {}
    for name, fn in functions.items():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                out = fn()
        keep[name] = out
        graphs[name] = graph
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in graphs}
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / inner)
    result = {}
    for name, values in samples.items():
        ordered = sorted(values)

        def percentile(fraction):
            position = fraction * (len(ordered) - 1)
            low, high = math.floor(position), math.ceil(position)
            return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

        result[name] = dict(
            median_us=statistics.median(values),
            p10_us=percentile(0.1),
            p90_us=percentile(0.9),
            min_us=min(values),
            max_us=max(values),
            samples_us=values,
        )
    return result


def oracle(q, keys, scales, weights):
    dots = q.cpu().double() @ keys.cpu().double().T
    return (
        (dots * scales.cpu().double()).clamp_min(0) * weights.cpu().double()[:, None]
    ).sum(0)


def build_case(stage, rows, capacity, native):
    torch.manual_seed(1234 + rows + capacity)
    q_raw = torch.randn(rows * HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k_new = torch.randn(rows, DIM, device="cuda", dtype=torch.bfloat16)
    raw_weights = torch.randn(rows, HEADS, device="cuda") / math.sqrt(HEADS)
    functions, checks, unsupported = {}, {}, {}
    if stage == "query":
        for precision in ("int8", "fp8"):
            functions[precision] = lambda p=precision: query(q_raw, p)
            got, scales = functions[precision]()
            ref, ref_scales = quant(q_raw.cpu(), precision)
            torch.testing.assert_close(
                got.cpu().view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(scales.cpu(), ref_scales, rtol=1e-6, atol=0)
            checks[precision] = "exact quantized bytes and scales"
        functions["torch"] = lambda: quant(q_raw, "int8")
        return functions, checks, unsupported

    cache_rows = (
        rows if stage == "cache" else capacity * (rows if "decode" in stage else 1)
    )
    pages = math.ceil(cache_rows / PAGE)
    caches = {
        p: torch.zeros(pages, PAGE, DIM + 4, device="cuda", dtype=torch.uint8)
        for p in ("int8", "fp8")
    }
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    if stage == "cache":
        for precision in caches:
            functions[precision] = lambda p=precision: write_cache(
                k_new, caches[p], slots, p
            )
            functions[precision]()
            values, scales = cache_parts(caches[precision], precision)
            ref, ref_scales = quant(k_new.cpu(), precision, 1e-4)
            got = values.reshape(-1, DIM)[:rows].cpu().view(torch.uint8)
            torch.testing.assert_close(got, ref.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(
                scales.reshape(-1)[:rows].cpu(), ref_scales[:, 0], rtol=1e-6, atol=0
            )
            checks[precision] = "exact inserted bytes and scales"
        torch_storage = torch.empty_like(caches["int8"])
        functions["torch"] = lambda: torch_cache(k_new, torch_storage, slots)
        return functions, checks, unsupported

    is_decode = "decode" in stage
    slots = slots * capacity + capacity - 1 if is_decode else slots + capacity - rows
    k_raw = torch.randn(cache_rows, DIM, device="cuda", dtype=torch.bfloat16)
    k_raw[slots] = k_new
    all_slots = torch.arange(cache_rows, device="cuda", dtype=torch.int64)
    for precision in caches:
        write_cache(k_raw, caches[precision], all_slots, precision)
    tables = torch.arange(pages, device="cuda", dtype=torch.int32).view(
        rows if is_decode else 1, -1
    )
    lengths = torch.full((rows, 1), capacity, device="cuda", dtype=torch.int32)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = lengths.view(-1)
    offsets = torch.tensor([0, capacity], device="cuda", dtype=torch.int32)
    topk_out = {
        p: torch.empty(rows, TOPK, device="cuda", dtype=torch.int32) for p in caches
    }
    data = {}
    for precision in caches:
        q, scale = query(q_raw, precision)
        values, ks = cache_parts(caches[precision], precision)
        keys = values.reshape(-1, DIM)
        ks = ks.reshape(-1)
        data[precision] = (q.view(rows, HEADS, DIM), scale.view(rows, HEADS), keys, ks)
        weights = raw_weights * scale.view(rows, HEADS)
        gathered = torch.empty(capacity, DIM, device="cuda", dtype=q.dtype)
        gathered_s = torch.empty(capacity, 4, device="cuda", dtype=torch.uint8)

        def call(p=precision, w=weights, values_out=gathered, scales_out=gathered_s):
            qv, qs, key, key_s = data[p]
            if stage.endswith("chain"):
                qv, qs = query(q_raw, p)
                qv, qs = qv.view(rows, HEADS, DIM), qs.view(rows, HEADS)
                w = raw_weights * qs
                write_cache(k_new, caches[p], slots, p)
            if is_decode:
                fn = (
                    gems.int8_paged_mqa_logits
                    if p == "int8"
                    else gems.fp8_fp4_paged_mqa_logits_graph
                )
                out = fn(
                    (qv.unsqueeze(1), None),
                    caches[p].unsqueeze(2),
                    w,
                    lengths,
                    tables,
                    None,
                    capacity,
                    False,
                )
            else:
                if stage.endswith("chain"):
                    gems.cp_gather_indexer_k_quant_cache(
                        caches[p], values_out, scales_out, tables, offsets
                    )
                    key, key_s = values_out, scales_out.view(torch.float32).view(-1)
                fn = gems.int8_mqa_logits if p == "int8" else gems.fp8_fp4_mqa_logits
                out = fn((qv, None), (key, key_s), w, starts, ends, False)
            if stage.endswith("chain"):
                if is_decode:
                    gems.top_k_per_row_decode(
                        out,
                        1,
                        lengths,
                        topk_out[p],
                        rows,
                        out.stride(0),
                        out.stride(1),
                        TOPK,
                    )
                else:
                    gems.top_k_per_row_prefill(
                        out,
                        starts,
                        ends,
                        topk_out[p],
                        rows,
                        out.stride(0),
                        out.stride(1),
                        TOPK,
                    )
                return topk_out[p]
            return out

        functions[precision] = call
        if is_decode and precision == "fp8" and caches[precision].numel() > 2**31:
            # The current FP8 graph kernel multiplies INT32 physical page IDs
            # by its byte stride. A >2GiB pool overflows that address expression.
            # The CPU oracle exposed incorrect results here; do not time them.
            unsupported["fp8"] = (
                "existing FP8 graph kernel INT32 address overflow above 2GiB"
            )
            del functions[precision]
            continue
        actual = call()
        max_error = 0.0
        # Keep the existing INT8 test tolerance; FP8 follows its existing
        # operator test's 5e-2 budget (also needed by native DeepGEMM FP8 MMA).
        atol = 2e-3 if precision == "int8" else 5e-2
        for row in sorted({0, rows - 1}):
            lo, hi = (
                (row * capacity, (row + 1) * capacity) if is_decode else (0, capacity)
            )
            ref = oracle(data[precision][0][row], keys[lo:hi], ks[lo:hi], weights[row])
            if stage.endswith("chain"):
                expected = ref.topk(TOPK).indices
                # Equal scores at the cutoff may choose different equivalent tokens.
                selected = actual[row].cpu().long()
                assert selected.unique().numel() == TOPK
                torch.testing.assert_close(
                    ref[selected].sort().values,
                    ref[expected].sort().values,
                    rtol=1e-5,
                    atol=atol,
                )
            else:
                torch.testing.assert_close(
                    actual[row].cpu(), ref.float(), rtol=1e-5, atol=atol
                )
                max_error = max(
                    max_error, (actual[row].cpu().double() - ref).abs().max().item()
                )
        checks[precision] = {
            "cpu_fp64_reference": "pass",
            "max_absolute_error": max_error,
            "atol": atol,
            "rtol": 1e-5,
        }

    qi, si, ki, ksi = data["int8"]
    wi = raw_weights * si
    if stage.endswith("chain"):
        torch_storage = caches["int8"].clone()

        def torch_chain():
            q, scales = quant(q_raw, "int8")
            torch_cache(k_new, torch_storage, slots)
            keys, key_scales = cache_parts(torch_storage, "int8")
            keys, key_scales = keys.reshape(-1, DIM), key_scales.reshape(-1)
            if is_decode:
                keys = keys.view(rows, capacity, DIM)
                key_scales = key_scales.view(rows, capacity)
            logits = torch_logits(
                q.view(rows, HEADS, DIM),
                keys,
                key_scales,
                raw_weights * scales.view(rows, HEADS),
            )
            return logits.topk(TOPK, dim=-1, sorted=False).indices.to(torch.int32)

        functions["torch"] = torch_chain
        actual = torch_chain()
        for row in sorted({0, rows - 1}):
            lo, hi = (
                (row * capacity, (row + 1) * capacity) if is_decode else (0, capacity)
            )
            ref = oracle(qi[row], ki[lo:hi], ksi[lo:hi], wi[row])
            selected = actual[row].cpu().long()
            torch.testing.assert_close(
                ref[selected].sort().values,
                ref.topk(TOPK).values.sort().values,
                rtol=1e-5,
                atol=2e-3,
            )
        checks["torch"] = "CPU FP64 quantized TopK reference passed"
    else:
        if is_decode:
            functions["torch"] = lambda: torch_logits(
                qi, ki.view(rows, capacity, DIM), ksi.view(rows, capacity), wi
            )
        else:
            functions["torch"] = lambda: torch_logits(qi, ki, ksi, wi)
        if native:
            from vllm.utils import deep_gemm as dg

            qf, sf, kf, ksf = data["fp8"]
            wf = raw_weights * sf
            if is_decode:
                schedule = dg.get_paged_mqa_logits_metadata(
                    lengths,
                    PAGE,
                    torch.cuda.get_device_properties(0).multi_processor_count,
                )
                functions["deepgemm_fp8"] = lambda: dg.fp8_fp4_paged_mqa_logits(
                    (qf.unsqueeze(1), None),
                    caches["fp8"].unsqueeze(2),
                    wf,
                    lengths,
                    tables,
                    schedule,
                    capacity,
                    False,
                )
            else:
                functions["deepgemm_fp8"] = lambda: dg.fp8_fp4_mqa_logits(
                    (qf, None), (kf, ksf), wf, starts, ends, False
                )
            try:
                got = functions["deepgemm_fp8"]()
                torch.cuda.synchronize()
            except (RuntimeError, NotImplementedError) as error:
                unsupported["deepgemm_fp8"] = str(error)
                del functions["deepgemm_fp8"]
            else:
                for row in sorted({0, rows - 1}):
                    lo, hi = (
                        (row * capacity, (row + 1) * capacity)
                        if is_decode
                        else (0, capacity)
                    )
                    ref = oracle(qf[row], kf[lo:hi], ksf[lo:hi], wf[row])
                    torch.testing.assert_close(
                        got[row].cpu(), ref.float(), rtol=1e-5, atol=5e-2
                    )
                checks["deepgemm_fp8"] = "CPU FP64 reference passed"
    return functions, checks, unsupported


def cases(suite, selected):
    if suite == "smoke":
        shape_map = {s: [(1, 4096), (7, 4096)] for s in selected}
    else:
        shape_map = {
            "query": [(m, 0) for m in (1, 7, 32, 128, 512)],
            "cache": [(m, 0) for m in (1, 7, 32, 128, 512)],
            "prefill": [(m, n) for m in (1, 7, 32, 128, 512) for n in (4096, 32768)],
            "decode": [(m, n) for m in (1, 7, 32, 128) for n in (4096, 32768, 131072)],
            "prefill_chain": [(m, n) for m in (1, 7, 128) for n in (4096, 32768)],
            "decode_chain": [(m, n) for m in (1, 7, 32) for n in (4096, 32768)],
        }
    for stage in selected:
        for rows, capacity in shape_map[stage]:
            yield stage, rows, capacity


def compiled_metrics():
    module = importlib.import_module("flaggems_vllm.ops.int8_sparse_indexer")
    result = []
    for name in ("_prefill", "_prefill_rows", "_decode"):
        if not hasattr(module, name):
            continue
        for entry in getattr(module, name).kernel_cache[0].values():
            kernel, constants = entry[:2]
            result.append(
                dict(
                    name=name,
                    constants=constants,
                    registers=kernel.n_regs,
                    spills=kernel.n_spills,
                    shared_bytes=kernel.metadata.shared,
                    mma_instructions=kernel.asm["ptx"].count("mma.sync"),
                    wgmma_instructions=kernel.asm["ptx"].count("wgmma.mma_async"),
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "core"), default="core")
    parser.add_argument(
        "--stages", default="query,cache,prefill,decode,prefill_chain,decode_chain"
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--inner", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--native-fp8", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        parser.error("FP8 comparison requires an SM90+ NVIDIA GPU")
    if min(args.warmup, args.inner, args.rounds) < 1:
        parser.error("warmup, inner and rounds must be positive")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    props = torch.cuda.get_device_properties(0)
    report = dict(
        gpu=props.name,
        sm_count=props.multi_processor_count,
        torch=torch.__version__,
        heads=HEADS,
        head_dim=DIM,
        page_size=PAGE,
        topk=TOPK,
        dtype="bfloat16",
        timing="CUDA graph, interleaved provider order; no clock changes",
        decode_pages="disjoint per request",
        warmup=args.warmup,
        inner=args.inner,
        rounds=args.rounds,
        results=[],
        operator_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        operator_source_sha256=hashlib.sha256(
            Path(gems.__file__)
            .with_name("ops")
            .joinpath("int8_sparse_indexer.py")
            .read_bytes()
        ).hexdigest(),
        configs_sha256=hashlib.sha256(
            Path(gems.__file__)
            .with_name("runtime")
            .joinpath("backend/_nvidia/tune_configs.yaml")
            .read_bytes()
        ).hexdigest(),
        skipped=[],
        gpu_state=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,power.draw",
                "--format=csv",
            ],
            text=True,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    deadline = time.perf_counter() + 0.25
    while time.perf_counter() < deadline:
        torch.mm(a, a)
    torch.cuda.synchronize()
    for stage, rows, capacity in cases(args.suite, args.stages.split(",")):
        # Keep other workloads' allocations untouched. Large disjoint decode
        # pools plus the dense Torch oracle need a substantial temporary budget.
        free, _ = torch.cuda.mem_get_info()
        tokens = rows * capacity if "decode" in stage else capacity
        estimated = tokens * DIM * 8 + rows * capacity * 16 + 256 * 1024**2
        if estimated > free:
            report["skipped"].append(
                dict(
                    stage=stage,
                    rows=rows,
                    capacity=capacity,
                    reason="insufficient free memory",
                    estimated_bytes=estimated,
                    free_bytes=free,
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"SKIP {stage} M={rows} N={capacity}: need {estimated}, free {free}",
                flush=True,
            )
            continue
        functions, checks, unsupported = build_case(
            stage, rows, capacity, args.native_fp8
        )
        torch_temporary_bytes = rows * HEADS * capacity * 12
        if "torch" in functions and estimated + torch_temporary_bytes > free:
            del functions["torch"]
            unsupported["torch"] = "dense Torch temporary exceeds free GPU memory"
        latencies = measure(
            functions, args.warmup, args.inner, args.rounds, rows + capacity
        )
        speedup = (
            latencies["fp8"]["median_us"] / latencies["int8"]["median_us"]
            if "fp8" in latencies
            else None
        )
        result = dict(
            stage=stage,
            rows=rows,
            capacity=capacity,
            correctness=checks,
            latency=latencies,
            fp8_over_int8=speedup,
            unsupported=unsupported,
            compiled_int8=compiled_metrics(),
        )
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{stage:14} M={rows:4} N={capacity:6} "
            + " ".join(f"{k}={v['median_us']:.3f}us" for k, v in latencies.items())
            + (
                f" FP8/INT8={speedup:.3f}x"
                if speedup is not None
                else " FP8 baseline unsupported"
            ),
            flush=True,
        )
        del functions
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
