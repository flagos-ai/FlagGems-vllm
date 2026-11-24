# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run paired Ascend MiniMax tests through the FlagGems-vllm public API."""

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import socket
import statistics
import subprocess
import tempfile
import traceback
from importlib import import_module
from pathlib import Path

from baseline.minimax_qknorm_rope_cache import (
    OP_NAME,
    Q_HEADS,
    REPLAY_ORIGINAL_SHA256,
    TOKENS,
    build_case,
    check_call,
    load_baseline,
)


def make_sampler(torch, fn, timer, warmup, active):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    if timer == "do-bench-npu":
        from triton.backends.ascend.testing import do_bench_npu

        def sample():
            with tempfile.TemporaryDirectory(prefix="minimax-profile-") as folder:
                return float(
                    do_bench_npu(
                        fn,
                        warmup=warmup,
                        active=active,
                        clear_l2_cache=False,
                        prof_dir=folder,
                    )
                )

        return sample

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(active):
            fn()
    graph.replay()
    torch.npu.synchronize()

    def sample():
        begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / active

    return sample


def measure_case(
    torch, candidate, baseline, case, timer, rounds, warmup, active, samples
):
    methods = {"baseline": baseline, "flaggems_vllm": candidate}
    for fn in methods.values():
        check_call(fn, case, check_untouched=False)
    kwargs = case[0]

    samplers = {
        name: make_sampler(torch, lambda fn=fn: fn(**kwargs), timer, warmup, active)
        for name, fn in methods.items()
    }
    values = {name: [] for name in methods}
    for round_index in range(rounds):
        order = tuple(samplers)
        if round_index % 2:
            order = order[::-1]
        for name in order:
            block = [samplers[name]() for _ in range(samples)]
            if any(not math.isfinite(value) or value <= 0 for value in block):
                raise RuntimeError(f"{timer} returned invalid ms for {name}: {block}")
            values[name].append(block)

    stats = {}
    for name, blocks in values.items():
        medians = [statistics.median(block) for block in blocks]
        stats[name] = dict(
            rounds=len(medians),
            samples=sum(len(block) for block in blocks),
            median_ms=statistics.median(medians),
            mean_ms=statistics.mean(medians),
            sd_ms=statistics.stdev(medians) if len(medians) > 1 else 0.0,
            per_round_ms=medians,
        )
    base, fg = (stats[name]["median_ms"] for name in methods)
    deltas = [
        x - y
        for x, y in zip(
            stats["baseline"]["per_round_ms"], stats["flaggems_vllm"]["per_round_ms"]
        )
    ]
    return dict(
        correctness="pass",
        speedup=base / fg,
        latency_drop_pct=(base - fg) / base * 100,
        flaggems_faster_rounds=sum(delta > 0 for delta in deltas),
        paired_baseline_minus_flaggems_ms=deltas,
        statistics=stats,
        samples_ms=values,
    )


def read_source(out, label, callable_):
    path = Path(inspect.getsourcefile(callable_))
    data = path.read_bytes()
    (out / f"source_{label}.py").write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def write_results(out, rows):
    (out / "results.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    with (out / "performance_ms.csv").open("w", newline="") as f:
        fields = (
            "q_heads",
            "tokens",
            "method",
            "status",
            "rounds",
            "samples",
            "median_ms",
            "mean_ms",
            "sd_ms",
            "speedup",
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            binding = {k: row[k] for k in ("q_heads", "tokens", "status")}
            if row["status"] != "ok":
                writer.writerow(binding)
                continue
            for method, metrics in row["statistics"].items():
                writer.writerow(
                    {
                        **binding,
                        "method": method,
                        "speedup": row["speedup"],
                        **{key: metrics[key] for key in fields if key in metrics},
                    }
                )

    with (out / "samples_ms.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("q_heads", "tokens", "method", "round", "sample", "ms"))
        for row in rows:
            for method, blocks in row.get("samples_ms", {}).items():
                for block_index, block in enumerate(blocks, 1):
                    for sample_index, value in enumerate(block, 1):
                        writer.writerow(
                            (
                                row["q_heads"],
                                row["tokens"],
                                method,
                                block_index,
                                sample_index,
                                value,
                            )
                        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument(
        "--heads", nargs="+", type=int, default=Q_HEADS, choices=Q_HEADS
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKENS, choices=TOKENS)
    parser.add_argument("--rounds", default=6, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--active", default=30, type=int)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument(
        "--baseline", choices=("source-replay", "native"), default="source-replay"
    )
    parser.add_argument(
        "--timer", choices=("graph-event", "do-bench-npu"), default="graph-event"
    )
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.active, args.samples) < 1:
        parser.error("rounds, warmup, active and samples must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.glob("source_*.py")) or (args.out / "results.json").exists():
        parser.error("--out already contains benchmark output")

    import flag_gems
    import torch
    import torch_npu
    import triton

    import flaggems_vllm

    if flaggems_vllm.vendor_name != "ascend":
        raise RuntimeError(f"expected Ascend backend, got {flaggems_vllm.vendor_name}")
    device = torch.device("npu", args.device)
    torch.npu.set_device(device)
    torch.npu.config.allow_internal_format = False
    backend = import_module(f"flaggems_vllm.runtime.backend._ascend.ops.{OP_NAME}")
    baseline = load_baseline(args.baseline)
    candidate = flaggems_vllm.minimax_qknorm_rope_cache
    samples = 1 if args.timer == "do-bench-npu" else args.samples
    env = dict(
        host=socket.gethostname(),
        device=torch.npu.get_device_name(device),
        vector_cores=backend.get_vectorcore_num(),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        triton_path=triton.__file__,
        flag_gems_path=flag_gems.__file__,
        flaggems_vllm_path=flaggems_vllm.__file__,
        task_queue_enable=os.getenv("TASK_QUEUE_ENABLE"),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        baseline=args.baseline,
        replay_upstream_sha256=(
            REPLAY_ORIGINAL_SHA256 if args.baseline == "source-replay" else None
        ),
        timer=args.timer,
        samples_per_round=samples,
        options=vars(args),
        sources={
            "public": read_source(args.out, "public", candidate),
            "ascend": read_source(
                args.out, "ascend", backend.minimax_qknorm_rope_cache
            ),
            "baseline": read_source(args.out, "baseline", baseline),
        },
    )
    (args.out / "environment.json").write_text(
        json.dumps(env, indent=2, default=str) + "\n"
    )
    if args.timer == "do-bench-npu":
        data = torch.ones(4096, device=device)
        sample = make_sampler(
            torch, lambda: data + data, args.timer, args.warmup, args.active
        )
        value = sample()
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"do_bench_npu profiler preflight returned {value}; use --timer graph-event"
            )

    results = []
    with torch.inference_mode():
        for heads in args.heads:
            for tokens in args.tokens:
                row = dict(q_heads=heads, tokens=tokens, status="ok")
                try:
                    case = build_case(tokens, heads, device)
                    row["input"] = {
                        name: dict(
                            shape=list(value.shape),
                            stride=list(value.stride()),
                            dtype=str(value.dtype),
                        )
                        for name, value in case[0].items()
                        if isinstance(value, torch.Tensor)
                    }
                    row["programs"] = min(tokens, env["vector_cores"])
                    row.update(
                        measure_case(
                            torch,
                            candidate,
                            baseline,
                            case,
                            args.timer,
                            args.rounds,
                            args.warmup,
                            args.active,
                            samples,
                        )
                    )
                    print(
                        f"Q={heads} T={tokens} speedup={row['speedup']:.4f}x",
                        flush=True,
                    )
                except Exception:
                    row.update(status="failed", error=traceback.format_exc())
                    print(row["error"], flush=True)
                finally:
                    results.append(row)
                    write_results(args.out, results)
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
