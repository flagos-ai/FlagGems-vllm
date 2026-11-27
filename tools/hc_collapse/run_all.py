"""No-graph vLLM-Ascend versus FlagGems-vllm. Run from repository root."""

import argparse
import csv
import importlib.metadata
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import traceback
from pathlib import Path

import flag_gems
import torch
import torch_npu
import triton
from triton.backends.ascend.testing import do_bench_npu

import flaggems_vllm
from baseline.hc_collapse import (
    VLLM_ASCEND_REPLAY_SOURCE,
    all_cases,
    build_inputs,
    vendor_native,
    vllm_ascend_replay,
)
from flaggems_vllm.runtime.backend._ascend.ops.hc_collapse import (
    _general,
    dispatch_path,
)


def sample(fn, warmup, active):
    # One callable per profile allows the backend to account for multiple tasks.
    with tempfile.TemporaryDirectory(prefix="hc-do-bench-npu-") as directory:
        value = float(
            do_bench_npu(
                fn,
                warmup=warmup,
                active=active,
                clear_l2_cache=False,
                prof_dir=directory,
            )
        )
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"do_bench_npu returned an invalid duration: {value}")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--active", "--calls", dest="active", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--baseline", choices=("native", "source-replay"), default="native")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if min(a.rounds, a.active) < 1 or a.warmup < 0:
        ap.error("rounds/active must be positive; warmup nonnegative")
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "results.json").exists():
        raise FileExistsError("Use a fresh output directory")
    torch.npu.set_device(a.device)
    if flaggems_vllm.vendor_name != "ascend":
        raise RuntimeError("expected ascend backend")
    torch.npu.config.allow_internal_format = False
    if a.baseline == "source-replay":
        baseline = vllm_ascend_replay
        baseline_method = "source_replay"
        baseline_label = VLLM_ASCEND_REPLAY_SOURCE
    else:
        baseline = vendor_native("ascend", required=True)
        baseline_method = "vendor"
        baseline_label = f"{inspect.getsourcefile(baseline)}:{baseline.__qualname__}"
    speedup_key = f"speedup_vs_{baseline_method}"
    print(f"vLLM-Ascend baseline ({a.baseline}): {baseline_label}", flush=True)
    versions = {}
    for name in ("vllm", "vllm-ascend"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    env = dict(
        host=platform.node(),
        device=torch.npu.get_device_name(a.device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        triton_path=triton.__file__,
        flag_gems_path=flag_gems.__file__,
        flaggems_vllm_path=flaggems_vllm.__file__,
        implementation=inspect.getsourcefile(flaggems_vllm.hc_collapse),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        baseline=baseline_label,
        baseline_method=baseline_method,
        package_versions=versions,
        task_queue_enable=os.environ.get("TASK_QUEUE_ENABLE"),
        allow_internal_format=False,
        timing="do_bench_npu via torch_npu.profiler; device task duration sum per call, averaged within each round",
        timer_source=inspect.getsourcefile(do_bench_npu),
        clear_l2_cache=False,
        options=vars(a),
    )
    (dest / "environment.json").write_text(json.dumps(env, indent=2))
    for source in [
        inspect.getsourcefile(flaggems_vllm.hc_collapse),
        inspect.getsourcefile(_general),
    ]:
        src = Path(source)
        (dest / ("source_" + src.parent.name + "_" + src.name)).write_text(
            src.read_text()
        )
    data = []
    samples = []
    failed = False
    for binding in all_cases():
        row = dict(binding, dispatch=None, correctness="pending", methods={})
        try:
            torch.manual_seed(0)
            (x, p), _ = build_inputs(binding, torch.bfloat16, f"npu:{a.device}")
            row["dispatch"] = dispatch_path(x)
            ref = (p.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)
            methods = {
                baseline_method: lambda: baseline(x, p),
                "merged": lambda: flaggems_vllm.hc_collapse(x, p),
                "general": lambda: _general(x, p),
            }
            valid = {}
            for name, fn in methods.items():
                try:
                    actual = fn()
                    torch.testing.assert_close(actual, ref, atol=0.016, rtol=0.01)
                    assert (
                        actual.is_contiguous() and torch.isfinite(actual).all().item()
                    )
                    if x.shape[0]:
                        torch.testing.assert_close(actual[0], x[0, -1], atol=0, rtol=0)
                    if x.shape[0] > 1 and x.shape[1] == 4:
                        torch.testing.assert_close(
                            actual[1], torch.zeros_like(actual[1]), atol=0, rtol=0
                        )
                    valid[name] = fn
                    row["methods"][name] = {
                        "status": "correct",
                        "max_abs": (
                            float((actual.float() - ref.float()).abs().max())
                            if actual.numel()
                            else 0
                        ),
                    }
                except Exception:
                    failed = True
                    row["methods"][name] = {
                        "status": "correctness_failed",
                        "error": traceback.format_exc(),
                    }
            row["correctness"] = "pass" if len(valid) == len(methods) else "failed"
            if x.numel():
                per = {name: [] for name in valid}
                for rnd in range(a.rounds):
                    names = list(per)
                    if rnd % 2:
                        names.reverse()
                    for name in names:
                        if row["methods"][name]["status"] == "benchmark_failed":
                            continue
                        try:
                            value = sample(valid[name], a.warmup, a.active)
                        except Exception:
                            failed = True
                            row["methods"][name].update(
                                status="benchmark_failed", error=traceback.format_exc()
                            )
                            continue
                        per[name].append(value)
                        samples.append(
                            dict(
                                shape=str(binding["shape"]),
                                strided=binding["strided"],
                                method=name,
                                round=rnd,
                                active=a.active,
                                do_bench_npu_ms=value,
                            )
                        )
                for name, vals in per.items():
                    if row["methods"][name]["status"] == "benchmark_failed":
                        continue
                    row["methods"][name].update(
                        status="ok",
                        n=len(vals),
                        active_per_round=a.active,
                        rounds_ms=vals,
                        median_ms=statistics.median(vals),
                        mean_ms=statistics.mean(vals),
                        sd_ms=statistics.stdev(vals) if len(vals) > 1 else 0,
                        min_ms=min(vals),
                        max_ms=max(vals),
                    )
                merged = row["methods"]["merged"].get("median_ms")
                if merged:
                    for name in ("general", baseline_method):
                        base = row["methods"].get(name, {}).get("median_ms")
                        row[f"speedup_vs_{name}"] = (
                            base / merged if base is not None else None
                        )
            else:
                for name in valid:
                    row["methods"][name]["status"] = "empty_pass"
        except Exception:
            failed = True
            row["error"] = traceback.format_exc()
        data.append(row)
        (dest / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False))
        print(
            binding, row["correctness"], {speedup_key: row.get(speedup_key)}, flush=True
        )
    with (dest / "samples_ms.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "strided",
                "method",
                "round",
                "active",
                "do_bench_npu_ms",
            ],
        )
        w.writeheader()
        w.writerows(samples)
    with (dest / "performance_ms.csv").open("w") as f:
        fields = [
            "shape",
            "strided",
            "dispatch",
            "method",
            "status",
            "n",
            "active_per_round",
            "median_ms",
            "mean_ms",
            "sd_ms",
            "min_ms",
            "max_ms",
            "speedup_vs_general",
            "speedup_vs_vendor",
            "speedup_vs_source_replay",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in data:
            for method, m in row["methods"].items():
                out = {k: row.get(k) for k in fields}
                out.update({k: m.get(k) for k in fields if k in m})
                out.update(shape=str(row["shape"]), method=method)
                w.writerow(out)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
