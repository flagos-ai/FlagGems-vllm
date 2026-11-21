#!/usr/bin/env python3
"""Benchmark quant LightningIndexer with the project NPU timing protocol.
 
Correctness is checked against the official Ascend operator before timing. Device
latency is obtained from ``workspace.utils.utils.perf_npu``. Host Graph latency
uses ``workspace.utils.utils.host_perf`` on ACLGraph ``graph.replay``; the host
comparison uses official/triton AB blocks.
"""
 
from __future__ import annotations
 
import argparse
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable
 
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
 
import torch
import torch_npu
import triton
 
import flaggems_vllm
from tests.quant_lightning_indexer_utils import (
    CASES,
    QuantLightningIndexerCase,
    assert_matches,
    call_candidate,
    call_official,
    make_inputs,
    prepare_metadata,
    register_official_ops,
)
 
 
def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
 
 
def _load_perf_utils(path: Path) -> Any:
    return _load_module(path, "quant_lightning_indexer_perf_utils")
 
 
def _make_executor(fn: Callable[[], Any]) -> Callable[[], Any]:
    def execute() -> Any:
        return fn()
 
    return execute
 
 
def _kernel_perf(
    fn: Callable[[], Any],
    perf_npu: Callable[..., dict[str, Any]],
    profile_dir: Path,
    active: int,
) -> dict[str, Any]:
    result = perf_npu(
        _make_executor(fn),
        profiling_dir=str(profile_dir),
        active=active,
    )
    if not isinstance(result, dict) or result.get("device_latency_us") is None:
        raise RuntimeError(f"perf_npu did not produce device latency: {result}")
    return {
        "latency_us": float(result["device_latency_us"]),
        "profiling_path": result.get("kernel_profiling_path"),
    }
 
 
def _bind_actual_launches(
    module: Any,
    fn: Callable[[], Any],
) -> tuple[Callable[[], Any], list[dict[str, Any]]]:
    """Move Triton host dispatch out of the ACLGraph capture."""
 
    records: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]] = []
    originals = {
        name: kernel
        for name, kernel in vars(module).items()
        if isinstance(kernel, triton.runtime.JITFunction)
    }
 
    class LaunchRecorder:
        def __init__(self, kernel: Any):
            self.kernel = kernel
 
        def __getitem__(self, grid: Any) -> Callable[..., Any]:
            launcher = self.kernel[grid]
 
            def launch(*args: Any, **kwargs: Any) -> Any:
                records.append((self.kernel, grid, args, dict(kwargs)))
                return launcher(*args, **kwargs)
 
            return launch
 
    try:
        for name, kernel in originals.items():
            setattr(module, name, LaunchRecorder(kernel))
        output = fn()
        torch.npu.synchronize()
    finally:
        for name, kernel in originals.items():
            setattr(module, name, kernel)
 
    if not records:
        raise RuntimeError("candidate wrapper did not launch a Triton kernel")
 
    def replay_launches() -> Any:
        for kernel, grid, args, kwargs in records:
            kernel[grid](*args, **kwargs)
        return output
 
    launches = [
        {
            "kernel": kernel.__name__,
            "grid": list(grid) if isinstance(grid, tuple) else grid,
            "options": {
                key: value
                for key, value in kwargs.items()
                if not isinstance(value, torch.Tensor)
            },
        }
        for kernel, grid, _, kwargs in records
    ]
    return replay_launches, launches
 
 
def _capture_graph(
    fn: Callable[[], Any],
    label: str,
    capture_warmup: int,
) -> dict[str, Any]:
    eager_output = fn()
    torch.npu.synchronize()
    eager_indices = eager_output[0]
    eager_cpu = eager_indices.detach().cpu().clone()
    eager_sorted = torch.sort(eager_cpu, dim=-1).values
    del eager_output
 
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(capture_stream):
        for _ in range(capture_warmup):
            fn()
    capture_stream.synchronize()
 
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        static_output = fn()
    torch.npu.current_stream().wait_stream(capture_stream)
    torch.npu.synchronize()
 
    def replay() -> Any:
        graph.replay()
        return static_output
 
    replay_output = replay()
    torch.npu.synchronize()
    replay_indices = replay_output[0]
    replay_cpu = replay_indices.detach().cpu().clone()
    if eager_cpu.shape != replay_cpu.shape or not torch.equal(
        eager_sorted, torch.sort(replay_cpu, dim=-1).values
    ):
        del graph
        raise RuntimeError(f"ACLGraph replay changed quant LightningIndexer output: {label}")
    return {"graph": graph, "replay": replay}
 
 
def _host_ab(
    calls: dict[str, Callable[[], Any]],
    host_perf: Callable[..., float],
    warmup: int,
    repeat: int,
    cycles: int,
    label: str,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    for cycle in range(cycles):
        for position, name in enumerate(("official", "triton")):
            latency_ms = float(
                host_perf(
                    calls[name],
                    "npu",
                    warmup=warmup,
                    repeat=repeat,
                    label=f"{label}_{name}_c{cycle}_p{position}",
                )
            )
            blocks.append(
                {
                    "cycle": cycle,
                    "position": position,
                    "implementation": name,
                    "latency_ms": latency_ms,
                }
            )
    result: dict[str, Any] = {"blocks": blocks}
    for name in ("official", "triton"):
        values = [
            block["latency_ms"]
            for block in blocks
            if block["implementation"] == name
        ]
        result[name] = {
            "latency_ms": math.exp(statistics.fmean(math.log(value) for value in values)),
            "blocks_ms": values,
        }
    return result
 
 
def _case_report(case: QuantLightningIndexerCase) -> dict[str, Any]:
    return {
        "scene": case.scene,
        "query_shape": [case.query_t, 64, 128],
        "key_shape": [4422, 128, 1, 128],
        "weights_shape": [case.query_t, 64],
        "query_scale_shape": [case.query_t, 64],
        "key_scale_shape": [4422, 128, 1],
        "source_q_lengths": list(case.source_q_lengths),
        "physical_query_cumulative_lengths": [
            sum(case.local_q_lengths[: index + 1])
            for index in range(len(case.local_q_lengths))
        ],
        "key_lengths_raw": list(case.key_lengths),
        "compressed_key_lengths_floor_div4": [length // 4 for length in case.key_lengths],
    }
 
 
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated case IDs, or all for the full R1-R12 suite",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--active", type=int, default=6)
    parser.add_argument(
        "--utils",
        type=Path,
        default=Path(
            os.environ.get(
                "QUANT_LIGHTNING_INDEXER_UTILS",
                Path(__file__).resolve().parents[2] / "workspace" / "utils" / "utils.py",
            )
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quant_lightning_indexer_perf.json"),
    )
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
 
    if args.repeat < 3 or args.cycles < 1 or args.active < 1:
        raise ValueError("repeat must be >= 3, cycles and active must be positive")
    all_case_ids = {case.case_id for case in CASES}
    requested = (
        list(CASES)
        if args.cases == "all"
        else [case for case in CASES if case.case_id in set(args.cases.split(","))]
    )
    unknown = (
        []
        if args.cases == "all"
        else [case_id for case_id in args.cases.split(",") if case_id and case_id not in all_case_ids]
    )
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    if not requested:
        raise ValueError("no cases selected")
    if not args.utils.is_file():
        raise FileNotFoundError(
            f"performance utilities not found: {args.utils}; pass --utils workspace/utils/utils.py"
        )
 
    torch_npu.npu.set_device(args.device)
    if flaggems_vllm.vendor_name != "ascend":
        raise RuntimeError("quant LightningIndexer benchmark requires the Ascend backend")
    register_official_ops()
    perf_utils = _load_perf_utils(args.utils.resolve())
    module = _load_module(
        Path(__file__).resolve().parents[1]
        / "src"
        / "flaggems_vllm"
        / "runtime"
        / "backend"
        / "_ascend"
        / "ops"
        / "quant_lightning_indexer.py",
        "quant_lightning_indexer_target",
    )
    report: dict[str, Any] = {
        "environment": {
            "device": args.device,
            "device_name": torch.npu.get_device_name(args.device),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "triton": triton.__version__,
        },
        "protocol": {
            "kernel": "workspace.utils.utils.perf_npu",
            "host_graph": "workspace.utils.utils.host_perf(graph.replay)",
            "host_order": "AB: official then triton per cycle",
            "correctness": "exact sorted [T,1,512] int32 index rows",
        },
        "cases": {},
    }
 
    for case in requested:
        inputs = make_inputs(case, torch.device(f"npu:{args.device}"))
        prepare_metadata(inputs, case)
        official = call_official(inputs)
        candidate = call_candidate(module, case, inputs)
        torch.npu.synchronize()
        assert_matches(official, candidate)
        item = _case_report(case)
        item["correctness"] = "passed"
        if args.correctness_only:
            report["cases"][case.case_id] = item
            del inputs, official, candidate
            gc.collect()
            torch.npu.empty_cache()
            continue
 
        calls = {
            "official": lambda inputs=inputs: call_official(inputs),
            "triton": lambda inputs=inputs, case=case: call_candidate(module, case, inputs),
        }
        item["kernel"] = {
            name: _kernel_perf(
                fn,
                perf_utils.perf_npu,
                args.output.parent / "device_profiles" / f"{case.case_id}_{name}",
                args.active,
            )
            for name, fn in calls.items()
        }
        candidate_graph_fn, launches = _bind_actual_launches(module, calls["triton"])
        graph_calls = {
            "official": _capture_graph(calls["official"], f"{case.case_id}_official", 3),
            "triton": _capture_graph(candidate_graph_fn, f"{case.case_id}_triton", 3),
        }
        os.environ["HOST_PERF_TRACE"] = "1"
        item["host_graph"] = _host_ab(
            {name: value["replay"] for name, value in graph_calls.items()},
            perf_utils.host_perf,
            args.warmup,
            args.repeat,
            args.cycles,
            f"{case.case_id}_graph",
        )
        item["triton_launches"] = launches
        item["speedup_official_over_triton"] = {
            "kernel": item["kernel"]["official"]["latency_us"]
            / item["kernel"]["triton"]["latency_us"],
            "host_graph": item["host_graph"]["official"]["latency_ms"]
            / item["host_graph"]["triton"]["latency_ms"],
        }
        report["cases"][case.case_id] = item
        for value in graph_calls.values():
            del value["graph"]
        del graph_calls, candidate_graph_fn, inputs, official, candidate
        gc.collect()
        torch.npu.empty_cache()
 
    report["status"] = "completed_correctness_only" if args.correctness_only else "completed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())