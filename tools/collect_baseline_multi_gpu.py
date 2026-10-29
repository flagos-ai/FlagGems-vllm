#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dynamically schedule isolated operator collectors, one consumer per GPU."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 算子固定取自 baseline/ 目录。
OPS_DIR = ROOT / "baseline"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_devices(value):
    devices = [v.strip() for v in value.split(",")]
    if not all(devices) or len(devices) != len(set(devices)):
        raise ValueError("devices must be nonempty and unique")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not set(devices).issubset(visible.split(",")):
        raise ValueError(
            "--devices must select tokens from inherited CUDA_VISIBLE_DEVICES"
        )
    return devices


def name_list(value):
    if value is None:
        return None
    path = Path(value)
    # A comma-separated selection can exceed the filesystem filename limit.
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    content = path.read_text() if is_file else value.replace(",", "\n")
    return {s for line in content.splitlines() if (s := line.split("#")[0].strip())}


def select_operators(only=None):
    if not OPS_DIR.is_dir():
        raise ValueError(f"operator directory not found: {OPS_DIR}")
    names = sorted(p.stem for p in OPS_DIR.glob("*.py") if p.stem != "__init__")
    unknown = (only or set()) - set(names)
    if unknown:
        raise ValueError(f"unknown operator modules: {sorted(unknown)}")
    return [n for n in names if only is None or n in only]


def source_digest():
    digest = hashlib.sha256()
    paths = sorted(OPS_DIR.rglob("*.py"))
    paths += sorted(OPS_DIR.rglob("*.json"))
    paths += sorted((ROOT / "tools").glob("*.py"))
    paths += [ROOT / "tools" / "hardware_specs.json"]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_process(command, environment, log_path, timeout, stop):
    """Terminate the entire group, including NCU and its Python child."""
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if stop.wait(0.2) or time.monotonic() >= deadline:
                    return "interrupted" if stop.is_set() else "timeout"
            return process.returncode
        finally:
            # NCU descendants can outlive their direct parent after a crash.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def collect(args, *, execute=run_process):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.with_suffix(output.suffix + ".lock")).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another collector owns this output") from exc
        return _collect(args, execute=execute)


def record_quality(entry):
    """Identify measured workloads and whether each has profiler evidence."""
    return {
        (shape, dtype): bool(record.get("num_kernels"))
        for shape, values in entry.get("shapes", {}).items()
        for dtype, record in values.items()
    }


def covers_previous(previous, current):
    old, new = record_quality(previous), record_quality(current)
    return old.keys() <= new.keys() and all(
        not profiled or new[key] for key, profiled in old.items()
    )


def report_directory(report_root, output, operator, attempt_directory):
    if not report_root:
        return attempt_directory / "ncu_reports"
    # Basenames alone collide for /run/a/data.json and /run/b/data.json.
    digest = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    namespace = f"{output.name[:80]}-{digest}"
    return Path(report_root).resolve() / namespace / operator / attempt_directory.name


def merge_event(results, origins, name, record, data):
    """Checkpoint latest task state without degrading a retained result snapshot.

    Running events have data=None; terminal events carry the child JSON (possibly
    empty after failure). Attempts are kept separately rather than mixing records
    collected on different GPUs/environments into one operator entry.
    """
    previous = results["_collection"]["tasks"].get(name, {})
    history = list(previous.get("previous_attempts", []))
    if data is None and previous:
        history.append(
            {
                key: value
                for key, value in previous.items()
                if key != "previous_attempts"
            }
        )
    if history:
        record["previous_attempts"] = history
    if data is None and name in results:
        record["retained_result"] = origins[name]
    if data is not None:
        candidate = data.get(name)
        if candidate is not None and (
            name not in results or covers_previous(results[name], candidate)
        ):
            results[name] = candidate
            origins[name] = {
                "device": record["device"],
                "result": record["result"],
                "environment": data.get("_environment"),
            }
        elif name in results:
            record["retained_result"] = origins[name]
        if "_environment" in data:
            results["_environments"][record["device"]] = data["_environment"]
            results.setdefault("_environment", data["_environment"])
        if "_scaling_factors" in data:
            results["_scaling_factors"] = data["_scaling_factors"]
        print(f"[{record['device']}] {name}: {record['status']}", flush=True)
    results["_collection"]["tasks"][name] = record


def _collect(args, *, execute=run_process):
    if not math.isfinite(args.op_timeout) or args.op_timeout <= 0:
        raise ValueError("--op-timeout must be finite and positive")
    devices = parse_devices(args.devices)
    names = select_operators(name_list(args.ops))
    if not names:
        raise ValueError("no operators selected")
    output = Path(args.output).resolve()
    work = output.with_suffix(output.suffix + ".work")
    signature = {
        "operators": names,
        "devices": devices,
        "op_timeout": args.op_timeout,
        "reference_chip": args.reference_chip,
        "source_sha256": source_digest(),
        "report_dir": (
            str(Path(args.report_dir).resolve())
            if getattr(args, "report_dir", None)
            else None
        ),
    }
    if output.exists() and not args.resume:
        raise ValueError("output exists; choose another path or use --resume")
    if args.resume:
        if not output.exists():
            raise ValueError("--resume requires an existing output")
        results = json.loads(output.read_text())
        if results.get("_collection", {}).get("configuration") != signature:
            raise ValueError(
                "resume configuration/source differs from the existing run"
            )
    else:
        results = {
            "_collection": {
                "configuration": signature,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "tasks": {},
            },
            "_environments": {},
        }
    tasks = results["_collection"]["tasks"]
    origins = {
        name: task.get("retained_result")
        or {
            "device": task.get("device"),
            "result": task.get("result"),
            "environment": results["_environments"].get(task.get("device")),
        }
        for name, task in tasks.items()
        if name in results
    }
    pending = queue.Queue()
    for name in names:
        if tasks.get(name, {}).get("status") != "completed":
            pending.put(name)
    work.mkdir(parents=True, exist_ok=True)
    collection = results["_collection"]
    collection.pop("finished_at_utc", None)
    collection["interrupted"] = False
    collection["attempt_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, results)
    stop = threading.Event()
    events = queue.Queue()

    def consume(device):
        while not stop.is_set():
            try:
                name = pending.get_nowait()
            except queue.Empty:
                return
            folder = work / name / uuid.uuid4().hex
            folder.mkdir(parents=True, exist_ok=True)
            part = folder / "result.json"
            command = [
                sys.executable,
                str(ROOT / "tools" / "collect_baseline_nvidia.py"),
                "--ops",
                name,
                "--output",
                str(part),
                "--report-dir",
                str(
                    report_directory(
                        getattr(args, "report_dir", None), output, name, folder
                    )
                ),
                "--op-timeout",
                str(args.op_timeout),
                "--reference-chip",
                args.reference_chip,
            ]
            # BASELINE_TASK_EXIT=1 让子采集返回 0/1 退出码（内部信号，经环境变量传入）。
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=device,
                PYTHONUNBUFFERED="1",
                BASELINE_TASK_EXIT="1",
            )
            events.put(
                (
                    name,
                    {
                        "status": "running",
                        "device": device,
                        "log": str(folder / "collector.log"),
                        "result": str(part),
                    },
                    None,
                )
            )
            try:
                code = execute(
                    command,
                    environment,
                    folder / "collector.log",
                    args.op_timeout + 120,
                    stop,
                )
                data = json.loads(part.read_text()) if part.exists() else {}
                entry = data.get(name)
                status = (
                    "failed"
                    if code != 0
                    else ("skipped" if entry is None else "completed")
                )
                if status == "completed":
                    records = [
                        r
                        for shape in entry.get("shapes", {}).values()
                        for r in shape.values()
                    ]
                    if not records:
                        status = "skipped"
                    elif any(not r.get("num_kernels") for r in records):
                        status = "profiling_incomplete"
                record = {
                    "status": status,
                    "device": device,
                    "returncode": code,
                    "log": str(folder / "collector.log"),
                    "result": str(part),
                }
                events.put((name, record, data))
            except Exception as exc:
                events.put(
                    (
                        name,
                        {
                            "status": "failed",
                            "device": device,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        {},
                    )
                )

    previous = {}

    def interrupted(signum, frame):
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(consume, device) for device in devices]
            try:
                while not all(f.done() for f in futures) or not events.empty():
                    for future in futures:
                        if future.done():
                            future.result()
                    try:
                        name, record, data = events.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    merge_event(results, origins, name, record, data)
                    atomic_json(output, results)
                for future in futures:
                    future.result()
            except BaseException:
                # Set this before executor.__exit__ joins the consumers. Otherwise
                # a broken checkpoint leaves them draining the entire GPU queue.
                stop.set()
                raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    results["_collection"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    results["_collection"]["interrupted"] = stop.is_set()
    atomic_json(output, results)
    return int(
        stop.is_set()
        or any(tasks.get(n, {}).get("status") != "completed" for n in names)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices", required=True, help="GPU visibility tokens, e.g. 0,1,2,3"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--report-dir", help="Optional report root (partitioned by output and operator)"
    )
    parser.add_argument("--ops")
    parser.add_argument("--op-timeout", type=float, default=1800)
    parser.add_argument("--reference-chip", default="H800")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.op_timeout <= 0:
        parser.error("--op-timeout must be positive")
    try:
        return collect(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
