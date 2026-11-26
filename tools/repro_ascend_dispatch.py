#!/usr/bin/env python3
"""Collect and compare Ascend generic/TensorView dispatch without a separate test process."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest


class Collector:
    def __init__(self):
        self.nodes = []

    def pytest_collection_finish(self, session):
        self.nodes = [item.nodeid for item in session.items]


class DispatchTrace:
    """Trace dispatch decisions by wrapping private eligibility and launch gates."""
    def __init__(self, force_fallback=False):
        self.events = []
        self.originals = []
        self.item = None
        self.force_fallback = force_fallback

    def _record(self, event):
        self.events.append({"nodeid": self.item.nodeid if self.item else None, **event})

    def pytest_collection_finish(self, session):
        import sys
        for name, module in tuple(sys.modules.items()):
            if not name.startswith("flaggems_vllm.runtime.backend._ascend.ops."):
                continue
            for eligible_name in ("_eligible", "_tv_eligible", "_unpack_tv_eligible"):
                eligible = getattr(module, eligible_name, None)
                if not callable(eligible):
                    continue
                def make_eligible(fn, module_name, attr_name):
                    def wrapped(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        self._record({"op_module": module_name, "route": "tensorview_eligible" if result else "generic_ineligible", "eligibility": attr_name})
                        return result
                    return wrapped
                self.originals.append((module, eligible_name, eligible))
                setattr(module, eligible_name, make_eligible(eligible, name, eligible_name))
            gate = getattr(module, "tensorview_call", None)
            if callable(gate):
                def make_gate(fn, module_name):
                    def wrapped(op_name, fast_fn, generic_fn, *args, **kwargs):
                        if self.force_fallback:
                            self._record({"op_module": module_name, "op": op_name, "route": "forced_generic_fallback"})
                            return generic_fn(*args, **kwargs)
                        def traced_fast(*a, **k):
                            self._record({"op_module": module_name, "op": op_name, "route": "tensorview_fast_attempt"})
                            try:
                                value = fast_fn(*a, **k)
                            except Exception as exc:
                                self._record({"op_module": module_name, "op": op_name, "route": "tensorview_fast_error", "error": repr(exc)})
                                raise
                            self._record({"op_module": module_name, "op": op_name, "route": "tensorview_fast_success"})
                            return value
                        def traced_generic(*a, **k):
                            self._record({"op_module": module_name, "op": op_name, "route": "generic_fallback"})
                            return generic_fn(*a, **k)
                        return fn(op_name, traced_fast, traced_generic, *args, **kwargs)
                    return wrapped
                self.originals.append((module, "tensorview_call", gate))
                setattr(module, "tensorview_call", make_gate(gate, name))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        self.item = item
        try:
            yield
        finally:
            self.item = None

    def pytest_unconfigure(self, config):
        for module, attr, original in self.originals:
            setattr(module, attr, original)


class Recorder:
    def __init__(self, report: Path, mode: str, tracer: DispatchTrace | None):
        self.report = report
        self.mode = mode
        self.tracer = tracer
        self.nodes = {}

    def pytest_runtest_logreport(self, report):
        entry = self.nodes.setdefault(report.nodeid, {})
        entry[report.when] = report.outcome
        if report.when == "setup" and report.skipped:
            entry["result"] = "skipped"
            entry["reason"] = str(report.longrepr)
        elif report.when == "call":
            entry["result"] = report.outcome
            if report.failed or report.skipped:
                entry["reason"] = str(report.longrepr)
        if report.failed and report.when != "call":
            entry["reason"] = str(report.longrepr)

    def pytest_sessionfinish(self, session, exitstatus):
        existing = json.loads(self.report.read_text()) if self.report.exists() else {}
        existing[self.mode] = {
            "exit_status": int(exitstatus),
            "nodes": self.nodes,
            "dispatch_events": self.tracer.events if self.tracer else [],
        }
        self.report.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", help="pytest target paths")
    parser.add_argument("--output-dir", type=Path, default=Path("ascend-dispatch-repro"))
    parser.add_argument("--execute", action="store_true", help="execute on configured hardware")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    targets = [str(root / t) if not Path(t).is_absolute() else t for t in args.targets]
    collector = Collector()
    status = pytest.main(["-q", "--collect-only", *targets], plugins=[collector])
    (out / "collection.log").write_text(
        f"pytest collection exit status: {int(status)}\ncollected nodes: {len(collector.nodes)}\n",
        encoding="utf-8",
    )
    if int(status) != 0 or not collector.nodes:
        print(f"Collection failed or returned zero nodes; see {out / 'collection.log'}", file=sys.stderr)
        return 2
    (out / "nodeids.txt").write_text("\n".join(collector.nodes) + "\n", encoding="utf-8")
    print(f"Collected {len(collector.nodes)} nodes: {out / 'nodeids.txt'}")
    if not args.execute:
        print("Collection only. --execute runs tests on hardware and captures TensorView routes.")
        return 0
    if os.environ.get("FLAGGEMS_VENDOR", os.environ.get("DNN_VENDOR", "")).lower() != "ascend":
        print("Refusing execution: set FLAGGEMS_VENDOR=ascend on an Ascend host.", file=sys.stderr)
        return 2
    base = (["--quick"] if args.quick else []) + collector.nodes
    results = []
    for mode, force_fallback in (("ascend_native_dispatch", False), ("ascend_forced_generic_fallback", True)):
        os.environ["FLAGGEMS_ASCEND_REPRO_MODE"] = mode
        tracer = DispatchTrace(force_fallback=force_fallback)
        recorder = Recorder(out / "results.json", mode, tracer)
        tracer.__pytest_plugin_name__ = "repro_ascend_trace"
        recorder.__pytest_plugin_name__ = "repro_ascend_recorder"
        results.append(int(pytest.main([
            "-q", "--repro-ascend-report", str(out / "results.json"), *base
        ], plugins=[tracer, recorder])))
    return int(any(results))


if __name__ == "__main__":
    raise SystemExit(main())
