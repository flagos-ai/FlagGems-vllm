# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Verify source/evidence identities and recalculate every retained timing ratio."""

import ast
import gzip
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def functions(data):
    tree = ast.parse(data)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with tarfile.open(
        fileobj=io.BytesIO(
            gzip.decompress((ROOT / "measured_sources.tar.gz").read_bytes())
        )
    ) as tar:
        sources = {member.name: tar.extractfile(member).read() for member in tar}
    for name, expected in manifest["measured_source_sha256"].items():
        assert hashlib.sha256(sources[name]).hexdigest() == expected, name
    available = {hashlib.sha256(data).hexdigest() for data in sources.values()}
    with tarfile.open(
        fileobj=io.BytesIO(
            gzip.decompress((ROOT / "source_history.tar.gz").read_bytes())
        )
    ) as tar:
        for member in tar:
            data = tar.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == manifest["historical_source_sha256"][member.name]
            available.add(digest)
    for name, expected in manifest["receipt_source_sha256"].items():
        data = (ROOT / name).read_bytes()
        receipt = json.loads(gzip.decompress(data) if name.endswith(".gz") else data)
        assert receipt["identity"]["sources"] == expected, name
        assert set(expected.values()) <= available, name
    for name in ("tle_pipeline.py", "tle_nows.py"):
        assert functions((ROOT / "kernels" / name).read_bytes()) == functions(
            sources["src/" + name]
        ), name
    library = ROOT.parents[2] / "src/flaggems_vllm/ops/minimax_m3_sparse_attn.py"
    assert (
        hashlib.sha256(library.read_bytes()).hexdigest()
        == manifest["pr979_operator_sha256"]
    ), "Existing PR operator changed"

    summary = json.loads((ROOT / "data/summary.json").read_text())
    calls = 0
    graph_count = 0
    stress_replays = 0
    graph_ratios = []
    for variant in summary["variants"]:
        data = gzip.decompress((ROOT / variant["benchmark"]).read_bytes())
        assert hashlib.sha256(data).hexdigest() == variant["benchmark_sha256"]
        bench = json.loads(data)
        gate_data = gzip.decompress((ROOT / variant["correctness"]).read_bytes())
        assert hashlib.sha256(gate_data).hexdigest() == variant["correctness_sha256"]
        gate = json.loads(gate_data)
        assert gate["status"] == "PASS"
        assert all(case["check"]["pass_"] for case in gate["cases"])
        assert variant["all_graphs_pass"] and variant["all_eager_bitwise"]
        contexts = gate["graph"] if isinstance(gate["graph"], list) else [gate["graph"]]
        for context in contexts:
            assert context["pass_"]
            graph_count += len(context["streams"])
            stress_replays += (
                len(context["streams"]) * context["stress_replays_per_graph"]
            )
        for case in bench["cases"]:
            assert case["check"]["bitwise"]
            for mode in ("eager", "graph"):
                result = case[mode]
                raw = result["raw"]
                counts = Counter((row["round"], row["arm"]) for row in raw)
                assert len(counts) == 6 and set(counts.values()) == {100}
                medians = {
                    arm: statistics.median(
                        row["gpu_us"] for row in raw if row["arm"] == arm
                    )
                    for arm in ("baseline", "candidate")
                }
                ratio = medians["baseline"] / medians["candidate"]
                assert math.isclose(ratio, result["speedup"], rel_tol=1e-12)
                for round_index in range(3):
                    round_medians = {
                        arm: statistics.median(
                            row["gpu_us"]
                            for row in raw
                            if row["arm"] == arm and row["round"] == round_index
                        )
                        for arm in ("baseline", "candidate")
                    }
                    round_ratio = round_medians["baseline"] / round_medians["candidate"]
                    assert math.isclose(
                        round_ratio,
                        result["round_speedups"][round_index],
                        rel_tol=1e-12,
                    )
                    assert round_ratio < 1
                calls += len(raw)
                if mode == "graph":
                    graph_ratios.append(ratio)
    assert calls == summary["timed_wrapper_invocations"] == 28800
    assert len(graph_ratios) == 24
    assert graph_count == 22
    assert stress_replays == summary["stress_graph_replays"] == 2200
    subprocess.run([sys.executable, str(ROOT / "ncu-20261026/verify.py")], check=True)
    print(
        json.dumps(
            {
                "status": "PASS_OFFLINE_ARTIFACT_VERIFICATION",
                "decision": summary["status"],
                "variants": len(summary["variants"]),
                "timed_wrapper_invocations": calls,
                "correctness_graphs": graph_count,
                "stress_graph_replays": stress_replays,
                "graph_speedup_range": [min(graph_ratios), max(graph_ratios)],
                "existing_pr_operator_unchanged": True,
                "gpu_work_started": False,
                "ncu_followup_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
