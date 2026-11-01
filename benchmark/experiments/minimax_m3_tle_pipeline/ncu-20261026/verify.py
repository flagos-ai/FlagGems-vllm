# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Verify the NCU follow-up and optionally prepare its GPU source workspace."""

import argparse
import gzip
import hashlib
import io
import json
import math
import statistics
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    data = (ROOT / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    args = parser.parse_args()
    manifest = read("MANIFEST.json")
    for name, expected in manifest["files"].items():
        assert sha((ROOT / name).read_bytes()) == expected, name
    with tarfile.open(
        fileobj=io.BytesIO(
            gzip.decompress((ROOT / "measured_sources.tar.gz").read_bytes())
        )
    ) as tar:
        members = tar.getmembers()
        assert all(member.isfile() for member in members)
        sources = {member.name: tar.extractfile(member).read() for member in members}
    assert set(sources) == set(manifest["source_sha256"])
    for name, expected in manifest["source_sha256"].items():
        assert sha(sources[name]) == expected, name
    assert sha(sources["reference/baseline.py"]) == manifest["baseline_sha256"]
    assert sha(sources["reference/flagattention_pr85.py"]) == (
        "6c7bb5b5768b6c7c925da9457c20d812195a5b7f66c2fc9608d071d86ae64a60"
    )

    summary = read("data/ncu-summary.json.gz")
    exports = {}
    actions = 0
    for name, expected in manifest["ncu_export_sha256"].items():
        data = gzip.decompress((ROOT / name).read_bytes())
        assert sha(data) == expected, name
        export = json.loads(data)
        tag = Path(name).name.removesuffix(".json.gz")
        exports[tag] = export
        assert export["report_sha256"] == summary[tag]["report_sha256"]
        assert len(export["actions"]) == len(summary[tag]["actions"])
        for exported, recorded in zip(export["actions"], summary[tag]["actions"]):
            assert exported["name"] == recorded["name"]
            assert sha(exported["metrics_json"].encode()) == exported["metrics_sha256"]
            assert (
                sha(exported["source_stalls_text"].encode())
                == exported["source_stalls_sha256"]
            )
            metrics = json.loads(exported["metrics_json"])
            assert all(
                metrics[key] == value for key, value in recorded["metrics"].items()
            )
            actions += 1
    assert len(exports) == len(summary) == manifest["binary_report_count"] == 20
    assert actions == manifest["ncu_actions"] == 45

    derived = read("data/metrics.json")
    derived_timings = {row["receipt"]: row for row in derived["timings"]}
    calls = 0
    rejected = 0
    accepted = 0
    for name, expected in manifest["timing_sha256"].items():
        data = gzip.decompress((ROOT / name).read_bytes())
        assert sha(data) == expected, name
        record = json.loads(data)
        if record["status"] == "NUMERICAL_REJECTION":
            assert not record["accepted"]
            assert record["configuration"]["arm"] == "fa85"
            assert record["configuration"]["storage"] == "fp8"
            assert "eager" not in record and "graph" not in record
            rejected += 1
            continue
        assert record["accepted"]
        accepted += 1
        if record["configuration"]["arm"] != "fa85":
            assert record["check"]["bitwise"]
        for mode in ("eager", "graph"):
            result = record[mode]
            raw = result["raw"]
            counts = Counter((row["round"], row["arm"]) for row in raw)
            assert len(counts) == 6 and set(counts.values()) == {100}
            ratios = []
            for round_index in (None, 0, 1, 2):
                medians = {
                    arm: statistics.median(
                        row["gpu_us"]
                        for row in raw
                        if row["arm"] == arm
                        and (round_index is None or row["round"] == round_index)
                    )
                    for arm in ("baseline", "candidate")
                }
                ratios.append(medians["baseline"] / medians["candidate"])
            assert math.isclose(ratios[0], result["speedup"], rel_tol=1e-12)
            assert all(
                math.isclose(a, b, rel_tol=1e-12)
                for a, b in zip(ratios[1:], result["round_speedups"])
            )
            if record["configuration"]["arm"] != "fa85":
                assert all(ratio < 1 for ratio in ratios)
            calls += len(raw)
        row = derived_timings[Path(name).name.removesuffix(".gz")]
        assert row["graph_speedup"] == record["graph"]["speedup"]
        for arm, field in (
            ("baseline", "baseline_graph_us"),
            ("candidate", "candidate_graph_us"),
        ):
            assert row[field] == record["graph"]["summary"][arm]["gpu_us"]["median"]
    assert accepted == 12 and rejected == 3
    assert calls == derived["timed_wrapper_invocations"] == 14400

    def counter(tag, name):
        metrics = json.loads(exports[tag]["actions"][0]["metrics_json"])
        return metrics[name]["value"]

    ratio = counter(
        "extra-fa85-bf16-contig", "lts__t_sectors_srcunit_tex_op_read.sum"
    ) / counter("extra-baseline-bf16-contig", "lts__t_sectors_srcunit_tex_op_read.sum")
    assert ratio == derived["findings"]["bf16_compact_fa85_over_union_l2_read_sectors"]
    oracle = read("data/oracle-control.json")
    assert oracle == derived["oracle_checks"]
    assert all(row["baseline_vs_literal"]["pass_"] for row in oracle)
    assert [
        row["fa85_vs_literal"]["failures"] for row in oracle if row["storage"] == "fp8"
    ] == [635, 632, 525]
    assert all(
        row["fa85_vs_literal"]["pass_"] for row in oracle if row["storage"] == "bf16"
    )

    if args.prepare:
        destination = args.prepare.resolve()
        if destination.exists() and any(destination.iterdir()):
            raise SystemExit("Choose a new or empty preparation directory")
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in sources.items():
            target = (destination / name).resolve()
            assert destination in target.parents
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for folder in ("receipts", "reports", "analysis"):
            (destination / folder).mkdir(exist_ok=True)
    print(
        json.dumps(
            {
                "status": "PASS_CPU_ONLY_NCU_EXPORT_VERIFICATION",
                "reports": len(exports),
                "kernel_actions": actions,
                "accepted_timing_pairs": accepted,
                "unscored_fp8_fa85_pairs": rejected,
                "timed_wrapper_invocations": calls,
                "bf16_compact_l2_sector_ratio": ratio,
                "binary_reports_included": False,
                "gpu_work_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
