# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MegaMoE benchmarks for NVIDIA Hopper and MetaX backends."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

_MEGAMOE_BACKEND = os.environ.get("DNN_VENDOR", "").lower()
if not _MEGAMOE_BACKEND:
    try:
        _MEGAMOE_BACKEND = "metax" if "metax" in version("torch").lower() else "nvidia"
    except PackageNotFoundError:
        _MEGAMOE_BACKEND = "nvidia"


if _MEGAMOE_BACKEND == "metax":
    import argparse
    import collections
    import csv
    import ctypes
    import hashlib
    import io
    import json
    import math
    import os
    import re
    import signal
    import statistics
    import subprocess
    import sys
    import tempfile
    from contextlib import contextmanager
    from dataclasses import replace
    from functools import lru_cache
    from pathlib import Path
    from typing import Callable, Optional

    # Copyright 2026 FlagOS Contributors
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    _summary_TOKENS = [
        1,
        2,
        4,
        8,
        16,
        *range(24, 257, 8),
        *range(272, 513, 16),
        2048,
        16384,
    ]
    _summary_EPS = (2, 4, 8)
    _summary_CANONICAL_TOKENS = tuple(_summary_TOKENS)
    _summary_GRAPH_PROTOCOL = "pr774-full-ep-both-cudagraph-5-mean-v2"
    _summary_BASELINE_TIMING_PROTOCOL = "pr774-vllm-native-apply-events-5-mean-v1"
    _summary_REPEATS = 5
    _summary_TIMING_PROTOCOL = "pr774-isolated-events-5-mean-v1"
    _summary_BAD_FIELDS = (
        "combine_bad",
        "reduction_bound_bad",
        "dispatch_metadata_bad",
        "dispatch_route_set_bad",
        "dispatch_expert_bad",
        "dispatch_token_bad",
        "dispatch_scale_bad",
        "dispatch_weight_bad",
        "native_q_bad",
        "native_sf_bad",
        "native_metadata_bad",
        "native_partial_bad",
    )
    _summary_NONFINITE_FIELDS = (
        "output_nonfinite",
        "native_sf_nonfinite",
        "native_partial_nonfinite",
    )
    _summary_BENCH = re.compile(
        r"BENCH_REPEAT\s+rank=(\d+)\s+tokens=(\d+)\s+repeat=(\d+)\s+mean_us=([0-9.eE+-]+)"
    )

    def _summary_number(value, positive=False):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and (not positive or value > 0)
        )

    def _summary_read_source(path, sources):
        if not path.is_file():
            sources[path.name] = {"exists": False}
            return None
        data = path.read_bytes()
        sources[path.name] = {
            "exists": True,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        return data.decode("utf-8", errors="replace")

    def _summary_json_objects(text):
        """Accept MPI prefixes and multiple adjacent JSON objects on one line."""
        decoder = json.JSONDecoder()
        for line in text.splitlines():
            offset = 0
            while True:
                start = line.find("{", offset)
                if start < 0:
                    break
                try:
                    value, end = decoder.raw_decode(line, start)
                except ValueError:
                    offset = start + 1
                    continue
                offset = end
                if isinstance(value, dict):
                    yield value

    def _summary_validate_correctness(record):
        errors = []
        formula = record.get("quantization_contract") == "symmetric-int8-rne-v1"
        prefix = "formula" if formula else "native"
        for key in (
            "input_hashes_match",
            f"{prefix}_checked",
            f"{prefix}_exact",
            "output_ok",
            "dispatch_ok",
        ):
            if record.get(key) is not True:
                errors.append(f"{key} is not true")
        if record.get("status") != "PASS":
            errors.append("status is not PASS")
        error = record.get("rel_l2")
        if not _summary_number(error) or not 0 <= error <= 0.005:
            errors.append("rel_l2 is missing, nonfinite, or outside [0,0.005]")
        for original_key in (*_summary_BAD_FIELDS, *_summary_NONFINITE_FIELDS):
            key = original_key.replace("native_", f"{prefix}_")
            if not _summary_number(record.get(key)) or record[key] != 0:
                errors.append(f"{key} is missing or nonzero")
        for key, value in record.items():
            if key.endswith("_bad") and key not in _summary_BAD_FIELDS:
                if key == "input_hash_bad":
                    if value != []:
                        errors.append("input_hash_bad is not empty")
                elif not _summary_number(value) or value != 0:
                    errors.append(f"{key} is nonzero or nonnumeric")
            if key.endswith("_error") and value:
                errors.append(f"{key}: {value}")
        if record.get("dispatch_expected_routes") != record.get(
            "dispatch_actual_routes"
        ):
            errors.append("dispatch route counts differ")
        return errors

    def _summary_validate_shape(record):
        errors = []
        if record.get("status") != "PASS":
            errors.append("status is not PASS")
        checked = (
            "formula_checked"
            if record.get("quantization_contract") == "symmetric-int8-rne-v1"
            else "native_checked"
        )
        for key in ("input_hashes_match", checked, "pre_passed", "post_passed"):
            if record.get(key) is not True:
                errors.append(f"{key} is not true")
        if record.get("input_hash_bad") != []:
            errors.append("input_hash_bad is missing or not empty")
        for key, value in record.items():
            if key.endswith("_bad") and key != "input_hash_bad":
                if not _summary_number(value) or value != 0:
                    errors.append(f"{key} is nonzero or nonnumeric")
            if key.endswith("_error") and value:
                errors.append(f"{key}: {value}")
        return errors

    def _summary_summarize_ep(directory, ep):
        sources, errors = {}, []
        baseline_by_token = collections.defaultdict(list)
        measurements = collections.defaultdict(list)
        records = collections.defaultdict(list)
        exits = {}
        for kind in ("baseline", "candidate"):
            raw = _summary_read_source(directory / f"{kind}-np{ep}.exit", sources)
            try:
                exits[kind] = int(raw.strip()) if raw is not None else None
            except ValueError:
                exits[kind] = None
                errors.append(f"{kind} exit file is not an integer")
            if exits[kind] not in (None, 0):
                errors.append(f"{kind} exited with {exits[kind]}")

        raw = _summary_read_source(directory / f"baseline-np{ep}.json", sources)
        baseline = []
        if raw is not None:
            try:
                baseline = json.loads(raw)
                if not isinstance(baseline, list):
                    raise ValueError("expected a list of baseline records")
            except ValueError as error:
                errors.append(f"baseline JSON: {error}")
                baseline = []
        for record in baseline:
            if not isinstance(record, dict):
                errors.append("baseline contains a non-object record")
                continue
            token = record.get("tokens_per_rank")
            if token not in _summary_TOKENS or record.get("world_size") != ep:
                errors.append(
                    f"baseline EP/shape mismatch: world={record.get('world_size')}, T={token}"
                )
                continue
            baseline_by_token[token].append(record)

        raw = _summary_read_source(directory / f"candidate-np{ep}.log", sources) or ""
        for match in _summary_BENCH.finditer(raw):
            rank, token, repeat = map(int, match.groups()[:3])
            if (
                rank not in range(ep)
                or token not in _summary_TOKENS
                or repeat not in range(_summary_REPEATS)
            ):
                errors.append(
                    f"unexpected BENCH_REPEAT rank={rank}, T={token}, repeat={repeat}"
                )
                continue
            try:
                mean = float(match[4]) / 1000.0
            except ValueError:
                mean = None
            measurements[token, rank, repeat].append(mean)
        for record in _summary_json_objects(raw):
            kind = record.get("kind")
            if kind not in ("correctness", "shape_summary", "benchmark"):
                continue
            token, rank = record.get("tokens"), record.get("rank")
            if (
                token not in _summary_TOKENS
                or rank not in range(ep)
                or record.get("world") != ep
            ):
                errors.append(
                    f"candidate EP/shape/rank mismatch: {kind}, world={record.get('world')}, T={token}, rank={rank}"
                )
                continue
            phase = record.get("correctness_phase") if kind == "correctness" else kind
            if phase not in ("pre", "post", "shape_summary", "benchmark"):
                errors.append(f"unexpected correctness phase: {phase}")
                continue
            records[token, rank, phase].append(record)

        rows = []
        for token in _summary_TOKENS:
            row_errors, baseline_errors, timing_errors = [], [], []
            base_records = baseline_by_token[token]
            base_means = {}
            for record in base_records:
                repeat = record.get("repeat")
                if repeat not in range(_summary_REPEATS) or repeat in base_means:
                    baseline_errors.append(f"invalid or duplicate repeat {repeat}")
                    continue
                performance = record.get("performance") or {}
                baseline_protocol = (
                    _summary_GRAPH_PROTOCOL
                    if _summary_TIMING_PROTOCOL == _summary_GRAPH_PROTOCOL
                    else _summary_BASELINE_TIMING_PROTOCOL
                )
                if performance.get("timing_protocol") != baseline_protocol:
                    baseline_errors.append(
                        f"repeat {repeat}: baseline timing protocol mismatch"
                    )
                mean = performance.get("primary_latency_ms")
                base_means[repeat] = (
                    mean if _summary_number(mean, positive=True) else None
                )
                if not _summary_number(mean, positive=True):
                    baseline_errors.append(
                        f"repeat {repeat}: latency is not positive finite"
                    )
                correctness = record.get("correctness") or {}
                if correctness.get("pass") is not True:
                    baseline_errors.append(
                        f"repeat {repeat}: correctness.pass is not true"
                    )
                rel_l2 = correctness.get("relative_l2_error")
                if not _summary_number(rel_l2) or not 0 <= rel_l2 <= 0.005:
                    baseline_errors.append(
                        f"repeat {repeat}: baseline rel_l2 outside [0,0.005]"
                    )
                shape = json.loads((directory / "manifest.json").read_text())["shape"]
                dimensions = {
                    "hidden_size": shape["hidden_size"],
                    "intermediate_size": shape["moe_intermediate_size"],
                    "num_experts": shape["num_experts"],
                    "local_experts": shape["num_experts"] // ep,
                    "top_k": shape["num_experts_per_tok"],
                }
                for key, expected in dimensions.items():
                    if record.get(key) != expected:
                        baseline_errors.append(f"repeat {repeat}: {key} != {expected}")
                if (
                    performance.get("warmup") != 20
                    or performance.get("iterations") != 100
                ):
                    baseline_errors.append(
                        f"repeat {repeat}: expected warmup=20, iterations=100"
                    )
                if record.get("first_input_quantization_in_timed_region") is not False:
                    baseline_errors.append(
                        f"repeat {repeat}: initial quantization timing differs"
                    )
                means = performance.get("per_rank_mean_ms")
                if (
                    not isinstance(means, list)
                    or len(means) != ep
                    or not all(_summary_number(x, positive=True) for x in means)
                ):
                    baseline_errors.append(
                        f"repeat {repeat}: per-rank means missing or invalid"
                    )
                elif not _summary_number(mean) or not math.isclose(
                    mean, max(means), rel_tol=1e-9, abs_tol=1e-12
                ):
                    baseline_errors.append(
                        f"repeat {repeat}: primary latency != max rank mean"
                    )
            if len(base_records) != _summary_REPEATS or set(base_means) != set(
                range(_summary_REPEATS)
            ):
                baseline_errors.append(
                    f"expected 5 unique repeats, found {len(base_records)} records"
                )

            rank_means = {}
            for rank in range(ep):
                for phase in ("pre", "post", "shape_summary"):
                    items = records[token, rank, phase]
                    if len(items) != 1:
                        row_errors.append(
                            f"rank {rank}: expected one {phase}, found {len(items)}"
                        )
                    for record in items:
                        validator = (
                            _summary_validate_shape
                            if phase == "shape_summary"
                            else _summary_validate_correctness
                        )
                        row_errors.extend(
                            f"rank {rank} {phase}: {error}"
                            for error in validator(record)
                        )
                for repeat in range(_summary_REPEATS):
                    values = measurements[token, rank, repeat]
                    if len(values) != 1 or not _summary_number(
                        values[0], positive=True
                    ):
                        timing_errors.append(
                            f"rank {rank} repeat {repeat}: expected one positive finite measurement, found {values}"
                        )
                    else:
                        rank_means[rank, repeat] = values[0]
                items = records[token, rank, "benchmark"]
                if len(items) != 1:
                    timing_errors.append(
                        f"rank {rank}: expected one benchmark record, found {len(items)}"
                    )
                for record in items:
                    if record.get("timing_protocol") != _summary_TIMING_PROTOCOL:
                        timing_errors.append(
                            f"rank {rank}: candidate timing protocol mismatch"
                        )
                    if (
                        record.get("warmup") != 20
                        or record.get("iters") != 100
                        or record.get("repeats") != _summary_REPEATS
                    ):
                        timing_errors.append(
                            f"rank {rank}: benchmark warmup/iters/repeats differs"
                        )
                    if record.get("correctness") != "PASS":
                        timing_errors.append(
                            f"rank {rank}: benchmark correctness is not PASS"
                        )
                    reported = record.get("mean_us")
                    if (
                        not isinstance(reported, list)
                        or len(reported) != _summary_REPEATS
                    ):
                        timing_errors.append(f"rank {rank}: benchmark means missing")
                    else:
                        for repeat, value in enumerate(reported):
                            measured = rank_means.get((rank, repeat))
                            if (
                                not _summary_number(value, positive=True)
                                or measured is None
                                or not math.isclose(
                                    value / 1000.0,
                                    measured,
                                    rel_tol=1e-8,
                                    abs_tol=1e-10,
                                )
                            ):
                                timing_errors.append(
                                    f"rank {rank} repeat {repeat}: JSON/line means differ"
                                )
            base_values = [base_means.get(k) for k in range(_summary_REPEATS)]
            maxima = (
                [
                    max(rank_means[r, k] for r in range(ep))
                    for k in range(_summary_REPEATS)
                ]
                if not timing_errors
                else []
            )
            base = statistics.mean(base_values) if not baseline_errors else None
            candidate = statistics.mean(maxima) if maxima else None
            ratio = (
                base / candidate if base is not None and candidate is not None else None
            )
            rows.append(
                {
                    "tokens_per_rank": token,
                    "vllm_ms": base,
                    "megamoe_ms": candidate,
                    "speedup": ratio,
                    "correctness": not row_errors,
                    "baseline_valid": not baseline_errors,
                    "timing_valid": not timing_errors,
                    "vllm_repeat_max_ms": base_values,
                    "megamoe_repeat_max_ms": maxima,
                    "errors": {
                        "candidate_correctness": row_errors,
                        "candidate_timing": timing_errors,
                        "baseline": baseline_errors,
                    },
                }
            )
        ratios = [row["speedup"] for row in rows if row["speedup"] is not None]
        if _summary_TIMING_PROTOCOL == _summary_GRAPH_PROTOCOL:
            errors.extend(_summary_validate_graph(directory, ep))
        all_rows_valid = all(
            row["correctness"] and row["baseline_valid"] and row["timing_valid"]
            for row in rows
        )
        complete = (
            all_rows_valid
            and not errors
            and all(value == 0 for value in exits.values())
        )
        mean = statistics.mean(ratios) if len(ratios) == len(_summary_TOKENS) else None
        target_pass = bool(complete and mean >= 0.95)
        finished = all(value is not None for value in exits.values())
        return {
            "ranks": ep,
            "expected_shapes": len(_summary_TOKENS),
            "correctness_passed_shapes": sum(row["correctness"] for row in rows),
            "baseline_valid_shapes": sum(row["baseline_valid"] for row in rows),
            "timing_valid_shapes": sum(row["timing_valid"] for row in rows),
            "paired_timing_shapes": len(ratios),
            "complete": complete,
            "target_pass": target_pass,
            "status": "PASS" if target_pass else "FAIL" if finished else "INCOMPLETE",
            "arithmetic_mean_speedup": mean,
            "observed_arithmetic_mean_speedup": (
                statistics.mean(ratios) if ratios else None
            ),
            "geometric_mean_speedup": (
                math.exp(statistics.mean(map(math.log, ratios)))
                if len(ratios) == len(_summary_TOKENS)
                else None
            ),
            "min_speedup": min(ratios) if len(ratios) == len(_summary_TOKENS) else None,
            "num_below_0_95": (
                sum(ratio < 0.95 for ratio in ratios)
                if len(ratios) == len(_summary_TOKENS)
                else None
            ),
            "total_latency_ratio": (
                sum(row["vllm_ms"] for row in rows)
                / sum(row["megamoe_ms"] for row in rows)
                if len(ratios) == len(_summary_TOKENS)
                else None
            ),
            "acceptance": (
                "Each EP: all 53 shapes correct; mean(vLLM_ms / MegaMoE_ms) >= 0.95. "
                "Latency: mean of 5 repeats of max-rank mean, 20 warmup, 100 iterations."
            ),
            "exit_codes": exits,
            "errors": errors,
            "sources": sources,
            "rows": rows,
        }

    def _summary_validate_graph(directory, ep):
        """Require complete native communication and dynamic replay checks on all ranks."""
        errors = []
        phases = {
            "original_pre",
            "remote_input_changed",
            "remote_routes_changed",
            "original_restored_pre",
            "post_measure",
        }
        for kind in ("baseline", "candidate"):
            expected = [
                t
                for t in _summary_TOKENS
                for _ in range(_summary_REPEATS if kind == "baseline" else 1)
            ]
            for rank in range(ep):
                path = directory / f"audit-{kind}/np{ep}-rank{rank}.jsonl"
                try:
                    records = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    assert [r["tokens"] for r in records] == expected
                    for r in records:
                        assert (
                            r["rank"] == rank
                            and r["world"] == ep
                            and r["capture_complete"]
                        )
                        assert r["timing_protocol"] == _summary_GRAPH_PROTOCOL
                        assert len(r["checks"]) == len(phases)
                        assert {c["phase"] for c in r["checks"]} == phases
                        assert all(
                            c["finite"] and c["bit_mismatches"] == 0
                            for c in r["checks"]
                        )
                        if kind == "baseline":
                            assert r["captured_collectives"] == {
                                "all_gather": 2,
                                "reduce_scatter": 1,
                            }
                        if rank == 0:
                            assert all(
                                c["changed_from_original"] > 0
                                for c in r["checks"]
                                if c["phase"].startswith("remote_")
                            )
                except (OSError, ValueError, KeyError, AssertionError) as error:
                    errors.append(
                        f"Graph audit {kind} rank {rank}: {type(error).__name__}: {error}"
                    )
        return errors

    def _summary_write_text(path, text):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)

    def _summary_write_report(directory, result):
        ep = result["ranks"]
        _summary_write_text(
            directory / f"summary-np{ep}.json",
            json.dumps(result, indent=2, allow_nan=False) + "\n",
        )
        text = io.StringIO()
        fields = [
            "tokens_per_rank",
            "vllm_ms",
            "megamoe_ms",
            "speedup",
            "correctness",
            "baseline_valid",
            "timing_valid",
        ]
        writer = csv.DictWriter(text, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in result["rows"])
        _summary_write_text(directory / f"summary-np{ep}.csv", text.getvalue())

    def _summary_main():
        global _summary_TOKENS, _summary_TIMING_PROTOCOL
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("directory", type=Path)
        parser.add_argument("--eps", type=int, nargs="+", choices=_summary_EPS)
        parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
        parser.add_argument(
            "--tokens", default=",".join(map(str, _summary_CANONICAL_TOKENS))
        )
        args = parser.parse_args()
        _summary_TOKENS = list(map(int, args.tokens.split(",")))
        _summary_TIMING_PROTOCOL = (
            _summary_GRAPH_PROTOCOL
            if args.mode == "graph"
            else "pr774-isolated-events-5-mean-v1"
        )
        if not args.directory.is_dir():
            parser.error("result directory does not exist")
        eps = sorted(
            set(
                args.eps
                or [
                    ep
                    for ep in _summary_EPS
                    if any(
                        (args.directory / name).exists()
                        for name in (
                            f"baseline-np{ep}.json",
                            f"candidate-np{ep}.log",
                            f"baseline-np{ep}.exit",
                            f"candidate-np{ep}.exit",
                        )
                    )
                ]
            )
        )
        results = {}
        for ep in eps:
            result = _summary_summarize_ep(args.directory, ep)
            _summary_write_report(args.directory, result)
            results[str(ep)] = {
                key: value for key, value in result.items() if key != "rows"
            }
            mean = result["arithmetic_mean_speedup"]
            print(
                f"EP={ep} {result['status']} "
                f"correctness={result['correctness_passed_shapes']}/{len(_summary_TOKENS)} "
                f"timing={result['timing_valid_shapes']}/{len(_summary_TOKENS)} "
                f"baseline={result['baseline_valid_shapes']}/{len(_summary_TOKENS)} "
                f"mean={mean if mean is not None else 'pending'} "
                f"target_pass={result['target_pass']}"
            )
        matrix_complete = bool(eps) and all(
            item["complete"] for item in results.values()
        )
        summary = {
            "required_eps": list(_summary_EPS),
            "mode": args.mode,
            "tokens": _summary_TOKENS,
            "full_53_shape_ep248": set(eps) == set(_summary_EPS)
            and tuple(_summary_TOKENS) == _summary_CANONICAL_TOKENS,
            "observed_eps": eps,
            "complete": matrix_complete,
            "target_pass": bool(
                matrix_complete
                and all(item["target_pass"] for item in results.values())
            ),
            "results": results,
        }
        _summary_write_text(
            args.directory / "summary.json",
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
        )
        print(
            f"MATRIX complete={summary['complete']} target_pass={summary['target_pass']} observed_eps={eps}"
        )

    # Copyright 2026 FlagOS Contributors
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    ROOT = Path(__file__).resolve().parents[1]
    LOCAL = Path(__file__).resolve().parent

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run(command, *, name, output, env, timeout):
        (output / f"{name}.command.json").write_text(
            json.dumps(
                dict(
                    command=command,
                    devices=env["CUDA_VISIBLE_DEVICES"],
                ),
                indent=2,
            )
            + "\n"
        )
        print(f"START {output.name}/{name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                rc = 124
        (output / f"{name}.exit").write_text(f"{rc}\n")
        if rc:
            raise RuntimeError(f"{name} exited {rc}; see {output / (name + '.log')}")
        print(f"PASS {output.name}/{name}", flush=True)

    def validate_runtime_preparation(directory, ep, protocol_id, devices):
        for phase in ("baseline", "candidate"):
            folder = directory / f"runtime-{phase}-np{ep}" / "resource-preinit"
            paths = [folder / "ready" / f"rank{rank}.json" for rank in range(ep)]
            hashes = {str(rank): digest(path) for rank, path in enumerate(paths)}
            for rank, path in enumerate(paths):
                ready = json.loads(path.read_text())
                certificate = json.loads(
                    (folder / "certificates" / f"rank{rank}.json").read_text()
                )
                assert (
                    ready["protocol_sha256"],
                    ready["phase"],
                    ready["rank"],
                    str(ready["gpu"]),
                ) == (protocol_id, phase, rank, str(devices[rank]))
                assert ready["implementation_sha256"] == digest(
                    LOCAL / "test_megamoe.py"
                )
                assert ready["setup_complete"] and ready["all_events_complete"]
                assert (
                    ready["outside_all_timers"] and ready["stream_pool_requests"] == 32
                )
                assert ready["setup_cuda_events"] == []
                assert certificate["ready_sha256"] == hashes
                assert (
                    certificate["protocol_sha256"] == protocol_id
                    and certificate["before_target_import"]
                )

            if phase == "candidate":
                expected = json.loads((directory / "manifest.json").read_text())[
                    "source_sha256"
                ]
                for rank in range(ep):
                    proof = json.loads(
                        (
                            folder.parent / f"candidate-source-rank{rank}.json"
                        ).read_text()
                    )
                    assert proof["rank"] == rank and proof["protocol_id"] == protocol_id
                    assert set(proof["files"]) == {"kernel", "launch_args", "factory"}
                    for name, path in proof["files"].items():
                        assert (
                            proof["sha256"][name]
                            == expected[path]
                            == digest(Path(path))
                        )

    def validate_kernel_profiles(directory, ep, tokens):
        for phase in ("baseline", "candidate"):
            for rank in range(ep):
                path = directory / f"audit-{phase}" / f"np{ep}-rank{rank}.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                expected = [
                    token
                    for token in tokens
                    for _ in range(5 if phase == "baseline" else 1)
                ]
                assert [record["tokens"] for record in records] == expected
                for record in records:
                    names = record.get("profile_kernel_names", [])
                    assert names, (phase, rank, record["tokens"], "missing GPU profile")
                    if phase == "candidate":
                        assert len(names) == 1 and "_megamoe" in names[0], names

    def main(argv=None):
        global _summary_TOKENS, _summary_TIMING_PROTOCOL
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("--data", type=Path, required=True)
        parser.add_argument("--model", type=Path, required=True)
        parser.add_argument(
            "--host-lib", type=Path, default=os.environ.get("MEGAMOE_HOST_LIB")
        )
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument(
            "--devices", required=True, help="Physical GPU indices, in rank order"
        )
        parser.add_argument(
            "--eps", type=int, nargs="+", choices=(2, 4, 8), default=[2, 4, 8]
        )
        parser.add_argument(
            "--tokens", default=",".join(map(str, _summary_CANONICAL_TOKENS))
        )
        parser.add_argument(
            "--mode", choices=("eager", "graph", "both"), default="graph"
        )
        parser.add_argument(
            "--device-visibility",
            choices=("per-rank", "all"),
            default="per-rank",
            help="vLLM device visibility; physical rank assignment is unchanged",
        )
        parser.add_argument("--timeout", type=int, default=1200)
        parser.add_argument("--graph-profile", action="store_true", default=True)
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Write the complete launch plan without GPU execution",
        )
        parser.add_argument(
            "--require-target",
            action="store_true",
            help="Exit nonzero if any selected EP/mode mean speedup is below 0.95",
        )
        args = parser.parse_args(argv)
        if args.host_lib is None:
            parser.error("--host-lib or MEGAMOE_HOST_LIB is required")
        devices, eps = args.devices.split(","), sorted(set(args.eps))
        tokens = list(map(int, args.tokens.split(",")))
        if len(devices) < max(eps) or len(set(devices)) != len(devices):
            parser.error("--devices must provide enough distinct GPU indices")
        if (
            not tokens
            or len(set(tokens)) != len(tokens)
            or not set(tokens) <= set(_summary_CANONICAL_TOKENS)
        ):
            parser.error(
                "--tokens must be a nonempty distinct subset of the canonical 53 shapes"
            )
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "tmp").mkdir()
        env = dict(os.environ)
        env.update(
            PYTHONPATH=os.pathsep.join(
                [str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")]
            ),
            TMPDIR=str(output / "tmp"),
            TRITON_CACHE_DIR=str(output / "triton-cache"),
            DNN_VENDOR="metax",
            MXSHMEM_BOOTSTRAP="MPI",
            MXSHMEM_SYMMETRIC_SIZE="2G",
            MEGAMOE_DATA_VERIFY_SHA256="1",
            USE_PRECOMPILED_KERNEL="1",
            USE_VLLM_TRITON_EXPERT="0",
            MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE="0",
            MACA_VLLM_ENABLE_MCTLASS_PYTHON_API="0",
            VLLM_PLUGINS="metax,metax_enhanced_customized,metax_enhanced_model",
        )
        env.setdefault("OMP_NUM_THREADS", "4")
        data, model = args.data.resolve(), args.model.resolve()
        dataset = json.loads((data / "manifest.json").read_text())
        sources = [
            ROOT / "benchmark/test_megamoe.py",
            ROOT / "tests/test_megamoe.py",
            ROOT / "src/flaggems_vllm/testing/megamoe.py",
            ROOT / "benchmark/megamoe_host.cpp",
            ROOT / "src/flaggems_vllm/runtime/backend/_metax/ops/megamoe.py",
            *sorted(
                (ROOT / "src/flaggems_vllm/runtime/backend/_metax/mega").rglob("*.py")
            ),
            args.host_lib.resolve(),
            data / "manifest.json",
            Path(env["MEGAMOE_DEVICE_LIB"]),
            Path(env["MXSHMEM_DEVICE_LIB"]),
        ]
        frozen = {str(p): digest(p) for p in sources}
        modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
        manifest = dict(
            source_sha256=frozen,
            eps=eps,
            tokens=tokens,
            devices=devices,
            modes=modes,
            shape=dataset["shape"],
            warmup=20,
            iterations=100,
            repeats=5,
            device_visibility=args.device_visibility,
            full_53_shape_ep248=(
                eps == [2, 4, 8] and tokens == list(_summary_CANONICAL_TOKENS)
            ),
            baseline="MetaX vLLM0.24 / mcoplib0.4.9 native W8A8 + full EP",
            initial_quantization_in_timing=False,
            graph_capture_in_timing=False,
            timing_protocols={
                mode: {
                    "baseline": (
                        _summary_GRAPH_PROTOCOL
                        if mode == "graph"
                        else _summary_BASELINE_TIMING_PROTOCOL
                    ),
                    "candidate": (
                        _summary_GRAPH_PROTOCOL
                        if mode == "graph"
                        else "pr774-isolated-events-5-mean-v1"
                    ),
                }
                for mode in modes
            },
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results = {mode: {} for mode in modes}
        launch_plan = []
        for mode in modes:
            directory = output / mode
            directory.mkdir()
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        for ep in eps:
            group_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(devices[:ep]))
            for mode in modes:
                directory = output / mode
                references = directory / f"references-np{ep}"
                native_script = LOCAL / "test_megamoe.py"
                native_role = "graph-baseline" if mode == "graph" else "baseline"
                baseline = [
                    sys.executable,
                    "-u",
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node",
                    str(ep),
                ]
                baseline += [
                    str(LOCAL / "test_megamoe.py"),
                    "--rank-launcher",
                    "--phase",
                    "baseline",
                    "--out",
                    str(directory / f"runtime-baseline-np{ep}"),
                    "--protocol-id",
                    digest(output / "manifest.json"),
                    "--visibility",
                    args.device_visibility,
                    "--",
                ]
                baseline += [
                    str(native_script),
                    "--worker",
                    native_role,
                    "--model",
                    str(model),
                    "--shared-data-dir",
                    str(data),
                    "--tokens-list",
                    args.tokens,
                    "--warmup",
                    "20",
                    "--iters",
                    "100",
                    "--repeats",
                    "5",
                    "--diff-tol",
                    "0.005",
                    "--capture-dir",
                    str(references),
                    "--json-output",
                    str(directory / f"baseline-np{ep}.json"),
                ]
                maca = Path(env["MACA_PATH"])
                candidate = [
                    str(maca / "ompi/bin/mpirun"),
                    "--allow-run-as-root",
                    "--mca",
                    "btl",
                    "self,tcp",
                    "--mca",
                    "plm",
                    "isolated",
                    "--bind-to",
                    "none",
                    "-np",
                    str(ep),
                    sys.executable,
                    "-u",
                    str(LOCAL / "test_megamoe.py"),
                    "--rank-launcher",
                    "--phase",
                    "candidate",
                    "--out",
                    str(directory / f"runtime-candidate-np{ep}"),
                    "--protocol-id",
                    digest(output / "manifest.json"),
                    "--visibility",
                    "all",
                    "--",
                ]
                candidate += [
                    str(LOCAL / "test_megamoe.py"),
                    "--worker",
                    "graph-candidate" if mode == "graph" else "candidate",
                ]
                candidate += [
                    "--host-lib",
                    str(args.host_lib.resolve()),
                    "--data",
                    str(data),
                    "--references",
                    str(references),
                    "--tokens",
                    args.tokens,
                    "--check-formula",
                    "--benchmark",
                    "--warmup",
                    "20",
                    "--iters",
                    "100",
                    "--repeats",
                    "5",
                ]
                for kind, command in (("baseline", baseline), ("candidate", candidate)):
                    if mode == "graph":
                        command += [
                            "--graph-audit-dir",
                            str(directory / f"audit-{kind}"),
                        ]
                        if args.graph_profile:
                            command.append("--graph-profile")
                    launch_plan.append(
                        dict(ep=ep, mode=mode, role=kind, command=command)
                    )
                    if args.dry_run:
                        continue
                    run(
                        command,
                        name=f"{kind}-np{ep}",
                        output=directory,
                        env=group_env,
                        timeout=args.timeout,
                    )
                if args.dry_run:
                    continue
                validate_runtime_preparation(
                    directory, ep, digest(output / "manifest.json"), devices[:ep]
                )
                if mode == "graph":
                    validate_kernel_profiles(directory, ep, tokens)
                if any(digest(Path(p)) != h for p, h in frozen.items()):
                    raise RuntimeError("Measured source/library/input manifest changed")
                _summary_TOKENS = tokens
                _summary_TIMING_PROTOCOL = (
                    _summary_GRAPH_PROTOCOL
                    if mode == "graph"
                    else "pr774-isolated-events-5-mean-v1"
                )
                result = _summary_summarize_ep(directory, ep)
                _summary_write_report(directory, result)
                if not result["complete"]:
                    raise RuntimeError(
                        f"{mode} EP{ep} failed strict validation; see summary-np{ep}.json"
                    )
                results[mode][str(ep)] = result
                report = dict(
                    complete=all(len(v) == len(eps) for v in results.values()),
                    full_53_shape_ep248=manifest["full_53_shape_ep248"],
                    target_pass=all(
                        r["target_pass"]
                        for group in results.values()
                        for r in group.values()
                    ),
                    results=results,
                )
                (output / "summary.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(
                    f"{mode} EP{ep} mean={result['arithmetic_mean_speedup']:.6f}",
                    flush=True,
                )
        (output / "launch-plan.json").write_text(
            json.dumps(dict(dry_run=args.dry_run, launches=launch_plan), indent=2)
            + "\n"
        )
        if args.dry_run:
            print(
                f"PLAN_ONLY: {len(launch_plan)} launches, {len(tokens)} shapes, EP{eps}; no GPU measurements",
                flush=True,
            )
            return
        if args.require_target and not report["target_pass"]:
            raise SystemExit(1)

    def _launch_rank():
        # Copyright 2026 FlagOS Contributors
        # SPDX-License-Identifier: Apache-2.0

        import argparse
        import hashlib
        import json
        import os
        import runpy
        import sys
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def create(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(
                path.name + f".tmp-{os.getpid()}-{time.time_ns()}"
            )
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            try:
                os.link(temporary, path)
            finally:
                temporary.unlink()

        def preinitialize_gpu_runtime(
            torch, device_index, out, protocol_id, phase, rank, devices
        ):
            # Complete lazy stream-pool and activity-profiler allocation before any rank
            # can enter a resident MoE. These are event commands only, outside all timers.
            folder = out / "resource-preinit"
            folder.mkdir(parents=True, exist_ok=True)
            streams = [torch.cuda.Stream(device=device_index) for _ in range(32)]
            events = []
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for stream in streams:
                    event = torch.cuda.Event()
                    event.record(stream)
                    events.append(event)
                for event in events:
                    event.synchronize()
                torch.cuda.synchronize(device_index)
            cuda_events = [
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            assert not cuda_events, "runtime preparation must not launch GPU kernels"
            prof.export_chrome_trace(str(folder / f"profile-rank{rank}.json"))
            proof = dict(
                protocol_sha256=protocol_id,
                phase=phase,
                rank=rank,
                gpu=devices[rank],
                implementation_sha256=digest(__file__),
                stream_pool_requests=len(streams),
                unique_streams=len({s.cuda_stream for s in streams}),
                all_events_complete=all(e.query() for e in events),
                setup_cuda_events=cuda_events,
                setup_complete=True,
                outside_all_timers=True,
                created_utc=datetime.now(timezone.utc).isoformat(),
            )
            create(folder / "ready" / f"rank{rank}.json", proof)
            started = time.monotonic()
            while True:
                paths = [
                    folder / "ready" / f"rank{r}.json" for r in range(len(devices))
                ]
                if all(p.exists() for p in paths):
                    peers = [json.loads(p.read_text()) for p in paths]
                    assert all(
                        (
                            x["protocol_sha256"],
                            x["phase"],
                            x["rank"],
                            x["gpu"],
                            x["setup_complete"],
                        )
                        == (protocol_id, phase, r, devices[r], True)
                        for r, x in enumerate(peers)
                    )
                    break
                if time.monotonic() - started > 1200:
                    raise TimeoutError("pre-MoE runtime readiness rendezvous")
                time.sleep(0.05)
            create(
                folder / "certificates" / f"rank{rank}.json",
                dict(
                    protocol_sha256=protocol_id,
                    phase=phase,
                    rank=rank,
                    ready_sha256={str(r): digest(p) for r, p in enumerate(paths)},
                    before_target_import=True,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                ),
            )
            print(
                f'RANK_RUNTIME_PREFLIGHT_READY phase={phase} rank={rank} streams={proof["unique_streams"]}',
                flush=True,
            )
            return streams, events, prof

        def main():
            parser = argparse.ArgumentParser(description=__doc__)
            parser.add_argument(
                "--phase", choices=("baseline", "candidate"), required=True
            )
            parser.add_argument("--out", type=Path, required=True)
            parser.add_argument("--protocol-id", required=True)
            parser.add_argument(
                "--visibility", choices=("per-rank", "all"), default="per-rank"
            )
            parser.add_argument("target", nargs=argparse.REMAINDER)
            args = parser.parse_args()
            target = args.target[1:] if args.target[:1] == ["--"] else args.target
            if not target:
                parser.error("a target worker script is required after --")
            rank_key = (
                "LOCAL_RANK"
                if args.phase == "baseline"
                else "OMPI_COMM_WORLD_LOCAL_RANK"
            )
            rank = int(os.environ[rank_key])
            devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            physical = devices[rank]
            isolated = args.phase == "baseline" and args.visibility == "per-rank"
            if isolated:
                os.environ["CUDA_VISIBLE_DEVICES"] = physical
                os.environ["LOCAL_RANK"] = "0"
            os.environ["MEGAMOE_PHYSICAL_GPU"] = physical
            device_index = 0 if isolated else rank
            import torch

            assert (
                not torch.cuda.is_initialized()
            ), "rank launcher must precede CUDA initialization"
            torch.cuda.set_device(device_index)
            context_guard = torch.empty(
                1, device=f"cuda:{device_index}", dtype=torch.uint8
            )
            torch.cuda.synchronize(device_index)
            runtime_preflight = preinitialize_gpu_runtime(
                torch,
                device_index,
                args.out,
                args.protocol_id,
                args.phase,
                rank,
                devices,
            )
            assert runtime_preflight[0] and context_guard.numel() == 1
            # Verify the installed candidate actually comes from this invocation's frozen sources.
            if args.phase == "candidate":
                import importlib
                import inspect

                manifest = json.loads((args.out.parent / "manifest.json").read_text())
                kernel = importlib.import_module(
                    "flaggems_vllm.runtime.backend._metax.mega.megamoe.kernel"
                )
                benchmark = importlib.import_module("benchmark.test_megamoe")
                paths = {
                    "kernel": Path(inspect.getsourcefile(kernel._megamoe.fn)).resolve(),
                    "launch_args": Path(
                        inspect.getsourcefile(kernel._launch_args)
                    ).resolve(),
                    "factory": Path(
                        inspect.getsourcefile(
                            inspect.unwrap(benchmark.create_workspace)
                        )
                    ).resolve(),
                }
                hashes = {name: digest(path) for name, path in paths.items()}
                for name, path in paths.items():
                    assert hashes[name] == manifest["source_sha256"][str(path)]
                create(
                    args.out / f"candidate-source-rank{rank}.json",
                    {
                        "rank": rank,
                        "protocol_id": args.protocol_id,
                        "files": {name: str(path) for name, path in paths.items()},
                        "sha256": hashes,
                    },
                )
            sys.argv = target
            runpy.run_path(target[0], run_name="__main__")

        main()

    def _prepare_bridge():
        # Copyright 2026 FlagOS Contributors
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.

        import argparse
        import hashlib
        import json
        import os
        import shlex
        import subprocess
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]

        def build(args):
            sdk, output = args.sdk.resolve(), args.output.resolve()
            output.mkdir(parents=True, exist_ok=False)
            source = ROOT / "benchmark/megamoe_host.cpp"
            host, device = output / "megamoe_host.so", output / "megamoe_sync.bc"
            commands = [
                [
                    args.compiler,
                    f"-offload-arch={args.arch}",
                    "-x",
                    "maca",
                    "-shared",
                    "-fPIC",
                    "-fgpu-rdc",
                    "-std=c++17",
                    f"-I{sdk / 'include'}",
                    f"-I{sdk / 'include/mxshmem'}",
                    f"-I{sdk / 'ompi/include'}",
                    f"-L{sdk / 'lib'}",
                    f"-L{sdk / 'ompi/lib'}",
                    str(source),
                    "-lmxshmem_host",
                    "-lmxshmem",
                    "-lmpi",
                    "-o",
                    str(host),
                ],
                [
                    args.compiler,
                    "-x",
                    "maca",
                    "-maca-device-only",
                    "-emit-llvm",
                    "-c",
                    "-O2",
                    "-fgpu-rdc",
                    "-DMEGAMOE_DEVICE_LIBRARY",
                    str(source),
                    "-o",
                    str(device),
                ],
            ]
            for command in commands:
                print(shlex.join(command), flush=True)
                subprocess.run(command, check=True, timeout=600)
            variables = {
                "MACA_PATH": str(sdk),
                "MEGAMOE_HOST_LIB": str(host),
                "MEGAMOE_DEVICE_LIB": str(device),
                "MXSHMEM_DEVICE_LIB": str(sdk / "lib/libmxshmem_device.bc"),
                "DNN_VENDOR": "metax",
                "MXSHMEM_BOOTSTRAP": "MPI",
                "MXSHMEM_SYMMETRIC_SIZE": "2G",
                "OMP_NUM_THREADS": "4",
            }
            text = "\n".join(
                f"export {k}={shlex.quote(v)}" for k, v in variables.items()
            )
            text += (
                f'\nexport PYTHONPATH={shlex.quote(str(ROOT) + os.pathsep + str(ROOT / "src"))}'
                ':"${PYTHONPATH:-}"\n'
            )
            (output / "env.sh").write_text(text)
            (output / "build.json").write_text(
                json.dumps(
                    dict(
                        commands=commands,
                        arch=args.arch,
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(f"source {shlex.quote(str(output / 'env.sh'))}", flush=True)

        def main():
            parser = argparse.ArgumentParser(
                description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            sub = parser.add_subparsers(dest="command", required=True)
            compile_parser = sub.add_parser(
                "build", help="Ahead-of-time mxSHMEM host and device adapters"
            )
            compile_parser.add_argument(
                "--sdk", type=Path, default=os.environ.get("MACA_PATH")
            )
            compile_parser.add_argument(
                "--arch",
                required=True,
                help="Target offload architecture, e.g. xcore1000",
            )
            compile_parser.add_argument("--compiler", default="mxcc")
            compile_parser.add_argument("--output", type=Path, required=True)
            args = parser.parse_args()
            if args.command == "build":
                if args.sdk is None:
                    parser.error("--sdk or MACA_PATH is required")
                build(args)

        main()

    # The controller and rank launcher run before importing the GPU benchmark stack.
    if __name__ == "__main__":
        if "--rank-launcher" in sys.argv:
            sys.argv.remove("--rank-launcher")
            _launch_rank()
            raise SystemExit(0)
        if "--prepare" in sys.argv:
            sys.argv.remove("--prepare")
            _prepare_bridge()
            raise SystemExit(0)
        if "--summarize" in sys.argv:
            sys.argv.remove("--summarize")
            _summary_main()
            raise SystemExit(0)
        if "--worker" not in sys.argv:
            main()
            raise SystemExit(0)

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_qwen3_canonical(data_dir, *, torch, verify_sha256: bool | None = None):
        """Load shared activations/routes and global expert tensors from a manifest."""
        root = Path(data_dir).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if verify_sha256 is None:
            verify_sha256 = os.environ.get("MEGAMOE_DATA_VERIFY_SHA256", "0") == "1"
        tensors = {}
        for name in ("shared", "experts"):
            entry = manifest["files"][name]
            path = root / entry["name"]
            if verify_sha256 and _sha256(path) != entry["sha256"]:
                raise ValueError(f"SHA256 mismatch for {path}")
            tensors[name] = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        return {"root": root, "manifest": manifest, **tensors}

    # The standalone comparison is CPU-only until its isolated workers select devices.

    import torch  # noqa: E402
    import triton  # noqa: E402
    import triton.language as tl  # noqa: E402
    from triton import knobs  # noqa: E402

    from flaggems_vllm.testing.megamoe import make_inputs as make_inputs  # noqa: E402
    from flaggems_vllm.testing.megamoe import reference as reference  # noqa: E402

    __all__ = [
        "KernelTimer",
        "create_workspace",
        "make_inputs",
        "measure",
        "reference",
        "resident_blocks",
        "zero_peer",
        "TIMING_PROTOCOL",
    ]

    TIMING_PROTOCOL = "pr774-isolated-events-5-mean-v1"

    @triton.jit
    def _local_peer(storage, table, size: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(storage + idx, 0, mask=idx < size)
        if tl.program_id(0) == 0:
            tl.store(table, storage.to(tl.int64))

    @triton.jit
    def _init_state(state, epoch_counter):
        tl.store(state + tl.arange(0, 32), 0)
        tl.store(epoch_counter, 0)

    @lru_cache(maxsize=1)
    def _occupancy_query():
        # PyTorch/FlagTree has already loaded the platform runtime.
        runtime = ctypes.CDLL(None)
        query = runtime.mcModuleOccupancyMaxActiveBlocksPerMultiprocessor
        query.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        return query

    @lru_cache(maxsize=128)
    def resident_blocks(function, num_warps, shared):
        """Query the compiled kernel's resident CTA count, without launching it."""
        from triton.runtime import driver

        count = ctypes.c_int()
        threads = num_warps * driver.active.get_current_target().warp_size
        status = _occupancy_query()(
            ctypes.byref(count), ctypes.c_void_p(function), threads, shared
        )
        if status or count.value < 1:
            raise RuntimeError(
                f"MegaMoE occupancy query failed: {status}, {count.value}"
            )
        return count.value

    @lru_cache(maxsize=1)
    def _device_libraries():
        """The caller supplies SDK-matched, prebuilt device libraries."""
        return (
            ("megamoe_sync", os.environ["MEGAMOE_DEVICE_LIB"]),
            ("mxshmem", os.environ["MXSHMEM_DEVICE_LIB"]),
        )

    _shmem_modules = {}

    @contextmanager
    def _shmem_module(compiled, shmem):
        """Reference-count registrations across overlapping caller workspaces."""
        key = (torch.cuda.current_device(), compiled.module)
        if key not in _shmem_modules:
            for name in ("mxshmemx_cumodule_init", "mxshmemx_cumodule_finalize"):
                function = getattr(shmem, name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_int
            status = shmem.mxshmemx_cumodule_init(ctypes.c_void_p(compiled.module))
            if status:
                raise RuntimeError(f"mxSHMEM module initialization failed: {status}")
            _shmem_modules[key] = [compiled, 0]
        _shmem_modules[key][1] += 1
        try:
            yield
        finally:
            torch.cuda.synchronize()
            _shmem_modules[key][1] -= 1
            if _shmem_modules[key][1] == 0:
                status = shmem.mxshmemx_cumodule_finalize(
                    ctypes.c_void_p(compiled.module)
                )
                del _shmem_modules[key]
                if status:
                    raise RuntimeError(f"mxSHMEM module finalization failed: {status}")

    @contextmanager
    def create_workspace(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        *,
        shmem,
        extern_libs=None,
        max_tokens=None,
        rank=0,
        world_size=1,
        allocate_symmetric=None,
        release_symmetric=None,
        a1_scale=None,
        expert_map=None,
        out=None,
        workers=None,
        ctas_per_sm=2,
        local_output=None,
        rank_reduce=None,
        block_m=None,
        block_n=128,
        block_k=None,
        stages=None,
    ):
        """Test-owned allocation and residency planning; nothing runs in op timing.

        ctas_per_sm is a configurable tuning budget, not a device or algorithm limit.
        workers overrides the requested total grid. The compiled occupancy may reduce
        it; all grid CTAs must reside simultaneously because the kernel spin-waits.
        Each EP rank must construct matching symmetric layouts/capacities in order.
        """
        from flaggems_vllm.runtime.backend._metax.mega.megamoe.api import MegaMoEContext
        from flaggems_vllm.runtime.backend._metax.mega.megamoe.kernel import (
            _launch_args,
            _megamoe,
        )

        capacity = max(hidden_states.shape[0], 1) if max_tokens is None else max_tokens
        top_k = topk_ids.shape[1]
        e, _, h = w1.shape
        i = w2.shape[2]
        sms = torch.cuda.get_device_properties(w1.device).multi_processor_count
        from flaggems_vllm.runtime.backend._metax.mega.megamoe.kernel import (
            plan_workspace,
        )

        selection = plan_workspace(
            capacity=capacity,
            top_k=top_k,
            local_experts=e,
            hidden=h,
            intermediate=i,
            world_size=world_size,
            multiprocessors=sms,
            workers=workers,
            ctas_per_sm=ctas_per_sm,
            local_output=local_output,
            rank_reduce=rank_reduce,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
        )
        bm, block_n, block_k, stages = (
            selection.bm,
            selection.bn,
            selection.bk,
            selection.stages,
        )
        local_output, rank_reduce, requested = (
            selection.local_output,
            selection.rank_reduce,
            selection.workers,
        )
        if requested < 1:
            raise ValueError("workspace requires a positive CTA budget")
        align = lambda n: triton.cdiv(n, 256) * 256
        xoff = 256
        soff = align(xoff + capacity * h)
        roff = align(soff + capacity * 4)
        woff = align(roff + capacity * top_k * 4)
        layout = (align(woff + capacity * top_k * 4), xoff, soff, roff, woff)
        storage = []

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=w1.device)

        if allocate_symmetric is None:
            if world_size != 1:
                raise ValueError("EP setup requires a symmetric allocator")

            def allocate_symmetric(nbytes):
                allocation = empty((nbytes,), torch.uint8)
                table = empty((1,), torch.int64)
                _local_peer[(triton.cdiv(nbytes, 1024),)](
                    allocation, table, nbytes, 1024
                )
                storage.append(allocation)
                return table

        try:
            inputs = allocate_symmetric(layout[0] * world_size)
            partials = allocate_symmetric(top_k * capacity * h * 2)
            signals = allocate_symmetric(3 * world_size * 4)
            blocks = triton.cdiv(capacity * top_k * world_size, bm) + e
            rows = blocks * bm
            chunks = triton.cdiv(capacity * top_k, 64)
            buffers = (
                empty((32,), torch.int32),
                empty((1,), torch.int32),
                empty((world_size * chunks * e,), torch.int32),
                empty((world_size * chunks * e,), torch.int32),
                empty((e * world_size,), torch.int32),
                empty((e,), torch.int32),
                empty((blocks, 2), torch.int32),
                empty((rows, 3), torch.int32),
                empty((rows, h), torch.int8),
                empty((rows,), torch.float32),
                empty((rows,), torch.float32),
                empty((rows, i), torch.bfloat16),
                empty((rows, i), torch.int8),
                empty((rows,), torch.float32),
            )
            expert_partials = (
                empty(
                    ((blocks if rank_reduce else min(blocks, 4 * requested)) * bm, h),
                    torch.bfloat16,
                )
                if local_output
                else partials
            )
            _init_state[(1,)](buffers[0], buffers[1])
            ws = MegaMoEContext(
                rank=rank,
                world_size=world_size,
                capacity=capacity,
                top_k=top_k,
                model_shape=(e, h, i),
                layout=layout,
                workers=requested,
                bm=bm,
                bn=block_n,
                bk=block_k,
                stages=stages,
                local_output=local_output,
                rank_reduce=rank_reduce,
                route_rows=(
                    empty(
                        (world_size * capacity * top_k + triton.cdiv(h, block_n),),
                        torch.int32,
                    )
                    if rank_reduce
                    else None
                ),
                inputs=inputs,
                partials=partials,
                signals=signals,
                peer_alignment=(
                    32
                    if all(
                        int(address) % 32 == 0
                        for table in (inputs, partials, signals)
                        for address in table.cpu().tolist()
                    )
                    else 1
                ),
                buffers=buffers,
                expert_partials=expert_partials,
                extern_libs=(
                    tuple(extern_libs)
                    if extern_libs is not None
                    else _device_libraries()
                ),
            )
            with torch.cuda.device(w1.device):
                while True:
                    _, args, options = _launch_args(
                        hidden_states,
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        w1_scale,
                        w2_scale,
                        workspace=ws,
                        a1_scale=a1_scale,
                        expert_map=expert_map,
                        out=out,
                    )
                    compiled = _megamoe.warmup(*args, grid=(ws.workers,), **options)
                    compiled._init_handles()
                    limit = sms * resident_blocks(
                        compiled.function,
                        compiled.metadata.num_warps,
                        compiled.metadata.shared,
                    )
                    if ws.workers <= limit:
                        break
                    ws = replace(ws, workers=limit)
            with _shmem_module(compiled, shmem):
                yield ws
        finally:
            torch.cuda.synchronize(w1.device)
            if release_symmetric is not None:
                release_symmetric()
            storage.clear()

    class KernelTimer:
        """Time exactly one MegaMoE launch; keep stream barriers outside events.

        Warmup/JIT, argument binding and occupancy queries precede the enter hook.
        The unchanged device kernel includes its input publication, route planning,
        compute, partial return and internal rank barriers.
        """

        def __init__(
            self, barrier: Optional[Callable[[torch.cuda.Stream], None]] = None
        ) -> None:
            self.barrier = barrier
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.armed = False
            self.launches = 0

        def __enter__(self) -> KernelTimer:
            knobs.runtime.launch_enter_hook.add(self._enter)
            knobs.runtime.launch_exit_hook.add(self._exit)
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.armed = False
            knobs.runtime.launch_exit_hook.remove(self._exit)
            knobs.runtime.launch_enter_hook.remove(self._enter)

        def arm(self) -> None:
            self.launches = 0
            self.armed = True

        def _enter(self, metadata) -> None:
            if not self.armed:
                return
            name = metadata.get()["name"]
            assert name == "_megamoe", f"unexpected kernel inside timed call: {name}"
            stream = torch.cuda.current_stream()
            if self.barrier is not None:
                self.barrier(stream)
            self.start.record(stream)
            self.launches += 1

        def _exit(self, metadata) -> None:
            if not self.armed:
                return
            stream = torch.cuda.current_stream()
            self.end.record(stream)
            if self.barrier is not None:
                self.barrier(stream)

        def finish(self) -> float:
            """Return milliseconds after completing this individual sample."""
            assert self.launches == 1, "each sample must contain one MegaMoE launch"
            self.end.synchronize()
            self.armed = False
            return self.start.elapsed_time(self.end)

    @triton.jit
    def zero_peer(table, rank: tl.constexpr, size: tl.constexpr, BLOCK: tl.constexpr):
        ptr = tl.load(table + rank).to(tl.pointer_type(tl.uint8))
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(ptr + idx, 0, mask=idx < size)

    def measure(run, lib, rank, tokens, warmup, iters, repeats):
        """PR #774 cadence: complete every CUDA-event sample before the next."""

        lib.mxshmemx_barrier_wrapper.argtypes = [ctypes.c_void_p]
        lib.mxshmemx_barrier_wrapper.restype = None

        def barrier(stream):
            lib.mxshmemx_barrier_wrapper(ctypes.c_void_p(stream.cuda_stream))

        means = []
        with KernelTimer(barrier) as timer:
            for repeat in range(repeats):
                for _ in range(warmup):
                    run()
                torch.cuda.synchronize()
                lib.mxshmem_all_success_wrapper(1)
                times = []
                for _ in range(iters):
                    timer.arm()
                    run()
                    times.append(timer.finish() * 1000.0)
                    # Complete the untimed stream barrier before the host collective.
                    torch.cuda.synchronize()
                    lib.mxshmem_all_success_wrapper(1)
                mean_us = sum(times) / len(times)
                means.append(mean_us)
                # All device work is complete before the repeat-level host collective.
                torch.cuda.synchronize()
                lib.mxshmem_all_success_wrapper(1)
                print(
                    f"BENCH_REPEAT rank={rank} tokens={tokens} repeat={repeat} "
                    f"mean_us={mean_us:.9f}",
                    flush=True,
                )
        return means

    import pytest  # noqa: E402

    from benchmark import base  # noqa: E402
    from benchmark.conftest import Config  # noqa: E402
    from benchmark.consts import (  # noqa: E402
        DEFAULT_ITER_TIME,
        DEFAULT_WARMUP_TIME,
        BenchLevel,
        BenchMode,
    )

    CORE_TOKENS = (1, 64, 128, 256, 512, 2048, 16384)
    EPS = (2, 4, 8)

    def _baseline_latency(row):
        return row["vllm_ms"]

    def _candidate_latency(row):
        return row["megamoe_ms"]

    def _iteration_count(option, configured, default, fallback):
        explicit = any(
            arg == option or arg.startswith(f"{option}=") for arg in sys.argv
        )
        return configured if explicit or configured != default else fallback

    class MegaMoEBenchmark(base.Benchmark):
        DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
        DEFAULT_DTYPES = [torch.int8]
        DEFAULT_SHAPE_DESC = (
            "tokens, hidden, intermediate, num_experts, topk, num_ranks"
        )

        def __init__(self):
            super().__init__(
                op_name="megamoe",
                torch_op=_baseline_latency,
                gems_op=_candidate_latency,
                dtypes=self.DEFAULT_DTYPES,
            )
            self._paired_rows = None

        def set_shapes(self, shape_file_path=None):
            tokens = (
                _summary_CANONICAL_TOKENS
                if Config.bench_level is BenchLevel.COMPREHENSIVE
                else CORE_TOKENS
            )
            data = os.environ.get("MEGAMOE_DATA_DIR")
            shape = (
                json.loads((Path(data) / "manifest.json").read_text())["shape"]
                if data
                else {
                    "hidden_size": 2048,
                    "moe_intermediate_size": 512,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                }
            )
            dimensions = (
                shape["hidden_size"],
                shape["moe_intermediate_size"],
                shape["num_experts"],
                shape["num_experts_per_tok"],
            )
            self.shapes = [(t, *dimensions, ep) for ep in EPS for t in tokens]
            self.shape_desc = self.DEFAULT_SHAPE_DESC

        def _ensure_paired_rows(self):
            if self._paired_rows is not None:
                return
            if Config.mode not in (BenchMode.KERNEL, BenchMode.CUDAGRAPH):
                raise ValueError("MegaMoE benchmark measures full CUDA Graph replay")
            warmup = _iteration_count(
                "--warmup", Config.warm_up, DEFAULT_WARMUP_TIME, 20
            )
            iterations = _iteration_count(
                "--iter", Config.repetition, DEFAULT_ITER_TIME, 100
            )
            if (warmup, iterations) != (20, 100):
                raise ValueError(
                    "The paired acceptance protocol requires --warmup 20 --iter 100 and 5 repeats"
                )
            required = ("MEGAMOE_DATA_DIR", "MEGAMOE_MODEL_DIR", "MEGAMOE_HOST_LIB")
            missing = [name for name in required if not os.environ.get(name)]
            if missing:
                raise ValueError("Set benchmark prerequisites: " + ", ".join(missing))
            devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if not devices:
                devices = ",".join(map(str, range(torch.cuda.device_count())))
            destination = os.environ.get("MEGAMOE_BENCHMARK_OUTPUT")
            output = (
                Path(destination)
                if destination
                else Path(tempfile.mkdtemp(prefix="megamoe-")) / "results"
            )
            tokens = sorted({shape[0] for shape in self.shapes})
            command = [
                sys.executable,
                "-u",
                "-m",
                "benchmark.test_megamoe",
                "--data",
                os.environ["MEGAMOE_DATA_DIR"],
                "--model",
                os.environ["MEGAMOE_MODEL_DIR"],
                "--host-lib",
                os.environ["MEGAMOE_HOST_LIB"],
                "--devices",
                devices,
                "--eps",
                *map(str, EPS),
                "--tokens",
                ",".join(map(str, tokens)),
                "--mode",
                "graph",
                "--output",
                str(output),
                "--timeout",
                "1800",
            ]
            subprocess.run(command, check=True)
            report = json.loads((output / "summary.json").read_text())
            if not report["complete"]:
                raise RuntimeError(f"Incomplete paired benchmark: {output}")
            rows = {}
            for ep in EPS:
                result = report["results"]["graph"][str(ep)]
                if not result["complete"]:
                    raise RuntimeError(f"Incomplete EP{ep}: {output}")
                for row in result["rows"]:
                    if not all(
                        row[key]
                        for key in ("correctness", "baseline_valid", "timing_valid")
                    ):
                        raise RuntimeError(f"Invalid paired result: EP{ep} {row}")
                    rows[ep, row["tokens_per_rank"]] = row
            if set(rows) != {(ep, token) for ep in EPS for token in tokens}:
                raise RuntimeError("Missing or unexpected paired benchmark rows")
            self._paired_rows = rows

        def get_input_iter(self, dtype):
            yield from self.shapes

        def get_latency(
            self, op, tokens, hidden, intermediate, num_experts, topk, num_ranks
        ):
            self._ensure_paired_rows()
            return op(self._paired_rows[num_ranks, tokens])

    @pytest.mark.megamoe
    def test_megamoe():
        if Config.query:
            MegaMoEBenchmark().run()
            return
        from flaggems_vllm.runtime import device

        if device.vendor_name != "metax":
            pytest.skip("requires MetaX")
        if torch.cuda.device_count() < max(EPS):
            pytest.skip("requires eight visible GPUs for paired EP2/4/8")
        MegaMoEBenchmark().run()

    def _run_vendor_worker(*, graph):
        import argparse
        import faulthandler
        import gc
        import hashlib
        import importlib
        import importlib.metadata
        import inspect
        import json
        import math
        import os
        import signal
        import statistics
        import sys
        import weakref
        from collections import Counter
        from contextlib import contextmanager, nullcontext
        from pathlib import Path
        from typing import Iterable, Iterator

        import torch
        import torch.distributed as dist
        import vllm
        from packaging.version import Version
        from vllm.config import set_current_vllm_config
        from vllm.distributed import get_ep_group
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
            prepare_communication_buffer_for_model,
        )
        from vllm.engine.arg_utils import EngineArgs
        from vllm.forward_context import get_forward_context, set_forward_context
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
        from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
        from vllm.model_executor.model_loader.utils import (
            configure_quant_config,
            get_model_cls,
            process_weights_after_loading,
        )
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            reset_workspace_manager,
        )

        # Copyright 2026 FlagOS Contributors
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.

        INT8_DTYPE = torch.int8

        def _get_tensor(mapping: dict[str, torch.Tensor], *names: str) -> torch.Tensor:
            for name in names:
                value = mapping.get(name)
                if isinstance(value, torch.Tensor):
                    return value
            raise KeyError(f"none of the required tensor keys are present: {names}")

        def _normalize_channel_scale(scale: torch.Tensor, *, name: str) -> torch.Tensor:
            scale = scale.float() if scale.dtype != torch.float32 else scale
            if scale.ndim == 2:
                scale = scale.unsqueeze(-1)
            if scale.ndim != 3 or scale.shape[-1] != 1:
                raise ValueError(
                    f"{name} must be [E,N] or [E,N,1], got {tuple(scale.shape)}"
                )
            return scale.contiguous()

        def _first_present(obj, names: Iterable[str], default=None):
            for name in names:
                value = getattr(obj, name, None)
                if value is not None:
                    return value
            return default

        def resolve_moe_spec(hf_config) -> dict[str, object]:
            values = {
                "hidden_size": _first_present(hf_config, ("hidden_size", "d_model")),
                "intermediate_size": _first_present(
                    hf_config,
                    (
                        "moe_intermediate_size",
                        "expert_intermediate_size",
                        "intermediate_size",
                    ),
                ),
                "num_experts": _first_present(
                    hf_config, ("num_experts", "n_routed_experts", "num_local_experts")
                ),
                "top_k": _first_present(
                    hf_config,
                    (
                        "num_experts_per_tok",
                        "num_experts_per_token",
                        "num_selected_experts",
                        "moe_top_k",
                    ),
                ),
            }
            missing = [name for name, value in values.items() if value is None]
            if missing:
                raise ValueError("Could not infer MoE fields: " + ", ".join(missing))
            return {
                **{name: int(value) for name, value in values.items()},
                "renormalize": bool(
                    _first_present(hf_config, ("norm_topk_prob",), True)
                ),
                "activation": str(_first_present(hf_config, ("hidden_act",), "silu")),
            }

        def _local_expert_ids(moe: FusedMoE, num_experts: int) -> list[int]:
            expert_map = getattr(moe, "_expert_map", None)
            if expert_map is None:
                return list(range(num_experts))
            return (
                torch.nonzero(expert_map.detach().cpu() >= 0, as_tuple=False)
                .flatten()
                .tolist()
            )

        def _iter_real_expert_weights(
            *,
            local: dict[str, torch.Tensor],
            local_ids: list[int],
            intermediate_size: int,
        ) -> Iterator[tuple[str, torch.Tensor]]:
            l1 = _get_tensor(local, "l1_weight_int8", "l1_weight_q")
            l2 = _get_tensor(local, "l2_weight_int8", "l2_weight_q")
            l1_scale = _normalize_channel_scale(
                _get_tensor(local, "l1_weight_scale", "l1_weight_scales"),
                name="l1_weight_scale",
            )
            l2_scale = _normalize_channel_scale(
                _get_tensor(local, "l2_weight_scale", "l2_weight_scales"),
                name="l2_weight_scale",
            )
            if l1.dtype != INT8_DTYPE or l2.dtype != INT8_DTYPE:
                raise TypeError(
                    f"expected INT8 expert weights, got l1={l1.dtype}, l2={l2.dtype}"
                )
            if l1.shape[0] != len(local_ids) or l2.shape[0] != len(local_ids):
                raise ValueError(
                    f"dataset/vLLM local-expert mismatch: {l1.shape[0]}, {l2.shape[0]}, {len(local_ids)}"
                )
            if l1.ndim != 3 or l1.shape[1] != 2 * intermediate_size:
                raise ValueError(f"expected l1 shape [E,2I,H], got {tuple(l1.shape)}")
            if (
                l2.ndim != 3
                or l2.shape[1] != l1.shape[2]
                or l2.shape[2] != intermediate_size
            ):
                raise ValueError(f"expected l2 shape [E,H,I], got {tuple(l2.shape)}")
            if l1_scale.shape[:2] != l1.shape[:2] or l2_scale.shape[:2] != l2.shape[:2]:
                raise ValueError("INT8 scale and weight shapes do not match")
            for slot, expert_id in enumerate(local_ids):
                stem = str(expert_id)
                yield f"{stem}.gate_proj.weight", l1[slot, :intermediate_size]
                yield f"{stem}.up_proj.weight", l1[slot, intermediate_size:]
                yield f"{stem}.down_proj.weight", l2[slot]
                yield f"{stem}.gate_proj.weight_scale", l1_scale[
                    slot, :intermediate_size
                ]
                yield f"{stem}.up_proj.weight_scale", l1_scale[slot, intermediate_size:]
                yield f"{stem}.down_proj.weight_scale", l2_scale[slot]

        def load_real_expert_weights(
            *, moe: FusedMoE, local: dict[str, torch.Tensor], spec: dict[str, object]
        ) -> list[int]:
            local_ids = _local_expert_ids(moe, int(spec["num_experts"]))
            loaded = list(
                moe.load_weights(
                    _iter_real_expert_weights(
                        local=local,
                        local_ids=local_ids,
                        intermediate_size=int(spec["intermediate_size"]),
                    )
                )
            )
            required = {
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            }
            missing = sorted(required - set(loaded))
            if missing:
                raise RuntimeError(
                    f"vLLM did not load required INT8 parameters: {missing}"
                )
            return local_ids

        def _rank_value(value, rank: int):
            if isinstance(value, (list, tuple)):
                return value[rank]
            if isinstance(value, torch.Tensor) and value.ndim >= 3:
                return value[rank]
            return value

        def _load_prequantized_input(*, shared: dict, rank: int, device):
            hidden_bf16 = (
                _rank_value(shared["hidden_states_bf16"], rank).to(device).contiguous()
            )
            input_key = next(
                (
                    k
                    for k in ("input_int8", "input_int8_values", "hidden_states_int8")
                    if k in shared
                ),
                None,
            )
            scale_key = next(
                (
                    k
                    for k in (
                        "input_scales",
                        "input_int8_scales",
                        "hidden_states_scale",
                    )
                    if k in shared
                ),
                None,
            )
            if input_key is None or scale_key is None:
                raise KeyError(
                    "METAX baseline requires pre-quantized input_int8 and input scale tensors"
                )
            hidden_q = _rank_value(shared[input_key], rank).to(device).contiguous()
            hidden_scale = (
                _rank_value(shared[scale_key], rank)
                .to(device=device, dtype=torch.float32)
                .contiguous()
            )
            return (
                hidden_bf16,
                hidden_q,
                hidden_scale,
                f"shared-dataset:{input_key}+{scale_key}",
            )

        def validate_inputs(hidden_q, hidden_scale, topk_weights, topk_ids, spec):
            tokens = hidden_q.shape[0]
            if hidden_q.dtype != INT8_DTYPE or not hidden_q.is_contiguous():
                raise TypeError(
                    f"input must be contiguous torch.int8, got {hidden_q.dtype}, {hidden_q.is_contiguous()}"
                )
            if hidden_scale.dtype != torch.float32 or tuple(hidden_scale.shape) != (
                tokens,
                1,
            ):
                raise TypeError(
                    f"input scales must be FP32 {(tokens, 1)}, got {hidden_scale.dtype}, {tuple(hidden_scale.shape)}"
                )
            expected_topk = (tokens, int(spec["top_k"]))
            if topk_ids.dtype != torch.int64 or tuple(topk_ids.shape) != expected_topk:
                raise TypeError(f"topk ids must be int64 {expected_topk}")
            if (
                topk_weights.dtype != torch.float32
                or tuple(topk_weights.shape) != expected_topk
            ):
                raise TypeError(f"topk weights must be FP32 {expected_topk}")

        def _dp_sizes_context(moe: FusedMoE):
            metadata = get_forward_context().dp_metadata
            if metadata is None:
                return nullcontext()
            return metadata.sp_local_sizes(moe.moe_config.sp_size)

        def _all_reduce_scalar(value: float, *, op, device) -> float:
            tensor = torch.tensor(value, dtype=torch.float64, device=device)
            dist.all_reduce(tensor, op=op)
            return float(tensor.item())

        @torch.inference_mode()
        def run_correctness(
            *,
            moe,
            adapter,
            reference_kernel,
            vllm_config,
            hidden_bf16,
            hidden_q,
            hidden_scale,
            topk_weights,
            topk_ids,
            diff_tol,
            device,
        ):
            qcfg = moe.moe_quant_config
            stock_q, stock_scale = moe_kernel_quantize_input(
                hidden_bf16,
                qcfg.a1_scale,
                qcfg.quant_dtype,
                qcfg.per_act_token_quant,
                qcfg.block_shape,
            )
            mismatch = int(
                _all_reduce_scalar(
                    float((stock_q.contiguous() != hidden_q).sum()),
                    op=dist.ReduceOp.SUM,
                    device=device,
                )
            )
            scale_error = _all_reduce_scalar(
                float((stock_scale.float() - hidden_scale).abs().max()),
                op=dist.ReduceOp.MAX,
                device=device,
            )
            dist.barrier()
            with set_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                num_tokens=hidden_q.shape[0],
            ):
                with _dp_sizes_context(moe):
                    reference = reference_kernel.apply(
                        hidden_states=hidden_bf16,
                        w1=moe.w13_weight,
                        w2=moe.w2_weight,
                        topk_ids=topk_ids,
                        topk_weights=topk_weights,
                        activation=moe.activation,
                        global_num_experts=moe.global_num_experts,
                        expert_map=moe.expert_map,
                        apply_router_weight_on_input=moe.apply_router_weight_on_input,
                    )
                    output = adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
            torch.cuda.synchronize(device)
            if output.dtype != hidden_bf16.dtype or output.shape != hidden_bf16.shape:
                raise RuntimeError(
                    f"invalid output dtype/shape: {output.dtype} {tuple(output.shape)}"
                )
            delta = output.float() - reference.float()
            diff_sq = _all_reduce_scalar(
                float(delta.square().sum()), op=dist.ReduceOp.SUM, device=device
            )
            ref_sq = _all_reduce_scalar(
                float(reference.float().square().sum()),
                op=dist.ReduceOp.SUM,
                device=device,
            )
            rel_l2 = math.sqrt(diff_sq / max(ref_sq, 1e-30))
            if rel_l2 > diff_tol:
                raise AssertionError(
                    f"relative L2 error {rel_l2:.6g} exceeds {diff_tol:.6g}"
                )
            return {
                "pass": True,
                "relative_l2_error": rel_l2,
                "max_absolute_error": _all_reduce_scalar(
                    float(delta.abs().max()), op=dist.ReduceOp.MAX, device=device
                ),
                "input_int8_value_mismatches": mismatch,
                "input_scale_max_absolute_error": scale_error,
            }

        @torch.inference_mode()
        def _eager_benchmark(
            *,
            moe,
            adapter,
            vllm_config,
            hidden_q,
            hidden_scale,
            topk_weights,
            topk_ids,
            warmup,
            iters,
            device,
        ):
            if warmup < 0 or iters <= 0:
                raise ValueError("--warmup must be >= 0 and --iters must be > 0")
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
            with set_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                num_tokens=hidden_q.shape[0],
            ):
                with _dp_sizes_context(moe):
                    for _ in range(warmup):
                        adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    for start, end in zip(starts, ends):
                        adapter(
                            hidden_q,
                            hidden_scale,
                            topk_weights,
                            topk_ids,
                            timing_events=(start, end),
                        )
                        end.synchronize()
                        dist.barrier()
            local = torch.tensor(
                [s.elapsed_time(e) for s, e in zip(starts, ends)],
                dtype=torch.float64,
                device=device,
            )
            rank_max = local.clone()
            dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
            means = [
                torch.empty(1, dtype=torch.float64, device=device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(means, local.mean().reshape(1))
            return rank_max.cpu().tolist(), [float(x.item()) for x in means]

        def parse_args() -> argparse.Namespace:
            p = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            p.add_argument("--model", type=Path, required=True)
            p.add_argument("--shared-data-dir", type=Path, required=True)
            p.add_argument("--layer", type=int, default=0)
            p.add_argument("--num-tokens", type=int, default=512)
            p.add_argument("--tokens-list", default="")
            p.add_argument("--repeats", type=int, default=5)
            p.add_argument("--json-output", type=Path)
            p.add_argument("--capture-dir", type=Path)
            p.add_argument("--warmup", type=int, default=20)
            p.add_argument("--iters", type=int, default=100)
            p.add_argument("--diff-tol", type=float, default=0.005)
            p.add_argument("--correctness-only", action="store_true")
            p.add_argument("--skip-correctness", action="store_true")
            p.add_argument(
                "--all2all-backend",
                default="allgather_reducescatter",
                choices=("allgather_reducescatter", "naive"),
            )
            p.add_argument("--dist-backend", default="nccl")
            return p.parse_args()

        def load_qwen3_rank_data(
            data_dir,
            *,
            rank: int,
            num_ranks: int,
            num_tokens: int,
            hidden: int,
            intermediate: int,
            num_experts: int,
            topk: int,
            torch,
        ):
            """Select a deterministic tokens-per-rank prefix and linear EP weight slice."""
            dataset = load_qwen3_canonical(data_dir, torch=torch)
            shape = dataset["manifest"]["shape"]
            requested = {
                "hidden_size": hidden,
                "moe_intermediate_size": intermediate,
                "num_experts": num_experts,
                "num_experts_per_tok": topk,
            }
            mismatches = {
                key: {"dataset": shape.get(key), "requested": value}
                for key, value in requested.items()
                if int(shape.get(key, -1)) != int(value)
            }
            if mismatches:
                raise ValueError(f"canonical shape mismatch: {mismatches}")
            if num_ranks not in (1, 2, 4, 8) or num_experts % num_ranks:
                raise ValueError(f"unsupported EP world size: {num_ranks}")
            if not 0 <= rank < num_ranks:
                raise ValueError(f"rank {rank} is outside [0, {num_ranks})")
            capacity = int(shape["tokens_per_rank_capacity"])
            if not 0 < num_tokens <= capacity:
                raise ValueError(f"num_tokens={num_tokens} is outside [1, {capacity}]")

            shared = {
                key: value[rank, :num_tokens].contiguous()
                for key, value in dataset["shared"].items()
            }
            experts_per_rank = num_experts // num_ranks
            begin = rank * experts_per_rank
            end = begin + experts_per_rank
            local = {
                key: value[begin:end].contiguous()
                for key, value in dataset["experts"].items()
            }
            return {"manifest": dataset["manifest"], "shared": shared, "local": local}

        # Copyright 2026 FlagOS Contributors
        # SPDX-License-Identifier: Apache-2.0

        VLLM_REQUIRED_VERSION = "0.24.0"
        VENDOR_EXPERTS_MODULE = (
            "vllm_metax.model_executor.layers.fused_moe.experts.triton_moe"
        )

        def vendor_evidence():
            import vllm_metax.envs as envs
            from vllm_metax.utils.fused_moe import get_triton_experts_cls

            for package, required in (("vllm-metax", "0.24.0"), ("mcoplib", "0.4.9")):
                found = importlib.metadata.version(package)
                if Version(found).base_version != required:
                    raise RuntimeError(f"Require {package} {required}, found {found}")
            cls = get_triton_experts_cls()
            if cls.__module__ != VENDOR_EXPERTS_MODULE:
                raise RuntimeError("The baseline requires MetaX vendor TritonExperts")
            if not envs.USE_PRECOMPILED_KERNEL or envs.USE_VLLM_TRITON_EXPERT:
                raise RuntimeError("Require vendor precompiled mcoplib experts")
            if envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE:
                raise RuntimeError("This baseline freezes the mcoplib W8A8 backend")
            source = inspect.getsource(cls.apply)
            if "torch.int8" not in source:
                raise RuntimeError("Installed vendor experts lack native INT8 support")
            modules = [
                VENDOR_EXPERTS_MODULE,
                "vllm_metax.model_executor.layers.fused_moe.fused_moe",
                "vllm_metax.model_executor.layers.fused_moe.utils",
                "mcoplib.triton_fused_moe",
                "vllm.model_executor.layers.fused_moe.modular_kernel",
                "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep",
            ]
            sources = {}
            for name in modules:
                path = Path(importlib.import_module(name).__file__)
                sources[name] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            return {
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("vllm", "vllm-metax", "mcoplib", "torch")
                },
                "sources": sources,
                "experts_source_patch": False,
                "public_entrypoint": "FusedMoEKernel.apply",
                "prepare_adapter": "prequantized INT8 + FP32 scales; native EP dispatch",
                "output_dtype_carrier": "uninitialized BF16 metadata; data never read",
                "first_input_quantization_in_timed_region": False,
            }

        def build_fused_moe(*, vllm_config, spec, layer, device):
            old_dtype = torch.get_default_dtype()
            torch.set_default_dtype(vllm_config.model_config.dtype)
            try:
                with torch.device(device):
                    runner = FusedMoE(
                        num_experts=int(spec["num_experts"]),
                        top_k=int(spec["top_k"]),
                        hidden_size=int(spec["hidden_size"]),
                        intermediate_size=int(spec["intermediate_size"]),
                        params_dtype=vllm_config.model_config.dtype,
                        renormalize=bool(spec["renormalize"]),
                        quant_config=vllm_config.quant_config,
                        prefix=f"model.language_model.layers.{layer}.mlp.experts",
                        scoring_func="softmax",
                        activation=str(spec["activation"]),
                    )
            finally:
                torch.set_default_dtype(old_dtype)
            moe = runner.routed_experts
            moe.expert_mapping = RoutedExperts.make_expert_params_mapping(
                moe,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=int(spec["num_experts"]),
                num_redundant_experts=0,
                routed_experts_prefix="",
            )
            return moe

        def build_direct_kernel(moe, *, use_dp_ep, prepare=None):
            from vllm_metax.utils.fused_moe import get_triton_experts_cls

            cls = get_triton_experts_cls()
            if cls.__module__ != VENDOR_EXPERTS_MODULE:
                raise RuntimeError("Non-vendor expert implementation is forbidden")
            experts = cls(moe.moe_config, moe.moe_quant_config)
            if experts.expects_unquantized_inputs:
                raise NotImplementedError("LoRA / deferred input quantization")
            if prepare is None:
                prepare = (
                    MoEPrepareAndFinalizeNaiveDPEPModular(
                        is_sequence_parallel=moe.moe_config.is_sequence_parallel
                    )
                    if use_dp_ep
                    else MoEPrepareAndFinalizeNoDPEPModular()
                )
            return FusedMoEKernel(prepare, experts)

        class _PrequantizedPrepare:
            """Only the first input quantizer is bypassed; native finalize is inherited."""

            bound_input = None
            use_dp_ep = False

            def prepare(
                self,
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant=False,
            ):
                if self.bound_input is None:
                    raise RuntimeError("No prequantized input bound")
                if apply_router_weight_on_input or defer_input_quant:
                    raise NotImplementedError(
                        "Input router weighting / deferred quantization"
                    )
                if (
                    quant_config.quant_dtype != torch.int8
                    or not quant_config.per_act_token_quant
                    or quant_config.block_shape is not None
                ):
                    raise NotImplementedError("Requires dynamic per-token INT8 W8A8")
                hidden_q, hidden_scale = self.bound_input
                if a1.dtype != torch.bfloat16 or a1.shape != hidden_q.shape:
                    raise ValueError("Invalid BF16 output metadata carrier")
                if self.use_dp_ep:
                    hidden_q, topk_weights, topk_ids, scales = get_ep_group().dispatch(
                        hidden_q,
                        topk_weights,
                        topk_ids,
                        is_sequence_parallel=self.is_sequence_parallel,
                        extra_tensors=[hidden_scale],
                    )
                    if len(scales) != 1:
                        raise RuntimeError(
                            "INT8 scale dispatch returned an invalid result"
                        )
                    hidden_scale = scales[0]
                return hidden_q, hidden_scale, None, topk_ids, topk_weights

        class _PrequantizedDPEP(
            _PrequantizedPrepare, MoEPrepareAndFinalizeNaiveDPEPModular
        ):
            use_dp_ep = True

        class _PrequantizedLocal(
            _PrequantizedPrepare, MoEPrepareAndFinalizeNoDPEPModular
        ):
            pass

        class PrequantizedInt8MoE:
            def __init__(self, moe, local_tokens, output_dtype):
                if output_dtype != torch.bfloat16:
                    raise NotImplementedError("Validated output dtype is BF16")
                self.moe = moe
                self.use_dp_ep = moe.moe_config.dp_size > 1 and moe.moe_config.use_ep
                self.prepare = (
                    _PrequantizedDPEP(
                        is_sequence_parallel=moe.moe_config.is_sequence_parallel
                    )
                    if self.use_dp_ep
                    else _PrequantizedLocal()
                )
                self.kernel = build_direct_kernel(
                    moe, use_dp_ep=self.use_dp_ep, prepare=self.prepare
                )
                self.metadata_carrier = torch.empty(
                    (local_tokens, moe.w2_weight.size(1)),
                    dtype=output_dtype,
                    device=moe.w13_weight.device,
                )

            @property
            def experts_name(self):
                cls = self.kernel.fused_experts.__class__
                return f"{cls.__module__}.{cls.__qualname__}"

            @torch.inference_mode()
            def __call__(
                self,
                hidden_q,
                hidden_scale,
                topk_weights,
                topk_ids,
                *,
                timing_events=None,
            ):
                if self.prepare.bound_input is not None:
                    raise RuntimeError(
                        "Concurrent calls on one adapter are unsupported"
                    )
                if hidden_q.dtype != torch.int8 or hidden_scale.dtype != torch.float32:
                    raise TypeError("Expected INT8 activations and FP32 scales")
                if hidden_scale.shape != (hidden_q.shape[0], 1):
                    raise ValueError("Expected one scale per token")
                self.prepare.bound_input = (hidden_q, hidden_scale)
                try:
                    apply = self.kernel.apply
                    kwargs = dict(
                        hidden_states=self.metadata_carrier,
                        w1=self.moe.w13_weight,
                        w2=self.moe.w2_weight,
                        topk_ids=topk_ids,
                        topk_weights=topk_weights,
                        activation=self.moe.activation,
                        global_num_experts=self.moe.global_num_experts,
                        expert_map=self.moe.expert_map,
                        apply_router_weight_on_input=False,
                    )
                    if timing_events is None:
                        return apply(**kwargs)
                    start, end = timing_events
                    start.record()
                    output = apply(**kwargs)
                    end.record()
                    return output
                finally:
                    self.prepare.bound_input = None

        @torch.inference_mode()
        def audit_adapter(adapter, hidden_q, hidden_scale, topk_weights, topk_ids):
            """Untimed execution audit: two vendor GEMMs and only intermediate quant."""
            from unittest.mock import patch

            expert_module = importlib.import_module(VENDOR_EXPERTS_MODULE)
            fused_module = importlib.import_module(
                "vllm_metax.model_executor.layers.fused_moe.fused_moe"
            )
            native_gemm = fused_module.fused_moe_triton_kernel
            native_quantize = expert_module.moe_kernel_quantize_input
            gemms, quantizers = [], []

            def gemm(*args, **kwargs):
                gemms.append(
                    {
                        "input_dtype": str(args[1].dtype),
                        "input_shape": list(args[1].shape),
                        "weight_dtype": str(args[2].dtype),
                        "output_dtype": str(args[3].dtype),
                        "use_int8_w8a8": bool(kwargs.get("use_int8_w8a8")),
                    }
                )
                return native_gemm(*args, **kwargs)

            def quantize(*args, **kwargs):
                quantizers.append(str(args[0].dtype))
                return native_quantize(*args, **kwargs)

            adapter.metadata_carrier.fill_(float("nan"))
            with patch.object(
                fused_module, "fused_moe_triton_kernel", gemm
            ), patch.object(expert_module, "moe_kernel_quantize_input", quantize):
                poisoned = adapter(
                    hidden_q, hidden_scale, topk_weights, topk_ids
                ).clone()
            adapter.metadata_carrier.zero_()
            clean = adapter(hidden_q, hidden_scale, topk_weights, topk_ids)
            if len(gemms) != 2 or any(
                g["input_dtype"] != "torch.int8"
                or g["weight_dtype"] != "torch.int8"
                or g["output_dtype"] != "torch.bfloat16"
                or not g["use_int8_w8a8"]
                for g in gemms
            ):
                raise AssertionError(f"Expected two vendor W8A8 GEMMs, found {gemms}")
            if quantizers != ["torch.bfloat16"]:
                raise AssertionError(f"Unexpected quantization calls: {quantizers}")
            equal = bool(torch.equal(poisoned, clean))
            finite = bool(torch.isfinite(clean).all())
            if not equal or not finite:
                raise AssertionError(
                    "Metadata carrier affected output or output is nonfinite"
                )
            return {
                "vendor_gemm_calls": gemms,
                "intermediate_quantizer_input_dtypes": quantizers,
                "metadata_carrier_nan_invariance": equal,
                "output_finite": finite,
            }

        benchmark = _eager_benchmark

        def main() -> None:
            args = parse_args()
            tokens_list = (
                [int(x) for x in args.tokens_list.split(",")]
                if args.tokens_list
                else [args.num_tokens]
            )
            args.num_tokens = max(tokens_list)
            reports = []
            if args.correctness_only and args.skip_correctness:
                raise ValueError(
                    "--correctness-only and --skip-correctness are mutually exclusive"
                )
            if Version(vllm.__version__).base_version != VLLM_REQUIRED_VERSION:
                raise RuntimeError(
                    f"This METAX baseline requires vLLM {VLLM_REQUIRED_VERSION}, found {vllm.__version__}"
                )
            if "MACA_PATH" not in os.environ:
                raise RuntimeError(
                    "METAX_SDK is not set; refusing to run the METAX baseline on another platform"
                )
            import vllm_metax.envs as mx_envs

            vendor_flags = {
                "USE_PRECOMPILED_KERNEL": mx_envs.USE_PRECOMPILED_KERNEL,
                "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE,
                "MACA_VLLM_ENABLE_MCTLASS_PYTHON_API": mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API,
                "USE_VLLM_TRITON_EXPERT": mx_envs.USE_VLLM_TRITON_EXPERT,
            }
            if vendor_flags["USE_VLLM_TRITON_EXPERT"]:
                raise RuntimeError(
                    "Upstream expert debug mode is not a vendor baseline"
                )
            if vendor_flags["MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE"]:
                vendor_backend = "MCTLASS W8A8"
            elif vendor_flags["USE_PRECOMPILED_KERNEL"]:
                vendor_backend = "mcoplib.triton_fused_moe"
            else:
                vendor_backend = "vllm_metax Triton JIT"
            evidence = vendor_evidence()
            if (
                not args.model.is_dir()
                or not (args.shared_data_dir / "manifest.json").is_file()
            ):
                raise FileNotFoundError(
                    "model directory or INT8 shared-data manifest is missing"
                )

            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size > 1 and args.all2all_backend != "allgather_reducescatter":
                raise RuntimeError(
                    "pre-quantized INT8 scales are validated only with allgather_reducescatter"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("METAX CUDA-compatible runtime is unavailable")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            init_workspace_manager(device, num_ubatches=1)

            engine_args = EngineArgs(
                model=str(args.model),
                skip_tokenizer_init=True,
                load_format="safetensors",
                tensor_parallel_size=1,
                data_parallel_size=world_size,
                enable_expert_parallel=world_size > 1,
                enable_eplb=False,
                expert_placement_strategy="linear",
                all2all_backend=args.all2all_backend,
                distributed_executor_backend="external_launcher",
                enforce_eager=False,
                moe_backend="triton",
            )
            vllm_config = engine_args.create_engine_config()
            if vllm_config.quant_config is None:
                raise RuntimeError("INT8 model has no quantization_config")
            spec = resolve_moe_spec(vllm_config.model_config.hf_text_config)
            configure_quant_config(
                vllm_config.quant_config, get_model_cls(vllm_config.model_config)
            )
            initialized = False
            report = None
            try:
                with set_current_vllm_config(vllm_config):
                    init_distributed_environment(
                        world_size, rank, "env://", local_rank, args.dist_backend
                    )
                    initialized = True
                    initialize_model_parallel(1, 1, 1, 1, backend=args.dist_backend)
                    dataset = load_qwen3_rank_data(
                        args.shared_data_dir,
                        rank=rank,
                        num_ranks=world_size,
                        num_tokens=args.num_tokens,
                        hidden=int(spec["hidden_size"]),
                        intermediate=int(spec["intermediate_size"]),
                        num_experts=int(spec["num_experts"]),
                        topk=int(spec["top_k"]),
                        torch=torch,
                    )
                    manifest, shared, local = (
                        dataset["manifest"],
                        dataset["shared"],
                        dataset["local"],
                    )
                    moe = build_fused_moe(
                        vllm_config=vllm_config,
                        spec=spec,
                        layer=args.layer,
                        device=device,
                    )
                    moe.eval()
                    local_ids = load_real_expert_weights(
                        moe=moe, local=local, spec=spec
                    )
                    process_weights_after_loading(moe, vllm_config.model_config, device)
                    prepare_communication_buffer_for_model(moe)
                    moe.moe_quant_config = moe.quant_method.get_fused_moe_quant_config(
                        moe
                    )
                    qcfg = moe.moe_quant_config
                    if (
                        qcfg is None
                        or not qcfg.use_int8_w8a8
                        or qcfg.quant_dtype != torch.int8
                    ):
                        raise RuntimeError("model must resolve to dynamic INT8 W8A8")
                    if not qcfg.per_act_token_quant or qcfg.block_shape is not None:
                        raise RuntimeError(
                            "METAX baseline requires dynamic per-token, channelwise INT8"
                        )
                    full_shared = shared
                    for current_tokens in tokens_list:
                        args.num_tokens = current_tokens
                        shared = {
                            key: tensor[:current_tokens]
                            for key, tensor in full_shared.items()
                        }
                        for repeat in range(args.repeats):
                            hidden_bf16, hidden_q, hidden_scale, input_source = (
                                _load_prequantized_input(
                                    shared=shared, rank=rank, device=device
                                )
                            )
                            topk_ids = (
                                _rank_value(shared["topk_idx"], rank)
                                .to(device)
                                .contiguous()
                            )
                            topk_weights = (
                                _rank_value(shared["topk_weights"], rank)
                                .to(device)
                                .contiguous()
                            )
                            validate_inputs(
                                hidden_q, hidden_scale, topk_weights, topk_ids, spec
                            )
                            adapter = PrequantizedInt8MoE(
                                moe, args.num_tokens, vllm_config.model_config.dtype
                            )
                            correctness = None
                            if not args.skip_correctness:
                                reference_kernel = build_direct_kernel(
                                    moe, use_dp_ep=adapter.use_dp_ep
                                )
                                correctness = run_correctness(
                                    moe=moe,
                                    adapter=adapter,
                                    reference_kernel=reference_kernel,
                                    vllm_config=vllm_config,
                                    hidden_bf16=hidden_bf16,
                                    hidden_q=hidden_q,
                                    hidden_scale=hidden_scale,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                    diff_tol=args.diff_tol,
                                    device=device,
                                )
                            if correctness is not None:
                                import math

                                if not math.isfinite(correctness["relative_l2_error"]):
                                    raise AssertionError("Nonfinite reference error")
                                if repeat == 0:
                                    with set_forward_context(
                                        attn_metadata=None,
                                        vllm_config=vllm_config,
                                        num_tokens=current_tokens,
                                    ):
                                        with _dp_sizes_context(moe):
                                            correctness["execution_audit"] = (
                                                audit_adapter(
                                                    adapter,
                                                    hidden_q,
                                                    hidden_scale,
                                                    topk_weights,
                                                    topk_ids,
                                                )
                                            )
                            performance = None
                            if not args.correctness_only:
                                samples, rank_means = benchmark(
                                    moe=moe,
                                    adapter=adapter,
                                    vllm_config=vllm_config,
                                    hidden_q=hidden_q,
                                    hidden_scale=hidden_scale,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                    warmup=args.warmup,
                                    iters=args.iters,
                                    device=device,
                                )
                                slowest = max(rank_means)
                                performance = {
                                    "warmup": args.warmup,
                                    "iterations": args.iters,
                                    "timing": (
                                        "CUDA events around native kernel.apply; full EP; "
                                        "individual sample completion"
                                    ),
                                    "timing_protocol": TIMING_PROTOCOL,
                                    "per_rank_mean_ms": rank_means,
                                    "primary_latency_ms": slowest,
                                    "mean_of_iteration_rank_max_ms": statistics.mean(
                                        samples
                                    ),
                                    "p50_ms": sorted(samples)[len(samples) // 2],
                                    "min_ms": min(samples),
                                    "max_ms": max(samples),
                                    "global_tokens_per_second": world_size
                                    * args.num_tokens
                                    / (slowest / 1e3),
                                }
                            if rank == 0:
                                report = {
                                    "implementation": "MetaX vendor vLLM 0.24.0 pre-quantized INT8 fused MoE + EP",
                                    "vendor_evidence": evidence,
                                    "vendor_backend": vendor_backend,
                                    "vendor_flags": vendor_flags,
                                    "vllm_version": vllm.__version__,
                                    "torch_version": torch.__version__,
                                    "gpu": torch.cuda.get_device_name(device),
                                    "model_config": str(args.model),
                                    "shared_data": str(args.shared_data_dir),
                                    "source_model": manifest.get("source_model"),
                                    "source_revision": manifest.get("source_revision"),
                                    "source_layer": manifest.get("layer"),
                                    "world_size": world_size,
                                    "tokens_per_rank": args.num_tokens,
                                    "hidden_size": spec["hidden_size"],
                                    "intermediate_size": spec["intermediate_size"],
                                    "num_experts": spec["num_experts"],
                                    "local_experts": len(local_ids),
                                    "top_k": spec["top_k"],
                                    "input_dtype": str(hidden_q.dtype),
                                    "input_scale_dtype": str(hidden_scale.dtype),
                                    "input_source": input_source,
                                    "weight_dtype": str(moe.w13_weight.dtype),
                                    "weight_quantization": "channelwise",
                                    "output_dtype": str(vllm_config.model_config.dtype),
                                    "all2all_backend": args.all2all_backend,
                                    "first_input_quantization_in_timed_region": False,
                                    "experts_implementation": adapter.experts_name,
                                    "correctness_reference": (
                                        "same METAX vLLM modular kernel with BF16 "
                                        "input quantized before experts"
                                    ),
                                    "routing": manifest.get("routing", {}).get(
                                        "source", "unknown"
                                    ),
                                    "correctness": correctness,
                                    "performance": performance,
                                }
                            if args.capture_dir is not None and repeat == 0:
                                import hashlib

                                args.capture_dir.mkdir(parents=True, exist_ok=True)
                                with set_forward_context(
                                    attn_metadata=None,
                                    vllm_config=vllm_config,
                                    num_tokens=current_tokens,
                                ):
                                    with _dp_sizes_context(moe):
                                        captured = adapter(
                                            hidden_q,
                                            hidden_scale,
                                            topk_weights,
                                            topk_ids,
                                        ).cpu()
                                tensors = dict(
                                    hidden_q=hidden_q,
                                    hidden_scale=hidden_scale,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                )
                                digests = {
                                    key: hashlib.sha256(
                                        value.cpu()
                                        .contiguous()
                                        .view(torch.uint8)
                                        .numpy()
                                        .tobytes()
                                    ).hexdigest()
                                    for key, value in tensors.items()
                                }
                                torch.save(
                                    dict(output=captured, input_sha256=digests),
                                    args.capture_dir
                                    / f"t{current_tokens}_rank{rank}.pt",
                                )
                            if rank == 0:
                                report["repeat"] = repeat
                                reports.append(report)
                                print(
                                    "BASELINE_RECORD " + json.dumps(report), flush=True
                                )
                                if args.json_output:
                                    args.json_output.parent.mkdir(
                                        parents=True, exist_ok=True
                                    )
                                    args.json_output.write_text(
                                        json.dumps(reports, indent=2)
                                    )
            finally:
                try:
                    if initialized:
                        try:
                            destroy_model_parallel()
                        finally:
                            destroy_distributed_environment()
                finally:
                    reset_workspace_manager()
            if report is not None:
                print("\n" + "=" * 88)
                print(
                    "METAX vLLM 0.24.0 | direct INT8-input FusedMoE + EP | final report"
                )
                print("=" * 88)
                print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

        if not graph:
            main()
            return

        faulthandler.register(signal.SIGUSR1, all_threads=True)

        PROTOCOL = "pr774-full-ep-both-cudagraph-5-mean-v2"
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--graph-audit-dir", type=Path, required=True)
        parser.add_argument("--graph-profile", action="store_true")
        extra, passthrough = parser.parse_known_args()
        sys.argv = [sys.argv[0], *passthrough]
        extra.graph_audit_dir.mkdir(parents=True, exist_ok=True)
        AUDITS = []
        ARGS = None
        RawAdapter = PrequantizedInt8MoE
        LIVE_ADAPTERS = weakref.WeakSet()
        original_destroy = destroy_model_parallel
        original_parse = parse_args

        def parse_args():
            nonlocal ARGS
            ARGS = original_parse()
            return ARGS

        class GraphAdapter(RawAdapter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                LIVE_ADAPTERS.add(self)

            @torch.inference_mode()
            def __call__(self, *tensors):
                graph = getattr(self, "_captured_graph", None)
                if graph is None:
                    return super().__call__(*tensors)
                assert all(
                    a.data_ptr() == b.data_ptr()
                    for a, b in zip(tensors, self._static_inputs)
                )
                graph.replay()
                return self.output

        @contextmanager
        def count_capture_collectives():
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            counts = Counter()
            originals = {}
            for name in (
                "all_gather",
                "all_gatherv",
                "reduce_scatter",
                "reduce_scatterv",
            ):
                original = getattr(PyNcclCommunicator, name, None)
                if original is None:
                    continue
                originals[name] = original

                def wrapped(self, *args, _name=name, _original=original, **kwargs):
                    if torch.cuda.is_current_stream_capturing():
                        counts[_name] += 1
                    return _original(self, *args, **kwargs)

                setattr(PyNcclCommunicator, name, wrapped)
            try:
                yield counts
            finally:
                for name, original in originals.items():
                    setattr(PyNcclCommunicator, name, original)

        def digest(tensor):
            return hashlib.sha256(
                tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()

        @torch.inference_mode()
        def graph_benchmark(
            *,
            moe,
            adapter,
            vllm_config,
            hidden_q,
            hidden_scale,
            topk_weights,
            topk_ids,
            warmup,
            iters,
            device,
        ):
            from vllm.distributed.parallel_state import graph_capture

            rank, world = dist.get_rank(), dist.get_world_size()
            tensors = (hidden_q, hidden_scale, topk_weights, topk_ids)
            tokens = hidden_q.shape[0]
            audit = dict(
                rank=rank,
                world=world,
                tokens=tokens,
                sequence=len(AUDITS),
                execution_mode="cuda_graph",
                timing_protocol=PROTOCOL,
                checks=[],
                input_sha256=[digest(t) for t in tensors],
            )

            def raw():
                return RawAdapter.__call__(adapter, *tensors)

            with set_forward_context(
                attn_metadata=None, vllm_config=vllm_config, num_tokens=tokens
            ):
                with _dp_sizes_context(moe):
                    # Warm all JIT/native kernels and the communicator on a side stream.
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        for _ in range(max(warmup, 3)):
                            raw()
                    torch.cuda.current_stream(device).wait_stream(stream)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with graph_capture(device=device) as capture_context:
                        with count_capture_collectives() as counts:
                            with torch.cuda.graph(graph, stream=capture_context.stream):
                                captured_output = raw()
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    audit["captured_collectives"] = dict(counts)
                    assert (
                        sum(v for k, v in counts.items() if k.startswith("all_gather"))
                        > 0
                    ), counts
                    assert (
                        sum(
                            v
                            for k, v in counts.items()
                            if k.startswith("reduce_scatter")
                        )
                        > 0
                    ), counts
                    snapshots = [t.clone() for t in tensors]
                    original_expected = None

                    def validate(phase):
                        # Replay FIRST: eager must not refresh any shared communication
                        # or scratch buffer before the replay being validated.
                        graph.replay()
                        actual = captured_output.clone()
                        torch.cuda.synchronize(device)
                        expected = raw().clone()
                        torch.cuda.synchronize(device)
                        mismatch = int((actual != expected).sum())
                        relative_l2 = float(
                            (actual.float() - expected.float()).norm()
                            / expected.float().norm().clamp_min(1e-20)
                        )
                        finite = bool(torch.isfinite(actual).all())
                        changed = (
                            int((expected != original_expected).sum())
                            if original_expected is not None
                            else 0
                        )
                        check = dict(
                            phase=phase,
                            bit_mismatches=mismatch,
                            rel_l2=relative_l2,
                            finite=finite,
                            changed_from_original=changed,
                        )
                        audit["checks"].append(check)
                        good = torch.tensor(
                            int(mismatch == 0 and finite), device=device
                        )
                        dist.all_reduce(good, op=dist.ReduceOp.MIN)
                        if not good.item():
                            print(
                                "GRAPH_VALIDATION_FAIL " + json.dumps(check), flush=True
                            )
                            raise AssertionError(
                                "Graph differs from full eager EP output"
                            )
                        return expected

                    original_expected = validate("original_pre")
                    # Only source rank 0 changes: its computation is routed to remote experts.
                    if rank == 0:
                        hidden_q.neg_()
                        hidden_scale.mul_(1.125)
                    validate("remote_input_changed")
                    for tensor, saved in zip(tensors, snapshots):
                        tensor.copy_(saved)
                    if rank == 0:
                        local_experts = moe.w13_weight.shape[0]
                        changed_ids = (
                            torch.arange(topk_ids.shape[1], device=device)
                            + local_experts
                        ) % (local_experts * world)
                        topk_ids.copy_(changed_ids.expand_as(topk_ids))
                        topk_weights.mul_(0.5)
                    validate("remote_routes_changed")
                    for tensor, saved in zip(tensors, snapshots):
                        tensor.copy_(saved)
                    validate("original_restored_pre")
                    assert [digest(t) for t in tensors] == audit["input_sha256"]
                    # Check that mutation really exercised a data-dependent result on source 0.
                    if rank == 0:
                        assert all(
                            c["changed_from_original"] > 0
                            for c in audit["checks"]
                            if c["phase"]
                            in ("remote_input_changed", "remote_routes_changed")
                        )
                    dist.barrier()
                    if extra.graph_profile:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof:
                            graph.replay()
                            torch.cuda.synchronize(device)
                        prof.export_chrome_trace(
                            str(
                                extra.graph_audit_dir
                                / f"replay-np{world}-t{tokens}-seq{len(AUDITS)}-rank{rank}.json"
                            )
                        )
                        audit["profile_kernel_names"] = [
                            event.name
                            for event in prof.events()
                            if event.device_type == torch.autograd.DeviceType.CUDA
                        ]
                    for _ in range(warmup):
                        graph.replay()
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    starts = [
                        torch.cuda.Event(enable_timing=True) for _ in range(iters)
                    ]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                    for start, end in zip(starts, ends):
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        dist.barrier()
                    # Validate the last measured output BEFORE executing any new MoE call.
                    measured_mismatches = int(
                        (captured_output != original_expected).sum()
                    )
                    finite = bool(torch.isfinite(captured_output).all())
                    audit["checks"].append(
                        dict(
                            phase="post_measure",
                            bit_mismatches=measured_mismatches,
                            finite=finite,
                            rel_l2=0.0 if measured_mismatches == 0 else None,
                        )
                    )
                    good = torch.tensor(
                        int(measured_mismatches == 0 and finite), device=device
                    )
                    dist.all_reduce(good, op=dist.ReduceOp.MIN)
                    assert good.item(), "last measured Graph output differs"
                    local = torch.tensor(
                        [s.elapsed_time(e) for s, e in zip(starts, ends)],
                        dtype=torch.float64,
                        device=device,
                    )
                    rank_max = local.clone()
                    dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
                    means = [
                        torch.empty(1, dtype=torch.float64, device=device)
                        for _ in range(world)
                    ]
                    dist.all_gather(means, local.mean().reshape(1))
                    adapter.output = captured_output
                    adapter._captured_graph = graph
                    adapter._static_inputs = tensors
                    audit["capture_complete"] = True
                    AUDITS.append(audit)
                    with (extra.graph_audit_dir / f"np{world}-rank{rank}.jsonl").open(
                        "a"
                    ) as f:
                        f.write(json.dumps(audit) + "\n")
                    print("GRAPH_VALIDATION " + json.dumps(audit), flush=True)
                    dist.barrier()
                    return rank_max.cpu().tolist(), [float(x.item()) for x in means]

        def destroy_after_graphs():
            # MCCL communicators can retain captured graph user objects. Destroy the
            # graph execs before communicator teardown, while the device is alive.
            torch.cuda.synchronize()
            for adapter in list(LIVE_ADAPTERS):
                graph = getattr(adapter, "_captured_graph", None)
                if graph is not None:
                    graph.reset()
                    adapter._captured_graph = None
                    adapter._static_inputs = None
            gc.collect()
            torch.cuda.synchronize()
            print("GRAPH_CLEANUP before_communicator_destroy", flush=True)
            original_destroy()
            print("GRAPH_CLEANUP communicator_destroyed", flush=True)

        destroy_model_parallel = destroy_after_graphs
        parse_args = parse_args
        PrequantizedInt8MoE = GraphAdapter
        benchmark = graph_benchmark
        TIMING_PROTOCOL = PROTOCOL
        main()
        if (
            int(__import__("os").environ.get("RANK", "0")) == 0
            and ARGS.json_output is not None
        ):
            records = json.loads(ARGS.json_output.read_text())
            assert len(records) == len(AUDITS)
            for record, audit in zip(records, AUDITS):
                assert record["tokens_per_rank"] == audit["tokens"]
                record["execution_mode"] = "cuda_graph"
                record["performance"][
                    "timing"
                ] = "CUDA events around full native EP graph.replay; individual sample completion"
                record["graph_validation"] = audit
            ARGS.json_output.write_text(json.dumps(records, indent=2) + "\n")

    def _run_candidate_worker(*, graph):
        import argparse
        import builtins
        import ctypes
        import hashlib
        import importlib
        import json
        import os
        import sys
        from pathlib import Path

        import torch
        import triton

        from benchmark.test_megamoe import TIMING_PROTOCOL as _eager_protocol
        from benchmark.test_megamoe import create_workspace, make_inputs
        from benchmark.test_megamoe import measure as _eager_measure
        from benchmark.test_megamoe import reference, zero_peer
        from flaggems_vllm.testing.megamoe import assert_moe_output
        from flaggems_vllm.testing.megamoe import validate as validate

        print = builtins.print
        # Copyright 2026 FlagOS Contributors
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.

        measure = _eager_measure
        TIMING_PROTOCOL = _eager_protocol

        def _run_worker():
            ap = argparse.ArgumentParser()
            ap.add_argument("--worker", action="store_true")
            ap.add_argument("--synthetic", action="store_true")
            ap.add_argument("--num-ranks", type=int)
            ap.add_argument("--hidden", type=int, default=2048)
            ap.add_argument("--intermediate", type=int, default=512)
            ap.add_argument("--num-experts", type=int, default=256)
            ap.add_argument("--topk", type=int, default=8)
            ap.add_argument(
                "--host-lib",
                default=os.environ.get("MEGAMOE_HOST_LIB", "libmegamoe_host.so"),
            )
            ap.add_argument("--data")
            ap.add_argument("--references")
            ap.add_argument("--tokens", default="1,128,256")
            ap.add_argument("--check-formula", action="store_true")
            ap.add_argument("--benchmark", action="store_true")
            ap.add_argument("--stress-batched-partials", action="store_true")
            ap.add_argument("--warmup", type=int, default=20)
            ap.add_argument("--iters", type=int, default=100)
            ap.add_argument("--repeats", type=int, default=5)
            args = ap.parse_args()
            if args.warmup < 0 or args.iters < 1 or args.repeats < 1:
                ap.error(
                    "warmup must be nonnegative; iters and repeats must be positive"
                )
            if args.data and not args.references:
                ap.error("--data requires --references")
            lib = ctypes.CDLL(args.host_lib)
            lib.mxshmem_alloc_bytes_wrapper.argtypes = [ctypes.c_size_t]
            lib.mxshmem_alloc_bytes_wrapper.restype = ctypes.c_void_p
            lib.mxshmem_ptr_wrapper.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.mxshmem_ptr_wrapper.restype = ctypes.c_void_p
            lib.mxshmem_free_wrapper.argtypes = [ctypes.c_void_p]
            lib.mxshmem_broadcast_bytes_wrapper.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.mxshmem_all_success_wrapper.argtypes = [ctypes.c_int]
            lib.mxshmem_all_success_wrapper.restype = ctypes.c_int
            # Initialize native schemas before library compatibility schemas.
            importlib.import_module("vllm._custom_ops")
            from flaggems_vllm import megamoe

            torch.cuda.set_device(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
            # MACA's SHMEM stream setup needs the selected device's primary context.
            context_guard = torch.empty(1, device="cuda")  # noqa: F841
            torch.cuda.synchronize()
            lib.mxshmem_init_wrapper()
            rank, world = lib.mxshmem_mype_wrapper(), lib.mxshmem_npes_wrapper()
            if args.num_ranks is not None:
                assert world == args.num_ranks
            allocations = []
            failures = 0

            def release_allocations():
                torch.cuda.synchronize()
                lib.mxshmem_barrier_all_wrapper()
                for allocation in allocations:
                    lib.mxshmem_free_wrapper(allocation)
                allocations.clear()

            def allocate(nbytes):
                lib.mxshmem_barrier_all_wrapper()
                pointer = lib.mxshmem_alloc_bytes_wrapper(nbytes)
                if not lib.mxshmem_all_success_wrapper(int(bool(pointer))):
                    raise RuntimeError(
                        "symmetric allocation failed on one or more ranks"
                    )
                allocations.append(pointer)
                addresses = [
                    pointer if peer == rank else lib.mxshmem_ptr_wrapper(pointer, peer)
                    for peer in range(world)
                ]
                if not lib.mxshmem_all_success_wrapper(int(all(addresses))):
                    raise RuntimeError(
                        "symmetric peer mapping failed on one or more ranks"
                    )
                table = torch.tensor(addresses, device="cuda", dtype=torch.int64)
                zero_peer[(triton.cdiv(nbytes, 1024),)](table, rank, nbytes, 1024)
                torch.cuda.synchronize()
                lib.mxshmem_barrier_all_wrapper()
                return table

            if args.data:

                dataset = load_qwen3_canonical(args.data, torch=torch)
                shared, experts = dataset["shared"], dataset["experts"]
                e = experts["l1_weight_int8"].shape[0] // world
                owned = slice(rank * e, (rank + 1) * e)
                w1 = experts["l1_weight_int8"][owned].to("cuda")
                w2 = experts["l2_weight_int8"][owned].to("cuda")
                s1 = experts["l1_weight_scale"][owned].reshape(e, -1).to("cuda")
                s2 = experts["l2_weight_scale"][owned].reshape(e, -1).to("cuda")
                sizes = [int(t) for t in args.tokens.split(",")]
                for t in sizes:
                    diagnostics = dict(rank=rank, world=world, tokens=t)
                    try:
                        fixture = torch.load(
                            Path(args.references) / f"t{t}_rank{rank}.pt",
                            map_location="cpu",
                            weights_only=False,
                        )
                        inputs = dict(
                            hidden_q=shared["input_int8"][rank, :t],
                            hidden_scale=shared["input_scales"][rank, :t],
                            topk_ids=shared["topk_idx"][rank, :t],
                            topk_weights=shared["topk_weights"][rank, :t],
                        )
                        hashes = {
                            name: hashlib.sha256(
                                tensor.contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest()
                            for name, tensor in inputs.items()
                        }
                        diagnostics["input_hash_bad"] = [
                            name
                            for name, digest in hashes.items()
                            if digest != fixture["input_sha256"][name]
                        ]
                        diagnostics["input_hashes_match"] = not diagnostics[
                            "input_hash_bad"
                        ]
                    except Exception as error:
                        diagnostics["input_error"] = f"{type(error).__name__}: {error}"
                        diagnostics["input_hashes_match"] = False
                    if not lib.mxshmem_all_success_wrapper(
                        int(diagnostics["input_hashes_match"])
                    ):
                        failures += 1
                        diagnostics.update(kind="shape_summary", status="FAIL")
                        print(json.dumps(diagnostics), flush=True)
                        continue
                    x, xs, routes, weights = (
                        inputs[name].to("cuda")
                        for name in (
                            "hidden_q",
                            "hidden_scale",
                            "topk_ids",
                            "topk_weights",
                        )
                    )
                    # Capacity is a deployment choice, so benchmark each real shape with
                    # its own capacity; a 16384-token workspace distorts every small case.
                    with create_workspace(
                        x,
                        w1,
                        w2,
                        weights,
                        routes,
                        s1,
                        s2,
                        a1_scale=xs,
                        max_tokens=t,
                        rank=rank,
                        world_size=world,
                        allocate_symmetric=allocate,
                        release_symmetric=release_allocations,
                        shmem=lib,
                    ) as ws:
                        out = torch.empty(
                            (t, x.shape[1]), device=x.device, dtype=torch.bfloat16
                        )

                        def run():
                            return megamoe(
                                x,
                                w1,
                                w2,
                                weights,
                                routes,
                                s1,
                                s2,
                                workspace=ws,
                                a1_scale=xs,
                                out=out,
                            )

                        y = run()
                        torch.cuda.synchronize()
                        pre_passed = validate(
                            ws,
                            w1,
                            w2,
                            s1,
                            s2,
                            y,
                            routes,
                            fixture,
                            shared,
                            lib,
                            args.check_formula,
                            "pre",
                        )
                        post_passed = None
                        if args.benchmark and pre_passed:
                            means = measure(
                                run, lib, rank, t, args.warmup, args.iters, args.repeats
                            )
                            # Validate the final measured invocation without replacing its output.
                            post_passed = validate(
                                ws,
                                w1,
                                w2,
                                s1,
                                s2,
                                out,
                                routes,
                                fixture,
                                shared,
                                lib,
                                args.check_formula,
                                "post",
                            )
                            print(
                                json.dumps(
                                    dict(
                                        kind="benchmark",
                                        rank=rank,
                                        world=world,
                                        tokens=t,
                                        warmup=args.warmup,
                                        iters=args.iters,
                                        repeats=args.repeats,
                                        mean_us=means,
                                        timing="CUDA events at launch hooks; individual sample completion",
                                        timing_protocol=TIMING_PROTOCOL,
                                        external_barriers_in_timing=False,
                                        internal_barriers_in_timing=True,
                                        includes=(
                                            "dispatch, dynamic routing, both GEMMs, "
                                            "activation, quantization, combine"
                                        ),
                                        capacity=ws.capacity,
                                        block_m=ws.bm,
                                        correctness="PASS" if post_passed else "FAIL",
                                    )
                                ),
                                flush=True,
                            )
                        passed = pre_passed and (post_passed is not False)
                        diagnostics.update(
                            kind="shape_summary",
                            status="PASS" if passed else "FAIL",
                            pre_passed=pre_passed,
                            post_passed=post_passed,
                            formula_checked=args.check_formula,
                            quantization_contract="symmetric-int8-rne-v1",
                            capacity=ws.capacity,
                            block_m=ws.bm,
                            block_n=ws.bn,
                            block_k=ws.bk,
                            workers=ws.workers,
                        )
                        print(json.dumps(diagnostics), flush=True)
                        if not passed:
                            failures += 1
                        del run, ws, y, out
            elif args.synthetic:
                assert args.num_experts % world == 0
                for t in map(int, args.tokens.split(",")):
                    x, xs, full_w1, full_w2, full_s1, full_s2, routes, weights = (
                        make_inputs(
                            t,
                            args.hidden,
                            args.intermediate,
                            args.num_experts,
                            args.topk,
                            device="cpu",
                        )
                    )
                    e = args.num_experts // world
                    owned = slice(rank * e, (rank + 1) * e)
                    x, xs, routes, weights = (
                        value.to("cuda") for value in (x, xs, routes, weights)
                    )
                    w1, w2, s1, s2 = (
                        value[owned].to("cuda")
                        for value in (full_w1, full_w2, full_s1, full_s2)
                    )
                    del full_w1, full_w2, full_s1, full_s2
                    with create_workspace(
                        x,
                        w1,
                        w2,
                        weights,
                        routes,
                        s1,
                        s2,
                        a1_scale=xs,
                        max_tokens=t,
                        rank=rank,
                        world_size=world,
                        allocate_symmetric=allocate,
                        release_symmetric=release_allocations,
                        shmem=lib,
                    ) as ws:
                        out = torch.empty(
                            (t, args.hidden), device="cuda", dtype=torch.bfloat16
                        )

                        def run():
                            return megamoe(
                                x,
                                w1,
                                w2,
                                weights,
                                routes,
                                s1,
                                s2,
                                workspace=ws,
                                a1_scale=xs,
                                out=out,
                            )

                        # Warm the explicit workspace specialization before timing, including --warmup=0.
                        run()
                        means = measure(
                            run, lib, rank, t, args.warmup, args.iters, args.repeats
                        )
                        if not lib.mxshmem_all_success_wrapper(
                            int(torch.isfinite(out).all())
                        ):
                            failures += 1
                        for repeat, mean in enumerate(means):
                            print(
                                f"MEGAMOE_RESULT scope=kernel rank={rank} world={world} "
                                f"repeat={repeat} latency_us={mean:.9f}",
                                flush=True,
                            )
                        del run, out
            else:
                sizes = (
                    (1025, 33, 0, 129)
                    if args.stress_batched_partials
                    else (19, 3, 0, 17)
                )
                capacity = (
                    torch.cuda.get_device_properties(0).multi_processor_count * 64
                    if args.stress_batched_partials
                    else 19
                )
                x, xs, full_w1, full_w2, full_s1, full_s2, routes, weights = (
                    make_inputs(max(sizes), 128, 70, world * 3, 2)
                )
                owned = slice(rank * 3, (rank + 1) * 3)
                w1, w2, s1, s2 = (
                    a[owned] for a in (full_w1, full_w2, full_s1, full_s2)
                )
                with create_workspace(
                    x,
                    w1,
                    w2,
                    weights,
                    routes,
                    s1,
                    s2,
                    a1_scale=xs,
                    workers=2 if args.stress_batched_partials else None,
                    max_tokens=capacity,
                    rank=rank,
                    world_size=world,
                    allocate_symmetric=allocate,
                    release_symmetric=release_allocations,
                    shmem=lib,
                ) as ws:
                    for iteration in range(8):
                        t = sizes[(iteration + rank) % 4]
                        current = routes[:t].clone()
                        current[:, 0] = (iteration + 1) % (world * 3)
                        current[:, 1] = -1
                        y = megamoe(
                            x[:t],
                            w1,
                            w2,
                            weights[:t],
                            current,
                            s1,
                            s2,
                            workspace=ws,
                            a1_scale=xs[:t],
                        )
                        expected = reference(
                            x[:t],
                            xs[:t],
                            full_w1,
                            full_w2,
                            full_s1,
                            full_s2,
                            current,
                            weights[:t],
                        ).to("cuda")
                        try:
                            assert_moe_output(y, expected)
                            good = True
                        except AssertionError:
                            good = False
                        close = y == expected
                        if not good:
                            bad = (~close).nonzero()[:8]
                            print(
                                json.dumps(
                                    dict(
                                        rank=rank,
                                        iteration=iteration,
                                        kind="dynamic_mismatch",
                                        count=int((~close).sum()),
                                        coordinates=bad.cpu().tolist(),
                                        actual=y[bad[:, 0], bad[:, 1]]
                                        .float()
                                        .cpu()
                                        .tolist(),
                                        expected=expected[bad[:, 0], bad[:, 1]]
                                        .float()
                                        .cpu()
                                        .tolist(),
                                        relative_l2=float(
                                            (y.float() - expected.float()).norm()
                                            / expected.float().norm().clamp_min(1e-10)
                                        ),
                                    )
                                ),
                                flush=True,
                            )
                        if not lib.mxshmem_all_success_wrapper(int(good)):
                            failures += 1
                            print(
                                json.dumps(
                                    dict(
                                        rank=rank,
                                        iteration=iteration,
                                        status="DYNAMIC_FAIL",
                                    )
                                ),
                                flush=True,
                            )
                    print(
                        json.dumps(
                            dict(
                                rank=rank,
                                world=world,
                                iterations=8,
                                status="DYNAMIC_FAIL" if failures else "DYNAMIC_PASS",
                            )
                        ),
                        flush=True,
                    )
                    t = sizes[1] if rank % 2 == 0 else sizes[0]
                    current = routes[:t].clone()
                    out = torch.empty((t, 128), device="cuda", dtype=torch.bfloat16)
                    megamoe(
                        x[:t],
                        w1,
                        w2,
                        weights[:t],
                        current,
                        s1,
                        s2,
                        workspace=ws,
                        a1_scale=xs[:t],
                        out=out,
                    )
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        megamoe(
                            x[:t],
                            w1,
                            w2,
                            weights[:t],
                            current,
                            s1,
                            s2,
                            workspace=ws,
                            a1_scale=xs[:t],
                            out=out,
                        )
                    graph_failures = 0
                    for iteration in range(4):
                        current[:, 0] = iteration % (world * 3)
                        current[:, 1] = -1
                        graph.replay()
                        expected = reference(
                            x[:t],
                            xs[:t],
                            full_w1,
                            full_w2,
                            full_s1,
                            full_s2,
                            current,
                            weights[:t],
                        ).to("cuda")
                        try:
                            assert_moe_output(out, expected)
                            good = True
                        except AssertionError:
                            good = False
                        if not lib.mxshmem_all_success_wrapper(int(good)):
                            graph_failures += 1
                            print(
                                json.dumps(
                                    dict(
                                        rank=rank,
                                        iteration=iteration,
                                        status="GRAPH_FAIL",
                                    )
                                ),
                                flush=True,
                            )
                    failures += graph_failures
                    print(
                        json.dumps(
                            dict(
                                rank=rank,
                                world=world,
                                replays=4,
                                status="GRAPH_FAIL" if graph_failures else "GRAPH_PASS",
                            )
                        ),
                        flush=True,
                    )
                    del graph, ws
            release_allocations()
            lib.mxshmem_finalize_wrapper()
            del context_guard
            return int(failures > 0)

        if not graph:
            return _run_worker()

        # Copyright 2026 FlagOS Contributors
        # SPDX-License-Identifier: Apache-2.0

        PROTOCOL = "pr774-full-ep-both-cudagraph-5-mean-v2"
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--graph-audit-dir", type=Path, required=True)
        p.add_argument("--graph-profile", action="store_true")
        extra, rest = p.parse_known_args()
        sys.argv = [sys.argv[0], *rest]
        extra.graph_audit_dir.mkdir(parents=True, exist_ok=True)

        def graph_measure(run, lib, rank, tokens, warmup, iters, repeats):
            values = dict(
                zip(
                    run.__code__.co_freevars, (x.cell_contents for x in run.__closure__)
                )
            )
            inputs = tuple(values[k] for k in ("x", "xs", "weights", "routes"))
            out, ws = values["out"], values["ws"]
            device = out.device
            audit = dict(
                rank=rank,
                world=ws.world_size,
                tokens=tokens,
                checks=[],
                capacity=ws.capacity,
                workers=ws.workers,
                block_m=ws.bm,
                block_n=ws.bn,
                block_k=ws.bk,
                timing_protocol=PROTOCOL,
            )
            lib.mxshmemx_barrier_wrapper.argtypes = [ctypes.c_void_p]
            lib.mxshmemx_barrier_wrapper.restype = None

            def sync():
                torch.cuda.synchronize(device)
                assert lib.mxshmem_all_success_wrapper(1)

            def barrier():
                lib.mxshmemx_barrier_wrapper(
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
                )

            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(max(warmup, 3)):
                    run()
            torch.cuda.current_stream(device).wait_stream(stream)
            sync()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run()
            assert captured.data_ptr() == out.data_ptr()
            sync()
            snapshots = [x.clone() for x in inputs]
            original = None

            def validate(phase):
                graph.replay()
                actual = out.clone()
                sync()
                expected = run().clone()
                sync()
                mismatch = int((actual != expected).sum())
                finite = bool(torch.isfinite(actual).all())
                changed = (
                    int((expected != original).sum()) if original is not None else 0
                )
                check = dict(
                    phase=phase,
                    bit_mismatches=mismatch,
                    finite=finite,
                    changed_from_original=changed,
                )
                audit["checks"].append(check)
                assert lib.mxshmem_all_success_wrapper(
                    int(mismatch == 0 and finite)
                ), check
                return expected

            original = validate("original_pre")
            if rank == 0:
                values["x"].neg_()
                values["xs"].mul_(1.125)
            validate("remote_input_changed")
            for x, saved in zip(inputs, snapshots):
                x.copy_(saved)
            if rank == 0:
                routes = values["routes"]
                ids = (
                    torch.arange(routes.shape[1], device=device) + ws.model_shape[0]
                ) % (ws.model_shape[0] * ws.world_size)
                routes.copy_(ids.expand_as(routes))
                values["weights"].mul_(0.5)
            validate("remote_routes_changed")
            for x, saved in zip(inputs, snapshots):
                x.copy_(saved)
            validate("original_restored_pre")
            if rank == 0:
                assert all(
                    c["changed_from_original"] > 0
                    for c in audit["checks"]
                    if c["phase"].startswith("remote_")
                )
            sync()
            if extra.graph_profile:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    graph.replay()
                    torch.cuda.synchronize(device)
                prof.export_chrome_trace(
                    str(
                        extra.graph_audit_dir
                        / f"replay-np{ws.world_size}-t{tokens}-rank{rank}.json"
                    )
                )
                audit["profile_kernel_names"] = [
                    e.name
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
            means = []
            for repeat in range(repeats):
                for _ in range(warmup):
                    graph.replay()
                sync()
                times = []
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                for _ in range(iters):
                    barrier()
                    start.record()
                    graph.replay()
                    end.record()
                    barrier()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 1000.0)
                    sync()
                mean = sum(times) / len(times)
                means.append(mean)
                print(
                    f"BENCH_REPEAT rank={rank} tokens={tokens} repeat={repeat} mean_us={mean:.9f}",
                    flush=True,
                )
            # Inspect measured output before any eager refresh.
            mismatch = int((out != original).sum())
            finite = bool(torch.isfinite(out).all())
            audit["checks"].append(
                dict(phase="post_measure", bit_mismatches=mismatch, finite=finite)
            )
            assert lib.mxshmem_all_success_wrapper(int(mismatch == 0 and finite))
            audit["capture_complete"] = True
            path = extra.graph_audit_dir / f"np{ws.world_size}-rank{rank}.jsonl"
            with path.open("a") as f:
                f.write(json.dumps(audit) + "\n")
            print("GRAPH_VALIDATION " + json.dumps(audit), flush=True)
            sync()
            graph.reset()
            return means

        def emit(*args, **kwargs):
            import builtins

            if len(args) == 1 and isinstance(args[0], str) and args[0].startswith("{"):
                record = json.loads(args[0])
                if record.get("kind") == "benchmark":
                    record["timing"] = (
                        "CUDA events around full graph.replay; individual sample completion"
                    )
                    record["execution_mode"] = "cuda_graph"
                    args = (json.dumps(record),)
            builtins.print(*args, **kwargs)

        print = emit
        measure = graph_measure
        TIMING_PROTOCOL = PROTOCOL
        _run_worker()

    if __name__ == "__main__":
        position = sys.argv.index("--worker")
        role = sys.argv[position + 1]
        del sys.argv[position : position + 2]
        if role in ("baseline", "graph-baseline"):
            _run_vendor_worker(graph=role == "graph-baseline")
        elif role in ("candidate", "graph-candidate"):
            _run_candidate_worker(graph=role == "graph-candidate")
        else:
            raise SystemExit(f"Unknown MegaMoE worker role: {role}")
else:
    import argparse
    import os
    import re
    import signal
    import subprocess
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    # FlagTree's raw CUDA cache must be private to each worker process. Establish
    # this before importing Triton through the benchmark harness or MegaMoE module.
    _IS_WORKER = "--worker" in sys.argv
    if _IS_WORKER:
        _local_rank = os.environ.get("LOCAL_RANK", "0")
        _cache_root = Path(
            os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")
        )
        os.environ["TRITON_CACHE_DIR"] = str(
            _cache_root.parent / f"{_cache_root.name}-megamoe-rank-{_local_rank}"
        )

    import torch  # noqa: E402

    if _IS_WORKER:
        torch.cuda.set_device(int(_local_rank))

    import pytest  # noqa: E402

    from flaggems_vllm.testing.megamoe import StandaloneNvshmem  # noqa: E402
    from flaggems_vllm.testing.megamoe import (  # noqa: E402
        configure_descriptor_allocator,
        interleave_l1_gate_up_rows,
        make_context,
        pool_tokens,
    )

    from . import base  # noqa: E402
    from .conftest import Config  # noqa: E402
    from .consts import DEFAULT_WARMUP_TIME  # noqa: E402
    from .consts import DEFAULT_ITER_TIME, BenchLevel, BenchMode  # noqa: E402

    OP_NAME = "megamoe"
    NUM_RANKS = 8
    STAGES = 4
    CORE_TOKENS = (1, 64, 128, 256, 512, 2048, 16384)
    COMPREHENSIVE_TOKENS = (
        1,
        2,
        4,
        8,
        16,
        *range(24, 257, 8),
        *range(272, 513, 16),
        2048,
        16384,
    )
    DEFAULT_WARMUP = 10
    DEFAULT_ITERS = 30
    DEFAULT_TIMEOUT_S = 1800
    DEFAULT_MAX_RECV = 4096
    _RESULT = re.compile(
        r"MEGAMOE_RESULT scope=kernel rank=(\d+) world=(\d+) latency_us=([0-9.]+)"
    )

    @dataclass(frozen=True)
    class _BenchmarkConfig:
        num_ranks: int
        tokens: int
        hidden: int
        intermediate: int
        num_experts: int
        topk: int
        stages: int
        warmup: int
        iterations: int
        timeout: int = DEFAULT_TIMEOUT_S
        max_recv: int = DEFAULT_MAX_RECV

    def _make_inputs(config: _BenchmarkConfig, runtime: StandaloneNvshmem):
        rank = runtime.rank
        world_size = runtime.world_size
        device = runtime.device
        experts_per_rank = config.num_experts // world_size
        n_l1_tiles = config.intermediate // 64
        n_l2_tiles = config.hidden // 128
        n_k1_tiles = config.hidden // 128
        n_k2_tiles = config.intermediate // 128
        scale_columns = config.hidden // 128
        fp8 = torch.float8_e4m3fn

        cpu_generator = torch.Generator().manual_seed(1234)
        scores = torch.randn(
            world_size * config.tokens,
            config.num_experts,
            generator=cpu_generator,
        )
        topk_all = torch.topk(
            scores, config.topk, dim=-1, largest=True, sorted=False
        ).indices.reshape(world_size, config.tokens, config.topk)
        x = torch.randn(config.tokens, config.hidden, generator=cpu_generator) * 0.5
        x = x.to(device=device, dtype=fp8)
        x_scales = (
            torch.rand(config.tokens, scale_columns, generator=cpu_generator).mul_(0.5)
            + 0.5
        ).to(device)
        topk_weights = (
            torch.rand(config.tokens, config.topk, generator=cpu_generator).mul_(0.8)
            + 0.2
        ).to(device)

        gpu_generator = torch.Generator(device=device).manual_seed(9000 + rank)
        w1 = torch.randn(
            experts_per_rank,
            n_l1_tiles * 128,
            config.hidden,
            generator=gpu_generator,
            device=device,
        ).mul_(0.4)
        w1 = interleave_l1_gate_up_rows(w1.to(fp8))
        w1_scales = (
            torch.rand(
                experts_per_rank,
                n_l1_tiles,
                n_k1_tiles,
                generator=gpu_generator,
                device=device,
            ).mul_(0.4)
            + 0.3
        )
        w2 = torch.randn(
            experts_per_rank,
            n_l2_tiles * 128,
            config.intermediate,
            generator=gpu_generator,
            device=device,
        ).mul_(0.4)
        w2 = w2.to(fp8)
        w2_scales = (
            torch.rand(
                experts_per_rank,
                n_l2_tiles,
                n_k2_tiles,
                generator=gpu_generator,
                device=device,
            ).mul_(0.4)
            + 0.3
        )

        return (
            x,
            x_scales,
            topk_all[rank].to(device),
            topk_weights,
            w1,
            w1_scales,
            w2,
            w2_scales,
            topk_all,
        )

    def _run_worker(config: _BenchmarkConfig) -> None:
        from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megamoe import megamoe_fp8

        assert config.num_experts % config.num_ranks == 0
        with StandaloneNvshmem() as runtime:
            assert runtime.world_size == config.num_ranks
            configure_descriptor_allocator(runtime.device)
            inputs = _make_inputs(config, runtime)
            pool_size = pool_tokens(inputs[-1], config)
            context = make_context(config, runtime, pool_size)
            output = torch.empty(
                (config.tokens, config.hidden),
                dtype=torch.bfloat16,
                device=runtime.device,
            )
            operands = (*inputs[:-1], context, output)

            for _ in range(config.warmup):
                megamoe_fp8(*operands, stages=config.stages)
            torch.cuda.synchronize(runtime.device)

            samples = []
            for _ in range(config.iterations):
                runtime.arm_kernel_timing()
                megamoe_fp8(*operands, stages=config.stages)
                samples.append(runtime.finish_kernel_timing())

            latency = sum(samples) / len(samples)
            print(
                f"MEGAMOE_RESULT scope=kernel rank={runtime.rank} "
                f"world={runtime.world_size} "
                f"latency_us={latency:.3f}",
                flush=True,
            )

    def _worker_arguments(config: _BenchmarkConfig) -> list[str]:
        arguments = [
            "--worker",
            "--num-ranks",
            str(config.num_ranks),
            "--tokens",
            str(config.tokens),
            "--hidden",
            str(config.hidden),
            "--intermediate",
            str(config.intermediate),
            "--num-experts",
            str(config.num_experts),
            "--topk",
            str(config.topk),
            "--stages",
            str(config.stages),
            "--warmup",
            str(config.warmup),
            "--iterations",
            str(config.iterations),
            "--max-recv",
            str(config.max_recv),
        ]
        return arguments

    def _launch_megamoe(config: _BenchmarkConfig) -> float:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(config.num_ranks),
            "--module",
            "benchmark.test_megamoe",
            *_worker_arguments(config),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=config.timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, command, output=stdout
            )
        latencies = {
            int(rank): float(latency) / 1e3
            for rank, world, latency in _RESULT.findall(stdout)
            if int(world) == config.num_ranks
        }
        if set(latencies) != set(range(config.num_ranks)):
            raise RuntimeError(f"missing rank results:\n{stdout}")
        return max(latencies.values())

    def _iteration_count(option, configured, default, fallback):
        explicitly_set = any(
            arg == option or arg.startswith(f"{option}=") for arg in sys.argv
        )
        return configured if explicitly_set or configured != default else fallback

    class MegaMoEBenchmark(base.Benchmark):
        DEFAULT_METRICS = ["latency"]
        DEFAULT_DTYPES = [torch.float8_e4m3fn]
        DEFAULT_SHAPE_DESC = (
            "tokens, hidden, intermediate, num_experts, topk, num_ranks"
        )

        def __init__(self):
            super().__init__(
                op_name=OP_NAME,
                torch_op=None,
                gems_op=_launch_megamoe,
                dtypes=self.DEFAULT_DTYPES,
            )

        def set_shapes(self, shape_file_path=None):
            tokens = (
                COMPREHENSIVE_TOKENS
                if Config.bench_level is BenchLevel.COMPREHENSIVE
                else CORE_TOKENS
            )
            self.shapes = [(value, 2048, 512, 256, 8, NUM_RANKS) for value in tokens]
            self.shape_desc = self.DEFAULT_SHAPE_DESC

        def get_input_iter(self, dtype):
            yield from self.shapes

        def get_latency(
            self,
            op,
            tokens,
            hidden,
            intermediate,
            num_experts,
            topk,
            num_ranks,
        ):
            assert Config.mode is BenchMode.KERNEL
            return op(
                _BenchmarkConfig(
                    num_ranks=num_ranks,
                    tokens=tokens,
                    hidden=hidden,
                    intermediate=intermediate,
                    num_experts=num_experts,
                    topk=topk,
                    stages=STAGES,
                    warmup=_iteration_count(
                        "--warmup", Config.warm_up, DEFAULT_WARMUP_TIME, DEFAULT_WARMUP
                    ),
                    iterations=_iteration_count(
                        "--iter", Config.repetition, DEFAULT_ITER_TIME, DEFAULT_ITERS
                    ),
                )
            )

    @pytest.mark.megamoe
    def test_megamoe():
        if Config.query:
            MegaMoEBenchmark().run()
            return
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            pytest.skip(f"requires SM90, got SM{major}0")
        if torch.cuda.device_count() < NUM_RANKS:
            pytest.skip(f"requires {NUM_RANKS} visible GPUs")
        MegaMoEBenchmark().run()

    def _parse_worker_config() -> _BenchmarkConfig:
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true", required=True)
        parser.add_argument("--num-ranks", type=int, required=True)
        parser.add_argument("--tokens", type=int, required=True)
        parser.add_argument("--hidden", type=int, required=True)
        parser.add_argument("--intermediate", type=int, required=True)
        parser.add_argument("--num-experts", type=int, required=True)
        parser.add_argument("--topk", type=int, required=True)
        parser.add_argument("--stages", type=int, required=True)
        parser.add_argument("--warmup", type=int, required=True)
        parser.add_argument("--iterations", type=int, required=True)
        parser.add_argument("--max-recv", type=int, required=True)
        arguments = vars(parser.parse_args())
        arguments.pop("worker")
        return _BenchmarkConfig(**arguments)

    if __name__ == "__main__" and _IS_WORKER:
        _run_worker(_parse_worker_config())
