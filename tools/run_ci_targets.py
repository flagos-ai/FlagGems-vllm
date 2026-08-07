#!/usr/bin/env python3

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

"""Validate JSON CI targets and invoke pytest without shell expansion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

TARGET_ROOTS = {"tests": "tests", "benchmarks": "benchmark"}


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON array of strings")
    return value


def load_targets(value: str) -> dict[str, list[str]]:
    payload = json.loads(value)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("targets must be a schema_version 1 JSON object")
    return {
        "tests": _string_list(payload.get("tests"), "tests"),
        "benchmarks": _string_list(payload.get("benchmarks"), "benchmarks"),
    }


def validate_target(repo_root: Path, target: str, kind: str) -> str:
    path = PurePosixPath(target)
    expected_root = TARGET_ROOTS[kind]
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {kind} target: {target!r}")
    if len(path.parts) < 2 or path.parts[0] != expected_root:
        raise ValueError(f"invalid {kind} target root: {target!r}")
    if not path.name.startswith("test_") or path.suffix != ".py":
        raise ValueError(f"invalid {kind} target name: {target!r}")
    if not (repo_root / Path(*path.parts)).is_file():
        raise ValueError(f"selected {kind} target does not exist: {target!r}")
    return target


def load_policy(path: Path, backend: str) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("backend capabilities must use schema_version 1")

    defaults = config.get("defaults")
    backends = config.get("backends")
    if not isinstance(defaults, dict) or not isinstance(backends, dict):
        raise ValueError("backend capabilities require defaults and backends objects")

    override = backends.get(backend, {})
    if not isinstance(override, dict):
        raise ValueError(f"capabilities for {backend!r} must be an object")
    policy = {**defaults, **override}
    policy["tests_allow"] = _string_list(policy.get("tests_allow"), "tests_allow")
    policy["benchmarks_allow"] = _string_list(
        policy.get("benchmarks_allow"), "benchmarks_allow"
    )
    for field in (
        "allow_all_tests",
        "benchmarks_enabled",
        "allow_all_benchmarks",
    ):
        if not isinstance(policy.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    return policy


def apply_policy(
    targets: dict[str, list[str]], policy: dict[str, Any]
) -> dict[str, list[str]]:
    allowed_tests = set(policy["tests_allow"])
    tests = (
        targets["tests"]
        if policy["allow_all_tests"]
        else [target for target in targets["tests"] if target in allowed_tests]
    )

    benchmarks: list[str] = []
    if policy["benchmarks_enabled"]:
        allowed_benchmarks = set(policy["benchmarks_allow"])
        benchmarks = (
            targets["benchmarks"]
            if policy["allow_all_benchmarks"]
            else [
                target
                for target in targets["benchmarks"]
                if target in allowed_benchmarks
            ]
        )
    return {
        "tests": list(dict.fromkeys(tests)),
        "benchmarks": list(dict.fromkeys(benchmarks)),
    }


def build_commands(targets: dict[str, list[str]]) -> list[list[str]]:
    commands = []
    if targets["tests"]:
        commands.append(
            [sys.executable, "-m", "pytest", "-q", "--quick", *targets["tests"]]
        )
    if targets["benchmarks"]:
        commands.append(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--level",
                "core",
                "--warmup",
                "1",
                "--iter",
                "1",
                *targets["benchmarks"],
            ]
        )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--targets-env",
        default="CI_TARGETS_JSON",
        help="environment variable containing the targets JSON",
    )
    parser.add_argument(
        "--capabilities",
        type=Path,
        default=Path(".github/backend-capabilities.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_targets = os.environ.get(args.targets_env)
    if raw_targets is None:
        raise ValueError(f"environment variable {args.targets_env!r} is not set")

    requested_targets = load_targets(raw_targets)
    for kind, values in requested_targets.items():
        requested_targets[kind] = [
            validate_target(args.repo_root, target, kind) for target in values
        ]

    policy = load_policy(args.capabilities, args.backend)
    targets = apply_policy(requested_targets, policy)
    for kind in ("tests", "benchmarks"):
        dropped = [
            target for target in requested_targets[kind] if target not in targets[kind]
        ]
        if dropped:
            print(
                f"Not approved for {args.backend} ({kind}):",
                json.dumps(dropped, ensure_ascii=False),
            )

    commands = build_commands(targets)
    if not commands:
        print(f"No approved tests or benchmarks for backend {args.backend}.")
        return 0

    for command in commands:
        print("Running:", json.dumps(command, ensure_ascii=False))
        if not args.dry_run:
            subprocess.run(command, cwd=args.repo_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
