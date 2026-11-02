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

"""Derive which operators are affected by a PR based on git diff.

Outputs (as GitHub Actions outputs):
  - changed_operators: JSON list of operator IDs
  - changed_files: JSON list of changed file paths
  - has_changes: 'true' or 'false'

Exit codes:
  0 - success
  2 - script internal error
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# Paths relative to repo root
OPERATORS_YAML = "conf/operators.yaml"
OPS_DIR = "src/flaggems_vllm/ops"
TESTS_DIR = "tests"
INIT_FILE = "src/flaggems_vllm/__init__.py"
OPS_INIT_FILE = "src/flaggems_vllm/ops/__init__.py"

# Generic ops file: src/flaggems_vllm/ops/<name>.py
OPS_FILE_RE = re.compile(r"^src/flaggems_vllm/ops/([^/]+)\.py$")

# Subpackage ops file: src/flaggems_vllm/ops/<subpkg>/<name>.py
# e.g. ops/FLA/chunk_kda.py, ops/DSA/bin_topk.py, ops/mhc/mhc_post.py, ops/qwen4/qsa.py
SUBPKG_OPS_FILE_RE = re.compile(r"^src/flaggems_vllm/ops/([^/]+)/([^/]+)\.py$")

# Backend operator implementations, e.g.
#   src/flaggems_vllm/runtime/backend/_nvidia/ops/xxx.py
#   src/flaggems_vllm/runtime/backend/_nvidia/hopper/ops/xxx.py
BACKEND_OPS_FILE_RE = re.compile(
    r"^src/flaggems_vllm/runtime/backend/_[^/]+/(?:[^/]+/)*ops/(.+)\.py$"
)

# Test files: tests/test_<name>.py or tests/test_<subdir>/test_<name>.py
TEST_FILE_RE = re.compile(r"^tests/test_(.+)\.py$")
TEST_SUBDIR_FILE_RE = re.compile(r"^tests/[^/]+/test_(.+)\.py$")

# Benchmark files
BENCHMARK_FILE_RE = re.compile(r"^benchmark/test_(.+)\.py$")


def get_diff_files(base_sha: str, head_sha: str) -> list[str]:
    """Get list of changed files introduced by the PR branch.

    Uses a three-dot diff (``base...head``), which compares ``head`` against
    the merge-base of ``base`` and ``head``.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{base_sha}...{head_sha}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError as e:
        print(f"::error::Failed to get git diff: {e.stderr}", file=sys.stderr)
        sys.exit(2)


def load_operators_yaml() -> dict[str, dict]:
    """Load operators.yaml and return a dict keyed by operator id."""
    yaml_path = Path(OPERATORS_YAML)
    if not yaml_path.exists():
        print(f"::error::Cannot find {OPERATORS_YAML}", file=sys.stderr)
        sys.exit(2)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    ops = data.get("ops", [])
    return {op["id"]: op for op in ops if "id" in op}


def derive_operators(changed_files: list[str], all_operators: dict) -> list[str]:
    """Given changed files, derive which operator IDs are affected."""
    changed_ops = set()

    for filepath in changed_files:
        # Case 1: operators.yaml itself changed -> flag for full check
        if filepath == OPERATORS_YAML:
            changed_ops.add("__operators_yaml_changed__")
            continue

        # Case 2: generic ops source file (src/flaggems_vllm/ops/<name>.py)
        m = OPS_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            if stem == "__init__":
                changed_ops.add("__init_changed__")
                continue
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                # Still track it for downstream checks
                changed_ops.add(stem)
            continue

        # Case 3: subpackage ops (src/flaggems_vllm/ops/<subpkg>/<name>.py)
        m = SUBPKG_OPS_FILE_RE.match(filepath)
        if m:
            subpkg = m.group(1)
            stem = m.group(2)
            if stem == "__init__":
                continue
            # Try exact match with stem
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                # Try subpkg_stem combination
                combined = f"{subpkg}_{stem}"
                if combined in all_operators:
                    changed_ops.add(combined)
                else:
                    changed_ops.add(stem)
            continue

        # Case 4: backend ops (src/flaggems_vllm/runtime/backend/_<vendor>/[arch/]ops/<name>.py)
        m = BACKEND_OPS_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            if stem == "__init__":
                continue
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                changed_ops.add(stem)
            continue

        # Case 5: test file changed (tests/test_<name>.py)
        m = TEST_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                changed_ops.add(stem)
            continue

        # Case 6: subdirectory test file (tests/<subdir>/test_<name>.py)
        m = TEST_SUBDIR_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            if stem in all_operators:
                changed_ops.add(stem)
            else:
                changed_ops.add(stem)
            continue

        # Case 7: benchmark file changed
        m = BENCHMARK_FILE_RE.match(filepath)
        if m:
            stem = m.group(1)
            # Strip _perf suffix if present
            if stem.endswith("_perf"):
                stem = stem[:-5]
            if stem in all_operators:
                changed_ops.add(stem)
            continue

        # Case 8: main __init__.py or ops __init__.py changed
        if filepath in (INIT_FILE, OPS_INIT_FILE):
            changed_ops.add("__init_changed__")
            continue

    # Remove sentinel markers from operator list for downstream
    sentinel_markers = {"__operators_yaml_changed__", "__init_changed__"}
    real_ops = sorted(changed_ops - sentinel_markers)

    return real_ops


def set_output(name: str, value: str):
    """Set a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            if "\n" in value:
                f.write(f"{name}<<EOF\n{value}\nEOF\n")
            else:
                f.write(f"{name}={value}\n")
    else:
        # Running locally, just print
        print(f"  {name}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Derive changed operators from PR diff"
    )
    parser.add_argument("--base", help="Base commit SHA (three-dot diff with --head)")
    parser.add_argument("--head", help="Head commit SHA")
    parser.add_argument(
        "--changed-files",
        help="Space- or newline-separated list of changed file paths, used "
        "instead of a git diff.",
    )
    args = parser.parse_args()

    if args.changed_files:
        changed_files = [f for f in args.changed_files.split() if f.strip()]
    elif args.base and args.head:
        changed_files = get_diff_files(args.base, args.head)
    else:
        parser.error("provide either --changed-files or both --base and --head")

    all_operators = load_operators_yaml()
    changed_ops = derive_operators(changed_files, all_operators)

    if args.base and args.head:
        print(f"Comparing {args.base}..{args.head}")
    print(f"Changed files ({len(changed_files)}):")
    for f in changed_files[:20]:
        print(f"  {f}")
    if len(changed_files) > 20:
        print(f"  ... and {len(changed_files) - 20} more")

    print(f"Total operators in registry: {len(all_operators)}")
    print(f"Changed operators ({len(changed_ops)}):")
    for op in changed_ops[:20]:
        print(f"  {op}")
    if len(changed_ops) > 20:
        print(f"  ... and {len(changed_ops) - 20} more")

    # Set outputs
    ops_json = json.dumps(changed_ops)
    files_json = json.dumps(changed_files)
    has_changes = "true" if changed_ops else "false"

    set_output("changed_operators", ops_json)
    set_output("changed_files", files_json)
    set_output("has_changes", has_changes)

    print(f"\nhas_changes={has_changes}")


if __name__ == "__main__":
    main()
