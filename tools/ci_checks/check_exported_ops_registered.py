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

"""Check that every statically exported op is registered in operators.yaml.

workflow.md requires: "An op is not 'done' until it is exported and
documented in conf/operators.yaml." The existing rule-check.yaml pipeline
(check_operators_yaml.py, check_operator_markers.py, check_aten_operators.py)
only validates operators that are *already present* in operators.yaml; an
operator exported from src/flaggems_vllm/ops/__init__.py's static __all__
but never added to operators.yaml passes all of those checks silently,
because each one filters its input down to `op_id in all_operators` first.
This script closes that gap by comparing __all__ directly against the
registry's id column.

Only the literal `__all__ = [...]` assignment is read; vendor-conditional
exports appended at import time (`if vendor == ...: __all__ += [...]`) are
out of scope, since evaluating those needs every vendor's runtime.

A pre-existing backlog of unregistered exports is carried in
tools/ci_checks/unregistered_exports_baseline.txt, generated with
--update-baseline, so this check can be enforced going forward without
failing every PR on the existing gap at once. New exports must either be
registered in conf/operators.yaml, or (temporarily, with justification)
added to the baseline file.

Exit codes:
  0 - all checks pass
  1 - new unregistered exports found, or the baseline is stale
  2 - script internal error
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

OPS_INIT = Path("src/flaggems_vllm/ops/__init__.py")
OPERATORS_YAML = Path("conf/operators.yaml")
BASELINE_FILE = Path("tools/ci_checks/unregistered_exports_baseline.txt")

ID_PATTERN = re.compile(r"^\s*-\s*id:\s*(\S+)", re.MULTILINE)


def static_all(path: Path) -> list[str]:
    """Extract the literal __all__ = [...] list; see module docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.List):
            return [
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
    raise ValueError(f"no literal __all__ list assignment found in {path}")


def registered_ids(path: Path) -> set[str]:
    return set(ID_PATTERN.findall(path.read_text(encoding="utf-8")))


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline file to match the current gap and exit 0",
    )
    args = parser.parse_args()

    if not OPS_INIT.exists() or not OPERATORS_YAML.exists():
        print(f"::error::Cannot find {OPS_INIT} or {OPERATORS_YAML}", file=sys.stderr)
        return 2

    try:
        exported = set(static_all(OPS_INIT))
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    registered = registered_ids(OPERATORS_YAML)
    unregistered = exported - registered

    if args.update_baseline:
        BASELINE_FILE.write_text(
            "\n".join(sorted(unregistered)) + "\n" if unregistered else "",
            encoding="utf-8",
        )
        print(f"Wrote {len(unregistered)} entries to {BASELINE_FILE}")
        return 0

    baseline = load_baseline(BASELINE_FILE)
    new_gaps = sorted(unregistered - baseline)
    stale_baseline = sorted(baseline - unregistered)

    if stale_baseline:
        print("Baseline entries no longer needed (now registered or not exported):")
        for name in stale_baseline:
            print(f"  • {name}")
        print(
            f"Remove these from {BASELINE_FILE} "
            "(or regenerate it with --update-baseline)."
        )

    if new_gaps:
        print("Exported ops missing from conf/operators.yaml:")
        for name in new_gaps:
            print(
                f"::error::'{name}' is exported from ops/__init__.py but not registered in conf/operators.yaml"
            )
            print(f"  • {name}")
        print(
            "Add an entry to conf/operators.yaml, or if this export is not a "
            f"user-facing op, add it to {BASELINE_FILE} with a comment "
            "explaining why."
        )

    if new_gaps or stale_baseline:
        return 1

    print(
        f"✅ {len(exported)} exported op(s) checked, "
        f"{len(baseline)} pre-existing baseline exemption(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
