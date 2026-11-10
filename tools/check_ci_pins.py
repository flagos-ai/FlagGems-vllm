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

"""Verify that every version pin in this repo agrees with conf/ci-pins.yaml.

Checks:
  1. The three FlagGems SHA pins in basic-ci.yml and
     .github/actions/setup-flaggems/action.yml all match the
     `flaggems_sha` key in conf/ci-pins.yaml.
  2. The fallback version defaults in tools/setup.sh match the
     corresponding keys in conf/ci-pins.yaml.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_PINS_FILE = REPO_ROOT / "conf" / "ci-pins.yaml"

SHA = r"[0-9a-f]{40}"

# Patterns that locate the three FlagGems SHA pins.
REUSABLE_PATTERN = re.compile(
    rf"flagos-ai/FlagGems/\.github/workflows/backend-test\.yaml@({SHA})"
)
CHECKOUT_PATTERN = re.compile(
    rf"repository:\s*flagos-ai/FlagGems\s+ref:\s*({SHA})", re.MULTILINE
)

# Patterns for version fallbacks in tools/setup.sh.
# Each maps a conf/ci-pins.yaml key → regex that matches the shell default.
SETUP_SH_PATTERNS: dict[str, re.Pattern[str]] = {
    "vllm": re.compile(r'VLLM_VERSION="\$\{VLLM_VERSION:-([^}]+)\}"'),
    "torch": re.compile(r'TORCH_VERSION="\$\{TORCH_VERSION:-([^}]+)\}"'),
    "torchaudio": re.compile(r'TORCHAUDIO_VERSION="\$\{TORCHAUDIO_VERSION:-([^}]+)\}"'),
    "torchvision": re.compile(
        r'TORCHVISION_VERSION="\$\{TORCHVISION_VERSION:-([^}]+)\}"'
    ),
    "flagtree": re.compile(r'FLAGTREE_VERSION="\$\{FLAGTREE_VERSION:-([^}]+)\}"'),
}


def load_pins() -> dict[str, str]:
    text = CI_PINS_FILE.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
        return {k: str(v) for k, v in data.items() if isinstance(v, str)}
    # Fallback: simple key: "value" regex (handles quoted scalar only).
    result: dict[str, str] = {}
    for m in re.finditer(r'^(\w+):\s*"([^"]+)"', text, re.MULTILINE):
        result[m.group(1)] = m.group(2)
    return result


def extract_pins(repo_root: Path) -> list[str]:
    """Return the list of FlagGems SHA strings found in workflow/action files.

    Kept for backward compatibility with test_ci_helpers.py.
    """
    workflow = (repo_root / ".github/workflows/basic-ci.yml").read_text(
        encoding="utf-8"
    )
    setup_action = (repo_root / ".github/actions/setup-flaggems/action.yml").read_text(
        encoding="utf-8"
    )
    found = REUSABLE_PATTERN.findall(workflow)
    found.extend(CHECKOUT_PATTERN.findall(workflow))
    found.extend(CHECKOUT_PATTERN.findall(setup_action))
    return found


def check_flaggems_sha(pins: dict[str, str]) -> list[str]:
    expected = pins.get("flaggems_sha", "")
    if not expected:
        return ["conf/ci-pins.yaml: missing 'flaggems_sha' key"]

    workflow = (REPO_ROOT / ".github/workflows/basic-ci.yml").read_text(
        encoding="utf-8"
    )
    setup_action = (REPO_ROOT / ".github/actions/setup-flaggems/action.yml").read_text(
        encoding="utf-8"
    )

    found = REUSABLE_PATTERN.findall(workflow)
    found.extend(CHECKOUT_PATTERN.findall(workflow))
    found.extend(CHECKOUT_PATTERN.findall(setup_action))

    errors = []
    if len(found) != 3:
        errors.append(
            f"expected 3 FlagGems SHA pins in workflow/action files, found {len(found)}: {found}"
        )
        return errors
    if len(set(found)) != 1:
        errors.append(f"FlagGems SHA pins are inconsistent: {found}")
        return errors
    actual = found[0]
    if actual != expected:
        errors.append(
            f"FlagGems SHA mismatch: conf/ci-pins.yaml has {expected!r}, "
            f"workflow/action files have {actual!r}"
        )
    return errors


def check_setup_sh_versions(pins: dict[str, str]) -> list[str]:
    setup_sh = (REPO_ROOT / "tools/setup.sh").read_text(encoding="utf-8")
    errors = []
    for key, pattern in SETUP_SH_PATTERNS.items():
        expected = pins.get(key)
        if expected is None:
            errors.append(f"conf/ci-pins.yaml: missing key '{key}'")
            continue
        m = pattern.search(setup_sh)
        if m is None:
            errors.append(
                f"tools/setup.sh: could not find fallback default for '{key}' "
                f"(expected pattern {pattern.pattern!r})"
            )
            continue
        actual = m.group(1)
        if actual != expected:
            errors.append(
                f"tools/setup.sh fallback for '{key}' is {actual!r}, "
                f"but conf/ci-pins.yaml says {expected!r}"
            )
    return errors


def main() -> int:
    if not CI_PINS_FILE.exists():
        print(
            f"::error::conf/ci-pins.yaml not found at {CI_PINS_FILE}", file=sys.stderr
        )
        return 2

    pins = load_pins()
    errors: list[str] = []
    errors.extend(check_flaggems_sha(pins))
    errors.extend(check_setup_sh_versions(pins))

    if errors:
        for err in errors:
            print(f"::error::{err}")
        return 1

    sha = pins.get("flaggems_sha", "")
    print(f"FlagGems CI pin: {sha}")
    print("All version pins consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
