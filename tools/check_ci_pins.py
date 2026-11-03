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

"""Ensure every FlagGems CI integration point uses the same full commit SHA."""

from __future__ import annotations

import re
from pathlib import Path

SHA = r"[0-9a-f]{40}"
REUSABLE_PATTERN = re.compile(
    rf"flagos-ai/FlagGems/\.github/workflows/backend-test\.yaml@({SHA})"
)
CHECKOUT_PATTERN = re.compile(
    rf"repository:\s*flagos-ai/FlagGems\s+ref:\s*({SHA})", re.MULTILINE
)


def extract_pins(repo_root: Path) -> list[str]:
    workflow = (repo_root / ".github/workflows/basic-ci.yml").read_text(
        encoding="utf-8"
    )
    setup_action = (repo_root / ".github/actions/setup-flaggems/action.yml").read_text(
        encoding="utf-8"
    )
    pins = REUSABLE_PATTERN.findall(workflow)
    pins.extend(CHECKOUT_PATTERN.findall(workflow))
    pins.extend(CHECKOUT_PATTERN.findall(setup_action))
    return pins


def main() -> int:
    pins = extract_pins(Path.cwd())
    if len(pins) != 3:
        raise RuntimeError(f"expected 3 FlagGems CI pins, found {len(pins)}: {pins}")
    if len(set(pins)) != 1:
        raise RuntimeError(f"FlagGems CI pins are inconsistent: {pins}")
    print(f"FlagGems CI pin: {pins[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
