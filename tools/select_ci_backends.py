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

"""Resolve PR-level container CI backends into a GitHub Actions matrix.

Reads `.github/configs/ci-backends.yaml`, whose schema mirrors FlagOps'
`flagos-user-tests/resource_map.yaml`, and emits one matrix entry per enabled
platform. Keeping the runner/image data in YAML (rather than inline in the
workflow) is what allows this file to move to FlagOps later by changing only
the read path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FIELDS = (
    "backend",
    "runner_labels",
    "container_image",
    "gpu_check_script",
)


def load_platforms(path: Path) -> dict[str, dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("ci-backends config must be a mapping")

    platforms = config.get("platforms")
    if not isinstance(platforms, dict) or not platforms:
        raise ValueError("ci-backends config requires a non-empty platforms mapping")

    for name, entry in platforms.items():
        if not isinstance(entry, dict):
            raise ValueError(f"platform {name!r} must be a mapping")
        missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            raise ValueError(f"platform {name!r} is missing: {', '.join(missing)}")
        for field in REQUIRED_FIELDS:
            if not isinstance(entry[field], str):
                raise ValueError(f"platform {name!r}.{field} must be a string")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"platform {name!r}.enabled must be a boolean")
        volumes = entry.get("container_volumes") or []
        if not isinstance(volumes, list) or not all(
            isinstance(volume, str) for volume in volumes
        ):
            raise ValueError(
                f"platform {name!r}.container_volumes must be a list of strings"
            )
        test_env = entry.get("test_env") or {}
        if not isinstance(test_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in test_env.items()
        ):
            raise ValueError(f"platform {name!r}.test_env must be a mapping of strings")
    return platforms


def select_platforms(
    platforms: dict[str, dict[str, Any]],
    only: set[str],
) -> list[dict[str, Any]]:
    unknown = only - set(platforms)
    if unknown:
        raise ValueError(f"unknown platforms requested: {', '.join(sorted(unknown))}")

    selected = []
    for name, entry in platforms.items():
        # An explicit request overrides `enabled`, so a maintainer can run a
        # platform that is still being brought up without editing the config.
        if only:
            if name not in only:
                continue
        elif not entry.get("enabled", True):
            continue

        selected.append(
            {
                "platform": name,
                "backend": entry["backend"],
                "runner_labels": entry["runner_labels"],
                "container_image": " ".join(entry["container_image"].split()),
                "container_options": " ".join(
                    (entry.get("container_options") or "").split()
                ),
                # Kept as a real list so `container.volumes` can consume it
                # directly, matching how ops-test.yaml passes volumes.
                "container_volumes": entry.get("container_volumes") or [],
                # JSON-encoded so the workflow can fromJSON it into `env:`,
                # matching how ops-test.yaml passes test_env.
                "test_env": json.dumps(
                    entry.get("test_env") or {}, separators=(",", ":")
                ),
                "gpu_check_script": entry["gpu_check_script"],
            }
        )
    return selected


def parse_only(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".github/configs/ci-backends.yaml"),
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated platforms to select, overriding `enabled`",
    )
    parser.add_argument("--format", choices=("github", "json", "list"), default="list")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_platforms(
        load_platforms(args.config),
        parse_only(args.only),
    )
    matrix = {"include": selected}

    if args.format == "github":
        print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
        print(f"has_backends={'true' if selected else 'false'}")
    elif args.format == "json":
        print(json.dumps(matrix, indent=2))
    else:
        print("\n".join(entry["platform"] for entry in selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
