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

"""Select non-NVIDIA CI backends from the FlagGems registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("backend", "runner_label", "label", "gpu_check", "enabled")
BACKEND_SOURCE_ROOT = "src/flaggems_vllm/runtime/backend"


def load_registry(path: Path) -> list[dict[str, Any]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, list):
        raise ValueError("backend registry must be a JSON array")

    seen_backends = set()
    for index, entry in enumerate(registry):
        if not isinstance(entry, dict):
            raise ValueError(f"backend registry entry {index} must be an object")
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"backend registry entry {index} is missing: {', '.join(missing)}"
            )
        for field in ("backend", "runner_label", "label", "gpu_check"):
            if not isinstance(entry[field], str):
                raise ValueError(
                    f"backend registry entry {index}.{field} must be a string"
                )
        if not isinstance(entry["enabled"], bool):
            raise ValueError(
                f"backend registry entry {index}.enabled must be a boolean"
            )
        if entry["backend"] in seen_backends:
            raise ValueError(f"duplicate backend name: {entry['backend']!r}")
        seen_backends.add(entry["backend"])
    return registry


def parse_labels(value: str) -> set[str]:
    labels = json.loads(value)
    if labels is None:
        return set()
    if not isinstance(labels, list) or not all(
        isinstance(label, str) for label in labels
    ):
        raise ValueError("pull request labels must be a JSON array of strings")
    return set(labels)


def read_changed_files(path: Path | None) -> set[str]:
    if path is None:
        return set()
    data = path.read_bytes()
    entries = data.split(b"\0") if b"\0" in data else data.splitlines()
    return {
        entry.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        for entry in entries
        if entry
    }


def backend_changed(backend: str, changed_files: set[str]) -> bool:
    vendor = backend.split("-", maxsplit=1)[0]
    prefix = f"{BACKEND_SOURCE_ROOT}/_{vendor}/"
    return any(path.startswith(prefix) for path in changed_files)


def select_backends(
    registry: list[dict[str, Any]],
    labels: set[str],
    all_enabled: bool,
    changed_files: set[str] | None = None,
) -> list[dict[str, str]]:
    changed_files = changed_files or set()
    selected = []
    for entry in registry:
        backend = entry["backend"]
        if not entry["enabled"] or backend.startswith("nvidia"):
            continue
        if (
            not all_enabled
            and entry["label"] not in labels
            and not backend_changed(backend, changed_files)
        ):
            continue

        selected.append(
            {
                "backend": backend,
                "runner_label": entry["runner_label"],
                "gpu_check": entry["gpu_check"],
            }
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--changed-files", type=Path)
    parser.add_argument("--all-enabled", action="store_true")
    parser.add_argument("--format", choices=("github", "json", "list"), default="list")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_backends(
        load_registry(args.registry),
        parse_labels(args.labels_json),
        args.all_enabled,
        read_changed_files(args.changed_files),
    )
    matrix = {"include": selected}

    if args.format == "github":
        print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
        print(f"has_backends={'true' if selected else 'false'}")
    elif args.format == "json":
        print(json.dumps(matrix, indent=2))
    else:
        print("\n".join(entry["backend"] for entry in selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
