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


def load_auto_selected_backends(path: Path) -> set[str]:
    """Return backends approved for automatic path-based CI routing."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("backend capabilities must use schema_version 1")

    defaults = config.get("defaults")
    backends = config.get("backends")
    if not isinstance(defaults, dict) or not isinstance(backends, dict):
        raise ValueError("backend capabilities require defaults and backends objects")

    default = defaults.get("auto_select")
    if default is not False:
        raise ValueError("defaults.auto_select must be false")

    selected = set()
    for backend, override in backends.items():
        if not isinstance(backend, str) or not isinstance(override, dict):
            raise ValueError("backend capability entries must be named objects")
        value = override.get("auto_select", default)
        if not isinstance(value, bool):
            raise ValueError(f"auto_select for {backend!r} must be a boolean")
        if value:
            selected.add(backend)
    return selected


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
    *,
    auto_selected_backends: set[str],
) -> list[dict[str, str]]:
    changed_files = changed_files or set()
    registry_backends = {entry["backend"] for entry in registry}
    unknown_backends = auto_selected_backends - registry_backends
    if unknown_backends:
        unknown = ", ".join(sorted(unknown_backends))
        raise ValueError(f"auto-selected backends missing from registry: {unknown}")

    selected = []
    for entry in registry:
        backend = entry["backend"]
        if not entry["enabled"] or backend.startswith("nvidia"):
            continue
        explicitly_selected = all_enabled or entry["label"] in labels
        automatically_selected = backend_changed(backend, changed_files) and (
            backend in auto_selected_backends
        )
        if not explicitly_selected and not automatically_selected:
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
    parser.add_argument("--capabilities", required=True, type=Path)
    parser.add_argument("--all-enabled", action="store_true")
    parser.add_argument("--format", choices=("github", "json", "list"), default="list")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auto_selected_backends = load_auto_selected_backends(args.capabilities)
    selected = select_backends(
        load_registry(args.registry),
        parse_labels(args.labels_json),
        args.all_enabled,
        read_changed_files(args.changed_files),
        auto_selected_backends=auto_selected_backends,
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
