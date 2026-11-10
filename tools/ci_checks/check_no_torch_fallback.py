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

"""Static check for NO_TORCH_COMPUTE_FALLBACK in production operator code.

Flags calls to torch compute APIs (torch.<op>, torch.nn.functional.*,
torch.ops.aten.*) under src/flaggems_vllm/ops and the vendor ops/fused/FLA
directories. Memory/creation/metadata helpers are allowed. A line may opt out
with a trailing '# allow-torch-compute' comment, which reviewers should question.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ALLOWED_TORCH_CALLS = {
    "empty",
    "empty_like",
    "empty_strided",
    "zeros",
    "zeros_like",
    "as_strided",
    "device",
    "Size",
    "finfo",
    "iinfo",
    "get_default_dtype",
    "is_tensor",
    "is_floating_point",
    "no_grad",
    "inference_mode",
    # creation / metadata only, no device compute
    "tensor",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "arange",
    "from_numpy",
    "result_type",
    "promote_types",
    "is_grad_enabled",
    "set_grad_enabled",
}
# torch sub-namespaces that are runtime/infra rather than tensor compute.
ALLOWED_TORCH_NAMESPACES = {
    "cuda",
    "version",
    "library",
    "distributed",
    "utils",
    "jit",
    "compiler",
    "_C",
    "testing",
    "amp",
    # vendor device runtimes (stream/event/graph/device management)
    "npu",
    "musa",
    "mlu",
    "xpu",
    "mps",
    "gcu",
    "ixuca",
}
PRAGMA = "allow-torch-compute"
VENDOR_SUBDIRS = ("ops", "fused", "FLA")


def dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def functional_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "torch.nn.functional" and item.asname:
                    aliases.add(item.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch.nn":
                for item in node.names:
                    if item.name == "functional":
                        aliases.add(item.asname or item.name)
    return aliases


def classify(name: str, aliases: set[str]) -> bool:
    parts = name.split(".")
    if parts[0] in aliases:
        return True
    if parts[0] != "torch" or len(parts) < 2:
        return False
    if name.startswith("torch.nn.functional."):
        return True
    if name.startswith("torch.ops.aten."):
        return True
    if parts[1] in ALLOWED_TORCH_NAMESPACES or parts[1] in ALLOWED_TORCH_CALLS:
        return False
    # torch.ops.<custom> and torch.nn.* modules are not compute calls we can judge.
    if parts[1] in ("ops", "nn"):
        return False
    # Only lowercase names are functions; torch.Tensor / torch.dtype are types.
    return parts[1][:1].islower()


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: cannot parse ({exc.msg})"]
    lines = source.splitlines()
    aliases = functional_aliases(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if not name or not classify(name, aliases):
            continue
        if PRAGMA in lines[node.lineno - 1]:
            continue
        found.append(f"{path}:{node.lineno}: forbidden torch compute call {name}()")
    return found


def default_files(root: Path) -> list[Path]:
    pkg = root / "src" / "flaggems_vllm"
    files = set((pkg / "ops").rglob("*.py"))
    for vendor in (pkg / "runtime" / "backend").glob("_*"):
        for sub in VENDOR_SUBDIRS:
            files.update((vendor / sub).rglob("*.py"))
    return sorted(p for p in files if p.is_file())


def in_scope(path: Path) -> bool:
    posix = path.as_posix()
    if "src/flaggems_vllm/ops/" in posix:
        return True
    if "src/flaggems_vllm/runtime/backend/_" in posix:
        return any(f"/{sub}/" in posix for sub in VENDOR_SUBDIRS)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files to check (default: all production operator files)",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--warn-only", action="store_true", help="report violations but exit 0"
    )
    args = parser.parse_args()

    if args.paths:
        files = [
            p for p in args.paths if p.suffix == ".py" and p.is_file() and in_scope(p)
        ]
    else:
        files = default_files(args.root)

    problems: list[str] = []
    for path in files:
        problems.extend(check_file(path))

    for line in problems:
        print(line)
    print(f"checked {len(files)} files, {len(problems)} violation(s)")
    if problems and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
