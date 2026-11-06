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

"""
Auto-sort __all__ exports in Python files and operators in operators.yaml.

Usage:
    # Check if files are sorted (exit 0 if sorted, 1 if not)
    python tools/ci_checks/sort_exports.py --check

    # Fix sorting in all files
    python tools/ci_checks/sort_exports.py --fix

    # Fix specific files only
    python tools/ci_checks/sort_exports.py --fix --files src/flaggems_vllm/ops/__init__.py

    # Dry run (show what would be changed)
    python tools/ci_checks/sort_exports.py --fix --dry-run
"""

import argparse
import ast
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def sort_python_all(file_path: Path, fix: bool = False, dry_run: bool = False) -> bool:
    """
    Sort __all__ list in a Python file by casefold.

    Args:
        file_path: Path to Python file
        fix: If True, modify the file; if False, only check
        dry_run: If True with fix, show changes but don't write

    Returns:
        True if file is already sorted, False otherwise
    """
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return True

    source = file_path.read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"❌ {file_path}: cannot parse ({exc})")
        return False

    all_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        and isinstance(node.value, ast.List)
    ]

    if not all_nodes:
        return True

    if len(all_nodes) > 1:
        lines_at = ", ".join(str(n.lineno) for n in all_nodes)
        print(
            f"❌ {file_path}: found {len(all_nodes)} `__all__` list definitions "
            f"(lines {lines_at}); expected exactly one. Skipping to avoid data loss."
        )
        return False

    node = all_nodes[0]
    items = [
        elt.value
        for elt in node.value.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]

    node_start = node.lineno - 1
    node_end = node.end_lineno - 1
    prefix = "__all__ = ["
    suffix = "]"

    if not items:
        return True

    # Remove duplicates while preserving order, then sort by casefold
    seen = set()
    unique_items = []
    duplicates = []
    for item in items:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
            unique_items.append(item)

    sorted_items = sorted(unique_items, key=str.casefold)

    if items == sorted_items:
        return True

    if not fix:
        print(f"❌ {file_path}: __all__ is not sorted by casefold")
        if duplicates:
            print(
                f"   Found {len(duplicates)} duplicate(s): {', '.join(set(duplicates))}"
            )
        for i, (actual, expected) in enumerate(zip(unique_items, sorted_items)):
            if actual != expected:
                print(f"   Position {i}: got '{actual}', expected '{expected}'")
                break
        return False

    # Detect indent and quote style from the existing __all__ block
    source_lines = source.split("\n")
    block_lines = source_lines[node_start : node_end + 1]
    quote = '"'
    indent = 4
    for line in block_lines[1:]:
        stripped = line.strip()
        if stripped and stripped != "]":
            if line != line.lstrip():
                indent = len(line) - len(line.lstrip())
            quote = '"' if '"' in stripped else "'"
            break

    indent_str = " " * indent

    if dry_run:
        msg = f"Would sort {file_path}: {len(unique_items)} items"
        if duplicates:
            msg += f" (removing {len(duplicates)} duplicate(s))"
        print(msg)
        print(f"  First item: '{sorted_items[0]}'")
        print(f"  Last item: '{sorted_items[-1]}'")
        return False

    new_block = [prefix]
    new_block += [f"{indent_str}{quote}{item}{quote}," for item in sorted_items]
    new_block.append(suffix)

    new_source = "\n".join(
        source_lines[:node_start] + new_block + source_lines[node_end + 1 :]
    )

    file_path.write_text(new_source)
    msg = f"✅ {file_path}: sorted {len(sorted_items)} items in __all__"
    if duplicates:
        msg += f" (removed {len(duplicates)} duplicate(s): {', '.join(sorted(set(duplicates)))})"
    print(msg)
    return False


def sort_operators_yaml(
    file_path: Path, fix: bool = False, dry_run: bool = False
) -> bool:
    """
    Sort operators.yaml by id.casefold().

    Args:
        file_path: Path to operators.yaml
        fix: If True, modify the file; if False, only check
        dry_run: If True with fix, show changes but don't write

    Returns:
        True if file is already sorted, False otherwise
    """
    if yaml is None:
        print(
            "Error: pyyaml is required for sorting operators.yaml. "
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(2)

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return True

    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    ops = data.get("ops", [])
    if not ops:
        return True

    ids = [op["id"] for op in ops]
    sorted_ids = sorted(ids, key=str.casefold)

    if ids == sorted_ids:
        return True

    if not fix:
        print(f"❌ {file_path}: operators not sorted by id.casefold()")
        for i, (actual, expected) in enumerate(zip(ids, sorted_ids)):
            if actual != expected:
                print(f"   Position {i}: got '{actual}', expected '{expected}'")
                break
        return False

    sorted_ops = sorted(ops, key=lambda x: x["id"].casefold())

    if dry_run:
        print(f"Would sort {file_path}: {len(ops)} operators")
        print(f"  First: '{sorted_ops[0]['id']}'")
        print(f"  Last: '{sorted_ops[-1]['id']}'")
        return False

    # Read original file to preserve header comments
    original = file_path.read_text()
    lines = original.splitlines()

    ops_line_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "ops:":
            ops_line_idx = i
            break

    if ops_line_idx is None:
        print(f"Error: Cannot find 'ops:' line in {file_path}", file=sys.stderr)
        return False

    header = "\n".join(lines[: ops_line_idx + 1]) + "\n"

    data["ops"] = sorted_ops
    yaml_content = yaml.dump(
        data, default_flow_style=False, allow_unicode=True, sort_keys=False
    )

    yaml_lines = yaml_content.splitlines()
    ops_start = None
    for i, line in enumerate(yaml_lines):
        if line.strip() == "ops:":
            ops_start = i + 1
            break

    if ops_start is None:
        print("Error: Cannot parse dumped YAML", file=sys.stderr)
        return False

    ops_body = "\n".join(yaml_lines[ops_start:])

    new_content = header + ops_body + "\n"
    file_path.write_text(new_content)
    print(f"✅ {file_path}: sorted {len(ops)} operators")
    return False


def discover_init_files() -> list[Path]:
    """Discover all __init__.py files that should be checked.

    Same logic as check_init_exports.py for consistency.
    """
    root = Path("src/flaggems_vllm")
    if not root.exists():
        return []

    files = []

    ops_init = root / "ops" / "__init__.py"
    if ops_init.exists():
        files.append(ops_init)

    backend_root = root / "runtime" / "backend"
    if backend_root.exists():
        for vendor_dir in sorted(backend_root.iterdir()):
            if not vendor_dir.is_dir() or not vendor_dir.name.startswith("_"):
                continue

            vendor_ops_init = vendor_dir / "ops" / "__init__.py"
            if vendor_ops_init.exists():
                files.append(vendor_ops_init)

            for arch_dir in sorted(vendor_dir.iterdir()):
                if not arch_dir.is_dir():
                    continue
                arch_ops_init = arch_dir / "ops" / "__init__.py"
                if arch_ops_init.exists():
                    files.append(arch_ops_init)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Sort __all__ exports and operators.yaml by casefold"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are sorted (don't modify)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix sorting issues by modifying files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to process (default: auto-discover all __init__.py + operators.yaml)",
    )

    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.error("Must specify either --check or --fix")

    if args.check and args.fix:
        parser.error("Cannot use both --check and --fix")

    if args.files:
        files_to_check = [Path(f) for f in args.files]
    else:
        files_to_check = discover_init_files()
        files_to_check.append(Path("conf/operators.yaml"))

    print(f"Processing {len(files_to_check)} file(s)...\n")

    all_sorted = True

    for file_path in files_to_check:
        if file_path.name == "operators.yaml":
            sorted_ok = sort_operators_yaml(
                file_path, fix=args.fix, dry_run=args.dry_run
            )
        else:
            sorted_ok = sort_python_all(file_path, fix=args.fix, dry_run=args.dry_run)

        all_sorted = all_sorted and sorted_ok

    if not all_sorted:
        if args.check:
            print("\n❌ Some files are not sorted. Run this to fix:")
            print("    python tools/ci_checks/sort_exports.py --fix")
            sys.exit(1)
        elif args.dry_run:
            print("\nRun without --dry-run to apply changes:")
            print("    python tools/ci_checks/sort_exports.py --fix")
            sys.exit(0)
        else:
            print("\n✅ All files have been sorted")
            sys.exit(0)
    else:
        if args.check:
            print("✅ All files are correctly sorted")
        sys.exit(0)


if __name__ == "__main__":
    main()
