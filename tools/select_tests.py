#!/usr/bin/env python3
"""Select pytest and benchmark targets for CI from changed files."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


SMOKE_TESTS = [
    "tests/test_outer.py",
    "tests/test_bincount.py",
    "tests/test_silu_and_mul.py",
    "tests/test_moe_align_block_size.py",
]

SMOKE_BENCHMARKS = [
    "benchmark/test_outer.py",
    "benchmark/test_bincount.py",
    "benchmark/test_silu_and_mul.py",
    "benchmark/test_moe_align_block_size_triton.py",
]

BROAD_TEST_FILES = {
    "tests/conftest.py",
    "tests/accuracy_utils.py",
    "pytest.ini",
}

BROAD_SOURCE_PREFIXES = (
    "src/flaggems_vllm/runtime/",
    "src/flaggems_vllm/testing/",
    "src/flaggems_vllm/utils/",
)

NON_TEST_PREFIXES = (
    "docs/",
)

NON_TEST_FILES = {
    ".flake8",
    ".gitignore",
    ".pre-commit-config.yaml",
    "LICENSE",
    "README.md",
    "README_cn.md",
    "workflow.md",
}

EXPLICIT_SOURCE_TO_TESTS = {
    "src/flaggems_vllm/ops/rotary_embedding.py": ["tests/test_apply_rotary_pos_emb.py"],
    "src/flaggems_vllm/ops/flashmla_sparse.py": ["tests/test_flash_mla_sparse_fwd.py"],
    "src/flaggems_vllm/ops/fused_moe.py": ["tests/test_fused_experts_impl.py"],
    "src/flaggems_vllm/ops/sparse_attention.py": ["tests/test_flash_attention.py"],
    "src/flaggems_vllm/ops/quant.py": ["tests/test_quant.py"],
}

EXPLICIT_SOURCE_TO_BENCHMARKS = {
    "src/flaggems_vllm/ops/rotary_embedding.py": [
        "benchmark/test_apply_rotary_pos_emb.py"
    ],
    "src/flaggems_vllm/ops/flashmla_sparse.py": [
        "benchmark/test_flash_mla_sparse_fwd.py"
    ],
    "src/flaggems_vllm/ops/fused_moe.py": [
        "benchmark/test_fused_moe.py",
        "benchmark/test_fused_moe_fp8.py",
        "benchmark/test_fused_moe_fp8_blockwise.py",
        "benchmark/test_fused_moe_int4_w4a16.py",
        "benchmark/test_fused_moe_int8.py",
        "benchmark/test_fused_moe_int8_w8a16.py",
        "benchmark/test_fused_moe_w8a16.py",
    ],
    "src/flaggems_vllm/ops/sparse_attention.py": [
        "benchmark/test_sparse_attention.py",
    ],
    "src/flaggems_vllm/ops/moe_align_block_size.py": [
        "benchmark/test_moe_align_block_size_triton.py",
    ],
}

def normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def existing_tests(repo_root: Path) -> list[str]:
    return sorted(
        path.as_posix()
        for path in (repo_root / "tests").rglob("test_*.py")
        if path.is_file()
    )


def existing_benchmarks(repo_root: Path) -> list[str]:
    return sorted(
        path.as_posix()
        for path in (repo_root / "benchmark").rglob("test_*.py")
        if path.is_file()
    )


def add_target(targets: set[str], target: str, existing_targets: set[str]) -> None:
    normalized = normalize_path(target)
    if normalized in existing_targets:
        targets.add(normalized)


def tests_for_source(path: str, tests: set[str]) -> list[str]:
    if path in EXPLICIT_SOURCE_TO_TESTS:
        return [test for test in EXPLICIT_SOURCE_TO_TESTS[path] if test in tests]

    if path.startswith("src/flaggems_vllm/ops/FLA/"):
        return sorted(test for test in tests if test.startswith("tests/test_FLA/"))

    if path.startswith("src/flaggems_vllm/ops/DSA/"):
        return sorted(test for test in tests if test.startswith("tests/test_DSA/"))

    if path.startswith("src/flaggems_vllm/ops/mhc/"):
        return ["tests/test_mhc_ops.py"] if "tests/test_mhc_ops.py" in tests else []

    if not path.startswith("src/flaggems_vllm/ops/") or not path.endswith(".py"):
        return []

    stem = Path(path).stem
    candidates = [
        f"tests/test_{stem}.py",
        f"tests/test_{stem.replace('layernorm', 'layer_norm')}.py",
        f"tests/test_{stem.replace('weightnorm', 'weight_norm')}.py",
    ]
    return [candidate for candidate in candidates if candidate in tests]


def benchmarks_for_source(path: str, benchmarks: set[str]) -> list[str]:
    if path in EXPLICIT_SOURCE_TO_BENCHMARKS:
        return [
            benchmark
            for benchmark in EXPLICIT_SOURCE_TO_BENCHMARKS[path]
            if benchmark in benchmarks
        ]

    if path.startswith("src/flaggems_vllm/ops/FLA/"):
        return sorted(
            benchmark
            for benchmark in benchmarks
            if benchmark.startswith("benchmark/test_FLA/")
        )

    if path.startswith("src/flaggems_vllm/ops/DSA/"):
        return []

    if path.startswith("src/flaggems_vllm/ops/mhc/"):
        return ["benchmark/test_mhc.py"] if "benchmark/test_mhc.py" in benchmarks else []

    if not path.startswith("src/flaggems_vllm/ops/") or not path.endswith(".py"):
        return []

    stem = Path(path).stem
    candidates = [
        f"benchmark/test_{stem}.py",
        f"benchmark/test_{stem.replace('layernorm', 'layer_norm')}.py",
        f"benchmark/test_{stem.replace('weightnorm', 'weight_norm')}.py",
    ]
    return [candidate for candidate in candidates if candidate in benchmarks]


def is_non_test_change(path: str) -> bool:
    return path in NON_TEST_FILES or path.startswith(NON_TEST_PREFIXES)


def select_targets(
    repo_root: Path, changed_files: list[str]
) -> tuple[str, list[str], list[str]]:
    tests = set(existing_tests(repo_root))
    benchmarks = set(existing_benchmarks(repo_root))
    test_targets: set[str] = set()
    benchmark_targets: set[str] = set()
    broad_change = False
    code_change = False

    for raw_path in changed_files:
        path = normalize_path(raw_path)
        if not path:
            continue

        if path in BROAD_TEST_FILES or path.startswith(".github/workflows/"):
            broad_change = True

        if path.startswith(BROAD_SOURCE_PREFIXES) or path in {
            "src/flaggems_vllm/__init__.py",
            "src/flaggems_vllm/config.py",
            "pyproject.toml",
        }:
            broad_change = True

        if (
            path.startswith(("src/", "tests/", "benchmark/"))
            or path in {"pyproject.toml", "pytest.ini"}
        ):
            code_change = True

        if path.startswith("tests/test_") and path.endswith(".py"):
            add_target(test_targets, path, tests)

        if path.startswith("benchmark/test_") and path.endswith(".py"):
            add_target(benchmark_targets, path, benchmarks)

        for target in tests_for_source(path, tests):
            add_target(test_targets, target, tests)

        for target in benchmarks_for_source(path, benchmarks):
            add_target(benchmark_targets, target, benchmarks)

    if test_targets or benchmark_targets:
        return "selected", sorted(test_targets), sorted(benchmark_targets)

    if broad_change or code_change:
        return (
            "smoke",
            [test for test in SMOKE_TESTS if test in tests],
            [benchmark for benchmark in SMOKE_BENCHMARKS if benchmark in benchmarks],
        )

    if changed_files and all(is_non_test_change(normalize_path(path)) for path in changed_files):
        return "skip", [], []

    return (
        "smoke",
        [test for test in SMOKE_TESTS if test in tests],
        [benchmark for benchmark in SMOKE_BENCHMARKS if benchmark in benchmarks],
    )


def read_changed_files(path: str | None) -> list[str]:
    if not path:
        return []

    changed_files_path = Path(path)
    if not changed_files_path.exists():
        return []

    return changed_files_path.read_text(encoding="utf-8").splitlines()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("--changed-files", help="file containing changed file paths")
    parser.add_argument(
        "--format",
        choices=("shell", "list"),
        default="list",
        help="output format",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode, tests, benchmarks = select_targets(
        Path(args.repo_root),
        read_changed_files(args.changed_files),
    )

    if args.format == "shell":
        print(f"TEST_SELECTION_MODE={shlex.quote(mode)}")
        print(f"SELECTED_TESTS={shlex.quote(' '.join(tests))}")
        print(f"SELECTED_BENCHMARKS={shlex.quote(' '.join(benchmarks))}")
    else:
        print("\n".join(tests + benchmarks))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
