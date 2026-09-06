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

"""Static Triton-only provenance gate for the complete production mHC package."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PDL_FILES = (
    "src/flaggems_vllm/ops/mhc/mhc_post.py",
    "src/flaggems_vllm/ops/mhc/mhc_prenorm.py",
    "src/flaggems_vllm/ops/mhc/mhc_pre_with_norm.py",
    "src/flaggems_vllm/ops/mhc/mhc_pre.py",
    "src/flaggems_vllm/ops/mhc/mhc_fused_post_pre.py",
)
MHC_SOURCE_FILES = tuple(
    path.relative_to(REPOSITORY_ROOT).as_posix()
    for path in sorted((REPOSITORY_ROOT / "src/flaggems_vllm/ops/mhc").glob("*.py"))
)
FORBIDDEN_IMPORT_ROOTS = {
    "cutlass",
    "deep_gemm",
    "tilelang",
    "taichi",
}
ALLOWED_HOST_TORCH_CALLS = {
    "torch.cuda.Event",
    "torch.cuda.current_device",
    "torch.cuda.current_stream",
    "torch.cuda.get_device_capability",
    "torch.cuda.is_current_stream_capturing",
    "torch.empty",
    "torch.empty_like",
}
FORBIDDEN_TENSOR_METHODS = {
    "add_",
    "bfloat16",
    "bmm",
    "clone",
    "contiguous",
    "copy_",
    "cpu",
    "cuda",
    "exp",
    "fill_",
    "float",
    "half",
    "matmul",
    "mm",
    "mul_",
    "normal_",
    "reshape",
    "rsqrt",
    "sigmoid",
    "softmax",
    "square",
    "sum",
    "to",
    "type",
    "type_as",
    "zero_",
}


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_triton_kernel(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _dotted_name(target) in {"triton.jit", "tl.jit"}:
            return True
    return False


class _ProductionVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.failures: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                self.failures.append(f"line {node.lineno}: import {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root in FORBIDDEN_IMPORT_ROOTS or "deep_gemm" in module:
            self.failures.append(f"line {node.lineno}: from {module} import ...")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if _is_triton_kernel(node):
            return
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if _is_triton_kernel(node):
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        if (
            dotted is not None
            and dotted.startswith("torch.")
            and dotted not in ALLOWED_HOST_TORCH_CALLS
        ):
            self.failures.append(f"line {node.lineno}: forbidden call {dotted}")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in FORBIDDEN_TENSOR_METHODS
        ):
            self.failures.append(
                f"line {node.lineno}: forbidden host method .{node.func.attr}()"
            )
        self.generic_visit(node)


@pytest.mark.parametrize("relative_path", MHC_SOURCE_FILES)
def test_mhc_production_is_triton_only(relative_path: str) -> None:
    path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _ProductionVisitor()
    visitor.visit(tree)
    assert not visitor.failures, f"{relative_path}:\n" + "\n".join(visitor.failures)


def test_mhc_forward_has_no_external_gpu_dsl_source_tokens() -> None:
    forbidden_tokens = ("import tilelang", "@tilelang", "deep_gemm", "torch.ops")
    failures: list[str] = []
    for relative_path in MHC_SOURCE_FILES:
        source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in source:
                failures.append(f"{relative_path}: {token}")
    assert not failures, "\n".join(failures)


def test_mhc_pdl_uses_distinct_kernel_and_launcher_keywords() -> None:
    failures: list[str] = []
    pdl_launches = 0
    for relative_path in PDL_FILES:
        path = REPOSITORY_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                _is_triton_kernel(node)
            ):
                argument_names = {argument.arg for argument in node.args.args}
                if "launch_pdl" in argument_names:
                    failures.append(
                        f"{relative_path}:{node.lineno}: kernel constexpr shadows "
                        "the Triton launch_pdl option"
                    )
            if isinstance(node, ast.Call):
                keywords = {keyword.arg for keyword in node.keywords}
                if "LAUNCH_PDL" in keywords:
                    pdl_launches += 1
                    if "launch_pdl" not in keywords:
                        failures.append(
                            f"{relative_path}:{node.lineno}: LAUNCH_PDL has no "
                            "matching Triton launcher option"
                        )
    assert pdl_launches > 0
    assert not failures, "\n".join(failures)
