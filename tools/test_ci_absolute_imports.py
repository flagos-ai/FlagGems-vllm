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

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src/flaggems_vllm"
BACKEND_ROOT = PACKAGE_ROOT / "runtime/backend"
CANONICAL_PACKAGE = "flaggems_vllm"


def production_operator_files() -> list[Path]:
    """Return operator sources and the vendor packages that register them."""
    paths = set((PACKAGE_ROOT / "ops").rglob("*.py"))
    paths.add(PACKAGE_ROOT / "ops_moe_mxq.py")
    for vendor_root in BACKEND_ROOT.glob("_*"):
        if vendor_root.is_dir():
            paths.update(vendor_root.rglob("*.py"))
    return sorted(paths)


def repository_module_names() -> set[str]:
    """Return names that would be ambiguous when imported without a package."""
    names = {CANONICAL_PACKAGE}
    for source in PACKAGE_ROOT.rglob("*.py"):
        if source.name == "__init__.py":
            names.add(source.parent.name)
        else:
            names.add(source.stem)
    return names


def non_absolute_internal_imports(
    source: str,
    internal_names: set[str],
) -> list[tuple[int, str]]:
    """Find relative imports and bare imports that match repository modules."""
    lines = source.splitlines()
    violations = []
    for node in ast.walk(ast.parse(source)):
        imported_modules: list[str]
        if isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append((node.lineno, lines[node.lineno - 1].strip()))
                continue
            imported_modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
        else:
            continue

        for module_name in imported_modules:
            if module_name == CANONICAL_PACKAGE or module_name.startswith(
                f"{CANONICAL_PACKAGE}."
            ):
                continue
            if module_name.partition(".")[0] in internal_names:
                violations.append((node.lineno, lines[node.lineno - 1].strip()))
                break
    return violations


class AbsoluteOperatorImportTest(unittest.TestCase):
    def test_production_operator_imports_are_package_qualified(self):
        internal_names = repository_module_names()
        violations = []
        for path in production_operator_files():
            source = path.read_text(encoding="utf-8")
            for line_number, statement in non_absolute_internal_imports(
                source, internal_names
            ):
                relative_path = path.relative_to(REPO_ROOT)
                violations.append(f"{relative_path}:{line_number}: {statement}")

        self.assertEqual(
            violations,
            [],
            "Production operators must use absolute 'flaggems_vllm.*' imports:\n"
            + "\n".join(violations),
        )

    def test_scanner_rejects_relative_and_bare_repository_imports(self):
        source = "\n".join(
            (
                "from .scaled_int8_quant import scaled_int8_quant",
                "from backend_utils import VendorDescriptor",
                "import flaggems_vllm.ops",
                "import torch",
            )
        )

        self.assertEqual(
            non_absolute_internal_imports(
                source,
                {"backend_utils", "scaled_int8_quant"},
            ),
            [
                (1, "from .scaled_int8_quant import scaled_int8_quant"),
                (2, "from backend_utils import VendorDescriptor"),
            ],
        )


class OptionalTritonImportTest(unittest.TestCase):
    def test_paged_mqa_import_without_tensor_descriptor(self):
        names = (
            "torch",
            "triton",
            "triton.language",
            "triton.tools",
            "flag_gems",
            "flag_gems.utils",
            "flag_gems.utils.device_info",
            "flag_gems.utils.triton_version_utils",
        )
        modules = {name: types.ModuleType(name) for name in names}
        for module in modules.values():
            module.__path__ = []
        modules["triton"].jit = lambda fn: fn
        modules["triton"].language = modules["triton.language"]
        modules["triton.language"].constexpr = object()
        modules["torch"].float8_e4m3fn = object()
        modules["flag_gems.utils.triton_version_utils"].has_triton_tle = (
            lambda *_version: False
        )
        capability = mock.Mock(side_effect=AssertionError("unexpected device query"))
        modules["flag_gems.utils.device_info"].get_device_capability = capability
        # Simulate Triton 3.1 even if a newer Triton is installed on the runner.
        modules["triton.tools.tensor_descriptor"] = None
        source = PACKAGE_ROOT / "ops/fp8_fp4_paged_mqa_logits.py"
        spec = importlib.util.spec_from_file_location("_test_paged_mqa", source)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        operator = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, modules):
            spec.loader.exec_module(operator)
            self.assertTrue(callable(operator.fp8_fp4_paged_mqa_logits))
            self.assertFalse(operator.HAS_TLE)
            self.assertFalse(operator._can_use_tle(16384, 256, 128))
            capability.assert_not_called()

            # On a supporting backend, descriptor construction still uses TMA.
            descriptor = types.ModuleType("triton.tools.tensor_descriptor")
            descriptor.TensorDescriptor = mock.Mock()
            sys.modules[descriptor.__name__] = descriptor
            kv_data = mock.Mock()
            result = operator._build_kv_descriptor(kv_data, 2, 256, 128)
            kv_data.view.assert_called_once_with(2, 256, 128)
            kv_data.view.return_value.view.assert_called_once_with(
                modules["torch"].float8_e4m3fn
            )
            descriptor.TensorDescriptor.from_tensor.assert_called_once_with(
                kv_data.view.return_value.view.return_value,
                block_shape=[1, 256, 128],
            )
            self.assertIs(result, descriptor.TensorDescriptor.from_tensor.return_value)


class CanonicalBackendLoaderTest(unittest.TestCase):
    module_names = ("backend_utils", "_nvidia")

    def setUp(self):
        self.saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == CANONICAL_PACKAGE
            or name.startswith(f"{CANONICAL_PACKAGE}.")
            or name in self.module_names
        }
        self._remove_test_modules()

    def tearDown(self):
        self._remove_test_modules()
        sys.modules.update(self.saved_modules)

    def _remove_test_modules(self):
        for name in list(sys.modules):
            if (
                name == CANONICAL_PACKAGE
                or name.startswith(f"{CANONICAL_PACKAGE}.")
                or name in self.module_names
            ):
                sys.modules.pop(name, None)

    @staticmethod
    def _package(name: str, path: Path) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        return module

    def test_vendor_loader_ignores_bare_module_sentinels(self):
        sys_path_before = tuple(sys.path)
        package = self._package(CANONICAL_PACKAGE, PACKAGE_ROOT)
        runtime = self._package(
            f"{CANONICAL_PACKAGE}.runtime", PACKAGE_ROOT / "runtime"
        )
        sys.modules[package.__name__] = package
        sys.modules[runtime.__name__] = runtime

        common = types.ModuleType(f"{CANONICAL_PACKAGE}.runtime.common")
        common.vendors = types.SimpleNamespace(get_all_vendors=lambda: {})
        sys.modules[common.__name__] = common

        canonical_backend_utils = types.ModuleType(
            f"{CANONICAL_PACKAGE}.runtime.backend.backend_utils"
        )
        canonical_backend_utils.BackendEventBase = object
        canonical_backend_utils.VendorDescriptor = lambda **values: (
            types.SimpleNamespace(**values)
        )
        sys.modules[canonical_backend_utils.__name__] = canonical_backend_utils

        bare_backend_utils = types.ModuleType("backend_utils")

        def reject_bare_backend_utils(**_values):
            self.fail("vendor package imported the bare backend_utils sentinel")

        bare_backend_utils.VendorDescriptor = reject_bare_backend_utils
        sys.modules[bare_backend_utils.__name__] = bare_backend_utils

        bare_vendor = types.ModuleType("_nvidia")
        bare_vendor.vendor_info = object()
        sys.modules[bare_vendor.__name__] = bare_vendor

        module_name = f"{CANONICAL_PACKAGE}.runtime.backend"
        module_path = BACKEND_ROOT / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_path,
            submodule_search_locations=[str(BACKEND_ROOT)],
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        backend = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = backend
        spec.loader.exec_module(backend)

        loaded_vendor = backend.get_vendor_module("nvidia", query=True)
        loaded_vendor_with_prefix = backend.get_vendor_module("_nvidia", query=True)
        state_vendor = backend.get_vendor_module("nvidia")

        self.assertEqual(
            loaded_vendor.__name__,
            f"{CANONICAL_PACKAGE}.runtime.backend._nvidia",
        )
        self.assertIs(loaded_vendor_with_prefix, loaded_vendor)
        self.assertIs(state_vendor, loaded_vendor)
        self.assertIs(backend.get_vendor_info(), loaded_vendor.vendor_info)
        self.assertIsNot(loaded_vendor, bare_vendor)
        self.assertEqual(loaded_vendor.vendor_info.vendor_name, "nvidia")
        self.assertIs(sys.modules["backend_utils"], bare_backend_utils)
        self.assertIs(sys.modules["_nvidia"], bare_vendor)
        self.assertEqual(tuple(sys.path), sys_path_before)


if __name__ == "__main__":
    unittest.main()
