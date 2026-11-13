# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only startup regressions for the MetaX einsum benchmark launcher."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MetaxEinsumEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.vendor = Path(self.temporary.name)
        files = {
            "torch/__init__.py": '__version__ = "2.8.0+metax"\n',
            "triton/__init__.py": '__version__ = "3.6.0"\n',
            "triton/compiler/__init__.py": "",
            "triton/compiler/compiler.py": "",
            "triton/runtime/__init__.py": "",
            "triton/runtime/cache.py": 'def triton_key():\n    return "runtime"\n',
            "deep_gemm/__init__.py": "from .einsum import einsum\n",
            "deep_gemm/einsum.py": "def einsum(*args, **kwargs):\n    pass\n",
        }
        for name, content in files.items():
            path = self.vendor / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def run_launcher(self, program, overrides=None):
        environment = os.environ.copy()
        environment.update(
            METAX_EINSUM_PYTHON=sys.executable,
            METAX_DEEP_GEMM_ROOT=str(self.vendor),
            PYTHONPATH="",
        )
        for name in ("METAX_EINSUM_GPU", "METAX_EINSUM_EXPECTED_SHA256"):
            environment.pop(name, None)
        environment.update(overrides or {})
        return subprocess.run(
            [
                "bash",
                str(Path(__file__).with_name("run_metax_einsum.sh")),
                "-c",
                program,
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_missing_cache_fingerprint_is_available_to_the_child(self):
        result = self.run_launcher(
            "from triton.compiler.compiler import triton_key; "
            "assert triton_key() == 'runtime'"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_existing_cache_fingerprint_is_preserved(self):
        (self.vendor / "triton/compiler/compiler.py").write_text(
            'def triton_key():\n    return "existing"\n'
        )
        result = self.run_launcher(
            "from triton.compiler.compiler import triton_key; "
            "assert triton_key() == 'existing'"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_baseline_mismatch_stops_before_the_user_program(self):
        result = self.run_launcher(
            "print('USER_PROGRAM_RAN')",
            {"METAX_EINSUM_EXPECTED_SHA256": "0" * 64},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 mismatch", result.stderr)
        self.assertNotIn("USER_PROGRAM_RAN", result.stdout)

    def test_gpu_selection_reaches_all_vendor_masks(self):
        result = self.run_launcher(
            "import os; "
            "assert all(os.environ[name] == '3' for name in "
            "('CUDA_VISIBLE_DEVICES', 'MACA_VISIBLE_DEVICES', 'METAX_VISIBLE_DEVICES'))",
            {"METAX_EINSUM_GPU": "3", "CUDA_VISIBLE_DEVICES": "0"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
